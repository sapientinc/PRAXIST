"""Post-generation boundary actions for the research-loop stage."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    is_committed_runtime_fact_file,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    evidence_maturity_snapshot,
    has_explicit_false_completion,
    resolve_result_snapshot_producers,
    result_snapshot_key,
    same_result_snapshot,
    task_authorizes_descriptive_maturity,
)
from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
    finding_source_published_after,
    findings_at_boundary_cutoff,
    include_finding_sources_in_snapshot,
    preserve_scoreless_finding_sources,
    reconcile_result_source_snapshot,
    result_source_snapshot_at_cutoff,
    scoreless_findings_with_frozen_sources,
)
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    BOUNDARY_CHECKPOINT_CUTOFF_KEY,
    BOUNDARY_CHECKPOINT_SNAPSHOT_KEY,
    read_boundary_evidence_checkpoint,
    write_boundary_evidence_checkpoint,
    write_boundary_marker,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import (
    is_scoreless,
    read_scoreless_evidence_manifest,
    write_scoreless_evidence_manifest,
)

logger = logging.getLogger(__name__)

_DIVERSITY_TELEMETRY_KEYS = frozenset(
    {
        "diversity_overlap_count",
        "diversity_overlap_total",
        "diversity_overlap_fraction",
        "diversity_most_similar_anchor",
        "diversity_overlap_status",
        "diversity_overlap_no_data_reason",
        "diversity_violated",
        "diversity_narrow_variation",
    }
)
_REFRESH_VOLATILE_KEYS = frozenset(
    {
        "timestamp",
        "created_at",
        "updated_at",
        "observed_at",
        "promoted_at",
    }
)


def _stable_evidence_payload(value: Any) -> Any:
    """Remove observation-only metadata before boundary refresh comparison."""
    if isinstance(value, dict):
        return {
            str(key): _stable_evidence_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _DIVERSITY_TELEMETRY_KEYS and str(key) not in _REFRESH_VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_evidence_payload(item) for item in value]
    if isinstance(value, set):
        return sorted((_stable_evidence_payload(item) for item in value), key=repr)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _evidence_field(item: dict[str, Any], key: str) -> str:
    for container in (
        item,
        item.get("metrics"),
        item.get("details"),
        item.get("current_aggregate"),
    ):
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _canonical_evidence_signature(
    items: list[dict[str, Any]],
) -> dict[tuple[str, str, str], tuple[str, ...]]:
    """Fingerprint canonical evidence without diversity/display telemetry."""
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for item in items:
        payload = _stable_evidence_payload(item)
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        identity = (
            _evidence_field(item, "id") or _evidence_field(item, "finding_id"),
            _evidence_field(item, "source_result_path"),
            _evidence_field(item, "source_result_sha256"),
        )
        if not any(identity):
            identity = ("anonymous", digest, "")
        grouped.setdefault(identity, []).append(digest)
    return {identity: tuple(sorted(digests)) for identity, digests in grouped.items()}


def _annotate_diversity_overlap(
    loop: Any, *, gen_id: int, findings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Annotate findings with diversity telemetry when frontier anchors exist."""
    strategy = loop._strategy_for_gen(gen_id)
    if strategy not in ("explore", "mixed", "pi_directed") or gen_id <= 0:
        return findings
    current_anchors = loop.frontier.get_summary()
    if not current_anchors:
        return findings

    configured_dims = getattr(loop.task_spec.evaluation, "diversity_dimensions", None) or []
    expected_dim_count = len(configured_dims) if configured_dims else 4
    from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
        annotate_findings_with_diversity_overlap,
    )

    findings = annotate_findings_with_diversity_overlap(
        findings,
        current_anchors,
        expected_dim_count,
    )
    clones = [
        f for f in findings if f.get("metrics", {}).get("diversity_overlap_status") == "clone"
    ]
    narrows = [
        f for f in findings if f.get("metrics", {}).get("diversity_overlap_status") == "narrow"
    ]
    cleans = [
        f for f in findings if f.get("metrics", {}).get("diversity_overlap_status") == "clean"
    ]
    no_data = [
        f for f in findings if f.get("metrics", {}).get("diversity_overlap_status") == "no_data"
    ]
    logger.info(
        "Gen %d (%s phase) diversity report (3-tier): "
        "clones=%d, narrows=%d, cleans=%d, no_data=%d (out of %d findings).",
        gen_id,
        strategy,
        len(clones),
        len(narrows),
        len(cleans),
        len(no_data),
        len(findings),
    )
    for clone in clones:
        metrics = clone.get("metrics", {})
        logger.warning(
            "  diversity-CLONE: variant=%s matched %d/%d dims with anchor=%s "
            "(overlap=%.0f%%) — this looks like a duplicate of an existing frontier entry.",
            clone.get("variant_name", "?"),
            metrics.get("diversity_overlap_count", "?"),
            metrics.get("diversity_overlap_total", "?"),
            metrics.get("diversity_most_similar_anchor", "?"),
            100 * metrics.get("diversity_overlap_fraction", 0),
        )
    for narrow in narrows:
        metrics = narrow.get("metrics", {})
        logger.info(
            "  diversity-narrow: variant=%s matched %d/%d dims with anchor=%s "
            "(overlap=%.0f%%) — narrow refinement (acceptable).",
            narrow.get("variant_name", "?"),
            metrics.get("diversity_overlap_count", "?"),
            metrics.get("diversity_overlap_total", "?"),
            metrics.get("diversity_most_similar_anchor", "?"),
            100 * metrics.get("diversity_overlap_fraction", 0),
        )
    if no_data:
        reason_counts: dict[str, int] = {}
        for finding in no_data:
            reason = str(
                finding.get("metrics", {}).get("diversity_overlap_no_data_reason", "unspecified")
            )
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        logger.warning(
            "  %d/%d findings have no comparable design-dimension overlap (reasons=%s).",
            len(no_data),
            len(findings),
            ", ".join(f"{key}:{value}" for key, value in sorted(reason_counts.items())),
        )
    return findings


def _sync_graph_before_next_generation(loop: Any, *, gen_id: int) -> None:
    """Run a blocking graph sync so next-generation prompts see fresh edges."""
    if loop._graph_maintainer is None:
        return
    try:
        result = loop._graph_maintainer.sync_once_blocking(timeout=300.0)
        if result.get("status") == "timeout":
            logger.warning(
                "inter-generation graph sync hit 5-min timeout; gen %d prompts may see stale edges",
                gen_id + 1,
            )
    except Exception as e:  # noqa: BLE001 - graph context is advisory.
        logger.debug("inter-generation graph sync failed: %s", e)


@contextlib.contextmanager
def _hold_findings_sync_for_gems(loop: Any):
    """Serialize Gems archive/reset against the local FindingsSync daemon.

    Gems resets prune the active findings filesystem and SQLite rows. The local
    FindingsSync sidecar is bidirectional, so a concurrent sync pass could
    otherwise materialize a stale SQLite snapshot back into shared_findings.
    Holding its private mutex here is intentional: the sidecar already exposes
    the mutex as its sync_once serialization boundary.
    """

    sync = getattr(loop, "_findings_sync", None)
    mutex = getattr(sync, "_sync_mutex", None)
    if mutex is None:
        yield sync
        return
    mutex.acquire()
    try:
        yield sync
    finally:
        mutex.release()


def _sync_findings_locked_once(sync: Any, *, reason: str) -> None:
    if sync is None:
        return
    try:
        if hasattr(sync, "_sync_once_locked"):
            sync._sync_once_locked()
        elif hasattr(sync, "sync_once"):
            sync.sync_once()
    except Exception as exc:  # noqa: BLE001 - sync is advisory before/after reset.
        logger.warning("gems: findings sync %s failed: %s", reason, exc)


def _write_boundary_marker_if_possible(
    loop: Any,
    *,
    gen_id: int,
    promoted_count: int,
    pi_status: str,
    agenda_path: str | None = None,
    error: str | None = None,
    stop_audit: dict[str, Any] | None = None,
    peer_mix: dict[str, Any] | None = None,
    evidence_cutoff_at: str | None = None,
    evidence_source_snapshot_at_cutoff: dict[str, str] | None = None,
) -> None:
    run_dir = getattr(loop, "run_dir", None)
    if run_dir is None:
        return
    try:
        write_boundary_marker(
            run_dir,
            gen_id=gen_id,
            promoted_count=promoted_count,
            pi_status=pi_status,
            agenda_path=agenda_path,
            error=error,
            stop_audit=stop_audit,
            peer_mix=peer_mix,
            evidence_cutoff_at=evidence_cutoff_at,
            evidence_source_snapshot_at_cutoff=evidence_source_snapshot_at_cutoff,
        )
    except Exception as exc:  # noqa: BLE001 - an unmarked boundary is not committed.
        logger.error("generation boundary marker write failed: %s", exc)
        raise RuntimeError(f"generation {gen_id} boundary could not be committed") from exc
    _clear_boundary_evidence_cutoff(loop, gen_id=gen_id)


def _clear_boundary_evidence_cutoff(loop: Any, *, gen_id: int) -> None:
    """Retire a generation cutoff after its canonical marker commits."""

    findings_sync = getattr(loop, "_findings_sync", None)
    clear_cutoff = getattr(findings_sync, "clear_boundary_evidence_cutoff", None)
    if callable(clear_cutoff):
        clear_cutoff(gen_id)
    active_boundary = getattr(loop, "_boundary_evidence_cutoff", None)
    if active_boundary is not None and active_boundary[0] == int(gen_id):
        loop._boundary_evidence_cutoff = None


def _activate_boundary_evidence_cutoff(
    loop: Any,
    *,
    gen_id: int,
    cutoff: datetime,
    evidence_source_snapshot: dict[str, str],
) -> None:
    """Align the sidecar with the final evidence snapshot until commit."""

    findings_sync = getattr(loop, "_findings_sync", None)
    begin_cutoff = getattr(findings_sync, "begin_boundary_evidence_cutoff", None)
    if callable(begin_cutoff):
        begin_cutoff(gen_id, cutoff, evidence_source_snapshot)
    loop._boundary_evidence_cutoff = (
        int(gen_id),
        cutoff,
        dict(evidence_source_snapshot),
    )


def _parse_signal_payload(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, Any] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        out[key] = _coerce_signal_value(value.strip())
    return out


def _coerce_signal_value(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _generation_stop_audit(loop: Any, *, gen_id: int) -> dict[str, Any]:
    run_dir = Path(getattr(loop, "run_dir", ""))
    gen_dir = run_dir / f"gen_{gen_id}"
    if not gen_dir.exists():
        return {}
    for name in ("STOP_SIGNAL", "STOP_SIGNAL_POSTGEN", "CLOSING_SIGNAL"):
        path = gen_dir / name
        if not path.exists():
            continue
        payload = _parse_signal_payload(path)
        payload.pop(BOUNDARY_CHECKPOINT_CUTOFF_KEY, None)
        payload.pop(BOUNDARY_CHECKPOINT_SNAPSHOT_KEY, None)
        payload["signal_file"] = str(path.relative_to(run_dir)) if run_dir else str(path)
        payload.setdefault("generation_id", gen_id)
        if is_scoreless(getattr(loop, "task_spec", None)):
            payload["mode"] = "scoreless"
            payload["evidence_status"] = "not_scored"
            payload["evidence_sufficient"] = None
            return payload
        if name == "STOP_SIGNAL_POSTGEN" or payload.get("trigger_reason") in {
            "safety_cap",
            "generation_wall_timeout",
        }:
            required = _safe_int(payload.get("required_mature_result_peers"), 0)
            mature = _safe_int(payload.get("mature_result_peers"), 0)
            payload.setdefault("evidence_sufficient", required > 0 and mature >= required)
        elif payload.get("trigger_reason") in {
            "cohort_drained",
            "cohort_drained_insufficient_mature",
            "peers_finished_before_trigger",
        }:
            payload.setdefault("evidence_sufficient", False)
        else:
            payload.setdefault("evidence_sufficient", True)
        return payload
    if is_scoreless(getattr(loop, "task_spec", None)):
        return {
            "generation_id": gen_id,
            "signal_file": "",
            "mode": "scoreless",
            "evidence_status": "not_scored",
            "evidence_sufficient": None,
        }
    return {"generation_id": gen_id, "signal_file": "", "evidence_sufficient": False}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


_NON_CONSTRUCTIVE_TOKENS = {
    "ablation",
    "ablate",
    "audit",
    "benchmark",
    "diagnostic",
    "diagnose",
    "falsifier",
    "falsify",
    "negative_control",
    "reference",
}
_CONSTRUCTIVE_TOKENS = {
    "bridge",
    "candidate",
    "combine",
    "construct",
    "exploit",
    "explore",
    "forward",
    "improve",
    "innovate",
    "mechanism",
    "new",
    "optimize",
    "parent",
    "repair",
    "solution",
    "variant",
}


def _is_constructive_payload(payload: dict[str, Any]) -> bool | None:
    fields: list[str] = []
    for key in (
        "role",
        "peer_role",
        "intent",
        "intent_preference",
        "next_step_intent",
        "parent_usage",
        "source_lane",
        "target_lane",
        "frontier_lane",
        "promotion_lane",
        "lane",
        "finding_type",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            fields.extend(str(item) for item in value)
        elif value is not None:
            fields.append(str(value))
    text = " ".join(fields).lower().replace("-", "_")
    tokens = {token for token in re.split(r"[^a-z0-9_]+", text) if token}
    parts = set(tokens)
    for token in tokens:
        parts.update(part for part in token.split("_") if part)
    if _NON_CONSTRUCTIVE_TOKENS & tokens or "negative_control" in tokens:
        return False
    if any(
        part in {"ablation", "audit", "benchmark", "diagnostic", "falsifier", "reference"}
        for part in parts
    ):
        return False
    if _CONSTRUCTIVE_TOKENS & parts:
        return True
    return None


def _candidate_payload(finding: dict[str, Any]) -> dict[str, Any]:
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    details = finding.get("details") if isinstance(finding.get("details"), dict) else {}
    extra = finding.get("extra") if isinstance(finding.get("extra"), dict) else {}
    nested = extra.get("extra") if isinstance(extra.get("extra"), dict) else {}
    return {**finding, **details, **extra, **nested, **metrics}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _falsey(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 0
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "n", "off"}
    return False


def _has_validation_only_marker(payload: dict[str, Any]) -> bool:
    if any(
        _truthy(payload.get(key))
        for key in (
            "validation_only",
            "validation_only_result",
            "late_after_generation_boundary",
        )
    ):
        return True
    for key in ("artifact_signal_status", "late_result_policy", "durability_scope"):
        token = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(payload.get(key) or "").strip().lower(),
        ).strip("_")
        if token in {
            "late_after_generation_boundary",
            "late_quarantined_protected_job",
            "quarantined_signal",
            "validation_signal_only",
            "validation_only",
        }:
            return True
    return False


def _has_hard_non_mature_marker(payload: dict[str, Any]) -> bool:
    if _has_validation_only_marker(payload):
        return True
    if has_explicit_false_completion(payload):
        return True
    for key in (
        "incomplete_eval",
        "is_incomplete_eval",
        "summary_only",
        "is_summary_only",
        "unscored_artifact",
        "suspect",
        "suspect_protocol",
        "suspect_fixed_weight_eval",
        "protocol_integrity_failed",
    ):
        if _truthy(payload.get(key)):
            return True
    if payload.get("protocol_integrity_passed") is False:
        return True
    status = " ".join(
        str(payload.get(key) or "")
        for key in (
            "status",
            "final_status",
            "tier_status",
            "result_status",
            "protocol_integrity_status",
        )
    ).lower()
    tokens = set(token for token in re.split(r"[^a-z0-9]+", status) if token)
    if (
        {"scored", "complete", "false"}.issubset(tokens)
        or {"not", "scored", "complete"}.issubset(tokens)
        or {"not", "complete"}.issubset(tokens)
    ):
        return True
    return any(
        token in status
        for token in (
            "failed",
            "crashed",
            "incomplete",
            "running",
            "invalid",
            "error",
            "summary_only",
            "unscored",
            "not_scored",
        )
    )


def _has_soft_non_mature_marker(payload: dict[str, Any]) -> bool:
    for key in (
        "partial",
        "partial_cohort",
        "partial_eval",
        "is_partial_eval",
        "scout_only",
        "is_scout_eval",
        "is_smoke_eval",
        "smoke_only",
        "capped",
        "is_capped",
        "result_capped",
    ):
        if _truthy(payload.get(key)):
            return True
    status = " ".join(
        str(payload.get(key) or "")
        for key in ("status", "final_status", "tier_status", "result_status")
    ).lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", status) if token}
    return bool(tokens & {"partial", "scout", "smoke", "preliminary", "prelim", "capped"})


def _is_mature_result_payload(payload: dict[str, Any], maturity_policy: Any) -> bool:
    if _has_hard_non_mature_marker(payload):
        return False
    snapshot = evidence_maturity_snapshot(payload, maturity_policy)
    decision = snapshot.get("mature_enough")
    task_stage_is_complete = bool(
        decision is True
        and task_authorizes_descriptive_maturity(
            payload,
            maturity_policy,
            maturity=snapshot,
        )
    )
    if _has_soft_non_mature_marker(payload) and not task_stage_is_complete:
        return False
    if decision is not None:
        return bool(decision)
    if isinstance(maturity_policy, dict) and maturity_policy.get("require_ratio_gate"):
        return False
    return False


def _is_explicitly_non_mature_result_payload(
    payload: dict[str, Any],
    maturity_policy: Any,
) -> bool:
    if _has_hard_non_mature_marker(payload):
        return True
    snapshot = evidence_maturity_snapshot(payload, maturity_policy)
    if snapshot.get("mature_enough") is False:
        return True
    task_stage_is_complete = bool(
        snapshot.get("mature_enough") is True
        and task_authorizes_descriptive_maturity(
            payload,
            maturity_policy,
            maturity=snapshot,
        )
    )
    return _has_soft_non_mature_marker(payload) and not task_stage_is_complete


def _load_generation_agenda(loop: Any, *, gen_id: int) -> dict[str, Any]:
    run_dir = getattr(loop, "run_dir", None)
    if run_dir is None:
        return {}
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import (
            load_agenda_for_gen,
        )

        task_spec = getattr(loop, "task_spec", None)
        gen_policy = getattr(task_spec, "generation_policy", None)
        agenda = load_agenda_for_gen(
            Path(run_dir),
            gen_id,
            cohort_size=int(getattr(gen_policy, "cohort_size", 0) or 0),
        )
    except Exception:
        return {}
    return agenda if isinstance(agenda, dict) else {}


def _generation_peer_mix(
    loop: Any, *, gen_id: int, findings: list[dict[str, Any]]
) -> dict[str, Any]:
    task_spec = getattr(loop, "task_spec", None)
    if is_scoreless(task_spec):
        return {
            "mode": "scoreless",
            "retained_finding_count": len(findings),
            "contributing_peer_count": len(
                {str(finding["peer_id"]) for finding in findings if finding.get("peer_id")}
            ),
            "evidence_status": "not_scored",
            "advisory_only": True,
        }
    evaluation = getattr(task_spec, "evaluation", None)
    gen_policy = getattr(task_spec, "generation_policy", None)
    if not bool(getattr(evaluation, "constructive_peer_mix_enabled", True)):
        return {}
    raw_target = getattr(evaluation, "constructive_target_ratio", 0.75)
    try:
        target = float(raw_target)
    except (TypeError, ValueError):
        target = 0.75
    if not math.isfinite(target):
        target = 0.75
    target = max(0.0, min(1.0, target))
    cohort_size = int(getattr(gen_policy, "cohort_size", 0) or 0)
    maturity_policy = getattr(evaluation, "maturity_policy", {}) if evaluation is not None else {}

    agenda = _load_generation_agenda(loop, gen_id=gen_id)
    contracts = (
        agenda.get("peer_contracts") if isinstance(agenda.get("peer_contracts"), dict) else {}
    )
    contract_total = len(contracts)
    contract_constructive = 0
    contract_nonconstructive = 0
    for contract in contracts.values():
        if not isinstance(contract, dict):
            continue
        verdict = _is_constructive_payload(contract)
        if verdict is True:
            contract_constructive += 1
        elif verdict is False:
            contract_nonconstructive += 1

    candidates = [
        (finding, _candidate_payload(finding))
        for finding in findings
        if finding.get("finding_type") in {"result", "insight", "intermediate_result"}
    ]
    resolved_artifacts = resolve_result_snapshot_producers(
        [
            (result_snapshot_key(finding), finding.get("variant_name"))
            for finding, _payload in candidates
        ]
    )
    artifact_by_finding = {
        id(finding): artifact
        for (finding, _payload), artifact in zip(candidates, resolved_artifacts, strict=True)
    }
    quarantined_artifacts = [
        artifact
        for finding, payload in candidates
        if _is_explicitly_non_mature_result_payload(payload, maturity_policy)
        and (artifact := artifact_by_finding[id(finding)]) is not None
        and bool(artifact[0])
    ]
    counted_artifacts: list[tuple[str, str, str]] = []
    mature_result_total = 0
    mature_constructive = 0
    mature_nonconstructive = 0
    unknown_constructive = 0
    for finding, payload in candidates:
        if not _is_mature_result_payload(payload, maturity_policy):
            continue
        artifact = artifact_by_finding[id(finding)]
        if artifact is not None and (
            any(
                same_result_snapshot(artifact, quarantined) for quarantined in quarantined_artifacts
            )
            or any(same_result_snapshot(artifact, counted) for counted in counted_artifacts)
        ):
            continue
        if artifact is not None:
            counted_artifacts.append(artifact)
        mature_result_total += 1
        verdict = _is_constructive_payload(payload)
        if verdict is True:
            mature_constructive += 1
        elif verdict is False:
            mature_nonconstructive += 1
        else:
            unknown_constructive += 1

    mature_ratio = _safe_ratio(mature_constructive, mature_result_total)
    contract_ratio = _safe_ratio(contract_constructive, contract_total)
    recommended_floor = int((cohort_size * target) + 0.9999) if cohort_size > 0 else 0
    return {
        "target_constructive_ratio": round(target, 4),
        "contract_total": contract_total,
        "contract_constructive_count": contract_constructive,
        "contract_nonconstructive_count": contract_nonconstructive,
        "contract_constructive_ratio": contract_ratio,
        "mature_result_total": mature_result_total,
        "mature_constructive_count": mature_constructive,
        "mature_nonconstructive_count": mature_nonconstructive,
        "mature_unknown_intent_count": unknown_constructive,
        "mature_constructive_ratio": mature_ratio,
        "constructive_deficit": round(max(0.0, target - mature_ratio), 4),
        "recommended_next_gen_constructive_floor": recommended_floor,
        "advisory_only": True,
    }


async def _complete_generation_boundary(
    loop: Any,
    *,
    gen_id: int,
    pi_agent: Any,
    pi_cfg: Any,
) -> None:
    """Promote findings, update advisory state, and run PI synthesis."""
    run_dir = getattr(loop, "run_dir", None)
    if (
        is_scoreless(loop.task_spec)
        and run_dir is not None
        and is_committed_runtime_fact_file(
            Path(run_dir) / f"gen_{gen_id}" / "generation_boundary.json"
        )
    ):
        # A terminal-delivery retry may revisit a completed generation. Its
        # frozen evidence must not be replaced by newly published source files.
        if read_scoreless_evidence_manifest(Path(run_dir), gen_id) is None:
            raise RuntimeError(f"generation {gen_id} frozen scoreless evidence is unavailable")
        return
    persisted_checkpoint = (
        read_boundary_evidence_checkpoint(Path(run_dir), gen_id) if run_dir is not None else None
    )
    active_checkpoint: tuple[datetime, dict[str, str]] | None = None
    active_boundary = getattr(loop, "_boundary_evidence_cutoff", None)
    if (
        isinstance(active_boundary, tuple)
        and len(active_boundary) == 3
        and active_boundary[0] == int(gen_id)
        and isinstance(active_boundary[1], datetime)
        and isinstance(active_boundary[2], dict)
    ):
        active_checkpoint = (active_boundary[1], dict(active_boundary[2]))
    recovered_checkpoint = persisted_checkpoint or active_checkpoint
    if recovered_checkpoint is not None:
        recovered_cutoff, recovered_snapshot = recovered_checkpoint
        _activate_boundary_evidence_cutoff(
            loop,
            gen_id=gen_id,
            cutoff=recovered_cutoff,
            evidence_source_snapshot=recovered_snapshot,
        )
    raw_findings = loop._collect_findings_for_generation(gen_id)
    raw_findings_collected_at = datetime.now(UTC)
    initial_evidence = _canonical_evidence_signature(raw_findings)

    # Materialization can reveal a source-generation result immediately after
    # the first resolve. Re-read the existing canonical inputs once before PI
    # consumes the manifest; this is idempotent when no new result arrived and
    # avoids a boundary-time race turning a finished result into a later-only
    # validation signal.
    if recovered_checkpoint is not None:
        evidence_cutoff, evidence_source_snapshot = recovered_checkpoint
    else:
        # Order canonical SQLite rows atomically around the cutoff. A source
        # scan on each side reconciles filesystem publications against the same
        # instant without trusting self-reported finding timestamps.
        if run_dir is not None:
            run_path = Path(run_dir)
            evidence_source_snapshot = result_source_snapshot_at_cutoff(run_path)
            snapshot_findings = raw_findings
            declared_mode = getattr(loop, "local_mode", None)
            canonical_findings = declared_mode is not None
            evidence_cutoff = raw_findings_collected_at
            if bool(getattr(loop, "local_mode", False)):
                try:
                    from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
                        snapshot_findings_at_cutoff,
                    )

                    evidence_cutoff, snapshot_findings = snapshot_findings_at_cutoff(gen_id)
                    canonical_findings = True
                except Exception as exc:  # noqa: BLE001 - filesystem cutoff remains usable.
                    logger.warning(
                        "Generation %d canonical finding cutoff snapshot failed: %s; "
                        "using the already collected conservative snapshot.",
                        gen_id,
                        exc,
                    )
            elif declared_mode is None:
                # Lightweight integrations that do not declare a store mode
                # retain their historical refresh semantics. Production
                # server mode explicitly declares False and uses the completed
                # collection above as its conservative canonical snapshot.
                evidence_cutoff = datetime.now(UTC)
            evidence_source_snapshot = reconcile_result_source_snapshot(
                run_path,
                evidence_cutoff,
                evidence_source_snapshot,
            )
            evidence_source_snapshot = include_finding_sources_in_snapshot(
                evidence_source_snapshot,
                snapshot_findings,
                run_dir=run_path,
                findings_dir=Path(getattr(loop, "findings_dir", run_path / "shared_findings")),
                gen_id=gen_id,
                cutoff=evidence_cutoff,
                canonical_findings=canonical_findings,
            )
            if is_scoreless(loop.task_spec):
                evidence_source_snapshot = preserve_scoreless_finding_sources(
                    evidence_source_snapshot, run_dir=run_path, cutoff=evidence_cutoff
                )
        else:
            evidence_cutoff = datetime.now(UTC)
            evidence_source_snapshot = {}
        _activate_boundary_evidence_cutoff(
            loop,
            gen_id=gen_id,
            cutoff=evidence_cutoff,
            evidence_source_snapshot=evidence_source_snapshot,
        )
        if run_dir is not None:
            write_boundary_evidence_checkpoint(
                Path(run_dir),
                gen_id=gen_id,
                cutoff=evidence_cutoff,
                evidence_source_snapshot=evidence_source_snapshot,
            )
    if recovered_checkpoint is not None and persisted_checkpoint is None and run_dir is not None:
        write_boundary_evidence_checkpoint(
            Path(run_dir),
            gen_id=gen_id,
            cutoff=evidence_cutoff,
            evidence_source_snapshot=evidence_source_snapshot,
        )
    evidence_cutoff_at = evidence_cutoff.isoformat()
    boundary_collector = getattr(loop, "_collect_findings_for_boundary", None)
    refreshed_findings: list[dict[str, Any]]
    if callable(boundary_collector):
        refreshed_findings = cast(
            list[dict[str, Any]],
            boundary_collector(
                gen_id,
                evidence_cutoff=evidence_cutoff,
                evidence_source_snapshot=evidence_source_snapshot,
            ),
        )
        post_cutoff_count = sum(
            1
            for finding in refreshed_findings
            if isinstance(finding.get("metrics"), dict)
            and finding["metrics"].get("generation_boundary_pending_commit") is True
        )
    else:
        # Keep lightweight integrations compatible while production loops use
        # the boundary-aware collector that persists these as late signals.
        refreshed_findings = loop._collect_findings_for_generation(gen_id)
        cutoff_findings: list[dict[str, Any]] = []
        post_cutoff_count = 0
        for finding in refreshed_findings:
            if run_dir is not None and finding_source_published_after(
                finding,
                run_dir=Path(run_dir),
                cutoff=evidence_cutoff,
                evidence_source_snapshot=evidence_source_snapshot,
            ):
                post_cutoff_count += 1
            else:
                cutoff_findings.append(finding)
        refreshed_findings = cutoff_findings
    if post_cutoff_count:
        logger.info(
            "Generation %d final evidence cutoff retained %d newly published result(s) "
            "as late validation signals.",
            gen_id,
            post_cutoff_count,
        )
    promotion_findings = findings_at_boundary_cutoff(
        evidence_source_snapshot,
        refreshed_findings,
    )
    promotion_evidence = _canonical_evidence_signature(promotion_findings)
    if promotion_evidence != initial_evidence:
        initial_keys = set(initial_evidence)
        promotion_keys = set(promotion_evidence)
        added = promotion_keys - initial_keys
        removed = initial_keys - promotion_keys
        updated = {
            identity
            for identity in initial_keys & promotion_keys
            if initial_evidence[identity] != promotion_evidence[identity]
        }
        logger.info(
            "Generation %d final evidence refresh: added=%d updated=%d removed=%d; "
            "frontier input refreshed before its single boundary commit.",
            gen_id,
            len(added),
            len(updated),
            len(removed),
        )
    scoreless = is_scoreless(loop.task_spec)
    if scoreless:
        findings = scoreless_findings_with_frozen_sources(
            evidence_source_snapshot, promotion_findings
        )
        observed_findings = refreshed_findings
        promoted = []
        if run_dir is not None:
            write_scoreless_evidence_manifest(
                Path(run_dir),
                gen_id=gen_id,
                findings=findings,
                evidence_cutoff_at=evidence_cutoff_at,
                evidence_source_snapshot=evidence_source_snapshot,
            )
    else:
        findings = _annotate_diversity_overlap(
            loop,
            gen_id=gen_id,
            findings=promotion_findings,
        )
        observed_findings = (
            findings
            if promotion_findings is refreshed_findings
            else _annotate_diversity_overlap(
                loop,
                gen_id=gen_id,
                findings=refreshed_findings,
            )
        )
        promoted = loop.frontier.promote(gen_id, findings)
    stop_audit = _generation_stop_audit(loop, gen_id=gen_id)
    peer_mix = _generation_peer_mix(loop, gen_id=gen_id, findings=observed_findings)
    if scoreless:
        logger.info("Generation %d complete: retained %d unscored findings", gen_id, len(findings))
    else:
        logger.info(
            "Generation %d complete: promoted %d findings to frontier",
            gen_id,
            len(promoted),
        )

    rm_cfg_post = getattr(loop.task_spec, "research_memory", None)
    if rm_cfg_post and getattr(rm_cfg_post, "enabled", False):
        try:
            loop._update_research_memory_post_gen(
                gen_id=gen_id,
                findings=observed_findings,
                promoted=promoted,
            )
        except Exception as e:  # noqa: BLE001 - ledger updates are non-blocking.
            logger.warning(
                "research_memory: post-gen ledger update failed: %s. "
                "Continuing — ledger updates are non-blocking.",
                e,
            )

    _sync_graph_before_next_generation(loop, gen_id=gen_id)

    max_generations = loop.task_spec.generation_policy.max_generations
    is_last_gen = max_generations is not None and gen_id == max_generations - 1
    gems_result = None
    gems = getattr(loop, "gems", None)
    if not scoreless and gems is not None and getattr(gems, "enabled", False) and not is_last_gen:
        try:
            with _hold_findings_sync_for_gems(loop) as findings_sync:
                _sync_findings_locked_once(findings_sync, reason="before Gems reset")
                gems_result = gems.maybe_trigger_after_boundary(completed_gen_id=gen_id)
                if gems_result and gems_result.triggered:
                    _sync_findings_locked_once(findings_sync, reason="after Gems reset")
        except Exception as e:  # noqa: BLE001 - Gems is opt-in but should not corrupt boundary.
            logger.exception("gems: boundary trigger failed for gen %d: %s", gen_id, e)
            raise
        if gems_result and gems_result.triggered:
            _sync_graph_before_next_generation(loop, gen_id=gen_id)
            _write_boundary_marker_if_possible(
                loop,
                gen_id=gen_id,
                promoted_count=len(promoted),
                pi_status="skipped_gems_reset",
                error=(
                    f"gems_reset_count={gems_result.reset_count}; "
                    f"admitted={gems_result.admitted_count}; "
                    f"archive={gems_result.archive_dir}"
                ),
                stop_audit=stop_audit,
                peer_mix=peer_mix,
                evidence_cutoff_at=evidence_cutoff_at,
                evidence_source_snapshot_at_cutoff=evidence_source_snapshot,
            )
            return

    if not scoreless and gems is not None and getattr(gems, "enabled", False) and is_last_gen:
        logger.info(
            "gems: terminal generation %d keeps the active frontier unchanged; "
            "no reset is committed without a successor generation",
            gen_id,
        )

    if pi_agent is None or is_last_gen:
        _write_boundary_marker_if_possible(
            loop,
            gen_id=gen_id,
            promoted_count=len(promoted),
            pi_status="skipped_last_generation" if is_last_gen else "disabled",
            stop_audit=stop_audit,
            peer_mix=peer_mix,
            evidence_cutoff_at=evidence_cutoff_at,
            evidence_source_snapshot_at_cutoff=evidence_source_snapshot,
        )
        return

    try:
        pi_result = await pi_agent.run(completed_gen_id=gen_id)
    except Exception as e:
        if pi_cfg.strict:
            raise
        logger.warning("PI agent raised (strict=False, continuing): %s", e)
        _write_boundary_marker_if_possible(
            loop,
            gen_id=gen_id,
            promoted_count=len(promoted),
            pi_status="raised_non_strict",
            error=str(e),
            stop_audit=stop_audit,
            peer_mix=peer_mix,
            evidence_cutoff_at=evidence_cutoff_at,
            evidence_source_snapshot_at_cutoff=evidence_source_snapshot,
        )
        return

    if pi_result.success:
        logger.info(
            "PI synthesis complete for gen %d → agenda for gen %d at %s (%.0fs)",
            gen_id,
            pi_result.next_gen_id,
            pi_result.agenda_path,
            pi_result.duration_seconds,
        )
        _write_boundary_marker_if_possible(
            loop,
            gen_id=gen_id,
            promoted_count=len(promoted),
            pi_status="succeeded",
            agenda_path=str(pi_result.agenda_path),
            stop_audit=stop_audit,
            peer_mix=peer_mix,
            evidence_cutoff_at=evidence_cutoff_at,
            evidence_source_snapshot_at_cutoff=evidence_source_snapshot,
        )
        return

    msg = f"PI synthesis FAILED for gen {gen_id} → gen {pi_result.next_gen_id}: {pi_result.error}"
    if pi_cfg.strict:
        logger.error(msg + " (strict mode → aborting run)")
        raise RuntimeError(msg)
    logger.warning(
        msg + " (strict=False → next gen will run without a fresh agenda; "
        "peers will fall back to baseline behavior)"
    )
    _write_boundary_marker_if_possible(
        loop,
        gen_id=gen_id,
        promoted_count=len(promoted),
        pi_status="failed_non_strict",
        error=pi_result.error,
        stop_audit=stop_audit,
        peer_mix=peer_mix,
        evidence_cutoff_at=evidence_cutoff_at,
        evidence_source_snapshot_at_cutoff=evidence_source_snapshot,
    )


def record_completed_generation_observation(
    loop: Any,
    *,
    gen_id: int,
    generation_results: list[dict[str, Any]],
) -> None:
    """Publish a privacy-bounded observation of an already durable boundary."""
    try:
        observer = getattr(loop, "run_lifecycle_observer", None)
        if observer is None:
            return
        from praxist.plugins.workflow_stages.research_loop.lifecycle import (
            record_generation_finished_safely,
        )

        record_generation_finished_safely(
            observer,
            generation_ordinal=gen_id,
            planned_peer_count=loop.task_spec.generation_policy.cohort_size,
            results=generation_results,
        )
    except Exception:
        logger.debug("generation lifecycle observation failed", exc_info=True)


async def complete_generation_boundary(
    loop: Any,
    *,
    gen_id: int,
    pi_agent: Any,
    pi_cfg: Any,
    generation_results: list[dict[str, Any]] | None = None,
) -> None:
    """Commit a generation boundary, then publish its aggregate observation."""
    await _complete_generation_boundary(
        loop,
        gen_id=gen_id,
        pi_agent=pi_agent,
        pi_cfg=pi_cfg,
    )
    if generation_results is not None:
        record_completed_generation_observation(
            loop,
            gen_id=gen_id,
            generation_results=generation_results,
        )
