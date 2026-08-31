"""Legacy two-round panel executor.

Orchestrates Round 0 (evidence freeze) → Round 1 (independent
memos, parallel) → Round 2 (cross-review, parallel) → optional Round 2.5
(boundary revision) → Chair synthesis.

Usage from generation_loop:

    from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import run_panel
    result = await run_panel(
        run_dir, workspace, model,
        completed_gen_id=N, panel_mode="full",
        ...,
    )
    if result.success:
        publish_agenda(result.agenda)
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from praxist.core.panel_topology import PanelRoleSpec, panel_topology_for_ref
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    COMMITTED,
    DERIVED_AUDIT_SNAPSHOT,
    FAILED,
    PARTIAL_OUTPUT,
    attach_artifact_semantics,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_metadata import (
    normalize_agenda_research_metadata,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.agenda_validator_v2 import (
    PI_ROLES,
    VALIDATION_PARENT_IDENTITY_KEYS,
    _normalize_required_peer_roles,
    _validation_parent_tokens,
    normalize_validation_candidate_parent_usages,
    validate_agenda_v2,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.chair_arbiter import (
    ChairArbiter,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.pi_roles import BasePI
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.role_bindings import (
    instantiate_pi_roles,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.context_auditor import (
    audit_agenda,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.context_firewall import (
    BUDGETS,
    fit_pack_to_budget,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
    _digest_validation_candidates,
    _validation_candidate_aliases_from_manifest,
    build_evidence_pack,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.metrics_logger import (
    log_synthesis_metrics,
)


def _atomic_write_text(path: Path, payload_writer: Callable[[Any], None]) -> None:
    """Adv-audit-R7 fix: temp + fsync + os.replace pattern lifted to module
    scope so memo / round2 / evidence_pack writes can also use it (was only
    used for final_agenda.yaml after Chair-audit-R4#2). A mid-write crash
    leaves either the previous file or no file, never a partial.

    `payload_writer` is called with an open file handle (text mode, utf-8).
    """
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    # SIM115 ignored: this is the atomic-write pattern (temp file + fsync +
    # os.replace). `with` would close the file at block exit, but we need to
    # close *before* the rename, then handle cleanup explicitly on error.
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w",
        dir=path.parent,
        delete=False,
        suffix=path.suffix + ".tmp",
        encoding="utf-8",
    )
    try:
        payload_writer(tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        try:
            if not tmp.closed:
                tmp.close()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            os.unlink(tmp.name)
        raise


def _validation_candidate_ids_from_pack(pack_dict: dict[str, Any]) -> set[str]:
    audit = pack_dict.get("audit") if isinstance(pack_dict, dict) else {}
    audit_ids = audit.get("validation_candidate_ids") if isinstance(audit, dict) else []
    if isinstance(audit_ids, list):
        ids: set[str] = set()
        for item in audit_ids:
            ids.update(_validation_parent_tokens(item))
        if ids:
            return ids
    shared_core = pack_dict.get("shared_core") if isinstance(pack_dict, dict) else {}
    candidates = shared_core.get("validation_candidates") if isinstance(shared_core, dict) else []
    ids: set[str] = set()
    if not isinstance(candidates, list):
        return ids
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        for key in VALIDATION_PARENT_IDENTITY_KEYS:
            value = entry.get(key)
            ids.update(_validation_parent_tokens(value))
        aliases = entry.get("identity_aliases")
        if isinstance(aliases, list):
            for value in aliases:
                ids.update(_validation_parent_tokens(value))
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            for key in VALIDATION_PARENT_IDENTITY_KEYS:
                ids.update(_validation_parent_tokens(metrics.get(key)))
            aliases = metrics.get("identity_aliases")
            if isinstance(aliases, list):
                for value in aliases:
                    ids.update(_validation_parent_tokens(value))
    return {item for item in ids if item}


def _role_coverage_retry_issues(blocking_issues: list[str]) -> tuple[str, ...]:
    """Return the one validator failure class a Chair retry can safely repair."""

    issues = tuple(str(issue).strip() for issue in blocking_issues if str(issue).strip())
    if issues and all(issue.startswith("peer_contracts missing roles:") for issue in issues):
        return issues
    return ()


def _validation_candidate_ids_from_manifest(run_dir: Path, completed_gen_id: int) -> set[str]:
    ids: set[str] = set(
        _validation_candidate_aliases_from_manifest(
            Path(run_dir),
            current_gen_id=completed_gen_id,
        )
    )
    for entry in _digest_validation_candidates(
        Path(run_dir),
        max_entries=10_000,
        current_gen_id=completed_gen_id,
    ):
        if not isinstance(entry, dict):
            continue
        for key in VALIDATION_PARENT_IDENTITY_KEYS:
            ids.update(_validation_parent_tokens(entry.get(key)))
        aliases = entry.get("identity_aliases")
        if isinstance(aliases, list):
            for value in aliases:
                ids.update(_validation_parent_tokens(value))
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            for key in VALIDATION_PARENT_IDENTITY_KEYS:
                ids.update(_validation_parent_tokens(metrics.get(key)))
            aliases = metrics.get("identity_aliases")
            if isinstance(aliases, list):
                for value in aliases:
                    ids.update(_validation_parent_tokens(value))
    return {item for item in ids if item}


logger = logging.getLogger(__name__)


class PanelMode(StrEnum):
    """Supported execution modes for the legacy two-round PI panel."""

    MINI = "mini"
    FULL = "full"
    HIGH_STAKES = "high_stakes"


@dataclass
class PanelResult:
    """Terminal result of a PI panel run, including memos and generated agenda."""

    success: bool
    panel_mode: str
    agenda: dict[str, Any] = field(default_factory=dict)
    pi_memos: dict[str, Any] = field(default_factory=dict)
    cross_reviews: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    raw_chair_text: str = ""
    error: str | None = None
    duration_seconds: float = 0.0


_DEFAULT_TOPOLOGY_REF = "panel_topology:legacy_multi_pi_two_round"
"""Legacy fallback when callers don't pass an explicit topology_ref.

Custom task projects that declare ``praxist_plugins.panel.topology``
flow that ref through ``TaskSpec.panel_topology_ref`` → ``run_panel`` →
these helpers; the default is only hit by tests / single-task call sites
that haven't been updated. Keep both branches working until call-site
migration is complete.
"""


def _select_pi_roles(
    panel_mode: str,
    *,
    topology_ref: str = _DEFAULT_TOPOLOGY_REF,
    registry: Any | None = None,
) -> list[str]:
    return panel_topology_for_ref(topology_ref, registry=registry).roles_for_mode(panel_mode)


def _select_pi_role_specs(
    panel_mode: str,
    *,
    topology_ref: str = _DEFAULT_TOPOLOGY_REF,
    registry: Any | None = None,
) -> list[PanelRoleSpec]:
    return panel_topology_for_ref(topology_ref, registry=registry).role_specs_for_mode(panel_mode)


def _select_chair_role_ref(
    *,
    topology_ref: str = _DEFAULT_TOPOLOGY_REF,
    registry: Any | None = None,
) -> str:
    topology = panel_topology_for_ref(topology_ref, registry=registry)
    return topology.chair_role_ref or "task_role:chair"


def _normalize_peer_role(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.strip().lower()
    text = re.sub(r"[\s./-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _role_repair_success_signal(missing_role: str, old_signal: str) -> str:
    context = f" Original success signal for context: {old_signal}" if old_signal else ""
    if missing_role == "theorist":
        return (
            "Theorist deliverable: publish a mechanism-level hypothesis map, "
            "failure-boundary analysis, and concrete next-generation experiment "
            f"contracts for the target mechanism.{context}"
        ).strip()
    if missing_role == "bridge":
        return (
            "Bridge deliverable: connect at least two surviving mechanisms, "
            "publish a coverage matrix, and specify a minimal combined test."
            f"{context}"
        ).strip()
    if missing_role == "falsifier":
        return (
            "Falsifier deliverable: run or specify the smallest decisive control "
            "that can refute the target mechanism without changing the task contract."
            f"{context}"
        ).strip()
    if missing_role == "anti_mainline":
        return (
            "Anti-mainline deliverable: preserve a high-upside minority mechanism "
            "with a bounded test and explicit kill condition."
            f"{context}"
        ).strip()
    return (
        "Exploit deliverable: improve the target mechanism while preserving the "
        f"task contract and required evidence schema.{context}"
    ).strip()


def _ensure_bridge_hypothesis_for_repair(agenda: dict[str, Any], old_target: Any) -> str:
    bridge = agenda.get("bridge_hypothesis")
    hypotheses = [
        h
        for h in (agenda.get("cross_peer_hypotheses") or [])
        if isinstance(h, dict) and h.get("id")
    ]
    source_ids = [str(h["id"]) for h in hypotheses[:2]]
    if len(source_ids) == 1 and old_target:
        source_ids.append(str(old_target))
    while len(source_ids) < 2:
        source_ids.append("auto_repair_missing_source")

    def _anchor(source_id: str) -> dict[str, str]:
        return {
            "variant": source_id,
            "extracted_mechanism": (
                "auto-repair bridge source; inspect agenda hypotheses and "
                "prior findings before implementation"
            ),
        }

    if isinstance(bridge, dict) and bridge.get("id"):
        bridge.setdefault("source_anchor_A", _anchor(source_ids[0]))
        bridge.setdefault("source_anchor_B", _anchor(source_ids[1]))
        bridge.setdefault(
            "coverage_check",
            "Auto-completed bridge coverage metadata for repaired bridge peer.",
        )
        return str(bridge["id"])

    repaired_bridge = {
        "id": "B_auto_role_repair",
        "source_anchor_A": _anchor(source_ids[0]),
        "source_anchor_B": _anchor(source_ids[1]),
        "rationale": (
            "Auto-generated because the full/high-stakes agenda omitted a bridge "
            "peer contract while requiring canonical role coverage."
        ),
        "experiment_matrix": [
            "Compare the two source mechanisms under the same evaluation protocol.",
            "Test a minimal combined variant only if both sources remain viable.",
        ],
        "expected_pareto_movement": "Recover diversity without weakening evidence gates.",
        "coverage_check": "Document which mechanisms or lanes the bridge spans before publishing.",
    }
    agenda["bridge_hypothesis"] = repaired_bridge
    return repaired_bridge["id"]


def _sanitize_round2_peer_labels(
    parsed: dict[str, Any],
    label_map: dict[str, str],
) -> dict[str, Any]:
    """Drop invalid peer label references before Chair sees cross-review."""
    if not isinstance(parsed, dict) or not label_map:
        return parsed
    valid_labels = set(label_map)
    for label_field in (
        "strongest_agreement",
        "strongest_objection",
        "private_kb_revealed_blind_spot",
        "singleton_high_upside_idea_to_preserve",
    ):
        value = parsed.get(label_field)
        if not isinstance(value, dict):
            continue
        label = value.get("peer_label")
        if label is None:
            continue
        if str(label) not in valid_labels:
            logger.warning(
                "PI %s round2: dropped invalid peer_label %r in %s; valid labels=%s",
                parsed.get("role", "?"),
                label,
                label_field,
                sorted(valid_labels),
            )
            value["peer_label"] = None
            value["_invalid_peer_label_dropped"] = str(label)
    return parsed


def _apply_role_repair_contract_fields(
    agenda: dict[str, Any],
    contract: dict[str, Any],
    missing_role: str,
    old_target: Any,
) -> None:
    """Rewrite role-specific contract fields after a full-panel auto repair."""
    if missing_role == "bridge":
        bridge_target = _ensure_bridge_hypothesis_for_repair(agenda, old_target)
        contract["target_hypothesis"] = bridge_target
        contract["coverage_check"] = (
            "Auto-repaired bridge: compare at least two surviving mechanisms or "
            "lanes from the agenda, document overlap/gaps, and propose a combined "
            "minimal test before publishing a result."
        )
        contract["parent_candidate"] = str(old_target or bridge_target)
        contract["parent_usage"] = "compare"
        contract["source"] = "auto_repaired_bridge_contract"
        return
    if missing_role == "falsifier":
        contract["target_hypothesis"] = "falsification_contract"
        contract["required_controls"] = [
            "Run or specify the smallest control that would refute the target mechanism.",
            f"Preserve original target context for audit: {old_target}",
        ]
        contract["source"] = "auto_repaired_falsification_contract"
        return
    if missing_role == "anti_mainline":
        contract["target_hypothesis"] = "anti_mainline_contract"
        contract["forbidden_actions"] = [
            "Do not simply replicate the current dominant mechanism family."
        ]
        contract["source"] = "auto_repaired_anti_mainline_contract"
        return
    if missing_role == "theorist":
        contract["mechanism_hypothesis_deliverable"] = (
            "Map the mechanism, expected failure boundary, evidence needed, and "
            "next-generation experiment contracts."
        )
        contract.setdefault("source", "auto_repaired_theorist_contract")
        return
    contract.setdefault("source", "auto_repaired_exploit_contract")


def _ensure_full_panel_required_roles(
    agenda: dict[str, Any],
    panel_mode: str,
    cohort_size: int,
    required_peer_roles: Any = None,
) -> list[dict[str, Any]]:
    """Ensure full/high-stakes agendas contain every canonical peer role.

    Chairs sometimes over-focus the cohort and omit `theorist`, which then
    triggers v1 back-compat warnings and weakens diversity. For full panels,
    repair by converting duplicate role contracts into missing roles before
    validation. The contract keeps its hypothesis/source but gets an explicit
    role-repair annotation and a role-appropriate success signal.
    """
    if not isinstance(agenda, dict):
        return []
    agenda["_runtime_panel_mode"] = panel_mode
    required_roles = _normalize_required_peer_roles(required_peer_roles)
    if panel_mode not in {"full", "high_stakes"} or cohort_size < len(required_roles):
        return []
    if set(required_roles) != set(PI_ROLES):
        return []
    contracts = agenda.get("peer_contracts")
    if not isinstance(contracts, dict):
        return []

    role_by_pid: dict[str, str] = {}
    counts: dict[str, int] = {}
    for pid, contract in contracts.items():
        if not isinstance(contract, dict):
            continue
        role = _normalize_peer_role(contract.get("role"))
        role_by_pid[pid] = role
        counts[role] = counts.get(role, 0) + 1

    missing = [role for role in required_roles if counts.get(role, 0) <= 0]
    if not missing:
        return []

    repairs: list[dict[str, Any]] = []
    for missing_role in missing:
        candidates = [
            pid
            for pid, role in role_by_pid.items()
            if role in PI_ROLES and role != missing_role and counts.get(role, 0) > 1
        ]
        if not candidates:
            repairs.append(
                {
                    "missing_role": missing_role,
                    "status": "unrepaired",
                    "reason": "no duplicate role contract available",
                }
            )
            continue
        pid = max(candidates, key=lambda p: (counts.get(role_by_pid.get(p, ""), 0), p))
        contract = contracts[pid]
        old_role = role_by_pid[pid]
        old_target = contract.get("target_hypothesis")
        old_signal = str(contract.get("success_signal", "")).strip()
        contract["role"] = missing_role
        contract["success_signal"] = _role_repair_success_signal(missing_role, old_signal)
        _apply_role_repair_contract_fields(agenda, contract, missing_role, old_target)
        contract["_auto_role_repair"] = {
            "old_role": old_role,
            "new_role": missing_role,
            "reason": f"{panel_mode} panel must cover every canonical peer role",
            "original_target_hypothesis": old_target,
            "original_success_signal": old_signal,
            "semantic_repair_note": (
                "Role and success_signal were rewritten before validation so the "
                "full PI panel retains canonical role coverage."
            ),
        }
        counts[old_role] -= 1
        counts[missing_role] = counts.get(missing_role, 0) + 1
        role_by_pid[pid] = missing_role
        repairs.append(
            {
                "peer_id": pid,
                "old_role": old_role,
                "new_role": missing_role,
                "status": "repaired",
            }
        )

    if repairs:
        normalize_agenda_research_metadata(agenda)
        agenda.setdefault("auto_repairs", [])
        if isinstance(agenda["auto_repairs"], list):
            agenda["auto_repairs"].extend(repairs)
        logger.warning(
            "panel_runner: repaired missing full-panel peer roles: %s",
            repairs,
        )
    return repairs


def _has_high_stakes_signal(
    shared_core: dict[str, Any],
    *,
    topology_ref: str = _DEFAULT_TOPOLOGY_REF,
    registry: Any | None = None,
) -> bool:
    """Check if shared_core's CLAIM-level fields contain universal /
    dominance / obsolete language.

    Scope the substring check to claim-bearing fields only, avoiding false
    positives from findings titled
    'Why X is not obsolete' or 'rho<0.20 scale-pattern candidate'.
    """
    return panel_topology_for_ref(topology_ref, registry=registry).has_high_stakes_signal(
        shared_core
    )


async def _run_pi_parallel(
    pis: list[BasePI],
    shared_core: dict[str, Any],
    private_packs: dict[str, list[dict[str, Any]]],
    target_decisions: list[str],
) -> dict[str, Any]:
    """Run all PIs concurrently and collect their memos."""

    async def _one(pi):
        try:
            # Deep-copy per-PI so concurrent template rendering can't mutate
            # the shared state seen by sibling PIs. (R1#3 + R1#7 defense.)
            memo = await pi.run(
                shared_core=copy.deepcopy(shared_core),
                private_pack=copy.deepcopy(private_packs.get(pi.role_name, [])),
                target_decisions=list(target_decisions),
            )
            return pi.role_name, memo
        except Exception as e:
            logger.exception("PI %s panic", pi.role_name)
            return pi.role_name, {"error": str(e), "_panic": True}

    results = await asyncio.gather(*[_one(p) for p in pis], return_exceptions=False)
    return {role: m for role, m in results}


def _anonymize_peers(
    pi_memos: dict[str, Any],
    me: str,
    rng_seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Map peer roles to opaque labels (PI #A, PI #B, ...) using a
    deterministic shuffle keyed by `rng_seed`. Excludes `me`. Used in
    Round 2 to prevent role-based stereotyping.

    R1#5 fix: sort peer keys before shuffle so the input ordering is
    deterministic regardless of dict construction order.

    Returns (label_to_memo, label_to_role). The `label_to_role` mapping
    is per-PI (each PI's "PI #A" generally refers to a different role)
    and is later forwarded to Chair so Chair can de-anonymize when
    cross-referencing the Round 2 outputs.
    """
    import random

    # Sort role keys for deterministic anonymization across calls.
    peers = sorted(
        ((role, memo) for role, memo in pi_memos.items() if role != me),
        key=lambda kv: kv[0],
    )
    if not peers:
        return {}, {}
    items = list(peers)
    rng = random.Random(rng_seed)
    rng.shuffle(items)
    label_to_memo: dict[str, dict[str, Any]] = {}
    label_to_role: dict[str, str] = {}
    for i, (role, memo) in enumerate(items):
        label = f"PI #{chr(ord('A') + i)}"
        label_to_memo[label] = memo if isinstance(memo, dict) else {}
        label_to_role[label] = role
    return label_to_memo, label_to_role


def _truncate_memo_for_round2(memo: Any, max_tokens: int = 2000) -> Any:
    """Cap a peer memo's size before passing to Round 2 LLM input.

    R1#4 fix: an unbounded Round 1 memo could push Round 2 prompt over
    token budget. Keep the structurally important fields (top_claims,
    objections_or_warnings, claim_boundaries, private_knowledge_used)
    and truncate verbose long-strings.
    """
    if not isinstance(memo, dict):
        # Multi-round-audit-R8 fix: returning a non-dict (e.g. None) here
        # would propagate through to the round2_cross_review.jinja2 template
        # which calls memo.get("_pi_unavailable") on each anon_peer. Coerce
        # to an empty dict so the template iterates safely.
        return {}
    # R2#2 fix: include proposed_experiments — it's part of Round 1
    # schema and other PIs need to see what experiments were proposed.
    keep_keys = (
        "role",
        "top_claims",
        "objections_or_warnings",
        "proposed_experiments",
        "proposed_peer_contracts",
        "claim_boundaries",
        "private_knowledge_used",
        "private_knowledge_contribution",
        "confidence_scores",
        "_pi_unavailable",
        "error_summary",
    )
    out = {k: memo[k] for k in keep_keys if k in memo}
    # Coarse char-count proxy for tokens (4 chars/token).
    target_chars = max_tokens * 4
    blob = json.dumps(out, default=str)
    if len(blob) <= target_chars:
        return out
    # Iterative truncation: trim long strings inside nested structures.
    # R3#4 fix: NEVER truncate identifier-like fields (id, claim_id,
    # boundary, target_hypothesis, finding_id, evidence_id) so peer
    # references and own_revisions stay resolvable.
    PRESERVE_VERBATIM = frozenset(
        {
            "id",
            "claim_id",
            "target_hypothesis",
            "boundary",
            "finding_id",
            "evidence_id",
            "peer_id",
            "role",
        }
    )
    # R7#4 fix: preserve full lists of identifier-bearing items so a peer
    # PI doesn't reference a claim_id that Round 2 truncation hid (would
    # be silently dropped by the own_revisions whitelist check).
    PRESERVE_LIST_FULL = frozenset(
        {
            "top_claims",
            "claim_boundaries",
            "objections_or_warnings",
        }
    )

    def _shrink(v: Any, parent_key: str = "") -> Any:
        if isinstance(v, str):
            if parent_key in PRESERVE_VERBATIM:
                return v  # don't touch identifiers
            if len(v) > 300:
                return v[:300] + "...[truncated for round2]"
            return v
        if isinstance(v, list):
            limit = len(v) if parent_key in PRESERVE_LIST_FULL else 8
            return [_shrink(x, parent_key) for x in v[:limit]]
        if isinstance(v, dict):
            return {k: _shrink(x, k) for k, x in v.items()}
        return v

    out = {k: _shrink(v, k) for k, v in out.items()}
    return out


async def _run_round2_parallel(
    pis: list[BasePI],
    pi_memos: dict[str, Any],
    round2_max_runtime_minutes: int | None,
    rng_seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Round 2 cross-review: each PI sees anonymized peers and answers 6 Qs.

    Returns (cross_reviews, per_pi_label_maps):
      cross_reviews: {role -> parsed cross_review dict}
      per_pi_label_maps: {role -> {label -> peer_role}}
    Each PI gets its own label map (different shuffle excluding self).
    Chair receives both so it can de-anonymize peer references.
    Failed/skipped PIs get a stub indicating round2_failed.
    """
    # Outer-scope: collect per-PI label→role maps so Chair can de-anonymize.
    # Pre-initialize empty slots so a panic in _anonymize_peers() can't leave
    # the dict missing a PI's entry (Chair's de-anonymization would KeyError).
    per_pi_label_maps: dict[str, dict[str, str]] = {p.role_name: {} for p in pis}

    async def _one(pi):
        anon_memos, label_map = _anonymize_peers(pi_memos, pi.role_name, rng_seed)
        available = {
            label: memo
            for label, memo in anon_memos.items()
            if not (isinstance(memo, dict) and memo.get("_pi_unavailable"))
        }
        if len(available) != len(anon_memos):
            label_map = {label: role for label, role in label_map.items() if label in available}
            anon_memos = available
        # R1#4: cap each peer memo's size before LLM input
        anon_memos = {label: _truncate_memo_for_round2(memo) for label, memo in anon_memos.items()}
        per_pi_label_maps[pi.role_name] = label_map
        own = pi_memos.get(pi.role_name) or {}
        # Skip Round 2 for PIs that already had _pi_unavailable in Round 1
        if isinstance(own, dict) and own.get("_pi_unavailable"):
            return pi.role_name, {
                "_round2_skipped": True,
                "reason": "round1_unavailable",
            }
        # R1#4: also truncate own memo to stay within Round 2 budget
        own_for_r2 = _truncate_memo_for_round2(own, max_tokens=3000)
        try:
            memo = await pi.run_cross_review(
                own_memo=own_for_r2,
                anon_peers=anon_memos,
                round2_max_runtime_minutes=round2_max_runtime_minutes,
            )
            return pi.role_name, memo
        except Exception as e:
            logger.exception("PI %s: round2 panic", pi.role_name)
            return pi.role_name, {"_round2_failed": True, "error": str(e)}

    results = await asyncio.gather(*[_one(p) for p in pis], return_exceptions=False)
    out: dict[str, Any] = {}
    for role, r in results:
        if hasattr(r, "parsed"):
            parsed = r.parsed or {}
            if isinstance(parsed, dict) and (
                parsed.get("_parse_error")
                or parsed.get("_round2_failed")
                or parsed.get("_round2_skipped")
                or parsed.get("_no_peers")  # R3#1: only-PI case
            ):
                out[role] = {
                    "_round2_unavailable": True,
                    "role": role,
                    "error_summary": parsed.get(
                        "error", parsed.get("reason", "round2 unavailable")
                    ),
                    "_no_peers": bool(parsed.get("_no_peers")),
                }
            else:
                # R2#1 fix: validate own_revisions reference real claim_ids
                # from this PI's Round 1 memo. Drop hallucinated entries
                # (LLM may invent ids it can't see in truncated context).
                own_round1 = pi_memos.get(role) or {}
                round1_claim_ids = set()
                if isinstance(own_round1, dict):
                    for c in own_round1.get("top_claims") or []:
                        if isinstance(c, dict) and c.get("id"):
                            round1_claim_ids.add(str(c["id"]))
                revisions = parsed.get("own_revisions") or []
                if isinstance(revisions, list) and round1_claim_ids:
                    valid = []
                    dropped = []
                    for rev in revisions:
                        if isinstance(rev, dict):
                            cid = str(rev.get("claim_id", ""))
                            if cid and cid in round1_claim_ids:
                                valid.append(rev)
                            else:
                                dropped.append(cid)
                    if dropped:
                        logger.warning(
                            "PI %s round 2: dropped %d own_revisions with "
                            "unknown claim_ids %r (round1 had %r)",
                            role,
                            len(dropped),
                            dropped,
                            sorted(round1_claim_ids),
                        )
                    parsed["own_revisions"] = valid
                out[role] = _sanitize_round2_peer_labels(
                    parsed,
                    per_pi_label_maps.get(role, {}),
                )
        else:
            out[role] = r  # error dict
    return out, per_pi_label_maps


def _shared_core_for_panel(
    shared_core: dict[str, Any],
    quality_diversity_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a prompt view with optional QD config, leaving evidence unchanged."""

    if not isinstance(quality_diversity_policy, dict) or not quality_diversity_policy.get(
        "enabled"
    ):
        return shared_core
    prompt_core = copy.deepcopy(shared_core)
    prompt_core["quality_diversity_policy"] = copy.deepcopy(quality_diversity_policy)
    return prompt_core


async def run_panel(
    run_dir: Path,
    workspace: Path,
    model: str,
    completed_gen_id: int,
    panel_mode: str = "full",
    cohort_size: int = 5,
    target_decisions: list[str] | None = None,
    findings_summary: dict[str, Any] | None = None,
    mcp_servers: dict[str, Any] | None = None,
    pi_max_runtime_minutes: int | None = 12,
    chair_max_runtime_minutes: int | None = 8,
    stop_check_fn: Callable[[], bool] | None = None,
    auto_escalate_to_high_stakes: bool = True,
    n_rounds: int = 2,
    round2_max_runtime_minutes: int | None = 6,
    # 2026-05-07: bound pair with 1M beta — activates adaptive thinking +
    # max effort on all PI (Round 1 + Round 2) + Chair BaseAgent calls.
    premium_mode: bool = False,
    # Issue #75 batch 3: threaded through to BasePI so task_role refs
    # resolve without a PRAXIST_TASK_PROJECT_PATH env read in role_skills.
    task_project_path: Path | None = None,
    # Panel topology to instantiate. ``TaskSpec.panel_topology_ref`` is
    # populated from the task descriptor's
    # ``praxist_plugins.panel.topology`` field; custom projects can
    # override the legacy three-PI shape this way (e.g. the four-PI
    # ``rocket_8e_panel`` with its interpreter role). Default preserves
    # the legacy single-task behaviour for callers that haven't migrated.
    topology_ref: str = _DEFAULT_TOPOLOGY_REF,
    # Issue #151: when the orchestrator has assembled a registry that
    # already includes the task project's plugin roots (via
    # ``--task-path`` / ``PluginRoots.defaults(task_path=...)``), pass
    # it through so ``panel_topology_for_ref`` can find a task-level
    # ``praxist_plugins.panel.topology`` plugin. Without this, the
    # resolver fell back to ``PluginLoader(PluginRoots.defaults(workspace))``
    # whose ``bundled`` root only covers the Praxist repo's own plugins, so
    # custom topologies were silently invisible.
    registry: Any | None = None,
    # Optional prompt-only policy for selecting the next generation from the
    # existing PI proposal pool. It may add advisory contract plan metadata,
    # but does not alter evidence or create another artifact.
    quality_diversity_policy: dict[str, Any] | None = None,
    reasoning_effort: str = "max",
) -> PanelResult:
    """Run the full multi-PI panel and return the final agenda."""
    run_dir = Path(run_dir)
    workspace = Path(workspace)
    next_gen_id = completed_gen_id + 1
    target_decisions = target_decisions or [
        f"allocate gen{next_gen_id} peer contracts",
        "validate or downgrade strongest active claim",
        "decide retirement boundaries for dominated mechanisms",
    ]
    started = datetime.now(UTC)

    # Resolve the topology spec once so its plugin-supplied ``prompts_dir``
    # (if any) can be threaded through to PI roles and the Chair. The
    # ``_select_*`` helpers below all key off the same ``topology_ref`` so
    # role selection, role-spec lookup, and high-stakes signalling stay
    # consistent within a single run_panel invocation.
    _topology_spec = panel_topology_for_ref(topology_ref, registry=registry)
    topology_prompts_dir = _topology_spec.prompts_dir

    # ---------- Round 0: Build evidence pack ----------
    pi_roles = _select_pi_roles(panel_mode, topology_ref=topology_ref, registry=registry)
    pack = build_evidence_pack(
        run_dir=run_dir,
        panel_mode=panel_mode,
        current_gen_id=completed_gen_id,
        target_decisions=target_decisions,
        pi_roles=pi_roles,
        max_cards_total=BUDGETS[panel_mode].max_cards * 2,
        max_cards_per_pack=BUDGETS[panel_mode].max_cards,
        findings_summary=findings_summary,
    )

    # auto-escalate if high-stakes signal detected in pack
    if (
        auto_escalate_to_high_stakes
        and panel_mode != "high_stakes"
        and _has_high_stakes_signal(pack.shared_core, topology_ref=topology_ref, registry=registry)
    ):
        logger.info(
            "panel_runner: high-stakes signal detected in shared_core; "
            "escalating panel_mode %s -> high_stakes",
            panel_mode,
        )
        panel_mode = "high_stakes"
        pi_roles = _select_pi_roles(panel_mode, topology_ref=topology_ref, registry=registry)
        pack = build_evidence_pack(
            run_dir=run_dir,
            panel_mode=panel_mode,
            current_gen_id=completed_gen_id,
            target_decisions=target_decisions,
            pi_roles=pi_roles,
            max_cards_total=BUDGETS[panel_mode].max_cards * 2,
            max_cards_per_pack=BUDGETS[panel_mode].max_cards,
            findings_summary=findings_summary,
        )

    # Deep-copy after fitting so each concurrent PI gets its own dict tree
    # and a Jinja filter / downstream code mutating one PI's view cannot
    # corrupt another PI's view. (R1#3 + R1#7 fix.)
    pack_dict = copy.deepcopy(fit_pack_to_budget(pack, panel_mode))
    shared_core = pack_dict["shared_core"]
    private_packs = pack_dict["private_packs"]
    # QD policy is run configuration, not evidence. Add it only to the bounded
    # prompt copy; do not duplicate it into evidence_pack.json or alter the
    # evidence-derived shared_core_id.
    panel_shared_core = _shared_core_for_panel(shared_core, quality_diversity_policy)

    # Persist pack for post-mortem
    out_dir = run_dir / "research_memory" / f"synth_gen{completed_gen_id}_to_{next_gen_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # R8#5 fix: pre-sanitize via json round-trip so non-JSON-native
    # types (datetime, np scalars) become strings explicitly rather
    # than relying on default=str alone.
    try:
        sanitized_pack = json.loads(json.dumps(pack_dict, default=str))
        # Adv-audit-R7 fix: atomic write.
        _atomic_write_text(
            out_dir / "evidence_pack.json",
            lambda f: json.dump(sanitized_pack, f, indent=2, ensure_ascii=False),
        )
    except Exception as e:
        logger.warning("panel_runner: persist pack failed: %s", e)

    # ---------- Round 1: Independent memos (parallel) ----------
    pi_role_specs = _select_pi_role_specs(panel_mode, topology_ref=topology_ref, registry=registry)
    pis: list[BasePI] = instantiate_pi_roles(
        pi_role_specs,
        run_dir=run_dir,
        workspace=workspace,
        model=model,
        max_runtime_minutes=pi_max_runtime_minutes,
        mcp_servers=mcp_servers,
        stop_check_fn=stop_check_fn,
        premium_mode=premium_mode,
        reasoning_effort=reasoning_effort,
        prompts_dir=topology_prompts_dir,
        task_project_path=task_project_path,
        plugin_registry=registry,
    )
    expected_roles = {role.legacy_role_id for role in pi_role_specs}
    missing_roles = sorted(expected_roles - {pi.role_name for pi in pis})
    for role in missing_roles:
        logger.warning("legacy_two_round_executor: unknown role %s; skipping", role)

    memo_results = await _run_pi_parallel(
        pis,
        panel_shared_core,
        private_packs,
        target_decisions,
    )

    # Convert PIMemo objects to dicts (for chair input).
    # R4#2 fix: when a PI failed (panicked, timed out, or produced
    # unparseable YAML), substitute a brief error stub so the chair
    # doesn't see a half-broken memo and try to extract claims from it.
    pi_memos = {}
    for role, memo in memo_results.items():
        if hasattr(memo, "parsed"):
            parsed = memo.parsed or {}
            # _parse_error / _failed sentinels: replace with explicit "PI unavailable"
            if isinstance(parsed, dict) and (parsed.get("_parse_error") or parsed.get("_failed")):
                pi_memos[role] = {
                    "_pi_unavailable": True,
                    "role": role,
                    "error_summary": parsed.get("error", "memo not parseable"),
                    "top_claims": [],
                    "objections_or_warnings": [],
                    "proposed_experiments": [],
                    "proposed_peer_contracts": [],
                    "private_knowledge_used": [],
                }
            else:
                pi_memos[role] = parsed
        elif isinstance(memo, dict) and memo.get("_panic"):
            # _one() returned a dict for panicked PI
            pi_memos[role] = {
                "_pi_unavailable": True,
                "role": role,
                "error_summary": memo.get("error", "PI panic"),
                "top_claims": [],
                "objections_or_warnings": [],
                "proposed_experiments": [],
                "proposed_peer_contracts": [],
                "private_knowledge_used": [],
            }
        else:
            pi_memos[role] = memo  # other error dict

    # Persist memos. R6#6 fix: serialize through json+reload so any
    # non-YAML-native types in the parsed memo (datetime, numpy scalars,
    # custom classes) become strings/floats before yaml.safe_dump tries
    # to emit them. Using default=str on the json side covers the
    # remaining edge cases.
    def _yaml_safe_obj(o: Any) -> Any:
        try:
            return json.loads(json.dumps(o, default=str))
        except Exception:
            return repr(o)

    try:
        import yaml

        for role, m in pi_memos.items():
            # Adv-audit-R7 fix: atomic write so a crash mid-write leaves
            # either the previous (or no) file, never a truncated YAML
            # that would later poison post-mortem analysis.
            _atomic_write_text(
                out_dir / f"memo_{role}.yaml",
                lambda f, _m=m: yaml.safe_dump(
                    _yaml_safe_obj(_m), f, sort_keys=False, allow_unicode=True
                ),
            )
    except Exception as e:
        logger.warning("panel_runner: persist memos failed: %s", e)

    # ---------- Round 2: cross-review ----------
    # When n_rounds >= 2, each PI gets a NEW LLM call: read anonymized
    # peer memos, answer 6 fixed questions, optionally revise own
    # confidence/boundary. When n_rounds == 1, fall back to the v1
    # structured-derive behavior.
    # R3#5 defense: clamp n_rounds at run-time too (config validation
    # may be bypassed by direct API callers).
    if n_rounds not in (1, 2):
        logger.warning(
            "run_panel: n_rounds=%d invalid; clamping to 2",
            n_rounds,
        )
        n_rounds = 2

    cross_reviews: dict[str, Any] = {}
    # Adv-R1.7 fix: pre-populate per_pi_label_maps with empty entries for
    # all PIs in the panel. If Round 2 is skipped (e.g., stop signal between
    # R1 and R2, or n_rounds=1), this still gives Chair a dict with the
    # right shape (role → {}) instead of a fully empty {} that breaks
    # de-anonymization template iteration.
    per_pi_label_maps: dict[str, dict[str, str]] = {p.role_name: {} for p in pis}
    # R5#9 fix: if a stop signal arrives between Round 1 and Round 2,
    # skip Round 2 instead of starting fresh LLM calls that will be
    # cancelled mid-flight (waste of budget + risk of partial outputs).
    if n_rounds >= 2 and pis and stop_check_fn and stop_check_fn():
        logger.info(
            "panel_runner: stop signal between Round 1 and Round 2; skipping Round 2 cross-review",
        )
        n_rounds = 1
    if n_rounds >= 2 and pis:
        # R2#6 fix: high_stakes mode has 4 PIs → each Round 2 sees
        # 3 peer memos plus own memo. Bump the per-PI Round 2 budget
        # by 50% so we have headroom for slower API responses + parsing.
        effective_r2_max = round2_max_runtime_minutes
        if (
            panel_mode == "high_stakes"
            and shared_core.get("research_loop_mode") != "scoreless"
            and effective_r2_max is not None
            and effective_r2_max < 9
        ):
            logger.info(
                "panel_runner: high_stakes mode auto-bumps Round 2 "
                "budget %d → 9 min for headroom (4 peers vs 2 in full)",
                effective_r2_max,
            )
            effective_r2_max = 9
        # Use shared_core_id as RNG seed so the anonymization is
        # deterministic for a given pack (debuggable) but rotates
        # across packs.
        # R1#7 defense: validate shared_core_id is hex before int(... 16).
        sc_id = str(shared_core.get("shared_core_id", "0"))
        if len(sc_id) >= 8 and all(c in "0123456789abcdefABCDEF" for c in sc_id[:8]):
            try:
                rng_seed = int(sc_id[:8], 16)
            except (ValueError, TypeError):
                rng_seed = 0
        else:
            logger.warning(
                "panel_runner: shared_core_id=%r not hex-prefixed; "
                "using rng_seed=0 (deterministic but not unique)",
                sc_id,
            )
            rng_seed = 0
        cross_reviews, per_pi_label_maps = await _run_round2_parallel(
            pis,
            pi_memos,
            effective_r2_max,
            rng_seed,
        )
        # Persist Round 2 outputs alongside Round 1 memos.
        # Adv-audit-R7 fix: atomic.
        try:
            import yaml as _yaml

            for role, cr in cross_reviews.items():
                _atomic_write_text(
                    out_dir / f"round2_{role}.yaml",
                    lambda f, _cr=cr: _yaml.safe_dump(
                        json.loads(json.dumps(_cr, default=str)),
                        f,
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                )
            _atomic_write_text(
                out_dir / "round2_label_maps.yaml",
                lambda f: _yaml.safe_dump(
                    per_pi_label_maps, f, sort_keys=False, allow_unicode=True
                ),
            )
        except Exception as e:
            logger.warning("panel_runner: persist round2 outputs failed: %s", e)
    else:
        # Fallback: v1 structured derive (no new LLM call).
        # Chair-audit-R8#1 fix: skip _pi_unavailable PIs so the n_rounds=1
        # path doesn't fabricate empty cross_reviews entries for PIs whose
        # Round 1 failed (mirrors the n_rounds=2 _round2_unavailable filter).
        for role, memo in pi_memos.items():
            if not isinstance(memo, dict) or memo.get("_pi_unavailable"):
                continue
            cross_reviews[role] = {
                "objections": memo.get("objections_or_warnings", []),
                "claim_boundaries": memo.get("claim_boundaries", []),
                "private_kb_used": memo.get("private_knowledge_used", []),
            }

    # ---------- Round 2.5: confidence revision (high-stakes only) ----------
    confidence_revisions: dict[str, Any] = {}
    if panel_mode == "high_stakes":
        # Look for blocking objections from skeptic / external_validity
        for role in ("skeptic", "external_validity"):
            memo = pi_memos.get(role) or {}
            for obj in memo.get("objections_or_warnings") or []:
                if isinstance(obj, dict) and obj.get("severity") == "blocking":
                    target_claim = obj.get("target_claim")
                    if target_claim:
                        confidence_revisions.setdefault(target_claim, []).append(
                            {
                                "by": role,
                                "objection": obj.get("objection"),
                                "resolving_experiment": obj.get("resolving_experiment"),
                            }
                        )

    # ---------- Chair synthesis ----------
    # R4#2 fix: filter out _round2_unavailable stubs so Chair's
    # `{% if cross_reviews %}` doesn't enter the Round-2 path with a
    # dict-of-stubs (which would invite the LLM to hallucinate
    # own_revisions / singleton ideas from empty entries).
    cross_reviews_for_chair = {
        role: cr
        for role, cr in cross_reviews.items()
        if not (isinstance(cr, dict) and cr.get("_round2_unavailable"))
    }
    # R8#1 fix: log when filtering empties the dict so post-mortem can
    # tell "Round 2 not run" from "Round 2 ran but every PI failed".
    if cross_reviews and not cross_reviews_for_chair:
        logger.warning(
            "panel_runner: all %d Round 2 outputs were _round2_unavailable; "
            "Chair will synthesize from Round 1 memos only "
            "(persisted round2_*.yaml stubs retained for audit)",
            len(cross_reviews),
        )
    chair = ChairArbiter(
        run_dir=run_dir,
        workspace=workspace,
        model=model,
        max_runtime_minutes=chair_max_runtime_minutes,
        peer_budget=cohort_size,
        mcp_servers=mcp_servers,
        stop_check_fn=stop_check_fn,
        premium_mode=premium_mode,
        reasoning_effort=reasoning_effort,
        prompts_dir=topology_prompts_dir,
        peer_role_rotation=_topology_spec.peer_role_rotation,
        role_ref=_topology_spec.chair_role_ref or "task_role:chair",
        task_project_path=task_project_path,
        plugin_registry=registry,
    )
    chair_deadline = (
        asyncio.get_running_loop().time() + max(0.0, float(chair_max_runtime_minutes) * 60.0)
        if chair_max_runtime_minutes is not None
        else None
    )
    chair_result = await chair.run(
        shared_core_digest=panel_shared_core,
        pi_memos=pi_memos,
        cross_reviews=cross_reviews_for_chair,
        confidence_revisions=confidence_revisions,
        next_gen_id=next_gen_id,
        completed_gen_id=completed_gen_id,
        panel_mode=panel_mode,
        shared_core_id=shared_core.get("shared_core_id", "?"),
        per_pi_label_maps=per_pi_label_maps,
    )

    if not chair_result.success:
        logger.error("panel_runner: chair failed: %s", chair_result.error)
        with contextlib.suppress(Exception):
            _atomic_write_text(
                out_dir / "chair_raw_failed.txt",
                lambda f: f.write(getattr(chair_result, "raw_text", "") or ""),
            )
            _atomic_write_text(
                out_dir / "chair_error.txt",
                lambda f: f.write(str(chair_result.error or "unknown chair failure")),
            )
        # Multi-round-audit-R5 fix: include raw_chair_text on failure so
        # post-mortem can read whatever Chair produced (often the raw text
        # contains a parseable-but-malformed agenda that was rejected at
        # the dict-isinstance check). Earlier this field was only set on
        # the success path, leaving operators no way to inspect a failed
        # Chair's actual output.
        return PanelResult(
            success=False,
            panel_mode=panel_mode,
            agenda={},
            pi_memos=pi_memos,
            cross_reviews=cross_reviews,
            raw_chair_text=getattr(chair_result, "raw_text", "") or "",
            error=f"chair: {chair_result.error}",
            duration_seconds=(datetime.now(UTC) - started).total_seconds(),
        )

    validation_candidate_ids = _validation_candidate_ids_from_manifest(
        run_dir, completed_gen_id
    ) or _validation_candidate_ids_from_pack(pack_dict)
    diversity_dimensions = (
        quality_diversity_policy.get("diversity_dimensions")
        if isinstance(quality_diversity_policy, dict) and quality_diversity_policy.get("enabled")
        else None
    )

    def _normalize_and_validate_chair_agenda(result):
        _ensure_full_panel_required_roles(
            result.agenda,
            panel_mode=panel_mode,
            cohort_size=cohort_size,
            required_peer_roles=_topology_spec.peer_role_rotation,
        )
        normalize_agenda_research_metadata(result.agenda)
        validation_parent_repairs = normalize_validation_candidate_parent_usages(
            result.agenda,
            validation_candidate_ids=validation_candidate_ids,
        )
        if validation_parent_repairs:
            result.agenda.setdefault("auto_repairs", []).extend(validation_parent_repairs)
        # R1#11: pass pi_memos so validator can detect Chair fabrications
        # (target_hypothesis not in any PI memo or agenda hypothesis).
        return validate_agenda_v2(
            result.agenda,
            next_gen_id=next_gen_id,
            cohort_size=cohort_size,
            require_high_stakes_external=(panel_mode == "high_stakes"),
            pi_memos=pi_memos,
            validation_candidate_ids=validation_candidate_ids,
            required_peer_roles=_topology_spec.peer_role_rotation,
            diversity_dimensions=diversity_dimensions,
        )

    # ---------- Validate ----------
    validation = _normalize_and_validate_chair_agenda(chair_result)
    retry_issues = _role_coverage_retry_issues(validation.blocking_issues)
    if retry_issues:
        logger.warning(
            "panel_runner: Chair agenda omitted required peer roles; issuing one correction "
            "within the original Chair time budget: %s",
            retry_issues,
        )
        original_result = chair_result
        with contextlib.suppress(Exception):
            _atomic_write_text(
                out_dir / "chair_raw_failed.txt",
                lambda f: f.write(original_result.raw_text),
            )
            _atomic_write_text(
                out_dir / "chair_error.txt",
                lambda f: f.write("machine validation failed: " + "; ".join(retry_issues)),
            )
        remaining_seconds = (
            chair_deadline - asyncio.get_running_loop().time()
            if chair_deadline is not None
            else None
        )
        retry_result = None
        if remaining_seconds is None or remaining_seconds > 0:
            try:
                retry_result = await asyncio.wait_for(
                    chair.run(
                        shared_core_digest=panel_shared_core,
                        pi_memos=pi_memos,
                        cross_reviews=cross_reviews_for_chair,
                        confidence_revisions=confidence_revisions,
                        next_gen_id=next_gen_id,
                        completed_gen_id=completed_gen_id,
                        panel_mode=panel_mode,
                        shared_core_id=shared_core.get("shared_core_id", "?"),
                        per_pi_label_maps=per_pi_label_maps,
                        validation_feedback=retry_issues,
                        validation_candidate=copy.deepcopy(original_result.agenda),
                    ),
                    timeout=remaining_seconds,
                )
            except TimeoutError:
                logger.warning("panel_runner: Chair role-coverage correction exhausted its budget")
        if retry_result is not None and retry_result.success:
            chair_result = retry_result
            validation = _normalize_and_validate_chair_agenda(chair_result)
        elif retry_result is not None:
            logger.warning(
                "panel_runner: bounded Chair validation retry failed; retaining the "
                "original rejected candidate for audit: %s",
                retry_result.error,
            )

    # Persist the final Chair attempt (Adv-audit-R7 fix: atomic).
    with contextlib.suppress(Exception):
        _atomic_write_text(
            out_dir / "chair_raw.txt",
            lambda f: f.write(chair_result.raw_text),
        )
        if chair_result.error:
            _atomic_write_text(
                out_dir / "chair_recovery.json",
                lambda f: json.dump(
                    {
                        "recovered": True,
                        "reason": chair_result.error,
                        "next_gen_id": next_gen_id,
                        "completed_gen_id": completed_gen_id,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                ),
            )

    # ---------- Audit ----------
    audit_report = audit_agenda(
        agenda=chair_result.agenda,
        pack_dict=pack_dict,
        pi_memos=pi_memos,
        audit_id=f"audit_gen{completed_gen_id}_to_{next_gen_id}",
        completed_gen_id=completed_gen_id,  # R5#6 fix
        expected_peer_contract_count=cohort_size,
    )

    # Persist agenda + audit.
    # R4#1 defense: write the agenda under a .candidate suffix first; only
    # rename to final_agenda.yaml if validation+audit pass. Otherwise the
    # debug copy stays as final_agenda.yaml.candidate for inspection but
    # cannot be confused with a publishable agenda.
    overall_pass = validation.valid and audit_report.pass_
    final_agenda_name = "final_agenda.yaml" if overall_pass else "final_agenda.yaml.candidate"
    agenda_snapshot = attach_artifact_semantics(
        chair_result.agenda,
        role=DERIVED_AUDIT_SNAPSHOT if overall_pass else PARTIAL_OUTPUT,
        status=COMMITTED if overall_pass else FAILED,
        stage="multi_pi_chair_agenda_snapshot",
        generation_id=next_gen_id,
        actor="research_loop:multi_pi_chair",
        derived_from=[
            "frontier/frontier_manifest.json",
            "shared_store.db",
            "research_memory/*",
            "gems/gems_state.json",
        ],
        canonical_sources=[
            "frontier/frontier_manifest.json",
            "shared_store.db",
            "research_memory/*",
            "gems/gems_state.json",
            "shared_findings/*",
            "results/*",
        ],
        runtime_fact_source=False,
        notes=(
            "Multi-PI chair agenda snapshot. The canonical agenda path may be "
            "used as the next-generation plan only after PIAgent commits it; "
            "measured evidence remains owned by canonical result/frontier state."
        ),
    )
    audit_payload = attach_artifact_semantics(
        {
            "pass": audit_report.pass_,
            "blocking_issues": audit_report.blocking_issues,
            "warnings": audit_report.warnings,
            "metrics": audit_report.metrics,
        },
        role=DERIVED_AUDIT_SNAPSHOT,
        status=COMMITTED if overall_pass else FAILED,
        stage="multi_pi_agenda_audit",
        generation_id=next_gen_id,
        actor="research_loop:multi_pi_auditor",
        derived_from=["research_memory/*", "frontier/frontier_manifest.json"],
        runtime_fact_source=False,
    )
    validation_payload = attach_artifact_semantics(
        {
            "valid": validation.valid,
            "blocking_issues": validation.blocking_issues,
            "warnings": validation.warnings,
        },
        role=DERIVED_AUDIT_SNAPSHOT,
        status=COMMITTED if validation.valid else FAILED,
        stage="multi_pi_agenda_validation",
        generation_id=next_gen_id,
        actor="research_loop:multi_pi_validator",
        derived_from=["research_memory/*", "frontier/frontier_manifest.json"],
        runtime_fact_source=False,
    )
    # Chair-audit-R4#2 fix: atomic write via temp file + os.replace.
    # The inner _atomic_write was hoisted to module-level _atomic_write_text
    # (Adv-audit-R7) so memo / round2 / evidence_pack writes also benefit.
    _atomic_write = _atomic_write_text
    try:
        import yaml

        _atomic_write(
            out_dir / final_agenda_name,
            lambda f: yaml.safe_dump(agenda_snapshot, f, sort_keys=False, allow_unicode=True),
        )
        _atomic_write(
            out_dir / "audit.json",
            lambda f: json.dump(audit_payload, f, indent=2, default=str, ensure_ascii=False),
        )
        _atomic_write(
            out_dir / "validation.json",
            lambda f: json.dump(validation_payload, f, indent=2, default=str, ensure_ascii=False),
        )
    except Exception as e:
        logger.warning("panel_runner: persist final outputs failed: %s", e)

    # ---------- Metrics ----------
    pack_size = len(json.dumps(pack_dict, default=str))
    prompt_size = sum(len(json.dumps(m, default=str)) for m in pi_memos.values())
    log_synthesis_metrics(
        run_dir=run_dir,
        generation_id=completed_gen_id,
        prompt_size_bytes=prompt_size,
        pack_size_bytes=pack_size,
        n_evidence_cards=len(pack.all_cards),
        citation_coverage=audit_report.metrics.get("citation_coverage", 0.0),
        negative_evidence_ratio=pack.audit.get("negative_evidence_ratio_global", 0.0),
        panel_mode=panel_mode,
        pi_count=len(pis),
        audit_warnings=len(audit_report.warnings) + len(validation.warnings),
        audit_blocking=len(audit_report.blocking_issues) + len(validation.blocking_issues),
        extra={"shared_core_id": shared_core.get("shared_core_id", "?")},
    )

    duration = (datetime.now(UTC) - started).total_seconds()
    return PanelResult(
        success=validation.valid and audit_report.pass_,
        panel_mode=panel_mode,
        agenda=chair_result.agenda,
        pi_memos=pi_memos,
        cross_reviews=cross_reviews,
        audit={
            "pass": audit_report.pass_,
            "blocking_issues": audit_report.blocking_issues,
            "warnings": audit_report.warnings,
            "metrics": audit_report.metrics,
        },
        validation={
            "valid": validation.valid,
            "blocking_issues": validation.blocking_issues,
            "warnings": validation.warnings,
        },
        raw_chair_text=chair_result.raw_text,
        duration_seconds=duration,
    )
