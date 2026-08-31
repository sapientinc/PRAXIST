"""Orchestrator status snapshot composition for the research-loop stage."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    durable_promotion_exclusion,
    evidence_maturity_snapshot,
    result_artifact_key,
)
from praxist.plugins.workflow_stages.research_loop.backend.orchestrator_status import (
    OrchestratorSnapshot,
    describe_promotion_blocker,
    describe_promotion_criteria,
    operator_manifest_paths,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import is_scoreless

logger = logging.getLogger(__name__)

_NON_PROMOTABLE_AUDIT_TAGS = frozenset(
    {
        "suspect",
        "suspect_protocol",
        "protocol_integrity_failed",
    }
)


@dataclass
class _ScorelessOrchestratorSnapshot(OrchestratorSnapshot):
    """Keep existing snapshot consumers compatible while declaring selection absent."""

    research_mode: str = "scoreless"
    selection_status: str = "disabled"
    evaluation_status: str = "not_configured"
    retention_policy: str = "all_findings"
    per_generation_hours: float | None = None


def _resource_scheduler_status(run_dir: Path) -> dict[str, Any]:
    try:
        scheduler_raw = json.loads(
            (Path(run_dir) / "resource_scheduler" / "status.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(scheduler_raw, dict):
        return {}
    return {
        key: scheduler_raw.get(key)
        for key in (
            "mode",
            "queued",
            "running",
            "completed",
            "failed",
            "rejected",
            "concurrency_limit",
            "admission_closed",
            "frozen_generations",
            "worker_error",
            "queue_blocked_reasons",
            "accelerator_probe",
        )
    }


def _scoreless_snapshot(
    *,
    run_started_at: str | None,
    run_dir: Path,
    task_spec: Any,
    current_gen: int,
    gens_completed: int,
    findings: list[dict[str, Any]],
) -> OrchestratorSnapshot:
    started = run_started_at or datetime.now(UTC).isoformat()
    stop_audit, _ = _last_boundary_control(run_dir, gens_completed)
    policy = task_spec.generation_policy
    return _ScorelessOrchestratorSnapshot(
        run_started_at=started,
        updated_at=datetime.now(UTC).isoformat(),
        run_dir=str(run_dir),
        task_id=task_spec.task_id,
        task_name=task_spec.task_name,
        current_generation=current_gen,
        max_generations=policy.max_generations,
        per_generation_hours=getattr(policy, "per_generation_hours", None),
        cohort_size=policy.cohort_size,
        strategy="scoreless",
        generations_completed=gens_completed,
        findings_total=len(findings),
        operator_manifest_paths=operator_manifest_paths(run_dir),
        logical_generation=current_gen,
        gen_promotion_criteria="Scoreless mode: retain all findings; metric selection is disabled.",
        last_stop_audit=stop_audit,
        resource_scheduler=_resource_scheduler_status(run_dir),
        wall_clock_elapsed_seconds=time.time() - datetime.fromisoformat(started).timestamp(),
    )


def _last_boundary_control(
    run_dir: Path, gens_completed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return compact control telemetry from the last completed generation."""

    if gens_completed <= 0:
        return {}, {}
    marker = Path(run_dir) / f"gen_{gens_completed - 1}" / "generation_boundary.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    if not isinstance(payload, dict):
        return {}, {}
    stop_audit = payload.get("stop_audit")
    peer_mix = payload.get("peer_mix")
    return (
        stop_audit if isinstance(stop_audit, dict) else {},
        peer_mix if isinstance(peer_mix, dict) else {},
    )


def _mature_quorum_required(task_spec: Any, stop_audit: dict[str, Any] | None = None) -> int:
    if isinstance(stop_audit, dict):
        audited = _finite_float(stop_audit.get("required_mature_result_peers"))
        if audited is not None:
            return max(0, int(audited))
    generation_policy = getattr(task_spec, "generation_policy", None)
    synthesis_trigger = getattr(task_spec, "synthesis_trigger", None)
    cohort_size = int(getattr(generation_policy, "cohort_size", 0) or 0)
    fraction = _finite_float(getattr(synthesis_trigger, "mature_quorum_fraction", 0.0) or 0.0)
    if fraction is None:
        return 0
    if cohort_size <= 0 or fraction <= 0:
        return 0
    return int(math.ceil(cohort_size * fraction))


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _compact_result_view(
    finding: dict[str, Any],
    value: float,
    *,
    primary_metric: str,
    baseline_relation: str,
    maturity: dict[str, Any],
) -> dict[str, Any]:
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    variant = (
        finding.get("variant_name")
        or metrics.get("variant_name")
        or finding.get("title")
        or finding.get("finding_id")
        or "unknown"
    )
    source_path = (
        finding.get("source_result_path")
        or metrics.get("source_result_path")
        or finding.get("result_artifact_path")
        or metrics.get("result_artifact_path")
        or ""
    )
    promotion_values = [
        source.get(key)
        for source in (finding, metrics)
        for key in ("promotion_eligible", "clean_promotion_eligible")
        if isinstance(source.get(key), bool)
    ]
    promotion_status = (
        "not_eligible"
        if False in promotion_values
        else "eligible"
        if True in promotion_values
        else "not_declared"
    )
    return {
        "variant_name": str(variant),
        "metric_name": primary_metric,
        "metric_value": value,
        "generation_id": finding.get("generation_id", metrics.get("generation_id")),
        "evidence_stage": finding.get("evidence_stage", metrics.get("evidence_stage", "")),
        "baseline_relation": baseline_relation,
        "maturity_basis": maturity.get("maturity_basis"),
        "promotion_status": promotion_status,
        "source_result_path": str(source_path),
    }


def _best_result_views(
    variant_findings: list[tuple[dict[str, Any], float]],
    *,
    primary_metric: str,
    direction: str,
    baseline_test: Callable[[Any], bool],
    maturity_policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mature: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    signals: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    consolidated: list[tuple[dict[str, Any], float]] = []
    artifact_positions: dict[tuple[str, str], int] = {}
    for finding, value in variant_findings:
        artifact = result_artifact_key(finding)
        immutable = artifact if artifact is not None and all(artifact) else None
        position = artifact_positions.get(immutable) if immutable is not None else None
        if position is None:
            if immutable is not None:
                artifact_positions[immutable] = len(consolidated)
            consolidated.append((finding, value))
            continue
        current, _current_value = consolidated[position]
        current_restricted = _result_view_is_restricted(current, maturity_policy)
        finding_restricted = _result_view_is_restricted(finding, maturity_policy)
        if finding_restricted and not current_restricted:
            consolidated[position] = (finding, value)

    for finding, value in consolidated:
        maturity = evidence_maturity_snapshot(finding, maturity_policy)
        audit_tags = {str(tag) for tag in maturity.get("audit_tags", []) if ":" not in str(tag)}
        routing_exclusion = durable_promotion_exclusion(finding)
        target = (
            mature
            if maturity.get("mature_enough") is True
            and not audit_tags.intersection(_NON_PROMOTABLE_AUDIT_TAGS)
            and routing_exclusion is None
            else signals
        )
        target.append((value, finding, maturity))

    reverse = direction == "maximize"

    def select(
        rows: list[tuple[float, dict[str, Any], dict[str, Any]]],
        *,
        validation: bool,
    ) -> dict[str, Any]:
        if not rows:
            return {}
        value, finding, maturity = sorted(rows, key=lambda item: item[0], reverse=reverse)[0]
        relation = "above_baseline" if baseline_test(value) else "not_above_baseline"
        view = _compact_result_view(
            finding,
            value,
            primary_metric=primary_metric,
            baseline_relation=relation,
            maturity=maturity,
        )
        if validation:
            blockers = [str(tag) for tag in maturity.get("audit_tags", []) if ":" not in str(tag)]
            routing_exclusion = durable_promotion_exclusion(finding)
            if routing_exclusion:
                blockers.append(routing_exclusion)
            view["validation_reason"] = ", ".join(blockers) or str(
                maturity.get("maturity_basis") or "validation required"
            )
        return view

    return select(mature, validation=False), select(signals, validation=True)


def _result_view_is_restricted(
    finding: dict[str, Any],
    maturity_policy: dict[str, Any],
) -> bool:
    maturity = evidence_maturity_snapshot(finding, maturity_policy)
    audit_tags = {str(tag) for tag in maturity.get("audit_tags", []) if ":" not in str(tag)}
    return (
        maturity.get("mature_enough") is not True
        or bool(audit_tags.intersection(_NON_PROMOTABLE_AUDIT_TAGS))
        or durable_promotion_exclusion(finding) is not None
    )


def build_orchestrator_status_snapshot(
    *,
    run_started_at: str | None,
    run_dir: Path,
    task_spec: Any,
    frontier: Any,
    current_gen: int,
    gens_completed: int,
    frontier_strategy: str,
    strategy_for_gen: Callable[[int], str],
    findings: list[dict[str, Any]],
    gems_context: dict[str, Any] | None = None,
) -> OrchestratorSnapshot:
    """Compose a thread-safe status snapshot from already-collected state."""
    if is_scoreless(task_spec):
        return _scoreless_snapshot(
            run_started_at=run_started_at,
            run_dir=run_dir,
            task_spec=task_spec,
            current_gen=current_gen,
            gens_completed=gens_completed,
            findings=findings,
        )
    gp = task_spec.generation_policy
    primary = task_spec.evaluation.primary_metric

    variant_findings: list[tuple[dict[str, Any], float]] = []
    for f in findings:
        metrics = f.get("metrics")
        if f.get("finding_type") != "result" or not isinstance(metrics, dict):
            continue
        primary_value = _finite_float(metrics.get(primary))
        if primary_value is None:
            continue
        variant_findings.append((f, primary_value))

    def _baseline_metric_value(b: Any) -> float:
        raw = getattr(b, "metric_value", None)
        if raw is None:
            raw = getattr(b, "expected_acc", 0.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    baselines = [b for b in task_spec.baselines]
    if baselines:
        if task_spec.evaluation.direction == "maximize":
            baseline_threshold = max(_baseline_metric_value(b) for b in baselines)

            def _above(v: Any) -> bool:
                return v > baseline_threshold

        else:
            baseline_threshold = min(_baseline_metric_value(b) for b in baselines)

            def _above(v: Any) -> bool:
                return v < baseline_threshold

    else:
        # No baselines defined — every variant that reports the primary
        # metric is treated as "above baseline".  The first generation
        # establishes the baseline for subsequent generations via the
        # frontier promotion path.
        logger.warning(
            "No baselines defined in task spec — all variants that "
            "report the primary metric will be treated as above "
            "baseline. Promotion will rank by %s only. "
            "To filter low-quality results, add baselines to task.yaml.",
            primary,
        )

        def _above(v: Any) -> bool:  # pragma: no cover - no baselines defined.
            return True

    above = sum(1 for _finding, value in variant_findings if _above(value))
    best_mature_result, best_validation_signal = _best_result_views(
        variant_findings,
        primary_metric=primary,
        direction=task_spec.evaluation.direction,
        baseline_test=_above,
        maturity_policy=dict(getattr(task_spec.evaluation, "maturity_policy", {}) or {}),
    )

    multi_seed = 0
    for f, _value in variant_findings:
        m = f.get("metrics", {})
        seeds_val = m.get("seeds")
        n_seeds = m.get("n_seeds") or m.get("num_seeds")
        try:
            n = 0
            if isinstance(seeds_val, (list, tuple)):
                n = len(seeds_val)
            elif n_seeds is not None:
                n = int(n_seeds)
            if n >= 3:
                multi_seed += 1
        except (TypeError, ValueError):
            pass

    frontier_count = 0
    with contextlib.suppress(Exception):
        frontier_count = len(frontier.get_summary() or [])
    gems_context = gems_context or {}
    raw_gems = gems_context.get("gems")
    gems_refs: list[dict[str, Any]] = []
    if isinstance(raw_gems, list):
        for entry in raw_gems:
            if not isinstance(entry, dict):
                continue
            gems_refs.append(
                {
                    "gem_finding_id": entry.get("gem_finding_id", ""),
                    "variant_name": entry.get("variant_name", ""),
                    "frontier_lane": entry.get("frontier_lane", ""),
                    "metric_name": entry.get("metric_name", ""),
                    "metric_value": entry.get("metric_value"),
                }
            )

    blocker = describe_promotion_blocker(
        variants_with_primary_metric=len(variant_findings),
        variants_above_baseline=above,
        promote_top_k=gp.promote_top_k,
        lane_based=bool(getattr(task_spec.evaluation, "frontier_lanes", None)),
    )

    started = run_started_at or datetime.now(UTC).isoformat()
    last_stop_audit, last_peer_mix = _last_boundary_control(run_dir, gens_completed)
    if not bool(getattr(task_spec.evaluation, "constructive_peer_mix_enabled", True)):
        last_peer_mix = {}
    resource_scheduler = _resource_scheduler_status(run_dir)
    return OrchestratorSnapshot(
        run_started_at=started,
        updated_at=datetime.now(UTC).isoformat(),
        run_dir=str(run_dir),
        task_id=task_spec.task_id,
        task_name=task_spec.task_name,
        current_generation=current_gen,
        max_generations=gp.max_generations,
        cohort_size=gp.cohort_size,
        strategy=(
            strategy_for_gen(current_gen)
            if current_gen is not None and current_gen >= 0
            else frontier_strategy
        ),
        generations_completed=gens_completed,
        variants_total=len(variant_findings),
        variants_above_baseline=above,
        variants_validated_multi_seed=multi_seed,
        findings_total=len(findings),
        frontier_candidates=frontier_count,
        best_mature_result=best_mature_result,
        best_validation_signal=best_validation_signal,
        operator_manifest_paths=operator_manifest_paths(run_dir),
        gems_cycle_index=int(gems_context.get("cycle_index", 0) or 0),
        gems_reset_count=int(gems_context.get("reset_count", 0) or 0),
        gems_count=int(gems_context.get("gems_count", 0) or 0),
        gems_refs=gems_refs,
        logical_generation=int(gems_context.get("logical_generation", current_gen) or 0),
        gen_promotion_criteria=describe_promotion_criteria(
            promote_top_k=gp.promote_top_k,
            promote_criterion=gp.promote_criterion,
            primary_metric=primary,
            direction=task_spec.evaluation.direction,
            frontier_lanes=getattr(task_spec.evaluation, "frontier_lanes", None),
        ),
        gen_promotion_blocker=blocker,
        last_stop_audit=last_stop_audit,
        last_peer_mix=last_peer_mix,
        mature_quorum_required=_mature_quorum_required(task_spec, last_stop_audit),
        resource_scheduler=resource_scheduler,
        wall_clock_elapsed_seconds=time.time()
        - (datetime.fromisoformat(started).timestamp() if started else time.time()),
    )
