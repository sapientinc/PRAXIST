"""Prompt context assembly for research-loop peers."""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    explicit_entry_generation_id,
    is_committed_runtime_fact_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    EFFECTIVE_CONFIG_METADATA_KEYS,
    has_effective_config_metadata,
)
from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
    _is_committed_frontier_entry,
    _resolved_boolish_candidate_field,
)
from praxist.plugins.workflow_stages.research_loop.backend.prompt_strategy import (
    _generate_variant_hint,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import (
    is_scoreless,
    load_scoreless_evidence,
)

logger = logging.getLogger(__name__)

_SIBLING_ROSTER_CAP = 8

_FRONTIER_ENTRY_KEYS = (
    *EFFECTIVE_CONFIG_METADATA_KEYS,
    "generation_id",
    "rank",
    "finding_id",
    "variant",
    "variant_name",
    "title",
    "metric_name",
    "metric_value",
    "frontier_lane",
    "promoted_for_lane",
    "lane_metric_name",
    "lane_metric_value",
    "source_frontier_lane",
    "evidence_stage",
    "evidence_maturity_rank",
    "frontier_entity_key",
    "scout_only",
    "scored_complete",
    "risk_violating_frontier_candidate",
    "risk_repair_required",
    "risk_violation_reason",
    "mature_enough",
    "parent_eligible",
    "maturity_basis",
    "effort_ratio",
    "coverage_ratio",
    "min_effort_ratio",
    "min_coverage_ratio",
    "promoted_for_anchor",
    "anchor_metric_value",
    "diversity_cell",
    "source_lane",
    "target_lane",
    "coverage_check",
    "mechanism_hypothesis_deliverable",
    "mechanism_family",
    "intervention_surface",
    "intent",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
)

_FRONTIER_METRIC_KEYS = (
    *EFFECTIVE_CONFIG_METADATA_KEYS,
    "mean_score",
    "score",
    "taskscore",
    "task_score",
    "test_score",
    "eval_score",
    "positive_cell_fraction",
    "n_cells",
    "n_hard_constraint_violations",
    "tier",
    "tier_reached",
    "tier_status",
    "final_status",
    "promotion_eligible",
    "clean_promotion_eligible",
    "frontier_lane",
    "strategy_family",
    "evidence_stage",
    "evidence_maturity_rank",
    "peer_role",
    "target_hypothesis",
    "bottleneck_target",
    "tradeoff_class",
    "primary_tradeoff",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
    "source_lane",
    "target_lane",
    "coverage_check",
    "mechanism_hypothesis_deliverable",
    "dig_selected_contract_path",
    "dig_expected_vs_actual_alignment",
    "mechanism_family",
    "intervention_surface",
    "intent",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
    "risk_violating_frontier_candidate",
    "risk_repair_required",
    "risk_violation_reason",
)

_FRONTIER_METRIC_PREFIXES = ("compute_overhead_",)

_VALIDATION_SIGNAL_PROMPT_CAP = 16
_STRONG_PARENT_VIEW_CAP = 8

_VALIDATION_SIGNAL_KEYS = (
    *EFFECTIVE_CONFIG_METADATA_KEYS,
    "generation_id",
    "finding_id",
    "variant_name",
    "metric_name",
    "metric_value",
    "metric_direction",
    "signal_source",
    "artifact_signal_source",
    "artifact_signal_status",
    "durability_scope",
    "evidence_stage",
    "evidence_maturity_rank",
    "result_status",
    "protocol_integrity_status",
    "scout_only",
    "scored_cell_count",
    "mature_enough",
    "maturity_basis",
    "effort_ratio",
    "coverage_ratio",
    "min_effort_ratio",
    "min_coverage_ratio",
    "frontier_entity_key",
    "submitted_frontier_lane",
    "matched_frontier_lanes",
    "signal_axis_lanes",
    "exclusion_reason",
    "recommended_next_step",
    "bottleneck_target",
    "tradeoff_class",
    "primary_tradeoff",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
    "mechanism_family",
    "intervention_surface",
    "intent",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
)

_DIG_PROVENANCE_KEYS = (
    "selected_contract_path",
    "selected_candidate_id",
    "final_selected_candidate_id",
    "local_selected_candidate_id",
    "selected_contract_source",
    "override_reason",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
    "diversity_cell",
    "canonical_labels",
    "contract_amended",
    "expected_vs_actual_alignment",
)

_AGENDA_TOP_LEVEL_KEYS = (
    "agenda_version",
    "generation",
    "synthesized_from_gen",
    "panel_mode",
    "shared_core_id",
)

_AGENDA_SIBLING_ROSTER_KEYS = (
    "role",
    "target_hypothesis",
    "success_signal",
    "source_lane",
    "target_lane",
    "coverage_check",
    "bottleneck_target",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
    "mechanism_family",
    "mechanism_family_preferences",
    "intervention_surface",
    "intervention_surface_preferences",
    "intent",
    "intent_preference",
    "diversity_cell",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
)

_AGENDA_SOURCE_SECTIONS = (
    "consensus_actions",
    "DISSENT_TO_EXPERIMENT",
    "minority_high_upside",
    "claim_boundary_updates",
)

_AGENDA_SOURCE_ID_KEYS = {
    "consensus_actions": ("action_id", "id", "claim_or_hypothesis"),
    "DISSENT_TO_EXPERIMENT": ("dissent_id", "id", "disputed_claim"),
    "minority_high_upside": ("idea_id", "id"),
    "claim_boundary_updates": ("claim_id", "id"),
}


def _frontier_entry_generation(entry: dict[str, Any]) -> int | None:
    return explicit_entry_generation_id(entry)


def _frontier_summary_up_to_generation(frontier: Any, gen_id: int) -> list[dict[str, Any]]:
    if hasattr(frontier, "get_summary_up_to_generation"):
        try:
            rows = frontier.get_summary_up_to_generation(gen_id)
            return [dict(row) for row in rows if isinstance(row, dict)]
        except Exception as exc:  # noqa: BLE001 - prompt context should degrade.
            logger.warning("frontier cutoff summary failed, falling back: %s", exc)
    rows = frontier.get_summary() if hasattr(frontier, "get_summary") else []
    cutoff = int(gen_id)
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        generation = _frontier_entry_generation(row)
        if generation is None or generation > cutoff:
            continue
        out.append(dict(row))
    return out


def _parent_frontier_summary_up_to_generation(frontier: Any, gen_id: int) -> list[dict[str, Any]]:
    if hasattr(frontier, "get_parent_summary_up_to_generation"):
        try:
            rows = frontier.get_parent_summary_up_to_generation(gen_id)
            return [dict(row) for row in rows if isinstance(row, dict)]
        except Exception as exc:  # noqa: BLE001 - prompt context should degrade.
            logger.warning("parent-eligible frontier summary failed: %s", exc)
    # Legacy frontier providers do not expose lane policy. Their summary keeps
    # the historical single-frontier semantics, where every durable entry was
    # parentable.
    return _frontier_summary_up_to_generation(frontier, gen_id)


def _compact_text(value: Any, *, max_chars: int = 600) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _compact_value(value: Any, *, max_chars: int = 600, max_items: int = 8) -> Any:
    if isinstance(value, str):
        return _compact_text(value, max_chars=max_chars)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [
            _compact_value(item, max_chars=max_chars, max_items=max_items)
            for item in value[:max_items]
        ]
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, max_chars=max_chars, max_items=max_items)
            for key, item in list(value.items())[:max_items]
        }
    return _compact_text(str(value), max_chars=max_chars)


def _task_primary_metric(task_spec: Any) -> str:
    evaluation = getattr(task_spec, "evaluation", None)
    return str(getattr(evaluation, "primary_metric", "") or "")


def _metric_name_from_axis(axis: Any) -> str:
    if isinstance(axis, dict):
        return str(axis.get("name") or axis.get("metric") or "").strip()
    if isinstance(axis, (list, tuple)) and axis:
        return str(axis[0] or "").strip()
    if isinstance(axis, str):
        return axis.strip()
    return ""


def _task_prompt_metric_names(task_spec: Any) -> set[str]:
    evaluation = getattr(task_spec, "evaluation", None)
    if evaluation is None:
        return set()

    names: set[str] = set()
    primary_metric = _task_primary_metric(task_spec)
    if primary_metric:
        names.add(primary_metric)

    aux_metrics = getattr(evaluation, "aux_metrics", None) or []
    if isinstance(aux_metrics, (list, tuple)):
        names.update(str(metric).strip() for metric in aux_metrics if str(metric).strip())

    anchor_metrics = getattr(evaluation, "anchor_metrics", None) or []
    if isinstance(anchor_metrics, (list, tuple)):
        for axis in anchor_metrics:
            name = _metric_name_from_axis(axis)
            if name:
                names.add(name)

    frontier_lanes = getattr(evaluation, "frontier_lanes", None) or []
    if isinstance(frontier_lanes, (list, tuple)):
        for lane in frontier_lanes:
            if not isinstance(lane, dict):
                continue
            for field_name in (
                "axes",
                "optional_axes",
                "require_metrics",
                "require_truthy_metrics",
                "require_falsey_metrics",
            ):
                raw_values = lane.get(field_name) or []
                if not isinstance(raw_values, (list, tuple)):
                    continue
                for axis in raw_values:
                    name = _metric_name_from_axis(axis)
                    if name:
                        names.add(name)
            for field_name in ("min_metrics", "max_metrics"):
                raw_bounds = lane.get(field_name) or {}
                if isinstance(raw_bounds, dict):
                    names.update(
                        str(metric).strip() for metric in raw_bounds if str(metric).strip()
                    )
    gems = getattr(task_spec, "gems", None)
    if gems is not None:
        for field_name in (
            "primary_metric_keys",
            "secondary_metric_keys",
            "lower_tail_metric_keys",
            "validation_metric_keys",
            "cost_metric_keys",
        ):
            raw_values = getattr(gems, field_name, None) or []
            if isinstance(raw_values, (list, tuple, set)):
                names.update(str(metric).strip() for metric in raw_values if str(metric).strip())
        derivations = getattr(gems, "result_cell_metric_derivations", None) or []
        if isinstance(derivations, (list, tuple)):
            for rule in derivations:
                if isinstance(rule, dict):
                    name = str(rule.get("name") or "").strip()
                    if name:
                        names.add(name)
        aliases = getattr(gems, "result_metric_aliases", None) or {}
        if isinstance(aliases, dict):
            for out_key, source_key in aliases.items():
                for key in (out_key, source_key):
                    text = str(key or "").strip()
                    if text:
                        names.add(text)
    return names


def _compact_frontier_metrics(metrics: Any, task_spec: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    dynamic_metric_keys = _task_prompt_metric_names(task_spec)
    compact: dict[str, Any] = {}
    for key in dict.fromkeys((*_FRONTIER_METRIC_KEYS, *sorted(dynamic_metric_keys))):
        if key in metrics:
            compact[key] = _compact_value(metrics[key], max_chars=240, max_items=6)
    if has_effective_config_metadata(compact) and metrics.get("source_result_path") not in (
        None,
        "",
    ):
        compact["source_result_path"] = _compact_value(
            metrics["source_result_path"], max_chars=240, max_items=6
        )
    dig_provenance = _compact_dig_provenance(metrics.get("dig_provenance"))
    if dig_provenance:
        compact["dig_provenance"] = dig_provenance
    for key, value in metrics.items():
        if key in compact:
            continue
        if any(str(key).startswith(prefix) for prefix in _FRONTIER_METRIC_PREFIXES):
            compact[key] = _compact_value(value, max_chars=240, max_items=6)
    omitted = max(0, len(metrics) - len(compact))
    if omitted:
        compact["_omitted_metric_count"] = omitted
    return compact


def _compact_dig_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in _DIG_PROVENANCE_KEYS:
        if key in value:
            compact[key] = _compact_value(value[key], max_chars=240, max_items=6)
    return compact


def _compact_frontier_entry(entry: dict[str, Any], task_spec: Any) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in _FRONTIER_ENTRY_KEYS:
        if key in entry:
            compact[key] = _compact_value(entry[key], max_chars=320, max_items=8)
    if "content" in entry:
        compact["content"] = _compact_text(entry["content"], max_chars=500)
    metrics = _compact_frontier_metrics(entry.get("metrics"), task_spec)
    if metrics:
        compact["metrics"] = metrics
    if has_effective_config_metadata(compact) and entry.get("source_result_path") not in (
        None,
        "",
    ):
        compact["source_result_path"] = _compact_value(
            entry["source_result_path"], max_chars=320, max_items=8
        )
    return compact


def _compact_frontier_summary_for_prompt(
    frontier_summary: list[dict[str, Any]], task_spec: Any
) -> list[dict[str, Any]]:
    return [_compact_frontier_entry(entry, task_spec) for entry in frontier_summary]


def _compact_validation_signal_entry(entry: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in _VALIDATION_SIGNAL_KEYS:
        if key in entry:
            compact[key] = _compact_value(entry[key], max_chars=280, max_items=8)
    metrics = entry.get("metrics")
    if isinstance(metrics, dict):
        metric_subset = {
            str(key): _compact_value(value, max_chars=160, max_items=4)
            for key, value in metrics.items()
            if key
            in {
                *EFFECTIVE_CONFIG_METADATA_KEYS,
                "score",
                "metric_value",
                "taskscore",
                "test_score",
                "eval_score",
                "mean_score",
                "risk_metric",
                "evidence_stage",
                "scored_cell_count",
                "mature_enough",
                "maturity_basis",
                "effort_ratio",
                "coverage_ratio",
                "min_effort_ratio",
                "min_coverage_ratio",
                "protocol_integrity_status",
            }
        }
        if has_effective_config_metadata(metric_subset) and metrics.get(
            "source_result_path"
        ) not in (None, ""):
            metric_subset["source_result_path"] = _compact_value(
                metrics["source_result_path"], max_chars=160, max_items=4
            )
        if metric_subset:
            compact["metrics"] = metric_subset
    if has_effective_config_metadata(compact) and entry.get("source_result_path") not in (
        None,
        "",
    ):
        compact["source_result_path"] = _compact_value(
            entry["source_result_path"], max_chars=280, max_items=8
        )
    return compact


def _validation_signals_for_prompt(
    run_dir: Path,
    *,
    completed_gen_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if completed_gen_id < 0:
        return [], {"returned": 0, "truncated": False, "cap": _VALIDATION_SIGNAL_PROMPT_CAP}
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            _digest_validation_candidates,
        )

        raw = _digest_validation_candidates(
            run_dir,
            max_entries=_VALIDATION_SIGNAL_PROMPT_CAP + 1,
            current_gen_id=completed_gen_id,
        )
    except Exception as exc:  # noqa: BLE001 - signal view is advisory.
        logger.debug("validation signal prompt digest failed for gen %d: %s", completed_gen_id, exc)
        return [], {
            "returned": 0,
            "truncated": False,
            "cap": _VALIDATION_SIGNAL_PROMPT_CAP,
            "error": type(exc).__name__,
        }
    truncated = len(raw) > _VALIDATION_SIGNAL_PROMPT_CAP
    compact = [
        _compact_validation_signal_entry(entry)
        for entry in raw[:_VALIDATION_SIGNAL_PROMPT_CAP]
        if isinstance(entry, dict)
    ]
    return compact, {
        "returned": len(compact),
        "truncated": truncated,
        "cap": _VALIDATION_SIGNAL_PROMPT_CAP,
    }


def _frontier_manifest_for_prompt(frontier: Any) -> dict[str, Any]:
    try:
        manifest = frontier.get_manifest() if hasattr(frontier, "get_manifest") else {}
    except Exception as exc:  # noqa: BLE001 - prompt context should degrade.
        logger.debug("frontier manifest prompt digest failed: %s", exc)
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _lane_entries_for_prompt(
    manifest: dict[str, Any],
    *,
    parent_eligible: bool,
    task_spec: Any,
    completed_gen_id: int,
    cap: int = _STRONG_PARENT_VIEW_CAP,
) -> list[dict[str, Any]]:
    lane_frontiers = manifest.get("lane_frontiers")
    if not isinstance(lane_frontiers, dict):
        return []
    lane_policies = {
        str(lane.get("name") or ""): lane
        for lane in manifest.get("frontier_lanes", [])
        if isinstance(lane, dict) and lane.get("name")
    }
    entries: list[dict[str, Any]] = []
    evaluation = getattr(task_spec, "evaluation", None)
    maturity_policy = getattr(evaluation, "maturity_policy", None)
    trust_committed_membership = is_committed_runtime_fact_source(
        manifest,
        legacy_ok=False,
    )
    for lane_name, lane_entries in lane_frontiers.items():
        policy = lane_policies.get(str(lane_name or ""), {})
        allow_lower_tier = bool(policy.get("allow_lower_tier"))
        lane_parent_eligible = bool(policy.get("parent_eligible", not allow_lower_tier))
        if lane_parent_eligible is not parent_eligible:
            continue
        if not isinstance(lane_entries, list):
            continue
        for entry in lane_entries:
            if isinstance(entry, dict):
                entry_generation = _frontier_entry_generation(entry)
                if entry_generation is None or entry_generation > int(completed_gen_id):
                    continue
                if not trust_committed_membership and not _is_committed_frontier_entry(
                    entry,
                    maturity_policy,
                ):
                    continue
                explicit_parent_eligible = _resolved_boolish_candidate_field(
                    entry,
                    "parent_eligible",
                )
                if parent_eligible and explicit_parent_eligible is False:
                    continue
                compact = _compact_frontier_entry(entry, task_spec)
                compact.setdefault("frontier_lane", str(lane_name or ""))
                compact["visibility_scope"] = (
                    "parentable" if lane_parent_eligible else "diagnostic_only"
                )
                entries.append(compact)
    return entries[:cap]


def _strong_parent_views_for_prompt(
    *,
    frontier: Any,
    validation_candidates: list[dict[str, Any]],
    task_spec: Any,
    completed_gen_id: int,
) -> dict[str, Any]:
    manifest = _frontier_manifest_for_prompt(frontier)
    incubator = _lane_entries_for_prompt(
        manifest,
        parent_eligible=True,
        task_spec=task_spec,
        completed_gen_id=completed_gen_id,
    )
    diagnostics = _lane_entries_for_prompt(
        manifest,
        parent_eligible=False,
        task_spec=task_spec,
        completed_gen_id=completed_gen_id,
    )
    validation = [
        {**entry, "visibility_scope": "revalidate_only"}
        for entry in validation_candidates[:_STRONG_PARENT_VIEW_CAP]
    ]
    return {
        "incubator_top_k": incubator,
        "validation_candidate_top_k": validation,
        "diagnostic_control_top_k": diagnostics,
        "policy": {
            "incubator": "parentable_when_task_protocol_allows",
            "validation_candidate": "revalidate_only_not_a_clean_parent",
            "diagnostic_control": "diagnostic_only_not_a_parent",
        },
    }


def _last_boundary_control_for_prompt(run_dir: Path, *, completed_gen_id: int) -> dict[str, Any]:
    if completed_gen_id < 0:
        return {}
    path = run_dir / f"gen_{completed_gen_id}" / "generation_boundary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("stop_audit", "peer_mix"):
        value = payload.get(key)
        if isinstance(value, dict):
            out[key] = _compact_value(value, max_chars=240, max_items=16)
    return out


def _copy_prompt_mapping(value: dict[str, Any]) -> dict[str, Any]:
    """Return a prompt-safe copy without dropping fields from the selected slice."""

    return {str(key): copy.deepcopy(item) for key, item in value.items()}


def _agenda_token_strings(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        tokens: set[str] = set()
        for item in value:
            tokens.update(_agenda_token_strings(item))
        return tokens
    text = str(value).strip()
    return {text} if text else set()


def _agenda_source_tokens(contract: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("source", "target_hypothesis", "parent_candidate"):
        tokens.update(_agenda_token_strings(contract.get(key)))
    return tokens


def _agenda_item_identity_tokens(section: str, item: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in _AGENDA_SOURCE_ID_KEYS.get(section, ()):
        tokens.update(_agenda_token_strings(item.get(key)))
    return tokens


def _compact_current_peer_source_context(
    agenda: dict[str, Any],
    current_contract: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    source_text = str(current_contract.get("source") or "").strip()
    source_tokens = _agenda_source_tokens(current_contract)
    out: dict[str, list[dict[str, Any]]] = {}
    for section in _AGENDA_SOURCE_SECTIONS:
        raw_items = agenda.get(section)
        if not isinstance(raw_items, list):
            continue
        items = [item for item in raw_items if isinstance(item, dict)]
        if not items:
            continue
        section_requested = section == source_text or section in source_text
        matching = [
            item for item in items if _agenda_item_identity_tokens(section, item) & source_tokens
        ]
        selected = matching or (items[:2] if section_requested else [])
        if selected:
            out[section] = [_copy_prompt_mapping(item) for item in selected[:3]]
    return out


def _agenda_source_context_hypothesis_ids(
    source_context: dict[str, list[dict[str, Any]]],
) -> set[str]:
    referenced: set[str] = set()
    for section, items in source_context.items():
        if not isinstance(items, list):
            continue
        source_keys = set(_AGENDA_SOURCE_ID_KEYS.get(section, ()))
        source_keys.update(
            {
                "claim_or_hypothesis",
                "disputed_claim",
                "target_hypothesis",
                "hypothesis_id",
                "claim_id",
            }
        )
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in source_keys:
                referenced.update(_agenda_token_strings(item.get(key)))
    return {item for item in referenced if item}


def _agenda_referenced_hypothesis_ids(agenda: dict[str, Any], contract: dict[str, Any]) -> set[str]:
    targets = _agenda_token_strings(contract.get("target_hypothesis"))
    referenced: set[str] = set(targets)
    bridge = agenda.get("bridge_hypothesis")
    bridge_ids = {"bridge_hypothesis", "bridge_contract"}
    if isinstance(bridge, dict):
        bridge_ids.update(_agenda_token_strings(bridge.get("id")))
    for target in list(targets):
        if target == "falsification_contract":
            falsification = agenda.get("falsification_contract")
            if isinstance(falsification, dict):
                referenced.update(_agenda_token_strings(falsification.get("target_hypothesis")))
        elif target in bridge_ids:
            if isinstance(bridge, dict):
                referenced.update(_agenda_token_strings(bridge.get("id")))
                for anchor_key in ("source_anchor_A", "source_anchor_B"):
                    anchor = bridge.get(anchor_key)
                    if isinstance(anchor, dict):
                        referenced.update(_agenda_token_strings(anchor.get("target_hypothesis")))
                        referenced.update(_agenda_token_strings(anchor.get("hypothesis_id")))
                        referenced.update(_agenda_token_strings(anchor.get("extracted_mechanism")))
                        referenced.update(_agenda_token_strings(anchor.get("variant")))
    return {item for item in referenced if item}


def _peer_contract_for_prompt(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the current peer's own role contract.

    The important boundary is slicing, not lossy compression: a peer should see
    its own contract exactly, while sibling contracts become a short
    coordination roster.
    """

    return _copy_prompt_mapping(contract)


def _sibling_roster_item(peer: str, contract: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {"peer_id": str(peer)}
    for key in _AGENDA_SIBLING_ROSTER_KEYS:
        if key in contract:
            item[key] = _compact_value(contract[key], max_chars=180, max_items=4)
    return item


def _contract_token_set(contract: dict[str, Any], keys: tuple[str, ...]) -> set[str]:
    tokens: set[str] = set()
    for key in keys:
        for token in _agenda_token_strings(contract.get(key)):
            if token:
                tokens.add(token)
    return tokens


def _sibling_relevance_score(current: dict[str, Any], sibling: dict[str, Any]) -> int:
    score = 0
    current_targets = _contract_token_set(
        current,
        ("target_hypothesis", "parent_candidate", "bottleneck_target"),
    )
    sibling_targets = _contract_token_set(
        sibling,
        ("target_hypothesis", "parent_candidate", "bottleneck_target"),
    )
    if current_targets & sibling_targets:
        score += 4
    current_lanes = _contract_token_set(current, ("source_lane", "target_lane"))
    sibling_lanes = _contract_token_set(sibling, ("source_lane", "target_lane"))
    if current_lanes & sibling_lanes:
        score += 2
    role = str(sibling.get("role") or "").strip().lower()
    if role in {"bridge", "replication", "replicator", "falsifier", "skeptic"}:
        score += 1
    return score


def _top_counter(counter: Counter[str], *, limit: int = 12) -> dict[str, int]:
    return {key: value for key, value in counter.most_common(limit) if key}


def _sibling_coordination_summary(contracts: dict[str, Any], peer_id: str) -> dict[str, Any]:
    role_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    lane_pair_counts: Counter[str] = Counter()
    for other_peer, contract in contracts.items():
        if str(other_peer) == peer_id or not isinstance(contract, dict):
            continue
        role = str(contract.get("role") or "").strip()
        if role:
            role_counts[role] += 1
        intent = str(contract.get("intent") or contract.get("intent_preference") or "").strip()
        if intent:
            intent_counts[intent] += 1
        source_lane = str(contract.get("source_lane") or "").strip()
        target_lane = str(contract.get("target_lane") or "").strip()
        if source_lane or target_lane:
            lane_pair_counts[f"{source_lane}->{target_lane}"] += 1
    return {
        "role_counts": _top_counter(role_counts),
        "intent_counts": _top_counter(intent_counts),
        "lane_pair_counts": _top_counter(lane_pair_counts),
    }


def _include_top_level_contract_for_peer(
    key: str,
    contract: dict[str, Any] | None,
    referenced_hypothesis_ids: set[str],
) -> bool:
    if not isinstance(contract, dict):
        return False
    role = str(contract.get("role") or "").strip().lower()
    target = str(contract.get("target_hypothesis") or "").strip()
    if key == "bridge_hypothesis":
        return (
            role == "bridge"
            or target == "bridge_hypothesis"
            or "bridge_hypothesis" in referenced_hypothesis_ids
        )
    if key == "anti_mainline_contract":
        return role == "anti_mainline" or target == "anti_mainline_contract"
    if key == "falsification_contract":
        return role == "falsifier" or target == "falsification_contract"
    return False


def _compact_research_agenda_for_prompt(
    agenda: Any,
    peer_id: str,
    *,
    full_agenda_path: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(agenda, dict):
        return None
    compact: dict[str, Any] = {
        key: copy.deepcopy(agenda[key]) for key in _AGENDA_TOP_LEVEL_KEYS if key in agenda
    }
    if full_agenda_path:
        compact["full_agenda_path"] = full_agenda_path

    mainline = agenda.get("mainline_observation")
    if isinstance(mainline, dict):
        compact["mainline_observation"] = _copy_prompt_mapping(mainline)

    contracts = agenda.get("peer_contracts")
    target_hypothesis_ids: set[str] = set()
    current_contract: dict[str, Any] | None = None
    if isinstance(contracts, dict):
        raw_current_contract = contracts.get(peer_id)
        current_contract = raw_current_contract if isinstance(raw_current_contract, dict) else None
        if current_contract is not None:
            target_hypothesis_ids.update(
                _agenda_referenced_hypothesis_ids(agenda, current_contract)
            )
            source_context = _compact_current_peer_source_context(agenda, current_contract)
            if source_context:
                target_hypothesis_ids.update(_agenda_source_context_hypothesis_ids(source_context))
                compact["current_peer_source_context"] = source_context
            compact["peer_contracts"] = {peer_id: _peer_contract_for_prompt(current_contract)}
        else:
            compact["peer_contracts"] = {}

        sibling_candidates = [
            (
                _sibling_relevance_score(current_contract or {}, contract),
                str(other_peer),
                _sibling_roster_item(str(other_peer), contract),
            )
            for other_peer, contract in contracts.items()
            if str(other_peer) != peer_id and isinstance(contract, dict)
        ]
        sibling_candidates.sort(key=lambda item: (-item[0], item[1]))
        sibling_roster = [item for _score, _peer, item in sibling_candidates[:_SIBLING_ROSTER_CAP]]
        total_siblings = len(sibling_candidates)
        if sibling_roster:
            compact["sibling_roster"] = sibling_roster
        compact["sibling_roster_total_count"] = total_siblings
        compact["sibling_contracts_omitted_count"] = max(0, total_siblings - len(sibling_roster))
        summary = _sibling_coordination_summary(contracts, peer_id)
        if any(summary.values()):
            compact["sibling_coordination_summary"] = summary

    hypotheses = agenda.get("cross_peer_hypotheses")
    if isinstance(hypotheses, list):
        selected_hypotheses: list[Any] = []
        selected_ids: set[int] = set()
        for index, hypothesis in enumerate(hypotheses):
            if len(selected_hypotheses) >= 8:
                break
            selected_hypotheses.append(hypothesis)
            selected_ids.add(index)
        if target_hypothesis_ids:
            for index, hypothesis in enumerate(hypotheses):
                if index in selected_ids or not isinstance(hypothesis, dict):
                    continue
                if str(hypothesis.get("id") or "").strip() in target_hypothesis_ids:
                    selected_hypotheses.append(hypothesis)
                    selected_ids.add(index)
        compact["cross_peer_hypotheses"] = [
            copy.deepcopy(hypothesis) for hypothesis in selected_hypotheses
        ]
        omitted = max(0, len(hypotheses) - len(selected_hypotheses))
        if omitted:
            compact["cross_peer_hypotheses_omitted_count"] = omitted

    for key in ("bridge_hypothesis", "anti_mainline_contract", "falsification_contract"):
        if isinstance(agenda.get(key), dict) and _include_top_level_contract_for_peer(
            key,
            current_contract,
            target_hypothesis_ids,
        ):
            compact[key] = _copy_prompt_mapping(agenda[key])

    success_metrics = agenda.get("success_metrics")
    if isinstance(success_metrics, dict):
        compact["success_metrics"] = _copy_prompt_mapping(success_metrics)

    omitted_top_level_keys = sorted(set(agenda) - set(compact))
    if omitted_top_level_keys:
        compact["_prompt_slicing"] = {
            "omitted_top_level_keys": omitted_top_level_keys,
            "source": "prompt_context_peer_agenda_slice_v1",
        }
    return compact


def _agent_system_from_runtime_ref(runtime_ref: str) -> str:
    """Return the short agent-system name from a ``agent_runtime:*`` ref.

    Mirrors the reverse of :data:`praxist.cli.start.AGENT_SYSTEM_TO_RUNTIME`;
    empty or non-matching input returns an empty string.
    """
    if not runtime_ref:
        return ""
    _, sep, short = runtime_ref.partition(":")
    return short if sep else ""


def _literature_lookup_enabled(
    task_spec: Any,
    available_tool_server_names: Iterable[str] | None = None,
) -> bool:
    """Return whether the effective research-loop tool set includes literature lookup."""
    if available_tool_server_names is not None:
        try:
            from praxist.core.tool_servers import LITERATURE_LOOKUP_SERVER_NAME

            return LITERATURE_LOOKUP_SERVER_NAME in set(available_tool_server_names)
        except Exception:
            return False
    try:
        from praxist.core.tool_servers import (
            LITERATURE_LOOKUP_TOOL_SERVER_REF,
            effective_research_tool_server_refs_from_task_descriptor,
        )

        refs = effective_research_tool_server_refs_from_task_descriptor(
            getattr(task_spec, "_raw", {}) or {}
        )
    except Exception:
        return False
    return LITERATURE_LOOKUP_TOOL_SERVER_REF in refs


def build_prompt_context(
    *,
    task_spec: Any,
    workspace: Path,
    run_dir: Path,
    results_dir: Path,
    variants_dir: Path,
    findings_dir: Path,
    frontier: Any,
    local_mode: bool,
    gen_id: int,
    peer_index: int,
    cohort_size: int,
    strategy: str,
    peer_role_rotation: tuple[str, ...] = (),
    peer_role_descriptions: dict[str, str] | None = None,
    available_tool_server_names: Iterable[str] | None = None,
    # Runtime identity is retained for attribution and compatibility. Prompt
    # tool instructions are shared because every production runtime exposes
    # the same resolved MCP surface.
    runtime_ref: str = "",
    gems_context: dict[str, Any] | None = None,
    logical_gen_id: int | None = None,
) -> dict[str, Any]:
    """Build the render context passed to PromptLayout for a peer.

    ``peer_role_rotation`` (issue #83) lets the panel topology supply a
    task-local role vocabulary that is applied at Gen 0, before any
    Chair-produced agenda exists. When non-empty, ``gen0_peer{i}`` is
    assigned ``rotation[i % len]`` via a synthesized minimal agenda so the
    existing ``prompt_generation.jinja2`` contract branch renders. Gen ≥ 1
    behavior is unchanged — the real Chair agenda is loaded as before.

    ``peer_role_descriptions`` (issue #85) supplies the topology's
    description text for each peer role, used by the peer prompt template's
    role-description include. When the assigned role appears in this map,
    a single-bullet task-local description renders instead of the bundled
    five-bullet vocabulary block.
    """
    diversity_dimensions = getattr(task_spec.evaluation, "diversity_dimensions", None) or None
    must_explore_axes = getattr(task_spec.evaluation, "must_explore_axes", None) or None
    effective_gen_id = gen_id if logical_gen_id is None else int(logical_gen_id)
    scoreless = is_scoreless(task_spec)
    if scoreless:
        parent_frontier_summary = []
        variant_hint = (
            "Build on the retained evidence and assigned research question. "
            "Record sources, uncertainty, contradictory evidence, and useful next steps. "
            "Retained findings are research context, not validated performance."
        )
    else:
        frontier_summary = _frontier_summary_up_to_generation(frontier, gen_id - 1)
        parent_frontier_summary = _parent_frontier_summary_up_to_generation(frontier, gen_id - 1)
        hint_frontier_summary = (
            parent_frontier_summary if strategy in {"exploit", "mixed"} else frontier_summary
        )
        variant_hint = _generate_variant_hint(
            effective_gen_id,
            peer_index,
            cohort_size,
            strategy,
            frontier,
            diversity_dimensions=diversity_dimensions,
            must_explore_axes=must_explore_axes,
            frontier_summary=hint_frontier_summary,
        )

    peer_id = f"gen{gen_id}_peer{peer_index}"
    prompt_frontier_summary = _compact_frontier_summary_for_prompt(
        parent_frontier_summary,
        task_spec,
    )
    if scoreless:
        validation_candidates, validation_candidates_meta = [], {}
        strong_parent_views = {
            "incubator_top_k": [],
            "validation_candidate_top_k": [],
            "diagnostic_control_top_k": [],
            "policy": {},
        }
    else:
        validation_candidates, validation_candidates_meta = _validation_signals_for_prompt(
            run_dir,
            completed_gen_id=gen_id - 1,
        )
        strong_parent_views = _strong_parent_views_for_prompt(
            frontier=frontier,
            validation_candidates=validation_candidates,
            task_spec=task_spec,
            completed_gen_id=gen_id - 1,
        )
    research_loop_control = (
        {}
        if scoreless
        else _last_boundary_control_for_prompt(
            run_dir,
            completed_gen_id=gen_id - 1,
        )
    )
    if not bool(getattr(task_spec.evaluation, "constructive_peer_mix_enabled", True)):
        research_loop_control.pop("peer_mix", None)

    try:
        from praxist.plugins.graph_maintainers.finding_graph_mvp.engine import (
            build_session_start_graph_context,
        )

        graph_session_context = build_session_start_graph_context(peer_id)
    except Exception as e:  # noqa: BLE001 - graph context is advisory.
        logger.debug("graph session context failed for %s: %s", peer_id, e)
        graph_session_context = ""

    research_agenda = None
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import (
            load_agenda_for_gen,
        )

        research_agenda = load_agenda_for_gen(
            run_dir,
            gen_id,
            cohort_size=cohort_size,
        )
        if research_agenda is None and gen_id > 0:
            logger.warning(
                "no research_agenda for gen %d — peers will fall back to %s",
                gen_id,
                "evidence-driven research" if scoreless else "frontier-driven free-explore mode",
            )
    except Exception as e:  # noqa: BLE001 - prompt template handles missing agenda.
        logger.debug("research_agenda load failed for gen %d: %s", gen_id, e)
        research_agenda = None

    if research_agenda is None and gen_id == 0 and peer_role_rotation:
        # Issue #83: synthesize a minimal Gen 0 agenda so cohorts whose
        # panel topology declares a peer_role_rotation see per-peer role
        # contracts from the first generation, instead of byte-identical
        # free-explore prompts that only differ in peer_id. The agenda
        # carries only the peer_contract — there are no PI-produced
        # hypotheses or cross-peer direction at this point.
        assigned_role = peer_role_rotation[peer_index % len(peer_role_rotation)]
        research_agenda = {
            "synthesized_from_gen": -1,
            "mainline_observation": {},
            "peer_contracts": {
                peer_id: {
                    "role": assigned_role,
                    "target_hypothesis": None,
                    "forbidden_actions": [],
                    "success_signal": (
                        "Produce findings consistent with your role; refer to your role's "
                        "skill.md for what counts as success."
                    ),
                    "source": "gen0_role_rotation",
                }
            },
        }

    prompt_research_agenda = _compact_research_agenda_for_prompt(
        research_agenda,
        peer_id,
        full_agenda_path=str(run_dir / "agendas" / f"research_agenda_gen{gen_id}.yaml"),
    )
    agent_system = _agent_system_from_runtime_ref(runtime_ref)
    rendered_gems_context = {} if scoreless else (gems_context or {})

    return {
        "peer_id": peer_id,
        "gen_id": gen_id,
        "logical_gen_id": effective_gen_id,
        "cohort_size": cohort_size,
        "workspace_dir": str(workspace),
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "variants_dir": str(variants_dir),
        "notebook_path": str(
            (run_dir / "work" / "notebooks" if scoreless else run_dir) / f"notebook_{peer_id}.json"
        ),
        **({"notebook_dir": str(run_dir / "work" / "notebooks")} if scoreless else {}),
        "task_spec": task_spec,
        "research_loop_mode": "scoreless" if scoreless else "metric",
        "scoreless_evidence": load_scoreless_evidence(run_dir, gen_id - 1) if scoreless else [],
        "frontier_summary": prompt_frontier_summary,
        "validation_candidates": validation_candidates,
        "validation_candidates_meta": validation_candidates_meta,
        "incubator_top_k": strong_parent_views["incubator_top_k"],
        "validation_candidate_top_k": strong_parent_views["validation_candidate_top_k"],
        "diagnostic_control_top_k": strong_parent_views["diagnostic_control_top_k"],
        "strong_parent_visibility_policy": strong_parent_views["policy"],
        "research_loop_control": research_loop_control,
        "variant_hint": variant_hint,
        "diversity_dimensions": copy.deepcopy(diversity_dimensions or []),
        "findings_dir": str(findings_dir),
        "logs_dir": str(run_dir / f"gen_{gen_id}"),
        "local_mode": local_mode,
        "graph_session_context": graph_session_context,
        "research_agenda": prompt_research_agenda,
        "peer_role_descriptions": dict(peer_role_descriptions or {}),
        "agent_system": agent_system,
        "literature_lookup_enabled": _literature_lookup_enabled(
            task_spec,
            available_tool_server_names=available_tool_server_names,
        ),
        "gems_context": rendered_gems_context,
        "effective_config_provenance_available": has_effective_config_metadata(
            (
                prompt_frontier_summary,
                validation_candidates,
                strong_parent_views,
                rendered_gems_context,
            )
        ),
    }
