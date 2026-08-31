"""Evidence pack builder — assembles shared_evidence_core + role-specific
private packs from ledgers and cards.

This is the single entry point that PI panel uses; all PI prompts derive
from a pack produced here.

Inputs:
  run_dir
  panel_mode: "mini" | "full" | "high_stakes"
  current_gen_id
  ledgers (loaded by caller)

Outputs:
  EvidencePack {
    shared_core: dict (the digest passed to ALL PIs)
    private_packs: dict[role -> list of evidence_cards]
    pack_id, built_at
  }
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    DERIVED_AUDIT_SNAPSHOT,
    attach_artifact_semantics,
    explicit_entry_generation_id,
    is_committed_runtime_fact_source,
    is_readable_signal_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    EFFECTIVE_CONFIG_METADATA_KEYS,
)
from praxist.plugins.workflow_stages.research_loop.backend.gems import (
    _entry_source_generation_id,
    _resolved_persisted_gem_source_generation_id,
    load_active_gems_for_prompt,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.card_builder import (
    build_cards_from_db,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers import (
    ClaimLedger,
    CoverageMatrix,
    DissentLedger,
    FrontierDeltaLedger,
    NegativeEvidenceLedger,
    RetiredClaimLedger,
    RoleROILedger,
)
from praxist.plugins.workflow_stages.research_loop.backend.research_memory.retrieval_policy import (
    HIGH_STAKES_MIX,
    NORMAL_MIX,
    negative_evidence_ratio,
    select_cards_with_mix,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import load_scoreless_evidence
from praxist.task_spec_compat import legacy_primary_metric_keys

logger = logging.getLogger(__name__)

_VALIDATION_CANDIDATES_SHARED_CORE_CAP = 16

_RESEARCH_METADATA_KEYS = (
    "bottleneck_target",
    "evidence_stage",
    "tradeoff_class",
    "primary_tradeoff",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
    "source_lane",
    "target_lane",
    "coverage_check",
    "mechanism_hypothesis_deliverable",
)


def _load_frontier_manifest_for_context(
    run_dir: Path,
    *,
    purpose: str,
    allow_signal_source: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(run_dir) / "frontier" / "frontier_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "could not read frontier manifest for %s %s: %s", purpose, manifest_path, exc
        )
        return {}
    if not isinstance(manifest, dict):
        return {}
    if is_committed_runtime_fact_source(manifest, legacy_ok=True):
        return manifest
    if allow_signal_source and is_readable_signal_source(manifest, legacy_ok=True):
        logger.warning(
            "using non-runtime frontier manifest only as research signal source for %s: %s",
            purpose,
            manifest_path,
        )
        semantics = manifest.get("artifact_semantics")
        signal_status = (
            str(semantics.get("status") or "").strip().lower()
            if isinstance(semantics, dict)
            else "legacy"
        )
        signal_manifest = dict(manifest)
        signal_manifest["_non_runtime_signal_source"] = True
        signal_manifest["_non_runtime_signal_status"] = signal_status or "unknown"
        return signal_manifest
    logger.warning(
        "ignoring non-committed runtime frontier manifest for %s: %s",
        purpose,
        manifest_path,
    )
    return {}


_VALIDATION_DIVERSITY_METADATA_KEYS = (
    "diversity_overlap_status",
    "diversity_most_similar_anchor",
    "diversity_overlap_score",
    "diversity_overlap_fraction",
    "diversity_overlap_count",
    "diversity_overlap_total",
    "diversity_violated",
    "diversity_narrow_variation",
    "diversity_violation",
    "mechanism_family",
    "intervention_surface",
    "intent",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
)
_VALIDATION_IDENTITY_KEYS = (
    "finding_id",
    "variant_name",
    "variant_id",
    "canonical_variant_id",
    "canonical_variant_name",
    "frontier_entity_key",
    "candidate_entity_key",
    "child_variant_name",
    "child_variant_id",
    "source_path",
    "result_path",
    "source_result_path",
    "canonical_source_result_path",
    "best_available_summary_path",
    "result_artifact_path",
    "summary_path",
)


def _configured_manifest_metric_keys(manifest: dict[str, Any]) -> tuple[str, ...]:
    gems = manifest.get("gems")
    gems = gems if isinstance(gems, dict) else {}
    keys: list[str] = []
    for field_name in (
        "primary_metric_keys",
        "secondary_metric_keys",
        "lower_tail_metric_keys",
        "validation_metric_keys",
        "cost_metric_keys",
    ):
        value = gems.get(field_name)
        if isinstance(value, (list, tuple, set)):
            keys.extend(str(item).strip() for item in value if str(item or "").strip())
    derivations = gems.get("result_cell_metric_derivations")
    if isinstance(derivations, list):
        for rule in derivations:
            if isinstance(rule, dict):
                name = str(rule.get("name") or "").strip()
                if name:
                    keys.append(name)
    aliases = gems.get("result_metric_aliases")
    if isinstance(aliases, dict):
        for out_key, source_key in aliases.items():
            for key in (out_key, source_key):
                text = str(key or "").strip()
                if text:
                    keys.append(text)
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


def _configured_manifest_metric_aliases(manifest: dict[str, Any]) -> dict[str, str]:
    gems = manifest.get("gems")
    gems = gems if isinstance(gems, dict) else {}
    aliases = gems.get("result_metric_aliases")
    if not isinstance(aliases, dict):
        return {}
    out: dict[str, str] = {}
    for out_key, source_key in aliases.items():
        out_text = str(out_key or "").strip()
        source_text = str(source_key or "").strip()
        if out_text and source_text:
            out[out_text] = source_text
    return out


def _safe_prompt_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        return text[:200] if text else None
    if isinstance(value, list):
        out = []
        for item in value[:12]:
            if isinstance(item, (str, int, float, bool)):
                safe = _safe_prompt_value(item)
                if safe is not None:
                    out.append(safe)
        return out or None
    return None


def _iter_validation_identity_values(entry: Any):
    if not isinstance(entry, dict):
        return
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    for source in (entry, metrics):
        for key in _VALIDATION_IDENTITY_KEYS:
            value = source.get(key)
            if isinstance(value, list):
                for item in value:
                    yield item
            else:
                yield value
        aliases = source.get("identity_aliases")
        if isinstance(aliases, list):
            for item in aliases:
                yield item


def _iter_validation_retirement_values(entry: Any):
    if not isinstance(entry, dict):
        return
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
            _candidate_retirement_aliases,
        )

        yield from _candidate_retirement_aliases(entry)
        return
    except (AttributeError, ImportError):
        logger.debug("falling back to validation identity aliases for retirement", exc_info=True)
    yield from _iter_validation_identity_values(entry)


def _ledger_entry_generation(entry: Any) -> int | None:
    data = getattr(entry, "data", None)
    if not isinstance(data, dict):
        data = {}
    candidates: list[Any] = [
        data.get("generation_id"),
        data.get("source_generation_id"),
        data.get("gen_id"),
        getattr(entry, "generation_id", None),
        getattr(entry, "id", None),
        getattr(entry, "created_by", None),
    ]
    for source in (
        data.get("source_evidence_id"),
        data.get("evidence_id"),
        data.get("finding_id"),
    ):
        candidates.append(source)
    sources = data.get("sources")
    if isinstance(sources, list):
        candidates.extend(sources)
    generations: list[int] = []
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, int):
            generations.append(value)
            continue
        if isinstance(value, float) and value.is_integer():
            generations.append(int(value))
            continue
        text = str(value).strip()
        if not text:
            continue
        import re

        for match in re.finditer(
            r"(?:^|[^a-z0-9])gen(?:eration)?[_:\-\s]*(\d+)",
            text,
            re.IGNORECASE,
        ):
            with contextlib.suppress(ValueError):
                generations.append(int(match.group(1)))
    return max(generations) if generations else None


def _within_generation_cutoff(entry: Any, current_gen_id: int | None) -> bool:
    if current_gen_id is None:
        return True
    generation = _ledger_entry_generation(entry)
    return generation is not None and generation <= int(current_gen_id)


# R3#6 fix: keep card metrics safe for downstream tojson rendering.
# NaN/Inf would otherwise produce invalid JSON literals that break the
# LLM's prompt parsing.
def _sanitize_value(v: Any) -> Any:
    """Recursively sanitize values for safe inclusion in PI prompts:
    - NaN/Inf -> None (R3#6: tojson cannot emit these)
    - {{ / }} / {% / %} in user-supplied strings -> escaped (R3#1:
      defense against accidental Jinja syntax appearing inside finding
      text or claim summaries; the LLM still sees the literal text but
      stray jinja-syntax can't sneak through if a downstream consumer
      re-renders).
    """
    import math

    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, str):
        # Escape jinja delimiters defensively. Use unicode look-alikes
        # (U+007B FULLWIDTH would change meaning); instead we insert a
        # zero-width space between the two braces so the LLM still reads
        # the same characters but jinja cannot match its delimiters.
        if "{{" in v or "}}" in v or "{%" in v or "%}" in v:
            return v.replace("{{", "{​{").replace("}}", "}​}").replace("{%", "{​%").replace("%}", "%​}")
        return v
    if isinstance(v, dict):
        return {k: _sanitize_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_value(x) for x in v]
    return v


@dataclass
class EvidencePack:
    """Evidence bundle assembled for PI panel synthesis and role-private review."""

    pack_id: str
    built_at: str
    panel_mode: str
    target_decisions: list[str]
    shared_core: dict[str, Any]
    private_packs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    all_cards: list[dict[str, Any]] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared core construction


def _digest_claims(
    claim_ledger: ClaimLedger,
    max_active: int = 8,
    current_gen_id: int | None = None,
) -> dict[str, Any]:
    active = [
        entry
        for entry in claim_ledger.list_active()
        if _within_generation_cutoff(entry, current_gen_id)
    ]
    active.sort(key=lambda e: e.last_updated_at, reverse=True)
    killed = [
        entry
        for entry in claim_ledger.list_recently_killed(n=30)
        if _within_generation_cutoff(entry, current_gen_id)
    ][:3]

    def _ent_summary(e):
        return {
            "id": e.id,
            "title": e.data.get("title", ""),
            "status": e.data.get("status", "active"),
            "confidence": e.data.get("confidence"),
            "boundary": e.data.get("boundary", ""),
            "supports_count": len(e.data.get("supports", [])),
            "challenges_count": len(e.data.get("challenges", [])),
            "missing_tests": list(e.data.get("missing_tests", []))[:5],
        }

    return {
        "active": [_ent_summary(e) for e in active[:max_active]],
        "recently_killed": [_ent_summary(e) for e in killed],
    }


def _digest_frontier(fd: FrontierDeltaLedger, current_gen_id: int | None = None) -> dict[str, Any]:
    if current_gen_id is None:
        latest = fd.latest_per_axis()
    else:
        latest: dict[str, Any] = {}
        for entry in fd.all():
            axis = entry.data.get("axis")
            if not isinstance(axis, str):
                continue
            try:
                generation_raw = entry.data.get("generation_id")
                if generation_raw is None:
                    continue
                gen_id = int(generation_raw)
            except (TypeError, ValueError):
                continue
            if gen_id > int(current_gen_id):
                continue
            current = latest.get(axis)
            if current is None or gen_id > int(current.data.get("generation_id", -1)):
                latest[axis] = entry
    out: dict[str, Any] = {}
    for axis, e in latest.items():
        out[axis] = {
            "current_anchor": e.data.get("current_anchor", {}),
            "previous_anchor": e.data.get("previous_anchor", {}),
            "raw_delta": e.data.get("raw_delta"),
            "since_gen": e.data.get("generation_id"),
        }
    return out


def _digest_lane_frontiers(
    run_dir: Path,
    max_entries_per_lane: int = 12,
    current_gen_id: int | None = None,
    total_entries_by_lane: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Read lane-based frontier summaries from frontier_manifest.json.

    This is intentionally generic: it does not know what a task's lanes mean,
    but it preserves lane names, variants, metric names, metric values, tiers,
    and repair flags so PI/Chair prompts can distinguish deployable candidates
    from durable-candidate, reference, diagnostic, or process lanes.
    """
    manifest = _load_frontier_manifest_for_context(run_dir, purpose="lane frontier digest")
    if not manifest:
        return {}
    raw_lanes = manifest.get("lane_frontiers")
    if not isinstance(raw_lanes, dict):
        return {}
    lane_policies = {
        str(lane.get("name") or ""): lane
        for lane in manifest.get("frontier_lanes", [])
        if isinstance(lane, dict) and lane.get("name")
    }
    configured_metric_keys = _configured_manifest_metric_keys(manifest)
    configured_metric_aliases = _configured_manifest_metric_aliases(manifest)
    trust_committed_membership = is_committed_runtime_fact_source(
        manifest,
        legacy_ok=False,
    )
    out: dict[str, Any] = {}
    for lane_name, entries in raw_lanes.items():
        if not isinstance(entries, list):
            continue
        lane_policy = lane_policies.get(str(lane_name or ""), {})
        allow_lower_tier = bool(lane_policy.get("allow_lower_tier"))
        lane_parent_eligible = bool(lane_policy.get("parent_eligible", not allow_lower_tier))
        compact = []
        eligible_count = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_gen = (
                explicit_entry_generation_id(entry)
                if trust_committed_membership
                else _entry_source_generation_id(entry)
            )
            if current_gen_id is not None and (
                entry_gen is None or entry_gen > int(current_gen_id)
            ):
                continue
            if not trust_committed_membership:
                try:
                    from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                        _is_committed_frontier_entry,
                    )

                    if not _is_committed_frontier_entry(entry):
                        continue
                except (AttributeError, ImportError):
                    logger.debug("could not import frontier durable filter", exc_info=True)
            eligible_count += 1
            if len(compact) >= max_entries_per_lane:
                if total_entries_by_lane is None:
                    break
                continue
            metrics = entry.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}

            def pick(
                *keys: str,
                default: Any = "",
                _entry: dict[str, Any] = entry,
                _metrics: dict[str, Any] = metrics,
            ) -> Any:
                for key in keys:
                    value = _entry.get(key)
                    if value is None:
                        value = _metrics.get(key)
                    if value is not None:
                        return value
                return default

            def pick_bool(*keys: str, default: bool | None = None) -> bool | None:
                value = pick(*keys, default=None)
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if value == 1:
                        return True
                    if value == 0:
                        return False
                if isinstance(value, str):
                    token = value.strip().lower()
                    if token in {"true", "yes", "1", "promotable", "passed"}:
                        return True
                    if token in {"false", "no", "0", "non-promotable", "failed"}:
                        return False
                return default

            promoted_for_lane = pick("promoted_for_lane", default="")
            submitted_lane = pick("source_frontier_lane", "frontier_lane", default="")
            entry_parent_eligible = pick_bool("parent_eligible", default=None)
            item = {
                "finding_id": entry.get("finding_id", ""),
                "variant_name": entry.get("variant_name", ""),
                "metric_name": pick("lane_metric_name", "metric_name", default=""),
                "metric_value": pick("lane_metric_value", "metric_value", default=None),
                "frontier_lane": promoted_for_lane or str(lane_name),
                "promoted_for_lane": promoted_for_lane or str(lane_name),
                "parent_eligible": bool(
                    lane_parent_eligible and entry_parent_eligible is not False
                ),
                "submitted_frontier_lane": submitted_lane,
                "source_frontier_lane": pick("source_frontier_lane", default=""),
                "tier": pick("tier", "tier_reached", "candidate_tier", default=""),
                "candidate_tier": pick("candidate_tier", "tier_reached", "tier", default=""),
                "generation_id": entry_gen,
                "canonical_variant_id": pick("canonical_variant_id", default=""),
                "source_result_path": pick("source_result_path", default=""),
                "canonical_source_result_path": pick("canonical_source_result_path", default=""),
                "best_available_summary_path": pick("best_available_summary_path", default=""),
                "source_selection_warning": pick("source_selection_warning", default=""),
                "source_selection_reason": pick("source_selection_reason", default=""),
                "canonical_metric_value": pick("canonical_metric_value", default=None),
                "selected_metric_value": pick("selected_metric_value", default=None),
                "tier_status": pick("tier_status", default=""),
                "promotion_eligible": pick("promotion_eligible", default=None),
                "clean_promotion_eligible": pick("clean_promotion_eligible", default=None),
                "evidence_stage": pick("evidence_stage", default=""),
                "evidence_maturity_rank": pick("evidence_maturity_rank", default=None),
                "mature_enough": pick("mature_enough", default=None),
                "maturity_basis": pick("maturity_basis", default=""),
                "scout_only": pick_bool("scout_only"),
                "evaluation_units": pick(
                    "evaluation_units",
                    "completed_required_eval_units",
                    "actual_eval_units",
                    "scored_cell_count",
                    "n_scored_cells",
                    "n_eval_cells",
                    default=None,
                ),
                "frontier_entity_key": pick("frontier_entity_key", default=""),
                "scored_complete": pick("scored_complete", default=None),
                "risk_violating_frontier_candidate": pick_bool("risk_violating_frontier_candidate"),
                "risk_repair_required": pick_bool("risk_repair_required"),
                "risk_violation_reason": pick("risk_violation_reason", default=""),
                **{key: pick(key, default=None) for key in EFFECTIVE_CONFIG_METADATA_KEYS},
                **{key: pick(key, default="") for key in _RESEARCH_METADATA_KEYS},
                "lane_lower_tier_candidate": pick_bool("lane_lower_tier_candidate"),
                "lane_non_promotable_candidate": pick_bool("lane_non_promotable_candidate"),
                "lane_missing_tier_candidate": pick_bool("lane_missing_tier_candidate"),
                "n_hard_constraint_violations": pick("n_hard_constraint_violations", default=None),
            }
            for task_key in configured_metric_keys:
                safe_value = _safe_prompt_value(pick(task_key, default=None))
                if safe_value is not None:
                    item[task_key] = safe_value
            for out_key, source_key in configured_metric_aliases.items():
                if item.get(out_key) not in (None, "", [], {}):
                    continue
                safe_value = _safe_prompt_value(pick(source_key, default=None))
                if safe_value is not None:
                    item[out_key] = safe_value
            for task_key in (
                *EFFECTIVE_CONFIG_METADATA_KEYS,
                "tier",
                "candidate_tier",
                "tier_status",
                "promotion_eligible",
                "clean_promotion_eligible",
                "evidence_stage",
                "evidence_maturity_rank",
                "mature_enough",
                "maturity_basis",
                "scout_only",
                "scored_complete",
                "parent_eligible",
                "risk_violating_frontier_candidate",
                "risk_repair_required",
                "risk_violation_reason",
                "canonical_variant_id",
                "source_result_path",
                "canonical_source_result_path",
                "best_available_summary_path",
                "source_selection_warning",
                "source_selection_reason",
                "canonical_metric_value",
                "selected_metric_value",
                "lane_lower_tier_candidate",
                "lane_non_promotable_candidate",
                "lane_missing_tier_candidate",
                "n_hard_constraint_violations",
            ):
                if item.get(task_key) in ("", None, [], {}):
                    item.pop(task_key, None)
            compact.append(item)
        if total_entries_by_lane is not None:
            total_entries_by_lane[str(lane_name)] = eligible_count
        out[str(lane_name)] = compact
    return out


def _digest_validation_candidates(
    run_dir: Path,
    max_entries: int = 16,
    current_gen_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return compact non-frontier candidates that need mature validation.

    These entries are deliberately separate from lane_frontiers/cumulative_top:
    they preserve promising scout/partial signals for PI follow-up without
    making them durable frontier entries or Gems parents.
    """

    manifest = _load_frontier_manifest_for_context(
        run_dir,
        purpose="validation candidate digest",
        allow_signal_source=True,
    )
    if not manifest:
        return []
    manifest_signal_only = bool(manifest.get("_non_runtime_signal_source"))
    manifest_signal_status = str(manifest.get("_non_runtime_signal_status") or "").strip()
    trust_committed_membership = not manifest_signal_only and is_committed_runtime_fact_source(
        manifest, legacy_ok=False
    )
    raw = manifest.get("validation_candidates")
    if not isinstance(raw, dict):
        if not manifest_signal_only:
            return []
        raw = {}

    def coerce_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def coerce_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def entry_generation(entry: dict[str, Any], generation_hint: int | None) -> int:
        if trust_committed_membership:
            explicit = explicit_entry_generation_id(
                entry,
                generation_hint=generation_hint,
            )
            if explicit is not None:
                return explicit
        return coerce_int(
            entry.get("generation_id"),
            generation_hint if generation_hint is not None else -1,
        )

    def entity_key(entry: dict[str, Any]) -> str:
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                _candidate_entity_key,
            )

            return _candidate_entity_key(entry)
        except (AttributeError, ImportError):
            logger.debug("falling back to local validation entity key", exc_info=True)
        frontier_key = normalized_token(entry.get("frontier_entity_key"))
        if frontier_key:
            if ":" in frontier_key and "::" not in frontier_key:
                prefix, _, payload = frontier_key.partition(":")
                if payload:
                    return f"{prefix}::{payload}"
            if "::" in frontier_key:
                prefix, _, payload = frontier_key.partition("::")
                return f"{prefix}::{payload}" if payload else ""
            return frontier_key
        variant = normalized_token(entry.get("variant_name"))
        if variant:
            return f"variant::{variant}"
        finding_id = normalized_token(entry.get("finding_id"))
        if finding_id:
            return f"finding::{finding_id}"
        return normalized_token(json.dumps(entry, sort_keys=True, default=str)) or "entry"

    def durable_entity_keys() -> set[str]:
        if manifest_signal_only:
            return set()
        keys: set[str] = set()

        def add_entries(entries: Any, generation_hint: int | None = None) -> None:
            if not isinstance(entries, list):
                return
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                gen_id = entry_generation(entry, generation_hint)
                if current_gen_id is not None and (gen_id < 0 or gen_id > int(current_gen_id)):
                    continue
                if not is_durable_entry(entry):
                    continue
                keys.add(entity_key(entry))

        add_entries(manifest.get("cumulative_top"))
        generations = manifest.get("generations")
        if isinstance(generations, dict):
            for gen_key, entries in generations.items():
                add_entries(entries, generation_hint=coerce_int(gen_key, -1))
        lane_frontiers = manifest.get("lane_frontiers")
        if isinstance(lane_frontiers, dict):
            for entries in lane_frontiers.values():
                add_entries(entries)
        gems = manifest.get("gems")
        gem_entries = gems.get("entries") if isinstance(gems, dict) else None
        if isinstance(gem_entries, list):
            for entry in gem_entries:
                if not isinstance(entry, dict) or not is_durable_gem_entry(entry):
                    continue
                if not gem_has_acceptable_source_generation(entry):
                    continue
                keys.add(entity_key(gem_identity_entry(entry)))
        return {key for key in keys if key}

    def durable_entity_aliases() -> set[str]:
        if manifest_signal_only:
            return set()
        aliases: set[str] = set()

        def add_entries(entries: Any, generation_hint: int | None = None) -> None:
            if not isinstance(entries, list):
                return
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                gen_id = entry_generation(entry, generation_hint)
                if current_gen_id is not None and (gen_id < 0 or gen_id > int(current_gen_id)):
                    continue
                if not is_durable_entry(entry):
                    continue
                aliases.update(
                    str(value).strip()
                    for value in _iter_validation_retirement_values(entry)
                    if value not in (None, "", [], {})
                )
                key = entity_key(entry)
                if key:
                    aliases.add(key)

        add_entries(manifest.get("cumulative_top"))
        generations = manifest.get("generations")
        if isinstance(generations, dict):
            for gen_key, entries in generations.items():
                add_entries(entries, generation_hint=coerce_int(gen_key, -1))
        lane_frontiers = manifest.get("lane_frontiers")
        if isinstance(lane_frontiers, dict):
            for entries in lane_frontiers.values():
                add_entries(entries)
        gems = manifest.get("gems")
        gem_entries = gems.get("entries") if isinstance(gems, dict) else None
        if isinstance(gem_entries, list):
            for entry in gem_entries:
                if not isinstance(entry, dict) or not is_durable_gem_entry(entry):
                    continue
                if not gem_has_acceptable_source_generation(entry):
                    continue
                payload = gem_identity_entry(entry)
                aliases.update(
                    str(value).strip()
                    for value in _iter_validation_retirement_values(payload)
                    if value not in (None, "", [], {})
                )
                key = entity_key(payload)
                if key:
                    aliases.add(key)
        return {alias for alias in aliases if alias}

    def boolish(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        token = str(value or "").strip().lower()
        if token in {"true", "yes", "1", "full", "complete", "completed", "confirmed"}:
            return True
        if token in {"false", "no", "0", "partial", "incomplete", "scout", "smoke"}:
            return False
        return None

    def is_preliminary_entry(entry: dict[str, Any]) -> bool:
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                _is_preliminary_or_incomplete_evidence,
            )

            return _is_preliminary_or_incomplete_evidence(entry)
        except (AttributeError, ImportError):
            logger.debug("falling back to local validation maturity check", exc_info=True)
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        for key in (
            "excluded_from_durable_frontier",
            "scout_only",
            "is_scout_eval",
            "is_smoke_eval",
            "smoke_only",
            "unscored_artifact",
            "summary_only",
            "is_summary_only",
            "partial_cohort",
            "partial_eval",
            "is_partial_eval",
            "incomplete_eval",
            "is_incomplete_eval",
            "capped",
            "is_capped",
            "result_capped",
        ):
            if boolish(entry.get(key, metrics.get(key))) is True:
                return True
        for key in (
            "scored_complete",
            "is_scored_complete",
            "complete_eval",
            "is_complete_eval",
        ):
            if boolish(entry.get(key, metrics.get(key))) is False:
                return True
        for key in (
            "tier_status",
            "final_status",
            "result_status",
            "status",
            "completion_status",
            "eval_status",
            "scoring_status",
        ):
            status = str(entry.get(key) or metrics.get(key) or "").strip().lower()
            if status in {
                "smoke",
                "unscored",
                "un_scored",
                "unscored_artifact",
                "not_scored",
                "scout",
                "cheap_probe",
                "partial",
                "preliminary",
                "prelim",
                "summary_only",
                "capped",
                "capped_at",
                "cap_at",
                "incomplete",
                "scored_complete_false",
                "complete_eval_false",
            }:
                return True
        if (
            str(entry.get("exclusion_reason") or metrics.get("exclusion_reason") or "")
            .strip()
            .lower()
            == "preliminary_or_incomplete_evidence"
        ):
            return True
        for key in ("evidence_stage", "eval_stage", "stage"):
            stage = str(entry.get(key) or metrics.get(key) or "").strip().lower().replace("-", "_")
            if stage in {
                "scout",
                "cheap_probe",
                "probe",
                "smoke",
                "sanity",
                "partial",
                "preliminary",
                "prelim",
                "incomplete",
                "partial_cohort",
                "partial_eval",
                "summary_only",
                "unscored",
                "un_scored",
                "unscored_artifact",
                "failed_or_unscored",
                "capped",
                "capped_at",
                "cap_at",
            }:
                return True
        return False

    def is_durable_entry(entry: dict[str, Any]) -> bool:
        if trust_committed_membership:
            return True
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                _is_committed_frontier_entry,
            )

            return _is_committed_frontier_entry(entry)
        except (AttributeError, ImportError):
            logger.debug("falling back to local durable frontier check", exc_info=True)
        return (
            isinstance(entry, dict)
            and not is_preliminary_entry(entry)
            and has_complete_evidence(entry)
        )

    def has_complete_evidence(entry: dict[str, Any]) -> bool:
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        for key in (
            "mature_enough",
            "complete_eval",
            "is_complete_eval",
            "scored_complete",
            "is_scored_complete",
        ):
            value = entry.get(key, metrics.get(key))
            if isinstance(value, bool):
                if value:
                    return True
                continue
            text = str(value or "").strip().lower()
            if text in {"true", "yes", "1", "complete", "completed", "confirmed"}:
                return True
        for key in (
            "evidence_stage",
            "tier",
            "tier_reached",
            "completed_tier",
            "candidate_tier",
            "tier_status",
            "final_status",
            "result_status",
            "completion_status",
            "eval_status",
        ):
            text = str(entry.get(key) or metrics.get(key) or "").strip().lower()
            normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
            if normalized in {
                "scored_complete",
                "complete_eval",
                "complete_eval_true",
                "is_complete_eval_true",
                "scored_complete_true",
                "is_scored_complete_true",
                "full_eval",
                "full_evaluation",
            }:
                return True
        return False

    def has_nonclean_gem_marker(entry: dict[str, Any]) -> bool:
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        for key in ("promotion_eligible", "clean_promotion_eligible"):
            value = entry.get(key, metrics.get(key))
            if isinstance(value, bool) and not value:
                return True
            text = str(value or "").strip().lower()
            if text in {"false", "no", "0", "ineligible", "nonpromotable", "non_promotable"}:
                return True
        for key in (
            "hard_constraint_violation_count",
            "hard_constraint_violations",
            "hard_constraint_failures",
            "constraint_violation_count",
            "constraint_violations",
        ):
            value = entry.get(key, metrics.get(key))
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (list, tuple, set, dict)):
                if len(value) > 0:
                    return True
                continue
            if isinstance(value, (int, float)):
                if value > 0:
                    return True
                continue
            text = str(value).strip().lower()
            if text and text not in {"0", "none", "no", "false", "[]", "{}"}:
                return True
        return False

    def is_legacy_persisted_gem_entry(entry: dict[str, Any]) -> bool:
        if not entry.get("gem_finding_id") or not isinstance(entry.get("admission_metrics"), dict):
            return False
        payload = gem_identity_entry(entry)
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        for cell_key in (
            "scored_cell_count",
            "n_scored_cells",
            "n_eval_cells",
            "cell_count",
            "n_cells",
        ):
            if coerce_int(payload.get(cell_key, metrics.get(cell_key)), 0) > 0:
                return False
        for key in (
            *legacy_primary_metric_keys(),
            "score",
            "metric_value",
            "lane_metric_value",
        ):
            if coerce_float(payload.get(key, metrics.get(key))) is not None:
                return True
        return False

    def is_durable_gem_entry(entry: dict[str, Any]) -> bool:
        if trust_committed_membership:
            return True
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.gems import (
                _entry_has_complete_eval_evidence,
                _entry_has_explicit_scout_or_partial_marker,
                _entry_has_nonclean_gem_marker,
            )

            return (
                not _entry_has_explicit_scout_or_partial_marker(entry)
                and not _entry_has_nonclean_gem_marker(entry)
                and _entry_has_complete_eval_evidence(entry)
            )
        except (AttributeError, ImportError):
            logger.debug("falling back to local durable Gem check", exc_info=True)
        payload = gem_identity_entry(entry)
        return (
            isinstance(entry.get("admission_metrics"), dict)
            and not is_preliminary_entry(payload)
            and not has_nonclean_gem_marker(payload)
            and (has_complete_evidence(payload) or is_legacy_persisted_gem_entry(entry))
        )

    def gem_identity_entry(entry: dict[str, Any]) -> dict[str, Any]:
        payload = dict(entry)
        admission = entry.get("admission_metrics")
        metrics = dict(admission) if isinstance(admission, dict) else {}
        existing_metrics = entry.get("metrics")
        if isinstance(existing_metrics, dict):
            metrics.update(existing_metrics)
        if metrics:
            payload["metrics"] = metrics
        return payload

    def gem_has_acceptable_source_generation(entry: dict[str, Any]) -> bool:
        if current_gen_id is None:
            return True
        source_generation = (
            _resolved_persisted_gem_source_generation_id(
                run_dir,
                entry,
                allow_identity_inference=False,
            )
            if trust_committed_membership
            else _resolved_persisted_gem_source_generation_id(run_dir, entry)
        )
        if source_generation is not None:
            return source_generation <= int(current_gen_id)
        return bool(
            is_legacy_persisted_gem_entry(entry)
            and (entry.get("variant_name") or entry.get("variant_id"))
        )

    def normalized_token(value: Any) -> str:
        if value in (None, ""):
            return ""
        try:
            text = str(value).strip().lower()
        except (TypeError, ValueError):
            return ""
        return " ".join(text.split())

    def sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        metric_value = coerce_float(entry.get("metric_value"))
        if metric_value is None:
            directional_value = float("-inf")
        else:
            direction = str(entry.get("metric_direction") or "maximize")
            if direction not in {"maximize", "minimize"}:
                direction = "maximize"
            directional_value = metric_value if direction == "maximize" else -metric_value
        return (
            coerce_int(entry.get("signal_source_priority"), 0),
            coerce_int(entry.get("evidence_maturity_rank"), 0),
            directional_value,
            coerce_int(entry.get("generation_id"), -1),
            str(entry.get("variant_name") or entry.get("finding_id") or ""),
        )

    def diversity_facets(entry: dict[str, Any]) -> set[str]:
        facets: set[str] = set()
        for key in (
            "bottleneck_target",
            "evidence_stage",
            "tradeoff_class",
            "primary_tradeoff",
            "next_step_intent",
            "source_lane",
            "target_lane",
            "submitted_frontier_lane",
            "mechanism_family",
            "intervention_surface",
            "intent",
            "semantic_family",
            "parent_lineage",
            "novelty_axis",
            "diversity_overlap_status",
        ):
            value = str(entry.get(key) or "").strip().lower()
            if value:
                facets.add(f"{key}:{value}")
        for key in (
            "matched_frontier_lanes",
            "signal_axis_lanes",
            "retained_validation_lanes",
        ):
            values = entry.get(key)
            if isinstance(values, list):
                for value in values:
                    text = str(value or "").strip().lower()
                    if text:
                        facets.add(f"{key}:{text}")
        return facets

    def select_validation_candidates(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(entries, key=sort_key, reverse=True)
        if max_entries <= 0:
            return []
        if len(ranked) <= max_entries:
            return ranked
        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        covered_facets: set[str] = set()

        def add(entry: dict[str, Any]) -> bool:
            key = validation_record_key(entry, str(entry.get("frontier_entity_key") or ""))
            if key in selected_keys:
                return False
            selected.append(entry)
            selected_keys.add(key)
            covered_facets.update(diversity_facets(entry))
            return True

        score_floor = min(max_entries, max(1, max_entries // 2))
        for entry in ranked[:score_floor]:
            add(entry)
        for entry in ranked[score_floor:]:
            if len(selected) >= max_entries:
                break
            facets = diversity_facets(entry)
            if facets and not facets.issubset(covered_facets):
                add(entry)
        for entry in ranked:
            if len(selected) >= max_entries:
                break
            add(entry)
        return selected

    def validation_record_key(entry: dict[str, Any], entry_entity_key: str = "") -> str:
        parts = [
            str(entry.get("generation_id") or ""),
            str(entry.get("finding_id") or ""),
            str(entry.get("variant_name") or ""),
            str(entry.get("metric_name") or ""),
            str(entry.get("signal_source") or ""),
            str(entry.get("source_path") or ""),
            str(entry.get("result_path") or ""),
            str(entry.get("source_result_path") or ""),
            str(entry.get("result_artifact_path") or ""),
        ]
        key = "\x1f".join(part.strip().lower() for part in parts if part and part != "None")
        return key or entry_entity_key or json.dumps(entry, sort_keys=True, default=str)

    compact_by_record: dict[str, dict[str, Any]] = {}
    durable_keys = durable_entity_keys()
    durable_aliases = durable_entity_aliases()

    def merge_aliases(target: dict[str, Any], source: dict[str, Any]) -> None:
        aliases = {
            str(value).strip()
            for value in _iter_validation_identity_values(target)
            if value not in (None, "", [], {})
        }
        aliases.update(
            str(value).strip()
            for value in _iter_validation_identity_values(source)
            if value not in (None, "", [], {})
        )
        for key in _VALIDATION_IDENTITY_KEYS:
            if target.get(key) not in (None, "", [], {}):
                continue
            value = source.get(key)
            if value not in (None, "", [], {}):
                target[key] = value
        if aliases:
            target["identity_aliases"] = sorted(aliases)

    def add_entry(
        entry: dict[str, Any],
        generation_hint: int | None = None,
        *,
        replace_existing: bool = True,
    ) -> None:
        entry_gen = entry_generation(entry, generation_hint)
        if current_gen_id is not None and (entry_gen < 0 or entry_gen > int(current_gen_id)):
            return
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}

        def pick(
            *keys: str,
            default: Any = "",
            _entry: dict[str, Any] = entry,
            _metrics: dict[str, Any] = metrics,
        ) -> Any:
            for key in keys:
                value = _entry.get(key)
                if value is None:
                    value = _metrics.get(key)
                if value is not None:
                    return _sanitize_value(value)
            return default

        metric_direction = str(entry.get("metric_direction") or "")
        if metric_direction not in {"maximize", "minimize"}:
            metric_direction = ""
        item = {
            "finding_id": entry.get("finding_id", ""),
            "variant_name": entry.get("variant_name", ""),
            "metric_name": entry.get("metric_name", ""),
            "metric_value": coerce_float(pick("metric_value", default=None)),
            "metric_direction": metric_direction,
            "metrics": _sanitize_value(metrics),
            "signal_source": entry.get("signal_source", "")
            or ("non_runtime_frontier_manifest" if manifest_signal_only else ""),
            "signal_source_priority": entry.get("signal_source_priority"),
            "generation_id": entry_gen if entry_gen >= 0 else None,
            "submitted_frontier_lane": entry.get("submitted_frontier_lane", ""),
            "matched_frontier_lanes": entry.get("matched_frontier_lanes", []),
            "signal_axis_lanes": entry.get("signal_axis_lanes", []),
            "retained_validation_lanes": entry.get("retained_validation_lanes", []),
            "evidence_stage": pick("evidence_stage", default=""),
            "evidence_maturity_rank": pick("evidence_maturity_rank", default=None),
            "scout_only": pick("scout_only", default=None),
            "evaluation_units": pick(
                "evaluation_units",
                "completed_required_eval_units",
                "actual_eval_units",
                "scored_cell_count",
                "n_scored_cells",
                "n_eval_cells",
                default=None,
            ),
            "frontier_entity_key": pick("frontier_entity_key", default=""),
            "exclusion_reason": entry.get("exclusion_reason", ""),
            "recommended_next_step": entry.get("recommended_next_step", ""),
            **{key: pick(key, default="") for key in _RESEARCH_METADATA_KEYS},
            **{key: pick(key, default="") for key in _VALIDATION_DIVERSITY_METADATA_KEYS},
        }
        if manifest_signal_only:
            item["artifact_signal_source"] = "non_runtime_frontier_manifest"
            item["artifact_signal_status"] = manifest_signal_status
            item["durability_scope"] = "validation_signal_only"
        raw_aliases = entry.get("identity_aliases")
        metric_aliases = metrics.get("identity_aliases")
        aliases = []
        for raw in (raw_aliases, metric_aliases):
            if isinstance(raw, list):
                aliases.extend(item for item in raw if item not in (None, ""))
        if aliases:
            item["identity_aliases"] = sorted({str(item).strip() for item in aliases if item})
        for key in _VALIDATION_IDENTITY_KEYS:
            if item.get(key) not in (None, "", [], {}):
                continue
            value = pick(key, default="")
            if value not in (None, "", [], {}):
                item[key] = value
        for optional_key in (
            "metric_value",
            "metrics",
            "evidence_stage",
            "evidence_maturity_rank",
            "scout_only",
            "evaluation_units",
            "frontier_entity_key",
            "submitted_frontier_lane",
            "matched_frontier_lanes",
            "signal_axis_lanes",
            "retained_validation_lanes",
            "exclusion_reason",
            "recommended_next_step",
            "artifact_signal_source",
            "artifact_signal_status",
            "durability_scope",
            "identity_aliases",
            *_VALIDATION_IDENTITY_KEYS,
            *_RESEARCH_METADATA_KEYS,
            *_VALIDATION_DIVERSITY_METADATA_KEYS,
        ):
            if item.get(optional_key) in ("", None, [], {}):
                item.pop(optional_key, None)
        key = entity_key(entry)
        if key and not item.get("frontier_entity_key"):
            item["frontier_entity_key"] = key
        item_aliases = {
            str(value).strip()
            for value in _iter_validation_retirement_values(item)
            if value not in (None, "", [], {})
        }
        if key in durable_keys or item_aliases & durable_aliases:
            return
        record_key = validation_record_key(item, key)
        incumbent = compact_by_record.get(record_key)
        if not replace_existing and incumbent is not None:
            merge_aliases(incumbent, item)
            return
        if not replace_existing and key:
            for existing in compact_by_record.values():
                if existing.get("frontier_entity_key") == key:
                    merge_aliases(existing, item)
                    return
        if incumbent is None or sort_key(item) > sort_key(incumbent):
            if incumbent is not None:
                merge_aliases(item, incumbent)
            compact_by_record[record_key] = item
        elif incumbent is not None:
            merge_aliases(incumbent, item)

    generations = raw.get("generations")
    if isinstance(generations, dict):
        for gen_key, gen_entries in sorted(
            generations.items(),
            key=lambda item: coerce_int(item[0], -1),
        ):
            generation_hint = coerce_int(gen_key, -1)
            if current_gen_id is not None and (
                generation_hint < 0 or generation_hint > int(current_gen_id)
            ):
                continue
            if not isinstance(gen_entries, list):
                continue
            for entry in gen_entries:
                if isinstance(entry, dict):
                    add_entry(entry, generation_hint=generation_hint)

    entries = raw.get("cumulative")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                add_entry(entry, replace_existing=False)

    if manifest_signal_only:
        for entry in manifest.get("cumulative_top") or []:
            if isinstance(entry, dict):
                add_entry(entry, replace_existing=False)
        signal_generations = manifest.get("generations")
        if isinstance(signal_generations, dict):
            for gen_key, gen_entries in sorted(
                signal_generations.items(),
                key=lambda item: coerce_int(item[0], -1),
            ):
                generation_hint = coerce_int(gen_key, -1)
                if not isinstance(gen_entries, list):
                    continue
                for entry in gen_entries:
                    if isinstance(entry, dict):
                        add_entry(entry, generation_hint=generation_hint, replace_existing=False)
        signal_lanes = manifest.get("lane_frontiers")
        if isinstance(signal_lanes, dict):
            for lane_name, lane_entries in signal_lanes.items():
                if not isinstance(lane_entries, list):
                    continue
                for entry in lane_entries:
                    if not isinstance(entry, dict):
                        continue
                    candidate = dict(entry)
                    candidate.setdefault("submitted_frontier_lane", lane_name)
                    add_entry(candidate, replace_existing=False)

    return select_validation_candidates(list(compact_by_record.values()))


def _validation_candidate_aliases_from_manifest(
    run_dir: Path,
    *,
    current_gen_id: int | None = None,
) -> set[str]:
    raw = _load_frontier_manifest_for_context(
        run_dir,
        purpose="validation candidate aliases",
        allow_signal_source=True,
    )
    if not raw:
        return set()
    trust_committed_membership = is_committed_runtime_fact_source(raw, legacy_ok=False)
    validation = raw.get("validation_candidates")
    if not isinstance(validation, dict):
        return set()
    compact_validation = _digest_validation_candidates(
        run_dir,
        current_gen_id=current_gen_id,
        max_entries=10_000,
    )
    active_candidate_aliases: set[str] = set()
    for entry in compact_validation:
        active_candidate_aliases.update(
            str(value).strip()
            for value in _iter_validation_identity_values(entry)
            if value not in (None, "", [], {})
        )

    def coerce_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def coerce_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def entry_generation(entry: dict[str, Any], generation_hint: int | None) -> int:
        if trust_committed_membership:
            explicit = explicit_entry_generation_id(
                entry,
                generation_hint=generation_hint,
            )
            if explicit is not None:
                return explicit
        return coerce_int(
            entry.get("generation_id"),
            generation_hint if generation_hint is not None else -1,
        )

    def is_durable_entry(entry: dict[str, Any]) -> bool:
        if trust_committed_membership:
            return True
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                _is_committed_frontier_entry,
            )

            return _is_committed_frontier_entry(entry)
        except (AttributeError, ImportError):
            logger.debug("falling back to local durable alias check", exc_info=True)
        return isinstance(entry, dict) and has_complete_evidence(entry)

    def entity_key(entry: dict[str, Any]) -> str:
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                _candidate_entity_key,
            )

            return _candidate_entity_key(entry)
        except (AttributeError, ImportError):
            logger.debug("falling back to local durable alias key", exc_info=True)
        key = str(entry.get("frontier_entity_key") or "").strip()
        if key:
            return key
        variant = str(entry.get("variant_name") or "").strip()
        return f"variant::{variant}" if variant else ""

    def gem_identity_entry(entry: dict[str, Any]) -> dict[str, Any]:
        payload = dict(entry)
        admission = entry.get("admission_metrics")
        metrics = dict(admission) if isinstance(admission, dict) else {}
        existing_metrics = entry.get("metrics")
        if isinstance(existing_metrics, dict):
            metrics.update(existing_metrics)
        if metrics:
            payload["metrics"] = metrics
        return payload

    def has_complete_evidence(entry: dict[str, Any]) -> bool:
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        for key in ("complete_eval", "is_complete_eval", "scored_complete", "is_scored_complete"):
            value = entry.get(key, metrics.get(key))
            if isinstance(value, bool):
                if value:
                    return True
                continue
            text = str(value or "").strip().lower()
            if text in {"true", "yes", "1", "complete", "completed", "confirmed"}:
                return True
        for key in (
            "evidence_stage",
            "tier",
            "tier_reached",
            "completed_tier",
            "candidate_tier",
            "tier_status",
            "final_status",
            "result_status",
            "completion_status",
            "eval_status",
        ):
            normalized = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(entry.get(key) or metrics.get(key) or "").strip().lower(),
            ).strip("_")
            if normalized in {
                "scored_complete",
                "complete_eval",
                "complete_eval_true",
                "is_complete_eval_true",
                "scored_complete_true",
                "is_scored_complete_true",
                "full_eval",
                "full_evaluation",
            }:
                return True
        return False

    def has_nonclean_gem_marker(entry: dict[str, Any]) -> bool:
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        for key in ("promotion_eligible", "clean_promotion_eligible"):
            value = entry.get(key, metrics.get(key))
            if isinstance(value, bool) and not value:
                return True
            text = str(value or "").strip().lower()
            if text in {"false", "no", "0", "ineligible", "nonpromotable", "non_promotable"}:
                return True
        for key in (
            "hard_constraint_violation_count",
            "hard_constraint_violations",
            "hard_constraint_failures",
            "constraint_violation_count",
            "constraint_violations",
        ):
            value = entry.get(key, metrics.get(key))
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (list, tuple, set, dict)):
                if len(value) > 0:
                    return True
                continue
            if isinstance(value, (int, float)):
                if value > 0:
                    return True
                continue
            text = str(value).strip().lower()
            if text and text not in {"0", "none", "no", "false", "[]", "{}"}:
                return True
        return False

    def is_legacy_persisted_gem_entry(entry: dict[str, Any]) -> bool:
        if not entry.get("gem_finding_id") or not isinstance(entry.get("admission_metrics"), dict):
            return False
        payload = gem_identity_entry(entry)
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        for cell_key in (
            "scored_cell_count",
            "n_scored_cells",
            "n_eval_cells",
            "cell_count",
            "n_cells",
        ):
            if coerce_int(payload.get(cell_key, metrics.get(cell_key)), 0) > 0:
                return False
        for key in (
            *legacy_primary_metric_keys(),
            "score",
            "metric_value",
            "lane_metric_value",
        ):
            if coerce_float(payload.get(key, metrics.get(key))) is not None:
                return True
        return False

    def is_durable_gem_entry(entry: dict[str, Any]) -> bool:
        if trust_committed_membership:
            return True
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.gems import (
                _entry_has_complete_eval_evidence,
                _entry_has_explicit_scout_or_partial_marker,
                _entry_has_nonclean_gem_marker,
            )

            return (
                not _entry_has_explicit_scout_or_partial_marker(entry)
                and not _entry_has_nonclean_gem_marker(entry)
                and _entry_has_complete_eval_evidence(entry)
            )
        except (AttributeError, ImportError):
            logger.debug("falling back to durable Gem alias check", exc_info=True)
        payload = gem_identity_entry(entry)
        return (
            isinstance(entry.get("admission_metrics"), dict)
            and not has_nonclean_gem_marker(payload)
            and (has_complete_evidence(payload) or is_legacy_persisted_gem_entry(entry))
        )

    def gem_has_acceptable_source_generation(entry: dict[str, Any]) -> bool:
        if current_gen_id is None:
            return True
        source_generation = (
            _resolved_persisted_gem_source_generation_id(
                run_dir,
                entry,
                allow_identity_inference=False,
            )
            if trust_committed_membership
            else _resolved_persisted_gem_source_generation_id(run_dir, entry)
        )
        if source_generation is not None:
            return source_generation <= int(current_gen_id)
        return bool(
            is_legacy_persisted_gem_entry(entry)
            and (entry.get("variant_name") or entry.get("variant_id"))
        )

    durable_aliases: set[str] = set()
    durable_keys: set[str] = set()
    raw_signal_only = bool(raw.get("_non_runtime_signal_source"))

    def add_durable_aliases(entries: Any, generation_hint: int | None = None) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            gen_id = entry_generation(entry, generation_hint)
            if current_gen_id is not None and (gen_id < 0 or gen_id > int(current_gen_id)):
                continue
            if not is_durable_entry(entry):
                continue
            durable_aliases.update(
                str(value).strip()
                for value in _iter_validation_retirement_values(entry)
                if value not in (None, "", [], {})
            )
            key = entity_key(entry)
            if key:
                durable_keys.add(key)
                durable_aliases.add(key)

    if not raw_signal_only:
        add_durable_aliases(raw.get("cumulative_top"))
        generations = raw.get("generations")
        if isinstance(generations, dict):
            for gen_key, entries in generations.items():
                add_durable_aliases(entries, generation_hint=coerce_int(gen_key, -1))
        lane_frontiers = raw.get("lane_frontiers")
        if isinstance(lane_frontiers, dict):
            for entries in lane_frontiers.values():
                add_durable_aliases(entries)
        gems = raw.get("gems")
        gem_entries = gems.get("entries") if isinstance(gems, dict) else None
        if isinstance(gem_entries, list):
            for entry in gem_entries:
                if not isinstance(entry, dict) or not is_durable_gem_entry(entry):
                    continue
                payload = gem_identity_entry(entry)
                if not gem_has_acceptable_source_generation(entry):
                    continue
                durable_aliases.update(
                    str(value).strip()
                    for value in _iter_validation_retirement_values(payload)
                    if value not in (None, "", [], {})
                )
                key = entity_key(payload)
                if key:
                    durable_keys.add(key)
                    durable_aliases.add(key)
    stale_validation_aliases: set[str] = set()

    def add_stale_validation_aliases(entries: Any, generation_hint: int | None = None) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            gen_id = entry_generation(entry, generation_hint)
            if current_gen_id is not None and (gen_id < 0 or gen_id > int(current_gen_id)):
                continue
            entry_aliases = {
                str(value).strip()
                for value in _iter_validation_retirement_values(entry)
                if value not in (None, "", [], {})
            }
            if entity_key(entry) not in durable_keys and not (entry_aliases & durable_aliases):
                continue
            stale_validation_aliases.update(
                str(value).strip()
                for value in _iter_validation_identity_values(entry)
                if value not in (None, "", [], {})
            )

    validation_generations = validation.get("generations")
    if isinstance(validation_generations, dict):
        for gen_key, entries in validation_generations.items():
            add_stale_validation_aliases(entries, generation_hint=coerce_int(gen_key, -1))
    add_stale_validation_aliases(validation.get("cumulative"))
    aliases: set[str] = set()
    by_generation = validation.get("validator_identity_aliases_by_generation")
    if isinstance(by_generation, dict):
        for gen_key, values in by_generation.items():
            try:
                gen_id = int(gen_key)
            except (TypeError, ValueError):
                continue
            if current_gen_id is not None and gen_id > int(current_gen_id):
                continue
            if isinstance(values, list):
                aliases.update(str(value).strip() for value in values if value not in (None, ""))
    values = validation.get("validator_identity_aliases")
    if current_gen_id is None and isinstance(values, list):
        aliases.update(str(value).strip() for value in values if value not in (None, ""))
    aliases.update(active_candidate_aliases)
    retired_aliases = durable_aliases | stale_validation_aliases
    return {alias for alias in aliases if alias and alias not in retired_aliases}


def _write_validation_candidates_artifact(
    run_dir: Path,
    *,
    current_gen_id: int,
    validation_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not validation_candidates:
        return None
    rel_path = Path("research_memory") / "validation_candidates" / f"gen_{current_gen_id}_full.json"
    path = Path(run_dir) / rel_path
    payload = {
        "generation_id": int(current_gen_id),
        "total": len(validation_candidates),
        "validation_candidates": validation_candidates,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(_sanitize_value(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError as exc:
        logger.warning("could not write validation candidates artifact %s: %s", path, exc)
        return None
    return {
        "result_path": str(rel_path),
        "generation_id": int(current_gen_id),
        "total": len(validation_candidates),
    }


def _digest_frontier_lane_metadata(
    run_dir: Path,
    *,
    total_entries_by_lane: dict[str, int] | None = None,
    returned_entries_by_lane: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Return compact task-defined lane metadata for PI/Chair interpretation."""
    manifest = _load_frontier_manifest_for_context(
        run_dir,
        purpose="frontier lane metadata",
        allow_signal_source=True,
    )
    if not manifest:
        return []
    raw_lanes = manifest.get("frontier_lanes")
    if not isinstance(raw_lanes, list):
        return []

    compact: list[dict[str, Any]] = []
    for lane in raw_lanes:
        if not isinstance(lane, dict):
            continue
        item = {
            "name": lane.get("name", ""),
            "description": lane.get("description", ""),
            "include_lanes": lane.get("include_lanes", []),
            "k": lane.get("k"),
            "cumulative_cap": lane.get("cumulative_cap"),
            "require_metrics": lane.get("require_metrics", []),
            "require_truthy_metrics": lane.get("require_truthy_metrics", []),
            "require_falsey_metrics": lane.get("require_falsey_metrics", []),
            "min_metrics": lane.get("min_metrics", {}),
            "max_metrics": lane.get("max_metrics", {}),
            "exclude_roles": lane.get("exclude_roles", []),
            "exclude_families": lane.get("exclude_families", []),
            "exclude_tags": lane.get("exclude_tags", []),
            "axes": lane.get("axes", []),
            "optional_axes": lane.get("optional_axes", []),
            "allow_lower_tier": bool(lane.get("allow_lower_tier", False)),
            "parent_eligible": bool(
                lane.get(
                    "parent_eligible",
                    not bool(lane.get("allow_lower_tier", False)),
                )
            ),
            "allow_non_promotable": bool(lane.get("allow_non_promotable", False)),
            "allow_missing_tier": bool(lane.get("allow_missing_tier", False)),
            "allow_risk_violating": bool(lane.get("allow_risk_violating", False)),
            "requires_tier": lane.get("requires_tier"),
            "requires_promotion_eligible": lane.get("requires_promotion_eligible"),
            "filters": lane.get("filters", {}),
        }
        lane_name = str(lane.get("name") or "")
        if total_entries_by_lane is not None:
            total = int(total_entries_by_lane.get(lane_name, 0))
            returned = len((returned_entries_by_lane or {}).get(lane_name, []))
            item.update(
                {
                    "available_entry_count": total,
                    "returned_entry_count": returned,
                    "entries_truncated": total > returned,
                }
            )
        compact.append(item)
    return compact


def _digest_gems(
    run_dir: Path,
    max_entries: int | None = None,
    current_gen_id: int | None = None,
) -> dict[str, Any]:
    """Return compact durable Gems context for PI/Chair shared core."""
    filtered = load_active_gems_for_prompt(
        Path(run_dir),
        max_entries=max_entries,
        max_generation_id=current_gen_id,
    )
    if not filtered:
        return {}
    compact = []
    for entry in filtered.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        item = {
            "gem_finding_id": entry.get("gem_finding_id", ""),
            "variant_name": entry.get("variant_name", ""),
            "frontier_lane": entry.get("frontier_lane", ""),
            "metric_name": entry.get("metric_name", ""),
            "metric_value": entry.get("metric_value"),
            "source_finding_id": entry.get("source_finding_id", ""),
            "source_generation_id": entry.get("source_generation_id"),
            "strategy_family": entry.get("strategy_family", ""),
            "mechanism_family": entry.get("mechanism_family", ""),
            "innovation_surface": entry.get("innovation_surface", ""),
            **{key: entry.get(key, "") for key in _RESEARCH_METADATA_KEYS},
            **{key: entry.get(key, "") for key in _VALIDATION_DIVERSITY_METADATA_KEYS},
            "gem_variant_ref": entry.get("gem_variant_ref", ""),
            "finding_path": entry.get("finding_path", ""),
            "admission_metrics": entry.get("admission_metrics", {}) or {},
        }
        compact.append(item)
    return {
        "cycle_index": filtered.get("cycle_index", 0),
        "reset_count": filtered.get("reset_count", 0),
        "cycle_start_generation": filtered.get("cycle_start_generation", 0),
        "entries": compact,
        "bottleneck_reports": list(filtered.get("bottleneck_reports", []) or [])[-5:],
        "latest_soft_agenda_priors": filtered.get("latest_soft_agenda_priors", {}) or {},
    }


def _digest_role_roi(role_roi: RoleROILedger, current_gen_id: int) -> dict[str, Any]:
    """Return the most recent recorded gen ROI summary, plus deltas.

    R7#8 fix: when no entries exist (e.g., research_memory.enabled=False
    or first ever synthesis), emit an explicit note so PI panels know
    the empty role data is structural, not a sign of zero peer activity.
    """
    all_entries = role_roi.all()
    if not all_entries:
        return {
            "note": "no role_roi entries yet (first synthesis or research_memory disabled)",
            "per_role": {},
        }
    all_entries = [
        entry for entry in all_entries if _within_generation_cutoff(entry, current_gen_id)
    ]
    if not all_entries:
        return {
            "note": "no role_roi entries available at this generation cutoff",
            "per_role": {},
        }
    all_entries.sort(key=lambda e: e.data.get("generation_id", 0), reverse=True)
    return {
        "latest_recorded_gen": all_entries[0].data.get("generation_id"),
        "per_role": all_entries[0].data.get("per_role", {}),
    }


def build_shared_core(
    run_dir: Path,
    panel_mode: str,
    current_gen_id: int,
    target_decisions: list[str],
    claim_ledger: ClaimLedger,
    frontier_delta_ledger: FrontierDeltaLedger,
    coverage_matrix: CoverageMatrix,
    negative_evidence_ledger: NegativeEvidenceLedger,
    retired_claim_ledger: RetiredClaimLedger,
    dissent_ledger: DissentLedger,
    role_roi_ledger: RoleROILedger,
    findings_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the digest that ALL PIs see verbatim."""
    scoreless = (findings_summary or {}).get("research_loop_mode") == "scoreless"
    bridge_grids = []
    grid_summaries = []
    for e in coverage_matrix.all():
        if not _within_generation_cutoff(e, current_gen_id):
            continue
        d = e.data
        if d.get("relation") == "bridge":
            bridge_grids.append(
                {
                    "pair": d.get("variant_pair"),
                    "dimension": d.get("grid_dimension"),
                    "bridge_points_tested": d.get("bridge_points_tested", []),
                }
            )
        else:
            grid_summaries.append(
                {
                    "variant_family": d.get("variant_family"),
                    "parameter": d.get("parameter"),
                    "values_tested": d.get("values_tested", []),
                }
            )

    validation_candidates = (
        []
        if scoreless
        else _digest_validation_candidates(
            run_dir,
            max_entries=_VALIDATION_CANDIDATES_SHARED_CORE_CAP,
            current_gen_id=current_gen_id,
        )
    )
    all_validation_candidates = (
        []
        if scoreless
        else _digest_validation_candidates(
            run_dir,
            max_entries=10_000,
            current_gen_id=current_gen_id,
        )
    )
    total_validation_candidates = len(all_validation_candidates)
    validation_candidates_source_ref = (
        ""
        if scoreless
        else _write_validation_candidates_artifact(
            run_dir,
            current_gen_id=current_gen_id,
            validation_candidates=all_validation_candidates,
        )
    )
    validation_candidate_ids = (
        []
        if scoreless
        else sorted(
            _validation_candidate_aliases_from_manifest(
                run_dir,
                current_gen_id=current_gen_id,
            )
            | {
                str(value).strip()
                for entry in all_validation_candidates
                if isinstance(entry, dict)
                for value in _iter_validation_identity_values(entry)
                if value not in (None, "", [], {})
            }
        )
    )

    lane_entry_counts: dict[str, int] = {}
    lane_frontiers = (
        {}
        if scoreless
        else _digest_lane_frontiers(
            run_dir,
            current_gen_id=current_gen_id,
            total_entries_by_lane=lane_entry_counts,
        )
    )
    shared_core = {
        "shared_core_id": "",  # filled below
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "panel_mode": panel_mode,
        "run_metadata": {
            "current_gen_id": current_gen_id,
            "target_decisions": list(target_decisions),
        },
        "current_frontier": {}
        if scoreless
        else _digest_frontier(frontier_delta_ledger, current_gen_id),
        "current_frontier_scope": "not_applicable"
        if scoreless
        else "latest_per_axis_generation_delta_anchors",
        "lane_frontiers": lane_frontiers,
        "validation_candidates": validation_candidates,
        "validation_candidates_meta": {
            "total": total_validation_candidates,
            "returned": len(validation_candidates),
            "truncated": total_validation_candidates > len(validation_candidates),
            "cap": _VALIDATION_CANDIDATES_SHARED_CORE_CAP,
            "selection_policy": "score_floor_plus_diversity_facets",
            "validator_id_count": len(validation_candidate_ids),
            "full_source_ref": validation_candidates_source_ref,
        },
        "gems": {} if scoreless else _digest_gems(run_dir, current_gen_id=current_gen_id),
        "frontier_lane_metadata": {}
        if scoreless
        else _digest_frontier_lane_metadata(
            run_dir,
            total_entries_by_lane=lane_entry_counts,
            returned_entries_by_lane=lane_frontiers,
        ),
        "claim_ledger_digest": _digest_claims(claim_ledger, current_gen_id=current_gen_id),
        "retired_claims": [
            {
                "id": e.id,
                "title": e.data.get("title", ""),
                "boundary": e.data.get("boundary", ""),
                "scope": e.data.get("scope", {}),
                "revive_if": e.data.get("revive_if", []),
            }
            for e in retired_claim_ledger.all()
            if _within_generation_cutoff(e, current_gen_id)
        ],
        "open_objections": [
            {
                "id": e.id,
                "disputed_claim_id": e.data.get("disputed_claim_id"),
                "status": e.data.get("status"),
                "resolving_experiment": e.data.get("resolving_experiment", ""),
            }
            for e in dissent_ledger.list_open()
            if _within_generation_cutoff(e, current_gen_id)
        ],
        "coverage_matrix_digest": {
            "single_family_grids": grid_summaries[:30],
            "bridge_grids": bridge_grids[:20],
        },
        "negative_evidence_digest": [
            {
                "id": e.id,
                "title": e.data.get("title", ""),
                "category": e.data.get("category", ""),
                "summary": e.data.get("summary", "")[:200],
            }
            for e in negative_evidence_ledger.list_recent(n=60)
            if _within_generation_cutoff(e, current_gen_id)
        ][:12],
        "role_performance": {} if scoreless else _digest_role_roi(role_roi_ledger, current_gen_id),
        "findings_summary": findings_summary or {},
    }

    if scoreless:
        evidence = load_scoreless_evidence(run_dir, current_gen_id)
        retained_ids = {row.get("id") for row in evidence if row.get("id")}
        graph_edges = []
        raw_edges = (findings_summary or {}).get("advisory_graph_edges")
        for edge in (raw_edges if isinstance(raw_edges, list) else [])[:100]:
            if not isinstance(edge, dict) or not all(
                edge.get(key) in retained_ids for key in ("src_finding_id", "dst_finding_id")
            ):
                continue
            graph_edges.append(
                {
                    key: value[:1200] if isinstance(value, str) else _sanitize_value(value)
                    for key in (
                        "edge_id",
                        "src_finding_id",
                        "dst_finding_id",
                        "edge_type",
                        "confidence",
                        "created_by",
                        "rationale",
                    )
                    if (value := edge.get(key)) is not None
                }
            )
        shared_core.update(
            {
                "research_loop_mode": "scoreless",
                "evidence_status": "not_scored",
                "scoreless_evidence": evidence,
                "advisory_graph_edges": graph_edges,
                "graph_evidence_scope": "advisory_edges_between_retained_findings",
                "research_memory_scope": "advisory_generation_filtered_ledgers_not_frozen_evidence",
                "validation_candidates_meta": {"status": "not_applicable"},
                "findings_summary": {
                    key: value
                    for key, value in (findings_summary or {}).items()
                    if key not in {"scoreless_evidence", "advisory_graph_edges"}
                },
            }
        )

    # stable id for caching / cross-PI consistency check
    blob = json.dumps(shared_core, sort_keys=True, default=str).encode("utf-8")
    shared_core["shared_core_id"] = hashlib.sha256(blob).hexdigest()[:16]
    return shared_core


# ---------------------------------------------------------------------------
# Role-specific private packs


def _role_filter(role: str, card: dict[str, Any]) -> float:
    """Return a relevance score (0..1) of a card for a given PI role."""
    score = 0.5  # base
    is_neg = bool(card.get("quality", {}).get("is_negative"))
    is_retired = bool(card.get("quality", {}).get("is_retired"))
    promotion_eligible = bool(card.get("metrics", {}).get("promotion_eligible"))
    # R8#2 fix: defensive guard against interpretation.short being None
    # or interpretation being a non-dict.
    interp_dict = card.get("interpretation") or {}
    if isinstance(interp_dict, dict):
        short_raw = interp_dict.get("short")
    else:
        short_raw = ""
    interp = (short_raw or "").lower() if isinstance(short_raw, str) else ""

    if role == "builder":
        # Builder wants successful lineages
        if promotion_eligible and not is_neg:
            score += 0.3
        if "synergy" in interp or "scaling" in interp or "champion" in interp:
            score += 0.1
    elif role == "skeptic":
        if is_neg:
            score += 0.3
        if "baseline" in interp or "fairness" in interp or "control" in interp:
            score += 0.2
        if promotion_eligible:
            score += 0.1  # skeptic still wants the strong claims to attack
    elif role == "portfolio":
        if is_neg or is_retired:
            score += 0.25
        if "anti_mainline" in interp or "online" in interp or "random" in interp:
            score += 0.2
        if "bridge" in interp:
            score += 0.1
    elif role == "external_validity":
        if "cross" in interp or "sentinel" in interp or "long" in interp:
            score += 0.4
        if is_neg:
            score += 0.1
    return max(0.0, min(1.0, score))


def build_role_private_pack(
    role: str,
    cards: list[dict[str, Any]],
    shared_core: dict[str, Any],
    panel_mode: str,
    max_cards: int,
) -> list[dict[str, Any]]:
    """Build the evidence pack visible to one PI role under the retrieval policy."""
    if not cards:
        return []
    # rank by role relevance
    scored = [(c, _role_filter(role, c)) for c in cards if isinstance(c, dict)]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, _ in scored[: max_cards * 3]]  # candidate pool
    mix = HIGH_STAKES_MIX if panel_mode == "high_stakes" else NORMAL_MIX
    return select_cards_with_mix(top, mix, max_cards, high_stakes=(panel_mode == "high_stakes"))


# ---------------------------------------------------------------------------
# Top-level


def build_evidence_pack(
    run_dir: Path,
    panel_mode: str,
    current_gen_id: int,
    target_decisions: list[str],
    pi_roles: list[str],
    max_cards_total: int = 40,
    max_cards_per_pack: int = 25,
    findings_summary: dict[str, Any] | None = None,
) -> EvidencePack:
    """Assemble the complete evidence pack."""
    run_dir = Path(run_dir)
    scoreless = (findings_summary or {}).get("research_loop_mode") == "scoreless"
    canonical_sources = (
        ["gen_*/scoreless_evidence.json", "research_memory/*"]
        if scoreless
        else [
            "frontier/frontier_manifest.json",
            "shared_store.db",
            "research_memory/*",
            "gems/gems_state.json",
            "shared_findings/*",
            "results/*",
        ]
    )

    # Load all ledgers
    claim_ledger = ClaimLedger(run_dir)
    coverage = CoverageMatrix(run_dir)
    neg = NegativeEvidenceLedger(run_dir)
    retired = RetiredClaimLedger(run_dir)
    dissent = DissentLedger(run_dir)
    fd = FrontierDeltaLedger(run_dir)
    role_roi = RoleROILedger(run_dir)

    shared_core = build_shared_core(
        run_dir,
        panel_mode,
        current_gen_id,
        target_decisions,
        claim_ledger,
        fd,
        coverage,
        neg,
        retired,
        dissent,
        role_roi,
        findings_summary=findings_summary,
    )
    shared_core = attach_artifact_semantics(
        shared_core,
        role=DERIVED_AUDIT_SNAPSHOT,
        stage="pi_evidence_pack_shared_core",
        generation_id=current_gen_id,
        actor="research_loop:pi_panel",
        canonical_sources=canonical_sources,
        runtime_fact_source=False,
        notes=(
            "Scoreless evidence is read from frozen generation manifests; research-memory "
            "ledgers and graph edges remain advisory. Retention does not establish quality."
            if scoreless
            else "PI evidence packs are regenerated from canonical frontier, findings, "
            "Gems, and research-memory state. Persisted packs are audit snapshots, "
            "not current fact owners."
        ),
    )

    # Build cards from existing DB. In Phase >0, card_builder may also pull
    # from a dedicated cards file; for now we rebuild on demand.
    all_cards = [] if scoreless else build_cards_from_db(run_dir, max_gen=current_gen_id)

    private_packs: dict[str, list[dict[str, Any]]] = {}
    for role in pi_roles:
        private_packs[role] = build_role_private_pack(
            role,
            all_cards,
            shared_core,
            panel_mode,
            max_cards_per_pack,
        )

    pack_id = "EP::" + shared_core["shared_core_id"]
    audit = attach_artifact_semantics(
        {
            "pack_id": pack_id,
            "shared_core_id": shared_core["shared_core_id"],
            "n_cards_total": len(all_cards),
            "private_pack_sizes": {k: len(v) for k, v in private_packs.items()},
            "negative_evidence_ratio_global": None
            if scoreless
            else negative_evidence_ratio(all_cards),
            "validation_candidate_ids": []
            if scoreless
            else sorted(
                _validation_candidate_aliases_from_manifest(
                    run_dir,
                    current_gen_id=current_gen_id,
                )
                | {
                    str(value).strip()
                    for entry in _digest_validation_candidates(
                        run_dir,
                        max_entries=10_000,
                        current_gen_id=current_gen_id,
                    )
                    if isinstance(entry, dict)
                    for value in _iter_validation_identity_values(entry)
                    if value not in (None, "", [], {})
                }
            ),
        },
        role=DERIVED_AUDIT_SNAPSHOT,
        stage="pi_evidence_pack",
        generation_id=current_gen_id,
        actor="research_loop:pi_panel",
        canonical_sources=canonical_sources,
        runtime_fact_source=False,
    )
    # R3#6 fix: sanitize NaN/Inf out of every dict the templates will see.
    shared_core = _sanitize_value(shared_core)
    private_packs = {k: _sanitize_value(v) for k, v in private_packs.items()}
    pack = EvidencePack(
        pack_id=pack_id,
        built_at=shared_core["built_at"],
        panel_mode=panel_mode,
        target_decisions=list(target_decisions),
        shared_core=shared_core,
        private_packs=private_packs,
        all_cards=_sanitize_value(all_cards),
        audit=audit,
    )
    return pack
