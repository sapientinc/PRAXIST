"""
Filesystem → SQLite finding ingestion.

Problem this solves (evidence from the 2026-04-17 post-hardening run):
    Agents have two ways to publish a finding:
      1. share_finding MCP tool  → writes filesystem AND SQLite (dual-write)
      2. Write tool → shared_findings/*.json → only filesystem

    In the first 17 hours of the post-hardening run, peers published 289
    findings via path (2) but only 2 via path (1). The orchestrator's
    frontier.promote() only sees SQLite, so the 287 missing findings meant
    frontier/ stayed empty and Gen 1 had no seed data to build from.

This module closes the loop: it scans shared_findings/*.json and upserts
any file not yet present in SQLite. Called from FindingsSync's daemon
cycle and from GenerationLoop._collect_findings_for_generation before
any SQLite read.

Design goals:
  - Idempotent: re-running ingest leaves the DB unchanged.
  - Tolerant: agent-written files have inconsistent schemas; we extract
    what we can (metrics by walking nested JSON, finding_type by
    heuristics on title/content).
  - Stable identity: a given filename always maps to the same SQLite id,
    even if the file is rewritten. UUID-prefixed filenames (from the
    share_finding path) preserve their original uuid; agent-written
    files get a deterministic "fs_<hash>" id derived from the filename.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.core.controller_state import controller_state_enabled
from praxist.core.storage import read_file_bytes
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    CANONICAL_STATE,
    COMMITTED,
    DERIVED_VIEW,
    PARTIAL,
)
from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    result_effective_config_metadata,
    strip_effective_config_fields,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    _RESULT_ARTIFACT_PATH_KEYS,
    _RESULT_PRODUCER_IDENTITY_KEYS,
    compact_result_identity_container,
    result_artifact_key,
)

logger = logging.getLogger(__name__)
_INGEST_SCHEMA_VERSION = 5

# UUID v4 regex for detecting share_finding-produced filenames like
# "84d85198-d1b1-4290-a8c0-0b667a4ef593_Session_12_....json".
_UUID_PATTERN = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_GEM_ID_PATTERN = re.compile(r"^gem_r\d{2}_\d{2}_[A-Za-z0-9_.-]{1,120}$")


def _is_system_result_reference(content: dict[str, Any]) -> bool:
    metrics = content.get("metrics")
    semantics = content.get("artifact_semantics")
    if not isinstance(metrics, dict) or not isinstance(semantics, dict):
        return False
    source_path = str(metrics.get("source_result_path") or "").strip()
    canonical_sources = semantics.get("canonical_sources")
    return bool(
        source_path
        and metrics.get("auto_materialized_from_result_artifact") is True
        and semantics.get("role") == DERIVED_VIEW
        and semantics.get("status") == COMMITTED
        and semantics.get("stage") == "result_finding_reference"
        and semantics.get("actor") == "research_loop:findings_collection"
        and isinstance(canonical_sources, list)
        and source_path in canonical_sources
    )


def _result_reference_effective_config_metadata(
    filepath: Path,
    content: dict[str, Any],
    *,
    maturity_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not _is_system_result_reference(content):
        return {}
    metrics = content.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    source_path = Path(str(metrics.get("source_result_path")))
    if (
        source_path.is_absolute()
        or ".." in source_path.parts
        or source_path.parts[:1] != ("results",)
    ):
        return {}
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        _slug,
        is_supported_result_summary_filename,
        normalized_result_summary,
        result_summary_control_digest,
        result_summary_variant_name,
    )

    if not is_supported_result_summary_filename(source_path.name):
        return {}
    summary_path = filepath.parent.parent / source_path
    try:
        summary = json.loads(read_file_bytes(summary_path).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(summary, dict):
        return {}
    summary = normalized_result_summary(
        summary,
        summary_path=summary_path,
        maturity_policy=maturity_policy,
    )
    if not (
        isinstance(summary.get("current_aggregate"), dict)
        and (summary.get("variant_name") or source_path.parent.name)
    ):
        return {}
    run_dir = filepath.parent.parent
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_dir.resolve()}::{source_path}"))
    variant = result_summary_variant_name(summary_path, summary, run_dir).strip()
    metrics = content.get("metrics")
    if (
        str(content.get("id") or "").strip() != expected_id
        or filepath.name != f"{expected_id}_{_slug(variant)}.json"
        or str(content.get("variant_name") or "").strip() != variant
        or not isinstance(metrics, dict)
        or str(metrics.get("source_result_sha256") or "").strip()
        != result_summary_control_digest(summary)
    ):
        return {}
    return result_effective_config_metadata(summary)


def sanitize_finding_effective_config_provenance(
    filepath: Path,
    finding: dict[str, Any],
    *,
    maturity_policy: dict[str, Any] | None = None,
) -> None:
    """Keep only metadata recomputed from a referenced evaluator summary."""

    strip_effective_config_fields(finding)
    metadata = _result_reference_effective_config_metadata(
        filepath,
        finding,
        maturity_policy=maturity_policy,
    )
    metrics = finding.get("metrics")
    if metadata and isinstance(metrics, dict):
        metrics.update(metadata)


# Metric key aliases. First match wins during the nested walk.
# NOTE: we deliberately do NOT list dataset-specific aliases here — those
# are handled by infer_dataset() + canonical key assignment in extract_metrics().
_ACC_KEY_ALIASES = (
    "test_accuracy",
    "test_acc",
    "best_test_acc",
    "final_test_acc",
    "mean_acc",
    "acc",
    "accuracy",
)
_GAP_KEY_ALIASES = (
    "train_test_gap",
    "generalization_gap",
    "test_gap",
    "gap",
)

# NOTE: ``method`` and ``optimizer`` are deliberately excluded. In practice
# peers use these nested fields to describe task-local procedures, not the
# variant under study. Generic ingest should only infer identity from explicit
# variant fields or bounded title tokens.
_VARIANT_KEY_ALIASES = ("variant", "variant_name")

# Last-resort: extract a compact uppercase variant token from the title.
# Matches tokens joined by hyphens and ending in a short all-caps method suffix.
# Rejects noise like "DEFINITIVE", "SESSION18", "GEN0" that a naive leading-
# caps regex would grab.
_VARIANT_TITLE_RX = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*){1,4})\b")

# Frontier lane promotion needs a few categorical fields from ``metrics``.
# The ingest path used to keep only numeric metric values plus tier metadata;
# that stripped fields such as task lane and strategy-family labels before
# frontier promotion read from SQLite.
# Keep this allow-list narrow so arbitrary prose does not enter the metric bag.
_IDENTITY_METRIC_KEYS = {
    "frontier_entity_key",
    "candidate_entity_key",
    "variant_id",
    "canonical_variant_id",
    "child_id",
    "child_variant_id",
    "result_variant_id",
    "sweep_child_id",
    "trial_id",
    "child_variant_name",
    "result_variant_name",
    "canonical_variant_name",
    "reported_variant_name",
    "source_result_path",
    "source_result_config_sha256",
    "source_result_effective_config_sha256",
    "replication_of_effective_config_sha256",
    "result_artifact_path",
    "result_path",
    "summary_path",
    "source_path",
    "gem_variant_ref",
    "identity_aliases",
}

_DIVERSITY_METADATA_KEYS = {
    "diversity_overlap_status",
    "diversity_most_similar_anchor",
    "diversity_violation",
    "diversity_violated",
    "diversity_narrow_variation",
    "diversity_overlap_no_data_reason",
    "mechanism_family",
    "intervention_surface",
    "intent",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
}

_NUMERIC_PRESERVED_METRIC_KEYS = {
    "diversity_overlap_score",
    "diversity_overlap_fraction",
    "diversity_overlap_count",
    "diversity_overlap_total",
    "effort_ratio",
    "coverage_ratio",
    "eval_coverage_ratio",
    "evidence_coverage_ratio",
    "compute_effort_ratio",
    "full_eval_effort_ratio",
}

_CATEGORICAL_METRIC_KEYS = (
    {
        "frontier_lane",
        "promotion_lane",
        "lane",
        "strategy_family",
        "family",
        "variant_family",
        "benchmark_type",
        "tier_reached",
        "tier_status",
        "candidate_tier",
        "final_status",
        "robustness_label",
        "historical_promotion_status",
        "bottleneck_target",
        "evidence_stage",
        "evidence_valence",
        "failure_mode",
        "diagnostic_role",
        "tradeoff_class",
        "primary_tradeoff",
        "next_step_intent",
        "parent_candidate",
        "parent_usage",
        "source_lane",
        "target_lane",
        "coverage_check",
        "mechanism_hypothesis_deliverable",
        "result_status",
        "protocol_integrity_status",
        "source_result_path",
        "source_result_kind",
        "source_result_sha256",
        "source_result_effective_config_status",
        "replication_effective_config_status",
        "source_generation_inference",
        "artifact_signal_status",
        "generation_boundary_path",
        "generation_boundary_mtime",
        "generation_boundary_evidence_cutoff_at",
        "source_result_mtime",
        "late_result_policy",
        "durability_scope",
        "exclusion_reason",
        "recommended_next_step",
        "tags",
        "labels",
        "roles",
    }
    | _IDENTITY_METRIC_KEYS
    | _DIVERSITY_METADATA_KEYS
)

_BOOLEAN_METRIC_KEYS = {
    "promotion_eligible",
    "source_result_effective_config_complete",
    "replication_effective_config_match",
    "clean_promotion_eligible",
    "mature_enough",
    "scored_complete",
    "is_scored_complete",
    "complete_eval",
    "is_complete_eval",
    "_inferred_scored_complete",
    "partial_cohort",
    "partial",
    "partial_eval",
    "is_partial_eval",
    "incomplete",
    "incomplete_eval",
    "is_incomplete_eval",
    "capped",
    "is_capped",
    "result_capped",
    "validation_only",
    "validation_only_result",
    "is_negative",
    "scout_only",
    "is_scout_eval",
    "unscored_artifact",
    "summary_only",
    "is_summary_only",
    "is_smoke_eval",
    "smoke_only",
    "risk_violating_frontier_candidate",
    "risk_repair_required",
    "lane_lower_tier_candidate",
    "lane_non_promotable_candidate",
    "lane_missing_tier_candidate",
    "feature_ablation_done",
    "auto_materialized_from_result_artifact",
    "suspect",
    "suspect_protocol",
    "protocol_integrity_failed",
    "protocol_integrity_passed",
    "excluded_from_durable_frontier",
    "source_generation_low_confidence",
    "late_after_generation_boundary",
    "generation_boundary_pending_commit",
    "diversity_violated",
    "diversity_narrow_variation",
    "diversity_violation",
}

_LIST_CATEGORICAL_METRIC_KEYS = {
    "tags",
    "labels",
    "roles",
    "identity_aliases",
}

_PROVENANCE_STRING_SUFFIXES = (
    "_sha256",
    "_hash",
    "_digest",
    "_checksum",
    "_id",
)

# Extract numeric leading value from strings like "0.7965 ± 0.0014" or "79.65%".
_NUMERIC_LEAD = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_numeric(value: Any) -> float | None:
    """Extract a float from ``value``. Handles raw numbers, 'x ± y' strings,
    and percentage-annotated strings like '79.65%'."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return None
        return float(value)
    if isinstance(value, str):
        m = _NUMERIC_LEAD.search(value)
        if not m:
            return None
        try:
            v = float(m.group(0))
        except ValueError:
            return None
        # Rescale 79.65 → 0.7965 only if string obviously represents a
        # percentage (has '%', or the numeric is >1 and we're extracting
        # an accuracy-like metric — caller handles normalization).
        if "%" in value:
            v = v / 100.0
        return v
    return None


def _is_provenance_string_key(key: Any) -> bool:
    normalized = str(key or "").strip().lower()
    return normalized == "id" or normalized.endswith(_PROVENANCE_STRING_SUFFIXES)


def _preserve_provenance_string(metrics: dict[str, Any], key: Any, value: Any) -> None:
    if value is None or isinstance(value, (bool, dict, list)):
        return
    token = str(value).strip()
    if token:
        metrics[str(key)] = token


def _walk_find(data: Any, keys: tuple[str, ...], depth: int = 0) -> float | None:
    """Walk a nested JSON structure for any of ``keys``. DFS; returns the
    first numeric value found. Depth-limited to prevent pathological input."""
    if depth > 6:
        return None
    if isinstance(data, dict):
        # Direct hits first at this level
        for k in keys:
            if k in data:
                v = _parse_numeric(data[k])
                if v is not None:
                    return v
        # Recurse
        for v in data.values():
            r = _walk_find(v, keys, depth + 1)
            if r is not None:
                return r
    elif isinstance(data, list):
        for item in data:
            r = _walk_find(item, keys, depth + 1)
            if r is not None:
                return r
    return None


def _normalize_accuracy(v: float) -> float:
    """If value looks like a percentage (>1.5), convert to fraction."""
    if v > 1.5:
        return v / 100.0
    return v


def _preserve_categorical_metric(metrics: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, str) and value.strip():
        metrics[key] = value.strip()
    elif key in _LIST_CATEGORICAL_METRIC_KEYS and isinstance(value, list):
        metrics[key] = [str(item).strip() for item in value if str(item).strip()]
    elif (
        key in _IDENTITY_METRIC_KEYS
        and value is not None
        and not isinstance(value, (bool, dict, list))
    ):
        token = str(value).strip()
        if token:
            metrics[key] = token


def _copy_preserved_metric_fields(metrics: dict[str, Any], content: dict[str, Any]) -> None:
    sources: list[dict[str, Any]] = [content]
    content_metrics = content.get("metrics")
    if isinstance(content_metrics, dict):
        sources.append(content_metrics)
    for container_name in ("details", "extra"):
        container = content.get(container_name)
        if isinstance(container, dict):
            sources.append(container)
            nested_extra = container.get("extra")
            if isinstance(nested_extra, dict):
                sources.append(nested_extra)
    for source in sources:
        for key in _CATEGORICAL_METRIC_KEYS:
            if key == "source_result_sha256":
                continue
            if metrics.get(key) not in (None, "", [], {}):
                continue
            if key in source:
                _preserve_categorical_metric(metrics, key, source.get(key))
        for key in _NUMERIC_PRESERVED_METRIC_KEYS:
            if metrics.get(key) is not None:
                continue
            if key in source:
                parsed = _parse_numeric(source.get(key))
                if parsed is not None:
                    metrics[key] = parsed
        for key, value in source.items():
            if key == "source_result_sha256" or not _is_provenance_string_key(key):
                continue
            if metrics.get(key) in (None, "", [], {}):
                _preserve_provenance_string(metrics, key, value)
        for key in _BOOLEAN_METRIC_KEYS:
            if metrics.get(key) not in (None, ""):
                continue
            if key in source and isinstance(source[key], bool):
                metrics[key] = source[key]  # type: ignore[assignment]
        if isinstance(source.get("suspect_fixed_weight_eval"), bool) and bool(
            source["suspect_fixed_weight_eval"]
        ):
            metrics["suspect_protocol"] = bool(source["suspect_fixed_weight_eval"])
        metrics.pop("suspect_fixed_weight_eval", None)
    artifact_key = result_artifact_key(content)
    if artifact_key is not None:
        source_path, source_sha256 = artifact_key
        if source_path:
            metrics.setdefault("source_result_path", source_path)
        if source_sha256:
            metrics["source_result_sha256"] = source_sha256


def _walk_find_str(data: Any, keys: tuple[str, ...], depth: int = 0) -> str | None:
    """DFS for the first string value under any of ``keys``."""
    if depth > 6:
        return None
    if isinstance(data, dict):
        for k in keys:
            if k in data and isinstance(data[k], str) and data[k].strip():
                return data[k].strip()
        for v in data.values():
            r = _walk_find_str(v, keys, depth + 1)
            if r is not None:
                return r
    elif isinstance(data, list):
        for item in data:
            r = _walk_find_str(item, keys, depth + 1)
            if r is not None:
                return r
    return None


def _dataset_tag(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", "", text)


def _dataset_alias_records(content: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Return task-declared dataset aliases from a finding payload.

    Core deliberately has no built-in dataset vocabulary. A task can still ask
    filesystem ingest to infer a dataset from title/filename by including
    ``dataset_aliases`` (dict or list) or ``datasets`` (list) in the finding.
    """

    raw = content.get("dataset_aliases")
    if raw is None:
        raw = content.get("datasets")
    records: list[tuple[str, list[str]]] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            tag = _dataset_tag(key)
            if not tag:
                continue
            aliases = [str(key)]
            if isinstance(value, list):
                aliases.extend(str(item) for item in value)
            elif value not in (None, ""):
                aliases.append(str(value))
            records.append((tag, aliases))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                tag = _dataset_tag(item.get("name") or item.get("tag") or item.get("id"))
                aliases = [str(item.get("name") or item.get("tag") or item.get("id") or "")]
                raw_aliases = item.get("aliases")
                if isinstance(raw_aliases, list):
                    aliases.extend(str(alias) for alias in raw_aliases)
                elif raw_aliases not in (None, ""):
                    aliases.append(str(raw_aliases))
            else:
                tag = _dataset_tag(item)
                aliases = [str(item)]
            if tag:
                records.append((tag, aliases))
    deduped: dict[str, list[str]] = {}
    for tag, aliases in records:
        bucket = deduped.setdefault(tag, [])
        for alias in aliases:
            alias_text = str(alias).strip()
            if alias_text and alias_text not in bucket:
                bucket.append(alias_text)
        if tag not in bucket:
            bucket.append(tag)
    return list(deduped.items())


def _dataset_alias_hits(blob: str, records: list[tuple[str, list[str]]]) -> list[str]:
    normalized_blob = _dataset_tag(blob)
    hits: list[str] = []
    for tag, aliases in records:
        alias_norms = sorted(
            {_dataset_tag(alias) for alias in [tag, *aliases] if _dataset_tag(alias)},
            key=len,
            reverse=True,
        )
        if any(alias_norm and alias_norm in normalized_blob for alias_norm in alias_norms):
            hits.append(tag)
    return list(dict.fromkeys(hits))


def infer_dataset(filepath: Path, content: dict[str, Any]) -> str | None:
    """Return the canonical dataset tag for this finding, or None.

    Precedence:
      1. Explicit nested ``dataset`` field (authoritative — peer-declared).
      2. Title / summary alias match, but only when the finding declares
         task-owned ``dataset_aliases`` or ``datasets``.
      3. Filename alias match using the same task-owned aliases.

    Returns None when nothing matches or more than one declared dataset alias
    matches, because metric assignment would be ambiguous.
    """
    explicit = _walk_find_str(content, ("dataset",))
    if explicit:
        return _dataset_tag(explicit) or None

    title_blob = " ".join(
        [
            str(content.get("title", "")),
            str(content.get("summary", "")),
            str(content.get("notes", "")),
        ]
    )
    alias_records = _dataset_alias_records(content)
    if not alias_records:
        return None
    hits = _dataset_alias_hits(title_blob, alias_records)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return None

    hits = _dataset_alias_hits(filepath.name, alias_records)
    if len(hits) == 1:
        return hits[0]

    return None


def extract_variant_name(content: dict[str, Any]) -> str:
    """Pull a variant/method name from the finding.

    Checks, in order:
      1. top-level ``variant_name``.
      2. any nested explicit ``variant`` / ``variant_name`` string.
      3. Compact uppercase token in the title/summary (e.g. "METHOD-A1").
         Uses a bounded regex to avoid grabbing noise words like "SESSION18"
         or "DEFINITIVE" that a naive leading-caps regex would accept.
    Returns empty string if nothing usable is found.
    """
    explicit = content.get("variant_name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()[:200]
    nested = _walk_find_str(content, _VARIANT_KEY_ALIASES)
    if nested:
        return nested[:200]
    blob = " ".join(
        [
            str(content.get("title", "")),
            str(content.get("summary", "")),
        ]
    )
    m = _VARIANT_TITLE_RX.search(blob)
    if m:
        return m.group(1)[:200]
    return ""


def derive_finding_id(filepath: Path, content: dict[str, Any]) -> str:
    """Return a stable id for the file. Rules:

    1. ``content["id"]`` if explicitly set and UUID-shaped (share_finding path).
    2. UUID prefix of filename (share_finding path, pre-refactor files).
    3. Deterministic ``fs_<sha256[:32]>`` of the source path — stable across
       re-ingests of the same file while keeping same-name files from
       root/gen-local finding directories distinct.
    """
    explicit = content.get("id")
    if isinstance(explicit, str) and _UUID_PATTERN.match(explicit):
        return explicit

    match = _UUID_PATTERN.match(filepath.name)
    if match:
        return match.group(1)

    digest = hashlib.sha256(_source_cache_key(filepath).encode("utf-8")).hexdigest()
    return f"fs_{digest[:32]}"


def _source_cache_key(filepath: Path) -> str:
    """Return a stable per-source key for finding ingest id/cache logic."""

    path = Path(filepath)
    parts = path.parts
    for marker in ("shared_findings", "results", "gems"):
        if marker in parts:
            idx = len(parts) - 1 - list(reversed(parts)).index(marker)
            prefix: tuple[str, ...] = ()
            if idx >= 1 and re.fullmatch(r"gen_\d+", parts[idx - 1]):
                prefix = (parts[idx - 1],)
            return "/".join((*prefix, *parts[idx:]))
    return str(path.resolve(strict=False))


def _declared_gem_paths_in_state(filepath: Path) -> dict[str, set[str]]:
    run_dir = filepath.parent.parent if filepath.parent.name == "shared_findings" else None
    if run_dir is None:
        return {}
    state_path = run_dir / "gems" / "gems_state.json"
    if not state_path.exists():
        return {}
    try:
        state = json.loads(read_file_bytes(state_path).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(state, dict):
        return {}
    semantics = state.get("artifact_semantics")
    if isinstance(semantics, dict):
        role = str(semantics.get("role") or "").strip()
        status = str(semantics.get("status") or "").strip().lower()
        pending_reset = isinstance(state.get("pending_reset"), dict)
        committed_runtime = (
            role == CANONICAL_STATE
            and status == COMMITTED
            and semantics.get("runtime_fact_source") is True
        )
        pending_repair_state = role == CANONICAL_STATE and status == PARTIAL and pending_reset
        if not (committed_runtime or pending_repair_state):
            return {}
    out: dict[str, set[str]] = {}

    def add_records(records: Any) -> None:
        if not isinstance(records, list):
            return
        for record in records:
            if not isinstance(record, dict):
                continue
            gem_id = str(record.get("gem_finding_id") or "").strip()
            rel_path = str(record.get("finding_path") or "").strip()
            if not gem_id or not rel_path:
                continue
            rel = Path(rel_path)
            if rel.is_absolute() or ".." in rel.parts or rel.parts[:1] != ("shared_findings",):
                continue
            out.setdefault(gem_id, set()).add(rel_path)

    add_records(state.get("gems"))
    pending = state.get("pending_reset")
    if isinstance(pending, dict):
        add_records(pending.get("gem_records"))
    return out


def _trusted_declared_gem_id(filepath: Path, explicit: Any) -> str:
    if not isinstance(explicit, str):
        return ""
    token = explicit.strip()
    if not token or not _GEM_ID_PATTERN.match(token):
        return ""
    allowed_paths = _declared_gem_paths_in_state(filepath).get(token, set())
    if not allowed_paths:
        return ""
    if filepath.is_symlink():
        return ""
    try:
        run_dir = filepath.parent.parent
        rel_path = str(filepath.relative_to(run_dir))
    except ValueError:
        return ""
    return token if rel_path in allowed_paths else ""


def _untrusted_declared_gem_update(parsed: dict[str, Any], declared_id: str) -> bool:
    if not _GEM_ID_PATTERN.match(declared_id):
        return False
    metrics = parsed.get("metrics") if isinstance(parsed.get("metrics"), dict) else {}
    return metrics.get("is_gem_finding") is not True


def extract_metrics(
    content: dict[str, Any],
    dataset: str | None = None,
    primary_metric: str | None = None,
) -> dict[str, Any]:
    """Best-effort metric extraction.

    Order of preference:
      1. ``content["metrics"]`` if it is a flat dict of measurements.
      2. Hoist the task-declared primary metric from nested JSON.
      3. Apply a legacy alias walk only when the task explicitly declares
         one of those aliases as its primary metric.

    Praxist never guesses a domain metric when ``primary_metric`` is absent.
    Original numeric fields remain available in ``metrics`` and the source
    artifact; only task-declared semantics are promoted to a canonical key.

    Values from the source ``metrics`` dict are trusted: if a peer wrote
    ``metrics: {"test_accuracy_dataset": 0.7965}``, that stays as-is even
    if dataset inference would have said otherwise.

    #150: when ``primary_metric`` is supplied (the task's
    ``evaluation.primary_metric``) and that key is still missing after
    the accuracy-alias walk, fall back to a strict-canonical walk for
    the primary key itself and hoist any nested numeric to
    ``metrics[primary_metric]``. Without this, filesystem-written
    findings whose peer stashed the primary value under ``final_results``
    / ``aggregated`` / ``details`` are silently rejected by
    ``frontier.promote`` (and counted as ``variants_total=0``) — the
    operator sees ``findings_total: 15, variants_total: 0`` and
    concludes the run made no progress when in fact the peer produced
    real measurements that nobody read.
    """
    metrics: dict[str, Any] = {}
    primary_metric_text = str(primary_metric or "").strip()

    # (1) Standard location — copy numeric entries as-is.
    direct = content.get("metrics")
    if isinstance(direct, dict):
        # R6-N5 fix: preserve string `tier` and bool `promotion_eligible`
        # BEFORE the numeric-only filter strips them. Without this, any
        # finding written via the `Write` tool (which historically was
        # 287/289 of all findings in one prior run) loses tier metadata
        # during ingest, completely bypassing the frontier tier filter.
        # We use the metrics dict as a heterogeneous payload here so that
        # frontier.promote() can read these gating fields back.
        # Note: bool is an int subclass in Python, so without this fix
        # `_parse_numeric(False)` returns 0.0 — preserving the value but
        # losing the boolean type, breaking the `if f_promo is False`
        # check in frontier.promote().
        if "tier" in direct and isinstance(direct["tier"], str):
            metrics["tier"] = direct["tier"]  # type: ignore[assignment]
        for key, value in direct.items():
            if isinstance(value, bool) and key != primary_metric_text:
                metrics[key] = value
        for key in _CATEGORICAL_METRIC_KEYS:
            if key == "source_result_sha256":
                continue
            _preserve_categorical_metric(metrics, key, direct.get(key))
        for k, v in direct.items():
            if k in ("tier",) or isinstance(v, bool) or k in _CATEGORICAL_METRIC_KEYS:
                continue  # already handled with type preservation
            if _is_provenance_string_key(k):
                _preserve_provenance_string(metrics, k, v)
                continue
            parsed = _parse_numeric(v)
            if parsed is not None:
                metrics[k] = parsed

    # (2) Legacy aliases are task-local compatibility. Never infer an ML
    # metric merely because no task primary metric was supplied.
    accuracy_alias_enabled = primary_metric_text in _ACC_KEY_ALIASES
    has_any_accuracy = ("test_accuracy" in metrics) or any(
        k.startswith("test_accuracy_") for k in metrics
    )
    if accuracy_alias_enabled and not has_any_accuracy:
        aliases = (primary_metric_text,) + tuple(
            key for key in _ACC_KEY_ALIASES if key != primary_metric_text
        )
        v = _walk_find(content, aliases)
        if v is not None:
            v_norm = _normalize_accuracy(v)
            metrics["test_accuracy"] = v_norm
            metrics.setdefault(primary_metric_text, v_norm)
            if dataset:
                metrics[f"test_accuracy_{dataset}"] = v_norm

    # (3) A gap alias is canonical only when the task declares it as primary.
    gap_alias_enabled = primary_metric_text in _GAP_KEY_ALIASES
    if gap_alias_enabled and "train_test_gap" not in metrics:
        aliases = (primary_metric_text,) + tuple(
            key for key in _GAP_KEY_ALIASES if key != primary_metric_text
        )
        v = _walk_find(content, aliases)
        if v is not None:
            normalized_gap = v if v < 1.0 else v / 100.0
            metrics["train_test_gap"] = normalized_gap
            metrics.setdefault(primary_metric_text, normalized_gap)

    # (4) #150: hoist the task-declared primary_metric when still missing
    # so downstream filters (status snapshot variant count + frontier
    # promotion) see filesystem-written findings in canonical shape.
    if (
        primary_metric
        and primary_metric not in metrics
        and primary_metric not in _ACC_KEY_ALIASES
        and primary_metric not in _GAP_KEY_ALIASES
    ):
        v = _walk_find(content, (primary_metric,))
        if v is not None:
            metrics[primary_metric] = v

    _copy_preserved_metric_fields(metrics, content)
    strip_effective_config_fields(metrics)
    return metrics


_FINDING_TYPES = ("result", "hypothesis", "insight", "challenge", "error")


def infer_finding_type(content: dict[str, Any], extracted_metrics: dict[str, float]) -> str:
    """Infer finding_type when not explicit. Heuristic:

    - Explicit value wins if it's one of the 4 valid types.
    - Otherwise: keywords in title/summary/notes.
    - Otherwise: if we extracted a primary metric ⇒ "result".
    - Fallback: "insight".
    """
    explicit = content.get("finding_type")
    if isinstance(explicit, str) and explicit in _FINDING_TYPES:
        return explicit

    title_bits = [
        str(content.get("title", "")),
        str(content.get("summary", "")),
        str(content.get("notes", "")),
    ]
    blob = " ".join(title_bits).lower()

    if any(kw in blob for kw in ("negative", "failed", "does not", "dead end", "rejected")):
        return "error"
    if any(kw in blob for kw in ("hypothesis", "proposal", "propose", "plan")):
        return "hypothesis"
    if extracted_metrics:
        return "result"
    if any(kw in blob for kw in ("insight", "observation", "pattern")):
        return "insight"
    return "insight"


def _infer_peer_and_gen(filepath: Path, content: dict[str, Any]) -> tuple[str, int]:
    """Extract peer_id and generation_id from content or filename."""
    peer_id = str(content.get("peer_id") or "")
    if not peer_id:
        m = re.match(r"(gen\d+_peer\d+)", filepath.name)
        if m:
            peer_id = m.group(1)

    gen_id: int | None = None
    if "generation_id" in content:
        with contextlib.suppress(TypeError, ValueError):
            gen_id = int(content["generation_id"])
    if gen_id is None and peer_id:
        m = re.match(r"gen(\d+)_", peer_id)
        if m:
            gen_id = int(m.group(1))
    if gen_id is None:
        m = re.match(r"gen_?(\d+)_", filepath.name)
        if m:
            gen_id = int(m.group(1))
    if gen_id is None:
        for parent in filepath.parents:
            m = re.fullmatch(r"gen_?(\d+)", parent.name)
            if m:
                gen_id = int(m.group(1))
                break

    return peer_id, gen_id if gen_id is not None else 0


def _lenient_json_loads(text: str) -> Any:
    r"""JSON loader that tolerates a few common agent-written mistakes.

    Real agent files observed in the 2026-04-17 run that stock ``json.loads``
    rejected (both due to leading ``+`` before a number):
      - ``"delta_acc": +0.0013``
      - ``"delta_pp": +2.54``

    We also clean trailing commas before ``}`` or ``]`` since those are
    common and the ``,\s*[}\]]`` pattern cannot occur inside a JSON string
    (a comma inside a string is not followed by a literal close-brace).

    What we deliberately do NOT touch:
      - ``//`` comments. A naive ``//[^\n]*`` strip destroys the ``//`` in
        any URL (``"https://example.com"``) or Unix-like path inside a
        JSON string. Observed malformed files do not use JS comments, so
        the cost/benefit doesn't justify the risk.

    String-safety: the ``+`` substitution is anchored to ``([:,\[])\s*``
    — a JSON structural token immediately preceding a value position —
    which cannot match inside a string literal. Content like
    ``"note": "jump from X to +42 units"`` is preserved.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"([:,\[])(\s*)\+(\d)", r"\1\2\3", text)
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return json.loads(cleaned)


def parse_finding_file(
    filepath: Path,
    primary_metric: str | None = None,
    *,
    result_maturity_policy: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Parse a filesystem finding file into the SQLite ``findings`` row shape.

    Returns ``None`` if the file cannot be parsed as JSON (even after
    lenient cleanup) or the content is not a dict (e.g. a JSONL-shaped
    file, which isn't a valid finding).

    #150: ``primary_metric`` (the task's ``evaluation.primary_metric``)
    is forwarded into ``extract_metrics`` so a nested primary value gets
    hoisted to the canonical ``metrics[primary_metric]`` location. When
    ``None``, the legacy accuracy-alias-only behaviour is preserved.
    """
    try:
        read_path = filepath
        if not controller_state_enabled() and filepath.is_symlink():
            # Legacy aliases remain readable, but never acquire target provenance.
            read_path = filepath.resolve()
        text = read_file_bytes(read_path).decode("utf-8", errors="replace")
    except (OSError, RuntimeError) as e:
        logger.debug("findings_ingest: read failed %s (%s)", filepath.name, e)
        return None
    try:
        content = _lenient_json_loads(text)
    except json.JSONDecodeError as e:
        logger.debug("findings_ingest: skipping %s (%s)", filepath.name, e)
        return None
    if not isinstance(content, dict):
        return None

    dataset = infer_dataset(filepath, content)
    metrics = extract_metrics(content, dataset=dataset, primary_metric=primary_metric)
    content_metrics = content.get("metrics")
    if isinstance(content_metrics, dict):
        for container_name in ("metrics", "details", "extra"):
            compact = compact_result_identity_container(content_metrics.get(container_name))
            if compact:
                strip_effective_config_fields(compact)
                metrics[container_name] = compact
        if isinstance(content_metrics.get("current_aggregate"), dict):
            current_aggregate = copy.deepcopy(content_metrics["current_aggregate"])
            strip_effective_config_fields(current_aggregate)
            metrics["current_aggregate"] = current_aggregate
    metrics.update(
        _result_reference_effective_config_metadata(
            filepath,
            content,
            maturity_policy=result_maturity_policy,
        )
    )
    declared_id = content.get("id")
    trusted_gem_id = _trusted_declared_gem_id(filepath, declared_id)
    if metrics.get("is_gem_finding") and not trusted_gem_id:
        metrics.pop("is_gem_finding", None)
    finding_type = infer_finding_type(content, metrics)
    peer_id, gen_id = _infer_peer_and_gen(filepath, content)
    variant_name = extract_variant_name(content)

    if "tier" in content and "tier" not in metrics:
        metrics["tier"] = content["tier"]  # type: ignore[assignment]
    if "promotion_eligible" in content and "promotion_eligible" not in metrics:
        metrics["promotion_eligible"] = content["promotion_eligible"]  # type: ignore[assignment]

    row: dict[str, Any] = {
        "id": derive_finding_id(filepath, content),
        "finding_type": finding_type,
        "title": str(content.get("title") or filepath.stem)[:500],
        "content": str(content.get("content") or content.get("summary") or "")[:5000],
        "metrics": metrics,
        "variant_name": variant_name,
        "notes": str(content.get("notes") or content.get("summary") or "")[:2000],
        "peer_id": peer_id,
        "generation_id": gen_id,
        "timestamp": str(content.get("timestamp") or datetime.now(UTC).isoformat()),
        # Preserve source-file provenance and inferred dataset in `extra`.
        # source_mtime_ns is the cache key for the hot-path "has this file
        # been seen and is it unchanged?" check in ingest_findings_directory.
        "dataset": dataset or "",
        "source_declared_finding_id": str(declared_id or "").strip()
        if isinstance(declared_id, str)
        else "",
        "source_filename": filepath.name,
        "source_filepath": str(filepath),
        "source_mtime_ns": _safe_mtime_ns(filepath),
        "ingest_schema_version": _INGEST_SCHEMA_VERSION,
        "ingested_at": datetime.now(UTC).isoformat(),
    }
    for key in (
        *_RESULT_PRODUCER_IDENTITY_KEYS,
        *_RESULT_ARTIFACT_PATH_KEYS,
        "source_result_sha256",
    ):
        value = content.get(key)
        if value not in (None, "") and not isinstance(value, (bool, dict, list)):
            row[key] = value
    if "tier" in content:
        row["tier"] = content["tier"]
    if "promotion_eligible" in content:
        row["promotion_eligible"] = content["promotion_eligible"]
    if isinstance(content.get("extra"), dict):
        row["extra"] = copy.deepcopy(content["extra"])
        strip_effective_config_fields(row["extra"])
    if isinstance(content.get("details"), dict):
        row["details"] = copy.deepcopy(content["details"])
        strip_effective_config_fields(row["details"])
    if isinstance(content.get("current_aggregate"), dict):
        row["current_aggregate"] = copy.deepcopy(content["current_aggregate"])
        strip_effective_config_fields(row["current_aggregate"])
    for key in sorted(_DIVERSITY_METADATA_KEYS | _NUMERIC_PRESERVED_METRIC_KEYS):
        value = content.get(key)
        if value in (None, "", [], {}):
            continue
        row.setdefault("extra", {}).setdefault(key, value)
    if isinstance(content.get("peer_role"), str) and content["peer_role"].strip():
        row["peer_role"] = content["peer_role"].strip()
        row.setdefault("extra", {})["peer_role"] = content["peer_role"].strip()
    if isinstance(content.get("links"), list):
        row["links"] = content["links"]
    dimension_sources = [content]
    if isinstance(content_metrics, dict):
        dimension_sources.append(content_metrics)
    for dimension_source in dimension_sources:
        for key in ("design_dimensions", "realized_dimensions"):
            value = dimension_source.get(key)
            if isinstance(value, (dict, list)):
                # Keep one canonical realized-evidence field. Auto-materialized
                # result findings carry it inside ``metrics``; peer-authored
                # findings may use either supported top-level spelling.
                row["design_dimensions"] = copy.deepcopy(value)
                break
        if "design_dimensions" in row:
            break
    return row


def _safe_mtime_ns(filepath: Path) -> int:
    try:
        return filepath.stat().st_mtime_ns
    except OSError:
        return 0


def ingest_findings_directory(
    findings_dir: Path,
    primary_metric: str | None = None,
    *,
    result_maturity_policy: dict[str, Any] | None = None,
) -> int:
    """Scan ``findings_dir`` for ``*.json`` files and upsert into SQLite.

    Returns the number of newly-inserted or updated rows.

    #150: ``primary_metric`` (the task's ``evaluation.primary_metric``)
    is threaded into ``parse_finding_file`` so filesystem-written
    findings whose peer nested the primary value under ``final_results``
    / ``aggregated`` / etc. get hoisted to ``metrics[primary_metric]``
    at ingest time. Without this, the row reaches SQLite but is
    silently dropped by ``frontier.promote`` and counted as zero in
    ``variants_total``.

    Hot-path cache: we index existing rows by ``source_filename`` and
    compare ``source_mtime_ns`` before parsing. A file whose mtime matches
    the stored value is skipped without opening it. This is what makes the
    daemon's 60-second poll cheap on a 300-file directory (hundreds of
    JSON loads collapse to hundreds of stat() calls).

    Covers two cases correctly:

    - **Overwrite**: agent rewrites ``gen0_peer4_result.json`` with updated
      metrics. The filename is unchanged, so the derived id is the same,
      but the mtime changed → we re-parse and ``INSERT OR REPLACE`` picks
      up the new content. Without the mtime check, the old "skip if id
      already present" logic silently kept the stale row.

    - **New file**: an agent just wrote a file we've never seen. No cache
      hit → parse + insert.

    Errors during parse/insert are logged but do not stop the scan — one
    bad file should not prevent the rest from being ingested.
    """
    if not findings_dir.exists():
        return 0

    try:
        from .local_store import _get_conn, init_db, insert_finding
    except ImportError as e:
        logger.warning("findings_ingest: local_store import failed: %s", e)
        return 0

    try:
        init_db()
    except Exception as e:
        logger.warning("findings_ingest: init_db failed: %s", e)
        return 0

    # Build two caches from SQLite in one pass:
    #   source_cache: source cache key/path → (stored mtime_ns, ingest_schema_version)
    #     for the hot path. Schema version makes metadata-preservation fixes
    #     refresh old rows once even when the source file mtime is unchanged.
    #   existing_ids: set of known ids (for the non-cached fallback path,
    #     e.g. share_finding MCP rows that lack source_filename).
    source_cache: dict[str, tuple[int, int]] = {}
    existing_ids: set = set()
    try:
        with _get_conn(readonly=True) as conn:
            rows = conn.execute(
                "SELECT id, json_extract(extra, '$.source_filepath'), "
                "json_extract(extra, '$.source_filename'), "
                "json_extract(extra, '$.source_mtime_ns'), "
                "json_extract(extra, '$.ingest_schema_version') FROM findings"
            ).fetchall()
        for row in rows:
            existing_ids.add(row[0])
            source_key = row[1] or row[2]
            if source_key:
                try:
                    mtime = int(row[3]) if row[3] is not None else 0
                except (TypeError, ValueError):
                    mtime = 0
                try:
                    schema_version = int(row[4]) if row[4] is not None else 0
                except (TypeError, ValueError):
                    schema_version = 0
                source_cache[str(source_key)] = (mtime, schema_version)
    except Exception as e:
        logger.warning("findings_ingest: SELECT existing rows failed: %s", e)

    touched = 0
    for filepath in sorted(findings_dir.glob("*.json")):
        # Skip lock/tmp files that might sneak into the glob on some FS.
        if filepath.suffix in (".lock", ".tmp"):
            continue

        source_key = str(filepath)

        # Hot path: cache hit on (source path, mtime) — no read, no parse.
        cached = source_cache.get(source_key)
        if cached is not None:
            cached_mtime, cached_schema_version = cached
            current_mtime = _safe_mtime_ns(filepath)
            if (
                current_mtime
                and current_mtime == cached_mtime
                and cached_schema_version >= _INGEST_SCHEMA_VERSION
            ):
                continue

        parsed = parse_finding_file(
            filepath,
            primary_metric=primary_metric,
            result_maturity_policy=result_maturity_policy,
        )
        if parsed is None:
            continue

        # Fallback-path dedup: share_finding-produced rows and orchestrator
        # Gem sidecars inserted directly into SQLite may not have
        # source_filename stored. Skip on either the derived id or the
        # file-declared id, but never use a non-UUID declared id as the
        # insertion identity for a new row.
        declared_id = str(parsed.get("source_declared_finding_id") or "")
        if (
            cached is None
            and declared_id
            and declared_id in existing_ids
            and parsed["id"] != declared_id
        ):
            if _untrusted_declared_gem_update(parsed, declared_id):
                source_cache[source_key] = (
                    int(parsed.get("source_mtime_ns", 0) or 0),
                    _INGEST_SCHEMA_VERSION,
                )
                continue
            parsed_for_declared_id = dict(parsed)
            parsed_for_declared_id["id"] = declared_id
            try:
                insert_finding(parsed_for_declared_id)
                source_cache[source_key] = (
                    int(parsed.get("source_mtime_ns", 0) or 0),
                    _INGEST_SCHEMA_VERSION,
                )
                touched += 1
            except Exception as e:
                logger.warning("findings_ingest: upsert failed for %s: %s", filepath.name, e)
            continue

        if cached is None and parsed["id"] in existing_ids:
            try:
                insert_finding(parsed)
                source_cache[source_key] = (
                    int(parsed.get("source_mtime_ns", 0) or 0),
                    _INGEST_SCHEMA_VERSION,
                )
                touched += 1
            except Exception as e:
                logger.warning("findings_ingest: upsert failed for %s: %s", filepath.name, e)
            continue

        try:
            insert_finding(parsed)
            existing_ids.add(parsed["id"])
            source_cache[source_key] = (
                int(parsed.get("source_mtime_ns", 0) or 0),
                _INGEST_SCHEMA_VERSION,
            )
            touched += 1
        except Exception as e:
            logger.warning("findings_ingest: upsert failed for %s: %s", filepath.name, e)

    if touched:
        logger.info(
            "findings_ingest: %d findings inserted/updated from %s",
            touched,
            findings_dir,
        )
    return touched
