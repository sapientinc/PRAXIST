"""Finding collection and store fallback logic for generation boundaries."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import os
import re
import shlex
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from praxist.core.storage import open_readonly_file, read_file_bytes
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    COMMITTED,
    DERIVED_VIEW,
    attach_artifact_semantics,
)
from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    EFFECTIVE_CONFIG_METADATA_KEYS,
    result_effective_config_metadata,
    strip_effective_config_fields,
)
from praxist.plugins.workflow_stages.research_loop.backend.evidence_maturity import (
    _RESULT_ARTIFACT_PATH_KEYS,
    _RESULT_PRODUCER_IDENTITY_KEYS,
    RESULT_RESEARCH_METADATA_KEYS,
    evidence_maturity_snapshot,
    task_authorizes_descriptive_maturity,
)
from praxist.plugins.workflow_stages.research_loop.backend.tools.atomic_io import (
    atomic_write_json as _atomic_write_json_file,
)

logger = logging.getLogger(__name__)

_RESULT_MATERIALIZATION_LOCK = threading.RLock()


_RESULT_SUMMARY_NAMES = (
    "tiered_eval_summary.json",
    "result_summary.json",
    "evaluation_summary.json",
    "eval_summary.json",
    "final_summary.json",
    "summary.json",
)
_RESULT_SUMMARY_SUFFIXES = _RESULT_SUMMARY_NAMES
_DURABLE_ROUTING_MARKER_KEYS = (
    "validation_only_result",
    "late_after_generation_boundary",
    "artifact_signal_status",
    "late_result_policy",
    "durability_scope",
)
_BOUNDARY_FINGERPRINT_IGNORED_KEYS = frozenset(
    {
        "timestamp",
        "created_at",
        "updated_at",
        "observed_at",
        "promoted_at",
        "source_filename",
        "source_filepath",
        "source_mtime_ns",
        "ingest_schema_version",
        "ingested_at",
    }
)
_FINDING_SNAPSHOT_PREFIX = "canonical-finding:"
_SCORELESS_SOURCE_SNAPSHOT_PREFIX = "scoreless-source-finding:"
_FINDING_SNAPSHOT_MARKER = "canonical-finding-snapshot:v1"
_FINDING_SNAPSHOT_PAYLOAD_PREFIX = "canonical-finding-payload:v1:"
_RESULT_SNAPSHOT_ROOT_MARKER = "result-source-root:v1"
_RESULT_SIGNAL_METADATA_KEYS = (
    "frontier_lane",
    "promotion_lane",
    "lane",
    "evidence_stage",
    "evidence_valence",
    "is_negative",
    "failure_mode",
    "diagnostic_role",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
    "tradeoff_class",
    "primary_tradeoff",
    "protocol_name",
    "ranking_allowed",
    "clean_promotion_eligible",
    "parent_eligible",
    "validation_only",
    "excluded_from_durable_frontier",
    "exclusion_reason",
    "incubator_candidate_reason",
    "incubator_axis",
)
_RESULT_DIMENSION_METADATA_KEYS = (
    "design_dimensions",
    "realized_dimensions",
)

_EVALUATION_UNIT_COUNT_KEYS = (
    "completed_required_eval_units",
    "actual_eval_units",
    "evaluation_units",
    # Read-only compatibility aliases from older task artifacts.
    "scored_cell_count",
    "n_scored_cells",
    "n_eval_cells",
    "cell_count",
)
_LEGACY_EVALUATION_UNIT_COUNT_KEYS = _EVALUATION_UNIT_COUNT_KEYS[3:]
_RESULT_CHILD_IDENTITY_KEYS = (
    "child_id",
    "sweep_child_id",
    "child_variant_id",
    "result_variant_id",
    "child_variant_name",
    "result_variant_name",
    "canonical_variant_name",
)
_RESULT_IDENTITY_PRESERVE_KEYS = (
    *_RESULT_PRODUCER_IDENTITY_KEYS,
    "canonical_variant_id",
    *_RESULT_ARTIFACT_PATH_KEYS,
    "source_result_sha256",
)
_COMPLETION_FLAG_KEYS = (
    "scored_complete",
    "is_scored_complete",
    "complete_eval",
    "is_complete_eval",
)
_CANONICAL_PEER_ID_PATTERN = re.compile(r"gen(?P<generation>\d+)_peer(?P<index>\d+)")
_SHELL_COMMAND_NAMES = {"bash", "dash", "sh", "zsh"}
_SCHEDULER_OUTPUT_OPTION_NAMES = frozenset(
    {
        "artifact-dir",
        "artifact-root",
        "out-dir",
        "output",
        "output-dir",
        "output-path",
        "result",
        "result-dir",
        "result-path",
        "results",
        "results-dir",
        "results-path",
        "save-dir",
        "save-path",
        "summary-dir",
        "summary-path",
    }
)
_SCHEDULER_ENV_PATTERN = re.compile(
    r"\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)|\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}"
)


def _compact_summary_result_identity(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    compact = {key: source[key] for key in _RESULT_IDENTITY_PRESERVE_KEYS if key in source}
    for container_name in ("metrics", "details", "extra", "current_aggregate"):
        nested = _compact_summary_result_identity(source.get(container_name))
        if nested:
            compact[container_name] = nested
    return compact


def is_supported_result_summary_filename(name: str) -> bool:
    """Return whether ``name`` is part of the task result-summary contract."""

    filename = str(name or "")
    return filename in _RESULT_SUMMARY_NAMES or (
        filename.startswith("custom_")
        and any(filename.endswith(f"_{suffix}") for suffix in _RESULT_SUMMARY_SUFFIXES)
    )


def result_summary_filename_variant(name: str) -> str:
    """Return a custom summary's filename-owned variant identity, if any."""

    filename = str(name or "")
    if not filename.startswith("custom_"):
        return ""
    for suffix in _RESULT_SUMMARY_SUFFIXES:
        marker = f"_{suffix}"
        if filename.endswith(marker):
            return filename[len("custom_") : -len(marker)]
    return ""


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_json_file(path, data)


def canonicalize_evaluation_unit_metadata(
    payload: dict[str, Any],
    *sources: dict[str, Any],
) -> Any:
    """Expose one generic evaluation-unit count while accepting old inputs."""
    value = None
    for source in (payload, *sources):
        if not isinstance(source, dict):
            continue
        for key in _EVALUATION_UNIT_COUNT_KEYS:
            candidate = source.get(key)
            if candidate is not None and not isinstance(candidate, bool):
                value = candidate
                break
        if value is not None:
            break
    for key in _LEGACY_EVALUATION_UNIT_COUNT_KEYS:
        payload.pop(key, None)
    if value is not None:
        payload["evaluation_units"] = value
    return value


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:120] or "unknown"


def _run_dir_from_findings_dir(findings_dir: Path) -> Path:
    findings_dir = Path(findings_dir)
    if findings_dir.name == "shared_findings":
        return findings_dir.parent
    return findings_dir


def _load_json(
    path: Path, *, source_content_hashes: dict[Path, str] | None = None
) -> dict[str, Any] | None:
    try:
        source_bytes = read_file_bytes(path)
        data = json.loads(source_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if source_content_hashes is not None:
        source_content_hashes[path] = hashlib.sha256(source_bytes).hexdigest()
    return data


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(read_file_bytes(path).decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _dig_contract_provenance(
    *,
    run_dir: Path,
    variant: str,
    source_gen_id: int,
    metrics: dict[str, Any],
) -> dict[str, Any] | None:
    """Return DIG provenance for a materialized result artifact when available."""

    gen_dirs = [run_dir / f"gen_{source_gen_id}"]
    gen_dirs.extend(sorted(path for path in run_dir.glob("gen_*") if path not in gen_dirs))
    best: tuple[int, Path, dict[str, Any]] | None = None
    variant_token = _slug(variant)
    for gen_dir in gen_dirs:
        peers_dir = gen_dir / "peers"
        if not peers_dir.exists():
            continue
        for contract_path in peers_dir.glob("*/dig/selected_contract.yaml"):
            contract = _load_yaml(contract_path)
            if not contract:
                continue
            score = 0
            contract_variant = _slug(contract.get("variant_name"))
            selected_id = _slug(contract.get("selected_candidate_id"))
            peer_id = contract_path.parents[1].name
            if contract_variant and contract_variant == variant_token:
                score += 100
            if selected_id and selected_id in variant_token:
                score += 40
            if peer_id and _slug(peer_id) in variant_token:
                score += 30
            with contextlib.suppress(ValueError):
                if int(str(gen_dir.name).split("_")[-1]) == int(source_gen_id):
                    score += 5
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, contract_path, contract)
    if best is None:
        return None
    _, contract_path, contract = best
    contract_provenance = dict(contract.get("dig_provenance") or {})
    expected = dict(contract.get("expected_metric_signature") or {})
    return {
        **contract_provenance,
        "selected_contract_path": str(contract_path.relative_to(run_dir)),
        "variant_name": contract.get("variant_name"),
        "selected_candidate_id": contract.get("selected_candidate_id"),
        "semantic_family": contract.get("semantic_family"),
        "parent_lineage": contract.get("parent_lineage"),
        "novelty_axis": contract.get("novelty_axis"),
        "diversity_cell": contract.get("diversity_cell") or {},
        "canonical_labels": contract.get("canonical_labels") or {},
        "contract_amended": (contract_path.parent / "contract_amendment.yaml").is_file(),
        "expected_metric_signature": expected,
        "actual_result_keys": sorted(
            key
            for key in (
                "mean_score",
                "score",
                "tier",
                "tier_status",
                "n_hard_constraint_violations",
                *expected.keys(),
            )
            if key in metrics
        ),
        "expected_vs_actual_alignment": "inconclusive",
    }


def iter_result_summary_paths(run_dir: Path) -> list[Path]:
    """Return result summary artifacts written by supported task evaluators."""

    results_dir = Path(run_dir) / "results"
    if not results_dir.exists():
        return []

    def walk(directory: Path, ancestor_dirs: frozenset[tuple[int, int]]):
        """Follow directory links without merging distinct logical producers."""

        try:
            stat_result = directory.stat()
            directory_id = (int(stat_result.st_dev), int(stat_result.st_ino))
        except OSError:
            return
        if directory_id in ancestor_dirs:
            return
        branch_dirs = ancestor_dirs | {directory_id}
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            return
        for entry in entries:
            path = directory / entry.name
            try:
                if entry.is_dir(follow_symlinks=True):
                    yield from walk(path, branch_dirs)
                elif entry.is_file(follow_symlinks=True) and path.suffix.lower() == ".json":
                    yield path
            except OSError:
                continue

    paths: list[Path] = []
    for path in walk(results_dir, frozenset()):
        if is_supported_result_summary_filename(path.name):
            paths.append(path)
    return sorted(paths, key=lambda path: str(path))


def _sync_generation_local_findings(
    *,
    run_dir: Path,
    gen_id: int,
    primary_metric: str | None = None,
    result_maturity_policy: dict[str, Any] | None = None,
) -> int:
    """Ingest generation-local findings without creating a second file copy."""
    src_dir = run_dir / f"gen_{gen_id}" / "shared_findings"
    if not src_dir.exists():
        return 0
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest import (
            ingest_findings_directory,
        )

        return ingest_findings_directory(
            src_dir,
            primary_metric=primary_metric,
            result_maturity_policy=result_maturity_policy,
        )
    except Exception as exc:  # noqa: BLE001 - caller keeps filesystem fallback.
        logger.debug("generation-local finding ingest failed: %s", exc)
        return 0


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v == v]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _q25(values: list[float]) -> float | None:
    clean = sorted(v for v in values if v == v)
    if not clean:
        return None
    idx = int(0.25 * (len(clean) - 1))
    return float(clean[idx])


def _cell_float(cell: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _as_float(cell.get(key))
        if value is not None:
            return value
    return None


_SCORING_METRIC_KEYS = (
    "mean_score",
    "score",
    "metric_value",
    "lane_metric_value",
    "primary_metric_value",
    "taskscore",
    "task_score",
    "test_score",
    "eval_score",
    "positive_cell_fraction",
)


def _infer_strategy_family(variant_name: str, summary: dict[str, Any]) -> str:
    metrics = summary.get("current_aggregate")
    metrics = metrics if isinstance(metrics, dict) else {}
    explicit = metrics.get("strategy_family") or summary.get("strategy_family")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return "task_candidate"


def _status_text(*values: Any) -> str:
    return " ".join(str(v or "").lower() for v in values)


def _normalized_status_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _status_has_any(text: str, *needles: str) -> bool:
    normalized = _normalized_status_text(text)
    if not normalized:
        return False
    tokens = [token for token in normalized.split("_") if token]
    if not tokens:
        return False
    negators = {"not", "non", "no", "without"}
    false_tokens = {"false", "no", "0"}
    for needle in needles:
        needle_tokens = [token for token in _normalized_status_text(needle).split("_") if token]
        if not needle_tokens:
            continue
        width = len(needle_tokens)
        for idx in range(0, len(tokens) - width + 1):
            if tokens[idx : idx + width] != needle_tokens:
                continue
            if any(token in negators for token in tokens[max(0, idx - 3) : idx]):
                continue
            next_idx = idx + width
            if (
                next_idx < len(tokens)
                and tokens[next_idx] in false_tokens
                and not any(token in false_tokens for token in needle_tokens)
            ):
                continue
            return True
    return False


def _status_marker_text(summary: dict[str, Any], summary_path: Path | None) -> str:
    parts = [
        summary.get("status"),
        summary.get("result_status"),
        summary.get("tier_status"),
        summary.get("final_status"),
        summary.get("evidence_stage"),
    ]
    if summary_path is not None:
        parts.append(summary_path.name)
        parent_marker = _terminal_path_status_marker(summary_path.parent.name)
        if parent_marker:
            parts.append(parent_marker)
    return _normalized_status_text(" ".join(str(part or "") for part in parts))


def _terminal_path_status_marker(name: str) -> str | None:
    tokens = [token for token in _normalized_status_text(name).split("_") if token]
    if not tokens:
        return None
    if tokens[-2:] == ["summary", "only"]:
        return "summary_only"
    if tokens[-2:] == ["cheap", "probe"]:
        return "cheap_probe"
    if tokens[-1] in {"scout", "smoke", "capped"}:
        if len(tokens) >= 2 and tokens[-2] in {"not", "non", "no", "without"}:
            return None
        return tokens[-1]
    return None


def _is_summary_only_status(text: str) -> bool:
    return _status_has_any(text, "summary_only")


def _is_scout_or_smoke_status(text: str) -> bool:
    return _status_has_any(text, "scout", "smoke", "cheap_probe")


def _is_capped_status(text: str) -> bool:
    return _status_has_any(text, "capped", "capped_at", "cap_at")


def _is_not_scored_status(text: str) -> bool:
    return _status_has_any(
        text,
        "not_scored",
        "unscored",
        "scored_complete_false",
        "complete_eval_false",
        "protocol_invalid",
    )


def _is_explicit_complete_status(text: str) -> bool:
    """Return True only for an evaluator-authored terminal success status."""

    normalized = _normalized_status_text(text)
    return normalized in {
        "complete_eval",
        "full_evaluation",
        "scored_complete",
    }


def _summary_declares_completion(
    summary: dict[str, Any],
    aggregate: dict[str, Any],
    *status_values: Any,
) -> bool:
    explicit_flag = _summary_completion_flag(summary, aggregate)
    if explicit_flag is not None:
        return explicit_flag
    return any(_is_explicit_complete_status(str(value or "")) for value in status_values)


def _apply_path_status_markers(
    summary: dict[str, Any], summary_path: Path | None
) -> dict[str, Any]:
    marker_text = _status_marker_text(summary, summary_path)
    if _is_summary_only_status(marker_text):
        summary["summary_only"] = True
    if _is_scout_or_smoke_status(marker_text):
        summary["scout_only"] = True
        if "smoke" in marker_text:
            summary["is_smoke_eval"] = True
        else:
            summary["is_scout_eval"] = True
    if _is_capped_status(marker_text):
        summary["capped"] = True
        summary["result_capped"] = True
    if _is_not_scored_status(marker_text):
        summary["scored_complete"] = False
    return summary


def _summary_flag(summary: dict[str, Any], aggregate: dict[str, Any], *keys: str) -> bool:
    value = _summary_flag_value(summary, aggregate, *keys)
    return bool(value) if value is not None else False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
    return False


def _summary_sources(
    summary: dict[str, Any], aggregate: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    roots = (summary, aggregate)
    nested = tuple(
        source for parent in roots if isinstance((source := parent.get("metrics")), dict)
    )
    return (*roots, *nested)


def _durable_routing_marker_value(
    summary: dict[str, Any],
    aggregate: dict[str, Any],
    key: str,
) -> Any:
    for source in _summary_sources(summary, aggregate):
        value = source.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value != 0:
                return value
            continue
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"0", "false", "no", "n", "off"}:
                continue
            return value.strip()
        return value
    return None


def _summary_flag_value(
    summary: dict[str, Any], aggregate: dict[str, Any], *keys: str
) -> bool | None:
    for source in _summary_sources(summary, aggregate):
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return bool(value)
            if isinstance(value, str):
                text = value.strip().lower()
                if text in {"1", "true", "yes", "y", "on"}:
                    return True
                if text in {"0", "false", "no", "n", "off"}:
                    return False
    return None


def _summary_completion_flag(
    summary: dict[str, Any],
    aggregate: dict[str, Any],
    *,
    aggregate_first: bool = False,
) -> bool | None:
    def explicit_source(source: dict[str, Any]) -> dict[str, Any]:
        nested_metrics = source.get("metrics")
        root_inferred = _truthy(source.get("_inferred_scored_complete"))
        nested_inferred = isinstance(nested_metrics, dict) and _truthy(
            nested_metrics.get("_inferred_scored_complete")
        )
        if not root_inferred and not nested_inferred:
            return source
        explicit = dict(source)
        if root_inferred:
            explicit.pop("scored_complete", None)
            explicit.pop("is_scored_complete", None)
        if nested_inferred and isinstance(nested_metrics, dict):
            explicit_metrics = dict(nested_metrics)
            explicit_metrics.pop("scored_complete", None)
            explicit_metrics.pop("is_scored_complete", None)
            explicit["metrics"] = explicit_metrics
        return explicit

    first, second = (aggregate, summary) if aggregate_first else (summary, aggregate)
    return _summary_flag_value(
        explicit_source(first),
        explicit_source(second),
        *_COMPLETION_FLAG_KEYS,
    )


def _canonicalize_completion_flag(
    aggregate: dict[str, Any],
    summary: dict[str, Any],
    *,
    aggregate_first: bool = False,
) -> None:
    decision = _summary_completion_flag(
        summary,
        aggregate,
        aggregate_first=aggregate_first,
    )
    if decision is None:
        if _truthy(summary.get("_inferred_scored_complete")) or _truthy(
            aggregate.get("_inferred_scored_complete")
        ):
            aggregate.pop("scored_complete", None)
            aggregate.pop("is_scored_complete", None)
        return
    for key in _COMPLETION_FLAG_KEYS:
        aggregate.pop(key, None)
    aggregate["scored_complete"] = decision


def _summary_explicit_false(summary: dict[str, Any], aggregate: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = summary.get(key, aggregate.get(key))
        if value is False:
            return True
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
            return True
        if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "n", "off"}:
            return True
    return False


def _scoring_metric_keys(extra_keys: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    configured_keys: list[str] = []
    for key in extra_keys or []:
        text = str(key).strip()
        if text and text not in configured_keys:
            configured_keys.append(text)
    if configured_keys:
        return tuple(configured_keys)
    return tuple(_SCORING_METRIC_KEYS)


def _metric_name_from_axis(axis: Any) -> str:
    if isinstance(axis, (list, tuple)) and axis:
        return str(axis[0] or "").strip()
    if isinstance(axis, dict):
        return str(axis.get("name") or axis.get("metric") or "").strip()
    return str(axis or "").strip()


def result_artifact_options_from_task_spec(task_spec: Any) -> dict[str, Any]:
    """Return task-configured result materialization options."""

    from praxist.plugins.workflow_stages.research_loop.backend.scoreless import is_scoreless

    evaluation = getattr(task_spec, "evaluation", None)
    gems_cfg = getattr(task_spec, "gems", None)
    primary_metric = getattr(evaluation, "primary_metric", None)
    scoring_metric_keys: list[str] = []

    def add_scoring_metric(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in scoring_metric_keys:
            scoring_metric_keys.append(text)

    add_scoring_metric(primary_metric)
    for metric in getattr(evaluation, "aux_metrics", None) or []:
        add_scoring_metric(metric)
    for axis in getattr(evaluation, "anchor_metrics", None) or []:
        add_scoring_metric(_metric_name_from_axis(axis))
    for lane in getattr(evaluation, "frontier_lanes", None) or []:
        if not isinstance(lane, dict):
            continue
        for field_name in ("require_metrics",):
            for metric in lane.get(field_name) or []:
                add_scoring_metric(metric)
        for field_name in ("axes", "optional_axes"):
            for axis in lane.get(field_name) or []:
                add_scoring_metric(_metric_name_from_axis(axis))
        for field_name in ("min_metrics", "max_metrics"):
            bounds = lane.get(field_name) or {}
            if isinstance(bounds, dict):
                for metric in bounds:
                    add_scoring_metric(metric)
    for field_name in (
        "primary_metric_keys",
        "secondary_metric_keys",
        "lower_tail_metric_keys",
        "validation_metric_keys",
    ):
        for metric in getattr(gems_cfg, field_name, None) or []:
            add_scoring_metric(metric)
    return {
        "materialize_result_artifacts": not is_scoreless(task_spec)
        and bool(getattr(gems_cfg, "result_artifact_materialization", True)),
        "result_artifact_default_lane": str(
            getattr(gems_cfg, "result_artifact_default_lane", "performance") or "performance"
        ),
        "result_artifact_default_family": str(
            getattr(gems_cfg, "result_artifact_default_family", "task_candidate")
            or "task_candidate"
        ),
        "result_cell_metric_derivations": list(
            getattr(gems_cfg, "result_cell_metric_derivations", []) or []
        ),
        "result_metric_aliases": dict(getattr(gems_cfg, "result_metric_aliases", {}) or {}),
        # Keep the task primary metric first.  The order is also used when a
        # compact derived finding renders one representative score.
        "result_scoring_metric_keys": scoring_metric_keys,
        "result_maturity_policy": dict(getattr(evaluation, "maturity_policy", {}) or {}),
    }


def collect_loop_findings(
    loop: Any,
    gen_id: int,
    *,
    do_ingest: bool = True,
) -> list[dict[str, Any]]:
    """Collect generation findings using the loop's task and boundary context."""

    primary_metric = getattr(getattr(loop.task_spec, "evaluation", None), "primary_metric", None)
    boundary_options: dict[str, Any] = {}
    active_boundary = getattr(loop, "_boundary_evidence_cutoff", None)
    if active_boundary is not None and active_boundary[0] == int(gen_id):
        boundary_options = {
            "evidence_cutoff": active_boundary[1],
            "evidence_source_snapshot": dict(active_boundary[2]),
        }
    findings = collect_findings_for_generation(
        findings_dir=loop.findings_dir,
        gen_id=gen_id,
        local_mode=loop.local_mode,
        do_ingest=do_ingest,
        primary_metric=primary_metric,
        task_spec=loop.task_spec,
        **boundary_options,
        **result_artifact_options_from_task_spec(loop.task_spec),
    )
    if not boundary_options:
        return findings
    findings = annotate_late_boundary_findings(
        findings,
        run_dir=_run_dir_from_findings_dir(loop.findings_dir),
        findings_dir=loop.findings_dir,
        gen_id=gen_id,
        cutoff=boundary_options["evidence_cutoff"],
        evidence_source_snapshot=boundary_options["evidence_source_snapshot"],
    )
    if loop.local_mode and do_ingest:
        persist_boundary_validation_findings(findings)
    return findings


def _has_scored_metrics(
    aggregate: dict[str, Any],
    extra_keys: list[str] | tuple[str, ...] | None = None,
) -> bool:
    return any(
        _as_float(aggregate.get(key)) is not None for key in _scoring_metric_keys(extra_keys)
    )


def _has_scored_cells(
    cells: list[Any],
    extra_keys: list[str] | tuple[str, ...] | None = None,
) -> bool:
    keys = _scoring_metric_keys(extra_keys)
    return any(
        isinstance(cell, dict) and any(_as_float(cell.get(key)) is not None for key in keys)
        for cell in cells
    )


def _cell_derivation_source_keys(
    derivations: list[dict[str, Any]] | None,
    *,
    validation_only: bool,
) -> tuple[str, ...]:
    keys: list[str] = []
    for rule in derivations or []:
        if not isinstance(rule, dict):
            continue
        if _truthy(rule.get("validation_only")) != validation_only:
            continue
        source_keys = rule.get("source_keys") or []
        if not isinstance(source_keys, list):
            continue
        for key in source_keys:
            text = str(key).strip()
            if text and text not in keys:
                keys.append(text)
    return tuple(keys)


def _first_scoring_metric_value(
    metrics: dict[str, Any],
    extra_keys: list[str] | tuple[str, ...] | None = None,
) -> Any:
    for key in _scoring_metric_keys(extra_keys):
        value = metrics.get(key)
        if _as_float(value) is not None:
            return value
    return None


def _failure_evidence_count(*sources: Any) -> int:
    keys = (
        "failed_units",
        "failed_eval_units",
        "error_units",
        "missing_units",
        "incomplete_units",
        "failed_unit_count",
        "n_failed_units",
        "error_unit_count",
        "n_error_units",
        "missing_unit_count",
        "n_missing_units",
        "incomplete_unit_count",
        "n_incomplete_units",
        "failed_cells",
        "failed_eval_cells",
        "error_cells",
        "missing_cells",
        "incomplete_cells",
        "failed_cell_count",
        "n_failed_cells",
        "error_cell_count",
        "n_error_cells",
        "missing_cell_count",
        "n_missing_cells",
        "incomplete_cell_count",
        "n_incomplete_cells",
    )
    total = 0
    for source in _sources_with_nested_metrics(*sources):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (list, dict)):
                total += len(value)
            elif isinstance(value, (int, float)):
                try:
                    total += max(0, int(value))
                except (TypeError, ValueError):
                    continue
            elif isinstance(value, str):
                text = value.strip()
                if not text:
                    continue
                try:
                    total += max(0, int(float(text)))
                except (TypeError, ValueError):
                    total += 1
    return total


def _sources_with_nested_metrics(*sources: Any) -> tuple[dict[str, Any], ...]:
    expanded: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        for candidate in (source, source.get("metrics")):
            if not isinstance(candidate, dict) or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            expanded.append(candidate)
    return tuple(expanded)


def _has_explicit_failed_cell_details(*sources: Any) -> bool:
    detail_keys = (
        "failed_units",
        "failed_eval_units",
        "error_units",
        "missing_units",
        "incomplete_units",
        "failed_cells",
        "failed_eval_cells",
        "error_cells",
        "missing_cells",
        "incomplete_cells",
    )
    for source in _sources_with_nested_metrics(*sources):
        for key in detail_keys:
            value = source.get(key)
            if isinstance(value, (list, dict)) and len(value) > 0:
                return True
    return False


def _has_list_failed_cell_details(*sources: Any) -> bool:
    detail_keys = (
        "failed_units",
        "failed_eval_units",
        "error_units",
        "missing_units",
        "incomplete_units",
        "failed_cells",
        "failed_eval_cells",
        "error_cells",
        "missing_cells",
        "incomplete_cells",
    )
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in detail_keys:
            value = source.get(key)
            if isinstance(value, list) and len(value) > 0:
                return True
    return False


def _protocol_integrity_fact_source(
    summary: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "protocol_integrity_passed",
        "protocol_integrity_failed",
        "protocol_integrity_status",
        "protocol_status",
        "protocol_integrity_violation_count",
        "suspect_protocol",
        "suspect_fixed_weight_eval",
    )
    return next(
        (
            source
            for source in _summary_sources(summary, aggregate)
            if any(key in source for key in keys)
        ),
        {},
    )


def _protocol_integrity_passed_value(
    summary: dict[str, Any], aggregate: dict[str, Any]
) -> bool | None:
    source = _protocol_integrity_fact_source(summary, aggregate)
    return _summary_flag_value(source, {}, "protocol_integrity_passed")


def _protocol_integrity_failure_status(
    summary: dict[str, Any], aggregate: dict[str, Any]
) -> str | None:
    source = _protocol_integrity_fact_source(summary, aggregate)
    for key in ("protocol_integrity_status", "protocol_status"):
        status = str(source.get(key) or "")
        if _status_has_any(status, "failed", "fail", "invalid", "protocol_invalid"):
            return status
    if source:
        return None
    result_source = next(
        (
            item
            for item in _summary_sources(summary, aggregate)
            if item.get("result_status") not in (None, "")
        ),
        {},
    )
    result_status = str(result_source.get("result_status") or "")
    if _status_has_any(result_status, "protocol_invalid"):
        return result_status
    return None


def _protocol_integrity_failed(summary: dict[str, Any], aggregate: dict[str, Any]) -> bool:
    source = _protocol_integrity_fact_source(summary, aggregate)
    count = source.get("protocol_integrity_violation_count")
    if not isinstance(count, bool):
        try:
            if count is not None and float(count) > 0:
                return True
        except (TypeError, ValueError):
            pass
    if _protocol_integrity_passed_value(summary, aggregate) is False:
        return True
    if _protocol_integrity_failure_status(summary, aggregate) is not None:
        return True
    return _summary_flag(
        source,
        {},
        "suspect_protocol",
        # Legacy artifact alias. New task outputs should use suspect_protocol.
        "suspect_fixed_weight_eval",
        "protocol_integrity_failed",
    )


def _is_bad_result_status(text: str, *, allow_partial: bool = False) -> bool:
    normalized = _normalized_status_text(text)
    tokens = {token for token in normalized.split("_") if token}
    bad_tokens = (
        "crash",
        "crashed",
        "cancel",
        "cancelled",
        "canceled",
        "error",
        "incomplete",
        "interrupt",
        "interrupted",
        "killed",
        "oom",
        "pending",
        "running",
        "stale",
        "timeout",
        "timed_out",
        "unscored",
        "protocol_invalid",
    )
    if not allow_partial:
        bad_tokens = (*bad_tokens, "partial")
    if _status_has_any(normalized, *bad_tokens):
        return True
    return bool(
        _status_has_any(normalized, "failed", "failure")
        and not tokens
        & {
            "constraint",
            "constraints",
            "hard",
            "promotion",
            "promotable",
            "eligible",
            "eligibility",
            "repair",
            "risk",
        }
    )


def _copy_research_metadata(metrics: dict[str, Any], *sources: Any) -> None:
    containers: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        containers.append(source)
        nested_extra = source.get("extra")
        if isinstance(nested_extra, dict):
            containers.append(nested_extra)
    for key in RESULT_RESEARCH_METADATA_KEYS:
        if metrics.get(key) not in (None, ""):
            continue
        for source in containers:
            value = source.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            if key == "is_negative" and isinstance(value, bool):
                metrics[key] = value
                break
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                metrics[key] = value
                break


def _copy_result_signal_metadata(metrics: dict[str, Any], *sources: Any) -> None:
    """Copy a bounded generic signal contract without promoting it to a score."""

    for key in _RESULT_SIGNAL_METADATA_KEYS:
        if metrics.get(key) not in (None, ""):
            continue
        for source in sources:
            if not isinstance(source, dict) or key not in source:
                continue
            value = source.get(key)
            if isinstance(value, (bool, int, float, str)) and not (
                isinstance(value, str) and not value.strip()
            ):
                metrics[key] = value.strip() if isinstance(value, str) else value
                break


def _copy_result_dimension_metadata(metrics: dict[str, Any], *sources: Any) -> None:
    """Preserve shallow realized-dimension maps without copying arbitrary payloads."""

    containers: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        containers.append(source)
        nested_extra = source.get("extra")
        if isinstance(nested_extra, dict):
            containers.append(nested_extra)
    for key in _RESULT_DIMENSION_METADATA_KEYS:
        for source in containers:
            raw = source.get(key)
            if not isinstance(raw, dict):
                continue
            compact = {
                str(name).strip(): value.strip() if isinstance(value, str) else value
                for name, value in raw.items()
                if str(name).strip()
                and isinstance(value, (bool, int, float, str))
                and not (isinstance(value, str) and not value.strip())
            }
            if compact:
                metrics["design_dimensions"] = compact
                metrics.pop("realized_dimensions", None)
                return


def _scalar_dict(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {
        str(key): value
        for key, value in source.items()
        if isinstance(value, (bool, int, float, str)) or value is None
    }


def _per_dataset_metric_suffix(value: Any) -> str:
    suffix = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    suffix = re.sub(r"_+", "_", suffix).strip("_")
    return suffix[:80] or "dataset"


def _per_dataset_metric_name(value: Any) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip())
    name = re.sub(r"_+", "_", name).strip("._-")
    return name[:80]


def _per_dataset_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return None


def _flatten_per_dataset_metrics(source: Any) -> dict[str, Any]:
    """Expose bounded per-dataset diagnostics from nested task summaries."""

    if not isinstance(source, dict):
        return {}
    per_dataset = source.get("per_dataset")
    if not isinstance(per_dataset, dict):
        return {}
    flattened: dict[str, Any] = {}
    for dataset, payload in per_dataset.items():
        if not isinstance(payload, dict):
            continue
        suffix = _per_dataset_metric_suffix(dataset)
        for raw_metric_name, raw_value in payload.items():
            metric_name = _per_dataset_metric_name(raw_metric_name)
            if not metric_name:
                continue
            value = _per_dataset_metric_value(raw_value)
            if value is not None:
                flattened[f"{metric_name}_{suffix}"] = value
            if len(flattened) >= 64:
                return flattened
    return flattened


def _result_path_from_tier(summary_path: Path | None, tier: dict[str, Any]) -> Path | None:
    raw = tier.get("result_path")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute() and summary_path is not None:
        path = (summary_path.parent / path).resolve()
    return path


def normalized_result_summary(
    summary: dict[str, Any],
    *,
    summary_path: Path | None = None,
    maturity_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize supported task tier summaries into the canonical aggregate shape."""

    if isinstance(summary.get("current_aggregate"), dict):
        normalized = dict(summary)
        aggregate = dict(summary["current_aggregate"])
        aggregate.update(_flatten_per_dataset_metrics(aggregate))
        canonicalize_evaluation_unit_metadata(aggregate, summary)
        _canonicalize_completion_flag(aggregate, summary, aggregate_first=True)
        normalized["current_aggregate"] = aggregate
        canonicalize_evaluation_unit_metadata(normalized, aggregate)
        return _apply_path_status_markers(normalized, summary_path)
    if isinstance(summary.get("metrics"), dict) and not summary.get("tiers"):
        normalized = dict(summary)
        aggregate = _scalar_dict(summary.get("metrics"))
        # Generic task evaluators commonly write ``summary.json`` with
        # metrics under ``metrics`` and task-level scalar metadata such as
        # ``frontier_lane`` or ``evidence_stage`` at the top level. Keep the
        # same canonical aggregate shape as tiered summaries so boundary
        # promotion cannot miss a completed result merely because the agent's
        # later share_finding call was interrupted.
        for key, value in _scalar_dict(summary).items():
            aggregate.setdefault(key, value)
        aggregate.update(_flatten_per_dataset_metrics(summary.get("metrics")))
        canonicalize_evaluation_unit_metadata(aggregate, summary)
        _canonicalize_completion_flag(aggregate, summary)
        normalized["current_aggregate"] = aggregate
        canonicalize_evaluation_unit_metadata(normalized, aggregate)
        return _apply_path_status_markers(normalized, summary_path)
    tiers = summary.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        normalized = dict(summary)
        aggregate = _scalar_dict(summary)
        if aggregate:
            _canonicalize_completion_flag(aggregate, summary)
            normalized["current_aggregate"] = aggregate
        return _apply_path_status_markers(normalized, summary_path)
    normalized = dict(summary)
    tier_records = [tier for tier in tiers if isinstance(tier, dict)]
    if not tier_records:
        return normalized
    latest = tier_records[-1]
    metrics_summary = latest.get("metrics_summary")
    aggregate = _scalar_dict(metrics_summary)
    aggregate.update(_flatten_per_dataset_metrics(metrics_summary))
    result_path = _result_path_from_tier(summary_path, latest)
    result_payload = (
        _load_json(result_path) if result_path is not None and result_path.exists() else None
    )
    if isinstance(result_payload, dict):
        aggregate.update(_scalar_dict(result_payload))
        aggregate.update(_flatten_per_dataset_metrics(result_payload))
    _copy_result_dimension_metadata(
        aggregate,
        metrics_summary,
        result_payload,
        latest,
        summary,
    )
    _canonicalize_completion_flag(aggregate, summary)
    canonicalize_evaluation_unit_metadata(aggregate, summary)
    gate = latest.get("gate") if isinstance(latest.get("gate"), dict) else {}
    tier = latest.get("tier") or aggregate.get("tier")
    aggregate.setdefault("promotion_eligible", aggregate.get("promotion_eligible"))
    if aggregate.get("promotion_eligible") is None and isinstance(gate, dict) and "passed" in gate:
        aggregate["promotion_eligible"] = bool(gate.get("passed", False))
    normalized["current_aggregate"] = aggregate
    normalized["tier_reached"] = tier
    normalized["completed_tier"] = tier
    normalized["tier_status"] = (
        latest.get("status") or normalized.get("final_status") or latest.get("returncode")
    )
    for key in ("evidence_stage", "eval_stage", "stage"):
        value = latest.get(key)
        if value not in (None, ""):
            normalized.setdefault(key, value)
            aggregate.setdefault(key, value)
    canonicalize_evaluation_unit_metadata(normalized, aggregate)
    normalized = _apply_path_status_markers(normalized, summary_path)
    status_text = _status_text(
        normalized.get("tier_status"),
        normalized.get("final_status"),
        normalized.get("result_status"),
        aggregate.get("result_status"),
    )
    summary_only = _summary_flag(
        summary, aggregate, "summary_only", "is_summary_only"
    ) or _is_summary_only_status(status_text)
    scout_only = _summary_flag(
        summary, aggregate, "scout_only", "is_scout_eval", "is_smoke_eval"
    ) or _is_scout_or_smoke_status(status_text)
    capped = _summary_flag(
        summary, aggregate, "capped", "is_capped", "result_capped"
    ) or _is_capped_status(status_text)
    completion_flag = _summary_completion_flag(summary, aggregate)
    explicit_not_scored = completion_flag is False or _is_not_scored_status(status_text)
    explicit_scored = (
        _summary_flag_value(
            summary,
            aggregate,
            "scored_complete",
            "is_scored_complete",
        )
        is True
    )
    maturity = evidence_maturity_snapshot(normalized, maturity_policy)
    task_stage_authorized_maturity = bool(
        maturity.get("mature_enough") is True
        and task_authorizes_descriptive_maturity(normalized, maturity_policy)
    )
    descriptive_mode_authorized = bool(
        maturity.get("mature_enough") is True
        and task_authorizes_descriptive_maturity(
            normalized,
            maturity_policy,
            maturity=maturity,
        )
    )
    has_scores = _has_scored_metrics(aggregate) or explicit_scored
    completion_declared = (
        completion_flag
        if completion_flag is not None
        else (
            task_stage_authorized_maturity
            or _summary_declares_completion(
                summary,
                aggregate,
                normalized.get("tier_status"),
                normalized.get("final_status"),
                summary.get("result_status"),
                aggregate.get("result_status"),
            )
        )
    )
    is_complete = bool(
        has_scores
        and completion_declared
        and not summary_only
        and (descriptive_mode_authorized or not scout_only)
        and (descriptive_mode_authorized or not capped)
        and not explicit_not_scored
    )
    if summary_only:
        default_status = "summary_only"
    elif scout_only and not descriptive_mode_authorized:
        default_status = "scout_or_smoke"
    elif capped and not descriptive_mode_authorized:
        default_status = "capped"
    elif explicit_not_scored:
        default_status = "not_scored_complete"
    elif has_scores and not completion_declared:
        default_status = "unknown_maturity"
    else:
        default_status = "scored_complete" if is_complete else "unscored_artifact"
    explicit_result_status = summary.get("result_status")
    normalized["result_status"] = explicit_result_status or default_status
    if not explicit_result_status:
        normalized["_inferred_result_status"] = True
    has_explicit_completion_flag = any(
        key in summary or key in aggregate
        for key in (
            "scored_complete",
            "is_scored_complete",
            "complete_eval",
            "is_complete_eval",
        )
    )
    if is_complete:
        normalized["scored_complete"] = True
        if not has_explicit_completion_flag:
            normalized["_inferred_scored_complete"] = True
    elif (
        explicit_not_scored
        or summary_only
        or ((scout_only or capped) and not descriptive_mode_authorized)
    ):
        normalized["scored_complete"] = False
    elif not has_explicit_completion_flag:
        # Unknown is not false. The task-owned maturity policy may still prove
        # this result complete from effort/coverage ratios or stage labels.
        normalized.pop("scored_complete", None)
    if summary_only:
        normalized["summary_only"] = True
    if scout_only:
        normalized["scout_only"] = True
    if capped:
        normalized["capped"] = True
    normalized["source_result_kind"] = summary_path.name if summary_path is not None else ""
    return normalized


def _result_summary_producer_identity(summary: dict[str, Any]) -> str:
    """Return a producer-owned top-level identity, never a nested fallback."""

    for key in (*_RESULT_CHILD_IDENTITY_KEYS, "variant_id"):
        reported = str(summary.get(key) or "").strip()
        if reported:
            return reported
    return ""


def result_summary_variant_name(summary_path: Path, summary: dict[str, Any], run_dir: Path) -> str:
    """Return the concrete variant identity represented by a result summary path.

    A producer-owned top-level id is authoritative. Historical summaries often
    copied one parent/family id into ``metrics`` or ``current_aggregate`` for
    several children; those nested values must not collapse distinct artifact
    paths into one entity.
    """

    producer_identity = _result_summary_producer_identity(summary)
    if producer_identity:
        return producer_identity
    # A child-specific legacy id remains more precise than an artifact path.
    # Do not apply this fallback to nested ``variant_id`` or ``variant_name``:
    # historical sweep summaries often copied a shared parent/display label
    # into those fields for every child.
    for source_key in ("current_aggregate", "metrics"):
        source = summary.get(source_key)
        if not isinstance(source, dict):
            continue
        for key in _RESULT_CHILD_IDENTITY_KEYS:
            reported = str(source.get(key) or "").strip()
            if reported:
                return reported
    results_dir = Path(run_dir) / "results"
    try:
        relative_parts = summary_path.relative_to(results_dir).parts
    except ValueError:
        relative_parts = ()
    name = summary_path.name
    filename_variant = result_summary_filename_variant(name)
    if filename_variant:
        return filename_variant
    if len(relative_parts) > 1:
        # Without a producer-authored stable id, preserve the entire relative
        # parent path. Praxist cannot infer which task-owned directory level is the
        # protocol, variant, seed, fold, or another identity dimension.
        return "/".join(relative_parts[:-1])
    # ``variant_name`` is a display label, not a stable identity. It is only a
    # fallback when the artifact layout provides no concrete child path.
    reported = str(summary.get("variant_name") or "").strip()
    if reported:
        return reported
    return summary_path.stem.replace("_tiered_eval_summary", "")


def _apply_result_cell_metric_derivations(
    metrics: dict[str, Any],
    *,
    cells: list[Any],
    derivations: list[dict[str, Any]] | None,
    aliases: dict[str, str] | None,
) -> None:
    primary = [c for c in cells if isinstance(c, dict) and not _truthy(c.get("validation_only"))]
    validation = [c for c in cells if isinstance(c, dict) and _truthy(c.get("validation_only"))]
    for rule in derivations or []:
        if not isinstance(rule, dict):
            continue
        out_key = str(rule.get("name") or "").strip()
        source_keys = rule.get("source_keys") or []
        if not out_key or not isinstance(source_keys, list):
            continue
        if metrics.get(out_key) is not None:
            continue
        selected = validation if _truthy(rule.get("validation_only")) else primary
        values = [
            _cell_float(c, *(str(key) for key in source_keys if str(key).strip())) for c in selected
        ]
        valid_values = [v for v in values if v is not None]
        aggregate = str(rule.get("aggregate") or "mean").strip().lower()
        derived = _q25(valid_values) if aggregate == "q25" else _mean(valid_values)
        if derived is not None:
            metrics[out_key] = derived
    for out_key, source_key in (aliases or {}).items():
        if metrics.get(out_key) is None and metrics.get(source_key) is not None:
            metrics[out_key] = metrics[source_key]


def _result_summary_metrics(
    summary: dict[str, Any],
    *,
    cell_metric_derivations: list[dict[str, Any]] | None = None,
    metric_aliases: dict[str, str] | None = None,
    scoring_metric_keys: list[str] | tuple[str, ...] | None = None,
    maturity_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate = summary.get("current_aggregate")
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    summary_metrics = summary.get("metrics")
    summary_metrics = summary_metrics if isinstance(summary_metrics, dict) else {}
    summary_extra = summary.get("extra")
    summary_extra = summary_extra if isinstance(summary_extra, dict) else {}
    metrics: dict[str, Any] = {}
    for key, value in aggregate.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            metrics[key] = value
    for key in _RESULT_IDENTITY_PRESERVE_KEYS:
        value = summary_metrics.get(key)
        if value not in (None, "") and not isinstance(value, (bool, dict, list)):
            metrics[key] = value
    _copy_research_metadata(metrics, summary_metrics, aggregate, summary_extra, summary)
    _copy_result_signal_metadata(
        metrics,
        summary_metrics,
        aggregate,
        summary_extra,
        summary,
    )
    _copy_result_dimension_metadata(
        metrics,
        summary_metrics,
        aggregate,
        summary_extra,
        summary,
    )
    for key in EFFECTIVE_CONFIG_METADATA_KEYS:
        metrics.pop(key, None)
    maturity = evidence_maturity_snapshot(summary, maturity_policy)
    for key in ("effort_ratio", "coverage_ratio"):
        if maturity[key] is not None:
            metrics[key] = maturity[key]
    canonical_variant_id = summary.get("canonical_variant_id")
    if canonical_variant_id not in (None, "") and not isinstance(
        canonical_variant_id, (dict, list)
    ):
        metrics["canonical_variant_id"] = canonical_variant_id
    tier_history = summary.get("tier_history")
    latest_tier = tier_history[-1] if isinstance(tier_history, list) and tier_history else {}
    latest_tier = latest_tier if isinstance(latest_tier, dict) else {}
    tier = summary.get("tier_reached") or summary.get("completed_tier") or latest_tier.get("tier")
    tier_status = (
        summary.get("tier_status") or summary.get("final_status") or latest_tier.get("status")
    )

    cells = summary.get("evaluation_unit_records")
    if not isinstance(cells, list):
        cells = summary.get("all_eval_units")
    if not isinstance(cells, list):
        cells = summary.get("all_eval_cells")
    if not isinstance(cells, list):
        cells = summary.get("all_paired_cells")
    cells = cells if isinstance(cells, list) else []
    _apply_result_cell_metric_derivations(
        metrics,
        cells=cells,
        derivations=cell_metric_derivations,
        aliases=metric_aliases,
    )
    status_values = [
        tier_status,
        summary.get("final_status"),
        aggregate.get("result_status"),
        summary.get("evidence_stage"),
        aggregate.get("evidence_stage"),
    ]
    if not bool(summary.get("_inferred_result_status")):
        status_values.append(summary.get("result_status"))
    status_text = _status_text(*status_values)
    task_stage_authorized_maturity = bool(
        maturity.get("mature_enough") is True
        and task_authorizes_descriptive_maturity(summary, maturity_policy)
    )
    descriptive_mode_authorized = bool(
        maturity.get("mature_enough") is True
        and task_authorizes_descriptive_maturity(
            summary,
            maturity_policy,
            maturity=maturity,
        )
    )
    primary_cells = [
        cell
        for cell in cells
        if isinstance(cell, dict) and not _truthy(cell.get("validation_only"))
    ]
    validation_cells = [
        cell for cell in cells if isinstance(cell, dict) and _truthy(cell.get("validation_only"))
    ]
    aggregate_has_scores = _has_scored_metrics(metrics, scoring_metric_keys)
    primary_cells_have_scores = _has_scored_cells(primary_cells, scoring_metric_keys)
    validation_cells_have_scores = _has_scored_cells(validation_cells, scoring_metric_keys)
    primary_cells_have_configured_evidence = _has_scored_cells(
        primary_cells,
        _cell_derivation_source_keys(cell_metric_derivations, validation_only=False),
    )
    validation_cells_have_configured_evidence = _has_scored_cells(
        validation_cells,
        _cell_derivation_source_keys(cell_metric_derivations, validation_only=True),
    )
    validation_only_scored_evidence = (
        bool(validation_cells)
        and not (
            primary_cells_have_scores
            or (aggregate_has_scores and primary_cells_have_configured_evidence)
        )
        and (
            aggregate_has_scores
            or validation_cells_have_scores
            or validation_cells_have_configured_evidence
        )
    )
    has_scores = aggregate_has_scores or primary_cells_have_scores or validation_cells_have_scores
    failed_count = _failure_evidence_count(summary, aggregate, latest_tier)
    is_bad_status = _is_bad_result_status(
        status_text,
        allow_partial=descriptive_mode_authorized,
    )
    protocol_failed = _protocol_integrity_failed(summary, aggregate)
    summary_only = _summary_flag(
        summary, aggregate, "summary_only", "is_summary_only"
    ) or _is_summary_only_status(status_text)
    scout_only = _summary_flag(
        summary, aggregate, "scout_only", "is_scout_eval", "is_smoke_eval"
    ) or _is_scout_or_smoke_status(status_text)
    capped = _summary_flag(
        summary, aggregate, "capped", "is_capped", "result_capped"
    ) or _is_capped_status(status_text)
    partial_status = _status_has_any(status_text, "partial", "partial_cohort", "partial_eval")
    unknown_maturity = _status_has_any(status_text, "unknown")
    completion_flag = _summary_completion_flag(summary, aggregate, aggregate_first=True)
    explicit_not_scored = (
        completion_flag is False
        or _is_not_scored_status(status_text)
        or validation_only_scored_evidence
    )
    completion_declared = (
        completion_flag
        if completion_flag is not None
        else (
            task_stage_authorized_maturity
            or _summary_declares_completion(
                summary,
                aggregate,
                tier_status,
                summary.get("final_status"),
                aggregate.get("result_status"),
                (
                    summary.get("result_status")
                    if not summary.get("_inferred_result_status")
                    else None
                ),
            )
        )
    )
    is_complete = bool(
        has_scores
        and completion_declared
        and failed_count == 0
        and not is_bad_status
        and not protocol_failed
        and not summary_only
        and (descriptive_mode_authorized or not scout_only)
        and (descriptive_mode_authorized or not capped)
        and (descriptive_mode_authorized or not partial_status)
        and not unknown_maturity
        and not explicit_not_scored
    )
    result_status = "scored_complete" if is_complete else "unknown_maturity"
    if protocol_failed:
        result_status = "protocol_invalid"
    elif failed_count or (partial_status and not descriptive_mode_authorized):
        result_status = "partial_cohort"
    elif summary_only:
        result_status = "summary_only"
    elif scout_only and not descriptive_mode_authorized:
        result_status = "scout_or_smoke"
    elif capped and not descriptive_mode_authorized:
        result_status = "capped"
    elif unknown_maturity:
        result_status = "unknown_maturity"
    elif explicit_not_scored:
        result_status = "not_scored_complete"
    elif is_bad_status:
        result_status = "failed_or_unscored"
    elif not has_scores:
        result_status = "unscored_artifact"

    metrics.update(
        {
            "tier": tier,
            "tier_reached": tier,
            "tier_status": tier_status,
            "final_status": summary.get("final_status")
            or (result_status if descriptive_mode_authorized else tier_status),
            "force_all_tiers": bool(summary.get("force_all_tiers", False)),
            "wall_time_s": summary.get("wall_time_s"),
            "result_status": result_status,
            "partial_cohort": bool(failed_count or partial_status),
            "unscored_artifact": bool(not has_scores or is_bad_status or explicit_not_scored),
            "validation_only_result": bool(validation_only_scored_evidence),
        }
    )
    hard_incomplete = bool(
        protocol_failed
        or failed_count
        or (partial_status and not descriptive_mode_authorized)
        or summary_only
        or (scout_only and not descriptive_mode_authorized)
        or (capped and not descriptive_mode_authorized)
        or explicit_not_scored
        or is_bad_status
        or not has_scores
    )
    if is_complete:
        metrics["scored_complete"] = True
        metrics["scored_complete_score"] = 1.0
    elif hard_incomplete:
        metrics["scored_complete"] = False
        metrics["scored_complete_score"] = 0.0
    else:
        # Preserve unknown maturity as unknown. Frontier/Gems apply the task's
        # ratio and stage-label contract after materialization.
        metrics.pop("scored_complete", None)
        metrics.pop("scored_complete_score", None)
    if protocol_failed:
        metrics["protocol_integrity_status"] = (
            _protocol_integrity_failure_status(summary, aggregate) or "failed"
        )
        metrics["suspect_protocol"] = bool(
            summary.get("suspect_protocol")
            or aggregate.get("suspect_protocol")
            or summary.get("suspect_fixed_weight_eval")
            or aggregate.get("suspect_fixed_weight_eval")
            or protocol_failed
        )
        metrics["unscored_artifact"] = True
    explicit_flags = {
        "summary_only": _summary_flag_value(summary, aggregate, "summary_only"),
        "is_summary_only": _summary_flag_value(summary, aggregate, "is_summary_only"),
        "scout_only": _summary_flag_value(summary, aggregate, "scout_only"),
        "is_scout_eval": _summary_flag_value(summary, aggregate, "is_scout_eval"),
        "is_smoke_eval": _summary_flag_value(summary, aggregate, "is_smoke_eval"),
        "partial": _summary_flag_value(summary, aggregate, "partial"),
        "partial_eval": _summary_flag_value(summary, aggregate, "partial_eval"),
        "is_partial_eval": _summary_flag_value(summary, aggregate, "is_partial_eval"),
        "incomplete_eval": _summary_flag_value(summary, aggregate, "incomplete_eval"),
        "is_incomplete_eval": _summary_flag_value(summary, aggregate, "is_incomplete_eval"),
        "capped": _summary_flag_value(summary, aggregate, "capped"),
        "is_capped": _summary_flag_value(summary, aggregate, "is_capped"),
        "result_capped": _summary_flag_value(summary, aggregate, "result_capped"),
        "complete_eval": _summary_flag_value(summary, aggregate, "complete_eval"),
        "is_complete_eval": _summary_flag_value(summary, aggregate, "is_complete_eval"),
    }
    protocol_passed = _protocol_integrity_passed_value(summary, aggregate)
    if protocol_passed is not None:
        explicit_flags["protocol_integrity_passed"] = protocol_passed
    for key, value in explicit_flags.items():
        if value is not None:
            metrics[key] = value
    for key in _DURABLE_ROUTING_MARKER_KEYS:
        value = _durable_routing_marker_value(summary, aggregate, key)
        if value is not None:
            metrics[key] = value
    if summary_only and "summary_only" not in metrics:
        metrics["summary_only"] = True
        metrics["is_summary_only"] = True
    if scout_only and "scout_only" not in metrics:
        metrics["scout_only"] = True
    if capped and "capped" not in metrics:
        metrics["capped"] = True
        metrics["result_capped"] = True

    hard = latest_tier.get("n_hard_constraint_violations")
    if hard is None:
        violations = aggregate.get("hard_constraint_violations")
        if isinstance(violations, list):
            hard = len(violations)
    if hard is None:
        blocking = aggregate.get("promotion_blocking_hard_constraint_violations")
        if isinstance(blocking, list):
            hard = len(blocking)
    try:
        hard_count = max(0, int(hard or 0))
    except (TypeError, ValueError):
        hard_count = 0
    metrics["n_hard_constraint_violations"] = hard_count

    explicit_promotion_eligible = _summary_flag_value(
        summary,
        aggregate,
        "promotion_eligible",
    )
    explicit_clean_promotion_eligible = _summary_flag_value(
        summary,
        aggregate,
        "clean_promotion_eligible",
    )
    if explicit_promotion_eligible is not None:
        # Tier labels are opaque task metadata. Core must not infer promotion
        # eligibility from a tier name; projects/runners that use staged evals
        # should emit these generic flags explicitly.
        metrics["promotion_eligible"] = explicit_promotion_eligible
        if explicit_clean_promotion_eligible is not None:
            metrics["clean_promotion_eligible"] = explicit_clean_promotion_eligible

    canonicalize_evaluation_unit_metadata(metrics, summary, aggregate)

    return metrics


def _json_digest(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def result_summary_control_digest(summary: dict[str, Any]) -> str:
    """Hash legacy result facts without optional config provenance fields."""

    digest_source = copy.deepcopy(summary)
    strip_effective_config_fields(digest_source)
    return _json_digest(digest_source)


def _stable_boundary_finding_payload(value: Any) -> Any:
    """Remove observation and boundary-routing fields from finding identity."""

    if isinstance(value, dict):
        return {
            str(key): _stable_boundary_finding_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _BOUNDARY_FINGERPRINT_IGNORED_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_boundary_finding_payload(item) for item in value]
    if isinstance(value, set):
        return sorted((_stable_boundary_finding_payload(item) for item in value), key=repr)
    return value


def _finding_snapshot_entry(finding: dict[str, Any]) -> tuple[str, str] | None:
    finding_id = str(finding.get("id") or finding.get("finding_id") or "").strip()
    if not finding_id:
        return None
    payload = _stable_boundary_finding_payload(finding)
    if not isinstance(payload, dict):
        return None
    return f"{_FINDING_SNAPSHOT_PREFIX}{finding_id}", _json_digest(payload)


def _finding_snapshot_payload(finding: dict[str, Any]) -> str:
    serialized = json.dumps(finding, sort_keys=True, default=str, separators=(",", ":"))
    return f"{_FINDING_SNAPSHOT_PAYLOAD_PREFIX}{serialized}"


def _finding_digest_from_snapshot_value(value: str) -> str | None:
    if not value.startswith(_FINDING_SNAPSHOT_PAYLOAD_PREFIX):
        return value or None
    try:
        finding = json.loads(value.removeprefix(_FINDING_SNAPSHOT_PAYLOAD_PREFIX))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(finding, dict):
        return None
    entry = _finding_snapshot_entry(finding)
    return entry[1] if entry is not None else None


def canonical_findings_from_snapshot(snapshot: dict[str, str]) -> list[dict[str, Any]]:
    """Recover the transient canonical rows captured before boundary work."""

    findings: list[dict[str, Any]] = []
    for source_ref, value in snapshot.items():
        if not source_ref.startswith(_FINDING_SNAPSHOT_PREFIX) or not value.startswith(
            _FINDING_SNAPSHOT_PAYLOAD_PREFIX
        ):
            continue
        try:
            finding = json.loads(value.removeprefix(_FINDING_SNAPSHOT_PAYLOAD_PREFIX))
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(finding, dict):
            findings.append(finding)
    return findings


def preserve_scoreless_finding_sources(
    snapshot: dict[str, str], *, run_dir: Path, cutoff: datetime
) -> dict[str, str]:
    """Enrich a new cutoff snapshot with complete matching source JSON.

    Args:
        snapshot: Newly captured canonical rows and cutoff source identities.
        run_dir: Run root containing the peer-authored source files.
        cutoff: Publication deadline for the frozen source bytes.

    Returns:
        A snapshot with separate full source payloads when bytes match the
        captured source digest. Canonical comparison fingerprints remain
        unchanged. Unavailable sources keep their canonical row with explicit
        provenance uncertainty. Recovered
        checkpoints must be used directly, without invoking this function.
    """
    from praxist.core.redaction import redact_json
    from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
        read_bounded_file_under_root_no_follow,
    )

    enriched = dict(snapshot)
    root = Path(run_dir)
    for finding in canonical_findings_from_snapshot(snapshot):
        original = dict(finding)
        finding["source_payload_status"] = "canonical_row_only"
        source = str(finding.get("source_filepath") or "").strip()
        if source:
            path = Path(source)
            if not path.is_absolute():
                path = root / path
            try:
                relative = str(path.relative_to(root))
                size = path.lstat().st_size
            except (ValueError, OSError):
                relative = ""
                size = -1
            expected = _stable_content_part(snapshot.get(relative))
            data = (
                read_bounded_file_under_root_no_follow(path, root, max_bytes=size)
                if size >= 0 and expected is not None
                else None
            )
            published_at = (
                _result_publication_mtime(path, run_dir=root) if data is not None else None
            )
            if (
                data is not None
                and expected == f"content:file:{hashlib.sha256(data).hexdigest()}"
                and published_at is not None
                and published_at <= cutoff.timestamp()
                and _logical_result_identity(path, run_dir=root) == snapshot.get(relative)
            ):
                try:
                    payload = json.loads(data)
                except (ValueError, UnicodeError):
                    payload = None
                if isinstance(payload, dict):
                    finding = {**payload, **original}
                    for key in ("content", "title", "summary", "notes"):
                        if key in payload:
                            finding[key] = payload[key]
                    finding["source_payload"] = payload
                    finding["source_payload_sha256"] = hashlib.sha256(data).hexdigest()
                    finding["source_payload_status"] = "frozen"
            if finding.get("source_payload_status") != "frozen":
                finding["source_payload_status"] = "source_unavailable_or_changed"
        entry = _finding_snapshot_entry(finding)
        if entry is not None:
            source_ref = entry[0].replace(
                _FINDING_SNAPSHOT_PREFIX, _SCORELESS_SOURCE_SNAPSHOT_PREFIX, 1
            )
            redacted, _ = redact_json(finding)
            if isinstance(redacted, dict):
                enriched[source_ref] = _finding_snapshot_payload(redacted)
    return enriched


def scoreless_findings_with_frozen_sources(
    snapshot: dict[str, str], findings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Restore complete scoreless evidence without changing cutoff comparisons.

    Args:
        snapshot: Frozen checkpoint including separate full source payloads.
        findings: Findings already accepted by the canonical cutoff filter.

    Returns:
        Accepted findings with their frozen complete source payloads restored.
    """
    restored = []
    for finding in findings:
        finding_id = str(finding.get("id") or finding.get("finding_id") or "").strip()
        serialized = snapshot.get(f"{_SCORELESS_SOURCE_SNAPSHOT_PREFIX}{finding_id}", "")
        if serialized.startswith(_FINDING_SNAPSHOT_PAYLOAD_PREFIX):
            try:
                payload = json.loads(serialized.removeprefix(_FINDING_SNAPSHOT_PAYLOAD_PREFIX))
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict):
                restored.append(payload)
                continue
        restored.append(finding)
    return restored


def compact_boundary_source_snapshot(snapshot: dict[str, str]) -> dict[str, str]:
    """Replace transient canonical payloads with fingerprints for the marker."""

    compacted: dict[str, str] = {}
    for source_ref, value in snapshot.items():
        digest = _finding_digest_from_snapshot_value(value)
        compacted[source_ref] = digest if digest is not None else value
    return compacted


def findings_at_boundary_cutoff(
    snapshot: dict[str, str],
    refreshed_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use pre-cutoff row values while retaining accepted filesystem findings."""

    originals: dict[str, dict[str, Any]] = {}
    for finding in canonical_findings_from_snapshot(snapshot):
        finding_id = str(finding.get("id") or finding.get("finding_id") or "").strip()
        if finding_id:
            originals[finding_id] = finding
    if not originals and snapshot.get(_FINDING_SNAPSHOT_MARKER) != "captured":
        return refreshed_findings
    selected = list(originals.values())
    for finding in refreshed_findings:
        finding_id = str(finding.get("id") or finding.get("finding_id") or "").strip()
        if finding_id in originals:
            continue
        metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
        if metrics.get("late_after_generation_boundary") is not True:
            selected.append(finding)
    return selected


def _scheduler_command_path_values(command: Any) -> list[str]:
    """Return values explicitly declared as result output paths."""

    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return []
    values = list(command)
    if command and Path(command[0]).name in _SHELL_COMMAND_NAMES:
        for raw in command[1:]:
            if not any(char.isspace() for char in raw):
                continue
            with contextlib.suppress(ValueError):
                values.extend(shlex.split(raw))
    path_values: list[str] = []
    index = 0
    while index < len(values):
        raw = values[index]
        value = raw.strip()
        if not value:
            index += 1
            continue
        if "=" in value:
            option, option_value = value.split("=", 1)
            option_name = option.lstrip("-").lower().replace("_", "-")
            if option_name in _SCHEDULER_OUTPUT_OPTION_NAMES and option_value:
                path_values.append(option_value)
            index += 1
            continue
        option_name = value.lstrip("-").lower().replace("_", "-")
        if (
            value.startswith("-")
            and option_name in _SCHEDULER_OUTPUT_OPTION_NAMES
            and index + 1 < len(values)
        ):
            path_values.append(values[index + 1])
            index += 2
            continue
        index += 1
    return path_values


def _scheduler_environment_path_values(environment: dict[str, str]) -> list[str]:
    """Return output paths from explicitly output-named environment entries."""

    values: list[str] = []
    for raw_name, raw_value in environment.items():
        name = str(raw_name).strip().lower().replace("_", "-")
        value = str(raw_value).strip()
        if name in _SCHEDULER_OUTPUT_OPTION_NAMES and value:
            values.append(value)
    return values


def _expand_scheduler_environment(value: str, environment: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group("plain") or match.group("braced") or ""
        return environment.get(key, match.group(0))

    return _SCHEDULER_ENV_PATTERN.sub(replace, value)


def _lexical_absolute_path(value: str, *, cwd: Path | None) -> Path | None:
    token = value.strip().strip("\"'")
    if not token or "\x00" in token or "://" in token:
        return None
    try:
        path = Path(token)
        if not path.is_absolute():
            if cwd is None:
                return None
            path = cwd / path
        return Path(os.path.abspath(path))
    except (OSError, ValueError):
        return None


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _scheduler_result_ownership_candidates(
    run_dir: Path,
) -> list[tuple[int, str, Path]]:
    """Load canonical result-output ownership from central scheduler submissions."""

    events_path = run_dir / "resource_scheduler" / "events.jsonl"
    try:
        event_lines = read_file_bytes(events_path).decode("utf-8").splitlines()
    except (OSError, UnicodeError):
        return []

    run_dir = Path(os.path.abspath(run_dir))
    results_dir = run_dir / "results"
    resolved_results_dir = Path(os.path.realpath(results_dir))
    candidates: set[tuple[int, str, Path]] = set()
    for line in event_lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("event") != "submitted":
            continue
        raw_generation = event.get("generation_id", event.get("generation"))
        if raw_generation is None or isinstance(raw_generation, bool):
            continue
        try:
            generation_id = int(str(raw_generation))
        except (TypeError, ValueError):
            continue
        peer_id = str(event.get("peer_id") or "").strip()
        peer_match = _CANONICAL_PEER_ID_PATTERN.fullmatch(peer_id)
        if peer_match is None or int(peer_match.group("generation")) != generation_id:
            continue
        canonical_peer_id = f"gen{generation_id}_peer{int(peer_match.group('index'))}"
        if peer_id != canonical_peer_id:
            continue

        raw_environment = event.get("environment_values")
        environment = (
            {str(key): str(value) for key, value in raw_environment.items()}
            if isinstance(raw_environment, dict)
            else {}
        )
        raw_cwd = str(event.get("cwd") or "").strip()
        cwd = _lexical_absolute_path(
            _expand_scheduler_environment(raw_cwd, environment),
            cwd=run_dir,
        )
        raw_output_values = [
            *_scheduler_command_path_values(event.get("command")),
            *_scheduler_environment_path_values(environment),
        ]
        if cwd is not None and (
            _path_contains(results_dir, cwd)
            or _path_contains(resolved_results_dir, Path(os.path.realpath(cwd)))
        ):
            raw_output_values.append(str(cwd))
        for raw_value in raw_output_values:
            output_path = _lexical_absolute_path(
                _expand_scheduler_environment(raw_value, environment),
                cwd=cwd,
            )
            if output_path is None:
                continue
            resolved_output_path = Path(os.path.realpath(output_path))
            if not (
                _path_contains(results_dir, output_path)
                or _path_contains(resolved_results_dir, resolved_output_path)
            ):
                continue
            candidates.add((generation_id, canonical_peer_id, resolved_output_path))
    return sorted(candidates, key=lambda item: (item[0], item[1], str(item[2])))


def _scheduler_result_owner(
    summary_path: Path,
    *,
    generation_id: int,
    candidates: list[tuple[int, str, Path]],
) -> tuple[str, bool]:
    """Return unique owner and whether any submitted output path matched."""

    summary_path = Path(os.path.realpath(os.path.abspath(summary_path)))
    matches = [
        (len(output_path.parts), peer_id)
        for event_generation, peer_id, output_path in candidates
        if event_generation == generation_id and _path_contains(output_path, summary_path)
    ]
    if not matches:
        return "", False
    longest_match = max(length for length, _peer_id in matches)
    owners = {peer_id for length, peer_id in matches if length == longest_match}
    return (next(iter(owners)) if len(owners) == 1 else ""), True


def _is_materializer_owned_result_finding(
    *, findings_dir: Path, path: Path, data: dict[str, Any], source_result_path: str
) -> bool:
    source_path = Path(source_result_path)
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    semantics = data.get("artifact_semantics")
    canonical_sources = semantics.get("canonical_sources") if isinstance(semantics, dict) else None
    variant = str(data.get("variant_name") or "").strip()
    run_dir = findings_dir.parent
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_dir.resolve()}::{source_result_path}"))
    return bool(
        not source_path.is_absolute()
        and ".." not in source_path.parts
        and source_path.parts[:1] == ("results",)
        and is_supported_result_summary_filename(source_path.name)
        and variant
        and str(data.get("id") or "").strip() == expected_id
        and path.name == f"{expected_id}_{_slug(variant)}.json"
        and str(data.get("source_result_path") or "").strip() == source_result_path
        and str(metrics.get("source_result_path") or "").strip() == source_result_path
        and metrics.get("auto_materialized_from_result_artifact") is True
        and isinstance(semantics, dict)
        and semantics.get("role") == DERIVED_VIEW
        and semantics.get("status") == COMMITTED
        and semantics.get("stage") == "result_finding_reference"
        and semantics.get("actor") == "research_loop:findings_collection"
        and isinstance(canonical_sources, list)
        and source_result_path in canonical_sources
    )


def _existing_materialized_results(findings_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not findings_dir.exists():
        return results
    for path in findings_dir.glob("*.json"):
        data = _load_json(path)
        if not data:
            continue
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        if not bool(metrics.get("auto_materialized_from_result_artifact")):
            continue
        raw = metrics.get("source_result_path") or data.get("source_result_path")
        if raw:
            source_result_path = str(raw)
            record = {
                "path": path,
                "peer_id": data.get("peer_id"),
                "metrics": dict(metrics),
                "source_result_sha256": metrics.get("source_result_sha256"),
                "source_result_config_sha256": metrics.get("source_result_config_sha256"),
                "source_generation_inference": metrics.get("source_generation_inference"),
                "source_generation_low_confidence": metrics.get("source_generation_low_confidence"),
                "timestamp": data.get("timestamp"),
                "trusted_materializer_record": _is_materializer_owned_result_finding(
                    findings_dir=findings_dir,
                    path=path,
                    data=data,
                    source_result_path=source_result_path,
                ),
            }
            previous = results.get(source_result_path)
            if (
                previous
                and previous.get("trusted_materializer_record") is True
                and not record["trusted_materializer_record"]
            ):
                continue
            results[source_result_path] = record
    return results


def _unlink_existing_materialized_result(
    existing_results: dict[str, dict[str, Any]],
    rel: str,
) -> None:
    existing = existing_results.get(rel)
    existing_path = existing.get("path") if isinstance(existing, dict) else None
    if isinstance(existing_path, Path):
        with contextlib.suppress(OSError):
            existing_path.unlink()


def _remove_existing_materialized_results(findings_dir: Path) -> None:
    for existing in _existing_materialized_results(findings_dir).values():
        existing_path = existing.get("path") if isinstance(existing, dict) else None
        if isinstance(existing_path, Path):
            with contextlib.suppress(OSError):
                existing_path.unlink()


def _is_auto_materialized_result_finding(row: dict[str, Any]) -> bool:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return bool(metrics.get("auto_materialized_from_result_artifact"))


def _is_low_confidence_auto_materialized_result(row: dict[str, Any]) -> bool:
    if not _is_auto_materialized_result_finding(row):
        return False
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    value = metrics.get("source_generation_low_confidence")
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "y"}


def _is_validation_only_auto_materialized_result(row: dict[str, Any]) -> bool:
    if not _is_auto_materialized_result_finding(row):
        return False
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return bool(metrics.get("excluded_from_durable_frontier")) and bool(
        str(metrics.get("exclusion_reason") or "").strip()
    )


def _materialized_finding_visible_at_boundary(finding: dict[str, Any], gen_id: int) -> bool:
    source_gen_id = _extract_generation_id(finding.get("generation_id"))
    if source_gen_id == gen_id:
        return True
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    return bool(metrics.get("late_after_generation_boundary")) and bool(
        _is_validation_only_auto_materialized_result(finding)
    )


def _late_materialization_pending_boundary(
    finding: dict[str, Any],
    *,
    run_dir: Path,
) -> bool:
    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    if metrics.get("late_after_generation_boundary") is not True:
        return False
    observed_gen = _extract_generation_id(metrics.get("late_observed_generation_id"))
    if observed_gen is None:
        return False
    return not (run_dir / f"gen_{observed_gen}" / "generation_boundary.json").exists()


def _auto_materialized_source_path(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return str(metrics.get("source_result_path") or row.get("source_result_path") or "")


def _auto_materialized_source_exists(row: dict[str, Any], run_dir: Path) -> bool:
    source = _auto_materialized_source_path(row)
    if not source:
        return False
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = run_dir / source_path
    return source_path.exists()


def _delete_stale_auto_materialized_rows_from_store(
    live_sources: set[str] | None,
) -> None:
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store,
        )

        stale_ids: list[str] = []
        for row in local_store.get_all_findings():
            if not _is_auto_materialized_result_finding(row):
                continue
            source = _auto_materialized_source_path(row)
            if live_sources is None or source not in live_sources:
                stale_ids.append(str(row.get("id") or ""))
        if stale_ids:
            local_store.delete_findings_by_ids(stale_ids)
    except Exception as exc:  # noqa: BLE001 - stale rows are also filtered locally.
        logger.debug("auto-materialized local-store cleanup failed: %s", exc)


def _extract_generation_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    text = str(value)
    match = re.search(r"(?<![A-Za-z0-9])gen[_-]?(\d+)(?=$|[^A-Za-z0-9])", text)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _infer_result_generation(
    *,
    run_dir: Path,
    summary_path: Path | None = None,
    summary: dict[str, Any],
    variant: str,
    boundary_gen_id: int,
) -> tuple[int, str]:
    aggregate = summary.get("current_aggregate")
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    for source_name, value in (
        ("summary_generation_id", summary.get("generation_id")),
        ("summary_source_generation_id", summary.get("source_generation_id")),
        ("summary_gen_id", summary.get("gen_id")),
        ("aggregate_generation_id", aggregate.get("generation_id")),
        ("aggregate_source_generation_id", aggregate.get("source_generation_id")),
        ("summary_peer_id", summary.get("peer_id")),
        ("aggregate_peer_id", aggregate.get("peer_id")),
    ):
        gen = _extract_generation_id(value)
        if gen is not None:
            return gen, source_name

    if summary_path is not None:
        try:
            rel_path = str(summary_path.relative_to(run_dir))
        except ValueError:
            rel_path = str(summary_path)
        for source_name, value in (
            ("summary_path", rel_path),
            ("summary_parent_path", str(summary_path.parent)),
        ):
            gen = _extract_generation_id(value)
            if gen is not None:
                return gen, source_name

    needle = variant.lower()
    summary_rel = ""
    if summary_path is not None:
        try:
            summary_rel = str(summary_path.relative_to(run_dir)).lower()
        except ValueError:
            summary_rel = ""
    if needle:
        gen_dirs = [
            path
            for path in run_dir.glob("gen_*/shared_findings")
            if _extract_generation_id(path.parent.name) is not None
        ]
        for gen_dir in sorted(
            gen_dirs,
            key=lambda path: int(_extract_generation_id(path.parent.name) or -1),
            reverse=True,
        ):
            gen = _extract_generation_id(gen_dir.parent.name)
            if gen is None:
                continue
            for path in sorted(gen_dir.glob("*.json")):
                payload = _load_json(path)
                if not payload:
                    continue
                metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
                payload_variant = str(payload.get("variant_name") or "").strip().lower()
                payload_source = str(metrics.get("source_result_path") or "").strip().lower()
                explicit_child_refs = metrics.get("child_variants")
                if isinstance(explicit_child_refs, list):
                    child_refs = {str(item).strip().lower() for item in explicit_child_refs}
                else:
                    child_refs = set()
                if (
                    payload_variant == needle
                    or (summary_rel and payload_source == summary_rel)
                    or needle in child_refs
                ):
                    return gen, "generation_local_finding_reference"

        root_findings = run_dir / "shared_findings"
        if root_findings.exists():
            for path in sorted(root_findings.glob("*.json")):
                payload = _load_json(path)
                if not payload:
                    continue
                metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
                if bool(metrics.get("auto_materialized_from_result_artifact")):
                    continue
                payload_variant = str(payload.get("variant_name") or "").strip().lower()
                payload_source = str(metrics.get("source_result_path") or "").strip().lower()
                if payload_variant == needle or (summary_rel and payload_source == summary_rel):
                    gen = _extract_generation_id(payload.get("generation_id"))
                    if gen is not None:
                        return gen, "root_finding_reference"

    return int(boundary_gen_id), "boundary_fallback"


def _timestamp_from_mtime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _result_publication_mtime(path: Path, *, run_dir: Path) -> float | None:
    """Return the publication time of a result's logical file path.

    ``mtime`` captures ordinary writes while ``ctime`` captures atomic file
    replacement. Directory publication is detected by the boundary's path
    snapshot instead of ancestor timestamps, because sibling result writes can
    legitimately change a shared protocol directory.
    """

    try:
        target_stat = path.stat()
    except OSError:
        return None
    published_at = max(target_stat.st_mtime, target_stat.st_ctime)
    try:
        logical_stat = path.lstat()
        published_at = max(published_at, logical_stat.st_mtime, logical_stat.st_ctime)
    except OSError:
        pass
    for parent in path.parents:
        if parent == run_dir:
            break
        try:
            if parent.is_symlink():
                link_stat = parent.lstat()
                published_at = max(published_at, link_stat.st_mtime, link_stat.st_ctime)
        except OSError:
            continue
    return published_at


def _logical_result_identity(path: Path, *, run_dir: Path) -> str | None:
    """Identify content plus same-run replacement of a logical result path."""

    try:
        target = path.stat()
    except OSError:
        return None
    content_identity = _stable_result_content_identity(path)
    if content_identity is None:
        return None
    parts = [f"content:{content_identity}", f"target:{int(target.st_dev)}:{int(target.st_ino)}"]
    for candidate in (path, *path.parents):
        if candidate == run_dir:
            break
        try:
            if candidate.is_symlink():
                link = candidate.lstat()
                parts.append(f"link:{int(link.st_dev)}:{int(link.st_ino)}")
        except OSError:
            return None
    return "|".join(parts)


def _stable_result_content_identity(path: Path) -> str | None:
    """Return relocation-stable content identity for a result file or directory."""

    if path.is_file():
        digest = _file_sha256(path)
        return f"file:{digest}" if digest is not None else None
    if not path.is_dir():
        return None

    digest = hashlib.sha256()
    found = False
    for summary_path in _iter_supported_summaries(path):
        summary_digest = _file_sha256(summary_path)
        if summary_digest is None:
            return None
        try:
            relative = summary_path.relative_to(path)
        except ValueError:
            relative = summary_path
        digest.update(str(relative).encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(summary_digest.encode("ascii"))
        digest.update(b"\0")
        found = True
    return f"directory:{digest.hexdigest()}" if found else "directory:empty"


def _file_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with open_readonly_file(path) as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _iter_supported_summaries(root: Path) -> list[Path]:
    paths: list[Path] = []

    def walk(directory: Path, ancestors: frozenset[tuple[int, int]]) -> None:
        try:
            stat_result = directory.stat()
            directory_id = (int(stat_result.st_dev), int(stat_result.st_ino))
        except OSError:
            return
        if directory_id in ancestors:
            return
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            return
        branch = ancestors | {directory_id}
        for entry in entries:
            candidate = directory / entry.name
            try:
                if entry.is_dir(follow_symlinks=True):
                    walk(candidate, branch)
                elif entry.is_file(follow_symlinks=True) and is_supported_result_summary_filename(
                    candidate.name
                ):
                    paths.append(candidate)
            except OSError:
                continue

    walk(root, frozenset())
    return sorted(paths, key=str)


def _run_root_location_identity(run_dir: Path) -> str | None:
    try:
        stat_result = Path(run_dir).stat()
    except OSError:
        return None
    return f"root:{int(stat_result.st_dev)}:{int(stat_result.st_ino)}"


def _stable_content_part(identity: str | None) -> str | None:
    for part in str(identity or "").split("|"):
        if part.startswith("content:"):
            return part
    return None


def _source_identity_matches_after_relocation(
    expected: str,
    current: str | None,
    *,
    run_dir: Path,
    snapshot: dict[str, str],
) -> bool:
    expected_root = snapshot.get(_RESULT_SNAPSHOT_ROOT_MARKER)
    current_root = _run_root_location_identity(run_dir)
    if expected_root is None or current_root is None or expected_root == current_root:
        return False
    expected_content = _stable_content_part(expected)
    return expected_content is not None and expected_content == _stable_content_part(current)


def result_source_snapshot_at_cutoff(run_dir: Path) -> dict[str, str]:
    """Snapshot logical result-summary paths and stable filesystem identities."""

    root = Path(run_dir)
    snapshot: dict[str, str] = {}
    for path in iter_result_summary_paths(root):
        try:
            source_ref = str(path.relative_to(root))
        except ValueError:
            source_ref = str(path)
        identity = _logical_result_identity(path, run_dir=root)
        if identity is not None:
            snapshot[source_ref] = identity
    return snapshot


def result_source_snapshot_with_cutoff(
    run_dir: Path,
) -> tuple[datetime, dict[str, str]]:
    """Return a cutoff and a reconciled snapshot of sources published before it."""

    root = Path(run_dir)
    snapshot = result_source_snapshot_at_cutoff(root)
    cutoff = datetime.now(UTC)
    return cutoff, reconcile_result_source_snapshot(root, cutoff, snapshot)


def reconcile_result_source_snapshot(
    run_dir: Path,
    cutoff: datetime,
    snapshot: dict[str, str],
) -> dict[str, str]:
    """Reconcile sources observed around an already established cutoff."""

    root = Path(run_dir)
    reconciled = dict(snapshot)
    observed_after_cutoff = result_source_snapshot_at_cutoff(root)
    for source_ref, identity in observed_after_cutoff.items():
        if reconciled.get(source_ref) == identity:
            continue
        source_path = Path(source_ref)
        if not source_path.is_absolute():
            source_path = root / source_path
        published_at = _result_discovery_mtime(source_path, run_dir=root)
        if published_at is not None and published_at <= cutoff.timestamp():
            reconciled[source_ref] = identity
    return reconciled


def _finding_source_index(
    *,
    findings_dir: Path,
    run_dir: Path,
    gen_id: int,
    source_content_hashes: dict[Path, str] | None = None,
) -> dict[str, list[Path]]:
    """Index raw finding files by both authored and canonical ingest ids."""

    from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest import (
        derive_finding_id,
    )

    by_id: dict[str, list[Path]] = {}
    source_dirs = (findings_dir, run_dir / f"gen_{int(gen_id)}" / "shared_findings")
    seen_dirs: set[Path] = set()
    for source_dir in source_dirs:
        if source_dir in seen_dirs or not source_dir.exists():
            continue
        seen_dirs.add(source_dir)
        for path in source_dir.glob("*.json"):
            payload = _load_json(path, source_content_hashes=source_content_hashes)
            if payload is None:
                continue
            source_ids = {
                str(payload.get("id") or "").strip(),
                str(payload.get("finding_id") or "").strip(),
                derive_finding_id(path, payload),
            }
            for source_id in source_ids - {""}:
                by_id.setdefault(source_id, []).append(path)
            by_id.setdefault(f"source_filepath:{path}", []).append(path)
    return by_id


def _finding_source_paths(
    finding: dict[str, Any],
    source_index: dict[str, list[Path]],
) -> list[Path]:
    source_filepath = str(finding.get("source_filepath") or "").strip()
    if source_filepath:
        exact_paths = source_index.get(f"source_filepath:{Path(source_filepath)}", [])
        if exact_paths:
            return sorted(set(exact_paths))
    source_ids = {
        str(finding.get("id") or "").strip(),
        str(finding.get("finding_id") or "").strip(),
        str(finding.get("source_finding_id") or "").strip(),
    }
    paths = {path for source_id in source_ids - {""} for path in source_index.get(source_id, [])}
    return sorted(paths)


def include_finding_sources_in_snapshot(
    snapshot: dict[str, str],
    findings: list[dict[str, Any]],
    *,
    run_dir: Path,
    findings_dir: Path,
    gen_id: int,
    cutoff: datetime,
    canonical_findings: bool = True,
) -> dict[str, str]:
    """Add canonical findings and their pre-cutoff sources to a boundary snapshot."""

    root = Path(run_dir)
    enriched = dict(snapshot)
    root_identity = _run_root_location_identity(root)
    if root_identity is not None:
        enriched[_RESULT_SNAPSHOT_ROOT_MARKER] = root_identity
    source_content_hashes: dict[Path, str] = {}
    source_index = _finding_source_index(
        findings_dir=findings_dir,
        run_dir=root,
        gen_id=gen_id,
        source_content_hashes=source_content_hashes,
    )
    if canonical_findings:
        enriched[_FINDING_SNAPSHOT_MARKER] = "captured"
    for finding in findings:
        if canonical_findings:
            finding_entry = _finding_snapshot_entry(finding)
            if finding_entry is not None:
                finding_ref, _finding_digest = finding_entry
                enriched[finding_ref] = _finding_snapshot_payload(finding)
        metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
        source = str(finding.get("source_result_path") or metrics.get("source_result_path") or "")
        source = source.strip()
        source_paths = [
            (finding_source, False)
            for finding_source in _finding_source_paths(finding, source_index)
        ]
        if source:
            result_source = Path(source)
            if not result_source.is_absolute():
                result_source = root / result_source
            source_paths.append((result_source, True))
        for source_path, include_directory_publication in source_paths:
            published_at = (
                _result_discovery_mtime(source_path, run_dir=root)
                if include_directory_publication
                else _result_publication_mtime(source_path, run_dir=root)
            )
            identity = _logical_result_identity(source_path, run_dir=root)
            if published_at is None or published_at > cutoff.timestamp() or identity is None:
                continue
            # A timestamp can cover multiple writes on a coarse filesystem clock.
            # Bind admission to the exact bytes used to identify this finding.
            indexed_digest = source_content_hashes.get(source_path)
            if (
                indexed_digest is not None
                and _stable_content_part(identity) != f"content:file:{indexed_digest}"
            ):
                continue
            try:
                source_ref = str(source_path.relative_to(root))
            except ValueError:
                source_ref = str(source_path)
            enriched[source_ref] = identity
    return enriched


def _result_discovery_mtime(path: Path, *, run_dir: Path) -> float | None:
    """Include nested directory publication times for snapshot reconciliation."""

    published_at = _result_publication_mtime(path, run_dir=run_dir)
    results_root = run_dir / "results"
    for parent in path.parents:
        if parent in {results_root, run_dir}:
            break
        try:
            parent_stat = parent.lstat()
        except OSError:
            continue
        parent_published_at = max(parent_stat.st_mtime, parent_stat.st_ctime)
        published_at = (
            parent_published_at if published_at is None else max(published_at, parent_published_at)
        )
    return published_at


def finding_source_published_after(
    finding: dict[str, Any],
    *,
    run_dir: Path,
    cutoff: datetime,
    evidence_source_snapshot: dict[str, str] | None = None,
    finding_source_paths: list[Path] | None = None,
) -> bool:
    """Return whether a result-backed finding was published after ``cutoff``."""

    metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
    raw_source = finding.get("source_result_path") or metrics.get("source_result_path")
    source = str(raw_source or "").strip()
    sources: list[Path] = []
    if source:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = Path(run_dir) / source_path
        sources.append(source_path)
    sources.extend(finding_source_paths or [])
    if (
        evidence_source_snapshot is not None
        and evidence_source_snapshot.get(_FINDING_SNAPSHOT_MARKER) == "captured"
    ):
        finding_entry = _finding_snapshot_entry(finding)
        if finding_entry is not None:
            finding_ref, finding_digest = finding_entry
            expected_finding = evidence_source_snapshot.get(finding_ref)
            if expected_finding is None:
                return True
            expected_digest = _finding_digest_from_snapshot_value(expected_finding)
            if expected_digest is None or expected_digest != finding_digest:
                return True
    for source_path in sources:
        if not source_path.exists():
            continue
        try:
            source_ref = str(source_path.relative_to(run_dir))
        except ValueError:
            source_ref = str(source_path)
        published_at = _result_publication_mtime(source_path, run_dir=Path(run_dir))
        if evidence_source_snapshot is not None:
            expected_identity = evidence_source_snapshot.get(source_ref)
            if expected_identity is None:
                return True
            current_identity = _logical_result_identity(source_path, run_dir=Path(run_dir))
            relocated_equivalent = _source_identity_matches_after_relocation(
                expected_identity,
                current_identity,
                run_dir=Path(run_dir),
                snapshot=evidence_source_snapshot,
            )
            if current_identity != expected_identity and not relocated_equivalent:
                return True
            if source_path.is_dir() or relocated_equivalent:
                continue
        if published_at is not None and published_at > cutoff.timestamp():
            return True
    return False


def annotate_late_boundary_findings(
    findings: list[dict[str, Any]],
    *,
    run_dir: Path,
    findings_dir: Path,
    gen_id: int,
    cutoff: datetime,
    evidence_source_snapshot: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Retain arbitrary post-cutoff result findings as durable validation signals."""

    annotated: list[dict[str, Any]] = []
    source_index = _finding_source_index(
        findings_dir=findings_dir,
        run_dir=run_dir,
        gen_id=gen_id,
    )
    for finding in findings:
        metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
        raw_finding_sources = _finding_source_paths(finding, source_index)
        if metrics.get(
            "late_after_generation_boundary"
        ) is True or not finding_source_published_after(
            finding,
            run_dir=run_dir,
            cutoff=cutoff,
            evidence_source_snapshot=evidence_source_snapshot,
            finding_source_paths=raw_finding_sources,
        ):
            annotated.append(finding)
            continue
        updated = dict(finding)
        updated_metrics = dict(metrics)
        source = str(
            updated.get("source_result_path") or updated_metrics.get("source_result_path") or ""
        ).strip()
        source_path = Path(source)
        if source and not source_path.is_absolute():
            source_path = run_dir / source_path
        publication_times = [
            published_at
            for finding_source in raw_finding_sources
            if (published_at := _result_publication_mtime(finding_source, run_dir=run_dir))
            is not None
        ]
        if source:
            source_published_at = _result_publication_mtime(source_path, run_dir=run_dir)
            if source_published_at is not None:
                publication_times.append(source_published_at)
        published_at = max(publication_times, default=None)
        updated_metrics.update(
            {
                "generation_boundary_path": f"gen_{int(gen_id)}/generation_boundary.json",
                "generation_boundary_pending_commit": True,
                "generation_boundary_evidence_cutoff_at": cutoff.isoformat(),
                "late_after_generation_boundary": True,
                "late_observed_generation_id": int(gen_id),
                "artifact_signal_status": "late_after_generation_boundary",
                "validation_only_result": True,
                "promotion_eligible": False,
                "clean_promotion_eligible": False,
                "excluded_from_durable_frontier": True,
                "exclusion_reason": "late_after_generation_boundary",
            }
        )
        if published_at is not None:
            updated_metrics["source_result_mtime"] = _timestamp_from_mtime(published_at)
        updated_metrics.setdefault(
            "recommended_next_step", "review_or_revalidate_late_result_before_promotion"
        )
        updated["metrics"] = updated_metrics
        annotated.append(updated)
    return annotated


def persist_boundary_validation_findings(findings: list[dict[str, Any]]) -> int:
    """Persist only boundary status fields into existing canonical rows."""

    from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
        mark_finding_boundary_validation,
    )

    updated = 0
    for finding in findings:
        metrics = finding.get("metrics") if isinstance(finding.get("metrics"), dict) else {}
        if metrics.get("late_after_generation_boundary") is not True:
            continue
        finding_id = str(finding.get("id") or finding.get("finding_id") or "").strip()
        try:
            updated += int(mark_finding_boundary_validation(finding_id, metrics))
        except Exception as exc:  # noqa: BLE001 - cutoff replay remains authoritative.
            logger.debug("boundary finding state update failed: %s", exc)
    return updated


def _late_generation_boundary_info(
    *,
    run_dir: Path,
    summary_path: Path,
    source_gen_id: int,
    evidence_cutoff: datetime | None = None,
    evidence_source_snapshot: dict[str, str] | None = None,
    current_result_control_digest: str = "",
    prior_result_control_digest: str = "",
) -> dict[str, Any] | None:
    boundary_path = run_dir / f"gen_{source_gen_id}" / "generation_boundary.json"
    summary_mtime = _result_publication_mtime(summary_path, run_dir=run_dir)
    if summary_mtime is None:
        return None

    boundary_exists = boundary_path.exists()
    boundary_mtime: float | None = None
    if boundary_exists:
        try:
            boundary_mtime = boundary_path.stat().st_mtime
        except OSError:
            return None
    elif evidence_cutoff is None:
        return None

    if boundary_mtime is not None:
        comparison_timestamp = boundary_mtime
    elif evidence_cutoff is not None:
        comparison_timestamp = evidence_cutoff.timestamp()
    else:  # The boundary-exists/active-cutoff checks above make this unreachable.
        return None
    boundary_cutoff_at = ""
    boundary_payload = _load_json(boundary_path) if boundary_exists else None
    boundary_source_snapshot: dict[str, str] | None = None
    if boundary_payload is not None:
        boundary_cutoff_at = str(boundary_payload.get("evidence_cutoff_at") or "").strip()
        if boundary_cutoff_at:
            try:
                parsed_cutoff = datetime.fromisoformat(boundary_cutoff_at.replace("Z", "+00:00"))
                if parsed_cutoff.tzinfo is None:
                    parsed_cutoff = parsed_cutoff.replace(tzinfo=UTC)
                comparison_timestamp = parsed_cutoff.timestamp()
            except ValueError:
                boundary_cutoff_at = ""
        raw_source_snapshot = boundary_payload.get("evidence_source_snapshot_at_cutoff")
        if isinstance(raw_source_snapshot, dict):
            boundary_source_snapshot = {
                str(path): str(identity)
                for path, identity in raw_source_snapshot.items()
                if str(path or "").strip() and str(identity or "").strip()
            }
    elif evidence_cutoff is not None:
        boundary_cutoff_at = evidence_cutoff.isoformat()
    try:
        boundary_rel = str(boundary_path.relative_to(run_dir))
    except ValueError:
        boundary_rel = str(boundary_path)
    try:
        summary_rel = str(summary_path.relative_to(run_dir))
    except ValueError:
        summary_rel = str(summary_path)
    source_snapshot = (
        boundary_source_snapshot
        if boundary_source_snapshot is not None
        else evidence_source_snapshot
    )
    if source_snapshot is None:
        published_after_cutoff = summary_mtime > comparison_timestamp
    else:
        expected_identity = source_snapshot.get(summary_rel)
        current_identity = _logical_result_identity(summary_path, run_dir=run_dir)
        relocated_equivalent = (
            expected_identity is not None
            and _source_identity_matches_after_relocation(
                expected_identity,
                current_identity,
                run_dir=run_dir,
                snapshot=source_snapshot,
            )
        )
        published_after_cutoff = (
            expected_identity is None
            or (current_identity != expected_identity and not relocated_equivalent)
            or (summary_mtime > comparison_timestamp and not relocated_equivalent)
        )
    if (
        published_after_cutoff
        and current_result_control_digest
        and (current_result_control_digest == prior_result_control_digest)
    ):
        published_after_cutoff = False
    if not published_after_cutoff:
        return None
    info: dict[str, Any] = {
        "generation_boundary_path": boundary_rel,
        "source_result_mtime": _timestamp_from_mtime(summary_mtime),
    }
    if boundary_mtime is not None:
        info["generation_boundary_mtime"] = _timestamp_from_mtime(boundary_mtime)
    else:
        info["generation_boundary_pending_commit"] = True
    if boundary_cutoff_at:
        info["generation_boundary_evidence_cutoff_at"] = boundary_cutoff_at
    return info


def _preserved_late_boundary_info(
    existing: dict[str, Any] | None,
    *,
    run_dir: Path,
    source_gen_id: int,
) -> dict[str, Any] | None:
    """Keep an observed late result validation-only across boundary retries."""

    if not isinstance(existing, dict):
        return None
    metrics = existing.get("metrics") if isinstance(existing.get("metrics"), dict) else {}
    if metrics.get("late_after_generation_boundary") is not True:
        return None
    existing_gen_id = _extract_generation_id(metrics.get("source_generation_id"))
    if existing_gen_id is None:
        existing_gen_id = _extract_generation_id(existing.get("generation_id"))
    if existing_gen_id != int(source_gen_id):
        return None
    expected_boundary_path = f"gen_{int(source_gen_id)}/generation_boundary.json"
    observed_boundary_path = str(metrics.get("generation_boundary_path") or "").strip()
    if observed_boundary_path and observed_boundary_path != expected_boundary_path:
        return None
    boundary_exists = (run_dir / f"gen_{source_gen_id}" / "generation_boundary.json").exists()
    if not boundary_exists and metrics.get("generation_boundary_pending_commit") is not True:
        return None
    if not boundary_exists:
        from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
            read_boundary_evidence_checkpoint,
        )

        if read_boundary_evidence_checkpoint(run_dir, source_gen_id) is None:
            return None
    keys = (
        "generation_boundary_path",
        "generation_boundary_evidence_cutoff_at",
        "source_result_mtime",
        "generation_boundary_pending_commit",
    )
    info = {key: metrics[key] for key in keys if metrics.get(key) not in (None, "")}
    if boundary_exists:
        info.pop("generation_boundary_pending_commit", None)
    return info or None


def _generation_source_rank(source: Any) -> int:
    source_text = str(source or "")
    if source_text == "generation_local_finding_reference":
        return 3
    if source_text in {
        "root_finding_reference",
        "summary_generation_id",
        "summary_source_generation_id",
        "summary_gen_id",
        "aggregate_generation_id",
        "aggregate_source_generation_id",
        "summary_peer_id",
        "aggregate_peer_id",
    }:
        return 2
    if source_text == "boundary_fallback":
        return 1
    return 0


def _materialize_result_artifacts(
    *,
    run_dir: Path,
    gen_id: int,
    default_lane: str = "performance",
    default_family: str = "task_candidate",
    cell_metric_derivations: list[dict[str, Any]] | None = None,
    metric_aliases: dict[str, str] | None = None,
    scoring_metric_keys: list[str] | tuple[str, ...] | None = None,
    result_maturity_policy: dict[str, Any] | None = None,
    evidence_cutoff: datetime | None = None,
    evidence_source_snapshot: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Serialize result-summary materialization within one orchestrator process."""

    with _RESULT_MATERIALIZATION_LOCK:
        return _materialize_result_artifacts_locked(
            run_dir=run_dir,
            gen_id=gen_id,
            default_lane=default_lane,
            default_family=default_family,
            cell_metric_derivations=cell_metric_derivations,
            metric_aliases=metric_aliases,
            scoring_metric_keys=scoring_metric_keys,
            result_maturity_policy=result_maturity_policy,
            evidence_cutoff=evidence_cutoff,
            evidence_source_snapshot=evidence_source_snapshot,
        )


def _materialize_result_artifacts_locked(
    *,
    run_dir: Path,
    gen_id: int,
    default_lane: str = "performance",
    default_family: str = "task_candidate",
    cell_metric_derivations: list[dict[str, Any]] | None = None,
    metric_aliases: dict[str, str] | None = None,
    scoring_metric_keys: list[str] | tuple[str, ...] | None = None,
    result_maturity_policy: dict[str, Any] | None = None,
    evidence_cutoff: datetime | None = None,
    evidence_source_snapshot: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Create canonical result findings from task result summaries.

    Task evaluators may organize summaries under one or more protocol folders,
    while frontier promotion uses canonical findings.  This bridge
    keeps those surfaces synchronized and gives sweep children independent
    ``variant_name`` identities based on their result directory names.
    """

    results_dir = run_dir / "results"
    findings_dir = run_dir / "shared_findings"
    if not results_dir.exists():
        _remove_existing_materialized_results(findings_dir)
        return []
    findings_dir.mkdir(parents=True, exist_ok=True)
    existing_results = _existing_materialized_results(findings_dir)
    scheduler_ownership = _scheduler_result_ownership_candidates(run_dir)
    config_digest = _json_digest(
        {
            "cell_metric_derivations": cell_metric_derivations or [],
            "metric_aliases": metric_aliases or {},
            "scoring_metric_keys": list(scoring_metric_keys or []),
            "maturity_policy": result_maturity_policy or {},
        }
    )
    touched: list[dict[str, Any]] = []
    seen_summary_rels: set[str] = set()
    accepted_summary_count = 0
    summary_paths = iter_result_summary_paths(run_dir)
    if not summary_paths and any(path.is_file() for path in results_dir.rglob("*")):
        logger.warning(
            "Result artifacts exist under %s, but none use a supported summary filename: %s",
            results_dir,
            ", ".join(_RESULT_SUMMARY_NAMES),
        )
    for summary_path in summary_paths:
        rel = str(summary_path.relative_to(run_dir))
        seen_summary_rels.add(rel)
        summary = _load_json(summary_path)
        if summary is None:
            continue
        summary = normalized_result_summary(
            summary,
            summary_path=summary_path,
            maturity_policy=result_maturity_policy,
        )
        digest = result_summary_control_digest(summary)
        legacy_digest = _json_digest(summary)
        existing = existing_results.get(rel)
        preserved_timestamp = ""
        if not (
            isinstance(summary.get("current_aggregate"), dict)
            and (summary.get("variant_name") or summary_path.parent.name)
        ):
            _unlink_existing_materialized_result(existing_results, rel)
            continue
        variant = result_summary_variant_name(summary_path, summary, run_dir).strip()
        source_gen_id, gen_source = _infer_result_generation(
            run_dir=run_dir,
            summary_path=summary_path,
            summary=summary,
            variant=variant,
            boundary_gen_id=gen_id,
        )
        aggregate = summary.get("current_aggregate")
        aggregate = aggregate if isinstance(aggregate, dict) else {}
        scheduler_peer_id, scheduler_path_matched = _scheduler_result_owner(
            summary_path,
            generation_id=source_gen_id,
            candidates=scheduler_ownership,
        )
        reported_peer_id = str(summary.get("peer_id") or aggregate.get("peer_id") or "").strip()
        if scheduler_peer_id:
            source_peer_id = scheduler_peer_id
        elif scheduler_path_matched:
            # Conflicting scheduler claims are not resolved by trusting the
            # task-authored summary. Keep the result visible under the
            # generation-scoped synthetic producer until ownership is unique.
            source_peer_id = f"gen{source_gen_id}_unknown_peer"
        else:
            source_peer_id = reported_peer_id or f"gen{source_gen_id}_result_artifact"
        late_boundary_info = _late_generation_boundary_info(
            run_dir=run_dir,
            summary_path=summary_path,
            source_gen_id=source_gen_id,
            evidence_cutoff=(evidence_cutoff if source_gen_id == gen_id else None),
            evidence_source_snapshot=(
                evidence_source_snapshot if source_gen_id == gen_id else None
            ),
            current_result_control_digest=digest,
            prior_result_control_digest=str(
                existing.get("source_result_sha256")
                if isinstance(existing, dict)
                and existing.get("trusted_materializer_record") is True
                else ""
            ),
        )
        if late_boundary_info is None:
            late_boundary_info = _preserved_late_boundary_info(
                existing,
                run_dir=run_dir,
                source_gen_id=source_gen_id,
            )
        metrics = _result_summary_metrics(
            summary,
            cell_metric_derivations=cell_metric_derivations,
            metric_aliases=metric_aliases,
            scoring_metric_keys=scoring_metric_keys,
            maturity_policy=result_maturity_policy,
        )
        effective_config_metadata = result_effective_config_metadata(summary)
        metrics.update(effective_config_metadata)
        producer_identity = _result_summary_producer_identity(summary)
        if producer_identity:
            metrics["canonical_variant_name"] = producer_identity
        reported_variant = str(summary.get("variant_name") or "").strip()
        if reported_variant and reported_variant != variant:
            metrics["reported_variant_name"] = reported_variant
        protocol_invalid = (
            str(metrics.get("result_status") or "").strip().lower() == "protocol_invalid"
        )
        explicit_status_text = _status_text(
            summary.get("result_status"),
            summary.get("final_status"),
            summary.get("tier_status"),
            aggregate.get("result_status"),
            aggregate.get("final_status"),
            aggregate.get("tier_status"),
        )
        explicit_retainable_partial = _status_has_any(
            explicit_status_text,
            "partial",
            "partial_cohort",
            "partial_eval",
        )
        cells = summary.get("evaluation_unit_records")
        if not isinstance(cells, list):
            cells = summary.get("all_eval_units")
        if not isinstance(cells, list):
            cells = summary.get("all_eval_cells")
        if not isinstance(cells, list):
            cells = summary.get("all_paired_cells")
        cells = cells if isinstance(cells, list) else []
        has_materializable_score = _has_scored_metrics(
            metrics, scoring_metric_keys
        ) or _has_scored_cells(cells, scoring_metric_keys)
        structured_failed_cell_details = _has_explicit_failed_cell_details(summary, aggregate)
        explicit_negative_signal = bool(
            _truthy(metrics.get("is_negative"))
            or str(metrics.get("evidence_valence") or "").strip().lower() in {"negative", "mixed"}
            or str(metrics.get("failure_mode") or "").strip()
        )
        actionable_diagnostic = bool(
            str(metrics.get("failure_mode") or "").strip()
            or str(metrics.get("diagnostic_role") or "").strip()
            or str(metrics.get("next_step_intent") or "").strip()
        )
        negative_diagnostic_signal = bool(explicit_negative_signal and actionable_diagnostic)
        retainable_signal_only = bool(
            (not has_materializable_score and structured_failed_cell_details)
            or (
                negative_diagnostic_signal
                and (
                    not has_materializable_score
                    or _is_bad_result_status(explicit_status_text, allow_partial=True)
                    or str(metrics.get("result_status") or "").strip().lower()
                    == "failed_or_unscored"
                    or metrics.get("scored_complete") is False
                )
            )
        )
        expected_validation_only = bool(
            _truthy(metrics.get("validation_only_result")) or retainable_signal_only
        )
        normalized_partial = (
            str(metrics.get("result_status") or "").strip().lower() == "partial_cohort"
        )
        if (
            structured_failed_cell_details
            and not explicit_retainable_partial
            and not (normalized_partial and has_materializable_score)
            and not (protocol_invalid and has_materializable_score)
            and not retainable_signal_only
        ):
            _unlink_existing_materialized_result(existing_results, rel)
            continue
        preliminary_status = str(metrics.get("result_status") or "").strip().lower()
        preliminary_incomplete = (
            bool(
                metrics.get("scored_complete") is False
                or preliminary_status
                in {
                    "partial_cohort",
                    "scout_or_smoke",
                    "capped",
                    "not_scored_complete",
                    "failed_or_unscored",
                    "unscored_artifact",
                    "summary_only",
                }
            )
            and not protocol_invalid
        )
        retainable_preliminary = preliminary_status in {
            "partial_cohort",
            "scout_or_smoke",
            "capped",
            "unknown_maturity",
            "not_scored_complete",
        } or (
            preliminary_status == "failed_or_unscored"
            and has_materializable_score
            and (
                _status_has_any(explicit_status_text, "timeout", "timed_out")
                or (
                    _status_has_any(explicit_status_text, "incomplete")
                    and not _status_has_any(
                        explicit_status_text,
                        "running",
                        "pending",
                        "stale",
                        "crash",
                        "crashed",
                        "error",
                        "cancel",
                        "cancelled",
                        "canceled",
                        "interrupted",
                        "killed",
                        "oom",
                    )
                )
            )
        )
        if preliminary_incomplete and (
            (not has_materializable_score and not retainable_signal_only)
            or (not retainable_preliminary and not retainable_signal_only)
        ):
            _unlink_existing_materialized_result(existing_results, rel)
            continue
        accepted_summary_count += 1
        existing_metrics = (
            existing.get("metrics")
            if isinstance(existing, dict) and isinstance(existing.get("metrics"), dict)
            else {}
        )
        if (
            existing
            and existing.get("source_result_sha256") in {digest, legacy_digest}
            and existing.get("source_result_config_sha256") == config_digest
        ):
            should_refresh_control_digest = bool(existing.get("source_result_sha256") != digest)
            existing_inference = str(existing.get("source_generation_inference") or "")
            existing_low_confidence = bool(existing.get("source_generation_low_confidence"))
            should_refresh_provenance = (
                gen_source != "boundary_fallback"
                and existing_inference != gen_source
                and (
                    existing_low_confidence
                    or _generation_source_rank(gen_source)
                    > _generation_source_rank(existing_inference)
                )
            )
            should_refresh_protocol = bool(
                protocol_invalid
                and (
                    str(existing_metrics.get("result_status") or "").strip().lower()
                    != "protocol_invalid"
                    or existing_metrics.get("exclusion_reason") != "protocol_integrity_failed"
                    or existing_metrics.get("excluded_from_durable_frontier") is not True
                )
            )
            existing_exclusion_reason = str(existing_metrics.get("exclusion_reason") or "").strip()
            existing_preliminary = bool(
                existing_exclusion_reason == "preliminary_or_incomplete_evidence"
                or (
                    not existing_exclusion_reason
                    and existing_metrics.get("excluded_from_durable_frontier") is True
                    and existing_metrics.get("scored_complete") is False
                )
            )
            should_refresh_preliminary = bool(
                (
                    preliminary_incomplete
                    and (
                        existing_metrics.get("exclusion_reason")
                        != "preliminary_or_incomplete_evidence"
                        or existing_metrics.get("excluded_from_durable_frontier") is not True
                        or bool(existing_metrics.get("validation_only_result"))
                        != expected_validation_only
                    )
                )
                or (not preliminary_incomplete and existing_preliminary)
            )
            should_refresh_late_boundary = bool(
                (
                    late_boundary_info
                    and existing_metrics.get("late_after_generation_boundary") is not True
                )
                or (
                    not late_boundary_info
                    and existing_metrics.get("late_after_generation_boundary") is True
                )
            )
            should_refresh_late_observation = bool(
                late_boundary_info
                and _extract_generation_id(existing_metrics.get("late_observed_generation_id"))
                is None
            )
            should_refresh_late_metadata = bool(
                late_boundary_info
                and any(
                    existing_metrics.get(key) != value for key, value in late_boundary_info.items()
                )
            )
            should_refresh_late_pending_state = bool(
                late_boundary_info
                and bool(existing_metrics.get("generation_boundary_pending_commit"))
                != bool(late_boundary_info.get("generation_boundary_pending_commit"))
            )
            should_refresh_identity = bool(
                producer_identity
                and existing_metrics.get("canonical_variant_name") != producer_identity
            )
            should_refresh_peer_ownership = (
                str(existing.get("peer_id") or "").strip() != source_peer_id
            )
            should_refresh_maturity = any(
                _as_float(existing_metrics.get(key)) != _as_float(metrics.get(key))
                for key in ("effort_ratio", "coverage_ratio")
            )
            expected_dimensions = metrics.get("design_dimensions")
            should_refresh_dimensions = bool(
                isinstance(expected_dimensions, dict)
                and expected_dimensions
                and existing_metrics.get("design_dimensions") != expected_dimensions
            )
            existing_config_metadata = {
                key: existing_metrics[key]
                for key in EFFECTIVE_CONFIG_METADATA_KEYS
                if key in existing_metrics
            }
            should_refresh_effective_config = existing_config_metadata != effective_config_metadata
            if (should_refresh_effective_config or should_refresh_control_digest) and not any(
                (
                    should_refresh_provenance,
                    should_refresh_protocol,
                    should_refresh_preliminary,
                    should_refresh_late_boundary,
                    should_refresh_late_observation,
                    should_refresh_late_metadata,
                    should_refresh_late_pending_state,
                    should_refresh_identity,
                    should_refresh_peer_ownership,
                    should_refresh_maturity,
                    should_refresh_dimensions,
                )
            ):
                preserved_timestamp = str(existing.get("timestamp") or "").strip()
            if (
                not should_refresh_provenance
                and not should_refresh_protocol
                and not should_refresh_preliminary
                and not should_refresh_late_boundary
                and not should_refresh_late_observation
                and not should_refresh_late_metadata
                and not should_refresh_late_pending_state
                and not should_refresh_identity
                and not should_refresh_peer_ownership
                and not should_refresh_maturity
                and not should_refresh_dimensions
                and not should_refresh_effective_config
                and not should_refresh_control_digest
            ):
                existing_path = existing.get("path")
                if isinstance(existing_path, Path):
                    existing_finding = _load_json(existing_path)
                    if existing_finding and _late_materialization_pending_boundary(
                        existing_finding,
                        run_dir=run_dir,
                    ):
                        touched.append(existing_finding)
                continue
        family = _infer_strategy_family(variant, summary)
        if family in {"learned_candidate", "task_candidate"} and default_family:
            family = str(default_family)
        metrics["strategy_family"] = family
        if not str(metrics.get("frontier_lane") or "").strip():
            metrics["frontier_lane"] = (
                str(default_lane)
                if family in {"learned_candidate", str(default_family)}
                else family
            )
        metrics["source_result_path"] = rel
        metrics["source_result_kind"] = summary_path.name
        metrics["source_result_sha256"] = digest
        metrics["source_result_config_sha256"] = config_digest
        metrics["auto_materialized_from_result_artifact"] = True
        metrics["source_generation_id"] = source_gen_id
        metrics["source_generation_inference"] = gen_source
        if gen_source == "boundary_fallback":
            metrics["source_generation_low_confidence"] = True
            metrics["provenance_warning"] = "source_generation_boundary_fallback"
            metrics["promotion_eligible"] = False
            metrics["clean_promotion_eligible"] = False
            metrics["excluded_from_durable_frontier"] = True
            metrics["exclusion_reason"] = "source_generation_low_confidence"
            metrics["recommended_next_step"] = "rerun_or_relink_result_with_explicit_generation_id"
        if protocol_invalid:
            metrics["promotion_eligible"] = False
            metrics["clean_promotion_eligible"] = False
            metrics["excluded_from_durable_frontier"] = True
            metrics["exclusion_reason"] = "protocol_integrity_failed"
            metrics["recommended_next_step"] = "rerun_with_valid_evaluator_protocol"
        elif preliminary_incomplete:
            metrics["promotion_eligible"] = False
            metrics["clean_promotion_eligible"] = False
            metrics["excluded_from_durable_frontier"] = True
            metrics["exclusion_reason"] = "preliminary_or_incomplete_evidence"
            metrics["validation_only_result"] = expected_validation_only
            metrics["recommended_next_step"] = (
                str(metrics.get("next_step_intent") or "").strip()
                if retainable_signal_only
                else "complete_standard_evaluation_protocol"
            )
        if late_boundary_info:
            metrics.update(late_boundary_info)
            metrics["late_after_generation_boundary"] = True
            existing_metrics: dict[str, Any] = {}
            if isinstance(existing, dict):
                raw_existing_metrics = existing.get("metrics")
                if isinstance(raw_existing_metrics, dict):
                    existing_metrics = raw_existing_metrics
            observed_gen_id = _extract_generation_id(
                existing_metrics.get("late_observed_generation_id")
            )
            metrics["late_observed_generation_id"] = int(
                gen_id if observed_gen_id is None else observed_gen_id
            )
            metrics["artifact_signal_status"] = "late_after_generation_boundary"
            metrics["promotion_eligible"] = False
            metrics["clean_promotion_eligible"] = False
            metrics["excluded_from_durable_frontier"] = True
            if not str(metrics.get("exclusion_reason") or "").strip():
                metrics["exclusion_reason"] = "late_after_generation_boundary"
            if not str(metrics.get("recommended_next_step") or "").strip():
                metrics["recommended_next_step"] = (
                    "review_or_revalidate_late_result_before_promotion"
                )
        dig_provenance = _dig_contract_provenance(
            run_dir=run_dir,
            variant=variant,
            source_gen_id=source_gen_id,
            metrics=metrics,
        )
        if dig_provenance:
            metrics["dig_provenance"] = dig_provenance
            metrics["dig_selected_contract_path"] = dig_provenance.get("selected_contract_path")
            metrics["dig_expected_vs_actual_alignment"] = dig_provenance.get(
                "expected_vs_actual_alignment"
            )
        for key in list(metrics):
            if metrics.get(key) in (None, "", [], {}):
                metrics.pop(key, None)

        finding_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_dir.resolve()}::{rel}"))
        late_after_boundary = bool(metrics.get("late_after_generation_boundary"))
        title = (
            f"{variant} late result validation signal"
            if late_after_boundary
            else f"{variant} structured result validation signal"
            if retainable_signal_only
            else f"{variant} canonical result artifact"
        )
        if late_after_boundary:
            content_parts = [
                "Auto-materialized from a results artifact written after its generation boundary; "
                "retained as a validation signal for later agents and diagnostics, "
                "but excluded from durable frontier/Gems promotion until revalidated.",
            ]
        elif retainable_signal_only:
            content_parts = [
                "Auto-materialized structured negative evidence for validation, "
                "memory, and diagnostics; excluded from durable frontier/Gems "
                "promotion until independently revalidated.",
            ]
        else:
            content_parts = [
                "Auto-materialized from results artifact so leaderboard-visible "
                "scored, preliminary, or protocol-invalid variants are also visible to frontier promotion, "
                "Gems, and the finding graph.",
            ]
        if metrics.get("tier"):
            content_parts.append(f"tier={metrics.get('tier')}.")
        if metrics.get("tier_status"):
            content_parts.append(f"status={metrics.get('tier_status')}.")
        primary_score = _first_scoring_metric_value(metrics, scoring_metric_keys)
        if primary_score is not None:
            content_parts.append(f"primary_score={primary_score}.")
        if metrics.get("n_hard_constraint_violations") is not None:
            content_parts.append(f"hard_violations={metrics.get('n_hard_constraint_violations')}.")
        if metrics.get("failure_mode"):
            content_parts.append(f"failure_mode={metrics.get('failure_mode')}.")
        if metrics.get("next_step_intent"):
            content_parts.append(f"next_step={metrics.get('next_step_intent')}.")
        if structured_failed_cell_details:
            content_parts.append(
                "Structured failure details remain available in the source result."
            )
        content = " ".join(content_parts)
        finding = {
            "id": finding_id,
            "finding_type": "result",
            "title": title,
            "content": content,
            "summary": content,
            "metrics": metrics,
            "variant_name": variant,
            "dig_provenance": dig_provenance or {},
            "notes": (
                "Late result artifact retained as a validation signal; one independent finding per result directory."
                if late_after_boundary
                else "Structured negative result retained as a validation signal; one independent finding per result directory."
                if retainable_signal_only
                else "Canonicalized result artifact; one independent finding per result directory."
            ),
            "peer_id": source_peer_id,
            "generation_id": int(source_gen_id),
            "timestamp": preserved_timestamp or datetime.now(UTC).isoformat(),
            "source_result_path": rel,
        }
        for key in (*_RESULT_PRODUCER_IDENTITY_KEYS, "canonical_variant_id"):
            value = summary.get(key)
            if value not in (None, "") and not isinstance(value, (bool, dict, list)):
                finding[key] = value
        for container_name in ("details", "extra", "current_aggregate"):
            compact_identity = _compact_summary_result_identity(summary.get(container_name))
            if compact_identity:
                finding[container_name] = compact_identity
        finding = attach_artifact_semantics(
            finding,
            role=DERIVED_VIEW,
            stage="result_finding_reference",
            generation_id=source_gen_id,
            actor="research_loop:findings_collection",
            derived_from=[rel],
            canonical_sources=[rel],
            runtime_fact_source=False,
            notes=(
                "Rebuildable finding-graph reference. Metric facts remain owned "
                "by the source result summary."
            ),
        )
        out = findings_dir / f"{finding_id}_{_slug(variant)}.json"
        _atomic_write_json(out, finding)
        touched.append(finding)
        existing_path = existing.get("path") if existing else None
        if isinstance(existing_path, Path) and existing_path != out:
            with contextlib.suppress(OSError):
                existing_path.unlink()
    for rel in set(existing_results) - seen_summary_rels:
        _unlink_existing_materialized_result(existing_results, rel)
    if touched:
        logger.info(
            "findings_collection: materialized %d result artifact finding(s) from %s",
            len(touched),
            results_dir,
        )
    elif summary_paths and accepted_summary_count == 0:
        logger.warning(
            "findings_collection: discovered %d supported result summary file(s) under %s "
            "but materialized no canonical findings; verify score, completion, and result-shape fields",
            len(summary_paths),
            results_dir,
        )
    return touched


def _materialize_late_generation_signals(*, run_dir: Path, gen_id: int) -> list[dict[str, Any]]:
    """Bridge protected late-job records into the canonical finding stream."""
    path = run_dir / f"gen_{int(gen_id)}" / "generation_results.json"
    try:
        rows = json.loads(read_file_bytes(path))
    except (OSError, UnicodeError, json.JSONDecodeError):
        rows = None
    if not isinstance(rows, list):
        return []
    findings_dir = run_dir / "shared_findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    touched: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("status") or "").startswith("late_"):
            continue
        peer_id = str(row.get("peer_id") or f"gen{gen_id}_late_signal")
        finding_id = (
            "late_signal_"
            + hashlib.sha256(
                json.dumps(row, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:16]
        )
        finding = {
            "id": finding_id,
            "finding_type": "late_result_signal",
            "title": "Late protected job signal",
            "content": str(row.get("message") or "Protected work remained active at boundary."),
            "summary": str(row.get("message") or "Protected work remained active at boundary."),
            "metrics": {
                **{key: value for key, value in row.items() if key not in {"message", "peer_id"}},
                "validation_only_result": True,
                "late_after_generation_boundary": True,
                "artifact_signal_status": "late_after_generation_boundary",
                "promotion_eligible": False,
            },
            "variant_name": peer_id,
            "peer_id": peer_id,
            "generation_id": int(row.get("generation_id", gen_id) or gen_id),
            "source_generation_results": str(path.relative_to(run_dir)),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        out = findings_dir / f"{finding_id}.json"
        if _load_json(out) is not None:
            continue
        _atomic_write_json(out, finding)
        touched.append(finding)
    return touched


def collect_findings_for_generation(
    *,
    findings_dir: Path,
    gen_id: int,
    local_mode: bool,
    do_ingest: bool = True,
    primary_metric: str | None = None,
    materialize_result_artifacts: bool = True,
    result_artifact_default_lane: str = "performance",
    result_artifact_default_family: str = "task_candidate",
    result_cell_metric_derivations: list[dict[str, Any]] | None = None,
    result_metric_aliases: dict[str, str] | None = None,
    result_scoring_metric_keys: list[str] | tuple[str, ...] | None = None,
    result_maturity_policy: dict[str, Any] | None = None,
    evidence_cutoff: datetime | None = None,
    evidence_source_snapshot: dict[str, str] | None = None,
    task_spec: Any | None = None,
) -> list[dict[str, Any]]:
    """Collect findings from the local store with a filesystem fallback.

    The research loop treats SQLite as the canonical local source when local
    mode is active, but result preservation wins over store availability. If
    ingestion or SQLite reads fail, the filesystem fallback still returns
    generation-scoped finding JSON files.

    #150: ``primary_metric`` (from ``task_spec.evaluation.primary_metric``)
    is forwarded into ``ingest_findings_directory`` so filesystem-written
    findings get the primary value hoisted into canonical
    ``metrics[primary_metric]`` shape during ingest. Without it, frontier
    promotion silently drops them and ``variants_total`` stays at 0.
    Explicit scoreless tasks disable result materialization and primary-metric
    hoisting while retaining peer-authored findings of every supported type.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.scoreless import is_scoreless

    if is_scoreless(task_spec):
        materialize_result_artifacts = False
        primary_metric = None
        result_maturity_policy = None
    run_dir = _run_dir_from_findings_dir(findings_dir)
    materialized_findings: list[dict[str, Any]] = []
    if local_mode and do_ingest:
        _sync_generation_local_findings(
            run_dir=run_dir,
            gen_id=gen_id,
            primary_metric=primary_metric,
            result_maturity_policy=result_maturity_policy,
        )
    if materialize_result_artifacts:
        with _RESULT_MATERIALIZATION_LOCK:
            materialized_findings = _materialize_result_artifacts(
                run_dir=run_dir,
                gen_id=gen_id,
                default_lane=result_artifact_default_lane,
                default_family=result_artifact_default_family,
                cell_metric_derivations=result_cell_metric_derivations,
                metric_aliases=result_metric_aliases,
                scoring_metric_keys=result_scoring_metric_keys,
                result_maturity_policy=result_maturity_policy,
                evidence_cutoff=evidence_cutoff,
                evidence_source_snapshot=evidence_source_snapshot,
            )
            materialized_findings.extend(
                _materialize_late_generation_signals(run_dir=run_dir, gen_id=gen_id)
            )
    else:
        # Task-local configurations can opt out of automatic result-artifact
        # bridging. In that mode stale auto-materialized files must not continue
        # to feed SQLite/frontier/Gems through either ingest or filesystem scans.
        _remove_existing_materialized_results(findings_dir)
    if local_mode and do_ingest:
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest import (
                ingest_findings_directory,
            )

            ingest_findings_directory(
                findings_dir,
                primary_metric=primary_metric,
                result_maturity_policy=result_maturity_policy,
            )
        except Exception as e:  # noqa: BLE001 - fallback keeps findings usable.
            logger.debug("findings_ingest failed: %s", e)

        if materialize_result_artifacts:
            _delete_stale_auto_materialized_rows_from_store(
                set(_existing_materialized_results(findings_dir))
            )
        else:
            _delete_stale_auto_materialized_rows_from_store(None)

        try:
            from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
                get_findings,
            )

            rows = get_findings(generation_id=gen_id)
            if not materialize_result_artifacts:
                rows = [row for row in rows if not _is_auto_materialized_result_finding(row)]
            else:
                live_materialized_sources = set(_existing_materialized_results(findings_dir))
                if live_materialized_sources:
                    rows = [
                        row
                        for row in rows
                        if not _is_auto_materialized_result_finding(row)
                        or (
                            _auto_materialized_source_path(row) in live_materialized_sources
                            and (
                                not _is_low_confidence_auto_materialized_result(row)
                                or _is_validation_only_auto_materialized_result(row)
                            )
                        )
                    ]
                else:
                    rows = [row for row in rows if not _is_auto_materialized_result_finding(row)]
            seen_ids = {str(row.get("id") or "") for row in rows}
            for finding in materialized_findings:
                finding_id = str(finding.get("id") or "")
                if not _materialized_finding_visible_at_boundary(finding, gen_id):
                    continue
                if _is_low_confidence_auto_materialized_result(
                    finding
                ) and not _is_validation_only_auto_materialized_result(finding):
                    continue
                if finding_id and finding_id not in seen_ids:
                    rows.append(finding)
                    seen_ids.add(finding_id)
            return rows
        except Exception as e:  # noqa: BLE001 - raw files remain a valid recovery surface.
            logger.debug("SQLite read failed, falling back to filesystem: %s", e)

    findings = [
        finding
        for finding in materialized_findings
        if _materialized_finding_visible_at_boundary(finding, gen_id)
        and (
            not _is_low_confidence_auto_materialized_result(finding)
            or _is_validation_only_auto_materialized_result(finding)
        )
    ]
    from praxist.plugins.tools.evaluation_tools.adapter import _gen_id_from_peer_id
    from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest import (
        derive_finding_id,
        sanitize_finding_effective_config_provenance,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
        _source_scoped_finding_id,
    )

    source_dirs = [findings_dir, run_dir / f"gen_{gen_id}" / "shared_findings"]
    if not any(source_dir.exists() for source_dir in source_dirs):
        return findings
    seen_finding_ids = {str(finding.get("id") or "") for finding in findings}
    for f_path in (
        path
        for source_dir in source_dirs
        if source_dir.exists()
        for path in source_dir.glob("*.json")
    ):
        try:
            finding = _load_json(f_path)
            if finding is None:
                continue
            sanitize_finding_effective_config_provenance(
                f_path,
                finding,
                maturity_policy=result_maturity_policy,
            )
            effective_gen_id = _gen_id_from_peer_id(finding.get("peer_id", ""))
            if effective_gen_id is None:
                effective_gen_id = finding.get("generation_id")
            if effective_gen_id == gen_id and (
                (
                    materialize_result_artifacts
                    and (
                        not _is_auto_materialized_result_finding(finding)
                        or (
                            _auto_materialized_source_exists(finding, run_dir)
                            and (
                                not _is_low_confidence_auto_materialized_result(finding)
                                or _is_validation_only_auto_materialized_result(finding)
                            )
                        )
                    )
                )
                or not _is_auto_materialized_result_finding(finding)
            ):
                finding = dict(finding)
                declared_id = str(finding.get("id") or finding.get("finding_id") or "").strip()
                finding_id = declared_id or derive_finding_id(f_path, finding)
                if finding_id in seen_finding_ids:
                    if _is_auto_materialized_result_finding(finding):
                        continue
                    finding_id = _source_scoped_finding_id(finding_id, f_path)
                if finding_id not in seen_finding_ids:
                    finding["id"] = finding_id
                    finding["source_filepath"] = str(f_path)
                    finding["source_filename"] = f_path.name
                    if declared_id and declared_id != finding_id:
                        finding["source_declared_finding_id"] = declared_id
                    findings.append(finding)
                    seen_finding_ids.add(finding_id)
            elif _is_auto_materialized_result_finding(
                finding
            ) and not _auto_materialized_source_exists(finding, run_dir):
                with contextlib.suppress(OSError):
                    f_path.unlink()
        except (json.JSONDecodeError, KeyError, OSError):
            continue

    return findings
