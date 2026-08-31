"""
Task specification loader.

Reads task_spec.yaml and provides structured access to task configuration.
"""

import logging
import math
import re
import shlex
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import yaml

from praxist.core.runtimes import REASONING_EFFORT_POLICIES
from praxist.task_spec_compat import migrate_legacy_gems_config

logger = logging.getLogger(__name__)


_VALID_ANCHOR_DIRECTIONS = ("maximize", "minimize")


def _normalize_diversity_dimensions(raw: Any) -> list[dict[str, str]]:
    """Normalize diversity_dimensions from task_spec.yaml.

    Each entry must have a `name` (str). Optional `description` and
    `examples`. Empty / missing returns []; downstream consumers
    fall back to the generic-science default in
    ``generation_loop._GENERIC_DIVERSITY_DIMENSIONS_DEFAULT``.

    Tolerates two YAML styles:
        diversity_dimensions:
          - name: foo
            description: bar
          - name: baz
    Or shorthand list of strings:
        diversity_dimensions:
          - foo
          - baz
    """
    out: list[dict[str, str]] = []
    if not raw:
        return out
    if not isinstance(raw, list):
        logger.warning(
            "evaluation.diversity_dimensions must be a list; got %r — ignored", type(raw).__name__
        )
        return out
    for entry in raw:
        try:
            if isinstance(entry, str):
                if entry.strip():
                    out.append({"name": entry.strip(), "description": "", "examples": ""})
            elif isinstance(entry, dict):
                name = str(entry.get("name", "")).strip()
                if not name:
                    logger.warning("Ignoring diversity_dimensions entry without name: %r", entry)
                    continue
                out.append(
                    {
                        "name": name,
                        "description": str(entry.get("description", "")),
                        "examples": str(entry.get("examples", "")),
                    }
                )
            else:
                logger.warning("Ignoring malformed diversity_dimensions entry: %r", entry)
        except (TypeError, ValueError) as e:
            logger.warning("Ignoring diversity_dimensions entry %r: %s", entry, e)
    return out


def _normalize_must_explore_axes(raw: Any) -> list[dict[str, str]]:
    """Normalize must_explore_axes from task_spec.yaml.

    Each axis is a free-text description of an under-explored research
    direction. The orchestrator round-robin-assigns these axes to peers
    at the start of each annealing cycle's explore phase so the cohort
    doesn't all converge on the same direction at cold-start.

    Tolerates two YAML styles (parity with diversity_dimensions):
        must_explore_axes:
          - name: cosine rho schedule
            description: rho varying along task progress
          - name: lookahead k-step
    Or shorthand list of strings:
        must_explore_axes:
          - cosine rho schedule (rho varying along task progress)
          - lookahead k-step / multi-step adversarial ascent

    Empty / missing returns [] → no axis pre-assignment, all peers
    get the generic explore hint (legacy behavior preserved).

    Round 3 (Plan C): an axis is a coordination signal, not a hard
    constraint. Peers may still pick anything that satisfies the
    research scope; the assigned axis is a strong hint for the
    explore-phase peer and is decoupled from frontier promotion.
    """
    out: list[dict[str, str]] = []
    if not raw:
        return out
    if not isinstance(raw, list):
        logger.warning(
            "evaluation.must_explore_axes must be a list; got %r — ignored",
            type(raw).__name__,
        )
        return out
    for entry in raw:
        try:
            if isinstance(entry, str):
                if entry.strip():
                    out.append({"name": entry.strip(), "description": ""})
            elif isinstance(entry, dict):
                # R3 Issue 3 fix: handle YAML `name: null` → Python None
                # (str(None) == "None" would otherwise pass the truthy
                # check and emit an axis literally named "None").
                raw_name = entry.get("name")
                name = str(raw_name).strip() if raw_name is not None else ""
                if not name:
                    logger.warning(
                        "Ignoring must_explore_axes entry without name: %r",
                        entry,
                    )
                    continue
                raw_desc = entry.get("description")
                desc = str(raw_desc).strip() if raw_desc is not None else ""
                out.append({"name": name, "description": desc})
            else:
                logger.warning(
                    "Ignoring malformed must_explore_axes entry: %r",
                    entry,
                )
        except (TypeError, ValueError) as e:
            logger.warning(
                "Ignoring must_explore_axes entry %r: %s",
                entry,
                e,
            )
    return out


def _normalize_anchor_metrics(raw: Any) -> list[tuple[str, str]]:
    """Normalize anchor_metrics from task_spec.yaml into a list of
    (metric_name, direction) tuples. Tolerates dict-style or
    list/tuple-style entries. Drops malformed entries with a warning.

    Pulled out of generation_loop.py per review round 1 m9: defensive
    parsing should live with the spec parser, not the consumer.

    Round 3 M1 fix: validate ``direction`` against the canonical set
    {maximize, minimize}. Anything else (e.g. "asc", "low") falls
    through both branches in FrontierStore.promote and silently picks
    the last-iterated finding rather than the intended best — wrong
    user intent, no error. We now drop with warning.
    """
    out: list[tuple[str, str]] = []
    if not raw:
        return out
    if not isinstance(raw, list):
        logger.warning(
            "evaluation.anchor_metrics must be a list; got %r — ignored", type(raw).__name__
        )
        return out
    for entry in raw:
        try:
            name: str | None = None
            direction: str | None = None
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                name, direction = str(entry[0]), str(entry[1])
            elif isinstance(entry, dict) and ("name" in entry or "metric" in entry):
                raw_name = entry.get("name") or entry.get("metric")
                if raw_name in (None, ""):
                    name = ""
                else:
                    name = str(raw_name)
                direction = str(entry.get("direction", "maximize"))
            else:
                logger.warning("Ignoring malformed anchor_metrics entry %r", entry)
                continue

            if not name:
                logger.warning("Ignoring anchor_metrics entry with empty name: %r", entry)
                continue
            if direction not in _VALID_ANCHOR_DIRECTIONS:
                logger.warning(
                    "Ignoring anchor_metrics entry %r: direction must be one of %s (got %r)",
                    entry,
                    _VALID_ANCHOR_DIRECTIONS,
                    direction,
                )
                continue
            out.append((name, direction))
        except (TypeError, ValueError) as e:
            logger.warning("Ignoring anchor_metrics entry %r: %s", entry, e)
    return out


def _normalize_frontier_lane_axes(
    raw: Any, *, default_direction: str = "maximize"
) -> list[tuple[str, str]]:
    """Normalize frontier-lane axes.

    Lane axes use the same structured forms as anchor_metrics, plus the common
    shorthand ``axes: ["score"]``. A string axis is interpreted with the lane's
    default direction so task-local compact YAML does not silently fall back to
    the primary metric.
    """
    if default_direction not in _VALID_ANCHOR_DIRECTIONS:
        default_direction = "maximize"
    if not raw:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "evaluation.frontier_lanes.axes must be a list; got %r — ignored",
            type(raw).__name__,
        )
        return []
    expanded: list[Any] = []
    for entry in raw:
        if isinstance(entry, str):
            name = entry.strip()
            if name:
                expanded.append({"name": name, "direction": default_direction})
            continue
        expanded.append(entry)
    return _normalize_anchor_metrics(expanded)


def _optional_float(raw: Any, *, field_name: str) -> float | None:
    if raw in (None, ""):
        return None
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        logger.warning("%s must be numeric; got %r — ignored", field_name, raw)
        return None
    if not math.isfinite(parsed):
        logger.warning("%s must be finite; got %r — ignored", field_name, raw)
        return None
    return parsed


def _float_or_default(raw: Any, default: float, *, field_name: str) -> float:
    parsed = _optional_float(raw, field_name=field_name)
    return default if parsed is None else parsed


def _optional_positive_limit(raw: Any, *, field_name: str) -> float | None:
    if raw is None:
        return None
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be null or a positive finite number") from exc
    if isinstance(raw, bool) or not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field_name} must be null or a positive finite number")
    return parsed


def _optional_positive_count(raw: Any, *, field_name: str) -> int | None:
    parsed = _optional_positive_limit(raw, field_name=field_name)
    if parsed is None:
        return None
    if not parsed.is_integer():
        raise ValueError(f"{field_name} must be null or a positive whole number")
    return int(parsed)


def _nonnegative_interval(raw: Any, *, field_name: str) -> float:
    if raw is None:
        return 0.0
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be null or a nonnegative finite number") from exc
    if isinstance(raw, bool) or not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field_name} must be null or a nonnegative finite number")
    return parsed


def _int_or_default(raw: Any, default: int, *, field_name: str) -> int:
    parsed = _optional_float(raw, field_name=field_name)
    if parsed is None:
        return default
    return int(parsed)


def _bool_or_default(raw: Any, default: bool, *, field_name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if not math.isfinite(float(raw)):
            logger.warning("%s must be finite boolean-like value; using %s", field_name, default)
            return default
        return raw != 0
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off", ""}:
            return False
    logger.warning("%s must be boolean-like; got %r — using %s", field_name, raw, default)
    return default


def _reasoning_effort_or_default(raw: Any) -> str:
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in REASONING_EFFORT_POLICIES:
            return normalized
    logger.warning(
        "agent.reasoning_effort must be one of %s; got %r - using max",
        ", ".join(sorted(REASONING_EFFORT_POLICIES)),
        raw,
    )
    return "max"


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in raw_items:
        try:
            text = str(item).strip()
        except (TypeError, ValueError):
            continue
        if text:
            out.append(text)
    return out


def _normalize_str_int_map(value: Any, *, field_name: str) -> dict[str, int]:
    if not value:
        return {}
    if not isinstance(value, dict):
        logger.warning("%s must be a mapping; got %r", field_name, type(value).__name__)
        return {}
    out: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        try:
            out[key] = max(1, int(raw_value))
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed %s entry %r=%r", field_name, raw_key, raw_value)
    return out


def _normalize_str_str_map(value: Any, *, field_name: str) -> dict[str, str]:
    if not value:
        return {}
    if not isinstance(value, dict):
        logger.warning("%s must be a mapping; got %r", field_name, type(value).__name__)
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        val = str(raw_value).strip()
        if key and val:
            out[key] = val
    return out


def _normalize_result_cell_metric_derivations(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if not isinstance(value, list):
        logger.warning(
            "gems.result_cell_metric_derivations must be a list; got %r",
            type(value).__name__,
        )
        return []
    out: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            logger.warning("Ignoring malformed result_cell_metric_derivations entry: %r", entry)
            continue
        name = str(entry.get("name") or entry.get("output") or "").strip()
        if not name:
            logger.warning("Ignoring result_cell_metric_derivations entry without name: %r", entry)
            continue
        source_keys = _normalize_str_list(entry.get("source_keys") or entry.get("cell_keys"))
        if not source_keys:
            logger.warning(
                "Ignoring result_cell_metric_derivations.%s without source_keys",
                name,
            )
            continue
        aggregate = str(entry.get("aggregate") or "mean").strip().lower()
        if aggregate not in {"mean", "q25"}:
            logger.warning(
                "gems.result_cell_metric_derivations.%s aggregate %r is invalid; using mean",
                name,
                aggregate,
            )
            aggregate = "mean"
        out.append(
            {
                "name": name,
                "source_keys": source_keys,
                "aggregate": aggregate,
                "validation_only": _bool_or_default(
                    entry.get("validation_only", False),
                    False,
                    field_name=f"gems.result_cell_metric_derivations.{name}.validation_only",
                ),
            }
        )
    return out


def _normalize_maturity_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}

    def finite_float(key: str, default: float) -> float:
        raw = value.get(key, default)
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "evaluation.maturity_policy.%s must be numeric; using %.2f", key, default
            )
            return default
        if not math.isfinite(parsed) or parsed < 0:
            logger.warning(
                "evaluation.maturity_policy.%s must be finite and non-negative; using %.2f",
                key,
                default,
            )
            return default
        return parsed

    return {
        "min_effort_ratio": finite_float("min_effort_ratio", 0.75),
        "min_coverage_ratio": finite_float("min_coverage_ratio", 0.80),
        # Compatibility default: when old task artifacts lack explicit ratios,
        # existing evidence-stage semantics remain usable. New task templates
        # should emit ratios and may set this true.
        "require_ratio_gate": _bool_or_default(
            value.get("require_ratio_gate", False),
            False,
            field_name="evaluation.maturity_policy.require_ratio_gate",
        ),
        # Stage labels are task-owned opaque metadata. A task that needs a
        # label-based compatibility mapping must declare it explicitly.
        "complete_stage_labels": _normalize_str_list(value.get("complete_stage_labels")),
        "preliminary_stage_labels": _normalize_str_list(value.get("preliminary_stage_labels")),
    }


def _normalize_launch_guard(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}

    def finite_float(key: str, default: float) -> float:
        raw = value.get(key, default)
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            logger.warning("evaluation.launch_guard.%s must be numeric; using %.2f", key, default)
            return default
        if not math.isfinite(parsed) or parsed < 0:
            logger.warning(
                "evaluation.launch_guard.%s must be finite and non-negative; using %.2f",
                key,
                default,
            )
            return default
        return parsed

    return {
        "enabled": _bool_or_default(
            value.get("enabled", True),
            True,
            field_name="evaluation.launch_guard.enabled",
        ),
        "estimated_heavy_eval_minutes": finite_float("estimated_heavy_eval_minutes", 0.0),
        "estimated_close_grade_eval_minutes": finite_float(
            "estimated_close_grade_eval_minutes",
            0.0,
        ),
        "safety_factor": max(1.0, finite_float("safety_factor", 1.25)),
    }


_GENERATION_DRAIN_MARGIN_MINUTES = 30.0


def _validate_declared_evaluation_horizon(
    evaluation: "EvaluationSpec",
    generation_policy: "GenerationPolicy",
    synthesis_trigger: "SynthesisTriggerConfig",
    *,
    close_grade_estimate_declared: bool,
) -> None:
    """Reject a task whose declared evaluator cannot reach its hard boundary.

    This check is intentionally conditional. Tasks may opt into reduced or
    late-signal workflows, but a task that requires formal close evidence must
    leave enough time for its own calibrated heavy evaluator plus the
    established drain margin.
    """

    launch_guard = evaluation.launch_guard if isinstance(evaluation.launch_guard, dict) else {}
    if not synthesis_trigger.enabled:
        return
    close_grade_minutes = float(launch_guard.get("estimated_close_grade_eval_minutes") or 0.0)
    legacy_heavy_minutes = float(launch_guard.get("estimated_heavy_eval_minutes") or 0.0)
    use_close_grade_estimate = close_grade_estimate_declared and close_grade_minutes > 0
    estimated_minutes = close_grade_minutes if use_close_grade_estimate else legacy_heavy_minutes
    if estimated_minutes <= 0:
        return
    safety_factor = max(1.0, float(launch_guard.get("safety_factor") or 1.0))
    adaptive = synthesis_trigger.adaptive if isinstance(synthesis_trigger.adaptive, dict) else {}
    adaptive_enabled = _bool_or_default(
        adaptive.get("enabled", False),
        False,
        field_name="synthesis_trigger.adaptive.enabled",
    )
    formal_close_required = synthesis_trigger.mature_quorum_fraction > 0
    if not formal_close_required:
        return

    peer_horizon = (
        float(generation_policy.per_generation_hours) * 60.0
        if generation_policy.per_generation_hours is not None
        else None
    )
    close_horizon = peer_horizon
    if synthesis_trigger.enabled:
        adaptive_ceiling = (
            _float_or_default(
                adaptive.get("max_interval_ceiling_minutes", 0.0),
                0.0,
                field_name="synthesis_trigger.adaptive.max_interval_ceiling_minutes",
            )
            if adaptive_enabled
            else 0.0
        )
        trigger_horizons = [
            value
            for value in (synthesis_trigger.max_interval_minutes, adaptive_ceiling)
            if value is not None and value > 0
        ]
        if trigger_horizons:
            trigger_horizon = max(trigger_horizons)
            close_horizon = (
                min(peer_horizon, trigger_horizon) if peer_horizon is not None else trigger_horizon
            )
    if close_horizon is None:
        return
    required_runtime = estimated_minutes * safety_factor
    usable_runtime = close_horizon - _GENERATION_DRAIN_MARGIN_MINUTES
    if required_runtime < usable_runtime:
        return

    if not use_close_grade_estimate:
        logger.warning(
            "legacy task timing may be unreachable: "
            "estimated_heavy_eval_minutes=%g * safety_factor=%g requires %.1f "
            "minutes, but the effective generation close horizon is %.1f minutes. "
            "The heavy evaluator may be optional or a user-authorized shorter "
            "protocol may own close, so this legacy task remains runnable. Declare "
            "estimated_close_grade_eval_minutes to validate the actual close-grade "
            "protocol.",
            estimated_minutes,
            safety_factor,
            required_runtime,
            close_horizon,
        )
        return

    minimum_horizon = required_runtime + _GENERATION_DRAIN_MARGIN_MINUTES
    estimate_field = (
        "estimated_close_grade_eval_minutes"
        if use_close_grade_estimate
        else "estimated_heavy_eval_minutes"
    )
    raise ValueError(
        "task evaluation timing is unreachable: "
        f"{estimate_field}={estimated_minutes:g} * "
        f"safety_factor={safety_factor:g} requires {required_runtime:.1f} minutes, "
        f"but the effective generation close horizon is {close_horizon:.1f} minutes "
        f"with a {_GENERATION_DRAIN_MARGIN_MINUTES:.0f}-minute drain margin. "
        f"Calibrate from observed complete-evaluator p90 runtime and raise both "
        f"synthesis_trigger.max_interval_minutes (or its adaptive ceiling) and "
        f"generation_policy.per_generation_hours so the effective horizon exceeds "
        f"{minimum_horizon:.1f} minutes. If the required close-grade protocol is "
        "shorter than the optional heavy evaluator, declare its calibrated runtime "
        "with estimated_close_grade_eval_minutes. Tasks that intentionally accept "
        "only late signals may keep formal close quorum disabled."
    )


def _is_python_command_name(value: str) -> bool:
    name = Path(value).name
    suffix = name.removeprefix("python")
    if not name.startswith("python"):
        return False
    if not suffix:
        return True
    parts = suffix.split(".")
    return bool(parts) and all(part and part.isdigit() for part in parts)


def shell_command_script_index(command: list[str]) -> int | None:
    """Return the command-string index for a static ``sh``/``bash`` argv."""

    index = 1
    options_with_value = {"-O", "-o", "--init-file", "--rcfile"}
    while index < len(command):
        value = command[index]
        if value in options_with_value:
            index += 2
            continue
        if value == "--" or not value.startswith("-"):
            return None
        if not value.startswith("--") and "c" in value[1:]:
            return index + 1 if index + 1 < len(command) else None
        index += 1
    return None


def _declared_entrypoint_token(tokens: list[str]) -> str:
    if not tokens:
        return ""
    assignment_count = 0
    while assignment_count < len(tokens) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*=.*",
        tokens[assignment_count],
    ):
        assignment_count += 1
    if assignment_count:
        return _declared_entrypoint_token(tokens[assignment_count:])
    program = Path(tokens[0]).name
    if program == "exec":
        index = 1
        while index < len(tokens):
            value = tokens[index]
            if value == "--":
                index += 1
                break
            if value == "-a":
                index += 2
                continue
            if value in {"-c", "-l"}:
                index += 1
                continue
            if value.startswith("-"):
                return ""
            break
        return _declared_entrypoint_token(tokens[index:])
    if program == "command":
        index = 1
        while index < len(tokens):
            value = tokens[index]
            if value == "--":
                index += 1
                break
            if value == "-p":
                index += 1
                continue
            if value in {"-v", "-V"} or value.startswith("-"):
                return ""
            break
        return _declared_entrypoint_token(tokens[index:])
    if program == "env":
        index = 1
        while index < len(tokens):
            value = tokens[index]
            if value == "--":
                index += 1
                break
            if value in {"-C", "--chdir"}:
                if index + 1 >= len(tokens) or any(
                    marker in tokens[index + 1] for marker in ("$", "{", "}", "*", "?")
                ):
                    return ""
                index += 2
                continue
            if value.startswith("--chdir="):
                if any(marker in value for marker in ("$", "{", "}", "*", "?")):
                    return ""
                index += 1
                continue
            if value in {"-S", "--split-string"} or value.startswith("--split-string="):
                return ""
            if value in {"-u", "--unset"}:
                index += 2
                continue
            if value.startswith("-") or ("=" in value and not value.startswith(("/", "."))):
                index += 1
                continue
            break
        return _declared_entrypoint_token(tokens[index:])
    if _is_python_command_name(tokens[0]):
        index = 1
        while index < len(tokens):
            value = tokens[index]
            if value in {"-c", "-m"}:
                return ""
            if value in {"-W", "-X", "--check-hash-based-pycs"}:
                index += 2
                continue
            if value == "--":
                return tokens[index + 1] if index + 1 < len(tokens) else ""
            if value.startswith("-"):
                index += 1
                continue
            return value
        return ""
    if program in {"sh", "bash"}:
        script_index = shell_command_script_index(tokens)
        if script_index is not None:
            return _declared_shell_launch(tokens[script_index])[0]
        for index, value in enumerate(tokens[1:], start=1):
            if value == "--":
                return tokens[index + 1] if index + 1 < len(tokens) else ""
            if not value.startswith("-"):
                return value
        return ""
    return tokens[0]


def _declared_env_chdir(tokens: list[str]) -> str:
    """Return the last static ``env --chdir`` value, if one is declared."""

    if not tokens:
        return ""
    assignment_count = 0
    while assignment_count < len(tokens) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*=.*",
        tokens[assignment_count],
    ):
        assignment_count += 1
    if assignment_count:
        return _declared_env_chdir(tokens[assignment_count:])
    program = Path(tokens[0]).name
    if program in {"exec", "command"}:
        index = 1
        while index < len(tokens):
            value = tokens[index]
            if value == "--":
                index += 1
                break
            if program == "exec" and value == "-a":
                index += 2
                continue
            if program == "exec" and value in {"-c", "-l"}:
                index += 1
                continue
            if program == "command" and value == "-p":
                index += 1
                continue
            if value.startswith("-"):
                return ""
            break
        return _declared_env_chdir(tokens[index:])
    if program in {"sh", "bash"}:
        script_index = shell_command_script_index(tokens)
        return _declared_shell_launch(tokens[script_index])[1] if script_index is not None else ""
    if program != "env":
        return ""
    selected = ""
    index = 1
    while index < len(tokens):
        value = tokens[index]
        if value == "--":
            break
        if value in {"-C", "--chdir"}:
            if index + 1 >= len(tokens):
                return ""
            selected = tokens[index + 1]
            index += 2
            continue
        if value.startswith("--chdir="):
            selected = value.split("=", 1)[1]
        if value in {"-u", "--unset", "-S", "--split-string"}:
            index += 2
            continue
        if value.startswith("-") or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", value):
            index += 1
            continue
        # The first non-option/non-assignment is env's program.  Arguments
        # after it belong to that program, even when they are named --chdir.
        break
    if any(marker in selected for marker in ("$", "{", "}", "*", "?")):
        return ""
    return selected


def _declared_shell_launch(command: str) -> tuple[str, str]:
    """Extract one static evaluator token and its shell-local cwd."""

    if not command or any(marker in command for marker in ("`", "\n", "\r")):
        return "", ""
    redirect_probe = re.sub(r"\d*>\s*&\s*\d+", "", command)
    if "#" in command or re.search(r"(?<!\s)[<>]|[<>](?!\s)", redirect_probe):
        # Preserve runtime resolution for syntax whose token boundaries are
        # ambiguous. Ordinary descriptor redirection (for example 2>&1) is
        # removed above and remains statically resolvable.
        return "", ""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return "", ""

    segments: list[tuple[list[str], str]] = []
    segment: list[str] = []
    for token in [*tokens, ""]:
        if token in {"&&", "||", ";", "|", "&", ""}:
            if segment:
                segments.append((segment, token))
            segment = []
        else:
            segment.append(token)

    active_cwd = ""
    selected_token = ""
    selected_cwd = ""
    for words, separator in segments:
        index = 0
        while index < len(words) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*",
            words[index],
        ):
            index += 1
        if index >= len(words):
            continue
        program = Path(words[index]).name
        if program == "cd" and index + 1 < len(words):
            previous_cwd = active_cwd
            raw_cwd = words[index + 1]
            if not raw_cwd.startswith("-") and not any(
                marker in raw_cwd for marker in ("$", "{", "}", "*", "?", "~")
            ):
                path = Path(raw_cwd)
                if not path.is_absolute() and active_cwd:
                    path = Path(active_cwd) / path
                active_cwd = str(path)
            if separator in {"||", "|", "&"}:
                # These operators do not carry a successful cd into the next
                # foreground command: || observes failure, while | and & run
                # the cd in a separate process context.
                active_cwd = previous_cwd
            continue
        if program in {".", "source", "export", "unset", "readonly"}:
            continue

        candidate = _declared_entrypoint_token(words)
        if not candidate:
            continue
        candidate_path = Path(candidate)
        candidate_is_path = bool(
            "/" in candidate
            or "\\" in candidate
            or candidate.startswith(".")
            or candidate_path.suffix in {".py", ".sh"}
        )
        segment_cwd = _declared_env_chdir(words)
        if segment_cwd and active_cwd and not Path(segment_cwd).is_absolute():
            segment_cwd = str(Path(active_cwd) / segment_cwd)
        elif not segment_cwd:
            segment_cwd = active_cwd
        if not selected_token or candidate_is_path:
            selected_token = candidate
            selected_cwd = segment_cwd

    return selected_token, selected_cwd


def declared_evaluation_entrypoint_token(command: str) -> str:
    """Return the static evaluator path token from a task entrypoint command."""

    raw = str(command or "").strip()
    if not raw:
        return ""
    return _declared_shell_launch(raw)[0]


def declared_evaluation_entrypoint_chdir(command: str) -> str:
    """Return the static working-directory component of an evaluator command."""

    return _declared_shell_launch(str(command or "").strip())[1]


def _declared_evaluation_entrypoint_candidates(
    command: str,
    *,
    task_dir: Path,
    runtime_cwd: object = None,
) -> list[Path]:
    """Return task-owned candidates for a statically declared evaluator."""

    raw = str(command or "").strip()
    if not raw:
        return []
    candidate = declared_evaluation_entrypoint_token(raw)
    if not candidate or any(marker in candidate for marker in ("$", "{", "}", "*", "?")):
        return []
    candidate_path = Path(candidate)
    if (
        "/" not in candidate
        and "\\" not in candidate
        and not candidate.startswith(".")
        and candidate_path.suffix not in {".py", ".sh"}
    ):
        # Bare commands may be external executables resolved through PATH.
        return []
    path = candidate_path.expanduser()
    if path.is_absolute():
        return [path.resolve()]

    raw_cwd = str(runtime_cwd or "task_project").strip()
    # The task descriptor owns evaluator identity. An explicitly configured
    # task cwd is part of that declaration, while ``run_dir`` is only an
    # execution location and must not let a stale run-local copy shadow it.
    base_candidates: list[Path] = []
    if raw_cwd not in {"", ".", "task_project", "run_dir"}:
        configured_cwd = Path(raw_cwd).expanduser()
        if not configured_cwd.is_absolute():
            configured_cwd = task_dir / configured_cwd
        base_candidates.append(configured_cwd)
    if task_dir not in base_candidates:
        base_candidates.append(task_dir)
    env_chdir = declared_evaluation_entrypoint_chdir(raw)
    candidates: list[Path] = []
    if env_chdir:
        declared_chdir = Path(env_chdir).expanduser()
        if declared_chdir.is_absolute():
            candidates.append(declared_chdir / path)
        else:
            candidates.extend(base / declared_chdir / path for base in base_candidates)
    else:
        candidates.extend(base / path for base in base_candidates)
    return list(dict.fromkeys(candidate.resolve() for candidate in candidates))


def resolve_declared_evaluation_entrypoint(
    command: str,
    *,
    task_dir: Path,
    runtime_cwd: object = None,
) -> Path | None:
    """Resolve a static evaluator command to its existing task-owned file."""

    return next(
        (
            candidate
            for candidate in _declared_evaluation_entrypoint_candidates(
                command,
                task_dir=task_dir,
                runtime_cwd=runtime_cwd,
            )
            if candidate.is_file()
        ),
        None,
    )


def _validate_declared_evaluation_entrypoint(
    command: str,
    *,
    task_dir: Path,
    runtime_cwd: object = None,
) -> None:
    """Validate a static evaluator against its configured cwd or task root."""

    raw = str(command or "").strip()
    candidates = _declared_evaluation_entrypoint_candidates(
        raw,
        task_dir=task_dir,
        runtime_cwd=runtime_cwd,
    )
    if not candidates:
        return
    raw_cwd = str(runtime_cwd or "task_project").strip()
    if raw_cwd == "run_dir" and not any(candidate.exists() for candidate in candidates):
        # The run directory is created after task-spec loading. Runtime command
        # preparation will resolve and validate this dynamic location.
        return
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    if existing is None:
        if re.search(r"&&|\|\||[;|&]", raw):
            # A preceding command may generate the evaluator or select a
            # conditional branch. Preserve custom protocols for runtime.
            return
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "task evaluation entrypoint not found relative to task root or configured "
            f"runtime cwd: {searched}"
        )
    if not existing.is_file():
        raise ValueError(f"task evaluation entrypoint must be a file: {existing}")


def _float_between(
    value: Any, *, default: float, field_name: str, minimum: float, maximum: float
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning("task_spec.%s must be numeric; using %.2f", field_name, default)
        return default
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        logger.warning(
            "task_spec.%s=%.3f is outside [%.3f, %.3f]; using %.2f",
            field_name,
            parsed,
            minimum,
            maximum,
            default,
        )
        return default
    return parsed


def _normalize_metric_bounds(value: Any, *, field_name: str) -> dict[str, float]:
    if not value:
        return {}
    if not isinstance(value, dict):
        logger.warning(
            "evaluation.frontier_lanes.%s must be a mapping; got %r",
            field_name,
            type(value).__name__,
        )
        return {}
    out: dict[str, float] = {}
    for metric, raw in value.items():
        try:
            metric_name = str(metric).strip()
            metric_value = float(raw)
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed frontier lane metric bound %r=%r", metric, raw)
            continue
        if not math.isfinite(metric_value):
            logger.warning("Ignoring non-finite frontier lane metric bound %r=%r", metric, raw)
            continue
        if metric_name:
            out[metric_name] = metric_value
    return out


def _normalize_frontier_lanes(raw: Any) -> list[dict[str, Any]]:
    """Normalize optional lane-based frontier promotion config.

    The default empty value preserves the legacy single-primary-metric
    promotion path. Tasks that opt in can reserve independent promotion lanes
    for deployable candidates, benchmark floors, diagnostic controls, or process
    audits without hard-coding any task-specific taxonomy into Praxist core.
    """
    if not raw:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "evaluation.frontier_lanes must be a list; got %r — ignored",
            type(raw).__name__,
        )
        return []
    lanes: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("Ignoring malformed frontier_lanes entry: %r", entry)
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            logger.warning("Ignoring frontier_lanes entry without name: %r", entry)
            continue
        try:
            k = max(0, int(entry.get("k", 1)))
        except (TypeError, ValueError):
            logger.warning("frontier_lanes[%s].k must be an int; using 1", name)
            k = 1
        cumulative_cap = entry.get("cumulative_cap")
        if cumulative_cap is not None:
            try:
                cumulative_cap = max(1, int(cumulative_cap))
            except (TypeError, ValueError):
                logger.warning(
                    "frontier_lanes[%s].cumulative_cap must be an int; ignored",
                    name,
                )
                cumulative_cap = None
        lane_default_direction = str(entry.get("direction", "maximize")).strip()
        axes = _normalize_frontier_lane_axes(
            entry.get("axes", []),
            default_direction=lane_default_direction,
        )
        allow_lower_tier = _bool_or_default(
            entry.get("allow_lower_tier", False),
            False,
            field_name=f"evaluation.frontier_lanes.{name}.allow_lower_tier",
        )
        lane = {
            "name": name,
            "k": k,
            "axes": axes,
            "optional_axes": _normalize_frontier_lane_axes(
                entry.get("optional_axes", []),
                default_direction=lane_default_direction,
            ),
            "include_lanes": _normalize_str_list(entry.get("include_lanes")),
            "exclude_lanes": _normalize_str_list(entry.get("exclude_lanes")),
            "include_families": _normalize_str_list(entry.get("include_families")),
            "exclude_families": _normalize_str_list(entry.get("exclude_families")),
            "include_tags": _normalize_str_list(entry.get("include_tags")),
            "exclude_tags": _normalize_str_list(entry.get("exclude_tags")),
            "include_roles": _normalize_str_list(entry.get("include_roles")),
            "exclude_roles": _normalize_str_list(entry.get("exclude_roles")),
            "require_metrics": _normalize_str_list(entry.get("require_metrics")),
            "require_truthy_metrics": _normalize_str_list(entry.get("require_truthy_metrics")),
            "require_falsey_metrics": _normalize_str_list(entry.get("require_falsey_metrics")),
            "min_metrics": _normalize_metric_bounds(
                entry.get("min_metrics"), field_name="min_metrics"
            ),
            "max_metrics": _normalize_metric_bounds(
                entry.get("max_metrics"), field_name="max_metrics"
            ),
            "allow_risk_violating": _bool_or_default(
                entry.get("allow_risk_violating", False),
                False,
                field_name=f"evaluation.frontier_lanes.{name}.allow_risk_violating",
            ),
            "allow_lower_tier": allow_lower_tier,
            # Lower-tier lanes are revalidation queues by default. Tasks may
            # explicitly allow their mature entries to act as parents without
            # relying on domain-specific lane names.
            "parent_eligible": _bool_or_default(
                entry.get("parent_eligible", not allow_lower_tier),
                not allow_lower_tier,
                field_name=f"evaluation.frontier_lanes.{name}.parent_eligible",
            ),
            "allow_non_promotable": _bool_or_default(
                entry.get("allow_non_promotable", False),
                False,
                field_name=f"evaluation.frontier_lanes.{name}.allow_non_promotable",
            ),
            "allow_missing_tier": _bool_or_default(
                entry.get("allow_missing_tier", False),
                False,
                field_name=f"evaluation.frontier_lanes.{name}.allow_missing_tier",
            ),
            "admit_new_high": _bool_or_default(
                entry.get("admit_new_high", False),
                False,
                field_name=f"evaluation.frontier_lanes.{name}.admit_new_high",
            ),
            "description": str(entry.get("description", "")),
        }
        if cumulative_cap is not None:
            lane["cumulative_cap"] = cumulative_cap
        lanes.append(lane)
    return lanes


@dataclass
class ComputeBudget:
    """Task-declared compute budget and optional scheduler hints.

    Missing values mean unknown and must be measured or supplied by the task.
    Praxist does not infer an accelerator model, memory size, utilization envelope,
    or CPU demand from the research domain.
    """

    per_experiment_gpu_hours: float = 0.0
    max_parallel_runs_per_peer: int | None = None
    peer_gpu_memory_gb: float | None = None
    peer_gpu_util_pct: float | None = None
    peer_cpu_cores: int | None = None
    # New tasks may opt into the central experiment scheduler.  A plain dict
    # keeps the task contract extensible without leaking scheduler internals
    # into the stable TaskSpec API.  Legacy/absent configuration is unchanged.
    resource_scheduler: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSpec:
    """Legacy task-spec evaluation settings for benchmark and metric selection."""

    primary_metric: str = "metric_value"
    direction: str = "maximize"  # maximize | minimize
    aux_metrics: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    aggregation: str = "mean_and_std"  # mean_and_std | median_and_iqr
    # Multi-anchor frontier: optional secondary anchor metrics. Each
    # entry adds ONE additional finding to the per-generation
    # promotion set: the variant with the best value on that
    # secondary metric (subject to dedup against primary picks).
    # Format: list of dicts {name: ..., direction: maximize|minimize},
    # or list of [name, direction] tuples. Empty / missing falls back
    # to single-metric behavior (pre-2026-04-30 default).
    anchor_metrics: list[Any] = field(default_factory=list)
    # Diversity-penalty dimensions used by the explore-phase prompt.
    # Each entry: {name, description, examples?}. The "differ in
    # ≥ ⌈M/2⌉ of M" rule scales with len(diversity_dimensions).
    # Empty / missing falls back to a 4-dim generic-science default
    # (problem formulation / methodological approach / key novel
    # mechanism / operating regime). RL or AI tasks should override
    # with their own domain-specific dimensions (architecture,
    # rl_algorithm, etc.); chemistry/optimization/SE tasks should
    # use their own; the default is intentionally domain-agnostic.
    diversity_dimensions: list[Any] = field(default_factory=list)
    # R6-N1 hardening: when True, the frontier rejects findings that
    # don't carry a `tier` field in metrics. Used by tiered-evaluation tasks
    # whose protocol mandates tiered eval and where peers MUST self-report
    # which tier produced each finding. Default False preserves the
    # legacy behavior for non-tiered tasks.
    requires_tier: bool = False
    # Optional lane-based frontier promotion. When empty, FrontierStore keeps
    # the legacy behavior: top-K by primary_metric plus secondary anchors.
    # When configured, each lane independently selects up to k findings using
    # its own filters and Pareto axes, so task authors can separate deployable
    # candidates, benchmark floors, controls, and process-audit artifacts.
    frontier_lanes: list[dict[str, Any]] = field(default_factory=list)
    # Must-explore axes (Praxist v2026-05-01 Plan C):
    # A list of free-text descriptions of under-explored research
    # directions for THIS task. At the start of each annealing cycle's
    # explore phase, the orchestrator round-robin-assigns these axes
    # to the first N peers (one axis per peer, where N = min(len(axes),
    # cohort_size)). Peers beyond N get the normal generic-explore
    # hint. In mixed/exploit phases, no assignment is applied —
    # peer roles return to homogeneous (Plan C: assignment is one-shot
    # per cycle, not persistent across the cycle). Empty / missing
    # disables the feature (no behavior change for tasks that don't
    # opt in).
    must_explore_axes: list[dict[str, str]] = field(default_factory=list)
    # Generic mature-evidence predicate. Tasks should emit effort_ratio and
    # coverage_ratio (or the equivalent actual/reference fields) in result
    # metrics. Stage labels remain audit context, not default hard gates.
    maturity_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "min_effort_ratio": 0.75,
            "min_coverage_ratio": 0.80,
            "require_ratio_gate": False,
        }
    )
    # Desired minimum fraction of constructive solution-producing work. This is
    # advisory feedback to PI/DIG; it does not hard-fail generations.
    constructive_peer_mix_enabled: bool = True
    constructive_target_ratio: float = 0.75
    # Generation-close launch freeze for new evaluations. Existing work drains
    # naturally; result publication and notebook/memory updates remain allowed.
    launch_guard: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "estimated_heavy_eval_minutes": 0.0,
            "estimated_close_grade_eval_minutes": 0.0,
            "safety_factor": 1.25,
        }
    )


@dataclass
class Baseline:
    """Legacy baseline descriptor used by compatibility task specifications."""

    name: str = ""
    # Backward-compatible alias for older examples/tests that used accuracy.
    expected_acc: float | None = None
    metric_name: str = "metric_value"
    metric_value: float | None = None
    direction: str = ""

    def __post_init__(self) -> None:
        if self.metric_value is None:
            self.metric_value = float(self.expected_acc or 0.0)
        if self.expected_acc is None:
            self.expected_acc = float(self.metric_value)


@dataclass
class Toolchain:
    """Legacy toolchain descriptor for task-local harness execution."""

    framework: str = "task_defined"
    entrypoint_template: str = ""
    eval_entrypoint: str = ""
    benchmark_entrypoint: str = ""


@dataclass
class GenerationPolicy:
    """Generation policy fields controlling cohort and generation counts.

    Task projects should declare values measured from their unchanged baseline.
    The package defaults retain the established execution shape for existing
    task specs that predate explicit calibration fields.
    Loaded scoreless tasks use ``None`` for omitted or null generation and
    per-peer runtime caps; resource-aware cohort sizing remains active.
    """

    max_generations: int | None = 8
    cohort_size: int = 5
    # `per_generation_hours` retained as the hard upper bound (safety cap)
    # for any single peer. The actual gen termination is event-driven via
    # `synthesis_trigger` — see SynthesisTriggerConfig below.
    per_generation_hours: float | None = 5.0
    promote_top_k: int = 2
    promote_criterion: str = "primary_metric"


@dataclass
class RunLifecyclePolicy:
    """Generic run-level stop policy evaluated before starting a generation."""

    max_wall_clock_hours: float | None = None
    stop_signal_path: str = ""


@dataclass
class SynthesisTriggerConfig:
    """Event-driven generation termination via synthesis trigger.

    A generation runs until ANY of these conditions is satisfied (whichever
    comes first):

      (A) Information-density trigger:
          findings_in_gen >= min_findings
          AND minutes_since_gen_start >= min_interval_minutes
          AND distinct_contributing_peers >= min_contributing_peers

      (B) Safety cap:
          minutes_since_gen_start >= max_interval_minutes

    When the trigger fires, the orchestrator writes a sentinel file
    (<gen_dir>/STOP_SIGNAL) that peers detect at their next safe checkpoint.
    Peers drain in-flight work then exit gracefully.

    Defaults preserve the established event-driven behavior. Task
    initialization should still calibrate all density and time fields from the
    unchanged baseline.
    """

    enabled: bool = True
    min_findings: int = 30
    min_interval_minutes: float = 120.0
    max_interval_minutes: float | None = 240.0
    min_contributing_peers: int = 3
    mature_quorum_fraction: float = 0.0
    # Legacy field name. The event-driven trigger treats this as a minimum
    # heartbeat fallback, not as a short polling cadence.
    poll_interval_seconds: int = 30
    adaptive: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """2026-05-07: Top-level agent runtime config.

    Applies to ALL LLM agent calls in the run: peers (AutonomousAgentLoop),
    PIs (Multi-PI panel Round 1/2), Chair (synthesizer). Set via
    `agent:` block in task_spec.yaml.

    ``reasoning_effort`` is a provider-neutral policy. Runtime adapters map it
    to their supported wire contract. ``premium_mode`` is retained as a
    compatibility alias for ``reasoning_effort: max`` when the latter is
    ``auto``.
    """

    premium_mode: bool = False
    reasoning_effort: str = "max"


@dataclass
class PIAgentConfig:
    """v2026-05-04: PI / Synthesis agent configuration.

    The PI agent runs ONCE between every pair of generations (so 7 times
    in an 8-generation run). It reads the just-completed generation's
    state and outputs a research_agenda_gen{N+1}.yaml that the next
    generation's peers receive in their prompts as explicit role contracts.
    """

    enabled: bool = True
    max_runtime_minutes: int | None = 15
    # When True, run will abort if PI fails to produce a valid agenda.
    # When False, fall back to the previous generation's agenda (or no
    # agenda if gen 0 was just completed and no agenda exists yet).
    strict: bool = False


@dataclass
class MultiPIConfig:
    """v2026-05-05: Multi-PI panel configuration.

    Default `enabled: False` keeps the v2026-05-04 single-PI behavior intact.
    See the workflow and state sections of docs/concepts/architecture.md.
    """

    enabled: bool = False
    panel_mode_default: str = "full"  # mini | full | high_stakes
    auto_escalate_to_high_stakes: bool = True
    pi_max_runtime_minutes: int | None = 12
    chair_max_runtime_minutes: int | None = 8
    chair_peer_budget: int = 5
    shared_core_ratio_target: float = 0.65
    private_kb_ratio_target: float = 0.35
    fallback_to_single_pi_on_panel_failure: bool = True
    # v2026-05-05+: number of LLM rounds per panel synthesis.
    #   1 = Round 1 only (independent memos), Chair sees no cross-review LLM output
    #   2 = + Round 2 cross-review (each PI reads anonymized peers, answers 6 Q's)
    # Round 2.5 (boundary revision) is still derived structurally from Round 2
    # outputs — not a separate LLM call.
    n_rounds: int = 2
    round2_max_runtime_minutes: int | None = 6


@dataclass
class ResearchMemoryConfig:
    """v2026-05-05: Research memory (ledgers + evidence pack) configuration.

    Default `enabled: False` keeps the v2026-05-04 raw-prompt path intact.
    """

    enabled: bool = False
    rollout_phase: int = 0  # -1=shadow, 0=ledgers-only, 1=single-PI uses pack, 2+=multi-PI
    evidence_pack_max_cards: int = 40
    citation_coverage_min: float = 0.95
    negative_evidence_ratio_min: float = 0.20
    bridge_coverage_check_required: bool = True


@dataclass
class PromptLayoutConfig:
    """Optional task-local prompt template overrides.

    Empty fields preserve the bundled research-loop templates. Task projects
    may provide prompt templates when the bundled peer framing is too domain
    specific, while still using the same Praxist prompt-layout renderer.
    """

    base_template: str = ""
    generation_template: str = ""


@dataclass
class GemsConfig:
    """Opt-in periodic Gems reset mechanism.

    When enabled, the research loop snapshots the current Pareto/lane frontier
    into durable Gems every ``reset_interval_generations`` completed logical
    generations, archives ordinary findings, and starts a new logical
    generation-0 cycle without resetting the absolute generation budget.
    """

    enabled: bool = False
    reset_interval_generations: int = 6
    max_resets: int = 3
    max_gems_per_reset: int = 4
    max_gems_total: int = 4
    max_gems_per_family: int = 2
    min_frontier_entries: int = 1
    archive_ordinary_findings: bool = True
    signature_top_k: int = 16
    signature_entries_per_lane: int = 8
    prompt_max_gems: int = 4
    include_lanes: list[str] = field(default_factory=list)
    selection_policy: str = "frontier_lane_balanced"
    min_mature_eval_units: int = 1
    evidence_stage_min_units: dict[str, int] = field(default_factory=dict)
    primary_metric_keys: list[str] = field(default_factory=list)
    secondary_metric_keys: list[str] = field(default_factory=list)
    lower_tail_metric_keys: list[str] = field(default_factory=list)
    validation_metric_keys: list[str] = field(default_factory=list)
    cost_metric_keys: list[str] = field(default_factory=list)
    result_cell_metric_derivations: list[dict[str, Any]] = field(default_factory=list)
    result_metric_aliases: dict[str, str] = field(default_factory=dict)
    gem_seeded_independent_peers: int = 0
    performance_lanes: list[str] = field(default_factory=list)
    control_lanes: list[str] = field(default_factory=list)
    bottleneck_detector_mode: str = "generic"
    result_artifact_materialization: bool = True
    result_artifact_default_lane: str = "performance"
    result_artifact_default_family: str = "task_candidate"


GEMS_MATURE_EVIDENCE_TOP_K = "mature_evidence_top_k"


def normalize_gems_selection_policy(value: Any) -> str:
    """Normalize a task-facing generic Gems policy token."""

    return str(value or "frontier_lane_balanced").strip().lower()


@dataclass
class RuntimeEnvironmentConfig:
    """Runtime environment fields consumed by backend-neutral helpers."""

    # Task-root-relative child paths that peers may execute/read but must not mutate.
    protected_child_paths: list[str] = field(default_factory=list)
    # Optional task-owned aliases for the resolved data directory. Praxist core treats
    # these as opaque env names; tasks opt in when legacy harnesses need them.
    data_env_aliases: list[str] = field(default_factory=list)


@dataclass
class TaskSpec:
    """Structured representation of a task_spec.yaml file."""

    task_id: str = ""
    task_name: str = ""
    description_file: str = ""
    research_direction: str = ""
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    compute_budget: ComputeBudget = field(default_factory=ComputeBudget)
    toolchain: Toolchain = field(default_factory=Toolchain)
    baselines: list[Baseline] = field(default_factory=list)
    generation_policy: GenerationPolicy = field(default_factory=GenerationPolicy)
    run_lifecycle: RunLifecyclePolicy = field(default_factory=RunLifecyclePolicy)
    research_loop: dict[str, Any] = field(default_factory=dict)
    execution_policy: dict[str, Any] = field(default_factory=dict)
    synthesis_trigger: SynthesisTriggerConfig = field(default_factory=SynthesisTriggerConfig)
    pi_agent: PIAgentConfig = field(default_factory=PIAgentConfig)
    multi_pi: MultiPIConfig = field(default_factory=MultiPIConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    research_memory: ResearchMemoryConfig = field(default_factory=ResearchMemoryConfig)
    prompt_layout: PromptLayoutConfig = field(default_factory=PromptLayoutConfig)
    gems: GemsConfig = field(default_factory=GemsConfig)
    runtime_environment: RuntimeEnvironmentConfig = field(default_factory=RuntimeEnvironmentConfig)
    # Panel topology ref read from ``praxist_plugins.panel.topology`` in
    # the task project's descriptor. The legacy default keeps single-task
    # behavior backwards-compatible; custom projects (e.g. ``rocket_8e_panel``
    # with its four-PI interpreter role) override it via the descriptor. The
    # research loop's ``GenerationLoop`` / ``PIAgent`` / ``run_panel`` chain
    # threads this through so the panel actually instantiates the declared
    # PIs instead of the legacy three.
    panel_topology_ref: str = "panel_topology:legacy_multi_pi_two_round"
    # Optional project-owned tiered eval config. Core treats tier labels as
    # opaque metadata; this dict is for prompt rendering and operator-side
    # documentation. Empty for tasks that don't use tiered eval.
    tiered_eval: dict[str, dict[str, Any]] = field(default_factory=dict)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)
    _task_dir: str = ""

    @property
    def task_dir(self) -> Path:
        return Path(self._task_dir)

    @property
    def description_path(self) -> Path:
        return self.task_dir / self.description_file

    def get_description(self) -> str:
        p = self.description_path
        if p.exists():
            return p.read_text()
        return self.research_direction

    def get_prompt_task_path(self) -> Path:
        return self.task_dir / "prompt_task.jinja2"

    def get_prompt_base_path(self, default_path: Path) -> Path:
        """Return the task-local base prompt template or ``default_path``.

        ``prompt_layout.base_template`` is resolved relative to the task
        directory unless absolute. The file is validated here so runtime prompt
        rendering fails with a clear task-spec error instead of a late Jinja
        file-read traceback.
        """

        raw = (self.prompt_layout.base_template or "").strip()
        if not raw:
            return Path(default_path)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.task_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"prompt_layout.base_template not found: {candidate}")
        if not candidate.is_file():
            raise ValueError(f"prompt_layout.base_template must be a file: {candidate}")
        return candidate

    def get_prompt_generation_path(self, default_path: Path) -> Path:
        """Return the task-local generation prompt template or ``default_path``.

        Generation prompts are rendered for gen>=1. If only the base template
        can be overridden, task-local prompt isolation is incomplete: bundled
        follow-up instructions can reintroduce domain-specific contracts. This
        resolver mirrors ``get_prompt_base_path`` and keeps the fallback
        explicit.
        """

        raw = (self.prompt_layout.generation_template or "").strip()
        if not raw:
            return Path(default_path)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.task_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"prompt_layout.generation_template not found: {candidate}")
        if not candidate.is_file():
            raise ValueError(f"prompt_layout.generation_template must be a file: {candidate}")
        return candidate


def _research_loop_config(raw: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    """Validate opt-in research-loop behavior before any task code executes."""
    config = raw.get("research_loop", {})
    if not isinstance(config, dict):
        raise ValueError("research_loop must be an object")
    unknown = set(config) - {"mode", "lifecycle"}
    if unknown:
        raise ValueError(f"unknown research_loop fields: {sorted(unknown)}")
    mode = config.get("mode", "metric")
    if mode not in {"metric", "scoreless"}:
        raise ValueError("research_loop.mode must be metric or scoreless")
    result = {**config, "mode": mode}
    if mode == "scoreless":
        evaluation = raw.get("evaluation") or {}
        for key in (
            "primary_metric",
            "aux_metrics",
            "anchor_metrics",
            "frontier_lanes",
            "requires_tier",
            "maturity_policy",
        ):
            if evaluation.get(key):
                raise ValueError(f"scoreless research cannot configure evaluation.{key}")
        if raw.get("baselines"):
            raise ValueError("scoreless research cannot declare scored baselines")
        if _bool_or_default(
            (raw.get("gems") or {}).get("enabled", False), False, field_name="gems.enabled"
        ):
            raise ValueError("scoreless research cannot enable Gems")
        quorum = (raw.get("synthesis_trigger") or {}).get("mature_quorum_fraction", 0)
        if quorum and float(quorum) != 0:
            raise ValueError("scoreless research cannot require scored maturity quorum")
    lifecycle = config.get("lifecycle", {})
    if not isinstance(lifecycle, dict):
        raise ValueError("research_loop.lifecycle must be an object")
    if not lifecycle:
        return result
    if mode != "scoreless":
        raise ValueError("task lifecycle callbacks require research_loop.mode=scoreless")
    unknown = set(lifecycle) - {
        "entrypoint",
        "initial_seconds",
        "finalization_seconds",
        "config",
        "after_generation",
    }
    if unknown:
        raise ValueError(f"unknown research_loop.lifecycle fields: {sorted(unknown)}")
    entrypoint = lifecycle.get("entrypoint")
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise ValueError("lifecycle.entrypoint must name a task-relative Python file:function")
    filename, function = entrypoint.rsplit(":", 1)
    relative = Path(filename)
    resolved = (task_dir / relative).resolve()
    if (
        relative.is_absolute()
        or relative.suffix != ".py"
        or not function.isidentifier()
        or not resolved.is_relative_to(task_dir.resolve())
        or not resolved.is_file()
    ):
        raise ValueError("lifecycle.entrypoint must resolve to an existing task-owned Python file")
    total_hours = (raw.get("run_lifecycle") or {}).get("max_wall_clock_hours")
    total_seconds = None
    if total_hours is not None:
        if isinstance(total_hours, bool):
            raise ValueError("lifecycle requires a finite positive run wall-clock budget")
        try:
            total_seconds = float(total_hours) * 3600
        except (TypeError, ValueError) as exc:
            raise ValueError("lifecycle requires a finite positive run wall-clock budget") from exc
        if not math.isfinite(total_seconds) or total_seconds <= 0:
            raise ValueError("lifecycle requires a finite positive run wall-clock budget")
    normalized = dict(lifecycle)
    if "after_generation" in lifecycle and not isinstance(lifecycle["after_generation"], bool):
        raise ValueError("lifecycle.after_generation must be boolean")
    for field_name in ("initial_seconds", "finalization_seconds"):
        value = lifecycle.get(field_name, 1800 if total_seconds is not None else None)
        if value is None:
            normalized[field_name] = None
            continue
        if isinstance(value, bool):
            raise ValueError(f"lifecycle.{field_name} must be a positive finite number or null")
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"lifecycle.{field_name} must be a positive finite number or null"
            ) from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"lifecycle.{field_name} must be a positive finite number or null")
        normalized[field_name] = seconds
    reserved_seconds = (normalized["initial_seconds"] or 0) + (
        normalized["finalization_seconds"] or 0
    )
    if total_seconds is not None and reserved_seconds >= total_seconds:
        raise ValueError("lifecycle phase reserves must leave time for research")
    if not isinstance(lifecycle.get("config", {}), dict):
        raise ValueError("lifecycle.config must be an object")
    result["lifecycle"] = normalized
    return result


def load_task_spec(path: str | Path) -> TaskSpec:
    """Load a TaskSpec from a YAML file."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Task spec not found: {path_obj}")

    with open(path_obj) as f:
        # R8-H4 fix: yaml.safe_load returns None for empty files; coerce
        # to {} so downstream `raw.get(...)` doesn't AttributeError.
        raw = yaml.safe_load(f) or {}

    task_dir = str(path_obj.parent)
    research_loop = _research_loop_config(raw, path_obj.parent)
    scoreless = research_loop["mode"] == "scoreless"
    execution_policy = {}
    if "execution_policy" in raw:
        from praxist.core.execution_policy import validate_task_execution_policy

        execution_policy = validate_task_execution_policy(raw["execution_policy"])

    eval_raw = raw.get("evaluation", {})
    evaluation = EvaluationSpec(
        primary_metric="" if scoreless else eval_raw.get("primary_metric", "metric_value"),
        direction=eval_raw.get("direction", "maximize"),
        aux_metrics=eval_raw.get("aux_metrics", []),
        seeds=eval_raw.get("seeds", [42, 43, 44, 45, 46]),
        aggregation=eval_raw.get("aggregation", "mean_and_std"),
        # Normalize at load time (m9 from review round 1) — list of
        # (name, direction) tuples. Empty if absent / malformed.
        anchor_metrics=_normalize_anchor_metrics(eval_raw.get("anchor_metrics", [])),
        diversity_dimensions=_normalize_diversity_dimensions(
            eval_raw.get("diversity_dimensions", [])
        ),
        requires_tier=_bool_or_default(
            eval_raw.get("requires_tier", False),
            False,
            field_name="evaluation.requires_tier",
        ),
        frontier_lanes=_normalize_frontier_lanes(eval_raw.get("frontier_lanes", [])),
        must_explore_axes=_normalize_must_explore_axes(eval_raw.get("must_explore_axes", [])),
        maturity_policy=_normalize_maturity_policy(eval_raw.get("maturity_policy", {})),
        constructive_peer_mix_enabled=_bool_or_default(
            eval_raw.get("constructive_peer_mix_enabled", True),
            True,
            field_name="evaluation.constructive_peer_mix_enabled",
        ),
        constructive_target_ratio=_float_between(
            eval_raw.get("constructive_target_ratio", 0.75),
            default=0.75,
            field_name="evaluation.constructive_target_ratio",
            minimum=0.0,
            maximum=1.0,
        ),
        launch_guard=_normalize_launch_guard(eval_raw.get("launch_guard", {})),
    )

    budget_raw = raw.get("compute_budget", {})
    scheduler_raw = budget_raw.get("resource_scheduler", {})
    if not isinstance(scheduler_raw, dict):
        raise ValueError("compute_budget.resource_scheduler must be an object")
    if scheduler_raw:
        # Resolve-only must reject malformed explicit scheduler policy before a
        # research-loop sidecar is started. The runtime parser remains the
        # single source of truth for this optional plugin-owned block.
        from praxist.plugins.workflow_stages.research_loop.backend.resource_scheduler import (
            SchedulerSettings,
        )

        SchedulerSettings.from_dict(scheduler_raw)
    compute_budget = ComputeBudget(
        per_experiment_gpu_hours=_float_or_default(
            budget_raw.get("per_experiment_gpu_hours", 0.0),
            0.0,
            field_name="compute_budget.per_experiment_gpu_hours",
        ),
        max_parallel_runs_per_peer=(
            _int_or_default(
                budget_raw.get("max_parallel_runs_per_peer"),
                1,
                field_name="compute_budget.max_parallel_runs_per_peer",
            )
            if budget_raw.get("max_parallel_runs_per_peer") is not None
            else None
        ),
        peer_gpu_memory_gb=_optional_float(
            budget_raw.get("peer_gpu_memory_gb"),
            field_name="compute_budget.peer_gpu_memory_gb",
        ),
        peer_gpu_util_pct=_optional_float(
            budget_raw.get("peer_gpu_util_pct"),
            field_name="compute_budget.peer_gpu_util_pct",
        ),
        peer_cpu_cores=(
            _int_or_default(
                budget_raw.get("peer_cpu_cores"),
                1,
                field_name="compute_budget.peer_cpu_cores",
            )
            if budget_raw.get("peer_cpu_cores") is not None
            else None
        ),
        resource_scheduler=dict(scheduler_raw),
    )

    runtime_raw = raw.get("runtime_environment", {}) or {}
    if not isinstance(runtime_raw, dict):
        logger.warning("task_spec.runtime_environment must be a dict; ignoring")
        runtime_raw = {}

    task_entrypoints_raw = raw.get("task_entrypoints", {}) or {}
    if not isinstance(task_entrypoints_raw, dict):
        logger.warning("task_spec.task_entrypoints must be a dict; ignoring")
        task_entrypoints_raw = {}
    evaluation_entrypoint_raw = task_entrypoints_raw.get("evaluation", {}) or {}
    if not isinstance(evaluation_entrypoint_raw, dict):
        logger.warning("task_spec.task_entrypoints.evaluation must be a dict; ignoring")
        evaluation_entrypoint_raw = {}
    declared_evaluation_command = evaluation_entrypoint_raw.get("command", "")
    if not isinstance(declared_evaluation_command, str):
        logger.warning("task_spec.task_entrypoints.evaluation.command must be a string; ignoring")
        declared_evaluation_command = ""

    tc_raw = raw.get("toolchain", {})
    if not isinstance(tc_raw, dict):
        logger.warning("task_spec.toolchain must be a dict; ignoring")
        tc_raw = {}
    toolchain = Toolchain(
        framework=tc_raw.get("framework", "task_defined"),
        entrypoint_template=tc_raw.get("entrypoint_template", ""),
        eval_entrypoint=tc_raw.get("eval_entrypoint", "") or declared_evaluation_command,
        benchmark_entrypoint=tc_raw.get("benchmark_entrypoint", ""),
    )
    _validate_declared_evaluation_entrypoint(
        toolchain.eval_entrypoint,
        task_dir=Path(task_dir),
        runtime_cwd=runtime_raw.get("cwd"),
    )

    baselines = []
    for b in raw.get("baselines", []):
        metric_value = b.get("metric_value", b.get("expected_acc", 0.0))
        baselines.append(
            Baseline(
                name=b.get("name", ""),
                metric_name=b.get("metric_name", evaluation.primary_metric),
                metric_value=metric_value,
                direction=b.get("direction", evaluation.direction),
                expected_acc=b.get("expected_acc", metric_value),
            )
        )

    gp_raw = raw.get("generation_policy", {})
    missing_generation_fields = [
        field_name
        for field_name in ("max_generations", "cohort_size", "per_generation_hours")
        if field_name not in gp_raw
    ]
    if missing_generation_fields and not scoreless:
        logger.warning(
            "task_spec omits calibrated generation_policy field(s) %s; using the established "
            "package defaults. Calibrate these values from the unchanged task baseline.",
            ", ".join(missing_generation_fields),
        )
    gen_policy = GenerationPolicy(
        max_generations=(
            _optional_positive_count(
                gp_raw.get("max_generations"), field_name="generation_policy.max_generations"
            )
            if scoreless
            else _int_or_default(
                gp_raw.get("max_generations", 8),
                8,
                field_name="generation_policy.max_generations",
            )
        ),
        cohort_size=_int_or_default(
            gp_raw.get("cohort_size", 5),
            5,
            field_name="generation_policy.cohort_size",
        ),
        per_generation_hours=(
            _optional_positive_limit(
                gp_raw.get("per_generation_hours"),
                field_name="generation_policy.per_generation_hours",
            )
            if scoreless
            else _float_or_default(
                gp_raw.get("per_generation_hours", 5.0),
                5.0,
                field_name="generation_policy.per_generation_hours",
            )
        ),
        promote_top_k=(
            0
            if scoreless
            else _int_or_default(
                gp_raw.get("promote_top_k", 2),
                2,
                field_name="generation_policy.promote_top_k",
            )
        ),
        promote_criterion=gp_raw.get("promote_criterion", "primary_metric"),
    )

    lifecycle_raw = raw.get("run_lifecycle", {}) or {}
    if not isinstance(lifecycle_raw, dict):
        logger.warning("task_spec.run_lifecycle must be a dict; ignoring")
        lifecycle_raw = {}
    run_lifecycle = RunLifecyclePolicy(
        max_wall_clock_hours=_optional_float(
            lifecycle_raw.get("max_wall_clock_hours"),
            field_name="run_lifecycle.max_wall_clock_hours",
        ),
        stop_signal_path=str(lifecycle_raw.get("stop_signal_path") or ""),
    )

    # v2026-05-04: synthesis_trigger config (event-driven gen termination)
    st_raw = raw.get("synthesis_trigger", {})
    missing_trigger_fields = [
        field_name
        for field_name in (
            "min_findings",
            "min_interval_minutes",
            "max_interval_minutes",
            "min_contributing_peers",
        )
        if field_name not in st_raw
    ]
    if missing_trigger_fields and not scoreless:
        logger.warning(
            "task_spec omits calibrated synthesis_trigger field(s) %s; using the established "
            "event-driven defaults. Calibrate them from observed task throughput.",
            ", ".join(missing_trigger_fields),
        )
    adaptive_raw = (
        st_raw.get("adaptive", {}) if isinstance(st_raw.get("adaptive", {}), dict) else {}
    )
    adaptive_cfg = dict(adaptive_raw)
    if "enabled" in adaptive_cfg:
        adaptive_cfg["enabled"] = _bool_or_default(
            adaptive_cfg.get("enabled", False),
            False,
            field_name="synthesis_trigger.adaptive.enabled",
        )
    synth_trigger = SynthesisTriggerConfig(
        enabled=_bool_or_default(
            st_raw.get("enabled", True),
            True,
            field_name="synthesis_trigger.enabled",
        ),
        min_findings=_int_or_default(
            st_raw.get("min_findings", 30),
            30,
            field_name="synthesis_trigger.min_findings",
        ),
        min_interval_minutes=(
            _nonnegative_interval(
                st_raw.get("min_interval_minutes"),
                field_name="synthesis_trigger.min_interval_minutes",
            )
            if scoreless
            else _float_or_default(
                st_raw.get("min_interval_minutes", 120.0),
                120.0,
                field_name="synthesis_trigger.min_interval_minutes",
            )
        ),
        max_interval_minutes=(
            _optional_positive_limit(
                st_raw.get("max_interval_minutes"),
                field_name="synthesis_trigger.max_interval_minutes",
            )
            if scoreless
            else _float_or_default(
                st_raw.get("max_interval_minutes", 240.0),
                240.0,
                field_name="synthesis_trigger.max_interval_minutes",
            )
        ),
        min_contributing_peers=_int_or_default(
            st_raw.get("min_contributing_peers", 3),
            3,
            field_name="synthesis_trigger.min_contributing_peers",
        ),
        mature_quorum_fraction=min(
            1.0,
            max(
                0.0,
                _float_or_default(
                    st_raw.get("mature_quorum_fraction", 0.0),
                    0.0,
                    field_name="synthesis_trigger.mature_quorum_fraction",
                ),
            ),
        ),
        poll_interval_seconds=_int_or_default(
            st_raw.get("poll_interval_seconds", 30),
            30,
            field_name="synthesis_trigger.poll_interval_seconds",
        ),
        adaptive=adaptive_cfg,
    )
    reachable_contributors = max(1, int(gen_policy.cohort_size))
    if synth_trigger.min_contributing_peers > reachable_contributors:
        logger.warning(
            "synthesis_trigger.min_contributing_peers=%d exceeds cohort_size=%d; "
            "clamping to %d so the event-driven trigger remains reachable.",
            synth_trigger.min_contributing_peers,
            gen_policy.cohort_size,
            reachable_contributors,
        )
        synth_trigger = dataclass_replace(
            synth_trigger, min_contributing_peers=reachable_contributors
        )

    # v2026-05-04: PI agent config
    pi_raw = raw.get("pi_agent", {})
    pi_agent = PIAgentConfig(
        enabled=_bool_or_default(
            pi_raw.get("enabled", True),
            True,
            field_name="pi_agent.enabled",
        ),
        max_runtime_minutes=(
            _optional_positive_count(
                pi_raw.get("max_runtime_minutes"), field_name="pi_agent.max_runtime_minutes"
            )
            if scoreless
            else pi_raw.get("max_runtime_minutes", 15)
        ),
        strict=_bool_or_default(
            pi_raw.get("strict", False),
            False,
            field_name="pi_agent.strict",
        ),
    )

    # v2026-05-05: Multi-PI panel config (default off → v2026-05-04 behavior)
    mp_raw = raw.get("multi_pi", {}) or {}
    if not isinstance(mp_raw, dict):
        logger.warning("task_spec.multi_pi must be a dict; ignoring")
        mp_raw = {}
    multi_pi = MultiPIConfig(
        enabled=_bool_or_default(
            mp_raw.get("enabled", False),
            False,
            field_name="multi_pi.enabled",
        ),
        panel_mode_default=str(mp_raw.get("panel_mode_default", "full")),
        auto_escalate_to_high_stakes=_bool_or_default(
            mp_raw.get("auto_escalate_to_high_stakes", True),
            True,
            field_name="multi_pi.auto_escalate_to_high_stakes",
        ),
        pi_max_runtime_minutes=(
            _optional_positive_count(
                mp_raw.get("pi_max_runtime_minutes"), field_name="multi_pi.pi_max_runtime_minutes"
            )
            if scoreless
            else int(mp_raw.get("pi_max_runtime_minutes", 12))
        ),
        chair_max_runtime_minutes=(
            _optional_positive_count(
                mp_raw.get("chair_max_runtime_minutes"),
                field_name="multi_pi.chair_max_runtime_minutes",
            )
            if scoreless
            else int(mp_raw.get("chair_max_runtime_minutes", 8))
        ),
        chair_peer_budget=int(mp_raw.get("chair_peer_budget", 5)),
        shared_core_ratio_target=float(mp_raw.get("shared_core_ratio_target", 0.65)),
        private_kb_ratio_target=float(mp_raw.get("private_kb_ratio_target", 0.35)),
        fallback_to_single_pi_on_panel_failure=_bool_or_default(
            mp_raw.get("fallback_to_single_pi_on_panel_failure", True),
            True,
            field_name="multi_pi.fallback_to_single_pi_on_panel_failure",
        ),
        n_rounds=int(mp_raw.get("n_rounds", 2)),
        round2_max_runtime_minutes=(
            _optional_positive_count(
                mp_raw.get("round2_max_runtime_minutes"),
                field_name="multi_pi.round2_max_runtime_minutes",
            )
            if scoreless
            else int(mp_raw.get("round2_max_runtime_minutes", 6))
        ),
    )
    if multi_pi.n_rounds not in (1, 2):
        logger.warning(
            "multi_pi.n_rounds=%d is not in {1, 2}; clamping to 2",
            multi_pi.n_rounds,
        )
        multi_pi.n_rounds = 2

    rm_raw = raw.get("research_memory", {}) or {}
    if not isinstance(rm_raw, dict):
        logger.warning("task_spec.research_memory must be a dict; ignoring")
        rm_raw = {}
    research_memory = ResearchMemoryConfig(
        enabled=_bool_or_default(
            rm_raw.get("enabled", False),
            False,
            field_name="research_memory.enabled",
        ),
        rollout_phase=int(rm_raw.get("rollout_phase", 0)),
        evidence_pack_max_cards=int(rm_raw.get("evidence_pack_max_cards", 40)),
        citation_coverage_min=float(rm_raw.get("citation_coverage_min", 0.95)),
        negative_evidence_ratio_min=float(rm_raw.get("negative_evidence_ratio_min", 0.20)),
        bridge_coverage_check_required=_bool_or_default(
            rm_raw.get("bridge_coverage_check_required", True),
            True,
            field_name="research_memory.bridge_coverage_check_required",
        ),
    )

    prompt_layout_raw = raw.get("prompt_layout", {}) or {}
    if not isinstance(prompt_layout_raw, dict):
        logger.warning("task_spec.prompt_layout must be a dict; ignoring")
        prompt_layout_raw = {}
    prompt_layout = PromptLayoutConfig(
        base_template=str(prompt_layout_raw.get("base_template") or ""),
        generation_template=str(prompt_layout_raw.get("generation_template") or ""),
    )

    if multi_pi.enabled and not research_memory.enabled:
        logger.warning(
            "task_spec: multi_pi.enabled=True but research_memory.enabled=False. "
            "Multi-PI requires the evidence pack pipeline. Auto-enabling "
            "research_memory at rollout_phase=max(2, current). User-provided "
            "fields (evidence_pack_max_cards, *_min, bridge_coverage_check_required) "
            "are preserved verbatim. (R1#6 fix.)"
        )
        # Preserve all user-provided fields by rebuilding from the existing
        # instance + only overriding `enabled` and `rollout_phase`. This
        # avoids silently overriding e.g. evidence_pack_max_cards=50.
        from dataclasses import replace

        research_memory = replace(
            research_memory,
            enabled=True,
            rollout_phase=max(2, research_memory.rollout_phase),
        )

    if multi_pi.panel_mode_default not in ("mini", "full", "high_stakes"):
        logger.warning(
            "multi_pi.panel_mode_default=%r is not in {mini,full,high_stakes}; using 'full'",
            multi_pi.panel_mode_default,
        )
        multi_pi.panel_mode_default = "full"

    # R3#4 fix: chair_peer_budget MUST equal generation_policy.cohort_size,
    # otherwise the chair prompt asks for N peers but the validator expects
    # M, producing immediate validation failure.
    # R5#7 fix: use dataclasses.replace so this code keeps working if
    # MultiPIConfig is later marked frozen=True.
    if multi_pi.enabled and multi_pi.chair_peer_budget != gen_policy.cohort_size:
        logger.warning(
            "task_spec: multi_pi.chair_peer_budget=%d != "
            "generation_policy.cohort_size=%d. Forcing chair_peer_budget to "
            "match cohort_size to avoid agenda validation mismatches. (R3#4 fix.)",
            multi_pi.chair_peer_budget,
            gen_policy.cohort_size,
        )
        from dataclasses import replace as _dc_replace

        multi_pi = _dc_replace(multi_pi, chair_peer_budget=gen_policy.cohort_size)

    # Sanity: synthesis_trigger.max_interval_minutes should usually leave
    # 30 min slack before per_generation_hours*60. If it exceeds the peer
    # safety cap, peers can time out before the orchestrator's synthesis
    # cap fires, creating mixed termination semantics. Auto-clamp that
    # case; equality is allowed for tasks that intentionally align both caps.
    safety_cap_min = (
        gen_policy.per_generation_hours * 60
        if gen_policy.per_generation_hours is not None
        else None
    )
    adaptive_max = 0.0
    adaptive_enabled = bool(
        isinstance(synth_trigger.adaptive, dict)
        and _bool_or_default(
            synth_trigger.adaptive.get("enabled", False),
            False,
            field_name="synthesis_trigger.adaptive.enabled",
        )
    )
    if adaptive_enabled:
        try:
            adaptive_max = float(synth_trigger.adaptive.get("max_interval_ceiling_minutes") or 0)
        except (TypeError, ValueError):
            adaptive_max = 0.0
        if not math.isfinite(adaptive_max):
            adaptive_max = 0.0
    synthesis_limits = [
        value
        for value in (synth_trigger.max_interval_minutes, adaptive_max)
        if value is not None and value > 0
    ]
    effective_synthesis_max = max(synthesis_limits) if synthesis_limits else None
    min_usable_cap = 15  # below this, misconfiguration is too severe to auto-correct
    if (
        safety_cap_min is not None
        and effective_synthesis_max is not None
        and effective_synthesis_max > safety_cap_min
    ):
        if safety_cap_min < min_usable_cap:
            raise ValueError(
                f"per_generation_hours={gen_policy.per_generation_hours}h "
                f"({safety_cap_min} min) is too small for synthesis_trigger "
                f"({effective_synthesis_max:.1f} min). "
                f"Raise per_generation_hours to at least "
                f"{(int(effective_synthesis_max) + 30) // 60 + 1}h."
            )
        # Keep the long-standing absolute finalization backstop. Task-specific
        # intervals remain configurable below it; an absurd value must not turn
        # a stalled generation into a multi-day wait.
        clamped = max(min_usable_cap, min(safety_cap_min - 30, 240))
        logger.warning(
            "effective synthesis max interval=%.1f exceeds the "
            "per_generation_hours=%dh safety cap (%d min). Clamping to "
            "%d min to keep generation finalization bounded. "
            "Raise synthesis_trigger.max_interval_minutes within the peer safety "
            "cap if a longer generation interval is intentional.",
            effective_synthesis_max,
            gen_policy.per_generation_hours,
            safety_cap_min,
            clamped,
        )
        updates = {**synth_trigger.__dict__}
        if synth_trigger.max_interval_minutes is not None:
            updates["max_interval_minutes"] = min(int(synth_trigger.max_interval_minutes), clamped)
        if adaptive_enabled and isinstance(updates.get("adaptive"), dict):
            adaptive_cfg = dict(updates["adaptive"])
            if adaptive_max > clamped:
                adaptive_cfg["max_interval_ceiling_minutes"] = clamped
            updates["adaptive"] = adaptive_cfg
        synth_trigger = type(synth_trigger)(**updates)
    elif (
        safety_cap_min is not None
        and effective_synthesis_max is not None
        and effective_synthesis_max + 30 > safety_cap_min
    ):
        # Slack < 30 min: warn but don't clamp (still within tolerance)
        logger.warning(
            "effective synthesis max interval=%.1f minutes is too close to "
            "per_generation_hours=%dh (=%d min). Peer-level timeout could "
            "fire close to the orchestrator's synthesis trigger, leaving "
            "the run with tight termination semantics. "
            "Consider raising per_generation_hours by ≥30min.",
            effective_synthesis_max,
            gen_policy.per_generation_hours,
            safety_cap_min,
        )

    raw_launch_guard = eval_raw.get("launch_guard", {})
    _validate_declared_evaluation_horizon(
        evaluation,
        gen_policy,
        synth_trigger,
        close_grade_estimate_declared=(
            isinstance(raw_launch_guard, dict)
            and "estimated_close_grade_eval_minutes" in raw_launch_guard
        ),
    )

    if gen_policy.max_generations is None:
        logger.info("task_spec has no configured generation-count limit")
    else:
        logger.info(
            "task_spec resolved max_generations=%d; the run will execute at most "
            "%d generation(s) with %d PI synthesis event(s) between them.",
            gen_policy.max_generations,
            gen_policy.max_generations,
            max(0, gen_policy.max_generations - 1),
        )

    # Tiered eval block (optional, for prompt rendering only).
    tiered_raw = raw.get("tiered_eval", {})
    tiered_eval: dict[str, dict[str, Any]] = {}
    if isinstance(tiered_raw, dict):
        for tname, tcfg in tiered_raw.items():
            if isinstance(tcfg, dict):
                tiered_eval[str(tname)] = dict(tcfg)
    maturity_raw = eval_raw.get("maturity_policy", {})
    maturity_raw = maturity_raw if isinstance(maturity_raw, dict) else {}
    if (
        tiered_eval
        and not evaluation.maturity_policy.get("require_ratio_gate")
        and not maturity_raw.get("complete_stage_labels")
        and not maturity_raw.get("preliminary_stage_labels")
    ):
        # Declared stage names are opaque. Treating every tier as complete can
        # promote a preliminary result, while guessing that the final mapping
        # entry is terminal makes YAML order a scientific contract. Preserve
        # the evidence and require producer completion facts or task policy for
        # durable maturity instead.
        logger.warning(
            "task_spec evaluation.maturity_policy omits stage labels for tiered stages %s; "
            "stage names remain advisory until the evaluator emits generic completion facts "
            "or the task declares complete_stage_labels/preliminary_stage_labels.",
            ", ".join(tiered_eval),
        )

    # Run-wide agent reasoning policy. Explicit reasoning_effort wins over the
    # legacy premium_mode compatibility switch at the runtime boundary.
    agent_raw = raw.get("agent", {})
    if not isinstance(agent_raw, dict):
        logger.warning("task_spec.agent must be a dict; using defaults")
        agent_raw = {}
    agent_cfg = AgentConfig(
        premium_mode=_bool_or_default(
            agent_raw.get("premium_mode", False),
            False,
            field_name="agent.premium_mode",
        ),
        reasoning_effort=_reasoning_effort_or_default(agent_raw.get("reasoning_effort", "max")),
    )

    # Pull the panel topology ref off the task descriptor — same field the
    # plugin resolver walks at ``praxist_plugins.panel.topology``. We
    # only honor strings (not dicts/lists) and fall back to the legacy
    # default when the field is absent or malformed; the rocket_8e_panel
    # bug it fixes is exactly the silent drop to the legacy three-PI set.
    praxist_plugins_raw = raw.get("praxist_plugins") or {}
    panel_raw = (
        praxist_plugins_raw.get("panel", {}) if isinstance(praxist_plugins_raw, dict) else {}
    ) or {}
    panel_topology_ref = panel_raw.get("topology") if isinstance(panel_raw, dict) else None
    if not isinstance(panel_topology_ref, str) or not panel_topology_ref.strip():
        panel_topology_ref = "panel_topology:legacy_multi_pi_two_round"

    gems_raw = raw.get("gems", {}) or {}
    if not isinstance(gems_raw, dict):
        logger.warning("task_spec.gems must be a dict; ignoring")
        gems_raw = {}
    gems_raw, migrated_gems_fields = migrate_legacy_gems_config(gems_raw)
    for field_name in migrated_gems_fields:
        logger.warning(
            "task_spec.gems.%s is deprecated and was migrated to the current "
            "task-agnostic Gems contract",
            field_name,
        )

    def _int_at_least(name: str, default: int, minimum: int) -> int:
        try:
            value = int(gems_raw.get(name, default))
        except (TypeError, ValueError):
            logger.warning("task_spec.gems.%s must be an int; using %d", name, default)
            value = default
        return max(minimum, value)

    def _bool_value(section: dict[str, Any], name: str, default: bool) -> bool:
        value = section.get(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "on", "y"}:
                return True
            if token in {"0", "false", "no", "off", "n"}:
                return False
        logger.warning(
            "task_spec boolean %s=%r is invalid; using %s",
            name,
            value,
            default,
        )
        return bool(default)

    gems_enabled = _bool_value(gems_raw, "enabled", False)
    raw_selection_policy = str(gems_raw.get("selection_policy") or "frontier_lane_balanced")
    selection_policy = normalize_gems_selection_policy(raw_selection_policy)
    mature_evidence_policy = selection_policy == GEMS_MATURE_EVIDENCE_TOP_K
    if (
        gems_enabled
        and mature_evidence_policy
        and not (
            "min_mature_eval_units" in gems_raw
            or isinstance(gems_raw.get("evidence_stage_min_units"), dict)
        )
    ):
        logger.warning(
            "task_spec.gems.selection_policy=mature_evidence_top_k requires explicit "
            "gems.min_mature_eval_units or gems.evidence_stage_min_units for strict "
            "maturity-gated tasks; using generic default min_mature_eval_units=1."
        )

    bottleneck_detector_mode = str(gems_raw.get("bottleneck_detector_mode") or "generic")
    if gems_enabled and bottleneck_detector_mode.strip().lower() not in {
        "generic",
        "disabled",
        "off",
        "none",
        "false",
        "0",
    }:
        logger.warning(
            "task_spec.gems.bottleneck_detector_mode must be generic or disabled; "
            "task-specific detectors must be implemented outside Praxist core. "
            "Falling back to generic."
        )
        bottleneck_detector_mode = "generic"

    gems_cfg = GemsConfig(
        enabled=gems_enabled,
        reset_interval_generations=_int_at_least("reset_interval_generations", 6, 1),
        max_resets=_int_at_least("max_resets", 3, 0),
        max_gems_per_reset=_int_at_least("max_gems_per_reset", 4, 1),
        max_gems_total=_int_at_least("max_gems_total", 4, 1),
        max_gems_per_family=_int_at_least("max_gems_per_family", 2, 1),
        min_frontier_entries=_int_at_least("min_frontier_entries", 1, 1),
        archive_ordinary_findings=_bool_value(gems_raw, "archive_ordinary_findings", True),
        signature_top_k=_int_at_least("signature_top_k", 16, 1),
        signature_entries_per_lane=_int_at_least("signature_entries_per_lane", 8, 1),
        prompt_max_gems=_int_at_least("prompt_max_gems", 4, 1),
        include_lanes=_normalize_str_list(gems_raw.get("include_lanes")),
        selection_policy=selection_policy,
        min_mature_eval_units=_int_at_least("min_mature_eval_units", 1, 1),
        evidence_stage_min_units=_normalize_str_int_map(
            gems_raw.get("evidence_stage_min_units"),
            field_name="gems.evidence_stage_min_units",
        ),
        primary_metric_keys=_normalize_str_list(gems_raw.get("primary_metric_keys")),
        secondary_metric_keys=_normalize_str_list(gems_raw.get("secondary_metric_keys")),
        lower_tail_metric_keys=_normalize_str_list(gems_raw.get("lower_tail_metric_keys")),
        validation_metric_keys=_normalize_str_list(gems_raw.get("validation_metric_keys")),
        cost_metric_keys=_normalize_str_list(gems_raw.get("cost_metric_keys")),
        result_cell_metric_derivations=_normalize_result_cell_metric_derivations(
            gems_raw.get("result_cell_metric_derivations")
        ),
        result_metric_aliases=_normalize_str_str_map(
            gems_raw.get("result_metric_aliases"),
            field_name="gems.result_metric_aliases",
        ),
        gem_seeded_independent_peers=_int_at_least("gem_seeded_independent_peers", 0, 0),
        performance_lanes=_normalize_str_list(gems_raw.get("performance_lanes")),
        control_lanes=_normalize_str_list(gems_raw.get("control_lanes")),
        bottleneck_detector_mode=bottleneck_detector_mode,
        result_artifact_materialization=_bool_value(
            gems_raw, "result_artifact_materialization", True
        ),
        result_artifact_default_lane=str(
            gems_raw.get("result_artifact_default_lane") or "performance"
        ),
        result_artifact_default_family=str(
            gems_raw.get("result_artifact_default_family") or "task_candidate"
        ),
    )

    runtime_environment = RuntimeEnvironmentConfig(
        protected_child_paths=_normalize_str_list(runtime_raw.get("protected_child_paths")),
        data_env_aliases=_normalize_str_list(
            runtime_raw.get("data_env_aliases") or runtime_raw.get("data_dir_env_aliases")
        ),
    )

    return TaskSpec(
        task_id=raw.get("task_id", ""),
        task_name=raw.get("task_name", ""),
        description_file=raw.get("description_file", ""),
        research_direction=raw.get("research_direction", ""),
        evaluation=evaluation,
        compute_budget=compute_budget,
        toolchain=toolchain,
        baselines=baselines,
        generation_policy=gen_policy,
        run_lifecycle=run_lifecycle,
        research_loop=research_loop,
        execution_policy=execution_policy,
        synthesis_trigger=synth_trigger,
        pi_agent=pi_agent,
        multi_pi=multi_pi,
        agent=agent_cfg,
        research_memory=research_memory,
        prompt_layout=prompt_layout,
        gems=gems_cfg,
        runtime_environment=runtime_environment,
        tiered_eval=tiered_eval,
        panel_topology_ref=panel_topology_ref,
        _raw=raw,
        _task_dir=task_dir,
    )
