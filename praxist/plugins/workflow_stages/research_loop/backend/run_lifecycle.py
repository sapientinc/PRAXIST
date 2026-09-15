"""Generic run-level stop gates for the research-loop stage."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RUN_STOP_REPORT_SCHEMA_VERSION = "praxist.run_stop_report.v1"
RUN_STOP_SIGNAL_SCHEMA_VERSION = "praxist.run_stop_signal.v1"


@dataclass(frozen=True)
class RunStopDecision:
    """Decision produced before starting a generation."""

    should_stop: bool
    exit_condition: str
    reason: str
    next_generation: int
    generations_completed: int
    elapsed_seconds: float
    run_dir: str
    source: str
    max_wall_clock_seconds: float | None = None
    stop_signal_path: str | None = None
    signal_evidence: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    schema_version: str = RUN_STOP_REPORT_SCHEMA_VERSION

    def to_report(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_seconds"] = round(float(self.elapsed_seconds), 3)
        if self.max_wall_clock_seconds is not None:
            payload["max_wall_clock_seconds"] = round(float(self.max_wall_clock_seconds), 3)
        payload["warnings"] = list(self.warnings)
        return payload


def evaluate_run_stop_gate(
    *,
    task_spec: Any,
    run_dir: Path,
    run_started_at_seconds: float,
    next_generation: int,
    generations_completed: int,
    now_seconds: float | None = None,
) -> RunStopDecision:
    """Evaluate generic run-level stop conditions before a generation starts."""
    run_dir = Path(run_dir)
    now = time.time() if now_seconds is None else float(now_seconds)
    elapsed = max(0.0, now - float(run_started_at_seconds))
    lifecycle = getattr(task_spec, "run_lifecycle", None)

    signal_path = _resolve_stop_signal_path(
        run_dir,
        getattr(lifecycle, "stop_signal_path", "") if lifecycle is not None else "",
    )
    for candidate_signal_path in _candidate_stop_signal_paths(run_dir, signal_path):
        if candidate_signal_path.exists():
            evidence, warnings = _read_stop_signal_evidence(candidate_signal_path)
            return RunStopDecision(
                should_stop=True,
                exit_condition="external_stop_signal",
                reason=str(evidence.get("reason") or "external_stop_signal"),
                next_generation=next_generation,
                generations_completed=generations_completed,
                elapsed_seconds=elapsed,
                run_dir=str(run_dir),
                source=str(evidence.get("source") or "external_stop_signal"),
                stop_signal_path=str(candidate_signal_path),
                signal_evidence=evidence,
                warnings=warnings,
            )

    max_hours = getattr(lifecycle, "max_wall_clock_hours", None) if lifecycle is not None else None
    max_seconds = None if max_hours is None else float(max_hours) * 3600.0
    if max_seconds is not None and elapsed >= max_seconds:
        return RunStopDecision(
            should_stop=True,
            exit_condition="wall_clock_limit",
            reason="wall_clock_limit",
            next_generation=next_generation,
            generations_completed=generations_completed,
            elapsed_seconds=elapsed,
            run_dir=str(run_dir),
            source="run_lifecycle",
            max_wall_clock_seconds=max_seconds,
            stop_signal_path=str(signal_path) if signal_path is not None else None,
        )

    return RunStopDecision(
        should_stop=False,
        exit_condition="",
        reason="continue",
        next_generation=next_generation,
        generations_completed=generations_completed,
        elapsed_seconds=elapsed,
        run_dir=str(run_dir),
        source="run_lifecycle",
        max_wall_clock_seconds=max_seconds,
        stop_signal_path=str(signal_path) if signal_path is not None else None,
    )


def max_generations_stop_report(
    *,
    run_dir: Path,
    max_generations: int,
    generations_completed: int,
    run_started_at_seconds: float,
    now_seconds: float | None = None,
) -> RunStopDecision:
    """Build the terminal stop report for an exhausted generation budget."""

    now = time.time() if now_seconds is None else float(now_seconds)
    elapsed = max(0.0, now - float(run_started_at_seconds))
    return RunStopDecision(
        should_stop=True,
        exit_condition="max_generations",
        reason="max_generations",
        next_generation=max_generations,
        generations_completed=generations_completed,
        elapsed_seconds=elapsed,
        run_dir=str(Path(run_dir)),
        source="generation_policy",
    )


def write_run_stop_report(run_dir: Path, decision: RunStopDecision) -> dict[str, Any]:
    """Persist a replayable stop report and return its JSON payload."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = decision.to_report()
    path = run_dir / "run_stop_report.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
    return payload


def write_external_stop_signal(
    run_dir: Path,
    payload: dict[str, Any],
    *,
    stop_signal_path: str = "run_control/stop.json",
) -> dict[str, Any]:
    """Persist a task-produced generic stop signal through Praxist control-plane code."""

    run_dir = Path(run_dir)
    signal_path = _resolve_stop_signal_path(run_dir, stop_signal_path)
    if signal_path is None:
        raise ValueError("stop_signal_path must not be empty")
    resolved_run_dir = run_dir.resolve(strict=False)
    resolved_signal = signal_path.resolve(strict=False)
    try:
        resolved_signal.relative_to(resolved_run_dir)
    except ValueError as exc:
        raise ValueError("external stop signal must be written under run_dir") from exc

    data = dict(payload)
    data.setdefault("schema_version", RUN_STOP_SIGNAL_SCHEMA_VERSION)
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = signal_path.with_name(f"{signal_path.name}.{time.time_ns()}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(signal_path)
    return data


def _resolve_stop_signal_path(run_dir: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    run_root = Path(run_dir)
    resolved_run_root = run_root.resolve(strict=False)
    if path.is_absolute():
        candidate = path
    else:
        candidate = run_root / path
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_run_root)
    except ValueError:
        return None
    if _has_symlink_component(resolved_run_root, resolved_candidate):
        return None
    return candidate


def _candidate_stop_signal_paths(run_dir: Path, configured: Path | None) -> list[Path]:
    paths: list[Path] = []
    if configured is not None:
        paths.append(configured)
    default = _resolve_stop_signal_path(Path(run_dir), "ORCHESTRATOR_SHUTDOWN")
    if default is not None:
        paths.append(default)
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def _has_symlink_component(run_root: Path, candidate: Path) -> bool:
    """Return True when an existing run-local component is a symlink."""

    try:
        relative = candidate.relative_to(run_root)
    except ValueError:
        return True
    cursor = run_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            try:
                if cursor.is_symlink():
                    return True
            except OSError:
                return True
    return False


def _read_stop_signal_evidence(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return (
            {
                "reason": "external_stop_signal",
                "source": "external_stop_signal",
                "read_error": str(exc),
            },
            (f"could not read stop signal: {exc}",),
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        kv_data = _parse_key_value_stop_signal(text)
        if kv_data:
            return kv_data, ()
        return (
            {
                "reason": "external_stop_signal",
                "source": "external_stop_signal",
                "parse_error": str(exc),
                "raw_preview": text[:500],
            },
            (f"stop signal is not valid JSON: {exc}",),
        )
    if not isinstance(data, dict):
        return (
            {
                "reason": "external_stop_signal",
                "source": "external_stop_signal",
                "raw_type": type(data).__name__,
            },
            ("stop signal JSON must be an object",),
        )
    return dict(data), ()


def _parse_key_value_stop_signal(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        data[key] = value.strip()
    if not data:
        return {}
    if "reason" not in data:
        data["reason"] = "orchestrator_shutdown"
    if "source" not in data:
        data["source"] = "orchestrator_shutdown"
    return data
