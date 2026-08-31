"""
Orchestrator status writer.

Problem this solves:
    `frontier/` empty, no `gen0_complete.json`, no log entry explaining why
    Gen 1 hasn't launched. Operators have no externalized view of orchestrator
    state.

Design:
  - One file `<run>/orchestrator_status.json` is rewritten every N seconds
    (default 300 = 5 min) while the run is alive.
  - On exit, a sibling `<run>/orchestrator_status.final.json` is written with
    `exit_condition` ∈ {completed, plateau, error, in_progress_snapshot}.
  - Content is intentionally small and JSON-serializable so any tool or
    tail -f equivalent can consume it.
  - The writer runs in a daemon thread — it does not affect orchestrator
    shutdown semantics. On `stop()` the daemon exits cleanly and writes the
    final snapshot.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .tools.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


DEFAULT_POLL_SECONDS = 300
CGROUP_MEMORY_WARNING_RATIO = 0.80
CGROUP_MEMORY_SHUTDOWN_RATIO = 0.92
CGROUP_ROOT = Path("/sys/fs/cgroup")


def read_effective_orchestrator_status(run_dir: Path) -> dict[str, Any]:
    """Return the authoritative readable orchestrator snapshot for ``run_dir``.

    A final snapshot normally supersedes the periodic snapshot left behind by
    the writer. A resumed run is the exception: its newer, explicitly
    in-progress periodic snapshot belongs to the active segment and therefore
    supersedes an older final snapshot from the previous segment.
    """

    run_dir = Path(run_dir)
    periodic_path = run_dir / "orchestrator_status.json"
    final_path = run_dir / "orchestrator_status.final.json"
    periodic = _read_status_object(periodic_path)
    final = _read_status_object(final_path)
    if not final:
        return periodic
    if not periodic:
        return final
    if _is_explicitly_in_progress(periodic) and _mtime_ns(periodic_path) > _mtime_ns(final_path):
        return periodic
    return final


def _read_status_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_explicitly_in_progress(payload: dict[str, Any]) -> bool:
    return str(payload.get("exit_condition") or payload.get("status") or "").strip().lower() in {
        "in_progress",
        "running",
    }


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


@dataclass
class OrchestratorSnapshot:
    """Serialized orchestrator state at a single moment."""

    run_started_at: str
    updated_at: str
    run_dir: str
    task_id: str
    task_name: str
    current_generation: int
    max_generations: int | None
    cohort_size: int
    strategy: str
    generations_completed: int
    current_session_per_peer: dict[str, str] = field(default_factory=dict)
    variants_total: int = 0
    variants_above_baseline: int = 0
    variants_validated_multi_seed: int = 0
    findings_total: int = 0
    frontier_candidates: int = 0
    best_mature_result: dict[str, Any] = field(default_factory=dict)
    best_validation_signal: dict[str, Any] = field(default_factory=dict)
    operator_manifest_paths: dict[str, str] = field(default_factory=dict)
    gems_cycle_index: int = 0
    gems_reset_count: int = 0
    gems_count: int = 0
    gems_refs: list[dict[str, Any]] = field(default_factory=list)
    logical_generation: int = 0
    gen_promotion_criteria: str = ""
    gen_promotion_blocker: str = ""
    last_stop_audit: dict[str, Any] = field(default_factory=dict)
    last_peer_mix: dict[str, Any] = field(default_factory=dict)
    mature_quorum_required: int = 0
    resource_scheduler: dict[str, Any] = field(default_factory=dict)
    wall_clock_elapsed_seconds: float = 0.0
    exit_condition: str = "in_progress"  # overwritten on final snapshot

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SnapshotFn = Callable[[], OrchestratorSnapshot]


class OrchestratorStatusWriter:
    """Background thread that periodically writes orchestrator status JSON.

    Usage:
        writer = OrchestratorStatusWriter(run_dir, snapshot_fn, interval=300)
        writer.start()
        ...
        writer.stop(exit_condition="completed")
    """

    def __init__(
        self,
        run_dir: Path,
        snapshot_fn: SnapshotFn,
        *,
        interval_seconds: float = DEFAULT_POLL_SECONDS,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_fn = snapshot_fn
        # Guard against interval_seconds <= 0: if set to 0 or negative, the
        # daemon thread's _stop_event.wait(timeout=<=0) returns immediately
        # and the loop spins at CPU speed. Clamp to a sensible minimum.
        iv = float(interval_seconds)
        if iv <= 0:
            raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")
        self.interval_seconds = iv
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        # stop() may be called more than once (e.g. from both an inner finally
        # and an outer wrapper). Idempotency avoids writing the final file
        # twice — possibly overwriting a clean snapshot with a second
        # fallback if snapshot_fn has become permanently broken.
        self._stopped = False
        self._stop_lock = threading.Lock()
        logger.info("orchestrator status manifests: %s", self.operator_manifest_paths())

    @property
    def status_path(self) -> Path:
        return self.run_dir / "orchestrator_status.json"

    @property
    def final_status_path(self) -> Path:
        return self.run_dir / "orchestrator_status.final.json"

    def operator_manifest_paths(self) -> dict[str, str]:
        """Return the related run surfaces operators commonly inspect together."""
        return operator_manifest_paths(self.run_dir)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="orchestrator-status-writer",
            daemon=True,
        )
        self._thread.start()
        # Write an immediate snapshot so operators see status right away.
        self._write_once("in_progress")

    def stop(self, *, exit_condition: str = "in_progress_snapshot") -> None:
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        # Final snapshot. If snapshot_fn raises, fall back to a skeletal
        # snapshot so operators still get *something* — knowing "the run
        # stopped at ${exit_condition}" is itself valuable.
        try:
            snap = self.snapshot_fn()
        except Exception as e:
            logger.warning(
                "orchestrator status final snapshot failed, writing skeletal fallback: %s",
                e,
            )
            snap = OrchestratorSnapshot(
                run_started_at="",
                updated_at="",
                run_dir=str(self.run_dir),
                task_id="",
                task_name="",
                current_generation=-1,
                max_generations=-1,
                cohort_size=-1,
                strategy="",
                generations_completed=-1,
            )
        snap.exit_condition = exit_condition
        snap.updated_at = datetime.now(UTC).isoformat()
        if not snap.operator_manifest_paths:
            snap.operator_manifest_paths = self.operator_manifest_paths()
        try:
            atomic_write_json(self.final_status_path, snap.to_dict())
        except OSError as e:
            logger.warning("orchestrator final status write failed: %s", e)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._write_once("in_progress")
            except Exception as e:  # pragma: no cover — defensive
                self._last_error = str(e)
                logger.debug("orchestrator status write failed: %s", e)
            self._stop_event.wait(timeout=self.interval_seconds)

    def _write_once(self, exit_condition: str) -> None:
        # If snapshot_fn fails, still update the status file — otherwise
        # operators see `updated_at` frozen at the last-successful moment
        # and cannot distinguish "orchestrator crashed" from "snapshot_fn
        # has been broken for hours". Write a degraded snapshot containing
        # only the bits we can produce without snapshot_fn.
        snap: OrchestratorSnapshot | None = None
        snap_error: str | None = None
        try:
            snap = self.snapshot_fn()
        except Exception as e:
            snap_error = f"{type(e).__name__}: {e}"
            logger.debug("orchestrator snapshot_fn failed: %s", e)
            snap = OrchestratorSnapshot(
                run_started_at="",
                updated_at="",
                run_dir=str(self.run_dir),
                task_id="",
                task_name="",
                current_generation=-1,
                max_generations=-1,
                cohort_size=-1,
                strategy="",
                generations_completed=-1,
            )
        snap.exit_condition = exit_condition
        snap.updated_at = datetime.now(UTC).isoformat()
        if not snap.operator_manifest_paths:
            snap.operator_manifest_paths = self.operator_manifest_paths()
        payload = snap.to_dict()
        if snap_error:
            payload["last_snapshot_error"] = snap_error
        try:
            atomic_write_json(self.status_path, payload)
        except OSError as e:
            logger.debug("orchestrator status atomic_write_json failed: %s", e)
        self._check_cgroup_memory_pressure()

    def _check_cgroup_memory_pressure(self) -> None:
        ratio = cgroup_memory_ratio()
        if ratio is None:
            return
        if ratio >= CGROUP_MEMORY_SHUTDOWN_RATIO:
            shutdown_path = self.run_dir / "ORCHESTRATOR_SHUTDOWN"
            if shutdown_path.exists():
                return
            logger.error(
                "cgroup memory pressure %.1f%% exceeds %.1f%%; touching %s",
                100 * ratio,
                100 * CGROUP_MEMORY_SHUTDOWN_RATIO,
                shutdown_path,
            )
            try:
                shutdown_path.write_text(
                    "reason=cgroup_memory_pressure\n"
                    f"ratio={ratio:.3f}\n"
                    f"threshold={CGROUP_MEMORY_SHUTDOWN_RATIO:.3f}\n"
                    f"at={datetime.now(UTC).isoformat()}\n",
                    encoding="utf-8",
                )
            except OSError as e:
                logger.warning("orchestrator shutdown sentinel write failed: %s", e)
        elif ratio >= CGROUP_MEMORY_WARNING_RATIO:
            logger.warning(
                "cgroup memory usage %.1f%% exceeds %.1f%%",
                100 * ratio,
                100 * CGROUP_MEMORY_WARNING_RATIO,
            )


# ---------------------------------------------------------------------------
# Gen-promotion criteria helper — extracted so tests can exercise it.
# ---------------------------------------------------------------------------


def describe_promotion_criteria(
    promote_top_k: int,
    promote_criterion: str,
    primary_metric: str,
    direction: str,
    frontier_lanes: list[dict] | None = None,
) -> str:
    """Render a one-line human-readable string of the gen-promotion rule.

    Surfacing this in the status file makes gen-gating visible.
    """
    arrow = "↑" if direction == "maximize" else "↓"
    if frontier_lanes:
        lane_bits = []
        for lane in frontier_lanes:
            name = str(lane.get("name") or "").strip()
            if not name:
                continue
            try:
                k = int(lane.get("k", 1) or 0)
            except (TypeError, ValueError):
                k = 1
            lane_bits.append(f"{name}(k={k})")
        lanes_text = ", ".join(lane_bits) if lane_bits else "configured lanes"
        return (
            "lane-based frontier promotion "
            f"[{lanes_text}] using {primary_metric} {arrow} plus lane-specific axes"
        )
    return (
        f"promote top-{promote_top_k} findings per generation by "
        f"{promote_criterion} ({primary_metric} {arrow})"
    )


def cgroup_memory_ratio(cgroup_root: Path = CGROUP_ROOT) -> float | None:
    """Return current cgroup memory usage divided by limit when bounded."""
    root = Path(cgroup_root)
    for current_path, max_path in (
        (root / "memory.current", root / "memory.max"),
        (
            root / "memory" / "memory.usage_in_bytes",
            root / "memory" / "memory.limit_in_bytes",
        ),
    ):
        try:
            current = int(current_path.read_text(encoding="utf-8").strip())
            max_raw = max_path.read_text(encoding="utf-8").strip()
            if max_raw == "max":
                return None
            maximum = int(max_raw)
            if maximum <= 0:
                continue
            return current / maximum
        except (FileNotFoundError, OSError, ValueError):
            continue
    return None


def describe_promotion_blocker(
    *,
    variants_with_primary_metric: int,
    variants_above_baseline: int,
    promote_top_k: int,
    lane_based: bool = False,
) -> str:
    """Heuristic free-text blocker description based on current counts.

    Returns empty string if nothing is blocking promotion. Otherwise returns
    a single sentence explaining why.
    """
    if variants_with_primary_metric == 0:
        return "no variants have reported the primary metric yet"
    if variants_above_baseline == 0:
        if lane_based:
            return (
                f"{variants_with_primary_metric} variant(s) reported; none above baseline yet. "
                "Lane-based frontier may still retain eligible durable-candidate, reference, or diagnostic evidence."
            )
        return (
            f"{variants_with_primary_metric} variant(s) reported, but none "
            "above baseline — nothing eligible for promotion"
        )
    if lane_based:
        return ""
    if variants_above_baseline < promote_top_k:
        return (
            f"only {variants_above_baseline} variant(s) above baseline; "
            f"policy requires top-{promote_top_k}"
        )
    return ""


def operator_manifest_paths(run_dir: Path) -> dict[str, str]:
    """Return stable pointers from status JSON to adjacent run manifests."""
    run_dir = Path(run_dir)
    return {
        "orchestrator_status": str(run_dir / "orchestrator_status.json"),
        "orchestrator_status_final": str(run_dir / "orchestrator_status.final.json"),
        "run_summary": str(run_dir / "run_summary.json"),
        "trajectory": str(run_dir / "trajectory.jsonl"),
        "artifact_index": str(run_dir / "artifact_index.jsonl"),
        "budget_ledger": str(run_dir / "budget_ledger.jsonl"),
        "credentials_redacted": str(run_dir / "credentials_redacted.json"),
        "shared_findings": str(run_dir / "shared_findings"),
        "frontier_manifest": str(run_dir / "frontier" / "frontier_manifest.json"),
        "launcher_log": str(run_dir / "logs" / "launcher.nohup.log"),
        "resource_scheduler": str(run_dir / "resource_scheduler" / "status.json"),
    }
