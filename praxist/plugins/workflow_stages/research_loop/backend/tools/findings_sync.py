"""
Background sync for shared findings.

Dual-mode:
  - Local mode (single server): reads from shared SQLite, writes JSON files
    for agents to discover via Glob/Read.
  - Server mode (multi-machine): polls HTTP endpoint for cross-machine sync.

Agents always read findings from the filesystem directory — this thread
just ensures the directory stays up-to-date with the canonical store.
"""

import asyncio
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from praxist.core.storage import read_file_bytes
from praxist.plugins.workflow_stages.research_loop.backend.event_wait import (
    wait_for_filesystem_event,
)

from .atomic_io import atomic_write_json
from .http_utils import DEFAULT_HEADERS, HAS_HTTPX, HAS_REQUESTS, get_server_url

if HAS_HTTPX:
    import httpx
if HAS_REQUESTS:
    import requests

logger = logging.getLogger(__name__)


def _finding_generation_id(finding: dict[str, Any]) -> int | None:
    value = finding.get("generation_id")
    if value is None and isinstance(finding.get("metrics"), dict):
        value = finding["metrics"].get("source_generation_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_sync_source_event(path: Path, *, findings_dir: Path, results_dir: Path) -> bool:
    """Wake for source changes while ignoring materializer-owned output writes."""

    candidate = Path(path).resolve(strict=False)
    findings_dir = Path(findings_dir).resolve(strict=False)
    results_dir = Path(results_dir).resolve(strict=False)
    if candidate.suffix.lower() != ".json":
        return False
    try:
        candidate.relative_to(results_dir)
        return True
    except ValueError:
        pass
    try:
        candidate.relative_to(findings_dir)
    except ValueError:
        return False
    try:
        # Preserve the original path so the safe reader can reject its links.
        payload = json.loads(read_file_bytes(Path(path)).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return True
    if not isinstance(payload, dict):
        return True
    metrics = payload.get("metrics")
    return not (
        isinstance(metrics, dict) and metrics.get("auto_materialized_from_result_artifact") is True
    )


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename component."""
    if not name:
        return "unknown"
    sanitized = ""
    for c in name:
        if c.isalnum() or c in "-_":
            sanitized += c
        else:
            sanitized += "_"
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")[:100]


def finding_filename(finding: dict) -> str:
    """Build the canonical filename for a finding."""
    finding_id = finding.get("id")
    variant = finding.get("variant_name") or finding.get("title") or "unknown"
    id_part = _sanitize_filename(str(finding_id or "unknown"))
    name_part = _sanitize_filename(variant)
    return f"{id_part}_{name_part}.json"


def save_finding_to_dir(finding: dict, findings_dir: Path) -> Path | None:
    """Save a single finding dict as a JSON file.

    Returns the file path if written, ``None`` if already exists or failed.

    Refuses to write when the finding came from reverse-ingest of an
    existing filesystem file (``source_filename`` set and that file is
    still on disk). Without this check, the forward-sync half of the
    daemon would mint a new UUID-shaped filename for every agent-written
    file on every cycle: the new file would then be re-ingested under a
    different ``fs_<hash>`` id, kicking off an unbounded file explosion
    (O(N × cycles) where N is the agent file count).
    """
    findings_dir.mkdir(parents=True, exist_ok=True)

    source_filename = finding.get("source_filename")
    source_filepath = finding.get("source_filepath")
    if source_filepath:
        source_path = Path(str(source_filepath)).expanduser()
        if not source_path.is_absolute():
            source_path = findings_dir.parent / source_path
        if source_path.exists():
            return None
    if source_filename:
        source_path = findings_dir / source_filename
        if source_path.exists():
            return None

    filename = finding_filename(finding)
    filepath = findings_dir / filename

    if filepath.exists():
        return None

    try:
        atomic_write_json(filepath, finding)
        return filepath
    except Exception as e:
        logger.warning(f"Failed to write {filepath}: {e}")
        return None


class FindingsSync:
    """Background thread that syncs findings to local filesystem.

    In local mode: reads from shared SQLite → writes JSON files.
    In server mode: polls HTTP endpoint → writes JSON files.
    """

    def __init__(
        self,
        findings_dir: Path,
        poll_interval: int = 60,
        local_mode: bool = False,
        primary_metric: str | None = None,
        run_dir: Path | None = None,
        materialize_result_artifacts: bool = False,
        result_artifact_default_lane: str = "performance",
        result_artifact_default_family: str = "task_candidate",
        result_cell_metric_derivations: list[dict[str, Any]] | None = None,
        result_metric_aliases: dict[str, str] | None = None,
        result_scoring_metric_keys: list[str] | tuple[str, ...] | None = None,
        result_maturity_policy: dict[str, Any] | None = None,
    ):
        self.findings_dir = Path(findings_dir)
        self.findings_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(run_dir) if run_dir is not None else self._infer_run_dir()
        self.poll_interval = poll_interval
        self.local_mode = local_mode
        # #150: when the orchestrator hands us the task's primary metric,
        # threaded into ingest so filesystem-written findings get the
        # primary value hoisted to canonical ``metrics[primary_metric]``.
        # Otherwise frontier.promote silently rejects them.
        self.primary_metric = primary_metric or None
        self.materialize_result_artifacts = bool(materialize_result_artifacts)
        self.result_artifact_default_lane = result_artifact_default_lane
        self.result_artifact_default_family = result_artifact_default_family
        self.result_cell_metric_derivations = list(result_cell_metric_derivations or [])
        self.result_metric_aliases = dict(result_metric_aliases or {})
        self.result_scoring_metric_keys = list(result_scoring_metric_keys or [])
        self.result_maturity_policy = dict(result_maturity_policy or {})
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # R6#5 fix: mutex bounding sync_once to one execution at a time.
        # Without this, the synthesis-trigger callback + 60s daemon +
        # status-snapshot-direct-call + finally-block can all enter
        # sync_once concurrently, doubling DB write rate during overlap
        # windows and contending _DB_LOCK 100s of times per call.
        self._sync_mutex = threading.Lock()
        self._boundary_evidence_cutoff: tuple[int, datetime, dict[str, str]] | None = None

    def begin_boundary_evidence_cutoff(
        self,
        gen_id: int,
        cutoff: datetime,
        evidence_source_snapshot: dict[str, str],
    ) -> None:
        """Keep background materialization aligned with the active boundary snapshot."""

        self._boundary_evidence_cutoff = (
            int(gen_id),
            cutoff,
            dict(evidence_source_snapshot),
        )

    def clear_boundary_evidence_cutoff(self, gen_id: int) -> None:
        """Retire an active cutoff after its generation marker commits."""

        active = self._boundary_evidence_cutoff
        if active is not None and active[0] == int(gen_id):
            self._boundary_evidence_cutoff = None

    def _evidence_cutoff_for_generation(self, gen_id: int) -> datetime | None:
        active = self._boundary_evidence_cutoff
        if active is None or active[0] != int(gen_id):
            return None
        return active[1]

    def _evidence_source_snapshot_for_generation(self, gen_id: int) -> dict[str, str] | None:
        active = self._boundary_evidence_cutoff
        if active is None or active[0] != int(gen_id):
            return None
        return dict(active[2])

    def start(self):
        """Start background sync thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background sync thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def sync_once(self) -> int:
        """Run one bidirectional sync cycle. Returns total touched count.

        Two-phase pipeline:
          1. **Reverse** (filesystem → SQLite): ingest any *.json in
             findings_dir that's not yet in SQLite. Agent-written files
             (via the Write tool, bypassing share_finding MCP) are picked
             up here and given stable ``fs_<hash>`` ids.
          2. **Forward** (SQLite → filesystem): materialize any SQLite row
             whose UUID-prefixed file isn't on disk yet.

        Both phases are idempotent and tolerant of per-entry failure.

        R6#5 fix: serialized via _sync_mutex so concurrent callers
        (60s daemon, trigger pre-eval callback, status-writer, finally
        block) cannot run overlapping sync passes.
        """
        if not self._sync_mutex.acquire(blocking=False):
            # Another sync is already in progress — return 0 (caller
            # should treat as "no new work")
            logger.debug("findings_sync: skipping concurrent sync_once invocation")
            return 0
        try:
            return self._sync_once_locked()
        finally:
            self._sync_mutex.release()

    def sync_once_blocking(self, *, timeout: float = 2.0) -> int:
        """Run one sync cycle, waiting briefly if the daemon is mid-sync."""

        if not self._sync_mutex.acquire(timeout=max(0.0, float(timeout))):
            logger.debug("findings_sync: timed out waiting for sync mutex")
            return 0
        try:
            return self._sync_once_locked()
        finally:
            self._sync_mutex.release()

    def _infer_run_dir(self) -> Path:
        return self.findings_dir.parent

    def _current_generation_hint(self) -> int:
        status_path = self.run_dir / "orchestrator_status.json"
        try:
            status = json.loads(read_file_bytes(status_path).decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return 0
        if not isinstance(status, dict):
            return 0
        for key in ("current_generation", "active_generation", "generation"):
            raw_value = status.get(key)
            if raw_value is None:
                continue
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            return max(0, value)
        raw_completed = status.get("completed_generations")
        if raw_completed is None:
            return 0
        try:
            completed = int(raw_completed)
        except (TypeError, ValueError):
            return 0
        return max(0, completed)

    def _materialize_result_artifacts_once(self) -> int:
        if not self.materialize_result_artifacts:
            return 0
        try:
            from ..findings_collection import (
                _delete_stale_auto_materialized_rows_from_store,
                _existing_materialized_results,
                _materialize_result_artifacts,
            )

            active_boundary = self._boundary_evidence_cutoff
            gen_id = (
                int(active_boundary[0])
                if active_boundary is not None
                else self._current_generation_hint()
            )
            materialized = _materialize_result_artifacts(
                run_dir=self.run_dir,
                gen_id=gen_id,
                default_lane=self.result_artifact_default_lane,
                default_family=self.result_artifact_default_family,
                cell_metric_derivations=self.result_cell_metric_derivations,
                metric_aliases=self.result_metric_aliases,
                scoring_metric_keys=self.result_scoring_metric_keys,
                result_maturity_policy=self.result_maturity_policy,
                evidence_cutoff=self._evidence_cutoff_for_generation(gen_id),
                evidence_source_snapshot=self._evidence_source_snapshot_for_generation(gen_id),
            )
            if self.local_mode:
                _delete_stale_auto_materialized_rows_from_store(
                    set(_existing_materialized_results(self.findings_dir))
                )
            return len(materialized)
        except Exception as e:
            logger.debug("result artifact materialization sync failed: %s", e)
            return 0

    def _sync_once_locked(self) -> int:
        """Inner sync implementation; assumes _sync_mutex is held."""
        # Phase 0: materialize completed result summaries into the existing
        # finding channel. This catches evaluator/train descendants that finish
        # after a generation boundary without retroactively rewriting the
        # boundary or frontier state.
        materialized_count = self._materialize_result_artifacts_once()

        # Phase 1: reverse sync. Only meaningful in local mode where we own
        # the SQLite file. In server mode, the HTTP server is source of
        # truth and reverse-ingest would fight it.
        reverse_count = 0
        if self.local_mode:
            try:
                from .findings_ingest import ingest_findings_directory

                reverse_count = ingest_findings_directory(
                    self.findings_dir,
                    primary_metric=self.primary_metric,
                    result_maturity_policy=self.result_maturity_policy,
                )
                for local_dir in sorted(self.run_dir.glob("gen_*/shared_findings")):
                    reverse_count += ingest_findings_directory(
                        local_dir,
                        primary_metric=self.primary_metric,
                        result_maturity_policy=self.result_maturity_policy,
                    )
            except Exception as e:
                logger.debug("reverse sync (filesystem→SQLite) failed: %s", e)

        # Phase 2: forward sync (existing behavior).
        findings = self._fetch_all_findings()
        active_boundary = self._boundary_evidence_cutoff
        if self.local_mode and active_boundary is not None:
            gen_id, cutoff, source_snapshot = active_boundary
            generation_findings = [
                finding for finding in findings if _finding_generation_id(finding) == int(gen_id)
            ]
            try:
                from ..findings_collection import (
                    annotate_late_boundary_findings,
                    persist_boundary_validation_findings,
                )

                annotated = annotate_late_boundary_findings(
                    generation_findings,
                    run_dir=self.run_dir,
                    findings_dir=self.findings_dir,
                    gen_id=gen_id,
                    cutoff=cutoff,
                    evidence_source_snapshot=source_snapshot,
                )
                if persist_boundary_validation_findings(annotated):
                    findings = self._fetch_all_findings()
            except Exception as e:
                logger.debug("boundary cutoff reverse sync failed: %s", e)
        forward_count = 0
        for finding in findings:
            path = save_finding_to_dir(finding, self.findings_dir)
            if path:
                forward_count += 1

        return materialized_count + reverse_count + forward_count

    def _fetch_all_findings(self) -> list:
        """Fetch findings from SQLite (local) or HTTP (server)."""
        if self.local_mode:
            return self._fetch_from_sqlite()
        return self._fetch_from_http()

    def _fetch_from_sqlite(self) -> list:
        """Read all findings from shared SQLite store."""
        try:
            from .local_store import get_all_findings, init_db

            init_db()
            return get_all_findings()
        except Exception as e:
            logger.debug(f"SQLite fetch failed: {e}")
            return []

    def _fetch_from_http(self) -> list:
        """Fetch all findings from HTTP server."""
        try:
            server_url = get_server_url()
        except ValueError:
            return []

        url = f"{server_url}/api/findings/all"
        try:
            if HAS_HTTPX:
                resp = httpx.get(url, headers=DEFAULT_HEADERS, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            elif HAS_REQUESTS:
                resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            else:
                return []

            if isinstance(data, list):
                return data
            return data.get("findings", [])
        except Exception as e:
            logger.debug(f"HTTP findings fetch failed: {e}")
            return []

    def _run(self):
        """Background thread loop.

        Local mode is event-driven: wake on external finding and result-summary
        changes, with a sparse heartbeat for missed inotify events. Writes made
        by the result materializer are ignored so the sync loop cannot wake
        itself. Server mode still
        polls the remote HTTP API because there is no local filesystem event
        that represents another machine's write.
        """
        while not self._stop_event.is_set():
            try:
                self.sync_once()
            except Exception as e:
                logger.debug(f"Findings sync cycle failed: {e}")
            if self._stop_event.is_set():
                break
            if not self.local_mode:
                self._stop_event.wait(timeout=self.poll_interval)
                continue
            try:
                watch_paths = [self.findings_dir]
                results_dir = self.run_dir / "results"
                if self.materialize_result_artifacts and results_dir.exists():
                    watch_paths.append(results_dir)
                asyncio.run(
                    wait_for_filesystem_event(
                        watch_paths,
                        timeout_seconds=max(300, int(self.poll_interval)),
                        stop_check=self._stop_event.is_set,
                        recursive=self.materialize_result_artifacts,
                        max_dirs=256 if self.materialize_result_artifacts else 64,
                        fallback_interval_seconds=max(300, int(self.poll_interval)),
                        stop_check_interval_seconds=30.0,
                        event_filter=lambda p, findings_dir=self.findings_dir, results_dir=results_dir: (
                            _is_sync_source_event(
                                Path(p),
                                findings_dir=findings_dir,
                                results_dir=results_dir,
                            )
                        ),
                    )
                )
            except Exception as e:
                logger.debug("Findings sync event wait failed: %s", e)
                self._stop_event.wait(timeout=max(300, int(self.poll_interval)))
