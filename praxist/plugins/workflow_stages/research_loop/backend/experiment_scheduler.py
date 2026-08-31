"""Run-local central experiment queue, launcher, and recovery service."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import socketserver
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from praxist.core.redaction import redact_text

from .experiment_process import process_group_alive, terminate_process_group
from .experiment_scheduler_client import (
    ENV_SCHEDULER_ENDPOINT,
    ExperimentRejected,
    is_sensitive_environment_entry,
    is_sensitive_environment_name,
    prepare_task_subprocess,
    rebase_recovered_task_context,
    recover_environment,
    resource_supply_signal_path,
    semantic_experiment_key,
    sensitive_environment_matches,
    task_runtime_context_changed,
)
from .resource_scheduler import (
    Allocation,
    ResourceAllocator,
    ResourceProfile,
    SchedulerSettings,
    _pid_start_time,
    _posix_file_locking,
)
from .tools.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

INFRASTRUCTURE_RETRY_EXIT_CODE = 75
_MAX_SUPPLY_WAITERS = 4096
_LEGACY_SUPPLY_LEASE_SECONDS = 180.0
_SUPPLY_DECLINE_COOLDOWN_SECONDS = 60.0
_SUPPLY_MAX_DECLINE_COOLDOWN_SECONDS = 900.0
_MATURITY_REFRESH_SECONDS = 5.0
_ENV_REFERENCE_RE = re.compile(r"__PRAXIST_ENV_REF_([A-Za-z_][A-Za-z0-9_]*)__")
_JOB_DETAIL_LIMIT = 200
_FAILURE_LOG_TAIL_BYTES = 8192
_FAILURE_LOG_TAIL_CHARS = 4096
_FAILURE_SIGNATURE_CHARS = 320
_ACCELERATOR_PROBE_GRACE_MAX_SECONDS = 120.0


def _classify_failure(exit_code: int, evidence: str) -> str:
    if exit_code == INFRASTRUCTURE_RETRY_EXIT_CODE:
        return "infrastructure"
    if exit_code < 0 or any(
        marker in evidence
        for marker in ("terminated by signal", "keyboardinterrupt", "cancelled", "canceled")
    ):
        return "interrupted"
    if any(
        marker in evidence
        for marker in (
            "out of memory",
            "resource exhausted",
            "no space left on device",
            "disk quota exceeded",
        )
    ):
        return "resource"
    if any(
        marker in evidence
        for marker in (
            "command not found",
            "no such file or directory",
            "modulenotfounderror",
            "importerror",
            "exec format error",
            "shared object file",
            "library not loaded",
        )
    ):
        return "environment"
    if any(
        marker in evidence
        for marker in ("permission denied", "operation not permitted", "read-only file system")
    ):
        return "permission"
    return "experiment"


@dataclass
class _ExperimentJob:
    job_id: str
    run_id: str
    generation_id: int
    peer_id: str
    experiment_id: str
    profile: str
    work_class: str
    command: list[str]
    cwd: str | None
    environment: dict[str, str]
    eta_seconds: int
    submitted_at: float
    state: str = "queued"
    attempts: int = 0
    pid: int = 0
    pgid: int = 0
    gpu_uuids: list[str] = field(default_factory=list)
    exit_code: int | None = None
    completed_at: float | None = None
    error: str = ""
    log_path: str = ""
    failure_category: str = ""
    failure_signature: str = ""
    failure_log_tail: str = ""
    binding_status: str = "not_applicable"
    supply_claim_id: str = ""

    @property
    def semantic_key(self) -> str:
        return semantic_experiment_key(self.run_id, self.generation_id, self.experiment_id)

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["supply_release_pending"] = bool(self.supply_claim_id)
        data.pop("command", None)
        data.pop("environment", None)
        data.pop("supply_claim_id", None)
        return data


@dataclass
class _ActiveJob:
    job: _ExperimentJob
    process: subprocess.Popen[bytes] | None
    allocation: Allocation | None
    log_handle: Any | None
    attempt_dir: Path | None
    pid_start_time: int | str | None = None


@dataclass(frozen=True)
class _SupplyLease:
    lease_id: str
    peer_id: str
    generation_id: int
    admissible_profiles: tuple[str, ...]
    priority: str
    issued_at: float
    expires_at: float

    def public(self) -> dict[str, Any]:
        return asdict(self)

    def locator(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "peer_id": self.peer_id,
            "generation_id": self.generation_id,
            "expires_at": self.expires_at,
        }


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        service: ExperimentSchedulerService = self.server.service  # type: ignore[attr-defined]
        try:
            raw = self.rfile.readline(4 * 1024 * 1024)
            request = json.loads(raw.decode("utf-8"))
            response = service.handle_request(request)
        except Exception as exc:  # noqa: BLE001 - RPC boundary returns structured errors.
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            self.wfile.write((json.dumps(response, default=str) + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("scheduler client disconnected before receiving its response")


class ExperimentSchedulerService:
    """Run-local queue with host-global atomic resource reservations."""

    def __init__(
        self,
        *,
        run_dir: Path,
        settings: SchedulerSettings,
        allocator: ResourceAllocator | None = None,
        max_parallel_runs_per_peer: int | None = None,
        recovery_rerun_generation: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        owner_digest = hashlib.sha256(str(self.run_dir).encode("utf-8")).hexdigest()
        self.resource_owner_id = "run:" + owner_digest
        self.supply_owner_id = "supply:" + owner_digest
        self.settings = settings
        self.allocator = allocator or ResourceAllocator(settings)
        self.max_parallel_runs_per_peer = (
            max(1, int(max_parallel_runs_per_peer))
            if max_parallel_runs_per_peer is not None
            else None
        )
        self._recovery_rerun_generation = recovery_rerun_generation
        self.state_dir = self.run_dir / "resource_scheduler"
        self.logs_dir = self.run_dir / "logs" / "experiments"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        uid = getattr(os, "getuid", lambda: 0)()
        digest = hashlib.sha256(str(self.run_dir.resolve()).encode()).hexdigest()[:16]
        socket_root = Path(f"/tmp/praxist-scheduler-{uid}")
        socket_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.endpoint = socket_root / f"{digest}.sock"
        self._condition = threading.Condition()
        self._jobs: dict[str, _ExperimentJob] = {}
        self._semantic_jobs: dict[str, str] = {}
        self._queue: list[str] = []
        self._active: dict[str, _ActiveJob] = {}
        self._frozen_generations: set[int] = set()
        self._assessment_generations: set[int] = set()
        self._generation_deadlines: dict[int, float | None] = {}
        self._generation_cohort_sizes: dict[int, int] = {}
        self._generation_peer_ids: dict[int, tuple[str, ...]] = {}
        self._generation_mature_targets: dict[int, int] = {}
        self._mature_count_callbacks: dict[int, Callable[[], int]] = {}
        self._mature_identity_callbacks: dict[int, Callable[[], set[str]]] = {}
        self._maturity_distinct_peers: set[int] = set()
        self._mature_completed: dict[int, int | None] = {}
        self._mature_completed_ids: dict[int, set[str]] = {}
        self._maturity_sample_at = 0.0
        self._maturity_refreshed_at = 0.0
        self._runtime_log_ratios: list[float] = []
        self._admission_closed = False
        self._stopping = False
        self._server: _UnixServer | None = None
        self._server_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._transport_cleanup_complete = False
        self._worker_error = ""
        self._accelerator_probe_unknown_since: float | None = None
        self._owner_lock_fd: int | None = None
        self._owner_finalize_lock = threading.Lock()
        self._last_snapshot_write = 0.0
        self._supply_idle_samples = 0
        self._supply_sample_at = 0.0
        self._idle_supply_waiters: dict[str, tuple[int, float]] = {}
        self._supply_leases: dict[str, _SupplyLease] = {}
        self._pending_supply_releases: dict[str, tuple[str, str, dict[str, Any] | None, bool]] = {}
        self._consumed_supply_leases: dict[str, tuple[float, str]] = {}
        self._declined_supply_peers: dict[tuple[str, int, str], tuple[float, int]] = {}
        self._supply_stats = {
            "granted": 0,
            "consumed": 0,
            "declined": 0,
            "expired": 0,
            "revoked": 0,
            "stale_submission": 0,
            "reuse_ignored": 0,
        }
        self._supply_stats_by_priority: dict[str, dict[str, int]] = {}

    def start(self) -> None:
        if not self.settings.enabled:
            return
        self._acquire_owner_lock()
        try:
            set_owner = getattr(self.allocator, "set_owner", None)
            if callable(set_owner):
                set_owner(self.resource_owner_id)
            clear_supply = getattr(self.allocator, "clear_supply", None)
            if callable(clear_supply):
                clear_supply(self.supply_owner_id)
            self._remove_stale_supply_files()
            self._recover_terminal_events()
            self._recover_active_process_groups()
            self.endpoint.unlink(missing_ok=True)
            server = _UnixServer(str(self.endpoint), _RequestHandler)
            server.service = self  # type: ignore[attr-defined]
            self._server = server
            self._server_thread = threading.Thread(
                target=server.serve_forever,
                name="experiment-scheduler-rpc",
                daemon=True,
            )
            self._worker_thread = threading.Thread(
                target=self._worker,
                name="experiment-scheduler-worker",
                daemon=True,
            )
            atomic_write_json(self.state_dir / "endpoint.json", {"endpoint": str(self.endpoint)})
            self._write_snapshot(force=True)
            os.environ[ENV_SCHEDULER_ENDPOINT] = str(self.endpoint)
            self._server_thread.start()
            for active in tuple(self._active.values()):
                if active.attempt_dir is not None and not (active.attempt_dir / "GO.json").exists():
                    self._release_launch_barrier(active.attempt_dir)
            self._worker_thread.start()
            logger.info("Central experiment scheduler started at %s", self.endpoint)
        except BaseException:
            self._begin_failed_start_drain()
            self._close_transport()
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5)
            worker_alive = self._worker_thread is not None and self._worker_thread.is_alive()
            if not worker_alive and not self._active and self._transport_cleanup_complete:
                self._finalize_owner_resources()
            raise

    def stop(self) -> None:
        owns_run = self._owner_lock_fd is not None
        if not owns_run:
            return
        stop_error: BaseException | None = None
        try:
            self._stop_admission_and_supply()
        except BaseException as exc:  # Preserve the audit failure after transport cleanup.
            stop_error = exc
            with self._condition:
                self._condition.notify_all()
        try:
            self._write_snapshot(force=True)
        except BaseException as exc:
            if stop_error is None:
                stop_error = exc
        transport_error = self._close_transport()
        if stop_error is None and transport_error is not None:
            stop_error = transport_error
        if self._worker_thread is not None and self._worker_thread.is_alive() and self._stopping:
            self._worker_thread.join(timeout=5)
        with self._condition:
            cleanup_pending = self._shutdown_cleanup_pending_locked()
        if (
            (self._worker_thread is None or not self._worker_thread.is_alive())
            and not cleanup_pending
            and self._transport_cleanup_complete
        ):
            if not self._finalize_owner_resources() and stop_error is None:
                stop_error = RuntimeError(
                    "central scheduler resource-controller cleanup remains pending"
                )
        else:
            logger.info(
                "Central experiment scheduler retains run ownership while %d active job(s) drain",
                len(self._active),
            )
        if stop_error is not None:
            raise stop_error

    def _begin_failed_start_drain(self) -> None:
        worker_alive = self._worker_thread is not None and self._worker_thread.is_alive()
        if not worker_alive and not self._active:
            return
        with self._condition:
            self._admission_closed = True
            self._stopping = True
            self._condition.notify_all()
        if self._active and not worker_alive:
            try:
                self._worker_thread = threading.Thread(
                    target=self._worker,
                    name="experiment-scheduler-drain",
                    daemon=True,
                )
                self._worker_thread.start()
            except Exception as exc:  # Retain ownership if no drainer can start.
                self._worker_error = f"{type(exc).__name__}: {exc}"

    def _close_transport(self) -> BaseException | None:
        first_error: BaseException | None = None
        server_clean = True
        if self._server is not None:
            if self._server_thread is not None and self._server_thread.is_alive():
                try:
                    self._server.shutdown()
                except BaseException as exc:
                    first_error = first_error or exc
                    server_clean = False
            try:
                self._server.server_close()
            except BaseException as exc:
                first_error = first_error or exc
                server_clean = False
        if self._server_thread is not None and self._server_thread.is_alive():
            try:
                self._server_thread.join(timeout=5)
            except BaseException as exc:
                first_error = first_error or exc
                server_clean = False
            if self._server_thread.is_alive():
                first_error = first_error or RuntimeError(
                    "central scheduler RPC thread did not stop"
                )
                server_clean = False
        if server_clean:
            self._server = None
        for path in (self.endpoint, self.state_dir / "endpoint.json"):
            try:
                path.unlink(missing_ok=True)
            except BaseException as exc:
                first_error = first_error or exc
        os.environ.pop(ENV_SCHEDULER_ENDPOINT, None)
        with self._condition:
            self._transport_cleanup_complete = server_clean
            self._condition.notify_all()
        return first_error

    def _stop_admission_and_supply(self) -> None:
        supply_owner_cleared = False
        with self._condition:
            # Freeze local admission before the durable fence is attempted. If
            # storage is unavailable, ownership and the worker stay alive but
            # no queued work can launch before a later stop retry succeeds.
            self._admission_closed = True
            self._append_event(
                {"event": "admission_closed", "reason": "scheduler_stopped"},
                required=True,
            )
            self._stopping = True
            self._revoke_supply_locked(reason="scheduler_stopped")
            clear_supply = getattr(self.allocator, "clear_supply", None)
            if self._pending_supply_releases and callable(clear_supply):
                try:
                    clear_supply(self.supply_owner_id)
                except Exception as exc:  # noqa: BLE001 - pending claims remain visible.
                    logger.warning("could not clear pending supply claims during stop: %s", exc)
                else:
                    supply_owner_cleared = True
                    for lease_id, pending in list(self._pending_supply_releases.items()):
                        lease = self._supply_leases.get(lease_id)
                        if lease is None:
                            self._pending_supply_releases.pop(lease_id, None)
                            continue
                        outcome, reason, event_details, terminal_recorded = pending
                        if reason == "grant_audit_failed":
                            self._pending_supply_releases.pop(lease_id, None)
                            self._supply_leases.pop(lease_id, None)
                            continue
                        if not terminal_recorded:
                            event = {
                                "event": f"supply_{outcome}",
                                "lease_id": lease_id,
                                "generation_id": lease.generation_id,
                                "peer_id": lease.peer_id,
                                "priority": lease.priority,
                                "reason": reason,
                            }
                            if event_details:
                                event.update(event_details)
                            try:
                                self._append_event(event, required=True)
                            except OSError as exc:
                                logger.warning(
                                    "could not persist cleared supply claim %s during stop: %s",
                                    lease_id,
                                    exc,
                                )
                                continue
                        self._pending_supply_releases.pop(lease_id, None)
                        self._supply_leases.pop(lease_id, None)
                        self._increment_supply_stat_locked(outcome, lease)
            self._consumed_supply_leases.clear()
            try:
                for job_id in list(self._queue):
                    self._finish_without_launch(self._jobs[job_id], "scheduler_stopped")
            finally:
                clear_supply = getattr(self.allocator, "clear_supply", None)
                if callable(clear_supply):
                    if not supply_owner_cleared:
                        try:
                            clear_supply(self.supply_owner_id)
                        except Exception as exc:  # noqa: BLE001 - claims remain visible on jobs.
                            logger.warning(
                                "could not clear terminal supply claims during stop: %s", exc
                            )
                        else:
                            supply_owner_cleared = True
                    if supply_owner_cleared:
                        for job in self._jobs.values():
                            if job.state in {
                                "completed",
                                "failed",
                                "rejected",
                                "drained_unknown",
                            }:
                                job.supply_claim_id = ""
            self._queue.clear()
            self._condition.notify_all()

    def _finalize_owner_resources(self) -> bool:
        with self._owner_finalize_lock:
            if self._owner_lock_fd is None:
                return True
            close_allocator = getattr(self.allocator, "close", None)
            if callable(close_allocator):
                try:
                    close_allocator()
                except Exception as exc:  # Retain ownership so stop can retry cleanup.
                    self._worker_error = f"{type(exc).__name__}: {exc}"
                    return False
            self._release_owner_lock()
            return True

    def _acquire_owner_lock(self) -> None:
        if self._owner_lock_fd is not None:
            raise RuntimeError("central experiment scheduler is already started")
        file_locking = _posix_file_locking()
        path = self.state_dir / "owner.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            file_locking.flock(fd, file_locking.LOCK_EX | file_locking.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError(
                f"another central experiment scheduler already owns run {self.run_dir}"
            ) from exc
        except BaseException:
            os.close(fd)
            raise
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(fd)
        except BaseException:
            with contextlib.suppress(OSError):
                file_locking.flock(fd, file_locking.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)
            raise
        self._owner_lock_fd = fd

    def _release_owner_lock(self) -> None:
        fd = self._owner_lock_fd
        if fd is None:
            return
        self._owner_lock_fd = None
        file_locking = _posix_file_locking()
        with contextlib.suppress(OSError):
            file_locking.flock(fd, file_locking.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        action = str(request.get("action", ""))
        if action == "ping":
            return {"ok": True, "run_id": self.run_id}
        if action == "submit":
            job, deduplicated = self._submit(request)
            existing_terminal = bool(
                deduplicated
                and request.get("retry_terminal") is not True
                and job.state in {"failed", "rejected"}
            )
            return {
                "ok": True,
                "job": job.public(),
                "existing_terminal_job": existing_terminal,
                "retry_requires_explicit_request": existing_terminal,
            }
        if action == "wait":
            return self.wait(str(request.get("job_id", "")), request.get("timeout_seconds"))
        if action == "status":
            return {"ok": True, "status": self.status()}
        if action == "validate_attempt":
            return {
                "ok": True,
                "active": self.attempt_is_active(
                    str(request.get("attempt_id", "")),
                    int(request.get("pgid", 0) or 0),
                ),
            }
        if action == "active_process_groups":
            return {"ok": True, "groups": self.active_process_groups()}
        if action == "cancel_queued":
            return {"ok": True, **self.cancel_queued(str(request.get("job_id", "")))}
        if action == "register_idle_supply":
            supply = self.register_idle_supply(
                str(request.get("peer_id", "")), int(request.get("generation_id", 0))
            )
            return {"ok": True, "supply": supply}
        if action == "unregister_idle_supply":
            self.unregister_idle_supply(
                str(request.get("peer_id", "")), int(request.get("generation_id", 0))
            )
            return {"ok": True}
        if action == "get_supply_lease":
            supply = self.get_supply_lease(
                str(request.get("peer_id", "")),
                int(request.get("generation_id", 0)),
                str(request.get("lease_id", "")),
            )
            return {"ok": True, "supply": supply}
        if action == "release_supply_lease":
            self.release_supply_lease(
                str(request.get("lease_id", "")),
                str(request.get("peer_id", "")),
                declined=bool(request.get("declined", False)),
                reason=str(request.get("reason", "")),
            )
            return {"ok": True}
        if action == "generation_advice":
            return {
                "ok": True,
                "advice": self.generation_advice(
                    str(request.get("peer_id", "")),
                    int(request.get("generation_id", 0)),
                ),
            }
        if action == "begin_assessment":
            self.begin_assessment(
                int(request["generation_id"]),
                str(request.get("reason", "assessment")),
            )
            return {"ok": True}
        if action == "freeze":
            self.freeze_generation(int(request["generation_id"]), str(request.get("reason", "")))
            return {"ok": True}
        if action == "freeze_all":
            self.freeze_all(str(request.get("reason", "external_stop")))
            return {"ok": True}
        raise ValueError(f"unknown scheduler action: {action!r}")

    def attempt_is_active(self, attempt_id: str, pgid: int) -> bool:
        """Confirm nested work against the scheduler's live launch authority."""

        if not attempt_id or pgid <= 1:
            return False
        from .protected_pids import _pid_start_time as protected_pid_start_time

        with self._condition:
            for active in self._active.values():
                job = active.job
                process = active.process
                if (
                    job.state != "running"
                    or active.attempt_dir is None
                    or job.pgid != pgid
                    or f"{job.job_id}-a{job.attempts}" != attempt_id
                ):
                    continue
                launcher_live = (
                    active.pid_start_time is not None
                    and protected_pid_start_time(job.pid) == active.pid_start_time
                )
                if process is not None and process.poll() is None:
                    if process.pid != job.pid or not launcher_live:
                        continue
                    try:
                        return os.getpgid(job.pid) == pgid
                    except (ProcessLookupError, PermissionError, OSError):
                        continue
                if process is None and launcher_live:
                    try:
                        if os.getpgid(job.pid) == pgid:
                            return True
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                if process_group_alive(pgid):
                    return True
        return False

    def active_process_groups(self) -> list[dict[str, int | str]]:
        """Return identity-bound process groups still owned by live launchers."""

        from .protected_pids import _pid_start_time as protected_pid_start_time

        groups: list[dict[str, int | str]] = []
        with self._condition:
            for active in self._active.values():
                job = active.job
                process = active.process
                if (
                    job.state != "running"
                    or process is None
                    or process.poll() is not None
                    or process.pid != job.pid
                    or job.pid <= 1
                    or job.pgid <= 1
                    or active.pid_start_time is None
                    or protected_pid_start_time(job.pid) != active.pid_start_time
                ):
                    continue
                try:
                    if os.getpgid(job.pid) != job.pgid:
                        continue
                except (ProcessLookupError, PermissionError, OSError):
                    continue
                groups.append(
                    {
                        "pgid": job.pgid,
                        "pid": job.pid,
                        "pid_start_time": active.pid_start_time,
                    }
                )
        return sorted(groups, key=lambda group: (int(group["pgid"]), int(group["pid"])))

    def submit(self, request: dict[str, Any]) -> _ExperimentJob:
        job, _deduplicated = self._submit(request)
        return job

    def _submit(self, request: dict[str, Any]) -> tuple[_ExperimentJob, bool]:
        command = request.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(x, str) for x in command)
        ):
            raise ExperimentRejected("command must be a non-empty argv list")
        experiment_id = str(request.get("experiment_id", "")).strip()
        if not experiment_id:
            raise ExperimentRejected("experiment_id must describe the semantic experiment")
        generation_id = int(request.get("generation_id", 0))
        peer_id = str(request.get("peer_id", "")).strip()
        work_class = str(request.get("work_class", "ordinary")).strip().lower()
        if work_class not in {"mature", "ordinary", "scout"}:
            work_class = "ordinary"
        requested_profile = str(request.get("profile", "")).strip()
        if requested_profile and requested_profile not in self.settings.profiles:
            raise ExperimentRejected(f"unknown resource profile: {requested_profile!r}")
        profile = self.settings.profile(requested_profile).name
        raw_environment = request.get("environment")
        normalized_command, environment, cwd = prepare_task_subprocess(
            list(command),
            {str(key): str(value) for key, value in raw_environment.items()}
            if isinstance(raw_environment, dict)
            else dict(os.environ),
            cwd=request.get("cwd"),
        )
        candidate = _ExperimentJob(
            job_id=uuid.uuid4().hex,
            run_id=self.run_id,
            generation_id=generation_id,
            peer_id=peer_id,
            experiment_id=experiment_id,
            profile=profile,
            work_class=work_class,
            command=normalized_command,
            cwd=cwd,
            environment=environment,
            eta_seconds=max(0, int(request.get("eta_seconds", 0) or 0)),
            submitted_at=time.time(),
        )
        retry_terminal = request.get("retry_terminal") is True
        prior_terminal_state = ""
        with self._condition:
            existing_id = self._semantic_jobs.get(candidate.semantic_key)
            if existing_id is not None:
                existing = self._jobs[existing_id]
                if not retry_terminal:
                    return existing, True
                if existing.state not in {"failed", "rejected"}:
                    if self._same_submission(existing, candidate):
                        return existing, True
                    raise ExperimentRejected(
                        "explicit terminal retry requires an existing failed or rejected "
                        f"experiment; current state is {existing.state!r}"
                    )
                inherited_peer_ids = {
                    value
                    for key in ("PRAXIST_PEER_ID", "PEER_ID")
                    if (value := candidate.environment.get(key, "").strip())
                }
                if inherited_peer_ids and inherited_peer_ids != {candidate.peer_id}:
                    raise ExperimentRejected(
                        "explicit terminal retry peer does not match the inherited runtime identity"
                    )
                if existing.peer_id != candidate.peer_id:
                    raise ExperimentRejected(
                        "explicit terminal retry must be submitted by the peer that owns "
                        "the semantic experiment"
                    )
                if existing.supply_claim_id:
                    raise ExperimentRejected(
                        "previous terminal experiment resource release is still pending; "
                        "retry after scheduler reconciliation"
                    )
                prior_terminal_state = existing.state
                candidate.job_id = existing.job_id
                candidate.attempts = existing.attempts
            elif retry_terminal:
                raise ExperimentRejected(
                    "explicit terminal retry requires a retained failed or rejected experiment"
                )
            unavailable = self._accelerator_block_reason(candidate, allow_transient=True)
            if unavailable:
                raise ExperimentRejected(unavailable)
            supply_lease_id = str(request.get("supply_lease_id", ""))
            if supply_lease_id in self._consumed_supply_leases:
                _expires_at, consumed_priority = self._consumed_supply_leases[supply_lease_id]
                self._increment_supply_stat_locked(
                    "reuse_ignored", None, priority=consumed_priority
                )
                self._append_event(
                    {
                        "event": "supply_reuse_ignored",
                        "lease_id": supply_lease_id,
                        "generation_id": generation_id,
                        "peer_id": peer_id,
                        "priority": consumed_priority,
                    }
                )
                supply_lease_id = ""
            elif (
                supply_lease_id
                and (pending_job := self._pending_supply_job_locked(supply_lease_id)) is not None
            ):
                lease = self._supply_leases.get(supply_lease_id)
                priority = lease.priority if lease is not None else ""
                self._increment_supply_stat_locked("reuse_ignored", lease, priority=priority)
                self._append_event(
                    {
                        "event": "supply_reuse_ignored",
                        "lease_id": supply_lease_id,
                        "generation_id": generation_id,
                        "peer_id": peer_id,
                        "priority": priority,
                        "reason": f"accepted_by_{pending_job.experiment_id}",
                    }
                )
                supply_lease_id = ""
            elif supply_lease_id in self._pending_supply_releases:
                self._increment_supply_stat_locked("stale_submission", None)
                self._append_event(
                    {
                        "event": "supply_stale_submission",
                        "lease_id": supply_lease_id,
                        "generation_id": generation_id,
                        "peer_id": peer_id,
                        "reason": "lease_terminal_release_pending",
                    }
                )
                supply_lease_id = ""
            elif (
                supply_lease_id
                and (lease := self._supply_leases.get(supply_lease_id)) is not None
                and lease.expires_at <= time.time()
            ):
                self._increment_supply_stat_locked("stale_submission", lease)
                self._append_event(
                    {
                        "event": "supply_stale_submission",
                        "lease_id": supply_lease_id,
                        "generation_id": generation_id,
                        "peer_id": peer_id,
                        "priority": lease.priority,
                        "reason": "lease_expired",
                    }
                )
                supply_lease_id = ""
            elif supply_lease_id and supply_lease_id not in self._supply_leases:
                self._increment_supply_stat_locked("stale_submission", None)
                self._append_event(
                    {
                        "event": "supply_stale_submission",
                        "lease_id": supply_lease_id,
                        "generation_id": generation_id,
                        "peer_id": peer_id,
                        "reason": "lease_not_active",
                    }
                )
                supply_lease_id = ""
            if self._stopping or self._admission_closed:
                raise ExperimentRejected("scheduler is stopping")
            if generation_id in self._frozen_generations or self._generation_signal_exists(
                generation_id
            ):
                self._frozen_generations.add(generation_id)
                raise ExperimentRejected(
                    f"generation {generation_id} is closing; new work is frozen"
                )
            if generation_id in self._assessment_generations and work_class != "mature":
                raise ExperimentRejected(
                    f"generation {generation_id} is in assessment; only mature work may start"
                )
            if not self._fits_generation(candidate):
                raise ExperimentRejected(
                    f"experiment ETA does not fit generation {generation_id} remaining time"
                )
            lease = self._supply_leases.get(supply_lease_id)
            supply_lease_eligible = bool(
                lease is not None
                and lease.peer_id == peer_id
                and lease.generation_id == generation_id
                and lease.expires_at > time.time()
                and profile in lease.admissible_profiles
                and (lease.priority != "mature" or work_class == "mature")
            )
            submission_event = {
                "event": "retry_queued" if prior_terminal_state else "submitted",
                **self._event_identity(candidate, include_request=True),
                "supply_lease_id": supply_lease_id,
                "supply_lease_eligible": supply_lease_eligible,
            }
            if prior_terminal_state:
                submission_event.update(
                    {
                        "retry_reason": "explicit_terminal_retry",
                        "prior_terminal_state": prior_terminal_state,
                    }
                )
            self._append_event(submission_event, required=True)
            self._declined_supply_peers = {
                key: value
                for key, value in self._declined_supply_peers.items()
                if key[:2] != (peer_id, generation_id)
            }
            self._jobs[candidate.job_id] = candidate
            self._semantic_jobs[candidate.semantic_key] = candidate.job_id
            self._queue.append(candidate.job_id)
            self._idle_supply_waiters.pop(peer_id, None)
            try:
                candidate.supply_claim_id = self._consume_or_revoke_peer_supply_locked(
                    peer_id=peer_id,
                    generation_id=generation_id,
                    profile=profile,
                    lease_id=supply_lease_id,
                    work_class=work_class,
                )
            except OSError as exc:
                if supply_lease_eligible:
                    candidate.supply_claim_id = supply_lease_id
                logger.warning(
                    "supply transition audit failed for accepted experiment %s; "
                    "the semantic job remains queued: %s",
                    candidate.experiment_id,
                    exc,
                )
            self._condition.notify_all()
        self._write_snapshot(force=True)
        return candidate, False

    @staticmethod
    def _same_submission(existing: _ExperimentJob, candidate: _ExperimentJob) -> bool:
        def stable_environment(job: _ExperimentJob) -> dict[str, str]:
            return {
                key: value
                for key, value in job.environment.items()
                if key != "PRAXIST_RESOURCE_SUPPLY_LEASE_ID"
            }

        return (
            existing.peer_id == candidate.peer_id
            and existing.profile == candidate.profile
            and existing.work_class == candidate.work_class
            and existing.command == candidate.command
            and existing.cwd == candidate.cwd
            and stable_environment(existing) == stable_environment(candidate)
            and existing.eta_seconds == candidate.eta_seconds
        )

    def register_idle_supply(self, peer_id: str, generation_id: int) -> dict[str, Any]:
        if not self._valid_supply_peer_id(peer_id, generation_id):
            return {}
        with self._condition:
            if (
                not self.settings.supply_signal_enabled
                or self._stopping
                or self._admission_closed
                or generation_id in self._frozen_generations
                or self._generation_signal_exists(generation_id)
                or generation_id not in self._generation_deadlines
                or (
                    peer_id not in self._idle_supply_waiters
                    and len(self._idle_supply_waiters) >= _MAX_SUPPLY_WAITERS
                )
            ):
                return {}
            existing = next(
                (
                    lease
                    for lease in self._supply_leases.values()
                    if lease.peer_id == peer_id and lease.generation_id == generation_id
                ),
                None,
            )
            if existing is not None and existing.expires_at <= time.time():
                self._remove_supply_lease_locked(
                    existing.lease_id,
                    outcome="expired",
                    reason="idle_peer_reregistered_after_expiry",
                )
                existing = None
            if existing is not None:
                return existing.public()
            self._idle_supply_waiters[peer_id] = (generation_id, time.time())
            self._condition.notify_all()
        return {}

    def get_supply_lease(self, peer_id: str, generation_id: int, lease_id: str) -> dict[str, Any]:
        if not self._valid_supply_peer_id(peer_id, generation_id):
            return {}
        with self._condition:
            lease = self._supply_leases.get(lease_id)
            if lease is None or lease.peer_id != peer_id or lease.generation_id != generation_id:
                return {}
            if (
                lease_id in self._pending_supply_releases
                or self._pending_supply_job_locked(lease_id) is not None
            ):
                return {}
            if lease.expires_at <= time.time():
                self._remove_supply_lease_locked(
                    lease_id,
                    outcome="expired",
                    reason="lease_fetched_after_expiry",
                )
                return {}
            return lease.public()

    def release_supply_lease(
        self,
        lease_id: str,
        peer_id: str,
        *,
        declined: bool = False,
        reason: str = "",
    ) -> None:
        with self._condition:
            lease = self._supply_leases.get(lease_id)
            if lease is None or lease.peer_id != peer_id:
                return
            if self._consume_pending_supply_locked(
                lease_id,
                reason="accepted_submission_released_by_agent",
            ):
                self._condition.notify_all()
                return
            if lease.expires_at <= time.time():
                self._remove_supply_lease_locked(
                    lease_id,
                    outcome="expired",
                    reason="release_after_response_window",
                )
                self._condition.notify_all()
                return
            if declined:
                key = (peer_id, lease.generation_id, lease.priority)
                previous = self._declined_supply_peers.get(key)
                decline_count = (previous[1] if previous is not None else 0) + 1
                cooldown_seconds = min(
                    _SUPPLY_MAX_DECLINE_COOLDOWN_SECONDS,
                    _SUPPLY_DECLINE_COOLDOWN_SECONDS * (2 ** min(4, decline_count - 1)),
                )
            else:
                key = None
                decline_count = 0
                cooldown_seconds = 0.0
            self._remove_supply_lease_locked(
                lease_id,
                outcome="declined" if declined else "revoked",
                reason=reason
                or ("session_ended_without_submission" if declined else "peer_release"),
                event_details=(
                    {
                        "decline_count": decline_count,
                        "cooldown_seconds": cooldown_seconds,
                    }
                    if declined
                    else None
                ),
            )
            if key is not None:
                self._declined_supply_peers[key] = (
                    time.time() + cooldown_seconds,
                    decline_count,
                )
            self._condition.notify_all()

    def unregister_idle_supply(self, peer_id: str, generation_id: int) -> None:
        with self._condition:
            waiter = self._idle_supply_waiters.get(peer_id)
            if waiter is not None and waiter[0] == generation_id:
                self._idle_supply_waiters.pop(peer_id, None)
            for lease_id, lease in list(self._supply_leases.items()):
                if lease.peer_id == peer_id and lease.generation_id == generation_id:
                    self._remove_supply_lease_locked(
                        lease_id,
                        outcome="revoked",
                        reason="peer_unregistered_idle_supply",
                    )
            self._condition.notify_all()

    def wait(self, job_id: str, timeout_seconds: object = None) -> dict[str, Any]:
        timeout = None if timeout_seconds is None else max(0.0, float(str(timeout_seconds)))
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    return {"ok": False, "error": "unknown job_id"}
                if job.state in {"completed", "failed", "rejected", "drained_unknown"}:
                    return {"ok": True, "job": job.public()}
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return {"ok": True, "timeout": True, "job": job.public()}
                self._condition.wait(timeout=remaining)

    def freeze_generation(self, generation_id: int, reason: str = "") -> None:
        with self._condition:
            self._append_event(
                {
                    "event": "generation_frozen",
                    "generation_id": generation_id,
                    "reason": reason,
                },
                required=True,
            )
            self._frozen_generations.add(generation_id)
            self._revoke_supply_locked(generation_id, reason="generation_frozen")
            retained: list[str] = []
            for job_id in self._queue:
                job = self._jobs[job_id]
                if job.generation_id == generation_id:
                    self._finish_without_launch(job, f"generation_closing:{reason}")
                else:
                    retained.append(job_id)
            self._queue = retained
            self._condition.notify_all()
        self._write_snapshot(force=True)

    def begin_assessment(self, generation_id: int, reason: str = "assessment") -> None:
        """Stop ordinary admission while allowing deadline-safe mature top-ups."""

        with self._condition:
            if generation_id in self._frozen_generations:
                return
            self._append_event(
                {
                    "event": "generation_assessment",
                    "generation_id": generation_id,
                    "reason": reason,
                },
                required=True,
            )
            self._assessment_generations.add(generation_id)
            retained: list[str] = []
            for job_id in self._queue:
                job = self._jobs[job_id]
                if job.generation_id == generation_id and job.work_class != "mature":
                    self._finish_without_launch(job, f"generation_assessment:{reason}")
                else:
                    retained.append(job_id)
            self._queue = retained
            for lease_id, lease in list(self._supply_leases.items()):
                if lease.generation_id == generation_id and lease.priority != "mature":
                    self._remove_supply_lease_locked(
                        lease_id,
                        outcome="revoked",
                        reason="assessment_mature_only",
                    )
            self._condition.notify_all()
        self._write_snapshot(force=True)

    def cancel_queued(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return {"cancelled": False, "error": "unknown job_id"}
            if job.state != "queued" or job_id not in self._queue:
                return {"cancelled": False, "job": job.public()}
            self._queue.remove(job_id)
            self._finish_without_launch(job, "admission_timeout")
            self._condition.notify_all()
            return {"cancelled": True, "job": job.public()}

    def open_generation(
        self,
        generation_id: int,
        *,
        deadline: float | None,
        cohort_size: int = 0,
        peer_ids: Sequence[str] | None = None,
    ) -> None:
        """Open admission until an optional wall-clock deadline or explicit close.

        Args:
            generation_id: Generation whose admission should open.
            deadline: Absolute Unix timestamp, or None for no time cap.
            cohort_size: Number of peers participating in the generation.
            peer_ids: Explicit identities eligible for generation supply.
        """
        deadline = None if deadline is None else float(deadline)
        generation_peer_ids = tuple(
            peer_id
            for peer_id in dict.fromkeys(str(value) for value in (peer_ids or ()))
            if self._valid_supply_peer_id(peer_id, generation_id)
        )
        with self._condition:
            assessment_continues = generation_id in self._assessment_generations
            self._append_event(
                {
                    "event": "generation_open",
                    "generation_id": generation_id,
                    "deadline": deadline,
                    "cohort_size": max(0, int(cohort_size)),
                    "peer_ids": list(generation_peer_ids),
                    "assessment_continues": assessment_continues,
                },
                required=True,
            )
            self._revoke_supply_locked(reason="generation_reopened")
            self._declined_supply_peers.clear()
            self._mature_count_callbacks.clear()
            self._mature_identity_callbacks.clear()
            self._maturity_distinct_peers.clear()
            self._mature_completed.clear()
            self._mature_completed_ids.clear()
            self._maturity_sample_at = 0.0
            self._maturity_refreshed_at = 0.0
            self._generation_mature_targets.clear()
            self._generation_cohort_sizes.clear()
            self._generation_peer_ids.clear()
            self._generation_deadlines[generation_id] = deadline
            self._generation_cohort_sizes[generation_id] = max(0, int(cohort_size))
            if generation_peer_ids:
                self._generation_peer_ids[generation_id] = generation_peer_ids
            self._admission_closed = False
            if generation_id == self._recovery_rerun_generation:
                self._recovery_rerun_generation = None
            if not assessment_continues:
                self._assessment_generations.discard(generation_id)
            self._frozen_generations.discard(generation_id)
            self._condition.notify_all()
        self._write_snapshot(force=True)

    def configure_generation_maturity(
        self,
        generation_id: int,
        *,
        cohort_size: int,
        mature_target: int,
        count_callback: Callable[[], int],
        identity_callback: Callable[[], set[str]] | None = None,
        distinct_peer_commitments: bool = False,
    ) -> None:
        """Attach the existing evidence view to generation supply control."""

        with self._condition:
            self._generation_cohort_sizes[generation_id] = max(0, int(cohort_size))
            self._generation_mature_targets[generation_id] = max(0, int(mature_target))
            self._mature_count_callbacks[generation_id] = count_callback
            if identity_callback is not None:
                self._mature_identity_callbacks[generation_id] = identity_callback
            if distinct_peer_commitments:
                self._maturity_distinct_peers.add(generation_id)
            self._mature_completed[generation_id] = None
            self._mature_completed_ids[generation_id] = set()
            self._condition.notify_all()
        self._write_snapshot(force=True)

    def generation_advice(self, peer_id: str, generation_id: int) -> dict[str, Any]:
        """Return deterministic direct/scout first-wave advice for one peer."""

        if not self._valid_supply_peer_id(peer_id, generation_id):
            return {}
        with self._condition:
            cohort_size = self._generation_cohort_sizes.get(generation_id, 0)
            target = self._generation_mature_targets.get(generation_id, 0)
            if cohort_size <= 0 or target <= 0:
                return {}
            direct = min(target, max(1, cohort_size - 1)) if cohort_size > 1 else 1
            peer_ids = self._generation_peer_ids.get(generation_id, ())
            if peer_ids:
                return {
                    "generation_id": generation_id,
                    "mature_target": target,
                    "first_wave": ("direct_mature" if peer_id in peer_ids[:direct] else "explore"),
                }
            match = re.fullmatch(rf"gen{generation_id}_peer([0-9]+)", peer_id)
            if match is None:
                return {}
            return {
                "generation_id": generation_id,
                "mature_target": target,
                "first_wave": "direct_mature" if int(match.group(1)) < direct else "explore",
            }

    def freeze_all(self, reason: str = "external_stop") -> None:
        with self._condition:
            self._append_event(
                {"event": "admission_closed", "reason": reason},
                required=True,
            )
            self._admission_closed = True
            self._revoke_supply_locked(reason=f"admission_closed:{reason}")
            for job_id in list(self._queue):
                self._finish_without_launch(self._jobs[job_id], f"admission_closed:{reason}")
            self._queue.clear()
            self._condition.notify_all()
        self._write_snapshot(force=True)

    def status(self) -> dict[str, Any]:
        with self._condition:
            jobs = [job.public() for job in self._jobs.values()]
            retained_jobs = jobs[-_JOB_DETAIL_LIMIT:]
            failure_counts: dict[str, int] = {}
            for job in jobs:
                if job["state"] != "failed":
                    continue
                category = str(job.get("failure_category") or "unclassified")
                failure_counts[category] = failure_counts.get(category, 0) + 1
            activity_by_job: dict[str, dict[str, Any]] = {}
            for job_id, active in self._active.items():
                observed_activity: dict[str, Any] = {"state": "unknown"}
                if active.allocation is not None:
                    describe = getattr(self.allocator, "describe_allocation_activity", None)
                    if callable(describe):
                        described: Any = describe(active.allocation)
                        if isinstance(described, dict):
                            observed_activity = described
                activity_by_job[job_id] = observed_activity
            activity_counts: dict[str, int] = {}
            peer_capacity_blocked = 0
            queue_blocked_reasons: dict[str, int] = {}
            for job in jobs:
                if job["state"] == "queued":
                    queued_job = self._jobs[str(job["job_id"])]
                    reason = self._queue_block_reason(queued_job)
                    if reason:
                        job["queue_blocked_reason"] = reason
                        queue_blocked_reasons[reason] = queue_blocked_reasons.get(reason, 0) + 1
                        if reason == "per_peer_capacity":
                            peer_capacity_blocked += 1
                job_activity = activity_by_job.get(str(job["job_id"]))
                if job_activity is None:
                    continue
                job["resource_activity"] = job_activity
                state = str(job_activity.get("state", "unknown"))
                activity_counts[state] = activity_counts.get(state, 0) + 1
            running = sum(job["state"] == "running" for job in jobs)
            return {
                "mode": "central",
                "queued": sum(job["state"] == "queued" for job in jobs),
                "running": running,
                "running_activity": {
                    "lifecycle_running": running,
                    "by_resource_phase": activity_counts,
                },
                "peer_capacity_blocked": peer_capacity_blocked,
                "queue_blocked_reasons": queue_blocked_reasons,
                "completed": sum(job["state"] == "completed" for job in jobs),
                "failed": sum(job["state"] == "failed" for job in jobs),
                "rejected": sum(job["state"] == "rejected" for job in jobs),
                "recovered_running": sum(
                    job["state"] == "running" and job["attempts"] == 0 for job in jobs
                ),
                "jobs_total": len(jobs),
                "jobs_retained": len(retained_jobs),
                "jobs_omitted": len(jobs) - len(retained_jobs),
                "failure_counts_by_category": failure_counts,
                "concurrency_limit": self.allocator.concurrency_limit,
                "host": asdict(self.allocator.snapshot),
                "accelerator_probe": {
                    "state": self.allocator.snapshot.accelerator_probe_state,
                    "reason": self.allocator.snapshot.accelerator_probe_reason,
                },
                "frozen_generations": sorted(self._frozen_generations),
                "assessment_generations": sorted(self._assessment_generations),
                "generation_deadlines": dict(self._generation_deadlines),
                "admission_closed": self._admission_closed,
                "resource_supply": {
                    "idle_waiters": len(self._idle_supply_waiters),
                    "leases": [
                        {
                            **lease.public(),
                            "release_pending": lease_id in self._pending_supply_releases,
                        }
                        for lease_id, lease in self._supply_leases.items()
                    ],
                    "stats": self._supply_status_locked(),
                    "maturity": {
                        str(generation_id): self._maturity_status_locked(generation_id)
                        for generation_id in sorted(self._generation_mature_targets)
                    },
                },
                "jobs": retained_jobs,
                "worker_error": self._worker_error,
            }

    def _supply_status_locked(self) -> dict[str, Any]:
        def with_conversion(counts: dict[str, int]) -> dict[str, int | float | None]:
            granted = int(counts.get("granted", 0))
            consumed = int(counts.get("consumed", 0))
            return {
                **counts,
                "conversion_rate": consumed / granted if granted else None,
            }

        status: dict[str, Any] = with_conversion(dict(self._supply_stats))
        status["outstanding"] = len(self._supply_leases)
        priorities = set(self._supply_stats_by_priority)
        priorities.update(lease.priority for lease in self._supply_leases.values())
        by_priority: dict[str, dict[str, int | float | None]] = {}
        for priority in sorted(priorities):
            counts = with_conversion(dict(self._supply_stats_by_priority.get(priority, {})))
            counts["outstanding"] = sum(
                lease.priority == priority for lease in self._supply_leases.values()
            )
            by_priority[priority] = counts
        status["by_priority"] = by_priority
        return status

    def _increment_supply_stat_locked(
        self,
        outcome: str,
        lease: _SupplyLease | None = None,
        *,
        priority: str = "",
    ) -> None:
        self._supply_stats[outcome] = self._supply_stats.get(outcome, 0) + 1
        priority = priority or (lease.priority if lease is not None else "")
        if not priority:
            return
        counts = self._supply_stats_by_priority.setdefault(priority, {})
        counts[outcome] = counts.get(outcome, 0) + 1

    def _worker(self) -> None:
        while True:
            with self._condition:
                retry_stop_fence = (
                    self._admission_closed
                    and not self._stopping
                    and self._server is None
                    and self._owner_lock_fd is not None
                )
                if (
                    self._stopping
                    and not self._active
                    and not self._shutdown_cleanup_pending_locked()
                    and self._transport_cleanup_complete
                ):
                    if self._finalize_owner_resources():
                        return
                    self._condition.wait(timeout=1.0)
                    continue
            if retry_stop_fence:
                try:
                    self._stop_admission_and_supply()
                except Exception as exc:  # noqa: BLE001 - retry while retaining ownership.
                    self._worker_error = f"{type(exc).__name__}: {exc}"
                    logger.warning("could not persist scheduler stop fence; retrying: %s", exc)
                with self._condition:
                    self._condition.wait(timeout=1.0)
                continue
            try:
                self._reap()
                if self._stopping:
                    self._reconcile_supply(None)
                with self._condition:
                    latent_supply_demand = bool(self._idle_supply_waiters)
                snapshot = self.allocator.refresh(queued=bool(self._queue) or latent_supply_demand)
                self._refresh_mature_counts(snapshot)
                self._launch_ready_jobs()
                self._reconcile_supply(snapshot)
                self._worker_error = ""
                self._write_snapshot()
            except Exception as exc:  # noqa: BLE001 - keep queue observable and retryable.
                self._worker_error = f"{type(exc).__name__}: {exc}"
                logger.exception("central experiment scheduler worker iteration failed")
            with self._condition:
                self._condition.wait(timeout=1.0)

    def _shutdown_cleanup_pending_locked(self) -> bool:
        if self._active:
            return True
        if self._pending_supply_releases:
            return True
        return any(job.supply_claim_id for job in self._jobs.values())

    def _refresh_mature_counts(self, snapshot: Any) -> None:
        observed_at = float(getattr(snapshot, "observed_at", 0.0) or 0.0)
        refreshed_at = time.monotonic()
        with self._condition:
            if (
                observed_at == self._maturity_sample_at
                and refreshed_at - self._maturity_refreshed_at < _MATURITY_REFRESH_SECONDS
            ):
                return
            self._maturity_sample_at = observed_at
            self._maturity_refreshed_at = refreshed_at
            callbacks = dict(self._mature_count_callbacks)
            identity_callbacks = dict(self._mature_identity_callbacks)
        updates: dict[int, int | None] = {}
        identity_updates: dict[int, set[str]] = {}
        for generation_id, callback in callbacks.items():
            try:
                identity_callback = identity_callbacks.get(generation_id)
                if identity_callback is not None:
                    identities = {str(item) for item in identity_callback() if str(item)}
                    identity_updates[generation_id] = identities
                    updates[generation_id] = len(identities)
                else:
                    updates[generation_id] = max(0, int(callback()))
            except Exception as exc:  # noqa: BLE001 - unknown is safer than invented evidence.
                logger.debug("mature evidence refresh failed for gen %d: %s", generation_id, exc)
                updates[generation_id] = None
        with self._condition:
            self._mature_completed.update(updates)
            self._mature_completed_ids.update(identity_updates)

    def _maturity_status_locked(self, generation_id: int) -> dict[str, Any]:
        target = self._generation_mature_targets.get(generation_id, 0)
        completed = self._mature_completed.get(generation_id)
        debt = max(0, target - completed) if completed is not None else None
        if generation_id in self._maturity_distinct_peers:
            completed_ids = self._mature_completed_ids.get(generation_id, set())
            mature_job_ids = {
                job.peer_id
                for job in self._jobs.values()
                if job.generation_id == generation_id
                and job.work_class == "mature"
                and job.state in {"queued", "running"}
                and job.peer_id not in completed_ids
            }
            mature_lease_ids = {
                lease.peer_id
                for lease_id, lease in self._supply_leases.items()
                if lease.generation_id == generation_id
                and lease.priority == "mature"
                and lease.peer_id not in completed_ids
                and lease_id not in self._pending_supply_releases
                and self._pending_supply_job_locked(lease_id) is None
            }
            mature_jobs = len(mature_job_ids)
            mature_leases = len(mature_lease_ids - mature_job_ids)
        else:
            mature_jobs = sum(
                job.generation_id == generation_id
                and job.work_class == "mature"
                and job.state in {"queued", "running"}
                for job in self._jobs.values()
            )
            mature_leases = sum(
                lease.generation_id == generation_id
                and lease.priority == "mature"
                and lease_id not in self._pending_supply_releases
                and self._pending_supply_job_locked(lease_id) is None
                for lease_id, lease in self._supply_leases.items()
            )
        cohort_size = self._generation_cohort_sizes.get(generation_id, 0)
        target_inflight = (
            min(
                cohort_size,
                int(math.ceil(self.settings.mature_supply_redundancy * debt)),
            )
            if debt is not None and debt > 0 and cohort_size > 0
            else 0
        )
        commitments = mature_jobs + mature_leases
        return {
            "target": target,
            "completed": completed,
            "debt": debt,
            "target_inflight": target_inflight,
            "inflight_jobs": mature_jobs,
            "priority_leases": mature_leases,
            "needed_inflight": max(0, target_inflight - commitments),
        }

    def _reconcile_supply(self, snapshot: Any) -> None:
        with self._condition:
            now = time.time()
            self._consumed_supply_leases = {
                lease_id: consumed
                for lease_id, consumed in self._consumed_supply_leases.items()
                if consumed[0] > now
            }
            for lease_id, lease in list(self._supply_leases.items()):
                if lease_id in self._pending_supply_releases:
                    self._remove_supply_lease_locked(
                        lease_id,
                        outcome="revoked",
                        reason="retry_pending_release",
                    )
                    continue
                pending_job = self._pending_supply_job_locked(lease_id)
                if pending_job is not None:
                    if pending_job.state in {
                        "completed",
                        "failed",
                        "rejected",
                        "drained_unknown",
                    }:
                        continue
                    try:
                        self._consume_pending_supply_locked(
                            lease_id,
                            reason="accepted_submission_audit_retried",
                        )
                    except OSError:
                        continue
                    continue
                claim_valid = getattr(self.allocator, "supply_claim_valid", None)
                if lease.expires_at <= now:
                    self._remove_supply_lease_locked(
                        lease_id,
                        outcome="expired",
                        reason="response_window_elapsed",
                    )
                elif callable(claim_valid) and not claim_valid(lease_id):
                    self._remove_supply_lease_locked(
                        lease_id,
                        outcome="revoked",
                        reason="resource_claim_reclaimed",
                    )
            for job in self._jobs.values():
                if job.supply_claim_id and (
                    self._stopping
                    or job.state in {"completed", "failed", "rejected", "drained_unknown"}
                ):
                    claim_id = job.supply_claim_id
                    if claim_id in self._supply_leases:
                        try:
                            self._consume_pending_supply_locked(
                                claim_id,
                                reason="terminal_submission_audit_retried",
                            )
                        except OSError:
                            continue
                        job.supply_claim_id = claim_id
                    try:
                        self.allocator.release(claim_id)
                    except Exception as exc:  # noqa: BLE001 - retain for next reconcile.
                        logger.warning(
                            "could not release terminal job supply claim %s: %s",
                            claim_id,
                            exc,
                        )
                        job.supply_claim_id = claim_id
                    else:
                        job.supply_claim_id = ""
            for generation_id in self._generation_mature_targets:
                if generation_id in self._maturity_distinct_peers:
                    completed_ids = self._mature_completed_ids.get(generation_id, set())
                    for lease in list(self._supply_leases.values()):
                        if (
                            lease.generation_id == generation_id
                            and lease.priority == "mature"
                            and lease.peer_id in completed_ids
                        ):
                            self._remove_supply_lease_locked(
                                lease.lease_id,
                                outcome="revoked",
                                reason="mature_peer_completed",
                            )
                maturity = self._maturity_status_locked(generation_id)
                excess = max(
                    0,
                    maturity["inflight_jobs"]
                    + maturity["priority_leases"]
                    - maturity["target_inflight"],
                )
                removable = sorted(
                    (
                        lease
                        for lease in self._supply_leases.values()
                        if lease.generation_id == generation_id and lease.priority == "mature"
                    ),
                    key=lambda lease: lease.issued_at,
                    reverse=True,
                )
                for lease in removable[:excess]:
                    self._remove_supply_lease_locked(
                        lease.lease_id,
                        outcome="revoked",
                        reason="mature_demand_satisfied",
                    )
            if not self.settings.supply_signal_enabled or not self._idle_supply_waiters:
                self._supply_idle_samples = 0
                return

            headroom = getattr(self.allocator, "has_supply_headroom", None)
            if callable(headroom) and not any(
                headroom(snapshot, domains=set(profile.pressure_domains))
                for profile in self.settings.profiles.values()
            ):
                self._supply_idle_samples = 0
                return

            observed_at = float(getattr(snapshot, "observed_at", 0.0) or 0.0)
            if observed_at == self._supply_sample_at:
                return
            self._supply_sample_at = observed_at
            self._supply_idle_samples += 1
            if self._supply_idle_samples < self.settings.supply_idle_samples:
                return

            candidates = sorted(
                self._idle_supply_waiters.items(),
                key=lambda item: (item[1][1], item[0]),
            )
            fallback_slots = max(
                0,
                self.allocator.concurrency_limit - len(self._active) - len(self._supply_leases),
            )
            claim_supply = getattr(self.allocator, "claim_supply", None)
            for peer_id, (generation_id, _registered_at) in candidates:
                deadline = self._generation_deadlines.get(generation_id, 0.0)
                if (
                    generation_id in self._frozen_generations
                    or self._generation_signal_exists(generation_id)
                    or (deadline is not None and deadline <= now)
                ):
                    self._idle_supply_waiters.pop(peer_id, None)
                    continue
                maturity = self._maturity_status_locked(generation_id)
                if generation_id in self._assessment_generations:
                    if maturity["needed_inflight"] <= 0:
                        continue
                    priority = "mature"
                else:
                    priority = "mature" if maturity["needed_inflight"] > 0 else "frontier_followup"
                if (
                    priority == "mature"
                    and generation_id in self._maturity_distinct_peers
                    and peer_id in self._mature_completed_ids.get(generation_id, set())
                ):
                    continue
                declined = self._declined_supply_peers.get((peer_id, generation_id, priority))
                if declined is not None and declined[0] > now:
                    continue
                lease_id = f"supply-{uuid.uuid4().hex}"
                expires_at = now + float(self.settings.supply_lease_seconds)
                if callable(claim_supply):
                    claimed_profiles = claim_supply(
                        lease_id=lease_id,
                        run_id=self.supply_owner_id,
                        expires_at=expires_at,
                    )
                    admissible_profiles = (
                        tuple(str(item) for item in claimed_profiles)
                        if isinstance(claimed_profiles, (list, tuple, set))
                        else ()
                    )
                elif fallback_slots > 0:
                    admissible_profiles = tuple(sorted(self.settings.profiles))
                    fallback_slots -= 1
                else:
                    admissible_profiles = ()
                if not admissible_profiles:
                    break
                lease = _SupplyLease(
                    lease_id=lease_id,
                    peer_id=peer_id,
                    generation_id=generation_id,
                    admissible_profiles=admissible_profiles,
                    priority=priority,
                    issued_at=now,
                    expires_at=expires_at,
                )
                self._supply_leases[lease.lease_id] = lease
                try:
                    self._append_event(
                        {
                            "event": "supply_granted",
                            "lease_id": lease.lease_id,
                            "generation_id": generation_id,
                            "peer_id": peer_id,
                            "priority": priority,
                            "admissible_profiles": list(admissible_profiles),
                            "issued_at": lease.issued_at,
                            "expires_at": lease.expires_at,
                        },
                        required=True,
                    )
                except Exception:
                    try:
                        self.allocator.release(lease.lease_id)
                    except Exception as release_exc:  # noqa: BLE001 - retry during reconcile.
                        self._pending_supply_releases[lease.lease_id] = (
                            "revoked",
                            "grant_audit_failed",
                            None,
                            False,
                        )
                        logger.warning(
                            "could not release unaudited supply claim %s; retrying during "
                            "reconcile: %s",
                            lease.lease_id,
                            release_exc,
                        )
                    else:
                        self._supply_leases.pop(lease.lease_id, None)
                    raise
                self._increment_supply_stat_locked("granted", lease)
                try:
                    self._write_supply_signal(lease)
                except Exception:
                    try:
                        self._remove_supply_lease_locked(
                            lease.lease_id,
                            outcome="revoked",
                            reason="signal_write_failed",
                        )
                    except Exception as cleanup_exc:
                        self._pending_supply_releases[lease.lease_id] = (
                            "revoked",
                            "signal_write_failed",
                            None,
                            False,
                        )
                        try:
                            self._unlink_supply_signal(lease)
                        except OSError as unlink_exc:
                            logger.warning(
                                "could not remove failed supply signal %s: %s",
                                lease.lease_id,
                                unlink_exc,
                            )
                        try:
                            self.allocator.release(lease.lease_id)
                        except Exception as release_exc:  # noqa: BLE001 - retry in reconcile.
                            logger.warning(
                                "could not release failed supply claim %s; retrying during "
                                "reconcile: %s",
                                lease.lease_id,
                                release_exc,
                            )
                        else:
                            self._pending_supply_releases.pop(lease.lease_id, None)
                            removed = self._supply_leases.pop(lease.lease_id, None)
                            if removed is not None:
                                self._increment_supply_stat_locked("revoked", removed)
                        logger.warning(
                            "supply signal %s failed and its terminal audit was unavailable: %s",
                            lease.lease_id,
                            cleanup_exc,
                        )
                    raise
                self._idle_supply_waiters.pop(peer_id, None)
            self._supply_idle_samples = 0

    def _supply_path(self, lease: _SupplyLease) -> Path:
        return resource_supply_signal_path(
            self.run_dir / f"gen_{lease.generation_id}",
            lease.peer_id,
        )

    @staticmethod
    def _valid_supply_peer_id(peer_id: str, generation_id: int) -> bool:
        return bool(
            len(peer_id) <= 128 and re.fullmatch(rf"gen{generation_id}_peer[0-9]+", peer_id)
        )

    def _open_supply_dir(self, generation_id: int, *, create: bool) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        run_fd = os.open(self.run_dir, flags)
        gen_fd = -1
        try:
            gen_name = f"gen_{generation_id}"
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(gen_name, mode=0o755, dir_fd=run_fd)
            gen_fd = os.open(gen_name, flags, dir_fd=run_fd)
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir("resource_supply", mode=0o700, dir_fd=gen_fd)
            return os.open("resource_supply", flags, dir_fd=gen_fd)
        finally:
            if gen_fd >= 0:
                os.close(gen_fd)
            os.close(run_fd)

    def _write_supply_signal(self, lease: _SupplyLease) -> None:
        directory_fd = self._open_supply_dir(lease.generation_id, create=True)
        filename = self._supply_path(lease).name
        temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
        fd = -1
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            payload = json.dumps(lease.locator(), indent=2).encode("utf-8")
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, filename, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(temporary, dir_fd=directory_fd)
            os.close(directory_fd)

    def _unlink_supply_signal(self, lease: _SupplyLease) -> None:
        try:
            directory_fd = self._open_supply_dir(lease.generation_id, create=False)
        except OSError:
            return
        try:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self._supply_path(lease).name, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)

    def _remove_supply_lease_locked(
        self,
        lease_id: str,
        *,
        outcome: str,
        reason: str,
        release_claim: bool = True,
        event_details: dict[str, Any] | None = None,
    ) -> None:
        lease = self._supply_leases.get(lease_id)
        if lease is not None:
            pending_release = self._pending_supply_releases.get(lease_id)
            if pending_release is not None and pending_release[1] == "grant_audit_failed":
                try:
                    self.allocator.release(lease_id)
                except Exception as exc:  # noqa: BLE001 - retain for the next reconcile.
                    logger.warning(
                        "could not release unaudited supply claim %s; retrying during "
                        "reconcile: %s",
                        lease_id,
                        exc,
                    )
                    return
                self._pending_supply_releases.pop(lease_id, None)
                self._supply_leases.pop(lease_id, None)
                return
            terminal_recorded = bool(pending_release and pending_release[3])
            if pending_release is not None:
                outcome, reason, event_details, _recorded = pending_release
                release_claim = True
            pending_job = self._pending_supply_job_locked(lease_id)
            if not terminal_recorded and pending_job is not None and outcome != "consumed":
                original_outcome = outcome
                outcome = "consumed"
                reason = f"accepted_submission_before_{reason}"
                release_claim = False
                event_details = {
                    **(event_details or {}),
                    "profile": pending_job.profile,
                    "superseded_outcome": original_outcome,
                }
            if not terminal_recorded:
                event = {
                    "event": f"supply_{outcome}",
                    "lease_id": lease_id,
                    "generation_id": lease.generation_id,
                    "peer_id": lease.peer_id,
                    "priority": lease.priority,
                    "reason": reason,
                }
                if event_details:
                    event.update(event_details)
                try:
                    self._append_event(event, required=True)
                except OSError:
                    if release_claim:
                        self._pending_supply_releases[lease_id] = (
                            outcome,
                            reason,
                            event_details,
                            False,
                        )
                        try:
                            self._unlink_supply_signal(lease)
                        except OSError as unlink_exc:
                            logger.warning(
                                "could not remove unaudited terminal supply signal %s: %s",
                                lease_id,
                                unlink_exc,
                            )
                    raise
            try:
                self._unlink_supply_signal(lease)
            except OSError as exc:
                logger.warning("could not remove terminal supply signal %s: %s", lease_id, exc)
            if release_claim:
                try:
                    self.allocator.release(lease_id)
                except Exception as exc:  # noqa: BLE001 - retain retryable terminal state.
                    self._pending_supply_releases[lease_id] = (
                        outcome,
                        reason,
                        event_details,
                        True,
                    )
                    logger.warning(
                        "could not release terminal supply claim %s; retrying during reconcile: %s",
                        lease_id,
                        exc,
                    )
                    return
            self._pending_supply_releases.pop(lease_id, None)
            self._supply_leases.pop(lease_id, None)
            self._increment_supply_stat_locked(outcome, lease)
            if pending_job is not None and outcome == "consumed":
                self._consumed_supply_leases[lease_id] = (float("inf"), lease.priority)
                if pending_job.state != "queued":
                    pending_job.supply_claim_id = ""

    def _consume_or_revoke_peer_supply_locked(
        self,
        *,
        peer_id: str,
        generation_id: int,
        profile: str,
        lease_id: str,
        work_class: str,
    ) -> str:
        for current_id, lease in list(self._supply_leases.items()):
            if lease.peer_id != peer_id:
                continue
            if (
                current_id == lease_id
                and lease.generation_id == generation_id
                and lease.expires_at > time.time()
            ):
                priority_matches = lease.priority != "mature" or work_class == "mature"
                if profile in lease.admissible_profiles and priority_matches:
                    self._remove_supply_lease_locked(
                        current_id,
                        outcome="consumed",
                        reason="experiment_submitted",
                        release_claim=False,
                        event_details={"profile": profile},
                    )
                    self._consumed_supply_leases[current_id] = (
                        float("inf"),
                        lease.priority,
                    )
                    return current_id
                reason = (
                    "work_class_not_mature" if not priority_matches else "profile_not_admissible"
                )
                self._remove_supply_lease_locked(
                    current_id,
                    outcome="revoked",
                    reason=reason,
                    event_details={
                        "profile": profile,
                        "submitted_work_class": work_class,
                    },
                )
                return ""
            outcome = "expired" if lease.expires_at <= time.time() else "revoked"
            reason = (
                "response_window_elapsed"
                if outcome == "expired"
                else "submission_without_matching_lease"
            )
            self._remove_supply_lease_locked(
                current_id,
                outcome=outcome,
                reason=reason,
            )
        return ""

    def _revoke_supply_locked(self, generation_id: int | None = None, *, reason: str) -> None:
        self._idle_supply_waiters = {
            peer_id: waiter
            for peer_id, waiter in self._idle_supply_waiters.items()
            if generation_id is not None and waiter[0] != generation_id
        }
        for lease_id, lease in list(self._supply_leases.items()):
            if generation_id is None or lease.generation_id == generation_id:
                pending_job = self._pending_supply_job_locked(lease_id)
                try:
                    if pending_job is not None and self._consume_pending_supply_locked(
                        lease_id,
                        reason=f"accepted_submission_before_{reason}",
                    ):
                        continue
                    self._remove_supply_lease_locked(
                        lease_id,
                        outcome="revoked",
                        reason=reason,
                    )
                except OSError as exc:
                    logger.warning(
                        "could not persist supply revocation for %s; "
                        "releasing the operational claim: %s",
                        lease_id,
                        exc,
                    )
                    if lease_id in self._pending_supply_releases:
                        try:
                            self.allocator.release(lease_id)
                        except Exception as release_exc:  # noqa: BLE001 - retry in reconcile.
                            logger.warning(
                                "could not release unaudited supply claim %s; retrying during "
                                "reconcile: %s",
                                lease_id,
                                release_exc,
                            )
                        else:
                            outcome, _reason, _details, _recorded = (
                                self._pending_supply_releases.pop(lease_id)
                            )
                            self._supply_leases.pop(lease_id, None)
                            self._increment_supply_stat_locked(outcome, lease)
                        continue
                    outcome = "consumed" if pending_job is not None else "revoked"
                    if pending_job is not None:
                        self._supply_leases.pop(lease_id, None)
                        self._increment_supply_stat_locked(outcome, lease)
                        self._consumed_supply_leases[lease_id] = (
                            float("inf"),
                            lease.priority,
                        )
                        if pending_job.state == "running":
                            pending_job.supply_claim_id = ""
                    try:
                        self._unlink_supply_signal(lease)
                    except OSError as unlink_exc:
                        logger.warning(
                            "could not remove supply signal %s after audit failure: %s",
                            lease_id,
                            unlink_exc,
                        )
                    if pending_job is None:
                        self._pending_supply_releases[lease_id] = (
                            "revoked",
                            reason,
                            None,
                            False,
                        )
                        try:
                            self.allocator.release(lease_id)
                        except Exception as release_exc:  # noqa: BLE001 - retry in reconcile.
                            logger.warning(
                                "could not release unaudited supply claim %s; retrying during "
                                "reconcile: %s",
                                lease_id,
                                release_exc,
                            )
                        else:
                            self._pending_supply_releases.pop(lease_id, None)
                            self._supply_leases.pop(lease_id, None)
                            self._increment_supply_stat_locked(outcome, lease)
        self._supply_idle_samples = 0
        self._supply_sample_at = 0.0

    def _pending_supply_job_locked(self, lease_id: str) -> _ExperimentJob | None:
        return next(
            (job for job in self._jobs.values() if job.supply_claim_id == lease_id),
            None,
        )

    def _consume_pending_supply_locked(self, lease_id: str, *, reason: str) -> bool:
        pending_job = self._pending_supply_job_locked(lease_id)
        lease = self._supply_leases.get(lease_id)
        if pending_job is None or lease is None:
            return False
        self._remove_supply_lease_locked(
            lease_id,
            outcome="consumed",
            reason=reason,
            release_claim=False,
            event_details={"profile": pending_job.profile},
        )
        self._consumed_supply_leases[lease_id] = (float("inf"), lease.priority)
        if pending_job.state == "running":
            pending_job.supply_claim_id = ""
        return True

    def _remove_stale_supply_files(self) -> None:
        for gen_dir in self.run_dir.glob("gen_*"):
            match = re.fullmatch(r"gen_([0-9]+)", gen_dir.name)
            if match is None:
                continue
            try:
                directory_fd = self._open_supply_dir(int(match.group(1)), create=False)
            except OSError:
                continue
            try:
                for name in os.listdir(directory_fd):
                    if not name.endswith(".json"):
                        continue
                    try:
                        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISREG(metadata.st_mode):
                        with contextlib.suppress(OSError):
                            os.unlink(name, dir_fd=directory_fd)
            finally:
                os.close(directory_fd)

    def _launch_ready_jobs(self) -> None:
        while True:
            with self._condition:
                if (
                    self._stopping
                    or self._admission_closed
                    or len(self._active) >= self.allocator.concurrency_limit
                ):
                    return
                candidates = self._ordered_queue()
            launched = False
            for job_id in candidates:
                with self._condition:
                    if job_id not in self._queue:
                        continue
                    job = self._jobs[job_id]
                    if job.state != "queued":
                        self._queue.remove(job_id)
                        continue
                    if job_id in self._active:
                        # A completed attempt can remain here briefly while its resource
                        # release is retried. Do not launch the queued retry concurrently.
                        continue
                    if (
                        job.generation_id in self._assessment_generations
                        and job.work_class != "mature"
                    ):
                        self._queue.remove(job_id)
                        self._finish_without_launch(job, "generation_assessment")
                        continue
                    if (
                        job.generation_id in self._frozen_generations
                        or self._generation_signal_exists(job.generation_id)
                    ):
                        self._frozen_generations.add(job.generation_id)
                        self._queue.remove(job_id)
                        self._finish_without_launch(job, "generation_closing")
                        continue
                    if not self._fits_generation(job):
                        self._queue.remove(job_id)
                        self._finish_without_launch(job, "insufficient_generation_time")
                        continue
                    if not self._peer_has_launch_capacity(job):
                        continue
                    accelerator_block = self._accelerator_block_reason(
                        job,
                        allow_transient=False,
                    )
                    if accelerator_block:
                        self._queue.remove(job_id)
                        self._finish_without_launch(job, accelerator_block)
                        continue
                    if self._launch(job):
                        launched = True
                        break
            if not launched:
                return

    def _ordered_queue(self) -> list[str]:
        priorities = {"mature": 0, "ordinary": 1, "scout": 2}
        non_mature_active = sum(
            active.job.work_class != "mature" for active in self._active.values()
        )
        reserve_needed = max(0, self.settings.exploration_reserve - non_mature_active)

        def priority(job_id: str) -> tuple[int, float]:
            job = self._jobs[job_id]
            if reserve_needed and job.work_class != "mature":
                return (-1, job.submitted_at)
            return (priorities[job.work_class], job.submitted_at)

        return sorted(
            self._queue,
            key=priority,
        )

    def _peer_has_launch_capacity(self, job: _ExperimentJob) -> bool:
        limit = self.max_parallel_runs_per_peer
        if limit is None:
            return True
        active = sum(item.job.peer_id == job.peer_id for item in self._active.values())
        return active < limit

    def _accelerator_block_reason(
        self,
        job: _ExperimentJob,
        *,
        allow_transient: bool,
    ) -> str:
        profile = self._require_profile(job.profile)
        if not profile.needs_gpu:
            return ""
        snapshot = self.allocator.snapshot
        if snapshot.gpus:
            self._accelerator_probe_unknown_since = None
            if profile.gpu_count > len(snapshot.gpus):
                return (
                    "accelerator_profile_unsatisfied: "
                    f"profile {profile.name!r} requests {profile.gpu_count} GPUs, "
                    f"but the host exposes {len(snapshot.gpus)}"
                )
            known_capacities = [
                device.memory_total_mb for device in snapshot.gpus if device.memory_total_mb > 0
            ]
            if profile.gpu_memory_gb is not None and len(known_capacities) == len(snapshot.gpus):
                requested_mb = int(profile.gpu_memory_gb * 1024)
                capable_devices = sum(
                    requested_mb <= capacity_mb * 0.95 for capacity_mb in known_capacities
                )
                if capable_devices < profile.gpu_count:
                    return (
                        "accelerator_profile_unsatisfied: "
                        f"profile {profile.name!r} requests {profile.gpu_memory_gb:g} GiB "
                        f"on each of {profile.gpu_count} GPUs, but only "
                        f"{capable_devices} detected devices can satisfy it"
                    )
            return ""
        state = str(snapshot.accelerator_probe_state or "unknown")
        reason = str(snapshot.accelerator_probe_reason or "accelerator inventory unavailable")
        if state in {"unavailable", "unsupported"}:
            self._accelerator_probe_unknown_since = None
            return f"accelerator_{state}: {reason}"
        now = time.time()
        if self._accelerator_probe_unknown_since is None:
            self._accelerator_probe_unknown_since = now
        grace = min(
            _ACCELERATOR_PROBE_GRACE_MAX_SECONDS,
            max(10.0, self.settings.adjustment_interval_seconds * 3.0),
        )
        if allow_transient or now - self._accelerator_probe_unknown_since < grace:
            return ""
        return f"accelerator_probe_unknown: {reason}"

    def _queue_block_reason(self, job: _ExperimentJob) -> str:
        if not self._peer_has_launch_capacity(job):
            return "per_peer_capacity"
        profile = self._require_profile(job.profile)
        snapshot = self.allocator.snapshot
        if profile.needs_gpu and not snapshot.gpus:
            state = str(snapshot.accelerator_probe_state or "unknown")
            return f"accelerator_{state}"
        if len(self._active) >= self.allocator.concurrency_limit:
            return "concurrency_limit"
        return "resource_pressure_or_capacity"

    def _launch(self, job: _ExperimentJob) -> bool:
        profile = self._require_profile(job.profile)
        allocation_id = f"{self.resource_owner_id}:{job.job_id}:{job.attempts + 1}"
        supply_claim_id = job.supply_claim_id
        allocation = self.allocator.reserve(
            allocation_id=allocation_id,
            run_id=self.resource_owner_id,
            pid=os.getpid(),
            pgid=os.getpgrp(),
            profile=profile,
            supply_claim_id=supply_claim_id,
        )
        if allocation is None:
            return False
        if supply_claim_id in self._supply_leases:
            with contextlib.suppress(OSError):
                self._consume_pending_supply_locked(
                    supply_claim_id,
                    reason="accepted_submission_audit_retried_at_launch",
                )
        if supply_claim_id not in self._supply_leases:
            job.supply_claim_id = ""
        if job.job_id in self._queue:
            self._queue.remove(job.job_id)
        job.attempts += 1
        attempt_id = f"{job.job_id}-a{job.attempts}"
        log_path = self.logs_dir / f"{attempt_id}.log"
        attempt_dir = self.state_dir / "attempts" / attempt_id
        ready_path = attempt_dir / "READY.json"
        go_path = attempt_dir / "GO.json"
        log_handle: Any | None = None
        process: subprocess.Popen[bytes] | None = None
        stage = "launch_intent"
        try:
            self._append_event(
                {
                    "event": "launch_intent",
                    **self._event_identity(job, include_request=True),
                    "allocation_id": allocation_id,
                    "gpu_uuids": list(allocation.gpu_uuids),
                    "attempt_id": attempt_id,
                    "attempt_dir": str(attempt_dir),
                    "log_path": str(log_path),
                },
                required=True,
            )
            stage = "attempt_directory"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            stage = "log_open"
            log_handle = open(log_path, "ab", buffering=0)  # noqa: SIM115 - owned until reap.
            env = dict(job.environment)
            if allocation.gpu_uuids:
                mask = ",".join(allocation.gpu_uuids)
                env["CUDA_VISIBLE_DEVICES"] = mask
                env["NVIDIA_VISIBLE_DEVICES"] = mask
            elif profile.accelerator == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["NVIDIA_VISIBLE_DEVICES"] = ""
            env.update(
                {
                    "PRAXIST_EXPERIMENT_ID": job.experiment_id,
                    "PRAXIST_EXPERIMENT_ATTEMPT_ID": attempt_id,
                    "PRAXIST_EXPERIMENT_ATTEMPT_DIR": str(attempt_dir),
                    "PRAXIST_RESOURCE_PROFILE": profile.name,
                    "PRAXIST_ASSIGNED_GPU_UUIDS": ",".join(allocation.gpu_uuids),
                }
            )
            stage = "process_start"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-E",
                    "-S",
                    str(Path(__file__).with_name("experiment_exec.py")),
                    str(ready_path),
                    str(go_path),
                    attempt_id,
                    *job.command,
                ],
                cwd=job.cwd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._append_event(
                {
                    "event": "process_started",
                    **self._event_identity(job, include_request=True),
                    "pid": process.pid,
                    "pgid": process.pid,
                    "pid_start_time": _pid_start_time(process.pid),
                    "attempt_id": attempt_id,
                    "attempt_dir": str(attempt_dir),
                    "log_path": str(log_path),
                },
                required=True,
            )
            stage = "resource_binding"
            if not self.allocator.bind_process(allocation_id, pid=process.pid, pgid=process.pid):
                raise RuntimeError("reserved allocation disappeared before process binding")
            allocation = Allocation(
                **{**asdict(allocation), "pid": process.pid, "pgid": process.pid}
            )
            job.state = "running"
            job.pid = process.pid
            job.pgid = process.pid
            job.gpu_uuids = list(allocation.gpu_uuids)
            job.log_path = str(log_path)
            job.binding_status = "assigned" if allocation.gpu_uuids else "not_applicable"
            stage = "protected_manifest"
            self._register_protected(job)
            stage = "launch_commit"
            self._append_event(
                {
                    "event": "launched",
                    **self._event_identity(job),
                    "pid": job.pid,
                    "pgid": job.pgid,
                    "pid_start_time": _pid_start_time(job.pid),
                    "attempt_id": attempt_id,
                    "gpu_uuids": job.gpu_uuids,
                    "accelerator_delivery": job.binding_status,
                    "started_at": time.time(),
                    "log_path": str(log_path),
                    "attempt_dir": str(attempt_dir),
                },
                required=True,
            )
            atomic_write_json(go_path, {"go": True, "recorded_at": time.time()})
        except Exception as exc:  # noqa: BLE001 - compensate every partial launch.
            if process is not None:
                terminate_process_group(process.pid, process)
                with contextlib.suppress(Exception):  # partial manifest may not exist.
                    self._unregister_protected(job)
            if log_handle is not None:
                with contextlib.suppress(OSError):
                    log_handle.close()
            try:
                self.allocator.release(allocation_id)
            except Exception as release_exc:  # noqa: BLE001 - preserve terminal transition.
                logger.error(
                    "could not release failed allocation %s: %s", allocation_id, release_exc
                )
            job.error = f"{stage}_error:{type(exc).__name__}:{exc}"
            self._append_event(
                {
                    "event": "launch_failed",
                    "job_id": job.job_id,
                    "attempt": job.attempts,
                    "attempt_dir": str(attempt_dir),
                    "stage": stage,
                    "error": job.error,
                }
            )
            job.state = "queued"
            job.pid = 0
            job.pgid = 0
            job.gpu_uuids = []
            self._retry_or_finish(job, INFRASTRUCTURE_RETRY_EXIT_CODE)
            return True
        with self._condition:
            from .protected_pids import _pid_start_time as protected_pid_start_time

            self._active[job.job_id] = _ActiveJob(
                job,
                process,
                allocation,
                log_handle,
                attempt_dir,
                protected_pid_start_time(process.pid),
            )
            self._condition.notify_all()
        return True

    def _reap(self) -> None:
        for job_id, active in list(self._active.items()):
            completed_pid = active.job.pid
            code = active.process.poll() if active.process is not None else None
            if active.process is not None and code is None:
                continue
            if process_group_alive(active.job.pgid):
                continue
            with self._condition:
                if active.job.state == "running":
                    if active.process is None:
                        self._append_event(
                            {
                                "event": "completed",
                                **self._event_identity(active.job),
                                "state": "drained_unknown",
                                "exit_code": None,
                                "attempt_dir": "",
                            },
                            required=True,
                        )
                        active.job.state = "drained_unknown"
                        active.job.completed_at = time.time()
                        active.job.exit_code = None
                    else:
                        assert code is not None
                        if code == 0 and active.allocation is not None:
                            self._observe_runtime(
                                active.job,
                                started_at=active.allocation.started_at,
                            )
                        self._retry_or_finish(
                            active.job,
                            int(code),
                            attempt_dir=str(active.attempt_dir) if active.attempt_dir else "",
                        )
                if active.log_handle is not None:
                    with contextlib.suppress(OSError):
                        active.log_handle.close()
                if active.allocation is not None:
                    try:
                        self.allocator.release(active.allocation.allocation_id)
                    except Exception as exc:  # noqa: BLE001 - retry cleanup on the next reap.
                        logger.warning(
                            "could not release completed experiment allocation %s: %s",
                            active.allocation.allocation_id,
                            exc,
                        )
                        self._condition.notify_all()
                        continue
                try:
                    self._unregister_protected(active.job, pid=completed_pid)
                except Exception as exc:  # noqa: BLE001 - terminal event is already durable.
                    logger.warning("could not unregister completed experiment %s: %s", job_id, exc)
                self._active.pop(job_id, None)
                self._condition.notify_all()

    def _retry_or_finish(
        self, job: _ExperimentJob, exit_code: int, *, attempt_dir: str = ""
    ) -> None:
        failure = self._failure_details(job, exit_code) if exit_code != 0 else {}
        retryable = exit_code == INFRASTRUCTURE_RETRY_EXIT_CODE
        can_retry = (
            retryable
            and job.attempts <= self.settings.infrastructure_retries
            and job.generation_id not in self._frozen_generations
            and (
                job.generation_id not in self._assessment_generations or job.work_class == "mature"
            )
            and self._fits_generation(job)
            and not self._stopping
        )
        if can_retry:
            try:
                self._append_event(
                    {
                        "event": "retry_queued",
                        **self._event_identity(job, include_request=True),
                        "exit_code": exit_code,
                        "attempt_dir": attempt_dir,
                        **failure,
                    },
                    required=True,
                )
                job.state = "queued"
                job.pid = 0
                job.pgid = 0
                job.gpu_uuids = []
                if job.job_id not in self._queue:
                    self._queue.append(job.job_id)
            finally:
                if (
                    job.state == "queued"
                    and job.job_id not in self._active
                    and job.job_id not in self._queue
                ):
                    self._queue.append(job.job_id)
                with self._condition:
                    self._condition.notify_all()
            return
        completed_at = time.time()
        state = "completed" if exit_code == 0 else "failed"
        terminal_details = failure if state == "failed" else {}
        terminal_error = (
            {"error": terminal_details["failure_signature"]} if terminal_details else {}
        )
        try:
            self._append_event(
                {
                    "event": "completed",
                    **self._event_identity(job),
                    "state": state,
                    "exit_code": exit_code,
                    "attempt_dir": attempt_dir,
                    **terminal_details,
                    **terminal_error,
                },
                required=True,
            )
            job.exit_code = exit_code
            job.completed_at = completed_at
            job.state = state
            if terminal_details:
                job.failure_category = str(terminal_details["failure_category"])
                job.failure_signature = str(terminal_details["failure_signature"])
                job.failure_log_tail = str(terminal_details["failure_log_tail"])
                job.error = job.failure_signature
            else:
                job.failure_category = ""
                job.failure_signature = ""
                job.failure_log_tail = ""
                job.error = ""
        finally:
            if (
                job.state == "queued"
                and job.job_id not in self._active
                and job.job_id not in self._queue
            ):
                self._queue.append(job.job_id)
            with self._condition:
                self._condition.notify_all()

    def _failure_details(self, job: _ExperimentJob, exit_code: int) -> dict[str, str]:
        tail = ""
        if job.log_path:
            try:
                with open(job.log_path, "rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - _FAILURE_LOG_TAIL_BYTES))
                    tail = handle.read(_FAILURE_LOG_TAIL_BYTES).decode("utf-8", errors="replace")
            except OSError:
                pass
        redacted_tail, _hits = redact_text(tail[-_FAILURE_LOG_TAIL_CHARS:])
        redacted_tail = redacted_tail[-_FAILURE_LOG_TAIL_CHARS:]
        fallback, _hits = redact_text(job.error[-_FAILURE_LOG_TAIL_CHARS:])
        lines = [line.strip() for line in redacted_tail.splitlines() if line.strip()]
        signature = (lines[-1] if lines else fallback) or f"process exited with code {exit_code}"
        signature = signature[-_FAILURE_SIGNATURE_CHARS:]
        evidence = f"{fallback}\n{redacted_tail}".lower()
        return {
            "failure_category": _classify_failure(exit_code, evidence),
            "failure_signature": signature,
            "failure_log_tail": redacted_tail,
        }

    def _event_identity(
        self, job: _ExperimentJob, *, include_request: bool = False
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "experiment_id": job.experiment_id,
            "generation_id": job.generation_id,
            "peer_id": job.peer_id,
            "attempt": job.attempts,
            "profile": job.profile,
            "work_class": job.work_class,
            "eta_seconds": job.eta_seconds,
            "cwd": job.cwd,
        }
        if include_request:
            payload["command"] = self._durable_command(job.command, job.environment)
            payload["environment_values"] = {
                key: value
                for key, value in job.environment.items()
                if not is_sensitive_environment_entry(key, value)
            }
            payload["environment_sensitive_hashes"] = {
                key: hashlib.sha256(value.encode("utf-8")).hexdigest()
                for key, value in job.environment.items()
                if is_sensitive_environment_entry(key, value)
            }
            payload["submitted_at"] = job.submitted_at
        return payload

    @staticmethod
    def _durable_command(command: list[str], environment: dict[str, str]) -> list[str]:
        sensitive_values = sorted(
            (
                (value, key)
                for key, value in environment.items()
                if value and is_sensitive_environment_entry(key, value)
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        durable: list[str] = []
        previous = ""
        for raw_argument in command:
            argument = raw_argument
            for value, key in sensitive_values:
                if argument == value or (len(value) >= 8 and value in argument):
                    argument = argument.replace(value, f"__PRAXIST_ENV_REF_{key}__")
            option = previous.lstrip("-").replace("-", "_").lower()
            current_option, separator, current_value = argument.partition("=")
            current_name = current_option.lstrip("-").replace("-", "_").lower()
            sensitive_option_names = {
                "api_key",
                "auth",
                "bearer",
                "credential",
                "password",
                "passwd",
                "pat",
                "pwd",
                "secret",
                "token",
            }
            previous_sensitive = option in sensitive_option_names or (
                option and is_sensitive_environment_name(f"OPTION_{option}")
            )
            current_sensitive = current_name in sensitive_option_names or (
                current_name and is_sensitive_environment_name(f"OPTION_{current_name}")
            )
            referenced = bool(_ENV_REFERENCE_RE.search(argument))
            if (
                (previous_sensitive and not referenced)
                or (separator and current_sensitive and current_value and not referenced)
                or (is_sensitive_environment_entry("COMMAND_ARGUMENT", argument) and not referenced)
            ):
                raise ExperimentRejected(
                    "credential-bearing command arguments are not durable; pass the value "
                    "through a credential-named environment variable"
                )
            durable.append(argument)
            previous = raw_argument
        return durable

    @staticmethod
    def _restore_durable_command(command: list[str], environment: dict[str, str]) -> list[str]:
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in environment:
                raise RuntimeError(f"credential environment {key!r} is unavailable during recovery")
            return environment[key]

        return [_ENV_REFERENCE_RE.sub(replace, argument) for argument in command]

    def _finish_without_launch(self, job: _ExperimentJob, reason: str) -> None:
        if job.supply_claim_id in self._supply_leases:
            lease = self._supply_leases[job.supply_claim_id]
            try:
                self._consume_pending_supply_locked(
                    job.supply_claim_id,
                    reason=f"accepted_submission_before_{reason}",
                )
            except OSError:
                self._supply_leases.pop(job.supply_claim_id, None)
                self._increment_supply_stat_locked("consumed", lease)
                self._consumed_supply_leases[job.supply_claim_id] = (
                    float("inf"),
                    lease.priority,
                )
                try:
                    self._unlink_supply_signal(lease)
                except OSError as exc:
                    logger.warning(
                        "could not remove accepted supply signal %s after audit failure: %s",
                        job.supply_claim_id,
                        exc,
                    )
        try:
            self._append_event(
                {
                    "event": "completed",
                    **self._event_identity(job),
                    "state": "rejected",
                    "exit_code": None,
                    "error": reason,
                },
                required=True,
            )
            job.state = "rejected"
            job.error = reason
            job.completed_at = time.time()
            job.exit_code = None
            if self._terminal_rejection_is_retryable(job):
                # The host probe, not the semantic experiment, failed. Keep the
                # terminal observation but let the same stable experiment id be
                # submitted again after inventory recovers.
                self._semantic_jobs.pop(job.semantic_key, None)
            if job.supply_claim_id:
                try:
                    self.allocator.release(job.supply_claim_id)
                except Exception as exc:  # noqa: BLE001 - reconcile retries terminal claims.
                    logger.warning(
                        "could not release rejected job supply claim %s: %s",
                        job.supply_claim_id,
                        exc,
                    )
                else:
                    job.supply_claim_id = ""
        finally:
            if job.state == "queued" and job.job_id in self._jobs and job.job_id not in self._queue:
                self._queue.append(job.job_id)
            with self._condition:
                self._condition.notify_all()

    @staticmethod
    def _terminal_rejection_is_retryable(job: _ExperimentJob) -> bool:
        if job.state != "rejected":
            return False
        return job.attempts == 0 and (
            job.error == "admission_timeout" or job.error.startswith("accelerator_probe_unknown:")
        )

    def _fits_generation(self, job: _ExperimentJob) -> bool:
        deadline = self._generation_deadlines.get(job.generation_id)
        if deadline is None:
            return True
        if job.work_class == "mature" and job.generation_id in self._assessment_generations:
            if job.eta_seconds <= 0:
                return False
            return self._completion_probability(job, deadline=deadline) >= (
                self.settings.mature_assessment_min_completion_probability
            )
        if not self.settings.deadline_admission or job.eta_seconds <= 0:
            return True
        return time.time() + job.eta_seconds <= deadline

    def _observe_runtime(self, job: _ExperimentJob, *, started_at: float) -> None:
        if job.eta_seconds <= 0 or started_at <= 0:
            return
        elapsed = max(1e-6, time.time() - started_at)
        ratio = elapsed / job.eta_seconds
        if not math.isfinite(ratio) or ratio <= 0:
            return
        self._runtime_log_ratios.append(math.log(ratio))
        del self._runtime_log_ratios[:-128]

    def _completion_probability(self, job: _ExperimentJob, *, deadline: float) -> float:
        remaining = deadline - time.time()
        if remaining <= 0 or job.eta_seconds <= 0:
            return 0.0
        observations = self._runtime_log_ratios
        # Two neutral prior observations keep the first few jobs from making
        # deadline admission overconfident.
        location = sum(observations) / (len(observations) + 2.0)
        if len(observations) < 2:
            sigma = 0.8
        else:
            mean = sum(observations) / len(observations)
            variance = sum((value - mean) ** 2 for value in observations) / len(observations)
            sigma = max(0.2, min(1.25, math.sqrt(variance)))
        median_seconds = job.eta_seconds * math.exp(location)
        z_score = math.log(remaining / median_seconds) / sigma
        return 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))

    def _generation_signal_exists(self, generation_id: int) -> bool:
        gen_dir = self.run_dir / f"gen_{generation_id}"
        return any(
            (gen_dir / name).exists()
            for name in ("CLOSING_SIGNAL", "STOP_SIGNAL", "STOP_SIGNAL_POSTGEN")
        )

    def _register_protected(self, job: _ExperimentJob) -> None:
        from . import protected_pids

        entry = protected_pids.register_pid(
            job.pid,
            peer_id=job.peer_id,
            tag=job.experiment_id,
            eta_seconds=job.eta_seconds,
            run_dir=self.run_dir,
            allow_duplicate=True,
            max_active_per_peer=None,
        )
        entry.pgid = job.pgid
        path = protected_pids._manifest_path(job.peer_id, self.run_dir)
        with protected_pids._manifest_lock(path):
            entries = protected_pids._read_manifest(path)
            for current in entries:
                if current.pid == job.pid:
                    current.pgid = job.pgid
            protected_pids._write_manifest(path, entries)

    def _recover_active_process_groups(self) -> None:
        from . import protected_pids

        for entry in protected_pids.list_active_jobs(run_dir=self.run_dir):
            generation_id = protected_pids._generation_id_from_peer_id(entry.peer_id) or 0
            experiment_id = entry.tag or f"recovered-pgid-{entry.pgid or entry.pid}"
            semantic_key = semantic_experiment_key(
                self.run_id,
                generation_id,
                experiment_id,
            )
            if semantic_key in self._semantic_jobs:
                continue
            if (
                entry.pid_start_time is not None
                and protected_pids._is_pid_alive(entry.pid)
                and not protected_pids._entry_process_identity_matches(entry)
            ):
                continue
            process_env = self._read_process_environment(entry.pid)
            gpu_uuids = self._recovered_gpu_uuids(
                process_env,
                pid=entry.pid,
                pgid=entry.pgid or entry.pid,
            )
            profile = self._recovered_profile(process_env, gpu_uuids)
            job = _ExperimentJob(
                job_id=f"recovered-{entry.pgid or entry.pid}",
                run_id=self.run_id,
                generation_id=generation_id,
                peer_id=entry.peer_id,
                experiment_id=experiment_id,
                profile=profile.name,
                work_class="ordinary",
                command=[],
                cwd=None,
                environment={},
                eta_seconds=entry.eta_seconds,
                submitted_at=time.time(),
                state="running",
                attempts=0,
                pid=entry.pid,
                pgid=entry.pgid or entry.pid,
                gpu_uuids=list(gpu_uuids),
                binding_status="recovered_observation",
            )
            allocation_id = f"{self.resource_owner_id}:manifest:{entry.pgid or entry.pid}"
            allocation = self.allocator.recover_allocation(
                allocation_id=allocation_id,
                run_id=self.resource_owner_id,
                pid=entry.pid,
                pgid=entry.pgid or entry.pid,
                profile=profile,
                gpu_uuids=gpu_uuids,
                require_admission=False,
            )
            if allocation is None:
                allocation = self._recover_conservative_allocation(
                    job,
                    allocation_id=f"{allocation_id}:unclassified",
                )
            if allocation is None:
                self._adopt_unaccounted_live(
                    job,
                    {},
                    reason="manifest_allocation_unavailable",
                )
                continue
            self._jobs[job.job_id] = job
            self._semantic_jobs[job.semantic_key] = job.job_id
            self._active[job.job_id] = _ActiveJob(job, None, allocation, None, None)
            self._append_event(
                {
                    "event": "adopted",
                    **self._event_identity(job),
                    "pid": job.pid,
                    "pgid": job.pgid,
                    "gpu_uuids": job.gpu_uuids,
                    "allocation_id": allocation.allocation_id,
                    "recovery_reason": "protected_manifest",
                }
            )

    def _recover_terminal_events(self) -> None:
        path = self.state_dir / "events.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(event, dict):
                events.append(event)
        self._recover_supply_events(events)
        identities: dict[str, dict[str, Any]] = {}
        latest_states: dict[str, dict[str, Any]] = {}
        for event in events:
            event_name = str(event.get("event", ""))
            if event_name == "generation_open":
                generation_id = int(event.get("generation_id", 0) or 0)
                deadline = event.get("deadline", 0.0)
                self._generation_deadlines[generation_id] = (
                    None if deadline is None else float(deadline or 0.0)
                )
                self._generation_cohort_sizes[generation_id] = max(
                    0, int(event.get("cohort_size", 0) or 0)
                )
                peer_ids = event.get("peer_ids", ())
                if isinstance(peer_ids, list):
                    recovered_peer_ids = tuple(
                        peer_id
                        for peer_id in dict.fromkeys(str(value) for value in peer_ids)
                        if self._valid_supply_peer_id(peer_id, generation_id)
                    )
                    if recovered_peer_ids:
                        self._generation_peer_ids[generation_id] = recovered_peer_ids
                if event.get("assessment_continues"):
                    self._assessment_generations.add(generation_id)
                else:
                    self._assessment_generations.discard(generation_id)
                self._frozen_generations.discard(generation_id)
                self._admission_closed = False
                continue
            if event_name == "generation_assessment":
                self._assessment_generations.add(int(event.get("generation_id", 0) or 0))
                continue
            if event_name == "generation_frozen":
                self._frozen_generations.add(int(event.get("generation_id", 0) or 0))
                continue
            if event_name == "admission_closed":
                self._admission_closed = True
                continue
            if not event.get("job_id"):
                continue
            job_id = str(event["job_id"])
            if event.get("event") in {
                "submitted",
                "launch_intent",
                "process_started",
                "launched",
                "adopted",
                "retry_queued",
                "completed",
            }:
                previous = identities.get(job_id, {})
                event_attempt = int(event.get("attempt", 0) or 0)
                previous_attempt = int(previous.get("attempt", -1) or -1)
                if event_attempt < previous_attempt:
                    continue
                if (
                    event_name in {"submitted", "launch_intent", "retry_queued"}
                    or event_attempt > previous_attempt
                ):
                    identities[job_id] = dict(event)
                elif event_attempt == previous_attempt:
                    identities[job_id] = {**previous, **event}
                latest_states[job_id] = event
        if self._recovery_rerun_generation is not None:
            # Keep recovered queue entries dormant until the cohort publishes
            # its fresh deadline through ``open_generation``.
            self._admission_closed = True
        live_semantics = self._live_manifest_semantics()
        live_owners = self._recoverable_live_semantic_owners(identities, latest_states)
        recovered_unknown: list[_ExperimentJob] = []
        for job_id, terminal in latest_states.items():
            identity = identities.get(job_id)
            if identity is None:
                continue
            event_name = str(terminal.get("event", ""))
            semantic = semantic_experiment_key(
                self.run_id,
                int(identity.get("generation_id", 0) or 0),
                str(identity.get("experiment_id", "")),
            )
            preferred_live_job = live_owners.get(semantic)
            if preferred_live_job is not None and preferred_live_job != job_id:
                continue
            if semantic in live_semantics and preferred_live_job is None:
                continue
            existing_job_id = self._semantic_jobs.get(semantic)
            if existing_job_id is not None and existing_job_id != job_id:
                logger.warning(
                    "ignoring duplicate recovered job %s for semantic experiment already "
                    "owned by %s",
                    job_id,
                    existing_job_id,
                )
                continue
            if event_name == "submitted":
                if semantic in live_semantics:
                    continue
                if not sensitive_environment_matches(identity):
                    self._reject_environment_recovery(job_id, identity)
                    continue
                job = self._job_from_event(job_id, identity, state="queued")
                if not job.command or not job.experiment_id:
                    continue
                self._require_profile(job.profile)
                rejection = self._recovery_rejection_reason(job, allow_rerun_queue=True)
                if rejection:
                    self._finish_without_launch(job, rejection)
                    self._jobs[job_id] = job
                    self._semantic_jobs[job.semantic_key] = job_id
                    continue
                self._jobs[job_id] = job
                self._semantic_jobs[job.semantic_key] = job_id
                self._queue.append(job_id)
                continue
            if event_name == "retry_queued":
                if semantic in live_semantics:
                    continue
                if not sensitive_environment_matches(identity):
                    self._reject_environment_recovery(job_id, identity)
                    continue
                job = self._job_from_event(job_id, identity, state="queued")
                if not job.command or not job.experiment_id:
                    continue
                self._require_profile(job.profile)
                rejection = self._recovery_rejection_reason(job, allow_rerun_queue=True)
                if rejection:
                    self._finish_without_launch(job, rejection)
                    self._jobs[job_id] = job
                    self._semantic_jobs[job.semantic_key] = job_id
                    continue
                self._jobs[job_id] = job
                self._semantic_jobs[job.semantic_key] = job_id
                self._queue.append(job_id)
                continue
            if event_name in {"launch_intent", "process_started", "launched", "adopted"}:
                attempt_dir_text = str(identity.get("attempt_dir", ""))
                attempt_dir = Path(attempt_dir_text) if attempt_dir_text else None
                if (
                    event_name == "launch_intent"
                    and not identity.get("pid")
                    and attempt_dir
                    and not identity.get("_recovery_process_checked")
                ):
                    attempt_id = str(identity.get("attempt_id", ""))
                    ready = self._read_ready_process(attempt_dir, attempt_id)
                    if not ready:
                        ready = self._find_attempt_process(attempt_dir, attempt_id)
                    identity.update(ready)
                pgid = int(identity.get("pgid", 0) or 0)
                process_identity_verified = self._event_process_matches(
                    identity,
                    attempt_dir=attempt_dir,
                )
                legacy_manifest_verified = (
                    not process_identity_verified and self._legacy_manifest_matches_event(identity)
                )
                process_identity_verified = process_identity_verified or legacy_manifest_verified
                if (
                    event_name == "adopted"
                    and identity.get("recovery_reason") == "protected_manifest"
                ):
                    if semantic in live_semantics or process_group_alive(pgid):
                        # The protected manifest remains the live authority for
                        # eventless work. Re-observe and account for it below.
                        continue
                    job = self._job_from_event(job_id, identity, state="drained_unknown")
                    job.completed_at = time.time()
                    recovered_unknown.append(job)
                    self._jobs[job_id] = job
                    self._semantic_jobs[job.semantic_key] = job_id
                    continue
                if process_identity_verified and process_group_alive(pgid):
                    job = self._job_from_event(job_id, identity, state="running")
                    waiting_for_barrier = bool(
                        attempt_dir is not None and not (attempt_dir / "GO.json").exists()
                    )
                    persisted_task_environment = recover_environment(identity)
                    task_context_relocated = bool(
                        waiting_for_barrier
                        and task_runtime_context_changed(
                            persisted_task_environment,
                            dict(os.environ),
                        )
                    )
                    rejection = self._recovery_rejection_reason(job) if waiting_for_barrier else ""
                    if rejection:
                        terminate_process_group(pgid)
                        with contextlib.suppress(Exception):
                            self._unregister_protected(job)
                        allocation_id = str(identity.get("allocation_id", ""))
                        if allocation_id:
                            try:
                                self.allocator.release(allocation_id)
                            except Exception as exc:  # noqa: BLE001 - stale claims reconcile later.
                                logger.warning(
                                    "could not release rejected launch allocation %s: %s",
                                    allocation_id,
                                    exc,
                                )
                        self._finish_without_launch(job, rejection)
                        self._jobs[job_id] = job
                        if not self._recovery_terminal_rejection_is_retryable(job):
                            self._semantic_jobs[job.semantic_key] = job_id
                        continue
                    allocation = self._recover_allocation(
                        job,
                        identity,
                        require_admission=waiting_for_barrier,
                    )
                    if allocation is None:
                        if not waiting_for_barrier:
                            allocation = self._recover_conservative_allocation(
                                job,
                                allocation_id=(
                                    f"{identity.get('allocation_id', job.job_id)}:unclassified"
                                ),
                            )
                            if allocation is None:
                                self._adopt_unaccounted_live(
                                    job,
                                    identity,
                                    reason="live_allocation_unavailable",
                                )
                                continue
                        else:
                            logger.error("could not recover allocation for barrier job %s", job_id)
                            terminate_process_group(pgid)
                            with contextlib.suppress(Exception):  # manifest may not exist.
                                self._unregister_protected(job)
                            if sensitive_environment_matches(identity):
                                self._jobs[job_id] = job
                                self._semantic_jobs[job.semantic_key] = job_id
                                self._retry_or_finish(job, INFRASTRUCTURE_RETRY_EXIT_CODE)
                            else:
                                self._reject_environment_recovery(job_id, identity)
                            continue
                    if task_context_relocated:
                        terminate_process_group(pgid)
                        with contextlib.suppress(Exception):
                            self._unregister_protected(job)
                        try:
                            self.allocator.release(allocation.allocation_id)
                        except Exception as exc:  # noqa: BLE001 - allocator reconcile retries.
                            logger.warning(
                                "could not release relocated barrier allocation %s: %s",
                                allocation.allocation_id,
                                exc,
                            )
                        if not sensitive_environment_matches(identity):
                            self._reject_environment_recovery(job_id, identity)
                            continue
                        job.state = "queued"
                        job.pid = 0
                        job.pgid = 0
                        job.gpu_uuids = []
                        self._append_event(
                            {
                                "event": "retry_queued",
                                **self._event_identity(job, include_request=True),
                                "attempt_dir": str(attempt_dir or ""),
                                "recovery_reason": "task_context_relocated_before_launch",
                            },
                            required=True,
                        )
                        self._jobs[job_id] = job
                        self._semantic_jobs[job.semantic_key] = job_id
                        self._queue.append(job_id)
                        continue
                    self._jobs[job_id] = job
                    self._semantic_jobs[job.semantic_key] = job_id
                    self._active[job_id] = _ActiveJob(
                        job,
                        None,
                        allocation,
                        None,
                        attempt_dir,
                        _pid_start_time(job.pid),
                    )
                    if semantic not in live_semantics or legacy_manifest_verified:
                        try:
                            self._register_protected(job)
                        except Exception as exc:  # noqa: BLE001 - live PGID remains tracked.
                            logger.warning(
                                "could not restore protected manifest for %s: %s", job_id, exc
                            )
                    if attempt_dir:
                        self._append_event(
                            {
                                "event": "adopted",
                                **self._event_identity(job),
                                "pid": job.pid,
                                "pgid": job.pgid,
                                "attempt_dir": str(attempt_dir),
                                "recovery_reason": f"resume_from_{event_name}",
                            },
                            required=True,
                        )
                    continue
                if semantic in live_semantics and not process_identity_verified:
                    # The manifest pass below owns the live process identity;
                    # do not let a stale event shadow a different live PID.
                    continue
                if event_name == "launch_intent" or (
                    attempt_dir is not None and not (attempt_dir / "GO.json").exists()
                ):
                    if not sensitive_environment_matches(identity):
                        self._reject_environment_recovery(job_id, identity)
                        continue
                    job = self._job_from_event(job_id, identity, state="queued")
                    if job.command and job.experiment_id:
                        self._require_profile(job.profile)
                        allocation_id = str(identity.get("allocation_id", ""))
                        if allocation_id:
                            try:
                                self.allocator.release(allocation_id)
                            except Exception as exc:  # noqa: BLE001 - queue remains recoverable.
                                logger.warning(
                                    "could not release abandoned launch intent %s: %s",
                                    allocation_id,
                                    exc,
                                )
                        rejection = self._recovery_rejection_reason(
                            job,
                            allow_rerun_queue=True,
                        )
                        if rejection:
                            self._finish_without_launch(job, rejection)
                            self._jobs[job_id] = job
                            self._semantic_jobs[job.semantic_key] = job_id
                            continue
                        self._jobs[job_id] = job
                        self._semantic_jobs[job.semantic_key] = job_id
                        self._queue.append(job_id)
                    continue
                job = self._job_from_event(job_id, identity, state="drained_unknown")
                job.completed_at = time.time()
                recovered_unknown.append(job)
                self._jobs[job_id] = job
                self._semantic_jobs[job.semantic_key] = job_id
                continue
            if event_name != "completed":
                continue
            exit_code = terminal.get("exit_code")
            parsed_exit = None
            if exit_code is not None:
                try:
                    parsed_exit = int(exit_code)
                except (TypeError, ValueError):
                    parsed_exit = 2
            state = str(terminal.get("state", ""))
            if state not in {"completed", "failed", "rejected", "drained_unknown"}:
                state = "completed" if parsed_exit == 0 else "failed"
            job = self._job_from_event(job_id, identity, state=state)
            job.exit_code = parsed_exit
            job.completed_at = float(terminal.get("recorded_at", 0.0) or 0.0)
            if not job.experiment_id:
                continue
            self._jobs[job_id] = job
            if not self._recovery_terminal_rejection_is_retryable(job):
                self._semantic_jobs[job.semantic_key] = job_id
        for job in recovered_unknown:
            self._append_event(
                {
                    "event": "completed",
                    **self._event_identity(job),
                    "state": "drained_unknown",
                    "exit_code": None,
                    "recovery_reason": "process_group_absent_after_restart",
                }
            )

    def _recoverable_live_semantic_owners(
        self,
        identities: dict[str, dict[str, Any]],
        latest_states: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        owners: dict[str, tuple[float, str]] = {}
        for job_id, terminal in latest_states.items():
            event_name = str(terminal.get("event", ""))
            if event_name not in {"launch_intent", "process_started", "launched", "adopted"}:
                continue
            identity = identities.get(job_id)
            if identity is None:
                continue
            attempt_dir_text = str(identity.get("attempt_dir", ""))
            attempt_dir = Path(attempt_dir_text) if attempt_dir_text else None
            if event_name == "launch_intent" and not identity.get("pid") and attempt_dir:
                attempt_id = str(identity.get("attempt_id", ""))
                ready = self._read_ready_process(attempt_dir, attempt_id)
                if not ready:
                    ready = self._find_attempt_process(attempt_dir, attempt_id)
                identity.update(ready)
                identity["_recovery_process_checked"] = True
            pgid = int(identity.get("pgid", 0) or 0)
            verified = self._event_process_matches(identity, attempt_dir=attempt_dir)
            if not verified:
                verified = self._legacy_manifest_matches_event(identity)
            if not verified or not process_group_alive(pgid):
                continue
            semantic = semantic_experiment_key(
                self.run_id,
                int(identity.get("generation_id", 0) or 0),
                str(identity.get("experiment_id", "")),
            )
            recorded_at = float(terminal.get("recorded_at", 0.0) or 0.0)
            if semantic not in owners or recorded_at >= owners[semantic][0]:
                owners[semantic] = (recorded_at, job_id)
        return {semantic: job_id for semantic, (_recorded_at, job_id) in owners.items()}

    def _live_manifest_semantics(self) -> set[str]:
        from . import protected_pids

        result: set[str] = set()
        for entry in protected_pids.list_active_jobs(run_dir=self.run_dir):
            generation_id = protected_pids._generation_id_from_peer_id(entry.peer_id) or 0
            experiment_id = entry.tag or f"recovered-pgid-{entry.pgid or entry.pid}"
            result.add(semantic_experiment_key(self.run_id, generation_id, experiment_id))
        return result

    def _legacy_manifest_matches_event(self, event: dict[str, Any]) -> bool:
        """Correlate pre-start-time events with their independent live manifest."""

        if event.get("pid_start_time") is not None:
            return False
        pid = int(event.get("pid", 0) or 0)
        pgid = int(event.get("pgid", 0) or 0)
        if pid <= 0 or pgid <= 0 or _pid_start_time(pid) is None:
            return False
        peer_id = str(event.get("peer_id", ""))
        experiment_id = str(event.get("experiment_id", ""))
        if not peer_id or not experiment_id or not process_group_alive(pgid):
            return False
        process_environment = self._read_process_environment(pid)
        if process_environment.get("PRAXIST_EXPERIMENT_ID") != experiment_id:
            return False
        attempt_id = str(event.get("attempt_id", ""))
        if attempt_id and process_environment.get("PRAXIST_EXPERIMENT_ATTEMPT_ID") != attempt_id:
            return False
        from . import protected_pids

        return any(
            entry.pid_start_time is None
            and entry.pid == pid
            and (entry.pgid or entry.pid) == pgid
            and entry.peer_id == peer_id
            and entry.tag == experiment_id
            for entry in protected_pids.list_active_jobs(run_dir=self.run_dir)
        )

    def _adopt_unaccounted_live(
        self,
        job: _ExperimentJob,
        event: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        """Deduplicate a live legacy process without claiming resource authority."""

        self._jobs[job.job_id] = job
        self._semantic_jobs[job.semantic_key] = job.job_id
        self._active[job.job_id] = _ActiveJob(job, None, None, None, None)
        self._append_event(
            {
                "event": "adopted",
                **self._event_identity(job),
                "pid": job.pid,
                "pgid": job.pgid,
                "attempt_id": str(event.get("attempt_id", "")),
                "recovery_reason": reason,
            },
            required=True,
        )

    def _recover_supply_events(self, events: list[dict[str, Any]]) -> None:
        records: dict[str, dict[str, Any]] = {}
        accepted_submissions: dict[str, list[dict[str, Any]]] = {}
        attempts: list[tuple[str, str]] = []
        terminal_names = {"consumed", "declined", "expired", "revoked"}
        for event in events:
            event_name = str(event.get("event", ""))
            lease_id = str(event.get("lease_id", ""))
            accepted_submission = event_name == "submitted" or (
                event_name == "retry_queued"
                and event.get("retry_reason") == "explicit_terminal_retry"
            )
            if accepted_submission:
                if "supply_lease_id" in event:
                    submitted_lease = str(event.get("supply_lease_id", ""))
                else:
                    environment = event.get("environment_values", {})
                    submitted_lease = (
                        str(environment.get("PRAXIST_RESOURCE_SUPPLY_LEASE_ID", ""))
                        if isinstance(environment, dict)
                        else ""
                    )
                if submitted_lease:
                    submission = {
                        "accepted_at": float(event.get("recorded_at", 0.0) or 0.0),
                        "peer_id": str(event.get("peer_id", "")),
                        "generation_id": int(event.get("generation_id", 0) or 0),
                        "profile": str(event.get("profile", "")),
                        "work_class": str(event.get("work_class", "ordinary")),
                    }
                    if "supply_lease_eligible" in event:
                        submission["eligible"] = bool(event.get("supply_lease_eligible"))
                    accepted_submissions.setdefault(submitted_lease, []).append(submission)
            if event_name == "supply_granted" and lease_id:
                record = records.setdefault(lease_id, {})
                record["priority"] = str(event.get("priority", ""))
                record["peer_id"] = str(event.get("peer_id", ""))
                record["generation_id"] = int(event.get("generation_id", 0) or 0)
                record["admissible_profiles"] = list(event.get("admissible_profiles", []) or [])
                record["granted"] = "1"
                record["issued_at"] = str(event.get("issued_at", event.get("recorded_at", "")))
                record["expires_at"] = str(event.get("expires_at", ""))
                continue
            if event_name.startswith("supply_") and lease_id:
                outcome = event_name.removeprefix("supply_")
                if outcome in terminal_names:
                    record = records.setdefault(lease_id, {})
                    record.setdefault("priority", str(event.get("priority", "")))
                    record["terminal"] = outcome
                    continue
            if event_name == "supply_stale_submission":
                attempts.append(("stale_submission", str(event.get("priority", ""))))
            elif event_name == "supply_reuse_ignored":
                attempts.append(("reuse_ignored", str(event.get("priority", ""))))

        for name in self._supply_stats:
            self._supply_stats[name] = 0
        self._supply_stats_by_priority.clear()
        self._consumed_supply_leases.clear()
        dangling: list[tuple[str, str, str]] = []
        for lease_id, record in records.items():
            if record.get("granted") != "1":
                continue
            priority = str(record.get("priority", ""))
            self._increment_supply_stat_locked("granted", priority=priority)
            terminal = record.get("terminal", "")
            try:
                expires_at = float(record.get("expires_at", ""))
            except (TypeError, ValueError):
                expires_at = 0.0
            if expires_at <= 0:
                try:
                    issued_at = float(record.get("issued_at", ""))
                except (TypeError, ValueError):
                    issued_at = 0.0
                if issued_at > 0:
                    expires_at = issued_at + _LEGACY_SUPPLY_LEASE_SECONDS
            if not terminal and lease_id in accepted_submissions:
                for submission in accepted_submissions[lease_id]:
                    if expires_at <= 0 or float(submission["accepted_at"]) > expires_at:
                        continue
                    if "eligible" in submission:
                        eligible = bool(submission["eligible"])
                    elif record.get("peer_id") or record.get("admissible_profiles"):
                        eligible = bool(
                            submission["peer_id"] == record.get("peer_id")
                            and submission["generation_id"] == record.get("generation_id")
                            and submission["profile"] in record.get("admissible_profiles", [])
                            and (priority != "mature" or submission["work_class"] == "mature")
                        )
                    else:
                        eligible = True
                    if eligible:
                        terminal = "consumed"
                        break
            if terminal:
                self._increment_supply_stat_locked(terminal, priority=priority)
                if terminal == "consumed":
                    self._consumed_supply_leases[lease_id] = (float("inf"), priority)
            else:
                dangling.append(
                    (
                        lease_id,
                        priority,
                        "expired" if 0 < expires_at <= time.time() else "revoked",
                    )
                )
        for outcome, priority in attempts:
            self._increment_supply_stat_locked(outcome, priority=priority)
        for lease_id, priority, outcome in dangling:
            self._append_event(
                {
                    "event": f"supply_{outcome}",
                    "lease_id": lease_id,
                    "priority": priority,
                    "reason": (
                        "response_window_elapsed_before_restart"
                        if outcome == "expired"
                        else "scheduler_restarted_before_response"
                    ),
                },
                required=True,
            )
            self._increment_supply_stat_locked(outcome, priority=priority)

    def _reject_environment_recovery(self, job_id: str, event: dict[str, Any]) -> None:
        job = self._job_from_event(job_id, event, state="failed")
        job.error = "sensitive_environment_changed_during_recovery"
        job.exit_code = INFRASTRUCTURE_RETRY_EXIT_CODE
        job.completed_at = time.time()
        self._jobs[job_id] = job
        self._semantic_jobs[job.semantic_key] = job_id
        self._append_event(
            {"event": "completed", **self._event_identity(job), "state": "failed", "exit_code": 75}
        )

    def _recover_allocation(
        self,
        job: _ExperimentJob,
        event: dict[str, Any],
        *,
        require_admission: bool,
    ) -> Allocation | None:
        allocation_id = str(event.get("allocation_id", ""))
        if not allocation_id:
            return None
        profile = self._require_profile(job.profile)
        return self.allocator.recover_allocation(
            allocation_id=allocation_id,
            run_id=self.resource_owner_id,
            pid=job.pid,
            pgid=job.pgid,
            profile=profile,
            gpu_uuids=tuple(job.gpu_uuids),
            require_admission=require_admission,
        )

    def _recover_conservative_allocation(
        self,
        job: _ExperimentJob,
        *,
        allocation_id: str,
    ) -> Allocation | None:
        """Count live work host-wide when its original resource shape is unavailable."""

        profile = ResourceProfile(
            name="unclassified_recovery",
            accelerator="cpu",
            pressure_domains=("cpu", "memory", "io"),
        )
        return self.allocator.recover_allocation(
            allocation_id=allocation_id,
            run_id=self.resource_owner_id,
            pid=job.pid,
            pgid=job.pgid,
            profile=profile,
            gpu_uuids=(),
            require_admission=False,
        )

    def _require_profile(self, name: str) -> ResourceProfile:
        profile = self.settings.profiles.get(name)
        if profile is None:
            raise RuntimeError(
                f"persisted resource profile {name!r} is no longer declared; "
                "restore the task profile before resuming"
            )
        return profile

    def _recovery_rejection_reason(
        self,
        job: _ExperimentJob,
        *,
        allow_rerun_queue: bool = False,
    ) -> str:
        rerun_queue = allow_rerun_queue and job.generation_id == self._recovery_rerun_generation
        if self._admission_closed and not rerun_queue:
            return "recovery_admission_closed"
        if not rerun_queue and (
            job.generation_id in self._frozen_generations
            or self._generation_signal_exists(job.generation_id)
        ):
            self._frozen_generations.add(job.generation_id)
            return "recovery_generation_frozen"
        if (
            not rerun_queue
            and job.generation_id in self._assessment_generations
            and job.work_class != "mature"
        ):
            return "recovery_generation_assessment"
        if job.cwd and not Path(job.cwd).is_dir():
            return "recovery_cwd_unavailable"
        return ""

    def _recovery_terminal_rejection_is_retryable(self, job: _ExperimentJob) -> bool:
        if self._terminal_rejection_is_retryable(job):
            return True
        if job.state != "rejected":
            return False
        rerun_control_errors = (
            "admission_closed",
            "generation_closing",
            "generation_assessment",
        )
        return job.generation_id == self._recovery_rerun_generation and (
            any(
                job.error == error or job.error.startswith(f"{error}:")
                for error in rerun_control_errors
            )
            or job.error
            in {
                "recovery_admission_closed",
                "recovery_generation_frozen",
                "recovery_generation_assessment",
            }
        )

    @staticmethod
    def _read_process_environment(pid: int) -> dict[str, str]:
        try:
            payload = (Path("/proc") / str(pid) / "environ").read_bytes()
        except OSError:
            return {}
        if len(payload) > 1024 * 1024:
            return {}
        result: dict[str, str] = {}
        for item in payload.split(b"\0"):
            key, separator, value = item.partition(b"=")
            if not separator:
                continue
            with contextlib.suppress(UnicodeDecodeError):
                result[key.decode("utf-8")] = value.decode("utf-8")
        return result

    def _recovered_gpu_uuids(
        self,
        environment: dict[str, str],
        *,
        pid: int,
        pgid: int,
    ) -> tuple[str, ...]:
        snapshot = self.allocator.snapshot
        by_index = {str(device.index): device.uuid for device in snapshot.gpus}
        available = {device.uuid for device in snapshot.gpus}
        raw = environment.get("PRAXIST_ASSIGNED_GPU_UUIDS", "")
        if not raw:
            raw = environment.get("CUDA_VISIBLE_DEVICES", "")
        observed = {
            by_index.get(token.strip(), token.strip())
            for token in raw.split(",")
            if token.strip() and token.strip() not in {"-1", "none", "void"}
        }
        observed &= available
        if not observed:
            for process in snapshot.gpu_processes:
                try:
                    same_group = process.pid == pid or os.getpgid(process.pid) == pgid
                except OSError:
                    same_group = process.pid == pid
                if same_group and process.uuid in available:
                    observed.add(process.uuid)
        return tuple(sorted(observed))

    def _recovered_profile(
        self, environment: dict[str, str], gpu_uuids: tuple[str, ...]
    ) -> ResourceProfile:
        explicit = environment.get("PRAXIST_RESOURCE_PROFILE", "").strip()
        if explicit in self.settings.profiles:
            profile = self.settings.profiles[explicit]
            if profile.needs_gpu == bool(gpu_uuids) and (
                not profile.needs_gpu or profile.gpu_count == len(gpu_uuids)
            ):
                return profile
        candidates = [
            profile
            for profile in self.settings.profiles.values()
            if profile.needs_gpu == bool(gpu_uuids)
            and (not profile.needs_gpu or profile.gpu_count == len(gpu_uuids))
        ]
        default = self.settings.profiles[self.settings.default_profile]
        if default in candidates:
            return default
        if candidates:
            return candidates[0]
        return ResourceProfile(
            name="unclassified_recovery",
            accelerator="gpu" if gpu_uuids else "cpu",
            gpu_count=len(gpu_uuids),
            pressure_domains=("cpu", "memory", "io", "gpu")
            if gpu_uuids
            else ("cpu", "memory", "io"),
        )

    @staticmethod
    def _read_ready_process(attempt_dir: Path, attempt_id: str) -> dict[str, Any]:
        ready_path = attempt_dir / "READY.json"
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                payload = json.loads(ready_path.read_text(encoding="utf-8"))
                pid = int(payload.get("pid", 0) or 0)
                pgid = int(payload.get("pgid", 0) or 0)
                recorded_attempt = str(payload.get("attempt_id", ""))
                recorded_start = payload.get("pid_start_time")
                current_start = _pid_start_time(pid)
                command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
                ready_argument = str(ready_path).encode("utf-8")
                attempt_argument = attempt_id.encode("utf-8")
                if ready_argument not in command:
                    raise ValueError("READY process does not own this attempt path")
                if recorded_attempt and recorded_attempt != attempt_id:
                    raise ValueError("READY attempt identity mismatch")
                if recorded_attempt and attempt_argument not in command:
                    raise ValueError("READY process command lacks attempt identity")
                if recorded_start is not None and int(recorded_start) != current_start:
                    raise ValueError("READY process identity was reused")
                return {"pid": pid, "pgid": pgid}
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                time.sleep(0.05)
        return {}

    @staticmethod
    def _find_attempt_process(attempt_dir: Path, attempt_id: str) -> dict[str, Any]:
        ready_argument = str(attempt_dir / "READY.json").encode("utf-8")
        attempt_argument = attempt_id.encode("utf-8")
        try:
            entries = list(Path("/proc").iterdir())
        except OSError:
            return {}
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            arguments = command.split(b"\0")
            if (
                len(command) > 1024 * 1024
                or ready_argument not in arguments
                or (attempt_id and attempt_argument not in arguments)
            ):
                continue
            pid = int(entry.name)
            try:
                return {"pid": pid, "pgid": os.getpgid(pid)}
            except OSError:
                continue
        return {}

    @staticmethod
    def _event_process_matches(event: dict[str, Any], *, attempt_dir: Path | None) -> bool:
        pid = int(event.get("pid", 0) or 0)
        pgid = int(event.get("pgid", 0) or 0)
        if pid <= 0 or pgid <= 0:
            return False
        recorded_start = event.get("pid_start_time")
        if recorded_start is not None:
            try:
                return int(recorded_start) == _pid_start_time(pid)
            except (TypeError, ValueError):
                return False
        if attempt_dir is None:
            return False
        try:
            arguments = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        except OSError:
            return False
        ready_argument = str(attempt_dir / "READY.json").encode("utf-8")
        attempt_id = str(event.get("attempt_id", ""))
        return ready_argument in arguments and (
            not attempt_id or attempt_id.encode("utf-8") in arguments
        )

    @staticmethod
    def _release_launch_barrier(attempt_dir: Path) -> None:
        atomic_write_json(attempt_dir / "GO.json", {"go": True, "recorded_at": time.time()})

    def _job_from_event(self, job_id: str, event: dict[str, Any], *, state: str) -> _ExperimentJob:
        recovered_environment = recover_environment(event)
        recovered_cwd = event.get("cwd")
        recovered_cwd_path = Path(str(recovered_cwd or "")).expanduser()
        persisted_task_root = str(
            recovered_environment.get("PRAXIST_TASK_PROJECT_PATH") or ""
        ).strip()
        if recovered_cwd and not recovered_cwd_path.is_absolute() and persisted_task_root:
            recovered_cwd = str((Path(persisted_task_root) / recovered_cwd_path).resolve())
        restored_command = self._restore_durable_command(
            list(event.get("command", []) or []),
            recovered_environment,
        )
        restored_command, recovered_environment, recovered_cwd = rebase_recovered_task_context(
            restored_command,
            recovered_environment,
            recovered_cwd,
            current_environment=dict(os.environ),
        )
        command, environment, cwd = prepare_task_subprocess(
            restored_command,
            recovered_environment,
            cwd=recovered_cwd,
            require_cwd_exists=False,
        )
        return _ExperimentJob(
            job_id=job_id,
            run_id=self.run_id,
            generation_id=int(event.get("generation_id", 0) or 0),
            peer_id=str(event.get("peer_id", "")),
            experiment_id=str(event.get("experiment_id", "")),
            profile=str(event.get("profile", "")),
            work_class=str(event.get("work_class", "ordinary")),
            command=command,
            cwd=cwd,
            environment=environment,
            eta_seconds=int(event.get("eta_seconds", 0) or 0),
            submitted_at=float(event.get("submitted_at", event.get("recorded_at", 0.0)) or 0.0),
            state=state,
            attempts=int(event.get("attempt", 0) or 0),
            pid=int(event.get("pid", 0) or 0),
            pgid=int(event.get("pgid", 0) or 0),
            gpu_uuids=list(event.get("gpu_uuids", []) or []),
            log_path=str(event.get("log_path", "")),
            binding_status=str(event.get("accelerator_delivery", "unknown")),
            error=str(event.get("error", "")),
            failure_category=str(event.get("failure_category", "")),
            failure_signature=str(event.get("failure_signature", "")),
            failure_log_tail=str(event.get("failure_log_tail", "")),
        )

    def _unregister_protected(self, job: _ExperimentJob, *, pid: int | None = None) -> None:
        from . import protected_pids

        protected_pids.unregister_pid(
            job.pid if pid is None else pid,
            peer_id=job.peer_id,
            run_dir=self.run_dir,
        )

    def _write_snapshot(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_snapshot_write < 5.0:
            return
        try:
            atomic_write_json(self.state_dir / "status.json", self.status())
            self._last_snapshot_write = now
        except OSError as exc:
            logger.debug("resource scheduler status write failed: %s", exc)

    def _append_event(self, payload: dict[str, Any], *, required: bool = False) -> None:
        payload = {**payload, "recorded_at": time.time()}
        encoded = (json.dumps(payload, sort_keys=True, default=str) + "\n").encode("utf-8")
        path = self.state_dir / "events.jsonl"
        with self._condition:
            fd = -1
            original_size: int | None = None
            wrote_event = False
            try:
                fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_RDWR, 0o600)
                self._truncate_incomplete_event_tail(fd)
                original_size = os.fstat(fd).st_size
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("event audit write made no progress")
                    remaining = remaining[written:]
                wrote_event = True
                os.fsync(fd)
            except OSError as exc:
                rollback_complete = False
                if fd >= 0 and original_size is not None:
                    try:
                        os.ftruncate(fd, original_size)
                        os.fsync(fd)
                        rollback_complete = True
                    except OSError as rollback_exc:
                        logger.error("resource scheduler event rollback failed: %s", rollback_exc)
                if (
                    required
                    and wrote_event
                    and not rollback_complete
                    and self._event_line_present(path, encoded)
                ):
                    try:
                        os.fsync(fd)
                    except OSError as retry_exc:
                        logger.error(
                            "resource scheduler event durability remains uncertain; the "
                            "transition is not acknowledged: %s",
                            retry_exc,
                        )
                    else:
                        return
                if required:
                    raise
                logger.warning("resource scheduler event audit write failed: %s", exc)
            finally:
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError as close_exc:
                        logger.warning(
                            "resource scheduler event descriptor close failed: %s", close_exc
                        )

    @staticmethod
    def _truncate_incomplete_event_tail(fd: int) -> None:
        size = os.fstat(fd).st_size
        if size <= 0 or os.pread(fd, 1, size - 1) == b"\n":
            return
        cursor = size
        boundary = 0
        while cursor > 0:
            start = max(0, cursor - 65536)
            chunk = os.pread(fd, cursor - start, start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                boundary = start + newline + 1
                break
            cursor = start
        os.ftruncate(fd, boundary)
        os.fsync(fd)

    @staticmethod
    def _event_line_present(path: Path, encoded: bytes) -> bool:
        try:
            with path.open("rb") as handle:
                size = handle.seek(0, os.SEEK_END)
                handle.seek(max(0, size - len(encoded) - 65536))
                return encoded in handle.read()
        except OSError:
            return False
