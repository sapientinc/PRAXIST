"""
Protected-PID manifest.

Problem this solves:
    broad cleanup should not kill legitimate long-running task jobs.

Design:
  - Each peer maintains a per-peer manifest file under
        <run>/protected_pids/<peer_id>.json
    containing a list of {pid, tag, eta_seconds, started_at}.
  - Long-running jobs (for example expensive training, simulation, or
    evaluation jobs with >30 min ETA) are expected to register their PID before launch.
  - Cleanup scripts consult all manifest files; any matching PID is
    skipped unless `--force-kill-protected` is passed.
  - Dead PIDs are pruned on read (defensive; prevents stale entries from
    accumulating when the owning process crashes without unregistering).

The manifest remains cooperative process tracking. When the task enables the
launch guard, the existing generation close sentinel freezes new protected
launches while preserving every already-running process group for drain.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Generator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from praxist.core.execution_guards import emit_resource_event_from_env

from .tools.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


ENV_PROTECTED_DIR = "PROTECTED_PIDS_DIR"
ENV_MAX_ACTIVE_PER_PEER = "PRAXIST_MAX_PARALLEL_RUNS_PER_PEER"
ENV_LAUNCH_GUARD_ENABLED = "PRAXIST_LAUNCH_GUARD_ENABLED"


class DuplicateProtectedPidError(RuntimeError):
    """Raised when a peer tries to register a second live job for the same tag."""


class ProtectedPidCapacityError(RuntimeError):
    """Raised when a peer exceeds its configured live protected-job capacity."""


class GenerationClosingLaunchError(RuntimeError):
    """Raised when a peer tries to start new protected work after generation close begins."""


_PEER_GEN_RE = re.compile(r"\bgen(?P<generation>\d+)(?:[_/-]?peer|\b)")


def _generation_id_from_peer_id(peer_id: str) -> int | None:
    match = _PEER_GEN_RE.search(str(peer_id or ""))
    if match is None:
        return None
    try:
        return int(match.group("generation"))
    except (TypeError, ValueError):
        return None


def _run_dir_from_protected_env() -> Path | None:
    override = os.environ.get(ENV_PROTECTED_DIR)
    if not override:
        return None
    protected_dir = Path(override).expanduser().resolve(strict=False)
    return protected_dir.parent if protected_dir.name == "protected_pids" else None


def _run_dir_from_runtime_env() -> Path | None:
    """Return the canonical run boundary installed by the orchestrator."""

    values = {
        Path(raw).expanduser().resolve(strict=False)
        for name in ("PRAXIST_RUN_DIR", "AUTO_RESEARCH_RUN_DIR")
        if (raw := str(os.environ.get(name) or "").strip())
    }
    if len(values) > 1:
        raise ValueError("protected_pids: inherited run boundaries disagree")
    return next(iter(values), None)


def _generation_closing_signal(run_dir: Path | None, peer_id: str) -> Path | None:
    """Return the current generation close/stop sentinel for ``peer_id`` if present."""

    if run_dir is None:
        run_dir = _run_dir_from_protected_env()
    if run_dir is None:
        return None
    gen_id = _generation_id_from_peer_id(peer_id)
    if gen_id is None:
        return None
    gen_dir = Path(run_dir) / f"gen_{gen_id}"
    for name in ("CLOSING_SIGNAL", "STOP_SIGNAL", "STOP_SIGNAL_POSTGEN"):
        path = gen_dir / name
        try:
            if path.exists():
                return path
        except OSError:
            continue
    return None


def _run_shutdown_signal(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        run_dir = _run_dir_from_protected_env()
    if run_dir is None:
        return None
    path = Path(run_dir) / "ORCHESTRATOR_SHUTDOWN"
    try:
        return path if path.exists() else None
    except OSError:
        return None


def _raise_if_run_shutting_down(run_dir: Path | None, peer_id: str) -> None:
    signal = _run_shutdown_signal(run_dir)
    if signal is not None:
        raise GenerationClosingLaunchError(
            "protected_pids: run shutdown has started for "
            f"peer_id={peer_id!r}; refusing to launch new work because "
            f"{signal.name} exists at {signal}."
        )


def _inside_active_scheduler_attempt(run_dir: Path | None) -> bool:
    """Verify that this process still belongs to one scheduler-owned attempt."""

    if run_dir is None:
        return False
    attempt_id = str(os.environ.get("PRAXIST_EXPERIMENT_ATTEMPT_ID") or "").strip()
    attempt_dir_raw = str(os.environ.get("PRAXIST_EXPERIMENT_ATTEMPT_DIR") or "").strip()
    if not attempt_id or not attempt_dir_raw:
        return False
    try:
        attempts_root = (Path(run_dir) / "resource_scheduler" / "attempts").resolve()
        attempt_dir = Path(attempt_dir_raw).expanduser().resolve()
        if attempt_dir.parent != attempts_root or attempt_dir.name != attempt_id:
            return False
        ready = json.loads((attempt_dir / "READY.json").read_text(encoding="utf-8"))
        if not isinstance(ready, dict) or str(ready.get("attempt_id") or "") != attempt_id:
            return False
        recorded_pgid = int(ready.get("pgid", 0) or 0)
        if not (
            recorded_pgid > 1
            and (attempt_dir / "GO.json").is_file()
            and os.getpgrp() == recorded_pgid
        ):
            return False
        from .experiment_scheduler_client import scheduler_attempt_is_active

        return scheduler_attempt_is_active(Path(run_dir), attempt_id, recorded_pgid)
    except (OSError, ValueError, TypeError):
        return False


def _raise_if_generation_closing(run_dir: Path | None, peer_id: str) -> None:
    raw_enabled = str(os.environ.get(ENV_LAUNCH_GUARD_ENABLED, "0")).strip().lower()
    if raw_enabled not in {"1", "true", "yes", "on"}:
        return
    signal = _generation_closing_signal(run_dir, peer_id)
    if signal is None:
        return
    raise GenerationClosingLaunchError(
        "protected_pids: generation close has started for "
        f"peer_id={peer_id!r}; refusing to launch new protected work because "
        f"{signal.name} exists at {signal}. Finish/publish current results "
        "or record a lightweight follow-up instead."
    )


@contextlib.contextmanager
def _manifest_lock(manifest_path: Path) -> Generator[None, None, None]:
    """Hold an exclusive flock on `<manifest>.lock` for the caller's block.

    Serializes concurrent read-modify-write on a peer manifest so a
    `register_pid` and a `list_all_protected(prune_dead=True)` don't
    clobber each other's edits. O_CLOEXEC prevents subprocess inheritance.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@dataclass
class ProtectedEntry:
    """Protected process record used to avoid killing valuable long-running evaluations."""

    pid: int
    tag: str = ""
    eta_seconds: int = 0
    started_at: str = ""
    peer_id: str = ""
    pgid: int = 0
    pid_start_time: int | str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ProtectedEntry:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def _protected_dir(run_dir: Path | None) -> Path:
    override = os.environ.get(ENV_PROTECTED_DIR)
    if override:
        return Path(override)
    if run_dir:
        return Path(run_dir) / "protected_pids"
    raise ValueError("protected_pids: pass run_dir or set PROTECTED_PIDS_DIR env var")


def _manifest_path(peer_id: str, run_dir: Path | None) -> Path:
    base = _protected_dir(run_dir)
    base.mkdir(parents=True, exist_ok=True)
    safe_peer = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (peer_id or "unknown"))
    return base / f"{safe_peer}.json"


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another user — treat as alive.
        return True


def _is_process_group_alive(pgid: int) -> bool:
    """Return whether a launcher-owned process group still has a member."""

    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _darwin_pid_start_time(pid: int) -> str | None:  # pragma: no cover - Darwin-only
    """Query process start time on Darwin via libproc without subprocesses."""

    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        import ctypes
        from ctypes import Structure, c_int32, c_int64, c_uint32, sizeof

        class _ProcBsdInfo(Structure):
            _fields_ = [
                ("pbi_flags", c_uint32),
                ("pbi_status", c_uint32),
                ("pbi_xstatus", c_uint32),
                ("pbi_pid", c_uint32),
                ("pbi_ppid", c_uint32),
                ("pbi_uid", c_uint32),
                ("pbi_gid", c_uint32),
                ("pbi_ruid", c_uint32),
                ("pbi_rgid", c_uint32),
                ("pbi_svuid", c_uint32),
                ("pbi_svgid", c_uint32),
                ("rfu_1", c_uint32),
                ("pbi_comm", ctypes.c_char * 16),
                ("pbi_name", ctypes.c_char * 32),
                ("pbi_nfiles", c_uint32),
                ("pbi_pgid", c_uint32),
                ("pbi_pjobc", c_uint32),
                ("e_tdev", c_uint32),
                ("e_tpgid", c_uint32),
                ("pbi_nice", c_int32),
                ("pbi_start_tvsec", c_int64),
                ("pbi_start_tvusec", c_int64),
            ]

        libproc = ctypes.CDLL("libproc.dylib")
        info = _ProcBsdInfo()
        ret = libproc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), sizeof(info))
        if ret == sizeof(info):
            return f"darwin:{info.pbi_start_tvsec}.{info.pbi_start_tvusec}"
    except Exception:
        pass
    return None


def _pid_start_time(pid: int) -> int | str | None:
    """Return a stable POSIX process-start identity for PID-reuse detection."""

    try:
        suffix = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        return int(suffix.split()[19])
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform == "darwin":  # pragma: no cover - Darwin-only
        darwin_start = _darwin_pid_start_time(pid)
        if darwin_start is not None:
            return darwin_start
    from praxist.cli.registry import process_start_token

    return process_start_token(pid) or None


def _entry_process_identity_matches(entry: ProtectedEntry) -> bool:
    """Verify a manifest PID when the writer recorded a process identity."""

    if entry.pid_start_time is None:
        return True
    current = _pid_start_time(entry.pid)
    if current is None:
        return True
    return current == entry.pid_start_time


def _entry_is_alive(entry: ProtectedEntry) -> bool:
    """Keep an isolated launch alive while a descendant evaluator still runs."""

    if _is_pid_alive(entry.pid):
        return _entry_process_identity_matches(entry)
    # The original launcher may exit before descendants in its isolated process
    # group. Keep protecting that group, but recovery will not adopt it without
    # a verifiable live launcher identity.
    return _is_process_group_alive(entry.pgid)


def _read_manifest(path: Path) -> list[ProtectedEntry]:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("protected_pids: malformed manifest %s: %s", path, e)
        return []
    if not isinstance(raw, list):
        return []
    return [ProtectedEntry.from_dict(d) for d in raw if isinstance(d, dict)]


def _write_manifest(path: Path, entries: list[ProtectedEntry]) -> None:
    atomic_write_json(path, [asdict(e) for e in entries])


def _max_active_per_peer(explicit: int | None = None) -> int | None:
    """Resolve the live protected-job cap from an explicit value or run env."""

    if explicit is not None:
        return max(1, int(explicit))
    raw = os.environ.get(ENV_MAX_ACTIVE_PER_PEER)
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "protected_pids: invalid %s=%r; peer capacity cap disabled",
            ENV_MAX_ACTIVE_PER_PEER,
            raw,
        )
        return None


def _active_entries_for_peer(
    entries: list[ProtectedEntry],
    peer_id: str,
    *,
    ignore_pid: int | None = None,
) -> list[ProtectedEntry]:
    return [e for e in entries if e.peer_id == peer_id and e.pid != ignore_pid]


def _check_duplicate_tag(
    entries: list[ProtectedEntry],
    *,
    peer_id: str,
    tag: str,
    ignore_pid: int | None = None,
) -> None:
    if not tag:
        return
    duplicate = next(
        (e for e in entries if e.peer_id == peer_id and e.tag == tag and e.pid != ignore_pid),
        None,
    )
    if duplicate is not None:
        raise DuplicateProtectedPidError(
            "protected_pids: live job already registered for "
            f"peer_id={peer_id!r} tag={tag!r} pid={duplicate.pid}; "
            "pass allow_duplicate=True only when duplicate compute is intentional"
        )


def _check_peer_capacity(
    entries: list[ProtectedEntry],
    *,
    peer_id: str,
    max_active_per_peer: int | None,
    ignore_pid: int | None = None,
) -> None:
    cap = _max_active_per_peer(max_active_per_peer)
    if cap is None:
        return
    active = _active_entries_for_peer(entries, peer_id, ignore_pid=ignore_pid)
    if len(active) >= cap:
        pids = ", ".join(str(e.pid) for e in active)
        raise ProtectedPidCapacityError(
            "protected_pids: live job capacity reached for "
            f"peer_id={peer_id!r}: {len(active)}/{cap} active"
            + (f" (pids: {pids})" if pids else "")
        )


def _emit_registered(pid: int, *, peer_id: str, tag: str, eta_seconds: int) -> None:
    emit_resource_event_from_env(
        "resource.protected_pid_registered",
        action_type="protected_pid",
        actor_ref="resource_guard:protected_pids",
        payload={
            "pid": pid,
            "peer_id": peer_id,
            "tag": tag,
            "eta_seconds": int(eta_seconds or 0),
        },
    )


def register_pid(
    pid: int,
    *,
    peer_id: str,
    tag: str = "",
    eta_seconds: int = 0,
    run_dir: Path | None = None,
    allow_duplicate: bool = False,
    max_active_per_peer: int | None = None,
) -> ProtectedEntry:
    """Register `pid` as protected under `peer_id`.

    Idempotent: re-registering an existing pid updates its tag/eta **only if
    the new value is truthy** (non-empty / non-zero). Preserves prior values
    otherwise, and preserves `started_at` always. Serialized via flock so
    concurrent registers / prunes don't lose each other's writes.
    """
    path = _manifest_path(peer_id, run_dir)
    with _manifest_lock(path):
        entries = [e for e in _read_manifest(path) if _entry_is_alive(e)]
        existing = next((e for e in entries if e.pid == pid), None)
        if existing is None:
            if not allow_duplicate:
                _check_duplicate_tag(entries, peer_id=peer_id, tag=tag, ignore_pid=pid)
            _check_peer_capacity(
                entries,
                peer_id=peer_id,
                max_active_per_peer=max_active_per_peer,
                ignore_pid=pid,
            )
        # Update in place if pid already registered.
        updated = False
        for e in entries:
            if e.pid == pid:
                e.tag = tag or e.tag
                e.eta_seconds = int(eta_seconds) or e.eta_seconds
                e.peer_id = peer_id or e.peer_id
                if e.pid_start_time is None:
                    e.pid_start_time = _pid_start_time(pid)
                updated = True
                break
        if not updated:
            entries.append(
                ProtectedEntry(
                    pid=pid,
                    tag=tag,
                    eta_seconds=int(eta_seconds or 0),
                    started_at=datetime.now(UTC).isoformat(),
                    peer_id=peer_id,
                    pgid=0,
                    pid_start_time=_pid_start_time(pid),
                )
            )
        _write_manifest(path, entries)
        _emit_registered(pid, peer_id=peer_id, tag=tag, eta_seconds=eta_seconds)
        return next(e for e in entries if e.pid == pid)


def launch_command(
    command: list[str],
    *,
    peer_id: str,
    tag: str = "",
    eta_seconds: int = 0,
    run_dir: Path | None = None,
    allow_duplicate: bool = False,
    max_active_per_peer: int | None = None,
    cwd: Path | None = None,
    wait_timeout_seconds: float | None = None,
    poll_seconds: float = 2.0,
    resource_profile: str = "",
    work_class: str = "ordinary",
    retry_terminal: bool = False,
) -> int:
    """Launch ``command`` only after atomically reserving a protected PID slot.

    The child process is spawned while holding the peer manifest lock, then its
    real PID is registered before any competing launch can observe stale
    capacity. The wrapper unregisters the child PID after it exits.
    """

    if not command:
        raise ValueError("protected_pids: launch command must not be empty")
    inherited_peer_ids = {
        value
        for key in ("PRAXIST_PEER_ID", "PEER_ID")
        if (value := os.environ.get(key, "").strip())
    }
    if retry_terminal and inherited_peer_ids and inherited_peer_ids != {peer_id}:
        raise ValueError("protected_pids: retry peer does not match the inherited runtime identity")

    # Central mode keeps this long-standing CLI/API as a compatibility facade,
    # but the peer no longer calls Popen or chooses a GPU.  The scheduler owns
    # the final environment, process identity, retries, and release.
    from .experiment_scheduler_client import (
        ENV_SCHEDULER_CONFIG,
        SchedulerUnavailable,
        prepare_task_subprocess,
        scheduler_endpoint_for_run,
        submit_and_wait,
    )

    canonical_run_dir = _run_dir_from_runtime_env()
    protected_run_dir = _run_dir_from_protected_env()
    inherited_run_dir = canonical_run_dir or protected_run_dir
    for candidate in (protected_run_dir, Path(run_dir) if run_dir is not None else None):
        if inherited_run_dir is not None and candidate is not None:
            if candidate.expanduser().resolve(strict=False) == inherited_run_dir:
                continue
            raise ValueError(
                "protected_pids: explicit run_dir does not match the inherited run boundary"
            )
    resolved_run_dir = inherited_run_dir or run_dir
    _raise_if_run_shutting_down(resolved_run_dir, peer_id)
    if _inside_active_scheduler_attempt(resolved_run_dir):
        # This is a task-owned child launched inside an already admitted
        # experiment. Keep it in the scheduler-owned process group and resource
        # envelope instead of recursively queueing a second experiment, which
        # could deadlock behind its own parent allocation.
        child_command, child_environment, child_cwd = prepare_task_subprocess(
            command,
            dict(os.environ),
            cwd=cwd if cwd is not None else Path.cwd(),
        )
        return subprocess.run(
            child_command,
            cwd=child_cwd,
            env=child_environment,
            check=False,
        ).returncode

    scheduler_endpoint = scheduler_endpoint_for_run(resolved_run_dir)
    if scheduler_endpoint:
        return submit_and_wait(
            command,
            peer_id=peer_id,
            experiment_id=tag,
            profile=resource_profile,
            work_class=work_class,
            eta_seconds=eta_seconds,
            run_dir=resolved_run_dir,
            cwd=cwd,
            wait_timeout_seconds=wait_timeout_seconds,
            scheduler_endpoint=scheduler_endpoint,
            retry_terminal=retry_terminal,
        )

    try:
        scheduler_config = json.loads(os.environ.get(ENV_SCHEDULER_CONFIG, "{}"))
    except json.JSONDecodeError as exc:
        raise SchedulerUnavailable("central scheduler configuration is malformed") from exc
    if (
        isinstance(scheduler_config, dict)
        and "mode" in scheduler_config
        and str(scheduler_config.get("mode", "")).strip().lower() != "legacy"
    ):
        raise SchedulerUnavailable(
            "central scheduler is configured but its endpoint is unavailable"
        )

    child_command, child_environment, child_cwd = prepare_task_subprocess(
        command,
        dict(os.environ),
        cwd=cwd if cwd is not None else Path.cwd(),
    )

    path = _manifest_path(peer_id, resolved_run_dir)
    deadline = time.monotonic() + wait_timeout_seconds if wait_timeout_seconds is not None else None
    interval = max(0.05, float(poll_seconds))

    while True:
        _raise_if_run_shutting_down(resolved_run_dir, peer_id)
        _raise_if_generation_closing(resolved_run_dir, peer_id)
        proc: subprocess.Popen[bytes] | None = None
        with _manifest_lock(path):
            # Recheck as close to process creation as this shared launch path
            # permits. A peer can have entered this call before the orchestrator
            # wrote CLOSING_SIGNAL, so the check above alone is insufficient.
            _raise_if_run_shutting_down(resolved_run_dir, peer_id)
            _raise_if_generation_closing(resolved_run_dir, peer_id)
            entries = [e for e in _read_manifest(path) if _entry_is_alive(e)]
            try:
                if not allow_duplicate:
                    _check_duplicate_tag(entries, peer_id=peer_id, tag=tag)
                _check_peer_capacity(
                    entries,
                    peer_id=peer_id,
                    max_active_per_peer=max_active_per_peer,
                )
            except (DuplicateProtectedPidError, ProtectedPidCapacityError):
                if deadline is None or time.monotonic() >= deadline:
                    raise
            else:
                proc = subprocess.Popen(
                    child_command,
                    cwd=child_cwd,
                    env=child_environment,
                    start_new_session=True,
                )
                entry = ProtectedEntry(
                    pid=proc.pid,
                    tag=tag,
                    eta_seconds=int(eta_seconds or 0),
                    started_at=datetime.now(UTC).isoformat(),
                    peer_id=peer_id,
                    pgid=proc.pid,
                    pid_start_time=_pid_start_time(proc.pid),
                )
                entries.append(entry)
                _write_manifest(path, entries)
                _emit_registered(proc.pid, peer_id=peer_id, tag=tag, eta_seconds=eta_seconds)

        if proc is not None:
            try:
                result = proc.wait()
                # A shell launcher may return while an evaluator it spawned
                # remains in its isolated process group. Preserve the existing
                # manifest entry until that evaluator has naturally drained.
                while _is_process_group_alive(proc.pid):
                    time.sleep(interval)
                return result
            finally:
                unregister_pid(proc.pid, peer_id=peer_id, run_dir=run_dir)

        time.sleep(interval)


def unregister_pid(
    pid: int,
    *,
    peer_id: str,
    run_dir: Path | None = None,
) -> bool:
    """Remove `pid` from `peer_id`'s manifest. Returns True if removed."""
    path = _manifest_path(peer_id, run_dir)
    with _manifest_lock(path):
        entries = _read_manifest(path)
        new = [e for e in entries if e.pid != pid]
        if len(new) == len(entries):
            return False
        _write_manifest(path, new)
        emit_resource_event_from_env(
            "resource.protected_pid_unregistered",
            action_type="protected_pid",
            actor_ref="resource_guard:protected_pids",
            payload={"pid": pid, "peer_id": peer_id},
        )
        return True


def list_all_protected(
    run_dir: Path | None = None,
    prune_dead: bool = True,
) -> list[ProtectedEntry]:
    """Enumerate all protected PIDs across all peer manifests.

    If prune_dead is True (default), entries for non-existent PIDs are
    removed from their manifests to prevent stale accumulation.
    """
    try:
        base = _protected_dir(run_dir)
    except ValueError:
        return []
    if not base.exists():
        return []

    all_entries: list[ProtectedEntry] = []
    for manifest in base.glob("*.json"):
        # Skip our own lock files (`.json.lock`) that glob may include on
        # systems where `.json.lock` technically matches `*.json` — defensive.
        if manifest.suffix == ".lock":
            continue
        # Each manifest has its own lock; serialize prune-writes against
        # concurrent register/unregister calls on the same manifest.
        with _manifest_lock(manifest):
            entries = _read_manifest(manifest)
            live: list[ProtectedEntry] = []
            for e in entries:
                if _entry_is_alive(e):
                    live.append(e)
                else:
                    logger.debug(
                        "protected_pids: pruning dead pid=%d tag=%s",
                        e.pid,
                        e.tag,
                    )
            if prune_dead and len(live) != len(entries):
                _write_manifest(manifest, live)
            all_entries.extend(live)
    return all_entries


def get_protected_pids_set(run_dir: Path | None = None) -> set[int]:
    """Return just the PID set, useful for cleanup filtering."""
    return {e.pid for e in list_all_protected(run_dir=run_dir)}


def list_active_jobs(
    *,
    peer_id: str | None = None,
    tag: str | None = None,
    run_dir: Path | None = None,
) -> list[ProtectedEntry]:
    """Return live protected jobs, optionally filtered by peer and semantic tag."""
    entries = list_all_protected(run_dir=run_dir)
    if peer_id is not None:
        entries = [e for e in entries if e.peer_id == peer_id]
    if tag is not None:
        entries = [e for e in entries if e.tag == tag]
    return entries


def wait_job(
    *,
    peer_id: str,
    tag: str,
    run_dir: Path | None = None,
    timeout_seconds: float | None = None,
    poll_seconds: float = 2.0,
) -> bool:
    """Wait until no live job exists for ``(peer_id, tag)``.

    Returns True when the job slot is clear and False on timeout. This is an
    advisory coordination helper for peers; it does not supervise processes.
    """
    import time

    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    interval = max(0.05, float(poll_seconds))
    while True:
        if not list_active_jobs(peer_id=peer_id, tag=tag, run_dir=run_dir):
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(interval)


# ---------------------------------------------------------------------------
# CLI — retained for future run-control tooling and manual diagnostics.
# ---------------------------------------------------------------------------


def _cli():
    """CLI shim so bash scripts can query the manifest.

    Usage:
        python -m praxist.plugins.workflow_stages.research_loop.backend.protected_pids list \\
            [--run-dir=<path>] [--format=json|pids]
        python -m praxist.plugins.workflow_stages.research_loop.backend.protected_pids register \\
            --pid=$$ --peer=gen0_peer3 --tag=variant_long --eta=3600 \\
            [--run-dir=<path>]
        python -m praxist.plugins.workflow_stages.research_loop.backend.protected_pids unregister \\
            --pid=$$ --peer=gen0_peer3 [--run-dir=<path>]
        python -m praxist.plugins.workflow_stages.research_loop.backend.protected_pids launch \\
            --peer=gen0_peer3 --tag=variant_long -- python train.py
    """
    import argparse
    import sys

    from .experiment_scheduler_client import ExperimentRejected, SchedulerUnavailable

    parser = argparse.ArgumentParser(prog="protected_pids")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register")
    p_reg.add_argument("--pid", type=int, required=True)
    p_reg.add_argument("--peer", required=True)
    p_reg.add_argument("--tag", default="")
    p_reg.add_argument("--eta", type=int, default=0)
    p_reg.add_argument("--run-dir", default=None)
    p_reg.add_argument("--allow-duplicate", action="store_true")
    p_reg.add_argument("--max-active", type=int, default=None)

    p_unreg = sub.add_parser("unregister")
    p_unreg.add_argument("--pid", type=int, required=True)
    p_unreg.add_argument("--peer", required=True)
    p_unreg.add_argument("--run-dir", default=None)

    p_list = sub.add_parser("list")
    p_list.add_argument("--run-dir", default=None)
    p_list.add_argument("--peer", default=None)
    p_list.add_argument("--tag", default=None)
    p_list.add_argument("--format", choices=["json", "pids"], default="json")

    p_wait = sub.add_parser("wait")
    p_wait.add_argument("--peer", required=True)
    p_wait.add_argument("--tag", required=True)
    p_wait.add_argument("--run-dir", default=None)
    p_wait.add_argument("--timeout", type=float, default=None)
    p_wait.add_argument("--poll", type=float, default=2.0)

    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--peer", required=True)
    p_launch.add_argument("--tag", default="")
    p_launch.add_argument("--eta", type=int, default=0)
    p_launch.add_argument("--run-dir", default=None)
    p_launch.add_argument("--allow-duplicate", action="store_true")
    p_launch.add_argument("--max-active", type=int, default=None)
    p_launch.add_argument("--wait-timeout", type=float, default=None)
    p_launch.add_argument("--poll", type=float, default=2.0)
    p_launch.add_argument("--cwd", default=None)
    p_launch.add_argument("--profile", default="")
    p_launch.add_argument(
        "--work-class", choices=["mature", "ordinary", "scout"], default="ordinary"
    )
    p_launch.add_argument(
        "--retry-terminal",
        action="store_true",
        help="explicitly retry a corrected failed/rejected semantic experiment",
    )
    p_launch.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None

    if args.cmd == "register":
        try:
            entry = register_pid(
                args.pid,
                peer_id=args.peer,
                tag=args.tag,
                eta_seconds=args.eta,
                run_dir=run_dir,
                allow_duplicate=args.allow_duplicate,
                max_active_per_peer=args.max_active,
            )
        except (DuplicateProtectedPidError, ProtectedPidCapacityError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
        print(json.dumps(asdict(entry), default=str))
        sys.exit(0)

    if args.cmd == "unregister":
        ok = unregister_pid(args.pid, peer_id=args.peer, run_dir=run_dir)
        sys.exit(0 if ok else 1)

    if args.cmd == "list":
        entries = list_active_jobs(peer_id=args.peer, tag=args.tag, run_dir=run_dir)
        if args.format == "pids":
            # Bash-friendly: one PID per line, easy to read into an array.
            for e in entries:
                print(e.pid)
        else:
            print(json.dumps([asdict(e) for e in entries], indent=2, default=str))
        sys.exit(0)

    if args.cmd == "wait":
        ok = wait_job(
            peer_id=args.peer,
            tag=args.tag,
            run_dir=run_dir,
            timeout_seconds=args.timeout,
            poll_seconds=args.poll,
        )
        sys.exit(0 if ok else 1)

    if args.cmd == "launch":
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        try:
            code = launch_command(
                command,
                peer_id=args.peer,
                tag=args.tag,
                eta_seconds=args.eta,
                run_dir=run_dir,
                allow_duplicate=args.allow_duplicate,
                max_active_per_peer=args.max_active,
                cwd=Path(args.cwd) if args.cwd else None,
                wait_timeout_seconds=args.wait_timeout,
                poll_seconds=args.poll,
                resource_profile=args.profile,
                work_class=args.work_class,
                retry_terminal=args.retry_terminal,
            )
        except (
            DuplicateProtectedPidError,
            ProtectedPidCapacityError,
            GenerationClosingLaunchError,
            ExperimentRejected,
            SchedulerUnavailable,
            TimeoutError,
            ValueError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(2)
        sys.exit(code)


if __name__ == "__main__":
    _cli()
