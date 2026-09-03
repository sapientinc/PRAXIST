"""
Per-GPU process pool governor.

Problem this solves:
    Uncontrolled subprocess fan-out can oversubscribe accelerators and reduce
    useful throughput even when each individual launch appears valid.

Design:
  - One lock file per GPU id under
        <run>/process_governor/gpu_<id>.lock
  - Each file holds a JSONL manifest of currently-acquired slots:
        {"pid": 123, "peer": "gen0_peer3", "tag": "variant_seed42",
         "expected_seconds": 3600, "started_at": "<iso>"}
  - `acquire_slot` uses fcntl.flock + a read-modify-write cycle to atomically
    add an entry if the current count is below `max_per_gpu`.
  - `release_slot` removes the entry. `atexit` registration is offered but
    not forced — callers that forked training subprocesses should release
    explicitly when the subprocess completes.
  - `BYPASS_GPU_GOVERNOR=1` env var skips all checks (emergency escape).

This is a soft governor, not a scheduler. It prevents the "accidental 266
processes" cascade without replacing any real batch system. Agents are
expected to acquire a slot before launching a long-running training
subprocess and release when it ends.

Locking note: fcntl.flock is advisory and per-open-file-description. We
always open the lock file fresh inside the context manager so concurrent
callers serialize on the lock rather than on a shared FD.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from praxist.core.execution_guards import (
    emit_resource_event_from_env,
    gpu_hours_since,
    record_budgeted_action_from_env,
)
from praxist.plugins.workflow_stages.research_loop.backend.event_wait import (
    wait_for_filesystem_event,
)

logger = logging.getLogger(__name__)


DEFAULT_MAX_PER_GPU = 4
DEFAULT_SLOT_WAIT_SECONDS = 30.0
CLI_SCHEMA_VERSION = "praxist.gpu_governor.cli.v1"

# Environment knobs.
ENV_BYPASS = "BYPASS_GPU_GOVERNOR"
ENV_GOVERNOR_DIR = "GPU_GOVERNOR_DIR"
ENV_MAX_PER_GPU = "GPU_GOVERNOR_MAX_PER_GPU"


class GovernorBusy(RuntimeError):
    """Raised when a non-blocking acquire fails because the GPU is full."""


@dataclass
class SlotEntry:
    """GPU slot reservation record tracked by the local GPU governor."""

    pid: int
    peer: str = ""
    tag: str = ""
    expected_seconds: int = 0
    started_at: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> SlotEntry:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def _governor_dir(run_dir: Path | None = None) -> Path:
    override = os.environ.get(ENV_GOVERNOR_DIR)
    if override:
        return Path(override)
    if run_dir:
        return Path(run_dir) / "process_governor"
    raise ValueError("gpu_governor: either pass run_dir or set GPU_GOVERNOR_DIR env var")


def _lock_path(gpu_id: int, run_dir: Path | None = None) -> Path:
    base = _governor_dir(run_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"gpu_{gpu_id}.lock"


def _max_per_gpu(override: int | None) -> int:
    if override is not None:
        # R4-N10: clamp to ≥1 to defend against `=0` deadlock.
        return max(1, override)
    env_val = os.environ.get(ENV_MAX_PER_GPU)
    if env_val:
        try:
            v = int(env_val)
            if v <= 0:
                logger.warning(
                    "gpu_governor: env %s=%r is non-positive; clamping to 1 "
                    "(else acquire_slot would deadlock).",
                    ENV_MAX_PER_GPU,
                    env_val,
                )
                return 1
            return v
        except ValueError:
            pass
    return DEFAULT_MAX_PER_GPU


def _is_bypassed() -> bool:
    return os.environ.get(ENV_BYPASS, "").lower() in ("1", "true", "yes")


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[object]:
    """Acquire an exclusive flock on `path`. Releases on context exit.

    O_CLOEXEC prevents child processes from inheriting the fd (and therefore
    the flock), which would otherwise keep the lock held even after the
    parent's fd is closed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open for append+read so the file is created if absent and writable.
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _read_entries(fd: int) -> list[SlotEntry]:
    """Read JSONL manifest from an open lock file (assumes caller holds flock)."""
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, 10 * 1024 * 1024).decode("utf-8", errors="replace")
    entries: list[SlotEntry] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(SlotEntry.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("gpu_governor: skipping malformed manifest line: %s", e)
    return entries


def _write_entries(fd: int, entries: list[SlotEntry]) -> None:
    """Truncate + rewrite JSONL manifest (assumes caller holds flock)."""
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    payload = "".join(json.dumps(e.__dict__, default=str) + "\n" for e in entries)
    os.write(fd, payload.encode("utf-8"))
    with contextlib.suppress(OSError):
        os.fsync(fd)


# Hard ceiling on slot age — even with a live PID, an entry older than
# this is treated as orphan-after-pid-reuse (Linux PIDs wrap at 32768 by
# default; a long-running orchestrator can recycle PIDs over a 24-hour
# multi-cohort run). The ceiling is generous: 8 hours is well beyond any
# tier's expected_seconds budget × safety factor.
_SLOT_HARD_AGE_CEILING_SECONDS = 8 * 3600


def _entry_age_seconds(e: SlotEntry) -> float | None:
    """Best-effort age in seconds; None if started_at is unparseable."""
    if not e.started_at:
        return None
    try:
        ts = datetime.fromisoformat(e.started_at)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - ts).total_seconds())


def _is_zombie(pid: int) -> bool:
    """Check if a PID is a zombie (Linux: State='Z' in /proc/<pid>/status).

    R9-3 fix: zombies pass `os.kill(pid, 0)` because the process exists in
    the kernel's process table, but they're already dead from a workload
    perspective (the parent hasn't called wait() yet). At gen-transition
    boundaries the orchestrator may not have reaped child peers, leaving
    their slots looking "alive" to the picker. Returns False on any error
    or non-Linux platform (be conservative — keep the entry alive rather
    than risking false-positive prune).
    """
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    state = line.split()[1] if len(line.split()) > 1 else ""
                    return state == "Z"
    except (FileNotFoundError, PermissionError, OSError):
        return False
    return False


def _prune_dead(entries: list[SlotEntry]) -> list[SlotEntry]:
    """Drop entries whose pid is no longer alive OR whose started_at is older
    than the hard ceiling (defends against PID reuse on long runs).

    pid <= 0 is treated as dead — os.kill(0, 0) targets the caller's whole
    process group, which both "succeeds" misleadingly and is dangerous.
    """
    alive: list[SlotEntry] = []
    for e in entries:
        if e.pid <= 0:
            logger.debug(
                "gpu_governor: pruning invalid pid=%d peer=%s tag=%s",
                e.pid,
                e.peer,
                e.tag,
            )
            continue

        age = _entry_age_seconds(e)
        if age is not None and age > _SLOT_HARD_AGE_CEILING_SECONDS:
            logger.warning(
                "gpu_governor: pruning stale entry (age=%.0fs > %ds ceiling) "
                "pid=%d peer=%s tag=%s — likely PID reuse",
                age,
                _SLOT_HARD_AGE_CEILING_SECONDS,
                e.pid,
                e.peer,
                e.tag,
            )
            continue

        try:
            os.kill(e.pid, 0)  # signal 0 = existence check
            # R9-3 fix: zombies pass os.kill(0) but are effectively dead.
            # Treat them as dead so gen-transition cleanup doesn't keep
            # un-reaped child peers' slots alive.
            if _is_zombie(e.pid):
                logger.debug(
                    "gpu_governor: pruning zombie entry pid=%d peer=%s tag=%s",
                    e.pid,
                    e.peer,
                    e.tag,
                )
                continue
            alive.append(e)
        except ProcessLookupError:
            logger.debug(
                "gpu_governor: pruning dead entry pid=%d peer=%s tag=%s",
                e.pid,
                e.peer,
                e.tag,
            )
        except PermissionError:
            # Process exists but isn't ours — keep it, safer.
            alive.append(e)
    return alive


def _wait_for_slot_manifest_change(lock_path: Path, fallback_seconds: float) -> None:
    wait_seconds = max(5.0, float(fallback_seconds))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        time.sleep(wait_seconds)
        return
    try:
        asyncio.run(
            wait_for_filesystem_event(
                [lock_path],
                timeout_seconds=wait_seconds,
                recursive=False,
                max_dirs=16,
                fallback_interval_seconds=wait_seconds,
                stop_check_interval_seconds=wait_seconds,
                event_filter=lambda p: Path(p).name == lock_path.name,
            )
        )
    except RuntimeError:
        # If acquire_slot is ever called inside an already-running event loop,
        # keep the synchronous API stable and fall back to a sparse sleep.
        time.sleep(wait_seconds)
    except Exception as e:
        logger.debug("gpu_governor: slot event wait failed: %s", e)
        time.sleep(wait_seconds)


def acquire_slot(
    gpu_id: int,
    *,
    pid: int,
    peer: str = "",
    tag: str = "",
    expected_seconds: int = 0,
    max_per_gpu: int | None = None,
    run_dir: Path | None = None,
    blocking: bool = True,
    poll_interval: float = DEFAULT_SLOT_WAIT_SECONDS,
    timeout_seconds: float | None = None,
) -> bool:
    """Acquire a slot on `gpu_id` for `pid`.

    Returns True on success. If blocking=False and the GPU is full, returns
    False immediately. If blocking=True, waits for the slot manifest to change
    and uses ``poll_interval`` only as a sparse fallback.

    Parameters
    ----------
    gpu_id : int
    pid : int
        The training-subprocess PID you want to register. Use os.getpid()
        for callers that are themselves the training process.
    peer : str, optional
    tag : str, optional
        Human-readable job tag (e.g. "variant_seed42_long").
    expected_seconds : int, optional
        Hint for observability; not enforced.
    max_per_gpu : int, optional
        Override the default cap. If None, uses env GPU_GOVERNOR_MAX_PER_GPU
        or DEFAULT_MAX_PER_GPU.
    run_dir : Path, optional
        Run directory (contains process_governor/). If None, env
        GPU_GOVERNOR_DIR must be set.
    blocking : bool
        Poll until a slot frees.
    poll_interval : float
        Low-frequency fallback wait when filesystem events are unavailable.
    timeout_seconds : float, optional
        If blocking and set, abort with GovernorBusy after this wait.
    """
    if _is_bypassed():
        return True

    cap = _max_per_gpu(max_per_gpu)
    start = time.monotonic()

    while True:
        lock = _lock_path(gpu_id, run_dir)
        with _locked(lock) as fd:
            entries = _prune_dead(_read_entries(fd))
            # If this pid is already registered (e.g. retry), treat as success.
            if any(e.pid == pid for e in entries):
                return True
            if len(entries) < cap:
                new_entry = SlotEntry(
                    pid=pid,
                    peer=peer,
                    tag=tag,
                    expected_seconds=int(expected_seconds or 0),
                    started_at=datetime.now(UTC).isoformat(),
                )
                entries.append(new_entry)
                _write_entries(fd, entries)
                emit_resource_event_from_env(
                    "resource.gpu_slot_acquired",
                    action_type="gpu_slot",
                    actor_ref="resource_guard:gpu_governor",
                    payload={
                        "gpu_id": gpu_id,
                        "pid": pid,
                        "peer": peer,
                        "tag": tag,
                        "expected_seconds": int(expected_seconds or 0),
                    },
                )
                return True
            # else: full — fall through

        if not blocking:
            return False

        if timeout_seconds is not None:
            elapsed = time.monotonic() - start
            if elapsed >= timeout_seconds:
                raise GovernorBusy(f"gpu_{gpu_id} full ({cap} slots) — waited {timeout_seconds}s")
            wait_seconds = min(float(poll_interval), max(0.1, timeout_seconds - elapsed))
        else:
            wait_seconds = float(poll_interval)

        _wait_for_slot_manifest_change(lock, wait_seconds)


def release_slot(
    gpu_id: int,
    *,
    pid: int,
    run_dir: Path | None = None,
) -> bool:
    """Release the slot held by `pid` on `gpu_id`. Returns True if released."""
    if _is_bypassed():
        return True
    lock = _lock_path(gpu_id, run_dir)
    with _locked(lock) as fd:
        entries = _read_entries(fd)
        released = [e for e in entries if e.pid == pid]
        new = [e for e in entries if e.pid != pid]
        if len(new) == len(entries):
            return False
        _write_entries(fd, new)
    for entry in released:
        gpu_hours = gpu_hours_since(entry.started_at)
        usage = {"gpu_hours": gpu_hours} if gpu_hours is not None else {}
        record_budgeted_action_from_env(
            action_type="gpu_slot",
            actor_ref="resource_guard:gpu_governor",
            actual_usage=usage,
            expected_units=("gpu_hours",),
            reason="gpu_slot_usage",
            metadata={
                "gpu_id": gpu_id,
                "pid": pid,
                "peer": entry.peer,
                "tag": entry.tag,
            },
        )
        emit_resource_event_from_env(
            "resource.gpu_slot_released",
            action_type="gpu_slot",
            actor_ref="resource_guard:gpu_governor",
            payload={
                "gpu_id": gpu_id,
                "pid": pid,
                "peer": entry.peer,
                "tag": entry.tag,
                "gpu_hours": gpu_hours,
            },
        )
    return True


def transfer_slot(
    gpu_id: int,
    *,
    from_pid: int,
    to_pid: int,
    peer: str | None = None,
    tag: str | None = None,
    run_dir: Path | None = None,
) -> bool:
    """Transfer a held slot from one PID to another under the GPU lock.

    This is for launch wrappers that must reserve capacity before the real
    training child exists. The manifest count never changes; only ownership of
    an existing reservation changes. If ``to_pid`` already owns the slot, the
    operation is treated as success and any duplicate ``from_pid`` entry is
    removed.
    """
    if _is_bypassed():
        return True
    if from_pid <= 0 or to_pid <= 0:
        return False
    if from_pid == to_pid:
        return True
    lock = _lock_path(gpu_id, run_dir)
    with _locked(lock) as fd:
        entries = _prune_dead(_read_entries(fd))
        to_entries = [entry for entry in entries if entry.pid == to_pid]
        if to_entries:
            deduped = [entry for entry in entries if entry.pid != from_pid]
            if len(deduped) != len(entries):
                _write_entries(fd, deduped)
            return True

        transferred = False
        for entry in entries:
            if entry.pid != from_pid:
                continue
            entry.pid = to_pid
            if peer is not None:
                entry.peer = peer
            if tag is not None:
                entry.tag = tag
            transferred = True
            break
        if not transferred:
            _write_entries(fd, entries)
            return False
        _write_entries(fd, entries)
    emit_resource_event_from_env(
        "resource.gpu_slot_transferred",
        action_type="gpu_slot",
        actor_ref="resource_guard:gpu_governor",
        payload={
            "gpu_id": gpu_id,
            "from_pid": from_pid,
            "to_pid": to_pid,
            "peer": peer or "",
            "tag": tag or "",
        },
    )
    return True


def list_slots(
    gpu_id: int,
    run_dir: Path | None = None,
) -> list[SlotEntry]:
    """Snapshot the slot manifest for `gpu_id` (with dead-pid pruning)."""
    lock = _lock_path(gpu_id, run_dir)
    if not lock.exists():
        return []
    with _locked(lock) as fd:
        raw = _read_entries(fd)
        entries = _prune_dead(raw)
        # Write back only when pruning changed the manifest. Rewriting on every
        # sweep amplifies lock-file contention on slower filesystems.
        if len(entries) != len(raw) or any(
            (a.pid != b.pid or a.tag != b.tag) for a, b in zip(entries, raw, strict=False)
        ):
            _write_entries(fd, entries)
        return list(entries)


def list_all_slots(
    num_gpus: int,
    run_dir: Path | None = None,
) -> dict[int, list[SlotEntry]]:
    """Snapshot slot manifests across a range of GPU ids."""
    return {g: list_slots(g, run_dir=run_dir) for g in range(num_gpus)}


def _entry_payload(entry: SlotEntry) -> dict:
    return {
        "pid": entry.pid,
        "peer": entry.peer,
        "tag": entry.tag,
        "expected_seconds": entry.expected_seconds,
        "started_at": entry.started_at,
    }


def _governor_dir_payload(run_dir: Path | None) -> str | None:
    try:
        return str(_governor_dir(run_dir))
    except ValueError:
        return None


def _slot_state_payload(
    gpu_id: int,
    *,
    run_dir: Path | None,
    max_per_gpu: int | None,
) -> dict:
    cap = _max_per_gpu(max_per_gpu)
    slots = [_entry_payload(entry) for entry in list_slots(gpu_id, run_dir=run_dir)]
    occupied = len(slots)
    return {
        "gpu_id": gpu_id,
        "max_per_gpu": cap,
        "occupied": occupied,
        "available": max(0, cap - occupied),
        "slots": slots,
    }


def _base_cli_payload(
    *,
    command: str,
    status: str,
    ok: bool,
    run_dir: Path | None,
) -> dict:
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "command": command,
        "status": status,
        "ok": ok,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "governor_dir": _governor_dir_payload(run_dir),
    }


def _with_slot_state(payload: dict, state: dict) -> dict:
    payload.update(
        {
            "gpu_id": state["gpu_id"],
            "max_per_gpu": state["max_per_gpu"],
            "occupied": state["occupied"],
            "available": state["available"],
            "current_slots": state,
        }
    )
    return payload


def _print_cli_payload(payload: dict, *, pretty: bool) -> None:
    indent = 2 if pretty else None
    print(json.dumps(payload, indent=indent, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# CLI for shell scripts (used by wrapper .sh helpers)
# ---------------------------------------------------------------------------


def _cli():
    """Tiny CLI so bash scripts can acquire/release/list slots.

    Usage:
        python -m praxist.plugins.workflow_stages.research_loop.backend.gpu_governor acquire \\
            --gpu=0 --pid=$$ --peer=gen0_peer3 --tag=variant_seed42 \\
            --expected-seconds=3600 --run-dir=<path>
        python -m praxist.plugins.workflow_stages.research_loop.backend.gpu_governor release \\
            --gpu=0 --pid=$$ --run-dir=<path>
        python -m praxist.plugins.workflow_stages.research_loop.backend.gpu_governor list \\
            --gpus=8 --run-dir=<path>
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="gpu_governor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_acq = sub.add_parser("acquire")
    p_acq.add_argument("--gpu", type=int, required=True)
    p_acq.add_argument("--pid", type=int, required=True)
    p_acq.add_argument("--peer", default="")
    p_acq.add_argument("--tag", default="")
    p_acq.add_argument("--expected-seconds", type=int, default=0)
    p_acq.add_argument("--max-per-gpu", type=int, default=None)
    p_acq.add_argument("--run-dir", default=None)
    p_acq.add_argument("--non-blocking", action="store_true")
    p_acq.add_argument("--timeout", type=float, default=None)
    p_acq.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_rel = sub.add_parser("release")
    p_rel.add_argument("--gpu", type=int, required=True)
    p_rel.add_argument("--pid", type=int, required=True)
    p_rel.add_argument("--run-dir", default=None)
    p_rel.add_argument("--max-per-gpu", type=int, default=None)
    p_rel.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_list = sub.add_parser("list")
    p_list.add_argument("--gpus", type=int, default=8)
    p_list.add_argument("--run-dir", default=None)
    p_list.add_argument("--max-per-gpu", type=int, default=None)
    p_list.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None

    if args.cmd == "acquire":
        if _is_bypassed():
            payload = _base_cli_payload(
                command="acquire",
                status="bypassed",
                ok=True,
                run_dir=run_dir,
            )
            payload.update(
                {
                    "gpu_id": args.gpu,
                    "pid": args.pid,
                    "peer": args.peer,
                    "tag": args.tag,
                    "expected_seconds": int(args.expected_seconds or 0),
                    "max_per_gpu": _max_per_gpu(args.max_per_gpu),
                    "reason": "bypassed",
                }
            )
            _print_cli_payload(payload, pretty=args.pretty)
            sys.exit(0)
        try:
            ok = acquire_slot(
                args.gpu,
                pid=args.pid,
                peer=args.peer,
                tag=args.tag,
                expected_seconds=args.expected_seconds,
                max_per_gpu=args.max_per_gpu,
                run_dir=run_dir,
                blocking=not args.non_blocking,
                timeout_seconds=args.timeout,
            )
        except GovernorBusy as e:
            payload = _base_cli_payload(
                command="acquire",
                status="timeout",
                ok=False,
                run_dir=run_dir,
            )
            payload.update(
                {
                    "gpu_id": args.gpu,
                    "pid": args.pid,
                    "peer": args.peer,
                    "tag": args.tag,
                    "expected_seconds": int(args.expected_seconds or 0),
                    "reason": "timeout",
                    "error": str(e),
                }
            )
            with contextlib.suppress(Exception):
                _with_slot_state(
                    payload,
                    _slot_state_payload(
                        args.gpu,
                        run_dir=run_dir,
                        max_per_gpu=args.max_per_gpu,
                    ),
                )
            _print_cli_payload(payload, pretty=args.pretty)
            sys.exit(2)
        except Exception as e:
            payload = _base_cli_payload(
                command="acquire",
                status="error",
                ok=False,
                run_dir=run_dir,
            )
            payload.update(
                {
                    "gpu_id": args.gpu,
                    "pid": args.pid,
                    "peer": args.peer,
                    "tag": args.tag,
                    "expected_seconds": int(args.expected_seconds or 0),
                    "reason": type(e).__name__,
                    "error": str(e),
                }
            )
            _print_cli_payload(payload, pretty=args.pretty)
            sys.exit(2)
        status = "acquired" if ok else "busy"
        payload = _base_cli_payload(
            command="acquire",
            status=status,
            ok=ok,
            run_dir=run_dir,
        )
        payload.update(
            {
                "gpu_id": args.gpu,
                "pid": args.pid,
                "peer": args.peer,
                "tag": args.tag,
                "expected_seconds": int(args.expected_seconds or 0),
            }
        )
        if not ok:
            payload["reason"] = "capacity_reached"
        _with_slot_state(
            payload,
            _slot_state_payload(args.gpu, run_dir=run_dir, max_per_gpu=args.max_per_gpu),
        )
        _print_cli_payload(payload, pretty=args.pretty)
        sys.exit(0 if ok else 1)

    if args.cmd == "release":
        if _is_bypassed():
            payload = _base_cli_payload(
                command="release",
                status="bypassed",
                ok=True,
                run_dir=run_dir,
            )
            payload.update(
                {
                    "gpu_id": args.gpu,
                    "pid": args.pid,
                    "max_per_gpu": _max_per_gpu(args.max_per_gpu),
                    "reason": "bypassed",
                }
            )
            _print_cli_payload(payload, pretty=args.pretty)
            sys.exit(0)
        try:
            ok = release_slot(args.gpu, pid=args.pid, run_dir=run_dir)
            state = _slot_state_payload(
                args.gpu,
                run_dir=run_dir,
                max_per_gpu=args.max_per_gpu,
            )
        except Exception as e:
            payload = _base_cli_payload(
                command="release",
                status="error",
                ok=False,
                run_dir=run_dir,
            )
            payload.update(
                {
                    "gpu_id": args.gpu,
                    "pid": args.pid,
                    "reason": type(e).__name__,
                    "error": str(e),
                }
            )
            _print_cli_payload(payload, pretty=args.pretty)
            sys.exit(2)
        payload = _base_cli_payload(
            command="release",
            status="released" if ok else "not_found",
            ok=ok,
            run_dir=run_dir,
        )
        payload.update({"gpu_id": args.gpu, "pid": args.pid})
        if not ok:
            payload["reason"] = "slot_not_found"
        _with_slot_state(payload, state)
        _print_cli_payload(payload, pretty=args.pretty)
        sys.exit(0 if ok else 1)

    if args.cmd == "list":
        try:
            gpu_states = {
                str(gpu): _slot_state_payload(
                    gpu,
                    run_dir=run_dir,
                    max_per_gpu=args.max_per_gpu,
                )
                for gpu in range(args.gpus)
            }
        except Exception as e:
            payload = _base_cli_payload(
                command="list",
                status="error",
                ok=False,
                run_dir=run_dir,
            )
            payload.update(
                {
                    "gpus_requested": args.gpus,
                    "reason": type(e).__name__,
                    "error": str(e),
                }
            )
            _print_cli_payload(payload, pretty=args.pretty)
            sys.exit(2)
        total_occupied = sum(state["occupied"] for state in gpu_states.values())
        total_available = sum(state["available"] for state in gpu_states.values())
        payload = _base_cli_payload(
            command="list",
            status="listed",
            ok=True,
            run_dir=run_dir,
        )
        payload.update(
            {
                "gpus_requested": args.gpus,
                "max_per_gpu": _max_per_gpu(args.max_per_gpu),
                "occupied": total_occupied,
                "available": total_available,
                "gpus": gpu_states,
            }
        )
        _print_cli_payload(payload, pretty=args.pretty)
        sys.exit(0)


if __name__ == "__main__":
    _cli()
