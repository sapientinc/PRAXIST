"""Process-group lifecycle helpers for centrally launched experiments."""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def process_group_alive(pgid: int) -> bool:
    """Return whether a process group still contains executable work."""

    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass

    linux_activity = _linux_process_group_activity(pgid)
    return True if linux_activity is None else linux_activity


def _linux_process_group_activity(pgid: int) -> bool | None:
    """Distinguish a zombie-only group on Linux; defer elsewhere.

    ``killpg(..., 0)`` reports zombie members as present even though they can
    never make progress. Returning ``None`` preserves the portable killpg
    result when procfs is unavailable or no member can be observed.
    """

    if sys.platform != "linux":
        return None
    proc_root = Path("/proc")
    if not proc_root.is_dir():  # pragma: no cover - non-procfs Linux
        return None

    observed_member = False
    try:
        with os.scandir(proc_root) as entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    stat = Path(entry.path, "stat").read_bytes()
                    fields = stat.rsplit(b")", 1)[1].split()
                    state, member_pgid = fields[0].decode("ascii"), int(fields[2])
                    if member_pgid != pgid:
                        continue
                    observed_member = True
                    if state not in {"X", "x", "Z"}:
                        return True
                except (OSError, IndexError, ValueError):
                    continue
    except OSError:
        return None
    return False if observed_member else None


def terminate_process_group(
    pgid: int,
    process: subprocess.Popen[bytes] | None = None,
    *,
    grace_seconds: float = 5.0,
) -> None:
    """Terminate an experiment process group and optionally reap its leader."""

    try:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while process_group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if process_group_alive(pgid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        if process is not None:
            process.wait(timeout=grace_seconds)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("could not fully drain experiment process group %s: %s", pgid, exc)
        if process is not None:
            try:
                process.kill()
                process.wait(timeout=grace_seconds)
            except (OSError, subprocess.SubprocessError):
                pass
