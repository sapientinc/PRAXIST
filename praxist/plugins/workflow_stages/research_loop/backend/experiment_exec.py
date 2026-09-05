"""Tiny launch barrier used by the central experiment scheduler."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _pid_start_time(pid: int) -> int | str | None:
    """Return the launcher's process-instance token without project imports."""

    try:
        suffix = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        return int(suffix.split()[19])
    except (OSError, ValueError, IndexError):
        pass
    ps = shutil.which("ps")
    if not ps:
        return None
    try:
        completed = subprocess.run(
            [ps, "-p", str(pid), "-o", "lstart="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            env={"LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    started = " ".join(completed.stdout.split())
    return f"ps:{started}" if completed.returncode == 0 and started else None


def main() -> int:
    """Wait for scheduler commit, then replace this barrier with the task command."""

    if len(sys.argv) < 5:
        return 75
    ready_path = Path(sys.argv[1])
    go_path = Path(sys.argv[2])
    attempt_id = sys.argv[3]
    command = sys.argv[4:]
    pid_start_time = _pid_start_time(os.getpid())
    temporary = ready_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "attempt_id": attempt_id,
                "pid_start_time": pid_start_time,
            }
        ),
        encoding="utf-8",
    )
    os.replace(temporary, ready_path)
    deadline = time.monotonic() + 300.0
    while not go_path.exists():
        if time.monotonic() >= deadline:
            return 75
        time.sleep(0.05)
    try:
        os.execvpe(command[0], command, os.environ)
    except OSError:
        return 75
    return 75  # pragma: no cover - exec never returns on success.


if __name__ == "__main__":
    raise SystemExit(main())
