"""Praxist run registry — file-per-run state under ``$PRAXIST_STATE_DIR/runs/``.

The registry is the structured counterpart to the Phase 1 ``ps``-scan in
:mod:`praxist.cli.status`.  ``praxist start`` writes one
``<run_id>.json`` file per launched run; ``praxist stop`` reads it to find
the PID without re-parsing ``cmdline``; ``praxist status`` merges these
entries with the ``ps`` scan so operators see runs started both before
and after the registry landed.

Schema invariants:

* One file per run, atomically replaced on update so two concurrent
  writers can't corrupt each other.  No central index; ``list_entries``
  scans the directory.
* The file's ``run_id`` field always equals its basename (minus
  ``.json``).  Inconsistent files are reported as ``corrupt`` rather
  than silently dropped — that gives operators a chance to recover
  hand-edited state.
* ``schema_version`` is monotonic.  Readers refuse versions they don't
  understand rather than guess.  Version 1 is the only currently
  supported schema.

State directory selection uses ``$PRAXIST_STATE_DIR/runs/`` when set, else
``$XDG_DATA_HOME/praxist/runs/``, else ``~/.local/share/praxist/runs/``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

SCHEMA_VERSION = 1
"""Current registry schema version.  Bumped only for breaking changes."""


STATE_RUNNING = "running"
STATE_STOPPED = "stopped"
STATE_STALE = "stale"
STATE_VALUES: tuple[str, ...] = (STATE_RUNNING, STATE_STOPPED, STATE_STALE)

_DARWIN_BOOT_TIME_RE = re.compile(
    r"\bsec\s*=\s*(\d+)\s*,?\s*usec\s*=\s*(\d+)\b",
    re.IGNORECASE,
)


def state_dir() -> Path:
    """Return the operator state directory without creating it."""
    configured = os.environ.get("PRAXIST_STATE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return base / "praxist"


@dataclass(frozen=True)
class RegistryEntry:
    """One persisted ``praxist start`` run record.

    Frozen so callers cannot mutate the disk shape by accident; use
    :func:`dataclasses.replace` or :meth:`with_state` to derive new
    values.

    The ``command_prefix`` field is the literal prefix used by the
    TOCTOU guard in :mod:`praxist.cli.stop`: before signalling
    ``pid`` we re-read ``ps -p <pid> -o command=`` and confirm it
    starts with ``command_prefix``, so a recycled PID does not get an
    accidental SIGTERM.
    """

    schema_version: int
    run_id: str
    pid: int
    parent_pid: int
    run_dir: str
    log_file: str
    task_path: str
    model: str
    model_provider_ref: str
    runtime_ref: str
    command: tuple[str, ...]
    command_prefix: str
    started_at: str
    state: str = STATE_RUNNING
    stopped_at: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view (tuples → lists)."""
        data = asdict(self)
        data["command"] = list(self.command)
        return data

    def with_state(self, state: str, stopped_at: str | None = None) -> RegistryEntry:
        """Return a copy with ``state`` (and optionally ``stopped_at``) updated."""
        if state not in STATE_VALUES:
            raise ValueError(f"unknown registry state: {state!r}")
        return replace(self, state=state, stopped_at=stopped_at or self.stopped_at)


class RegistryError(RuntimeError):
    """Raised when a registry file cannot be read, written, or parsed."""


def runs_dir(*, create: bool = False) -> Path:
    """Return the registry directory.

    Args:
        create: When True, ensure the directory exists (write path).
            When False (default), return the would-be path without
            creating anything — useful for read paths so ``praxist status``
            on a fresh box does not silently materialize state dirs.
    """
    path = state_dir() / "runs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def entry_path(run_id: str, *, create: bool = False) -> Path:
    """Return the on-disk path for ``run_id`` (does not require the file to exist)."""
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return runs_dir(create=create) / f"{run_id}.json"


def write_entry(entry: RegistryEntry) -> Path:
    """Atomically persist ``entry`` to ``<runs_dir>/<run_id>.json``.

    Uses ``os.replace`` so two concurrent ``praxist start`` invocations
    cannot interleave a half-written file.  The temp file lives in the
    same directory as the target so the rename stays on one filesystem.
    """
    target = entry_path(entry.run_id, create=True)
    payload = json.dumps(entry.to_dict(), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        with contextlib.suppress(OSError):
            os.chmod(target, 0o600)
    finally:
        # If write_text succeeded but os.replace failed (e.g. ENOSPC),
        # clean up the temp file so the directory doesn't accumulate
        # debris.  Suppressed because the original error matters more.
        if tmp.exists():  # pragma: no cover - only reached on os.replace failure
            with contextlib.suppress(OSError):
                tmp.unlink()
    return target


def create_entry(entry: RegistryEntry) -> Path:
    """Atomically create a new entry without replacing an existing run id."""
    target = entry_path(entry.run_id, create=True)
    payload = json.dumps(entry.to_dict(), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        try:
            os.link(tmp, target)
        except FileExistsError as exc:
            raise RegistryError(f"run registry entry already exists: {entry.run_id}") from exc
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return target


@contextlib.contextmanager
def entry_lock(run_id: str):
    """Serialize lifecycle transitions for one run across CLI processes."""

    lock_path = entry_path(run_id, create=True).with_suffix(".lock")
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - process registry supports POSIX hosts.
        raise RegistryError("run lifecycle locking requires a POSIX host") from exc
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise RegistryError(f"could not open lifecycle lock for {run_id!r}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


@contextlib.contextmanager
def registry_lock():
    """Serialize registry-wide enumeration against new run reservations."""

    lock_path = runs_dir(create=True) / ".lifecycle.lock"
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - process registry supports POSIX hosts.
        raise RegistryError("registry lifecycle locking requires a POSIX host") from exc
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise RegistryError(f"could not open registry lifecycle lock: {exc}") from exc
    try:
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


def local_host_identity() -> dict[str, str]:
    """Return stable best-effort host fields for new registry entries."""
    identity = {"hostname": socket.gethostname()}
    configured_host_id = os.environ.get("PRAXIST_HOST_ID", "").strip()
    if configured_host_id:
        identity["host_id"] = hashlib.sha256(f"operator:{configured_host_id}".encode()).hexdigest()
    else:
        for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
            try:
                machine_id = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if machine_id:
                identity["host_id"] = hashlib.sha256(machine_id.encode()).hexdigest()
                break
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    try:
        boot_id = boot_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        boot_id = ""
    if boot_id:
        identity["boot_id"] = _normalized_boot_id(boot_id)
    elif sysctl := shutil.which("sysctl"):
        try:
            result = subprocess.run(
                [sysctl, "-n", "kern.boottime"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0 and result.stdout.strip():
            identity["boot_id"] = _normalized_boot_id(result.stdout)
    with contextlib.suppress(OSError):
        identity["pid_namespace"] = os.readlink("/proc/self/ns/pid")
    return identity


def _normalized_boot_id(value: str) -> str:
    """Normalize Darwin boot identity to its stable whole-second value."""

    token = str(value or "").strip()
    if token.startswith("darwin:"):
        parts = token.split(":")
        if len(parts) in {2, 3} and parts[1].isdigit():
            return f"darwin:{int(parts[1])}"
    match = _DARWIN_BOOT_TIME_RE.search(token)
    if match:
        return f"darwin:{int(match.group(1))}"
    return token


def entry_is_local(entry: RegistryEntry) -> bool | None:
    """Return whether ``entry`` belongs to this host.

    Legacy entries do not carry host identity and return ``None`` so existing
    local lifecycle behavior remains compatible.
    """
    hostname = entry.extra.get("hostname", "").strip()
    if not hostname:
        return None
    local = local_host_identity()
    recorded_host_id = entry.extra.get("host_id", "").strip()
    local_host_id = local.get("host_id", "")
    if recorded_host_id and local_host_id:
        if recorded_host_id != local_host_id:
            return False
        return _same_host_process_scope(entry, local)
    if hostname == local["hostname"]:
        return _same_host_process_scope(entry, local)
    recorded_boot = _normalized_boot_id(entry.extra.get("boot_id", ""))
    local_boot = _normalized_boot_id(local.get("boot_id", ""))
    if recorded_boot and local_boot:
        if recorded_boot != local_boot:
            return False
        recorded_namespace = entry.extra.get("pid_namespace", "").strip()
        local_namespace = local.get("pid_namespace", "")
        if recorded_namespace and local_namespace:
            return recorded_namespace == local_namespace
        return True
    return False


def _same_host_process_scope(entry: RegistryEntry, local: dict[str, str]) -> bool:
    """Distinguish an old host boot from another current container."""

    recorded_boot = _normalized_boot_id(entry.extra.get("boot_id", ""))
    local_boot = _normalized_boot_id(local.get("boot_id", ""))
    if recorded_boot and local_boot and recorded_boot != local_boot:
        return True
    recorded_namespace = entry.extra.get("pid_namespace", "").strip()
    local_namespace = local.get("pid_namespace", "")
    if recorded_namespace and local_namespace:
        return recorded_namespace == local_namespace
    return True


def entry_process_epoch_matches(entry: RegistryEntry) -> bool | None:
    """Return whether an entry belongs to the current boot/PID namespace."""

    local = local_host_identity()
    compared = False
    recorded_boot = _normalized_boot_id(entry.extra.get("boot_id", ""))
    local_boot = _normalized_boot_id(local.get("boot_id", ""))
    if recorded_boot and local_boot:
        compared = True
        if recorded_boot != local_boot:
            return False
    recorded_namespace = entry.extra.get("pid_namespace", "").strip()
    local_namespace = local.get("pid_namespace", "")
    if recorded_namespace and local_namespace:
        compared = True
        if recorded_namespace != local_namespace:
            return False
    return True if compared else None


def process_start_token(pid: int) -> str:
    """Return a process-instance token when the host exposes one.

    Linux ``/proc/<pid>/stat`` field 22 is the process start time in clock
    ticks since boot. Other POSIX hosts fall back to the process start timestamp
    reported by ``ps`` so lifecycle commands remain usable without ``/proc``.
    """
    if pid <= 0:
        return ""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        raw = ""
    if raw:
        close = raw.rfind(")")
        if close >= 0:
            fields = raw[close + 2 :].split()
            if len(fields) > 19:
                return f"proc:{fields[19]}"
    ps = shutil.which("ps")
    if not ps:
        return ""
    try:
        result = subprocess.run(
            [ps, "-p", str(pid), "-o", "lstart="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            env={"LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    started = " ".join(result.stdout.split())
    if result.returncode == 0 and started and started != "-":
        return f"ps:{started}"
    return ""


def process_identity_matches(entry: RegistryEntry) -> bool | None:
    """Compare a live PID with the process token recorded at launch."""
    expected = entry.extra.get("process_start_token", "").strip()
    if not expected:
        return None
    observed = process_start_token(entry.pid)
    if not observed:
        return None
    return observed == expected


def read_entry(run_id: str) -> RegistryEntry:
    """Read and validate a registry entry by ``run_id``.

    Raises:
        RegistryError: If the file is missing, unreadable, malformed, or
            written by a schema version this build does not understand.
    """
    path = entry_path(run_id)
    if not path.exists():
        raise RegistryError(f"no registry entry for run_id: {run_id!r}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"could not read registry entry {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"registry entry {path} is not valid JSON: {exc}") from exc
    return _entry_from_dict(data, source_path=path)


def list_entries() -> list[RegistryEntry]:
    """Return all readable registry entries, sorted by ``started_at``.

    Corrupt or unreadable entries are skipped silently here — operators
    discover them through :func:`iter_entries_with_errors` if they want
    explicit failures.  This is the read path most consumers (status,
    stop --all) want: a best-effort list.
    """
    entries: list[RegistryEntry] = []
    for entry, _err in iter_entries_with_errors():
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda e: e.started_at)
    return entries


def iter_entries_with_errors() -> Iterable[tuple[RegistryEntry | None, str | None]]:
    """Yield ``(entry, error_message)`` for every file under :func:`runs_dir`.

    Exactly one of the two is non-None per yielded pair.  Used by the
    operator-facing ``praxist status`` to surface corrupted entries rather
    than silently dropping them.
    """
    root = runs_dir(create=False)
    if not root.is_dir():
        return
    for path in sorted(root.glob("*.json")):
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            yield _entry_from_dict(data, source_path=path), None
        except (OSError, json.JSONDecodeError, RegistryError) as exc:
            yield None, f"{path.name}: {exc}"


def remove_entry(run_id: str) -> bool:
    """Delete the registry file for ``run_id``.  Returns ``True`` if removed."""
    path = entry_path(run_id)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def update_state(run_id: str, state: str, stopped_at: str | None = None) -> RegistryEntry:
    """Read, mutate, and rewrite an entry's ``state`` atomically.

    The read-modify-write is not transactional across processes; the
    invariant we care about is that the file on disk is never half
    written, which :func:`write_entry` guarantees via ``os.replace``.
    """
    entry = read_entry(run_id)
    new_entry = entry.with_state(state, stopped_at=stopped_at)
    write_entry(new_entry)
    return new_entry


def _entry_from_dict(data: object, source_path: Path) -> RegistryEntry:
    """Validate a raw JSON dict and construct a :class:`RegistryEntry`."""
    if not isinstance(data, dict):
        raise RegistryError(f"{source_path}: top-level JSON value must be an object")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RegistryError(
            f"{source_path}: schema_version {version!r} is not supported "
            f"(this build supports {SCHEMA_VERSION})"
        )
    required = (
        "run_id",
        "pid",
        "parent_pid",
        "run_dir",
        "log_file",
        "task_path",
        "model",
        "model_provider_ref",
        "runtime_ref",
        "command",
        "command_prefix",
        "started_at",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise RegistryError(f"{source_path}: missing required fields: {missing}")
    raw_run_id = data["run_id"]
    if source_path.stem != raw_run_id:
        raise RegistryError(f"{source_path}: run_id {raw_run_id!r} does not match filename")
    command = data["command"]
    if not isinstance(command, list) or not all(isinstance(c, str) for c in command):
        raise RegistryError(f"{source_path}: command must be a list of strings")
    state = data.get("state", STATE_RUNNING)
    if state not in STATE_VALUES:
        raise RegistryError(f"{source_path}: unknown state {state!r}")
    extra = data.get("extra", {})
    if not isinstance(extra, dict):
        raise RegistryError(f"{source_path}: extra must be an object")
    return RegistryEntry(
        schema_version=SCHEMA_VERSION,
        run_id=raw_run_id,
        pid=int(data["pid"]),
        parent_pid=int(data["parent_pid"]),
        run_dir=str(data["run_dir"]),
        log_file=str(data["log_file"]),
        task_path=str(data["task_path"]),
        model=str(data["model"]),
        model_provider_ref=str(data["model_provider_ref"]),
        runtime_ref=str(data["runtime_ref"]),
        command=tuple(command),
        command_prefix=str(data["command_prefix"]),
        started_at=str(data["started_at"]),
        state=state,
        stopped_at=data.get("stopped_at"),
        extra={str(k): str(v) for k, v in extra.items()},
    )
