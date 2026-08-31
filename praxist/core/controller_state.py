"""Private startup authority for controllers running separately from peer UIDs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

CONTROLLER_STATE_ENV = "PRAXIST_CONTROLLER_STATE_DIR"
_STARTUP_FILENAME = "startup_config.json"


def controller_state_enabled() -> bool:
    """Return whether the process explicitly selected private controller state."""
    return CONTROLLER_STATE_ENV in os.environ


def validate_controller_run_directory(run_dir: Path) -> None:
    """Require controller ownership of existing public run metadata roots.

    Args:
        run_dir: Public root; peers may write only explicitly provisioned children.

    Raises:
        ValueError: An existing root is a symlink, belongs to another UID, or is
            group/world writable. Unconfigured controller mode is unchanged.
    """
    if not controller_state_enabled():
        return
    try:
        descriptor = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("controller run root must be a safe directory") from exc
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
            raise ValueError("controller run root must be controller-owned and not peer-writable")
    finally:
        os.close(descriptor)


def _private_root(run_dir: Path) -> Path:
    raw = os.environ.get(CONTROLLER_STATE_ENV, "").strip()
    root = Path(raw)
    if not raw or not root.is_absolute():
        raise ValueError("private controller state requires an absolute directory")
    root = Path(os.path.abspath(root))
    canonical_run = run_dir.resolve()
    if root.is_relative_to(canonical_run) or canonical_run.is_relative_to(root):
        raise ValueError("private controller state must be separate from public run outputs")
    return root


def _check_directory(descriptor: int, *, private: bool) -> None:
    info = os.fstat(descriptor)
    uid = os.geteuid()
    if info.st_uid not in {0, uid}:
        raise ValueError("private controller state has an untrusted directory owner")
    mode = stat.S_IMODE(info.st_mode)
    if private:
        if info.st_uid != uid or mode != 0o700:
            raise ValueError(
                "private controller state directories must be controller-owned mode 0700"
            )
    elif mode & 0o022 and not (mode & stat.S_ISVTX):
        raise ValueError("private controller state has a writable ancestor directory")


@contextmanager
def _run_directory(run_dir: Path, *, create: bool) -> Iterator[tuple[Path, int]]:
    root = _private_root(run_dir)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(root.anchor, flags)
    try:
        _check_directory(descriptor, private=False)
        for index, component in enumerate(root.parts[1:], start=1):
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            _check_directory(descriptor, private=index == len(root.parts) - 1)
        run_hash = hashlib.sha256(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:24]
        if create:
            with suppress(FileExistsError):
                os.mkdir(run_hash, mode=0o700, dir_fd=descriptor)
        child = os.open(run_hash, flags, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = child
        _check_directory(descriptor, private=True)
        yield root / run_hash, descriptor
    except OSError as exc:
        raise ValueError("private controller startup authority is unavailable or unsafe") from exc
    finally:
        os.close(descriptor)


def private_controller_run_dir(run_dir: Path, *, create: bool = False) -> Path | None:
    """Return the validated private per-run directory when explicitly configured.

    Args:
        run_dir: Public run directory whose resolved path identifies the run.
        create: Whether to create missing private directories with mode 0700.

    Returns:
        The private directory, or None when controller state is not configured.

    Raises:
        ValueError: The configured directory is missing or unsafe.
    """
    if not controller_state_enabled():
        return None
    validate_controller_run_directory(run_dir)
    with _run_directory(Path(run_dir), create=create) as (path, _descriptor):
        return path


def _validate_startup_snapshot(payload: Any, run_dir: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != "praxist.startup.v1":
        raise ValueError("private startup authority must be a Praxist startup object")
    args = payload.get("canonical_args")
    if not isinstance(args, dict):
        raise ValueError("private startup authority is missing canonical arguments")
    for name in (
        "task",
        "task_path",
        "run_dir",
        "runtime",
        "model_provider",
        "budget_policy",
        "model",
    ):
        if not isinstance(args.get(name), str) or not args[name].strip():
            raise ValueError(f"private startup authority is missing canonical {name}")
    if Path(args["run_dir"]).resolve() != run_dir.resolve():
        raise ValueError("private startup authority belongs to another run directory")
    return payload


def read_private_startup_config(run_dir: Path) -> dict[str, Any] | None:
    """Read startup authority without following public artifacts or symlinks.

    Args:
        run_dir: Public run directory bound to this private startup snapshot.

    Returns:
        The original startup snapshot, or None only when controller mode is unset.

    Raises:
        ValueError: Configured private authority is absent, corrupt, or unsafe.
    """
    if not controller_state_enabled():
        return None
    validate_controller_run_directory(run_dir)
    try:
        with _run_directory(Path(run_dir), create=False) as (_path, directory):
            descriptor = os.open(_STARTUP_FILENAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    raise ValueError("private startup authority must be controller-owned mode 0600")
                if info.st_nlink != 1:
                    # A crash after atomic publication can leave its private
                    # temporary hardlink behind. Only recover matching links in
                    # this controller-owned directory, never unexplained ones.
                    for name in os.listdir(directory):
                        if not (name.startswith(".startup-") and name.endswith(".tmp")):
                            continue
                        try:
                            candidate = os.stat(name, dir_fd=directory, follow_symlinks=False)
                            if (candidate.st_dev, candidate.st_ino) == (info.st_dev, info.st_ino):
                                os.unlink(name, dir_fd=directory)
                        except FileNotFoundError:
                            pass
                    if os.fstat(stream.fileno()).st_nlink != 1:
                        raise ValueError("private startup authority has unexplained hardlinks")
                payload = json.load(stream)
        return _validate_startup_snapshot(payload, Path(run_dir))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("private startup authority is unavailable or corrupt") from exc


def write_private_startup_config(run_dir: Path, payload: Mapping[str, Any]) -> None:
    """Commit the initial startup snapshot once, before public run artifacts.

    Args:
        run_dir: Public run directory identifying this controller-owned snapshot.
        payload: Complete canonical startup configuration prepared by the controller.

    Raises:
        ValueError: Private authority already exists or cannot be safely committed.
    """
    if not controller_state_enabled():
        return
    validate_controller_run_directory(run_dir)
    snapshot = _validate_startup_snapshot(dict(payload), Path(run_dir))
    root = _private_root(Path(run_dir))
    if root.is_relative_to(Path(snapshot["canonical_args"]["task_path"]).resolve()):
        raise ValueError("private controller state must be separate from the task project")
    encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
    with _run_directory(Path(run_dir), create=True) as (_path, directory):
        temporary = f".startup-{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            # link is atomic and refuses to replace existing authority. An
            # operator must use a new run directory, never reseed from a mirror.
            os.link(
                temporary,
                _STARTUP_FILENAME,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory)
        os.fsync(directory)
