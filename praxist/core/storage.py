"""Run directory, JSONL, and artifact helpers for Gate A."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from praxist.core.redaction import redact_json


def utc_now() -> str:
    """Return the current UTC timestamp used by Praxist run artifacts."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")


def new_run_id(task_slug: str) -> str:
    """Generate a new opaque run identifier."""
    return f"{utc_stamp()}_{task_slug}_{secrets.token_hex(4)}"


@contextmanager
def _artifact_parent(path: Path, *, create: bool = True) -> Iterator[tuple[int, str]]:
    absolute = Path(os.path.abspath(path))
    if "PRAXIST_CONTROLLER_STATE_DIR" not in os.environ:
        # Ordinary operator-selected symlink storage remains supported. The
        # separate-UID controller opts into strict ancestor checks explicitly.
        absolute = absolute.parent.resolve() / absolute.name
    remaining = list(absolute.parent.parts[1:])
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    root_info = os.fstat(descriptor)
    aliases = 0
    try:
        while remaining:
            component = remaining.pop(0)
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                with suppress(FileExistsError):
                    os.mkdir(component, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError:
                parent_info = os.fstat(descriptor)
                link_info = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                # Permit only the real filesystem-root macOS aliases. Root
                # ownership alone does not make a link inside run data safe.
                if (
                    not stat.S_ISLNK(link_info.st_mode)
                    or component not in {"var", "tmp", "etc"}
                    or (parent_info.st_dev, parent_info.st_ino)
                    != (root_info.st_dev, root_info.st_ino)
                    or link_info.st_uid != 0
                    or parent_info.st_uid != 0
                    or stat.S_IMODE(parent_info.st_mode) & 0o022
                    or aliases >= 40
                ):
                    raise
                target = Path(os.readlink(component, dir_fd=descriptor))
                if target not in {
                    Path("private") / component,
                    Path(absolute.anchor) / "private" / component,
                }:
                    raise
                aliases += 1
                if target.is_absolute():
                    child = os.open(target.anchor, flags)
                    os.close(descriptor)
                    descriptor = child
                    remaining = list(target.parts[1:]) + remaining
                else:
                    remaining = list(target.parts) + remaining
                continue
            os.close(descriptor)
            descriptor = child
        yield descriptor, absolute.name
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o666) -> None:
    """Atomically replace a file without following peer-controlled symlinks.

    Args:
        path: Destination file. Missing parent directories are created.
        payload: Already encoded bytes; callers own any required redaction.
        mode: Creation permissions, further restricted by the process umask.

    Raises:
        OSError: A parent is an untrusted symlink in controller mode or the
            write cannot complete.
    """
    with _artifact_parent(path) as (directory, filename):
        temporary = f".{filename}.{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
            dir_fd=directory,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, filename, src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory)


def write_json(path: Path, value: Any) -> None:
    """Write a redacted JSON artifact with stable indentation and sorted keys."""
    redacted, _ = redact_json(value)
    atomic_write_bytes(
        path,
        (json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


@contextmanager
def open_append_file(path: Path, *, mode: int = 0o666) -> Iterator[BinaryIO]:
    """Open a regular append-only file without following untrusted links.

    Args:
        path: File to append to, creating missing parents and the file as needed.
        mode: Creation permissions, further restricted by the process umask.

    Yields:
        A binary append stream; callers may pass its descriptor to subprocesses.

    Raises:
        OSError: A directory is an untrusted symlink in controller mode, the file
            is a symlink or hardlink, or the destination is not a regular file.
    """
    with _artifact_parent(path) as (directory, filename):
        descriptor = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
            mode,
            dir_fd=directory,
        )
        with os.fdopen(descriptor, "ab") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError("append-only artifacts must be regular files without hardlinks")
            yield stream


@contextmanager
def open_readonly_file(path: Path) -> Iterator[BinaryIO]:
    """Open a regular file without following links to controller-private files.

    Args:
        path: File to read. Missing files or parent directories are not created.

    Yields:
        A binary stream reading through a pinned descriptor.

    Raises:
        OSError: The file is absent, not regular, or symlinked/hardlinked. In
            controller mode, peer-controlled ancestor symlinks also fail closed.
    """
    with _artifact_parent(path, create=False) as (directory, filename):
        descriptor = os.open(
            filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError("artifact reads require regular files without hardlinks")
            yield stream


def read_file_bytes(path: Path) -> bytes:
    """Read bytes with the no-follow regular-file checks of open_readonly_file.

    Args:
        path: File to read without creating missing files or directories.

    Returns:
        The complete file content.

    Raises:
        OSError: The file is missing, unsafe, or cannot be read.
    """
    with open_readonly_file(path) as stream:
        return stream.read()


def append_jsonl(path: Path, value: Any) -> None:
    """Append one redacted JSONL record to an append-only run ledger."""
    redacted, _ = redact_json(value)
    encoded = json.dumps(redacted, ensure_ascii=False, sort_keys=True) + "\n"
    with open_append_file(path) as stream:
        stream.write(encoded.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read JSONL records from a run ledger, skipping blank lines."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return records, [f"missing:{path.name}"]
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
            else:
                errors.append(f"{path.name}:{line_no}:not_object")
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_no}:json_decode:{exc.msg}")
    return records, errors


def rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically rewrite an entire JSONL ledger with redacted records."""
    encoded = []
    for record in records:
        redacted, _ = redact_json(record)
        encoded.append(json.dumps(redacted, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_write_bytes(path, "".join(encoded).encode("utf-8"))


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for raw bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


OUTPUT_LEDGER_RELS = (
    "findings/findings.jsonl",
    "findings/frontier.jsonl",
    "memory/research_memory.jsonl",
    "memory/graph_edges.jsonl",
)


def output_ledger_hashes(run_dir: Path) -> dict[str, str]:
    """Compute content hashes for canonical output ledgers in a run directory."""
    run_dir = Path(run_dir)
    hashes: dict[str, str] = {}
    for rel in OUTPUT_LEDGER_RELS:
        path = run_dir / rel
        hashes[rel] = sha256_bytes(path.read_bytes() if path.exists() else b"")
    return hashes


def ensure_run_dirs(run_dir: Path) -> None:
    """Create the minimum run directory layout used by startup and replay."""
    for rel in (
        "artifacts/by_id",
        "findings",
        "memory",
        "logs",
        "indexes",
        "replay",
    ):
        (run_dir / rel).mkdir(parents=True, exist_ok=True)


class ArtifactWriter:
    """Run-local artifact writer with stable ids, redaction, and artifact_index accounting."""

    def __init__(self, run_dir: Path, trajectory: Any | None = None) -> None:
        self.run_dir = run_dir
        self.trajectory = trajectory
        self.run_id = str(getattr(trajectory, "run_id", run_dir.name))
        self._seq = _existing_artifact_seq(run_dir)

    def persist_json(
        self,
        artifact_type: str,
        logical_path: str,
        payload: dict[str, Any],
        *,
        schema_ref: str | None,
        producer: dict[str, str],
        source_event_ids: list[str] | None = None,
        source_artifact_ids: list[str] | None = None,
        redaction_level: str = "redacted",
        artifact_role: str | None = None,
        artifact_status: str | None = None,
        runtime_fact_source: bool | None = None,
        derived_from: list[str] | None = None,
    ) -> dict[str, Any]:
        redacted_payload, hits = redact_json(payload)
        payload_bytes = (
            json.dumps(redacted_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        return self._persist_payload(
            artifact_type=artifact_type,
            logical_path=logical_path,
            payload_bytes=payload_bytes,
            payload_name="payload.json",
            content_type="application/json",
            schema_ref=schema_ref,
            producer=producer,
            source_event_ids=source_event_ids or [],
            source_artifact_ids=source_artifact_ids or [],
            redaction_level=redaction_level,
            redaction_hits=hits,
            artifact_role=artifact_role,
            artifact_status=artifact_status,
            runtime_fact_source=runtime_fact_source,
            derived_from=derived_from,
        )

    def persist_text(
        self,
        artifact_type: str,
        logical_path: str,
        payload: str,
        *,
        schema_ref: str | None,
        producer: dict[str, str],
        content_type: str = "text/markdown",
        source_event_ids: list[str] | None = None,
        artifact_role: str | None = None,
        artifact_status: str | None = None,
        runtime_fact_source: bool | None = None,
        derived_from: list[str] | None = None,
    ) -> dict[str, Any]:
        redacted_payload, hits = redact_json(payload)
        payload_bytes = str(redacted_payload).encode("utf-8")
        return self._persist_payload(
            artifact_type=artifact_type,
            logical_path=logical_path,
            payload_bytes=payload_bytes,
            payload_name="payload.md",
            content_type=content_type,
            schema_ref=schema_ref,
            producer=producer,
            source_event_ids=source_event_ids or [],
            source_artifact_ids=[],
            redaction_level="redacted",
            redaction_hits=hits,
            artifact_role=artifact_role,
            artifact_status=artifact_status,
            runtime_fact_source=runtime_fact_source,
            derived_from=derived_from,
        )

    def _persist_payload(
        self,
        *,
        artifact_type: str,
        logical_path: str,
        payload_bytes: bytes,
        payload_name: str,
        content_type: str,
        schema_ref: str | None,
        producer: dict[str, str],
        source_event_ids: list[str],
        source_artifact_ids: list[str],
        redaction_level: str,
        redaction_hits: list[str],
        artifact_role: str | None,
        artifact_status: str | None,
        runtime_fact_source: bool | None,
        derived_from: list[str] | None,
    ) -> dict[str, Any]:
        artifact_id, artifact_dir = self._next_artifact_dir()
        payload_path = artifact_dir / payload_name
        atomic_write_bytes(payload_path, payload_bytes)
        redacted_logical_path, logical_path_hits = redact_json(logical_path)
        metadata = {
            "schema_version": "praxist.artifact.v1",
            "artifact_id": artifact_id,
            "run_id": self.run_id,
            "artifact_type": artifact_type,
            "logical_path": str(redacted_logical_path),
            "payload_path": f"artifacts/by_id/{artifact_id}/{payload_name}",
            "content_hash": sha256_bytes(payload_bytes),
            "content_type": content_type,
            "schema_ref": schema_ref,
            "size_bytes": len(payload_bytes),
            "producer": producer,
            "source_event_ids": source_event_ids,
            "source_artifact_ids": source_artifact_ids,
            "redaction_level": redaction_level,
            "redaction_hits": sorted(set([*redaction_hits, *logical_path_hits])),
            "created_at": utc_now(),
        }
        if artifact_role:
            metadata["artifact_role"] = str(artifact_role)
        if artifact_status:
            metadata["artifact_status"] = str(artifact_status)
        if runtime_fact_source is not None:
            metadata["runtime_fact_source"] = bool(runtime_fact_source)
        if derived_from:
            metadata["derived_from"] = [str(item) for item in derived_from if str(item).strip()]
        write_json(artifact_dir / "metadata.json", metadata)
        append_jsonl(self.run_dir / "artifact_index.jsonl", metadata)
        if self.trajectory is not None:
            self.trajectory.emit(
                "artifact.persisted",
                actor={"type": "core", "id": "artifact_writer"},
                payload={
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "logical_path": logical_path,
                    "content_hash": metadata["content_hash"],
                },
                artifact_refs=[metadata],
            )
        return metadata

    def _next_artifact_dir(self) -> tuple[str, Path]:
        while True:
            self._seq += 1
            artifact_id = f"art_{self._seq:06d}"
            artifact_dir = self.run_dir / "artifacts" / "by_id" / artifact_id
            try:
                artifact_dir.mkdir(parents=True, exist_ok=False)
                return artifact_id, artifact_dir
            except FileExistsError:
                continue


def _existing_artifact_seq(run_dir: Path) -> int:
    highest = 0
    artifact_root = run_dir / "artifacts" / "by_id"
    if artifact_root.exists():
        for path in artifact_root.glob("art_*"):
            suffix = path.name.removeprefix("art_")
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    index_path = run_dir / "artifact_index.jsonl"
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            artifact_id = str(record.get("artifact_id", ""))
            suffix = artifact_id.removeprefix("art_")
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return highest
