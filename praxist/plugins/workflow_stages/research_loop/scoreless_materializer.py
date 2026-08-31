"""Archive committed scoreless evidence and task deliveries for replay."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from praxist.core.redaction import redact_text
from praxist.core.storage import ArtifactWriter
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    CANONICAL_STATE,
    COMMITTED,
    is_committed_runtime_fact_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.task_lifecycle import _open_directory


@dataclass(frozen=True)
class CommittedSnapshot:
    """A single source read used for both replay content and its source hash."""

    relative_path: str
    data: bytes
    payload: Any
    encoding: str
    artifact_type: str
    redaction_hits: tuple[str, ...] = ()

    @property
    def source_sha256(self) -> str:
        """Return the hash of the exact source bytes, before redaction."""
        return hashlib.sha256(self.data).hexdigest()


def _read_source(run_dir: Path, relative: str) -> bytes:
    if not isinstance(relative, str):
        raise ValueError("scoreless artifact path must be a string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("scoreless artifact path must be run-relative")
    directory = _open_directory(run_dir)
    try:
        for component in path.parts[:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("scoreless artifact must be a regular file without hardlinks")
            return source.read()
    finally:
        os.close(directory)


def _committed_boundaries(run_dir: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    generation = 0
    while True:
        relative = f"gen_{generation}/generation_boundary.json"
        try:
            marker = json.loads(_read_source(run_dir, relative))
        except FileNotFoundError:
            return
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError(f"scoreless generation boundary is unsafe: {relative}") from exc
        if (
            not isinstance(marker, dict)
            or marker.get("generation_id") != generation
            or not is_committed_runtime_fact_source(marker, legacy_ok=False)
        ):
            raise ValueError(f"scoreless generation boundary is not committed: {relative}")
        yield generation, marker
        generation += 1


def collect_committed_scoreless_sources(
    run_dir: Path, result: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[CommittedSnapshot]]:
    """Read frozen boundaries and hash-verified terminal task artifacts.

    Args:
        run_dir: Resolved canonical run root.
        result: Controller result containing any committed task delivery and
            its controller-owned artifact hashes.

    Returns:
        Full findings and immutable in-memory snapshots of their source files.
        Partial or live findings remain on disk and are not promoted to facts.

    Raises:
        ValueError: Committed evidence is missing, malformed, or changed.
        OSError: An artifact path is unsafe or cannot be read.
    """
    snapshots: list[CommittedSnapshot] = []
    findings: dict[str, dict[str, Any]] = {}
    for generation, marker in _committed_boundaries(run_dir):
        relative = f"gen_{generation}/scoreless_evidence.json"
        try:
            data = _read_source(run_dir, relative)
            manifest = json.loads(data)
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError(f"committed scoreless evidence is unavailable: {relative}") from exc
        if hashlib.sha256(data).hexdigest() != marker.get("scoreless_evidence_sha256"):
            raise ValueError(f"committed scoreless evidence hash changed or missing: {relative}")
        if (
            not isinstance(manifest, dict)
            or manifest.get("mode") != "scoreless"
            or manifest.get("generation_id") != generation
            or not isinstance(manifest.get("findings"), list)
            or any(not isinstance(finding, dict) for finding in manifest["findings"])
        ):
            raise ValueError(f"committed scoreless evidence is malformed: {relative}")
        snapshots.append(
            CommittedSnapshot(relative, data, manifest, "json", "scoreless_evidence_manifest")
        )
        for finding in manifest["findings"]:
            identity = finding.get("id") or finding.get("finding_id")
            if identity:
                findings[str(identity)] = finding

    delivery = result.get("task_delivery") or {}
    if delivery.get("status") == "completed":
        paths = delivery.get("artifacts")
        hashes = delivery.get("artifact_hashes")
        if not isinstance(paths, list) or not isinstance(hashes, dict):
            raise ValueError("committed task delivery requires controller artifact hashes")
        for relative in dict.fromkeys(paths):
            data = _read_source(run_dir, relative)
            if hashlib.sha256(data).hexdigest() != hashes.get(relative):
                raise ValueError(f"committed task delivery hash changed: {relative}")
            redaction_hits: tuple[str, ...] = ()
            try:
                payload = data.decode("utf-8")
                encoding = "utf-8"
            except UnicodeError:
                # Latin-1 preserves every byte while exposing ASCII secrets to
                # the ordinary redactor before base64 would conceal them.
                redacted, hits = redact_text(data.decode("latin-1"))
                redaction_hits = tuple(hits)
                payload = base64.b64encode(redacted.encode("latin-1")).decode("ascii")
                encoding = "base64"
            if Path(relative).suffix.lower() == ".json":
                try:
                    payload = json.loads(data)
                    encoding = "json"
                except (ValueError, UnicodeError):
                    # Artifact validity belongs to the task callback. Replay
                    # still retains exact unparsed content from failed formats.
                    pass
            snapshots.append(
                CommittedSnapshot(
                    relative, data, payload, encoding, "task_delivery", redaction_hits
                )
            )
    return list(findings.values()), snapshots


def archive_committed_scoreless_sources(
    writer: ArtifactWriter, snapshots: list[CommittedSnapshot]
) -> list[dict[str, Any]]:
    """Persist redacted replay copies with hashes of the original source bytes.

    Args:
        writer: Canonical artifact writer for this run.
        snapshots: Files already read and verified by the controller importer.

    Returns:
        Indexed artifact references with immutable replay payload paths.
    """
    return [
        writer.persist_json(
            snapshot.artifact_type,
            snapshot.relative_path,
            {
                "source_path": snapshot.relative_path,
                "source_sha256": snapshot.source_sha256,
                "source_encoding": snapshot.encoding,
                "source_redaction_hits": list(snapshot.redaction_hits),
                "payload": snapshot.payload,
            },
            schema_ref="research_loop:committed_source_snapshot.v1",
            producer={"stage_id": "research_loop", "role_ref": "workflow_stage:research_loop"},
            artifact_role=CANONICAL_STATE,
            artifact_status=COMMITTED,
            runtime_fact_source=True,
            derived_from=[snapshot.relative_path],
        )
        for snapshot in snapshots
    ]
