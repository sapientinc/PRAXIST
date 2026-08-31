"""Preserve generation evidence without implying an evaluated objective."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from praxist.core.storage import write_json

logger = logging.getLogger(__name__)
SCORELESS_EVIDENCE_FILENAME = "scoreless_evidence.json"


def is_scoreless(task_spec: Any) -> bool:
    """Return whether a task explicitly selects scoreless research.

    Args:
        task_spec: Loaded task specification or descriptor mapping. Older
            adapters may retain the descriptor only in ``_raw``.

    Returns:
        True only for an explicitly declared ``research_loop.mode: scoreless``.
    """
    config = (
        task_spec.get("research_loop")
        if isinstance(task_spec, dict)
        else getattr(task_spec, "research_loop", None)
    )
    if not isinstance(config, dict) or not config:
        raw = getattr(task_spec, "_raw", None)
        config = raw.get("research_loop") if isinstance(raw, dict) else None
    return isinstance(config, dict) and config.get("mode") == "scoreless"


def write_scoreless_evidence_manifest(
    run_dir: Path,
    *,
    gen_id: int,
    findings: list[dict[str, Any]],
    evidence_cutoff_at: str,
    evidence_source_snapshot: dict[str, str],
) -> Path:
    """Write complete frozen findings before committing a scoreless boundary.

    Args:
        run_dir: Canonical run directory.
        gen_id: Generation whose cutoff has been frozen.
        findings: Canonical findings accepted at that cutoff, of every type.
        evidence_cutoff_at: Timestamp of the existing generation cutoff.
        evidence_source_snapshot: Source identities from that same cutoff.

    Returns:
        Path of the atomically written, redacted evidence artifact.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
        compact_boundary_source_snapshot,
    )

    path = Path(run_dir) / f"gen_{gen_id}" / SCORELESS_EVIDENCE_FILENAME
    write_json(
        path,
        {
            "schema_version": 1,
            "mode": "scoreless",
            "generation_id": gen_id,
            "evidence_status": "not_scored",
            "evidence_cutoff_at": evidence_cutoff_at,
            "evidence_source_snapshot_at_cutoff": compact_boundary_source_snapshot(
                evidence_source_snapshot
            ),
            "retained_count": len(findings),
            "findings": findings,
        },
    )
    return path


def read_scoreless_evidence_manifest(run_dir: Path, gen_id: int) -> dict[str, Any] | None:
    """Read one complete frozen manifest without following symlinks.

    Args:
        run_dir: Canonical run directory.
        gen_id: Generation whose manifest is required.

    Returns:
        The full manifest, or None if unavailable, unsafe, or malformed.
    """
    from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
        read_bounded_file_under_root_no_follow,
    )

    root = Path(run_dir)
    path = root / f"gen_{gen_id}" / SCORELESS_EVIDENCE_FILENAME
    try:
        size = path.lstat().st_size
    except OSError:
        return None
    # Storage retains complete task-budgeted evidence. Only prompt projections
    # are truncated; a fixed storage cap could reject our own committed output.
    data = read_bounded_file_under_root_no_follow(path, root, max_bytes=size)
    if data is None:
        return None
    try:
        manifest = json.loads(data)
    except (ValueError, UnicodeError):
        logger.warning("Scoreless evidence manifest unreadable for generation %d", gen_id)
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("generation_id") != gen_id
        or manifest.get("mode") != "scoreless"
        or not isinstance(manifest.get("findings"), list)
    ):
        return None
    return manifest


def load_scoreless_evidence(
    run_dir: Path,
    completed_gen_id: int,
    *,
    max_findings: int = 100,
    max_content_chars: int = 1200,
) -> list[dict[str, Any]]:
    """Read bounded narrative context from frozen generation evidence.

    Args:
        run_dir: Canonical run directory.
        completed_gen_id: Latest generation the consumer may inspect.
        max_findings: Maximum number of findings, newest generations first.
        max_content_chars: Maximum characters in each narrative field.

    Returns:
        Finding summaries with source-manifest references. Full payloads stay
        in their manifests; arbitrary nested metadata and ranking metrics are
        deliberately not copied into the bounded prompt context.
    """
    if completed_gen_id < 0 or max_findings <= 0:
        return []
    root = Path(run_dir)
    limit = max(0, max_content_chars)
    rows: list[dict[str, Any]] = []
    for gen_id in range(completed_gen_id, -1, -1):
        path = root / f"gen_{gen_id}" / SCORELESS_EVIDENCE_FILENAME
        manifest = read_scoreless_evidence_manifest(root, gen_id)
        if manifest is None:
            continue
        for finding in manifest["findings"]:
            if not isinstance(finding, dict):
                continue
            row: dict[str, Any] = {
                "generation_id": gen_id,
                "evidence_manifest": str(path.relative_to(root)),
                "evidence_status": "not_scored",
            }
            for key in (
                "id",
                "finding_type",
                "peer_id",
                "variant_name",
                "timestamp",
                "source_finding_path",
                "source_result_path",
            ):
                value = finding.get(key)
                if isinstance(value, (str, int, float, bool)):
                    row[key] = value[:240] if isinstance(value, str) else value
            for key in ("title", "content", "summary"):
                value = str(finding.get(key) or "")
                row[key] = value[:limit]
                if len(value) > limit:
                    row[f"{key}_truncated"] = True
            rows.append(row)
            if len(rows) >= max_findings:
                return rows
    return rows
