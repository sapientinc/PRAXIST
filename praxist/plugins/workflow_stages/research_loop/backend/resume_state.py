"""Resume-state inspection for research-loop runs.

This module intentionally implements a conservative resume contract: Praxist may
continue from a completed generation boundary, or repair a generation whose
cohort finished but whose boundary/PI step did not. It never guesses that an
in-flight cohort is complete.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    CANONICAL_STATE,
    COMMITTED,
    PARTIAL,
    artifact_semantics,
    is_committed_runtime_fact_file,
    is_committed_runtime_fact_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.tools.atomic_io import (
    atomic_write_json,
)

BOUNDARY_MARKER_FILENAME = "generation_boundary.json"
BOUNDARY_CHECKPOINT_SIGNAL_FILENAME = "CLOSING_SIGNAL"
BOUNDARY_CHECKPOINT_CUTOFF_KEY = "boundary_evidence_cutoff_at"
BOUNDARY_CHECKPOINT_SNAPSHOT_KEY = "boundary_evidence_source_snapshot"
RESUME_EVENTS_FILENAME = "resume_events.jsonl"
RESUME_CANONICAL_ARG_KEYS = (
    "task",
    "runtime",
    "model_provider",
    "budget_policy",
    "model",
    "frontier_strategy",
    "run_dir",
)


def canonical_completed_generation_count(run_dir: Path) -> int:
    """Count contiguous committed generation-boundary markers on disk."""

    completed = 0
    while _valid_boundary_marker(Path(run_dir), completed):
        completed += 1
    return completed


def reported_completed_generations(result: dict[str, Any], run_dir: Path) -> int:
    """Prefer committed markers while retaining legacy markerless summaries."""

    marker_contract_start = _boundary_marker_contract_start(Path(run_dir))
    if marker_contract_start is not None:
        completed = marker_contract_start
        while _valid_boundary_marker(Path(run_dir), completed):
            completed += 1
        return completed
    try:
        return max(0, int(result.get("generations_completed", 0) or 0))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class ResumePlan:
    """Safe point from which a research-loop run can continue."""

    enabled: bool
    policy: str
    start_generation: int = 0
    completed_generations: int = 0
    pending_boundary_generation: int | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_pending_boundary(self) -> bool:
        return self.pending_boundary_generation is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "policy": self.policy,
            "start_generation": self.start_generation,
            "completed_generations": self.completed_generations,
            "pending_boundary_generation": self.pending_boundary_generation,
            "warnings": list(self.warnings),
        }


def inspect_resume_plan(
    run_dir: Path,
    *,
    max_generations: int,
    pi_enabled: bool,
    policy: str = "completed_generation",
) -> ResumePlan:
    """Inspect ``run_dir`` and return the safest continuation point.

    ``completed_generation`` means Praxist only skips generations whose cohort and
    boundary are both durably materialized. If a cohort result exists without a
    boundary marker/frontier/agenda, the plan asks the orchestrator to complete
    that boundary before launching the next cohort.
    """

    if policy != "completed_generation":
        raise ValueError(f"unsupported resume policy: {policy}")

    warnings: list[str] = []
    run_dir = Path(run_dir)
    frontier_generations = _frontier_generations(run_dir)
    marker_contract_start = _boundary_marker_contract_start(run_dir)

    completed = 0
    pending_boundary: int | None = None
    for gen_id in range(max_generations):
        if not _generation_results_valid(run_dir, gen_id):
            break

        boundary_done, reason = _generation_boundary_done(
            run_dir,
            gen_id,
            max_generations=max_generations,
            pi_enabled=pi_enabled,
            frontier_generations=frontier_generations,
            explicit_marker_required=(
                marker_contract_start is not None and gen_id >= marker_contract_start
            ),
        )
        if boundary_done:
            completed = gen_id + 1
            continue

        pending_boundary = gen_id
        warnings.append(f"generation {gen_id} has cohort results but incomplete boundary: {reason}")
        break

    return ResumePlan(
        enabled=True,
        policy=policy,
        start_generation=completed,
        completed_generations=completed,
        pending_boundary_generation=pending_boundary,
        warnings=tuple(warnings),
    )


def repair_inferred_gems_boundary_markers(
    run_dir: Path,
    *,
    max_generations: int,
    pi_enabled: bool,
) -> list[dict[str, Any]]:
    """Backfill markers for committed boundaries inferred by legacy runs.

    A crash can happen after Gems reset commits but before
    ``generation_boundary.json`` is written. Resume can infer that boundary,
    while predecessor runs can also have a frontier+agenda prefix that predates
    boundary markers entirely. Relying on either inference forever becomes
    brittle after a resumed generation writes its first marker. This helper
    materializes each inferred committed boundary before launching more work.

    The historical function name remains part of the internal API because Gems
    recovery was its original purpose.
    """

    run_dir = Path(run_dir)
    frontier_generations = _frontier_generations(run_dir)
    marker_contract_start = _boundary_marker_contract_start(run_dir)
    repaired: list[dict[str, Any]] = []
    for gen_id in range(max_generations):
        if not _generation_results_valid(run_dir, gen_id):
            break
        marker = run_dir / f"gen_{gen_id}" / BOUNDARY_MARKER_FILENAME
        if marker.exists():
            with contextlib.suppress(OSError):
                clear_boundary_evidence_checkpoint(run_dir, gen_id)
            continue
        boundary_done, reason = _generation_boundary_done(
            run_dir,
            gen_id,
            max_generations=max_generations,
            pi_enabled=pi_enabled,
            frontier_generations=frontier_generations,
            explicit_marker_required=(
                marker_contract_start is not None and gen_id >= marker_contract_start
            ),
        )
        if not boundary_done:
            continue
        is_gems_reset = reason == "gems reset event exists"
        write_boundary_marker(
            run_dir,
            gen_id=gen_id,
            promoted_count=0,
            pi_status=(
                "skipped_gems_reset_repaired"
                if is_gems_reset
                else "legacy_inferred_boundary_repaired"
            ),
            error=f"boundary marker repaired from {reason}",
        )
        repaired.append(
            {
                "generation_id": gen_id,
                "reason": reason,
                "marker_path": str(marker),
            }
        )
    return repaired


def load_generation_results(run_dir: Path, gen_id: int) -> list[dict[str, Any]]:
    """Load a generation result list, returning an empty list on malformed data."""

    path = Path(run_dir) / f"gen_{gen_id}" / "generation_results.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def read_boundary_evidence_checkpoint(
    run_dir: Path,
    gen_id: int,
) -> tuple[datetime, dict[str, str]] | None:
    """Read the pending boundary cutoff stored in the existing close signal."""

    path = Path(run_dir) / f"gen_{int(gen_id)}" / BOUNDARY_CHECKPOINT_SIGNAL_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    raw_cutoff = fields.get(BOUNDARY_CHECKPOINT_CUTOFF_KEY, "")
    raw_snapshot = fields.get(BOUNDARY_CHECKPOINT_SNAPSHOT_KEY, "")
    if not raw_cutoff or not raw_snapshot:
        return None
    try:
        cutoff = datetime.fromisoformat(raw_cutoff.replace("Z", "+00:00"))
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)
        decoded = json.loads(raw_snapshot)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    snapshot = {
        str(source): str(identity)
        for source, identity in decoded.items()
        if str(source or "").strip() and str(identity or "").strip()
    }
    return cutoff, snapshot


def write_boundary_evidence_checkpoint(
    run_dir: Path,
    *,
    gen_id: int,
    cutoff: datetime,
    evidence_source_snapshot: dict[str, str],
) -> bool:
    """Persist an uncommitted cutoff without creating another artifact."""

    gen_dir = Path(run_dir) / f"gen_{int(gen_id)}"
    if not (gen_dir / "generation_results.json").exists():
        return False
    path = gen_dir / BOUNDARY_CHECKPOINT_SIGNAL_FILENAME
    existing_checkpoint = read_boundary_evidence_checkpoint(run_dir, gen_id)
    if existing_checkpoint == (cutoff, evidence_source_snapshot):
        return True
    snapshot_json = json.dumps(
        evidence_source_snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    _create_text_durable_if_absent(
        path,
        f"trigger_reason=boundary_finalization\ngen_id={int(gen_id)}\n",
    )
    payload = (
        "\n"
        + f"{BOUNDARY_CHECKPOINT_CUTOFF_KEY}={cutoff.isoformat()}\n"
        + f"{BOUNDARY_CHECKPOINT_SNAPSHOT_KEY}={snapshot_json}\n"
    )
    # Append instead of replacing CLOSING_SIGNAL. The trigger owns its close
    # fields and may publish them concurrently; an append cannot restore a
    # stale copy over that newer signal. Verify the active path and retry if a
    # one-time atomic signal replacement raced the open file descriptor.
    for _attempt in range(3):
        _append_text_durable(path, payload)
        if read_boundary_evidence_checkpoint(run_dir, gen_id) == (
            cutoff,
            evidence_source_snapshot,
        ):
            return True
    raise OSError(
        f"could not persist generation {int(gen_id)} boundary evidence cutoff after 3 attempts"
    )


def clear_boundary_evidence_checkpoint(run_dir: Path, gen_id: int) -> None:
    """Remove transient cutoff fields after the canonical marker commits."""

    path = Path(run_dir) / f"gen_{int(gen_id)}" / BOUNDARY_CHECKPOINT_SIGNAL_FILENAME
    try:
        existing = path.read_text(encoding="utf-8")
    except OSError:
        return
    retained = _without_boundary_checkpoint_fields(existing)
    if retained == existing:
        return
    _atomic_write_text(path, retained)


def _without_boundary_checkpoint_fields(payload: str) -> str:
    keys = {
        BOUNDARY_CHECKPOINT_CUTOFF_KEY,
        BOUNDARY_CHECKPOINT_SNAPSHOT_KEY,
    }
    retained = [line for line in payload.splitlines() if line.partition("=")[0].strip() not in keys]
    return "\n".join(retained).rstrip() + "\n"


def _create_text_durable_if_absent(path: Path, payload: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    _fsync_parent(path)
    return True


def _append_text_durable(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    _fsync_parent(path)


def _fsync_parent(path: Path) -> None:
    try:
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError:
        pass


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(OSError):
            os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
        try:
            parent_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError:
            pass
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def write_boundary_marker(
    run_dir: Path,
    *,
    gen_id: int,
    promoted_count: int,
    pi_status: str,
    agenda_path: str | None = None,
    error: str | None = None,
    stop_audit: dict[str, Any] | None = None,
    peer_mix: dict[str, Any] | None = None,
    evidence_cutoff_at: str | None = None,
    evidence_source_snapshot_at_cutoff: dict[str, str] | None = None,
) -> None:
    """Write a durable marker after post-generation boundary work finishes."""

    run_dir = Path(run_dir)
    checkpoint = read_boundary_evidence_checkpoint(run_dir, gen_id)
    if checkpoint is not None:
        checkpoint_cutoff, checkpoint_snapshot = checkpoint
        if not evidence_cutoff_at:
            evidence_cutoff_at = checkpoint_cutoff.isoformat()
        if evidence_source_snapshot_at_cutoff is None:
            evidence_source_snapshot_at_cutoff = checkpoint_snapshot
    marker = run_dir / f"gen_{gen_id}" / BOUNDARY_MARKER_FILENAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    canonical_sources = [
        f"gen_{gen_id}/generation_results.json",
        "frontier/frontier_manifest.json",
    ]
    if is_committed_runtime_fact_file(run_dir / "gems" / "gems_state.json"):
        canonical_sources.append("gems/gems_state.json")
    payload = {
        "schema_version": "praxist.generation_boundary.v1",
        "artifact_semantics": artifact_semantics(
            role=CANONICAL_STATE,
            status=COMMITTED,
            stage="generation_boundary",
            generation_id=gen_id,
            actor="research_loop:orchestrator",
            canonical_sources=canonical_sources,
            runtime_fact_source=True,
            notes=(
                "This marker is the canonical committed boundary for resume; "
                "derived PI/Gems/prompt snapshots must not override it."
            ),
        ),
        "generation_id": gen_id,
        "promoted_count": promoted_count,
        "pi_status": pi_status,
        "agenda_path": agenda_path,
        "error": error,
        "written_at": datetime.now(UTC).isoformat(),
    }
    if evidence_cutoff_at:
        payload["evidence_cutoff_at"] = str(evidence_cutoff_at)
    if evidence_source_snapshot_at_cutoff is not None:
        from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
            compact_boundary_source_snapshot,
        )

        evidence_source_snapshot_at_cutoff = compact_boundary_source_snapshot(
            evidence_source_snapshot_at_cutoff
        )
        payload["evidence_source_snapshot_at_cutoff"] = {
            str(path): str(identity)
            for path, identity in sorted(evidence_source_snapshot_at_cutoff.items())
            if str(path).strip() and str(identity).strip()
        }
    if isinstance(stop_audit, dict) and stop_audit:
        payload["stop_audit"] = stop_audit
    if isinstance(peer_mix, dict) and peer_mix:
        payload["peer_mix"] = peer_mix
    atomic_write_json(marker, payload)
    with contextlib.suppress(OSError):
        clear_boundary_evidence_checkpoint(run_dir, gen_id)


def append_resume_event(run_dir: Path, event: dict[str, Any]) -> None:
    """Append an operator-visible resume audit event."""

    path = Path(run_dir) / RESUME_EVENTS_FILENAME
    payload = {
        "schema_version": "praxist.resume_event.v1",
        "written_at": datetime.now(UTC).isoformat(),
        **event,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def validate_resume_startup_identity(
    run_dir: Path,
    startup_config: dict[str, Any],
    candidate_task_project_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ensure resume startup cannot silently mix a run with a new task/model identity."""

    run_dir = Path(run_dir)
    existing_startup = _read_json_object_for_resume(run_dir / "startup_config.json")
    existing_run = _read_json_object_for_resume(run_dir / "run.json")
    mismatches = _resume_canonical_mismatches(
        existing_startup,
        startup_config,
        existing_run,
        run_dir,
        candidate_task_project_manifest,
    )
    event_payload = {
        "event": "startup.resume_identity_check",
        "resume_policy": str((startup_config.get("resume") or {}).get("policy") or ""),
        "canonical_args": dict(startup_config.get("canonical_args") or {}),
        "mismatches": mismatches,
    }
    if mismatches:
        append_resume_event(run_dir, {"status": "rejected", **event_payload})
        fields = ", ".join(item["field"] for item in mismatches)
        raise ValueError(
            "resume startup identity does not match existing run artifacts: "
            f"{fields}. Start a fresh run directory or use the original task/model settings."
        )
    append_resume_event(run_dir, {"status": "accepted", **event_payload})
    return existing_run


def ensure_resumable_run_dir(run_dir: Path) -> None:
    """Validate that an existing directory looks like a Praxist run."""

    if not run_dir.exists():
        raise ValueError(f"resume run_dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"resume run_dir is not a directory: {run_dir}")
    missing = [rel for rel in ("run.json", "startup_config.json") if not (run_dir / rel).exists()]
    if missing:
        raise ValueError(
            f"resume run_dir is missing Praxist startup artifacts: {run_dir} ({', '.join(missing)}). "
            "Manual inspection is required to verify or repair run artifacts before resuming."
        )


def _read_json_object_for_resume(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"resume artifact is not valid JSON: {path}. "
            "Manual inspection is required to verify or repair run artifacts before resuming."
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"resume artifact must be a JSON object: {path}. "
            "Manual inspection is required to verify or repair run artifacts before resuming."
        )
    return value


def _read_json_object_optional(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _resume_compare_value(key: str, value: Any) -> str:
    raw = str(value or "").strip()
    if key in {"task_path", "run_dir"} and raw:
        try:
            return str(Path(raw).expanduser().resolve())
        except OSError:
            return str(Path(raw).expanduser())
    return raw


def _resume_canonical_mismatches(
    existing_startup_config: dict[str, Any],
    startup_config: dict[str, Any],
    existing_run_metadata: dict[str, Any],
    run_dir: Path,
    candidate_task_project_manifest: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    existing_args = existing_startup_config.get("canonical_args")
    candidate_args = startup_config.get("canonical_args")
    mismatches: list[dict[str, str]] = []
    if not isinstance(existing_args, dict) or not existing_args:
        mismatches.append(
            {
                "field": "canonical_args",
                "existing": "<missing>",
                "candidate": "<present>" if isinstance(candidate_args, dict) else "<missing>",
            }
        )
        return mismatches
    if not isinstance(candidate_args, dict) or not candidate_args:
        mismatches.append(
            {
                "field": "canonical_args",
                "existing": "<present>",
                "candidate": "<missing>",
            }
        )
        return mismatches

    for key in RESUME_CANONICAL_ARG_KEYS:
        existing = _resume_compare_value(key, existing_args.get(key))
        candidate = _resume_compare_value(key, candidate_args.get(key))
        if existing != candidate:
            mismatches.append({"field": key, "existing": existing, "candidate": candidate})

    existing_run_id = str(existing_run_metadata.get("run_id") or "").strip()
    if existing_run_id and existing_run_id != run_dir.name:
        mismatches.append(
            {"field": "run_id", "existing": existing_run_id, "candidate": run_dir.name}
        )
    existing_task_ref = str(existing_run_metadata.get("task_ref") or "").strip()
    candidate_task_ref = str(candidate_args.get("task") or "").strip()
    if existing_task_ref and candidate_task_ref and existing_task_ref != candidate_task_ref:
        mismatches.append(
            {
                "field": "run.task_ref",
                "existing": existing_task_ref,
                "candidate": candidate_task_ref,
            }
        )
    existing_identity = (
        existing_startup_config.get("resume_identity")
        if isinstance(existing_startup_config.get("resume_identity"), dict)
        else {}
    )
    candidate_identity = (
        startup_config.get("resume_identity")
        if isinstance(startup_config.get("resume_identity"), dict)
        else {}
    )
    existing_task_project = (
        existing_run_metadata.get("task_project")
        if isinstance(existing_run_metadata.get("task_project"), dict)
        else {}
    )
    existing_manifest_file = _read_json_object_optional(run_dir / "task_project_manifest.json")
    existing_manifest = str(
        existing_identity.get("task_project_manifest_sha256")
        or existing_task_project.get("manifest_sha256")
        or existing_manifest_file.get("sha256")
        or ""
    ).strip()
    candidate_manifest = str(candidate_identity.get("task_project_manifest_sha256") or "").strip()
    if candidate_manifest and not existing_manifest:
        mismatches.append(
            {
                "field": "task_project_manifest_sha256",
                "existing": "<missing>",
                "candidate": candidate_manifest,
            }
        )
    elif (
        existing_manifest
        and candidate_manifest
        and existing_manifest != candidate_manifest
        and not _legacy_manifest_diff_is_generated_reports_only(
            existing_manifest_file,
            candidate_task_project_manifest,
            existing_sha256=existing_manifest,
            candidate_sha256=candidate_manifest,
        )
    ):
        mismatches.append(
            {
                "field": "task_project_manifest_sha256",
                "existing": existing_manifest,
                "candidate": candidate_manifest,
            }
        )
    existing_descriptor = str(
        existing_identity.get("effective_task_descriptor_sha256") or ""
    ).strip()
    candidate_descriptor = str(
        candidate_identity.get("effective_task_descriptor_sha256") or ""
    ).strip()
    if candidate_descriptor and not existing_descriptor:
        mismatches.append(
            {
                "field": "effective_task_descriptor_sha256",
                "existing": "<missing>",
                "candidate": candidate_descriptor,
            }
        )
    elif (
        existing_descriptor and candidate_descriptor and existing_descriptor != candidate_descriptor
    ):
        mismatches.append(
            {
                "field": "effective_task_descriptor_sha256",
                "existing": existing_descriptor,
                "candidate": candidate_descriptor,
            }
        )
    existing_local = existing_identity.get("local_mode", existing_startup_config.get("local_mode"))
    candidate_local = candidate_identity.get("local_mode")
    if candidate_local not in (None, "") and existing_local in (None, ""):
        mismatches.append(
            {
                "field": "local_mode",
                "existing": "<missing>",
                "candidate": str(candidate_local).strip(),
            }
        )
    elif existing_local not in (None, "") and candidate_local not in (None, ""):
        existing = str(existing_local).strip()
        candidate = str(candidate_local).strip()
        if existing != candidate:
            mismatches.append({"field": "local_mode", "existing": existing, "candidate": candidate})
    return mismatches


def _legacy_manifest_diff_is_generated_reports_only(
    existing_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any] | None,
    *,
    existing_sha256: str,
    candidate_sha256: str,
) -> bool:
    """Accept only the manifest migration that removes generated run reports."""

    if not isinstance(candidate_manifest, dict):
        return False
    root_value = candidate_manifest.get("path")
    if not isinstance(root_value, str) or not root_value.strip():
        return False
    candidate_root = Path(root_value).expanduser().resolve()
    existing_root_value = existing_manifest.get("path")
    existing_root = (
        Path(existing_root_value).expanduser().resolve()
        if isinstance(existing_root_value, str) and existing_root_value.strip()
        else candidate_root
    )
    if not existing_root.is_dir():
        existing_root = candidate_root
    existing_rows = _verified_manifest_rows(
        existing_manifest,
        existing_sha256,
        existing_root,
    )
    candidate_rows = _verified_manifest_rows(
        candidate_manifest,
        candidate_sha256,
        candidate_root,
    )
    if existing_rows is None or candidate_rows is None:
        return False

    report_prefix = "docs/praxist_reports/"
    if not any(path.startswith(report_prefix) for path, _sha, _size in existing_rows):
        return False

    existing_identity_rows = [row for row in existing_rows if not row[0].startswith(report_prefix)]
    candidate_identity_rows = [
        row for row in candidate_rows if not row[0].startswith(report_prefix)
    ]
    return existing_identity_rows == candidate_identity_rows


def _verified_manifest_rows(
    manifest: dict[str, Any],
    expected_sha256: str,
    root: Path,
) -> list[tuple[str, str, int]] | None:
    """Verify one v1 manifest against current bytes before using it for migration."""

    if manifest.get("schema_version") != "task_project_manifest.v1":
        return None
    claimed_sha256 = str(manifest.get("sha256") or "").strip().lower()
    if claimed_sha256 != str(expected_sha256 or "").strip().lower():
        return None
    if len(claimed_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in claimed_sha256):
        return None
    files = manifest.get("files")
    if not isinstance(files, list):
        return None

    digest = hashlib.sha256()
    rows: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            return None
        relative = str(item.get("path") or "")
        relative_path = Path(relative)
        if (
            not relative
            or relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            return None
        seen.add(relative)
        unresolved_path = root / relative_path
        path = unresolved_path.resolve()
        if unresolved_path.is_symlink() or not path.is_relative_to(root) or not path.is_file():
            return None
        expected_size = item.get("bytes")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool):
            return None
        try:
            expected_file_sha = str(item.get("sha256") or "").strip().lower()
            content = path.read_bytes()
        except OSError:
            return None
        if expected_size < 0 or expected_size != len(content):
            return None
        actual_file_sha = hashlib.sha256(content).hexdigest()
        if expected_file_sha != actual_file_sha:
            return None
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        rows.append((relative, expected_file_sha, expected_size))
    if digest.hexdigest() != claimed_sha256:
        return None
    return rows


def pid_is_alive(pid: int) -> bool:
    """Return whether ``pid`` appears live on this host."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_pid(lock_text: str) -> int | None:
    """Parse the PID from an ``orchestrator.lock`` payload."""

    for line in lock_text.splitlines():
        if line.startswith("pid="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _generation_results_valid(run_dir: Path, gen_id: int) -> bool:
    results = load_generation_results(run_dir, gen_id)
    return bool(results)


def _frontier_generations(run_dir: Path) -> set[int]:
    manifest = Path(run_dir) / "frontier" / "frontier_manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not is_committed_runtime_fact_source(data, legacy_ok=True):
        return set()
    raw_generations = data.get("generations")
    if not isinstance(raw_generations, dict):
        return set()
    out: set[int] = set()
    for key in raw_generations:
        try:
            out.add(int(key))
        except (TypeError, ValueError):
            continue
    return out


def _generation_boundary_done(
    run_dir: Path,
    gen_id: int,
    *,
    max_generations: int,
    pi_enabled: bool,
    frontier_generations: set[int],
    explicit_marker_required: bool | None = None,
) -> tuple[bool, str]:
    marker = Path(run_dir) / f"gen_{gen_id}" / BOUNDARY_MARKER_FILENAME
    if _has_pending_gems_reset(run_dir, gen_id):
        return False, "pending Gems reset transaction exists"
    if marker.exists():
        return True, "boundary marker exists"
    if _gems_reset_boundary_done(run_dir, gen_id):
        return True, "gems reset event exists"
    if explicit_marker_required is None:
        marker_contract_start = _boundary_marker_contract_start(run_dir)
        explicit_marker_required = (
            marker_contract_start is not None and gen_id >= marker_contract_start
        )
    if explicit_marker_required:
        return False, "required generation boundary marker is missing"
    if gen_id not in frontier_generations:
        return False, "frontier manifest has no entry for generation"
    if pi_enabled and gen_id < max_generations - 1:
        agenda = Path(run_dir) / "agendas" / f"research_agenda_gen{gen_id + 1}.yaml"
        if not _agenda_committed_for_resume(agenda):
            return False, "frontier entry exists but next-generation agenda is missing"
    return True, "inferred from frontier manifest and agenda"


def _valid_boundary_marker(run_dir: Path, gen_id: int) -> bool:
    marker = Path(run_dir) / f"gen_{gen_id}" / BOUNDARY_MARKER_FILENAME
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    try:
        return int(payload.get("generation_id", gen_id)) == gen_id
    except (TypeError, ValueError):
        return False


def _boundary_marker_contract_start(run_dir: Path) -> int | None:
    """Return the first generation whose completion requires a marker.

    Praxist runs were introduced after generation boundary markers, so their
    startup/run schema is a stable capability signal even when Gen0 marker
    publication fails and therefore require markers from Gen0. Predecessor runs
    retain frontier+agenda inference before their first marker; that prefix is
    backfilled on resume before any new work starts.
    """

    run_dir = Path(run_dir)
    startup = _read_json_object_optional(run_dir / "startup_config.json")
    if startup.get("schema_version") == "praxist.startup.v1":
        return 0
    run_metadata = _read_json_object_optional(run_dir / "run.json")
    if run_metadata.get("schema_version") == "praxist.run.v1":
        return 0

    marker_generations: list[int] = []
    for marker in run_dir.glob(f"gen_*/{BOUNDARY_MARKER_FILENAME}"):
        try:
            gen_id = int(marker.parent.name.removeprefix("gen_"))
        except ValueError:
            continue
        if _valid_boundary_marker(run_dir, gen_id):
            marker_generations.append(gen_id)
    return min(marker_generations) if marker_generations else None


def _agenda_committed_for_resume(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import (
            _parse_agenda_file,
        )

        data = _parse_agenda_file(path)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    semantics = data.get("artifact_semantics")
    if not isinstance(semantics, dict):
        return True
    return str(semantics.get("status") or "").strip().lower() in {"", COMMITTED}


def _has_pending_gems_reset(run_dir: Path, gen_id: int) -> bool:
    state_path = Path(run_dir) / "gems" / "gems_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(state, dict):
        return False
    semantics = state.get("artifact_semantics")
    if isinstance(semantics, dict):
        role = str(semantics.get("role") or "").strip()
        status = str(semantics.get("status") or "").strip().lower()
        if role != CANONICAL_STATE:
            return False
        if status not in {COMMITTED, PARTIAL}:
            return False
    pending = state.get("pending_reset")
    if not isinstance(pending, dict):
        return False
    try:
        return int(pending.get("completed_gen_id", -1)) == int(gen_id)
    except (TypeError, ValueError):
        return False


def _gems_reset_boundary_done(run_dir: Path, gen_id: int) -> bool:
    """Return true when Gems reset committed but marker write was interrupted."""

    state_path = Path(run_dir) / "gems" / "gems_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(state, dict):
        return False
    if not is_committed_runtime_fact_source(state, legacy_ok=True):
        return False
    pending = state.get("pending_reset")
    if isinstance(pending, dict):
        try:
            if int(pending.get("completed_gen_id", -1)) == int(gen_id):
                return False
        except (TypeError, ValueError):
            return False
    events = state.get("reset_events")
    if not isinstance(events, list):
        return False
    matching_event: dict[str, Any] | None = None
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            completed = int(event.get("completed_gen_id", -1))
            next_gen = int(event.get("next_absolute_generation", -1))
        except (TypeError, ValueError):
            continue
        if completed == int(gen_id) and next_gen == int(gen_id) + 1:
            matching_event = event
            if event.get("committed") is True:
                try:
                    return int(event.get("reset_count", -1)) <= int(state.get("reset_count", -2))
                except (TypeError, ValueError):
                    return False
            break
    if matching_event is None:
        return False

    # Legacy compatibility for pre-transaction Gems reset records that did not
    # carry `committed=true`: only accept the currently-active reset if the
    # frontier manifest also proves the active frontier was cleared.
    if int(state.get("cycle_start_generation", -1) or -1) != int(gen_id) + 1:
        return False
    return _gems_frontier_manifest_committed(run_dir, state)


def _gems_frontier_manifest_committed(run_dir: Path, state: dict[str, Any]) -> bool:
    """Check that the frontier manifest reflects a completed Gems reset.

    `gems_state.json` and `frontier_manifest.json` are separate files. If a
    process dies after state write but before manifest rewrite, treating the
    boundary as complete would restart the next cohort with stale active
    frontier entries. Resume therefore only accepts a Gems reset as durable
    when both files agree and the active frontier was cleared.
    """

    manifest_path = Path(run_dir) / "frontier" / "frontier_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    if not is_committed_runtime_fact_source(manifest, legacy_ok=True):
        return False
    gems = manifest.get("gems")
    if not isinstance(gems, dict):
        return False
    if int(gems.get("reset_count", -1) or -1) != int(state.get("reset_count", -2) or -2):
        return False
    if int(gems.get("cycle_start_generation", -1) or -1) != int(
        state.get("cycle_start_generation", -2) or -2
    ):
        return False
    if manifest.get("generations") not in ({}, None):
        return False
    if manifest.get("lane_frontiers") not in ({}, None):
        return False
    return manifest.get("cumulative_top") in ([], None)
