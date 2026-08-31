"""Validate startup identity before task-owned runtime preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from praxist.core.task_project import TaskProject
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    ensure_resumable_run_dir,
    validate_resume_startup_identity,
)


def ensure_fresh_run_dir(run_dir: Path, *, resume: bool = False) -> None:
    """Reject occupied new-run destinations or invalid resume directories.

    Args:
        run_dir: Candidate run destination.
        resume: Whether existing resumable artifacts are expected.

    Raises:
        ValueError: The directory conflicts with the requested startup mode.
    """
    if resume:
        ensure_resumable_run_dir(run_dir)
        return
    if not run_dir.exists():
        return
    for rel in (
        "run.json",
        "trajectory.jsonl",
        "budget_ledger.jsonl",
        "artifact_index.jsonl",
        "run_summary.json",
        "plugin_resolution.json",
        "startup_config.json",
    ):
        if (run_dir / rel).exists():
            raise ValueError(
                f"run_dir already contains Praxist run artifacts: {run_dir}. "
                "Resume mode is not implemented; choose a fresh run directory."
            )
    blocking_paths = [path for path in run_dir.iterdir() if not _is_ignorable_precreated_path(path)]
    if blocking_paths:
        raise ValueError(
            f"run_dir already exists and is not empty: {run_dir}. "
            "Resume mode is not implemented; choose a fresh run directory."
        )


def _is_ignorable_precreated_path(path: Path) -> bool:
    if path.is_file():
        return path.name in {".DS_Store", ".gitkeep"}
    if path.is_dir():
        if path.name == "logs":
            return all(child.name in {".gitkeep", "launcher.nohup.log"} for child in path.iterdir())
        return not any(path.iterdir())
    return False


def validate_private_resume_selection(
    authority: dict[str, Any],
    *,
    project: TaskProject,
    runtime_ref: str,
    model_provider_ref: str,
    budget_policy_ref: str,
    local_mode: bool,
) -> None:
    """Reject changed execution authority before loading selected plugins.

    Args:
        authority: Validated private startup checkpoint.
        project: Explicitly selected task project.
        runtime_ref: Selected agent runtime.
        model_provider_ref: Selected model provider.
        budget_policy_ref: Selected budget policy.
        local_mode: Whether execution uses the local mode.

    Raises:
        ValueError: Selection differs from the private startup authority.
    """
    args = authority["canonical_args"]
    selections = {
        "task": project.task_ref,
        "task_path": str(project.path.resolve()),
        "runtime": runtime_ref,
        "model_provider": model_provider_ref,
        "budget_policy": budget_policy_ref,
    }
    mismatches = [key for key, value in selections.items() if args.get(key) != value]
    identity = authority.get("resume_identity")
    if not isinstance(identity, dict):
        raise ValueError("private startup authority is missing task identity")
    if identity.get("task_project_manifest_sha256") != project.manifest["sha256"]:
        mismatches.append("task_project_manifest_sha256")
    if identity.get("local_mode") is not local_mode:
        mismatches.append("local_mode")
    if mismatches:
        raise ValueError(
            "private startup authority does not match resume selection: " + ", ".join(mismatches)
        )


def prepare_startup_identity(
    *,
    project: TaskProject,
    task_ref: str,
    run_dir: Path,
    runtime_ref: str,
    model_provider_ref: str,
    budget_policy_ref: str,
    model: str,
    frontier_strategy: str,
    local_mode: bool,
    effective_task_descriptor: dict[str, Any],
    resume: bool,
    resume_policy: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Capture effective identity and reject changed resumes before preparation.

    Args:
        project: Explicit task project with its source manifest.
        task_ref: Selected task reference.
        run_dir: Canonical run directory.
        runtime_ref: Selected agent runtime.
        model_provider_ref: Selected model provider.
        budget_policy_ref: Selected budget policy.
        model: Provider-normalized model name.
        frontier_strategy: Selected frontier strategy.
        local_mode: Whether execution uses the local mode.
        effective_task_descriptor: Task configuration after environment overrides.
        resume: Whether startup resumes existing research.
        resume_policy: Requested generation recovery policy.

    Returns:
        Canonical arguments, resume identity, and previous run metadata.

    Raises:
        ValueError: A resume differs from the authoritative existing identity.
    """
    canonical_args = {
        "task": task_ref,
        "task_path": str(project.path),
        "runtime": runtime_ref,
        "model_provider": model_provider_ref,
        "budget_policy": budget_policy_ref,
        "model": model,
        "frontier_strategy": frontier_strategy,
        "run_dir": str(run_dir),
    }
    identity = {
        "task_project_manifest_sha256": project.manifest["sha256"],
        "effective_task_descriptor_sha256": hashlib.sha256(
            json.dumps(
                effective_task_descriptor, sort_keys=True, default=str, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "local_mode": bool(local_mode),
    }
    previous = (
        validate_resume_startup_identity(
            run_dir,
            {
                "canonical_args": canonical_args,
                "resume_identity": identity,
                "resume": {"policy": resume_policy},
            },
            candidate_task_project_manifest=project.manifest,
        )
        if resume
        else {}
    )
    return canonical_args, identity, previous
