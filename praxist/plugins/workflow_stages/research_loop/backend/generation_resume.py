"""Resume-time helpers for generation-boundary recovery."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.cohort_runner import (
    clear_generation_runtime_signals,
)
from praxist.plugins.workflow_stages.research_loop.backend.generation_boundary import (
    _activate_boundary_evidence_cutoff,
    _clear_boundary_evidence_cutoff,
    _hold_findings_sync_for_gems,
    _sync_findings_locked_once,
    _sync_graph_before_next_generation,
    _write_boundary_marker_if_possible,
)
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    BOUNDARY_MARKER_FILENAME,
    ResumePlan,
    append_resume_event,
    clear_boundary_evidence_checkpoint,
    inspect_resume_plan,
    load_generation_results,
    read_boundary_evidence_checkpoint,
    repair_inferred_gems_boundary_markers,
)

logger = logging.getLogger(__name__)


def prepare_resume_for_sidecars(
    run_dir: Path,
    *,
    max_generations: int | None,
    pi_enabled: bool,
    policy: str,
) -> ResumePlan:
    """Plan resume and clear controls only for a cohort that will rerun."""

    plan = inspect_resume_plan(
        run_dir,
        max_generations=max_generations,
        pi_enabled=pi_enabled,
        policy=policy,
    )
    if not plan.has_pending_boundary and (
        max_generations is None or plan.start_generation < max_generations
    ):
        gen_dir = Path(run_dir) / f"gen_{plan.start_generation}"
        if gen_dir.exists():
            clear_generation_runtime_signals(gen_dir)
    return plan


def _cohort_results_are_complete(loop: Any, gen_id: int) -> bool:
    return bool(load_generation_results(loop.run_dir, gen_id))


def _discard_abandoned_boundary_checkpoint(loop: Any, gen_id: int) -> None:
    if bool(getattr(loop, "local_mode", False)):
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
                clear_pending_boundary_validation,
            )

            clear_pending_boundary_validation(gen_id)
        except Exception as exc:  # noqa: BLE001 - checkpoint removal must still proceed.
            logger.warning(
                "resume: could not clear provisional boundary state for generation %d: %s",
                gen_id,
                exc,
            )
    clear_boundary_evidence_checkpoint(loop.run_dir, gen_id)
    _clear_boundary_evidence_cutoff(loop, gen_id=gen_id)


def _checkpoint_generation_ids(run_dir: Path, max_generations: int | None) -> range | list[int]:
    if max_generations is not None:
        return range(max(0, int(max_generations)))
    # Sparse checkpoints can outlive an incomplete prefix. Scan actual directories
    # in numeric order so the latest generation's cutoff remains active.
    generations = []
    for directory in Path(run_dir).glob("gen_*"):
        suffix = directory.name.removeprefix("gen_")
        if not suffix.isascii() or not suffix.isdecimal() or not directory.is_dir():
            continue
        generation = int(suffix)
        if directory.name == f"gen_{generation}":
            generations.append(generation)
    return sorted(generations)


def prime_resume_boundary_evidence_cutoff(loop: Any, *, max_generations: int | None) -> None:
    """Restore the newest uncommitted cutoff before the first sidecar sync."""

    for gen_id in _checkpoint_generation_ids(loop.run_dir, max_generations):
        marker = loop.run_dir / f"gen_{gen_id}" / BOUNDARY_MARKER_FILENAME
        checkpoint = (
            None if marker.exists() else read_boundary_evidence_checkpoint(loop.run_dir, gen_id)
        )
        if checkpoint is not None:
            if not _cohort_results_are_complete(loop, gen_id):
                _discard_abandoned_boundary_checkpoint(loop, gen_id)
                continue
            cutoff, source_snapshot = checkpoint
            _activate_boundary_evidence_cutoff(
                loop,
                gen_id=gen_id,
                cutoff=cutoff,
                evidence_source_snapshot=source_snapshot,
            )


def repair_inferred_boundaries_for_resume(
    loop: Any,
    *,
    max_generations: int | None,
    pi_enabled: bool,
) -> list[dict[str, Any]]:
    """Reclassify checkpointed evidence before inferred markers commit."""

    checkpointed: list[int] = []
    for gen_id in _checkpoint_generation_ids(loop.run_dir, max_generations):
        marker = loop.run_dir / f"gen_{gen_id}" / BOUNDARY_MARKER_FILENAME
        checkpoint = (
            None if marker.exists() else read_boundary_evidence_checkpoint(loop.run_dir, gen_id)
        )
        if checkpoint is None:
            continue
        if not _cohort_results_are_complete(loop, gen_id):
            _discard_abandoned_boundary_checkpoint(loop, gen_id)
            continue
        cutoff, source_snapshot = checkpoint
        _activate_boundary_evidence_cutoff(
            loop,
            gen_id=gen_id,
            cutoff=cutoff,
            evidence_source_snapshot=source_snapshot,
        )
        boundary_collector = getattr(loop, "_collect_findings_for_boundary", None)
        if callable(boundary_collector):
            boundary_collector(
                gen_id,
                evidence_cutoff=cutoff,
                evidence_source_snapshot=source_snapshot,
            )
        else:
            sync_once = getattr(getattr(loop, "_findings_sync", None), "sync_once", None)
            if callable(sync_once):
                sync_once()
        checkpointed.append(gen_id)

    repairs = repair_inferred_gems_boundary_markers(
        loop.run_dir,
        max_generations=max_generations,
        pi_enabled=pi_enabled,
    )
    repaired_ids = {
        int(repair["generation_id"])
        for repair in repairs
        if repair.get("generation_id") is not None
    }
    for gen_id in checkpointed:
        if gen_id in repaired_ids:
            _clear_boundary_evidence_cutoff(loop, gen_id=gen_id)
    return repairs


def recover_pending_gems_reset_for_resume(loop: Any, *, pending_gen: int) -> Any:
    """Complete a pending Gems reset under the normal boundary locks."""

    checkpoint = read_boundary_evidence_checkpoint(loop.run_dir, pending_gen)
    if checkpoint is not None:
        cutoff, source_snapshot = checkpoint
        _activate_boundary_evidence_cutoff(
            loop,
            gen_id=pending_gen,
            cutoff=cutoff,
            evidence_source_snapshot=source_snapshot,
        )
    with _hold_findings_sync_for_gems(loop) as findings_sync:
        _sync_findings_locked_once(findings_sync, reason="before pending Gems reset recovery")
        recovered_gems = loop.gems.recover_pending_reset(completed_gen_id=pending_gen)
        if recovered_gems.triggered:
            _sync_findings_locked_once(findings_sync, reason="after pending Gems reset recovery")
    if recovered_gems.triggered:
        _sync_graph_before_next_generation(loop, gen_id=pending_gen)
        _write_boundary_marker_if_possible(
            loop,
            gen_id=pending_gen,
            promoted_count=0,
            pi_status="skipped_gems_reset_recovered",
            error=(
                f"gems_reset_count={recovered_gems.reset_count}; "
                f"admitted={recovered_gems.admitted_count}; "
                f"archive={recovered_gems.archive_dir}"
            ),
        )
        append_resume_event(
            loop.run_dir,
            {
                "event": "pending_gems_reset_recovered",
                "generation_id": pending_gen,
                "reset_count": recovered_gems.reset_count,
                "admitted_count": recovered_gems.admitted_count,
            },
        )
    return recovered_gems
