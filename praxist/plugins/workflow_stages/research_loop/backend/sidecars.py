"""Lifecycle helpers for research-loop sidecar services."""

from __future__ import annotations

import logging
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.orchestrator_status import (
    OrchestratorStatusWriter,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import is_scoreless

logger = logging.getLogger(__name__)


def start_sidecars(loop: Any, *, resume_plan: Any = None) -> None:
    """Start optional local sidecars plus the orchestrator status writer."""
    loop._experiment_scheduler = None
    compute_budget = getattr(getattr(loop, "task_spec", None), "compute_budget", None)
    raw = getattr(compute_budget, "resource_scheduler", {})
    scheduler_requested = bool(
        isinstance(raw, dict)
        and "mode" in raw
        and str(raw.get("mode", "")).strip().lower() != "legacy"
    )
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler import (
            ExperimentSchedulerService,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.resource_scheduler import (
            SchedulerSettings,
        )

        settings = SchedulerSettings.from_dict(raw)
        if settings.enabled:
            loop._experiment_scheduler = ExperimentSchedulerService(
                run_dir=loop.run_dir,
                settings=settings,
                max_parallel_runs_per_peer=getattr(
                    compute_budget,
                    "max_parallel_runs_per_peer",
                    None,
                ),
                recovery_rerun_generation=(
                    resume_plan.start_generation
                    if resume_plan is not None and not resume_plan.has_pending_boundary
                    else None
                ),
            )
            loop._experiment_scheduler.start()
    except Exception as e:  # noqa: BLE001 - central mode must not silently fall back.
        if scheduler_requested:
            raise RuntimeError(f"Central experiment scheduler could not start: {e}") from e
        loop._experiment_scheduler = None
        logger.warning("Experiment scheduler configuration could not be loaded: %s", e)
    if loop.local_mode:
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.findings_collection import (
                result_artifact_options_from_task_spec,
            )
            from praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync import (
                FindingsSync,
            )

            # #150: forward the task's primary metric so the background
            # reverse-sync cycle (filesystem → SQLite) puts the canonical
            # primary-metric key into each row's ``metrics``. Without
            # this, the daemon ingests rows that frontier.promote drops
            # and ``variants_total`` stays at 0 throughout the run.
            primary_metric = getattr(
                getattr(getattr(loop, "task_spec", None), "evaluation", None),
                "primary_metric",
                None,
            )
            if is_scoreless(getattr(loop, "task_spec", None)):
                primary_metric = None
            result_options = result_artifact_options_from_task_spec(
                getattr(loop, "task_spec", None)
            )
            loop._findings_sync = FindingsSync(
                findings_dir=loop.findings_dir,
                run_dir=loop.run_dir,
                poll_interval=60,
                local_mode=True,
                primary_metric=primary_metric,
                **result_options,
            )
            active_boundary = getattr(loop, "_boundary_evidence_cutoff", None)
            if active_boundary is not None:
                loop._findings_sync.begin_boundary_evidence_cutoff(*active_boundary)
            try:
                loop._findings_sync.sync_once()
            except Exception as e:  # noqa: BLE001 - initial sync is best-effort.
                logger.debug("initial findings sync failed: %s", e)
            loop._findings_sync.start()
            logger.info("FindingsSync daemon started (event-driven local sync)")
        except Exception as e:  # noqa: BLE001 - losing the daemon should not stop peers.
            logger.warning("FindingsSync daemon could not start: %s", e)
            loop._findings_sync = None

        try:
            from praxist.plugins.graph_maintainers.finding_graph_mvp.engine import (
                FindingGraphMaintainer,
            )

            loop._graph_maintainer = FindingGraphMaintainer(
                run_dir=loop.run_dir,
                poll_interval=120,
            )
            try:
                loop._graph_maintainer.sync_once()
            except Exception as e:  # noqa: BLE001 - initial graph pass is advisory.
                logger.debug("initial graph maintainer cycle failed: %s", e)
            loop._graph_maintainer.start()
            logger.info("FindingGraphMaintainer daemon started (event-driven)")
        except Exception as e:  # noqa: BLE001 - graph guidance is advisory.
            logger.warning("FindingGraphMaintainer could not start (non-fatal): %s", e)
            loop._graph_maintainer = None

    loop._status_writer = OrchestratorStatusWriter(
        loop.run_dir,
        snapshot_fn=loop._build_status_snapshot,
    )
    loop._status_writer.start()


def stop_sidecars(loop: Any, *, exit_condition: str) -> None:
    """Flush and stop sidecars without masking the workflow exit condition."""
    scheduler = getattr(loop, "_experiment_scheduler", None)
    if scheduler is not None:
        for attempt in range(2):
            try:
                scheduler.stop()
                break
            except Exception as e:  # noqa: BLE001
                if attempt:
                    logger.debug("experiment scheduler stop failed after retry: %s", e)
    if loop._findings_sync is not None:
        try:
            loop._findings_sync.sync_once()
        except Exception as e:  # noqa: BLE001
            logger.debug("final findings sync failed: %s", e)
        try:
            loop._findings_sync.stop()
        except Exception as e:  # noqa: BLE001
            logger.debug("findings sync stop failed: %s", e)
    if loop._graph_maintainer is not None:
        try:
            loop._graph_maintainer.sync_once()
        except Exception as e:  # noqa: BLE001
            logger.debug("final graph maintainer cycle failed: %s", e)
        try:
            loop._graph_maintainer.stop()
        except Exception as e:  # noqa: BLE001
            logger.debug("graph maintainer stop failed: %s", e)
    if loop._status_writer is not None:
        try:
            loop._status_writer.stop(exit_condition=exit_condition)
        except Exception as e:  # noqa: BLE001
            logger.debug("status writer stop failed: %s", e)


def close_sidecars_and_runtime(loop: Any, exit_condition: str, runtime_scope: Any) -> None:
    """Stop sidecars and always release the orchestrator runtime scope."""
    try:
        stop_sidecars(loop, exit_condition=exit_condition)
    finally:
        runtime_scope.close()
