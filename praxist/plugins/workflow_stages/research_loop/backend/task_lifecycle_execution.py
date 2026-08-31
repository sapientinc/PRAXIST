"""Task-owned research phases executed inside the generation-loop lifecycle."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from itertools import count
from typing import Any

from praxist.core.controller_state import private_controller_run_dir
from praxist.core.execution_policy import task_execution_deadline_scope
from praxist.core.protocol import AgentRunResult
from praxist.core.run_config import RunConfig
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
    load_generation_results,
)
from praxist.plugins.workflow_stages.research_loop.backend.run_lifecycle import (
    RunStopDecision,
    evaluate_run_stop_gate,
    max_generations_stop_report,
    write_run_stop_report,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import (
    is_scoreless,
    read_scoreless_evidence_manifest,
)
from praxist.plugins.workflow_stages.research_loop.backend.task_lifecycle import TaskLifecycle

logger = logging.getLogger(__name__)


class TaskDeliveryIncomplete(RuntimeError):
    """Carry a preserved task delivery that did not reach a committed result."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("task lifecycle delivery is incomplete")
        self.result = result


async def run_lifecycle_agent(
    loop: Any,
    prompt: str,
    *,
    role: str = "research",
    allowed_tools: list[str] | None = None,
    timeout_seconds: float | None = None,
) -> AgentRunResult:
    """Execute a trusted task callback's agent through the normal runtime."""
    from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

    config = RunConfig.from_environ(
        os.environ,
        overrides={
            "run_id": loop.run_dir.name,
            "run_dir": str(loop.run_dir),
            "stage_id": "research_loop",
            "role_ref": f"task_role:{role}",
            "agent_runtime_ref": loop.runtime_ref or "agent_runtime:claude_sdk",
        },
    )
    finalizing = loop._lifecycle_phase == "finalize"
    agent = BaseAgent(
        name=f"task_lifecycle_{role}",
        allowed_tools=list(loop._peer_allowed_tools) if allowed_tools is None else allowed_tools,
        workspace=loop.workspace,
        mcp_servers={} if finalizing else loop.mcp_servers,
        model=loop.model,
        plugin_registry=loop.plugin_registry,
        run_config=config,
        execution_role=role,
        runtime_timeout_seconds=(
            max(1, math.ceil(timeout_seconds)) if timeout_seconds is not None else None
        ),
        reasoning_effort=loop.task_spec.agent.reasoning_effort,
        runtime_sandbox_intent=(
            {"filesystem": "read_only" if role == "review" else "workspace_write", "network": "off"}
            if finalizing
            else None
        ),
    )
    return await agent.execute_normalized(prompt)


def check_task_stop(loop: Any, start_time: float, generations_completed: int) -> None:
    """Honor an operator stop before starting any additional task agents."""
    decision = evaluate_run_stop_gate(
        task_spec=loop.task_spec,
        run_dir=loop.run_dir,
        run_started_at_seconds=start_time,
        next_generation=generations_completed,
        generations_completed=generations_completed,
    )
    if decision.exit_condition == "external_stop_signal":
        raise TaskDeliveryIncomplete(
            {
                "status": "incomplete",
                "exit_condition": decision.exit_condition,
                "reason": decision.reason,
                "run_stop_report": write_run_stop_report(loop.run_dir, decision),
                "artifacts": [],
            }
        )


def frozen_findings_through(loop: Any, gen_id: int) -> list[dict[str, Any]]:
    """Read committed scoreless evidence instead of mutable peer sources."""
    findings: list[dict[str, Any]] = []
    for generation in range(gen_id + 1):
        manifest = read_scoreless_evidence_manifest(loop.run_dir, generation)
        if manifest is None:
            raise RuntimeError(f"committed evidence for generation {generation} is unavailable")
        findings.extend(manifest.get("findings") or [])
    return findings


async def run_task_generation_review(loop: Any, gen_id: int) -> dict[str, Any]:
    """Complete or recover a task review before successor research starts."""
    lifecycle = loop._task_lifecycle
    assert lifecycle is not None
    loop._lifecycle_phase = "review"
    try:
        with task_execution_deadline_scope(lifecycle.research_deadline_at):
            result = await lifecycle.run_phase(
                "review", loop._frozen_findings_through(gen_id), generation_id=gen_id
            )
    finally:
        loop._lifecycle_phase = ""
    if result.get("status") != "completed":
        logger.warning(
            "Task review for generation %d incomplete; preserving previous delivery", gen_id
        )
    return result


def initialize_task_lifecycle(
    loop: Any,
    start_time: float,
    configure_environment: Callable[..., Any],
    initialize_store: Callable[..., None],
    validate_baseline: Callable[..., None],
    run_logger: logging.Logger,
) -> float:
    """Prepare task execution and restore private lifecycle authority.

    Startup services are supplied by the orchestration facade so callers retain
    its established dependency-injection boundary and failure cleanup.
    """
    # UTC keeps elapsed-time math stable across timezone and DST changes.
    loop._run_started_at = datetime.now(UTC).isoformat()
    configure_environment(
        task_spec=loop.task_spec,
        run_dir=loop.run_dir,
        findings_dir=loop.findings_dir,
        local_mode=loop.local_mode,
    )
    initialize_store(local_mode=loop.local_mode)
    if not is_scoreless(loop.task_spec):
        validate_baseline(task_spec=loop.task_spec, workspace=loop.workspace, run_dir=loop.run_dir)
    state_dir = private_controller_run_dir(loop.run_dir, create=True)
    loop._task_lifecycle = TaskLifecycle(
        loop.task_spec,
        loop.task_project_path or loop.task_spec.task_dir,
        loop.run_dir,
        loop._run_lifecycle_agent,
        state_dir=state_dir,
    )
    if loop._task_lifecycle.started_at is not None:
        start_time = loop._task_lifecycle.started_at
        loop._run_started_at = datetime.fromtimestamp(start_time, UTC).isoformat()
    gp = loop.task_spec.generation_policy
    run_logger.info(
        f"\n{'#' * 60}\n"
        f"# Praxist — Generation Loop\n"
        f"# Task: {loop.task_spec.task_name}\n"
        f"# Generations: {gp.max_generations}\n"
        f"# Cohort size: {gp.cohort_size}\n"
        f"# Strategy: {loop.frontier_strategy}\n"
        f"# Mode: {'local' if loop.local_mode else 'server'}\n"
        f"{'#' * 60}"
    )
    return start_time


async def run_initial_task_phase(loop: Any, start_time: float) -> None:
    """Commit the initial task phase before any research cohort starts."""
    lifecycle = loop._task_lifecycle
    assert lifecycle is not None
    if not lifecycle.enabled:
        return
    loop._check_task_stop(start_time, loop._generations_completed)
    loop._lifecycle_phase = "initial"
    try:
        with task_execution_deadline_scope(lifecycle.deadline_at):
            initial = await lifecycle.run_phase("initial")
    finally:
        loop._lifecycle_phase = ""
    if initial.get("status") != "completed":
        raise TaskDeliveryIncomplete(initial)


async def review_completed_generations(
    loop: Any, start_time: float, completed: int, start_generation: int
) -> int:
    """Recover reviews and verify committed evidence before honoring completion."""
    lifecycle = loop._task_lifecycle
    assert lifecycle is not None
    if lifecycle.enabled and lifecycle.after_generation:
        for gen_id in range(completed):
            loop._check_task_stop(start_time, completed)
            result = await loop._run_task_generation_review(gen_id)
            if (
                result.get("status") == "completed"
                and result.get("summary", {}).get("research_complete") is True
            ):
                break
    if lifecycle.finalization_started:
        maximum = loop.task_spec.generation_policy.max_generations
        return start_generation if maximum is None else maximum
    return start_generation


def generation_indices(loop: Any, start_generation: int) -> Iterable[int]:
    """Yield research generations without inventing a limit for an uncapped task."""
    maximum = loop.task_spec.generation_policy.max_generations
    return count(start_generation) if maximum is None else range(start_generation, maximum)


def generation_limit_report(loop: Any, start_time: float, completed: int) -> dict[str, Any]:
    """Record exhaustion only when the task declares a generation limit."""
    maximum = loop.task_spec.generation_policy.max_generations
    if maximum is None:
        raise RuntimeError("uncapped generation iteration cannot exhaust its limit")
    return write_run_stop_report(
        loop.run_dir,
        max_generations_stop_report(
            run_dir=loop.run_dir,
            max_generations=maximum,
            generations_completed=completed,
            run_started_at_seconds=start_time,
        ),
    )


async def run_generation_with_deadline(loop: Any, gen_id: int) -> tuple[list[dict[str, Any]], bool]:
    """Run one cohort within its research deadline, preserving timed-out results."""
    lifecycle = loop._task_lifecycle
    assert lifecycle is not None
    remaining = lifecycle.remaining_seconds()
    with task_execution_deadline_scope(lifecycle.research_deadline_at):
        try:
            if remaining is None:
                return await loop._run_generation(gen_id), False
            return await asyncio.wait_for(loop._run_generation(gen_id), remaining), False
        except TimeoutError:
            if remaining is None:
                raise
            return load_generation_results(loop.run_dir, gen_id), True


def check_generation_start(
    loop: Any, start_time: float, gen_id: int, completed: int
) -> tuple[str, dict[str, Any] | None]:
    """Check run stop controls and reserved finalization time before a cohort."""
    decision = evaluate_run_stop_gate(
        task_spec=loop.task_spec,
        run_dir=loop.run_dir,
        run_started_at_seconds=start_time,
        next_generation=gen_id,
        generations_completed=completed,
    )
    if decision.should_stop:
        report = write_run_stop_report(loop.run_dir, decision)
        logger.info("Run lifecycle gate stopped before generation %d: %s", gen_id, decision.reason)
        return decision.exit_condition, report
    lifecycle = loop._task_lifecycle
    assert lifecycle is not None
    if getattr(lifecycle, "research_completed", False):
        decision = RunStopDecision(
            should_stop=True,
            exit_condition="research_complete",
            reason="research_complete",
            next_generation=gen_id,
            generations_completed=completed,
            elapsed_seconds=max(0.0, time.time() - start_time),
            run_dir=str(loop.run_dir),
            source="task_lifecycle",
        )
        return decision.exit_condition, write_run_stop_report(loop.run_dir, decision)
    if getattr(lifecycle, "finalization_started", False):
        return "finalization_resumed", None
    remaining = lifecycle.remaining_seconds()
    if remaining is not None and remaining <= 0:
        return "research_deadline", None
    return "", None


async def complete_boundary_with_deadline(
    loop: Any,
    gen_id: int,
    pi_agent: Any,
    pi_cfg: Any,
    generation_results: list[dict[str, Any]],
    complete_boundary: Callable[..., Awaitable[Any]],
) -> None:
    """Commit the normal boundary without launching PI work after the deadline."""
    lifecycle = loop._task_lifecycle
    assert lifecycle is not None
    remaining = lifecycle.remaining_seconds()
    with task_execution_deadline_scope(lifecycle.research_deadline_at):
        await complete_boundary(
            loop,
            gen_id=gen_id,
            pi_agent=None if remaining is not None and remaining <= 0 else pi_agent,
            pi_cfg=pi_cfg,
            generation_results=generation_results,
        )


async def review_generation_if_enabled(
    loop: Any, gen_id: int, start_time: float, completed: int
) -> None:
    """Apply the operator stop gate before an enabled task generation review."""
    lifecycle = loop._task_lifecycle
    assert lifecycle is not None
    if lifecycle.enabled and lifecycle.after_generation:
        loop._check_task_stop(start_time, completed)
        await loop._run_task_generation_review(gen_id)


async def run_final_task_phase(
    loop: Any, start_time: float, completed: int
) -> dict[str, Any] | None:
    """Commit final delivery from frozen generations before runtime teardown."""
    lifecycle = loop._task_lifecycle
    assert lifecycle is not None
    if not lifecycle.enabled:
        return None
    loop._check_task_stop(start_time, completed)
    frozen_findings = loop._frozen_findings_through(completed - 1)
    loop._lifecycle_phase = "finalize"
    try:
        with task_execution_deadline_scope(lifecycle.deadline_at):
            delivery = await lifecycle.run_phase("finalize", frozen_findings)
    finally:
        loop._lifecycle_phase = ""
    if delivery.get("status") != "completed":
        raise TaskDeliveryIncomplete(delivery)
    delivery["artifact_hashes"] = lifecycle.final_artifact_hashes()
    return delivery


def add_task_delivery_summary(
    loop: Any, summary: dict[str, Any], delivery: dict[str, Any] | None
) -> None:
    """Add truthful scoreless and task-delivery status to the run summary."""
    incomplete = delivery is not None and delivery.get("status") != "completed"
    summary["status"] = "incomplete" if incomplete else "succeeded"
    summary["exit_code"] = 1 if incomplete else 0
    if is_scoreless(loop.task_spec):
        summary["evaluation_status"] = "not_configured"
        summary["selection_status"] = "disabled"
    if delivery is not None:
        summary["task_delivery"] = delivery
    if loop._task_lifecycle and loop._task_lifecycle.deadline_at is not None:
        summary["deadline_at"] = loop._task_lifecycle.deadline_at
