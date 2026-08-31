"""Authorize and record the research stage budget from trusted task configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from praxist.core.budget import policy_for_ref
from praxist.core.ledgers import BudgetLedger
from praxist.core.protocol import BudgetRequest
from praxist.core.trajectory import TrajectoryWriter
from praxist.plugins.workflow_stages.research_loop.stage import planned_research_loop_usage
from praxist.task_spec import TaskSpec


def grant_stage_budget(
    *,
    run_dir: Path,
    run_id: str,
    task_ref: str,
    task_spec: TaskSpec,
    budget_policy_ref: str,
    trajectory: TrajectoryWriter,
    registry: Any | None = None,
) -> str | None:
    """Authorize a stage budget using controller-selected task limits.

    Args:
        run_dir: Run artifacts directory.
        run_id: Run identifier.
        task_ref: Explicit task project reference.
        task_spec: Validated task configuration, not agent-supplied request metadata.
        budget_policy_ref: Selected budget policy plugin.
        trajectory: Writer for budget lifecycle events.
        registry: Resolved plugin registry when available.

    Returns:
        The approved grant identifier, or None when the policy declines approval.
    """
    requested = planned_research_loop_usage(task_spec)
    uncapped = any(amount is None for amount in requested.values())
    request = BudgetRequest(
        request_id="budget_request_research_loop_start",
        requester_id="workflow_stage:research_loop",
        experiment_id=f"{task_ref}/research_loop",
        model_profile_ref="",
        requested=requested,
        expected_value={
            "confidence": "strong",
            "value": "stage_execution",
            "requires_full_stage_budget": True,
            **({"usage_estimate_status": "unknown"} if uncapped else {}),
        },
        evidence_refs=[task_ref],
        cheaper_alternatives=[],
        abort_conditions=["stage_startup_failed"],
    )
    policy = policy_for_ref(budget_policy_ref, registry=registry)
    # Only this trusted task-startup path supplies the out-of-band permission;
    # fields inside a BudgetRequest never confer uncapped authorization.
    decision = policy.decide(request, allow_uncapped=True) if uncapped else policy.decide(request)
    ledger = BudgetLedger(run_dir, run_id)
    if decision.grant and decision.grant.grant_id in ledger.active_grants():
        return decision.grant.grant_id
    ledger.append_request(
        request,
        actor_ref="workflow_stage:research_loop",
        stage_id="research_loop",
        action_type="stage_start",
        reason="legacy_research_loop_stage_budget_request",
    )
    ledger.append_decision(
        request,
        decision,
        actor_ref=budget_policy_ref,
        stage_id="research_loop",
        action_type="stage_start",
        reason="legacy_research_loop_stage_budget_decision",
    )
    trajectory.emit(
        "budget.requested",
        scope={"stage_id": "research_loop"},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={"request_id": request.request_id, "requested": request.requested},
    )
    trajectory.emit(
        "budget.granted" if decision.grant else "budget.review_required",
        scope={
            "stage_id": "research_loop",
            "grant_id": decision.grant.grant_id if decision.grant else "",
        },
        actor={"type": "budget_policy", "id": budget_policy_ref},
        payload={"request_id": request.request_id, "decision": decision.to_dict()},
    )
    return decision.grant.grant_id if decision.grant else None
