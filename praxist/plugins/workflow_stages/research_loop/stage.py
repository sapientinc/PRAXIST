"""Executable research_loop workflow stage plugin."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from praxist.core.budget import INFORMATIONAL_USAGE_UNITS
from praxist.core.ledgers import BudgetLedger
from praxist.core.role_skills import RoleSkill, load_role_skill
from praxist.core.runtimes import close_runtime_for_ref, collect_runtime_usage
from praxist.core.workflow import WorkflowStageResult, WorkflowStageSpec
from praxist.plugins.workflow_stages.research_loop.lifecycle import (
    ResearchRunLifecycleObserver,
)
from praxist.plugins.workflow_stages.research_loop.provider_env import (
    DEEPSEEK_CLAUDE_DEFAULT_EFFORT,
    DEEPSEEK_CLAUDE_DEFAULT_HAIKU_MODEL,
    DEEPSEEK_CLAUDE_DEFAULT_MODEL,
    DEEPSEEK_CLAUDE_SDK_BASE_URL,
    OPENROUTER_CLAUDE_SDK_BASE_URL,
    normalize_openrouter_base_url,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchLoopStageContext:
    """Execution context passed from core workflow dispatch into the research_loop stage."""

    task_spec: Any
    workspace: Path
    run_dir: Path
    local_mode: bool
    model: str
    model_provider_ref: str
    frontier_strategy: str
    budget_grant_id: str | None
    runtime_ref: str = "agent_runtime:claude_sdk"
    model_provider_credential_key_id: str | None = None
    provider_env: dict[str, str | None] | None = None
    tool_server_refs: tuple[str, ...] = ()
    plugin_registry: Any | None = None
    resolve_only: bool = False
    # Issue #75 batch 3: passed through to GenerationLoop → PIAgent →
    # run_panel → BasePI so ``task_role:*`` refs resolve without an
    # PRAXIST_TASK_PROJECT_PATH env read.
    task_project_path: Path | None = None
    resume: bool = False
    resume_policy: str = "completed_generation"
    run_lifecycle_observer: ResearchRunLifecycleObserver | None = None
    peer_role_ref: str | None = None
    peer_role_refs: tuple[str, ...] = ()


class ResearchLoopStage:
    """WorkflowStage wrapper around the plugin-local GenerationLoop backend."""

    ref = "workflow_stage:research_loop"
    stage_id = "research_loop"

    def describe(self, *, budget_policy_ref: str | None = None) -> WorkflowStageSpec:
        return WorkflowStageSpec(
            stage_id=self.stage_id,
            ref=self.ref,
            enabled_by_default=True,
            inputs=["task_spec", "plugin_resolution", "budget_grant"],
            outputs=["findings", "frontier", "research_memory", "budget_ledger"],
            budget_policy_ref=budget_policy_ref,
            retry_policy={"max_attempts": 1, "retry_on": []},
        )

    async def execute(self, context: ResearchLoopStageContext) -> WorkflowStageResult:
        if not context.budget_grant_id:
            return WorkflowStageResult(
                stage_id=self.stage_id,
                status="failed",
                success=False,
                error="research_loop stage requires an approved budget grant",
            )
        ledger = BudgetLedger(context.run_dir, context.run_dir.name)
        try:
            active_grant = ledger.require_active_grant(context.budget_grant_id)
        except ValueError as exc:
            return WorkflowStageResult(
                stage_id=self.stage_id,
                status="failed",
                success=False,
                error=str(exc),
            )

        if context.resolve_only:
            ledger.append_usage(
                request_id=active_grant.get("request_id"),
                grant_id=context.budget_grant_id,
                actor_ref=self.ref,
                stage_id=self.stage_id,
                action_type="resolve_only",
                actual_usage=_usage_for_grant(
                    active_grant,
                    wall_clock_seconds=0.0,
                    allow_zero_for_unmeasured=True,
                ),
                reason="research_loop_resolve_only_usage",
            )
            return WorkflowStageResult(
                stage_id=self.stage_id,
                status="succeeded",
                success=True,
                summary={
                    "generations_completed": 0,
                    "run_dir": str(context.run_dir),
                    "exit_condition": "resolve_only",
                },
            )

        shortfalls = _budget_shortfalls(
            planned=_planned_usage_for_task_spec(context.task_spec),
            approved=active_grant.get("granted_budget") or {},
        )
        if shortfalls:
            return WorkflowStageResult(
                stage_id=self.stage_id,
                status="failed",
                success=False,
                error="approved budget is below planned research_loop usage: "
                + ", ".join(shortfalls),
            )

        from praxist.plugins.workflow_stages.research_loop.backend.generation_loop import (
            GenerationLoop,
        )

        peer_role_skills: list[RoleSkill] = []
        role_refs = context.peer_role_refs or (
            (context.peer_role_ref,) if context.peer_role_ref else ()
        )
        for role_ref in role_refs:
            try:
                peer_role_skills.append(
                    load_role_skill(
                        role_ref,
                        registry=context.plugin_registry,
                        workspace=context.workspace,
                        task_project_path=context.task_project_path,
                    )
                )
            except Exception as exc:
                if role_ref.startswith("task_role:"):
                    return WorkflowStageResult(
                        stage_id=self.stage_id,
                        status="failed",
                        success=False,
                        error=f"cannot load selected peer role {role_ref}: {exc}",
                    )
                logger.debug("peer role %s has no loadable RoleSkill", role_ref)

        peer_role_skill = next(
            (skill for skill in peer_role_skills if skill.role_ref == context.peer_role_ref),
            peer_role_skills[0] if peer_role_skills else None,
        )

        loop = GenerationLoop(
            task_spec=context.task_spec,
            workspace=context.workspace,
            run_dir=context.run_dir,
            local_mode=context.local_mode,
            model=context.model,
            frontier_strategy=context.frontier_strategy,
            tool_server_refs=context.tool_server_refs,
            plugin_registry=context.plugin_registry,
            task_project_path=context.task_project_path,
            peer_role_ref=context.peer_role_ref,
            peer_role_skill=peer_role_skill,
            peer_role_skills=tuple(peer_role_skills),
            # Thread the runtime ref through for runtime-neutral request
            # construction and trajectory attribution.
            runtime_ref=context.runtime_ref,
            resume=context.resume,
            resume_policy=context.resume_policy,
            run_lifecycle_observer=context.run_lifecycle_observer,
        )
        started = time.monotonic()
        env_updates = dict(context.provider_env or _provider_env(context.model_provider_ref))
        env_updates.update(
            {
                "PRAXIST_RUN_DIR": str(context.run_dir),
                "PRAXIST_RUN_ID": context.run_dir.name,
                "PRAXIST_STAGE_ID": self.stage_id,
                "PRAXIST_AGENT_RUNTIME_REF": context.runtime_ref,
                "PRAXIST_BUDGET_GRANT_ID": context.budget_grant_id,
                "PRAXIST_BUDGET_REQUEST_ID": str(active_grant.get("request_id") or ""),
                "PRAXIST_MODEL_CREDENTIAL_KEY_ID": context.model_provider_credential_key_id,
                "PRAXIST_PROTECTED_CHILD_PATHS": None,
            }
        )
        previous_env = {key: os.environ.get(key) for key in env_updates}
        result: dict[str, Any] | None = None
        missing_usage_units: list[str] = []
        run_failed = False
        with collect_runtime_usage() as usage_collector:
            try:
                for key, value in env_updates.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                result = await loop.run()
            except BaseException:
                run_failed = True
                raise
            finally:
                elapsed = max(0.0, time.monotonic() - started)
                try:
                    await close_runtime_for_ref(context.runtime_ref, context.plugin_registry)
                except Exception as exc:  # noqa: BLE001 - teardown must not mask run outcome.
                    logger.warning(
                        "agent runtime teardown failed for %s: %s", context.runtime_ref, exc
                    )
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                measured_result = _with_runtime_usage(result, usage_collector.snapshot())
                if result is not None:
                    result = measured_result
                executed = run_failed or _stage_result_indicates_execution(measured_result)
                actual_usage = _usage_for_grant(
                    active_grant,
                    wall_clock_seconds=elapsed,
                    result=measured_result,
                    allow_zero_for_unmeasured=not executed,
                )
                if executed:
                    missing_usage_units = _missing_approved_usage_units(active_grant, actual_usage)
                ledger.append_usage(
                    request_id=active_grant.get("request_id"),
                    grant_id=context.budget_grant_id,
                    actor_ref=self.ref,
                    stage_id=self.stage_id,
                    action_type="stage_execution",
                    actual_usage=actual_usage,
                    reason=(
                        "research_loop_stage_execution_usage_partial"
                        if missing_usage_units
                        else "research_loop_stage_execution_usage"
                    ),
                )
                if missing_usage_units:
                    ledger.append_usage_unknown(
                        request_id=active_grant.get("request_id"),
                        grant_id=context.budget_grant_id,
                        actor_ref=self.ref,
                        stage_id=self.stage_id,
                        action_type="stage_execution",
                        unknown_units=missing_usage_units,
                        reason="research_loop_stage_execution_usage_unknown",
                    )
        if missing_usage_units:
            summary = dict(result or {})
            raw_warnings = summary.get("warnings")
            warnings = list(raw_warnings) if isinstance(raw_warnings, list) else []
            warnings.append(
                "budget usage measurement unavailable for approved units: "
                + ", ".join(missing_usage_units)
            )
            summary["warnings"] = warnings
            summary["usage_unknown_units"] = missing_usage_units
            return WorkflowStageResult(
                stage_id=self.stage_id,
                status="succeeded",
                success=True,
                summary=summary,
            )
        return WorkflowStageResult(
            stage_id=self.stage_id,
            status="succeeded",
            success=True,
            summary=result,
        )


def create_stage() -> ResearchLoopStage:
    """Manifest entrypoint that constructs the research_loop workflow stage."""
    return ResearchLoopStage()


def run_research_loop_stage(context: ResearchLoopStageContext) -> WorkflowStageResult:
    """Compatibility entrypoint for running research_loop without a prebuilt stage object."""
    return asyncio.run(ResearchLoopStage().execute(context))


def _usage_for_grant(
    grant_record: dict[str, Any],
    *,
    wall_clock_seconds: float,
    result: dict[str, Any] | None = None,
    allow_zero_for_unmeasured: bool = False,
) -> dict[str, float]:

    approved = grant_record.get("granted_budget") or {}
    usage: dict[str, float] = {}

    # Collect approved units
    if isinstance(approved, dict):
        for unit in approved:
            unit_name = str(unit)
            measured = _measured_usage_for_unit(unit_name, result)
            if measured is not None:
                usage[unit_name] = measured
            elif allow_zero_for_unmeasured:
                usage[unit_name] = 0.0

    # Keep provider-reported billing units even when the grant did not set a cost ceiling.
    if isinstance(result, dict) and isinstance(result.get("runtime_usage"), dict):
        for unit_name, value in result["runtime_usage"].items():
            if unit_name in INFORMATIONAL_USAGE_UNITS and unit_name not in usage:
                usage[unit_name] = float(value)

    if (
        "wall_clock_seconds" in usage
        or (isinstance(approved, dict) and "wall_clock_seconds" in approved)
        or not usage
    ):
        usage["wall_clock_seconds"] = wall_clock_seconds
    return usage


def _with_runtime_usage(
    result: dict[str, Any] | None,
    runtime_usage: dict[str, float],
) -> dict[str, Any] | None:
    if not runtime_usage:
        return result
    measured_result = dict(result or {})
    measured_result["runtime_usage"] = dict(runtime_usage)
    total_tokens = runtime_usage.get("total_tokens")
    if total_tokens is not None:
        measured_result["total_tokens"] = total_tokens
    return measured_result


def _measured_usage_for_unit(
    unit: str,
    result: dict[str, Any] | None,
) -> float | None:
    if unit == "tokens":
        measured = _first_positive_number(
            result,
            ("tokens", "total_tokens", "tokens_used", "llm_tokens", "total_llm_tokens"),
        )
        if measured is not None:
            return measured
    if unit == "gpu_hours":
        measured = _first_positive_number(
            result, ("gpu_hours", "total_gpu_hours", "gpu_hours_used")
        )
        if measured is not None:
            return measured
    return None


def _missing_approved_usage_units(
    grant_record: dict[str, Any], actual_usage: dict[str, float]
) -> list[str]:
    approved = grant_record.get("granted_budget") or {}
    if not isinstance(approved, dict):
        return ["approved budget payload is invalid"]
    missing = []
    for unit, raw_approved in approved.items():
        try:
            approved_amount = float(raw_approved)
        except (TypeError, ValueError):
            continue
        unit_name = str(unit)
        if approved_amount > 0 and unit_name not in actual_usage:
            missing.append(unit_name)
    return missing


def _stage_result_indicates_execution(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    for key in (
        "generations_completed",
        "frontier_records",
        "last_gen_findings_count",
        "last_gen_promoted_count",
    ):
        try:
            if float(result.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    frontier_summary = result.get("frontier_summary")
    if isinstance(frontier_summary, list) and frontier_summary:
        return True
    usage = result.get("usage")
    return isinstance(usage, dict) and bool(usage)


def planned_research_loop_usage(task_spec: Any) -> dict[str, float]:
    """Return the canonical planned usage for grant and stage validation."""

    max_generations = max(1, int(task_spec.generation_policy.max_generations))
    cohort_size = max(1, int(task_spec.generation_policy.cohort_size))
    per_generation_hours = max(
        1.0 / 60.0,
        float(task_spec.generation_policy.per_generation_hours),
    )
    per_experiment_gpu_hours = max(
        0.0,
        float(task_spec.compute_budget.per_experiment_gpu_hours or 0.0),
    )
    return {
        "tokens": float(max(50_000, max_generations * cohort_size * 250_000)),
        "wall_clock_seconds": float(max(60, max_generations * per_generation_hours * 3600)),
        "gpu_hours": float(max_generations * cohort_size * per_experiment_gpu_hours),
    }


def _planned_usage_for_task_spec(task_spec: Any) -> dict[str, float]:
    try:
        return planned_research_loop_usage(task_spec)
    except (AttributeError, TypeError, ValueError):
        return {}


def _budget_shortfalls(*, planned: dict[str, float], approved: dict[str, Any]) -> list[str]:
    shortfalls: list[str] = []
    if not isinstance(approved, dict):
        return ["approved budget payload is invalid"]
    for unit, planned_amount in planned.items():
        try:
            approved_amount = float(approved.get(unit, 0.0))
        except (TypeError, ValueError):
            shortfalls.append(f"{unit} approved is non-numeric")
            continue
        if approved_amount < float(planned_amount):
            shortfalls.append(f"{unit} {approved_amount} < {float(planned_amount)}")
    return shortfalls


def _first_positive_number(payload: dict[str, Any] | None, keys: tuple[str, ...]) -> float | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get("usage")
    candidates = [payload]
    runtime_usage = payload.get("runtime_usage")
    if isinstance(runtime_usage, dict):
        candidates.append(runtime_usage)
    if isinstance(nested, dict):
        candidates.append(nested)
    for candidate in candidates:
        for key in keys:
            if key not in candidate:
                continue
            try:
                value = float(candidate[key])
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
    return None


def _provider_env(model_provider_ref: str) -> dict[str, str | None]:
    base = {
        "PRAXIST_MODEL_PROVIDER_REF": model_provider_ref,
        "ANTHROPIC_API_KEY": None,
        "ANTHROPIC_BASE_URL": None,
        "ANTHROPIC_AUTH_TOKEN": None,
        "OPENROUTER_API_KEY": None,
        "OPENAI_API_KEY": None,
        "DEEPSEEK_API_KEY": None,
        "ANTHROPIC_MODEL": None,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": None,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": None,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": None,
        "CLAUDE_CODE_SUBAGENT_MODEL": None,
        "CLAUDE_CODE_EFFORT_LEVEL": None,
    }
    if model_provider_ref == "model_provider:openrouter":
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("OPENROUTER_API_KEY")
        base_url = normalize_openrouter_base_url(
            os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or OPENROUTER_CLAUDE_SDK_BASE_URL
        )
        updates: dict[str, str | None] = {
            **base,
            "ANTHROPIC_BASE_URL": base_url,
        }
        if auth_token:
            updates["ANTHROPIC_AUTH_TOKEN"] = auth_token
            updates["OPENROUTER_API_KEY"] = auth_token
        else:
            updates["ANTHROPIC_AUTH_TOKEN"] = None
            updates["OPENROUTER_API_KEY"] = None
        return updates
    if model_provider_ref == "model_provider:anthropic_messages":
        return {
            **base,
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        }
    if model_provider_ref == "model_provider:openai_compatible":
        return {**base, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")}
    if model_provider_ref == "model_provider:deepseek_alias":
        auth_token = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        return {
            **base,
            "ANTHROPIC_BASE_URL": os.environ.get("DEEPSEEK_ANTHROPIC_BASE_URL")
            or DEEPSEEK_CLAUDE_SDK_BASE_URL,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY") or auth_token,
            "ANTHROPIC_MODEL": os.environ.get("ANTHROPIC_MODEL") or DEEPSEEK_CLAUDE_DEFAULT_MODEL,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": os.environ.get("ANTHROPIC_DEFAULT_OPUS_MODEL")
            or DEEPSEEK_CLAUDE_DEFAULT_MODEL,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
            or DEEPSEEK_CLAUDE_DEFAULT_MODEL,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL")
            or DEEPSEEK_CLAUDE_DEFAULT_HAIKU_MODEL,
            "CLAUDE_CODE_SUBAGENT_MODEL": os.environ.get("CLAUDE_CODE_SUBAGENT_MODEL")
            or DEEPSEEK_CLAUDE_DEFAULT_HAIKU_MODEL,
            "CLAUDE_CODE_EFFORT_LEVEL": os.environ.get("CLAUDE_CODE_EFFORT_LEVEL")
            or DEEPSEEK_CLAUDE_DEFAULT_EFFORT,
        }
    if model_provider_ref == "model_provider:fake_provider":
        return base
    raise ValueError(f"Unsupported model provider env contract: {model_provider_ref}")
