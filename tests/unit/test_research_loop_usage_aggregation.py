from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.core import runtimes as runtime_core
from praxist.core.runtimes import (
    AgentRuntimeExecutionContext,
    collect_runtime_usage,
    execute_runtime,
)
from praxist.plugins.workflow_stages.research_loop import stage
from praxist.plugins.workflow_stages.research_loop.backend import agent


def _result(
    *,
    usage: dict[str, float],
    cost: dict[str, float] | None = None,
    success: bool = True,
    error: str | None = None,
) -> runtime_core.AgentRunResult:
    return runtime_core.AgentRunResult(
        success=success,
        events=[],
        text_output_refs=[],
        tool_uses=[],
        error=error,
        failover_reason=None,
        credential_ref=None,
        usage=usage,
        cost=dict(cost or {}),
    )


class _Ledger:
    instances: list[_Ledger] = []

    def __init__(self, _run_dir: Path, _run_id: str) -> None:
        self.usage_records: list[dict[str, object]] = []
        self.unknown_records: list[dict[str, object]] = []
        self.__class__.instances.append(self)

    def require_active_grant(self, grant_id: str) -> dict[str, object]:
        return {
            "grant_id": grant_id,
            "request_id": "request-1",
            "granted_budget": {"tokens": 1_000.0, "wall_clock_seconds": 60.0},
        }

    def append_usage(self, **kwargs: object) -> dict[str, object]:
        self.usage_records.append(kwargs)
        return {}

    def append_usage_unknown(self, **kwargs: object) -> dict[str, object]:
        self.unknown_records.append(kwargs)
        return {}


def _context(root: Path) -> stage.ResearchLoopStageContext:
    run_dir = root / "run"
    run_dir.mkdir()
    return stage.ResearchLoopStageContext(
        task_spec=object(),
        workspace=root,
        run_dir=run_dir,
        local_mode=True,
        model="gpt-test",
        model_provider_ref="model_provider:openai_compatible",
        frontier_strategy="mixed",
        budget_grant_id="grant-1",
        runtime_ref="agent_runtime:fake",
    )


class ResearchLoopUsageAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        _Ledger.instances.clear()

    def test_stage_aggregates_every_base_agent_role_into_summary_and_ledger(self) -> None:
        per_call_usage = [
            {"input_tokens": 10.0, "output_tokens": 2.0},
            {"input_tokens": 5.0, "output_tokens": 5.0, "total_tokens": 10.0},
            {"total_tokens": 3.0},
            {"input_tokens": 7.0, "output_tokens": 1.0},
        ]
        per_call_cost = [0.01, 0.02, 0.03, 0.04]
        observed_agent_usage: list[dict[str, float] | None] = []

        class FakeRuntime:
            async def execute(
                self, _request: object, _context: object
            ) -> runtime_core.AgentRunResult:
                index = len(observed_agent_usage)
                return _result(
                    usage=dict(per_call_usage[index]),
                    cost={"cost_usd": per_call_cost[index]},
                )

        class FakeGenerationLoop:
            def __init__(self, **kwargs: object) -> None:
                self.workspace = Path(str(kwargs["workspace"]))

            async def run(self) -> dict[str, object]:
                for role in ("peer", "pi", "dig", "chair"):
                    role_result = await agent.BaseAgent(
                        name=role,
                        allowed_tools=[],
                        workspace=self.workspace,
                        mcp_servers={},
                        model="gpt-test",
                    ).execute(role)
                    observed_agent_usage.append(role_result.usage)
                return {"generations_completed": 1, "exit_condition": "completed"}

        with tempfile.TemporaryDirectory() as tmp:
            context = _context(Path(tmp))
            with (
                patch.object(stage, "BudgetLedger", _Ledger),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop",
                    FakeGenerationLoop,
                ),
                patch.object(agent, "runtime_for_ref", return_value=FakeRuntime()),
                patch.object(agent, "_legacy_trajectory_writer", return_value=None),
            ):
                stage_result = asyncio.run(stage.ResearchLoopStage().execute(context))

        self.assertTrue(stage_result.success)
        self.assertEqual(observed_agent_usage, per_call_usage)
        self.assertEqual(
            stage_result.summary["runtime_usage"],
            {
                "input_tokens": 22.0,
                "output_tokens": 8.0,
                "total_tokens": 33.0,
                "cost_usd": 0.1,
            },
        )
        self.assertEqual(stage_result.summary["total_tokens"], 33.0)
        actual_usage = _Ledger.instances[0].usage_records[0]["actual_usage"]
        self.assertEqual(actual_usage["tokens"], 33.0)
        self.assertEqual(actual_usage["cost_usd"], 0.1)
        self.assertNotIn("input_tokens", actual_usage)
        self.assertNotIn("output_tokens", actual_usage)
        self.assertNotIn("total_tokens", actual_usage)
        self.assertIn("wall_clock_seconds", actual_usage)
        self.assertEqual(_Ledger.instances[0].unknown_records, [])

    def test_stage_threads_selected_peer_role_skill_into_generation_loop(self) -> None:
        captured: dict[str, object] = {}

        class FakeGenerationLoop:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            async def run(self) -> dict[str, object]:
                return {"generations_completed": 0, "exit_condition": "completed"}

        role_skills = [
            SimpleNamespace(role_ref="task_role:starter", content_hash="starter-hash"),
            SimpleNamespace(role_ref="task_role:analyst", content_hash="analyst-hash"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            context = stage.ResearchLoopStageContext(
                **{
                    **_context(Path(tmp)).__dict__,
                    "peer_role_ref": "task_role:starter",
                    "peer_role_refs": ("task_role:starter", "task_role:analyst"),
                    "task_project_path": Path(tmp),
                }
            )
            with (
                patch.object(stage, "BudgetLedger", _Ledger),
                patch.object(stage, "load_role_skill", side_effect=role_skills) as load,
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop",
                    FakeGenerationLoop,
                ),
            ):
                result = asyncio.run(stage.ResearchLoopStage().execute(context))

        self.assertTrue(result.success)
        self.assertEqual(load.call_count, 2)
        self.assertEqual(captured["peer_role_ref"], "task_role:starter")
        self.assertIs(captured["peer_role_skill"], role_skills[0])
        self.assertEqual(captured["peer_role_skills"], tuple(role_skills))

    def test_failed_agent_result_usage_is_recorded_in_stage_ledger(self) -> None:
        class FailedRuntime:
            async def execute(
                self, _request: object, _context: object
            ) -> runtime_core.AgentRunResult:
                return _result(
                    usage={"input_tokens": 4.0, "output_tokens": 1.0},
                    success=False,
                    error="provider failed",
                )

        class FailingGenerationLoop:
            def __init__(self, **kwargs: object) -> None:
                self.workspace = Path(str(kwargs["workspace"]))

            async def run(self) -> dict[str, object]:
                result = await agent.BaseAgent(
                    name="chair",
                    allowed_tools=[],
                    workspace=self.workspace,
                    mcp_servers={},
                    model="gpt-test",
                ).execute("synthesize")
                if not result.success:
                    raise RuntimeError(result.error)
                return {}

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(stage, "BudgetLedger", _Ledger),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop",
                FailingGenerationLoop,
            ),
            patch.object(agent, "runtime_for_ref", return_value=FailedRuntime()),
            patch.object(agent, "_legacy_trajectory_writer", return_value=None),
            self.assertRaisesRegex(RuntimeError, "provider failed"),
        ):
            asyncio.run(stage.ResearchLoopStage().execute(_context(Path(tmp))))

        actual_usage = _Ledger.instances[0].usage_records[0]["actual_usage"]
        self.assertEqual(actual_usage["tokens"], 5.0)
        self.assertIn("wall_clock_seconds", actual_usage)
        self.assertEqual(_Ledger.instances[0].unknown_records, [])

    def test_runtime_usage_collectors_are_context_local(self) -> None:
        class Runtime:
            def __init__(self, tokens: float) -> None:
                self.tokens = tokens

            async def execute(
                self, _request: object, _context: object
            ) -> runtime_core.AgentRunResult:
                await asyncio.sleep(0)
                return _result(usage={"total_tokens": self.tokens})

        async def worker(tokens: float) -> dict[str, float]:
            with collect_runtime_usage() as collector:
                await execute_runtime(
                    Runtime(tokens),
                    object(),
                    AgentRuntimeExecutionContext(),
                )
                await asyncio.sleep(0)
                return collector.snapshot()

        async def run_workers() -> list[dict[str, float]]:
            return list(await asyncio.gather(worker(2.0), worker(7.0)))

        self.assertEqual(
            asyncio.run(run_workers()),
            [{"total_tokens": 2.0}, {"total_tokens": 7.0}],
        )


if __name__ == "__main__":
    unittest.main()
