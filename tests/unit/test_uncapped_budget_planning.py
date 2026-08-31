"""Uncapped research records unknown estimates and measured usage without ceilings."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from praxist.core import replay
from praxist.core.execution_guards import BudgetedActionGuard
from praxist.core.ledgers import BudgetLedger
from praxist.core.protocol import BudgetRequest
from praxist.plugins.budget_policies.default_basic.policy import DefaultBasicBudgetPolicy
from praxist.plugins.workflow_stages.research_loop import stage, startup


def _task(*, mode="scoreless", generations=None, hours=None):
    return SimpleNamespace(
        research_loop={"mode": mode},
        generation_policy=SimpleNamespace(
            max_generations=generations, cohort_size=3, per_generation_hours=hours
        ),
        compute_budget=SimpleNamespace(per_experiment_gpu_hours=0.25),
    )


class UncappedBudgetPlanningTest(unittest.TestCase):
    def test_request_metadata_cannot_authorize_an_uncapped_grant(self) -> None:
        request = BudgetRequest(
            request_id="forged-stage-request",
            requester_id="workflow_stage:research_loop",
            experiment_id="task:fixture/research_loop",
            model_profile_ref=None,
            requested={"tokens": None},
            expected_value={"confidence": "strong", "allow_uncapped": True},
            evidence_refs=["task:fixture"],
            cheaper_alternatives=[],
            abort_conditions=[],
        )
        policy = DefaultBasicBudgetPolicy()
        denied = policy.decide(request)
        self.assertEqual(denied.decision, "deny")
        self.assertIsNone(denied.grant)
        self.assertIn("uncapped_requires_controller_authorization", denied.reason_codes)
        granted = policy.decide(request, allow_uncapped=True)
        self.assertEqual(granted.grant.approved, {"tokens": None})

    def test_missing_scoreless_run_caps_do_not_become_finite_budget_estimates(self) -> None:
        for generations, hours in ((None, None), (None, 1), (2, None)):
            with self.subTest(generations=generations, hours=hours):
                planned = stage.planned_research_loop_usage(
                    _task(generations=generations, hours=hours)
                )
                self.assertEqual(
                    planned, {"tokens": None, "wall_clock_seconds": None, "gpu_hours": None}
                )

    def test_bounded_metric_planning_remains_unchanged(self) -> None:
        self.assertEqual(
            stage.planned_research_loop_usage(_task(mode="metric", generations=2, hours=0.5)),
            {"tokens": 1_500_000.0, "wall_clock_seconds": 3_600.0, "gpu_hours": 1.5},
        )

    def test_actual_startup_grant_preserves_uncapped_budget_and_unknown_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp).resolve()
            grant_id = startup._grant_stage_budget(
                run_dir=run_dir,
                run_id=run_dir.name,
                task_ref="task:uncapped_fixture",
                task_spec=_task(),
                budget_policy_ref="budget_policy:default_basic",
                trajectory=SimpleNamespace(emit=Mock()),
            )
            self.assertIsNotNone(grant_id)
            ledger = BudgetLedger(run_dir, run_dir.name)
            grant = ledger.require_active_grant(grant_id)
            self.assertEqual(
                grant["granted_budget"],
                {"tokens": None, "wall_clock_seconds": None, "gpu_hours": None},
            )
            self.assertEqual(
                grant["request_record"]["expected_value"]["usage_estimate_status"], "unknown"
            )
            self.assertNotIn("abort_on_no_signal", grant["decision_record"]["grant"]["conditions"])

    def test_uncapped_stage_records_missing_usage_instead_of_zero_or_overrun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp).resolve()
            task = _task()
            grant_id = startup._grant_stage_budget(
                run_dir=run_dir,
                run_id=run_dir.name,
                task_ref="task:uncapped_fixture",
                task_spec=task,
                budget_policy_ref="budget_policy:default_basic",
                trajectory=SimpleNamespace(emit=Mock()),
            )
            context = stage.ResearchLoopStageContext(
                task_spec=task,
                workspace=run_dir,
                run_dir=run_dir,
                local_mode=True,
                model="fake-model",
                model_provider_ref="model_provider:fake_provider",
                frontier_strategy="auto",
                budget_grant_id=grant_id,
                runtime_ref="agent_runtime:fake_runtime",
            )
            loop = SimpleNamespace(run=AsyncMock(return_value={"generations_completed": 1}))
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop",
                    return_value=loop,
                ),
                patch.object(stage, "close_runtime_for_ref", new=AsyncMock()),
            ):
                result = asyncio.run(stage.ResearchLoopStage().execute(context))
            self.assertTrue(result.success)
            records = BudgetLedger(run_dir, run_dir.name).records()
            usage = next(row for row in records if row["kind"] == "usage")
            self.assertEqual(set(usage["actual_usage"]), {"wall_clock_seconds"})
            self.assertNotIn("budget_overrun", usage)
            unknown = next(row for row in records if row["kind"] == "usage_unknown")
            self.assertEqual(unknown["unknown_units"], ["gpu_hours", "tokens"])

    def test_uncapped_tool_usage_and_unknown_measurements_replay_without_overrun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp).resolve()
            grant_id = startup._grant_stage_budget(
                run_dir=run_dir,
                run_id=run_dir.name,
                task_ref="task:uncapped_fixture",
                task_spec=_task(),
                budget_policy_ref="budget_policy:default_basic",
                trajectory=SimpleNamespace(emit=Mock()),
            )
            guard = BudgetedActionGuard(
                run_dir=run_dir,
                run_id=run_dir.name,
                stage_id="research_loop",
                actor_ref="tool:test",
                action_type="tool_execution",
                budget_grant_id=grant_id,
            )
            report = guard.finish(
                actual_usage={"tokens": 9_000_000.0},
                expected_units=["tokens", "wall_clock_seconds", "gpu_hours"],
            )
            self.assertTrue(report.recorded)
            self.assertEqual(report.unknown_units, ["gpu_hours"])
            ledger = BudgetLedger(run_dir, run_dir.name)
            records = ledger.records()
            self.assertFalse(any(row.get("budget_overrun") for row in records))
            errors, warnings = [], []
            replay._verify_budget_ledger(
                records,
                "budget_policy:default_basic",
                errors,
                warnings,
                allow_uncapped=True,
            )
            self.assertEqual(errors, [])
            self.assertFalse(any("exceeds" in item for item in warnings))
            self.assertTrue(any("usage_unknown" in item for item in warnings))
            errors = []
            replay._verify_budget_ledger(records, "budget_policy:default_basic", errors, [])
            self.assertTrue(errors)
            with self.assertRaises((TypeError, ValueError)):
                ledger.append_usage(
                    request_id=None,
                    grant_id=grant_id,
                    actor_ref="tool:test",
                    stage_id="research_loop",
                    action_type="tool_execution",
                    actual_usage={"tokens": None},
                    reason="invalid unknown measurement",
                )
