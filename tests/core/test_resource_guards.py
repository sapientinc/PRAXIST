from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.execution_guards import (
    BudgetedActionGuard,
    ResourceBudgetError,
    emit_resource_event_from_run_config,
    record_budgeted_action_from_env,
    record_budgeted_action_from_run_config,
)
from praxist.core.ledgers import BudgetLedger
from praxist.core.protocol import BudgetDecision, BudgetGrant, BudgetRequest
from praxist.core.run_config import RunConfig
from praxist.core.tool_servers import execute_legacy_tool_handler
from praxist.plugins.tools.evaluation_tools.adapter import _handle_wait_for_file
from praxist.plugins.workflow_stages.research_loop.backend import gpu_governor
from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
    register_pid,
    unregister_pid,
)
from praxist.plugins.workflow_stages.research_loop.backend.tools.training_timeout import (
    TimeoutPolicy,
    monitor_subprocess_with_grace,
)


class Step16ResourceGuardMigrationTest(unittest.TestCase):
    def test_budgeted_action_records_usage_and_unknown_units_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_guard"
            run_dir.mkdir()
            grant_id, request_id = _write_budget_grant(
                run_dir,
                {"tokens": 1000.0, "wall_clock_seconds": 60.0, "gpu_hours": 0.5},
            )

            missing = BudgetedActionGuard(
                run_dir=run_dir,
                run_id=run_dir.name,
                stage_id="research_loop",
                actor_ref="resource_guard:test",
                action_type="eval_runner",
                require_budget_grant=True,
            )
            with self.assertRaises(ResourceBudgetError):
                missing.start()

            guard = BudgetedActionGuard(
                run_dir=run_dir,
                run_id=run_dir.name,
                stage_id="research_loop",
                actor_ref="resource_guard:test",
                action_type="eval_runner",
                budget_grant_id=grant_id,
                request_id=request_id,
                require_budget_grant=True,
            )
            guard.start()
            report = guard.finish(
                actual_usage={"tokens": 12.0},
                expected_units=("tokens", "wall_clock_seconds", "gpu_hours"),
                reason="test_eval_runner_usage",
            )

            self.assertTrue(report.recorded)
            self.assertEqual(report.unknown_units, ["gpu_hours"])
            records = BudgetLedger(run_dir, run_dir.name).records()
            usage = [record for record in records if record["kind"] == "usage"][-1]
            self.assertEqual(usage["actual_usage"]["tokens"], 12.0)
            self.assertIn("wall_clock_seconds", usage["actual_usage"])
            unknown = [record for record in records if record["kind"] == "usage_unknown"][-1]
            self.assertEqual(unknown["unknown_units"], ["gpu_hours"])

    def test_budgeted_action_allows_informational_cost_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_cost"
            run_dir.mkdir()
            grant_id, request_id = _write_budget_grant(
                run_dir,
                {"tokens": 1000.0},
            )

            guard = BudgetedActionGuard(
                run_dir=run_dir,
                run_id=run_dir.name,
                stage_id="research_loop",
                actor_ref="resource_guard:test",
                action_type="eval_runner",
                budget_grant_id=grant_id,
                request_id=request_id,
                require_budget_grant=True,
            )
            guard.start()
            report = guard.finish(
                actual_usage={"tokens": 500.0, "cost_usd": 0.05, "neurons": 200.0},
                expected_units=("tokens", "cost_usd", "neurons"),
                reason="test_cost_usage",
            )

            self.assertTrue(report.recorded)
            records = BudgetLedger(run_dir, run_dir.name).records()
            usage = [record for record in records if record["kind"] == "usage"][-1]
            self.assertEqual(usage["actual_usage"]["tokens"], 500.0)
            self.assertEqual(usage["actual_usage"]["cost_usd"], 0.05)
            self.assertEqual(usage["actual_usage"]["neurons"], 200.0)

    def test_wait_for_file_records_tool_wall_clock_usage_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_wait"
            run_dir.mkdir()
            target = run_dir / "result.json"
            target.write_text('{"status": "ok"}', encoding="utf-8")
            grant_id, request_id = _write_budget_grant(run_dir, {"wall_clock_seconds": 60.0})

            with patch.dict(
                os.environ,
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_RUN_ID": run_dir.name,
                    "PRAXIST_STAGE_ID": "research_loop",
                    "PRAXIST_BUDGET_GRANT_ID": grant_id,
                    "PRAXIST_BUDGET_REQUEST_ID": request_id,
                    "LOCAL_STORE_DIR": str(run_dir),
                },
                clear=False,
            ):
                result = asyncio.run(
                    _handle_wait_for_file(
                        {
                            "path": str(target),
                            "timeout_seconds": 2,
                            "poll_interval_seconds": 2,
                            "min_bytes": 1,
                        }
                    )
                )

            payload = json.loads(result["content"][0]["text"])
            self.assertEqual(payload["status"], "ready")
            usage = [
                record
                for record in BudgetLedger(run_dir, run_dir.name).records()
                if record["kind"] == "usage"
            ]
            self.assertEqual(usage[-1]["action_type"], "tool.wait_for_file")
            self.assertIn("wall_clock_seconds", usage[-1]["actual_usage"])

    def test_tool_handler_adapter_can_record_action_usage_with_explicit_budget_context(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_tool"
            run_dir.mkdir()
            grant_id, request_id = _write_budget_grant(run_dir, {"wall_clock_seconds": 60.0})

            with patch.dict(os.environ, {"FRONTIER_DIR": ""}, clear=False):
                result = execute_legacy_tool_handler(
                    "tool_server:frontier_tools",
                    "get_frontier",
                    {"top_k": 1},
                    run_dir=run_dir,
                    run_id=run_dir.name,
                    budget_grant_id=grant_id,
                    budget_request_id=request_id,
                )

            self.assertTrue(result.success)
            usage = [
                record
                for record in BudgetLedger(run_dir, run_dir.name).records()
                if record["kind"] == "usage"
            ]
            self.assertEqual(usage[-1]["action_type"], "tool.get_frontier")
            self.assertIn("wall_clock_seconds", usage[-1]["actual_usage"])

    def test_gpu_governor_release_records_gpu_hours_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_gpu"
            run_dir.mkdir()
            grant_id, request_id = _write_budget_grant(run_dir, {"gpu_hours": 1.0})
            env = {
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_RUN_ID": run_dir.name,
                "PRAXIST_STAGE_ID": "research_loop",
                "PRAXIST_BUDGET_GRANT_ID": grant_id,
                "PRAXIST_BUDGET_REQUEST_ID": request_id,
                "GPU_GOVERNOR_DIR": "",
            }
            with patch.dict(os.environ, env, clear=False):
                os.environ.pop("BYPASS_GPU_GOVERNOR", None)
                self.assertTrue(
                    gpu_governor.acquire_slot(
                        0,
                        pid=os.getpid(),
                        peer="gen0_peer0",
                        tag="unit-test",
                        run_dir=run_dir,
                        blocking=False,
                        max_per_gpu=1,
                    )
                )
                time.sleep(0.01)
                self.assertTrue(gpu_governor.release_slot(0, pid=os.getpid(), run_dir=run_dir))

            usage = [
                record
                for record in BudgetLedger(run_dir, run_dir.name).records()
                if record["kind"] == "usage"
            ]
            self.assertEqual(usage[-1]["action_type"], "gpu_slot")
            self.assertGreaterEqual(usage[-1]["actual_usage"]["gpu_hours"], 0.0)
            events = _trajectory_kinds(run_dir)
            self.assertIn("resource.gpu_slot_acquired", events)
            self.assertIn("resource.gpu_slot_released", events)

    def test_eval_runner_records_wall_clock_usage_when_budget_context_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_eval_runner"
            run_dir.mkdir()
            grant_id, request_id = _write_budget_grant(run_dir, {"wall_clock_seconds": 60.0})

            with patch.dict(
                os.environ,
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_RUN_ID": run_dir.name,
                    "PRAXIST_STAGE_ID": "research_loop",
                    "PRAXIST_BUDGET_GRANT_ID": grant_id,
                    "PRAXIST_BUDGET_REQUEST_ID": request_id,
                },
                clear=False,
            ):
                record_budgeted_action_from_env(
                    action_type="eval_runner",
                    actor_ref="evaluation_runner:test_task",
                    actual_usage={"wall_clock_seconds": 1.25},
                    expected_units=("wall_clock_seconds",),
                    status="succeeded",
                    reason="eval_runner_wall_clock_usage",
                    metadata={"tier": "T1"},
                )

            usage = [
                record
                for record in BudgetLedger(run_dir, run_dir.name).records()
                if record["kind"] == "usage"
            ]
            self.assertEqual(usage[-1]["action_type"], "eval_runner")
            self.assertEqual(usage[-1]["actual_usage"]["wall_clock_seconds"], 1.25)

    def test_protected_pid_and_training_timeout_emit_resource_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_survival"
            run_dir.mkdir()
            env = {
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_RUN_ID": run_dir.name,
                "PRAXIST_STAGE_ID": "research_loop",
                "PROTECTED_PIDS_DIR": str(run_dir / "protected_pids"),
            }
            with patch.dict(os.environ, env, clear=False):
                entry = register_pid(
                    os.getpid(), peer_id="gen0_peer0", tag="long-eval", run_dir=run_dir
                )
                self.assertEqual(entry.pid, os.getpid())
                self.assertTrue(unregister_pid(os.getpid(), peer_id="gen0_peer0", run_dir=run_dir))

                proc = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(10)"],
                    start_new_session=True,
                )
                rc = monitor_subprocess_with_grace(
                    proc,
                    log_path=run_dir / "missing.log",
                    total_epochs=10,
                    policy=TimeoutPolicy(
                        hard_cap_seconds=0,
                        grace_check_interval_seconds=1,
                        kill_grace_seconds=1,
                    ),
                )

            self.assertNotEqual(rc, 0)
            events = _trajectory_kinds(run_dir)
            self.assertIn("resource.protected_pid_registered", events)
            self.assertIn("resource.protected_pid_unregistered", events)
            self.assertIn("resource.training_timeout_abort", events)


def _write_budget_grant(run_dir: Path, approved: dict[str, float]) -> tuple[str, str]:
    ledger = BudgetLedger(run_dir, run_dir.name)
    request = BudgetRequest(
        request_id="req_step16",
        requester_id="workflow_stage:research_loop",
        experiment_id="step16",
        model_profile_ref=None,
        requested=approved,
        expected_value={"confidence": "strong"},
        evidence_refs=["test"],
        cheaper_alternatives=[],
        abort_conditions=[],
    )
    decision = BudgetDecision(
        decision="grant",
        reason_codes=["test"],
        grant=BudgetGrant(
            grant_id="grant_step16",
            approved=approved,
            conditions=["record_actual_usage"],
            expires_at_generation=None,
        ),
    )
    ledger.append_request(
        request,
        actor_ref="test",
        stage_id="research_loop",
        action_type="test_request",
        reason="test",
    )
    ledger.append_decision(
        request,
        decision,
        actor_ref="budget_policy:test",
        stage_id="research_loop",
        action_type="test_decision",
        reason="test",
    )
    return "grant_step16", "req_step16"


class Issue75Batch4FromRunConfigTest(unittest.TestCase):
    """``BudgetedActionGuard.from_run_config`` is the env-free constructor."""

    def test_from_run_config_uses_explicit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_cfg"
            cfg = RunConfig(
                run_id="explicit-run",
                run_dir=run_dir,
                stage_id="explicit-stage",
                budget_grant_id="grant-explicit",
                budget_request_id="req-explicit",
            )
            # Env values present after RunConfig construction must not leak.
            with patch.dict(
                os.environ,
                {
                    "PRAXIST_RUN_DIR": "/should/not/leak",
                    "PRAXIST_RUN_ID": "env-ignored",
                    "PRAXIST_STAGE_ID": "env-ignored",
                    "PRAXIST_BUDGET_GRANT_ID": "env-ignored",
                    "PRAXIST_BUDGET_REQUEST_ID": "env-ignored",
                },
                clear=False,
            ):
                guard = BudgetedActionGuard.from_run_config(
                    cfg, action_type="eval_runner", actor_ref="resource_guard:test"
                )
        self.assertEqual(guard.run_dir, run_dir)
        self.assertEqual(guard.run_id, "explicit-run")
        self.assertEqual(guard.stage_id, "explicit-stage")
        self.assertEqual(guard.budget_grant_id, "grant-explicit")
        self.assertEqual(guard.request_id, "req-explicit")

    def test_from_run_config_applies_same_defaults_as_from_env(self) -> None:
        """An empty RunConfig produces the same defaults as ``from_env`` does

        when no env vars are set: ``stage_id="research_loop"``,
        ``run_id="legacy_direct"`` (no run_dir to derive from), and
        ``None`` budget refs.
        """
        cfg = RunConfig()
        # Clear the relevant env vars to confirm we don't read them.
        with patch.dict(
            os.environ,
            {
                "PRAXIST_RUN_DIR": "",
                "PRAXIST_RUN_ID": "",
                "PRAXIST_STAGE_ID": "",
                "PRAXIST_BUDGET_GRANT_ID": "",
                "PRAXIST_BUDGET_REQUEST_ID": "",
            },
            clear=False,
        ):
            guard = BudgetedActionGuard.from_run_config(cfg, action_type="t", actor_ref="actor")
        self.assertIsNone(guard.run_dir)
        self.assertEqual(guard.run_id, "legacy_direct")
        self.assertEqual(guard.stage_id, "research_loop")
        self.assertIsNone(guard.budget_grant_id)
        self.assertIsNone(guard.request_id)

    def test_from_run_config_run_id_derives_from_run_dir_name(self) -> None:
        """When ``run_id`` is empty but ``run_dir`` is set, use the dir name.

        Mirrors ``from_env``'s fallback so the two constructors produce
        equivalent guards from equivalent inputs.
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_2026-05-17"
            cfg = RunConfig(run_dir=run_dir)
            guard = BudgetedActionGuard.from_run_config(cfg, action_type="t", actor_ref="actor")
        self.assertEqual(guard.run_id, run_dir.name)

    def test_record_budgeted_action_from_run_config_records_usage(self) -> None:
        """End-to-end: the ``_from_run_config`` wrapper records the same

        usage shape ``record_budgeted_action_from_env`` would, without
        touching ``os.environ``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_record"
            run_dir.mkdir()
            grant_id, request_id = _write_budget_grant(run_dir, {"wall_clock_seconds": 60.0})
            cfg = RunConfig(
                run_id=run_dir.name,
                run_dir=run_dir,
                stage_id="research_loop",
                budget_grant_id=grant_id,
                budget_request_id=request_id,
            )
            report = record_budgeted_action_from_run_config(
                cfg,
                action_type="eval_runner",
                actor_ref="resource_guard:test",
                actual_usage={"wall_clock_seconds": 5.0},
                expected_units=("wall_clock_seconds",),
                reason="test_from_run_config",
            )
            self.assertTrue(report.recorded)
            records = BudgetLedger(run_dir, run_dir.name).records()
            usage = [r for r in records if r["kind"] == "usage"][-1]
            self.assertEqual(usage["actual_usage"]["wall_clock_seconds"], 5.0)
            self.assertEqual(usage["grant_id"], grant_id)

    def test_emit_resource_event_from_run_config_emits_trajectory_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_emit"
            run_dir.mkdir()
            cfg = RunConfig(
                run_id=run_dir.name,
                run_dir=run_dir,
                stage_id="research_loop",
            )
            emit_resource_event_from_run_config(
                cfg,
                "resource.test_event",
                action_type="test",
                actor_ref="resource_guard:test",
                payload={"hello": "world"},
            )
            self.assertIn("resource.test_event", _trajectory_kinds(run_dir))


def _trajectory_kinds(run_dir: Path) -> list[str]:
    path = run_dir / "trajectory.jsonl"
    return [
        json.loads(line)["kind"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
