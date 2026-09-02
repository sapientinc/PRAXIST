from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core.ledgers import BudgetLedger
from praxist.core.panel_topology import panel_topology_for_ref
from praxist.core.protocol import BudgetDecision, BudgetGrant, BudgetRequest
from praxist.core.replay import verify_run
from praxist.core.workflow import (
    OptionalWorkflowStageContext,
    literature_scout_contract,
    optional_workflow_stage,
)
from praxist.plugins.agent_runtimes.claude_sdk.adapter import (
    extract_legacy_output,
    format_legacy_message,
    is_billing_error,
)
from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import panel_runner
from praxist.plugins.workflow_stages.research_loop.stage import ResearchLoopStage
from praxist.plugins.workflow_stages.research_loop.startup import (
    finalize_research_loop_plugin_run,
    prepare_research_loop_plugin_run,
)


class ResearchLoopWorkflowTest(unittest.TestCase):
    def test_panel_runner_is_facade_and_topology_plans_roles(self) -> None:
        source = inspect.getsource(panel_runner)
        self.assertNotIn("ROLE_REGISTRY", source)
        self.assertNotIn("build_evidence_pack", source)

        topology = panel_topology_for_ref("panel_topology:legacy_multi_pi_two_round")
        self.assertEqual(topology.roles_for_mode("mini"), ["builder", "skeptic"])
        self.assertEqual(topology.roles_for_mode("full"), ["builder", "skeptic", "portfolio"])
        self.assertEqual(
            topology.roles_for_mode("high_stakes"),
            ["builder", "skeptic", "portfolio", "external_validity"],
        )
        self.assertTrue(
            topology.has_high_stakes_signal(
                {
                    "claim_ledger_digest": {
                        "active": [
                            {"title": "architecture-independent scaling law", "boundary": ""}
                        ]
                    }
                }
            )
        )

    def test_claude_sdk_runtime_adapter_owns_message_normalization(self) -> None:
        class TextBlock:
            text = "hello"

        class ToolUseBlock:
            name = "Bash"
            input = {"command": "python train.py"}

        class AssistantMessage:
            content = [TextBlock(), ToolUseBlock()]

        class ResultMessage:
            result = {"ok": True}

        messages = [AssistantMessage(), ResultMessage()]
        output = extract_legacy_output(messages)
        self.assertEqual(output["text_outputs"], ["hello"])
        self.assertEqual(output["tool_uses"][0]["tool"], "Bash")
        self.assertIn("python train.py", format_legacy_message(messages[0], "agent"))
        self.assertTrue(is_billing_error("exceeded your current quota"))

    def test_budget_ledger_appends_and_enforces_grants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp), "run_budget")
            request = BudgetRequest(
                request_id="req",
                requester_id="workflow_stage:research_loop",
                experiment_id="exp",
                model_profile_ref="cheap_peer",
                requested={"tokens": 1000},
                expected_value={"confidence": "strong"},
                evidence_refs=["task:test"],
                cheaper_alternatives=[],
                abort_conditions=[],
            )
            decision = BudgetDecision(
                decision="grant",
                reason_codes=["test"],
                grant=BudgetGrant(
                    grant_id="grant_req",
                    approved={"tokens": 1000},
                    conditions=[],
                    expires_at_generation=None,
                ),
            )
            ledger.append_request(
                request,
                actor_ref="workflow_stage:research_loop",
                stage_id="research_loop",
                action_type="stage_start",
                reason="test_request",
            )
            ledger.append_decision(
                request,
                decision,
                actor_ref="budget_policy:test",
                stage_id="research_loop",
                action_type="stage_start",
                reason="test_decision",
            )
            ledger.append_usage(
                request_id=request.request_id,
                grant_id="grant_req",
                actor_ref="agent_runtime:fake_runtime",
                stage_id="research_loop",
                action_type="agent_turn",
                actual_usage={"tokens": 100},
                reason="test_usage",
            )
            self.assertEqual(
                [record["kind"] for record in ledger.records()], ["request", "decision", "usage"]
            )
            with self.assertRaises(ValueError):
                ledger.append_usage(
                    request_id="missing",
                    grant_id="missing_grant",
                    actor_ref="agent_runtime:fake_runtime",
                    stage_id="research_loop",
                    action_type="agent_turn",
                    actual_usage={"tokens": 1},
                    reason="missing",
                )
            overrun = ledger.append_usage(
                request_id=request.request_id,
                grant_id="grant_req",
                actor_ref="agent_runtime:fake_runtime",
                stage_id="research_loop",
                action_type="agent_turn",
                actual_usage={"tokens": 1000},
                reason="test_overrun",
            )
            self.assertTrue(overrun["budget_overrun"])
            self.assertEqual(overrun["overrun_units"], ["tokens"])

    def test_optional_workflow_stage_stubs_expose_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_optional_stubs"
            run_dir.mkdir()
            ideation = optional_workflow_stage("ideation")
            self.assertEqual(
                ideation.describe().outputs, ["research_plan", "task_draft", "initial_agenda"]
            )

            skipped = asyncio.run(
                ideation.execute(
                    OptionalWorkflowStageContext(
                        run_dir=run_dir,
                        run_id="run_optional_stubs",
                        enabled=False,
                    )
                )
            )
            self.assertTrue(skipped.success)
            self.assertEqual(skipped.status, "skipped")

            failed = asyncio.run(
                optional_workflow_stage("paper_writing").execute(
                    OptionalWorkflowStageContext(
                        run_dir=run_dir,
                        run_id="run_optional_stubs",
                        enabled=True,
                    )
                )
            )
            self.assertFalse(failed.success)
            self.assertIn("no implementation is configured", failed.error or "")

            reviewer = asyncio.run(
                optional_workflow_stage("reviewer").execute(
                    OptionalWorkflowStageContext(
                        run_dir=run_dir,
                        run_id="run_optional_stubs",
                        enabled=True,
                        mode="fake",
                    )
                )
            )
            self.assertTrue(reviewer.success)
            self.assertEqual(reviewer.output_artifacts[0]["schema_ref"], "core:review_artifact.v1")
            self.assertEqual(literature_scout_contract()["role_ref"], "task_role:literature_scout")
            self.assertEqual(literature_scout_contract()["role_scope"], "task_project")
            self.assertEqual(
                literature_scout_contract()["tool_ref"], "tool_server:literature_lookup"
            )
            self.assertEqual(
                literature_scout_contract()["resource_policy"], "current_environment_only"
            )

    def test_research_loop_stage_resolve_only_writes_budget_ledger(self) -> None:
        self.assertEqual(ResearchLoopStage().describe().stage_id, "research_loop")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_gate_c"
            with patch.dict(os.environ, {}, clear=False):
                prepared = prepare_research_loop_plugin_run(
                    task_project_path=Path.cwd() / "templates" / "tasks" / "toy_math",
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    model="fake-deterministic",
                    local_mode=True,
                    frontier_strategy="auto",
                    credential_profile="fake_multi_key",
                    command="gate c test",
                )
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 0,
                    "run_dir": str(run_dir),
                    "exit_condition": "resolve_only",
                },
            )
            self.assertTrue(verify_run(run_dir)["success"])
            records = [
                json.loads(line)
                for line in (run_dir / "budget_ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual([record["kind"] for record in records], ["request", "decision"])
            self.assertTrue(prepared.stage_budget_grant_id)
            trajectory = [
                json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()
            ]
            skipped = {
                (event.get("scope") or {}).get("stage_id")
                for event in trajectory
                if event.get("kind") == "workflow.stage_skipped"
            }
            self.assertEqual(skipped, {"ideation", "paper_writing", "reviewer"})
            disabled = prepared.resolution_manifest["disabled_optional"]
            self.assertTrue(any(item.get("stage_id") == "ideation" for item in disabled))

    def test_research_loop_startup_rejects_preexisting_artifacts_with_resume_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = (root / "run_existing").resolve()
            with patch.dict(os.environ, {}, clear=False):
                prepare_research_loop_plugin_run(
                    task_project_path=Path.cwd() / "templates" / "tasks" / "toy_math",
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    model="fake-deterministic",
                    local_mode=True,
                    frontier_strategy="auto",
                    credential_profile="fake_multi_key",
                    command="initial test",
                )
            with self.assertRaises(ValueError) as cm:
                prepare_research_loop_plugin_run(
                    task_project_path=Path.cwd() / "templates" / "tasks" / "toy_math",
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    model="fake-deterministic",
                    local_mode=True,
                    frontier_strategy="auto",
                    credential_profile="fake_multi_key",
                    command="second test",
                    resume=False,
                )
            self.assertIn("already contains Praxist run artifacts", str(cm.exception))
            self.assertIn("Use --resume or --resume-from", str(cm.exception))
            self.assertNotIn("Resume mode is not implemented", str(cm.exception))

            with patch.dict(os.environ, {}, clear=False):
                resumed = prepare_research_loop_plugin_run(
                    task_project_path=Path.cwd() / "templates" / "tasks" / "toy_math",
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    model="fake-deterministic",
                    local_mode=True,
                    frontier_strategy="auto",
                    credential_profile="fake_multi_key",
                    command="resumed test",
                    resume=True,
                )
            self.assertEqual(resumed.run_dir, run_dir)
            run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_json.get("resume_count"), 1)
            self.assertIn("resumed_at", run_json)

    def test_research_loop_startup_rejects_unrelated_nonempty_directory_without_resume_guidance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = (root / "run_unrelated").resolve()
            run_dir.mkdir()
            (run_dir / "random.txt").write_text("hello", encoding="utf-8")
            with self.assertRaises(ValueError) as cm:
                prepare_research_loop_plugin_run(
                    task_project_path=Path.cwd() / "templates" / "tasks" / "toy_math",
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    model="fake-deterministic",
                    local_mode=True,
                    frontier_strategy="auto",
                    credential_profile="fake_multi_key",
                    command="unrelated test",
                    resume=False,
                )
            self.assertIn("already exists and is not empty", str(cm.exception))
            self.assertIn("Choose a fresh run directory.", str(cm.exception))
            self.assertNotIn("Resume mode is not implemented", str(cm.exception))
            self.assertNotIn("--resume-from", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
