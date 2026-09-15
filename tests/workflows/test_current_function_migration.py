from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from praxist.core.redaction import scan_file
from praxist.core.replay import dry_run, verify_run
from praxist.core.storage import read_jsonl
from praxist.core.task_project import resolve_task_project, task_project_global_plugin_refs
from praxist.plugins.workflow_stages.research_loop.startup import (
    finalize_research_loop_plugin_run,
    prepare_research_loop_plugin_run,
)


class CurrentFunctionMigrationTest(unittest.TestCase):
    def test_terminal_finalization_requests_final_report_after_completed_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_dir = root / "run_terminal_report"
            with patch.dict(os.environ, {}, clear=False):
                prepared = prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    model="fake-deterministic",
                    local_mode=True,
                    frontier_strategy="auto",
                    credential_profile="fake_multi_key",
                    command="terminal report test",
                )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.startup."
                    "resume_state.reported_completed_generations",
                    return_value=2,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.run_report."
                    "generate_boundary_report_safely"
                ) as generate_report,
            ):
                finalize_research_loop_plugin_run(
                    prepared,
                    success=False,
                    result={"exit_condition": "interrupted"},
                    error="SIGTERM",
                    exit_code=143,
                )
            generate_report.assert_called_once_with(
                run_dir=run_dir,
                task_dir=_toy_task_path(),
                generation_id=1,
                final=True,
            )

    def test_failed_zero_generation_run_still_requests_final_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_dir = root / "run_failed_before_generation"
            with patch.dict(os.environ, {}, clear=False):
                prepared = prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    model="fake-deterministic",
                    local_mode=True,
                    frontier_strategy="auto",
                    credential_profile="fake_multi_key",
                    command="failed terminal report test",
                )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.startup."
                    "resume_state.reported_completed_generations",
                    return_value=0,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.run_report."
                    "generate_boundary_report_safely"
                ) as generate_report,
            ):
                finalize_research_loop_plugin_run(
                    prepared,
                    success=False,
                    result={"exit_condition": "failed"},
                    error="startup failed",
                    exit_code=1,
                )

            generate_report.assert_called_once_with(
                run_dir=run_dir,
                task_dir=_toy_task_path(),
                generation_id=-1,
                final=True,
            )

    def test_task_project_exports_current_functionality_refs_without_task_plugin(self) -> None:
        project = resolve_task_project(_toy_task_path(), workspace=Path.cwd())
        selected_refs = {
            ref.as_string() for ref in task_project_global_plugin_refs(project.descriptor)
        }

        expected_refs = {
            "workflow_stage:research_loop",
            "panel_topology:fake_two_round",
            "role:fake_peer",
            "role:fake_pi",
            "role:fake_chair",
            "audit_rule:fake_panel_audit",
            "evaluation:fake_pareto",
        }
        self.assertTrue(
            expected_refs.issubset(selected_refs), sorted(expected_refs - selected_refs)
        )

    def test_startup_loads_external_task_project_and_writes_replay_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_dir = root / "run_gate8_toy"
            with patch.dict(os.environ, {}, clear=False):
                prepared = prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    model="fake-deterministic",
                    local_mode=True,
                    frontier_strategy="auto",
                    credential_profile="fake_multi_key",
                    command="test gate8",
                )
            self.assertEqual(prepared.task_spec.task_id, "toy_math")
            self.assertEqual(prepared.task_spec.evaluation.primary_metric, "deterministic_score")
            self.assertEqual(prepared.task_project_path, _toy_task_path())
            self.assertTrue((run_dir / "task_project_manifest.json").exists())

            effective = yaml.safe_load(
                (run_dir / "effective_task_spec.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                effective["praxist_plugins"]["panel"]["topology"],
                "panel_topology:fake_two_round",
            )
            self.assertIn("role:fake_chair", effective["praxist_plugins"]["panel"]["roles"])

            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 0,
                    "run_dir": str(run_dir),
                    "exit_condition": "test",
                },
            )
            report = verify_run(run_dir)
            self.assertTrue(report["success"], report)
            dry_report = dry_run(run_dir)
            self.assertFalse(dry_report["success"])
            self.assertIn("dry-run expected", "\n".join(dry_report["errors"]))
            self.assertEqual(scan_file(run_dir / "credentials_redacted.json"), [])

            trajectory, errors = read_jsonl(run_dir / "trajectory.jsonl")
            self.assertEqual(errors, [])
            kinds = [event["kind"] for event in trajectory]
            self.assertIn("plugins.resolved", kinds)
            self.assertIn("run.finalized", kinds)

            resolution = json.loads(
                (run_dir / "plugin_resolution.json").read_text(encoding="utf-8")
            )
            selected_refs = {
                item["metadata"]["kind"] + ":" + item["metadata"]["name"]
                for item in resolution["selected"]
            }
            self.assertNotIn("task:toy_math", selected_refs)
            self.assertIn("model_provider:fake_provider", selected_refs)


def _toy_task_path() -> Path:
    return (Path.cwd() / "templates" / "tasks" / "toy_math").resolve()


if __name__ == "__main__":
    unittest.main()
