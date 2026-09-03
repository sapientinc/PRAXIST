from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from praxist.core.budget import policy_for_ref
from praxist.core.credentials import CredentialFailoverManager, CredentialResolver
from praxist.core.ledgers import BudgetLedger
from praxist.core.protocol import BudgetDecision, BudgetGrant, BudgetRequest
from praxist.core.redaction import scan_text
from praxist.core.registry import PluginLoader, PluginRoots, static_resolution_manifest
from praxist.core.replay import verify_run
from praxist.core.source_snapshot import build_core_source_snapshot
from praxist.core.storage import ArtifactWriter, sha256_bytes, write_json
from praxist.core.trajectory import TrajectoryWriter
from praxist.plugins.agent_runtimes.claude_sdk.adapter import (
    ClaudeSdkAgentRuntime,
    LegacyAgentResult,
    LegacyClaudeRuntimeOptions,
    extract_legacy_output,
    format_legacy_message,
)
from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent
from praxist.plugins.workflow_stages.research_loop.backend.run_summary import (
    write_run_summary,
)
from praxist.plugins.workflow_stages.research_loop.stage import (
    ResearchLoopStage,
    ResearchLoopStageContext,
)
from praxist.plugins.workflow_stages.research_loop.startup import (
    _task_runtime_env,
    finalize_research_loop_plugin_run,
    prepare_research_loop_plugin_run,
)
from praxist.testing.fake_workflow_fixture import run_fake_workflow_fixture


class RefactorHardeningTest(unittest.TestCase):
    def setUp(self) -> None:
        self._credential_env = patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "or-test-credential-for-refactor-hardening",
                "ANTHROPIC_API_KEY": "anthropic-test-credential-for-refactor-hardening",
            },
            clear=False,
        )
        self._credential_env.start()

    def tearDown(self) -> None:
        self._credential_env.stop()

    def test_plugin_dependency_cycles_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workflow_manifest(
                root,
                "stage_a",
                [{"kind": "workflow_stage", "name": "stage_b", "version": ">=0.1,<1.0"}],
            )
            self._write_workflow_manifest(
                root,
                "stage_b",
                [{"kind": "workflow_stage", "name": "stage_a", "version": ">=0.1,<1.0"}],
            )
            loader = PluginLoader(PluginRoots(bundled=[root], user=[], project=[]))
            with self.assertRaisesRegex(ValueError, "dependency cycle"):
                loader.resolve(["workflow_stage:stage_a"], root_task_ref="workflow_stage:stage_a")

    def test_selected_plugin_must_satisfy_later_version_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workflow_manifest(root, "stage_b", [])
            self._write_workflow_manifest(
                root,
                "stage_a",
                [{"kind": "workflow_stage", "name": "stage_b", "version": ">=9.0,<10.0"}],
            )
            loader = PluginLoader(PluginRoots(bundled=[root], user=[], project=[]))
            with self.assertRaisesRegex(ValueError, "does not satisfy required constraint"):
                loader.resolve(
                    ["workflow_stage:stage_b", "workflow_stage:stage_a"],
                    root_task_ref="workflow_stage:stage_a",
                )

    def test_unsupported_version_constraint_syntax_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workflow_manifest(root, "stage_a", [])
            loader = PluginLoader(PluginRoots(bundled=[root], user=[], project=[]))
            with self.assertRaisesRegex(ValueError, "Unsupported version constraint"):
                loader.resolve(
                    [{"kind": "workflow_stage", "name": "stage_a", "version": "~=0.1"}],
                    root_task_ref="workflow_stage:stage_a",
                )

    def test_cli_rejects_direct_task_spec_execution(self) -> None:
        from praxist.run import cmd_run

        args = SimpleNamespace(
            fake=False,
            task="",
            task_spec="deprecated/tasks/sam_optimizer/task_spec.yaml",
            workspace="",
            model="",
            runtime="",
            model_provider="",
            budget_policy="",
            credential_profile="",
            run_dir="",
            resolve_only=False,
            local=True,
            frontier_strategy="mixed",
        )
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as raised:
                cmd_run(args)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("direct --task-spec execution is disabled", stderr.getvalue())

    def test_cli_startup_failure_does_not_create_partial_run_dir(self) -> None:
        from praxist.run import cmd_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_missing_credential"
            task_path = _write_research_task(root / "task")
            args = SimpleNamespace(
                fake=False,
                task_path=str(task_path),
                task="",
                task_spec="",
                workspace=str(root),
                model="anthropic/claude-opus-4.7",
                runtime="agent_runtime:claude_sdk",
                model_provider="model_provider:openrouter",
                budget_policy="budget_policy:default_basic",
                credential_profile="",
                run_dir=str(run_dir),
                resolve_only=False,
                local=True,
                frontier_strategy="mixed",
            )
            with patch.dict(os.environ, _fixture_plugin_env(), clear=True):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        cmd_run(args)
            self.assertEqual(raised.exception.code, 3)
            self.assertIn("startup failed", stderr.getvalue())
            self.assertFalse(run_dir.exists())

    def test_cli_defaults_to_pi_directed_auto_strategy(self) -> None:
        config = Path("praxist/config.py").read_text(encoding="utf-8")
        # #75 batch 8b moved the cohort/frontier literals into
        # ``core/run_config.py``; ``config.py`` re-exports them as
        # ``MAX_GENERATIONS`` / ``COHORT_SIZE`` / ``PER_GENERATION_HOURS``
        # / ``PROMOTE_TOP_K`` / ``FRONTIER_STRATEGY``. Pin the new
        # re-export shape and the historic dogfood overrides we never
        # want to silently re-introduce.
        self.assertIn("DEFAULT_MAX_GENERATIONS as MAX_GENERATIONS", config)
        self.assertIn("DEFAULT_PER_GENERATION_HOURS as PER_GENERATION_HOURS", config)
        self.assertIn("DEFAULT_FRONTIER_STRATEGY as FRONTIER_STRATEGY", config)
        self.assertIn("legacy/direct caller fallbacks", config)
        self.assertNotIn("Single source of truth", config)
        self.assertNotIn('MAX_GENERATIONS = int(os.getenv("MAX_GENERATIONS", "4"))', config)
        self.assertNotIn(
            'PER_GENERATION_HOURS = int(os.getenv("PER_GENERATION_HOURS", "24"))', config
        )
        self.assertNotIn('FRONTIER_STRATEGY = os.getenv("FRONTIER_STRATEGY", "mixed")', config)

        run_config = Path("praxist/core/run_config.py").read_text(encoding="utf-8")
        self.assertIn("DEFAULT_MAX_GENERATIONS = 1", run_config)
        self.assertIn("DEFAULT_PER_GENERATION_HOURS = 5", run_config)
        self.assertIn('DEFAULT_FRONTIER_STRATEGY = "auto"', run_config)

        generation_loop = Path(
            "praxist/plugins/workflow_stages/research_loop/backend/generation_loop.py"
        ).read_text(encoding="utf-8")
        self.assertIn("gen >= 1 follows PI-directed per-peer role contracts", generation_loop)
        self.assertNotIn("explore→mixed→exploit", generation_loop)
        self.assertNotIn("annealed 3-cycle", generation_loop)

        help_result = subprocess.run(
            [sys.executable, "-m", "praxist.run", "run", "--help"],
            check=True,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
        )
        help_text = " ".join(help_result.stdout.split())
        self.assertIn("PI-directed per-peer role contracts", help_text)
        self.assertNotIn("3-cycle", help_result.stdout)

    def test_docs_exclude_one_off_improvement_plans_and_logs(self) -> None:
        removed_docs = (
            "docs/roadmap.md",
            "docs/history/refactor-implementation-log.md",
            "docs/analysis/run_artifact_system_simplification_2026-06-29.md",
            "docs/analysis/run_artifact_system_subagent_simulation_2026-06-29.md",
            "docs/analysis/scientific_research_support_improvement_plan_2026-07-01.md",
            "docs/guides/qdig-actual-verification-2026-06-19.md",
            "docs/guides/qdig-implementation-and-comparison-2026-06-19.md",
            "docs/guides/scientific-research-support-improvements.zh-CN.md",
            "docs/guides/mle-runtime-support.md",
        )
        for rel in removed_docs:
            with self.subTest(rel=rel):
                self.assertFalse(Path(rel).exists())

    def test_task_spec_loads_all_compute_budget_resource_hints(self) -> None:
        from praxist.task_spec import load_task_spec

        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "task.yaml"
            spec_path.write_text(
                yaml.safe_dump(
                    {
                        "task_id": "budget_test",
                        "compute_budget": {
                            "per_experiment_gpu_hours": 3.5,
                            "max_parallel_runs_per_peer": 7,
                            "peer_gpu_memory_gb": 24.0,
                            "peer_gpu_util_pct": 35.0,
                            "peer_cpu_cores": 12,
                        },
                    }
                ),
                encoding="utf-8",
            )
            budget = load_task_spec(str(spec_path)).compute_budget

        self.assertEqual(budget.per_experiment_gpu_hours, 3.5)
        self.assertEqual(budget.max_parallel_runs_per_peer, 7)
        self.assertEqual(budget.peer_gpu_memory_gb, 24.0)
        self.assertEqual(budget.peer_gpu_util_pct, 35.0)
        self.assertEqual(budget.peer_cpu_cores, 12)

    def test_deliverable_snapshot_extraction_skips_untrusted_tar_members(self) -> None:
        from praxist.deliver import extract_frontier_snapshots

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            out_dir = root / "deliverables"
            frontier_dir = run_dir / "frontier" / "gen_0"
            frontier_dir.mkdir(parents=True)
            snapshot = frontier_dir / "frontier_snapshot.tar.gz"

            with tarfile.open(snapshot, "w:gz") as tar:
                safe_payload = b"safe"
                safe = tarfile.TarInfo("safe.txt")
                safe.size = len(safe_payload)
                tar.addfile(safe, io.BytesIO(safe_payload))

                escape_payload = b"escape"
                escape = tarfile.TarInfo("../escape.txt")
                escape.size = len(escape_payload)
                tar.addfile(escape, io.BytesIO(escape_payload))

                link = tarfile.TarInfo("link_to_host")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                tar.addfile(link)

            self.assertEqual(extract_frontier_snapshots(run_dir, out_dir), 1)
            target = out_dir / "code" / "frontier_snapshot"
            self.assertEqual((target / "safe.txt").read_text(encoding="utf-8"), "safe")
            self.assertFalse((out_dir / "code" / "escape.txt").exists())
            self.assertFalse((target / "link_to_host").exists())

    def test_task_project_toolchain_paths_are_not_bundled_plugin_paths(self) -> None:
        descriptor = _toy_task_path().joinpath("task.yaml").read_text(encoding="utf-8")
        self.assertNotIn("praxist/plugins/tasks", descriptor)
        self.assertNotIn("legacy task data fallback", descriptor.lower())
        self.assertFalse(Path("praxist/plugins/tasks").exists())

    def test_legacy_startup_and_failure_summary_are_redacted_before_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_secret"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                budget_policy_ref="budget_policy:default_basic",
                model="anthropic/claude-opus-4.7",
                local_mode=True,
                frontier_strategy="mixed",
                command="python run --token sk-test-redaction-000000",
            )
            finalize_research_loop_plugin_run(
                prepared, success=False, error="bad token sk-test-redaction-000000"
            )

            for rel in ("startup_config.json", "run_summary.json", "trajectory.jsonl"):
                self.assertNotIn(
                    "sk-test-redaction-000000",
                    (run_dir / rel).read_text(encoding="utf-8"),
                    rel,
                )
            self.assertTrue(verify_run(run_dir)["success"])

    def test_secret_like_legacy_run_id_is_rejected(self) -> None:
        self.assertIn("openai_style_key", scan_text("run_sk-test-redaction-000000"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "run_id contains secret-looking content"):
                prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=root / "run_sk-test-redaction-000000",
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:openrouter",
                    budget_policy_ref="budget_policy:default_basic",
                    model="anthropic/claude-opus-4.7",
                    local_mode=True,
                    frontier_strategy="mixed",
                    command="test",
                )

    def test_legacy_plugin_run_rejects_existing_run_artifacts_but_allows_empty_precreated_dirs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_reuse"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                budget_policy_ref="budget_policy:default_basic",
                model="anthropic/claude-opus-4.7",
                local_mode=True,
                frontier_strategy="mixed",
                command="first",
            )
            finalize_research_loop_plugin_run(
                prepared, success=True, result={"exit_condition": "test"}
            )
            with self.assertRaisesRegex(ValueError, "Praxist run artifacts"):
                prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:openrouter",
                    budget_policy_ref="budget_policy:default_basic",
                    model="anthropic/claude-opus-4.7",
                    local_mode=True,
                    frontier_strategy="mixed",
                    command="second",
                )
            precreated = root / "run_precreated_empty_dirs"
            (precreated / "logs").mkdir(parents=True)
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=precreated,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                budget_policy_ref="budget_policy:default_basic",
                model="anthropic/claude-opus-4.7",
                local_mode=True,
                frontier_strategy="mixed",
                command="third",
            )
            self.assertTrue((prepared.run_dir / "run.json").exists())

    def test_run_summary_preserves_task_extension_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_summary.json"
            write_json(path, {"task_result": {"score": 0.75}, "status": "in_progress"})

            summary = write_run_summary(
                path,
                {
                    "status": "succeeded",
                    "exit_condition": "max_generations",
                    "generations_completed": 2,
                },
            )

            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(summary["task_result"], {"score": 0.75})
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["task_result"], {"score": 0.75})
            self.assertEqual(loaded["generations_completed"], 2)

    def test_replay_rejects_duplicate_lifecycle_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            TrajectoryWriter(run_dir, result["run_id"]).emit("run.started")
            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("run.started" in error for error in report["errors"]))

    def test_replay_rejects_corrupt_trajectory_sequence_and_duplicate_event_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            trajectory_path = Path(result["run_dir"]) / "trajectory.jsonl"
            records = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
            records[1]["seq"] = records[0]["seq"]
            records[1]["event_id"] = records[0]["event_id"]
            trajectory_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(any("seq mismatch" in error for error in report["errors"]))
            self.assertTrue(any("duplicate event_id" in error for error in report["errors"]))

    def test_replay_rejects_events_appended_after_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            TrajectoryWriter(run_dir, result["run_id"]).emit(
                "audit.verdict_recorded",
                scope={"stage_id": "research_loop"},
                payload={"audit_id": "late"},
            )

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("after run.finalized" in error for error in report["errors"]))

    def test_replay_rejects_malformed_trajectory_tail_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            trajectory_path = Path(result["run_dir"]) / "trajectory.jsonl"
            with trajectory_path.open("a", encoding="utf-8") as handle:
                handle.write("{not json}\n")

            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(any("json_decode" in error for error in report["errors"]))

    def test_replay_scans_logs_for_secret_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            log_path = Path(result["run_dir"]) / "logs" / "session.log"
            log_path.write_text("raw sk-test-redaction-000000\n", encoding="utf-8")
            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(any("logs/session.log" in error for error in report["errors"]))

    def test_replay_scans_legacy_generation_text_outputs_for_secret_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            result_file = run_dir / "results" / "leak.txt"
            result_file.parent.mkdir(parents=True)
            result_file.write_text("raw sk-test-redaction-000000\n", encoding="utf-8")
            extensionless_result = run_dir / "results" / "leak"
            extensionless_result.write_text("raw sk-test-redaction-000000\n", encoding="utf-8")
            peer_log = run_dir / "gen_0" / "gen0_peer0" / "session.log"
            peer_log.parent.mkdir(parents=True)
            peer_log.write_text("raw sk-test-redaction-000000\n", encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("results/leak.txt" in error for error in report["errors"]))
            self.assertTrue(any("results/leak:" in error for error in report["errors"]))
            self.assertTrue(
                any("gen_0/gen0_peer0/session.log" in error for error in report["errors"])
            )

    def test_replay_rejects_invalid_budget_usage_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            ledger_path = Path(result["run_dir"]) / "budget_ledger.jsonl"
            records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            for record in records:
                if record.get("kind") == "usage":
                    record["grant_id"] = "grant_does_not_exist"
                    record["actual_usage"] = {"tokens": 10**12}
            ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(any("unknown grant_id" in error for error in report["errors"]))

    def test_replay_rejects_wrong_run_id_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            trajectory_path = run_dir / "trajectory.jsonl"
            trajectory = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
            trajectory[0]["run_id"] = "other_run"
            trajectory_path.write_text(
                "\n".join(json.dumps(record) for record in trajectory) + "\n"
            )
            ledger_path = run_dir / "budget_ledger.jsonl"
            ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            ledger[0]["run_id"] = "other_run"
            ledger_path.write_text("\n".join(json.dumps(record) for record in ledger) + "\n")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any("trajectory:1 run_id mismatch" in error for error in report["errors"])
            )
            self.assertTrue(
                any("budget_ledger:1 run_id mismatch" in error for error in report["errors"])
            )

    def test_replay_rejects_negative_and_nonfinite_budget_usage(self) -> None:
        for bad_value in (-1, float("nan"), float("inf")):
            with tempfile.TemporaryDirectory() as tmp:
                result = run_fake_workflow_fixture(
                    workspace=Path(tmp),
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    credential_profile="fake_multi_key",
                )
                ledger_path = Path(result["run_dir"]) / "budget_ledger.jsonl"
                records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
                for record in records:
                    if record.get("kind") == "usage":
                        record["actual_usage"] = {"tokens": bad_value}
                ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

                report = verify_run(Path(result["run_dir"]))
                self.assertFalse(report["success"])
                self.assertTrue(any("invalid usage" in error for error in report["errors"]))

    def test_budget_rejects_nonfinite_grants_live_and_in_replay(self) -> None:
        request = BudgetRequest(
            request_id="nan_req",
            requester_id="peer",
            experiment_id="nan",
            model_profile_ref="cheap_peer",
            requested={"tokens": float("nan")},
            expected_value={"confidence": "strong"},
            evidence_refs=["finding:x"],
            cheaper_alternatives=[],
            abort_conditions=[],
        )
        self.assertEqual(
            policy_for_ref("budget_policy:default_basic").decide(request).decision, "deny"
        )

        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp), "run_budget")
            valid_request = BudgetRequest(
                request_id="req",
                requester_id="workflow_stage:research_loop",
                experiment_id="exp",
                model_profile_ref="cheap_peer",
                requested={"tokens": 100},
                expected_value={"confidence": "strong"},
                evidence_refs=["task:test"],
                cheaper_alternatives=[],
                abort_conditions=[],
            )
            with self.assertRaisesRegex(ValueError, "invalid amount"):
                ledger.append_decision(
                    valid_request,
                    BudgetDecision(
                        decision="grant",
                        reason_codes=["test"],
                        grant=BudgetGrant(
                            grant_id="grant_nan",
                            approved={"tokens": float("nan")},
                            conditions=[],
                            expires_at_generation=None,
                        ),
                    ),
                    actor_ref="budget_policy:test",
                    stage_id="research_loop",
                    action_type="stage_start",
                    reason="bad_grant",
                )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            ledger_path = Path(result["run_dir"]) / "budget_ledger.jsonl"
            records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            for record in records:
                if record.get("kind") == "decision":
                    record["granted_budget"] = {"tokens": float("nan")}
                if record.get("kind") == "usage":
                    record["actual_usage"] = {"tokens": 10**12}
            ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(any("invalid amount" in error for error in report["errors"]))

    def test_budget_rejects_invalid_requested_budget_live_and_in_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp), "run_budget")
            request = BudgetRequest(
                request_id="bad_req",
                requester_id="workflow_stage:research_loop",
                experiment_id="exp",
                model_profile_ref="cheap_peer",
                requested={"tokens": -1},
                expected_value={"confidence": "strong"},
                evidence_refs=["task:test"],
                cheaper_alternatives=[],
                abort_conditions=[],
            )
            with self.assertRaisesRegex(ValueError, "invalid amount"):
                ledger.append_request(
                    request,
                    actor_ref="workflow_stage:research_loop",
                    stage_id="research_loop",
                    action_type="stage_start",
                    reason="bad_request",
                )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            ledger_path = Path(result["run_dir"]) / "budget_ledger.jsonl"
            records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            for record in records:
                if record.get("kind") in {"request", "decision"}:
                    record["requested_budget"] = {"tokens": -1}
            ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(
                any(
                    "requested_budget" in error and "invalid amount" in error
                    for error in report["errors"]
                )
            )

    def test_budget_rejects_unknown_units_live_and_in_replay(self) -> None:
        request = BudgetRequest(
            request_id="bad_unit",
            requester_id="workflow_stage:research_loop",
            experiment_id="exp",
            model_profile_ref="cheap_peer",
            requested={"usd": 1_000_000},
            expected_value={"confidence": "strong"},
            evidence_refs=["task:test"],
            cheaper_alternatives=[],
            abort_conditions=[],
        )
        self.assertEqual(
            policy_for_ref("budget_policy:default_basic").decide(request).decision, "deny"
        )
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp), "run_budget")
            with self.assertRaisesRegex(ValueError, "unsupported unit"):
                ledger.append_request(
                    request,
                    actor_ref="workflow_stage:research_loop",
                    stage_id="research_loop",
                    action_type="stage_start",
                    reason="bad_unit",
                )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            ledger_path = Path(result["run_dir"]) / "budget_ledger.jsonl"
            records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            for record in records:
                if record.get("kind") in {"request", "decision"}:
                    record["requested_budget"] = {"usd": 1_000_000}
                    record["request_record"]["requested"] = {"usd": 1_000_000}
                if record.get("kind") == "decision":
                    record["granted_budget"] = {"usd": 1_000_000}
                    record["decision_record"]["decision"] = "grant"
                    record["decision_record"]["grant"] = {
                        "grant_id": record["grant_id"],
                        "approved": {"usd": 1_000_000},
                        "conditions": [],
                        "expires_at_generation": None,
                    }
                if record.get("kind") == "usage":
                    record["actual_usage"] = {"usd": 1_000_000}
            ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(any("unsupported unit" in error for error in report["errors"]))

    def test_replay_recomputes_budget_decisions_from_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            ledger_path = run_dir / "budget_ledger.jsonl"
            records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            for record in records:
                if record.get("kind") in {"request", "decision"}:
                    record["requested_budget"] = {"tokens": 10**12, "wall_clock_seconds": 30}
                    record["request_record"]["requested"] = dict(record["requested_budget"])
                if record.get("kind") == "decision":
                    record["granted_budget"] = dict(record["requested_budget"])
                    record["decision_record"]["grant"]["approved"] = dict(
                        record["requested_budget"]
                    )
                if record.get("kind") == "usage":
                    record["actual_usage"] = {"tokens": 10**12, "wall_clock_seconds": 1}
            ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("replayed policy" in error for error in report["errors"]))

    def test_replay_warns_when_reported_usage_omits_approved_budget_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            ledger_path = run_dir / "budget_ledger.jsonl"
            records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            for record in records:
                if record.get("kind") == "usage":
                    record["actual_usage"] = {"wall_clock_seconds": 1}
            ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            report = verify_run(run_dir)
            self.assertTrue(report["success"])
            self.assertTrue(
                any(
                    "missing approved unit and usage_unknown: tokens" in warning
                    for warning in report["warnings"]
                )
            )

    def test_replay_rejects_summary_tampering_that_hides_missing_budget_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            ledger_path = run_dir / "budget_ledger.jsonl"
            records = [
                json.loads(line)
                for line in ledger_path.read_text().splitlines()
                if json.loads(line).get("kind") != "usage"
            ]
            ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            summary_path = run_dir / "run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["status"] = "failed"
            summary["exit_condition"] = "resolve_only"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any("missing usage or usage_unknown" in warning for warning in report["warnings"])
            )
            self.assertTrue(any("status mismatch" in error for error in report["errors"]))

    def test_replay_rejects_retroactive_duplicate_budget_grants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            ledger_path = Path(result["run_dir"]) / "budget_ledger.jsonl"
            records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            decision = next(record for record in records if record.get("kind") == "decision")
            for record in records:
                if record.get("kind") == "decision":
                    record["granted_budget"] = {"tokens": 1, "wall_clock_seconds": 30}
                    record["decision_record"]["grant"]["approved"] = record["granted_budget"]
                if record.get("kind") == "usage":
                    record["actual_usage"] = {"tokens": 100}
            duplicate = dict(decision)
            duplicate["record_id"] = "budget_grant_duplicate"
            duplicate["granted_budget"] = {"tokens": 1000}
            records.append(duplicate)
            ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(any("duplicate grant_id" in error for error in report["errors"]))
            self.assertTrue(any("exceeds approved" in warning for warning in report["warnings"]))

    def test_replay_validates_source_event_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            frontier_path = Path(result["run_dir"]) / "findings" / "frontier.jsonl"
            records = [json.loads(line) for line in frontier_path.read_text().splitlines()]
            records[0]["source_event_ids"] = ["art_000001"]
            frontier_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(any("unknown source_event_id" in error for error in report["errors"]))

    def test_replay_validates_frontier_finding_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            frontier_path = Path(result["run_dir"]) / "findings" / "frontier.jsonl"
            records = [json.loads(line) for line in frontier_path.read_text().splitlines()]
            records[0]["finding_id"] = "finding_missing"
            frontier_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(any("unknown finding_id" in error for error in report["errors"]))

    def test_replay_reconciles_summary_stage_counts_with_actual_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            (run_dir / "findings" / "findings.jsonl").write_text("", encoding="utf-8")
            (run_dir / "findings" / "frontier.jsonl").write_text("", encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("frontier_records mismatch" in error for error in report["errors"]))
            self.assertTrue(any("stage findings mismatch" in error for error in report["errors"]))

    def test_replay_rejects_tampered_frontier_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            frontier_path = run_dir / "findings" / "frontier.jsonl"
            records = [
                json.loads(line) for line in frontier_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["metric_value"] = 999999.0
            records[0]["promotion_reason"] = "tampered promotion"
            frontier_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
            )

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("output_hashes mismatch" in error for error in report["errors"]))

    def test_replay_validates_budget_source_event_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            ledger_path = Path(result["run_dir"]) / "budget_ledger.jsonl"
            records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            records[0]["source_event_ids"] = ["evt_missing"]
            ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            report = verify_run(Path(result["run_dir"]))
            self.assertFalse(report["success"])
            self.assertTrue(
                any(
                    "budget_ledger" in error and "unknown source_event_id" in error
                    for error in report["errors"]
                )
            )

    def test_replay_rejects_unconfined_artifact_payload_paths(self) -> None:
        for payload_path in (
            "/tmp/praxist_external_payload.txt",
            "../praxist_external_payload.txt",
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result = run_fake_workflow_fixture(
                    workspace=root,
                    runtime_ref="agent_runtime:fake_runtime",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    credential_profile="fake_multi_key",
                )
                run_dir = Path(result["run_dir"])
                external = root / "praxist_external_payload.txt"
                external.write_text("external payload", encoding="utf-8")
                index_path = run_dir / "artifact_index.jsonl"
                records = [json.loads(line) for line in index_path.read_text().splitlines()]
                records[0]["payload_path"] = payload_path
                records[0]["content_hash"] = sha256_bytes(external.read_bytes())
                index_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

                report = verify_run(run_dir)
                self.assertFalse(report["success"])
                self.assertTrue(
                    any("run-relative confined path" in error for error in report["errors"])
                )

    def test_replay_requires_artifact_hashes_and_provenance_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            index_path = run_dir / "artifact_index.jsonl"
            records = [json.loads(line) for line in index_path.read_text().splitlines()]
            records[0]["content_hash"] = ""
            index_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("invalid content_hash" in error for error in report["errors"]))

        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            frontier_path = run_dir / "findings" / "frontier.jsonl"
            frontier = [json.loads(line) for line in frontier_path.read_text().splitlines()]
            frontier[0]["source_artifact_ids"] = ["art_missing"]
            frontier[0]["artifact_refs"][0]["content_hash"] = "sha256:" + "0" * 64
            frontier_path.write_text("\n".join(json.dumps(record) for record in frontier) + "\n")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any("unknown source_artifact_id" in error for error in report["errors"])
            )
            self.assertTrue(any("content_hash mismatch" in error for error in report["errors"]))

        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            findings_path = run_dir / "findings" / "findings.jsonl"
            findings = [json.loads(line) for line in findings_path.read_text().splitlines()]
            findings[0]["evidence_refs"][0]["artifact_id"] = "art_missing"
            findings[0]["evidence_refs"][0]["content_hash"] = "sha256:" + "0" * 64
            findings_path.write_text("\n".join(json.dumps(record) for record in findings) + "\n")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("evidence_refs" in error for error in report["errors"]))

        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            metadata_path = run_dir / "artifacts" / "by_id" / "art_000001" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["api_key"] = "plainsecret123456"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("metadata.json" in error for error in report["errors"]))

        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            extra = run_dir / "artifacts" / "by_id" / "art_999999" / "payload.txt"
            extra.parent.mkdir(parents=True)
            extra.write_text("unindexed sk-test-redaction-000000\n", encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any("unindexed artifact file" in warning for warning in report["warnings"])
            )
            self.assertTrue(any("art_999999/payload.txt" in error for error in report["errors"]))

    def test_replay_rejects_stripped_plugin_resolution_and_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            resolution_path = run_dir / "plugin_resolution.json"
            resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
            resolution["selected"] = []
            resolution_path.write_text(json.dumps(resolution), encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any("selected must be a non-empty list" in error for error in report["errors"])
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            run_json_path = run_dir / "run.json"
            run_json = json.loads(run_json_path.read_text(encoding="utf-8"))
            del run_json["workspace_hash"]
            run_json_path.write_text(json.dumps(run_json), encoding="utf-8")

            report = verify_run(run_dir)
            self.assertTrue(report["success"])
            self.assertTrue(any("workspace_hash" in warning for warning in report["warnings"]))

            report = verify_run(run_dir, locked=True)
            self.assertFalse(report["success"])
            self.assertTrue(any("workspace_hash" in error for error in report["errors"]))

    def test_replay_requires_final_summary_and_finalized_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            (run_dir / "run_summary.json").unlink()
            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("run_summary.json" in error for error in report["errors"]))

        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            trajectory_path = run_dir / "trajectory.jsonl"
            records = [
                json.loads(line)
                for line in trajectory_path.read_text().splitlines()
                if json.loads(line).get("kind") != "run.finalized"
            ]
            for index, record in enumerate(records, start=1):
                record["seq"] = index
                record["event_id"] = f"evt_{index:06d}"
            trajectory_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("no run.finalized" in error for error in report["errors"]))

    def test_replay_rejects_stripped_plugin_dependency_closure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            resolution_path = run_dir / "plugin_resolution.json"
            resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
            resolution["selected"] = [
                item
                for item in resolution["selected"]
                if item["metadata"]["kind"] + ":" + item["metadata"]["name"]
                not in {"role:fake_pi", "panel_topology:fake_two_round"}
            ]
            resolution["dependency_edges"] = []
            resolution_path.write_text(json.dumps(resolution), encoding="utf-8")
            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any(
                    "missing required dependency" in error
                    or "missing required selected ref" in error
                    for error in report["errors"]
                ),
                report,
            )

    def test_replay_rejects_stripped_requested_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
                resolve_only=True,
            )
            run_dir = Path(result["run_dir"])
            resolution_path = run_dir / "plugin_resolution.json"
            resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
            stripped_refs = {
                "role:fake_peer",
                "role:fake_pi",
                "role:fake_chair",
                "audit_rule:fake_panel_audit",
                "evaluation:fake_pareto",
            }
            resolution["selected"] = [
                item
                for item in resolution["selected"]
                if item["metadata"]["kind"] + ":" + item["metadata"]["name"] not in stripped_refs
            ]
            resolution["dependency_edges"] = [
                edge
                for edge in resolution["dependency_edges"]
                if edge.get("to") not in stripped_refs
            ]
            resolution_path.write_text(json.dumps(resolution), encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("role:fake_peer" in error for error in report["errors"]))

    def test_replay_rejects_mutated_plugin_resolution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            resolution_path = run_dir / "plugin_resolution.json"
            resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
            for selected in resolution["selected"]:
                if (
                    selected["metadata"]["kind"] == "agent_runtime"
                    and selected["metadata"]["name"] == "fake_runtime"
                ):
                    selected["metadata"]["description"] = "mutated description"
                    break
            resolution_path.write_text(json.dumps(resolution), encoding="utf-8")

            report = verify_run(run_dir)
            self.assertTrue(report["success"])
            self.assertTrue(
                any("plugin metadata drift" in warning for warning in report["warnings"])
            )

            report = verify_run(run_dir, locked=True)
            self.assertFalse(report["success"])
            self.assertTrue(any("plugin metadata drift" in error for error in report["errors"]))

            report = verify_run(run_dir, locked=True, allow_plugin_drift=True)
            self.assertTrue(report["success"])
            self.assertTrue(
                any("plugin metadata drift" in warning for warning in report["warnings"])
            )

    def test_replay_rejects_plugin_source_spoofing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
                resolve_only=True,
            )
            run_dir = Path(result["run_dir"])
            resolution_path = run_dir / "plugin_resolution.json"
            resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
            resolution["selected"][0]["source"] = "project"
            resolution_path.write_text(json.dumps(resolution), encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any(
                    "only supports bundled or task_project plugins" in error
                    for error in report["errors"]
                )
            )

    def test_replay_binds_runtime_and_provider_refs_to_selected_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            trajectory_path = run_dir / "trajectory.jsonl"
            records = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
            for record in records:
                actor = record.get("actor") or {}
                if actor.get("type") == "agent_runtime":
                    actor["id"] = "agent_runtime:unselected_runtime"
                if actor.get("type") == "model_provider":
                    actor["id"] = "model_provider:unselected_provider"
                payload = record.get("payload") or {}
                request = payload.get("request") if isinstance(payload, dict) else None
                if isinstance(request, dict):
                    request["agent_runtime_ref"] = "agent_runtime:unselected_runtime"
                    if isinstance(request.get("model_call"), dict):
                        request["model_call"]["provider_ref"] = "model_provider:unselected_provider"
                if isinstance(payload, dict) and isinstance(payload.get("model_call"), dict):
                    payload["model_call"]["provider_ref"] = "model_provider:unselected_provider"
                if isinstance(payload, dict) and "provider_ref" in payload:
                    payload["provider_ref"] = "model_provider:unselected_provider"
            trajectory_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("agent_runtime actor" in error for error in report["errors"]))
            self.assertTrue(any("model_provider actor" in error for error in report["errors"]))

    def test_fake_panel_requires_credentials_for_non_fake_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ, {"OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": ""}, clear=False
            ):
                with self.assertRaisesRegex(
                    ValueError, "requires a matching active model provider credential"
                ):
                    run_fake_workflow_fixture(
                        workspace=Path(tmp),
                        runtime_ref="agent_runtime:claude_sdk",
                        model_provider_ref="model_provider:openrouter",
                        budget_policy_ref="budget_policy:fake_tiered",
                    )

    def test_fake_panel_resolve_only_does_not_require_credentials(self) -> None:
        """Issue #86: --resolve-only is documented as no-LLM-call. It must

        succeed against environments without provider credentials so smoke
        tests / CI can verify task contracts without secrets.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ, {"OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": ""}, clear=False
            ):
                result = run_fake_workflow_fixture(
                    workspace=Path(tmp),
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:openrouter",
                    budget_policy_ref="budget_policy:fake_tiered",
                    resolve_only=True,
                )
            self.assertEqual(result.get("status"), "resolved")

    def test_replay_rejects_tampered_model_provider_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_tampered_credentials"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                budget_policy_ref="budget_policy:default_basic",
                model="anthropic/claude-opus-4.7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            finalize_research_loop_plugin_run(prepared, success=False, error="startup test failure")
            self.assertTrue(verify_run(run_dir)["success"])

            credentials_path = run_dir / "credentials_redacted.json"
            credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
            credentials["credential_profiles"] = []
            write_json(credentials_path, credentials)

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any(
                    "missing active credential for selected model provider" in error
                    for error in report["errors"]
                )
            )

    def test_replay_binds_model_profiles_to_selected_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
                resolve_only=True,
            )
            run_dir = Path(result["run_dir"])
            profiles_path = run_dir / "model_profiles.json"
            profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
            profiles["runtime_ref"] = "agent_runtime:unselected_runtime"
            profiles["provider_adapters"] = {"model_provider:unselected_provider": "fake"}
            for profile in profiles["profiles"].values():
                profile["provider_ref"] = "model_provider:unselected_provider"
            profiles_path.write_text(json.dumps(profiles), encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(
                any("model_profiles runtime_ref" in error for error in report["errors"])
            )
            self.assertTrue(
                any("provider_ref is not selected" in error for error in report["errors"])
            )

    def test_replay_rejects_static_builtin_plugin_resolution_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
                resolve_only=True,
            )
            run_dir = Path(result["run_dir"])
            static_manifest = static_resolution_manifest(
                result["run_id"],
                "task:fake_panel",
                [
                    "task:fake_panel",
                    "workflow_stage:research_loop",
                    "agent_runtime:fake_runtime",
                    "model_provider:fake_provider",
                    "budget_policy:fake_tiered",
                ],
            )
            (run_dir / "plugin_resolution.json").write_text(
                json.dumps(static_manifest), encoding="utf-8"
            )

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("algorithm_version" in error for error in report["errors"]))
            self.assertTrue(
                any("not replay-verifiable" in warning for warning in report["warnings"])
            )

    def test_optional_dependency_absence_does_not_emit_dangling_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workflow_manifest(
                root,
                "stage_with_missing_optional",
                [
                    {
                        "kind": "workflow_stage",
                        "name": "stage_optional_missing",
                        "version": ">=0.1,<1.0",
                        "required": False,
                    }
                ],
            )
            loader = PluginLoader(PluginRoots(bundled=[root], user=[], project=[]))
            manifest = loader.resolve(
                ["workflow_stage:stage_with_missing_optional"],
                root_task_ref="workflow_stage:stage_with_missing_optional",
            )
            self.assertEqual(manifest["dependency_edges"], [])
            self.assertEqual(
                [item["metadata"]["name"] for item in manifest["selected"]],
                ["stage_with_missing_optional"],
            )

    def test_source_snapshot_includes_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "praxist" / "config.py"
            config.parent.mkdir(parents=True)
            config.write_text("AGENT_MODEL = 'a'\n", encoding="utf-8")
            before = build_core_source_snapshot(root)["workspace_hash"]
            config.write_text("AGENT_MODEL = 'b'\n", encoding="utf-8")
            after = build_core_source_snapshot(root)["workspace_hash"]
            self.assertNotEqual(before, after)

    def test_source_snapshot_includes_execution_prompts_and_private_kb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = (
                root
                / "praxist"
                / "plugins"
                / "workflow_stages"
                / "research_loop"
                / "backend"
                / "prompt_base.jinja2"
            )
            private_kb = (
                root
                / "praxist"
                / "plugins"
                / "workflow_stages"
                / "research_loop"
                / "backend"
                / "multi_pi"
                / "private_kb"
                / "builder"
                / "note.md"
            )
            prompt.parent.mkdir(parents=True)
            private_kb.parent.mkdir(parents=True)
            prompt.write_text("prompt v1\n", encoding="utf-8")
            private_kb.write_text("kb v1\n", encoding="utf-8")
            before = build_core_source_snapshot(root)["workspace_hash"]
            prompt.write_text("prompt v2\n", encoding="utf-8")
            private_kb.write_text("kb v2\n", encoding="utf-8")
            after = build_core_source_snapshot(root)["workspace_hash"]
            self.assertNotEqual(before, after)

    def test_fake_panel_resolve_only_does_not_emit_findings_or_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
                resolve_only=True,
            )
            run_dir = Path(result["run_dir"])
            self.assertEqual(result["status"], "resolved")
            self.assertEqual((run_dir / "findings" / "findings.jsonl").read_text(), "")
            self.assertEqual((run_dir / "findings" / "frontier.jsonl").read_text(), "")
            self.assertTrue(verify_run(run_dir)["success"])

    def test_execution_filters_project_shadowed_plugins_until_registry_backed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin_dir = root / ".praxist" / "plugins" / "budget_policies" / "fake_tiered"
            plugin_dir.mkdir(parents=True)
            plugin_dir.joinpath("plugin.yaml").write_text(
                """schema_version: 1
name: fake_tiered
kind: budget_policy
version: 0.1.0
protocol_version: 1
stability: v1_stable
description: Project shadow budget policy.
compatibility:
  praxist_core: ">=0.1.0,<1.0"
  python: ">=3.11"
dependencies: []
capabilities: []
code: []
assets: []
""",
                encoding="utf-8",
            )
            result = run_fake_workflow_fixture(
                workspace=root,
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            resolution = json.loads(
                (Path(result["run_dir"]) / "plugin_resolution.json").read_text(encoding="utf-8")
            )
            selected_budget = next(
                item
                for item in resolution["selected"]
                if item["metadata"]["kind"] == "budget_policy"
                and item["metadata"]["name"] == "fake_tiered"
            )
            self.assertEqual(resolution["execution_source_policy"], "bundled_only")
            self.assertEqual(selected_budget["source"], "bundled")
            self.assertTrue(
                any(
                    item["kind"] == "budget_policy"
                    and item["name"] == "fake_tiered"
                    and item["shadowed_source"] == "project"
                    for item in resolution["shadowed"]
                )
            )

    def test_sam_legacy_accepts_deepseek_direct_key_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(
                os.environ,
                _fixture_plugin_env({"DEEPSEEK_API_KEY": "sk-test-redaction-000000"}),
                clear=True,
            ):
                prepared = prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=root / "run_deepseek_direct",
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:deepseek_alias",
                    budget_policy_ref="budget_policy:default_basic",
                    model="",
                    local_mode=True,
                    frontier_strategy="mixed",
                    command="test",
                )
            self.assertEqual(prepared.model_provider_ref, "model_provider:deepseek_alias")
            self.assertEqual(
                prepared.startup_config["canonical_args"]["model"], "deepseek-v4-pro[1m]"
            )
            self.assertEqual(prepared.provider_env["DEEPSEEK_API_KEY"], "sk-test-redaction-000000")
            self.assertEqual(
                prepared.provider_env["ANTHROPIC_BASE_URL"],
                "https://api.deepseek.com/anthropic",
            )
            self.assertEqual(
                prepared.provider_env["ANTHROPIC_AUTH_TOKEN"], "sk-test-redaction-000000"
            )
            self.assertIsNone(prepared.provider_env["OPENROUTER_API_KEY"])

    def test_sam_legacy_requires_selected_provider_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, _fixture_plugin_env(), clear=True):
                with self.assertRaisesRegex(ValueError, "model_provider:openrouter requires"):
                    prepare_research_loop_plugin_run(
                        task_project_path=_toy_task_path(),
                        workspace=Path(tmp),
                        run_dir=Path(tmp) / "run_no_creds",
                        runtime_ref="agent_runtime:claude_sdk",
                        model_provider_ref="model_provider:openrouter",
                        budget_policy_ref="budget_policy:default_basic",
                        model="anthropic/claude-opus-4.7",
                        local_mode=True,
                        frontier_strategy="mixed",
                        command="test",
                    )
            with patch.dict(
                os.environ,
                _fixture_plugin_env(
                    {
                        "ANTHROPIC_AUTH_TOKEN": "proxy-token-not-openrouter",
                        "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1",
                    }
                ),
                clear=True,
            ):
                with self.assertRaisesRegex(
                    ValueError, "model_provider:openrouter requires a matching active"
                ):
                    prepare_research_loop_plugin_run(
                        task_project_path=_toy_task_path(),
                        workspace=Path(tmp),
                        run_dir=Path(tmp) / "run_proxy_wrong_provider",
                        runtime_ref="agent_runtime:claude_sdk",
                        model_provider_ref="model_provider:openrouter",
                        budget_policy_ref="budget_policy:default_basic",
                        model="anthropic/claude-opus-4.7",
                        local_mode=True,
                        frontier_strategy="mixed",
                        command="test",
                    )

    def test_sam_legacy_rejects_provider_incompatible_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "not compatible"):
                prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=Path(tmp),
                    run_dir=Path(tmp) / "run_bad_openrouter_model",
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:openrouter",
                    budget_policy_ref="budget_policy:default_basic",
                    model="claude-opus-4-7",
                    local_mode=True,
                    frontier_strategy="mixed",
                    command="test",
                )
            with self.assertRaisesRegex(ValueError, "not compatible"):
                prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=Path(tmp),
                    run_dir=Path(tmp) / "run_bad_anthropic_model",
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:anthropic_messages",
                    budget_policy_ref="budget_policy:default_basic",
                    model="gpt-5.2",
                    local_mode=True,
                    frontier_strategy="mixed",
                    command="test",
                )

    def test_legacy_stage_resolve_only_records_budget_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_legacy_usage"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                budget_policy_ref="budget_policy:default_basic",
                model="anthropic/claude-opus-4.7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            stage_result = asyncio.run(
                ResearchLoopStage().execute(
                    ResearchLoopStageContext(
                        task_spec=prepared.task_spec,
                        workspace=root,
                        run_dir=run_dir,
                        local_mode=True,
                        model="anthropic/claude-opus-4.7",
                        model_provider_ref=prepared.model_provider_ref,
                        frontier_strategy="mixed",
                        budget_grant_id=prepared.stage_budget_grant_id,
                        resolve_only=True,
                    )
                )
            )
            finalize_research_loop_plugin_run(
                prepared, success=stage_result.success, result=stage_result.summary
            )
            records = [
                json.loads(line)
                for line in (run_dir / "budget_ledger.jsonl").read_text().splitlines()
            ]
            self.assertIn("usage", [record["kind"] for record in records])
            self.assertTrue(verify_run(run_dir)["success"])

    def test_legacy_stage_applies_openrouter_provider_env_during_execution(self) -> None:
        observed: dict[str, str | None] = {}

        async def fake_generation_run(self):
            observed["base_url"] = os.environ.get("ANTHROPIC_BASE_URL")
            observed["auth_token"] = os.environ.get("ANTHROPIC_AUTH_TOKEN")
            observed["openrouter_key"] = os.environ.get("OPENROUTER_API_KEY")
            return {
                "generations_completed": 1,
                "run_dir": str(self.run_dir),
                "exit_condition": "completed",
                "total_duration_seconds": 0.0,
                "frontier_summary": [],
                "usage": {"tokens": 1000.0, "gpu_hours": 0.01},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_provider_env"
            with patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "sk-test-redaction-000000",
                    "ANTHROPIC_BASE_URL": "",
                    "ANTHROPIC_AUTH_TOKEN": "",
                },
                clear=False,
            ):
                prepared = prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:openrouter",
                    budget_policy_ref="budget_policy:default_basic",
                    model="anthropic/claude-opus-4.7",
                    local_mode=True,
                    frontier_strategy="mixed",
                    command="test",
                )
                with patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop.run",
                    fake_generation_run,
                ):
                    stage_result = asyncio.run(
                        ResearchLoopStage().execute(
                            ResearchLoopStageContext(
                                task_spec=prepared.task_spec,
                                workspace=root,
                                run_dir=run_dir,
                                local_mode=True,
                                model="anthropic/claude-opus-4.7",
                                model_provider_ref=prepared.model_provider_ref,
                                frontier_strategy="mixed",
                                budget_grant_id=prepared.stage_budget_grant_id,
                                resolve_only=False,
                            )
                        )
                    )
                finalize_research_loop_plugin_run(
                    prepared, success=stage_result.success, result=stage_result.summary
                )

            self.assertEqual(observed["base_url"], "https://openrouter.ai/api")
            self.assertEqual(observed["auth_token"], "sk-test-redaction-000000")
            self.assertEqual(observed["openrouter_key"], "sk-test-redaction-000000")
            self.assertNotEqual(os.environ.get("ANTHROPIC_AUTH_TOKEN"), "sk-test-redaction-000000")

    def test_legacy_stage_uses_startup_frozen_provider_env(self) -> None:
        observed: dict[str, str | None] = {}

        async def fake_generation_run(self):
            observed["base_url"] = os.environ.get("ANTHROPIC_BASE_URL")
            observed["auth_token"] = os.environ.get("ANTHROPIC_AUTH_TOKEN")
            observed["openrouter_key"] = os.environ.get("OPENROUTER_API_KEY")
            return {
                "generations_completed": 0,
                "run_dir": str(self.run_dir),
                "exit_condition": "completed",
                "frontier_summary": [],
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_frozen_provider_env"
            with patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "or-startup-key",
                    "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1",
                    "ANTHROPIC_AUTH_TOKEN": "or-bashrc-compatible-token",
                },
                clear=False,
            ):
                prepared = prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:openrouter",
                    budget_policy_ref="budget_policy:default_basic",
                    model="anthropic/claude-opus-4.7",
                    local_mode=True,
                    frontier_strategy="mixed",
                    command="test",
                )
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-mutated-key"}, clear=False):
                with patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop.run",
                    fake_generation_run,
                ):
                    stage_result = asyncio.run(
                        ResearchLoopStage().execute(
                            ResearchLoopStageContext(
                                task_spec=prepared.task_spec,
                                workspace=root,
                                run_dir=run_dir,
                                local_mode=True,
                                model=prepared.startup_config["canonical_args"]["model"],
                                model_provider_ref=prepared.model_provider_ref,
                                frontier_strategy="mixed",
                                budget_grant_id=prepared.stage_budget_grant_id,
                                model_provider_credential_key_id=prepared.model_provider_credential_key_id,
                                provider_env=prepared.provider_env,
                                resolve_only=False,
                            )
                        )
                    )
            self.assertTrue(stage_result.success)
            self.assertEqual(observed["base_url"], "https://openrouter.ai/api")
            self.assertEqual(observed["auth_token"], "or-bashrc-compatible-token")
            self.assertEqual(observed["openrouter_key"], "or-bashrc-compatible-token")

    def test_legacy_runtime_scopes_env_to_selected_provider(self) -> None:
        observed: dict[str, str] = {}

        async def fake_execute_legacy(self, task, options):
            observed.update(options.env or {})
            return LegacyAgentResult(success=True, output={}, duration=0.0, iteration_count=0)

        with patch.object(ClaudeSdkAgentRuntime, "execute_legacy", fake_execute_legacy):
            with patch.dict(
                os.environ,
                {
                    "PRAXIST_MODEL_PROVIDER_REF": "model_provider:openrouter",
                    "ANTHROPIC_API_KEY": "anthropic-key-should-not-pass",
                    "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1",
                    "ANTHROPIC_AUTH_TOKEN": "openrouter-token",
                    "OPENROUTER_API_KEY": "openrouter-native-token",
                    "PRAXIST_RUN_DIR": "/tmp/praxist-run",
                    "PRAXIST_TASK_PROJECT_PATH": "/tmp/task-project",
                    "PRAXIST_DATA_DIR": "/tmp/data/sam_optimizer",
                    "PYTHONPATH": "/tmp/praxist-system",
                },
                clear=False,
            ):
                result = asyncio.run(
                    BaseAgent(
                        name="test",
                        allowed_tools=[],
                        workspace=Path.cwd(),
                        mcp_servers={},
                    ).execute("noop")
                )

        self.assertTrue(result.success)
        self.assertEqual(observed["ANTHROPIC_BASE_URL"], "https://openrouter.ai/api")
        self.assertEqual(observed["ANTHROPIC_AUTH_TOKEN"], "openrouter-token")
        self.assertEqual(observed["OPENROUTER_API_KEY"], "openrouter-native-token")
        self.assertEqual(observed["PRAXIST_RUN_DIR"], "/tmp/praxist-run")
        self.assertEqual(observed["PRAXIST_TASK_PROJECT_PATH"], "/tmp/task-project")
        self.assertEqual(observed["PRAXIST_DATA_DIR"], "/tmp/data/sam_optimizer")
        self.assertEqual(observed["PYTHONPATH"], "/tmp/praxist-system")
        self.assertNotIn("ANTHROPIC_API_KEY", observed)

    def test_task_runtime_env_makes_external_task_subprocesses_cwd_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "external_task"
            task_path.mkdir()
            data_dir = root / "praxist" / "data" / "sam_optimizer"
            data_dir.mkdir(parents=True)

            env = _task_runtime_env(
                task_project_path=task_path,
                workspace=root / "praxist",
                task_id="sam_optimizer",
                env={"PYTHONPATH": "/opt/existing"},
                task_descriptor={
                    "runtime_environment": {
                        "data_env_aliases": ["PRAXIST_SAM_DATA_DIR", "SAM_DATA_DIR"],
                    }
                },
            )

        self.assertEqual(env["PRAXIST_TASK_PROJECT_PATH"], str(task_path.resolve()))
        self.assertEqual(env["PRAXIST_WORKSPACE_ROOT"], str((root / "praxist").resolve()))
        self.assertEqual(env["PRAXIST_DATA_DIR"], str(data_dir.resolve()))
        self.assertEqual(env["PRAXIST_SAM_DATA_DIR"], str(data_dir.resolve()))
        self.assertEqual(env["PRAXIST_DATASETS_DIR"], str((root / "praxist" / "data").resolve()))
        self.assertEqual(
            env["PYTHONPATH"], f"{(root / 'praxist').resolve()}{os.pathsep}/opt/existing"
        )

    def test_legacy_base_agent_records_selected_credential_key_id(self) -> None:
        async def fake_execute_legacy(self, task, options):
            return LegacyAgentResult(success=True, output={}, duration=0.0, iteration_count=0)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_agent_provenance"
            run_dir.mkdir()
            with patch.object(ClaudeSdkAgentRuntime, "execute_legacy", fake_execute_legacy):
                with patch.dict(
                    os.environ,
                    {
                        "PRAXIST_RUN_DIR": str(run_dir),
                        "PRAXIST_RUN_ID": run_dir.name,
                        "PRAXIST_MODEL_PROVIDER_REF": "model_provider:openrouter",
                        "PRAXIST_MODEL_CREDENTIAL_KEY_ID": "openrouter:env:abc123",
                        "PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:claude_sdk",
                        "PRAXIST_BUDGET_GRANT_ID": "grant_test",
                        "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1",
                        "ANTHROPIC_AUTH_TOKEN": "openrouter-token",
                    },
                    clear=False,
                ):
                    asyncio.run(
                        BaseAgent(
                            name="test",
                            allowed_tools=[],
                            workspace=Path.cwd(),
                            mcp_servers={},
                            model="anthropic/claude-opus-4.7",
                        ).execute("noop")
                    )

            events = [
                json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()
            ]
            finished = next(event for event in events if event["kind"] == "agent.run_finished")
            self.assertEqual(
                finished["payload"]["model_call"]["credential_ref"], "openrouter:env:abc123"
            )
            self.assertEqual(finished["payload"]["budget_grant_id"], "grant_test")

    def test_legacy_finalize_materializes_outputs_and_replay_counts(self) -> None:
        async def fake_generation_run(self):
            TrajectoryWriter(self.run_dir, self.run_dir.name).emit(
                "agent.run_finished",
                scope={"stage_id": "research_loop", "agent_name": "gen0_peer0-session0"},
                actor={"type": "agent_runtime", "id": "agent_runtime:claude_sdk"},
                payload={
                    "success": True,
                    "agent_runtime_ref": "agent_runtime:claude_sdk",
                    "model_call": {
                        "provider_ref": prepared.model_provider_ref,
                        "model": self.model,
                        "credential_ref": prepared.model_provider_credential_key_id,
                    },
                    "budget_grant_id": prepared.stage_budget_grant_id,
                    "output_summary": {
                        "tool_uses": [
                            {
                                "tool": "mcp__evaluation-tools__share_finding",
                                "input": {
                                    "finding_type": "result",
                                    "title": "legacy promoted finding",
                                    "metrics": json.dumps({"test_accuracy_cifar100": 0.42}),
                                    "variant_name": "legacy_variant",
                                    "peer_id": "gen0_peer0",
                                },
                            }
                        ]
                    },
                },
            )
            shared = self.run_dir / "shared_findings"
            shared.mkdir(parents=True, exist_ok=True)
            (shared / "legacy_f1_result.json").write_text(
                json.dumps(
                    {
                        "id": "legacy_f1",
                        "finding_type": "result",
                        "title": "legacy promoted finding",
                        "metrics": {"test_accuracy_cifar100": 0.42},
                        "variant_name": "legacy_variant",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "timestamp": "2026-05-09T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            return {
                "generations_completed": 1,
                "run_dir": str(self.run_dir),
                "exit_condition": "completed",
                "total_duration_seconds": 1.0,
                "usage": {"tokens": 1000.0, "gpu_hours": 0.01},
                "frontier_summary": [
                    {
                        "finding_id": "legacy_f1",
                        "variant_name": "legacy_variant",
                        "metric_value": 0.42,
                        "metrics": {"test_accuracy_cifar100": 0.42},
                        "promoted_at": "2026-05-09T00:01:00Z",
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_materialize"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:anthropic_messages",
                budget_policy_ref="budget_policy:default_basic",
                model="claude-opus-4-6",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop.run",
                fake_generation_run,
            ):
                stage_result = asyncio.run(
                    ResearchLoopStage().execute(
                        ResearchLoopStageContext(
                            task_spec=prepared.task_spec,
                            workspace=root,
                            run_dir=run_dir,
                            local_mode=True,
                            model=prepared.startup_config["canonical_args"]["model"],
                            model_provider_ref=prepared.model_provider_ref,
                            frontier_strategy="mixed",
                            budget_grant_id=prepared.stage_budget_grant_id,
                            provider_env=prepared.provider_env,
                            resolve_only=False,
                        )
                    )
                )
            finalize_research_loop_plugin_run(
                prepared, success=stage_result.success, result=stage_result.summary
            )

            findings = [
                json.loads(line)
                for line in (run_dir / "findings" / "findings.jsonl").read_text().splitlines()
            ]
            frontier = [
                json.loads(line)
                for line in (run_dir / "findings" / "frontier.jsonl").read_text().splitlines()
            ]
            self.assertEqual([record["finding_id"] for record in findings], ["legacy_f1"])
            self.assertEqual([record["finding_id"] for record in frontier], ["legacy_f1"])
            self.assertTrue(findings[0]["evidence_refs"])
            self.assertTrue(findings[0]["source_event_ids"])
            self.assertTrue(frontier[0]["artifact_refs"])
            self.assertTrue(frontier[0]["source_artifact_ids"])
            self.assertTrue(frontier[0]["source_event_ids"])
            self.assertTrue(verify_run(run_dir)["success"])

    def test_legacy_finalize_imports_same_peer_finding_without_tool_provenance_as_weak(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_forged_same_peer"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:anthropic_messages",
                budget_policy_ref="budget_policy:default_basic",
                model="claude-opus-4-7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            TrajectoryWriter(run_dir, run_dir.name).emit(
                "agent.run_finished",
                scope={"stage_id": "research_loop", "agent_name": "gen0_peer0-session0"},
                actor={"type": "agent_runtime", "id": "agent_runtime:claude_sdk"},
                payload={
                    "success": True,
                    "agent_runtime_ref": "agent_runtime:claude_sdk",
                    "model_call": {
                        "provider_ref": prepared.model_provider_ref,
                        "model": "claude-opus-4-7",
                        "credential_ref": prepared.model_provider_credential_key_id,
                    },
                    "budget_grant_id": prepared.stage_budget_grant_id,
                    "output_summary": {"tool_uses": []},
                },
            )
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True, exist_ok=True)
            (shared / "forged.json").write_text(
                json.dumps(
                    {
                        "id": "forged_finding",
                        "finding_type": "result",
                        "title": "forged promoted finding",
                        "peer_id": "gen0_peer0",
                        "metrics": {"mean_test_accuracy": 999.0},
                    }
                ),
                encoding="utf-8",
            )
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 1,
                    "run_dir": str(run_dir),
                    "exit_condition": "completed",
                    "frontier_summary": [{"finding_id": "forged_finding", "metric_value": 999.0}],
                },
            )
            findings = [
                json.loads(line)
                for line in (run_dir / "findings" / "findings.jsonl").read_text().splitlines()
            ]
            self.assertEqual(findings[0]["provenance_quality"], "legacy_weak")
            report = verify_run(run_dir)
            self.assertTrue(report["success"])
            self.assertTrue(
                any("imported legacy provenance" in warning for warning in report["warnings"])
            )

    def test_cli_preserves_run_when_success_materialization_has_weak_provenance(self) -> None:
        from praxist.run import cmd_run

        async def fake_generation_run(self):
            shared = self.run_dir / "shared_findings"
            shared.mkdir(parents=True, exist_ok=True)
            (shared / "unproven.json").write_text(
                json.dumps(
                    {
                        "id": "unproven_finding",
                        "finding_type": "result",
                        "title": "unproven finding",
                        "peer_id": "gen0_peer0",
                        "metrics": {"mean_test_accuracy": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            return {
                "generations_completed": 1,
                "run_dir": str(self.run_dir),
                "exit_condition": "completed",
                "frontier_summary": [{"finding_id": "unproven_finding", "metric_value": 1.0}],
                "usage": {"tokens": 1000.0, "gpu_hours": 0.01},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_finalize_failure"
            task_path = _write_research_task(root / "task")
            args = SimpleNamespace(
                fake=False,
                task_path=str(task_path),
                task="",
                task_spec="",
                workspace=str(root),
                model="fake-deterministic",
                runtime="agent_runtime:fake_runtime",
                model_provider="model_provider:fake_provider",
                budget_policy="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
                run_dir=str(run_dir),
                resolve_only=False,
                local=True,
                frontier_strategy="mixed",
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop.run",
                fake_generation_run,
            ):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    cmd_run(args)
            unexpected_stderr = "\n".join(
                line
                for line in stderr.getvalue().splitlines()
                if not (
                    line.startswith("MCP ")
                    and "server unavailable: ImportError: claude_agent_sdk is required" in line
                )
            )
            self.assertEqual("", unexpected_stderr)
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            trajectory = [
                json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()
            ]
            findings = [
                json.loads(line)
                for line in (run_dir / "findings" / "findings.jsonl").read_text().splitlines()
            ]
            self.assertEqual(summary["status"], "succeeded")
            self.assertEqual(run_json["status"], "succeeded")
            self.assertEqual(trajectory[-1]["kind"], "run.finalized")
            self.assertEqual(findings[0]["provenance_quality"], "legacy_weak")
            self.assertTrue(verify_run(run_dir)["success"])

    def test_legacy_finalize_imports_outputs_without_agent_provenance_as_weak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_injected"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:anthropic_messages",
                budget_policy_ref="budget_policy:default_basic",
                model="claude-opus-4-7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True, exist_ok=True)
            (shared / "fake.json").write_text(
                json.dumps({"id": "injected_finding", "metrics": {"mean_test_accuracy": 999.0}}),
                encoding="utf-8",
            )
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 1,
                    "run_dir": str(run_dir),
                    "exit_condition": "completed",
                    "frontier_summary": [{"finding_id": "injected_finding", "metric_value": 999.0}],
                },
            )
            findings = [
                json.loads(line)
                for line in (run_dir / "findings" / "findings.jsonl").read_text().splitlines()
            ]
            self.assertEqual(findings[0]["provenance_quality"], "legacy_weak")
            report = verify_run(run_dir)
            self.assertTrue(report["success"])
            self.assertTrue(
                any("imported legacy provenance" in warning for warning in report["warnings"])
            )

    def test_legacy_finalize_imports_outputs_with_unrelated_agent_event_as_weak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_unrelated_agent"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:anthropic_messages",
                budget_policy_ref="budget_policy:default_basic",
                model="claude-opus-4-7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            TrajectoryWriter(run_dir, run_dir.name).emit(
                "agent.run_finished",
                scope={"stage_id": "research_loop", "agent_name": "other_peer-session0"},
                actor={"type": "agent_runtime", "id": "agent_runtime:claude_sdk"},
                payload={"success": True},
            )
            shared = run_dir / "shared_findings"
            shared.mkdir(parents=True, exist_ok=True)
            (shared / "evil.json").write_text(
                json.dumps(
                    {
                        "id": "evil_finding",
                        "peer_id": "gen0_peer0",
                        "metrics": {"mean_test_accuracy": 999.0},
                    }
                ),
                encoding="utf-8",
            )
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 1,
                    "run_dir": str(run_dir),
                    "exit_condition": "completed",
                    "frontier_summary": [{"finding_id": "evil_finding", "metric_value": 999.0}],
                },
            )
            findings = [
                json.loads(line)
                for line in (run_dir / "findings" / "findings.jsonl").read_text().splitlines()
            ]
            self.assertEqual(findings[0]["provenance_quality"], "legacy_weak")
            report = verify_run(run_dir)
            self.assertTrue(report["success"])
            self.assertTrue(
                any("imported legacy provenance" in warning for warning in report["warnings"])
            )
            self.assertTrue(
                any("missing replayable model_call" in warning for warning in report["warnings"])
            )
            self.assertTrue(
                any("missing budget_grant_id" in warning for warning in report["warnings"])
            )
            locked_report = verify_run(run_dir, locked=True)
            self.assertFalse(locked_report["success"])
            self.assertTrue(
                any("missing replayable model_call" in error for error in locked_report["errors"])
            )

    def test_legacy_startup_model_profiles_match_effective_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_model_profile"
            prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                budget_policy_ref="budget_policy:default_basic",
                model="anthropic/claude-opus-4.6",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )

            startup = json.loads((run_dir / "startup_config.json").read_text(encoding="utf-8"))
            profiles = json.loads((run_dir / "model_profiles.json").read_text(encoding="utf-8"))
            self.assertEqual(startup["canonical_args"]["model"], "anthropic/claude-opus-4.6")
            self.assertEqual(
                {profile["model"] for profile in profiles["profiles"].values()},
                {"anthropic/claude-opus-4.6"},
            )

    def test_legacy_startup_applies_generation_policy_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_generation_overrides"
            with patch.dict(os.environ, {"MAX_GENERATIONS": "1", "COHORT_SIZE": "1"}, clear=False):
                prepared = prepare_research_loop_plugin_run(
                    task_project_path=_toy_task_path(),
                    workspace=root,
                    run_dir=run_dir,
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:anthropic_messages",
                    budget_policy_ref="budget_policy:default_basic",
                    model="claude-opus-4-7",
                    local_mode=True,
                    frontier_strategy="mixed",
                    command="test",
                )

            self.assertEqual(prepared.task_spec.generation_policy.max_generations, 1)
            self.assertEqual(prepared.task_spec.generation_policy.cohort_size, 1)
            startup = json.loads((run_dir / "startup_config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {item["env"]: item["value"] for item in startup["env_overrides_seen"]},
                {"MAX_GENERATIONS": 1, "COHORT_SIZE": 1},
            )
            effective = yaml.safe_load(
                (run_dir / "effective_task_spec.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(effective["generation_policy"]["max_generations"], 1)
            self.assertEqual(effective["generation_policy"]["cohort_size"], 1)
            records = [
                json.loads(line)
                for line in (run_dir / "budget_ledger.jsonl").read_text().splitlines()
            ]
            request = next(record for record in records if record["kind"] == "request")
            self.assertEqual(request["requested_budget"]["gpu_hours"], 0.0)

    def test_legacy_stage_budget_requests_and_records_cost_units(self) -> None:
        async def fake_generation_run(self):
            return {
                "generations_completed": 1,
                "run_dir": str(self.run_dir),
                "exit_condition": "completed",
                "total_duration_seconds": 1.0,
                "frontier_summary": [],
                "usage": {"tokens": 1000.0, "gpu_hours": 0.01},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_budget_units"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:anthropic_messages",
                budget_policy_ref="budget_policy:default_basic",
                model="claude-opus-4-7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            records = [
                json.loads(line)
                for line in (run_dir / "budget_ledger.jsonl").read_text().splitlines()
            ]
            request_record = next(record for record in records if record["kind"] == "request")
            decision_record = next(record for record in records if record["kind"] == "decision")
            self.assertEqual(
                set(request_record["requested_budget"]),
                {"tokens", "wall_clock_seconds", "gpu_hours"},
            )
            self.assertEqual(decision_record["decision"], "grant")
            self.assertEqual(decision_record["granted_budget"], request_record["requested_budget"])

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop.run",
                fake_generation_run,
            ):
                stage_result = asyncio.run(
                    ResearchLoopStage().execute(
                        ResearchLoopStageContext(
                            task_spec=prepared.task_spec,
                            workspace=root,
                            run_dir=run_dir,
                            local_mode=True,
                            model="claude-opus-4-7",
                            model_provider_ref=prepared.model_provider_ref,
                            frontier_strategy="mixed",
                            budget_grant_id=prepared.stage_budget_grant_id,
                            resolve_only=False,
                        )
                    )
                )
            finalize_research_loop_plugin_run(
                prepared, success=stage_result.success, result=stage_result.summary
            )
            records = [
                json.loads(line)
                for line in (run_dir / "budget_ledger.jsonl").read_text().splitlines()
            ]
            decision = next(record for record in records if record["kind"] == "decision")
            usage = next(record for record in records if record["kind"] == "usage")
            self.assertEqual(set(usage["actual_usage"]), set(decision["granted_budget"]))
            self.assertEqual(usage["actual_usage"]["tokens"], 1000.0)
            self.assertEqual(usage["actual_usage"]["gpu_hours"], 0.01)
            self.assertGreaterEqual(usage["actual_usage"]["wall_clock_seconds"], 0.0)
            self.assertTrue(verify_run(run_dir)["success"])

    def test_legacy_stage_records_usage_unknown_without_failing_executed_run(self) -> None:
        async def fake_generation_run(self):
            return {
                "generations_completed": 1,
                "run_dir": str(self.run_dir),
                "exit_condition": "completed",
                "frontier_summary": [],
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_missing_usage_measurement"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:anthropic_messages",
                budget_policy_ref="budget_policy:default_basic",
                model="claude-opus-4-7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop.run",
                fake_generation_run,
            ):
                stage_result = asyncio.run(
                    ResearchLoopStage().execute(
                        ResearchLoopStageContext(
                            task_spec=prepared.task_spec,
                            workspace=root,
                            run_dir=run_dir,
                            local_mode=True,
                            model="claude-opus-4-7",
                            model_provider_ref=prepared.model_provider_ref,
                            frontier_strategy="mixed",
                            budget_grant_id=prepared.stage_budget_grant_id,
                            resolve_only=False,
                        )
                    )
                )
            self.assertTrue(stage_result.success)
            self.assertEqual(set(stage_result.summary["usage_unknown_units"]), {"tokens"})
            finalize_research_loop_plugin_run(
                prepared, success=stage_result.success, result=stage_result.summary
            )
            records = [
                json.loads(line)
                for line in (run_dir / "budget_ledger.jsonl").read_text().splitlines()
            ]
            usage_unknown = next(record for record in records if record["kind"] == "usage_unknown")
            self.assertEqual(usage_unknown["unknown_units"], ["tokens"])
            report = verify_run(run_dir)
            self.assertTrue(report["success"])
            self.assertTrue(any("usage_unknown" in warning for warning in report["warnings"]))

    def test_legacy_stage_uses_context_runtime_ref_for_provenance_env(self) -> None:
        async def fake_generation_run(self):
            return {
                "generations_completed": 0,
                "run_dir": str(self.run_dir),
                "exit_condition": "completed",
                "observed_runtime_ref": os.environ.get("PRAXIST_AGENT_RUNTIME_REF"),
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_runtime_env"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:anthropic_messages",
                budget_policy_ref="budget_policy:default_basic",
                model="claude-opus-4-7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop.run",
                fake_generation_run,
            ):
                stage_result = asyncio.run(
                    ResearchLoopStage().execute(
                        ResearchLoopStageContext(
                            task_spec=prepared.task_spec,
                            workspace=root,
                            run_dir=run_dir,
                            local_mode=True,
                            model="claude-opus-4-7",
                            model_provider_ref=prepared.model_provider_ref,
                            frontier_strategy="mixed",
                            budget_grant_id=prepared.stage_budget_grant_id,
                            runtime_ref="agent_runtime:custom_test",
                            resolve_only=False,
                        )
                    )
                )
            self.assertTrue(stage_result.success)
            self.assertEqual(
                stage_result.summary["observed_runtime_ref"], "agent_runtime:custom_test"
            )

    def test_legacy_stage_budget_shortfall_requires_review_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_budget_shortfall"
            task_path = _write_research_task(root / "task", per_experiment_gpu_hours=3.0)
            prepared = prepare_research_loop_plugin_run(
                task_project_path=task_path,
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:anthropic_messages",
                budget_policy_ref="budget_policy:fake_tiered",
                model="claude-opus-4-7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            self.assertIsNone(prepared.stage_budget_grant_id)
            records = [
                json.loads(line)
                for line in (run_dir / "budget_ledger.jsonl").read_text().splitlines()
            ]
            decision = next(record for record in records if record["kind"] == "decision")
            self.assertEqual(decision["decision"], "require_review")
            self.assertIsNone(decision["granted_budget"])

    def test_cli_budget_gate_failure_uses_budget_exit_code(self) -> None:
        from praxist.run import cmd_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_budget_gate_failure"
            task_path = _write_research_task(root / "task", per_experiment_gpu_hours=3.0)
            args = SimpleNamespace(
                fake=False,
                task_path=str(task_path),
                task="",
                task_spec="",
                workspace=str(root),
                model="fake-deterministic",
                runtime="agent_runtime:fake_runtime",
                model_provider="model_provider:fake_provider",
                budget_policy="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
                run_dir=str(run_dir),
                resolve_only=False,
                local=True,
                frontier_strategy="mixed",
            )
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    cmd_run(args)
            self.assertEqual(raised.exception.code, 5)
            self.assertIn("budget gate did not grant", stderr.getvalue())
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["exit_code"], 5)

    def test_replay_requires_usage_for_executed_legacy_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_missing_usage"
            prepared = prepare_research_loop_plugin_run(
                task_project_path=_toy_task_path(),
                workspace=root,
                run_dir=run_dir,
                runtime_ref="agent_runtime:claude_sdk",
                model_provider_ref="model_provider:openrouter",
                budget_policy_ref="budget_policy:default_basic",
                model="anthropic/claude-opus-4.7",
                local_mode=True,
                frontier_strategy="mixed",
                command="test",
            )
            finalize_research_loop_plugin_run(
                prepared,
                success=True,
                result={
                    "generations_completed": 1,
                    "run_dir": str(run_dir),
                    "exit_condition": "completed",
                    "total_duration_seconds": 1.0,
                },
            )

            report = verify_run(run_dir)
            self.assertTrue(report["success"])
            self.assertTrue(
                any("missing usage or usage_unknown" in warning for warning in report["warnings"])
            )

    def test_replay_checks_core_source_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            with patch(
                "praxist.core.replay.build_core_source_snapshot",
                return_value={"workspace_hash": "sha256:not-the-current-source"},
            ):
                report = verify_run(Path(result["run_dir"]))
            self.assertTrue(report["success"])
            self.assertTrue(
                any("core source hash drift" in warning for warning in report["warnings"])
            )

            with patch(
                "praxist.core.replay.build_core_source_snapshot",
                return_value={"workspace_hash": "sha256:not-the-current-source"},
            ):
                report = verify_run(Path(result["run_dir"]), locked=True)
            self.assertFalse(report["success"])
            self.assertTrue(any("core source hash drift" in error for error in report["errors"]))

    def test_research_loop_stage_requires_budget_grant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(
                ResearchLoopStage().execute(
                    ResearchLoopStageContext(
                        task_spec=object(),
                        workspace=Path(tmp),
                        run_dir=Path(tmp) / "run",
                        local_mode=True,
                        model="fake",
                        model_provider_ref="model_provider:anthropic_messages",
                        frontier_strategy="mixed",
                        budget_grant_id=None,
                        resolve_only=True,
                    )
                )
            )
            self.assertFalse(result.success)
            self.assertEqual(result.status, "failed")

    def test_research_loop_stage_rejects_unknown_budget_grant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(
                ResearchLoopStage().execute(
                    ResearchLoopStageContext(
                        task_spec=object(),
                        workspace=Path(tmp),
                        run_dir=Path(tmp) / "run",
                        local_mode=True,
                        model="fake",
                        model_provider_ref="model_provider:anthropic_messages",
                        frontier_strategy="mixed",
                        budget_grant_id="grant_missing",
                        resolve_only=True,
                    )
                )
            )
            self.assertFalse(result.success)
            self.assertIn("not found", result.error or "")

    def test_replay_scans_effective_task_spec_for_secret_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            effective_spec = run_dir / "effective_task_spec.yaml"
            effective_spec.write_text(
                effective_spec.read_text(encoding="utf-8") + "\nleaked: sk-test-redaction-000000\n",
                encoding="utf-8",
            )

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("effective_task_spec.yaml" in error for error in report["errors"]))

    def test_key_based_json_redaction_before_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = root / "secret.json"
            write_json(
                json_path, {"api_key": "plainsecret123456", "nested": {"password": "anothersecret"}}
            )
            encoded = json_path.read_text(encoding="utf-8")
            self.assertNotIn("plainsecret123456", encoded)
            self.assertNotIn("anothersecret", encoded)

            artifact = ArtifactWriter(root).persist_json(
                "secret_test",
                "secret.json",
                {"auth_token": "plainsecret123456"},
                schema_ref=None,
                producer={"type": "test", "id": "redaction"},
            )
            payload = root / artifact["payload_path"]
            self.assertNotIn("plainsecret123456", payload.read_text(encoding="utf-8"))

    def test_replay_catches_tampered_json_secret_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_fake_workflow_fixture(
                workspace=Path(tmp),
                runtime_ref="agent_runtime:fake_runtime",
                model_provider_ref="model_provider:fake_provider",
                budget_policy_ref="budget_policy:fake_tiered",
                credential_profile="fake_multi_key",
            )
            run_dir = Path(result["run_dir"])
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            summary["api_key"] = "plainsecret123456"
            (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

            report = verify_run(run_dir)
            self.assertFalse(report["success"])
            self.assertTrue(any("run_summary.json" in error for error in report["errors"]))

    def test_docs_do_not_contain_raw_secret_patterns(self) -> None:
        for rel in (
            "README.md",
            "docs/concepts/architecture.md",
            "docs/guides/credentials.md",
        ):
            with self.subTest(rel=rel):
                self.assertEqual(scan_text(Path(rel).read_text(encoding="utf-8")), [])

    def test_trajectory_writer_resumes_sequence_and_redacts_non_payload_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            TrajectoryWriter(run_dir, "run_hardening").emit("run.started")
            TrajectoryWriter(run_dir, "run_hardening").emit(
                "test.secret_scope",
                scope={"stage_id": "sk-test-redaction-000000"},
                actor={"type": "core", "id": "Bearer abcdefghijklmnop"},
                artifact_refs=[{"logical_path": "x/sk-test-redaction-000000"}],
            )

            records = [
                json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()
            ]
            self.assertEqual([record["seq"] for record in records], [1, 2])
            encoded = json.dumps(records)
            self.assertNotIn("sk-test-redaction-000000", encoded)
            self.assertNotIn("Bearer abcdefghijklmnop", encoded)
            self.assertTrue(records[1]["redaction"]["applied"])

    def test_artifact_writer_resumes_sequence_and_redacts_logical_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            first = ArtifactWriter(run_dir).persist_text(
                "note",
                "notes/sk-test-redaction-000000.md",
                "payload",
                schema_ref=None,
                producer={"type": "test", "id": "writer"},
            )
            second = ArtifactWriter(run_dir).persist_text(
                "note",
                "notes/second.md",
                "payload",
                schema_ref=None,
                producer={"type": "test", "id": "writer"},
            )

            self.assertEqual(first["artifact_id"], "art_000001")
            self.assertEqual(second["artifact_id"], "art_000002")
            index_text = (run_dir / "artifact_index.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("sk-test-redaction-000000", index_text)

    def test_claude_runtime_adapter_redacts_logs_and_exception_errors(self) -> None:
        class TextBlock:
            text = "hello sk-test-redaction-000000"

        class ToolUseBlock:
            name = "Bash"
            input = {"command": "echo sk-test-redaction-000000"}

        class AssistantMessage:
            content = [TextBlock(), ToolUseBlock()]

        formatted = format_legacy_message(AssistantMessage(), "agent")
        self.assertNotIn("sk-test-redaction-000000", formatted)
        extracted = extract_legacy_output([AssistantMessage()])
        self.assertNotIn("sk-test-redaction-000000", json.dumps(extracted))

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "praxist.plugins.agent_runtimes.claude_sdk.adapter._load_claude_sdk",
                side_effect=RuntimeError("bad key sk-test-redaction-000000"),
            ):
                with self.assertLogs(
                    "praxist.plugins.agent_runtimes.claude_sdk.adapter", level="ERROR"
                ) as logs:
                    result = asyncio.run(
                        ClaudeSdkAgentRuntime().execute_legacy(
                            "task",
                            LegacyClaudeRuntimeOptions(
                                name="agent",
                                allowed_tools=[],
                                workspace=Path(tmp),
                                mcp_servers={},
                                model="fake",
                                permission_mode="default",
                            ),
                        )
                    )
            self.assertFalse(result.success)
            self.assertNotIn("sk-test-redaction-000000", result.error or "")
            self.assertNotIn("sk-test-redaction-000000", "\n".join(logs.output))

    def test_deepseek_env_key_selects_deepseek_alias_provider(self) -> None:
        credential_set = CredentialResolver(
            {"DEEPSEEK_API_KEY": "sk-test-redaction-000000"}
        ).discover()
        manager = CredentialFailoverManager(credential_set)
        credential = manager.select(
            scope="model_provider",
            provider="deepseek_alias",
            target_ref="model_provider:deepseek_alias",
        )
        self.assertIsNotNone(credential)
        self.assertEqual(credential.provider, "deepseek_alias")

    def test_new_provider_aliases_env_keys_select_respective_providers(self) -> None:
        cases = (
            ("GROQ_API_KEY", "groq_alias", "model_provider:groq_alias"),
            ("MISTRAL_API_KEY", "mistral_alias", "model_provider:mistral_alias"),
            ("XAI_API_KEY", "xai_alias", "model_provider:xai_alias"),
        )
        for env_key, provider, target_ref in cases:
            credential_set = CredentialResolver({env_key: "sk-test-redaction-000000"}).discover()
            manager = CredentialFailoverManager(credential_set)
            credential = manager.select(
                scope="model_provider",
                provider=provider,
                target_ref=target_ref,
            )
            self.assertIsNotNone(credential, f"failed for {provider}")
            self.assertEqual(credential.provider, provider)

    def test_plugin_code_and_asset_paths_are_confined_to_plugin_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workflow_manifest(root, "stage_traversal", [], assets=["../outside.txt"])
            loader = PluginLoader(PluginRoots(bundled=[root], user=[], project=[]))
            with self.assertRaisesRegex(ValueError, "inside plugin root"):
                loader.resolve(
                    ["workflow_stage:stage_traversal"],
                    root_task_ref="workflow_stage:stage_traversal",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workflow_manifest(root, "stage_absolute", [], assets=["/etc/hosts"])
            loader = PluginLoader(PluginRoots(bundled=[root], user=[], project=[]))
            with self.assertRaisesRegex(ValueError, "inside plugin root"):
                loader.resolve(
                    ["workflow_stage:stage_absolute"], root_task_ref="workflow_stage:stage_absolute"
                )

    def _write_workflow_manifest(
        self,
        root: Path,
        name: str,
        dependencies: list[dict[str, str]],
        assets: list[str] | None = None,
    ) -> None:
        plugin_dir = root / "workflow_stages" / name
        plugin_dir.mkdir(parents=True)
        asset_lines = (
            ["assets:", *[f"  - {asset}" for asset in assets]] if assets else ["assets: []"]
        )
        plugin_dir.joinpath("plugin.yaml").write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    f"name: {name}",
                    "kind: workflow_stage",
                    "version: 0.1.0",
                    "protocol_version: 1",
                    "stability: v1_stable",
                    f"description: {name}",
                    "compatibility:",
                    '  praxist_core: ">=0.1.0,<1.0"',
                    '  python: ">=3.11"',
                    "dependencies:",
                    *[
                        "\n".join(
                            [
                                f"  - kind: {dep['kind']}",
                                f"    name: {dep['name']}",
                                f'    version: "{dep["version"]}"',
                                f"    required: {str(dep.get('required', True)).lower()}",
                            ]
                        )
                        for dep in dependencies
                    ],
                    "capabilities: []",
                    "code: []",
                    *asset_lines,
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _toy_task_path() -> Path:
    return (Path.cwd() / "templates" / "tasks" / "toy_math").resolve()


def _fixture_plugin_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {}
    fixture_roots = os.environ.get("PRAXIST_BUNDLED_PLUGIN_ROOTS", "")
    if fixture_roots:
        env["PRAXIST_BUNDLED_PLUGIN_ROOTS"] = fixture_roots
    env.update(extra or {})
    return env


def _write_research_task(root: Path, *, per_experiment_gpu_hours: float = 0.0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("task.yaml").write_text(
        f"""
task_id: test_research_task
task_name: Test Research Task
description_file: README.md
research_direction: Test the research loop startup path.
evaluation:
  primary_metric: deterministic_score
  direction: maximize
  aux_metrics: []
  seeds: [1]
  aggregation: mean_and_std
compute_budget:
  per_experiment_gpu_hours: {per_experiment_gpu_hours}
  max_parallel_runs_per_peer: 1
generation_policy:
  max_generations: 1
  cohort_size: 1
  per_generation_hours: 5
synthesis_trigger:
  min_findings: 30
  min_interval_minutes: 120
  max_interval_minutes: 240
  min_contributing_peers: 1
praxist_plugins:
  task_ref: task:test_research_task
  workflow:
    stage: workflow_stage:research_loop
  panel:
    topology: panel_topology:fake_two_round
    roles:
      - role:fake_peer
      - role:fake_pi
      - role:fake_chair
  audit_rules:
    - audit_rule:fake_panel_audit
  evaluations:
    - evaluation:fake_pareto
  tools:
    - tool_server:evaluation_tools
    - tool_server:frontier_tools
    - tool_server:finding_graph_query
    - tool_server:memory_tools
    - tool_server:prior_work_tools
  graph_maintainers:
    - graph_maintainer:finding_graph_mvp
  optional_workflow_stages:
    ideation: workflow_stage:ideation_stub
    paper_writing: workflow_stage:paper_writing_stub
    reviewer: workflow_stage:reviewer_stub
""".lstrip(),
        encoding="utf-8",
    )
    root.joinpath("README.md").write_text("Test research task.\n", encoding="utf-8")
    return root.resolve()


if __name__ == "__main__":
    unittest.main()
