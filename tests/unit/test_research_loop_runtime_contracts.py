from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch


class GenerationLivenessContractsTest(unittest.TestCase):
    def test_generation_runtime_cap_uses_enabled_adaptive_ceiling(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.cohort_runner import (
            _effective_generation_cap_seconds,
        )

        enabled = SimpleNamespace(
            max_interval_minutes=350,
            adaptive={"enabled": True, "max_interval_ceiling_minutes": 500},
        )
        disabled = SimpleNamespace(
            max_interval_minutes=350,
            adaptive={"enabled": False, "max_interval_ceiling_minutes": 500},
        )

        self.assertEqual(
            _effective_generation_cap_seconds(
                enabled,
                per_peer_safety_seconds=600 * 60,
            ),
            500 * 60,
        )
        self.assertEqual(
            _effective_generation_cap_seconds(
                disabled,
                per_peer_safety_seconds=600 * 60,
            ),
            350 * 60,
        )
        self.assertEqual(
            _effective_generation_cap_seconds(
                SimpleNamespace(
                    enabled=False,
                    max_interval_minutes=350,
                    adaptive={"enabled": True, "max_interval_ceiling_minutes": 500},
                ),
                per_peer_safety_seconds=600 * 60,
            ),
            600 * 60,
        )

    def test_generation_close_clock_starts_after_prelaunch_planning(self) -> None:
        import inspect

        from praxist.plugins.workflow_stages.research_loop.backend.cohort_runner import (
            run_generation_cohort,
        )

        source = inspect.getsource(run_generation_cohort)
        self.assertLess(
            source.index("prepared_peers = await asyncio.gather"),
            source.index("gen_start_time = time.time()"),
        )
        self.assertLess(
            source.index("gen_start_time = time.time()"),
            source.index("reset_start_time(gen_start_time)"),
        )

    def test_deadline_watchdog_fires_while_parent_event_loop_thread_is_blocked(self) -> None:
        import threading

        from praxist.plugins.workflow_stages.research_loop.backend.cohort_runner import (
            _start_generation_deadline_watchdog,
        )

        fired = threading.Event()
        observed_at: list[float] = []

        class Trigger:
            def fire_deadline(self) -> None:
                observed_at.append(time.time())
                fired.set()

        stop, worker = _start_generation_deadline_watchdog(
            Trigger(),
            deadline=time.time() + 0.03,
            gen_id=2,
        )
        blocked_until = time.time() + 0.15
        time.sleep(0.15)
        stop.set()
        worker.join(timeout=1)

        self.assertTrue(fired.is_set())
        self.assertLess(observed_at[0], blocked_until)

    def test_deadline_watchdog_can_be_retired_after_normal_generation_close(self) -> None:
        import threading

        from praxist.plugins.workflow_stages.research_loop.backend.cohort_runner import (
            _start_generation_deadline_watchdog,
        )

        fired = threading.Event()
        trigger = SimpleNamespace(fire_deadline=fired.set)
        stop, worker = _start_generation_deadline_watchdog(
            trigger,
            deadline=time.time() + 0.2,
            gen_id=0,
        )
        stop.set()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertFalse(fired.is_set())

    def test_signal_finalization_recovers_contiguous_canonical_generation_count(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import resume_state

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for gen_id in (0, 1):
                gen_dir = run_dir / f"gen_{gen_id}"
                gen_dir.mkdir()
                (gen_dir / "generation_boundary.json").write_text(
                    json.dumps({"generation_id": gen_id}),
                    encoding="utf-8",
                )
            gen2 = run_dir / "gen_2"
            gen2.mkdir()
            (gen2 / "generation_boundary.json").write_text("{partial", encoding="utf-8")

            self.assertEqual(resume_state.canonical_completed_generation_count(run_dir), 2)
            self.assertEqual(resume_state.reported_completed_generations({}, run_dir), 2)
            self.assertEqual(
                resume_state.reported_completed_generations({"generations_completed": 3}, run_dir),
                2,
            )


class OrchestratorSignalContractsTest(unittest.TestCase):
    def test_orchestrator_signal_handler_writes_shutdown_and_raises_system_exit(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.orchestrator_runtime import (
            enter_orchestrator_runtime_scope,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            sentinel = run_dir / "ORCHESTRATOR_SHUTDOWN"
            sentinel.write_text("reason=preexisting\n", encoding="utf-8")
            logger = SimpleNamespace(
                warning=lambda *args, **kwargs: None,
                debug=lambda *args, **kwargs: None,
            )
            scope = enter_orchestrator_runtime_scope(run_dir=run_dir, resume=False, logger=logger)
            try:
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "reason=preexisting\n")
                handler = signal.getsignal(signal.SIGTERM)
                self.assertTrue(callable(handler))
                with self.assertRaises(SystemExit) as cm:
                    handler(signal.SIGTERM, None)
                self.assertEqual(cm.exception.code, 128 + signal.SIGTERM)
                text = sentinel.read_text(encoding="utf-8")
                self.assertIn("signal=15", text)
            finally:
                scope.close()

    def test_orchestrator_signal_handler_chains_previous_callable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.orchestrator_runtime import (
            enter_orchestrator_runtime_scope,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            calls: list[int] = []

            def previous(signum: int, _frame: object) -> None:
                calls.append(signum)

            old = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, previous)
            logger = SimpleNamespace(
                warning=lambda *args, **kwargs: None,
                debug=lambda *args, **kwargs: None,
            )
            try:
                scope = enter_orchestrator_runtime_scope(
                    run_dir=run_dir, resume=False, logger=logger
                )
                try:
                    handler = signal.getsignal(signal.SIGTERM)
                    self.assertTrue(callable(handler))
                    handler(signal.SIGTERM, None)
                    self.assertEqual(calls, [signal.SIGTERM])
                    text = (run_dir / "ORCHESTRATOR_SHUTDOWN").read_text(encoding="utf-8")
                    self.assertIn("signal=15", text)
                finally:
                    scope.close()
            finally:
                signal.signal(signal.SIGTERM, old)

    def test_orchestrator_lock_resume_and_signal_install_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            orchestrator_runtime,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            logger = SimpleNamespace(
                warnings=[],
                debug_calls=[],
                warning=lambda *args, **kwargs: logger.warnings.append(args),
                debug=lambda *args, **kwargs: logger.debug_calls.append(args),
            )

            live_lock = run_dir / "orchestrator.lock"
            live_lock.write_text("pid=123\n", encoding="utf-8")
            with (
                patch.object(orchestrator_runtime, "pid_is_alive", return_value=True),
                self.assertRaisesRegex(RuntimeError, "live pid"),
            ):
                orchestrator_runtime.OrchestratorRuntimeScope(
                    run_dir=run_dir,
                    resume=True,
                    logger=logger,
                )._prepare_lock()

            live_lock.write_text("pid=123\nstarted=old\n", encoding="utf-8")
            with patch.object(orchestrator_runtime, "pid_is_alive", return_value=False):
                scope = orchestrator_runtime.OrchestratorRuntimeScope(
                    run_dir=run_dir,
                    resume=True,
                    logger=logger,
                )
                scope._prepare_lock()
            self.assertIn("pid=", live_lock.read_text(encoding="utf-8"))
            events = (run_dir / "resume_events.jsonl").read_text(encoding="utf-8")
            self.assertIn("stale_lock_removed", events)

            live_lock.write_text("unparseable", encoding="utf-8")
            scope = orchestrator_runtime.OrchestratorRuntimeScope(
                run_dir=run_dir,
                resume=False,
                logger=logger,
            )
            scope._prepare_lock()
            self.assertTrue(logger.warnings)

            scope = orchestrator_runtime.OrchestratorRuntimeScope(
                run_dir=run_dir,
                resume=False,
                logger=logger,
            )
            with patch.object(orchestrator_runtime.signal, "signal", side_effect=ValueError):
                scope._install_signal_handlers()
            self.assertFalse(scope._signals_installed)
            self.assertTrue(logger.debug_calls)

            old = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                scope = orchestrator_runtime.enter_orchestrator_runtime_scope(
                    run_dir=run_dir,
                    resume=False,
                    logger=logger,
                )
                try:
                    handler = signal.getsignal(signal.SIGINT)
                    self.assertTrue(callable(handler))
                    handler(signal.SIGINT, None)
                    self.assertIn("signal=2", (run_dir / "ORCHESTRATOR_SHUTDOWN").read_text())
                finally:
                    scope.close()
            finally:
                signal.signal(signal.SIGINT, old)


class BaselineCacheContractsTest(unittest.TestCase):
    def test_runtime_and_curated_baseline_cache_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import baseline_cache

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            entry = baseline_cache.record_measurement(
                "task",
                workspace,
                "adam",
                0.81,
                code_hash="old",
                hardware="test-accelerator",
                dataset_hash="ds",
                mean=0.80,
                std=0.01,
                seeds=[1, 2],
                epochs=5,
                notes="fresh measurement",
            )
            self.assertEqual(entry.name, "adam")

            old_entry = baseline_cache.CachedBaseline(
                name="sgd",
                accuracy=0.75,
                measured_at=(datetime.now(UTC) - timedelta(days=60)).isoformat(),
                code_hash="old",
                seeds=None,  # type: ignore[arg-type]
            )
            baseline_cache.save_cache("task", workspace, [entry, old_entry])
            curated_path = workspace / "curated.jsonl"
            curated_path.write_text(
                "\n".join(
                    [
                        json.dumps({"_protocol": "metadata"}),
                        "{bad json",
                        json.dumps({"optimizer": "vanilla_sam", "accuracy": 0.79}),
                        json.dumps(["not", "a", "row"]),
                    ]
                ),
                encoding="utf-8",
            )
            curated = baseline_cache.load_curated_baseline_entries(curated_path)

            with patch.object(
                baseline_cache,
                "get_changed_files_since",
                return_value=["baseline/train.py"],
            ):
                report = baseline_cache.validate_cache(
                    "task",
                    workspace,
                    ["adam", "sgd", "vanilla_sam", "missing"],
                    current_code_hash="new",
                    stale_after_days=30,
                    curated_entries=curated,
                )
            status_path = baseline_cache.write_report_for_peers(report, workspace / "run")
            status = json.loads(status_path.read_text(encoding="utf-8"))

            self.assertEqual(report.total, 2)
            self.assertEqual(report.stale, 2)
            self.assertEqual(report.curated_baseline_names, ["vanilla_sam"])
            self.assertEqual(report.missing_baselines, ["adam", "sgd", "missing"])
            self.assertEqual(
                status["missing_runtime_cache_baselines"], ["adam", "sgd", "vanilla_sam", "missing"]
            )
            self.assertEqual(baseline_cache.load_cache("missing", workspace), [])


class FindingsIngestAndSyncContractsTest(unittest.TestCase):
    def test_findings_ingest_helper_edge_cases_are_task_generic(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_ingest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uuid_name = "22222222-2222-4222-8222-222222222222_result.json"
            uuid_path = root / uuid_name
            uuid_path.write_text("{}", encoding="utf-8")

            self.assertIsNone(findings_ingest._parse_numeric("no number"))
            self.assertIsNone(findings_ingest._parse_numeric(float("nan")))
            self.assertEqual(findings_ingest._parse_numeric("79.5%"), 0.795)
            self.assertEqual(findings_ingest._normalize_accuracy(79.5), 0.795)
            self.assertIsNone(findings_ingest._walk_find({"a": {"b": []}}, ("missing",)))
            self.assertEqual(
                findings_ingest._walk_find({"a": [{"score": "1.5"}]}, ("score",)),
                1.5,
            )
            self.assertEqual(
                findings_ingest._walk_find_str({"a": [{"variant": "METHOD-A"}]}, ("variant",)),
                "METHOD-A",
            )
            self.assertIsNone(
                findings_ingest._walk_find_str(
                    {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": "too deep"}}}}}}}},
                    ("h",),
                )
            )

            self.assertEqual(
                findings_ingest.infer_dataset(root / "unknown.json", {"dataset": "CIFAR_100"}),
                "cifar100",
            )
            self.assertEqual(
                findings_ingest.infer_dataset(
                    root / "mnist_result.json",
                    {"title": "", "dataset_aliases": ["MNIST"]},
                ),
                "mnist",
            )
            self.assertEqual(
                findings_ingest.infer_dataset(root / "none.json", {"dataset": "custom"}),
                "custom",
            )
            self.assertEqual(
                findings_ingest.extract_variant_name({"title": "Try METHOD-A1 now"}),
                "METHOD-A1",
            )
            self.assertEqual(findings_ingest.extract_variant_name({"method": "not used"}), "")
            self.assertEqual(
                findings_ingest.derive_finding_id(
                    root / "x.json",
                    {"id": "33333333-3333-4333-8333-333333333333"},
                ),
                "33333333-3333-4333-8333-333333333333",
            )
            self.assertEqual(
                findings_ingest.derive_finding_id(uuid_path, {}),
                "22222222-2222-4222-8222-222222222222",
            )
            self.assertTrue(
                findings_ingest.derive_finding_id(root / "plain.json", {}).startswith("fs_")
            )

            metrics = findings_ingest.extract_metrics(
                {
                    "frontier_entity_key": (
                        "artifact::results/top_level_child/tiered_eval_summary.json"
                    ),
                    "details": {
                        "child_id": "details_child_id",
                        "sweep_child_id": "details_child",
                        "trial_id": 7,
                    },
                    "metrics": {
                        "tier": "T3",
                        "tier_reached": "T2",
                        "tier_status": "completed_T3_forced",
                        "promotion_eligible": False,
                        "clean_promotion_eligible": False,
                        "custom_gate": True,
                        "custom_block": False,
                        "scored_complete": True,
                        "bottleneck_target": "drawdown_regression",
                        "evidence_stage": "full_T1",
                        "tradeoff_class": "high_return_drawdown_repair_target",
                        "primary_tradeoff": "return_vs_mdd",
                        "next_step_intent": "repair_failure_mode",
                        "parent_candidate": "parent_v",
                        "parent_usage": "repair",
                        "loss": "0.25",
                        "ignored": "nope",
                    },
                    "nested": {"accuracy": "82%", "gap": "3%"},
                },
                dataset="cifar10",
                primary_metric="test_accuracy",
            )
            self.assertEqual(metrics["tier"], "T3")
            self.assertEqual(metrics["tier_reached"], "T2")
            self.assertEqual(metrics["tier_status"], "completed_T3_forced")
            self.assertFalse(metrics["promotion_eligible"])
            self.assertFalse(metrics["clean_promotion_eligible"])
            self.assertIs(metrics["custom_gate"], True)
            self.assertIs(metrics["custom_block"], False)
            self.assertTrue(metrics["scored_complete"])
            self.assertEqual(metrics["bottleneck_target"], "drawdown_regression")
            self.assertEqual(metrics["evidence_stage"], "full_T1")
            self.assertEqual(metrics["tradeoff_class"], "high_return_drawdown_repair_target")
            self.assertEqual(metrics["primary_tradeoff"], "return_vs_mdd")
            self.assertEqual(metrics["next_step_intent"], "repair_failure_mode")
            self.assertEqual(metrics["parent_candidate"], "parent_v")
            self.assertEqual(metrics["parent_usage"], "repair")
            self.assertEqual(
                metrics["frontier_entity_key"],
                "artifact::results/top_level_child/tiered_eval_summary.json",
            )
            self.assertEqual(metrics["sweep_child_id"], "details_child")
            self.assertEqual(metrics["child_id"], "details_child_id")
            self.assertEqual(metrics["trial_id"], "7")
            self.assertEqual(metrics["loss"], 0.25)
            self.assertEqual(metrics["test_accuracy_cifar10"], 0.82)

            preserved_from_non_metrics = findings_ingest.extract_metrics(
                {
                    "frontier_lane": "alpha",
                    "details": {
                        "strategy_family": "learned_alpha",
                        "source_result_path": ("results/details_child/tiered_eval_summary.json"),
                        "clean_promotion_eligible": True,
                    },
                    "extra": {
                        "evidence_stage": "full_T1",
                        "extra": {"parent_usage": "repair"},
                    },
                }
            )
            self.assertEqual(preserved_from_non_metrics["frontier_lane"], "alpha")
            self.assertEqual(preserved_from_non_metrics["strategy_family"], "learned_alpha")
            self.assertEqual(
                preserved_from_non_metrics["source_result_path"],
                "results/details_child/tiered_eval_summary.json",
            )
            self.assertTrue(preserved_from_non_metrics["clean_promotion_eligible"])
            self.assertEqual(preserved_from_non_metrics["evidence_stage"], "full_T1")
            self.assertEqual(preserved_from_non_metrics["parent_usage"], "repair")

            nested_metrics = findings_ingest.extract_metrics(
                {"nested": {"test_acc": "82%", "gap": "3%"}},
                dataset="cifar10",
                primary_metric="test_accuracy",
            )
            self.assertEqual(nested_metrics["test_accuracy_cifar10"], 0.82)
            self.assertNotIn("train_test_gap", nested_metrics)

            self.assertEqual(
                findings_ingest.infer_finding_type({"finding_type": "hypothesis"}, {}), "hypothesis"
            )
            self.assertEqual(
                findings_ingest.infer_finding_type({"finding_type": "challenge"}, {}),
                "challenge",
            )
            self.assertEqual(
                findings_ingest.infer_finding_type({"title": "negative dead end"}, {}), "error"
            )
            self.assertEqual(
                findings_ingest.infer_finding_type({"summary": "proposal"}, {}), "hypothesis"
            )
            self.assertEqual(
                findings_ingest.infer_finding_type({"notes": "pattern found"}, {}), "insight"
            )
            self.assertEqual(findings_ingest.infer_finding_type({}, {"score": 1}), "result")
            self.assertEqual(
                findings_ingest._infer_peer_and_gen(
                    root / "gen7_peer2_result.json",
                    {"generation_id": "bad"},
                ),
                ("gen7_peer2", 7),
            )
            self.assertEqual(findings_ingest._safe_mtime_ns(root / "missing.json"), 0)

    def test_findings_ingest_parses_agent_files_and_is_idempotent(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_ingest,
            local_store,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings_dir = root / "shared_findings"
            store_dir = root / "store"
            findings_dir.mkdir()
            (findings_dir / "gen2_peer4_candidate_CIFAR10.json").write_text(
                """
                {
                  "title": "METHOD-A1 CIFAR-10 result",
                  "dataset_aliases": {"cifar10": ["CIFAR-10"], "cifar100": ["CIFAR-100"]},
                  "summary": "hypothesis rejected? no, result",
                  "final_results": {"candidate": {"test_accuracy": "79.65%"}},
                  "metrics": {"tier": "T2", "promotion_eligible": true, "delta_acc": +0.013,},
                  "optimizer": "METHOD-A1",
                  "generation_id": "2",
                  "peer_role": "explorer",
                  "links": [{"target": "x"}],
                  "design_dimensions": {"mechanism": "adaptive"}
                }
                """,
                encoding="utf-8",
            )
            (findings_dir / "ambiguous.json").write_text(
                json.dumps(
                    {
                        "title": "CIFAR10 and CIFAR100 comparison",
                        "dataset_aliases": {"cifar10": ["CIFAR10"], "cifar100": ["CIFAR100"]},
                        "metrics": {"accuracy": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            (findings_dir / "gen2_peer4_alpha_incubator.json").write_text(
                json.dumps(
                    {
                        "id": "alpha-incubator-finding",
                        "finding_type": "result",
                        "title": "Early alpha result",
                        "variant_name": "early_alpha",
                        "candidate_entity_key": "artifact::results/early_alpha/tiered_eval_summary.json",
                        "details": {
                            "child_id": "early_alpha_child",
                            "result_variant_id": "early_alpha",
                            "trial_id": 7,
                        },
                        "peer_id": "gen2_peer4",
                        "generation_id": 2,
                        "metrics": {
                            "score": 0.2,
                            "tier": "T1",
                            "promotion_eligible": False,
                            "frontier_lane": "alpha_incubator",
                            "strategy_family": "learned_alpha",
                            "variant_id": "artifact-child-7",
                            "sweep_child_id": "bridge_l1_c007",
                            "frontier_entity_key": "variant::artifact-child-7",
                            "tags": ["active_alpha"],
                            "mean_active_alpha_vs_benchmark_pct": -0.4,
                            "mean_active_share": 0.02,
                            "active_ir": 0.1,
                            "scored_complete": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (findings_dir / "gen2_peer5_details_only_alpha.json").write_text(
                json.dumps(
                    {
                        "id": "details-only-alpha",
                        "finding_type": "result",
                        "title": "Details-only alpha routing",
                        "variant_name": "details_only_alpha",
                        "peer_id": "gen2_peer5",
                        "generation_id": 2,
                        "details": {
                            "frontier_lane": "alpha",
                            "strategy_family": "learned_alpha",
                            "source_result_path": (
                                "results/details_only_alpha/tiered_eval_summary.json"
                            ),
                            "clean_promotion_eligible": True,
                        },
                        "extra": {
                            "evidence_stage": "full_T1",
                        },
                        "metrics": {
                            "score": 0.4,
                            "tier": "T1",
                            "promotion_eligible": False,
                            "mean_active_alpha_vs_benchmark_pct": -0.2,
                            "mean_active_share": 0.03,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (findings_dir / "gen2_peer6_challenge.json").write_text(
                json.dumps(
                    {
                        "id": "challenge-alpha",
                        "finding_type": "challenge",
                        "title": "Challenge alpha parent",
                        "variant_name": "challenge_alpha",
                        "peer_id": "gen2_peer6",
                        "generation_id": 2,
                        "metrics": {"score": -1.0, "promotion_eligible": False},
                        "extra": {"is_negative": True, "peer_role": "falsifier"},
                    }
                ),
                encoding="utf-8",
            )
            (findings_dir / "bad.json").write_text("[1, 2, 3]", encoding="utf-8")

            parsed = findings_ingest.parse_finding_file(
                findings_dir / "gen2_peer4_candidate_CIFAR10.json"
            )
            self.assertIsNotNone(parsed)
            assert parsed is not None
            self.assertEqual(parsed["dataset"], "cifar10")
            self.assertEqual(parsed["peer_id"], "gen2_peer4")
            self.assertEqual(parsed["metrics"]["tier"], "T2")
            self.assertTrue(parsed["metrics"]["promotion_eligible"])
            self.assertAlmostEqual(parsed["metrics"]["delta_acc"], 0.013)
            self.assertEqual(parsed["variant_name"], "METHOD-A1")
            self.assertTrue(parsed["id"].startswith("fs_"))
            parsed_alpha = findings_ingest.parse_finding_file(
                findings_dir / "gen2_peer4_alpha_incubator.json"
            )
            self.assertIsNotNone(parsed_alpha)
            assert parsed_alpha is not None
            self.assertEqual(parsed_alpha["metrics"]["frontier_lane"], "alpha_incubator")
            self.assertEqual(parsed_alpha["metrics"]["strategy_family"], "learned_alpha")
            self.assertEqual(parsed_alpha["metrics"]["variant_id"], "artifact-child-7")
            self.assertEqual(parsed_alpha["metrics"]["sweep_child_id"], "bridge_l1_c007")
            self.assertEqual(
                parsed_alpha["metrics"]["frontier_entity_key"], "variant::artifact-child-7"
            )
            self.assertEqual(
                parsed_alpha["metrics"]["candidate_entity_key"],
                "artifact::results/early_alpha/tiered_eval_summary.json",
            )
            self.assertEqual(parsed_alpha["metrics"]["result_variant_id"], "early_alpha")
            self.assertEqual(parsed_alpha["metrics"]["child_id"], "early_alpha_child")
            self.assertEqual(parsed_alpha["metrics"]["trial_id"], "7")
            self.assertEqual(parsed_alpha["details"]["child_id"], "early_alpha_child")
            self.assertEqual(parsed_alpha["details"]["result_variant_id"], "early_alpha")
            self.assertEqual(parsed_alpha["metrics"]["tags"], ["active_alpha"])
            parsed_details_only = findings_ingest.parse_finding_file(
                findings_dir / "gen2_peer5_details_only_alpha.json"
            )
            self.assertIsNotNone(parsed_details_only)
            assert parsed_details_only is not None
            self.assertEqual(parsed_details_only["metrics"]["frontier_lane"], "alpha")
            self.assertEqual(parsed_details_only["metrics"]["strategy_family"], "learned_alpha")
            self.assertEqual(
                parsed_details_only["metrics"]["source_result_path"],
                "results/details_only_alpha/tiered_eval_summary.json",
            )
            self.assertTrue(parsed_details_only["metrics"]["clean_promotion_eligible"])
            self.assertEqual(parsed_details_only["metrics"]["evidence_stage"], "full_T1")
            parsed_challenge = findings_ingest.parse_finding_file(
                findings_dir / "gen2_peer6_challenge.json"
            )
            self.assertIsNotNone(parsed_challenge)
            assert parsed_challenge is not None
            self.assertEqual(parsed_challenge["finding_type"], "challenge")
            self.assertFalse(parsed_challenge["metrics"]["promotion_eligible"])
            self.assertIsNone(findings_ingest.parse_finding_file(findings_dir / "bad.json"))
            self.assertIsNone(
                findings_ingest.infer_dataset(
                    findings_dir / "ambiguous.json",
                    json.loads((findings_dir / "ambiguous.json").read_text(encoding="utf-8")),
                )
            )

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(store_dir)}):
                touched = findings_ingest.ingest_findings_directory(findings_dir)
                self.assertEqual(touched, 5)
                self.assertEqual(findings_ingest.ingest_findings_directory(findings_dir), 0)
                rows = local_store.get_all_findings()
                self.assertEqual(len(rows), 5)
                challenge_rows = [row for row in rows if row["variant_name"] == "challenge_alpha"]
                self.assertEqual(len(challenge_rows), 1)
                self.assertEqual(challenge_rows[0]["finding_type"], "challenge")
                alpha_rows = [row for row in rows if row["variant_name"] == "early_alpha"]
                self.assertEqual(len(alpha_rows), 1)
                self.assertEqual(
                    alpha_rows[0]["metrics"]["frontier_lane"],
                    "alpha_incubator",
                )
                self.assertEqual(
                    alpha_rows[0]["metrics"]["strategy_family"],
                    "learned_alpha",
                )
                self.assertEqual(
                    alpha_rows[0]["metrics"]["candidate_entity_key"],
                    "artifact::results/early_alpha/tiered_eval_summary.json",
                )
                self.assertEqual(alpha_rows[0]["details"]["trial_id"], 7)
                details_rows = [row for row in rows if row["variant_name"] == "details_only_alpha"]
                self.assertEqual(len(details_rows), 1)
                self.assertEqual(details_rows[0]["metrics"]["frontier_lane"], "alpha")
                self.assertEqual(details_rows[0]["metrics"]["strategy_family"], "learned_alpha")
                self.assertEqual(
                    details_rows[0]["metrics"]["source_result_path"],
                    "results/details_only_alpha/tiered_eval_summary.json",
                )
                self.assertEqual(details_rows[0]["details"]["frontier_lane"], "alpha")
                self.assertTrue(details_rows[0]["metrics"]["clean_promotion_eligible"])

                from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                    FrontierStore,
                )

                promoted = FrontierStore(
                    root / "frontier",
                    promote_top_k=4,
                    primary_metric="score",
                    metric_direction="maximize",
                    require_tier=True,
                    frontier_lanes=[
                        {
                            "name": "alpha_incubator",
                            "k": 10,
                            "include_lanes": ["alpha", "alpha_incubator"],
                            "exclude_families": ["benchmark", "control"],
                            "allow_lower_tier": True,
                            "allow_non_promotable": True,
                            "require_metrics": [
                                "mean_active_alpha_vs_benchmark_pct",
                                "mean_active_share",
                            ],
                            "min_metrics": {
                                "mean_active_alpha_vs_benchmark_pct": -5.0,
                                "mean_active_share": 0.005,
                            },
                            "axes": [
                                ("mean_active_alpha_vs_benchmark_pct", "maximize"),
                                ("active_ir", "maximize"),
                                ("score", "maximize"),
                            ],
                        }
                    ],
                ).promote(2, rows)
                self.assertEqual(len(promoted), 1)
                self.assertEqual(promoted[0]["variant_name"], "early_alpha")
                self.assertEqual(promoted[0]["frontier_lane"], "alpha_incubator")

    def test_findings_sync_materializes_local_and_http_findings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_sync

        with tempfile.TemporaryDirectory() as tmp:
            findings_dir = Path(tmp) / "shared"
            sync = findings_sync.FindingsSync(findings_dir, poll_interval=1, local_mode=True)
            self.assertEqual(findings_sync._sanitize_filename("A/B C!!"), "A_B_C")
            path = findings_sync.save_finding_to_dir(
                {"id": "f1", "variant_name": "Variant One"}, findings_dir
            )
            self.assertIsNotNone(path)
            self.assertIsNone(
                findings_sync.save_finding_to_dir(
                    {"id": "f2", "source_filename": path.name}, findings_dir
                )
            )

            with sync._sync_mutex:
                self.assertEqual(sync.sync_once(), 0)

            with (
                tempfile.TemporaryDirectory() as store_tmp,
                patch.dict(os.environ, {"LOCAL_STORE_DIR": store_tmp}),
                patch.object(
                    sync, "_fetch_all_findings", return_value=[{"id": "f3", "title": "New"}]
                ),
            ):
                self.assertEqual(sync.sync_once(), 2)

            http_sync = findings_sync.FindingsSync(findings_dir, poll_interval=1, local_mode=False)
            fake_response = type(
                "FakeResponse",
                (),
                {
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"findings": [{"id": "remote"}]},
                },
            )()
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync.get_server_url",
                    return_value="http://example.test",
                ),
                patch.object(findings_sync, "HAS_HTTPX", True),
                patch.object(
                    findings_sync,
                    "httpx",
                    SimpleNamespace(get=lambda *args, **kwargs: fake_response),
                    create=True,
                ),
            ):
                self.assertEqual(http_sync._fetch_from_http(), [{"id": "remote"}])


class RuntimeResourceGuardContractsTest(unittest.TestCase):
    def test_gpu_governor_slot_lifecycle_and_bypass(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gpu_governor

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch.dict(os.environ, {"BYPASS_GPU_GOVERNOR": "1"}):
                self.assertTrue(gpu_governor.acquire_slot(0, pid=123, run_dir=run_dir))
                self.assertTrue(gpu_governor.release_slot(0, pid=123, run_dir=run_dir))

            with (
                patch.dict(os.environ, {}, clear=False),
                patch.object(
                    gpu_governor,
                    "_prune_dead",
                    side_effect=lambda entries: entries,
                ),
            ):
                self.assertTrue(
                    gpu_governor.acquire_slot(
                        0,
                        pid=1001,
                        peer="gen0_peer1",
                        tag="t1",
                        expected_seconds=30,
                        max_per_gpu=1,
                        run_dir=run_dir,
                        blocking=False,
                    )
                )
                self.assertFalse(
                    gpu_governor.acquire_slot(
                        0,
                        pid=1002,
                        max_per_gpu=1,
                        run_dir=run_dir,
                        blocking=False,
                    )
                )
                slots = gpu_governor.list_slots(0, run_dir=run_dir)
                self.assertEqual(slots[0].peer, "gen0_peer1")
                self.assertTrue(
                    gpu_governor.transfer_slot(
                        0,
                        from_pid=1001,
                        to_pid=1001,
                        run_dir=run_dir,
                    )
                )
                same_pid_slots = gpu_governor.list_slots(0, run_dir=run_dir)
                self.assertEqual([slot.pid for slot in same_pid_slots], [1001])
                self.assertTrue(
                    gpu_governor.transfer_slot(
                        0,
                        from_pid=1001,
                        to_pid=3001,
                        peer="gen0_peer1",
                        tag="child",
                        run_dir=run_dir,
                    )
                )
                slots = gpu_governor.list_slots(0, run_dir=run_dir)
                self.assertEqual([slot.pid for slot in slots], [3001])
                self.assertEqual(slots[0].tag, "child")
                self.assertTrue(
                    gpu_governor.transfer_slot(
                        0,
                        from_pid=1001,
                        to_pid=3001,
                        run_dir=run_dir,
                    )
                )
                self.assertFalse(
                    gpu_governor.transfer_slot(
                        0,
                        from_pid=1001,
                        to_pid=3002,
                        run_dir=run_dir,
                    )
                )
                self.assertFalse(gpu_governor.release_slot(0, pid=1001, run_dir=run_dir))
                self.assertTrue(gpu_governor.release_slot(0, pid=3001, run_dir=run_dir))
                self.assertEqual(gpu_governor.list_slots(0, run_dir=run_dir), [])
                self.assertTrue(
                    gpu_governor.acquire_slot(
                        0,
                        pid=1001,
                        peer="gen0_peer1",
                        tag="t1",
                        expected_seconds=30,
                        max_per_gpu=1,
                        run_dir=run_dir,
                        blocking=False,
                    )
                )
                self.assertTrue(gpu_governor.release_slot(0, pid=1001, run_dir=run_dir))
                self.assertFalse(gpu_governor.release_slot(0, pid=1001, run_dir=run_dir))

            lock_path = run_dir / "process_governor" / "gpu_1.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text("{bad json}\n", encoding="utf-8")
            self.assertEqual(gpu_governor.list_all_slots(2, run_dir=run_dir)[1], [])

    def test_gpu_governor_helper_edges_are_event_driven_and_best_effort(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import gpu_governor

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            override_dir = run_dir / "override"
            with patch.dict(os.environ, {gpu_governor.ENV_GOVERNOR_DIR: str(override_dir)}):
                self.assertEqual(gpu_governor._governor_dir(None), override_dir)
            with patch.dict(os.environ, {}, clear=False), self.assertRaises(ValueError):
                gpu_governor._governor_dir(None)
            self.assertEqual(gpu_governor._max_per_gpu(0), 1)
            with patch.dict(os.environ, {gpu_governor.ENV_MAX_PER_GPU: "-5"}):
                self.assertEqual(gpu_governor._max_per_gpu(None), 1)
            with patch.dict(os.environ, {gpu_governor.ENV_MAX_PER_GPU: "7"}):
                self.assertEqual(gpu_governor._max_per_gpu(None), 7)
            with patch.dict(os.environ, {gpu_governor.ENV_MAX_PER_GPU: "bad"}):
                self.assertEqual(gpu_governor._max_per_gpu(None), gpu_governor.DEFAULT_MAX_PER_GPU)

            self.assertEqual(
                gpu_governor.SlotEntry.from_dict({"pid": 1, "peer": "p", "ignored": "x"}).peer,
                "p",
            )
            self.assertIsNone(gpu_governor._entry_age_seconds(gpu_governor.SlotEntry(pid=1)))
            self.assertIsNone(
                gpu_governor._entry_age_seconds(gpu_governor.SlotEntry(pid=1, started_at="bad"))
            )
            self.assertIsNotNone(
                gpu_governor._entry_age_seconds(
                    gpu_governor.SlotEntry(pid=1, started_at=datetime.now().isoformat())
                )
            )

            entries = [
                gpu_governor.SlotEntry(pid=0, tag="invalid"),
                gpu_governor.SlotEntry(
                    pid=10,
                    tag="stale",
                    started_at=(datetime.now(UTC) - timedelta(hours=9)).isoformat(),
                ),
                gpu_governor.SlotEntry(pid=11, tag="permission"),
                gpu_governor.SlotEntry(pid=12, tag="missing"),
                gpu_governor.SlotEntry(pid=13, tag="zombie"),
                gpu_governor.SlotEntry(pid=14, tag="alive"),
            ]

            def fake_kill(pid: int, sig: int) -> None:
                if pid == 11:
                    raise PermissionError
                if pid == 12:
                    raise ProcessLookupError

            with (
                patch.object(gpu_governor.os, "kill", side_effect=fake_kill),
                patch.object(gpu_governor, "_is_zombie", side_effect=lambda pid: pid == 13),
            ):
                alive = gpu_governor._prune_dead(entries)
            self.assertEqual([entry.tag for entry in alive], ["permission", "alive"])

            with patch("builtins.open", mock_open(read_data="State:\tZ (zombie)\n")):
                self.assertTrue(gpu_governor._is_zombie(999))
            with patch("builtins.open", side_effect=OSError):
                self.assertFalse(gpu_governor._is_zombie(999))

            lock_path = run_dir / "process_governor" / "gpu_0.lock"
            with (
                patch.object(gpu_governor.asyncio, "get_running_loop", return_value=object()),
                patch.object(gpu_governor.time, "sleep") as sleep,
            ):
                gpu_governor._wait_for_slot_manifest_change(lock_path, 0.1)
            sleep.assert_called_once_with(5.0)

            def fail_asyncio_run(coro):
                coro.close()
                raise RuntimeError

            with (
                patch.object(gpu_governor.asyncio, "get_running_loop", side_effect=RuntimeError),
                patch.object(gpu_governor.asyncio, "run", side_effect=fail_asyncio_run),
                patch.object(gpu_governor.time, "sleep") as fallback_sleep,
            ):
                gpu_governor._wait_for_slot_manifest_change(lock_path, 0.1)
            fallback_sleep.assert_called_once_with(5.0)

            with patch.object(gpu_governor, "_prune_dead", side_effect=lambda value: value):
                self.assertTrue(
                    gpu_governor.acquire_slot(
                        0,
                        pid=2001,
                        max_per_gpu=1,
                        run_dir=run_dir,
                        blocking=False,
                    )
                )
                self.assertTrue(
                    gpu_governor.acquire_slot(
                        0,
                        pid=2001,
                        max_per_gpu=1,
                        run_dir=run_dir,
                        blocking=False,
                    )
                )
                with (
                    patch.object(gpu_governor.time, "monotonic", side_effect=[0.0, 1.0]),
                    patch.object(gpu_governor, "_wait_for_slot_manifest_change"),
                    self.assertRaises(gpu_governor.GovernorBusy),
                ):
                    gpu_governor.acquire_slot(
                        0,
                        pid=2002,
                        max_per_gpu=1,
                        run_dir=run_dir,
                        timeout_seconds=0.5,
                    )

    def test_protected_pid_manifest_lifecycle(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            pid = os.getpid()
            entry = protected_pids.register_pid(
                pid,
                peer_id="gen0/peer1",
                tag="long_eval",
                eta_seconds=60,
                run_dir=run_dir,
            )
            self.assertEqual(entry.peer_id, "gen0/peer1")
            updated = protected_pids.register_pid(
                pid,
                peer_id="gen0/peer1",
                tag="updated",
                eta_seconds=0,
                run_dir=run_dir,
            )
            self.assertEqual(updated.tag, "updated")
            self.assertEqual(updated.eta_seconds, 60)
            self.assertIn(pid, protected_pids.get_protected_pids_set(run_dir=run_dir))
            with (
                patch.object(protected_pids, "_is_pid_alive", return_value=True),
                self.assertRaises(protected_pids.DuplicateProtectedPidError),
            ):
                protected_pids.register_pid(
                    pid + 1,
                    peer_id="gen0/peer1",
                    tag="updated",
                    eta_seconds=60,
                    run_dir=run_dir,
                )
            with (
                patch.dict(os.environ, {protected_pids.ENV_MAX_ACTIVE_PER_PEER: "1"}),
                patch.object(protected_pids, "_is_pid_alive", return_value=True),
                self.assertRaises(protected_pids.ProtectedPidCapacityError),
            ):
                protected_pids.register_pid(
                    pid + 2,
                    peer_id="gen0/peer1",
                    tag="different_long_eval",
                    eta_seconds=60,
                    run_dir=run_dir,
                )
            with patch.object(protected_pids, "_is_pid_alive", return_value=True):
                duplicate = protected_pids.register_pid(
                    pid + 1,
                    peer_id="gen0/peer1",
                    tag="updated",
                    eta_seconds=60,
                    run_dir=run_dir,
                    allow_duplicate=True,
                    max_active_per_peer=2,
                )
                self.assertEqual(duplicate.pid, pid + 1)
                active = protected_pids.list_active_jobs(
                    peer_id="gen0/peer1",
                    tag="updated",
                    run_dir=run_dir,
                )
            self.assertEqual({item.pid for item in active}, {pid, pid + 1})
            self.assertTrue(
                protected_pids.unregister_pid(pid + 1, peer_id="gen0/peer1", run_dir=run_dir)
            )
            self.assertTrue(
                protected_pids.unregister_pid(pid, peer_id="gen0/peer1", run_dir=run_dir)
            )
            self.assertFalse(
                protected_pids.unregister_pid(pid, peer_id="gen0/peer1", run_dir=run_dir)
            )

            manifest = run_dir / "protected_pids" / "bad.json"
            manifest.write_text("{bad json}", encoding="utf-8")
            self.assertEqual(protected_pids.list_all_protected(run_dir=run_dir), [])
            manifest.write_text("{}", encoding="utf-8")
            self.assertEqual(protected_pids.list_all_protected(run_dir=run_dir), [])

    def test_protected_pid_keeps_isolated_group_alive_after_launcher_exits(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        entry = protected_pids.ProtectedEntry(pid=101, pgid=202, peer_id="gen0_peer1")
        with (
            patch.object(protected_pids, "_is_pid_alive", return_value=False),
            patch.object(protected_pids, "_is_process_group_alive", return_value=True),
        ):
            self.assertTrue(protected_pids._entry_is_alive(entry))

        stale = protected_pids.ProtectedEntry(
            pid=101,
            pgid=101,
            peer_id="gen0_peer1",
            pid_start_time=1,
        )
        with (
            patch.object(protected_pids, "_is_pid_alive", return_value=True),
            patch.object(protected_pids, "_pid_start_time", return_value=2),
            patch.object(protected_pids, "_is_process_group_alive", return_value=True),
        ):
            self.assertFalse(protected_pids._entry_is_alive(stale))

    def test_protected_pid_rejects_reused_pid_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        entry = protected_pids.ProtectedEntry(
            pid=os.getpid(),
            pgid=0,
            peer_id="gen0_peer1",
            pid_start_time=-1,
        )
        self.assertFalse(protected_pids._entry_is_alive(entry))

    def test_protected_pid_launch_obeys_peer_capacity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaises(ValueError):
                protected_pids.launch_command(
                    [],
                    peer_id="gen0_peer1",
                    run_dir=run_dir,
                )

            code = protected_pids.launch_command(
                [sys.executable, "-c", "pass"],
                peer_id="gen0_peer1",
                tag="train_once",
                run_dir=run_dir,
                max_active_per_peer=1,
            )
            self.assertEqual(code, 0)
            self.assertEqual(protected_pids.list_active_jobs(run_dir=run_dir), [])

            protected_pids.register_pid(
                os.getpid(),
                peer_id="gen0_peer1",
                tag="still_running",
                run_dir=run_dir,
                max_active_per_peer=1,
            )
            with (
                patch.object(protected_pids.subprocess, "Popen") as popen,
                self.assertRaises(protected_pids.ProtectedPidCapacityError),
            ):
                protected_pids.launch_command(
                    [sys.executable, "-c", "raise SystemExit(99)"],
                    peer_id="gen0_peer1",
                    tag="second_train",
                    run_dir=run_dir,
                    max_active_per_peer=1,
                )
            popen.assert_not_called()

    def test_protected_pid_launch_uses_isolated_process_group(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            fake_proc = SimpleNamespace(pid=12345, wait=lambda: 0)
            with (
                patch.object(protected_pids.subprocess, "Popen", return_value=fake_proc) as popen,
                patch.object(
                    protected_pids,
                    "_is_process_group_alive",
                    side_effect=[True, False],
                ),
                patch.object(protected_pids, "_pid_start_time", return_value="fake-start"),
                patch.object(protected_pids.time, "sleep") as sleep,
            ):
                code = protected_pids.launch_command(
                    ["task-evaluator", "--full"],
                    peer_id="gen0_peer1",
                    run_dir=Path(tmp),
                )

        self.assertEqual(code, 0)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        sleep.assert_called_once()

    def test_legacy_launch_applies_task_python_path_and_environment_boundary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            task_root = root / "task"
            caller_cwd = root / "run" / "gen_0" / "peer"
            evaluator = task_root / "evaluations" / "run.py"
            output = root / "observed.json"
            evaluator.parent.mkdir(parents=True)
            caller_cwd.mkdir(parents=True)
            evaluator.write_text(
                "import json, os, pathlib, sys\n"
                "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
                "'cwd': os.getcwd(), 'pythonpath': os.getenv('PYTHONPATH'), "
                "'pythonhome': os.getenv('PYTHONHOME')}))\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                    "PYTHONPATH": "/runner/python313",
                    "PYTHONHOME": "/runner/python313",
                },
                clear=True,
            ):
                code = protected_pids.launch_command(
                    ["python3.11", "evaluations/run.py", str(output)],
                    peer_id="gen0_peer0",
                    tag="legacy-task-boundary",
                    run_dir=root / "run",
                    cwd=caller_cwd,
                )

            observed = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            observed,
            {"cwd": str(caller_cwd), "pythonpath": None, "pythonhome": None},
        )

    def test_protected_pid_launch_rejects_malformed_central_configuration(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids
        from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client import (
            ENV_SCHEDULER_CONFIG,
            SchedulerUnavailable,
        )

        with (
            patch.dict(os.environ, {ENV_SCHEDULER_CONFIG: "{"}, clear=True),
            self.assertRaisesRegex(SchedulerUnavailable, "configuration is malformed"),
        ):
            protected_pids.launch_command(["task"], peer_id="gen0_peer0")

    def test_protected_pid_launch_cli_reports_launch_validation_errors(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with (
            patch.object(
                protected_pids, "launch_command", side_effect=ValueError("invalid launch")
            ),
            patch("sys.argv", ["protected_pids", "launch", "--peer", "p", "--", "task"]),
            self.assertRaises(SystemExit) as exit_error,
        ):
            protected_pids._cli()
        self.assertEqual(exit_error.exception.code, 2)

    def test_protected_pid_launch_allows_late_work_when_close_freeze_is_disabled(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "CLOSING_SIGNAL").write_text("trigger_reason=mature_quorum\n")
            with patch.dict(os.environ, {"PRAXIST_LAUNCH_GUARD_ENABLED": "0"}):
                code = protected_pids.launch_command(
                    [sys.executable, "-c", "pass"],
                    peer_id="gen0_peer1",
                    tag="late_heavy_eval",
                    run_dir=run_dir,
                )
            self.assertEqual(code, 0)

    def test_protected_pid_launch_refuses_new_work_when_close_freeze_enabled(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "CLOSING_SIGNAL").write_text("trigger_reason=mature_quorum\n")
            with (
                patch.dict(os.environ, {"PRAXIST_LAUNCH_GUARD_ENABLED": "1"}),
                patch.object(protected_pids.subprocess, "Popen") as popen,
                self.assertRaises(protected_pids.GenerationClosingLaunchError),
            ):
                protected_pids.launch_command(
                    [sys.executable, "-c", "pass"],
                    peer_id="gen0_peer1",
                    tag="late_heavy_eval",
                    run_dir=run_dir,
                )
            popen.assert_not_called()

    def test_close_freeze_preserves_existing_protected_job_for_drain(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            protected_pids.register_pid(
                os.getpid(),
                peer_id="gen0_peer1",
                tag="pre_close_evaluation",
                run_dir=run_dir,
            )
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "CLOSING_SIGNAL").write_text("trigger_reason=mature_quorum\n")
            try:
                active = protected_pids.list_active_jobs(peer_id="gen0_peer1", run_dir=run_dir)
                self.assertEqual([entry.tag for entry in active], ["pre_close_evaluation"])
            finally:
                protected_pids.unregister_pid(os.getpid(), peer_id="gen0_peer1", run_dir=run_dir)

    def test_protected_pid_launch_rechecks_close_after_manifest_lock(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            original_lock = protected_pids._manifest_lock

            @contextlib.contextmanager
            def closing_lock(path):
                with original_lock(path):
                    (gen_dir / "CLOSING_SIGNAL").write_text("trigger_reason=mature_quorum\n")
                    yield

            with (
                patch.dict(os.environ, {"PRAXIST_LAUNCH_GUARD_ENABLED": "1"}),
                patch.object(protected_pids, "_manifest_lock", closing_lock),
                patch.object(protected_pids.subprocess, "Popen") as popen,
                self.assertRaises(protected_pids.GenerationClosingLaunchError),
            ):
                protected_pids.launch_command(
                    [sys.executable, "-c", "pass"],
                    peer_id="gen0_peer1",
                    tag="late_heavy_eval",
                    run_dir=run_dir,
                )
            popen.assert_not_called()

    def test_protected_pid_launch_guard_uses_protected_dir_env_when_run_dir_omitted(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            protected_dir = run_dir / "protected_pids"
            gen_dir = run_dir / "gen_0"
            protected_dir.mkdir()
            gen_dir.mkdir()
            (gen_dir / "CLOSING_SIGNAL").write_text("trigger_reason=mature_quorum\n")
            with (
                patch.dict(
                    os.environ,
                    {
                        protected_pids.ENV_PROTECTED_DIR: str(protected_dir),
                        "PRAXIST_LAUNCH_GUARD_ENABLED": "1",
                    },
                ),
                patch.object(protected_pids.subprocess, "Popen") as popen,
                self.assertRaises(protected_pids.GenerationClosingLaunchError),
            ):
                protected_pids.launch_command(
                    [sys.executable, "-c", "pass"],
                    peer_id="gen0_peer1",
                    tag="late_heavy_eval",
                    run_dir=None,
                )
            popen.assert_not_called()

    def test_protected_pid_launch_guard_can_be_disabled_by_task_runtime_env(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "CLOSING_SIGNAL").write_text("trigger_reason=mature_quorum\n")
            with patch.dict(os.environ, {"PRAXIST_LAUNCH_GUARD_ENABLED": "0"}):
                code = protected_pids.launch_command(
                    [sys.executable, "-c", "pass"],
                    peer_id="gen0_peer1",
                    tag="operator_allowed_late_eval",
                    run_dir=run_dir,
                )
            self.assertEqual(code, 0)

    def test_protected_pid_launch_waits_for_capacity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            protected_pids.register_pid(
                os.getpid(),
                peer_id="gen0_peer1",
                tag="first",
                run_dir=run_dir,
                max_active_per_peer=1,
            )

            alive_checks = {"count": 0}

            def alive_then_dead(pid):
                alive_checks["count"] += 1
                return alive_checks["count"] == 1

            with (
                patch.object(protected_pids, "_is_pid_alive", side_effect=alive_then_dead),
                patch.object(protected_pids.time, "sleep") as sleep,
            ):
                code = protected_pids.launch_command(
                    [sys.executable, "-c", "pass"],
                    peer_id="gen0_peer1",
                    tag="second",
                    run_dir=run_dir,
                    max_active_per_peer=1,
                    wait_timeout_seconds=2.0,
                    poll_seconds=0.01,
                )
            self.assertEqual(code, 0)
            sleep.assert_called_once_with(0.05)

    def test_protected_pid_launch_rechecks_close_signal_while_waiting_for_capacity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            protected_pids.register_pid(
                os.getpid(),
                peer_id="gen0_peer1",
                tag="first",
                run_dir=run_dir,
                max_active_per_peer=1,
            )

            alive_checks = {"count": 0}

            def alive_then_dead(pid):
                alive_checks["count"] += 1
                return alive_checks["count"] == 1

            def mark_closing(_seconds):
                (gen_dir / "CLOSING_SIGNAL").write_text("trigger_reason=mature_quorum\n")

            with (
                patch.dict(os.environ, {"PRAXIST_LAUNCH_GUARD_ENABLED": "1"}),
                patch.object(protected_pids, "_is_pid_alive", side_effect=alive_then_dead),
                patch.object(protected_pids.time, "sleep", side_effect=mark_closing),
                patch.object(protected_pids.subprocess, "Popen") as popen,
                self.assertRaises(protected_pids.GenerationClosingLaunchError),
            ):
                protected_pids.launch_command(
                    [sys.executable, "-c", "pass"],
                    peer_id="gen0_peer1",
                    tag="second",
                    run_dir=run_dir,
                    max_active_per_peer=1,
                    wait_timeout_seconds=2.0,
                    poll_seconds=0.01,
                )
            popen.assert_not_called()

    def test_protected_pid_env_and_cli_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        self.assertIsNone(protected_pids._max_active_per_peer())
        with patch.dict(os.environ, {protected_pids.ENV_MAX_ACTIVE_PER_PEER: "bad"}):
            self.assertIsNone(protected_pids._max_active_per_peer())
        protected_pids._check_duplicate_tag([], peer_id="p", tag="")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with (
                patch(
                    "sys.argv",
                    [
                        "protected_pids",
                        "register",
                        "--pid",
                        str(os.getpid()),
                        "--peer",
                        "p",
                        "--run-dir",
                        str(run_dir),
                    ],
                ),
                self.assertRaises(SystemExit) as reg_exit,
            ):
                protected_pids._cli()
            self.assertEqual(reg_exit.exception.code, 0)

            with (
                patch(
                    "sys.argv", ["protected_pids", "list", "--peer", "p", "--run-dir", str(run_dir)]
                ),
                self.assertRaises(SystemExit) as list_exit,
            ):
                protected_pids._cli()
            self.assertEqual(list_exit.exception.code, 0)

            with (
                patch.object(protected_pids, "wait_job", return_value=True),
                patch("sys.argv", ["protected_pids", "wait", "--peer", "p", "--tag", "t"]),
                self.assertRaises(SystemExit) as wait_exit,
            ):
                protected_pids._cli()
            self.assertEqual(wait_exit.exception.code, 0)

            with (
                patch.object(protected_pids, "launch_command", return_value=7) as launch,
                patch(
                    "sys.argv",
                    [
                        "protected_pids",
                        "launch",
                        "--peer",
                        "p",
                        "--tag",
                        "t",
                        "--run-dir",
                        str(run_dir),
                        "--",
                        sys.executable,
                        "-c",
                        "pass",
                    ],
                ),
                self.assertRaises(SystemExit) as launch_exit,
            ):
                protected_pids._cli()
            self.assertEqual(launch_exit.exception.code, 7)
            launch.assert_called_once()

            protected_pids.register_pid(
                os.getpid(),
                peer_id="capacity",
                tag="full",
                run_dir=run_dir,
                max_active_per_peer=1,
            )
            with (
                patch(
                    "sys.argv",
                    [
                        "protected_pids",
                        "register",
                        "--pid",
                        str(os.getpid() + 1),
                        "--peer",
                        "capacity",
                        "--tag",
                        "other",
                        "--run-dir",
                        str(run_dir),
                        "--max-active",
                        "1",
                    ],
                ),
                patch.object(protected_pids, "_is_pid_alive", return_value=True),
                self.assertRaises(SystemExit) as cap_exit,
            ):
                protected_pids._cli()
            self.assertEqual(cap_exit.exception.code, 2)

    def test_cohort_drained_active_work_counts_current_gen_protected_jobs(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner
        from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
            ProtectedEntry,
        )

        done_task = SimpleNamespace(done=lambda: True)
        active_task = SimpleNamespace(done=lambda: False)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            protected_jobs = [
                ProtectedEntry(pid=101, peer_id="gen0_peer0", tag="train"),
                ProtectedEntry(pid=102, peer_id="gen0/peer1", tag="train"),
                ProtectedEntry(pid=103, peer_id="gen0-peer2", tag="train"),
                ProtectedEntry(pid=201, peer_id="gen1_peer0", tag="train"),
            ]
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.protected_pids.list_active_jobs",
                return_value=protected_jobs,
            ):
                self.assertEqual(
                    cohort_runner._active_generation_work_count(
                        [done_task],
                        run_dir=run_dir,
                        gen_id=0,
                    ),
                    3,
                )
                self.assertEqual(
                    cohort_runner._active_generation_work_count(
                        [active_task],
                        run_dir=run_dir,
                        gen_id=1,
                    ),
                    2,
                )

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.protected_pids.list_active_jobs",
                return_value=[],
            ):
                self.assertEqual(
                    cohort_runner._active_generation_work_count(
                        [done_task],
                        run_dir=run_dir,
                        gen_id=0,
                    ),
                    0,
                )

    def test_protected_pid_wait_job_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with (
            patch.object(protected_pids, "list_active_jobs", side_effect=[[object()], []]),
            patch("time.sleep") as sleep_mock,
        ):
            self.assertTrue(
                protected_pids.wait_job(
                    peer_id="gen0_peer1",
                    tag="long_eval",
                    timeout_seconds=2.0,
                    poll_seconds=0.01,
                )
            )
            sleep_mock.assert_called_once_with(0.05)

        with (
            patch.object(protected_pids, "list_active_jobs", return_value=[object()]),
            patch("time.monotonic", side_effect=[0.0, 1.0]),
            patch("time.sleep"),
        ):
            self.assertFalse(
                protected_pids.wait_job(
                    peer_id="gen0_peer1",
                    tag="long_eval",
                    timeout_seconds=0.5,
                    poll_seconds=0.01,
                )
            )

    def test_training_timeout_policy_helpers_and_kill_path(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            training_timeout,
        )

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            log.write_text("Epoch 1/10\nnoise\nEpoch 9/10\nEpoch 2/5\n", encoding="utf-8")
            self.assertEqual(training_timeout.parse_current_epoch(log, expected_total=10), (9, 10))
            self.assertEqual(training_timeout.parse_current_epoch(log, expected_total=500), (9, 10))
            self.assertIsNone(training_timeout.parse_current_epoch(Path(tmp) / "missing.log"))
            self.assertEqual(training_timeout.should_emit_partial_summary(0, 10), (True, "ok"))
            self.assertFalse(training_timeout.should_emit_partial_summary(10, 10)[0])
            self.assertFalse(training_timeout.should_emit_partial_summary(4, 10)[0])
            self.assertAlmostEqual(training_timeout.apply_frontier_discount(10.0, True), 9.5)
            with self.assertRaises(ValueError):
                training_timeout.monitor_subprocess_with_grace(
                    type("P", (), {"poll": lambda self: None, "pid": 1})(),
                    log,
                    total_epochs=1,
                )

            class HangingProc:
                pid = 4321

                def poll(self):
                    return None

                def wait(self, timeout=None):
                    return -signal.SIGTERM

            policy = training_timeout.TimeoutPolicy(
                hard_cap_seconds=0,
                grace_check_interval_seconds=1,
                kill_grace_seconds=0,
            )
            with (
                patch.object(training_timeout, "parse_current_epoch", return_value=None),
                patch.object(training_timeout.os, "killpg"),
            ):
                rc = training_timeout.monitor_subprocess_with_grace(
                    HangingProc(),
                    log,
                    total_epochs=10,
                    policy=policy,
                )
            self.assertEqual(rc, -signal.SIGTERM)

    def test_training_timeout_monitor_preserves_result_and_bounds_grace_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            training_timeout,
        )

        class ScriptedProc:
            pid = 4321

            def __init__(
                self,
                polls: list[int | None],
                waits: list[object] | None = None,
            ) -> None:
                self.polls = list(polls)
                self.waits = list(waits or [])
                self.terminated = False
                self.killed = False

            def poll(self):
                if self.polls:
                    return self.polls.pop(0)
                return None

            def wait(self, timeout=None):
                if self.waits:
                    value = self.waits.pop(0)
                    if isinstance(value, BaseException):
                        raise value
                    return value
                return 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

        def clock(*values: float):
            sequence = list(values)

            def _next() -> float:
                if len(sequence) > 1:
                    return sequence.pop(0)
                return sequence[0]

            return _next

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            log.write_text("Epoch 9/10\n", encoding="utf-8")
            policy = training_timeout.TimeoutPolicy(
                hard_cap_seconds=0,
                grace_progress_threshold=0.9,
                grace_max_extension_seconds=3,
                grace_stall_max_polls=2,
                grace_check_interval_seconds=1,
                kill_grace_seconds=0,
            )

            self.assertIsNone(
                training_timeout.parse_current_epoch(log, epoch_pattern=re.compile(r"Epoch\s+x"))
            )
            self.assertEqual(
                training_timeout.parse_current_epoch(log, expected_total=None), (9, 10)
            )
            self.assertEqual(training_timeout.should_emit_partial_summary(1, 10)[0], True)
            self.assertEqual(
                training_timeout.should_emit_partial_summary(1, 0), (False, "total_cells <= 0")
            )
            self.assertAlmostEqual(
                training_timeout.apply_frontier_discount(
                    10.0,
                    True,
                    training_timeout.PartialSummaryPolicy(frontier_discount=0.8),
                ),
                8.0,
            )
            self.assertEqual(training_timeout.apply_frontier_discount(10.0, False), 10.0)

            exiting = ScriptedProc([None, 7])
            with (
                patch.object(training_timeout.time, "time", side_effect=clock(0, 1, 1, 2)),
                patch.object(training_timeout.time, "sleep"),
                patch.object(training_timeout, "emit_resource_event_from_env"),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        exiting,
                        log,
                        total_epochs=10,
                        policy=policy,
                    ),
                    7,
                )

            stale = ScriptedProc([None, None, None, None], waits=[-3])
            with (
                patch.object(training_timeout.time, "time", side_effect=clock(0, 1, 1, 2, 2, 3, 3)),
                patch.object(training_timeout.time, "sleep"),
                patch.object(
                    training_timeout, "parse_current_epoch", side_effect=[(9, 10), (9, 10), (9, 10)]
                ),
                patch.object(training_timeout.os, "killpg"),
                patch.object(training_timeout, "emit_resource_event_from_env"),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        stale,
                        log,
                        total_epochs=10,
                        policy=policy,
                    ),
                    -3,
                )

            no_progress = ScriptedProc([None, None, None, None], waits=[-3])
            with (
                patch.object(training_timeout.time, "time", side_effect=clock(0, 1, 1, 2, 2, 3, 3)),
                patch.object(training_timeout.time, "sleep"),
                patch.object(
                    training_timeout, "parse_current_epoch", side_effect=[(9, 10), None, None]
                ),
                patch.object(training_timeout.os, "killpg"),
                patch.object(training_timeout, "emit_resource_event_from_env"),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        no_progress,
                        log,
                        total_epochs=10,
                        policy=policy,
                    ),
                    -3,
                )

            too_long = ScriptedProc([None, None], waits=[-3])
            with (
                patch.object(training_timeout.time, "time", side_effect=clock(0, 1, 1, 10, 10)),
                patch.object(training_timeout.time, "sleep"),
                patch.object(training_timeout, "parse_current_epoch", return_value=(9, 10)),
                patch.object(training_timeout.os, "killpg"),
                patch.object(training_timeout, "emit_resource_event_from_env"),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        too_long,
                        log,
                        total_epochs=10,
                        policy=policy,
                    ),
                    -3,
                )

            below_threshold = ScriptedProc([None, None], waits=[-signal.SIGTERM])
            with (
                patch.object(training_timeout.time, "time", side_effect=clock(0, 1)),
                patch.object(training_timeout.time, "sleep"),
                patch.object(training_timeout, "parse_current_epoch", return_value=(5, 10)),
                patch.object(training_timeout.os, "killpg"),
                patch.object(training_timeout, "emit_resource_event_from_env"),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        below_threshold,
                        log,
                        total_epochs=10,
                        policy=policy,
                    ),
                    -signal.SIGTERM,
                )

            class MinimalOs:
                SEEK_END = os.SEEK_END

            fallback_proc = ScriptedProc(
                [None, None],
                waits=[
                    subprocess.TimeoutExpired(cmd="train", timeout=0),
                    -9,
                ],
            )
            with (
                patch.object(training_timeout.time, "time", side_effect=clock(0, 1)),
                patch.object(training_timeout.time, "sleep"),
                patch.object(training_timeout, "parse_current_epoch", return_value=None),
                patch.object(training_timeout, "os", MinimalOs),
                patch.object(training_timeout, "emit_resource_event_from_env"),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        fallback_proc,
                        log,
                        total_epochs=10,
                        policy=policy,
                    ),
                    -1,
                )
            self.assertTrue(fallback_proc.terminated)
            self.assertTrue(fallback_proc.killed)


class EventAndStatusContractsTest(unittest.TestCase):
    def test_pending_boundary_checkpoint_is_active_before_retry_collection(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_boundary

        cutoff = datetime(2026, 8, 4, 13, 0, tzinfo=UTC)
        source_snapshot = {"results/candidate/summary.json": "target:1:2"}
        observed: list[object] = []
        loop = SimpleNamespace(
            run_dir=Path("/tmp/run"),
            _findings_sync=None,
            _boundary_evidence_cutoff=None,
            _collect_findings_for_generation=lambda _gen_id: (
                observed.append(loop._boundary_evidence_cutoff) or []
            ),
            frontier=SimpleNamespace(promote=lambda _gen_id, _findings: []),
            task_spec=SimpleNamespace(
                generation_policy=SimpleNamespace(max_generations=1),
                research_memory=SimpleNamespace(enabled=False),
            ),
            gems=None,
        )

        with (
            patch.object(
                generation_boundary,
                "read_boundary_evidence_checkpoint",
                return_value=(cutoff, source_snapshot),
            ),
            patch.object(
                generation_boundary,
                "write_boundary_evidence_checkpoint",
            ) as write_checkpoint,
            patch.object(
                generation_boundary,
                "_annotate_diversity_overlap",
                side_effect=lambda _loop, gen_id, findings: findings,
            ),
            patch.object(generation_boundary, "_generation_stop_audit", return_value={}),
            patch.object(generation_boundary, "_generation_peer_mix", return_value={}),
            patch.object(generation_boundary, "_sync_graph_before_next_generation"),
            patch.object(generation_boundary, "_write_boundary_marker_if_possible"),
        ):
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    loop,
                    gen_id=0,
                    pi_agent=None,
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

        self.assertTrue(observed)
        self.assertEqual(observed[0], (0, cutoff, source_snapshot))
        write_checkpoint.assert_not_called()

    def test_generation_loop_collection_inherits_only_matching_boundary_cutoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
            generation_loop,
        )

        cutoff = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
        source_snapshot = {"results/candidate/summary.json": "target:1:2"}
        loop = object.__new__(generation_loop.GenerationLoop)
        loop.task_spec = SimpleNamespace(evaluation=SimpleNamespace(primary_metric="score"))
        loop.findings_dir = Path("/tmp/findings")
        loop.local_mode = True
        loop._boundary_evidence_cutoff = (4, cutoff, source_snapshot)

        with (
            patch.object(
                findings_collection,
                "result_artifact_options_from_task_spec",
                return_value={"materialize_result_artifacts": True},
            ),
            patch.object(
                findings_collection,
                "collect_findings_for_generation",
                return_value=[],
            ) as collect,
        ):
            loop._collect_findings_for_generation(4, do_ingest=False)
            matching_kwargs = collect.call_args.kwargs
            loop._collect_findings_for_generation(3, do_ingest=False)
            unrelated_kwargs = collect.call_args.kwargs

        self.assertEqual(matching_kwargs["evidence_cutoff"], cutoff)
        self.assertEqual(matching_kwargs["evidence_source_snapshot"], source_snapshot)
        self.assertFalse(matching_kwargs["do_ingest"])
        self.assertNotIn("evidence_cutoff", unrelated_kwargs)
        self.assertNotIn("evidence_source_snapshot", unrelated_kwargs)

    def test_generation_loop_boundary_collector_preserves_nonstandard_result_sources(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            findings_collection,
            generation_loop,
        )

        cutoff = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
        source_snapshot = {"results/candidate.json": "target:1:2"}
        finding = {
            "finding_id": "nonstandard-source",
            "source_result_path": "results/candidate.json",
        }
        loop = object.__new__(generation_loop.GenerationLoop)
        loop.task_spec = SimpleNamespace(evaluation=SimpleNamespace(primary_metric="score"))
        loop.findings_dir = Path("/tmp/findings")
        loop.local_mode = True
        loop._boundary_evidence_cutoff = None

        with patch.object(
            findings_collection,
            "collect_findings_for_generation",
            return_value=[finding],
        ) as collect:
            result = loop._collect_findings_for_boundary(
                4,
                evidence_cutoff=cutoff,
                evidence_source_snapshot=source_snapshot,
            )

        self.assertEqual(result, [finding])
        self.assertEqual(loop._boundary_evidence_cutoff, (4, cutoff, source_snapshot))
        self.assertEqual(collect.call_args.kwargs["evidence_cutoff"], cutoff)
        self.assertEqual(collect.call_args.kwargs["evidence_source_snapshot"], source_snapshot)

    def test_base_agent_request_execution_and_autonomous_loop_controls(self) -> None:
        from praxist.core.protocol import AgentEvent, AgentRunResult
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop_signal = root / "STOP_SIGNAL"
            checker = agent.StopChecker(max_runtime=999, stop_signal_path=stop_signal)
            self.assertIsNone(checker.check())
            stop_signal.write_text("stop", encoding="utf-8")
            self.assertEqual(checker.check(), agent.StopReason.SYNTHESIS_TRIGGER)
            timeout_checker = agent.StopChecker(max_runtime=0.001)
            timeout_checker.start_time -= 1
            self.assertEqual(timeout_checker.check(), agent.StopReason.TIMEOUT)
            timeout_checker.record_error()
            timeout_checker.record_success()
            self.assertEqual(timeout_checker.consecutive_errors, 0)

            manifest = {
                "schema_version": "praxist.prompt_layout.v1",
                "layout_hash": "layout",
                "frozen_prefix_hash": "frozen",
                "dynamic_payload_hash": "dynamic",
                "cache_mode": "runtime_auto_cache",
                "runtime_cache_strategy": "stable_prefix",
                "provider_cache_strategy": "automatic",
                "cache_breakpoints": ["frozen_prefix"],
            }
            base = agent.BaseAgent(
                name="peer/name",
                allowed_tools=["Read"],
                workspace=root,
                mcp_servers={"evaluation-tools": object()},
                model="openrouter/model",
                prompt_layout_manifest=manifest,
                premium_mode=True,
            )
            with patch.dict(
                os.environ,
                {
                    "PRAXIST_RUN_ID": "run",
                    "PRAXIST_STAGE_ID": "research_loop",
                    "PRAXIST_MODEL_PROVIDER_REF": "model_provider:openrouter",
                    "PRAXIST_MODEL_CREDENTIAL_KEY_ID": "cred1",
                    "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1/messages",
                    "PRAXIST_AGENT_TIMEOUT_SECONDS": "bad",
                },
                clear=False,
            ):
                env = agent._scoped_legacy_provider_env()
                request = base._build_agent_run_request("task", env)
            self.assertEqual(request.run_id, "run")
            self.assertEqual(request.prompt_ref["kind"], "prompt_layout_v1")
            self.assertEqual(request.model_call.provider_ref, "model_provider:openrouter")
            self.assertEqual(request.timeout_seconds, 0)
            self.assertIn("evaluation-tools", request.tool_servers[0]["server_name"])
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    agent._legacy_model_provider_ref("deepseek-chat"),
                    "model_provider:deepseek_alias",
                )
                self.assertEqual(
                    agent._legacy_model_provider_ref("gpt-4.1"),
                    "model_provider:openai_compatible",
                )
            self.assertEqual(agent._float_payload("bad"), 0.0)
            self.assertEqual(agent._int_payload("bad"), 0)
            self.assertIn("Bootstrap Recovery", agent._with_bootstrap_retry_directive("start"))

            async def fake_execute(runtime_self, request_arg, options):
                return AgentRunResult(
                    success=True,
                    events=[
                        AgentEvent(
                            event_id="e1",
                            run_id=request_arg.run_id,
                            agent_run_id=request_arg.request_id,
                            stage_id=request_arg.stage_id,
                            type="final_result",
                            payload={
                                "success": True,
                                "duration": "2.5",
                                "iteration_count": "3",
                                "legacy_output": {"text_outputs": ["done"], "tool_uses": []},
                            },
                            artifact_refs=[],
                            credential_refs=[],
                            timestamp_ms=1,
                        )
                    ],
                    text_output_refs=[],
                    tool_uses=[],
                    error=None,
                    failover_reason="none",
                    credential_ref=request_arg.credential_ref,
                )

            fake_runtime = SimpleNamespace(
                execute=lambda request_arg, options: fake_execute(
                    None,
                    request_arg,
                    options,
                )
            )
            with patch.object(agent, "runtime_for_ref", return_value=fake_runtime):
                result = asyncio.run(base.execute("task"))
            self.assertTrue(result.success)
            self.assertEqual(result.iteration_count, 3)

            loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer0",
                generation_id=0,
                task_prompt="do work",
                workspace=root,
                max_runtime_seconds=1,
                logs_dir=root / "logs",
                findings_dir=root / "findings",
                local_mode=True,
                allowed_tools=["Read"],
                stop_signal_path=root / "gen_0" / "STOP_SIGNAL",
            )
            self.assertTrue(loop._is_next_session_event(root / "findings" / "a.json"))
            self.assertFalse(loop._is_next_session_event(root / "shared_store.db-wal"))
            self.assertTrue(loop._is_next_session_event(root / "STOP_SIGNAL", productive=False))
            self.assertTrue(
                loop._session_was_bootstrap_wait(
                    agent.AgentResult(
                        success=True,
                        output={"text_outputs": ["What would you like me to do?"]},
                        duration=0,
                        iteration_count=0,
                    )
                )
            )
            with patch.dict(
                os.environ,
                {
                    "PRAXIST_LAUNCH_GUARD_ENABLED": "1",
                    "PROTECTED_PIDS_DIR": str(root / "protected_pids"),
                    "PRAXIST_MAX_PARALLEL_RUNS_PER_PEER": "2",
                    "GPU_GOVERNOR_DIR": str(root / "gpu_governor"),
                    "GPU_GOVERNOR_MAX_PER_GPU": "1",
                },
            ):
                created = loop._create_agent("session_check")
            self.assertEqual(created.runtime_env_overrides["PEER_ID"], "gen0_peer0")
            self.assertEqual(created.runtime_env_overrides["GENERATION_ID"], "0")
            self.assertEqual(created.runtime_env_overrides["AUTO_RESEARCH_RUN_DIR"], str(root))
            self.assertEqual(created.runtime_env_overrides["PRAXIST_LAUNCH_GUARD_ENABLED"], "1")
            self.assertEqual(
                created.runtime_env_overrides["PROTECTED_PIDS_DIR"],
                str(root / "protected_pids"),
            )
            self.assertEqual(
                created.runtime_env_overrides["PRAXIST_MAX_PARALLEL_RUNS_PER_PEER"], "2"
            )
            self.assertEqual(
                created.runtime_env_overrides["GPU_GOVERNOR_DIR"],
                str(root / "gpu_governor"),
            )
            self.assertEqual(created.runtime_env_overrides["GPU_GOVERNOR_MAX_PER_GPU"], "1")
            request = created._build_agent_run_request("task", {"PEER_ID": "gen0_peer0"})
            self.assertEqual(
                request.runtime_options["runtime_env_overrides"]["PEER_ID"],
                "gen0_peer0",
            )
            self.assertEqual(
                request.runtime_options["runtime_env_overrides"]["GENERATION_ID"],
                "0",
            )
            self.assertEqual(
                request.runtime_options["runtime_env_overrides"]["AUTO_RESEARCH_RUN_DIR"],
                str(root),
            )
            self.assertEqual(
                request.runtime_options["runtime_env_overrides"]["PRAXIST_LAUNCH_GUARD_ENABLED"],
                "1",
            )
            self.assertEqual(
                request.runtime_options["runtime_env_overrides"]["PROTECTED_PIDS_DIR"],
                str(root / "protected_pids"),
            )
            self.assertEqual(
                request.runtime_options["runtime_env_overrides"][
                    "PRAXIST_MAX_PARALLEL_RUNS_PER_PEER"
                ],
                "2",
            )
            self.assertEqual(
                request.runtime_options["runtime_env_overrides"]["GPU_GOVERNOR_DIR"],
                str(root / "gpu_governor"),
            )
            self.assertEqual(
                request.runtime_options["runtime_env_overrides"]["GPU_GOVERNOR_MAX_PER_GPU"],
                "1",
            )
            cohort_loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer0",
                generation_id=0,
                task_prompt="do work",
                workspace=root,
                max_runtime_seconds=1,
                logs_dir=root / "gen_0" / "gen0_peer0",
                findings_dir=root / "findings",
                local_mode=True,
                allowed_tools=["Read"],
            )
            self.assertEqual(
                cohort_loop._create_agent("session_check").runtime_env_overrides[
                    "AUTO_RESEARCH_RUN_DIR"
                ],
                str(root),
            )

            async def one_session():
                return agent.AgentResult(
                    success=True,
                    output={"text_outputs": ["worked"]},
                    duration=0.1,
                    iteration_count=1,
                )

            async def stop_after_wait(*, productive=True):
                loop.stop_checker.start_time -= 10

            loop._run_session = one_session  # type: ignore[method-assign]
            loop._wait_for_next_session_event = stop_after_wait  # type: ignore[method-assign]
            run_result = asyncio.run(loop.run())
            self.assertEqual(run_result["sessions"], 1)
            self.assertEqual(run_result["stop_reason"], "timeout")

    def test_autonomous_loop_bounds_repeated_empty_success_sessions(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer0",
                generation_id=0,
                task_prompt="do work",
                workspace=root,
                max_runtime_seconds=10,
                logs_dir=root / "logs",
                findings_dir=root / "findings",
                local_mode=True,
                allowed_tools=["Read"],
            )
            calls = 0

            async def empty_session():
                nonlocal calls
                calls += 1
                return agent.AgentResult(
                    success=True,
                    output={"text_outputs": [""], "tool_uses": [], "background_tasks": []},
                    duration=0,
                    iteration_count=0,
                )

            loop._run_session = empty_session  # type: ignore[method-assign]
            with patch.object(agent, "_EMPTY_SESSION_RETRY_SECONDS", 0):
                result = asyncio.run(loop.run())

            self.assertEqual(calls, 2)
            self.assertEqual(result["sessions"], 2)
            self.assertEqual(result["stop_reason"], "runtime_empty")

            terminal_result = agent.AgentResult(
                success=True,
                output={
                    "text_outputs": [],
                    "tool_uses": [],
                    "background_tasks": [{"task_id": "task-1", "status": "completed"}],
                },
                duration=0,
                iteration_count=0,
            )
            self.assertFalse(loop._session_was_empty(terminal_result))

    def test_autonomous_loop_bounds_repeated_runtime_failures_to_one_peer(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer0",
                generation_id=0,
                task_prompt="do work",
                workspace=root,
                max_runtime_seconds=10,
                logs_dir=root / "logs",
                findings_dir=root / "findings",
                local_mode=True,
                allowed_tools=["Read"],
            )
            calls = 0

            async def failed_session():
                nonlocal calls
                calls += 1
                raise RuntimeError("provider unavailable")

            loop._run_session = failed_session  # type: ignore[method-assign]
            with (
                patch.object(agent, "_RUNTIME_FAILURE_RETRY_SECONDS", 0),
                patch.object(agent.traceback, "print_exc"),
            ):
                result = asyncio.run(loop.run())

            self.assertEqual(calls, 2)
            self.assertEqual(result["sessions"], 0)
            self.assertEqual(result["stop_reason"], "runtime_failure")
            self.assertEqual(loop.stop_checker.consecutive_errors, 2)

    def test_terminal_background_only_sessions_retry_without_event_wait(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer0",
                generation_id=0,
                task_prompt="do work",
                workspace=root,
                max_runtime_seconds=10,
                logs_dir=root / "logs",
                findings_dir=root / "findings",
                local_mode=True,
                allowed_tools=["Read"],
            )
            calls = 0

            async def terminal_session():
                nonlocal calls
                calls += 1
                return agent.AgentResult(
                    success=True,
                    output={
                        "text_outputs": [],
                        "tool_uses": [{"tool": "Task", "input": {}}],
                        "background_tasks": [{"task_id": "task-1", "status": "completed"}],
                        "terminal_background_only": True,
                    },
                    duration=0,
                    iteration_count=1,
                )

            async def forbidden_wait(*, productive=True):
                raise AssertionError("terminal-only session entered the 900-second event wait")

            loop._run_session = terminal_session  # type: ignore[method-assign]
            loop._wait_for_next_session_event = forbidden_wait  # type: ignore[method-assign]
            with patch.object(agent, "_EMPTY_SESSION_RETRY_SECONDS", 0):
                result = asyncio.run(loop.run())

            self.assertEqual(calls, 2)
            self.assertEqual(result["stop_reason"], "runtime_empty")

    def test_autonomous_session_bootstrap_retry_and_s3_sync(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer0",
                generation_id=0,
                task_prompt="do work",
                workspace=root,
                max_runtime_seconds=10,
                logs_dir=root / "logs",
                findings_dir=root / "findings",
                local_mode=True,
            )

            class FakeAgent:
                calls = 0

                async def execute(self, task):
                    FakeAgent.calls += 1
                    if FakeAgent.calls == 1:
                        return agent.AgentResult(
                            success=True,
                            output={"text_outputs": ["please tell me what to do"]},
                            duration=0,
                            iteration_count=0,
                        )
                    return agent.AgentResult(
                        success=True,
                        output={"text_outputs": ["used tools"]},
                        duration=1,
                        iteration_count=2,
                    )

            loop._create_agent = lambda *args, **kwargs: FakeAgent()  # type: ignore[method-assign]
            session_result = asyncio.run(loop._run_session())
            self.assertTrue(session_result.success)
            self.assertEqual(FakeAgent.calls, 2)
            self.assertTrue(any(path.name.endswith(".log") for path in (root / "logs").iterdir()))

            loop.findings_path.write_text("{}", encoding="utf-8")
            uploads: list[str] = []
            with patch(
                "praxist.infrastructure.s3_utils.upload_file_to_s3",
                side_effect=lambda **kwargs: uploads.append(kwargs["s3_key"]),
            ):
                asyncio.run(loop._sync_to_s3())
            self.assertTrue(any(key.endswith("findings.json") for key in uploads))
            self.assertTrue(any("/logs/" in key for key in uploads))

    def test_run_entrypoint_command_paths_are_patchable_and_json_stable(self) -> None:
        from praxist import run as run_cli

        def invoke(func, args):
            out = io.StringIO()
            err = io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    func(args)
                code = 0
            except SystemExit as exc:
                code = int(exc.code or 0)
            return code, out.getvalue(), err.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            fake_args = SimpleNamespace(
                workspace=str(workspace),
                run_dir=str(workspace / "run"),
                fake=True,
                task_path="",
                task="",
                runtime="",
                model_provider="",
                budget_policy="",
                credential_profile="fake",
                resolve_only=False,
            )
            with patch(
                "praxist.testing.fake_workflow_fixture.run_fake_workflow_fixture",
                return_value={"status": "succeeded"},
            ) as fake_run:
                code, out, _err = invoke(run_cli.cmd_run, fake_args)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["status"], "succeeded")
            fake_run.assert_called_once()

            missing_args = SimpleNamespace(
                workspace=str(workspace),
                run_dir="",
                fake=False,
                task_path="",
                task="",
                task_spec="",
            )
            code, _out, err = invoke(run_cli.cmd_run, missing_args)
            self.assertEqual(code, 2)
            self.assertIn("--task-path is required", err)

            deprecated_args = SimpleNamespace(
                workspace=str(workspace),
                run_dir="",
                fake=False,
                task_path="",
                task="task:x",
                task_spec="",
            )
            code, _out, err = invoke(run_cli.cmd_run, deprecated_args)
            self.assertEqual(code, 2)
            self.assertIn("no longer discovered", err)

            peer_args = SimpleNamespace(
                peer_id="gen0_peer0",
                generation_id=0,
                max_runtime=7,
                prompt_file="prompt.md",
                model="fake-model",
                local=True,
            )
            with patch("praxist.infrastructure.execute_autonomous.main") as peer_main:
                run_cli.cmd_peer(peer_args)
            peer_main.assert_called_once()
            self.assertEqual(os.environ["PEER_ID"], "gen0_peer0")
            self.assertEqual(os.environ["LOCAL_MODE"], "true")

            with patch(
                "praxist.core.replay.inspect_run",
                return_value={"success": True, "mode": "inspect"},
            ):
                code, out, _err = invoke(
                    run_cli.cmd_replay,
                    SimpleNamespace(
                        run_dir=str(workspace),
                        mode="inspect",
                        strict_tail=False,
                        allow_plugin_drift=False,
                        locked=False,
                    ),
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["mode"], "inspect")

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.parity.verify_research_loop_parity",
                return_value={"success": False, "errors": ["bad"]},
            ):
                code, out, _err = invoke(
                    run_cli.cmd_parity,
                    SimpleNamespace(
                        run_dir=str(workspace),
                        deliverables_dir="",
                        strict=True,
                        write_report=False,
                    ),
                )
            self.assertEqual(code, 1)
            self.assertFalse(json.loads(out)["success"])

            code, out, _err = invoke(run_cli.cmd_server, SimpleNamespace(port=8000))
            self.assertEqual(code, 1)
            self.assertIn("not yet implemented", out)

    def test_claude_runtime_adapter_normalizes_legacy_messages_and_failures(self) -> None:
        from praxist.core.credentials import CredentialRef
        from praxist.core.protocol import (
            AgentRunRequest,
            CachePolicy,
            EnvPolicy,
            ModelCallSpec,
            ToolPermissionSet,
        )
        from praxist.plugins.agent_runtimes.claude_sdk import adapter

        class TextBlock:
            text = "hello sk-ant-secret"

        class ToolUseBlock:
            name = "Bash"
            input = {"command": "echo sk-ant-secret"}

        class ThinkingBlock:
            thinking = "private thought"

        class AssistantMessage:
            content = [TextBlock(), ToolUseBlock(), ThinkingBlock()]

        class ResultMessage:
            result = {"status": "done", "secret": "sk-ant-secret"}

        class TaskNotificationMessage:
            task_id = "task-1"
            status = "completed"
            output_file = "/tmp/task.output"
            summary = "completed with exit code 0"

        class TaskUpdatedMessage:
            task_id = "task-3"
            status = None
            patch = {"status": "failed", "summary": "failed before notification"}

        completed_task = TaskNotificationMessage()
        failed_task = TaskNotificationMessage()
        failed_task.task_id = "task-2"
        failed_task.status = "failed"
        failed_task.output_file = "/tmp/empty-task.output"
        failed_task.summary = "failed with exit code 7"
        terminal_update = TaskUpdatedMessage()
        running_update = TaskUpdatedMessage()
        running_update.task_id = "task-4"
        running_update.patch = {"status": "running"}
        messages = [
            AssistantMessage(),
            completed_task,
            failed_task,
            terminal_update,
            running_update,
            ResultMessage(),
        ]
        formatted = adapter.format_legacy_message(messages[0], "agent")
        task_formatted = adapter.format_legacy_message(messages[1], "agent")
        output = adapter.extract_legacy_output(messages)
        self.assertIn("agent", formatted)
        self.assertIn("Bash", formatted)
        self.assertIn("task-1", task_formatted)
        self.assertEqual(output["background_tasks"][0]["status"], "completed")
        self.assertEqual(output["background_tasks"][1]["status"], "failed")
        self.assertEqual(output["background_tasks"][2]["task_id"], "task-3")
        self.assertEqual(output["background_tasks"][2]["status"], "failed")
        self.assertEqual(len(output["background_tasks"]), 3)
        self.assertIn("task-3", adapter.format_legacy_message(terminal_update, "agent"))
        self.assertNotIn("sk-ant-secret", json.dumps(output))
        self.assertTrue(adapter.is_billing_error("Payment required"))
        self.assertEqual(adapter._classify_legacy_failure("rate limit exceeded"), "rate_limited")

        credential = CredentialRef(
            scope="model_provider",
            provider="fake",
            target_ref="model_provider:fake",
            key_id="key",
            source="test",
        )
        request = AgentRunRequest(
            request_id="req/1",
            run_id="run",
            stage_id="research_loop",
            role_ref="task_role:peer",
            agent_runtime_ref="agent_runtime:claude_sdk",
            prompt_ref={},
            system_prompt_ref=None,
            cwd=str(Path.cwd()),
            model_profile_ref="profile",
            model_call=ModelCallSpec(
                profile_id="profile",
                provider_ref="model_provider:fake",
                api_format="fake",
                model="fake-model",
                parameters={},
                credential_ref=credential,
            ),
            tool_permissions=ToolPermissionSet(allowed_tools=["Bash"]),
            tool_servers=[],
            env_policy=EnvPolicy(),
            credential_ref=credential,
            credential_mode="single",
            budget_grant_id=None,
            artifact_scope="peer",
            timeout_seconds=1,
            cache_policy=CachePolicy(mode="runtime_managed", frozen_prefix_hash="hash"),
        )
        result = adapter._agent_run_result_from_legacy(
            request,
            adapter.LegacyAgentResult(
                success=False,
                output={"text_outputs": ["hi"], "tool_uses": [{"tool": "Bash", "input": {}}]},
                duration=1.0,
                iteration_count=1,
                error="timeout",
            ),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failover_reason, "timeout")
        self.assertEqual(result.tool_uses[0].tool_name, "Bash")
        self.assertIn("req_1_event", result.events[0].event_id)
        with patch.dict(os.environ, {"PRAXIST_CLAUDE_SETTING_SOURCES": "local,user,bad,local"}):
            self.assertEqual(adapter.claude_setting_sources_from_env(), ["local", "user"])
        self.assertIsInstance(adapter.create_runtime(), adapter.ClaudeSdkAgentRuntime)

    def test_evaluation_and_prior_work_server_mode_paths(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter as eval_adapter
        from praxist.plugins.tools.prior_work_tools import adapter as prior_adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = Path("/tmp") / f"ready_{os.getpid()}.txt"
            ready.write_text("prefix target suffix", encoding="utf-8")
            other = root / "other.txt"
            other.write_text("ok", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "LOCAL_STORE_DIR": str(root),
                    "LOCAL_MODE": "false",
                    "SERVER_URL": "http://server",
                    "LOCAL_FINDINGS_DIR": str(root / "findings"),
                    "PRIMARY_METRIC": "score",
                },
            ):
                server_metrics: list[dict[str, object]] = []

                async def fake_post(url, json_data, timeout):
                    server_metrics.append({"url": url, "payload": json_data, "timeout": timeout})
                    return {"ok": True}

                async def fake_get(url, params=None, timeout=0):
                    return [{"variant_name": "remote", "metrics": {"score": 1.0}}]

                with (
                    patch.object(eval_adapter, "async_http_post", fake_post),
                    patch.object(eval_adapter, "async_http_get", fake_get),
                ):
                    logged = asyncio.run(
                        eval_adapter._handle_log_experiment_metrics(
                            {
                                "run_id": "r",
                                "variant_name": "v",
                                "metrics": {"score": 1},
                                "peer_id": "Gen3_Peer2",
                            }
                        )
                    )
                    shared = asyncio.run(
                        eval_adapter._handle_share_finding(
                            {
                                "finding_type": "result",
                                "title": "T",
                                "content": "c" * 600,
                                "metrics": {"score": 1},
                                "peer_id": "gen3_peer2",
                                "links": [{"target_finding_id": "old", "edge_type": "supports"}],
                                "design_dimensions": {"axis": "value"},
                                "extra": {"peer_role": "builder"},
                            }
                        )
                    )
                    leaderboard = asyncio.run(
                        eval_adapter._handle_get_leaderboard({"generation": "3", "top_k": "2"})
                    )
                self.assertEqual(json.loads(logged["content"][0]["text"])["status"], "recorded")
                self.assertEqual(len(server_metrics), 2)
                self.assertEqual(
                    server_metrics[0]["payload"]["generation_id"],
                    3,
                )
                self.assertEqual(
                    json.loads(shared["content"][0]["text"])["summary"], "c" * 500 + "..."
                )
                self.assertEqual(
                    json.loads(leaderboard["content"][0]["text"])["mode"], "server_legacy"
                )

                all_ready = asyncio.run(
                    eval_adapter._handle_wait_for_file_impl(
                        {
                            "path": f"{ready},{ready},{other}",
                            "timeout_seconds": 1,
                            "poll_interval_seconds": 2,
                            "min_bytes": 1,
                            "contains_text": "target",
                            "mode": "any",
                        }
                    )
                )
                self.assertEqual(json.loads(all_ready["content"][0]["text"])["status"], "ready")
                too_long = asyncio.run(
                    eval_adapter._handle_wait_for_file_impl(
                        {"path": str(ready), "contains_text": "x" * 1025}
                    )
                )
                self.assertTrue(too_long["is_error"])

                findings_dir = root / "findings"
                findings_dir.mkdir(exist_ok=True)
                (findings_dir / "a.json").write_text(
                    json.dumps(
                        {
                            "finding_type": "result",
                            "variant_name": "a",
                            "metrics": {"score": 2, "tier": "T3"},
                            "generation_id": 3,
                        }
                    ),
                    encoding="utf-8",
                )
                (findings_dir / "b.json").write_text(
                    json.dumps(
                        {
                            "finding_type": "result",
                            "variant_name": "b",
                            "metrics": {"score": 3, "tier": "T1"},
                            "generation_id": 3,
                        }
                    ),
                    encoding="utf-8",
                )
                fs_board = json.loads(eval_adapter._filesystem_leaderboard(3, 10))
                # Filesystem fallback no longer interprets tier labels; entries
                # are ranked by the primary metric and filtered only by generic
                # promotion metadata when present.
                self.assertEqual(fs_board["entries"][0]["variant_name"], "b")

            with patch.dict(
                os.environ, {"WORKSPACE_DIR": str(root), "SERVER_URL": "http://server"}
            ):

                async def snapshot_meta(url, timeout=0):
                    return {"snapshot_s3_key": "key", "variant_name": "v", "metrics": {"m": 1}}

                with (
                    patch.object(prior_adapter, "async_http_get", snapshot_meta),
                    patch(
                        "praxist.infrastructure.s3_utils.download_snapshot_from_s3",
                        return_value=["file1", "file2"],
                    ),
                ):
                    downloaded = asyncio.run(
                        prior_adapter._handle_download_snapshot(
                            {"snapshot_id": "snap_1", "target_dir": ""}
                        )
                    )
                self.assertEqual(json.loads(downloaded["content"][0]["text"])["files_count"], 2)
                bad = asyncio.run(
                    prior_adapter._handle_download_snapshot(
                        {"snapshot_id": "../bad", "target_dir": ""}
                    )
                )
                self.assertTrue(bad["is_error"])
                self.assertIn("download_snapshot", prior_adapter.create_tool_plugin()["tool_names"])
                with (
                    patch.object(prior_adapter, "create_sdk_mcp_server", None),
                    patch.object(prior_adapter, "tool", None),
                    self.assertRaises(ImportError),
                ):
                    prior_adapter.create_prior_work_tools_server()

    def test_evaluation_tools_error_fallback_and_wait_paths(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter as eval_adapter
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store,
        )

        self.assertIsNone(eval_adapter._gen_id_from_peer_id(""))
        self.assertIsNone(eval_adapter._gen_id_from_peer_id("peer"))
        self.assertEqual(eval_adapter._gen_id_from_peer_id(" Gen12_Peer3 "), 12)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            store = root / "store"
            findings_dir = root / "findings"
            findings_dir.mkdir()
            with patch.dict(
                os.environ,
                {
                    "LOGS_DIR": str(logs),
                    "LOCAL_STORE_DIR": str(store),
                    "LOCAL_FINDINGS_DIR": str(findings_dir),
                    "LOCAL_MODE": "true",
                    "GENERATION_ID": "5",
                },
                clear=False,
            ):
                invalid_metrics = asyncio.run(
                    eval_adapter._handle_log_experiment_metrics(
                        {"run_id": "r", "variant_name": "v", "metrics": "{bad"}
                    )
                )
                self.assertTrue(invalid_metrics["is_error"])

                with patch.object(local_store, "insert_metric", side_effect=RuntimeError("db")):
                    logged = asyncio.run(
                        eval_adapter._handle_log_experiment_metrics(
                            {
                                "run_id": "r",
                                "variant_name": "v",
                                "metrics": {"score": 1},
                                "peer_id": "gen5_peer1",
                            }
                        )
                    )
                self.assertEqual(json.loads(logged["content"][0]["text"])["status"], "recorded")
                self.assertTrue((logs / "metrics_log.jsonl").exists())

                bad_type = asyncio.run(
                    eval_adapter._handle_share_finding(
                        {"finding_type": "bad", "title": "T", "content": "C"}
                    )
                )
                self.assertTrue(bad_type["is_error"])

                (store / "agendas").mkdir(parents=True)
                (store / "agendas" / "research_agenda_gen5.yaml").write_text(
                    "agenda", encoding="utf-8"
                )
                with (
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync.save_finding_to_dir",
                        side_effect=RuntimeError("fs"),
                    ),
                    patch.object(local_store, "insert_finding", side_effect=RuntimeError("db")),
                ):
                    shared = asyncio.run(
                        eval_adapter._handle_share_finding(
                            {
                                "finding_type": "result",
                                "title": "T",
                                "content": "C",
                                "metrics": "{bad",
                                "variant_name": "V",
                                "peer_id": "gen5_peer1",
                                "links": "{bad",
                                "design_dimensions": {"axis": 1},
                                "extra": json.dumps(["not", "dict"]),
                            }
                        )
                    )
                self.assertEqual(json.loads(shared["content"][0]["text"])["status"], "shared")

                with patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.FindingGraphBuilder.propose_edges_for",
                    side_effect=RuntimeError("graph"),
                ):
                    shared_graph_error = asyncio.run(
                        eval_adapter._handle_share_finding(
                            {
                                "finding_type": "result",
                                "title": "T2",
                                "content": "C2",
                                "metrics": "{}",
                                "variant_name": "V",
                                "peer_id": "gen5_peer1",
                            }
                        )
                    )
                self.assertEqual(
                    json.loads(shared_graph_error["content"][0]["text"])["status"],
                    "shared",
                )

            with patch.dict(
                os.environ,
                {
                    "LOCAL_MODE": "false",
                    "LOGS_DIR": str(logs),
                    "SERVER_URL": "http://server",
                },
                clear=False,
            ):

                async def failing_post(*args, **kwargs):
                    raise RuntimeError("server down")

                with patch.object(eval_adapter, "async_http_post", failing_post):
                    logged = asyncio.run(
                        eval_adapter._handle_log_experiment_metrics(
                            {
                                "run_id": "r",
                                "variant_name": "v",
                                "metrics": {"score": 1},
                                "peer_id": "peer",
                            }
                        )
                    )
                    shared = asyncio.run(
                        eval_adapter._handle_share_finding(
                            {
                                "finding_type": "result",
                                "title": "T",
                                "content": "C",
                                "metrics": "{}",
                            }
                        )
                    )
                self.assertEqual(json.loads(logged["content"][0]["text"])["status"], "recorded")
                self.assertEqual(json.loads(shared["content"][0]["text"])["status"], "shared")

            self.assertTrue(
                asyncio.run(eval_adapter._handle_wait_for_file_impl({"path": ""}))["is_error"]
            )
            self.assertTrue(
                asyncio.run(eval_adapter._handle_wait_for_file_impl({"path": " , "}))["is_error"]
            )
            self.assertTrue(
                asyncio.run(
                    eval_adapter._handle_wait_for_file_impl({"path": ",".join(["/tmp/x"] * 33)})
                )["is_error"]
            )
            self.assertTrue(
                asyncio.run(eval_adapter._handle_wait_for_file_impl({"path": "bad\npath"}))[
                    "is_error"
                ]
            )
            self.assertTrue(
                asyncio.run(
                    eval_adapter._handle_wait_for_file_impl(
                        {"path": str(root / "x.txt"), "timeout_seconds": 0}
                    )
                )["is_error"]
            )

            ready = Path("/tmp") / f"ready_{os.getpid()}.txt"
            ready.write_text("needle", encoding="utf-8")
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": "/"}, clear=False):
                ready_result = asyncio.run(
                    eval_adapter._handle_wait_for_file(
                        {
                            "path": str(ready),
                            "timeout_seconds": "bad",
                            "poll_interval_seconds": "bad",
                            "min_bytes": "bad",
                            "contains_text": "needle",
                            "mode": "invalid",
                        }
                    )
                )
            self.assertEqual(json.loads(ready_result["content"][0]["text"])["status"], "ready")

            big = root / "big.txt"
            big.write_bytes(b"a" * (4 * 1024 * 1024 + 16) + b"tail-needle")
            with patch.dict(os.environ, {"LOGS_DIR": str(root)}, clear=False):
                tail_result = asyncio.run(
                    eval_adapter._handle_wait_for_file_impl(
                        {
                            "path": str(big),
                            "timeout_seconds": 1,
                            "poll_interval_seconds": 2,
                            "contains_text": "tail-needle",
                        }
                    )
                )
            self.assertEqual(json.loads(tail_result["content"][0]["text"])["status"], "ready")

            run_path = (
                root / "experiments_tracking" / "run_2026-05-12_demo" / "gen_0" / "result.txt"
            )
            run_path.parent.mkdir(parents=True)
            run_path.write_text("", encoding="utf-8")
            (run_path.parents[1] / "gen_0" / "STOP_SIGNAL").write_text("stop", encoding="utf-8")
            with patch.dict(os.environ, {"LOGS_DIR": str(root)}, clear=False):
                stop_result = asyncio.run(
                    eval_adapter._handle_wait_for_file_impl(
                        {
                            "path": str(run_path),
                            "timeout_seconds": 1,
                            "poll_interval_seconds": 2,
                        }
                    )
                )
            self.assertEqual(
                json.loads(stop_result["content"][0]["text"])["status"],
                "aborted_by_stop_signal",
            )

            env_calls: list[str] = []

            def flaky_env(name: str, default: str) -> str:
                if not env_calls:
                    env_calls.append(name)
                    raise RuntimeError("env")
                return os.environ.get(name, default)

            with (
                patch.dict(
                    os.environ,
                    {
                        "LOCAL_STORE_DIR": str(store),
                        "LOCAL_FINDINGS_DIR": str(findings_dir),
                    },
                    clear=False,
                ),
                patch.object(eval_adapter, "_get_env", side_effect=flaky_env),
            ):
                fallback = json.loads(eval_adapter._sqlite_leaderboard(-1, 2))
            self.assertEqual(fallback["mode"], "filesystem_fallback")

            with patch.dict(os.environ, {"ANCHOR_METRICS": "{bad"}, clear=False):
                self.assertEqual(eval_adapter._parse_anchor_metrics_env(), [])
            with patch.dict(os.environ, {"ANCHOR_METRICS": json.dumps({"bad": True})}, clear=False):
                self.assertEqual(eval_adapter._parse_anchor_metrics_env(), [])
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(eval_adapter._wait_for_file_hard_cap_seconds(), 18 * 3600)
            with patch.dict(os.environ, {"PRAXIST_WAIT_FOR_FILE_MAX_SECONDS": "64800"}, clear=True):
                self.assertEqual(eval_adapter._wait_for_file_hard_cap_seconds(), 64800)
            with patch.dict(
                os.environ, {"PRAXIST_WAIT_FOR_FILE_MAX_SECONDS": "999999"}, clear=True
            ):
                self.assertEqual(eval_adapter._wait_for_file_hard_cap_seconds(), 24 * 3600)

            missing_fs = json.loads(eval_adapter._filesystem_leaderboard(-1, 2))
            self.assertEqual(missing_fs["mode"], "filesystem_fallback")
            fs_dir = root / "fs_findings"
            fs_dir.mkdir()
            (fs_dir / "bad.json").write_text("{bad", encoding="utf-8")
            (fs_dir / "t1.json").write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "variant_name": "skip",
                        "metrics": {"score": 9, "tier": "T1"},
                    }
                ),
                encoding="utf-8",
            )
            (fs_dir / "good.json").write_text(
                json.dumps(
                    {
                        "finding_type": "insight",
                        "variant_name": "good",
                        "title": "G",
                        "metrics": {"score": 2},
                        "details": {"promotion_eligible": True},
                        "generation_id": 5,
                    }
                ),
                encoding="utf-8",
            )
            (fs_dir / "missing_metric.json").write_text(
                json.dumps(
                    {
                        "finding_type": "result",
                        "variant_name": "z",
                        "metrics": {"score": True},
                        "generation_id": 5,
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "LOCAL_FINDINGS_DIR": str(fs_dir),
                    "PRIMARY_METRIC": "score",
                    "METRIC_DIRECTION": "minimize",
                },
                clear=False,
            ):
                fs_board = json.loads(eval_adapter._filesystem_leaderboard(5, 10))
            self.assertEqual(
                [entry["variant_name"] for entry in fs_board["entries"]], ["good", "z"]
            )

            def fake_server(name, tools):
                return {"name": name, "tools": tools}

            with (
                patch.object(eval_adapter, "create_sdk_mcp_server", fake_server),
                patch.object(eval_adapter, "tool", object()),
            ):
                self.assertEqual(
                    eval_adapter.create_evaluation_tools_server()["name"], "evaluation-tools"
                )

    def test_schedule_helpers_are_epoch_fraction_based(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import schedule

        self.assertEqual(schedule.epoch_fraction(-1, 10), 0.0)
        self.assertEqual(schedule.epoch_fraction(15, 10), 1.0)
        self.assertEqual(schedule.epoch_fraction(15, 10, clamp=False), 1.5)
        self.assertAlmostEqual(schedule.linear_schedule(0.25, start=0.0, end=2.0), 0.5)
        self.assertAlmostEqual(schedule.cosine_schedule(0.5, start=0.0, end=2.0), 1.0)
        self.assertEqual(
            schedule.peaked_schedule(0.0, start=1.0, peak=3.0, peak_at=0.0),
            schedule.cosine_schedule(0.0, start=3.0, end=1.0),
        )
        self.assertEqual(
            schedule.peaked_schedule(1.0, start=1.0, peak=3.0, peak_at=1.0),
            schedule.cosine_schedule(1.0, start=1.0, end=3.0),
        )
        self.assertAlmostEqual(
            schedule.warmup_then_schedule(
                0.1,
                warmup_fraction=0.2,
                warmup_start=0.0,
                base_start=1.0,
                base_end=0.0,
            ),
            0.5,
        )
        self.assertAlmostEqual(
            schedule.warmup_then_schedule(
                0.6,
                warmup_fraction=0.2,
                warmup_start=0.0,
                base_start=1.0,
                base_end=0.0,
                base_kind="linear",
            ),
            0.5,
        )
        hits = schedule.scan_for_step_anti_pattern(
            '"""total_steps in docs"""\n# max_steps comment\nx = total_steps + 1\n'
        )
        self.assertEqual(hits, [(3, "x = total_steps + 1", "total_steps")])

    def test_generation_signal_parser_preserves_numeric_zero_one(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
        )

        self.assertEqual(generation_boundary._coerce_signal_value("0"), 0)
        self.assertEqual(generation_boundary._coerce_signal_value("1"), 1)
        self.assertEqual(generation_boundary._coerce_signal_value("1.5"), 1.5)

    def test_generation_peer_mix_lane_classification_uses_token_boundaries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
        )

        self.assertIsNone(
            generation_boundary._is_constructive_payload(
                {"frontier_lane": "controller_improvements"}
            )
        )
        self.assertIsNone(
            generation_boundary._is_constructive_payload({"frontier_lane": "process_optimizer"})
        )
        self.assertIsNone(
            generation_boundary._is_constructive_payload({"frontier_lane": "control"})
        )
        self.assertIsNone(
            generation_boundary._is_constructive_payload({"frontier_lane": "process"})
        )
        self.assertFalse(
            generation_boundary._is_constructive_payload({"frontier_lane": "negative_control"})
        )
        self.assertFalse(
            generation_boundary._is_constructive_payload({"frontier_lane": "benchmark_candidate"})
        )
        loop = SimpleNamespace(
            run_dir=Path("/nonexistent"),
            task_spec=SimpleNamespace(
                evaluation=SimpleNamespace(
                    constructive_target_ratio=0.0,
                    maturity_policy={},
                ),
                generation_policy=SimpleNamespace(cohort_size=5),
            ),
        )
        mix = generation_boundary._generation_peer_mix(loop, gen_id=0, findings=[])
        self.assertEqual(mix["target_constructive_ratio"], 0.0)
        self.assertEqual(mix["recommended_next_gen_constructive_floor"], 0)

    def test_generation_peer_mix_can_be_fully_disabled(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
        )

        loop = SimpleNamespace(
            run_dir=Path("/nonexistent"),
            task_spec=SimpleNamespace(
                evaluation=SimpleNamespace(
                    constructive_peer_mix_enabled=False,
                    constructive_target_ratio=0.75,
                    maturity_policy={},
                ),
                generation_policy=SimpleNamespace(cohort_size=8),
            ),
        )

        self.assertEqual(
            generation_boundary._generation_peer_mix(loop, gen_id=4, findings=[]),
            {},
        )

    def test_generation_peer_mix_does_not_count_hard_non_mature_ratio_payloads(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
        )

        loop = SimpleNamespace(
            run_dir=Path("/nonexistent"),
            task_spec=SimpleNamespace(
                evaluation=SimpleNamespace(
                    constructive_target_ratio=0.5,
                    maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80},
                ),
                generation_policy=SimpleNamespace(cohort_size=4),
            ),
        )
        findings = [
            {
                "finding_type": "result",
                "metrics": {
                    "effort_ratio": 0.95,
                    "coverage_ratio": 0.95,
                    "frontier_lane": "solution_candidate",
                    "scored_complete": True,
                },
            },
            {
                "finding_type": "result",
                "metrics": {
                    "effort_ratio": 0.95,
                    "coverage_ratio": 0.95,
                    "frontier_lane": "solution_candidate",
                    "incomplete_eval": True,
                },
            },
            {
                "finding_type": "result",
                "metrics": {
                    "effort_ratio": 0.95,
                    "coverage_ratio": 0.95,
                    "frontier_lane": "solution_candidate",
                    "suspect_protocol": True,
                },
            },
            {
                "finding_type": "result",
                "metrics": {
                    "effort_ratio": 0.95,
                    "coverage_ratio": 0.95,
                    "frontier_lane": "solution_candidate",
                    "validation_only_result": True,
                },
            },
            {
                "finding_type": "result",
                "metrics": {
                    "effort_ratio": 0.95,
                    "coverage_ratio": 0.95,
                    "frontier_lane": "solution_candidate",
                    "artifact_signal_status": "late_after_generation_boundary",
                },
            },
            {
                "finding_type": "result",
                "metrics": {
                    "effort_ratio": 0.95,
                    "coverage_ratio": 0.95,
                    "frontier_lane": "solution_candidate",
                    "late_result_policy": "quarantined_signal",
                },
            },
        ]

        mix = generation_boundary._generation_peer_mix(loop, gen_id=0, findings=findings)

        self.assertEqual(mix["mature_result_total"], 1)
        self.assertEqual(mix["mature_constructive_count"], 1)

    def test_generation_boundary_deduplicates_and_quarantines_exact_result_snapshots(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
        )

        loop = SimpleNamespace(
            run_dir=None,
            task_spec=SimpleNamespace(
                evaluation=SimpleNamespace(
                    constructive_target_ratio=0.75,
                    maturity_policy={"min_effort_ratio": 0.75, "min_coverage_ratio": 0.8},
                ),
                generation_policy=SimpleNamespace(cohort_size=4),
            ),
        )
        mature = {
            "finding_type": "result",
            "metrics": {
                "child_variant_id": "candidate",
                "effort_ratio": 1.0,
                "coverage_ratio": 1.0,
                "scored_complete": True,
                "frontier_lane": "solution_candidate",
                "source_result_path": "results/shared.json",
                "source_result_sha256": "shared-sha",
            },
        }
        duplicate = {
            **mature,
            "metrics": {**mature["metrics"], "variant_name": "display_alias"},
        }
        quarantined = {
            "finding_type": "result",
            "metrics": {
                **mature["metrics"],
                "validation_only_result": True,
            },
        }

        mix = generation_boundary._generation_peer_mix(
            loop,
            gen_id=0,
            findings=[mature, duplicate, quarantined],
        )

        self.assertEqual(mix["mature_result_total"], 0)

        split_coordinate_signal = {
            "finding_type": "result",
            "metrics": {
                "validation_only_result": True,
                "source_result_path": "results/shared.json",
            },
            "extra": {"source_result_sha256": "shared-sha"},
        }
        mix = generation_boundary._generation_peer_mix(
            loop,
            gen_id=0,
            findings=[mature, split_coordinate_signal],
        )
        self.assertEqual(mix["mature_result_total"], 1)

    def test_generation_boundary_quarantines_label_only_preliminary_snapshot(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
        )

        loop = SimpleNamespace(
            run_dir=None,
            task_spec=SimpleNamespace(
                evaluation=SimpleNamespace(
                    constructive_target_ratio=0.75,
                    maturity_policy={},
                ),
                generation_policy=SimpleNamespace(cohort_size=2),
            ),
        )
        shared_source = {
            "child_variant_id": "candidate",
            "source_result_path": "results/shared.json",
            "source_result_sha256": "shared-sha",
        }
        preliminary = {
            "finding_type": "result",
            "metrics": {
                **shared_source,
                "evidence_stage": "preliminary",
                "frontier_lane": "solution_candidate",
            },
        }
        mature_alias = {
            "finding_type": "result",
            "metrics": {
                **shared_source,
                "scored_complete": True,
                "frontier_lane": "solution_candidate",
            },
        }

        mix = generation_boundary._generation_peer_mix(
            loop,
            gen_id=0,
            findings=[preliminary, mature_alias],
        )

        self.assertEqual(mix["mature_result_total"], 0)

    def test_generation_boundary_scopes_shared_snapshot_by_producer_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
        )

        loop = SimpleNamespace(
            run_dir=None,
            task_spec=SimpleNamespace(
                evaluation=SimpleNamespace(
                    constructive_target_ratio=0.75,
                    maturity_policy={},
                ),
                generation_policy=SimpleNamespace(cohort_size=2),
            ),
        )
        shared = {
            "source_result_path": "results/shared.json",
            "source_result_sha256": "shared-sha",
        }
        preliminary = {
            "finding_type": "result",
            "metrics": {
                **shared,
                "child_variant_id": "child-a",
                "evidence_stage": "preliminary",
                "scored_complete": False,
            },
        }
        mature = {
            "finding_type": "result",
            "metrics": {
                **shared,
                "child_variant_id": "child-b",
                "scored_complete": True,
                "frontier_lane": "solution_candidate",
            },
        }

        mix = generation_boundary._generation_peer_mix(
            loop,
            gen_id=0,
            findings=[preliminary, mature],
        )

        self.assertEqual(mix["mature_result_total"], 1)

    def test_generation_boundary_accepts_ratio_mature_legacy_inference(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
        )

        policy = {"min_effort_ratio": 0.75, "min_coverage_ratio": 0.80}
        inferred = {
            "finding_type": "result",
            "effort_ratio": 1.0,
            "coverage_ratio": 1.0,
            "scored_complete": False,
            "_inferred_scored_complete": True,
        }
        explicit = {**inferred, "_inferred_scored_complete": False}

        self.assertTrue(generation_boundary._is_mature_result_payload(inferred, policy))
        self.assertFalse(generation_boundary._is_mature_result_payload(explicit, policy))

    def test_runtime_environment_sidecars_and_generation_boundary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            generation_boundary,
            runtime_environment,
            sidecars,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = SimpleNamespace(
                evaluation=SimpleNamespace(
                    primary_metric="score",
                    direction="maximize",
                    anchor_metrics=[
                        {"name": "score", "direction": "maximize"},
                        ("loss", "minimize"),
                        {"name": "bad", "direction": "invalid"},
                    ],
                    requires_tier=True,
                    launch_guard={"enabled": False},
                )
            )
            with patch.dict(os.environ, {"BYPASS_GPU_GOVERNOR": "1"}, clear=False):
                anchors = runtime_environment.configure_runtime_environment(
                    task_spec=task_spec,
                    run_dir=root / "run",
                    findings_dir=root / "findings",
                    local_mode=True,
                )
                self.assertEqual([a["name"] for a in anchors], ["score", "loss", "bad"])
                self.assertEqual(os.environ["LOCAL_MODE"], "true")
                self.assertEqual(os.environ["REQUIRES_TIER"], "true")
                self.assertEqual(os.environ["PRAXIST_LAUNCH_GUARD_ENABLED"], "0")
                self.assertEqual(
                    os.environ["PRAXIST_BASELINE_CACHE_DIR"], str(root / "run" / "baseline_cache")
                )
                self.assertEqual(os.environ["AUTO_RESEARCH_RUN_DIR"], str(root / "run"))
                self.assertNotIn("BYPASS_GPU_GOVERNOR", os.environ)

            class FakeFindingsSync:
                instances: list[FakeFindingsSync] = []

                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.calls: list[str] = []
                    FakeFindingsSync.instances.append(self)

                def sync_once(self):
                    self.calls.append("sync")

                def start(self):
                    self.calls.append("start")

                def stop(self):
                    self.calls.append("stop")

            class FakeGraphMaintainer(FakeFindingsSync):
                pass

            class FakeStatusWriter:
                def __init__(self, run_dir, snapshot_fn):
                    self.run_dir = run_dir
                    self.snapshot_fn = snapshot_fn
                    self.calls: list[str] = []

                def start(self):
                    self.calls.append("start")

                def stop(self, *, exit_condition):
                    self.calls.append(exit_condition)

            loop = SimpleNamespace(
                local_mode=True,
                findings_dir=root / "findings",
                run_dir=root / "run",
                task_spec=task_spec,
                _findings_sync=None,
                _graph_maintainer=None,
                _status_writer=None,
                _build_status_snapshot=lambda: SimpleNamespace(to_dict=lambda: {}),
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync.FindingsSync",
                    FakeFindingsSync,
                ),
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.FindingGraphMaintainer",
                    FakeGraphMaintainer,
                ),
                patch.object(sidecars, "OrchestratorStatusWriter", FakeStatusWriter),
            ):
                sidecars.start_sidecars(loop)
                sidecars.stop_sidecars(loop, exit_condition="completed")
            self.assertEqual(loop._findings_sync.calls, ["sync", "start", "sync", "stop"])
            self.assertEqual(loop._findings_sync.kwargs["run_dir"], root / "run")
            self.assertTrue(loop._findings_sync.kwargs["materialize_result_artifacts"])
            self.assertEqual(loop._findings_sync.kwargs["primary_metric"], "score")
            self.assertEqual(loop._graph_maintainer.calls, ["sync", "start", "sync", "stop"])
            self.assertEqual(loop._status_writer.calls, ["start", "completed"])

            class FakeFrontier:
                def __init__(self):
                    self.promoted = []

                def get_summary(self):
                    return [{"id": "a", "design_dimensions": {"x": 1}}]

                def promote(self, gen_id, findings):
                    self.promoted = findings
                    return findings[:1]

            class FakePIAgent:
                async def run(self, *, completed_gen_id):
                    return SimpleNamespace(
                        success=False,
                        next_gen_id=completed_gen_id + 1,
                        error="offline",
                    )

            boundary_loop = SimpleNamespace(
                _strategy_for_gen=lambda gen_id: "pi_directed",
                frontier=FakeFrontier(),
                task_spec=SimpleNamespace(
                    evaluation=SimpleNamespace(diversity_dimensions=[{"name": "x"}]),
                    research_memory=SimpleNamespace(enabled=True),
                    generation_policy=SimpleNamespace(max_generations=3),
                ),
                _collect_findings_for_generation=lambda gen_id: [
                    {
                        "id": "f1",
                        "variant_name": "v",
                        "metrics": {"score": 1.0},
                        "design_dimensions": {"x": 1},
                    }
                ],
                _update_research_memory_post_gen=lambda **kwargs: None,
                _graph_maintainer=SimpleNamespace(
                    sync_once_blocking=lambda timeout: {"status": "timeout"}
                ),
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    boundary_loop,
                    gen_id=1,
                    pi_agent=FakePIAgent(),
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )
            self.assertEqual(
                boundary_loop.frontier.promoted[0]["metrics"]["diversity_overlap_status"], "clone"
            )

            with self.assertRaises(RuntimeError):
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        boundary_loop,
                        gen_id=1,
                        pi_agent=FakePIAgent(),
                        pi_cfg=SimpleNamespace(strict=True),
                    )
                )

    def test_generation_cohort_runs_peers_and_writes_postgen_results(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        class FakeAgentLoop:
            instances: list[dict[str, object]] = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.__class__.instances.append(dict(kwargs))

            async def run(self):
                peer_id = self.kwargs["peer_id"]
                if peer_id.endswith("peer1"):
                    raise RuntimeError("peer failed")
                return {"peer_id": peer_id, "success": True}

        class FakeTrigger:
            fired = False

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def wait_until_fire(self, abort_event):
                return None

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="not_yet")

            def fire(self, snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                marker = self.kwargs["gen_dir"] / "STOP_SIGNAL_POSTGEN"
                marker.write_text(snapshot.reason, encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = SimpleNamespace(
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(cohort_size=2, per_generation_hours=0.001),
                    synthesis_trigger=SimpleNamespace(
                        enabled=False,
                        min_findings=1,
                        min_interval_minutes=1,
                        max_interval_minutes=2,
                        min_contributing_peers=1,
                        poll_interval_seconds=1,
                    ),
                    agent=SimpleNamespace(premium_mode=False, reasoning_effort="high"),
                ),
                run_dir=root / "run",
                workspace=root,
                base_template=root / "base.jinja2",
                task_prompt_path=root / "task.jinja2",
                gen_template=root / "gen.jinja2",
                findings_dir=root / "findings",
                model="fake",
                local_mode=True,
                mcp_servers={},
                _peer_allowed_tools=["Read"],
                plugin_registry=None,
                _findings_sync=SimpleNamespace(sync_once=lambda: None),
                _build_prompt_context=lambda gen_id, peer, cohort: {"peer": peer},
                _persist_prompt_layout_artifacts=lambda **kwargs: kwargs["manifest"],
            )
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    return_value=("prompt", {"layout": "ok"}),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))
            self.assertEqual(results[0]["success"], True)
            self.assertFalse(results[1]["success"])
            self.assertEqual(
                [kwargs["reasoning_effort"] for kwargs in FakeAgentLoop.instances],
                ["high", "high"],
            )
            gen_dir = root / "run" / "gen_0"
            self.assertTrue((gen_dir / "generation_results.json").exists())
            self.assertTrue((root / "run" / "results" / "gen_0").is_dir())
            self.assertEqual(
                (gen_dir / "STOP_SIGNAL_POSTGEN").read_text(encoding="utf-8"), "not_yet"
            )

    def test_generation_cohort_records_late_protected_jobs_without_stopping_generation(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        trigger_kwargs: list[dict[str, object]] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                trigger_kwargs.append(kwargs)
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                self.fired = True
                self.kwargs["gen_dir"].joinpath("STOP_SIGNAL").write_text("stop", encoding="utf-8")

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="not_yet")

            def fire(self, snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                self.kwargs["gen_dir"].joinpath("STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = SimpleNamespace(
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(cohort_size=2, per_generation_hours=0.001),
                    synthesis_trigger=SimpleNamespace(
                        enabled=True,
                        min_findings=1,
                        min_interval_minutes=1,
                        max_interval_minutes=2,
                        min_contributing_peers=3,
                        poll_interval_seconds=1,
                    ),
                    agent=SimpleNamespace(premium_mode=False),
                ),
                run_dir=root / "run",
                workspace=root,
                base_template=root / "base.jinja2",
                task_prompt_path=root / "task.jinja2",
                gen_template=root / "gen.jinja2",
                findings_dir=root / "findings",
                model="fake",
                local_mode=True,
                mcp_servers={},
                _peer_allowed_tools=["Read"],
                plugin_registry=None,
                _findings_sync=SimpleNamespace(sync_once=lambda: None),
                _build_prompt_context=lambda gen_id, peer, cohort: {"peer": peer},
                _persist_prompt_layout_artifacts=lambda **kwargs: kwargs["manifest"],
            )
            late_job = SimpleNamespace(
                peer_id="gen0_peer1",
                pid=12345,
                tag="long_t2_eval",
                eta_seconds=86400,
                started_at="2026-07-03T00:00:00+00:00",
            )
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    return_value=("prompt", {"layout": "ok"}),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.protected_pids.list_active_jobs",
                    return_value=[late_job],
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            self.assertEqual(trigger_kwargs[0]["min_contributing_peers"], 2)
            late = [
                item for item in results if item.get("status") == "late_quarantined_protected_job"
            ]
            self.assertEqual(len(late), 1)
            self.assertIs(late[0]["success"], True)
            self.assertIs(late[0]["promotion_eligible"], False)
            self.assertEqual(late[0]["pid"], 12345)
            self.assertEqual(late[0]["late_result_policy"], "quarantined_signal")
            persisted = json.loads(
                (root / "run" / "gen_0" / "generation_results.json").read_text(encoding="utf-8")
            )
            persisted_late = [
                item for item in persisted if item.get("status") == "late_quarantined_protected_job"
            ]
            self.assertEqual(
                persisted_late,
                late,
            )

    def test_late_protected_job_result_is_quarantined_then_ingested_for_later_agents(
        self,
    ) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            cohort_runner,
            protected_pids,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            findings_sync,
            local_store,
        )

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False

            async def wait_until_fire(self, abort_event):
                self.fired = True
                self.kwargs["gen_dir"].joinpath("STOP_SIGNAL").write_text("stop", encoding="utf-8")

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="not_yet")

            def fire(self, snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                self.kwargs["gen_dir"].joinpath("STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            loop = SimpleNamespace(
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(cohort_size=1, per_generation_hours=0.001),
                    synthesis_trigger=SimpleNamespace(
                        enabled=True,
                        min_findings=1,
                        min_interval_minutes=1,
                        max_interval_minutes=2,
                        min_contributing_peers=1,
                        poll_interval_seconds=1,
                    ),
                    agent=SimpleNamespace(premium_mode=False),
                ),
                run_dir=run_dir,
                workspace=root,
                base_template=root / "base.jinja2",
                task_prompt_path=root / "task.jinja2",
                gen_template=root / "gen.jinja2",
                findings_dir=run_dir / "shared_findings",
                model="fake",
                local_mode=True,
                mcp_servers={},
                _peer_allowed_tools=["Read"],
                plugin_registry=None,
                _findings_sync=SimpleNamespace(sync_once=lambda: None),
                _build_prompt_context=lambda gen_id, peer, cohort: {"peer": peer},
                _persist_prompt_layout_artifacts=lambda **kwargs: kwargs["manifest"],
            )
            protected_pids.register_pid(
                os.getpid(),
                peer_id="gen0_peer0",
                tag="spectral_risk_mdd_penalty_dual_eval",
                eta_seconds=3600,
                run_dir=run_dir,
                allow_duplicate=True,
            )
            try:
                with (
                    patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                    patch.object(
                        cohort_runner,
                        "resolve_prompt_with_layout",
                        return_value=("prompt", {"layout": "ok"}),
                    ),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                        FakeTrigger,
                    ),
                ):
                    results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))
            finally:
                protected_pids.unregister_pid(os.getpid(), peer_id="gen0_peer0", run_dir=run_dir)

            late_records = [
                item for item in results if item.get("status") == "late_quarantined_protected_job"
            ]
            self.assertEqual(len(late_records), 1)
            self.assertEqual(late_records[0]["tag"], "spectral_risk_mdd_penalty_dual_eval")
            self.assertEqual(late_records[0]["late_result_policy"], "quarantined_signal")
            self.assertIs(late_records[0]["success"], True)
            self.assertIs(late_records[0]["promotion_eligible"], False)

            gen_dir = run_dir / "gen_0"
            boundary = gen_dir / "generation_boundary.json"
            boundary.write_text(
                json.dumps({"generation_id": 0, "status": "complete"}),
                encoding="utf-8",
            )
            time.sleep(0.02)
            result_dir = run_dir / "results" / "spectral_risk_mdd_penalty_dual_eval"
            result_dir.mkdir(parents=True)
            (result_dir / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "spectral_risk_mdd_penalty_dual_eval",
                        "generation_id": 0,
                        "current_aggregate": {
                            "score": 1.25,
                            "scored_complete": True,
                            "promotion_eligible": True,
                        },
                        "n_eval_cells": 28,
                        "scored_complete": True,
                        "promotion_eligible": True,
                    }
                ),
                encoding="utf-8",
            )

            sync = findings_sync.FindingsSync(
                run_dir / "shared_findings",
                poll_interval=0,
                local_mode=True,
                run_dir=run_dir,
                materialize_result_artifacts=True,
                result_scoring_metric_keys=("score",),
            )
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root / "store")}):
                touched = sync.sync_once_blocking(timeout=0.1)
                rows = local_store.get_all_findings()

            self.assertGreaterEqual(touched, 1)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["variant_name"], "spectral_risk_mdd_penalty_dual_eval")
            self.assertEqual(row["generation_id"], 0)
            metrics = row["metrics"]
            self.assertEqual(metrics["score"], 1.25)
            self.assertIs(metrics["late_after_generation_boundary"], True)
            self.assertEqual(metrics["artifact_signal_status"], "late_after_generation_boundary")
            self.assertIs(metrics["promotion_eligible"], False)
            self.assertIs(metrics["clean_promotion_eligible"], False)
            self.assertIs(metrics["excluded_from_durable_frontier"], True)
            self.assertEqual(metrics["exclusion_reason"], "late_after_generation_boundary")
            self.assertTrue(any((run_dir / "shared_findings").glob("*.json")))

    def test_generation_cohort_waits_for_assessment_trigger_before_postgen(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        created: list[dict[str, object]] = []

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(kwargs)

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = False
                self.closing = False
                self.assessment_started = True
                self.adaptive_policy = SimpleNamespace(drain_grace_minutes=0.001)

            async def wait_until_fire(self, abort_event):
                await asyncio.sleep(0.01)
                self.fired = True
                (self.kwargs["gen_dir"] / "STOP_SIGNAL").write_text("stop", encoding="utf-8")

            async def evaluate_async(self):
                return SimpleNamespace(fired=False, reason="should_not_postgen")

            def fire(self, snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                (self.kwargs["gen_dir"] / "STOP_SIGNAL_POSTGEN").write_text(
                    snapshot.reason,
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = SimpleNamespace(
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(cohort_size=1, per_generation_hours=0.001),
                    synthesis_trigger=SimpleNamespace(
                        enabled=True,
                        min_findings=1,
                        min_interval_minutes=1,
                        max_interval_minutes=2,
                        min_contributing_peers=1,
                        poll_interval_seconds=1,
                    ),
                    agent=SimpleNamespace(premium_mode=False),
                ),
                run_dir=root / "run",
                workspace=root,
                base_template=root / "base.jinja2",
                task_prompt_path=root / "task.jinja2",
                gen_template=root / "gen.jinja2",
                findings_dir=root / "findings",
                model="fake",
                local_mode=True,
                mcp_servers={},
                _peer_allowed_tools=["Read"],
                plugin_registry=None,
                _findings_sync=SimpleNamespace(sync_once=lambda: None),
                _build_prompt_context=lambda gen_id, peer, cohort: {"peer": peer},
                _persist_prompt_layout_artifacts=lambda **kwargs: kwargs["manifest"],
            )
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    return_value=("prompt", {"layout": "ok"}),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            gen_dir = root / "run" / "gen_0"
            self.assertEqual(results, [{"peer_id": "gen0_peer0", "success": True}])
            self.assertTrue((gen_dir / "STOP_SIGNAL").exists())
            self.assertFalse((gen_dir / "STOP_SIGNAL_POSTGEN").exists())
            self.assertEqual(created[0]["closing_signal_path"], gen_dir / "CLOSING_SIGNAL")

    def test_generation_cohort_preserves_postgen_marker_on_trigger_eval_error(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        self.assertTrue(cohort_runner._protected_job_belongs_to_generation("gen3_peer1", 3))
        self.assertTrue(cohort_runner._protected_job_belongs_to_generation("gen3/peer1", 3))
        self.assertTrue(cohort_runner._protected_job_belongs_to_generation("gen3-peer1", 3))
        self.assertFalse(cohort_runner._protected_job_belongs_to_generation("gen2_peer1", 3))
        trigger = SimpleNamespace(adaptive_policy=SimpleNamespace(drain_grace_minutes=0.0))
        self.assertEqual(cohort_runner._closing_trigger_wait_seconds(trigger, []), 120.0)
        self.assertEqual(
            cohort_runner._closing_trigger_wait_seconds(
                trigger,
                [SimpleNamespace(eta_seconds=900)],
            ),
            1020.0,
        )
        self.assertEqual(
            cohort_runner._closing_trigger_wait_seconds(
                trigger,
                [SimpleNamespace(eta_seconds=99999)],
            ),
            3600.0,
        )
        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend.protected_pids.list_active_jobs",
            side_effect=RuntimeError("pid ledger"),
        ):
            self.assertEqual(
                cohort_runner._active_generation_work_count([], run_dir=Path("."), gen_id=0),
                0,
            )

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def run(self):
                return {"peer_id": self.kwargs["peer_id"], "success": True}

        class FakeTrigger:
            fired = False
            closing = False

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def wait_until_fire(self, abort_event):
                raise AssertionError("disabled trigger should not wait")

            async def evaluate_async(self):
                raise RuntimeError("postgen eval failed")

            def fire(self, snapshot):
                self.fired = True

            def write_postgen_marker(self, snapshot):
                raise AssertionError("exception path writes fallback marker directly")

        class FakeGems:
            def logical_generation(self, gen_id):
                return gen_id + 10

            def prompt_context(self, gen_id):
                return {"cycle_index": 4, "generation": gen_id}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = SimpleNamespace(
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(cohort_size=1, per_generation_hours=0.001),
                    synthesis_trigger=SimpleNamespace(
                        enabled=False,
                        min_findings=1,
                        min_interval_minutes=1,
                        max_interval_minutes=2,
                        min_contributing_peers=1,
                        poll_interval_seconds=1,
                    ),
                    agent=SimpleNamespace(premium_mode=False),
                ),
                run_dir=root / "run",
                workspace=root,
                base_template=root / "base.jinja2",
                task_prompt_path=root / "task.jinja2",
                gen_template=root / "gen.jinja2",
                findings_dir=root / "findings",
                model="fake",
                local_mode=True,
                mcp_servers={},
                _peer_allowed_tools=["Read"],
                plugin_registry=None,
                _findings_sync=None,
                gems=FakeGems(),
                _build_prompt_context=lambda gen_id, peer, cohort: {"peer": peer},
                _persist_prompt_layout_artifacts=lambda **kwargs: kwargs["manifest"],
            )
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    return_value=("prompt", {"layout": "ok"}),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
                patch.object(Path, "unlink", side_effect=OSError("stale")),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            gen_dir = root / "run" / "gen_0"
            marker = gen_dir / "STOP_SIGNAL_POSTGEN"
            self.assertEqual(results, [{"peer_id": "gen0_peer0", "success": True}])
            self.assertIn("postgen_eval_raised", marker.read_text(encoding="utf-8"))
            self.assertEqual(os.environ["PRAXIST_LOGICAL_GENERATION_ID"], "10")
            self.assertEqual(os.environ["PRAXIST_GEMS_CYCLE"], "4")

    def test_generation_cohort_cancels_peers_after_trigger_drain(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        class FakeAgentLoop:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def run(self):
                await asyncio.sleep(30)
                return {"peer_id": self.kwargs["peer_id"], "success": True}

        class FakeTrigger:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fired = True
                self.closing = False
                self.active_counts: list[int] = []

            async def wait_until_fire(self, abort_event):
                self.active_counts.append(self.kwargs["cohort_active_peers_callback"]())

            async def evaluate_async(self):
                raise AssertionError("already fired")

            def fire(self, snapshot):
                raise AssertionError("already fired")

            def write_postgen_marker(self, snapshot):
                raise AssertionError("already fired")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = SimpleNamespace(
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(cohort_size=1, per_generation_hours=0.001),
                    synthesis_trigger=SimpleNamespace(
                        enabled=True,
                        min_findings=1,
                        min_interval_minutes=1,
                        max_interval_minutes=2,
                        min_contributing_peers=1,
                        poll_interval_seconds=1,
                    ),
                    agent=SimpleNamespace(premium_mode=False),
                ),
                run_dir=root / "run",
                workspace=root,
                base_template=root / "base.jinja2",
                task_prompt_path=root / "task.jinja2",
                gen_template=root / "gen.jinja2",
                findings_dir=root / "findings",
                model="fake",
                local_mode=True,
                mcp_servers={},
                _peer_allowed_tools=["Read"],
                plugin_registry=None,
                _findings_sync=None,
                _build_prompt_context=lambda gen_id, peer, cohort: {"peer": peer},
                _persist_prompt_layout_artifacts=lambda **kwargs: kwargs["manifest"],
            )
            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakeAgentLoop),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    return_value=("prompt", {"layout": "ok"}),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
                patch.object(cohort_runner, "_PEER_DRAIN_GRACE_SECONDS", 0),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            self.assertEqual(len(results), 1)
            self.assertTrue((root / "run" / "gen_0" / "generation_results.json").exists())

    def test_research_memory_firewall_source_metrics_and_usage_helpers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            research_memory_update,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            context_firewall,
            metrics_logger,
            source_resolver,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.telemetry import (
            usage_tracker,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            huge = "x" * 5000
            pack = SimpleNamespace(
                pack_id="pack",
                built_at="now",
                panel_mode="mini",
                target_decisions=["decide"],
                shared_core={"summary": huge, "coverage_matrix_digest": huge},
                private_packs={
                    "builder": [{"id": str(i), "interpretation": {"short": huge}} for i in range(5)]
                },
                audit={"ok": True},
            )
            fitted = context_firewall.fit_pack_to_budget(pack, "mini")
            self.assertLessEqual(len(fitted["private_packs"]["builder"]), 2)
            self.assertTrue(
                context_firewall.forbid_raw_history({"raw_history": ["should not pass"]})
            )
            self.assertGreater(context_firewall.estimate_tokens({"文本": "内容"}), 0)

            json_path = run_dir / "finding.json"
            json_path.write_text(json.dumps({"id": "f1", "metric": 1}), encoding="utf-8")
            yaml_path = run_dir / "agenda.yaml"
            yaml_path.write_text("agenda: ok\n", encoding="utf-8")
            log_path = run_dir / "log.txt"
            log_path.write_text("hello", encoding="utf-8")
            resolver = source_resolver.SourceResolver(run_dir)
            self.assertEqual(
                resolver.resolve({"finding_path": "finding.json"})["content"]["id"],
                "f1",
            )
            self.assertEqual(
                resolver.resolve({"agenda_path": "agenda.yaml"})["content"]["agenda"], "ok"
            )
            self.assertEqual(resolver.resolve({"raw_log_path": "log.txt"})["content"], "hello")
            self.assertIn("rejected", resolver.resolve({"finding_path": "/etc/passwd"})["error"])
            self.assertIn("no resolvable", resolver.resolve({})["error"])

            metrics_logger.log_synthesis_metrics(
                run_dir,
                generation_id=0,
                prompt_size_bytes=100,
                pack_size_bytes=80,
                n_evidence_cards=2,
                citation_coverage=0.5,
                negative_evidence_ratio=0.2,
                panel_mode="mini",
                pi_count=2,
                audit_warnings=1,
                audit_blocking=0,
                extra={"shared_core_id": "abc"},
            )
            metrics_logger.log_synthesis_metrics(
                run_dir,
                generation_id=0,
                prompt_size_bytes=150,
                pack_size_bytes=90,
                n_evidence_cards=3,
                citation_coverage=0.6,
                negative_evidence_ratio=0.1,
                panel_mode="full",
                pi_count=3,
                audit_warnings=0,
                audit_blocking=0,
            )
            self.assertTrue((run_dir / "research_memory" / "metrics_gen0.prev.json").exists())
            self.assertIsInstance(metrics_logger.compute_prompt_kb_slope(run_dir), float)

            research_memory_update.update_research_memory_post_gen(
                run_dir=run_dir,
                gen_id=1,
                findings=[
                    {
                        "id": "neg1",
                        "title": "failed candidate",
                        "notes": "negative evidence",
                        "finding_type": "error",
                        "extra": json.dumps({"peer_role": "skeptic"}),
                    }
                ],
                promoted=[
                    {
                        "id": "p1",
                        "variant_name": "v1",
                        "metrics": {
                            "quality_score": {"mean": 4.0},
                            "failure_rate": 8.0,
                            "failure_rate_direction": "minimize",
                        },
                    },
                    {
                        "id": "p2",
                        "variant_name": "v2",
                        "metrics": {
                            "quality_score": {"mean": 3.0},
                            "failure_rate": 5.0,
                            "failure_rate_direction": "minimize",
                        },
                    },
                ],
            )
            self.assertTrue(
                (run_dir / "research_memory" / "ledgers" / "negative_evidence_ledger.yaml").exists()
            )
            from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.frontier_delta_ledger import (
                FrontierDeltaLedger,
            )

            latest_axes = FrontierDeltaLedger(run_dir).latest_per_axis()
            self.assertEqual(
                latest_axes["quality_score"].data["current_anchor"]["variant"],
                "v1",
            )
            self.assertEqual(
                latest_axes["failure_rate"].data["current_anchor"]["variant"],
                "v2",
            )
            self.assertEqual(
                latest_axes["failure_rate"].data["current_anchor"]["direction"],
                "minimize",
            )

            tracker = usage_tracker.UsageTracker(stats_dir=run_dir / "logs")
            tracker.record_skill("SkillA", duration_ms=1.0, metadata={"k": "v"})
            tracker.record_mcp_tool("mcp__server__tool", success=False)
            stats = tracker.get_stats()
            self.assertEqual(stats["skills"]["SkillA"]["successes"], 1)
            self.assertEqual(stats["tools"]["mcp__server__tool"]["failures"], 1)
            with patch.dict(os.environ, {"LOGS_DIR": str(run_dir / "logs")}):
                usage_tracker.reset_tracker()
                self.assertIs(usage_tracker.get_tracker(), usage_tracker.get_tracker())

    def test_research_memory_ignores_auxiliary_counts_but_keeps_declared_axes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory_update import (
            _configured_metric_directions,
            _numeric_axes_for_entry,
        )

        inferred = _numeric_axes_for_entry(
            {
                "metrics": {
                    "quality_score": 0.8,
                    "n_batches": 12,
                    "sample_count": 240,
                    "runtime_seconds": 30,
                }
            }
        )
        self.assertEqual(inferred, {"quality_score": (0.8, "maximize")})

        legacy_global_direction = _numeric_axes_for_entry(
            {
                "metric_direction": "minimize",
                "metrics": {"generic_objective": 0.8},
            }
        )
        self.assertEqual(
            legacy_global_direction,
            {"generic_objective": (0.8, "minimize")},
        )
        self.assertEqual(
            _configured_metric_directions(
                SimpleNamespace(
                    primary_metric="quality_score",
                    direction="maximize",
                    anchor_metrics=["secondary_score"],
                    frontier_lanes=[],
                )
            ),
            {"quality_score": "maximize", "secondary_score": "maximize"},
        )

        declared = _numeric_axes_for_entry(
            {
                "primary_metric_name": "failure_count",
                "primary_metric_value": 2,
                "metrics": {
                    "failure_count": 2,
                    "failure_count_direction": "minimize",
                },
            }
        )
        self.assertEqual(declared["failure_count"], (2.0, "minimize"))

        lane_declared = _numeric_axes_for_entry(
            {
                "lane_metric_name": "latency_seconds",
                "lane_metric_value": 2,
                "lane_metric_direction": "minimize",
                "metrics": {
                    "latency_seconds": 2,
                    "lane_metric_value": 2,
                },
            }
        )
        self.assertEqual(lane_declared, {"latency_seconds": (2.0, "minimize")})

    def test_research_memory_uses_task_directions_and_manifest_finding_identity(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.frontier_delta_ledger import (
            FrontierDeltaLedger,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory_update import (
            update_research_memory_post_gen,
        )

        evaluation = SimpleNamespace(
            primary_metric="quality_score",
            direction="maximize",
            anchor_metrics=[
                *((f"unused_metric_{index}", "maximize") for index in range(30)),
                ("stability_penalty", "minimize"),
                ("evaluation_count", "maximize"),
                ("ambiguous_metric", "maximize"),
                ("ambiguous_metric", "minimize"),
            ],
            frontier_lanes=[],
        )
        promoted = [
            {
                "finding_id": "finding-a",
                "variant_name": "candidate_a",
                "metric_direction": "maximize",
                "metrics": {
                    "quality_score": 9.0,
                    "stability_penalty": 7.0,
                    "evaluation_count": 20,
                    "undeclared_value": 100.0,
                    "undeclared_value_direction": "maximize",
                    "ambiguous_metric": 3.0,
                    "ambiguous_metric_direction": "maximize",
                    "metric_direction": "maximize",
                },
            },
            {
                "finding_id": "finding-b",
                "variant_name": "candidate_b",
                "metric_direction": "maximize",
                "metrics": {
                    "quality_score": 8.0,
                    "stability_penalty": 2.0,
                    "evaluation_count": 30,
                    "undeclared_value": 200.0,
                    "undeclared_value_direction": "maximize",
                    "ambiguous_metric": 4.0,
                    "ambiguous_metric_direction": "maximize",
                    "metric_direction": "maximize",
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_research_memory_post_gen(
                run_dir=run_dir,
                gen_id=0,
                findings=[],
                promoted=promoted,
                evaluation=None,
            )
            update_research_memory_post_gen(
                run_dir=run_dir,
                gen_id=0,
                findings=[],
                promoted=promoted,
                evaluation=evaluation,
            )
            latest = FrontierDeltaLedger(run_dir).latest_per_axis()

        self.assertEqual(set(latest), {"quality_score", "stability_penalty", "evaluation_count"})
        self.assertEqual(latest["quality_score"].data["current_anchor"]["finding_id"], "finding-a")
        self.assertEqual(latest["quality_score"].data["previous_anchor"], {})
        self.assertEqual(
            latest["stability_penalty"].data["current_anchor"]["finding_id"],
            "finding-b",
        )
        self.assertEqual(
            latest["stability_penalty"].data["current_anchor"]["direction"],
            "minimize",
        )
        self.assertEqual(
            latest["evaluation_count"].data["current_anchor"]["finding_id"],
            "finding-b",
        )

        with tempfile.TemporaryDirectory() as tmp:
            empty_run_dir = Path(tmp)
            update_research_memory_post_gen(
                run_dir=empty_run_dir,
                gen_id=0,
                findings=[],
                promoted=promoted,
                evaluation=evaluation,
            )
            update_research_memory_post_gen(
                run_dir=empty_run_dir,
                gen_id=0,
                findings=[],
                promoted=[],
                evaluation=evaluation,
            )
            self.assertEqual(FrontierDeltaLedger(empty_run_dir).latest_per_axis(), {})

    def test_http_utils_and_tool_usage_hook_offline_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.hooks import (
            log_tool_usage,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import http_utils

        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            http_utils.get_server_url()
        with patch.dict(os.environ, {"SERVER_URL": "http://server/"}):
            self.assertEqual(http_utils.get_server_url(), "http://server")
        self.assertEqual(http_utils.validate_safe_identifier("abc-123", "id"), "abc-123")
        with self.assertRaises(ValueError):
            http_utils.validate_safe_identifier("../bad", "id")
        with tempfile.TemporaryDirectory() as tmp:
            allowed = Path(tmp)
            inside = allowed / "file.txt"
            inside.write_text("x", encoding="utf-8")
            self.assertEqual(
                http_utils.validate_safe_path(str(inside), "path", allowed_base=str(allowed)),
                str(inside.resolve()),
            )
            with self.assertRaises(ValueError):
                http_utils.validate_safe_path("/etc/passwd", "path", allowed_base=str(allowed))

        class FakeResponse:
            def __init__(self, payload, status_code=200):
                self._payload = payload
                self.status_code = status_code

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError("http error")

        class FakeRequests:
            @staticmethod
            def post(url, json, headers, timeout):
                return FakeResponse({"ok": True, "url": url, "json": json})

            @staticmethod
            def get(url, params, headers, timeout):
                return FakeResponse({"ok": True, "params": params})

        with (
            patch.object(http_utils, "HAS_HTTPX", False),
            patch.object(http_utils, "HAS_REQUESTS", True),
            patch.object(http_utils, "requests", FakeRequests, create=True),
        ):
            self.assertEqual(
                asyncio.run(http_utils.async_http_post("http://x", {"a": 1}))["json"]["a"],
                1,
            )
            self.assertEqual(
                asyncio.run(http_utils.async_http_get("http://x", {"q": 1}))["params"]["q"],
                1,
            )
        with (
            patch.object(http_utils, "HAS_HTTPX", False),
            patch.object(http_utils, "HAS_REQUESTS", False),
            self.assertRaises(ImportError),
        ):
            asyncio.run(http_utils.async_http_get("http://x"))

        self.assertEqual(
            log_tool_usage.parse_mcp_tool_name("mcp__server__tool"), ("server", "tool")
        )
        self.assertEqual(log_tool_usage.parse_mcp_tool_name("Read"), (None, None))
        with patch.object(sys, "stdin", SimpleNamespace(read=lambda: '{"tool_name": "Read"}')):
            self.assertEqual(log_tool_usage.read_hook_input()["tool_name"], "Read")
        with patch.object(sys, "stdin", SimpleNamespace(read=lambda: "{bad")):
            self.assertIsNone(log_tool_usage.read_hook_input())

        class FakeTracker:
            def __init__(self):
                self.calls = []

            def record_mcp_tool(self, **kwargs):
                self.calls.append(("mcp", kwargs))

            def record_skill(self, **kwargs):
                self.calls.append(("skill", kwargs))

        tracker = FakeTracker()
        fake_usage_module = SimpleNamespace(get_tracker=lambda: tracker)
        fake_timing_module = SimpleNamespace(get_duration_ms=lambda **kwargs: 12.5)
        with patch.dict(
            sys.modules,
            {
                "praxist.plugins.workflow_stages.research_loop.backend.telemetry.usage_tracker": fake_usage_module,
                "praxist.plugins.workflow_stages.research_loop.backend.telemetry.tool_timing": fake_timing_module,
            },
        ):
            payload = json.dumps(
                {
                    "tool_name": "mcp__server__tool",
                    "tool_input": {"x": 1},
                    "tool_response": json.dumps({"error": "bad"}),
                    "session_id": "s",
                }
            )
            with patch.object(sys, "stdin", SimpleNamespace(read=lambda: payload)):
                log_tool_usage.main()
        self.assertEqual(tracker.calls[0][0], "mcp")
        self.assertFalse(tracker.calls[0][1]["success"])

    def test_finding_graph_query_handlers_use_local_store(self) -> None:
        from praxist.plugins.tools.finding_graph_query import adapter
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store,
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"LOCAL_STORE_DIR": str(Path(tmp) / "store")}),
        ):
            local_store.init_db()
            local_store.insert_finding(
                {
                    "id": "f1",
                    "finding_type": "result",
                    "title": "A",
                    "content": "a" * 1000,
                    "metrics": {"score": 1},
                    "variant_name": "v1",
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                }
            )
            local_store.insert_finding(
                {
                    "id": "f2",
                    "finding_type": "hypothesis",
                    "title": "B",
                    "content": "b",
                    "metrics": {},
                    "variant_name": "v2",
                    "peer_id": "gen0_peer1",
                    "generation_id": 0,
                }
            )
            local_store.insert_edge(
                {
                    "src_finding_id": "f1",
                    "dst_finding_id": "f2",
                    "edge_type": "supports",
                    "confidence": 0.9,
                    "created_by": "unit",
                    "rationale": "offline",
                }
            )
            missing = asyncio.run(adapter._handle_get_finding_neighbors({}))
            self.assertTrue(missing["is_error"])
            neighbors = asyncio.run(
                adapter._handle_get_finding_neighbors(
                    {"finding_id": "f1", "edge_types": '["supports"]'}
                )
            )
            text = neighbors["content"][0]["text"]
            self.assertIn("neighbor_findings", text)
            self.assertIn("truncated", text)
            subgraph = asyncio.run(adapter._handle_get_finding_subgraph({"finding_id": "f1"}))
            self.assertIn("nodes", subgraph["content"][0]["text"])
            unlinked = asyncio.run(adapter._handle_get_unlinked_recent_findings({"hours": 1}))
            self.assertIn("unlinked_findings", unlinked["content"][0]["text"])
            self.assertEqual(adapter._trim_finding({"id": "x", "content": "abc"})["content"], "abc")
            with (
                patch.object(adapter, "create_sdk_mcp_server", None),
                patch.object(adapter, "tool", None),
                self.assertRaises(ImportError),
            ):
                adapter.create_finding_graph_query_server()
            self.assertIn("get_finding_neighbors", adapter.create_tool_plugin()["tool_names"])

    def test_event_wait_fallback_stop_and_candidate_roots(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import event_wait

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "missing" / "file.txt"
            roots = event_wait._candidate_watch_roots([file_path, root])
            self.assertIn(root.resolve(), roots)

            result = asyncio.run(
                event_wait.wait_for_filesystem_event(
                    [root / "does_not_exist" / "x"],
                    timeout_seconds=0.01,
                    fallback_interval_seconds=0.01,
                    stop_check=lambda: True,
                    stop_check_interval_seconds=0.01,
                )
            )
            self.assertEqual(result.reason, "no_watch_paths")
            with patch.object(event_wait, "_InotifyWaiter", side_effect=OSError("no inotify")):
                result = asyncio.run(
                    event_wait.wait_for_filesystem_event(
                        [root],
                        timeout_seconds=0.01,
                        fallback_interval_seconds=0.01,
                        stop_check=lambda: True,
                        stop_check_interval_seconds=0.01,
                    )
                )
            self.assertEqual(result.reason, "stop")

    def test_orchestrator_status_writer_fallbacks_and_helpers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import orchestrator_status

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            snap = orchestrator_status.OrchestratorSnapshot(
                run_started_at="start",
                updated_at="",
                run_dir=str(run_dir),
                task_id="task",
                task_name="Task",
                current_generation=1,
                max_generations=2,
                cohort_size=3,
                strategy="pi_directed",
                generations_completed=1,
            )
            writer = orchestrator_status.OrchestratorStatusWriter(
                run_dir,
                lambda: snap,
                interval_seconds=100,
            )
            writer._write_once("in_progress")
            self.assertEqual(
                json.loads(writer.status_path.read_text(encoding="utf-8"))["task_id"],
                "task",
            )
            with patch.object(orchestrator_status, "cgroup_memory_ratio", return_value=0.93):
                writer._check_cgroup_memory_pressure()
            shutdown = run_dir / "ORCHESTRATOR_SHUTDOWN"
            shutdown_text = shutdown.read_text(encoding="utf-8")
            self.assertIn("reason=cgroup_memory_pressure", shutdown_text)
            self.assertIn("ratio=0.930", shutdown_text)
            shutdown.write_text("reason=operator\n", encoding="utf-8")
            with patch.object(orchestrator_status, "cgroup_memory_ratio", return_value=0.95):
                writer._check_cgroup_memory_pressure()
            self.assertEqual(shutdown.read_text(encoding="utf-8"), "reason=operator\n")
            self.assertIn(
                "artifact_index",
                json.loads(writer.status_path.read_text(encoding="utf-8"))[
                    "operator_manifest_paths"
                ],
            )
            failing = orchestrator_status.OrchestratorStatusWriter(
                run_dir / "bad",
                lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                interval_seconds=100,
            )
            failing._write_once("in_progress")
            self.assertIn(
                "last_snapshot_error",
                json.loads(failing.status_path.read_text(encoding="utf-8")),
            )
            failing.stop(exit_condition="error")
            final = json.loads(failing.final_status_path.read_text(encoding="utf-8"))
            self.assertEqual(final["exit_condition"], "error")

            cgroup = run_dir / "cgroup"
            cgroup.mkdir()
            (cgroup / "memory.current").write_text("92\n", encoding="utf-8")
            (cgroup / "memory.max").write_text("100\n", encoding="utf-8")
            self.assertEqual(orchestrator_status.cgroup_memory_ratio(cgroup), 0.92)
            (cgroup / "memory.max").write_text("max\n", encoding="utf-8")
            self.assertIsNone(orchestrator_status.cgroup_memory_ratio(cgroup))
            (cgroup / "memory.current").unlink()
            (cgroup / "memory.max").unlink()
            v1 = cgroup / "memory"
            v1.mkdir()
            (v1 / "memory.usage_in_bytes").write_text("81\n", encoding="utf-8")
            (v1 / "memory.limit_in_bytes").write_text("100\n", encoding="utf-8")
            self.assertEqual(orchestrator_status.cgroup_memory_ratio(cgroup), 0.81)

        self.assertIn(
            "top-2",
            orchestrator_status.describe_promotion_criteria(2, "score", "accuracy", "maximize"),
        )
        self.assertIn(
            "lane-based frontier promotion",
            orchestrator_status.describe_promotion_criteria(
                4,
                "primary_metric",
                "future_fitness",
                "maximize",
                frontier_lanes=[
                    {"name": "confirmed", "k": 2},
                    {"name": "incubator", "k": 10},
                ],
            ),
        )
        self.assertEqual(
            orchestrator_status.describe_promotion_blocker(
                variants_with_primary_metric=3,
                variants_above_baseline=1,
                promote_top_k=4,
                lane_based=True,
            ),
            "",
        )
        self.assertEqual(
            orchestrator_status.describe_promotion_blocker(
                variants_with_primary_metric=0,
                variants_above_baseline=0,
                promote_top_k=2,
            ),
            "no variants have reported the primary metric yet",
        )
        self.assertIn(
            "none above baseline",
            orchestrator_status.describe_promotion_blocker(
                variants_with_primary_metric=3,
                variants_above_baseline=0,
                promote_top_k=2,
            ),
        )
        self.assertEqual(
            orchestrator_status.describe_promotion_blocker(
                variants_with_primary_metric=3,
                variants_above_baseline=3,
                promote_top_k=2,
            ),
            "",
        )

    def test_status_snapshot_reads_generation_boundary_control_telemetry(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import status_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gen0 = run_dir / "gen_0"
            gen0.mkdir()
            (gen0 / "generation_boundary.json").write_text(
                json.dumps(
                    {
                        "stop_audit": {
                            "trigger_reason": "mature_quorum",
                            "mature_result_peers": 2,
                            "required_mature_result_peers": 2,
                        },
                        "peer_mix": {
                            "mature_constructive_ratio": 0.5,
                            "target_constructive_ratio": 0.75,
                        },
                    }
                ),
                encoding="utf-8",
            )
            task_spec = SimpleNamespace(
                task_id="task",
                task_name="Task",
                baselines=[],
                generation_policy=SimpleNamespace(
                    cohort_size=7,
                    max_generations=3,
                    promote_top_k=2,
                    promote_criterion="score",
                ),
                evaluation=SimpleNamespace(
                    primary_metric="score",
                    direction="maximize",
                    frontier_lanes=[],
                ),
                synthesis_trigger=SimpleNamespace(mature_quorum_fraction=0.4),
            )
            frontier = SimpleNamespace(
                current_generation=lambda: [],
                get_summary=lambda: [],
                get_manifest=lambda: {"cumulative_top": [], "lane_frontiers": {}},
            )
            snapshot = status_snapshot.build_orchestrator_status_snapshot(
                run_started_at="2026-07-07T00:00:00+00:00",
                run_dir=run_dir,
                task_spec=task_spec,
                frontier=frontier,
                current_gen=1,
                gens_completed=1,
                frontier_strategy="auto",
                strategy_for_gen=lambda _gen: "auto",
                findings=[],
            )

            task_spec.evaluation.constructive_peer_mix_enabled = False
            disabled_snapshot = status_snapshot.build_orchestrator_status_snapshot(
                run_started_at="2026-07-07T00:00:00+00:00",
                run_dir=run_dir,
                task_spec=task_spec,
                frontier=frontier,
                current_gen=1,
                gens_completed=1,
                frontier_strategy="auto",
                strategy_for_gen=lambda _gen: "auto",
                findings=[],
            )

        self.assertEqual(snapshot.last_stop_audit["trigger_reason"], "mature_quorum")
        self.assertEqual(snapshot.last_peer_mix["mature_constructive_ratio"], 0.5)
        self.assertEqual(snapshot.mature_quorum_required, 2)
        self.assertEqual(disabled_snapshot.last_peer_mix, {})

    def test_synthesis_trigger_evaluates_fires_and_tracks_postgen_stop(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import synthesis_trigger
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            local_store,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / "store"
            gen_dir = root / "gen0"
            gen_dir.mkdir()
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(store)}):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "f1",
                        "finding_type": "result",
                        "title": "result",
                        "content": "",
                        "metrics": {"score": 1.0},
                        "variant_name": "v",
                        "peer_id": "gen0_peer1",
                        "generation_id": 0,
                    }
                )
                # #75 batch 9: ``SynthesisTrigger`` no longer reads
                # ``LOCAL_STORE_DIR`` from os.environ — the store path
                # is an explicit kwarg now. Test inherits the original
                # intent (store separate from run_dir) by passing it.
                trigger = synthesis_trigger.SynthesisTrigger(
                    run_dir=root,
                    gen_dir=gen_dir,
                    gen_id=0,
                    gen_start_time=time.time() - 3600,
                    min_findings=1,
                    min_interval_minutes=1,
                    max_interval_minutes=100,
                    min_contributing_peers=1,
                    poll_interval_seconds=1,
                    pre_eval_sync_callback=lambda: None,
                    local_store_dir=store,
                )
                snapshot = trigger.evaluate()
                self.assertTrue(snapshot.fired)
                self.assertEqual(snapshot.reason, "info_density")
                trigger.fire(snapshot)
                trigger.fire(snapshot)
                self.assertTrue((gen_dir / "STOP_SIGNAL").exists())
                self.assertTrue(asyncio.run(trigger.evaluate_async()).fired)

                safety = synthesis_trigger.SynthesisTrigger(
                    run_dir=root,
                    gen_dir=gen_dir,
                    gen_id=1,
                    gen_start_time=time.time() - 3600,
                    min_findings=99,
                    min_interval_minutes=1,
                    max_interval_minutes=1,
                    min_contributing_peers=1,
                    local_store_dir=store,
                )
                self.assertEqual(safety.evaluate().reason, "safety_cap")


if __name__ == "__main__":
    unittest.main()
