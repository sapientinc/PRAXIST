from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend.generation_loop import (
    GenerationLoop,
)
from praxist.plugins.workflow_stages.research_loop.backend.run_lifecycle import (
    evaluate_run_stop_gate,
    max_generations_stop_report,
    write_external_stop_signal,
    write_run_stop_report,
)
from praxist.task_spec import RunLifecyclePolicy, load_task_spec


class RunLifecycleGateContractsTest(unittest.TestCase):
    def test_gate_reports_continue_wall_clock_and_stop_signal_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            spec = SimpleNamespace(
                run_lifecycle=SimpleNamespace(
                    max_wall_clock_hours=1.0,
                    stop_signal_path="run_control/stop.json",
                )
            )

            decision = evaluate_run_stop_gate(
                task_spec=spec,
                run_dir=run_dir,
                run_started_at_seconds=100.0,
                now_seconds=200.0,
                next_generation=1,
                generations_completed=1,
            )
            self.assertFalse(decision.should_stop)
            self.assertEqual(decision.reason, "continue")

            decision = evaluate_run_stop_gate(
                task_spec=spec,
                run_dir=run_dir,
                run_started_at_seconds=100.0,
                now_seconds=3700.0,
                next_generation=1,
                generations_completed=1,
            )
            self.assertTrue(decision.should_stop)
            self.assertEqual(decision.exit_condition, "wall_clock_limit")
            self.assertEqual(decision.reason, "wall_clock_limit")

            signal = run_dir / "run_control" / "stop.json"
            signal.parent.mkdir()
            signal.write_text(
                json.dumps({"reason": "task_success", "source": "task_local"}),
                encoding="utf-8",
            )
            decision = evaluate_run_stop_gate(
                task_spec=spec,
                run_dir=run_dir,
                run_started_at_seconds=100.0,
                now_seconds=3700.0,
                next_generation=1,
                generations_completed=1,
            )
            self.assertTrue(decision.should_stop)
            self.assertEqual(decision.exit_condition, "external_stop_signal")
            self.assertEqual(decision.reason, "task_success")
            self.assertEqual(decision.signal_evidence["source"], "task_local")

            report = write_run_stop_report(run_dir, decision)
            semantic_report = {
                key: value
                for key, value in report.items()
                if key not in {"run_dir", "stop_signal_path"}
            }
            text = json.dumps(semantic_report).lower()
            for forbidden in ("mle", "gold", "submission", "leaderboard"):
                self.assertNotIn(forbidden, text)

    def test_malformed_signal_still_stops_with_parse_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            signal = run_dir / "run_control" / "stop.json"
            signal.parent.mkdir(parents=True)
            signal.write_text("{not-json", encoding="utf-8")
            spec = SimpleNamespace(
                run_lifecycle=SimpleNamespace(
                    max_wall_clock_hours=None,
                    stop_signal_path="run_control/stop.json",
                )
            )

            decision = evaluate_run_stop_gate(
                task_spec=spec,
                run_dir=run_dir,
                run_started_at_seconds=10.0,
                now_seconds=20.0,
                next_generation=0,
                generations_completed=0,
            )
            self.assertTrue(decision.should_stop)
            self.assertEqual(decision.exit_condition, "external_stop_signal")
            self.assertIn("parse_error", decision.signal_evidence)
            self.assertTrue(decision.warnings)

    def test_default_orchestrator_shutdown_signal_stops_run_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "ORCHESTRATOR_SHUTDOWN").write_text(
                "signal=15\nat=123\n",
                encoding="utf-8",
            )
            spec = SimpleNamespace(
                run_lifecycle=SimpleNamespace(
                    max_wall_clock_hours=None,
                    stop_signal_path="",
                )
            )

            decision = evaluate_run_stop_gate(
                task_spec=spec,
                run_dir=run_dir,
                run_started_at_seconds=10.0,
                now_seconds=20.0,
                next_generation=3,
                generations_completed=3,
            )
            self.assertTrue(decision.should_stop)
            self.assertEqual(decision.exit_condition, "external_stop_signal")
            self.assertEqual(decision.reason, "orchestrator_shutdown")
            self.assertEqual(decision.source, "orchestrator_shutdown")
            self.assertEqual(decision.signal_evidence["signal"], "15")
            self.assertEqual(decision.stop_signal_path, str(run_dir / "ORCHESTRATOR_SHUTDOWN"))

    def test_external_stop_signal_writer_is_generic_and_run_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            payload = write_external_stop_signal(
                run_dir,
                {"reason": "task_success", "source": "task_local"},
                stop_signal_path="run_control/stop.json",
            )
            signal = run_dir / "run_control" / "stop.json"
            self.assertTrue(signal.is_file())
            self.assertEqual(payload["schema_version"], "praxist.run_stop_signal.v1")

            decision = evaluate_run_stop_gate(
                task_spec=SimpleNamespace(
                    run_lifecycle=SimpleNamespace(
                        max_wall_clock_hours=None,
                        stop_signal_path="run_control/stop.json",
                    )
                ),
                run_dir=run_dir,
                run_started_at_seconds=10.0,
                now_seconds=20.0,
                next_generation=1,
                generations_completed=1,
            )
            self.assertTrue(decision.should_stop)
            self.assertEqual(decision.exit_condition, "external_stop_signal")
            self.assertEqual(decision.signal_evidence["reason"], "task_success")

            text = signal.read_text(encoding="utf-8").lower()
            for forbidden in ("mle", "gold", "submission", "leaderboard"):
                self.assertNotIn(forbidden, text)

            with self.assertRaises(ValueError):
                write_external_stop_signal(
                    run_dir,
                    {"reason": "outside"},
                    stop_signal_path="../stop.json",
                )
            with self.assertRaises(ValueError):
                write_external_stop_signal(
                    run_dir,
                    {"reason": "absolute"},
                    stop_signal_path=str(Path(tmp) / "outside.json"),
                )

            absolute_signal = run_dir / "run_control" / "absolute_stop.json"
            write_external_stop_signal(
                run_dir,
                {"reason": "absolute_run_local"},
                stop_signal_path=str(absolute_signal),
            )
            absolute_decision = evaluate_run_stop_gate(
                task_spec=SimpleNamespace(
                    run_lifecycle=SimpleNamespace(
                        max_wall_clock_hours=None,
                        stop_signal_path=str(absolute_signal),
                    )
                ),
                run_dir=run_dir,
                run_started_at_seconds=10.0,
                now_seconds=20.0,
                next_generation=1,
                generations_completed=1,
            )
            self.assertTrue(absolute_decision.should_stop)
            self.assertEqual(
                absolute_decision.signal_evidence["reason"],
                "absolute_run_local",
            )

    def test_relative_run_dir_accepts_an_absolute_run_local_stop_signal(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp).relative_to(Path.cwd())
            run_dir = root / "run"
            run_dir.mkdir()
            signal = (run_dir / "run_control" / "stop.json").resolve()

            write_external_stop_signal(
                run_dir,
                {"reason": "absolute_run_local"},
                stop_signal_path=str(signal),
            )

            self.assertTrue(signal.is_file())
            decision = evaluate_run_stop_gate(
                task_spec=SimpleNamespace(
                    run_lifecycle=SimpleNamespace(
                        max_wall_clock_hours=None,
                        stop_signal_path=str(signal),
                    )
                ),
                run_dir=run_dir,
                run_started_at_seconds=10.0,
                now_seconds=20.0,
                next_generation=1,
                generations_completed=1,
            )
            self.assertTrue(decision.should_stop)
            self.assertEqual(decision.signal_evidence["reason"], "absolute_run_local")

    def test_stop_signal_paths_must_be_run_local_plain_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            outside = root / "outside_stop.json"
            outside.write_text(json.dumps({"reason": "outside"}), encoding="utf-8")

            decision = evaluate_run_stop_gate(
                task_spec=SimpleNamespace(
                    run_lifecycle=SimpleNamespace(
                        max_wall_clock_hours=None,
                        stop_signal_path=str(outside),
                    )
                ),
                run_dir=run_dir,
                run_started_at_seconds=10.0,
                now_seconds=20.0,
                next_generation=1,
                generations_completed=1,
            )
            self.assertFalse(decision.should_stop)
            self.assertIsNone(decision.stop_signal_path)

            linked_parent = run_dir / "linked_control"
            linked_parent.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                write_external_stop_signal(
                    run_dir,
                    {"reason": "symlink_parent"},
                    stop_signal_path="linked_control/stop.json",
                )

    def test_max_generations_report_shape(self) -> None:
        decision = max_generations_stop_report(
            run_dir=Path("/tmp/run"),
            max_generations=2,
            generations_completed=2,
            run_started_at_seconds=100.0,
            now_seconds=130.0,
        )
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.exit_condition, "max_generations")
        self.assertEqual(decision.next_generation, 2)
        self.assertEqual(decision.elapsed_seconds, 30.0)

    def test_task_spec_loads_run_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_yaml = root / "task.yaml"
            task_yaml.write_text(
                "\n".join(
                    [
                        "task_id: demo",
                        "task_name: Demo",
                        "run_lifecycle:",
                        "  max_wall_clock_hours: 24",
                        "  stop_signal_path: run_control/stop.json",
                    ]
                ),
                encoding="utf-8",
            )

            spec = load_task_spec(str(task_yaml))
            self.assertEqual(spec.run_lifecycle.max_wall_clock_hours, 24.0)
            self.assertEqual(spec.run_lifecycle.stop_signal_path, "run_control/stop.json")

    def test_generation_loop_stops_before_generation_when_gate_matches(self) -> None:
        async def fail_generation(_loop, _gen_id):
            raise AssertionError("generation should not start")

        with tempfile.TemporaryDirectory() as tmp:
            spec = load_task_spec("templates/tasks/toy_math/task.yaml")
            spec = replace(
                spec,
                generation_policy=replace(spec.generation_policy, max_generations=2, cohort_size=1),
                run_lifecycle=RunLifecyclePolicy(max_wall_clock_hours=0.0),
                pi_agent=replace(spec.pi_agent, enabled=False),
            )
            run_dir = Path(tmp) / "run"
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.run_generation_cohort",
                    fail_generation,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.configure_runtime_environment"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.initialize_local_store_if_needed"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.validate_baseline_cache_for_run"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.start_sidecars"
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.sidecars.stop_sidecars"
                ),
            ):
                loop = GenerationLoop(
                    task_spec=spec,
                    workspace=Path(tmp),
                    run_dir=run_dir,
                    local_mode=True,
                    tool_server_refs=[],
                )
                summary = asyncio.run(loop.run())

            self.assertEqual(summary["generations_completed"], 0)
            self.assertEqual(summary["exit_condition"], "wall_clock_limit")
            self.assertEqual(summary["stop_reason"], "wall_clock_limit")
            report = json.loads((run_dir / "run_stop_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["next_generation"], 0)
