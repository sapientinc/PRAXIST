from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class AtomicIoAndTelemetryContractsTest(unittest.TestCase):
    def test_atomic_json_cas_conflicts_and_timing_cleanup(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.telemetry import (
            tool_timing,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import atomic_io

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "state.json"
            written = atomic_io.atomic_write_json(state_path, {"hello": "world"})
            self.assertEqual(written, state_path)
            self.assertEqual(atomic_io.read_json_with_version(root / "missing.json"), (None, 0))
            self.assertEqual(atomic_io.read_json_with_version(state_path)[1], 0)

            created = atomic_io.atomic_write_json_cas(
                state_path,
                {"value": 1},
                expected_version=0,
                peer_id="peer/unsafe",
            )
            self.assertEqual(created.status, atomic_io.CAS_OK)
            self.assertEqual(created.new_version, 1)
            self.assertEqual(json.loads(state_path.read_text(encoding="utf-8"))["_version"], 1)

            conflict = atomic_io.atomic_write_json_cas(
                state_path,
                {"value": 2},
                expected_version=0,
                peer_id="peer/unsafe",
            )
            self.assertEqual(conflict.status, atomic_io.CAS_CONFLICT_DUMPED)
            self.assertIsNotNone(conflict.conflict_path)
            assert conflict.conflict_path is not None
            self.assertIn("peer_unsafe", conflict.conflict_path.name)
            self.assertTrue(conflict.conflict_path.exists())

            no_dump = atomic_io.atomic_write_json_cas(
                state_path,
                {"value": 3},
                expected_version=0,
                dump_on_conflict=False,
            )
            self.assertEqual(no_dump.status, atomic_io.CAS_CONFLICT)
            with self.assertRaises(TypeError):
                atomic_io.atomic_write_json_cas(state_path, [], expected_version=1)  # type: ignore[arg-type]

            bad_path = root / "bad.json"
            bad_path.write_text("{bad", encoding="utf-8")
            self.assertEqual(atomic_io.read_json_with_version(bad_path), (None, 0))
            list_path = root / "list.json"
            list_path.write_text("[1, 2]", encoding="utf-8")
            self.assertEqual(atomic_io.read_json_with_version(list_path), ([1, 2], 0))

            with patch.dict(os.environ, {"LOGS_DIR": str(root / "logs")}, clear=False):
                with patch.object(tool_timing.time, "time", return_value=1000.0):
                    tool_timing.record_start_time("session", "mcp__x", '{"a": 1}')
                with patch.object(tool_timing.time, "time", return_value=1001.25):
                    self.assertEqual(
                        tool_timing.get_duration_ms("session", "mcp__x", '{"a": 1}'),
                        1250.0,
                    )
                self.assertIsNone(tool_timing.get_duration_ms("session", "missing", ""))
                timing_file = root / "logs" / ".tool_timing.json"
                timing_file.write_text("{bad", encoding="utf-8")
                self.assertEqual(tool_timing._load_timing_state(timing_file), {})
                timing_file.write_text(
                    json.dumps(
                        {
                            "fresh": {"start_time": 4990.0},
                            "old": {"start_time": 1.0},
                            "malformed": {},
                        }
                    ),
                    encoding="utf-8",
                )
                with patch.object(tool_timing.time, "time", return_value=5000.0):
                    self.assertEqual(set(tool_timing._load_timing_state(timing_file)), {"fresh"})


class HooksAndRunPodContractsTest(unittest.TestCase):
    def test_hooks_are_best_effort_and_runpod_errors_are_classified(self) -> None:
        from praxist.infrastructure import runpod
        from praxist.plugins.workflow_stages.research_loop.backend.hooks import (
            log_tool_start,
            sync_to_s3,
        )

        with patch.object(sys, "stdin", io.StringIO("")):
            self.assertIsNone(log_tool_start.read_hook_input())
        with patch.object(sys, "stdin", io.StringIO("{bad")):
            self.assertIsNone(log_tool_start.read_hook_input())

        recorded: list[dict[str, str]] = []
        with (
            patch.object(sys, "stdin", io.StringIO(json.dumps({"tool_name": "Read"}))),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.telemetry.tool_timing.record_start_time",
                side_effect=lambda **kwargs: recorded.append(kwargs),
            ),
        ):
            log_tool_start.main()
        self.assertEqual(recorded, [])

        with (
            patch.object(
                sys,
                "stdin",
                io.StringIO(
                    json.dumps(
                        {
                            "tool_name": "mcp__evaluation-tools__log",
                            "tool_input": {"b": 2},
                            "session_id": "s1",
                        }
                    )
                ),
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.telemetry.tool_timing.record_start_time",
                side_effect=lambda **kwargs: recorded.append(kwargs),
            ),
        ):
            log_tool_start.main()
        self.assertEqual(recorded[-1]["session_id"], "s1")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (root / "findings.json").write_text("{}", encoding="utf-8")
            (logs / "usage_stats.json").write_text("{}", encoding="utf-8")
            (logs / "session_001.log").write_text("log", encoding="utf-8")
            uploads: list[tuple[str, str]] = []
            with (
                patch.dict(
                    os.environ,
                    {
                        "PEER_ID": "peer",
                        "RUN_ID": "run",
                        "S3_BUCKET": "bucket",
                        "S3_RESULTS_PREFIX": "prefix/",
                        "LOGS_DIR": str(logs),
                    },
                    clear=False,
                ),
                patch(
                    "praxist.infrastructure.s3_utils.upload_file_to_s3",
                    side_effect=lambda file_path, s3_key, **_kwargs: uploads.append(
                        (Path(file_path).name, s3_key)
                    ),
                ),
            ):
                sync_to_s3.sync()
            self.assertEqual(
                {name for name, _ in uploads},
                {"findings.json", "usage_stats.json", "session_001.log"},
            )

        self.assertIn(
            "export A='value with spaces'",
            runpod.create_run_command("python app.py", {"A": "value with spaces"}),
        )

        class FakeResponse:
            def __init__(self, status_code: int, text: str = "ok", payload: dict | None = None):
                self.status_code = status_code
                self.text = text
                self._payload = payload or {"id": "pod1", "status": "RUNNING"}

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(self.text)

        requests = SimpleNamespace(
            post=lambda *args, **kwargs: FakeResponse(200),
            get=lambda *args, **kwargs: FakeResponse(200, payload={"status": "RUNNING"}),
            delete=lambda *args, **kwargs: FakeResponse(200),
        )
        with patch.dict(sys.modules, {"requests": requests}):
            deployed = runpod.deploy_pod(
                "n",
                "img",
                "cmd",
                gpu_type="provider-accelerator",
                api_key="key",
                datacenter_ids=["DC"],
            )
            self.assertEqual(deployed["id"], "pod1")
            self.assertEqual(runpod.get_pod_status("pod1", api_key="key")["status"], "RUNNING")
            self.assertTrue(runpod.stop_pod("pod1", api_key="key"))
            self.assertTrue(runpod.delete_pod("pod1", api_key="key"))
        with (
            patch.dict(
                sys.modules,
                {"requests": SimpleNamespace(post=lambda *a, **k: FakeResponse(429, "capacity"))},
            ),
            self.assertRaises(runpod.RunPodCapacityError),
        ):
            runpod.deploy_pod("n", "img", "cmd", gpu_type="provider-accelerator", api_key="key")
        with (
            patch.dict(
                sys.modules,
                {"requests": SimpleNamespace(post=lambda *a, **k: FakeResponse(500, "bad"))},
            ),
            self.assertRaises(runpod.RunPodPermanentError),
        ):
            runpod.deploy_pod("n", "img", "cmd", gpu_type="provider-accelerator", api_key="key")
        with (
            patch.dict(sys.modules, {"requests": requests}),
            self.assertRaises(runpod.RunPodPermanentError),
        ):
            runpod.deploy_pod("n", "img", "cmd", gpu_type="provider-accelerator", api_key="")
        with (
            patch.dict(sys.modules, {"requests": requests}),
            self.assertRaisesRegex(runpod.RunPodPermanentError, "gpu_type"),
        ):
            runpod.deploy_pod("n", "img", "cmd", api_key="key")


class TrainingTimeoutAndContextContractsTest(unittest.TestCase):
    def test_training_timeout_policy_and_context_budget_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            context_firewall,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import (
            training_timeout,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "train.log"
            log.write_text(
                "Epoch 1/10\nEpoch 8/20\nEpoch 6/10\nEpoch 9/10\n",
                encoding="utf-8",
            )
            self.assertEqual(training_timeout.parse_current_epoch(log, expected_total=10), (9, 10))
            self.assertEqual(training_timeout.parse_current_epoch(log, expected_total=30), (8, 20))
            self.assertIsNone(training_timeout.parse_current_epoch(root / "missing.log"))
            bad_log = root / "bad.log"
            bad_log.write_text("Epoch x/y\nEpoch 0/0\n", encoding="utf-8")
            self.assertIsNone(training_timeout.parse_current_epoch(bad_log))

            class BadMatch:
                def findall(self, _tail):
                    return [("bad", "10"), ("5",), ("4", "0")]

            self.assertIsNone(training_timeout.parse_current_epoch(log, epoch_pattern=BadMatch()))

            self.assertEqual(training_timeout.should_emit_partial_summary(0, 10), (True, "ok"))
            self.assertFalse(training_timeout.should_emit_partial_summary(10, 10)[0])
            self.assertFalse(training_timeout.should_emit_partial_summary(5, 10)[0])
            self.assertTrue(training_timeout.should_emit_partial_summary(1, 10)[0])
            self.assertEqual(training_timeout.apply_frontier_discount(10.0, False), 10.0)
            self.assertEqual(training_timeout.apply_frontier_discount(10.0, True), 9.5)

            class FakeProc:
                pid = 1234

                def __init__(self, polls: list[int | None], wait_result: int = -15) -> None:
                    self.polls = polls
                    self.wait_result = wait_result
                    self.terminated = False
                    self.killed = False

                def poll(self):
                    return self.polls.pop(0) if self.polls else None

                def wait(self, timeout=None):
                    return self.wait_result

                def terminate(self):
                    self.terminated = True

                def kill(self):
                    self.killed = True

            with self.assertRaises(ValueError):
                training_timeout.monitor_subprocess_with_grace(FakeProc([None]), log, 1)

            policy = training_timeout.TimeoutPolicy(
                hard_cap_seconds=0,
                grace_check_interval_seconds=0,
                kill_grace_seconds=0,
            )
            with (
                patch.object(training_timeout.time, "sleep", lambda _seconds: None),
                patch.object(training_timeout.os, "killpg", lambda _pid, _sig: None),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        FakeProc([None], wait_result=-15),
                        root / "missing.log",
                        10,
                        policy=policy,
                    ),
                    -15,
                )
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        FakeProc([None, 0], wait_result=0),
                        log,
                        10,
                        policy=policy,
                    ),
                    0,
                )
            with (
                patch.object(training_timeout.time, "sleep", lambda _seconds: None),
                patch.object(training_timeout.os, "killpg", side_effect=ProcessLookupError),
            ):
                proc = FakeProc([None, 7], wait_result=7)
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        proc,
                        root / "missing.log",
                        10,
                        policy=policy,
                    ),
                    7,
                )

            no_killpg_policy = training_timeout.TimeoutPolicy(
                hard_cap_seconds=0,
                grace_check_interval_seconds=0,
                kill_grace_seconds=0,
            )
            with (
                patch.object(training_timeout.time, "sleep", lambda _seconds: None),
                patch.object(training_timeout, "os", SimpleNamespace()),
            ):
                proc = FakeProc([None], wait_result=-9)
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        proc,
                        root / "missing.log",
                        10,
                        policy=no_killpg_policy,
                    ),
                    -9,
                )

            grace_stall_policy = training_timeout.TimeoutPolicy(
                hard_cap_seconds=0,
                grace_progress_threshold=0.5,
                grace_stall_max_polls=1,
                grace_check_interval_seconds=0,
                grace_max_extension_seconds=999,
                kill_grace_seconds=0,
            )
            with (
                patch.object(training_timeout.time, "sleep", lambda _seconds: None),
                patch.object(training_timeout.os, "killpg", lambda _pid, _sig: None),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        FakeProc([None, None, None], wait_result=-3),
                        log,
                        10,
                        policy=grace_stall_policy,
                    ),
                    -3,
                )

            run_once_policy = training_timeout.TimeoutPolicy(
                hard_cap_seconds=999,
                grace_check_interval_seconds=0,
            )
            with patch.object(training_timeout.time, "sleep", lambda _seconds: None):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        FakeProc([None, 0]),
                        log,
                        10,
                        policy=run_once_policy,
                    ),
                    0,
                )

            grace_advances_policy = training_timeout.TimeoutPolicy(
                hard_cap_seconds=0,
                grace_progress_threshold=0.5,
                grace_stall_max_polls=3,
                grace_check_interval_seconds=0,
                grace_max_extension_seconds=999,
            )
            with (
                patch.object(training_timeout.time, "sleep", lambda _seconds: None),
                patch.object(
                    training_timeout,
                    "parse_current_epoch",
                    side_effect=[(9, 10), (10, 10)],
                ),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        FakeProc([None, None, 0]),
                        log,
                        10,
                        policy=grace_advances_policy,
                    ),
                    0,
                )

            class TimeoutOnceProc(FakeProc):
                def __init__(self, polls: list[int | None], wait_result: int = -2) -> None:
                    super().__init__(polls, wait_result=wait_result)
                    self.wait_calls = 0

                def wait(self, timeout=None):
                    self.wait_calls += 1
                    if self.wait_calls == 1:
                        raise training_timeout.subprocess.TimeoutExpired("train", timeout)
                    return self.wait_result

            with (
                patch.object(training_timeout.time, "sleep", lambda _seconds: None),
                patch.object(training_timeout, "os", SimpleNamespace()),
            ):
                proc = TimeoutOnceProc([None], wait_result=-2)
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        proc,
                        root / "missing.log",
                        10,
                        policy=no_killpg_policy,
                    ),
                    -1,
                )
                self.assertTrue(proc.killed)

            with (
                patch.object(training_timeout.time, "sleep", lambda _seconds: None),
                patch.object(training_timeout.os, "killpg", lambda _pid, _sig: None),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        TimeoutOnceProc([None]),
                        root / "missing.log",
                        10,
                        policy=policy,
                    ),
                    -1,
                )

            grace_expired_policy = training_timeout.TimeoutPolicy(
                hard_cap_seconds=0,
                grace_progress_threshold=0.5,
                grace_check_interval_seconds=0,
                grace_max_extension_seconds=-1,
                kill_grace_seconds=0,
            )
            with (
                patch.object(training_timeout.time, "sleep", lambda _seconds: None),
                patch.object(training_timeout.os, "killpg", lambda _pid, _sig: None),
            ):
                self.assertEqual(
                    training_timeout.monitor_subprocess_with_grace(
                        FakeProc([None, None], wait_result=-3),
                        log,
                        10,
                        policy=grace_expired_policy,
                    ),
                    -3,
                )

        pack = SimpleNamespace(
            pack_id="pack",
            built_at="now",
            panel_mode="mini",
            target_decisions=["d"],
            shared_core={
                "coverage_matrix_digest": "must stay",
                "negative_evidence_digest": "x" * 5000,
                "findings_summary": "x" * 5000,
                "long": ["x" * 1000 for _ in range(30)],
            },
            private_packs={
                "Builder": [{"interpretation": {"short": "x" * 1000}, "i": i} for i in range(10)]
            },
            audit={"ok": True},
        )
        fitted = context_firewall.fit_pack_to_budget(pack, "mini")
        self.assertLessEqual(len(fitted["private_packs"]["Builder"]), 2)
        self.assertIn("coverage_matrix_digest", fitted["shared_core"])
        self.assertTrue(context_firewall.forbid_raw_history({"raw_history": "x"}))
        self.assertFalse(context_firewall.forbid_raw_history({"summary": "x"}))
        self.assertGreater(context_firewall.estimate_tokens({"中文": "测试"}), 1)
        self.assertIn(
            "truncated",
            json.dumps(
                context_firewall.shrink_dict({"a": "x" * 2000}, budget_tokens=10),
                ensure_ascii=False,
            ),
        )


class CliAndSnapshotContractsTest(unittest.TestCase):
    def test_orchestrator_status_separates_mature_results_from_validation_signals(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import status_snapshot

        task_spec = SimpleNamespace(
            task_id="task",
            task_name="Task",
            generation_policy=SimpleNamespace(
                max_generations=3,
                cohort_size=2,
                promote_top_k=1,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                maturity_policy={
                    "complete_stage_labels": ["complete"],
                    "preliminary_stage_labels": ["preview"],
                },
            ),
            baselines=[SimpleNamespace(expected_acc=0.5)],
        )
        snapshot = status_snapshot.build_orchestrator_status_snapshot(
            run_started_at="2026-05-12T00:00:00+00:00",
            run_dir=Path("/tmp/run"),
            task_spec=task_spec,
            frontier=SimpleNamespace(get_summary=lambda: []),
            current_gen=1,
            gens_completed=0,
            frontier_strategy="auto",
            strategy_for_gen=lambda gen: f"strategy-{gen}",
            findings=[
                {
                    "finding_type": "result",
                    "variant_name": "mature",
                    "metrics": {
                        "score": 0.8,
                        "evidence_stage": "complete",
                        "scored_complete": True,
                    },
                },
                {
                    "finding_type": "result",
                    "variant_name": "preview",
                    "metrics": {
                        "score": 0.99,
                        "evidence_stage": "preview",
                        "scored_complete": False,
                        "partial": True,
                    },
                },
            ],
        )

        self.assertEqual(snapshot.best_mature_result["variant_name"], "mature")
        self.assertEqual(snapshot.best_mature_result["metric_value"], 0.8)
        self.assertEqual(snapshot.best_validation_signal["variant_name"], "preview")
        self.assertIn("partial", snapshot.best_validation_signal["validation_reason"])

    def test_orchestrator_status_separates_authorized_reduced_maturity_from_routing(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import status_snapshot

        task_spec = SimpleNamespace(
            task_id="task",
            task_name="Task",
            generation_policy=SimpleNamespace(
                max_generations=3,
                cohort_size=2,
                promote_top_k=1,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                maturity_policy={
                    "complete_stage_labels": ["reduced"],
                    "preliminary_stage_labels": ["diagnostic"],
                },
            ),
            baselines=[SimpleNamespace(expected_acc=0.5)],
        )
        snapshot = status_snapshot.build_orchestrator_status_snapshot(
            run_started_at="2026-05-12T00:00:00+00:00",
            run_dir=Path("/tmp/run"),
            task_spec=task_spec,
            frontier=SimpleNamespace(get_summary=lambda: []),
            current_gen=1,
            gens_completed=0,
            frontier_strategy="auto",
            strategy_for_gen=lambda gen: f"strategy-{gen}",
            findings=[
                {
                    "finding_type": "result",
                    "variant_name": "approved_reduced",
                    "metrics": {
                        "score": 0.8,
                        "evidence_stage": "reduced",
                        "scored_complete": True,
                        "partial": True,
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "promotion_eligible": True,
                    },
                },
                {
                    "finding_type": "result",
                    "variant_name": "validation_only_reduced",
                    "metrics": {
                        "score": 0.9,
                        "evidence_stage": "reduced",
                        "scored_complete": True,
                        "partial": True,
                        "validation_only": True,
                        "promotion_eligible": False,
                    },
                },
            ],
        )

        self.assertEqual(snapshot.best_mature_result["variant_name"], "approved_reduced")
        self.assertEqual(snapshot.best_mature_result["promotion_status"], "eligible")
        self.assertEqual(
            snapshot.best_validation_signal["variant_name"],
            "validation_only_reduced",
        )
        self.assertIn(
            "promotion_eligible=false", snapshot.best_validation_signal["validation_reason"]
        )

    def test_orchestrator_status_keeps_protocol_failed_result_out_of_mature_view(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import status_snapshot

        task_spec = SimpleNamespace(
            task_id="task",
            task_name="Task",
            generation_policy=SimpleNamespace(
                max_generations=3,
                cohort_size=2,
                promote_top_k=1,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                maturity_policy={
                    "complete_stage_labels": ["approved_reduced"],
                    "require_ratio_gate": False,
                },
            ),
            baselines=[SimpleNamespace(expected_acc=0.5)],
        )
        snapshot = status_snapshot.build_orchestrator_status_snapshot(
            run_started_at="2026-05-12T00:00:00+00:00",
            run_dir=Path("/tmp/run"),
            task_spec=task_spec,
            frontier=SimpleNamespace(get_summary=lambda: []),
            current_gen=1,
            gens_completed=0,
            frontier_strategy="auto",
            strategy_for_gen=lambda gen: f"strategy-{gen}",
            findings=[
                {
                    "finding_type": "result",
                    "variant_name": "protocol_failed",
                    "metrics": {
                        "score": 9.0,
                        "evidence_stage": "approved_reduced",
                        "scored_complete": True,
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "protocol_integrity_passed": False,
                    },
                }
            ],
        )

        self.assertEqual(snapshot.best_mature_result, {})
        self.assertEqual(
            snapshot.best_validation_signal["variant_name"],
            "protocol_failed",
        )
        self.assertEqual(
            snapshot.best_validation_signal["validation_reason"],
            "protocol_integrity",
        )

    def test_orchestrator_status_keeps_incomplete_ratio_result_as_signal(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import status_snapshot

        task_spec = SimpleNamespace(
            task_id="task",
            task_name="Task",
            generation_policy=SimpleNamespace(
                max_generations=3,
                cohort_size=2,
                promote_top_k=1,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                maturity_policy={"min_effort_ratio": 0.5, "min_coverage_ratio": 0.5},
            ),
            baselines=[SimpleNamespace(expected_acc=0.5)],
        )
        snapshot = status_snapshot.build_orchestrator_status_snapshot(
            run_started_at="2026-05-12T00:00:00+00:00",
            run_dir=Path("/tmp/run"),
            task_spec=task_spec,
            frontier=SimpleNamespace(get_summary=lambda: []),
            current_gen=1,
            gens_completed=0,
            frontier_strategy="auto",
            strategy_for_gen=lambda gen: f"strategy-{gen}",
            findings=[
                {
                    "finding_type": "result",
                    "variant_name": "unfinished",
                    "metrics": {
                        "score": 9.0,
                        "effort_ratio": 1.0,
                        "coverage_ratio": 1.0,
                        "incomplete_eval": True,
                    },
                }
            ],
        )

        self.assertEqual(snapshot.best_mature_result, {})
        self.assertEqual(snapshot.best_validation_signal["variant_name"], "unfinished")
        self.assertEqual(
            snapshot.best_validation_signal["validation_reason"],
            "incomplete_eval",
        )

    def test_orchestrator_status_rejects_non_result_artifacts_from_mature_view(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import status_snapshot

        task_spec = SimpleNamespace(
            task_id="task",
            task_name="Task",
            generation_policy=SimpleNamespace(
                max_generations=3,
                cohort_size=2,
                promote_top_k=1,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                maturity_policy={"complete_stage_labels": ["reduced"]},
            ),
            baselines=[SimpleNamespace(expected_acc=0.5)],
        )
        for marker in ("summary_only", "unscored_artifact"):
            snapshot = status_snapshot.build_orchestrator_status_snapshot(
                run_started_at="2026-05-12T00:00:00+00:00",
                run_dir=Path("/tmp/run"),
                task_spec=task_spec,
                frontier=SimpleNamespace(get_summary=lambda: []),
                current_gen=1,
                gens_completed=0,
                frontier_strategy="auto",
                strategy_for_gen=lambda gen: f"strategy-{gen}",
                findings=[
                    {
                        "finding_type": "result",
                        "variant_name": marker,
                        "metrics": {
                            "score": 9.0,
                            "evidence_stage": "reduced",
                            "scored_complete": True,
                            marker: True,
                        },
                    }
                ],
            )

            self.assertEqual(snapshot.best_mature_result, {})
            self.assertEqual(snapshot.best_validation_signal["variant_name"], marker)

    def test_orchestrator_status_does_not_infer_protocol_permission_from_tier(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import status_snapshot

        task_spec = SimpleNamespace(
            task_id="task",
            task_name="Task",
            generation_policy=SimpleNamespace(
                max_generations=3,
                cohort_size=2,
                promote_top_k=1,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                maturity_policy={"complete_stage_labels": ["full"]},
            ),
            baselines=[SimpleNamespace(expected_acc=0.5)],
        )
        snapshot = status_snapshot.build_orchestrator_status_snapshot(
            run_started_at="2026-05-12T00:00:00+00:00",
            run_dir=Path("/tmp/run"),
            task_spec=task_spec,
            frontier=SimpleNamespace(get_summary=lambda: []),
            current_gen=1,
            gens_completed=0,
            frontier_strategy="auto",
            strategy_for_gen=lambda gen: f"strategy-{gen}",
            findings=[
                {
                    "finding_type": "result",
                    "variant_name": "tier_conflict",
                    "metrics": {
                        "score": 9.0,
                        "evidence_stage": "scout",
                        "tier_reached": "full",
                        "scored_complete": True,
                    },
                }
            ],
        )

        self.assertEqual(snapshot.best_mature_result, {})
        self.assertEqual(snapshot.best_validation_signal["variant_name"], "tier_conflict")

    def test_orchestrator_status_restrictive_view_wins_for_same_result_artifact(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import status_snapshot

        task_spec = SimpleNamespace(
            task_id="task",
            task_name="Task",
            generation_policy=SimpleNamespace(
                max_generations=3,
                cohort_size=2,
                promote_top_k=1,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                maturity_policy={"complete_stage_labels": ["complete"]},
            ),
            baselines=[SimpleNamespace(expected_acc=0.5)],
        )
        clean = {
            "finding_type": "result",
            "variant_name": "clean_view",
            "metrics": {
                "score": 9.0,
                "evidence_stage": "complete",
                "scored_complete": True,
                "source_result_path": "results/shared/summary.json",
                "source_result_sha256": "shared-sha",
            },
        }
        restricted = {
            "finding_type": "result",
            "variant_name": "restricted_view",
            "metrics": {
                **clean["metrics"],
                "validation_only": True,
                "promotion_eligible": False,
            },
        }

        for findings in ([clean, restricted], [restricted, clean]):
            snapshot = status_snapshot.build_orchestrator_status_snapshot(
                run_started_at="2026-05-12T00:00:00+00:00",
                run_dir=Path("/tmp/run"),
                task_spec=task_spec,
                frontier=SimpleNamespace(get_summary=lambda: []),
                current_gen=1,
                gens_completed=0,
                frontier_strategy="auto",
                strategy_for_gen=lambda gen: f"strategy-{gen}",
                findings=findings,
            )
            self.assertEqual(snapshot.best_mature_result, {})
            self.assertEqual(
                snapshot.best_validation_signal["variant_name"],
                "restricted_view",
            )

    def test_cli_wrappers_and_orchestrator_snapshot(self) -> None:
        from praxist import run as cli
        from praxist.plugins.workflow_stages.research_loop.backend import status_snapshot

        class FakeFrontier:
            def get_summary(self):
                return [{"variant": "v"}]

        task_spec = SimpleNamespace(
            task_id="task",
            task_name="Task",
            generation_policy=SimpleNamespace(
                max_generations=3,
                cohort_size=2,
                promote_top_k=1,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(primary_metric="score", direction="maximize"),
            baselines=[SimpleNamespace(expected_acc=0.5)],
        )
        snapshot = status_snapshot.build_orchestrator_status_snapshot(
            run_started_at="2026-05-12T00:00:00+00:00",
            run_dir=Path("/tmp/run"),
            task_spec=task_spec,
            frontier=FakeFrontier(),
            current_gen=1,
            gens_completed=0,
            frontier_strategy="auto",
            strategy_for_gen=lambda gen: f"strategy-{gen}",
            findings=[
                {"finding_type": "result", "metrics": {"score": None}},
                {"finding_type": "result", "metrics": {"score": "nan"}},
                {
                    "finding_type": "result",
                    "metrics": {"score": 0.7, "seeds": [1, 2, 3]},
                },
                {"finding_type": "insight", "metrics": {"score": 0.9}},
            ],
        )
        self.assertEqual(snapshot.variants_total, 1)
        self.assertEqual(snapshot.variants_above_baseline, 1)
        self.assertEqual(snapshot.variants_validated_multi_seed, 1)
        self.assertEqual(snapshot.strategy, "strategy-1")

        task_spec_min = SimpleNamespace(
            **{
                **task_spec.__dict__,
                "evaluation": SimpleNamespace(primary_metric="loss", direction="minimize"),
                "baselines": [SimpleNamespace(expected_acc=1.0)],
            }
        )
        snapshot_min = status_snapshot.build_orchestrator_status_snapshot(
            run_started_at=None,
            run_dir=Path("/tmp/run"),
            task_spec=task_spec_min,
            frontier=SimpleNamespace(
                get_summary=lambda: (_ for _ in ()).throw(RuntimeError("frontier"))
            ),
            current_gen=-1,
            gens_completed=0,
            frontier_strategy="fallback",
            strategy_for_gen=lambda gen: "unused",
            findings=[{"finding_type": "result", "metrics": {"loss": 0.7, "n_seeds": "bad"}}],
        )
        self.assertEqual(snapshot_min.strategy, "fallback")
        self.assertEqual(snapshot_min.variants_above_baseline, 1)
        self.assertEqual(snapshot_min.frontier_candidates, 0)

        with (
            patch.object(cli, "cmd_run", side_effect=lambda _args: None) as cmd_run,
            patch.object(sys, "argv", ["praxist", "--log-level", "DEBUG", "run", "--fake"]),
        ):
            cli.main()
        cmd_run.assert_called_once()

        with (
            patch("praxist.infrastructure.execute_autonomous.main") as peer_main,
            patch.dict(os.environ, {}, clear=False),
        ):
            cli.cmd_peer(
                SimpleNamespace(
                    peer_id="p",
                    generation_id=2,
                    max_runtime=5,
                    prompt_file="prompt.txt",
                    model="model",
                    local=True,
                )
            )
            peer_main.assert_called_once()
            self.assertEqual(os.environ["PEER_ID"], "p")
            self.assertEqual(os.environ["LOCAL_MODE"], "true")

        with (
            patch.object(sys, "argv", ["praxist", "server"]),
            self.assertRaises(SystemExit) as cm,
        ):
            cli.main()
        self.assertEqual(cm.exception.code, 1)

    def test_empty_baselines_all_variants_above_and_not_blocked(self) -> None:
        """Issue #180: empty baselines must not deadlock promotion."""
        from praxist.plugins.workflow_stages.research_loop.backend import (
            status_snapshot,
        )

        task_spec = SimpleNamespace(
            task_id="t",
            task_name="T",
            generation_policy=SimpleNamespace(
                max_generations=2,
                cohort_size=2,
                promote_top_k=2,
                promote_criterion="primary_metric",
            ),
            evaluation=SimpleNamespace(primary_metric="score", direction="maximize"),
            baselines=[],  # empty
        )
        snapshot = status_snapshot.build_orchestrator_status_snapshot(
            run_started_at="2026-05-12T00:00:00+00:00",
            run_dir=Path("/tmp/run"),
            task_spec=task_spec,
            frontier=SimpleNamespace(get_summary=lambda: []),
            current_gen=0,
            gens_completed=0,
            frontier_strategy="auto",
            strategy_for_gen=lambda g: "auto",
            findings=[
                {"finding_type": "result", "metrics": {"score": 0.9}},
                {"finding_type": "result", "metrics": {"score": 0.7}},
                {"finding_type": "insight"},
            ],
        )
        self.assertEqual(snapshot.variants_above_baseline, 2)
        self.assertEqual(snapshot.gen_promotion_blocker, "")

    def test_subhour_generation_uses_one_canonical_budget_plan(self) -> None:
        from praxist.plugins.workflow_stages.research_loop import stage, startup

        task_spec = SimpleNamespace(
            generation_policy=SimpleNamespace(
                max_generations=1,
                cohort_size=1,
                per_generation_hours=0.5,
            ),
            compute_budget=SimpleNamespace(per_experiment_gpu_hours=0.0),
        )

        startup_plan = startup.planned_research_loop_usage(task_spec)
        stage_plan = stage._planned_usage_for_task_spec(task_spec)

        self.assertEqual(startup_plan, stage_plan)
        self.assertEqual(startup_plan["wall_clock_seconds"], 1800.0)
        self.assertEqual(startup_plan["gpu_hours"], 0.0)


if __name__ == "__main__":
    unittest.main()
