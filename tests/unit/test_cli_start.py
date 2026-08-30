"""Tests for ``praxist start`` — CLI lifecycle Phase 2."""

from __future__ import annotations

import io
import json
import os
import shlex
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch


class _FakeProc:
    def __init__(self, pid: int = 12345) -> None:
        self.pid = pid


def _make_task_dir(root: Path, name: str = "demo_task") -> Path:
    task = root / name
    task.mkdir()
    (task / "task.yaml").write_text(
        f"task_id: {name}\ntask_name: {name}\n",
        encoding="utf-8",
    )
    return task


class TaskProjectValidationTest(unittest.TestCase):
    def test_warns_when_measured_baseline_asset_is_not_declared(self) -> None:
        from praxist.cli import start

        with tempfile.TemporaryDirectory() as tmp:
            task = _make_task_dir(Path(tmp))
            baseline_dir = task / "assets" / "baselines"
            baseline_dir.mkdir(parents=True)
            (baseline_dir / "results.jsonl").write_text(
                json.dumps({"metric_name": "score", "metric_value": 0.75}) + "\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                start._validate_task_project(task)

        warning = stderr.getvalue()
        self.assertIn("WARNING: task.yaml declares no baselines", warning)
        self.assertIn("results.jsonl", warning)
        self.assertIn("not trusted or used", warning)

    def test_ignores_non_measurement_baseline_assets(self) -> None:
        from praxist.cli import start

        with tempfile.TemporaryDirectory() as tmp:
            task = _make_task_dir(Path(tmp))
            baseline_dir = task / "assets" / "baselines"
            baseline_dir.mkdir(parents=True)
            (baseline_dir / "results.jsonl").write_text("not-json\n", encoding="utf-8")
            (baseline_dir / "._results.jsonl").write_text(
                json.dumps({"metric_value": 1.0}) + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "README.md").write_text("# Baselines\n", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                start._validate_task_project(task)

        self.assertNotIn("WARNING: task.yaml declares no baselines", stderr.getvalue())

    def test_baseline_probe_tolerates_unrepresentable_numeric_values(self) -> None:
        from praxist.cli import start

        self.assertFalse(start._has_finite_numeric_value(10**10000))

    def test_declared_baseline_suppresses_asset_wiring_warning(self) -> None:
        from praxist.cli import start

        with tempfile.TemporaryDirectory() as tmp:
            task = _make_task_dir(Path(tmp))
            (task / "task.yaml").write_text(
                "task_id: demo\n"
                "task_name: demo\n"
                "baselines:\n"
                "  - name: measured\n"
                "    metric_name: score\n"
                "    metric_value: 0.75\n"
                "    direction: maximize\n",
                encoding="utf-8",
            )
            baseline_dir = task / "assets" / "baselines"
            baseline_dir.mkdir(parents=True)
            (baseline_dir / "results.jsonl").write_text(
                json.dumps({"metric_name": "score", "metric_value": 0.75}) + "\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                start._validate_task_project(task)

        self.assertNotIn("WARNING: task.yaml declares no baselines", stderr.getvalue())


class LaunchRunTest(unittest.TestCase):
    """``launch_run`` resolves defaults, spawns, and writes a registry entry."""

    def setUp(self) -> None:
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self._env_patch = patch.dict(
            os.environ,
            {
                "PRAXIST_STATE_DIR": self.state.name,
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "OPENROUTER_API_KEY": "",
                "ORCAROUTER_API_KEY": "",
                "OPENAI_API_KEY": "",
                "DEEPSEEK_API_KEY": "",
                "PRAXIST_MODEL_PROVIDER_REF": "",
                "MODEL_PROVIDER_REF": "",
                "MODEL": "",
                "PRAXIST_MODEL": "",
                "RUN_DIR": "",
                "TASK_PATH": "",
                "PRAXIST_AGENT_SYSTEM": "",
                "PRAXIST_LLM_PROVIDER": "",
                "PRAXIST_AGENT_RUNTIME_REF": "",
                "RUNTIME_REF": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

    def test_monitor_command_shell_quotes_run_id(self) -> None:
        from praxist.cli import start

        run_id = "run with spaces;printf unsafe"
        self.assertEqual(
            shlex.split(start.monitor_command(run_id)),
            ["praxist", "--monitor", "--run-id", run_id],
        )

    def test_operator_text_removes_terminal_and_bidi_controls(self) -> None:
        from praxist.cli import start

        rendered = start.operator_text("safe\x1b[2J\x07\u202evisible")
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x07", rendered)
        self.assertNotIn("\u202e", rendered)
        self.assertIn("visible", rendered)

    def test_default_provider_and_model_resolution_no_openrouter(self) -> None:
        from praxist.cli import registry, start

        task = _make_task_dir(Path(self.workspace.name))
        fake_spawn = MagicMock(return_value=_FakeProc(pid=4242))
        entry = start.launch_run(
            task_path=str(task),
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            spawn=fake_spawn,
        )
        self.assertEqual(entry.model_provider_ref, start.ANTHROPIC_PROVIDER_REF)
        self.assertEqual(entry.model, start.ANTHROPIC_DEFAULT_MODEL)
        self.assertEqual(entry.pid, 4242)
        # Registry entry persisted under the test state dir.
        on_disk = registry.read_entry(entry.run_id)
        self.assertEqual(on_disk.pid, 4242)

    def test_openrouter_provider_picked_when_only_openrouter_key_set(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="another")
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": "or-key"},
            clear=False,
        ):
            entry = start.launch_run(
                task_path=str(task),
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertEqual(entry.model_provider_ref, start.OPENROUTER_PROVIDER_REF)
        self.assertEqual(entry.model, start.OPENROUTER_DEFAULT_MODEL)

    def test_orcarouter_provider_picked_when_only_orcarouter_key_set(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="orcarouter")
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": "", "ORCAROUTER_API_KEY": "orca-key"},
            clear=False,
        ):
            entry = start.launch_run(
                task_path=str(task),
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertEqual(entry.model_provider_ref, start.ORCAROUTER_PROVIDER_REF)
        self.assertEqual(entry.model, start.ORCAROUTER_DEFAULT_MODEL)

    def test_deepseek_provider_picked_when_only_deepseek_key_set(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="deepseek_direct")
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "",
                "OPENROUTER_API_KEY": "",
                "DEEPSEEK_API_KEY": "ds-key",
            },
            clear=False,
        ):
            entry = start.launch_run(
                task_path=str(task),
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertEqual(entry.model_provider_ref, "model_provider:deepseek_alias")

    def test_deepseek_provider_preferred_when_deepseek_and_openrouter_keys_are_set(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="deepseek_preferred")
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "",
                "OPENROUTER_API_KEY": "or-key",
                "DEEPSEEK_API_KEY": "ds-key",
            },
            clear=False,
        ):
            entry = start.launch_run(
                task_path=str(task),
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertEqual(entry.model_provider_ref, "model_provider:deepseek_alias")
        self.assertEqual(entry.model, "deepseek-v4-pro[1m]")

    def test_credential_precheck_rejects_when_no_keys(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="creds")
        with (
            patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": "", "DEEPSEEK_API_KEY": ""},
                clear=False,
            ),
            self.assertRaises(start.StartError) as cm,
        ):
            start.launch_run(
                task_path=str(task),
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertIn("provider credential missing", str(cm.exception))

    def test_provider_specific_credential_required(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="provider-check")
        # Anthropic key is set, but operator forces openrouter provider.
        with self.assertRaises(start.StartError) as cm:
            start.launch_run(
                task_path=str(task),
                run_dir=None,
                model=None,
                model_provider_ref=start.OPENROUTER_PROVIDER_REF,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertIn("OPENROUTER_API_KEY", str(cm.exception))

    def test_missing_task_dir_rejected(self) -> None:
        from praxist.cli import start

        with self.assertRaises(start.StartError) as cm:
            start.launch_run(
                task_path="/path/that/does/not/exist",
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertIn("task project directory not found", str(cm.exception))

    def test_explicit_run_dir_honored(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="explicit_rd")
        explicit = Path(self.workspace.name) / "outer_runs" / "run_x"
        entry = start.launch_run(
            task_path=str(task),
            run_dir=str(explicit),
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            spawn=MagicMock(return_value=_FakeProc()),
        )
        self.assertEqual(Path(entry.run_dir), explicit.resolve())
        self.assertEqual(entry.run_id, explicit.name)
        # logs dir created.
        self.assertTrue((explicit / "logs").is_dir())

    def test_resume_from_is_threaded_to_child_command_and_registry(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="resume_task")
        previous_run = Path(self.workspace.name) / "resume_runs" / "run_old"
        previous_run.mkdir(parents=True)
        shutdown_path = previous_run / "ORCHESTRATOR_SHUTDOWN"
        shutdown_path.write_text("source=praxist_stop\n", encoding="utf-8")

        def spawn_after_reopen(*_args: object, **_kwargs: object) -> _FakeProc:
            self.assertFalse(shutdown_path.exists())
            return _FakeProc()

        entry = start.launch_run(
            task_path=str(task),
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            resume=False,
            resume_from=str(previous_run),
            resume_policy="completed_generation",
            spawn=spawn_after_reopen,
        )

        self.assertEqual(Path(entry.run_dir), previous_run.resolve())
        self.assertIn("--resume-from", entry.command)
        resume_idx = entry.command.index("--resume-from")
        self.assertEqual(Path(entry.command[resume_idx + 1]), previous_run.resolve())
        self.assertIn("--resume-policy", entry.command)
        self.assertEqual(entry.extra["resume"], "1")
        self.assertEqual(entry.extra["resume_policy"], "completed_generation")
        self.assertFalse(shutdown_path.exists())

    def test_direct_resume_restores_shutdown_fence_after_confirmed_startup_failure(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="failed_direct_resume")
        previous_run = Path(self.workspace.name) / "resume_runs" / "run_failed"
        previous_run.mkdir(parents=True)
        shutdown_path = previous_run / "ORCHESTRATOR_SHUTDOWN"
        original = b"source=praxist_stop\n"
        shutdown_path.write_bytes(original)

        def failed_startup(entry, *_args):
            return replace(entry, extra={**entry.extra, "startup_state": "failed"})

        with (
            patch("praxist.cli.start._default_spawn", return_value=_FakeProc()),
            patch("praxist.cli.start._wait_for_startup", side_effect=failed_startup),
        ):
            entry = start.launch_run(
                task_path=str(task),
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                resume_from=str(previous_run),
                startup_timeout=1,
            )

        self.assertEqual(entry.extra["startup_state"], "failed")
        self.assertEqual(shutdown_path.read_bytes(), original)

    def test_launch_holds_lifecycle_lock_through_spawn_and_registry_write(self) -> None:
        from praxist.cli import registry, start

        task = _make_task_dir(Path(self.workspace.name), name="locked_launch")
        run_dir = Path(self.workspace.name) / "runs" / "run_locked_launch"
        events: list[str] = []

        @contextmanager
        def observed_registry_lock():
            events.append("registry-enter")
            yield
            events.append("registry-exit")

        @contextmanager
        def observed_lock(run_id: str):
            self.assertEqual(run_id, run_dir.name)
            events.append("lock-enter")
            yield
            events.append("lock-exit")

        def observed_spawn(*_args: object, **_kwargs: object) -> _FakeProc:
            self.assertEqual(events, ["registry-enter", "lock-enter"])
            self.assertEqual(registry.read_entry(run_dir.name).pid, 0)
            events.append("spawn")
            return _FakeProc(pid=4321)

        with (
            patch(
                "praxist.cli.start.registry_lock",
                side_effect=observed_registry_lock,
            ),
            patch("praxist.cli.start.entry_lock", side_effect=observed_lock),
        ):
            entry = start.launch_run(
                task_path=str(task),
                run_dir=str(run_dir),
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=observed_spawn,
            )

        self.assertEqual(
            events,
            ["registry-enter", "lock-enter", "spawn", "lock-exit", "registry-exit"],
        )
        self.assertEqual(entry.pid, 4321)
        self.assertEqual(registry.read_entry(run_dir.name).pid, 4321)

    def test_direct_resume_rejects_live_controller_for_same_run(self) -> None:
        from praxist.cli import registry, start

        task = _make_task_dir(Path(self.workspace.name), name="live_direct_resume")
        run_dir = Path(self.workspace.name) / "runs" / "run_live_direct_resume"
        run_dir.mkdir(parents=True)
        existing = registry.RegistryEntry(
            schema_version=registry.SCHEMA_VERSION,
            run_id=run_dir.name,
            pid=7654,
            parent_pid=1,
            run_dir=str(run_dir),
            log_file=str(run_dir / "logs" / "launcher.nohup.log"),
            task_path=str(task),
            model=start.ANTHROPIC_DEFAULT_MODEL,
            model_provider_ref=start.ANTHROPIC_PROVIDER_REF,
            runtime_ref=start.DEFAULT_RUNTIME_REF,
            command=(
                "python",
                "-m",
                "praxist.run",
                "run",
                "--run-dir",
                str(run_dir),
            ),
            command_prefix="python -m praxist.run",
            started_at="2026-07-23T00:00:00+00:00",
            extra=registry.local_host_identity(),
        )
        registry.write_entry(existing)
        ps_rows = {
            existing.pid: (
                1,
                "00:01",
                f"python -m praxist.run run --run-dir {run_dir}",
            )
        }

        with (
            patch("praxist.cli.start.read_ps_table", return_value=ps_rows),
            self.assertRaises(start.StartError) as cm,
        ):
            start.launch_run(
                task_path=str(task),
                run_dir=str(run_dir),
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                resume=True,
                spawn=MagicMock(return_value=_FakeProc()),
            )

        self.assertIn("still appears to be running", str(cm.exception))

    def test_direct_resume_allows_recycled_pid_with_changed_start_identity(self) -> None:
        from praxist.cli import registry, start

        task = _make_task_dir(Path(self.workspace.name), name="recycled_direct_resume")
        run_dir = Path(self.workspace.name) / "runs" / "run_recycled_direct_resume"
        run_dir.mkdir(parents=True)
        existing = registry.RegistryEntry(
            schema_version=registry.SCHEMA_VERSION,
            run_id=run_dir.name,
            pid=7654,
            parent_pid=1,
            run_dir=str(run_dir),
            log_file=str(run_dir / "logs" / "launcher.nohup.log"),
            task_path=str(task),
            model=start.ANTHROPIC_DEFAULT_MODEL,
            model_provider_ref=start.ANTHROPIC_PROVIDER_REF,
            runtime_ref=start.DEFAULT_RUNTIME_REF,
            command=("python", "-m", "praxist.run", "run"),
            command_prefix="python -m praxist.run",
            started_at="2026-07-23T00:00:00+00:00",
            extra={
                **registry.local_host_identity(),
                "process_start_token": "proc:old",
            },
        )
        registry.write_entry(existing)

        with (
            patch("praxist.cli.start.process_identity_matches", return_value=False),
            patch("praxist.cli.start.read_ps_table") as read_ps,
        ):
            start._validate_resume_registry_slot(existing)

        read_ps.assert_not_called()

    def test_direct_resume_rejects_strong_identity_even_if_command_changes(self) -> None:
        from praxist.cli import registry, start

        run_dir = Path(self.workspace.name) / "runs" / "run_exec_controller"
        run_dir.mkdir(parents=True)
        existing = registry.RegistryEntry(
            schema_version=registry.SCHEMA_VERSION,
            run_id=run_dir.name,
            pid=7654,
            parent_pid=1,
            run_dir=str(run_dir),
            log_file=str(run_dir / "logs" / "launcher.nohup.log"),
            task_path="/tmp/task",
            model=start.ANTHROPIC_DEFAULT_MODEL,
            model_provider_ref=start.ANTHROPIC_PROVIDER_REF,
            runtime_ref=start.DEFAULT_RUNTIME_REF,
            command=("python", "-m", "praxist.run", "run"),
            command_prefix="python -m praxist.run",
            started_at="2026-07-23T00:00:00+00:00",
            extra={"process_start_token": "proc:live"},
        )
        registry.write_entry(existing)

        with (
            patch("praxist.cli.start.process_identity_matches", return_value=True),
            patch("praxist.cli.start.pid_is_alive", return_value=True),
            patch("praxist.cli.start.read_ps_table") as read_ps,
            self.assertRaisesRegex(start.StartError, "still appears to be running"),
        ):
            start._validate_resume_registry_slot(existing)

        read_ps.assert_not_called()

    def test_startup_ack_does_not_revive_concurrently_stopped_run(self) -> None:
        from praxist.cli import registry, start

        task = _make_task_dir(Path(self.workspace.name), name="stopped_startup")
        run_dir = Path(self.workspace.name) / "runs" / "run_stopped_startup"
        launched = start.launch_run(
            task_path=str(task),
            run_dir=str(run_dir),
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            spawn=MagicMock(return_value=_FakeProc(pid=5432)),
        )
        registry.update_state(launched.run_id, registry.STATE_STOPPED)

        observed = start._persist_startup_state(launched, "failed")

        self.assertEqual(observed.state, registry.STATE_STOPPED)
        self.assertEqual(
            registry.read_entry(launched.run_id).state,
            registry.STATE_STOPPED,
        )

    def test_startup_ack_rejects_reused_pid_with_different_start_token(self) -> None:
        from praxist.cli import registry, start

        task = _make_task_dir(Path(self.workspace.name), name="reused_startup_pid")
        run_dir = Path(self.workspace.name) / "runs" / "run_reused_startup_pid"
        launched = start.launch_run(
            task_path=str(task),
            run_dir=str(run_dir),
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            spawn=MagicMock(return_value=_FakeProc(pid=5432)),
        )
        launched = replace(
            launched,
            extra={**launched.extra, "process_start_token": "proc:old"},
        )
        registry.write_entry(
            replace(
                launched,
                extra={**launched.extra, "process_start_token": "proc:new"},
            )
        )

        with self.assertRaisesRegex(start.StartError, "controller changed"):
            start._persist_startup_state(launched, "running")
        with self.assertRaisesRegex(start.StartError, "controller changed"):
            start._current_launch_entry(launched)

    def test_resume_from_and_run_dir_must_match(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="resume_mismatch")
        previous_run = Path(self.workspace.name) / "resume_runs" / "run_old"
        other_run = Path(self.workspace.name) / "resume_runs" / "run_other"
        previous_run.mkdir(parents=True)
        other_run.mkdir(parents=True)
        with self.assertRaises(start.StartError) as cm:
            start.launch_run(
                task_path=str(task),
                run_dir=str(other_run),
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                resume=False,
                resume_from=str(previous_run),
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertIn("--resume-from and --run-dir", str(cm.exception))

    def test_server_flag_drops_local_arg(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="server_flag")
        entry = start.launch_run(
            task_path=str(task),
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=True,
            spawn=MagicMock(return_value=_FakeProc()),
        )
        self.assertNotIn("--local", entry.command)
        # And conversely: server=False adds --local.
        task2 = _make_task_dir(Path(self.workspace.name), name="local_flag")
        entry2 = start.launch_run(
            task_path=str(task2),
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            spawn=MagicMock(return_value=_FakeProc()),
        )
        self.assertIn("--local", entry2.command)

    def test_spawn_arguments_include_session_isolation(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="spawn_args")
        fake_spawn = MagicMock(return_value=_FakeProc(pid=7777))
        entry = start.launch_run(
            task_path=str(task),
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            spawn=fake_spawn,
        )
        # spawn invoked exactly once with start_new_session=True.
        self.assertEqual(fake_spawn.call_count, 1)
        _args, kwargs = fake_spawn.call_args
        self.assertTrue(kwargs.get("start_new_session"))
        self.assertEqual(kwargs.get("close_fds"), True)
        self.assertIn("env", kwargs)
        # log_file actually exists / will be writable.
        self.assertTrue(Path(entry.log_file).parent.is_dir())

    def test_cohort_and_generations_exported_into_child_env(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="cohort_env")
        fake_spawn = MagicMock(return_value=_FakeProc())
        start.launch_run(
            task_path=str(task),
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort="7",
            generations="3",
            server=False,
            spawn=fake_spawn,
        )
        _args, kwargs = fake_spawn.call_args
        env = kwargs["env"]
        self.assertEqual(env["COHORT_SIZE"], "7")
        self.assertEqual(env["MAX_GENERATIONS"], "3")

    def test_pythonunbuffered_is_set_in_child_env(self) -> None:
        """#167: spawn env must set PYTHONUNBUFFERED=1 so the workload's
        stdout / stderr line-buffer through ``launcher.nohup.log``.
        Without this, CPython block-buffers ~8 KB of log output when
        stdout is redirected to a file and the operator's ``tail -f``
        on ``launcher.nohup.log`` sees long quiet stretches.
        """
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="unbuffered_env")
        fake_spawn = MagicMock(return_value=_FakeProc())
        start.launch_run(
            task_path=str(task),
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            spawn=fake_spawn,
        )
        _args, kwargs = fake_spawn.call_args
        env = kwargs["env"]
        self.assertEqual(env.get("PYTHONUNBUFFERED"), "1")

    def test_pythonunbuffered_can_be_overridden_by_operator(self) -> None:
        """The flag uses ``setdefault`` so operators who really want
        buffered output (e.g. high-throughput CI runners) can set
        ``PYTHONUNBUFFERED`` to anything else in their parent shell
        and the launcher will pass that through unchanged.
        """
        import os
        from unittest.mock import patch

        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="unbuffered_override")
        fake_spawn = MagicMock(return_value=_FakeProc())
        with patch.dict(os.environ, {"PYTHONUNBUFFERED": ""}, clear=False):
            start.launch_run(
                task_path=str(task),
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=fake_spawn,
            )
        _args, kwargs = fake_spawn.call_args
        env = kwargs["env"]
        self.assertEqual(env["PYTHONUNBUFFERED"], "")


class CodexSdkAgentSystemTest(unittest.TestCase):
    """codex_sdk agent_system cascades into runtime ref, provider, env."""

    def setUp(self) -> None:
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self._env_patch = patch.dict(
            os.environ,
            {
                "PRAXIST_STATE_DIR": self.state.name,
                "ANTHROPIC_API_KEY": "",
                "OPENROUTER_API_KEY": "",
                "ORCAROUTER_API_KEY": "",
                "OPENAI_API_KEY": "sk-openai-test",
                "DEEPSEEK_API_KEY": "",
                "PRAXIST_MODEL_PROVIDER_REF": "",
                "MODEL_PROVIDER_REF": "",
                "MODEL": "",
                "PRAXIST_MODEL": "",
                "RUN_DIR": "",
                "TASK_PATH": "",
                "PRAXIST_AGENT_SYSTEM": "",
                "PRAXIST_LLM_PROVIDER": "",
                "PRAXIST_AGENT_RUNTIME_REF": "",
                "RUNTIME_REF": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

    def test_codex_sdk_default_provider_and_runtime(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="codex_default")
        fake_spawn = MagicMock(return_value=_FakeProc(pid=3333))
        entry = start.launch_run(
            task_path=str(task),
            agent_system="codex_sdk",
            runtime_ref=None,
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            spawn=fake_spawn,
        )
        self.assertEqual(entry.runtime_ref, "agent_runtime:codex_sdk")
        self.assertEqual(entry.model_provider_ref, start.OPENAI_PROVIDER_REF)
        # codex_sdk has no hard-coded default model; the runtime picks.
        self.assertEqual(entry.model, "")
        self.assertEqual(entry.extra["agent_system"], "codex_sdk")
        # --runtime threaded into the spawned command argv.
        self.assertIn("--runtime", entry.command)
        runtime_idx = entry.command.index("--runtime")
        self.assertEqual(entry.command[runtime_idx + 1], "agent_runtime:codex_sdk")
        # No --model flag when default-empty.
        self.assertNotIn("--model", entry.command)

    def test_codex_sdk_with_praxist_llm_provider_deepseek(self) -> None:
        """``PRAXIST_LLM_PROVIDER=deepseek`` routes through the deepseek alias."""
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="codex_deepseek")
        fake_spawn = MagicMock(return_value=_FakeProc())
        with patch.dict(
            os.environ,
            {"PRAXIST_LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "sk-deepseek"},
            clear=False,
        ):
            entry = start.launch_run(
                task_path=str(task),
                agent_system="codex_sdk",
                runtime_ref=None,
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=fake_spawn,
            )
        self.assertEqual(entry.model_provider_ref, "model_provider:deepseek_alias")
        self.assertEqual(entry.extra["auth_mode"], "configured-provider")
        self.assertEqual(fake_spawn.call_args.kwargs["env"]["DEEPSEEK_API_KEY"], "sk-deepseek")

    def test_codex_native_mode_isolates_api_configuration_from_spawn(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="codex_native")
        fake_spawn = MagicMock(return_value=_FakeProc())
        sensitive = {
            "OPENAI_API_KEY": "sk-api-billing",
            "CODEX_API_KEY": "codex-api-billing",
            "CODEX_ACCESS_TOKEN": "access-token",
            "OPENAI_BASE_URL": "https://custom.invalid/v1",
            "PRAXIST_CODEX_BIN": "/tmp/custom-codex",
            "PRAXIST_LLM_PROVIDER": "deepseek",
            "PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:claude_sdk",
            "RUNTIME_REF": "agent_runtime:claude_sdk",
            "PRAXIST_MODEL": "deepseek-v4-pro[1m]",
            "DEEPSEEK_API_KEY": "sk-deepseek-relay",
        }
        with patch.dict(os.environ, sensitive, clear=False):
            entry = start.launch_run(
                task_path=str(task),
                agent_system=None,
                runtime_ref=None,
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                codex_native=True,
                spawn=fake_spawn,
            )

        child_env = fake_spawn.call_args.kwargs["env"]
        self.assertEqual(entry.runtime_ref, "agent_runtime:codex_sdk")
        self.assertEqual(entry.model_provider_ref, start.OPENAI_PROVIDER_REF)
        self.assertEqual(entry.model, start.CODEX_NATIVE_DEFAULT_MODEL)
        self.assertEqual(entry.extra["auth_mode"], "codex-native")
        for variable in start.CODEX_NATIVE_BLOCKED_ENV:
            self.assertNotIn(variable, child_env)
        # Other provider credentials are not globally redefined by this mode.
        # They remain irrelevant because the selected provider is native OpenAI.
        self.assertEqual(child_env["DEEPSEEK_API_KEY"], "sk-deepseek-relay")

    def test_codex_native_mode_rejects_relay_provider_override(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="codex_native_conflict")
        with self.assertRaises(start.StartError) as cm:
            start.launch_run(
                task_path=str(task),
                agent_system="codex_sdk",
                runtime_ref="agent_runtime:codex_sdk",
                run_dir=None,
                model="gpt-5.6-luna",
                model_provider_ref=start.DEEPSEEK_PROVIDER_REF,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                codex_native=True,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertIn("openai_compatible", str(cm.exception))

    def test_codex_sdk_native_openai_defers_to_runtime_managed_auth(self) -> None:
        """The child runtime decides between saved ChatGPT auth and an API key."""
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="codex_creds")
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            entry = start.launch_run(
                task_path=str(task),
                agent_system="codex_sdk",
                runtime_ref=None,
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertEqual(entry.runtime_ref, "agent_runtime:codex_sdk")
        self.assertEqual(entry.model_provider_ref, start.OPENAI_PROVIDER_REF)

    def test_non_codex_runtime_still_requires_openai_api_key(self) -> None:
        from praxist.cli import start

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
            self.assertRaises(start.StartError),
        ):
            start._precheck_credentials(
                start.OPENAI_PROVIDER_REF,
                "agent_runtime:custom",
            )

    def test_explicit_runtime_ref_wins_over_agent_system_mapping(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="explicit_rt")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x"}, clear=False):
            entry = start.launch_run(
                task_path=str(task),
                agent_system="claude_sdk",
                runtime_ref="agent_runtime:custom_plugin",
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertEqual(entry.runtime_ref, "agent_runtime:custom_plugin")

    def test_explicit_builtin_runtime_overrides_persisted_agent_system(self) -> None:
        from praxist.cli import start

        with patch.dict(
            os.environ,
            {"PRAXIST_AGENT_SYSTEM": "claude_sdk"},
            clear=False,
        ):
            selected_agent, selected_runtime = start._resolve_runtime_selection(
                None,
                "agent_runtime:codex_sdk",
            )

        self.assertEqual(selected_agent, "codex_sdk")
        self.assertEqual(selected_runtime, "agent_runtime:codex_sdk")

    def test_canonical_runtime_env_overrides_stale_agent_system(self) -> None:
        from praxist.cli import start

        with patch.dict(
            os.environ,
            {
                "PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:codex_sdk",
                "RUNTIME_REF": "agent_runtime:claude_sdk",
                "PRAXIST_AGENT_SYSTEM": "claude_sdk",
            },
            clear=False,
        ):
            selected_agent, selected_runtime = start._resolve_runtime_selection(None, None)

        self.assertEqual(selected_agent, "codex_sdk")
        self.assertEqual(selected_runtime, "agent_runtime:codex_sdk")

    def test_legacy_runtime_env_is_loaded_when_canonical_is_empty(self) -> None:
        from praxist.cli import start

        with patch.dict(
            os.environ,
            {
                "PRAXIST_AGENT_RUNTIME_REF": "",
                "RUNTIME_REF": "agent_runtime:codex_sdk",
                "PRAXIST_AGENT_SYSTEM": "",
            },
            clear=False,
        ):
            selected_agent, selected_runtime = start._resolve_runtime_selection(None, None)
            helper_runtime = start._resolve_runtime_ref(None, "claude_sdk")

        self.assertEqual(selected_agent, "codex_sdk")
        self.assertEqual(selected_runtime, "agent_runtime:codex_sdk")
        self.assertEqual(helper_runtime, "agent_runtime:codex_sdk")

    def test_unknown_agent_system_rejected(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="bogus_as")
        with self.assertRaises(start.StartError) as cm:
            start.launch_run(
                task_path=str(task),
                agent_system="not_a_real_system",
                runtime_ref=None,
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertIn("unknown PRAXIST_AGENT_SYSTEM", str(cm.exception))

    def test_praxist_agent_runtime_ref_env_override(self) -> None:
        from praxist.cli import start

        with patch.dict(
            os.environ,
            {"PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:env_override"},
            clear=False,
        ):
            self.assertEqual(
                start._resolve_runtime_ref(None, "claude_sdk"),
                "agent_runtime:env_override",
            )

    def test_precheck_silent_for_unknown_provider_plugin_ref(self) -> None:
        """A custom provider ref is allowed through; runtime fails loudly later."""
        from praxist.cli import start

        # Returns None silently (no raise) — the inverse map has no entry.
        self.assertIsNone(start._provider_short_name("model_provider:custom_x"))
        start._precheck_credentials("model_provider:custom_x")  # no raise.

    def test_resolve_model_empty_for_claude_sdk_with_unusual_provider(self) -> None:
        from praxist.cli import start

        with patch.dict(os.environ, {"MODEL": "", "PRAXIST_MODEL": ""}, clear=False):
            resolved = start._resolve_model(None, "model_provider:custom", "claude_sdk")
        self.assertEqual(resolved, "")

    def test_praxist_agent_system_env_var_picks_codex_sdk(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="env_codex")
        with patch.dict(os.environ, {"PRAXIST_AGENT_SYSTEM": "codex_sdk"}, clear=False):
            entry = start.launch_run(
                task_path=str(task),
                agent_system=None,
                runtime_ref=None,
                run_dir=None,
                model=None,
                model_provider_ref=None,
                frontier_strategy="auto",
                cohort=None,
                generations=None,
                server=False,
                spawn=MagicMock(return_value=_FakeProc()),
            )
        self.assertEqual(entry.runtime_ref, "agent_runtime:codex_sdk")
        self.assertEqual(entry.extra["agent_system"], "codex_sdk")


class StartDefaultResolutionTest(unittest.TestCase):
    """Direct tests for the helper resolvers in ``praxist.cli.start``."""

    def test_task_path_precedence_is_explicit_then_env_then_invocation_cwd(self) -> None:
        from praxist.cli import start

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit = _make_task_dir(root, "explicit")
            env_task = _make_task_dir(root, "env")
            cwd_task = _make_task_dir(root, "cwd")
            with (
                patch.dict(os.environ, {"TASK_PATH": str(env_task)}, clear=False),
                patch("praxist.cli.start.Path.cwd", return_value=cwd_task),
            ):
                self.assertEqual(start._resolve_task_path(str(explicit)), explicit.resolve())
                self.assertEqual(start._resolve_task_path(None), env_task.resolve())
            with (
                patch.dict(os.environ, {"TASK_PATH": ""}, clear=False),
                patch("praxist.cli.start.Path.cwd", return_value=cwd_task),
            ):
                self.assertEqual(start._resolve_task_path(None), cwd_task.resolve())

    def test_provider_ref_from_env_override(self) -> None:
        from praxist.cli import start

        with patch.dict(
            os.environ,
            {
                "PRAXIST_MODEL_PROVIDER_REF": "",
                "MODEL_PROVIDER_REF": "model_provider:custom",
                "OPENROUTER_API_KEY": "",
                "PRAXIST_LLM_PROVIDER": "",
            },
            clear=False,
        ):
            self.assertEqual(
                start._resolve_provider_ref(None, start.DEFAULT_AGENT_SYSTEM),
                "model_provider:custom",
            )

    def test_canonical_provider_ref_precedes_legacy_selector(self) -> None:
        from praxist.cli import start

        with patch.dict(
            os.environ,
            {
                "PRAXIST_MODEL_PROVIDER_REF": "model_provider:canonical",
                "MODEL_PROVIDER_REF": "model_provider:legacy",
                "PRAXIST_LLM_PROVIDER": "openrouter",
            },
            clear=False,
        ):
            self.assertEqual(
                start._resolve_provider_ref(None, start.DEFAULT_AGENT_SYSTEM),
                "model_provider:canonical",
            )

    def test_model_explicit_value_wins(self) -> None:
        from praxist.cli import start

        self.assertEqual(
            start._resolve_model(
                "my-model", start.ANTHROPIC_PROVIDER_REF, start.DEFAULT_AGENT_SYSTEM
            ),
            "my-model",
        )

    def test_model_env_override(self) -> None:
        from praxist.cli import start

        with patch.dict(os.environ, {"MODEL": "env-model", "PRAXIST_MODEL": ""}, clear=False):
            self.assertEqual(
                start._resolve_model(
                    None, start.ANTHROPIC_PROVIDER_REF, start.DEFAULT_AGENT_SYSTEM
                ),
                "env-model",
            )

    def test_canonical_model_precedes_legacy_selector(self) -> None:
        from praxist.cli import start

        with patch.dict(
            os.environ,
            {"MODEL": "legacy-model", "PRAXIST_MODEL": "canonical-model"},
            clear=False,
        ):
            self.assertEqual(
                start._resolve_model(
                    None, start.ANTHROPIC_PROVIDER_REF, start.DEFAULT_AGENT_SYSTEM
                ),
                "canonical-model",
            )

    def test_run_dir_env_override(self) -> None:
        import datetime as _dt

        from praxist.cli import start

        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "task"
            task.mkdir()
            override_dir = Path(tmp) / "explicit_run"
            with patch.dict(os.environ, {"RUN_DIR": str(override_dir)}, clear=False):
                resolved = start._resolve_run_dir(
                    None, task, _dt.datetime(2026, 5, 19, tzinfo=_dt.UTC)
                )
            self.assertEqual(resolved, override_dir.resolve())

    def test_precheck_requires_anthropic_key_for_anthropic_provider(self) -> None:
        from praxist.cli import start

        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "", "OPENROUTER_API_KEY": "or-key"},
            clear=False,
        ):
            with self.assertRaises(start.StartError) as cm:
                start._precheck_credentials(start.ANTHROPIC_PROVIDER_REF)
            self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))

    def test_startup_stage_ack_ignores_prior_resume_events(self) -> None:
        from praxist.cli import start

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            trajectory = run_dir / "trajectory.jsonl"
            trajectory.write_text(
                json.dumps({"kind": "workflow.stage_started"}) + "\n",
                encoding="utf-8",
            )
            baseline = start._startup_artifact_signatures(run_dir)
            self.assertFalse(start._startup_stage_started(run_dir, baseline))
            with trajectory.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "workflow.stage_started"}) + "\n")
            self.assertTrue(start._startup_stage_started(run_dir, baseline))

    def test_startup_failure_requires_a_new_run_status(self) -> None:
        from praxist.cli import start

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            run_json = run_dir / "run.json"
            run_json.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            baseline = start._startup_artifact_signatures(run_dir)
            self.assertFalse(start._startup_failed(run_dir, baseline))
            run_json.write_text(
                json.dumps({"status": "failed", "error": "new failure"}),
                encoding="utf-8",
            )
            self.assertTrue(start._startup_failed(run_dir, baseline))


class StartCliEndToEndTest(unittest.TestCase):
    """Smoke-test ``praxist start`` via the top-level dispatcher."""

    def setUp(self) -> None:
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.task = _make_task_dir(Path(self.workspace.name), name="dispatch_task")
        config_file = Path(self.workspace.name) / "empty-praxist.env"
        config_file.write_text("", encoding="utf-8")
        self._env_patch = patch.dict(
            os.environ,
            {
                "PRAXIST_STATE_DIR": self.state.name,
                "PRAXIST_CONFIG_FILE": str(config_file),
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "OPENROUTER_API_KEY": "",
                "ORCAROUTER_API_KEY": "",
                "OPENAI_API_KEY": "",
                "DEEPSEEK_API_KEY": "",
                "PRAXIST_MODEL_PROVIDER_REF": "",
                "MODEL_PROVIDER_REF": "",
                "MODEL": "",
                "PRAXIST_MODEL": "",
                "RUN_DIR": "",
                "TASK_PATH": "",
                "PRAXIST_AGENT_SYSTEM": "",
                "PRAXIST_LLM_PROVIDER": "",
                "PRAXIST_AGENT_RUNTIME_REF": "",
                "RUNTIME_REF": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        from praxist.cli import main

        argv = list(argv)
        if argv and argv[0] == "start" and "--startup-timeout" not in argv:
            argv.extend(["--startup-timeout", "0"])
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with (
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                patch(
                    "praxist.cli.start._default_spawn",
                    return_value=_FakeProc(pid=8888),
                ),
            ):
                main(argv)
            code = 0
        except SystemExit as exc:
            code = int(exc.code or 0)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_dispatcher_starts_run_and_prints_run_id(self) -> None:
        with patch("praxist.cli.product_usage.prompt_for_consent_if_unset") as consent_prompt:
            code, out, err = self._run(["start", "--task-path", str(self.task)])
        self.assertEqual(code, 0)
        consent_prompt.assert_called_once_with()
        # Stdout = run_id only (machine-readable), stderr = operator hint.
        self.assertTrue(out.strip().startswith("run_"))
        self.assertIn("=== Praxist run launched ===", err)
        self.assertIn("praxist stop", err)

    def test_dispatcher_hint_points_at_both_log_streams(self) -> None:
        """#167: hint surfaces ``launcher.nohup.log`` AND
        ``trajectory.jsonl`` so the operator knows where the
        orchestrator's stdout/stderr lands AND where structured
        control-plane events go."""
        code, _out, err = self._run(["start", "--task-path", str(self.task)])
        self.assertEqual(code, 0)
        self.assertIn("launcher.nohup.log", err)
        self.assertIn("trajectory.jsonl", err)
        # The two ``tail -f`` lines should be annotated so the operator
        # knows which is which without having to read the docstrings.
        self.assertIn("orchestrator stdout/stderr", err)
        self.assertIn("control-plane events", err)
        self.assertIn("praxist --monitor --run-id", err)

    def test_dispatcher_hint_shell_quotes_explicit_run_dir(self) -> None:
        run_dir = Path(self.workspace.name) / "run with spaces;printf unsafe"
        code, out, err = self._run(
            ["start", "--task-path", str(self.task), "--run-dir", str(run_dir)]
        )
        self.assertEqual(code, 0)
        run_id = out.strip()
        self.assertEqual(run_id, run_dir.name)
        self.assertIn(shlex.quote(str(run_dir / "logs" / "launcher.nohup.log")), err)
        self.assertIn(shlex.quote(str(run_dir / "trajectory.jsonl")), err)
        self.assertIn(shlex.join(("praxist", "--monitor", "--run-id", run_id)), err)
        self.assertIn(shlex.join(("praxist", "stop", run_id)), err)

    def test_dispatcher_json_output(self) -> None:
        with patch("praxist.cli.product_usage.prompt_for_consent_if_unset") as consent_prompt:
            code, out, _err = self._run(["start", "--task-path", str(self.task), "--json"])
        self.assertEqual(code, 0)
        consent_prompt.assert_not_called()
        payload = json.loads(out)
        self.assertEqual(payload["pid"], 8888)
        self.assertEqual(payload["model_provider_ref"], "model_provider:anthropic_messages")
        self.assertIn("run_id", payload)
        self.assertEqual(
            payload["extra"]["monitor_command"],
            f"praxist --monitor --run-id {payload['run_id']}",
        )
        self.assertEqual(payload["extra"]["monitor_mode"], "foreground")

    def test_dispatcher_loads_explicit_config_file(self) -> None:
        config_file = Path(self.workspace.name) / "custom.env"
        config_file.write_text(
            "\n".join(
                [
                    "export PRAXIST_AGENT_SYSTEM=codex_sdk",
                    "export PRAXIST_LLM_PROVIDER=deepseek",
                    "export PRAXIST_MODEL=deepseek-v4-pro[1m]",
                    "export DEEPSEEK_API_KEY=test-deepseek-key",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        code, out, err = self._run(
            [
                "start",
                "--task-path",
                str(self.task),
                "--config-file",
                str(config_file),
                "--json",
            ]
        )
        self.assertEqual(code, 0, msg=out + err)
        payload = json.loads(out)
        self.assertEqual(payload["runtime_ref"], "agent_runtime:codex_sdk")
        self.assertEqual(payload["model_provider_ref"], "model_provider:deepseek_alias")
        self.assertEqual(payload["model"], "deepseek-v4-pro[1m]")

    def test_dispatcher_returns_nonzero_when_creds_missing(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            code, _out, err = self._run(["start", "--task-path", str(self.task)])
        self.assertEqual(code, 1)
        self.assertIn("provider credential missing", err)


class DaemonizeFlagTest(unittest.TestCase):
    """#99 follow-up: ``--daemonize`` routes through ``_spawn_daemonized``."""

    def setUp(self) -> None:
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self._env_patch = patch.dict(
            os.environ,
            {
                "PRAXIST_STATE_DIR": self.state.name,
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "OPENROUTER_API_KEY": "",
                "ORCAROUTER_API_KEY": "",
                "OPENAI_API_KEY": "",
                "DEEPSEEK_API_KEY": "",
                "PRAXIST_MODEL_PROVIDER_REF": "",
                "MODEL_PROVIDER_REF": "",
                "MODEL": "",
                "PRAXIST_MODEL": "",
                "RUN_DIR": "",
                "TASK_PATH": "",
                "PRAXIST_AGENT_SYSTEM": "",
                "PRAXIST_LLM_PROVIDER": "",
                "PRAXIST_AGENT_RUNTIME_REF": "",
                "RUNTIME_REF": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

    def test_daemonize_false_uses_popen_path(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="default_path")
        spawn = MagicMock(return_value=_FakeProc(pid=7000))
        daemon_spawn = MagicMock(return_value=8000)
        entry = start.launch_run(
            task_path=str(task),
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            daemonize=False,
            spawn=spawn,
            daemon_spawn=daemon_spawn,
        )
        self.assertEqual(entry.pid, 7000)
        self.assertEqual(entry.extra.get("daemonized"), "0")
        spawn.assert_called_once()
        daemon_spawn.assert_not_called()

    def test_daemonize_true_uses_daemon_spawn_path(self) -> None:
        from praxist.cli import start

        task = _make_task_dir(Path(self.workspace.name), name="daemon_path")
        spawn = MagicMock(return_value=_FakeProc(pid=7000))
        daemon_spawn = MagicMock(return_value=8000)
        entry = start.launch_run(
            task_path=str(task),
            run_dir=None,
            model=None,
            model_provider_ref=None,
            frontier_strategy="auto",
            cohort=None,
            generations=None,
            server=False,
            daemonize=True,
            spawn=spawn,
            daemon_spawn=daemon_spawn,
        )
        self.assertEqual(entry.pid, 8000)
        self.assertEqual(entry.extra.get("daemonized"), "1")
        spawn.assert_not_called()
        daemon_spawn.assert_called_once()
        # daemon_spawn(command, log_file, env)
        args, _kwargs = daemon_spawn.call_args
        self.assertEqual(args[0][0:4], list(entry.command[0:4]))

    def test_dispatcher_threads_daemonize_flag(self) -> None:
        """``praxist start --daemonize`` reaches ``launch_run(daemonize=True)``."""
        from praxist.cli import main

        task = _make_task_dir(Path(self.workspace.name), name="dispatch_daemon")
        captured: dict[str, object] = {}

        def fake_launch_run(**kwargs: object) -> object:
            captured.update(kwargs)
            # Return a minimal stand-in matching the e2e contract.
            from praxist.cli.registry import SCHEMA_VERSION, RegistryEntry

            return RegistryEntry(
                schema_version=SCHEMA_VERSION,
                run_id="run_x",
                pid=9999,
                parent_pid=1,
                run_dir="/tmp/r",
                log_file="/tmp/r/log",
                task_path=str(task),
                model="",
                model_provider_ref="model_provider:anthropic_messages",
                runtime_ref="agent_runtime:claude_sdk",
                command=("python", "-m", "praxist.run"),
                command_prefix="python -m praxist.run",
                started_at="2026-05-21T00:00:00+00:00",
                extra={"agent_system": "claude_sdk", "daemonized": "1"},
            )

        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with (
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                patch("praxist.cli.start.launch_run", side_effect=fake_launch_run),
            ):
                main(["start", "--task-path", str(task), "--daemonize"])
        except SystemExit:
            pass
        self.assertEqual(captured.get("daemonize"), True)
        self.assertIn("daemonize    : on", stderr.getvalue())


class SpawnDaemonizedIntegrationTest(unittest.TestCase):
    """Real-fork end-to-end test for ``_spawn_daemonized``.

    The double-fork path is awkward to mock in isolation; the
    simplest high-value test is to actually fork a benign child and
    confirm the original-parent side correctly reads the workload
    PID via the status pipe.
    """

    def test_real_double_fork_returns_workload_pid(self) -> None:
        import sys as _sys
        import time

        from praxist.cli import start

        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "logs" / "launcher.log"
            # Workload writes its own PID to stdout then exits.  The
            # ``_spawn_daemonized`` log redirection sends stdout into
            # ``log_file`` so we can read the workload's view of the
            # PID and compare it against what the original parent
            # got back through the pipe.
            command = [
                _sys.executable,
                "-c",
                "import os, sys; sys.stdout.write(str(os.getpid())); sys.stdout.flush()",
            ]
            pid = start._spawn_daemonized(command, log_file, dict(os.environ))
            self.assertGreater(pid, 0)
            # Give the workload a moment to write to the log; the
            # original parent returns as soon as it has the PID from
            # the pipe, which can happen before the workload's
            # stdout flush lands on disk. Keep the asserts INSIDE
            # the ``with`` block so ``tempfile.TemporaryDirectory``
            # cleanup doesn't wipe the log before we check it.
            for _ in range(50):
                if log_file.exists() and log_file.stat().st_size > 0:
                    break
                time.sleep(0.05)
            self.assertTrue(log_file.exists())
            workload_reported_pid = int(log_file.read_text().strip())
            self.assertEqual(workload_reported_pid, pid)

    def test_invalid_command_surfaces_start_error(self) -> None:
        """An ``ENOENT`` exec at the grandchild stage is surfaced as
        a clean :class:`StartError` rather than a hanging launcher.
        """
        from praxist.cli import start

        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "logs" / "launcher.log"
            with self.assertRaises(start.StartError):
                start._spawn_daemonized(
                    ["/no/such/binary"],
                    log_file,
                    dict(os.environ),
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
