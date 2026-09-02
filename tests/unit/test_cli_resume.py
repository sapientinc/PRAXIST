"""Tests for ``praxist resume``."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


class _FakeProc:
    def __init__(self, pid: int = 22345) -> None:
        self.pid = pid


def _make_task_dir(root: Path, name: str = "resume_task") -> Path:
    task = root / name
    task.mkdir(parents=True)
    return task


def _make_run_dir(root: Path, *, task_path: Path) -> Path:
    run_dir = root / "experiments" / "run_resume_demo"
    run_dir.mkdir(parents=True)
    startup_config = {
        "schema_version": "praxist.startup.v1",
        "canonical_args": {
            "task": "task:resume",
            "task_path": str(task_path),
            "runtime": "agent_runtime:claude_sdk",
            "model_provider": "model_provider:anthropic_messages",
            "budget_policy": "budget_policy:default",
            "model": "claude-opus-4-7",
            "frontier_strategy": "mixed",
            "run_dir": str(run_dir),
        },
    }
    (run_dir / "startup_config.json").write_text(
        json.dumps(startup_config),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": run_dir.name, "status": "interrupted"}),
        encoding="utf-8",
    )
    return run_dir


def _entry_kwargs(**overrides: object) -> dict[str, object]:
    base = {
        "schema_version": 1,
        "run_id": "run_resume_demo",
        "pid": 5000,
        "parent_pid": 1,
        "run_dir": "/tmp/resume_demo",
        "log_file": "/tmp/resume_demo/logs/launcher.nohup.log",
        "task_path": "/tmp/task",
        "model": "claude-opus-4-7",
        "model_provider_ref": "model_provider:anthropic_messages",
        "runtime_ref": "agent_runtime:claude_sdk",
        "command": (
            "python",
            "-m",
            "praxist.run",
            "run",
            "--task-path",
            "/tmp/task",
            "--run-dir",
            "/tmp/resume_demo",
            "--frontier-strategy",
            "mixed",
        ),
        "command_prefix": "python -m praxist.run",
        "started_at": "2026-06-27T00:00:00+00:00",
        "extra": {"agent_system": "claude_sdk"},
    }
    base.update(overrides)
    return base


class ResumeRunTest(unittest.TestCase):
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

    def test_registry_run_id_resumes_original_run_settings(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name))
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                state=registry.STATE_STOPPED,
            )
        )
        registry.write_entry(entry)
        shutdown_path = run_dir / "ORCHESTRATOR_SHUTDOWN"
        shutdown_path.write_text("source=praxist_stop\n", encoding="utf-8")

        def spawn_after_reopen(*_args: object, **_kwargs: object) -> _FakeProc:
            self.assertFalse(shutdown_path.exists())
            return _FakeProc(pid=7777)

        with (
            patch("praxist.cli.resume.read_ps_table", return_value={}),
            patch("praxist.cli.resume.pid_is_alive", return_value=False),
            patch("praxist.cli.start.read_ps_table", return_value={}),
            patch("praxist.cli.start.pid_is_alive", return_value=False),
        ):
            resumed = resume.resume_run(
                target=entry.run_id,
                spawn=spawn_after_reopen,
            )

        self.assertEqual(resumed.pid, 7777)
        self.assertEqual(Path(resumed.run_dir), run_dir.resolve())
        self.assertEqual(Path(resumed.task_path), task.resolve())
        self.assertEqual(resumed.model_provider_ref, "model_provider:anthropic_messages")
        self.assertEqual(resumed.runtime_ref, "agent_runtime:claude_sdk")
        self.assertIn("--resume-from", resumed.command)
        self.assertEqual(resumed.extra["resume"], "1")
        self.assertEqual(resumed.extra["resume_from"], str(run_dir.resolve()))
        self.assertIn("--frontier-strategy", resumed.command)
        strategy_idx = resumed.command.index("--frontier-strategy")
        self.assertEqual(resumed.command[strategy_idx + 1], "mixed")
        self.assertFalse(shutdown_path.exists())

    def test_failed_resume_restores_consumed_shutdown_fence(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="failed_resume_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                state=registry.STATE_STOPPED,
            )
        )
        registry.write_entry(entry)
        shutdown_path = run_dir / "ORCHESTRATOR_SHUTDOWN"
        original = b"source=praxist_stop\n"
        shutdown_path.write_bytes(original)

        with (
            patch("praxist.cli.resume.read_ps_table", return_value={}),
            patch("praxist.cli.resume.pid_is_alive", return_value=False),
            patch(
                "praxist.cli.resume.start.launch_run",
                side_effect=RuntimeError("spawn failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "spawn failed"),
        ):
            resume.resume_run(target=entry.run_id)

        self.assertEqual(shutdown_path.read_bytes(), original)

    def test_startup_wait_failure_restores_consumed_shutdown_fence(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="failed_startup_wait_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        stopped = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                state=registry.STATE_STOPPED,
            )
        )
        registry.write_entry(stopped)
        shutdown_path = run_dir / "ORCHESTRATOR_SHUTDOWN"
        original = b"source=praxist_stop\n"
        shutdown_path.write_bytes(original)
        launched = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                pid=7778,
                state=registry.STATE_RUNNING,
            )
        )
        failed = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                pid=7778,
                state=registry.STATE_RUNNING,
                extra={"agent_system": "claude_sdk", "startup_state": "failed"},
            )
        )

        with (
            patch("praxist.cli.resume.read_ps_table", return_value={}),
            patch("praxist.cli.resume.pid_is_alive", return_value=False),
            patch("praxist.cli.resume.start.launch_run", return_value=launched),
            patch("praxist.cli.resume.start._wait_for_startup", return_value=failed),
        ):
            result = resume.resume_run(target=stopped.run_id, startup_timeout=1)

        self.assertEqual(result, failed)
        self.assertEqual(shutdown_path.read_bytes(), original)

    def test_startup_wait_exception_restores_consumed_shutdown_fence(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="startup_exception_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        stopped = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                state=registry.STATE_STOPPED,
            )
        )
        registry.write_entry(stopped)
        shutdown_path = run_dir / "ORCHESTRATOR_SHUTDOWN"
        original = b"source=praxist_stop\n"
        shutdown_path.write_bytes(original)
        launched = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                pid=7780,
                state=registry.STATE_RUNNING,
            )
        )

        with (
            patch("praxist.cli.resume.read_ps_table", return_value={}),
            patch("praxist.cli.resume.pid_is_alive", return_value=False),
            patch("praxist.cli.resume.start.launch_run", return_value=launched),
            patch(
                "praxist.cli.resume.start._wait_for_startup",
                side_effect=RuntimeError("startup probe failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "startup probe failed"),
        ):
            resume.resume_run(target=stopped.run_id, startup_timeout=1)

        self.assertEqual(shutdown_path.read_bytes(), original)

    def test_resume_reports_shutdown_fence_io_failures(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="fence_io_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        stopped = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                state=registry.STATE_STOPPED,
            )
        )
        registry.write_entry(stopped)

        with (
            patch("praxist.cli.resume.read_ps_table", return_value={}),
            patch("praxist.cli.resume.pid_is_alive", return_value=False),
            patch(
                "praxist.cli.resume.start._consume_shutdown_fence",
                side_effect=OSError("read-only run"),
            ),
            self.assertRaisesRegex(resume.ResumeError, "could not reopen stopped run"),
        ):
            resume.resume_run(target=stopped.run_id)

        resume._restore_consumed_shutdown_fence(None)
        with (
            patch(
                "praxist.cli.resume.start._restore_shutdown_fence",
                side_effect=OSError("read-only run"),
            ),
            self.assertRaisesRegex(resume.ResumeError, "could not restore shutdown fence"),
        ):
            resume._restore_consumed_shutdown_fence(
                (run_dir / "ORCHESTRATOR_SHUTDOWN", b"source=praxist_stop\n")
            )

    def test_startup_wait_failure_does_not_overwrite_newer_shutdown_fence(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="concurrent_stop_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        stopped = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                state=registry.STATE_STOPPED,
            )
        )
        registry.write_entry(stopped)
        shutdown_path = run_dir / "ORCHESTRATOR_SHUTDOWN"
        shutdown_path.write_bytes(b"source=original_stop\n")
        launched = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                pid=7779,
                state=registry.STATE_RUNNING,
            )
        )
        failed = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                pid=7779,
                state=registry.STATE_RUNNING,
                extra={"agent_system": "claude_sdk", "startup_state": "failed"},
            )
        )

        def fail_after_new_stop(*_args: object, **_kwargs: object) -> registry.RegistryEntry:
            shutdown_path.write_bytes(b"source=newer_stop\n")
            return failed

        with (
            patch("praxist.cli.resume.read_ps_table", return_value={}),
            patch("praxist.cli.resume.pid_is_alive", return_value=False),
            patch("praxist.cli.resume.start.launch_run", return_value=launched),
            patch(
                "praxist.cli.resume.start._wait_for_startup",
                side_effect=fail_after_new_stop,
            ),
        ):
            result = resume.resume_run(target=stopped.run_id, startup_timeout=1)

        self.assertEqual(result, failed)
        self.assertEqual(shutdown_path.read_bytes(), b"source=newer_stop\n")

    def test_live_registry_run_is_rejected_without_force(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="live_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        entry = registry.RegistryEntry(**_entry_kwargs(run_dir=str(run_dir), task_path=str(task)))
        registry.write_entry(entry)
        ps_rows = {
            entry.pid: (
                1,
                "00:10",
                "python -m praxist.run run --task-path /tmp/task --run-dir /tmp/resume_demo",
            )
        }

        with (
            patch("praxist.cli.resume.read_ps_table", return_value=ps_rows),
            self.assertRaises(resume.ResumeError) as cm,
        ):
            resume.resolve_resume_target(entry.run_id)
        self.assertIn("still appears to be running", str(cm.exception))

    def test_same_prefix_other_run_dir_is_not_treated_as_live_resume_target(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="recycled_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        entry = registry.RegistryEntry(**_entry_kwargs(run_dir=str(run_dir), task_path=str(task)))
        registry.write_entry(entry)
        ps_rows = {
            entry.pid: (
                1,
                "00:10",
                "python -m praxist.run run --run-dir /tmp/different_run",
            )
        }

        with patch("praxist.cli.resume.read_ps_table", return_value=ps_rows):
            target = resume.resolve_resume_target(entry.run_id)

        self.assertEqual(target.source_run_id, entry.run_id)
        self.assertEqual(target.run_dir, run_dir.resolve())

    def test_empty_target_is_rejected(self) -> None:
        from praxist.cli import resume

        with self.assertRaises(resume.ResumeError) as cm:
            resume.resolve_resume_target("  ")
        self.assertIn("expected", str(cm.exception))

    def test_force_does_not_override_live_registry_controller(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="force_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                command=(
                    "python",
                    "-m",
                    "praxist.run",
                    "run",
                    "--frontier-strategy=mixed",
                ),
                run_dir=str(run_dir),
                task_path=str(task),
                extra={"process_start_token": "proc:live"},
            )
        )
        registry.write_entry(entry)

        ps_rows = {
            entry.pid: (
                1,
                "00:10",
                "python -m praxist.run run --run-dir /tmp/resume_demo --frontier-strategy=mixed",
            )
        }
        with (
            patch("praxist.cli.resume.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.resume.process_identity_matches", return_value=True),
            patch("praxist.cli.resume.pid_is_alive", return_value=True),
            self.assertRaises(resume.ResumeError) as cm,
        ):
            resume.resolve_resume_target(entry.run_id, force=True)
        self.assertIn("--force cannot bypass", str(cm.exception))

    def test_force_allows_unknown_legacy_registry_process(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="legacy_force_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                extra={},
            )
        )
        registry.write_entry(entry)
        ps_rows = {
            entry.pid: (
                1,
                "00:10",
                "python -m praxist.run run --run-dir /tmp/resume_demo",
            )
        }
        with (
            patch("praxist.cli.resume.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.resume.process_identity_matches", return_value=None),
        ):
            with self.assertRaises(resume.ResumeError) as cm:
                resume.resolve_resume_target(entry.run_id)
            target = resume.resolve_resume_target(entry.run_id, force=True)

        self.assertIn("ownership cannot be verified", str(cm.exception))
        self.assertEqual(target.source_run_id, entry.run_id)

    def test_matching_strong_identity_stays_live_when_command_changes(self) -> None:
        from praxist.cli import registry, resume

        entry = registry.RegistryEntry(**_entry_kwargs(extra={"process_start_token": "proc:live"}))
        with (
            patch("praxist.cli.resume.entry_process_epoch_matches", return_value=True),
            patch("praxist.cli.resume.process_identity_matches", return_value=True),
            patch("praxist.cli.resume.pid_is_alive", return_value=True),
            patch(
                "praxist.cli.resume.read_ps_table",
                return_value={entry.pid: (1, "00:01", "renamed-controller")},
            ) as read_ps,
        ):
            self.assertEqual(resume._entry_liveness(entry), "verified-live")

        read_ps.assert_not_called()

    def test_run_dir_path_cannot_bypass_live_registry_controller(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="path_live_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                command=(
                    "python",
                    "-m",
                    "praxist.run",
                    "run",
                    "--run-dir",
                    str(run_dir),
                ),
            )
        )
        registry.write_entry(entry)
        ps_rows = {
            entry.pid: (
                1,
                "00:10",
                f"python -m praxist.run run --run-dir {run_dir}",
            )
        }
        with (
            patch("praxist.cli.resume.read_ps_table", return_value=ps_rows),
            self.assertRaises(resume.ResumeError) as cm,
        ):
            resume.resolve_resume_target(str(run_dir))

        self.assertIn("still appears to be running", str(cm.exception))

    def test_run_dir_path_rejects_remote_registry_owner(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="path_remote_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                extra={"agent_system": "claude_sdk", "hostname": "remote-host"},
            )
        )
        registry.write_entry(entry)
        with (
            patch(
                "praxist.cli.registry.local_host_identity",
                return_value={"hostname": "local-host"},
            ),
            self.assertRaises(resume.ResumeError) as cm,
        ):
            resume.resolve_resume_target(str(run_dir))

        self.assertIn("belongs to", str(cm.exception))

    def test_run_dir_path_cannot_overwrite_same_id_from_other_directory(self) -> None:
        from praxist.cli import registry, resume

        first_root = Path(self.workspace.name) / "first"
        second_root = Path(self.workspace.name) / "second"
        task = _make_task_dir(Path(self.workspace.name), name="same_id_task")
        registered_dir = _make_run_dir(first_root, task_path=task)
        unregistered_dir = _make_run_dir(second_root, task_path=task)
        registry.write_entry(
            registry.RegistryEntry(
                **_entry_kwargs(
                    run_dir=str(registered_dir),
                    task_path=str(task),
                    state=registry.STATE_STOPPED,
                )
            )
        )

        with self.assertRaises(resume.ResumeError) as cm:
            resume.resolve_resume_target(str(unregistered_dir))

        self.assertIn("already registered", str(cm.exception))
        self.assertIn(str(registered_dir.resolve()), str(cm.exception))

    def test_run_dir_target_recovers_startup_config(self) -> None:
        from praxist.cli import resume

        task = _make_task_dir(Path(self.workspace.name), name="dir_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        target = resume.resolve_resume_target(str(run_dir))

        self.assertEqual(target.source, "run_dir")
        self.assertEqual(target.run_dir, run_dir.resolve())
        self.assertEqual(target.task_path, str(task))
        self.assertEqual(target.runtime_ref, "agent_runtime:claude_sdk")
        self.assertEqual(target.model_provider_ref, "model_provider:anthropic_messages")
        self.assertEqual(target.model, "claude-opus-4-7")
        self.assertEqual(target.frontier_strategy, "mixed")

    def test_run_dir_target_can_recover_task_project_path_fallback(self) -> None:
        from praxist.cli import resume

        task = _make_task_dir(Path(self.workspace.name), name="fallback_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        (run_dir / "startup_config.json").write_text(
            json.dumps(
                {
                    "task_project": {"path": str(task)},
                    "canonical_args": {"model": "", "frontier_strategy": ""},
                }
            ),
            encoding="utf-8",
        )

        target = resume.resolve_resume_target(str(run_dir))

        self.assertEqual(target.task_path, str(task))
        self.assertIsNone(target.model)
        self.assertEqual(target.frontier_strategy, "auto")

    def test_run_dir_target_decodes_string_server_flag(self) -> None:
        from praxist.cli import resume

        task = _make_task_dir(Path(self.workspace.name), name="server_flag_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        payload = json.loads((run_dir / "startup_config.json").read_text())
        payload["canonical_args"]["server"] = "false"
        (run_dir / "startup_config.json").write_text(json.dumps(payload), encoding="utf-8")

        self.assertFalse(resume.resolve_resume_target(str(run_dir)).server)
        payload["canonical_args"]["server"] = "true"
        (run_dir / "startup_config.json").write_text(json.dumps(payload), encoding="utf-8")
        self.assertTrue(resume.resolve_resume_target(str(run_dir)).server)

    def test_run_dir_target_recovers_server_mode_from_top_level_local_mode(self) -> None:
        from praxist.cli import resume

        task = _make_task_dir(Path(self.workspace.name), name="legacy_server_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        payload = json.loads((run_dir / "startup_config.json").read_text())
        payload["canonical_args"].pop("server", None)
        payload["local_mode"] = False
        (run_dir / "startup_config.json").write_text(json.dumps(payload), encoding="utf-8")

        self.assertTrue(resume.resolve_resume_target(str(run_dir)).server)

        payload["local_mode"] = True
        (run_dir / "startup_config.json").write_text(json.dumps(payload), encoding="utf-8")
        self.assertFalse(resume.resolve_resume_target(str(run_dir)).server)

    def test_run_dir_target_recovers_codex_native_from_redacted_credential(self) -> None:
        from praxist.cli import resume

        task = _make_task_dir(Path(self.workspace.name), name="native_artifact_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        payload = json.loads((run_dir / "startup_config.json").read_text())
        payload["canonical_args"].update(
            {
                "runtime": "agent_runtime:codex_sdk",
                "model_provider": "model_provider:openai_compatible",
            }
        )
        (run_dir / "startup_config.json").write_text(json.dumps(payload), encoding="utf-8")
        (run_dir / "credentials_redacted.json").write_text(
            json.dumps(
                {
                    "credential_profiles": [
                        {
                            "provider": "openai_compatible",
                            "source": "runtime_session",
                            "key_id": "openai_compatible:codex_sdk:chatgpt:account-hash",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        target = resume.resolve_resume_target(str(run_dir))

        self.assertTrue(target.codex_native)

    def test_resume_run_requires_inferable_task_path(self) -> None:
        from praxist.cli import resume

        run_dir = Path(self.workspace.name) / "missing_task_run"
        run_dir.mkdir()
        (run_dir / "run.json").write_text("{}", encoding="utf-8")
        (run_dir / "startup_config.json").write_text(
            json.dumps({"canonical_args": {}}),
            encoding="utf-8",
        )

        with self.assertRaises(resume.ResumeError) as cm:
            resume.resume_run(target=str(run_dir))
        self.assertIn("could not infer task", str(cm.exception))

    def test_resume_runtime_override_does_not_reuse_conflicting_agent_system(self) -> None:
        from praxist.cli import resume

        agent_system, runtime_ref = resume._resume_runtime_selection(
            agent_system=None,
            runtime_ref="agent_runtime:codex_sdk",
            inherited_agent_system="claude_sdk",
            inherited_runtime_ref="agent_runtime:claude_sdk",
        )
        self.assertEqual(agent_system, "codex_sdk")
        self.assertEqual(runtime_ref, "agent_runtime:codex_sdk")

        agent_system, runtime_ref = resume._resume_runtime_selection(
            agent_system="codex_sdk",
            runtime_ref=None,
            inherited_agent_system="claude_sdk",
            inherited_runtime_ref="agent_runtime:claude_sdk",
        )
        self.assertEqual(agent_system, "codex_sdk")
        self.assertIsNone(runtime_ref)

    def test_recycled_pid_start_identity_does_not_block_resume(self) -> None:
        from praxist.cli import registry, resume

        entry = registry.RegistryEntry(**_entry_kwargs(extra={"process_start_token": "proc:old"}))
        with (
            patch("praxist.cli.resume.entry_process_epoch_matches", return_value=True),
            patch("praxist.cli.resume.process_identity_matches", return_value=False),
            patch("praxist.cli.resume.read_ps_table") as read_ps,
        ):
            self.assertFalse(resume._entry_appears_live(entry))

        read_ps.assert_not_called()

    def test_resume_rechecks_target_after_lifecycle_lock(self) -> None:
        from praxist.cli import resume

        run_dir = Path(self.workspace.name) / "run_locked"
        run_dir.mkdir()
        initial = resume.ResumeTarget(run_dir=run_dir, task_path="/tmp/task")
        with (
            patch(
                "praxist.cli.resume.resolve_resume_target",
                side_effect=[initial, resume.ResumeError("competing controller is live")],
            ),
            patch("praxist.cli.resume.start.launch_run") as launch,
            self.assertRaises(resume.ResumeError),
        ):
            resume.resume_run(target=str(run_dir))
        launch.assert_not_called()

    def test_resume_waits_for_startup_after_releasing_lifecycle_lock(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="startup_wait_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        target = resume.ResumeTarget(run_dir=run_dir, task_path=str(task))
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                pid=9876,
                state=registry.STATE_RUNNING,
            )
        )
        lock_held = False

        @contextmanager
        def observed_lock(_run_id: str):
            nonlocal lock_held
            self.assertFalse(lock_held)
            lock_held = True
            try:
                yield
            finally:
                lock_held = False

        def observed_wait(
            current: registry.RegistryEntry,
            _timeout: float,
            _baseline: object,
        ) -> registry.RegistryEntry:
            self.assertFalse(lock_held)
            return current

        with (
            patch("praxist.cli.resume.resolve_resume_target", return_value=target),
            patch("praxist.cli.resume.entry_lock", side_effect=observed_lock),
            patch("praxist.cli.resume.start.launch_run", return_value=entry) as launch,
            patch(
                "praxist.cli.resume.start._wait_for_startup",
                side_effect=observed_wait,
            ) as wait,
        ):
            resumed = resume.resume_run(target=str(run_dir), startup_timeout=1)

        self.assertEqual(resumed, entry)
        launch.assert_called_once()
        self.assertTrue(launch.call_args.kwargs["_defer_startup_wait"])
        wait.assert_called_once()

    def test_registry_target_preserves_codex_native_auth_mode(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="native_resume_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                runtime_ref="agent_runtime:codex_sdk",
                model_provider_ref="model_provider:openai_compatible",
                extra={
                    "agent_system": "codex_sdk",
                    "auth_mode": "codex-native",
                },
            )
        )

        target = resume._target_from_registry(entry)

        self.assertTrue(target.codex_native)
        self.assertEqual(target.runtime_ref, "agent_runtime:codex_sdk")
        self.assertEqual(target.model_provider_ref, "model_provider:openai_compatible")

    def test_legacy_registry_target_recovers_codex_native_from_artifact(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="legacy_native_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        (run_dir / "credentials_redacted.json").write_text(
            json.dumps(
                {
                    "credential_profiles": [
                        {
                            "provider": "openai_compatible",
                            "source": "runtime_session",
                            "key_id": "openai_compatible:codex_sdk:chatgpt:account-hash",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_dir=str(run_dir),
                task_path=str(task),
                runtime_ref="agent_runtime:codex_sdk",
                model_provider_ref="model_provider:openai_compatible",
                extra={"agent_system": "codex_sdk"},
            )
        )

        self.assertTrue(resume._target_from_registry(entry).codex_native)

    def test_codex_native_resume_rejects_canonical_runtime_switch(self) -> None:
        from praxist.cli import resume

        task = _make_task_dir(Path(self.workspace.name), name="native_switch_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        target = resume.ResumeTarget(
            run_dir=run_dir,
            task_path=str(task),
            runtime_ref="agent_runtime:claude_sdk",
            model_provider_ref="model_provider:deepseek_alias",
        )
        with (
            patch("praxist.cli.resume.resolve_resume_target", return_value=target),
            patch("praxist.cli.resume.start.launch_run") as launch,
            self.assertRaises(resume.ResumeError) as cm,
        ):
            resume.resume_run(target=str(run_dir), codex_native=True)

        launch.assert_not_called()
        self.assertIn("canonical runtime", str(cm.exception))
        self.assertIn("Start a new", str(cm.exception))

    def test_resume_lifecycle_lock_open_failure_is_operator_error(self) -> None:
        from praxist.cli import resume

        task = _make_task_dir(Path(self.workspace.name), name="lock_error_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        with (
            patch("praxist.cli.registry.os.open", side_effect=OSError("read-only state")),
            self.assertRaises(resume.ResumeError) as cm,
        ):
            resume.resume_run(target=str(run_dir))

        self.assertIn("lifecycle lock", str(cm.exception))

    def test_malformed_resume_artifacts_are_rejected(self) -> None:
        from praxist.cli import resume

        task = _make_task_dir(Path(self.workspace.name), name="bad_artifact_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        (run_dir / "startup_config.json").write_text("[1, 2, 3]", encoding="utf-8")

        with self.assertRaises(resume.ResumeError) as cm:
            resume.resolve_resume_target(str(run_dir))
        self.assertIn("JSON object", str(cm.exception))
        self.assertIn("Manual inspection is required", str(cm.exception))

    def test_missing_resume_artifacts_are_rejected(self) -> None:
        from praxist.cli import resume

        missing_run_dir = Path(self.workspace.name) / "does_not_exist"

        with self.assertRaises(resume.ResumeError) as cm:
            resume.resolve_resume_target(str(missing_run_dir))
        self.assertIn("does not exist", str(cm.exception))

    def test_existing_directory_missing_startup_artifacts_requires_manual_inspection(self) -> None:
        from praxist.cli import resume

        empty_run_dir = Path(self.workspace.name) / "empty_existing_run"
        empty_run_dir.mkdir()

        with self.assertRaises(resume.ResumeError) as cm:
            resume.resolve_resume_target(str(empty_run_dir))
        self.assertIn("missing Praxist startup artifacts", str(cm.exception))
        self.assertIn("Manual inspection is required", str(cm.exception))

    def test_resume_malformed_json_artifact_requires_manual_inspection(self) -> None:
        from praxist.cli import resume

        task = _make_task_dir(Path(self.workspace.name), name="corrupt_json_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        (run_dir / "startup_config.json").write_text("{not-valid-json", encoding="utf-8")

        with self.assertRaises(resume.ResumeError) as cm:
            resume.resolve_resume_target(str(run_dir))
        self.assertIn("could not read resume artifact", str(cm.exception))
        self.assertIn("Manual inspection is required", str(cm.exception))

    def test_run_directory_with_multiple_registry_owners_is_rejected(self) -> None:
        from praxist.cli import registry, resume

        task = _make_task_dir(Path(self.workspace.name), name="ambiguous_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)
        for index in range(2):
            registry.write_entry(
                registry.RegistryEntry(
                    **_entry_kwargs(
                        run_id=f"run_ambiguous_{index}",
                        pid=5100 + index,
                        run_dir=str(run_dir),
                        task_path=str(task),
                        state=registry.STATE_STOPPED,
                    )
                )
            )

        with self.assertRaises(resume.ResumeError) as cm:
            resume.resolve_resume_target(str(run_dir))
        self.assertIn("multiple registry entries", str(cm.exception))

    def test_resume_helper_edges_preserve_conservative_identity_semantics(self) -> None:
        from praxist.cli import registry, resume

        entry = registry.RegistryEntry(
            **_entry_kwargs(
                command=(
                    "python",
                    "-m",
                    "praxist.run",
                    "--frontier-strategy=pareto",
                )
            )
        )
        with patch("praxist.cli.resume.entry_process_epoch_matches", return_value=False):
            self.assertEqual(resume._entry_liveness(entry), "stale")
        with (
            patch("praxist.cli.resume.entry_process_epoch_matches", return_value=None),
            patch("praxist.cli.resume.process_identity_matches", return_value=None),
            patch("praxist.cli.resume.read_ps_table", return_value={}),
            patch("praxist.cli.resume.pid_is_alive", return_value=True),
        ):
            self.assertEqual(resume._entry_liveness(entry), "unknown")
        self.assertEqual(resume._frontier_strategy_from_entry(entry), "pareto")
        self.assertTrue(resume._optional_bool(7, default=False))
        self.assertFalse(resume._optional_bool("unexpected", default=False))

    def test_codex_native_recovery_prefers_canonical_mode_and_valid_profile_list(self) -> None:
        from praxist.cli import resume

        run_dir = Path(self.workspace.name)
        self.assertTrue(resume._run_dir_used_codex_native(run_dir, {"codex_native": "yes"}))
        self.assertFalse(resume._run_dir_used_codex_native(run_dir, {"codex_native": "no"}))
        (run_dir / "credentials_redacted.json").write_text(
            json.dumps({"credential_profiles": "not-a-list"}),
            encoding="utf-8",
        )
        self.assertFalse(resume._run_dir_used_codex_native(run_dir, {}))

    def test_malformed_json_resume_artifact_reports_source_path(self) -> None:
        from praxist.cli import resume

        path = Path(self.workspace.name) / "malformed.json"
        path.write_text("{bad", encoding="utf-8")
        with self.assertRaises(resume.ResumeError) as cm:
            resume._read_json_object(path)
        self.assertIn(str(path), str(cm.exception))


class ResumeCliEndToEndTest(unittest.TestCase):
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
                "OPENAI_API_KEY": "",
                "DEEPSEEK_API_KEY": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        from praxist.cli import main

        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                main(argv)
            code = 0
        except SystemExit as exc:
            code = int(exc.code or 0)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_resume_subcommand_registered_and_returns_json(self) -> None:
        task = _make_task_dir(Path(self.workspace.name), name="cli_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)

        with patch(
            "praxist.cli.start._default_spawn",
            return_value=_FakeProc(pid=6789),
        ):
            code, out, err = self._run(["resume", str(run_dir), "--startup-timeout", "0", "--json"])

        self.assertEqual(code, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(payload["pid"], 6789)
        self.assertEqual(Path(payload["run_dir"]), run_dir.resolve())
        self.assertIn("--resume-from", payload["command"])
        self.assertEqual(
            payload["extra"]["monitor_command"],
            f"praxist --monitor --run-id {payload['run_id']}",
        )

    def test_resume_subcommand_text_output(self) -> None:
        task = _make_task_dir(Path(self.workspace.name), name="text_task")
        run_dir = _make_run_dir(Path(self.workspace.name), task_path=task)

        with patch(
            "praxist.cli.start._default_spawn",
            return_value=_FakeProc(pid=6790),
        ):
            code, out, err = self._run(["resume", str(run_dir), "--startup-timeout", "0"])

        self.assertEqual(code, 0, msg=err)
        self.assertEqual(out, "")
        self.assertIn("Praxist run resumed", err)
        self.assertIn("praxist --monitor --run-id", err)
        self.assertIn("praxist status", err)

    def test_resume_subcommand_returns_nonzero_for_confirmed_startup_failure(self) -> None:
        from praxist.cli import registry

        failed = registry.RegistryEntry(
            **_entry_kwargs(extra={"agent_system": "claude_sdk", "startup_state": "failed"})
        )
        with patch("praxist.cli.resume.resume_run", return_value=failed):
            code, out, err = self._run(["resume", failed.run_id])

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("resume failed during startup", err)
        self.assertNotIn("run resumed", err)

    def test_resume_subcommand_surfaces_errors(self) -> None:
        code, out, err = self._run(["resume", str(Path(self.workspace.name) / "missing")])

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("praxist resume:", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
