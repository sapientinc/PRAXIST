from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from praxist.plugins.workflow_stages.research_loop.backend import (
    experiment_exec,
    experiment_process,
    experiment_scheduler_client,
)
from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler import (
    ExperimentSchedulerService,
)
from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client import (
    ENV_SCHEDULER_ENDPOINT,
    ExperimentRejected,
    SchedulerUnavailable,
    _generation_from_peer,
    _rpc,
    freeze_all_for_run,
    freeze_generation,
    is_sensitive_environment_entry,
    is_sensitive_environment_name,
    recover_environment,
    semantic_experiment_key,
    sensitive_environment_matches,
    submit_and_wait,
)
from praxist.plugins.workflow_stages.research_loop.backend.resource_scheduler import (
    Allocation,
    GPUDevice,
    HostAllocationRegistry,
    HostObserver,
    HostSnapshot,
    ResourceAllocator,
    ResourceProfile,
    SchedulerSettings,
    _as_bool,
    _as_float,
    _as_int,
    _optional_positive_float,
    _pid_start_time,
    _process_group_alive,
)


class _StaticObserver:
    def __init__(self, *snapshots: HostSnapshot) -> None:
        self.snapshots = list(snapshots)

    def snapshot(self) -> HostSnapshot:
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]


class _MemoryRegistry:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.limits: list[tuple[str, int]] = []
        self.removed: list[str] = []

    def locked(self):
        rows = self.rows

        class _Lock:
            def __enter__(self):
                return rows

            def __exit__(self, *_args):
                return False

        return _Lock()

    def update_limit(self, *, run_id: str, pid: int, pgid: int, limit: int) -> None:
        del pid, pgid
        self.limits.append((run_id, limit))

    def remove_limit(self, run_id: str) -> None:
        self.removed.append(run_id)


class _LifecycleAllocator:
    def __init__(self) -> None:
        self.concurrency_limit = 1
        self.snapshot = HostSnapshot(1, 0, 0, 0)
        self.owners: list[str] = []
        self.closed = False

    def set_owner(self, run_id: str) -> None:
        self.owners.append(run_id)

    def close(self) -> None:
        self.closed = True

    def refresh(self, *, queued: bool) -> HostSnapshot:
        del queued
        return self.snapshot

    def reserve(self, **_kwargs):
        return None


def _settings(*, enabled: bool = True) -> SchedulerSettings:
    return SchedulerSettings.from_dict(
        {
            "mode": "central" if enabled else "legacy",
            "initial_concurrent_experiments": 2,
            "min_concurrent_experiments": 1,
            "max_concurrent_experiments": 3,
            "adjustment_interval_seconds": 2,
            "profiles": {
                "cpu": {"accelerator": "cpu"},
                "gpu": {
                    "accelerator": "gpu",
                    "gpu_count": 1,
                    "gpu_memory_gb": 1,
                    "gpu_utilization_pct": 20,
                },
            },
            "default_profile": "cpu",
        }
    )


class LaunchBarrierBoundaryTest(unittest.TestCase):
    def test_barrier_rejects_missing_arguments(self) -> None:
        with patch.object(sys, "argv", ["experiment_exec"]):
            self.assertEqual(experiment_exec.main(), 75)

    def test_barrier_records_identity_and_executes_only_after_go(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ready = Path(td) / "READY.json"
            go = Path(td) / "GO.json"
            go.write_text("{}", encoding="utf-8")
            with (
                patch.object(
                    sys,
                    "argv",
                    ["experiment_exec", str(ready), str(go), "attempt-1", "task"],
                ),
                patch.object(os, "execvpe", side_effect=OSError("not executable")) as execute,
            ):
                self.assertEqual(experiment_exec.main(), 75)
            payload = json.loads(ready.read_text(encoding="utf-8"))
            self.assertEqual(payload["pid"], os.getpid())
            self.assertEqual(payload["pgid"], os.getpgrp())
            self.assertEqual(payload["attempt_id"], "attempt-1")
            self.assertEqual(payload["pid_start_time"], _pid_start_time(os.getpid()))
            execute.assert_called_once()

    def test_barrier_tolerates_missing_process_start_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ready = Path(td) / "READY.json"
            go = Path(td) / "GO.json"
            go.write_text("{}", encoding="utf-8")
            with (
                patch.object(
                    sys,
                    "argv",
                    ["experiment_exec", str(ready), str(go), "attempt-2", "task"],
                ),
                patch.object(Path, "read_text", side_effect=OSError("proc unavailable")),
                patch.object(experiment_exec.shutil, "which", return_value=None),
                patch.object(os, "execvpe", side_effect=OSError("not executable")),
            ):
                self.assertEqual(experiment_exec.main(), 75)
            self.assertIsNone(json.loads(ready.read_text(encoding="utf-8"))["pid_start_time"])

    def test_barrier_uses_ps_identity_without_procfs(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["ps"],
            returncode=0,
            stdout="Thu Sep  5 10:11:12 2026\n",
            stderr="",
        )
        with (
            patch.object(Path, "read_text", side_effect=OSError("proc unavailable")),
            patch.object(experiment_exec.shutil, "which", return_value="/usr/bin/ps"),
            patch.object(experiment_exec.subprocess, "run", return_value=completed),
        ):
            observed = experiment_exec._pid_start_time(4321)

        self.assertEqual(observed, "ps:Thu Sep 5 10:11:12 2026")

    def test_barrier_tolerates_ps_identity_probe_failure(self) -> None:
        with (
            patch.object(Path, "read_text", side_effect=OSError("proc unavailable")),
            patch.object(experiment_exec.shutil, "which", return_value="/usr/bin/ps"),
            patch.object(experiment_exec.subprocess, "run", side_effect=OSError("ps unavailable")),
        ):
            self.assertIsNone(experiment_exec._pid_start_time(4321))

    def test_barrier_times_out_without_scheduler_commit(self) -> None:
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(
                sys,
                "argv",
                [
                    "experiment_exec",
                    f"{td}/READY.json",
                    f"{td}/GO.json",
                    "attempt-1",
                    "task",
                ],
            ),
            patch.object(experiment_exec.time, "monotonic", side_effect=[10.0, 10.0, 311.0]),
            patch.object(experiment_exec.time, "sleep") as sleep,
        ):
            self.assertEqual(experiment_exec.main(), 75)
        sleep.assert_called_once_with(0.05)


class ProcessGroupBoundaryTest(unittest.TestCase):
    def test_process_group_visibility_contract(self) -> None:
        self.assertFalse(experiment_process.process_group_alive(1))
        with patch.object(os, "killpg"):
            self.assertTrue(experiment_process.process_group_alive(42))
        with patch.object(os, "killpg", side_effect=ProcessLookupError):
            self.assertFalse(experiment_process.process_group_alive(42))
        with (
            patch.object(os, "killpg", side_effect=PermissionError),
            patch.object(experiment_process, "_linux_process_group_activity", return_value=None),
        ):
            self.assertTrue(experiment_process.process_group_alive(42))

    def test_permission_denied_zombie_group_is_not_active_work(self) -> None:
        with (
            patch.object(os, "killpg", side_effect=PermissionError),
            patch.object(
                experiment_process,
                "_linux_process_group_activity",
                return_value=False,
            ),
        ):
            self.assertFalse(experiment_process.process_group_alive(42))

    def test_linux_process_group_scan_defers_on_other_platforms(self) -> None:
        with patch.object(experiment_process.sys, "platform", "darwin"):
            self.assertIsNone(experiment_process._linux_process_group_activity(42))

    @unittest.skipUnless(hasattr(os, "fork") and Path("/proc/self/stat").exists(), "Linux procfs")
    def test_zombie_only_process_group_is_not_active_work(self) -> None:
        ready_read, ready_write = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(ready_read)
                os.setsid()
                os.write(ready_write, b"1")
            finally:
                os._exit(0)

        os.close(ready_write)
        try:
            self.assertEqual(os.read(ready_read, 1), b"1")
            deadline = time.monotonic() + 2
            state = ""
            while time.monotonic() < deadline:
                try:
                    state = (
                        Path(f"/proc/{pid}/stat")
                        .read_text(encoding="utf-8")
                        .rsplit(")", 1)[1]
                        .split()[0]
                    )
                except (OSError, IndexError):
                    break
                if state == "Z":
                    break
                time.sleep(0.01)
            self.assertEqual(state, "Z")
            self.assertFalse(experiment_process.process_group_alive(pid))
        finally:
            os.close(ready_read)
            os.waitpid(pid, 0)

    def test_terminate_reaps_after_graceful_exit(self) -> None:
        process = MagicMock()
        with (
            patch.object(
                experiment_process, "process_group_alive", side_effect=[True, False, False]
            ),
            patch.object(experiment_process.time, "monotonic", side_effect=[0.0, 0.1]),
            patch.object(experiment_process.time, "sleep"),
            patch.object(os, "killpg") as killpg,
        ):
            experiment_process.terminate_process_group(42, process, grace_seconds=1)
        killpg.assert_called_once_with(42, experiment_process.signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=1)

    def test_terminate_escalates_and_falls_back_to_process(self) -> None:
        with (
            patch.object(experiment_process, "process_group_alive", side_effect=[True, True]),
            patch.object(experiment_process.time, "monotonic", side_effect=[0.0, 1.0]),
            patch.object(os, "killpg") as killpg,
        ):
            experiment_process.terminate_process_group(42, grace_seconds=0)
        self.assertEqual(killpg.call_count, 2)

        process = MagicMock()
        with patch.object(os, "killpg", side_effect=OSError("host error")):
            experiment_process.terminate_process_group(42, process)
        process.kill.assert_called_once()
        process.wait.assert_called_once()


class SchedulerClientBoundaryTest(unittest.TestCase):
    def test_task_command_resolution_handles_common_wrappers_without_rewriting_shell_logic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            task_root = Path(td)
            evaluator = task_root / "evaluations" / "run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            environment = {"PRAXIST_TASK_PROJECT_PATH": str(task_root)}

            versioned = experiment_scheduler_client.resolve_task_command_path(
                ["python3.8", "-u", "evaluations/run.py"],
                environment,
            )
            wrapped = experiment_scheduler_client.resolve_task_command_path(
                ["env", "MODE=test", "python", "evaluations/run.py"],
                environment,
            )
            hash_checked = experiment_scheduler_client.resolve_task_command_path(
                [
                    "python",
                    "--check-hash-based-pycs",
                    "always",
                    "evaluations/run.py",
                ],
                environment,
            )
            shell = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "python -u evaluations/run.py"],
                environment,
            )
            compound = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "python evaluations/run.py > result.log"],
                environment,
            )
            attached_redirect = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "python evaluations/run.py 2>result.log"],
                environment,
            )
            descriptor_redirect = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "python evaluations/run.py >result.log 2>&1"],
                environment,
            )
            versioned_shell = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "python3.11 evaluations/run.py"],
                {**environment, "PRAXIST_TASK_PYTHON": sys.executable},
            )
            comment = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "python evaluations/run.py # preserve operator note"],
                environment,
            )
            activation = task_root / ".venv" / "bin" / "activate"
            activation.parent.mkdir(parents=True)
            activation.write_text("", encoding="utf-8")
            chained = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "source .venv/bin/activate && python evaluations/run.py"],
                environment,
            )
            env_chdir = experiment_scheduler_client.resolve_task_command_path(
                ["env", "-C", "evaluations", "python", "run.py"],
                environment,
            )
            assigned_shell = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "MODE=test python evaluations/run.py"],
                environment,
            )
            task_python_shell = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "$PRAXIST_TASK_PYTHON evaluations/run.py"],
                {**environment, "PRAXIST_TASK_PYTHON": sys.executable},
            )
            exec_shell = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "exec python evaluations/run.py"],
                environment,
            )
            command_shell = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", "command -p python evaluations/run.py"],
                environment,
            )
            non_executable_shadow = task_root / "echo"
            non_executable_shadow.write_text("not an executable\n", encoding="utf-8")
            bare_path_command = experiment_scheduler_client.resolve_task_command_path(
                ["echo", "ok"],
                environment,
            )

        self.assertEqual(versioned[-1], str(evaluator.resolve()))
        self.assertEqual(wrapped[-1], str(evaluator.resolve()))
        self.assertEqual(hash_checked[-1], str(evaluator.resolve()))
        self.assertIn(str(evaluator.resolve()), shell[-1])
        self.assertEqual(compound[-1], f"python {evaluator.resolve()} > result.log")
        self.assertEqual(attached_redirect[-1], f"python {evaluator.resolve()} 2>result.log")
        self.assertEqual(
            descriptor_redirect[-1],
            f"python {evaluator.resolve()} >result.log 2>&1",
        )
        self.assertEqual(versioned_shell[-1], f"{sys.executable} {evaluator.resolve()}")
        self.assertEqual(
            comment[-1],
            f"python {evaluator.resolve()} # preserve operator note",
        )
        self.assertEqual(
            chained[-1],
            f"source {activation.resolve()} && python {evaluator.resolve()}",
        )
        self.assertEqual(
            env_chdir,
            [
                "env",
                "-C",
                str(evaluator.parent.resolve()),
                "python",
                str(evaluator.resolve()),
            ],
        )
        self.assertEqual(
            assigned_shell[-1],
            f"MODE=test python {evaluator.resolve()}",
        )
        self.assertEqual(
            task_python_shell[-1],
            f"$PRAXIST_TASK_PYTHON {evaluator.resolve()}",
        )
        self.assertEqual(exec_shell[-1], f"exec python {evaluator.resolve()}")
        self.assertEqual(
            command_shell[-1],
            f"command -p python {evaluator.resolve()}",
        )
        self.assertEqual(bare_path_command, ["echo", "ok"])

    def test_task_environment_isolates_runner_python_paths_without_task_python(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            task_root = Path(td)
            command, child, cwd = experiment_scheduler_client.prepare_task_subprocess(
                [sys.executable, "-c", "print('ok')"],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PYTHONPATH": "/runner/python313",
                    "PYTHONHOME": "/runner/python313",
                },
            )
            _preserved_command, preserved, _preserved_cwd = (
                experiment_scheduler_client.prepare_task_subprocess(
                    [sys.executable, "-c", "print('ok')"],
                    {
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_TASK_RUNTIME_ENV_KEYS": "PYTHONPATH,PYTHONHOME",
                        "PYTHONPATH": "/task/imports",
                        "PYTHONHOME": "/task/python",
                    },
                )
            )
            guard_site = task_root / "run" / ".runtime_guards" / "gen0_peer0" / "python_site"
            guard_site.mkdir(parents=True)
            (guard_site / "sitecustomize.py").write_text("", encoding="utf-8")
            _guarded_command, guarded, _guarded_cwd = (
                experiment_scheduler_client.prepare_task_subprocess(
                    [sys.executable, "-c", "print('ok')"],
                    {
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_DELETE_GUARD_RUN_DIR": str(task_root / "run"),
                        "PRAXIST_DELETE_GUARD_AGENT": "gen0_peer0",
                        "PYTHONPATH": os.pathsep.join([str(guard_site), "/runner/python313"]),
                    },
                )
            )
            runner_root = task_root / "runner"
            helper_root = task_root / "legacy_helpers"
            runner_root.mkdir()
            helper_root.mkdir()
            (helper_root / "legacy_helper.py").write_text(
                "VALUE = 'legacy-import-ok'\n",
                encoding="utf-8",
            )
            legacy_command, legacy_environment, legacy_cwd = (
                experiment_scheduler_client.prepare_task_subprocess(
                    [
                        sys.executable,
                        "-c",
                        "import legacy_helper; print(legacy_helper.VALUE)",
                    ],
                    {
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_WORKSPACE_ROOT": str(runner_root),
                        "PYTHONPATH": os.pathsep.join([str(runner_root), str(helper_root)]),
                    },
                )
            )
            legacy_result = subprocess.run(
                legacy_command,
                cwd=legacy_cwd,
                env=legacy_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            _owned_command, owned_environment, _owned_cwd = (
                experiment_scheduler_client.prepare_task_subprocess(
                    [sys.executable, "-c", "print('ok')"],
                    {
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_WORKSPACE_ROOT": str(runner_root),
                        "PRAXIST_TASK_PYTHON": sys.executable,
                        "PRAXIST_TASK_RUNTIME_ENV_KEYS": "PYTHONPATH",
                        "PYTHONPATH": os.pathsep.join([str(runner_root), str(helper_root)]),
                    },
                )
            )

        self.assertEqual(command, [sys.executable, "-c", "print('ok')"])
        self.assertEqual(cwd, str(task_root.resolve()))
        self.assertNotIn("PYTHONPATH", child)
        self.assertEqual(child["PYTHONHOME"], "/runner/python313")
        self.assertEqual(preserved["PYTHONPATH"], "/task/imports")
        self.assertEqual(preserved["PYTHONHOME"], "/task/python")
        self.assertEqual(guarded["PYTHONPATH"], str(guard_site.resolve()))
        self.assertEqual(legacy_environment["PYTHONPATH"], str(helper_root))
        self.assertEqual(
            owned_environment["PYTHONPATH"],
            os.pathsep.join([str(runner_root), str(helper_root)]),
        )
        self.assertEqual(legacy_result.returncode, 0, legacy_result.stderr)
        self.assertEqual(legacy_result.stdout.strip(), "legacy-import-ok")

    def test_static_shell_wrappers_use_the_declared_task_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            task_root = root / "task"
            run_cwd = root / "run"
            evaluator = task_root / "evaluations" / "run.py"
            task_python = task_root / ".venv" / "bin" / "python"
            marker = root / "task-python-used.txt"
            evaluator.parent.mkdir(parents=True)
            task_python.parent.mkdir(parents=True)
            run_cwd.mkdir()
            evaluator.write_text("print('task-interpreter-ok')\n", encoding="utf-8")
            task_python.write_text(
                "#!/bin/sh\n"
                f"printf used > {shlex.quote(str(marker))}\n"
                f'exec {shlex.quote(sys.executable)} "$@"\n',
                encoding="utf-8",
            )
            task_python.chmod(0o755)

            for wrapper in ("exec", "command -p", "command -p env MODE=test"):
                with self.subTest(wrapper=wrapper):
                    marker.unlink(missing_ok=True)
                    command, environment, resolved_cwd = (
                        experiment_scheduler_client.prepare_task_subprocess(
                            [
                                "bash",
                                "-lc",
                                f"{wrapper} python evaluations/run.py",
                            ],
                            {
                                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                                "PRAXIST_TASK_PYTHON": str(task_python),
                            },
                            cwd=run_cwd,
                        )
                    )
                    completed = subprocess.run(
                        command,
                        cwd=resolved_cwd,
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertEqual(completed.stdout.strip(), "task-interpreter-ok")
                    self.assertEqual(marker.read_text(encoding="utf-8"), "used")

    def test_static_env_chdir_runs_task_evaluator_from_non_task_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            task_root = root / "task"
            run_cwd = root / "run"
            evaluator = task_root / "evaluations" / "v2" / "run.py"
            task_python = task_root / ".venv" / "bin" / "python"
            marker = root / "task-python-used.txt"
            evaluator.parent.mkdir(parents=True)
            task_python.parent.mkdir(parents=True)
            run_cwd.mkdir()
            evaluator.write_text("print('env-chdir-ok')\n", encoding="utf-8")
            task_python.write_text(
                "#!/bin/sh\n"
                f"printf used > {shlex.quote(str(marker))}\n"
                f'exec {shlex.quote(sys.executable)} "$@"\n',
                encoding="utf-8",
            )
            task_python.chmod(0o755)

            command, environment, resolved_cwd = (
                experiment_scheduler_client.prepare_task_subprocess(
                    ["env", "-C", "evaluations/v2", "python", "run.py"],
                    {
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_TASK_PYTHON": str(task_python),
                    },
                    cwd=run_cwd,
                )
            )
            completed = subprocess.run(
                command,
                cwd=resolved_cwd,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "env-chdir-ok")
            self.assertEqual(marker.read_text(encoding="utf-8"), "used")
            self.assertEqual(command[2], str(evaluator.parent.resolve()))
            self.assertEqual(command[3], str(task_python.resolve()))
            self.assertEqual(command[4], str(evaluator.resolve()))

    def test_dynamic_env_chdir_remains_runtime_resolved(self) -> None:
        command = ["env", "-C", "$TASK_EVAL_DIR", "python", "run.py"]

        resolved = experiment_scheduler_client.resolve_task_command_path(
            command,
            {"PRAXIST_TASK_PROJECT_PATH": "/unused/task"},
        )

        self.assertEqual(resolved, command)

    def test_static_shell_cd_preserves_shell_relative_evaluator_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            run_cwd = root / "run"
            task_python = task_root / ".venv" / "bin" / "python"
            root_evaluator = task_root / "run.py"
            nested_evaluator = task_root / "sub" / "run.py"
            task_python.parent.mkdir(parents=True)
            nested_evaluator.parent.mkdir(parents=True)
            run_cwd.mkdir()
            root_evaluator.write_text("print('wrong-root')\n", encoding="utf-8")
            nested_evaluator.write_text("print('nested-ok')\n", encoding="utf-8")
            task_python.write_text(
                f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n',
                encoding="utf-8",
            )
            task_python.chmod(0o755)

            command, environment, resolved_cwd = (
                experiment_scheduler_client.prepare_task_subprocess(
                    ["bash", "-lc", "cd sub && python run.py"],
                    {
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_TASK_PYTHON": str(task_python),
                    },
                    cwd=run_cwd,
                )
            )
            completed = subprocess.run(
                command,
                cwd=resolved_cwd,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "nested-ok")
        self.assertIn(f"cd {nested_evaluator.parent.resolve()}", command[-1])
        self.assertNotIn(str(root_evaluator.resolve()), command[-1])

    def test_resolved_shell_evaluator_runs_from_non_task_cwd_with_redirection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            run_cwd = root / "run" / "peer"
            evaluator = task_root / "evaluations" / "run.py"
            evaluator.parent.mkdir(parents=True)
            run_cwd.mkdir(parents=True)
            evaluator.write_text("print('resolved-from-task-root')\n", encoding="utf-8")
            command = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", f"{sys.executable} evaluations/run.py > result.log"],
                {"PRAXIST_TASK_PROJECT_PATH": str(task_root)},
            )

            completed = subprocess.run(command, cwd=run_cwd, check=False)

            self.assertEqual(completed.returncode, 0)
            self.assertEqual(
                (run_cwd / "result.log").read_text(encoding="utf-8").strip(),
                "resolved-from-task-root",
            )

            evaluator.write_text(
                "import sys\nprint('resolved-stderr', file=sys.stderr)\n",
                encoding="utf-8",
            )
            stderr_command = experiment_scheduler_client.resolve_task_command_path(
                ["bash", "-lc", f"{sys.executable} evaluations/run.py 2>stderr.log"],
                {"PRAXIST_TASK_PROJECT_PATH": str(task_root)},
            )
            completed = subprocess.run(stderr_command, cwd=run_cwd, check=False)
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(
                (run_cwd / "stderr.log").read_text(encoding="utf-8").strip(),
                "resolved-stderr",
            )

            activation = task_root / ".venv" / "bin" / "activate"
            activation.parent.mkdir(parents=True)
            activation.write_text("", encoding="utf-8")
            evaluator.write_text("print('resolved-from-task-root')\n", encoding="utf-8")
            chained = experiment_scheduler_client.prepare_task_subprocess(
                [
                    "bash",
                    "-lc",
                    "source .venv/bin/activate && python evaluations/run.py > chained.log",
                ],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                },
                cwd=run_cwd,
            )[0]
            completed = subprocess.run(chained, cwd=run_cwd, check=False)
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(
                (run_cwd / "chained.log").read_text(encoding="utf-8").strip(),
                "resolved-from-task-root",
            )

            dynamic = experiment_scheduler_client.prepare_task_subprocess(
                [
                    "bash",
                    "-lc",
                    'python3.8 evaluations/run.py > "${PRAXIST_RUN_DIR}/dynamic.log"',
                ],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                    "PRAXIST_RUN_DIR": str(run_cwd),
                },
                cwd=run_cwd,
            )[0]
            completed = subprocess.run(
                dynamic,
                cwd=run_cwd,
                env={**os.environ, "PRAXIST_RUN_DIR": str(run_cwd)},
                check=False,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(
                (run_cwd / "dynamic.log").read_text(encoding="utf-8").strip(),
                "resolved-from-task-root",
            )

    def test_explicit_cwd_evaluator_precedes_task_root_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            candidate_cwd = root / "run" / "candidate"
            task_root.mkdir()
            candidate_cwd.mkdir(parents=True)
            (task_root / "evaluate.py").write_text("print('task')\n", encoding="utf-8")
            candidate_evaluator = candidate_cwd / "evaluate.py"
            candidate_evaluator.write_text("print('candidate')\n", encoding="utf-8")

            command, _environment, resolved_cwd = (
                experiment_scheduler_client.prepare_task_subprocess(
                    ["python", "evaluate.py"],
                    {
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_TASK_PYTHON": sys.executable,
                    },
                    cwd=candidate_cwd,
                )
            )

        self.assertEqual(command, [sys.executable, str(candidate_evaluator.resolve())])
        self.assertEqual(resolved_cwd, str(candidate_cwd.resolve()))

    def test_declared_evaluator_prefers_task_root_over_stale_cwd_copy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            candidate_cwd = root / "run" / "candidate"
            task_evaluator = task_root / "evaluations" / "run.py"
            stale_evaluator = candidate_cwd / "evaluations" / "run.py"
            task_evaluator.parent.mkdir(parents=True)
            stale_evaluator.parent.mkdir(parents=True)
            task_evaluator.write_text("print('task')\n", encoding="utf-8")
            stale_evaluator.write_text("print('stale')\n", encoding="utf-8")
            environment = {
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                "PRAXIST_TASK_PYTHON": sys.executable,
                "PRAXIST_EVALUATION_ENTRYPOINT": "python evaluations/run.py",
            }

            direct, _environment, _cwd = experiment_scheduler_client.prepare_task_subprocess(
                ["python", "evaluations/run.py"],
                environment,
                cwd=candidate_cwd,
            )
            env_chdir, _environment, _cwd = experiment_scheduler_client.prepare_task_subprocess(
                ["env", "-C", "evaluations", "python", "run.py"],
                environment,
                cwd=candidate_cwd,
            )

        self.assertEqual(direct, [sys.executable, str(task_evaluator.resolve())])
        self.assertEqual(env_chdir[2], str(task_evaluator.parent.resolve()))
        self.assertEqual(env_chdir[-1], str(task_evaluator.resolve()))

    def test_declared_command_uses_resolved_runtime_cwd_evaluator_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            run_cwd = root / "run" / "peer"
            task_evaluator = task_root / "evaluations" / "run.py"
            stale_evaluator = run_cwd / "run.py"
            task_evaluator.parent.mkdir(parents=True)
            run_cwd.mkdir(parents=True)
            task_evaluator.write_text("print('task')\n", encoding="utf-8")
            stale_evaluator.write_text("print('stale')\n", encoding="utf-8")

            command, _environment, _cwd = experiment_scheduler_client.prepare_task_subprocess(
                ["python", "run.py"],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                    "PRAXIST_EVALUATION_ENTRYPOINT": "python run.py",
                    "PRAXIST_EVALUATION_ENTRYPOINT_PATH": str(task_evaluator),
                },
                cwd=run_cwd,
            )

        self.assertEqual(command, [sys.executable, str(task_evaluator.resolve())])

    def test_shell_declared_evaluator_resolves_static_cd_and_redirection(self) -> None:
        from praxist.task_spec import resolve_declared_evaluation_entrypoint

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            run_cwd = root / "run" / "peer"
            task_evaluator = task_root / "evals" / "run.py"
            stale_evaluator = run_cwd / "evals" / "run.py"
            task_evaluator.parent.mkdir(parents=True)
            stale_evaluator.parent.mkdir(parents=True)
            task_evaluator.write_text("print('task')\n", encoding="utf-8")
            stale_evaluator.write_text("print('stale')\n", encoding="utf-8")
            declaration = "bash -lc 'cd evals && python run.py 2>&1'"

            resolved = resolve_declared_evaluation_entrypoint(
                declaration,
                task_dir=task_root,
            )
            command, _environment, _cwd = experiment_scheduler_client.prepare_task_subprocess(
                ["bash", "-lc", "cd evals && python run.py 2>&1"],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                    "PRAXIST_EVALUATION_ENTRYPOINT": declaration,
                },
                cwd=run_cwd,
            )

        self.assertEqual(resolved, task_evaluator.resolve())
        self.assertIn(f"cd {task_evaluator.parent.resolve()}", command[-1])
        self.assertNotIn(str(stale_evaluator.parent.resolve()), command[-1])

    def test_combined_shell_options_resolve_and_normalize_task_evaluator(self) -> None:
        from praxist.task_spec import resolve_declared_evaluation_entrypoint

        with tempfile.TemporaryDirectory() as td:
            task_root = Path(td) / "task"
            evaluator = task_root / "evaluations" / "run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            environment = {
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                "PRAXIST_TASK_PYTHON": sys.executable,
                "PRAXIST_EVALUATION_ENTRYPOINT": "python evaluations/run.py",
                "PRAXIST_EVALUATION_ENTRYPOINT_PATH": str(evaluator),
            }

            for shell, options in (("bash", "-euc"), ("sh", "-xec")):
                with self.subTest(shell=shell, options=options):
                    declaration = f"{shell} {options} 'python evaluations/run.py'"
                    resolved = resolve_declared_evaluation_entrypoint(
                        declaration,
                        task_dir=task_root,
                    )
                    command, _environment, _cwd = (
                        experiment_scheduler_client.prepare_task_subprocess(
                            [shell, options, "python evaluations/run.py"],
                            environment,
                            cwd=task_root,
                        )
                    )
                    self.assertEqual(resolved, evaluator.resolve())
                    self.assertIn(str(evaluator.resolve()), command[-1])

    def test_combined_shell_options_honor_explicit_task_runtime_cwd(self) -> None:
        from praxist.task_spec import resolve_declared_evaluation_entrypoint

        with tempfile.TemporaryDirectory() as td:
            task_root = Path(td) / "task"
            evaluator = task_root / "evaluations" / "run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")

            resolved = resolve_declared_evaluation_entrypoint(
                "bash -euc 'python run.py'",
                task_dir=task_root,
                runtime_cwd="evaluations",
            )

        self.assertEqual(resolved, evaluator.resolve())

    def test_declared_evaluator_task_root_precedes_stale_runtime_copy(self) -> None:
        from praxist.task_spec import resolve_declared_evaluation_entrypoint

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            run_cwd = root / "run" / "peer"
            task_evaluator = task_root / "evaluations" / "run.py"
            stale_evaluator = run_cwd / "evaluations" / "run.py"
            task_evaluator.parent.mkdir(parents=True)
            stale_evaluator.parent.mkdir(parents=True)
            task_evaluator.write_text("print('task')\n", encoding="utf-8")
            stale_evaluator.write_text("print('stale')\n", encoding="utf-8")

            resolved = resolve_declared_evaluation_entrypoint(
                "python evaluations/run.py",
                task_dir=task_root,
                runtime_cwd="run_dir",
            )

        self.assertEqual(resolved, task_evaluator.resolve())

    def test_explicit_task_runtime_cwd_precedes_same_named_task_root_evaluator(self) -> None:
        from praxist.task_spec import resolve_declared_evaluation_entrypoint

        with tempfile.TemporaryDirectory() as td:
            task_root = Path(td) / "task"
            configured = task_root / "evaluations" / "run.py"
            root_copy = task_root / "run.py"
            configured.parent.mkdir(parents=True)
            configured.write_text("print('configured')\n", encoding="utf-8")
            root_copy.write_text("print('root')\n", encoding="utf-8")

            resolved = resolve_declared_evaluation_entrypoint(
                "python run.py",
                task_dir=task_root,
                runtime_cwd="evaluations",
            )

        self.assertEqual(resolved, configured.resolve())

    def test_shell_evaluator_path_is_rewritten_next_to_control_operator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            run_cwd = root / "run" / "peer"
            task_evaluator = task_root / "evaluations" / "run.py"
            stale_evaluator = run_cwd / "evaluations" / "run.py"
            task_evaluator.parent.mkdir(parents=True)
            stale_evaluator.parent.mkdir(parents=True)
            task_evaluator.write_text("print('task')\n", encoding="utf-8")
            stale_evaluator.write_text("print('stale')\n", encoding="utf-8")

            command, _environment, _cwd = experiment_scheduler_client.prepare_task_subprocess(
                ["bash", "-lc", "python evaluations/run.py&&echo done"],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                    "PRAXIST_EVALUATION_ENTRYPOINT": "python evaluations/run.py",
                    "PRAXIST_EVALUATION_ENTRYPOINT_PATH": str(task_evaluator),
                },
                cwd=run_cwd,
            )

        self.assertIn(str(task_evaluator.resolve()), command[-1])
        self.assertIn("&&echo done", command[-1])
        self.assertNotIn(str(stale_evaluator.resolve()), command[-1])

    def test_nested_shell_cd_chain_anchors_declared_evaluator_to_task_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            run_cwd = root / "run" / "peer"
            task_evaluator = task_root / "evaluations" / "v2" / "run.py"
            stale_evaluator = run_cwd / "evaluations" / "v2" / "run.py"
            task_evaluator.parent.mkdir(parents=True)
            stale_evaluator.parent.mkdir(parents=True)
            task_evaluator.write_text("print('task')\n", encoding="utf-8")
            stale_evaluator.write_text("print('stale')\n", encoding="utf-8")
            declaration = "bash -lc 'cd evaluations && cd v2 && python run.py'"

            command, _environment, _cwd = experiment_scheduler_client.prepare_task_subprocess(
                ["bash", "-lc", "cd evaluations && cd v2 && python run.py"],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                    "PRAXIST_EVALUATION_ENTRYPOINT": declaration,
                    "PRAXIST_EVALUATION_ENTRYPOINT_PATH": str(task_evaluator),
                },
                cwd=run_cwd,
            )

        self.assertIn(f"cd {task_evaluator.parent.parent.resolve()}", command[-1])
        self.assertIn(f"cd {task_evaluator.parent.resolve()}", command[-1])
        self.assertIn(str(task_evaluator.resolve()), command[-1])
        self.assertNotIn(str(stale_evaluator.parent.resolve()), command[-1])

    def test_declared_shell_resolves_task_setup_before_cd_chain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            run_cwd = root / "run" / "peer"
            setup = task_root / "scripts" / "setup.sh"
            evaluator = task_root / "evaluations" / "run.py"
            setup.parent.mkdir(parents=True)
            evaluator.parent.mkdir(parents=True)
            run_cwd.mkdir(parents=True)
            setup.write_text("export TASK_READY=1\n", encoding="utf-8")
            evaluator.write_text("print('task')\n", encoding="utf-8")
            declaration = "bash -lc 'source scripts/setup.sh && cd evaluations && python run.py'"

            command, _environment, _cwd = experiment_scheduler_client.prepare_task_subprocess(
                [
                    "bash",
                    "-lc",
                    "source scripts/setup.sh && cd evaluations && python run.py",
                ],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                    "PRAXIST_EVALUATION_ENTRYPOINT": declaration,
                    "PRAXIST_EVALUATION_ENTRYPOINT_PATH": str(evaluator),
                },
                cwd=run_cwd,
            )

        self.assertIn(f"source {setup.resolve()}", command[-1])
        self.assertIn(f"cd {evaluator.parent.resolve()}", command[-1])
        self.assertIn(str(evaluator.resolve()), command[-1])

    def test_failed_cd_alternative_keeps_declared_evaluator_at_task_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            run_cwd = root / "run" / "peer"
            evaluator = task_root / "evaluations" / "run.py"
            stale = run_cwd / "evaluations" / "run.py"
            evaluator.parent.mkdir(parents=True)
            stale.parent.mkdir(parents=True)
            evaluator.write_text("print('task')\n", encoding="utf-8")
            stale.write_text("print('stale')\n", encoding="utf-8")
            declaration = "bash -lc 'cd missing || python evaluations/run.py'"

            command, _environment, _cwd = experiment_scheduler_client.prepare_task_subprocess(
                ["bash", "-lc", "cd missing || python evaluations/run.py"],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                    "PRAXIST_EVALUATION_ENTRYPOINT": declaration,
                    "PRAXIST_EVALUATION_ENTRYPOINT_PATH": str(evaluator),
                },
                cwd=run_cwd,
            )

        self.assertIn(str(evaluator.resolve()), command[-1])
        self.assertNotIn(str(stale.resolve()), command[-1])

    def test_declared_env_chdir_does_not_capture_unrelated_same_basename_script(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            custom_cwd = root / "run" / "candidate"
            task_evaluator = task_root / "evaluations" / "run.py"
            custom_script = custom_cwd / "run.py"
            task_evaluator.parent.mkdir(parents=True)
            custom_cwd.mkdir(parents=True)
            task_evaluator.write_text("print('task')\n", encoding="utf-8")
            custom_script.write_text("print('custom')\n", encoding="utf-8")

            command, _environment, _cwd = experiment_scheduler_client.prepare_task_subprocess(
                ["python", "run.py"],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": sys.executable,
                    "PRAXIST_EVALUATION_ENTRYPOINT": ("env -C evaluations python run.py"),
                    "PRAXIST_EVALUATION_ENTRYPOINT_PATH": str(task_evaluator),
                },
                cwd=custom_cwd,
            )

        self.assertEqual(command, [sys.executable, str(custom_script.resolve())])

    def test_explicit_relative_cwd_uses_callers_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            candidate_cwd = root / "run" / "candidate"
            task_root.mkdir()
            candidate_cwd.mkdir(parents=True)
            (task_root / "evaluate.py").write_text("print('task')\n", encoding="utf-8")
            candidate_evaluator = candidate_cwd / "evaluate.py"
            candidate_evaluator.write_text("print('candidate')\n", encoding="utf-8")

            with patch.object(
                experiment_scheduler_client.Path,
                "cwd",
                return_value=candidate_cwd,
            ):
                command, _environment, resolved_cwd = (
                    experiment_scheduler_client.prepare_task_subprocess(
                        ["python", "evaluate.py"],
                        {
                            "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                            "PRAXIST_TASK_PYTHON": sys.executable,
                        },
                        cwd=Path("."),
                    )
                )

        self.assertEqual(command, [sys.executable, str(candidate_evaluator.resolve())])
        self.assertEqual(resolved_cwd, str(candidate_cwd.resolve()))

    def test_supply_and_control_client_wrappers_cover_endpoint_and_legacy_modes(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(experiment_scheduler_client.register_idle_supply("gen0_peer0", 0), {})
            self.assertEqual(experiment_scheduler_client.get_supply_lease("gen0_peer0", 0, ""), {})
            self.assertEqual(experiment_scheduler_client.generation_advice("gen0_peer0", 0), {})
            self.assertFalse(experiment_scheduler_client.begin_assessment(0))
            experiment_scheduler_client.unregister_idle_supply("gen0_peer0", 0)
            experiment_scheduler_client.release_supply_lease("", "gen0_peer0")
            experiment_scheduler_client.freeze_generation(0)

        responses = [
            {"supply": {"lease_id": "lease"}},
            {},
            {"supply": {"lease_id": "lease"}},
            {},
            {"advice": {"first_wave": "explore"}},
            {},
            {},
        ]
        with (
            patch.dict(os.environ, {ENV_SCHEDULER_ENDPOINT: "/tmp/scheduler"}, clear=True),
            patch.object(experiment_scheduler_client, "_rpc", side_effect=responses) as rpc,
        ):
            self.assertEqual(
                experiment_scheduler_client.register_idle_supply("gen0_peer0", 0)["lease_id"],
                "lease",
            )
            experiment_scheduler_client.unregister_idle_supply("gen0_peer0", 0)
            self.assertEqual(
                experiment_scheduler_client.get_supply_lease("gen0_peer0", 0, "lease")["lease_id"],
                "lease",
            )
            experiment_scheduler_client.release_supply_lease("lease", "gen0_peer0", declined=True)
            self.assertEqual(
                experiment_scheduler_client.generation_advice("gen0_peer0", 0)["first_wave"],
                "explore",
            )
            self.assertTrue(experiment_scheduler_client.begin_assessment(0, "ready"))
            experiment_scheduler_client.freeze_generation(0, "close")
        self.assertEqual(rpc.call_count, 7)

    def test_environment_classification_and_recovery(self) -> None:
        self.assertTrue(is_sensitive_environment_name("service_api_key"))
        self.assertTrue(is_sensitive_environment_name("OPENAI_API_KEY"))
        self.assertTrue(is_sensitive_environment_name("PASSWORD"))
        self.assertTrue(is_sensitive_environment_name("PGPASSWORD"))
        self.assertTrue(is_sensitive_environment_name("DATABASE_URL"))
        self.assertTrue(is_sensitive_environment_name("REDIS_URL"))
        self.assertTrue(is_sensitive_environment_name("KAGGLE_KEY"))
        self.assertTrue(is_sensitive_environment_name("AZURE_STORAGE_KEY"))
        self.assertFalse(is_sensitive_environment_name("MONKEY"))
        self.assertFalse(is_sensitive_environment_name("PUBLIC_KEY"))
        self.assertFalse(is_sensitive_environment_name("PUBLIC_URL"))
        self.assertTrue(
            is_sensitive_environment_entry(
                "PIP_INDEX_URL", "https://user:password@example.invalid/simple"
            )
        )
        self.assertTrue(
            is_sensitive_environment_entry(
                "CUSTOM_ENDPOINT", "https://example.invalid/path?access_token=secret"
            )
        )
        self.assertTrue(
            is_sensitive_environment_entry(
                "AZURE_STORAGE_CONNECTION_STRING",
                "DefaultEndpointsProtocol=https;AccountName=a;AccountKey=secret",
            )
        )
        self.assertFalse(
            is_sensitive_environment_entry("PUBLIC_URL", "https://example.invalid/documentation")
        )
        self.assertTrue(is_sensitive_environment_entry("HEADER", "Bearer credential"))
        self.assertFalse(is_sensitive_environment_entry("BROKEN_URL", "http://[::1"))
        self.assertTrue(
            is_sensitive_environment_entry(
                "CUSTOM_ENDPOINT", "https://user:password@example.invalid/path"
            )
        )
        secret = "current-secret"
        digest = hashlib.sha256(secret.encode()).hexdigest()
        event = {
            "environment_values": {"PLAIN": "value"},
            "environment_sensitive_hashes": {"SERVICE_TOKEN": digest},
        }
        with patch.dict(os.environ, {"SERVICE_TOKEN": secret}, clear=True):
            self.assertEqual(
                recover_environment(event), {"PLAIN": "value", "SERVICE_TOKEN": secret}
            )
            self.assertTrue(sensitive_environment_matches(event))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(recover_environment(event), {"PLAIN": "value"})
            self.assertFalse(sensitive_environment_matches(event))

    def test_legacy_environment_delta_and_semantic_key(self) -> None:
        with patch.dict(os.environ, {"KEEP": "yes", "DROP": "yes"}, clear=True):
            recovered = recover_environment(
                {"environment_unset": ["DROP"], "environment_delta": {"ADD": 4}}
            )
        self.assertEqual(recovered, {"KEEP": "yes", "ADD": "4"})
        self.assertEqual(
            semantic_experiment_key("run", 1, "experiment"),
            semantic_experiment_key("run", 1, "experiment"),
        )
        self.assertNotEqual(
            semantic_experiment_key("run", 1, "experiment"),
            semantic_experiment_key("run", 2, "experiment"),
        )

    def test_rpc_errors_are_structured(self) -> None:
        with self.assertRaises(SchedulerUnavailable):
            _rpc("", {"action": "ping"})

        client = MagicMock()
        client.__enter__.return_value = client
        client.recv.return_value = b'{"ok":false,"error":"closed"}\n'
        with (
            patch("socket.socket", return_value=client),
            self.assertRaisesRegex(ExperimentRejected, "closed"),
        ):
            _rpc("/tmp/scheduler.sock", {"action": "ping"})

        client.connect.side_effect = OSError("missing")
        with (
            patch("socket.socket", return_value=client),
            self.assertRaises(SchedulerUnavailable),
        ):
            _rpc("/tmp/missing.sock", {"action": "ping"})

        empty = MagicMock()
        empty.__enter__.return_value = empty
        empty.recv.return_value = b""
        with patch("socket.socket", return_value=empty), self.assertRaises(json.JSONDecodeError):
            _rpc("/tmp/empty.sock", {"action": "ping"})

    def test_freeze_helpers_and_generation_fallback(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client._rpc"
            ) as rpc,
        ):
            freeze_generation(2, "close")
            rpc.assert_not_called()
        with (
            patch.dict(os.environ, {ENV_SCHEDULER_ENDPOINT: "/tmp/s.sock"}, clear=True),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client._rpc"
            ) as rpc,
        ):
            freeze_generation(2, "close")
            self.assertEqual(rpc.call_args.args[1]["generation_id"], 2)

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            self.assertFalse(freeze_all_for_run(run_dir))
            state = run_dir / "resource_scheduler"
            state.mkdir()
            uid = getattr(os, "getuid", lambda: 0)()
            digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:16]
            endpoint = str(Path(f"/tmp/praxist-scheduler-{uid}") / f"{digest}.sock")
            (state / "endpoint.json").write_text(
                json.dumps({"endpoint": endpoint}), encoding="utf-8"
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client._rpc"
            ):
                self.assertTrue(freeze_all_for_run(run_dir))

        self.assertEqual(_generation_from_peer("gen7_peer3"), 7)
        with patch.dict(os.environ, {"GENERATION_ID": "9"}, clear=True):
            self.assertEqual(_generation_from_peer("peer"), 9)
        with patch.dict(os.environ, {"GENERATION_ID": "bad"}, clear=True):
            self.assertEqual(_generation_from_peer("peer"), 0)

    def test_run_endpoint_metadata_is_identity_bound_and_well_formed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            state_dir = run_dir / "resource_scheduler"
            state_dir.mkdir(parents=True)
            endpoint_path = state_dir / "endpoint.json"

            endpoint_path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(SchedulerUnavailable, "unreadable"):
                experiment_scheduler_client.scheduler_endpoint_for_run(run_dir)

            endpoint_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(SchedulerUnavailable, "invalid"):
                experiment_scheduler_client.scheduler_endpoint_for_run(run_dir)

            endpoint_path.unlink()
            with (
                patch.dict(
                    os.environ,
                    {ENV_SCHEDULER_ENDPOINT: "/tmp/other-run.sock"},
                    clear=True,
                ),
                self.assertRaisesRegex(SchedulerUnavailable, "requested run"),
            ):
                experiment_scheduler_client.scheduler_endpoint_for_run(run_dir)

    def test_scheduler_authority_queries_normalize_only_verified_groups(self) -> None:
        run_dir = Path("/tmp/run")
        with patch.object(
            experiment_scheduler_client,
            "scheduler_endpoint_for_run",
            return_value="",
        ):
            self.assertFalse(
                experiment_scheduler_client.scheduler_attempt_is_active(run_dir, "attempt", 42)
            )
            self.assertEqual(
                experiment_scheduler_client.scheduler_active_process_groups(run_dir),
                {},
            )

        with (
            patch.object(
                experiment_scheduler_client,
                "scheduler_endpoint_for_run",
                return_value="/tmp/scheduler.sock",
            ),
            patch.object(
                experiment_scheduler_client,
                "_rpc",
                return_value={"active": True},
            ) as rpc,
        ):
            self.assertTrue(
                experiment_scheduler_client.scheduler_attempt_is_active(run_dir, "attempt", 42)
            )
            self.assertEqual(rpc.call_args.args[1]["action"], "validate_attempt")

        with (
            patch.object(
                experiment_scheduler_client,
                "scheduler_endpoint_for_run",
                return_value="/tmp/scheduler.sock",
            ),
            patch.object(
                experiment_scheduler_client,
                "_rpc",
                side_effect=SchedulerUnavailable("gone"),
            ),
        ):
            self.assertFalse(
                experiment_scheduler_client.scheduler_attempt_is_active(run_dir, "attempt", 42)
            )
            self.assertEqual(
                experiment_scheduler_client.scheduler_active_process_groups(run_dir),
                {},
            )

        response = {
            "groups": [
                "not-a-row",
                {"pgid": "bad", "pid": 2, "pid_start_time": 1},
                {"pgid": 41, "pid": 42, "pid_start_time": 123},
                {"pgid": 43, "pid": 44, "pid_start_time": "ps:started"},
                {"pgid": 45, "pid": 46, "pid_start_time": "unverified"},
                {"pgid": 1, "pid": 2, "pid_start_time": 456},
            ]
        }
        with (
            patch.object(
                experiment_scheduler_client,
                "scheduler_endpoint_for_run",
                return_value="/tmp/scheduler.sock",
            ),
            patch.object(experiment_scheduler_client, "_rpc", return_value=response),
        ):
            groups = experiment_scheduler_client.scheduler_active_process_groups(run_dir)
        self.assertEqual(groups, {41: (42, "proc:123"), 43: (44, "ps:started")})

    def test_submit_timeout_cancels_queued_work(self) -> None:
        responses = [
            {"job": {"job_id": "job"}},
            {"timeout": True, "job": {"state": "queued"}},
            {"cancelled": True},
        ]
        with (
            patch.dict(os.environ, {ENV_SCHEDULER_ENDPOINT: "/tmp/s.sock"}, clear=True),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client._rpc",
                side_effect=responses,
            ),
            self.assertRaises(TimeoutError),
        ):
            submit_and_wait(
                ["task"], peer_id="gen0_peer0", experiment_id="candidate", wait_timeout_seconds=1
            )

    def test_submit_timeout_waits_for_running_work_and_surfaces_rejection(self) -> None:
        responses = [
            {"job": {"job_id": "job"}},
            {"timeout": True, "job": {"state": "running"}},
            {"job": {"state": "completed", "exit_code": None}},
        ]
        with (
            patch.dict(os.environ, {ENV_SCHEDULER_ENDPOINT: "/tmp/s.sock"}, clear=True),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client._rpc",
                side_effect=responses,
            ),
        ):
            self.assertEqual(
                submit_and_wait(
                    ["task"],
                    peer_id="gen0_peer0",
                    experiment_id="candidate",
                    wait_timeout_seconds=1,
                ),
                2,
            )
        responses = [
            {"job": {"job_id": "job"}},
            {"job": {"state": "rejected", "error": "closed"}},
        ]
        with (
            patch.dict(os.environ, {ENV_SCHEDULER_ENDPOINT: "/tmp/s.sock"}, clear=True),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client._rpc",
                side_effect=responses,
            ),
            self.assertRaisesRegex(ExperimentRejected, "closed"),
        ):
            submit_and_wait(["task"], peer_id="gen0_peer0", experiment_id="candidate")

    def test_submit_revalidates_supply_lease_and_strips_stale_context(self) -> None:
        requests: list[dict[str, Any]] = []

        def respond(_endpoint: str, request: dict[str, Any], **_kwargs):
            requests.append(request)
            if request["action"] == "get_supply_lease":
                return {"supply": {}}
            if request["action"] == "submit":
                return {"job": {"job_id": "job"}}
            return {"job": {"state": "completed", "exit_code": 0}}

        environment = {
            ENV_SCHEDULER_ENDPOINT: "/tmp/s.sock",
            "PRAXIST_RESOURCE_SUPPLY_LEASE_ID": "stale-lease",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client._rpc",
                side_effect=respond,
            ),
        ):
            self.assertEqual(
                submit_and_wait(["task"], peer_id="gen2_peer0", experiment_id="candidate"),
                0,
            )

        submit = next(request for request in requests if request["action"] == "submit")
        self.assertEqual(submit["supply_lease_id"], "")
        self.assertNotIn("PRAXIST_RESOURCE_SUPPLY_LEASE_ID", submit["environment"])

    def test_submit_preserves_canonical_supply_lease(self) -> None:
        requests: list[dict[str, Any]] = []

        def respond(_endpoint: str, request: dict[str, Any], **_kwargs):
            requests.append(request)
            if request["action"] == "get_supply_lease":
                return {"supply": {"lease_id": "active-lease"}}
            if request["action"] == "submit":
                return {"job": {"job_id": "job"}}
            return {"job": {"state": "completed", "exit_code": 0}}

        environment = {
            ENV_SCHEDULER_ENDPOINT: "/tmp/s.sock",
            "PRAXIST_RESOURCE_SUPPLY_LEASE_ID": "active-lease",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client._rpc",
                side_effect=respond,
            ),
        ):
            self.assertEqual(
                submit_and_wait(["task"], peer_id="gen2_peer0", experiment_id="candidate"),
                0,
            )

        submit = next(request for request in requests if request["action"] == "submit")
        self.assertEqual(submit["supply_lease_id"], "active-lease")
        self.assertEqual(
            submit["environment"]["PRAXIST_RESOURCE_SUPPLY_LEASE_ID"],
            "active-lease",
        )

    def test_submit_resolves_relative_evaluator_without_overriding_runtime_cwd(self) -> None:
        requests: list[dict[str, Any]] = []

        def respond(_endpoint: str, request: dict[str, Any], **_kwargs):
            requests.append(request)
            if request["action"] == "submit":
                return {"job": {"job_id": "job"}}
            return {"job": {"state": "completed", "exit_code": 0}}

        with tempfile.TemporaryDirectory() as td:
            task_root = Path(td) / "task"
            caller_cwd = Path(td) / "run" / "peer"
            task_root.mkdir()
            caller_cwd.mkdir(parents=True)
            evaluator = task_root / "evaluations" / "public" / "run.py"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(caller_cwd)
                with (
                    patch.dict(
                        os.environ,
                        {
                            ENV_SCHEDULER_ENDPOINT: "/tmp/s.sock",
                            "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        },
                        clear=True,
                    ),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler_client._rpc",
                        side_effect=respond,
                    ),
                ):
                    self.assertEqual(
                        submit_and_wait(
                            ["python", "evaluations/public/run.py"],
                            peer_id="gen0_peer0",
                            experiment_id="task-root-evaluator",
                        ),
                        0,
                    )
            finally:
                os.chdir(previous)

        submit = next(request for request in requests if request["action"] == "submit")
        self.assertEqual(submit["cwd"], str(caller_cwd.resolve()))
        self.assertEqual(submit["command"][1], str(evaluator.resolve()))

    def test_submit_explicit_cwd_still_overrides_task_root(self) -> None:
        requests: list[dict[str, Any]] = []

        def respond(_endpoint: str, request: dict[str, Any], **_kwargs):
            requests.append(request)
            if request["action"] == "submit":
                return {"job": {"job_id": "job"}}
            return {"job": {"state": "completed", "exit_code": 0}}

        with tempfile.TemporaryDirectory() as td:
            task_root = Path(td) / "task"
            explicit = Path(td) / "scratch"
            task_root.mkdir()
            explicit.mkdir()
            with (
                patch.dict(
                    os.environ,
                    {
                        ENV_SCHEDULER_ENDPOINT: "/tmp/s.sock",
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    },
                    clear=True,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client._rpc",
                    side_effect=respond,
                ),
            ):
                self.assertEqual(
                    submit_and_wait(
                        ["task"],
                        peer_id="gen0_peer0",
                        experiment_id="explicit-cwd",
                        cwd=explicit,
                    ),
                    0,
                )

        submit = next(request for request in requests if request["action"] == "submit")
        self.assertEqual(submit["cwd"], str(explicit.resolve()))

    def test_supply_revalidation_failure_does_not_block_submission(self) -> None:
        responses: list[object] = [
            SchedulerUnavailable("temporary read failure"),
            {"job": {"job_id": "job"}},
            {"job": {"state": "completed", "exit_code": 0}},
        ]
        environment = {
            ENV_SCHEDULER_ENDPOINT: "/tmp/s.sock",
            "PRAXIST_RESOURCE_SUPPLY_LEASE_ID": "lease",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client._rpc",
                side_effect=responses,
            ) as rpc,
        ):
            self.assertEqual(
                submit_and_wait(["task"], peer_id="gen0_peer0", experiment_id="candidate"),
                0,
            )
        self.assertEqual(rpc.call_args_list[1].args[1]["supply_lease_id"], "lease")


class HostResourceBoundaryTest(unittest.TestCase):
    def test_settings_reject_ambiguous_resource_identity_and_bound_policy_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid accelerator"):
            SchedulerSettings.from_dict(
                {
                    "mode": "central",
                    "profiles": {"gpu": {"accelerator": "custom"}},
                }
            )
        with self.assertRaisesRegex(ValueError, "is not declared"):
            SchedulerSettings.from_dict(
                {
                    "mode": "central",
                    "profiles": {"cpu": {"accelerator": "cpu"}},
                    "default_profile": "missing",
                }
            )

        settings = SchedulerSettings.from_dict(
            {
                "mode": "CENTRAL",
                "min_concurrent_experiments": "4",
                "max_concurrent_experiments": "2",
                "initial_concurrent_experiments": "2",
                "deadline_admission": "off",
                "mature_assessment_min_completion_probability": 2,
                "profiles": {
                    "cpu": {"accelerator": "cpu", "pressure_domains": ["cpu"]},
                },
            }
        )
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.initial_concurrency, 2)
        self.assertEqual(settings.min_concurrency, 2)
        self.assertFalse(settings.deadline_admission)
        self.assertEqual(settings.default_profile, "cpu")
        self.assertEqual(settings.profile("unknown").accelerator, "cpu")
        with self.assertRaisesRegex(
            ValueError, "initial_concurrent_experiments must be an integer"
        ):
            SchedulerSettings.from_dict(
                {"mode": "central", "initial_concurrent_experiments": "bad"}
            )
        self.assertEqual(settings.profile("cpu").pressure_domains, ("cpu",))
        self.assertEqual(settings.mature_assessment_min_completion_probability, 1.0)

        self.assertEqual(_as_int(object(), 7), 7)
        self.assertEqual(_as_float(float("nan"), 8.0), 8.0)
        self.assertTrue(_as_bool("yes", False))
        self.assertFalse(_as_bool("no", True))
        self.assertTrue(_as_bool(1, True))
        self.assertIsNone(_optional_positive_float(None))
        self.assertEqual(_optional_positive_float(200, maximum=100), 100)

    def test_linux_observer_parses_pressure_and_gpu_data(self) -> None:
        observer = HostObserver()

        def read_text(path: Path, **_kwargs) -> str:
            if str(path) == "/proc/stat":
                return "cpu  100 0 100 800 0\n"
            if str(path) == "/proc/meminfo":
                return "MemTotal: 1000 kB\nMemAvailable: 250 kB\n"
            if str(path) == "/proc/pressure/io":
                return "some avg10=12.50 avg60=1.0 total=2\n"
            raise OSError

        result = SimpleNamespace(
            returncode=0,
            stdout="0, GPU-a, 10000, 2000, 30\nbad\n1, GPU-b, x, 0, 0\n",
        )
        with (
            patch.object(Path, "read_text", autospec=True, side_effect=read_text),
            patch.object(subprocess, "run", return_value=result),
            patch.object(os, "cpu_count", return_value=8),
            patch.object(os, "getloadavg", return_value=(4.0, 0.0, 0.0)),
        ):
            first = observer.snapshot()
            second = observer.snapshot()
        self.assertEqual(first.memory_utilization_pct, 75.0)
        self.assertEqual(first.io_pressure_pct, 12.5)
        self.assertEqual(first.gpus[0].uuid, "GPU-a")
        self.assertEqual(first.accelerator_probe_state, "available")
        self.assertEqual(second.cpu_utilization_pct, 50.0)

        with patch.object(subprocess, "run", side_effect=OSError):
            self.assertEqual(HostObserver._gpus(), [])

    def test_linux_observer_failure_paths_degrade_to_zero_or_no_gpu(self) -> None:
        observer = HostObserver()
        with (
            patch.object(Path, "read_text", side_effect=OSError("proc unavailable")),
            patch.object(os, "getloadavg", side_effect=OSError("load unavailable")),
        ):
            self.assertEqual(observer._cpu_utilization(), 0.0)
            self.assertEqual(observer._memory_utilization(), 0.0)
            self.assertEqual(observer._io_pressure(), 0.0)
        failed = SimpleNamespace(returncode=1, stdout="")
        with patch.object(subprocess, "run", return_value=failed):
            self.assertEqual(observer._gpus(), [])
            _gpus, state, reason = observer._query_gpus()
        self.assertEqual(state, "unknown")
        self.assertTrue(reason)
        self.assertEqual(_as_float("bad", 3.0), 3.0)
        self.assertTrue(_as_bool(True, False))

    def test_linux_observer_reports_detected_unsupported_accelerator_backend(self) -> None:
        observer = HostObserver()
        failed = SimpleNamespace(returncode=1, stdout="", stderr="driver unavailable")
        with (
            patch.object(subprocess, "run", return_value=failed),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "resource_scheduler.shutil.which",
                return_value="/usr/bin/rocm-smi",
            ),
        ):
            gpus, state, reason = observer._query_gpus()

        self.assertEqual(gpus, [])
        self.assertEqual(state, "unsupported")
        self.assertIn("ROCm", reason)

    def test_registry_prunes_bad_rows_and_process_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            registry.path.write_text("not-json", encoding="utf-8")
            self.assertEqual(registry._read_and_prune(), [])
            registry.path.write_text("{}", encoding="utf-8")
            self.assertEqual(registry._read_and_prune(), [])
            with registry.locked() as rows:
                rows.append({"record_type": "controller", "run_id": "run", "pid": 0})
            self.assertTrue(registry.path.exists())

        self.assertFalse(_process_group_alive({"pid": 0, "pgid": 0}))
        with patch.object(os, "kill", side_effect=PermissionError):
            self.assertTrue(_process_group_alive({"pid": 42, "pgid": 0}))
        self.assertIsNone(_pid_start_time(-1))

    def test_allocator_adapts_concurrency_and_rejects_io_pressure(self) -> None:
        low = HostSnapshot(16, 10, 20, 0)
        high = HostSnapshot(16, 99, 20, 0)
        registry = _MemoryRegistry()
        allocator = ResourceAllocator(
            _settings(), observer=_StaticObserver(low, low, high), registry=registry
        )
        allocator.set_owner("run")
        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend.resource_scheduler."
            "time.monotonic",
            side_effect=[10.0, 20.0],
        ):
            allocator.refresh(queued=True)
            self.assertEqual(allocator.concurrency_limit, 3)
            allocator.refresh(queued=True)
            self.assertEqual(allocator.concurrency_limit, 2)
        allocator.close()
        self.assertEqual(registry.removed, ["run"])

        allocator.observer = _StaticObserver(HostSnapshot(16, 10, 10, 99))
        io_profile = ResourceProfile("io", pressure_domains=("io",))
        self.assertIsNone(
            allocator.reserve(
                allocation_id="io",
                run_id="run",
                pid=0,
                pgid=0,
                profile=io_profile,
            )
        )
        self.assertFalse(allocator.bind_process("missing", pid=2, pgid=2))

    def test_gpu_recovery_rejects_missing_or_busy_exact_device(self) -> None:
        gpu = GPUDevice(0, "GPU-a", 10_000, 0, 0)
        snapshot = HostSnapshot(16, 10, 10, 0, (gpu,))
        registry = _MemoryRegistry()
        allocator = ResourceAllocator(
            _settings(), observer=_StaticObserver(snapshot), registry=registry
        )
        profile = _settings().profile("gpu")
        self.assertIsNone(
            allocator.recover_allocation(
                allocation_id="missing",
                run_id="run",
                pid=2,
                pgid=2,
                profile=profile,
                gpu_uuids=("GPU-z",),
            )
        )
        registry.rows.append(
            {
                "record_type": "allocation",
                "allocation_id": "other",
                "gpu_uuids": ["GPU-a"],
                "gpu_memory_mb": 9500,
                "gpu_utilization_pct": 95,
                "started_at": 0,
            }
        )
        self.assertIsNone(
            allocator.recover_allocation(
                allocation_id="new",
                run_id="run",
                pid=2,
                pgid=2,
                profile=profile,
                gpu_uuids=("GPU-a",),
            )
        )

    def test_allocator_binds_releases_and_recovers_real_ledger_rows(self) -> None:
        snapshot = HostSnapshot(16, 10, 10, 0)
        registry = _MemoryRegistry()
        allocator = ResourceAllocator(
            _settings(), observer=_StaticObserver(snapshot), registry=registry
        )
        profile = _settings().profile("cpu")
        allocation = allocator.reserve(
            allocation_id="first", run_id="run", pid=2, pgid=2, profile=profile
        )
        self.assertIsNotNone(allocation)
        self.assertTrue(allocator.bind_process("first", pid=3, pgid=3))
        self.assertFalse(registry.rows[0]["pending"])
        allocator.release("first")
        self.assertEqual(registry.rows, [])

        recovered = allocator.recover_allocation(
            allocation_id="recovered",
            run_id="run",
            pid=4,
            pgid=4,
            profile=profile,
            gpu_uuids=(),
        )
        self.assertIsNotNone(recovered)
        self.assertEqual(registry.rows[0]["allocation_id"], "recovered")

        registry.rows.append(
            {
                "record_type": "controller",
                "run_id": "other",
                "concurrency_limit": 1,
            }
        )
        independent = allocator.recover_allocation(
            allocation_id="independent",
            run_id="run",
            pid=5,
            pgid=5,
            profile=profile,
            gpu_uuids=(),
        )
        self.assertIsNotNone(independent)


class SchedulerControlBoundaryTest(unittest.TestCase):
    def test_disabled_scheduler_is_a_legacy_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(enabled=False)
            )
            service.start()
            self.assertFalse(service.endpoint.exists())

    def test_enabled_scheduler_publishes_and_closes_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _LifecycleAllocator()
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(), allocator=allocator
            )
            service.start()
            try:
                self.assertEqual(len(allocator.owners), 1)
                self.assertRegex(allocator.owners[0], r"^run:[0-9a-f]{64}$")
                self.assertTrue(service.endpoint.exists())
                self.assertEqual(service.handle_request({"action": "ping"})["run_id"], "run")
            finally:
                service.stop()
            self.assertTrue(allocator.closed)
            self.assertFalse(service.endpoint.exists())

    def test_control_requests_and_validation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            self.assertEqual(service.handle_request({"action": "ping"})["run_id"], "run")
            self.assertEqual(service.handle_request({"action": "status"})["status"]["queued"], 0)
            self.assertFalse(service.wait("missing")["ok"])
            self.assertFalse(service.cancel_queued("missing")["cancelled"])
            service.open_generation(0, deadline=9999999999, cohort_size=2)
            service.configure_generation_maturity(
                0,
                cohort_size=2,
                mature_target=1,
                count_callback=lambda: 0,
            )
            self.assertEqual(
                service.handle_request(
                    {"action": "register_idle_supply", "peer_id": "gen0_peer0", "generation_id": 0}
                )["supply"],
                {},
            )
            self.assertEqual(
                service.handle_request(
                    {
                        "action": "get_supply_lease",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "lease_id": "missing",
                    }
                )["supply"],
                {},
            )
            self.assertEqual(
                service.handle_request(
                    {"action": "generation_advice", "peer_id": "gen0_peer0", "generation_id": 0}
                )["advice"]["first_wave"],
                "direct_mature",
            )
            service.handle_request(
                {"action": "release_supply_lease", "lease_id": "missing", "peer_id": "gen0_peer0"}
            )
            service.handle_request(
                {"action": "unregister_idle_supply", "peer_id": "gen0_peer0", "generation_id": 0}
            )
            queued = service.handle_request(
                {
                    "action": "submit",
                    "command": ["task"],
                    "experiment_id": "queued-control",
                    "peer_id": "gen0_peer1",
                    "generation_id": 0,
                }
            )["job"]
            self.assertTrue(
                service.handle_request({"action": "cancel_queued", "job_id": queued["job_id"]})[
                    "cancelled"
                ]
            )
            self.assertEqual(
                service.handle_request(
                    {"action": "wait", "job_id": queued["job_id"], "timeout_seconds": 0}
                )["job"]["state"],
                "rejected",
            )
            service.handle_request(
                {"action": "begin_assessment", "generation_id": 0, "reason": "ready"}
            )
            with self.assertRaises(ValueError):
                service.handle_request({"action": "unknown"})
            self.assertTrue(
                service.handle_request({"action": "freeze", "generation_id": 3, "reason": "close"})[
                    "ok"
                ]
            )
            self.assertTrue(service.handle_request({"action": "freeze_all"})["ok"])
            for command in (None, [], [1]):
                with self.assertRaises(ExperimentRejected):
                    service.submit({"command": command, "experiment_id": "candidate"})
            with self.assertRaises(ExperimentRejected):
                service.submit({"command": ["task"], "experiment_id": ""})

    def test_freeze_rejects_only_selected_queued_generation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            first = service.submit(
                {
                    "command": ["task"],
                    "experiment_id": "first",
                    "peer_id": "gen1_peer0",
                    "generation_id": 1,
                }
            )
            second = service.submit(
                {
                    "command": ["task"],
                    "experiment_id": "second",
                    "peer_id": "gen2_peer0",
                    "generation_id": 2,
                    "work_class": "unsupported",
                }
            )
            service.freeze_generation(1, "closing")
            self.assertEqual(first.state, "rejected")
            self.assertEqual(second.state, "queued")
            service.open_generation(1, deadline=9999999999)
            service.freeze_all("stop")
            self.assertEqual(second.state, "rejected")
            with self.assertRaises(ExperimentRejected):
                service.submit({"command": ["task"], "experiment_id": "third"})

            resumed = ExperimentSchedulerService(
                run_dir=service.run_dir,
                settings=_settings(),
            )
            resumed._recover_terminal_events()
            self.assertTrue(resumed.status()["admission_closed"])
            resumed.open_generation(2, deadline=9999999999)
            reopened = resumed.submit(
                {
                    "command": ["task"],
                    "experiment_id": "after-resume",
                    "peer_id": "gen2_peer1",
                    "generation_id": 2,
                }
            )
            self.assertEqual(reopened.state, "queued")

            assessing = ExperimentSchedulerService(
                run_dir=Path(td) / "assessing-run",
                settings=_settings(),
            )
            assessing.open_generation(3, deadline=9999999999)
            assessing.begin_assessment(3)
            assessing.freeze_all("stop")
            resumed_assessment = ExperimentSchedulerService(
                run_dir=assessing.run_dir,
                settings=_settings(),
            )
            resumed_assessment._recover_terminal_events()
            resumed_assessment.open_generation(3, deadline=9999999999)
            self.assertIn(3, resumed_assessment._assessment_generations)
            with self.assertRaisesRegex(ExperimentRejected, "only mature work"):
                resumed_assessment.submit(
                    {
                        "command": ["task"],
                        "experiment_id": "ordinary-after-resume",
                        "peer_id": "gen3_peer0",
                        "generation_id": 3,
                        "work_class": "ordinary",
                    }
                )

    def test_snapshot_and_event_io_failures_are_best_effort_unless_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler.atomic_write_json",
                side_effect=OSError("read-only"),
            ):
                service._write_snapshot(force=True)
            with patch.object(os, "open", side_effect=OSError("read-only")):
                service._append_event({"event": "optional"})
                with self.assertRaises(OSError):
                    service._append_event({"event": "required"}, required=True)

    def test_launch_queue_honors_close_signal_and_deadline_before_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            closing = service.submit(
                {
                    "command": ["task"],
                    "experiment_id": "closing",
                    "peer_id": "gen1_peer0",
                    "generation_id": 1,
                }
            )
            gen_dir = service.run_dir / "gen_1"
            gen_dir.mkdir()
            (gen_dir / "CLOSING_SIGNAL").touch()
            service._launch_ready_jobs()
            self.assertEqual(closing.state, "rejected")

            late = service.submit(
                {
                    "command": ["task"],
                    "experiment_id": "late",
                    "peer_id": "gen2_peer0",
                    "generation_id": 2,
                    "eta_seconds": 100,
                }
            )
            service.open_generation(2, deadline=0)
            service._launch_ready_jobs()
            self.assertEqual(late.state, "rejected")

    def test_recovery_rejects_changed_secrets_and_requeues_abandoned_intent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(run_dir=run_dir, settings=_settings())
            events = service.state_dir / "events.jsonl"
            changed = {
                "event": "retry_queued",
                "job_id": "changed",
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "changed-secret",
                "profile": "cpu",
                "command": ["task"],
                "environment_values": {},
                "environment_sensitive_hashes": {"SERVICE_TOKEN": "different"},
            }
            abandoned = {
                "event": "launch_intent",
                "job_id": "abandoned",
                "generation_id": 0,
                "peer_id": "gen0_peer1",
                "experiment_id": "abandoned-intent",
                "profile": "cpu",
                "command": ["task"],
                "environment_values": {},
                "allocation_id": "old-allocation",
            }
            events.write_text(
                "not-json\n" + json.dumps(changed) + "\n" + json.dumps(abandoned) + "\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"SERVICE_TOKEN": "current"}, clear=True):
                service._recover_terminal_events()
            self.assertEqual(service._jobs["changed"].state, "failed")
            self.assertEqual(service._jobs["changed"].exit_code, 75)
            self.assertEqual(service._jobs["abandoned"].state, "queued")
            self.assertIn("abandoned", service._queue)

    def test_abandoned_launch_intent_with_missing_cwd_is_not_requeued(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            task_root.mkdir()
            service = ExperimentSchedulerService(run_dir=root / "run", settings=_settings())
            event = {
                "event": "launch_intent",
                "job_id": "missing-cwd-intent",
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "missing-cwd-intent",
                "profile": "cpu",
                "command": ["task"],
                "cwd": str(task_root / "removed"),
                "environment_values": {"PRAXIST_TASK_PROJECT_PATH": str(task_root)},
                "allocation_id": "abandoned-allocation",
            }
            (service.state_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n",
                encoding="utf-8",
            )

            service._recover_terminal_events()

        recovered = service._jobs["missing-cwd-intent"]
        self.assertEqual(recovered.state, "rejected")
        self.assertEqual(recovered.error, "recovery_cwd_unavailable")
        self.assertNotIn("missing-cwd-intent", service._queue)

    def test_recovery_without_allocation_preserves_already_launched_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            event = {
                "event": "launched",
                "job_id": "live",
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "live-no-allocation",
                "profile": "cpu",
                "command": ["task"],
                "environment_values": {},
                "pid": 123,
                "pgid": 123,
                "attempt": 1,
            }
            (service.state_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n", encoding="utf-8"
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.process_group_alive",
                    return_value=True,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.terminate_process_group"
                ) as terminate,
                patch.object(service, "_event_process_matches", return_value=True),
                patch.object(service, "_unregister_protected", side_effect=OSError("missing")),
            ):
                service._recover_terminal_events()
            terminate.assert_not_called()
            self.assertEqual(service._jobs["live"].state, "running")
            self.assertIn("live", service._active)
            self.assertEqual(
                service._active["live"].allocation.profile,
                "unclassified_recovery",
            )

    def test_recovery_rebinds_live_launch_without_releasing_barrier_early(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            allocator = MagicMock()
            allocator.concurrency_limit = 1
            allocator.snapshot = HostSnapshot(1, 0, 0, 0)
            allocator.recover_allocation.return_value = Allocation(
                allocation_id="allocation",
                run_id="run",
                pid=123,
                pgid=123,
                profile="cpu",
                gpu_uuids=(),
                gpu_memory_mb=0,
                gpu_utilization_pct=0,
                started_at=1,
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir, settings=_settings(), allocator=allocator
            )
            attempt_dir = service.state_dir / "attempts" / "live-a1"
            attempt_dir.mkdir(parents=True)
            event = {
                "event": "launched",
                "job_id": "live",
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "live-launch",
                "profile": "cpu",
                "command": ["task"],
                "environment_values": {},
                "pid": 123,
                "pgid": 123,
                "attempt": 1,
                "allocation_id": "allocation",
                "attempt_dir": str(attempt_dir),
            }
            (service.state_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n", encoding="utf-8"
            )
            semantic = semantic_experiment_key("run", 0, "live-launch")
            with (
                patch.object(service, "_live_manifest_semantics", return_value={semantic}),
                patch.object(service, "_event_process_matches", return_value=True),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.process_group_alive",
                    return_value=True,
                ),
            ):
                service._recover_terminal_events()
            self.assertIn("live", service._active)
            self.assertFalse((attempt_dir / "GO.json").exists())
            events = (service.state_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn('"event": "adopted"', events)

    def test_recovery_does_not_release_live_barrier_after_generation_close(self) -> None:
        boundaries = (
            ("generation_frozen", {"event": "generation_frozen", "generation_id": 0}, ""),
            ("admission_closed", {"event": "admission_closed", "reason": "run_stopped"}, ""),
            ("closing_signal", None, "CLOSING_SIGNAL"),
        )
        for name, boundary_event, signal_name in boundaries:
            with self.subTest(boundary=name), tempfile.TemporaryDirectory() as td:
                allocator = MagicMock()
                allocator.concurrency_limit = 1
                allocator.snapshot = HostSnapshot(1, 0, 0, 0)
                service = ExperimentSchedulerService(
                    run_dir=Path(td) / "run",
                    settings=_settings(),
                    allocator=allocator,
                )
                attempt_dir = service.state_dir / "attempts" / "blocked-a1"
                attempt_dir.mkdir(parents=True)
                launch = {
                    "event": "launch_intent",
                    "job_id": "blocked",
                    "generation_id": 0,
                    "peer_id": "gen0_peer0",
                    "experiment_id": "blocked-launch",
                    "profile": "cpu",
                    "command": ["task"],
                    "environment_values": {},
                    "pid": 123,
                    "pgid": 123,
                    "attempt": 1,
                    "allocation_id": "allocation",
                    "attempt_dir": str(attempt_dir),
                }
                event_lines = [json.dumps(launch)]
                if boundary_event is not None:
                    event_lines.append(json.dumps(boundary_event))
                (service.state_dir / "events.jsonl").write_text(
                    "\n".join(event_lines) + "\n", encoding="utf-8"
                )
                if signal_name:
                    gen_dir = service.run_dir / "gen_0"
                    gen_dir.mkdir()
                    (gen_dir / signal_name).touch()
                semantic = semantic_experiment_key("run", 0, "blocked-launch")
                with (
                    patch.object(service, "_live_manifest_semantics", return_value={semantic}),
                    patch.object(service, "_event_process_matches", return_value=True),
                    patch.object(service, "_unregister_protected"),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler.process_group_alive",
                        return_value=True,
                    ),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler.terminate_process_group"
                    ) as terminate,
                ):
                    service._recover_terminal_events()

                terminate.assert_called_once_with(123)
                allocator.recover_allocation.assert_not_called()
                allocator.release.assert_called_once_with("allocation")
                self.assertFalse((attempt_dir / "GO.json").exists())
                self.assertNotIn("blocked", service._active)
                self.assertEqual(service._jobs["blocked"].state, "rejected")

    def test_recovery_normalizes_malformed_terminal_events(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            invalid_exit = {
                "event": "completed",
                "job_id": "bad-exit",
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "bad-exit",
                "profile": "cpu",
                "exit_code": "not-an-int",
                "state": "invented",
            }
            missing_identity = {
                "event": "completed",
                "job_id": "missing-identity",
                "generation_id": 0,
                "peer_id": "gen0_peer1",
                "experiment_id": "",
                "profile": "cpu",
                "exit_code": 0,
                "state": "completed",
            }
            (service.state_dir / "events.jsonl").write_text(
                "[]\n"
                + json.dumps({"event": "completed"})
                + "\n"
                + json.dumps(invalid_exit)
                + "\n"
                + json.dumps(missing_identity)
                + "\n",
                encoding="utf-8",
            )
            service._recover_terminal_events()
            self.assertEqual(service._jobs["bad-exit"].exit_code, 2)
            self.assertEqual(service._jobs["bad-exit"].state, "failed")
            self.assertNotIn("missing-identity", service._jobs)

    def test_real_peer_guard_allows_barrier_but_still_protects_scheduler_state(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            task_root = root / "task"
            task_root.mkdir()
            allocator = MagicMock()
            allocator.concurrency_limit = 4
            allocator.snapshot = HostSnapshot(8, 0, 0, 0)

            def reserve(**kwargs):
                return Allocation(
                    allocation_id=kwargs["allocation_id"],
                    run_id=kwargs["run_id"],
                    pid=kwargs["pid"],
                    pgid=kwargs["pgid"],
                    profile=kwargs["profile"].name,
                    gpu_uuids=(),
                    gpu_memory_mb=0,
                    gpu_utilization_pct=0,
                    started_at=1,
                )

            allocator.reserve.side_effect = reserve
            allocator.bind_process.return_value = True
            service = ExperimentSchedulerService(
                run_dir=run_dir, settings=_settings(), allocator=allocator
            )
            guarded = prepare_delete_guard_env(
                {
                    **os.environ,
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PEER_ID": "gen0_peer0",
                },
                workspace=run_dir,
                agent_name="gen0_peer0",
            )
            peer_workspace = Path(guarded["PRAXIST_PEER_WORKSPACE"])
            service.start()
            try:
                jobs = []
                for index in range(4):
                    output = peer_workspace / f"completed-{index}.txt"
                    jobs.append(
                        service.submit(
                            {
                                "command": [
                                    sys.executable,
                                    "-c",
                                    "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('ok')",
                                    str(output),
                                ],
                                "peer_id": f"gen0_peer{index}",
                                "generation_id": 0,
                                "experiment_id": f"guarded-{index}",
                                "environment": guarded,
                            }
                        )
                    )
                for index, job in enumerate(jobs):
                    result = service.wait(job.job_id, 20)["job"]
                    self.assertEqual(result["state"], "completed", result)
                    self.assertEqual((peer_workspace / f"completed-{index}.txt").read_text(), "ok")
                    attempt = service.state_dir / "attempts" / f"{job.job_id}-a1"
                    self.assertTrue((attempt / "READY.json").exists())

                protected = service.state_dir / "status.json"
                blocked = service.submit(
                    {
                        "command": [
                            sys.executable,
                            "-c",
                            "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('tampered')",
                            str(protected),
                        ],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "guard-still-active",
                        "environment": guarded,
                    }
                )
                self.assertEqual(service.wait(blocked.job_id, 20)["job"]["state"], "failed")
                self.assertNotEqual(protected.read_text(encoding="utf-8"), "tampered")
            finally:
                service.stop()


if __name__ == "__main__":
    unittest.main()
