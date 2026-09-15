from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock


class ClaudeDeleteGuardContractsTest(unittest.TestCase):
    def test_runtime_temp_prefix_cannot_escape_through_parent_symlink(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
            run_dir = Path(tmp) / "task" / "experiments" / "run"
            protected_dir = run_dir / "shared_findings"
            protected_dir.mkdir(parents=True)
            protected_file = protected_dir / "finding.json"
            protected_file.write_text("keep", encoding="utf-8")
            escape_link = Path("/tmp") / f"pym-praxist-guard-{uuid.uuid4().hex}"
            escape_link.symlink_to(protected_dir, target_is_directory=True)
            guarded = prepare_delete_guard_env(
                {**os.environ, "PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen0_peer0"},
                workspace=run_dir,
                agent_name="gen0_peer0",
            )
            runtime_tmp = Path(guarded["TMPDIR"])
            try:
                probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import os,sys; os.unlink(sys.argv[1])",
                        str(escape_link / protected_file.name),
                    ],
                    env=guarded,
                    cwd=run_dir,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                self.assertNotEqual(probe.returncode, 0)
                self.assertTrue(protected_file.exists())
            finally:
                escape_link.unlink(missing_ok=True)
                runtime_tmp.unlink(missing_ok=True)

    def test_chained_sitecustomize_runs_after_delete_guard_activation(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
            task_root = Path(tmp) / "task"
            run_dir = task_root / "experiments" / "run"
            protected_dir = run_dir / "shared_findings"
            startup_dir = task_root / "startup"
            protected_dir.mkdir(parents=True)
            startup_dir.mkdir(parents=True)
            protected_file = protected_dir / "finding.json"
            protected_file.write_text("keep", encoding="utf-8")
            (startup_dir / "sitecustomize.py").write_text(
                "import os\n"
                "try:\n"
                "    os.unlink(os.environ['PRAXIST_TEST_PROTECTED_FILE'])\n"
                "except PermissionError:\n"
                "    print('chained_sitecustomize_guarded')\n",
                encoding="utf-8",
            )
            guarded = prepare_delete_guard_env(
                {
                    **os.environ,
                    "PYTHONPATH": str(startup_dir),
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PEER_ID": "gen0_peer0",
                    "PRAXIST_TEST_PROTECTED_FILE": str(protected_file),
                },
                workspace=run_dir,
                agent_name="gen0_peer0",
            )
            runtime_tmp = Path(guarded["TMPDIR"])
            try:
                probe = subprocess.run(
                    [sys.executable, "-c", "pass"],
                    env=guarded,
                    cwd=run_dir,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                self.assertEqual(probe.returncode, 0, probe.stderr)
                self.assertIn("chained_sitecustomize_guarded", probe.stdout)
                self.assertTrue(protected_file.exists())
            finally:
                runtime_tmp.unlink(missing_ok=True)

    def test_long_run_path_uses_short_tmpdir_for_multiprocessing(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / ("task_" + "x" * 70) / "experiments" / ("run_" + "y" * 70)
            guarded = prepare_delete_guard_env(
                {**os.environ, "PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen0_peer11"},
                workspace=run_dir,
                agent_name="gen0_peer11",
            )
            runtime_tmp = Path(guarded["TMPDIR"])
            try:
                self.assertLessEqual(len(os.fsencode(runtime_tmp)), 64)
                self.assertTrue(runtime_tmp.is_dir())
                self.assertTrue(runtime_tmp.is_symlink())
                self.assertEqual(
                    runtime_tmp.resolve(),
                    run_dir.resolve() / "peer_workspaces" / "gen0_peer11" / "tmp",
                )
                safe_roots = {
                    Path(raw).resolve()
                    for raw in guarded["PRAXIST_SAFE_DELETE_ROOTS"].split(os.pathsep)
                    if raw
                }
                self.assertIn(runtime_tmp.resolve(), safe_roots)

                probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os,tempfile; "
                            "from multiprocessing.connection import Listener; "
                            "from multiprocessing.reduction import DupFd; "
                            "assert tempfile.gettempdir() == os.environ['TMPDIR']; "
                            "read_fd,write_fd=os.pipe(); shared=DupFd(read_fd); "
                            "copied_fd=shared.detach(); "
                            "os.close(copied_fd); os.close(read_fd); os.close(write_fd); "
                            "listener=Listener(address=None, family='AF_UNIX'); "
                            "address=listener.address; "
                            "assert len(os.fsencode(address)) < 108, address; "
                            "listener.close(); print(address)"
                        ),
                    ],
                    env=guarded,
                    cwd=run_dir,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                self.assertEqual(probe.returncode, 0, probe.stderr + probe.stdout)
                self.assertIn(str(runtime_tmp), probe.stdout)

                optional_probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import importlib.util\n"
                            "if importlib.util.find_spec('wandb') is not None:\n"
                            "    import wandb\n"
                            "if importlib.util.find_spec('torch') is not None:\n"
                            "    from torch.utils.data import DataLoader\n"
                            "    assert list(DataLoader(list(range(4)), batch_size=2, "
                            "num_workers=1))\n"
                        ),
                    ],
                    env=guarded,
                    cwd=run_dir,
                    text=True,
                    capture_output=True,
                    timeout=60,
                )
                self.assertEqual(
                    optional_probe.returncode,
                    0,
                    optional_probe.stderr + optional_probe.stdout,
                )
            finally:
                runtime_tmp.unlink(missing_ok=True)

    def test_stdlib_tempfile_probe_survives_explicit_system_tmp_override(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            guarded = prepare_delete_guard_env(
                {**os.environ, "PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen0_peer0"},
                workspace=run_dir,
                agent_name="gen0_peer0",
            )
            runtime_tmp = Path(guarded["TMPDIR"])
            with tempfile.NamedTemporaryFile(
                mode="w", prefix="praxist-guard-near-miss-", dir="/tmp", delete=False
            ) as handle:
                handle.write("keep")
                protected_tmp_file = Path(handle.name)
            guarded.update({"TMPDIR": "/tmp", "TEMP": "/tmp", "TMP": "/tmp"})
            try:
                probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os,sys,tempfile; "
                            "from multiprocessing.connection import Listener; "
                            "tempfile.tempdir=None; "
                            "assert tempfile.gettempdir() == '/tmp'; "
                            "handle=tempfile.NamedTemporaryFile(); temp_name=handle.name; "
                            "handle.close(); "
                            "assert not os.path.exists(temp_name); "
                            "directory=tempfile.TemporaryDirectory(); dir_name=directory.name; "
                            "link=os.path.join(dir_name, 'external-link'); "
                            "os.symlink('/tmp/nonexistent-praxist-guard-target', link); "
                            "directory.cleanup(); assert not os.path.exists(dir_name); "
                            "finalized=tempfile.TemporaryDirectory(); finalized_name=finalized.name; "
                            "del finalized; import gc; gc.collect(); "
                            "assert not os.path.exists(finalized_name); "
                            "listener=Listener(address=None, family='AF_UNIX'); "
                            "listener_name=listener.address; listener.close(); "
                            "assert not os.path.exists(listener_name); "
                            "target=sys.argv[1]; "
                            "\ntry: os.unlink(target)\n"
                            "except PermissionError: print('direct_delete_blocked')\n"
                            "else: raise SystemExit('direct delete unexpectedly allowed')"
                        ),
                        str(protected_tmp_file),
                    ],
                    env=guarded,
                    cwd=run_dir,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                self.assertEqual(probe.returncode, 0, probe.stderr + probe.stdout)
                self.assertIn("direct_delete_blocked", probe.stdout)
                self.assertTrue(protected_tmp_file.exists())
            finally:
                protected_tmp_file.unlink(missing_ok=True)
                runtime_tmp.unlink(missing_ok=True)

    def test_default_guarded_tmp_cleans_symlinks_and_nested_multiprocessing_tree(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            guarded = prepare_delete_guard_env(
                {**os.environ, "PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen0_peer0"},
                workspace=run_dir,
                agent_name="gen0_peer0",
            )
            runtime_tmp = Path(guarded["TMPDIR"])
            try:
                probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os,pathlib,shutil,tempfile; "
                            "named=tempfile.NamedTemporaryFile(); "
                            "named.write(b'x'); named.close(); "
                            "plain=tempfile.TemporaryFile(); "
                            "plain.write(b'x'); plain.close(); "
                            "spooled=tempfile.SpooledTemporaryFile(max_size=1); "
                            "spooled.write(b'xx'); spooled.close(); "
                            "directory=tempfile.TemporaryDirectory(); dir_name=directory.name; "
                            "link=os.path.join(dir_name, 'external-link'); "
                            "os.symlink('/tmp/nonexistent-praxist-guard-target', link); "
                            "directory.cleanup(); assert not os.path.lexists(dir_name); "
                            "pymp=pathlib.Path(tempfile.mkdtemp(prefix='pymp-')); "
                            "nested=pymp/'nested'; nested.mkdir(); "
                            "(nested/'payload').write_text('x'); "
                            "(nested/'external-link').symlink_to("
                            "'/tmp/nonexistent-praxist-pymp-target'); "
                            "shutil.rmtree(pymp); assert not pymp.exists(); "
                            "owned=pathlib.Path(os.environ['PRAXIST_PEER_WORKSPACE'])/'dirfd'; "
                            "owned.mkdir(); (owned/'payload').write_text('x'); "
                            "fd=os.open(owned, os.O_RDONLY); "
                            "os.remove('payload', dir_fd=fd); os.close(fd); "
                            "assert not (owned/'payload').exists()"
                        ),
                    ],
                    env=guarded,
                    cwd=run_dir,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                self.assertEqual(probe.returncode, 0, probe.stderr + probe.stdout)
            finally:
                runtime_tmp.unlink(missing_ok=True)

    def test_short_tmp_discovery_falls_back_without_host_temp(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import delete_guard

        with tempfile.TemporaryDirectory() as tmp:
            fallback = Path(tmp) / "peer_tmp"
            fallback.mkdir()
            with (
                mock.patch.object(
                    delete_guard.tempfile,
                    "gettempdir",
                    side_effect=FileNotFoundError("no host temp"),
                ),
                mock.patch.object(delete_guard.os, "access", return_value=False),
            ):
                selected = delete_guard._ensure_short_runtime_tmp(
                    run_dir=Path(tmp) / "run",
                    safe_agent="gen0_peer0",
                    fallback=fallback,
                    environment={},
                )
            self.assertEqual(selected, fallback)

            with mock.patch.object(
                delete_guard.Path,
                "resolve",
                side_effect=RuntimeError("symlink loop"),
            ):
                selected_after_loop = delete_guard._ensure_short_runtime_tmp(
                    run_dir=Path(tmp) / "run",
                    safe_agent="gen0_peer0",
                    fallback=fallback,
                    environment={},
                )
            self.assertEqual(selected_after_loop, fallback)

    def test_short_tmp_link_cleanup_only_removes_matching_link(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import delete_guard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / "target"
            target.mkdir()
            matching = root / "matching"
            matching.symlink_to(target, target_is_directory=True)
            delete_guard._cleanup_runtime_tmp_link(matching, target)
            self.assertFalse(matching.exists())
            self.assertFalse(matching.is_symlink())

            other_target = root / "other"
            other_target.mkdir()
            mismatch = root / "mismatch"
            mismatch.symlink_to(other_target, target_is_directory=True)
            delete_guard._cleanup_runtime_tmp_link(mismatch, target)
            self.assertTrue(mismatch.is_symlink())

    def test_task_cli_arguments_are_opaque_to_runtime_guard(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            variant_dir = run_dir / "variants" / "gen0_peer0_valid"
            variant_dir.mkdir(parents=True)
            task_script = variant_dir / "candidate.py"
            task_script.write_text(
                "from pathlib import Path\n"
                "variants = []\n"
                "def save_checkpoint_and_summary(root):\n"
                "    Path(root, 'checkpoint.json').write_text('summary')\n",
                encoding="utf-8",
            )
            guarded = prepare_delete_guard_env(
                {**os.environ, "PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen0_peer0"},
                workspace=run_dir,
                agent_name="gen0_peer0",
            )
            launcher = (
                "import os,sys; "
                "argv=[sys.executable,'-c','pass','--variant-main-script',sys.argv[1]]; "
                "os.execvpe(argv[0],argv,os.environ)"
            )
            allowed = subprocess.run(
                [sys.executable, "-c", launcher, str(task_script)],
                env=guarded,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_sitecustomize_allows_python_runtime_temp_cleanup_only(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            protected = run_dir / "shared_findings" / "keep.json"
            protected.parent.mkdir(parents=True)
            protected.write_text("{}", encoding="utf-8")

            env = prepare_delete_guard_env(
                os.environ.copy(),
                workspace=run_dir,
                agent_name="gen0_peer0",
            )
            env["PRAXIST_RUN_DIR"] = str(run_dir)
            env["PRAXIST_DELETE_GUARD_RUN_DIR"] = str(run_dir)
            env["PRAXIST_TASK_PROJECT_PATH"] = str(root / "task")

            script = f"""
import os
import pathlib

runtime_candidate = pathlib.Path('/dev/shm/pym-praxist-delete-guard-contract')
if runtime_candidate.parent.exists() and os.access(runtime_candidate.parent, os.W_OK):
    runtime_candidate.write_text('temporary runtime resource', encoding='utf-8')
    os.unlink(runtime_candidate)
    assert not runtime_candidate.exists()
else:
    print('dev_shm_unavailable')

protected = pathlib.Path({str(protected)!r})
try:
    os.unlink(protected)
except PermissionError:
    print('protected_blocked')
else:
    raise SystemExit('protected unlink unexpectedly allowed')

near_miss = pathlib.Path('/dev/shm/not-pym-praxist-delete-guard-contract')
if near_miss.parent.exists() and os.access(near_miss.parent, os.W_OK):
    near_miss.write_text('not a runtime resource', encoding='utf-8')
    try:
        os.unlink(near_miss)
    except PermissionError:
        print('near_miss_blocked')
    else:
        raise SystemExit('near-miss /dev/shm unlink unexpectedly allowed')
    finally:
        if near_miss.exists():
            # Clean up via a trusted parent process after the child guard test.
            pass
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            try:
                near_miss = Path("/dev/shm/not-pym-praxist-delete-guard-contract")
                if near_miss.exists():
                    near_miss.unlink()
            except OSError:
                pass
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("protected_blocked", result.stdout)

    def test_sitecustomize_allows_praxist_resource_guards_to_update_run_state(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            task_root = root / "task_project"
            peer_ws = run_dir / "peer_workspaces" / "gen0_peer0"
            task_root.mkdir()
            peer_ws.mkdir(parents=True)
            env = os.environ.copy()
            env.update(
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_DELETE_GUARD_RUN_DIR": str(run_dir),
                    "PRAXIST_WORKSPACE_ROOT": str(task_root),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "GPU_GOVERNOR_DIR": str(run_dir / "process_governor"),
                    "PROTECTED_PIDS_DIR": str(run_dir / "protected_pids"),
                    "PRAXIST_MAX_PARALLEL_RUNS_PER_PEER": "1",
                    "PYTHONPATH": os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")]),
                }
            )
            env.pop("PRAXIST_TRUSTED_PROJECT_EXTRA_ROOTS", None)
            guarded = prepare_delete_guard_env(
                env,
                workspace=run_dir,
                agent_name="gen0_peer0",
            )
            self.assertEqual(guarded["PRAXIST_WORKSPACE_ROOT"], str(task_root))
            self.assertNotEqual(guarded["PRAXIST_WORKSPACE_ROOT"], str(repo_root))
            self.assertIn(str(repo_root), guarded.get("PRAXIST_TRUSTED_PROJECT_EXTRA_ROOTS", ""))

            blocked_run_control_mkdir_script = f"""
from pathlib import Path
Path({str(run_dir / "run_control")!r}).mkdir(parents=True, exist_ok=True)
"""
            blocked_run_control_mkdir = subprocess.run(
                [sys.executable, "-c", blocked_run_control_mkdir_script],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertNotEqual(blocked_run_control_mkdir.returncode, 0)
            self.assertIn(
                "protected",
                blocked_run_control_mkdir.stderr + blocked_run_control_mkdir.stdout,
            )

            allowed_script = f"""
import os
import subprocess
import sys
from types import SimpleNamespace

from praxist.plugins.workflow_stages.research_loop.backend import gpu_governor, protected_pids
from praxist.plugins.workflow_stages.research_loop.backend.run_lifecycle import (
    evaluate_run_stop_gate,
    write_external_stop_signal,
)

run_dir = {str(run_dir)!r}
proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
try:
    assert gpu_governor.acquire_slot(
        0,
        pid=os.getpid(),
        peer="gen0_peer0",
        tag="guarded",
        run_dir=run_dir,
        blocking=False,
        max_per_gpu=1,
    )
    assert gpu_governor.transfer_slot(
        0,
        from_pid=os.getpid(),
        to_pid=proc.pid,
        peer="gen0_peer0",
        tag="guarded",
        run_dir=run_dir,
    )
    assert [entry.pid for entry in gpu_governor.list_slots(0, run_dir=run_dir)] == [proc.pid]
    protected_pids.register_pid(proc.pid, peer_id="gen0_peer0", tag="guarded", run_dir=run_dir)
    assert protected_pids.list_active_jobs(peer_id="gen0_peer0", run_dir=run_dir)
    assert protected_pids.unregister_pid(proc.pid, peer_id="gen0_peer0", run_dir=run_dir)
    assert gpu_governor.release_slot(0, pid=proc.pid, run_dir=run_dir)
    payload = write_external_stop_signal(
        run_dir,
        {{"reason": "task_success", "source": "task_local"}},
        stop_signal_path="run_control/stop.json",
    )
    assert payload["schema_version"] == "praxist.run_stop_signal.v1"
    decision = evaluate_run_stop_gate(
        task_spec=SimpleNamespace(
            run_lifecycle=SimpleNamespace(
                max_wall_clock_hours=None,
                stop_signal_path="run_control/stop.json",
            )
        ),
        run_dir=run_dir,
        run_started_at_seconds=1.0,
        now_seconds=2.0,
        next_generation=1,
        generations_completed=1,
    )
    assert decision.should_stop
    assert decision.exit_condition == "external_stop_signal"
finally:
    proc.terminate()
    proc.wait(timeout=10)
print("resource_guards_ok")
"""
            allowed = subprocess.run(
                [sys.executable, "-c", allowed_script],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr + allowed.stdout)
            self.assertIn("resource_guards_ok", allowed.stdout)

            blocked_script = f"""
from pathlib import Path
Path({str(run_dir / "process_governor" / "manual")!r}).mkdir(parents=True)
"""
            blocked = subprocess.run(
                [sys.executable, "-c", blocked_script],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("protected state", blocked.stderr + blocked.stdout)

            blocked_protected_pids_script = f"""
from pathlib import Path
target = Path({str(run_dir / "protected_pids" / "manual.json")!r})
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("{{}}", encoding="utf-8")
"""
            blocked_protected_pids = subprocess.run(
                [sys.executable, "-c", blocked_protected_pids_script],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertNotEqual(blocked_protected_pids.returncode, 0)
            protected_pid_output = blocked_protected_pids.stderr + blocked_protected_pids.stdout
            self.assertIn("protected", protected_pid_output)
            self.assertIn("outside $PRAXIST_PEER_WORKSPACE", protected_pid_output)

            blocked_run_control_script = f"""
from pathlib import Path
target = Path({str(run_dir / "run_control" / "stop.json")!r})
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("{{}}", encoding="utf-8")
"""
            blocked_run_control = subprocess.run(
                [sys.executable, "-c", blocked_run_control_script],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertNotEqual(blocked_run_control.returncode, 0)
            run_control_output = blocked_run_control.stderr + blocked_run_control.stdout
            self.assertIn("protected", run_control_output)
            self.assertIn("run_control/stop.json", run_control_output)

            for key, value in (
                ("GPU_GOVERNOR_DIR", "/tmp/other-governor"),
                ("BYPASS_GPU_GOVERNOR", "1"),
            ):
                env_mutation = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"import os; os.environ[{key!r}] = {value!r}",
                    ],
                    env=guarded,
                    cwd=run_dir,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                self.assertEqual(
                    env_mutation.returncode,
                    0,
                    env_mutation.stderr + env_mutation.stdout,
                )

            warning_log = Path(guarded["PRAXIST_GUARD_WARNINGS_PATH"])
            self.assertTrue(warning_log.exists())
            warning_text = warning_log.read_text(encoding="utf-8")
            self.assertIn("guard_env_mutation", warning_text)

            warning_delete = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; os.unlink(sys.argv[1])",
                    str(warning_log),
                ],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertNotEqual(warning_delete.returncode, 0)
            self.assertTrue(warning_log.exists())

            warning_overwrite = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; open(sys.argv[1], 'w').write('tampered')",
                    str(warning_log),
                ],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertNotEqual(warning_overwrite.returncode, 0)
            self.assertIn("guard_env_mutation", warning_log.read_text(encoding="utf-8"))

            from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
                validate_bash_command,
            )

            shell_env_mutation = validate_bash_command(
                "export BYPASS_GPU_GOVERNOR=1",
                env=guarded,
                cwd=run_dir,
            )
            self.assertTrue(shell_env_mutation.allowed)
            self.assertEqual(shell_env_mutation.severity, "warning")

            bypass_workload = validate_bash_command(
                "BYPASS_GPU_GOVERNOR=1 python train.py",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(bypass_workload.allowed)
            self.assertEqual(bypass_workload.rule_id, "operator_bypass_workload")

            exported_bypass_workload = validate_bash_command(
                "export BYPASS_GPU_GOVERNOR=1; python train.py",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(exported_bypass_workload.allowed)
            self.assertEqual(exported_bypass_workload.rule_id, "operator_bypass_workload")

    def test_sitecustomize_allows_normal_python_workloads_and_runtime_imports(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            task_root = root / "task_project"
            run_dir.mkdir()
            task_root.mkdir()
            env = os.environ.copy()
            env.update(
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_DELETE_GUARD_RUN_DIR": str(run_dir),
                    "PRAXIST_WORKSPACE_ROOT": str(task_root),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PYTHONPATH": os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")]),
                }
            )
            guarded = prepare_delete_guard_env(
                env,
                workspace=run_dir,
                agent_name="gen0_peer0",
            )

            script = """
import importlib.util
import os
import subprocess
import sys

_Alias = subprocess.Popen[bytes]
proc = subprocess.Popen(
    [sys.executable, "-c", "print('popen-ok')"],
    stdout=subprocess.PIPE,
    text=True,
)
out, _ = proc.communicate(timeout=10)
assert proc.returncode == 0, proc.returncode
assert out.strip() == "popen-ok", out
print(out.strip())

shell = subprocess.run(
    f"{sys.executable} -c \\"print('shell-python-ok')\\"",
    shell=True,
    text=True,
    capture_output=True,
    timeout=10,
)
assert shell.returncode == 0, shell.stderr + shell.stdout
assert "shell-python-ok" in shell.stdout
print(shell.stdout.strip())

system_code = os.system(f"{sys.executable} -c \\"print('os-system-python-ok')\\"")
assert system_code == 0, system_code

if importlib.util.find_spec("claude_agent_sdk") is not None:
    import claude_agent_sdk  # noqa: F401
    print("claude_agent_sdk_ok")
else:
    print("claude_agent_sdk_skipped")

if importlib.util.find_spec("torch") is not None:
    import torch
    print("torch_ok", getattr(torch, "__version__", "unknown"))
else:
    print("torch_skipped")
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("popen-ok", result.stdout)
            self.assertIn("shell-python-ok", result.stdout)
            self.assertIn("os-system-python-ok", result.stdout)
            self.assertRegex(result.stdout, r"claude_agent_sdk_(ok|skipped)")
            self.assertRegex(result.stdout, r"torch_(ok|skipped)")

            outside_target = root / "outside_delete_target"
            outside_target.mkdir()
            blocked_rm = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import os; os.system('/bin/rm -rf {outside_target}')",
                ],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertNotEqual(blocked_rm.returncode, 0)
            self.assertTrue(outside_target.exists())

    def test_generated_guards_use_shared_policy_without_task_specific_paths(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import delete_guard

        generated = delete_guard._bash_env_text() + delete_guard._sitecustomize_text()
        self.assertNotIn("__PRAXIST_", generated)
        self.assertNotIn("task-specific legacy workspace path", generated)
        self.assertIn("BYPASS_GPU_GOVERNOR", generated)
        self.assertIn("process_governor", generated)
        self.assertIn("protected_pids", generated)
        compile(delete_guard._sitecustomize_text(), "sitecustomize.py", "exec")

    def test_allows_diagnostics_and_inferred_task_eval_entrypoints(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
            validate_bash_command,
        )

        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "task"
            run_dir = task_root / "experiments" / "run_2026-06-07"
            peer_ws = run_dir / "peer_workspaces" / "gen0_peer0"
            eval_py = task_root / "evaluations" / "trading_pareto" / "run.py"
            harness_eval = task_root / "assets" / "harness" / "eval" / "tiered_eval.py"
            eval_py.parent.mkdir(parents=True)
            harness_eval.parent.mkdir(parents=True)
            peer_ws.mkdir(parents=True)
            eval_py.write_text("print('eval help')\n", encoding="utf-8")
            harness_eval.write_text("print('tiered eval help')\n", encoding="utf-8")
            (task_root / "scratch_module.py").write_text("print('safe module')\n", encoding="utf-8")
            env = {
                "PRAXIST_SAFE_DELETE_ROOTS": str(peer_ws),
                "PRAXIST_PEER_WORKSPACE": str(peer_ws),
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_DELETE_GUARD_RUN_DIR": str(run_dir),
                "PEER_ID": "gen0_peer0",
                "PATH": "/bin:/usr/bin:/usr/sbin",
            }

            diagnostic = validate_bash_command(
                "ps -p $$ >/dev/null; pgrep -f definitely-not-required || true; "
                "nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null || true; "
                f"printf '%s\\n' {run_dir / 'shared_findings'} | xargs echo >/dev/null",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(diagnostic.allowed, diagnostic.message)

            direct_eval = validate_bash_command(
                f"python {eval_py} --help",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(direct_eval.allowed, direct_eval.message)

            module_eval = validate_bash_command(
                "python -m evaluations.trading_pareto.run --help",
                env=env,
                cwd=task_root,
            )
            self.assertTrue(module_eval.allowed, module_eval.message)

            module_harness_eval = validate_bash_command(
                "python -m assets.harness.eval.tiered_eval --help",
                env=env,
                cwd=task_root,
            )
            self.assertTrue(module_harness_eval.allowed, module_harness_eval.message)

            allowed_project_module = validate_bash_command(
                "python -m scratch_module",
                env=env,
                cwd=task_root,
            )
            self.assertTrue(allowed_project_module.allowed, allowed_project_module.message)
            unresolved_safe_module = validate_bash_command(
                "python -m safe_package.safe_module --help",
                env=env,
                cwd=task_root,
            )
            self.assertTrue(unresolved_safe_module.allowed, unresolved_safe_module.message)
            self.assertTrue(unresolved_safe_module.warning)
            bad_module = peer_ws / "badmod.py"
            bad_module.write_text(
                (
                    "import os, pathlib, sitecustomize\n"
                    "sitecustomize._ROOTS = (pathlib.Path('/'),)\n"
                    f"os.unlink({str(run_dir / 'shared_findings' / 'keep.json')!r})\n"
                ),
                encoding="utf-8",
            )
            blocked_bad_python_m = validate_bash_command(
                "python -m badmod",
                env=env,
                cwd=peer_ws,
            )
            self.assertFalse(blocked_bad_python_m.allowed)
            blocked_bad_module = validate_bash_command(
                "python -m ../scratch_module",
                env=env,
                cwd=task_root,
            )
            self.assertFalse(blocked_bad_module.allowed)

            blocked_xargs_delete = validate_bash_command(
                f"printf '%s\\n' {run_dir / 'shared_findings'} | xargs /bin/rm -rf",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_xargs_delete.allowed)

            safe_python_introspection = validate_bash_command(
                'python -c "import json; print(dir(json)[:2]); print(vars(type))"',
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(safe_python_introspection.allowed, safe_python_introspection.message)

            blocked_guard_introspection = validate_bash_command(
                "python -c \"import sys; print(sys.modules.get('site'+'customize'))\"",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(
                blocked_guard_introspection.allowed, blocked_guard_introspection.message
            )
            self.assertTrue(blocked_guard_introspection.warning)

            blocked_guard_introspection_delete = validate_bash_command(
                "python -c \"import sys, os; print(sys.modules); os.unlink('shared_findings/keep.json')\"",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_guard_introspection_delete.allowed)

            guarded = prepare_delete_guard_env(env, workspace=run_dir, agent_name="gen0_peer0")
            runtime_diag = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,subprocess; "
                        "subprocess.run(['ps','-p',str(os.getpid())], check=True, "
                        "stdout=subprocess.DEVNULL); "
                        "subprocess.run(['printf','ok'], check=True, "
                        "stdout=subprocess.DEVNULL)"
                    ),
                ],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(runtime_diag.returncode, 0, runtime_diag.stderr + runtime_diag.stdout)

    def test_blocks_rm_targets_outside_peer_workspace(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
            validate_bash_command,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            task_root = Path(tmp) / "task"
            peer_ws = run_dir / "peer_workspaces" / "gen0_peer0"
            peer_ws.mkdir(parents=True)
            task_root.mkdir()
            env = {
                "PRAXIST_SAFE_DELETE_ROOTS": str(peer_ws),
                "PRAXIST_PEER_WORKSPACE": str(peer_ws),
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_DELETE_GUARD_RUN_DIR": str(run_dir),
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
            }

            allowed = validate_bash_command(
                "rm -rf $PRAXIST_PEER_WORKSPACE/tmp",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed.allowed, allowed.message)

            agendas = run_dir / "agendas"
            agendas.mkdir()
            pi_guarded = prepare_delete_guard_env(
                {**env, "PEER_ID": "gen0_peer0"},
                workspace=run_dir,
                agent_name="pi_synthesizer",
            )
            runtime_pi_candidate = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "Path('agendas/research_agenda_gen1.yaml.candidate').write_text('x')"
                    ),
                ],
                env=pi_guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(
                runtime_pi_candidate.returncode,
                0,
                runtime_pi_candidate.stderr + runtime_pi_candidate.stdout,
            )
            self.assertTrue((agendas / "research_agenda_gen1.yaml.candidate").exists())

            peer_guarded = prepare_delete_guard_env(env, workspace=run_dir, agent_name="gen0_peer0")
            runtime_peer_candidate = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "Path('agendas/research_agenda_gen2.yaml.candidate').write_text('x')"
                    ),
                ],
                env=peer_guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertNotEqual(runtime_peer_candidate.returncode, 0)

            blocked_root = validate_bash_command(
                f"rm -rf {run_dir}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_root.allowed)
            self.assertIn("outside this agent's scratch workspace", blocked_root.message)

            blocked_absolute = validate_bash_command(
                f"/bin/rm -rf {run_dir / 'shared_findings'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_absolute.allowed)

            blocked_unlink = validate_bash_command(
                f"unlink {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_unlink.allowed)

            blocked_command = validate_bash_command(
                f"command rm -rf {run_dir / 'results' / 'other_variant'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_command.allowed)

            blocked_command_p = validate_bash_command(
                f"command -p rm -rf {run_dir / 'results' / 'other_variant'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_command_p.allowed)

            blocked_command_p_dashdash = validate_bash_command(
                f"command -p -- rm -rf {run_dir / 'results' / 'other_variant'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_command_p_dashdash.allowed)

            blocked_env = validate_bash_command(
                f"env -i PATH=/bin:/usr/bin rm -rf {run_dir / 'shared_findings'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_env.allowed)

            blocked_env_shell = validate_bash_command(
                f"env -i PATH=/bin:/usr/bin sh -c 'rm -rf {run_dir / 'shared_findings'}'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_env_shell.allowed)

            blocked_env_shell_dashdash = validate_bash_command(
                f"env -i PATH=/bin:/usr/bin sh -c -- '/bin/rm -rf {run_dir / 'shared_findings'}'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_env_shell_dashdash.allowed)

            blocked_env_split = validate_bash_command(
                f"env -S '/bin/rm -rf {run_dir / 'shared_findings'}'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_env_split.allowed)

            blocked_bash_lc = validate_bash_command(
                f"bash -lc '/bin/rm -rf {run_dir / 'shared_findings'}'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_bash_lc.allowed)

            blocked_sh_ec = validate_bash_command(
                f"sh -ec '/bin/rm -rf {run_dir / 'shared_findings'}'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_sh_ec.allowed)

            blocked_sh_c_dashdash = validate_bash_command(
                f"sh -c -- '/bin/rm -rf {run_dir / 'shared_findings'}'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_sh_c_dashdash.allowed)

            blocked_xargs_shell = validate_bash_command(
                f"printf x | xargs sh -c '/bin/rm -rf {run_dir / 'shared_findings'}'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_xargs_shell.allowed)

            blocked_multiline = validate_bash_command(
                f"echo ok\n/bin/rm -rf {run_dir / 'shared_findings'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_multiline.allowed)

            blocked_backtick = validate_bash_command(
                f"echo `/bin/rm -rf {run_dir / 'shared_findings'}`",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_backtick.allowed)

            blocked_compound = validate_bash_command(
                f"if true; then /bin/rm -rf {run_dir / 'shared_findings'}; fi",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_compound.allowed)

            blocked_xargs = validate_bash_command(
                f"printf '%s\\n' {run_dir / 'shared_findings'} | xargs /bin/rm -rf",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_xargs.allowed)

            blocked_expanded_rm = validate_bash_command(
                'x=; p=$PRAXIST_DELETE_GUARD_RUN_DIR/shared_findings; /bin/r${x}m -rf "$p"',
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_expanded_rm.allowed)

            blocked_command_expanded_rm = validate_bash_command(
                'x=; p=$PRAXIST_DELETE_GUARD_RUN_DIR/shared_findings; command -p r${x}m -rf "$p"',
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_command_expanded_rm.allowed)

            blocked_expanded_node = validate_bash_command(
                "no$'de' -e 'console.log(1)'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_expanded_node.allowed)

            blocked_mv = validate_bash_command(
                f"mv {run_dir / 'shared_findings' / 'keep.txt'} $PRAXIST_PEER_WORKSPACE/tmp/",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_mv.allowed)
            allowed_atomic_mv = validate_bash_command(
                f"mv $PRAXIST_PEER_WORKSPACE/tmp/finding.tmp {run_dir / 'shared_findings' / 'gen0_peer0_atomic.json'}",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_atomic_mv.allowed, allowed_atomic_mv.message)
            blocked_other_peer_atomic_mv = validate_bash_command(
                f"mv $PRAXIST_PEER_WORKSPACE/tmp/finding.tmp {run_dir / 'shared_findings' / 'gen0_peer9_atomic.json'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_other_peer_atomic_mv.allowed)
            blocked_frontier_mv = validate_bash_command(
                f"mv $PRAXIST_PEER_WORKSPACE/tmp/finding.tmp {run_dir / 'frontier' / 'frontier.jsonl'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_frontier_mv.allowed)
            for form in (
                f"mv -t {task_root} $PRAXIST_PEER_WORKSPACE/tmp/payload.txt",
                f"mv -t{task_root} $PRAXIST_PEER_WORKSPACE/tmp/payload.txt",
                f"mv --target-directory={task_root} $PRAXIST_PEER_WORKSPACE/tmp/payload.txt",
            ):
                blocked_task_target_mv = validate_bash_command(form, env=env, cwd=run_dir)
                self.assertFalse(blocked_task_target_mv.allowed, form)

            blocked_tar_remove = validate_bash_command(
                f"tar --remove-files -cf $PRAXIST_PEER_WORKSPACE/tmp/a.tar {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_tar_remove.allowed)

            blocked_zip_move = validate_bash_command(
                f"zip -m $PRAXIST_PEER_WORKSPACE/tmp/a.zip {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_zip_move.allowed)

            blocked_shred_remove = validate_bash_command(
                f"shred -u {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_shred_remove.allowed)

            blocked_rsync_delete = validate_bash_command(
                f"rsync --delete $PRAXIST_PEER_WORKSPACE/tmp/ {run_dir / 'shared_findings'}/",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_rsync_delete.allowed)

            blocked_redirection = validate_bash_command(
                f": > {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_redirection.allowed)

            peer_memory = run_dir / "gen_0" / "peers" / "gen0_peer0" / "memory"
            peer_memory.mkdir(parents=True)
            allowed_memory_redirection = validate_bash_command(
                f": >> {peer_memory / 'experiment_ledger.jsonl'}",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_memory_redirection.allowed, allowed_memory_redirection.message)

            blocked_truncate = validate_bash_command(
                f"truncate -s 0 {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_truncate.allowed)

            blocked_dd = validate_bash_command(
                f"dd if=/dev/null of={run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_dd.allowed)

            blocked_cp = validate_bash_command(
                f"cp $PRAXIST_PEER_WORKSPACE/tmp/x {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_cp.allowed)

            blocked_cp_t = validate_bash_command(
                f"cp -t {run_dir / 'shared_findings'} $PRAXIST_PEER_WORKSPACE/tmp/x",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_cp_t.allowed)

            blocked_cp_target_directory = validate_bash_command(
                f"cp --target-directory={run_dir / 'shared_findings'} $PRAXIST_PEER_WORKSPACE/tmp/x",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_cp_target_directory.allowed)

            blocked_ln = validate_bash_command(
                f"ln -sf $PRAXIST_PEER_WORKSPACE/tmp/x {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_ln.allowed)

            blocked_chmod = validate_bash_command(
                f"chmod -R 000 {run_dir / 'shared_findings'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_chmod.allowed)

            blocked_awk = validate_bash_command(
                f"awk 'BEGIN {{print 1 > \"{run_dir / 'shared_findings' / 'keep.txt'}\"}}'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_awk.allowed)

            blocked_rsync_destination = validate_bash_command(
                f"rsync $PRAXIST_PEER_WORKSPACE/tmp/ {run_dir / 'shared_findings'}/",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_rsync_destination.allowed)

            blocked_hardlink_source = validate_bash_command(
                f"ln {run_dir / 'shared_findings' / 'keep.txt'} $PRAXIST_PEER_WORKSPACE/tmp/hard",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_hardlink_source.allowed)

            build_dir = peer_ws / "build"
            build_dir.mkdir(parents=True)
            (build_dir / "Makefile").write_text("all:\n\ttrue\n", encoding="utf-8")
            (build_dir / "build.ninja").write_text(
                "rule noop\n  command = true\nbuild all: noop\n",
                encoding="utf-8",
            )
            allowed_make = validate_bash_command(
                "make -C $PRAXIST_PEER_WORKSPACE/build all",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_make.allowed, allowed_make.message)

            allowed_make_after_cd = validate_bash_command(
                "cd $PRAXIST_PEER_WORKSPACE/build && make all",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_make_after_cd.allowed, allowed_make_after_cd.message)

            blocked_make_clean = validate_bash_command(
                "make -C $PRAXIST_PEER_WORKSPACE/build clean",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_make_clean.allowed)

            bad_build_dir = peer_ws / "bad_build"
            bad_build_dir.mkdir(parents=True)
            (bad_build_dir / "Makefile").write_text(
                f"all:\n\t/bin/rm -rf {run_dir / 'shared_findings'}\n",
                encoding="utf-8",
            )
            blocked_make_recipe = validate_bash_command(
                "make -C $PRAXIST_PEER_WORKSPACE/bad_build all",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_make_recipe.allowed)

            evasive_build_dir = peer_ws / "evasive_build"
            evasive_build_dir.mkdir(parents=True)
            (evasive_build_dir / "Makefile").write_text(
                'R=/bin/r\nM=m\nall:\n\t$(R)$(M) -f "$$P"\n',
                encoding="utf-8",
            )
            blocked_make_variable_recipe = validate_bash_command(
                "P=$PRAXIST_DELETE_GUARD_RUN_DIR/shared_findings/keep.txt make -C $PRAXIST_PEER_WORKSPACE/evasive_build all",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_make_variable_recipe.allowed)

            blocked_env_make = validate_bash_command(
                "env -i PATH=/usr/bin:/bin make -C $PRAXIST_PEER_WORKSPACE/build all",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_env_make.allowed)

            blocked_make = validate_bash_command(
                "make -f $PRAXIST_PEER_WORKSPACE/tmp/Makefile",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_make.allowed)

            allowed_ninja = validate_bash_command(
                "ninja -C $PRAXIST_PEER_WORKSPACE/build",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_ninja.allowed, allowed_ninja.message)

            allowed_ninja_after_cd = validate_bash_command(
                "cd $PRAXIST_PEER_WORKSPACE/build && ninja",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_ninja_after_cd.allowed, allowed_ninja_after_cd.message)

            blocked_cd_rm = validate_bash_command(
                "cd $PRAXIST_DELETE_GUARD_RUN_DIR/shared_findings && rm keep.txt",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_cd_rm.allowed)

            blocked_tar_checkpoint = validate_bash_command(
                f"tar --checkpoint=1 --checkpoint-action=exec='/bin/rm -f {run_dir / 'shared_findings' / 'keep.txt'}' -cf /dev/null $PRAXIST_PEER_WORKSPACE/tmp/x",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_tar_checkpoint.allowed)

            blocked_sort_output = validate_bash_command(
                f"sort /dev/null -o {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_sort_output.allowed)

            blocked_tar_extract = validate_bash_command(
                f"tar -xf $PRAXIST_PEER_WORKSPACE/tmp/payload.tar -C {run_dir / 'shared_findings'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_tar_extract.allowed)

            blocked_touch = validate_bash_command(
                f"touch {run_dir / 'shared_findings' / 'evil.json'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_touch.allowed)

            blocked_mkdir = validate_bash_command(
                f"mkdir {run_dir / 'shared_findings' / 'evil_dir'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_mkdir.allowed)

            blocked_fallocate = validate_bash_command(
                f"fallocate -l 1024 {run_dir / 'shared_findings' / 'keep.bin'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_fallocate.allowed)

            blocked_gcc_output = validate_bash_command(
                f"gcc $PRAXIST_PEER_WORKSPACE/tmp/a.c -o {run_dir / 'shared_findings' / 'keep.json'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_gcc_output.allowed)

            blocked_curl_output = validate_bash_command(
                f"curl -o {run_dir / 'shared_findings' / 'keep.json'} file:///tmp/source",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_curl_output.allowed)

            blocked_wget_output = validate_bash_command(
                f"wget -O {run_dir / 'shared_findings' / 'keep.json'} file:///tmp/source",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_wget_output.allowed)

            blocked_openssl_output = validate_bash_command(
                f"/usr/bin/openssl rand -out {run_dir / 'shared_findings' / 'keep.json'} 16",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_openssl_output.allowed)

            blocked_ex_editor = validate_bash_command(
                f"/usr/bin/ex -es {run_dir / 'shared_findings' / 'keep.json'} -c wq",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_ex_editor.allowed)

            allowed_openssl_scratch = validate_bash_command(
                "/usr/bin/openssl rand -out $PRAXIST_PEER_WORKSPACE/tmp/random.bin 16",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_openssl_scratch.allowed, allowed_openssl_scratch.message)

            blocked_cp_hardlink = validate_bash_command(
                f"cp -l {run_dir / 'shared_findings' / 'keep.txt'} $PRAXIST_PEER_WORKSPACE/tmp/linked.txt",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_cp_hardlink.allowed)

            blocked_cp_attached_target = validate_bash_command(
                "cp -t$PRAXIST_RUN_DIR/shared_findings $PRAXIST_PEER_WORKSPACE/tmp/source.txt",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_cp_attached_target.allowed)

            blocked_cp_recursive_target = validate_bash_command(
                "cp -rt $PRAXIST_RUN_DIR/shared_findings $PRAXIST_PEER_WORKSPACE/tmp/source.txt",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_cp_recursive_target.allowed)

            blocked_install_combined_target = validate_bash_command(
                "install -Dt $PRAXIST_RUN_DIR/shared_findings $PRAXIST_PEER_WORKSPACE/tmp/source.txt",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_install_combined_target.allowed)

            blocked_curl_output_dir = validate_bash_command(
                "curl --output-dir $PRAXIST_RUN_DIR/shared_findings -O file:///tmp/source.json",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_curl_output_dir.allowed)

            blocked_wget_output_dir = validate_bash_command(
                "wget -P $PRAXIST_RUN_DIR/shared_findings file:///tmp/source.json",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_wget_output_dir.allowed)

            blocked_curl_remote_name = validate_bash_command(
                "curl -O https://example.com/shared_store.db",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_curl_remote_name.allowed)

            blocked_wget_plain = validate_bash_command(
                "wget https://example.com/shared_store.db",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_wget_plain.allowed)

            blocked_tar_archive = validate_bash_command(
                "tar -cf $PRAXIST_RUN_DIR/frontier/out.tar $PRAXIST_PEER_WORKSPACE/tmp",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_tar_archive.allowed)

            blocked_zip_archive = validate_bash_command(
                "zip $PRAXIST_RUN_DIR/frontier/out.zip $PRAXIST_PEER_WORKSPACE/tmp/source.txt",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_zip_archive.allowed)

            blocked_unzip_implicit = validate_bash_command(
                "unzip /tmp/archive.zip",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_unzip_implicit.allowed)

            blocked_patch_implicit = validate_bash_command(
                "patch -p0 < /tmp/change.diff",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_patch_implicit.allowed)

            blocked_split_implicit = validate_bash_command(
                "split /tmp/input.bin",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_split_implicit.allowed)

            blocked_git_init = validate_bash_command(
                "git init $PRAXIST_RUN_DIR/frontier",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_git_init.allowed)

            blocked_git_clone = validate_bash_command(
                "git clone https://example.com/repo.git $PRAXIST_RUN_DIR/shared_findings/clone",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_git_clone.allowed)

            blocked_git_archive = validate_bash_command(
                "git archive -o $PRAXIST_RUN_DIR/shared_findings/out.tar HEAD",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_git_archive.allowed)

            blocked_git_worktree = validate_bash_command(
                "git worktree add $PRAXIST_RUN_DIR/frontier/wt HEAD",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_git_worktree.allowed)

            allowed_git_status = validate_bash_command(
                "git status --short",
                env=env,
                cwd=peer_ws,
            )
            self.assertTrue(allowed_git_status.allowed, allowed_git_status.message)

            allowed_git_diff = validate_bash_command(
                "git diff --stat",
                env=env,
                cwd=peer_ws,
            )
            self.assertTrue(allowed_git_diff.allowed, allowed_git_diff.message)

            blocked_git_diff_output = validate_bash_command(
                "git diff --output $PRAXIST_RUN_DIR/shared_findings/diff.patch",
                env=env,
                cwd=peer_ws,
            )
            self.assertFalse(blocked_git_diff_output.allowed)

            blocked_git_diff_output_equals = validate_bash_command(
                "git diff --output=$PRAXIST_RUN_DIR/shared_findings/diff.patch",
                env=env,
                cwd=peer_ws,
            )
            self.assertFalse(blocked_git_diff_output_equals.allowed)

            source = peer_ws / "tmp" / "source.txt"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("source", encoding="utf-8")
            guarded = prepare_delete_guard_env(env, workspace=run_dir, agent_name="gen0_peer0")
            runtime_cp = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os, subprocess; "
                        f"p={str(run_dir / 'shared_findings')!r}; "
                        f"s={str(source)!r}; "
                        "r=subprocess.run(['cp','-rt',p,s], text=True, capture_output=True); "
                        "assert r.returncode != 0, r.stdout + r.stderr"
                    ),
                ],
                env=guarded,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(runtime_cp.returncode, 0, runtime_cp.stderr + runtime_cp.stdout)

            (run_dir / "scratch_module.py").write_text("print('safe')\n", encoding="utf-8")
            allowed_python_m = validate_bash_command(
                "python -m scratch_module",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_python_m.allowed, allowed_python_m.message)

            allowed_protected_pids = validate_bash_command(
                "python -m praxist.plugins.workflow_stages.research_loop.backend.protected_pids register --pid 123 --label smoke",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_protected_pids.allowed, allowed_protected_pids.message)

            allowed_literal_eval = validate_bash_command(
                f"{task_root}/external_env/bin/python evaluations/generic/run.py "
                "--variant-name gen0_peer0_alpha "
                "--variant-main-script $PRAXIST_DELETE_GUARD_RUN_DIR/variants/gen0_peer0_alpha/train.py "
                "--output-dir $PRAXIST_DELETE_GUARD_RUN_DIR/results/gen0_peer0_alpha --epochs 1 --max-tier T1",
                env={
                    **env,
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TRUSTED_PROJECT_EXTRA_ROOTS": str(task_root),
                },
                cwd=task_root,
            )
            self.assertTrue(allowed_literal_eval.allowed, allowed_literal_eval.message)

            blocked_git_clean = validate_bash_command(
                f"git clean -fdx {run_dir / 'shared_findings'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_git_clean.allowed)

            (peer_ws / "subdir").mkdir()
            blocked_glob_escape = validate_bash_command(
                "rm -rf $PRAXIST_PEER_WORKSPACE/*/../../../shared_findings",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_glob_escape.allowed)

            fake_binary = peer_ws / "tmp" / "fake_elf"
            fake_binary.parent.mkdir(parents=True, exist_ok=True)
            fake_binary.write_bytes(b"\x7fELFnotreally")
            fake_binary.chmod(0o755)
            blocked_binary = validate_bash_command(
                f"{fake_binary} {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_binary.allowed)
            self.assertIn("binary executable", blocked_binary.message)

    def test_blocks_find_delete_and_python_rmtree_outside_peer_workspace(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            validate_bash_command,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            peer_ws = run_dir / "peer_workspaces" / "gen1_peer4"
            peer_ws.mkdir(parents=True)
            env = {
                "PRAXIST_SAFE_DELETE_ROOTS": str(peer_ws),
                "PRAXIST_PEER_WORKSPACE": str(peer_ws),
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_DELETE_GUARD_RUN_DIR": str(run_dir),
            }

            blocked_find = validate_bash_command(
                f"find {run_dir / 'gen_1'} -type f -delete",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_find.allowed)

            blocked_find_multi_root = validate_bash_command(
                f"find $PRAXIST_PEER_WORKSPACE {run_dir / 'shared_findings'} -type f -delete",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_find_multi_root.allowed)

            blocked_ambiguous_find = validate_bash_command(
                "find $UNKNOWN -type f -delete",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_ambiguous_find.allowed)

            blocked_find_exec = validate_bash_command(
                f"find {run_dir / 'shared_findings'} -type f -exec /bin/rm {{}} +",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_find_exec.allowed)

            blocked_find_exec_unlink = validate_bash_command(
                f"find {run_dir / 'shared_findings'} -type f -exec unlink {{}} ';'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_find_exec_unlink.allowed)

            blocked_find_exec_payload = validate_bash_command(
                f"find $PRAXIST_PEER_WORKSPACE -type f -exec /bin/rm -rf {run_dir / 'shared_findings'} {{}} +",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_find_exec_payload.allowed)

            blocked_find_exec_env_split = validate_bash_command(
                f"find $PRAXIST_PEER_WORKSPACE -type f -exec env -S '/bin/rm -rf {run_dir / 'shared_findings'}' {{}} +",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_find_exec_env_split.allowed)

            blocked_find_exec_cp = validate_bash_command(
                f"find $PRAXIST_PEER_WORKSPACE -type f -exec cp {{}} {run_dir / 'shared_findings' / 'evil.json'} ';'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_find_exec_cp.allowed)

            blocked_find_exec_touch = validate_bash_command(
                f"find $PRAXIST_PEER_WORKSPACE -type f -exec touch {run_dir / 'shared_findings' / 'evil.json'} ';'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_find_exec_touch.allowed)

            blocked_find_exec_mkdir = validate_bash_command(
                f"find $PRAXIST_PEER_WORKSPACE -type f -exec mkdir {run_dir / 'shared_findings' / 'evil_dir'} ';'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_find_exec_mkdir.allowed)

            blocked_python = validate_bash_command(
                f"python - <<'PY'\nimport shutil, os\nshutil.rmtree('{run_dir}')\nPY",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_python.allowed)

            blocked_python_import = validate_bash_command(
                f"python - <<'PY'\nfrom shutil import rmtree\nrmtree('{run_dir / 'frontier'}')\nPY",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_python_import.allowed)

            blocked_python_getattr = validate_bash_command(
                "python - <<'PY'\n"
                f"getattr(__import__('shutil'), 'rmt' + 'ree')('{run_dir / 'frontier'}')\nPY",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_python_getattr.allowed)

            blocked_python_getattr_relative = validate_bash_command(
                "python - <<'PY'\ngetattr(__import__('shutil'), 'rmt' + 'ree')('gen_1')\nPY",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_python_getattr_relative.allowed)

            blocked_python_system = validate_bash_command(
                f"python -c \"import os; os.system('/bin/rm -rf {run_dir / 'frontier'}')\"",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_python_system.allowed)

            blocked_python_subprocess = validate_bash_command(
                "python - <<'PY'\nimport subprocess\n"
                "subprocess.run(['/bin/' + 'rm', '-rf', 'results/other_variant'])\nPY",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_python_subprocess.allowed)

            introspection_script = peer_ws / "tmp" / "introspection_escape.py"
            introspection_script.parent.mkdir(parents=True, exist_ok=True)
            introspection_script.write_text(
                "import importlib\nm = importlib.import_module('site' + 'customize')\n"
                "m.__dict__['_ROOTS'] = ['/']\n",
                encoding="utf-8",
            )
            blocked_introspection_script = validate_bash_command(
                f"python {introspection_script}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_introspection_script.allowed)

            blocked_python_ctypes = validate_bash_command(
                'python -c "import ctypes; print(ctypes.CDLL(None))"',
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(blocked_python_ctypes.allowed, blocked_python_ctypes.message)
            self.assertTrue(blocked_python_ctypes.warning)

            blocked_python_sitecustomize = validate_bash_command(
                'python -c "import sitecustomize; print(sitecustomize.__dict__)"',
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(
                blocked_python_sitecustomize.allowed, blocked_python_sitecustomize.message
            )
            self.assertTrue(blocked_python_sitecustomize.warning)

            python_escape_script = peer_ws / "tmp" / "kwdefaults_escape.py"
            python_escape_script.parent.mkdir(parents=True, exist_ok=True)
            python_escape_script.write_text(
                "import os,sys\ngetattr(os, 'un'+'link').__kwdefaults__['_orig'](sys.argv[1])\n",
                encoding="utf-8",
            )
            blocked_python_script_escape = validate_bash_command(
                f"python {python_escape_script} {run_dir / 'shared_findings' / 'keep.txt'}",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_python_script_escape.allowed)

            blocked_python_s = validate_bash_command(
                'python -S -c "from pathlib import Path; import shutil; '
                "getattr(shutil, 'rmt' + 'ree')(Path.cwd()/('shared_' + 'findings'))\"",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_python_s.allowed)

            warned_python_isolated = validate_bash_command(
                "python -I -c 'print(1)'",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(warned_python_isolated.allowed, warned_python_isolated.message)
            self.assertEqual(warned_python_isolated.severity, "warning")

            warned_python_ignore_env = validate_bash_command(
                "python -E -c 'print(1)'",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(warned_python_ignore_env.allowed, warned_python_ignore_env.message)
            self.assertEqual(warned_python_ignore_env.severity, "warning")

            warned_pythonpath_assignment = validate_bash_command(
                "PYTHONPATH= python -c 'print(1)'",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(
                warned_pythonpath_assignment.allowed,
                warned_pythonpath_assignment.message,
            )
            self.assertEqual(warned_pythonpath_assignment.severity, "warning")

            warned_env_i_python = validate_bash_command(
                "env -i PATH=/bin:/usr/bin python -c 'print(1)'",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(warned_env_i_python.allowed, warned_env_i_python.message)
            self.assertEqual(warned_env_i_python.severity, "warning")

            warned_unset_pythonpath = validate_bash_command(
                "env -u PYTHONPATH python -c 'print(1)'",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(warned_unset_pythonpath.allowed, warned_unset_pythonpath.message)
            self.assertEqual(warned_unset_pythonpath.severity, "warning")

            blocked_perl_inline = validate_bash_command(
                f"perl -e 'unlink q({run_dir / 'shared_findings' / 'keep.txt'})'",
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_perl_inline.allowed)

            warned_perl_script = validate_bash_command(
                "perl $PRAXIST_PEER_WORKSPACE/tmp/delete.pl",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(warned_perl_script.allowed, warned_perl_script.message)
            self.assertEqual(warned_perl_script.severity, "warning")

            warned_pipe_to_sh = validate_bash_command(
                "cat $PRAXIST_PEER_WORKSPACE/tmp/evil.sh | sh",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(warned_pipe_to_sh.allowed, warned_pipe_to_sh.message)
            self.assertEqual(warned_pipe_to_sh.severity, "warning")

            warned_python_stdin = validate_bash_command(
                "python < $PRAXIST_PEER_WORKSPACE/tmp/evil.py",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(warned_python_stdin.allowed, warned_python_stdin.message)
            self.assertEqual(warned_python_stdin.severity, "warning")

            warned_bash_stdin = validate_bash_command(
                "cat $PRAXIST_PEER_WORKSPACE/tmp/evil.sh | bash",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(warned_bash_stdin.allowed, warned_bash_stdin.message)
            self.assertEqual(warned_bash_stdin.severity, "warning")

            warned_unbalanced_quote = validate_bash_command(
                'python3 -c "print(1)',
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(warned_unbalanced_quote.allowed, warned_unbalanced_quote.message)
            self.assertEqual(warned_unbalanced_quote.severity, "warning")
            self.assertIn("not fully parseable", warned_unbalanced_quote.message)
            self.assertIn("shell_tokenization_warning", warned_unbalanced_quote.rule_id)

            blocked_unbalanced_rm = validate_bash_command(
                f'rm -rf "{run_dir / "frontier"}',
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_unbalanced_rm.allowed)

            allowed_find = validate_bash_command(
                "find $PRAXIST_PEER_WORKSPACE/tmp -type f -delete",
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_find.allowed, allowed_find.message)

    def test_allows_safe_make_ninja_in_task_project_but_blocks_protected_recipe(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
            validate_bash_command,
        )

        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "task"
            run_dir = task_root / "experiments" / "run_1"
            build_dir = task_root / "build"
            run_dir.mkdir(parents=True)
            build_dir.mkdir(parents=True)
            (task_root / "Makefile").write_text("all:\n\tprintf ok\n", encoding="utf-8")
            (build_dir / "build.ninja").write_text(
                "rule noop\n  command = printf ok\nbuild all: noop\n", encoding="utf-8"
            )
            env = prepare_delete_guard_env(
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_WORKSPACE_ROOT": str(task_root),
                    "PEER_ID": "gen0_peer0",
                    "PATH": "/bin:/usr/bin:/usr/sbin",
                },
                workspace=task_root,
                agent_name="gen0_peer0",
            )

            allowed_make = validate_bash_command("make all", env=env, cwd=task_root)
            self.assertTrue(allowed_make.allowed, allowed_make.message)
            (build_dir / "Custom.mk").write_text("all:\n\tprintf ok\n", encoding="utf-8")
            allowed_make_custom = validate_bash_command(
                f"make -C {build_dir} -f Custom.mk all",
                env=env,
                cwd=task_root,
            )
            self.assertTrue(allowed_make_custom.allowed, allowed_make_custom.message)
            allowed_make_attached_c = validate_bash_command(
                f"make -C{build_dir} -f Custom.mk all",
                env=env,
                cwd=task_root,
            )
            self.assertTrue(allowed_make_attached_c.allowed, allowed_make_attached_c.message)
            multi_build_dir = task_root / "multi" / "child"
            multi_build_dir.mkdir(parents=True)
            (multi_build_dir / "Safe.mk").write_text("all:\n\tprintf ok\n", encoding="utf-8")
            (multi_build_dir / "Makefile").write_text(
                f"all:\n\tprintf bad > {run_dir / 'shared_findings' / 'default_should_not_run.json'}\n",
                encoding="utf-8",
            )
            allowed_make_multi_c_custom = validate_bash_command(
                f"make -C {multi_build_dir.parent} -C child -f Safe.mk all",
                env=env,
                cwd=task_root,
            )
            self.assertTrue(
                allowed_make_multi_c_custom.allowed,
                allowed_make_multi_c_custom.message,
            )
            outside_makefile = Path(tmp) / "outside.mk"
            outside_makefile.write_text("all:\n\tprintf ok\n", encoding="utf-8")
            blocked_outside_makefile = validate_bash_command(
                f"make -f {outside_makefile} all",
                env=env,
                cwd=task_root,
            )
            self.assertFalse(blocked_outside_makefile.allowed)
            allowed_ninja = validate_bash_command(
                f"ninja -C {build_dir} all", env=env, cwd=task_root
            )
            self.assertTrue(allowed_ninja.allowed, allowed_ninja.message)
            (build_dir / "custom.ninja").write_text(
                "rule noop\n  command = printf ok\nbuild all: noop\n",
                encoding="utf-8",
            )
            (build_dir / "build.ninja").write_text(
                f"rule bad\n  command = printf bad > {run_dir / 'shared_findings' / 'default_ninja_should_not_run.json'}\nbuild all: bad\n",
                encoding="utf-8",
            )
            allowed_ninja_custom = validate_bash_command(
                f"ninja -C {build_dir} -f custom.ninja all",
                env=env,
                cwd=task_root,
            )
            self.assertTrue(allowed_ninja_custom.allowed, allowed_ninja_custom.message)

            (task_root / "common.mk").write_text("all:\n\tprintf ok\n", encoding="utf-8")
            (task_root / "Makefile").write_text("include\tcommon.mk\n", encoding="utf-8")
            allowed_make_include = validate_bash_command("make all", env=env, cwd=task_root)
            self.assertTrue(allowed_make_include.allowed, allowed_make_include.message)

            (build_dir / "safe.ninja").write_text(
                "rule noop\n  command = printf ok\nbuild all: noop\n",
                encoding="utf-8",
            )
            (build_dir / "build.ninja").write_text("subninja safe.ninja\n", encoding="utf-8")
            allowed_ninja_subninja = validate_bash_command(
                f"ninja -C {build_dir} all", env=env, cwd=task_root
            )
            self.assertTrue(allowed_ninja_subninja.allowed, allowed_ninja_subninja.message)

            (task_root / "Makefile").write_text(
                f"all:\n\tprintf bad > {run_dir / 'shared_findings' / 'evil.json'}\n",
                encoding="utf-8",
            )
            blocked_make = validate_bash_command("make all", env=env, cwd=task_root)
            self.assertFalse(blocked_make.allowed)

            (task_root / "evil.mk").write_text(
                f"all:\n\tprintf bad > {run_dir / 'shared_findings' / 'evil_include.json'}\n",
                encoding="utf-8",
            )
            (task_root / "Makefile").write_text("include\tevil.mk\n", encoding="utf-8")
            blocked_make_include = validate_bash_command("make all", env=env, cwd=task_root)
            self.assertFalse(blocked_make_include.allowed)

            (build_dir / "evil.ninja").write_text(
                f"rule bad\n  command = printf bad > {run_dir / 'shared_findings' / 'evil_ninja_include.json'}\nbuild all: bad\n",
                encoding="utf-8",
            )
            (build_dir / "build.ninja").write_text("subninja evil.ninja\n", encoding="utf-8")
            blocked_ninja_subninja = validate_bash_command(
                f"ninja -C {build_dir} all", env=env, cwd=task_root
            )
            self.assertFalse(blocked_ninja_subninja.allowed)

            (task_root / "description.md").write_text("canonical\n", encoding="utf-8")
            (task_root / "Makefile").write_text(
                "all:\n\tprintf bad > description.md\n",
                encoding="utf-8",
            )
            blocked_task_write_make = validate_bash_command("make all", env=env, cwd=task_root)
            self.assertFalse(blocked_task_write_make.allowed)

            (task_root / "Makefile").write_text(
                "all:\n\ttouch description.md\n",
                encoding="utf-8",
            )
            blocked_touch_make = validate_bash_command("make all", env=env, cwd=task_root)
            self.assertFalse(blocked_touch_make.allowed)

            (task_root / "Makefile").write_text("clean:\n\trm -rf build\n", encoding="utf-8")
            blocked_clean = validate_bash_command("make clean", env=env, cwd=task_root)
            self.assertFalse(blocked_clean.allowed)

    def test_prepare_env_adds_sitecustomize_warning_log_and_safe_workspace(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            env = {"PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen2_peer7", "PATH": "/bin"}
            guarded = prepare_delete_guard_env(env, workspace=Path(tmp), agent_name="agent")

            self.assertNotIn("BASH_ENV", guarded)
            self.assertIn("PYTHONPATH", guarded)
            self.assertIn("PRAXIST_GUARD_WARNINGS_PATH", guarded)
            self.assertIn("PRAXIST_PEER_WORKSPACE", guarded)
            self.assertTrue(Path(guarded["PRAXIST_PEER_WORKSPACE"]).exists())
            self.assertLessEqual(len(os.fsencode(guarded["TMPDIR"])), 64)
            self.assertIn(
                Path(guarded["TMPDIR"]).resolve(),
                {
                    Path(raw).resolve()
                    for raw in guarded["PRAXIST_SAFE_DELETE_ROOTS"].split(os.pathsep)
                    if raw
                },
            )
            self.assertEqual(guarded["PATH"], "/bin")
            self.assertTrue(
                (
                    run_dir / ".runtime_guards" / "gen2_peer7" / "python_site" / "sitecustomize.py"
                ).exists()
            )

    def test_runtime_guard_allows_generation_scoped_peer_results(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
            validate_tool_use,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            guarded = prepare_delete_guard_env(
                {**os.environ, "PRAXIST_RUN_DIR": str(run_dir), "PEER_ID": "gen0_peer3"},
                workspace=run_dir,
                agent_name="gen0_peer3",
            )
            (run_dir / "results" / "gen_0").mkdir(parents=True)
            owned = run_dir / "results" / "gen_0" / "gen0_peer3" / "candidate"
            pretool_cases = (
                ("Write", {"file_path": str(owned / "summary.json")}),
                ("Bash", {"command": f"mkdir -p {shlex.quote(str(owned))}"}),
                (
                    "Bash",
                    {
                        "command": (
                            f"mv {shlex.quote(str(owned / 'summary.json.tmp'))} "
                            f"{shlex.quote(str(owned / 'summary.json'))}"
                        )
                    },
                ),
                ("Bash", {"command": f"rm -f {shlex.quote(str(owned / 'summary.json.tmp'))}"}),
            )
            for tool_name, tool_input in pretool_cases:
                decision = validate_tool_use(tool_name, tool_input, env=guarded, cwd=run_dir)
                self.assertTrue(decision.allowed, decision.message)

            allowed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,sys; from pathlib import Path; "
                        "target=Path(sys.argv[1]); target.mkdir(parents=True); "
                        "tmp=target/'summary.json.tmp'; tmp.write_text('{}'); "
                        "os.replace(tmp,target/'summary.json'); "
                        "capsule=target/'source_capsule'; capsule.mkdir(); "
                        "(capsule/'manifest.json').write_bytes(b'{}')"
                    ),
                    str(owned),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr + allowed.stdout)
            self.assertEqual((owned / "summary.json").read_text(encoding="utf-8"), "{}")
            self.assertTrue((owned / "source_capsule" / "manifest.json").is_file())

            other_peer = run_dir / "results" / "gen_0" / "gen0_peer9" / "candidate"
            blocked = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True)",
                    str(other_peer),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertFalse(other_peer.exists())
            blocked_pretool = validate_tool_use(
                "Write",
                {"file_path": str(other_peer / "summary.json")},
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_pretool.allowed)

    def test_blocks_write_tool_against_shared_run_state(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
            validate_tool_use,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            task_root = Path(tmp) / "task"
            peer_ws = run_dir / "peer_workspaces" / "gen0_peer0"
            peer_ws.mkdir(parents=True)
            task_root.mkdir()
            protected_task_file = task_root / "assets" / "harness" / "env" / "market_env.py"
            protected_task_file.parent.mkdir(parents=True)
            protected_task_file.write_text("# canonical env\n", encoding="utf-8")
            env = {
                "PRAXIST_SAFE_DELETE_ROOTS": str(peer_ws),
                "PRAXIST_PEER_WORKSPACE": str(peer_ws),
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_DELETE_GUARD_RUN_DIR": str(run_dir),
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
            }
            blocked = validate_tool_use(
                "Write",
                {"file_path": str(run_dir / "shared_findings" / "x.json")},
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked.allowed)
            agenda_path = run_dir / "agendas" / "research_agenda_gen9.yaml"
            blocked_peer_agenda = validate_tool_use(
                "Write",
                {"file_path": str(agenda_path)},
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_peer_agenda.allowed)
            pi_env = {**env, "PRAXIST_DELETE_GUARD_AGENT": "pi_synthesizer"}
            allowed_pi_agenda = validate_tool_use(
                "Write",
                {"file_path": str(agenda_path)},
                env=pi_env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_pi_agenda.allowed, allowed_pi_agenda.message)
            allowed_pi_redirect = validate_tool_use(
                "Bash",
                {"command": f"printf 'generation: 9\\n' > {agenda_path}"},
                env=pi_env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_pi_redirect.allowed, allowed_pi_redirect.message)
            blocked_task_write = validate_tool_use(
                "Write",
                {"file_path": str(protected_task_file)},
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_task_write.allowed)
            allowed_variant_write = validate_tool_use(
                "Write",
                {"file_path": str(run_dir / "variants" / "gen0_peer0_alpha" / "train.py")},
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_variant_write.allowed, allowed_variant_write.message)
            session_guarded_env = prepare_delete_guard_env(
                {**env, "PEER_ID": "gen0_peer0"},
                workspace=run_dir,
                agent_name="gen0_peer0-session_000_test",
            )
            self.assertEqual(session_guarded_env["PRAXIST_DELETE_GUARD_AGENT"], "gen0_peer0")
            allowed_session_variant_write = validate_tool_use(
                "Write",
                {"file_path": str(run_dir / "variants" / "gen0_peer0_beta" / "train.py")},
                env=session_guarded_env,
                cwd=run_dir,
            )
            self.assertTrue(
                allowed_session_variant_write.allowed,
                allowed_session_variant_write.message,
            )
            blocked_other_variant_write = validate_tool_use(
                "Write",
                {"file_path": str(run_dir / "variants" / "gen0_peer1_alpha" / "train.py")},
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_other_variant_write.allowed)
            blocked_new_project_shadow = validate_tool_use(
                "Write",
                {"file_path": str(task_root / "env_factory.py")},
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_new_project_shadow.allowed)
            allowed_result_write = validate_tool_use(
                "Write",
                {"file_path": str(run_dir / "results" / "gen0_peer0_alpha" / "summary.json")},
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_result_write.allowed, allowed_result_write.message)
            allowed_finding_write = validate_tool_use(
                "Write",
                {"file_path": str(run_dir / "shared_findings" / "gen0_peer0_alpha.json")},
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_finding_write.allowed, allowed_finding_write.message)
            allowed_dig_amendment_write = validate_tool_use(
                "Write",
                {
                    "file_path": str(
                        run_dir
                        / "gen_0"
                        / "peers"
                        / "gen0_peer0"
                        / "dig"
                        / "contract_amendment.yaml"
                    )
                },
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(
                allowed_dig_amendment_write.allowed,
                allowed_dig_amendment_write.message,
            )
            blocked_dig_contract_rewrite = validate_tool_use(
                "Write",
                {
                    "file_path": str(
                        run_dir
                        / "gen_0"
                        / "peers"
                        / "gen0_peer0"
                        / "dig"
                        / "selected_contract.yaml"
                    )
                },
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_dig_contract_rewrite.allowed)
            blocked_other_peer_dig_amendment_write = validate_tool_use(
                "Write",
                {
                    "file_path": str(
                        run_dir
                        / "gen_0"
                        / "peers"
                        / "gen0_peer1"
                        / "dig"
                        / "contract_amendment.yaml"
                    )
                },
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_other_peer_dig_amendment_write.allowed)
            memory_dir = run_dir / "gen_0" / "peers" / "gen0_peer0" / "memory"
            memory_dir.mkdir(parents=True)
            allowed_memory_write = validate_tool_use(
                "Write",
                {"file_path": str(memory_dir / "peer_state.yaml")},
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed_memory_write.allowed, allowed_memory_write.message)
            blocked_memory_non_contract_file = validate_tool_use(
                "Write",
                {"file_path": str(memory_dir / "notes.txt")},
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_memory_non_contract_file.allowed)
            other_memory_dir = run_dir / "gen_0" / "peers" / "gen0_peer1" / "memory"
            other_memory_dir.mkdir(parents=True)
            blocked_other_peer_memory_write = validate_tool_use(
                "Write",
                {"file_path": str(other_memory_dir / "peer_state.yaml")},
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_other_peer_memory_write.allowed)
            blocked_unowned_finding_write = validate_tool_use(
                "Write",
                {"file_path": str(run_dir / "shared_findings" / "alpha.json")},
                env=env,
                cwd=run_dir,
            )
            self.assertFalse(blocked_unowned_finding_write.allowed)
            allowed = validate_tool_use(
                "Write",
                {"file_path": "$PRAXIST_PEER_WORKSPACE/tmp/x.txt"},
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(allowed.allowed, allowed.message)

    def test_pretool_blocks_shell_side_effects_and_runtime_blocks_python_side_effects(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
            validate_bash_command,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            protected = run_dir / "shared_findings"
            task_root = Path(tmp) / "task"
            protected.mkdir(parents=True)
            task_root.mkdir()
            (protected / "keep.json").write_text("{}", encoding="utf-8")
            protected_task_file = task_root / "description.md"
            protected_task_file.write_text("canonical task\n", encoding="utf-8")
            guarded = prepare_delete_guard_env(
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PEER_ID": "gen0_peer3",
                    "PATH": "/bin:/usr/bin",
                },
                workspace=Path(tmp),
                agent_name="agent",
            )

            blocked = validate_bash_command(
                f"rm -rf {protected}",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked.allowed)
            self.assertTrue(protected.exists())

            scratch_file = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "delete_me"
            scratch_file.parent.mkdir(parents=True, exist_ok=True)
            scratch_file.write_text("x", encoding="utf-8")
            allowed = subprocess.run(
                ["bash", "-lc", f"rm -f {scratch_file}"],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            self.assertFalse(scratch_file.exists())

            unsafe_script = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "unsafe.sh"
            unsafe_script.write_text(
                f"#!/usr/bin/env bash\n/bin/rm -rf {protected}\n", encoding="utf-8"
            )
            blocked_script = validate_bash_command(
                f"bash {unsafe_script}",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_script.allowed)
            self.assertTrue(protected.exists())

            trap_escape_script = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "trap_escape.sh"
            trap_escape_script.write_text(
                f"#!/usr/bin/env bash\n/bin/rm -rf {protected}\n",
                encoding="utf-8",
            )
            blocked_trap_script = validate_bash_command(
                f"bash {trap_escape_script}",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_trap_script.allowed)
            self.assertTrue(protected.exists())

            sh_script = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "unsafe_sh.sh"
            sh_script.write_text("echo unsafe shell\n", encoding="utf-8")
            warned_sh_script = validate_bash_command(
                f"sh {sh_script}",
                env=guarded,
                cwd=run_dir,
            )
            self.assertTrue(warned_sh_script.allowed, warned_sh_script.message)
            self.assertEqual(warned_sh_script.severity, "warning")

            protected_python = protected / "python_delete"
            protected_python.mkdir()
            blocked_python_rmtree = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import shutil,sys; shutil.rmtree(sys.argv[1])",
                    str(protected_python),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_rmtree.returncode, 0)
            self.assertTrue(protected_python.exists())

            blocked_python_subprocess = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; subprocess.run(['/bin/rm','-rf',sys.argv[1]])",
                    str(protected_python),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_subprocess.returncode, 0)
            self.assertTrue(protected_python.exists())

            source_for_cp = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "cp_source.txt"
            source_for_cp.write_text("x", encoding="utf-8")
            subprocess_mutating_cases = [
                (
                    "touch",
                    "import subprocess,sys; subprocess.run(['/bin/touch', sys.argv[1]])",
                    [str(protected / "touch_subprocess.json")],
                    protected / "touch_subprocess.json",
                ),
                (
                    "mkdir",
                    "import subprocess,sys; subprocess.run(['/bin/mkdir', sys.argv[1]])",
                    [str(protected / "mkdir_subprocess")],
                    protected / "mkdir_subprocess",
                ),
                (
                    "cp",
                    "import subprocess,sys; subprocess.run(['/bin/cp', sys.argv[1], sys.argv[2]])",
                    [str(source_for_cp), str(protected / "cp_subprocess.json")],
                    protected / "cp_subprocess.json",
                ),
            ]
            for name, code, args_tail, target in subprocess_mutating_cases:
                blocked_subprocess_case = subprocess.run(
                    [sys.executable, "-c", code, *args_tail],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_subprocess_case.returncode, 0, name)
                self.assertFalse(target.exists())

            makefile_subprocess = (
                Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "Makefile.subprocess"
            )
            makefile_subprocess.write_text(
                f"all:\n\t/bin/touch {protected / 'make_subprocess.json'}\n", encoding="utf-8"
            )
            blocked_subprocess_make = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; subprocess.run(['/usr/bin/make','-f',sys.argv[1]])",
                    str(makefile_subprocess),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_subprocess_make.returncode, 0)
            self.assertFalse((protected / "make_subprocess.json").exists())

            if Path("/usr/bin/make").exists():
                build_dir = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "build"
                build_dir.mkdir(parents=True, exist_ok=True)
                (build_dir / "Makefile").write_text("all:\n\ttrue\n", encoding="utf-8")
                allowed_subprocess_make = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import subprocess,sys; raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'all']).returncode)",
                        str(build_dir),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    allowed_subprocess_make.returncode,
                    0,
                    allowed_subprocess_make.stderr + allowed_subprocess_make.stdout,
                )

                protected_make_target = protected / "make_variable_evasion.json"
                protected_make_target.write_text("keep", encoding="utf-8")
                evasive_build_dir = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "evasive_build"
                evasive_build_dir.mkdir(parents=True, exist_ok=True)
                (evasive_build_dir / "Makefile").write_text(
                    'R=/bin/r\nM=m\nall:\n\t$(R)$(M) -f "$$P"\n',
                    encoding="utf-8",
                )
                blocked_evasive_make = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os,subprocess,sys; "
                            "env=os.environ.copy(); env['P']=sys.argv[2]; "
                            "raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'all'], env=env).returncode)"
                        ),
                        str(evasive_build_dir),
                        str(protected_make_target),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_evasive_make.returncode, 0)
                self.assertTrue(protected_make_target.exists())

                protected_shell_recipe_target = protected / "make_shell_script_evasion.json"
                protected_shell_recipe_target.write_text("keep", encoding="utf-8")
                shell_recipe_dir = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "shell_recipe_build"
                shell_recipe_dir.mkdir(parents=True, exist_ok=True)
                (shell_recipe_dir / "cleanup.sh").write_text(
                    f"#!/bin/sh\n/bin/rm -f {protected_shell_recipe_target}\n",
                    encoding="utf-8",
                )
                (shell_recipe_dir / "Makefile").write_text(
                    "all:\n\t/bin/sh cleanup.sh\n",
                    encoding="utf-8",
                )
                blocked_shell_recipe_make = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import subprocess,sys; "
                            "raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'all']).returncode)"
                        ),
                        str(shell_recipe_dir),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_shell_recipe_make.returncode, 0)
                self.assertTrue(protected_shell_recipe_target.exists())

            task_makefile = task_root / "Makefile"
            task_makefile.write_text("all:\n\tprintf ok\n", encoding="utf-8")
            allowed_task_make = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'all']).returncode)",
                    str(task_root),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                allowed_task_make.returncode, 0, allowed_task_make.stderr + allowed_task_make.stdout
            )
            task_build_dir_for_make = task_root / "make_build"
            task_build_dir_for_make.mkdir(parents=True)
            (task_build_dir_for_make / "Custom.mk").write_text(
                "all:\n\tprintf ok\n",
                encoding="utf-8",
            )
            allowed_task_make_custom = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys; "
                        "raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'-f','Custom.mk','all']).returncode)"
                    ),
                    str(task_build_dir_for_make),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                allowed_task_make_custom.returncode,
                0,
                allowed_task_make_custom.stderr + allowed_task_make_custom.stdout,
            )
            (task_build_dir_for_make / "Makefile").write_text(
                f"all:\n\t/bin/touch {protected / 'make_default_should_not_run.json'}\n",
                encoding="utf-8",
            )
            allowed_task_make_multi_c_custom = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys; "
                        "raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'-C','make_build','-f','Custom.mk','all']).returncode)"
                    ),
                    str(task_root),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                allowed_task_make_multi_c_custom.returncode,
                0,
                allowed_task_make_multi_c_custom.stderr + allowed_task_make_multi_c_custom.stdout,
            )
            self.assertFalse((protected / "make_default_should_not_run.json").exists())

            (task_root / "common.mk").write_text("all:\n\tprintf ok\n", encoding="utf-8")
            task_makefile.write_text("include\tcommon.mk\n", encoding="utf-8")
            allowed_task_make_include = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'all']).returncode)",
                    str(task_root),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                allowed_task_make_include.returncode,
                0,
                allowed_task_make_include.stderr + allowed_task_make_include.stdout,
            )

            (task_root / "evil.mk").write_text(
                f"all:\n\tprintf bad > {protected / 'make_include_evil.json'}\n",
                encoding="utf-8",
            )
            task_makefile.write_text("include\tevil.mk\n", encoding="utf-8")
            blocked_task_make_include = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'all']).returncode)",
                    str(task_root),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_task_make_include.returncode, 0)
            self.assertFalse((protected / "make_include_evil.json").exists())

            task_build_dir = task_root / "ninja_build"
            task_build_dir.mkdir(parents=True)
            (task_build_dir / "build.ninja").write_text(
                "rule noop\n  command = printf ok\nbuild all: noop\n",
                encoding="utf-8",
            )
            ninja_path = "/usr/bin/ninja" if Path("/usr/bin/ninja").exists() else None
            if ninja_path:
                allowed_task_ninja = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import subprocess,sys; "
                            "raise SystemExit(subprocess.run([sys.argv[1],'-C',sys.argv[2],'all']).returncode)"
                        ),
                        ninja_path,
                        str(task_build_dir),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    allowed_task_ninja.returncode,
                    0,
                    allowed_task_ninja.stderr + allowed_task_ninja.stdout,
                )
                (task_build_dir / "safe.ninja").write_text(
                    "rule noop\n  command = printf ok\nbuild all: noop\n",
                    encoding="utf-8",
                )
                (task_build_dir / "custom.ninja").write_text(
                    "rule noop\n  command = printf ok\nbuild all: noop\n",
                    encoding="utf-8",
                )
                allowed_task_custom_ninja = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import subprocess,sys; "
                            "raise SystemExit(subprocess.run([sys.argv[1],'-C',sys.argv[2],'-f','custom.ninja','all']).returncode)"
                        ),
                        ninja_path,
                        str(task_build_dir),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    allowed_task_custom_ninja.returncode,
                    0,
                    allowed_task_custom_ninja.stderr + allowed_task_custom_ninja.stdout,
                )
                (task_build_dir / "build.ninja").write_text(
                    "subninja safe.ninja\n", encoding="utf-8"
                )
                allowed_task_subninja = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import subprocess,sys; "
                            "raise SystemExit(subprocess.run([sys.argv[1],'-C',sys.argv[2],'all']).returncode)"
                        ),
                        ninja_path,
                        str(task_build_dir),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    allowed_task_subninja.returncode,
                    0,
                    allowed_task_subninja.stderr + allowed_task_subninja.stdout,
                )
                (task_build_dir / "evil.ninja").write_text(
                    f"rule bad\n  command = printf bad > {protected / 'subninja_evil.json'}\nbuild all: bad\n",
                    encoding="utf-8",
                )
                (task_build_dir / "build.ninja").write_text(
                    "subninja evil.ninja\n", encoding="utf-8"
                )
                blocked_task_subninja = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import subprocess,sys; "
                            "raise SystemExit(subprocess.run([sys.argv[1],'-C',sys.argv[2],'all']).returncode)"
                        ),
                        ninja_path,
                        str(task_build_dir),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_task_subninja.returncode, 0)
                self.assertFalse((protected / "subninja_evil.json").exists())

            task_makefile.write_text(
                f"all:\n\tprintf bad > {protected / 'make_task_evil.json'}\n",
                encoding="utf-8",
            )
            blocked_task_make = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'all']).returncode)",
                    str(task_root),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_task_make.returncode, 0)
            self.assertFalse((protected / "make_task_evil.json").exists())

            protected_task_file.write_text("canonical task\n", encoding="utf-8")
            task_makefile.write_text("all:\n\tprintf bad > description.md\n", encoding="utf-8")
            blocked_task_description_make = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; raise SystemExit(subprocess.run(['/usr/bin/make','-C',sys.argv[1],'all']).returncode)",
                    str(task_root),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_task_description_make.returncode, 0)
            self.assertEqual(protected_task_file.read_text(encoding="utf-8"), "canonical task\n")

            blocked_subprocess_shell_write = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; subprocess.run('/bin/touch ' + sys.argv[1], shell=True)",
                    str(protected / "shell_subprocess.json"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_subprocess_shell_write.returncode, 0)
            self.assertFalse((protected / "shell_subprocess.json").exists())

            blocked_subprocess_tar_checkpoint = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys; "
                        "subprocess.run(['/bin/tar','--checkpoint=1',"
                        "'--checkpoint-action=exec=/bin/touch ' + sys.argv[1],"
                        "'-cf','/dev/null',sys.argv[2]])"
                    ),
                    str(protected / "tar_checkpoint_subprocess.json"),
                    str(source_for_cp),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_subprocess_tar_checkpoint.returncode, 0)
            self.assertFalse((protected / "tar_checkpoint_subprocess.json").exists())

            blocked_subprocess_cwd_escape = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,subprocess,sys; "
                        "os.chdir(os.environ['PRAXIST_PEER_WORKSPACE']); "
                        "subprocess.run(['/bin/rm','-rf','shared_findings'], cwd=sys.argv[1])"
                    ),
                    str(run_dir),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_subprocess_cwd_escape.returncode, 0)
            self.assertTrue(protected.exists())

            shell_rm_script = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "shell_rm.py"
            shell_rm_script.write_text(
                "import subprocess,sys\nsubprocess.run(['/bin/sh','-c','/bin/rm -f ' + sys.argv[1]])\n",
                encoding="utf-8",
            )
            protected_file = protected / "keep.json"
            blocked_python_shell_rm = subprocess.run(
                [sys.executable, str(shell_rm_script), str(protected_file)],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_shell_rm.returncode, 0)
            self.assertTrue(protected_file.exists())

            protected_move = protected / "move_me.txt"
            protected_move.write_text("x", encoding="utf-8")
            blocked_python_move = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import shutil,sys; shutil.move(sys.argv[1], sys.argv[2])",
                    str(protected_move),
                    str(Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "moved.txt"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_move.returncode, 0)
            self.assertTrue(protected_move.exists())

            protected_rename = protected / "rename_me.txt"
            protected_rename.write_text("x", encoding="utf-8")
            blocked_python_rename = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; os.rename(sys.argv[1], sys.argv[2])",
                    str(protected_rename),
                    str(Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "renamed.txt"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_rename.returncode, 0)
            self.assertTrue(protected_rename.exists())

            protected_posix = protected / "posix_unlink_me.txt"
            protected_posix.write_text("x", encoding="utf-8")
            blocked_python_posix = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import posix,sys; posix.unlink(sys.argv[1])",
                    str(protected_posix),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_posix.returncode, 0)
            self.assertTrue(protected_posix.exists())

            safe_python_dir = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "safe_python"
            safe_python_dir.mkdir(parents=True)
            allowed_python_rmtree = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import shutil,sys; shutil.rmtree(sys.argv[1])",
                    str(safe_python_dir),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(allowed_python_rmtree.returncode, 0, allowed_python_rmtree.stderr)
            self.assertFalse(safe_python_dir.exists())

            protected_write = protected / "write_me.txt"
            protected_write.write_text("x", encoding="utf-8")
            blocked_python_write = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('')",
                    str(protected_write),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_write.returncode, 0)
            self.assertEqual(protected_write.read_text(encoding="utf-8"), "x")

            import_standard_library = subprocess.run(
                [sys.executable, "-c", "import json; print(json.dumps({'guard': 'ok'}))"],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(import_standard_library.returncode, 0, import_standard_library.stderr)

            peer_variant_file = run_dir / "variants" / "gen0_peer3_alpha" / "train.py"
            peer_variant_file.parent.mkdir(parents=True)
            allowed_variant_create = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
                    str(peer_variant_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(allowed_variant_create.returncode, 0, allowed_variant_create.stderr)
            self.assertEqual(peer_variant_file.read_text(encoding="utf-8"), "ok")

            peer_dig_amendment = (
                run_dir / "gen_0" / "peers" / "gen0_peer3" / "dig" / "contract_amendment.yaml"
            )
            peer_dig_amendment.parent.mkdir(parents=True)
            allowed_dig_amendment_create = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
                    str(peer_dig_amendment),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                allowed_dig_amendment_create.returncode,
                0,
                allowed_dig_amendment_create.stderr,
            )
            self.assertEqual(peer_dig_amendment.read_text(encoding="utf-8"), "ok")

            peer_selected_contract = (
                run_dir / "gen_0" / "peers" / "gen0_peer3" / "dig" / "selected_contract.yaml"
            )
            blocked_selected_contract_rewrite = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('bad')",
                    str(peer_selected_contract),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_selected_contract_rewrite.returncode, 0)
            self.assertFalse(peer_selected_contract.exists())

            other_peer_dig_amendment = (
                run_dir / "gen_0" / "peers" / "gen0_peer9" / "dig" / "contract_amendment.yaml"
            )
            other_peer_dig_amendment.parent.mkdir(parents=True)
            blocked_other_peer_dig_amendment = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('bad')",
                    str(other_peer_dig_amendment),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_other_peer_dig_amendment.returncode, 0)
            self.assertFalse(other_peer_dig_amendment.exists())

            other_variant_file = run_dir / "variants" / "gen0_peer9_alpha" / "train.py"
            other_variant_file.parent.mkdir(parents=True)
            blocked_other_variant_create = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('bad')",
                    str(other_variant_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_other_variant_create.returncode, 0)
            self.assertFalse(other_variant_file.exists())

            owned_finding_tmp = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "finding.tmp"
            owned_finding_tmp.write_text("{}", encoding="utf-8")
            owned_finding = run_dir / "shared_findings" / "gen0_peer3_atomic.json"
            allowed_atomic_replace = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; os.replace(sys.argv[1], sys.argv[2])",
                    str(owned_finding_tmp),
                    str(owned_finding),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(allowed_atomic_replace.returncode, 0, allowed_atomic_replace.stderr)
            self.assertTrue(owned_finding.exists())

            atomic_publish_cases = [
                (
                    "shutil_move",
                    "import shutil,sys; shutil.move(sys.argv[1], sys.argv[2])",
                    "gen0_peer3_move.json",
                ),
                (
                    "os_rename",
                    "import os,sys; os.rename(sys.argv[1], sys.argv[2])",
                    "gen0_peer3_rename.json",
                ),
                (
                    "path_rename",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).rename(sys.argv[2])",
                    "gen0_peer3_path_rename.json",
                ),
            ]
            for case_name, code, filename in atomic_publish_cases:
                src_tmp = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / f"{case_name}.tmp"
                src_tmp.write_text("{}", encoding="utf-8")
                dst = run_dir / "shared_findings" / filename
                allowed_atomic_publish = subprocess.run(
                    [sys.executable, "-c", code, str(src_tmp), str(dst)],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    allowed_atomic_publish.returncode,
                    0,
                    case_name + allowed_atomic_publish.stderr + allowed_atomic_publish.stdout,
                )
                self.assertTrue(dst.exists(), case_name)

            blocked_atomic_move_cases = [
                (
                    "other_peer",
                    run_dir / "shared_findings" / "gen0_peer9_move.json",
                ),
                (
                    "frontier",
                    run_dir / "frontier" / "frontier.jsonl",
                ),
                (
                    "gems",
                    run_dir / "gems" / "gems_state.json",
                ),
            ]
            for case_name, dst in blocked_atomic_move_cases:
                src_tmp = (
                    Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / f"blocked_{case_name}.tmp"
                )
                src_tmp.write_text("{}", encoding="utf-8")
                blocked_atomic_publish = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import shutil,sys; shutil.move(sys.argv[1], sys.argv[2])",
                        str(src_tmp),
                        str(dst),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_atomic_publish.returncode, 0, case_name)
                self.assertFalse(dst.exists(), case_name)

            for case_name, target_arg in (
                ("space_t", ["-t", str(task_root)]),
                ("attached_t", [f"-t{task_root}"]),
                ("long_target", [f"--target-directory={task_root}"]),
            ):
                src_tmp = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / f"mv_t_{case_name}.tmp"
                src_tmp.write_text("payload", encoding="utf-8")
                blocked_mv_t = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import subprocess,sys; "
                            "raise SystemExit(subprocess.run(['/bin/mv',*sys.argv[1:-1],sys.argv[-1]]).returncode)"
                        ),
                        *target_arg,
                        str(src_tmp),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_mv_t.returncode, 0, case_name)
                self.assertFalse((task_root / src_tmp.name).exists(), case_name)

            allowed_existing_protected_mkdir = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)",
                    str(protected),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                allowed_existing_protected_mkdir.returncode,
                0,
                allowed_existing_protected_mkdir.stderr,
            )

            blocked_io_open = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import io,sys; io.open(sys.argv[1], 'w').write('bad')",
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_io_open.returncode, 0)
            self.assertEqual(protected_file.read_text(encoding="utf-8"), "{}")

            blocked__io_open = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import _io,sys; _io.open(sys.argv[1], 'w').write('bad')",
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked__io_open.returncode, 0)
            self.assertEqual(protected_file.read_text(encoding="utf-8"), "{}")

            shared_db = run_dir / "shared_store.db"
            blocked_sqlite = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sqlite3,sys; "
                        "conn=sqlite3.connect(sys.argv[1]); "
                        "conn.execute('create table if not exists findings(id text)'); "
                        "conn.execute('delete from findings'); conn.commit()"
                    ),
                    str(shared_db),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_sqlite.returncode, 0)
            self.assertFalse(shared_db.exists())

            blocked_sqlite_fake_ro = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sqlite3,sys; "
                        "conn=sqlite3.connect('file:' + sys.argv[1] + '?mode=rwc&x=mode=ro', uri=True); "
                        "conn.execute('create table findings(id text)'); conn.commit()"
                    ),
                    str(shared_db),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_sqlite_fake_ro.returncode, 0)
            self.assertFalse(shared_db.exists())

            blocked_ctypes_unlink = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "m=''.join(map(chr,[99,116,121,112,101,115])); "
                        "c=getattr(__import__(m), ''.join(map(chr,[67,68,76,76]))); "
                        "u=''.join(map(chr,[117,110,108,105,110,107])); "
                        "getattr(c(None), u)(sys.argv[1].encode())"
                    ),
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_ctypes_unlink.returncode, 0)
            self.assertTrue(protected_file.exists())

            blocked_libdl_ctypes = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    ("import ctypes; ctypes.CDLL('libdl.so.2')"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_libdl_ctypes.returncode, 0)

            peer_local_so = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "local.so"
            peer_local_so.write_bytes(b"\x7fELFnotreally")
            blocked_peer_ctypes = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import ctypes,sys; ctypes.CDLL(sys.argv[1])",
                    str(peer_local_so),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_peer_ctypes.returncode, 0)

            blocked_pydll_libc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import ctypes; ctypes.PyDLL('lib' + 'c.so.6')",
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_pydll_libc.returncode, 0)

            blocked_pydll_loadlibrary = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import ctypes; ctypes.pydll.LoadLibrary('lib' + 'c.so.6')",
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_pydll_loadlibrary.returncode, 0)

            blocked_cffi_import = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "__import__('c' + 'ffi')",
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_cffi_import.returncode, 0)

            import importlib.machinery

            ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
            peer_extension = (
                Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / f"evil_ext{ext_suffix}"
            )
            peer_extension.write_bytes(b"\x7fELFnotreally")
            ext_guarded = dict(guarded)
            ext_guarded["PYTHONPATH"] = (
                f"{peer_extension.parent}{os.pathsep}{ext_guarded['PYTHONPATH']}"
            )
            blocked_extension_import = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import evil_ext",
                ],
                env=ext_guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_extension_import.returncode, 0)
            self.assertIn(
                "delete guard",
                (blocked_extension_import.stderr + blocked_extension_import.stdout).lower(),
            )

            blocked_env_strip_child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,subprocess,sys; "
                        "os.environ['PYTHONPATH']=''; "
                        "subprocess.run([sys.executable,'-c',"
                        "'open(sys.argv[1],\\'w\\').write(\\'bad\\')',sys.argv[1]])"
                    ),
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_env_strip_child.returncode, 0)
            self.assertEqual(protected_file.read_text(encoding="utf-8"), "{}")

            other_variant_file = run_dir / "variants" / "gen0_peer9_alpha" / "train.py"
            other_variant_file.parent.mkdir(parents=True, exist_ok=True)
            blocked_agent_spoof = subprocess.run(
                [
                    "bash",
                    "-lc",
                    (
                        "PRAXIST_DELETE_GUARD_AGENT=gen0_peer9 "
                        f"{sys.executable} -c \"open('{other_variant_file}', 'w').write('bad')\""
                    ),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_agent_spoof.returncode, 0)
            self.assertFalse(other_variant_file.exists())

            blocked_subprocess_agent_spoof = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,subprocess,sys; env=os.environ.copy(); "
                        "env['PRAXIST_DELETE_GUARD_AGENT']='gen0_peer9'; "
                        "subprocess.run([sys.executable,'-c',"
                        "'open(sys.argv[1],\\'w\\').write(\\'bad\\')',sys.argv[1]], env=env)"
                    ),
                    str(other_variant_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_subprocess_agent_spoof.returncode, 0)
            self.assertFalse(other_variant_file.exists())

            blocked_subprocess_root_broaden = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,subprocess,sys; env=os.environ.copy(); "
                        "env['PRAXIST_SAFE_DELETE_ROOTS']=env['PRAXIST_SAFE_DELETE_ROOTS'] + ':' + sys.argv[1]; "
                        "subprocess.run([sys.executable,'-c',"
                        "'open(sys.argv[1],\\'w\\').write(\\'bad\\')',sys.argv[2]], env=env)"
                    ),
                    str(protected),
                    str(protected / "broadened.json"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_subprocess_root_broaden.returncode, 0)
            self.assertFalse((protected / "broadened.json").exists())

            blocked_ld_preload_assignment = validate_bash_command(
                f"LD_PRELOAD=$PRAXIST_PEER_WORKSPACE/tmp/fake.so {sys.executable} -c 'print(1)'",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_ld_preload_assignment.allowed)
            self.assertEqual(blocked_ld_preload_assignment.rule_id, "loader_injection_workload")

            low_level_cases = [
                (
                    "os_truncate",
                    "import os,sys; os.truncate(sys.argv[1], 0)",
                    protected_file,
                ),
                (
                    "os_open_truncate",
                    "import os,sys; os.open(sys.argv[1], os.O_WRONLY | os.O_TRUNC)",
                    protected_file,
                ),
                (
                    "os_chmod",
                    "import os,sys; os.chmod(sys.argv[1], 0o000)",
                    protected_file,
                ),
                (
                    "path_chmod",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).chmod(0o000)",
                    protected_file,
                ),
                (
                    "shutil_copyfile",
                    "import shutil,sys; shutil.copyfile(sys.argv[1], sys.argv[2])",
                    protected_task_file,
                ),
            ]
            source_file = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "source.txt"
            source_file.write_text("source", encoding="utf-8")
            for name, code, target in low_level_cases:
                args = [sys.executable, "-c", code]
                if name == "shutil_copyfile":
                    args.extend([str(source_file), str(target)])
                else:
                    args.append(str(target))
                blocked_case = subprocess.run(
                    args,
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_case.returncode, 0, name)

            dir_fd = os.open(str(protected), os.O_RDONLY)
            try:
                blocked_dirfd_unlink = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os,sys; "
                            "fd=os.open(sys.argv[1], os.O_RDONLY); "
                            "os.unlink('keep.json', dir_fd=fd)"
                        ),
                        str(protected),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            finally:
                os.close(dir_fd)
            self.assertNotEqual(blocked_dirfd_unlink.returncode, 0)
            self.assertTrue(protected_file.exists())

            blocked_fchmod = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); os.fchmod(fd, 0)",
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_fchmod.returncode, 0)

            protected_link = protected / "new_link"
            blocked_symlink = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; os.symlink(sys.argv[1], sys.argv[2])",
                    str(source_file),
                    str(protected_link),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_symlink.returncode, 0)
            self.assertFalse(protected_link.exists())

            hardlink_path = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "hard"
            blocked_hardlink = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; os.link(sys.argv[1], sys.argv[2])",
                    str(protected_file),
                    str(hardlink_path),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_hardlink.returncode, 0)
            self.assertFalse(hardlink_path.exists())

            blocked_orig_escape = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "m=next(v for k,v in sys.modules.items() if k.endswith('customize')); "
                        "vars(m)['_'+'orig'+'_'+'un'+'link'](sys.argv[1])"
                    ),
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_orig_escape.returncode, 0)
            self.assertTrue(protected_file.exists())

            script_orig_escape = (
                Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "kwdefaults_escape.py"
            )
            script_orig_escape.write_text(
                "import os,sys\ngetattr(os, 'un'+'link').__kwdefaults__['_orig'](sys.argv[1])\n",
                encoding="utf-8",
            )
            blocked_script_orig_escape = subprocess.run(
                ["bash", "-lc", f"{sys.executable} {script_orig_escape} {protected_file}"],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_script_orig_escape.returncode, 0)
            self.assertTrue(protected_file.exists())

            blocked_empty_env_subprocess = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys; "
                        "subprocess.run([sys.executable,'-c',"
                        "'import os,sys; os.unlink(sys.argv[1])',sys.argv[1]], env={})"
                    ),
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_empty_env_subprocess.returncode, 0)
            self.assertTrue(protected_file.exists())

            blocked_env_i_rm_subprocess = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys; "
                        "subprocess.run(['/usr/bin/env','-i','PATH=/bin:/usr/bin',"
                        "'/bin/rm','-f',sys.argv[1]])"
                    ),
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_env_i_rm_subprocess.returncode, 0)
            self.assertTrue(protected_file.exists())

            blocked_python_no_site_subprocess = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys; "
                        "subprocess.run([sys.executable,'-S','-c',"
                        "'import os,sys; os.unlink(sys.argv[1])',sys.argv[1]])"
                    ),
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_no_site_subprocess.returncode, 0)
            self.assertTrue(protected_file.exists())

            allowed_trusted_pythonpath_prepend = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,subprocess,sys; env=os.environ.copy(); "
                        "env['PYTHONPATH']=sys.argv[1] + os.pathsep + env['PYTHONPATH']; "
                        "raise SystemExit(subprocess.run([sys.executable,'-c','print(1)'], env=env).returncode)"
                    ),
                    str(task_root / "assets" / "harness" / "baseline"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                allowed_trusted_pythonpath_prepend.returncode,
                0,
                allowed_trusted_pythonpath_prepend.stderr,
            )

            shadow_dir = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "shadow_py"
            shadow_dir.mkdir(parents=True)
            (shadow_dir / "sitecustomize.py").write_text("print('shadow')\n", encoding="utf-8")
            warned_peer_pythonpath_shadow = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,subprocess,sys; env=os.environ.copy(); "
                        "env['PYTHONPATH']=sys.argv[1] + os.pathsep + env['PYTHONPATH']; "
                        "subprocess.run([sys.executable,'-c','print(1)'], env=env)"
                    ),
                    str(shadow_dir),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                warned_peer_pythonpath_shadow.returncode,
                0,
                warned_peer_pythonpath_shadow.stderr + warned_peer_pythonpath_shadow.stdout,
            )
            self.assertIn("shadow", warned_peer_pythonpath_shadow.stdout)

            blocked_os_system_unlink = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,shlex,sys; os.system('/bin/unlink ' + shlex.quote(sys.argv[1]))",
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_os_system_unlink.returncode, 0)
            self.assertTrue(protected_file.exists())

            blocked_os_system_redirection = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; os.system(': > ' + sys.argv[1])",
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_os_system_redirection.returncode, 0)
            self.assertTrue(protected_file.exists())

            blocked_spawn_unlink = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,sys; "
                        "os.spawnv(os.P_WAIT, '/bin/unlink', ['/bin/unlink', sys.argv[1]])"
                    ),
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_spawn_unlink.returncode, 0)
            self.assertTrue(protected_file.exists())

            blocked_shell_obfuscated_rm = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; subprocess.run('/bin/r${x}m -f ' + sys.argv[1], shell=True)",
                    str(protected_file),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_shell_obfuscated_rm.returncode, 0)
            self.assertTrue(protected_file.exists())

            direct_python_script = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "direct_py"
            direct_python_script.write_text(
                f"#!{sys.executable}\nimport os,sys\nos.truncate(sys.argv[1], 0)\n",
                encoding="utf-8",
            )
            direct_python_script.chmod(0o755)
            blocked_direct_python = subprocess.run(
                ["bash", "-lc", f"{direct_python_script} {protected_file}"],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_direct_python.returncode, 0)
            self.assertEqual(protected_file.read_text(encoding="utf-8"), "{}")

            direct_sh_script = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "direct_sh"
            direct_sh_script.write_text(
                f"#!/bin/sh\n/bin/rm -f {protected_file}\n",
                encoding="utf-8",
            )
            direct_sh_script.chmod(0o755)
            blocked_direct_sh = validate_bash_command(
                str(direct_sh_script),
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_direct_sh.allowed)
            self.assertTrue(protected_file.exists())

            path_sh_script = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "path_sh"
            path_sh_script.write_text(
                f"#!/bin/sh\n/bin/rm -f {protected_file}\n",
                encoding="utf-8",
            )
            path_sh_script.chmod(0o755)
            path_guarded = dict(guarded)
            path_guarded["PATH"] = f"{path_sh_script.parent}{os.pathsep}{path_guarded['PATH']}"
            blocked_path_script = validate_bash_command(
                "path_sh",
                env=path_guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_path_script.allowed)
            self.assertTrue(protected_file.exists())

            external_script = Path(tmp) / "evil.sh"
            external_script.write_text(
                f"#!/bin/sh\n/bin/rm -f {protected_file}\n", encoding="utf-8"
            )
            external_script.chmod(0o755)
            blocked_external_script = validate_bash_command(
                str(external_script),
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_external_script.allowed)
            self.assertTrue(protected_file.exists())

            fake_binary = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "fake_elf"
            fake_binary.write_bytes(b"\x7fELFnotreally")
            fake_binary.chmod(0o755)
            blocked_exec_binary = validate_bash_command(
                f"exec {fake_binary} {protected_file}",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_exec_binary.allowed)
            self.assertIn("delete guard", blocked_exec_binary.message.lower())
            self.assertTrue(protected_file.exists())

            blocked_python_mkdir = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; getattr(os, 'mk'+'dir')(sys.argv[1])",
                    str(protected / "evil_py_dir"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_mkdir.returncode, 0)
            self.assertFalse((protected / "evil_py_dir").exists())

            blocked_python_makedirs = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; getattr(os, 'make'+'dirs')(sys.argv[1])",
                    str(protected / "evil_nested" / "x"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_makedirs.returncode, 0)
            self.assertFalse((protected / "evil_nested").exists())

            blocked_path_mkdir = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).mkdir()",
                    str(protected / "evil_path_dir"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_path_mkdir.returncode, 0)
            self.assertFalse((protected / "evil_path_dir").exists())

            blocked_mkfifo = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; getattr(os, 'mk'+'fifo')(sys.argv[1])",
                    str(protected / "evil_fifo"),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_mkfifo.returncode, 0)
            self.assertFalse((protected / "evil_fifo").exists())

            safe_runtime_mkdir = (
                Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "safe_runtime_dir"
            )
            allowed_python_mkdir = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; os.makedirs(sys.argv[1])",
                    str(safe_runtime_mkdir),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(allowed_python_mkdir.returncode, 0, allowed_python_mkdir.stderr)
            self.assertTrue(safe_runtime_mkdir.exists())

            find_delete_file = protected / "find_delete_me.txt"
            find_delete_file.write_text("x", encoding="utf-8")
            blocked_python_find_delete = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys; "
                        "subprocess.run(['/usr/bin/find', sys.argv[1], '-type', 'f', '-delete'])"
                    ),
                    str(protected),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_python_find_delete.returncode, 0)
            self.assertTrue(find_delete_file.exists())

            safe_find_dir = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "safe_find"
            safe_find_dir.mkdir(parents=True)
            safe_find_file = safe_find_dir / "delete_me.txt"
            safe_find_file.write_text("x", encoding="utf-8")
            allowed_python_find_delete = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import subprocess,sys; "
                        "raise SystemExit(subprocess.run(['/usr/bin/find', sys.argv[1], '-type', 'f', '-delete']).returncode)"
                    ),
                    str(safe_find_dir),
                ],
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                allowed_python_find_delete.returncode, 0, allowed_python_find_delete.stderr
            )
            self.assertFalse(safe_find_file.exists())

            if importlib.util.find_spec("numpy") is not None:
                blocked_numpy_tofile = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import numpy as np,sys; np.array([1], dtype=np.int8).tofile(sys.argv[1])",
                        str(protected_file),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_numpy_tofile.returncode, 0)
                self.assertTrue(protected_file.exists())

                blocked_numpy_save = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import numpy as np,sys; np.save(sys.argv[1], np.array([1]))",
                        str(protected / "np_save"),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_numpy_save.returncode, 0)
                self.assertFalse((protected / "np_save.npy").exists())

                safe_numpy_path = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "safe.npy"
                allowed_numpy_save = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import numpy as np,sys; np.save(sys.argv[1], np.array([1]))",
                        str(safe_numpy_path),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(allowed_numpy_save.returncode, 0, allowed_numpy_save.stderr)
                self.assertTrue(safe_numpy_path.exists())

            if (
                importlib.util.find_spec("pandas") is not None
                and importlib.util.find_spec("pyarrow") is not None
            ):
                blocked_pandas_parquet = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import pandas as pd,sys; pd.DataFrame({'x':[1]}).to_parquet(sys.argv[1])",
                        str(protected / "frame.parquet"),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_pandas_parquet.returncode, 0)
                self.assertFalse((protected / "frame.parquet").exists())

            if Path("/usr/bin/openssl").exists():
                blocked_subprocess_openssl = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import subprocess,sys; "
                            "subprocess.run(['/usr/bin/openssl','rand','-out',sys.argv[1],'16'])"
                        ),
                        str(protected / "openssl.bin"),
                    ],
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(blocked_subprocess_openssl.returncode, 0)
                self.assertFalse((protected / "openssl.bin").exists())

            makefile = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "Makefile"
            makefile.write_text(f"all:\n\t/bin/rm -f {protected_file}\n", encoding="utf-8")
            blocked_make_runtime = validate_bash_command(
                f"make -f {makefile}",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_make_runtime.allowed)
            self.assertTrue(protected_file.exists())

            tar_source = Path(guarded["PRAXIST_PEER_WORKSPACE"]) / "tmp" / "tar_source"
            tar_source.write_text("x", encoding="utf-8")
            blocked_tar_checkpoint_runtime = validate_bash_command(
                f"tar --checkpoint=1 --checkpoint-action=exec='/bin/rm -f {protected_file}' -cf /dev/null {tar_source}",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_tar_checkpoint_runtime.allowed)
            self.assertTrue(protected_file.exists())

            blocked_touch_runtime = validate_bash_command(
                f"touch {protected / 'evil.json'}",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_touch_runtime.allowed)
            self.assertFalse((protected / "evil.json").exists())

            eval_entrypoint = task_root / "evaluations" / "trading_pareto" / "run.py"
            eval_entrypoint.parent.mkdir(parents=True)
            eval_entrypoint.write_text(
                "import argparse\n"
                "parser = argparse.ArgumentParser(description='task-local evaluation')\n"
                "parser.parse_args()\n",
                encoding="utf-8",
            )
            run_py_help = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"{shlex.quote(sys.executable)} {shlex.quote(str(eval_entrypoint))} --help",
                ],
                env={**guarded, "PRAXIST_TASK_PROJECT_PATH": str(task_root)},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run_py_help.returncode, 0, run_py_help.stderr[-1000:])

            blocked_implicit_ln = validate_bash_command(
                f"cd {protected} && ln -s {source_file}",
                env=guarded,
                cwd=run_dir,
            )
            self.assertFalse(blocked_implicit_ln.allowed)
            self.assertFalse((protected / source_file.name).exists())

    def test_runtime_shell_script_operand_blocks_subprocess_protected_write(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            task_root = Path(tmp) / "task"
            task_root.mkdir()
            shared_store = run_dir / "shared_store.db"
            shared_store.write_text("ORIGINAL", encoding="utf-8")
            guarded = prepare_delete_guard_env(
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PEER_ID": "gen0_peer0",
                    "PATH": "/bin:/usr/bin",
                },
                workspace=Path(tmp),
                agent_name="gen0_peer0",
            )
            peer_workspace = Path(guarded["PRAXIST_PEER_WORKSPACE"])
            bad_script = peer_workspace / "bad_operand.sh"
            bad_script.write_text(f"echo CORRUPT > {shared_store}\n", encoding="utf-8")
            shell_operand_cases = [
                (
                    "bin_sh",
                    "import subprocess; subprocess.run(['/bin/sh','bad_operand.sh'], check=True)",
                ),
                (
                    "bin_bash",
                    "import subprocess; subprocess.run(['/bin/bash','bad_operand.sh'], check=True)",
                ),
                (
                    "path_sh",
                    "import subprocess; subprocess.run(['sh','bad_operand.sh'], check=True)",
                ),
            ]
            for case_name, case_code in shell_operand_cases:
                runner = peer_workspace / f"runner_shell_script_operand_{case_name}.py"
                runner.write_text(case_code + "\n", encoding="utf-8")
                blocked_shell_operand = subprocess.run(
                    [sys.executable, str(runner)],
                    cwd=peer_workspace,
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(
                    blocked_shell_operand.returncode,
                    0,
                    case_name + blocked_shell_operand.stderr + blocked_shell_operand.stdout,
                )
                self.assertEqual(shared_store.read_text(encoding="utf-8"), "ORIGINAL")

    def test_runtime_exec_shell_script_operand_blocks_protected_write(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            task_root = Path(tmp) / "task"
            task_root.mkdir()
            shared_store = run_dir / "shared_store.db"
            shared_store.write_text("ORIGINAL", encoding="utf-8")
            guarded = prepare_delete_guard_env(
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PEER_ID": "gen0_peer0",
                    "PATH": "/bin:/usr/bin",
                },
                workspace=Path(tmp),
                agent_name="gen0_peer0",
            )
            peer_workspace = Path(guarded["PRAXIST_PEER_WORKSPACE"])
            bad_script = peer_workspace / "bad.sh"
            bad_script.write_text(f"echo CORRUPT > {shared_store}\n", encoding="utf-8")
            execv_runner = peer_workspace / "runner_exec_shell_script_operand.py"
            execv_runner.write_text(
                "import os\nos.execv('/bin/sh', ['/bin/sh', 'bad.sh'])\n",
                encoding="utf-8",
            )
            blocked_execv = subprocess.run(
                [sys.executable, str(execv_runner)],
                cwd=peer_workspace,
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(
                blocked_execv.returncode, 0, blocked_execv.stderr + blocked_execv.stdout
            )
            self.assertEqual(shared_store.read_text(encoding="utf-8"), "ORIGINAL")

            if hasattr(os, "posix_spawn"):
                posix_runner = peer_workspace / "runner_posix_spawn_shell_script_operand.py"
                posix_runner.write_text(
                    (
                        "import os\n"
                        "pid = os.posix_spawn('/bin/sh', ['/bin/sh', 'bad.sh'], os.environ.copy())\n"
                        "_, status = os.waitpid(pid, 0)\n"
                        "raise SystemExit(os.waitstatus_to_exitcode(status))\n"
                    ),
                    encoding="utf-8",
                )
                blocked_posix_spawn = subprocess.run(
                    [sys.executable, str(posix_runner)],
                    cwd=peer_workspace,
                    env=guarded,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(
                    blocked_posix_spawn.returncode,
                    0,
                    blocked_posix_spawn.stderr + blocked_posix_spawn.stdout,
                )
                self.assertEqual(shared_store.read_text(encoding="utf-8"), "ORIGINAL")

    def test_runtime_shell_script_operand_allows_safe_peer_script(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk.delete_guard import (
            prepare_delete_guard_env,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True)
            task_root = Path(tmp) / "task"
            task_root.mkdir()
            guarded = prepare_delete_guard_env(
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PEER_ID": "gen0_peer0",
                    "PATH": "/bin:/usr/bin",
                },
                workspace=Path(tmp),
                agent_name="gen0_peer0",
            )
            peer_workspace = Path(guarded["PRAXIST_PEER_WORKSPACE"])
            safe_script = peer_workspace / "safe.sh"
            safe_script.write_text("echo SAFE\n", encoding="utf-8")
            allowed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import subprocess; raise SystemExit(subprocess.run(['/bin/sh','safe.sh']).returncode)",
                ],
                cwd=peer_workspace,
                env=guarded,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr + allowed.stdout)

    def test_parser_helpers_cover_write_target_variants(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import delete_guard as guard

        self.assertEqual(
            guard._source_destructive_path_args(["-rf", "--", "a", "b"]),
            ["a", "b"],
        )
        self.assertEqual(
            guard._source_destructive_path_args(["-t", "dst", "src"]),
            ["dst", "src"],
        )
        self.assertEqual(
            guard._source_destructive_path_args(["--target-directory=dst", "src"]),
            ["dst", "src"],
        )
        self.assertEqual(guard._mv_sources_destinations(["src", "dst"]), (["src"], ["dst"]))
        self.assertEqual(
            guard._mv_sources_destinations(["--target-directory=dst", "src"]),
            (["src"], ["dst"]),
        )
        self.assertEqual(guard._mv_sources_destinations(["src"]), (["src"], []))

        self.assertEqual(guard._cp_install_destinations(["-t", "dst", "src"]), ["dst"])
        self.assertEqual(guard._cp_install_destinations(["-tcovered", "src"]), ["covered"])
        self.assertEqual(guard._cp_install_destinations(["src", "dst"]), ["dst"])
        self.assertEqual(guard._cp_install_sources(["-t", "dst", "src"]), ["dst"])
        self.assertEqual(guard._cp_install_sources(["src", "dst"]), ["src"])
        self.assertTrue(guard._cp_segment_hardlinks(["-al", "src", "dst"]))

        self.assertEqual(guard._ln_destinations(["src", "dst"]), ["dst"])
        self.assertEqual(guard._ln_destinations(["src"]), ["src"])
        self.assertEqual(guard._ln_destinations([]), [])
        self.assertEqual(guard._ln_sources(["src", "dst"]), ["src"])
        self.assertEqual(guard._ln_sources(["-s", "src", "dst"]), [])
        self.assertEqual(
            guard._awk_redirection_targets(["BEGIN", ">", "out", ">log"]), ["out", "log"]
        )

        self.assertEqual(
            guard._metadata_target_args(["--reference", "ref", "644", "file"]), ["file"]
        )
        self.assertEqual(
            guard._metadata_target_args(["--reference=ref", "user", "dir/file"]), ["dir/file"]
        )
        self.assertEqual(guard._rsync_destination(["src", "dst"]), "dst")
        self.assertIsNone(guard._rsync_destination(["src"]))
        self.assertEqual(
            guard._rsync_secondary_write_dirs(["--backup-dir", "backup", "--partial-dir=partial"]),
            ["backup", "partial"],
        )

        self.assertEqual(guard._option_value(["-oout"], "-o", "--output"), "out")
        self.assertEqual(guard._option_value(["--output=out"], "-o", "--output"), "out")
        self.assertIsNone(guard._option_value(["--"], "-o", "--output"))
        self.assertEqual(
            guard._downloader_output_dirs(["--output-dir=dl"], base="curl"),
            ["dl"],
        )
        self.assertEqual(guard._downloader_output_dirs(["-Pcache"], base="wget"), ["cache"])
        self.assertTrue(guard._downloader_writes_to_cwd(["-LO"], base="curl"))
        self.assertTrue(guard._downloader_writes_to_cwd(["https://example.test/a"], base="wget"))
        self.assertFalse(guard._downloader_writes_to_cwd(["-O", "out"], base="wget"))
        self.assertFalse(guard._downloader_writes_to_cwd([], base="other"))

        self.assertEqual(guard._tar_archive_outputs(["-cfout.tar", "src"]), ["out.tar"])
        self.assertEqual(guard._tar_archive_outputs(["--file=out.tar", "src"]), ["out.tar"])
        self.assertTrue(guard._tar_creates_archive(["-czf", "out.tgz"]))
        self.assertTrue(guard._tar_segment_extracts(["--extract", "-f", "in.tar"]))
        self.assertEqual(guard._zip_archive_outputs(["-q", "out.zip", "src"]), ["out.zip"])
        self.assertEqual(guard._zip_archive_outputs(["--"]), [])
        self.assertEqual(guard._split_output_prefix(["input.bin", "chunk_"]), "chunk_")
        self.assertIsNone(guard._split_output_prefix(["input.bin"]))

        self.assertEqual(guard._sed_inplace_files(["-i", "s/a/b/", "file.txt"]), ["file.txt"])
        self.assertEqual(
            guard._sed_inplace_files(["-e", "s/a/b/", "--", "file.txt"]),
            ["file.txt"],
        )
        self.assertEqual(guard._find_start_paths([".", "src", "-type", "f"]), [".", "src"])
        self.assertEqual(guard._find_start_paths(["-type", "f"]), [])
        self.assertTrue(guard._is_find_expression_token("!"))
        self.assertTrue(guard._is_find_expression_token("-type"))
        self.assertFalse(guard._is_find_expression_token("src"))

        self.assertTrue(guard._zip_segment_removes_sources(["-rm"]))
        self.assertTrue(guard._shred_segment_removes_sources(["--remove=wipe"]))
        self.assertTrue(guard._rsync_segment_deletes(["--delete-excluded"]))
        self.assertTrue(guard._git_segment_is_destructive(["clean", "-fdx"]))
        self.assertTrue(guard._git_segment_is_destructive(["checkout", "--force", "HEAD"]))
        self.assertFalse(guard._git_segment_is_destructive([]))
        self.assertEqual(
            guard._git_subcommand_and_args(["-C", "repo", "--git-dir=.git", "status", "-sb"]),
            ("status", ["-sb"]),
        )
        self.assertTrue(guard._git_segment_is_read_only(["-C", "repo", "log", "--oneline"]))
        self.assertFalse(guard._git_segment_is_read_only(["diff", "--output=patch"]))
        self.assertEqual(guard._git_write_paths(["-C", "repo", "init", "dst"]), ["repo", "dst"])
        self.assertEqual(
            guard._git_write_paths(["clone", "https://example.test/repo.git", "dst"]),
            ["dst"],
        )
        self.assertEqual(guard._git_write_paths(["diff", "--output=patch"]), ["patch"])
        self.assertEqual(guard._git_write_paths(["worktree", "add", "wt", "HEAD"]), ["wt"])

        self.assertEqual(guard._shell_c_argument(["-lc", "echo ok"]), "echo ok")
        self.assertEqual(guard._shell_c_argument(["-c", "--", "echo ok"]), "echo ok")
        self.assertIsNone(guard._shell_c_argument(["script.sh"]))
        self.assertTrue(guard._env_segment_strips_guard(["--ignore-environment", "python"]))
        self.assertTrue(guard._env_segment_strips_guard(["--unset=BASH_ENV", "bash"]))
        self.assertTrue(guard._env_segment_strips_guard(["PRAXIST_SAFE_DELETE_ROOTS=/", "python"]))
        self.assertFalse(guard._env_segment_strips_guard(["PATH=/bin", "true"]))
        self.assertTrue(guard._python_segment_disables_sitecustomize(["-SI", "-c", "pass"]))
        self.assertFalse(guard._python_segment_disables_sitecustomize(["-c", "pass"]))

    def test_delete_guard_helper_contracts_cover_scripts_and_paths(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import delete_guard as guard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            peer_ws = run_dir / "peer_workspaces" / "gen0_peer0"
            task_root = root / "task"
            package = task_root / "pkg"
            for path in (run_dir, peer_ws, task_root, package):
                path.mkdir(parents=True, exist_ok=True)
            (task_root / "description.md").write_text("task", encoding="utf-8")
            (task_root / "experiments").mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            module_file = package / "tool.py"
            module_file.write_text("print('safe')\n", encoding="utf-8")
            package_main = package / "__main__.py"
            package_main.write_text("print('main')\n", encoding="utf-8")

            env = {
                "PRAXIST_SAFE_DELETE_ROOTS": str(peer_ws),
                "PRAXIST_PEER_WORKSPACE": str(peer_ws),
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_DELETE_GUARD_RUN_DIR": str(run_dir),
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                "PEER_ID": "gen0_peer0",
                "PYTHONPATH": str(task_root),
                "PATH": "/bin:/usr/bin",
            }

            self.assertEqual(
                {
                    path.name
                    for path in guard._python_module_file_candidates(
                        "pkg.tool", env=env, cwd=run_dir
                    )
                },
                {"tool.py"},
            )
            self.assertIn(
                package_main.resolve(strict=False),
                guard._python_module_file_candidates("pkg", env=env, cwd=run_dir),
            )
            self.assertTrue(guard._is_safe_python_module_name("pkg.tool_1"))
            self.assertFalse(guard._is_safe_python_module_name("pkg.bad-name"))
            self.assertIsNone(guard._python_script_operand(["-m", "pkg.tool"]))
            self.assertEqual(
                guard._python_script_operand(["-W", "ignore", "script.py"]), "script.py"
            )
            self.assertEqual(guard._python_script_operand(["--", "script.py"]), "script.py")
            self.assertEqual(guard._shell_script_operand(["-eu", "--", "safe.sh"]), "safe.sh")

            safe_python = peer_ws / "safe.py"
            safe_python.write_text("print('ok')\n", encoding="utf-8")
            unsafe_python = peer_ws / "unsafe.py"
            unsafe_python.write_text("import shutil\nshutil.rmtree('frontier')\n", encoding="utf-8")
            binary_python = peer_ws / "binary.py"
            binary_python.write_bytes(b"\x7fELFbinary")
            huge_python = peer_ws / "huge.py"
            huge_python.write_text("#" * 200_001, encoding="utf-8")
            self.assertTrue(
                guard._validate_python_script_file(str(safe_python), env=env, cwd=run_dir).allowed
            )
            self.assertFalse(
                guard._validate_python_script_file(str(unsafe_python), env=env, cwd=run_dir).allowed
            )
            self.assertFalse(
                guard._validate_python_script_file(str(binary_python), env=env, cwd=run_dir).allowed
            )
            self.assertFalse(
                guard._validate_python_script_file(str(huge_python), env=env, cwd=run_dir).allowed
            )
            self.assertTrue(
                guard._validate_python_script_file("missing.py", env=env, cwd=run_dir).allowed
            )

            safe_shell = peer_ws / "safe.sh"
            safe_shell.write_text("echo ok\n", encoding="utf-8")
            unsafe_shell = peer_ws / "unsafe.sh"
            unsafe_shell.write_text("rm -rf $PRAXIST_RUN_DIR/frontier\n", encoding="utf-8")
            huge_shell = peer_ws / "huge.sh"
            huge_shell.write_text("#" * 200_001, encoding="utf-8")
            binary_shell = peer_ws / "binary.sh"
            binary_shell.write_bytes(b"\x00binary")
            self.assertTrue(
                guard._validate_script_file(str(safe_shell), env=env, cwd=run_dir, depth=0).allowed
            )
            self.assertFalse(
                guard._validate_script_file(
                    str(unsafe_shell), env=env, cwd=run_dir, depth=0
                ).allowed
            )
            self.assertFalse(
                guard._validate_script_file(
                    str(binary_shell), env=env, cwd=run_dir, depth=0
                ).allowed
            )
            self.assertFalse(
                guard._validate_script_file(str(huge_shell), env=env, cwd=run_dir, depth=0).allowed
            )
            self.assertTrue(
                guard._validate_script_file(
                    str(peer_ws / "missing.sh"), env=env, cwd=run_dir, depth=0
                ).allowed
            )
            self.assertFalse(
                guard._validate_script_file(
                    str(run_dir / "missing.sh"), env=env, cwd=run_dir, depth=0
                ).allowed
            )
            self.assertTrue(guard._looks_binary_payload(b"\xcf\xfa\xed\xfe"))
            self.assertTrue(guard._looks_binary_payload(b"\xca\xfe\xba\xbe"))
            self.assertFalse(guard._looks_binary_payload(b"#!/bin/sh\n"))

            self.assertTrue(guard._is_guard_assignment("BASH_ENV=/tmp/x"))
            self.assertFalse(guard._is_guard_assignment("-BASH_ENV=/tmp/x"))
            self.assertTrue(
                guard._find_execs_destructive_command([".", "-exec", "python", "-c", "pass", ";"])
            )
            self.assertFalse(guard._find_execs_destructive_command([".", "-name", "*.py"]))
            tokens = guard._shell_tokens(
                "A=1 command -- rm -rf x; env -u BASH_ENV bash -c 'echo ok'"
            )
            self.assertEqual(guard._iter_delete_invocations_anywhere(tokens)[0], ["-rf", "x"])
            self.assertEqual(guard._skip_command_prefixes(["A=1", "command", "--", "rm"], 0), 3)
            self.assertEqual(guard._skip_env_prefix(["--unset=BASH_ENV", "python"], 0), 1)
            self.assertEqual(guard._skip_nice_prefix(["--adjustment=5", "python"], 0), 1)
            self.assertEqual(guard._command_segment(["rm", "x", ";", "echo"], 1), ["x"])
            self.assertEqual(guard._rm_targets(["-rf", "--", "-literal"]), ["-literal"])

            globbed = peer_ws / "tmp" / "a.txt"
            globbed.parent.mkdir(parents=True, exist_ok=True)
            globbed.write_text("x", encoding="utf-8")
            self.assertEqual(
                guard._resolve_target("$PRAXIST_PEER_WORKSPACE/tmp/a.txt", env=env, cwd=run_dir),
                globbed.resolve(),
            )
            self.assertIsNone(guard._resolve_target("$UNKNOWN/path", env=env, cwd=run_dir))
            self.assertIsNotNone(
                guard._resolve_target_candidates(
                    str(peer_ws / "tmp" / "*.txt"), env=env, cwd=run_dir
                )
            )
            self.assertIsNone(guard._resolve_target_candidates("../*/bad", env=env, cwd=run_dir))
            self.assertEqual(guard._allowed_roots(env), [peer_ws.resolve()])
            self.assertEqual(guard._run_dir(env), run_dir.resolve())
            self.assertIn(task_root.resolve(), guard._protected_roots(env, cwd=run_dir))
            self.assertEqual(guard._peer_id({**env, "PEER_ID": "bad name!"}), "bad_name")
            self.assertEqual(
                guard._run_relative_parts(run_dir / "variants" / "gen0_peer0_a", env),
                ("variants", "gen0_peer0_a"),
            )
            self.assertTrue(
                guard._is_peer_owned_run_write_path(run_dir / "variants" / "gen0_peer0_a", env)
            )
            self.assertTrue(
                guard._is_peer_owned_run_write_path(
                    run_dir / "shared_findings" / "gen0_peer0_a.json", env
                )
            )
            self.assertTrue(
                guard._is_peer_owned_run_write_path(run_dir / "notebook_gen0_peer0.json", env)
            )
            self.assertFalse(
                guard._is_peer_owned_run_write_path(run_dir / "run_control" / "stop.json", env)
            )
            self.assertTrue(
                guard._is_peer_owned_run_delete_path(run_dir / "results" / "gen0_peer0_a", env)
            )
            self.assertFalse(
                guard._is_peer_owned_run_delete_path(
                    run_dir / "shared_findings" / "gen0_peer0_a.json", env
                )
            )
            self.assertIn(task_root.resolve(), guard._trusted_project_roots(env, cwd=task_root))
            self.assertIn(task_root.resolve(), guard._discover_task_roots(task_root / "subdir"))
            self.assertTrue(guard._is_trusted_project_script(module_file, env, cwd=run_dir))
            self.assertFalse(guard._is_trusted_system_executable(module_file))
            self.assertEqual(guard._command_token_basename("$(rm)"), "rm")
            self.assertTrue(guard._has_glob_meta("*.py"))
            self.assertTrue(guard._mentions_protected_target("$PRAXIST_RUN_DIR/frontier", env))
            self.assertEqual(guard._safe_name(" bad/name "), "bad_name")
            self.assertEqual(
                guard._join_paths([run_dir, peer_ws]), f"{run_dir}{os.pathsep}{peer_ws}"
            )
            self.assertNotIn("validate-bash", guard._bash_env_text())
            self.assertNotIn("DEBUG", guard._bash_env_text())
            self.assertFalse(hasattr(guard, "_rm_wrapper_text"))
            self.assertFalse(hasattr(guard, "_shell_wrapper_text"))
            self.assertIn("sqlite3", guard._sitecustomize_text())

    def test_delete_guard_command_matrix_covers_validator_edges(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import delete_guard as guard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            peer_ws = run_dir / "peer_workspaces" / "gen0_peer0"
            task_root = root / "task"
            for path in (run_dir, peer_ws, task_root):
                path.mkdir(parents=True, exist_ok=True)
            protected_file = task_root / "keep.txt"
            protected_file.write_text("keep", encoding="utf-8")
            unsafe_script = task_root / "unsafe.sh"
            unsafe_script.write_text('rm -rf "$PRAXIST_RUN_DIR/frontier"\n', encoding="utf-8")
            safe_makefile = peer_ws / "Makefile"
            safe_makefile.write_text("include included.mk\nall:\n\t@echo ok\n", encoding="utf-8")
            (peer_ws / "included.mk").write_text("noop:\n\t@echo noop\n", encoding="utf-8")
            unsafe_makefile = peer_ws / "Unsafe.mk"
            unsafe_makefile.write_text("SHELL := /bin/bash\nall:\n\t@echo bad\n", encoding="utf-8")
            unsafe_recipe = peer_ws / "UnsafeRecipe.mk"
            unsafe_recipe.write_text(f"all:\n\trm -rf {task_root}\n", encoding="utf-8")
            unsafe_ninja = peer_ws / "build.ninja"
            unsafe_ninja.write_text(
                "rule bad\n  command = rm -rf $PRAXIST_RUN_DIR/frontier\n", encoding="utf-8"
            )
            env = {
                "PRAXIST_SAFE_DELETE_ROOTS": str(peer_ws),
                "PRAXIST_PEER_WORKSPACE": str(peer_ws),
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_DELETE_GUARD_RUN_DIR": str(run_dir),
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                "PEER_ID": "gen0_peer0",
                "PATH": "/bin:/usr/bin",
            }

            disabled = guard.prepare_delete_guard_env(
                {"PRAXIST_DISABLE_DELETE_GUARD": "1"},
                workspace=run_dir,
                agent_name="gen0_peer0",
            )
            self.assertEqual(disabled, {"PRAXIST_DISABLE_DELETE_GUARD": "1"})
            self.assertTrue(
                guard.validate_tool_use(
                    "Read",
                    {"file_path": str(protected_file)},
                    env=env,
                    cwd=run_dir,
                ).allowed
            )
            self.assertTrue(guard.validate_bash_command("", env=env, cwd=run_dir).allowed)
            self.assertFalse(
                guard._validate_bash_command(
                    "echo nested",
                    env=env,
                    cwd=run_dir,
                    depth=4,
                ).allowed
            )
            self.assertTrue(
                guard.validate_bash_command(
                    f"rm -rf {shlex.quote(str(task_root))}",
                    env={"PATH": "/bin"},
                    cwd=run_dir,
                ).allowed
            )
            self.assertTrue(
                guard.validate_rm_argv(
                    ["-rf", str(task_root)],
                    env={"PATH": "/bin"},
                    cwd=run_dir,
                ).allowed
            )
            self.assertFalse(
                guard.validate_tool_use(
                    "Write",
                    {"file_path": str(run_dir / "frontier" / "manifest.json")},
                    env=env,
                    cwd=run_dir,
                ).allowed
            )
            self.assertFalse(
                guard._validate_run_write_path(
                    "$UNSET/out.txt",
                    env=env,
                    cwd=run_dir,
                    op="write",
                ).allowed
            )
            self.assertFalse(
                guard._validate_link_sources(["$UNSET/source"], env=env, cwd=run_dir).allowed
            )
            self.assertFalse(
                guard._validate_path_args(
                    [],
                    env=env,
                    cwd=run_dir,
                    allowed_roots=[peer_ws],
                    op="source",
                ).allowed
            )
            self.assertFalse(
                guard._validate_find_delete(
                    ["find", str(task_root), "-delete"],
                    env=env,
                    cwd=run_dir,
                    allowed_roots=[peer_ws],
                ).allowed
            )
            self.assertTrue(guard._build_include_targets("", is_ninja=False) is None)
            self.assertEqual(
                guard._build_include_targets("include included.mk", is_ninja=False),
                ("include", ["included.mk"], False),
            )
            self.assertEqual(
                guard._build_include_targets("-include missing.mk", is_ninja=False),
                ("-include", ["missing.mk"], True),
            )
            self.assertEqual(
                guard._build_include_targets("subninja child.ninja", is_ninja=True),
                ("subninja", ["child.ninja"], False),
            )
            self.assertFalse(
                guard._validate_build_include_target(
                    "$DYNAMIC",
                    source_path=safe_makefile,
                    env=env,
                )[0].allowed
            )
            self.assertFalse(
                guard._validate_build_file_content(
                    safe_makefile,
                    env=env,
                    depth=guard._MAX_BUILD_INCLUDE_DEPTH + 1,
                ).allowed
            )
            self.assertFalse(guard._validate_build_file_content(unsafe_makefile, env=env).allowed)
            self.assertFalse(guard._validate_build_file_content(unsafe_recipe, env=env).allowed)
            self.assertFalse(guard._validate_build_file_content(unsafe_ninja, env=env).allowed)
            self.assertTrue(guard._validate_build_file_content(safe_makefile, env=env).allowed)

            blocked_commands = [
                "cd -; rm -rf old",
                "cd $UNSET; rm -rf old",
                f"bash {shlex.quote(str(unsafe_script))}",
                f"cd {shlex.quote(str(task_root))}; tar --checkpoint-action=exec=sh x.tar",
                "tar --remove-files -cf archive.tar old",
                "tar -xf archive.tar",
                "zip -m archive.zip old",
                f"shred -u {shlex.quote(str(task_root / 'keep.txt'))}",
                "rsync --delete src/ dst/",
                "git clean -fdx",
                f"git -C {shlex.quote(str(task_root))} status",
                f"truncate -s 0 {shlex.quote(str(task_root / 'keep.txt'))}",
                f"dd if=/dev/null of={shlex.quote(str(task_root / 'keep.txt'))}",
                f"cp -l {shlex.quote(str(protected_file))} {shlex.quote(str(peer_ws / 'copy.txt'))}",
                f"cp {shlex.quote(str(peer_ws / 'copy.txt'))} {shlex.quote(str(task_root / 'copy.txt'))}",
                f"tee {shlex.quote(str(task_root / 'tee.txt'))}",
                f"sed -i '' {shlex.quote(str(task_root / 'keep.txt'))}",
                f"sort -o {shlex.quote(str(task_root / 'sorted.txt'))} input.txt",
                f"sort -T {shlex.quote(str(task_root))} input.txt",
                "tar -cf archive.tar input.txt",
                f"tar -cf {shlex.quote(str(task_root / 'archive.tar'))} input.txt",
                "zip archive.zip input.txt",
                f"gcc -o {shlex.quote(str(task_root / 'a.out'))} input.c",
                f"touch {shlex.quote(str(task_root / 'new.txt'))}",
                f"mkdir {shlex.quote(str(task_root / 'newdir'))}",
                f"fallocate -l 1 {shlex.quote(str(task_root / 'blob'))}",
                f"curl -o {shlex.quote(str(task_root / 'download.txt'))} https://example.invalid",
                f"wget -O {shlex.quote(str(task_root / 'download.txt'))} https://example.invalid",
                f"wget -P {shlex.quote(str(task_root))} https://example.invalid/file",
                "unzip archive.zip",
                f"unzip archive.zip -d {shlex.quote(str(task_root))}",
                "patch -p1 < fix.patch",
                "split input",
                f"split -b 1 input {shlex.quote(str(task_root / 'part'))}",
                f"ln {shlex.quote(str(protected_file))} {shlex.quote(str(peer_ws / 'ln.txt'))}",
                f"ln {shlex.quote(str(peer_ws / 'ln.txt'))} {shlex.quote(str(task_root / 'ln.txt'))}",
                f"chmod 600 {shlex.quote(str(task_root / 'keep.txt'))}",
                f"rsync src/ {shlex.quote(str(task_root / 'dst'))}",
                f"rsync --partial-dir={shlex.quote(str(task_root / 'partial'))} src/ dst/",
                f"echo hi > {shlex.quote(str(task_root / 'redir.txt'))}",
                "2> ",
                "GUARD=/tmp make clean",
                "BASH_ENV=/tmp/x make all",
                f"make -C {shlex.quote(str(peer_ws))} -f {shlex.quote(str(unsafe_makefile))} all",
                f"ninja -C {shlex.quote(str(peer_ws))}",
            ]
            for command in blocked_commands:
                with self.subTest(command=command):
                    self.assertFalse(
                        guard.validate_bash_command(command, env=env, cwd=run_dir).allowed
                    )

            allowed_commands = [
                f"touch {shlex.quote(str(peer_ws / 'ok.txt'))}",
                f"mkdir {shlex.quote(str(peer_ws / 'dir'))}",
                f"echo hi > {shlex.quote(str(peer_ws / 'redir.txt'))}",
                f"curl -o {shlex.quote(str(peer_ws / 'download.txt'))} https://example.invalid",
                f"rsync src/ {shlex.quote(str(peer_ws / 'dst'))}",
                f"gcc -o {shlex.quote(str(peer_ws / 'a.out'))} input.c",
                f"make -C {shlex.quote(str(peer_ws))} all",
            ]
            for command in allowed_commands:
                with self.subTest(command=command):
                    self.assertTrue(
                        guard.validate_bash_command(command, env=env, cwd=run_dir).allowed
                    )

    def test_delete_guard_argument_parser_helper_edges(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import delete_guard as guard

        self.assertEqual(
            guard._source_destructive_path_args(["-rf", "-t", "dst", "--", "src"]),
            ["dst", "src"],
        )
        self.assertEqual(guard._source_destructive_path_args(["-tdst", "src"]), ["dst", "src"])
        self.assertEqual(
            guard._source_destructive_path_args(["--target-directory=dst", "src"]),
            ["dst", "src"],
        )
        self.assertEqual(guard._mv_sources_destinations(["-tdst", "src"]), (["src"], ["dst"]))
        self.assertEqual(guard._mv_sources_destinations(["only"]), (["only"], []))
        self.assertTrue(guard._zip_segment_removes_sources(["-rm"]))
        self.assertTrue(guard._shred_segment_removes_sources(["--remove=wipesync"]))
        self.assertTrue(guard._rsync_segment_deletes(["--delete-after"]))

        self.assertTrue(guard._git_segment_is_destructive(["clean", "-fdx"]))
        self.assertTrue(guard._git_segment_is_destructive(["reset", "--hard"]))
        self.assertTrue(guard._git_segment_is_destructive(["checkout", "--force", "main"]))
        self.assertFalse(guard._git_segment_is_destructive(["checkout", "main"]))
        self.assertFalse(guard._git_segment_is_read_only(["diff", "--output=patch"]))
        self.assertEqual(
            guard._git_subcommand_and_args(
                ["-C", "repo", "--git-dir=.git", "-c", "a=b", "--", "status", "--short"]
            ),
            ("status", ["--short"]),
        )
        self.assertIsNone(guard._git_subcommand_and_args(["-C"]))
        self.assertEqual(
            guard._git_write_paths(["-C", "repo", "init", "--bare", "dst"]), ["repo", "dst"]
        )
        self.assertEqual(guard._git_write_paths(["clone", "--depth", "1", "url", "dst"]), ["dst"])
        self.assertEqual(
            guard._git_write_paths(["archive", "--output=out.tar", "HEAD"]), ["out.tar"]
        )
        self.assertEqual(
            guard._git_write_paths(["worktree", "add", "--detach", "wt", "HEAD"]), ["wt"]
        )
        self.assertIsNone(guard._last_path_arg([]))

        self.assertEqual(guard._cp_install_destinations(["-Dt", "dst", "src"]), ["dst"])
        self.assertEqual(guard._cp_install_destinations(["-tdir", "src"]), ["dir"])
        self.assertEqual(guard._cp_install_destinations(["src", "dst"]), ["dst"])
        self.assertEqual(guard._cp_install_sources(["-t", "dst", "src"]), ["dst"])
        self.assertEqual(guard._cp_install_sources(["src", "dst"]), ["src"])
        self.assertTrue(guard._cp_segment_hardlinks(["-al"]))
        self.assertEqual(guard._ln_destinations(["src"]), ["src"])
        self.assertEqual(guard._ln_destinations(["src", "dst"]), ["dst"])
        self.assertEqual(guard._ln_sources(["-s", "src", "dst"]), [])
        self.assertEqual(guard._ln_sources(["src", "dst"]), ["src"])
        self.assertEqual(guard._awk_redirection_targets([">out", ">>", "err"]), ["out", "err"])
        self.assertEqual(
            guard._metadata_target_args(["--reference", "ref", "644", "file", "/abs"]),
            ["file", "/abs"],
        )
        self.assertEqual(guard._rsync_destination(["src", "dst"]), "dst")
        self.assertEqual(
            guard._rsync_secondary_write_dirs(["--backup-dir", "bak", "--partial-dir=part"]),
            ["bak", "part"],
        )
        self.assertEqual(guard._option_value(["--output=out"], "-o", "--output"), "out")
        self.assertEqual(guard._option_value(["-oout"], "-o", "--output"), "out")
        self.assertEqual(
            guard._downloader_output_dirs(["--output-dir=downloads"], base="curl"),
            ["downloads"],
        )
        self.assertEqual(
            guard._downloader_output_dirs(["-Pdownloads", "--directory-prefix=more"], base="wget"),
            ["downloads", "more"],
        )
        self.assertTrue(guard._downloader_writes_to_cwd(["-LO"], base="curl"))
        self.assertFalse(guard._downloader_writes_to_cwd(["-Oout"], base="wget"))
        self.assertFalse(guard._downloader_writes_to_cwd([], base="unknown"))
        self.assertEqual(guard._tar_archive_outputs(["-cfout.tar", "src"]), ["out.tar"])
        self.assertEqual(guard._tar_archive_outputs(["--file=out.tar", "src"]), ["out.tar"])
        self.assertEqual(guard._tar_archive_outputs(["-c", "-f", "out.tar", "src"]), ["out.tar"])
        self.assertTrue(guard._tar_creates_archive(["--create"]))
        self.assertEqual(guard._zip_archive_outputs(["--", "out.zip", "src"]), ["out.zip"])
        self.assertEqual(guard._split_output_prefix(["input", "prefix"]), "prefix")
        self.assertIsNone(guard._split_output_prefix(["input"]))
        self.assertTrue(guard._tar_segment_extracts(["--get"]))
        self.assertEqual(
            guard._sed_inplace_files(["-e", "s/a/b/", "-i", "script", "file"]), ["file"]
        )
        self.assertEqual(guard._find_start_paths(["--", "root", "-name", "*.py"]), ["root"])
        self.assertTrue(guard._is_find_expression_token("!"))

        self.assertEqual(guard._shell_c_argument(["-ecpayload"]), "payload")
        self.assertEqual(guard._shell_c_argument(["-e", "-c", "--", "payload"]), "payload")
        self.assertTrue(guard._has_command_word_expansion("r${x}m"))
        self.assertTrue(guard._is_guard_assignment("BASH_ENV=/tmp/guard"))
        self.assertTrue(guard._segment_contains_guarded_launcher(["env", "python", "-c", "pass"]))
        self.assertTrue(guard._env_segment_strips_guard(["--unset=BASH_ENV", "python"]))
        self.assertTrue(guard._env_segment_strips_guard(["-u", "PYTHONPATH", "python"]))
        self.assertTrue(guard._python_segment_disables_sitecustomize(["-Es", "-c", "pass"]))
        self.assertFalse(guard._python_segment_disables_sitecustomize(["-c", "pass"]))
        self.assertEqual(guard._shell_script_operand(["--", "script.sh"]), "script.sh")
        self.assertIsNone(guard._python_script_operand(["-m", "module"]))
        self.assertEqual(
            guard._python_script_operand(["-W", "ignore", "--", "script.py"]), "script.py"
        )
        self.assertTrue(guard._is_safe_python_module_name("pkg._mod1"))
        self.assertFalse(guard._is_safe_python_module_name("pkg.bad-name"))
        self.assertIn(
            ["target"], guard._iter_delete_invocations_anywhere(["echo", "x", "rm", "target"])
        )
        self.assertEqual(guard._skip_command_options(["-p", "--", "rm"], 0), 2)
        self.assertEqual(guard._skip_env_prefix(["--chdir=/tmp", "A=B", "python"], 0), 2)
        self.assertEqual(guard._skip_simple_options(["-n", "--", "cmd"], 0), 2)
        self.assertEqual(guard._skip_nice_prefix(["--adjustment=5", "cmd"], 0), 1)
        self.assertEqual(guard._command_segment(["a", "b", ";", "c"], 1), ["b"])
        self.assertEqual(guard._rm_targets(["-rf", "--", "a", "b"]), ["a", "b"])
        self.assertTrue(guard._has_glob_meta("*.py"))
        self.assertEqual(guard._safe_name("gen/0 peer"), "gen_0_peer")

    def test_delete_guard_low_level_validation_edges(self) -> None:
        from praxist.plugins.agent_runtimes.claude_sdk import delete_guard as guard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_root = root / "task"
            run_dir = root / "run"
            peer_ws = run_dir / "peer_workspaces" / "gen0_peer0"
            peer_tmp = peer_ws / "tmp"
            protected = run_dir / "shared_findings" / "keep.json"
            task_file = task_root / "description.md"
            peer_tmp.mkdir(parents=True)
            protected.parent.mkdir(parents=True)
            task_root.mkdir()
            protected.write_text("{}", encoding="utf-8")
            task_file.write_text("canonical", encoding="utf-8")
            env = {
                "PRAXIST_SAFE_DELETE_ROOTS": str(peer_ws),
                "PRAXIST_PEER_WORKSPACE": str(peer_ws),
                "PRAXIST_RUN_DIR": str(run_dir),
                "PRAXIST_DELETE_GUARD_RUN_DIR": str(run_dir),
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                "PATH": f"{peer_tmp}{os.pathsep}/bin:/usr/bin",
            }
            allowed_roots = guard._allowed_roots(env)

            self.assertTrue(
                guard.validate_rm_argv(["rm", "-rf", str(run_dir)], env={}, cwd=run_dir).allowed
            )
            self.assertFalse(
                guard._validate_run_write_path(
                    "$UNKNOWN/output.json",
                    env=env,
                    cwd=run_dir,
                    op="write",
                ).allowed
            )
            ambiguous_mkdir = guard._validate_run_write_path(
                "$OUTDIR",
                env=env,
                cwd=run_dir,
                op="mkdir",
            )
            self.assertTrue(ambiguous_mkdir.allowed, ambiguous_mkdir.message)
            self.assertTrue(ambiguous_mkdir.warning)
            self.assertTrue(
                guard._validate_run_write_path(
                    str(peer_tmp / "out.json"),
                    env=env,
                    cwd=run_dir,
                    op="write",
                ).allowed
            )
            self.assertFalse(
                guard._validate_run_write_path(
                    str(protected), env=env, cwd=run_dir, op="write"
                ).allowed
            )
            self.assertFalse(
                guard._validate_run_write_path(
                    str(task_root / "new.txt"),
                    env=env,
                    cwd=run_dir,
                    op="write",
                ).allowed
            )
            self.assertFalse(
                guard._validate_implicit_cwd_write(env=env, cwd=task_root, op="tar").allowed
            )
            self.assertFalse(
                guard._validate_link_sources(["$UNKNOWN"], env=env, cwd=run_dir).allowed
            )
            self.assertFalse(
                guard._validate_link_sources([str(task_file)], env=env, cwd=run_dir).allowed
            )
            self.assertFalse(
                guard._validate_rm_args(
                    [],
                    env=env,
                    cwd=run_dir,
                    allowed_roots=allowed_roots,
                ).allowed
            )
            self.assertFalse(
                guard._validate_path_args(
                    [],
                    env=env,
                    cwd=run_dir,
                    allowed_roots=allowed_roots,
                    op="mv source",
                ).allowed
            )
            self.assertFalse(
                guard._validate_path_args(
                    ["$UNKNOWN"],
                    env=env,
                    cwd=run_dir,
                    allowed_roots=allowed_roots,
                    op="mv source",
                ).allowed
            )

            self.assertFalse(
                guard._validate_redirections(
                    "printf x >",
                    env=env,
                    cwd=run_dir,
                    allowed_roots=allowed_roots,
                ).allowed
            )
            self.assertTrue(
                guard._validate_redirections(
                    "printf x >&1 2>&2",
                    env=env,
                    cwd=run_dir,
                    allowed_roots=allowed_roots,
                ).allowed
            )
            self.assertFalse(
                guard._validate_redirections(
                    f"printf x 2> {protected}",
                    env=env,
                    cwd=run_dir,
                    allowed_roots=allowed_roots,
                ).allowed
            )
            self.assertFalse(
                guard._validate_build_command_prefixes(
                    [f"OUT={protected}"],
                    env=env,
                    cwd=run_dir,
                ).allowed
            )
            self.assertFalse(
                guard._validate_build_command_prefixes(
                    ["PYTHONPATH="],
                    env=env,
                    cwd=run_dir,
                ).allowed
            )
            self.assertFalse(
                guard._validate_make_ninja_segment(
                    ["make", "-C", "$UNKNOWN"],
                    env=env,
                    cwd=run_dir,
                    allowed_roots=allowed_roots,
                ).allowed
            )
            self.assertFalse(
                guard._validate_make_ninja_segment(
                    ["make", "-C", str(task_root)],
                    env=env,
                    cwd=run_dir,
                    allowed_roots=allowed_roots,
                ).allowed
            )
            self.assertFalse(
                guard._validate_make_ninja_segment(
                    ["make", "-C", str(peer_ws / "empty")],
                    env=env,
                    cwd=run_dir,
                    allowed_roots=allowed_roots,
                ).allowed
            )

            self.assertEqual(
                guard._explicit_build_file_candidates(
                    "$UNKNOWN",
                    build_paths=[peer_ws],
                    env=env,
                    cwd=run_dir,
                ),
                [],
            )
            self.assertIsNone(guard._build_include_targets("", is_ninja=False))
            self.assertIsNone(guard._build_include_targets("not an include", is_ninja=True))
            self.assertIsNone(guard._build_include_targets("include", is_ninja=False))
            self.assertIsNone(guard._build_include_targets("-include", is_ninja=False))
            self.assertEqual(
                guard._build_include_targets("include 'unterminated", is_ninja=False),
                ("include", [], False),
            )

            makefile = peer_ws / "Makefile"
            include = peer_ws / "safe.mk"
            include.write_text("all:\n\ttrue\n", encoding="utf-8")
            makefile.write_text("include safe.mk\n", encoding="utf-8")
            self.assertTrue(guard._validate_build_file_content(makefile, env=env).allowed)
            self.assertTrue(
                guard._validate_build_file_content(
                    makefile,
                    env=env,
                    seen={str(makefile.resolve(strict=False))},
                ).allowed
            )
            self.assertFalse(
                guard._validate_build_file_content(
                    makefile,
                    env=env,
                    depth=guard._MAX_BUILD_INCLUDE_DEPTH + 1,
                ).allowed
            )
            makefile.write_text("SHELL := /bin/sh\n", encoding="utf-8")
            self.assertFalse(guard._validate_build_file_content(makefile, env=env).allowed)
            makefile.write_text("include missing.mk\n", encoding="utf-8")
            self.assertFalse(guard._validate_build_file_content(makefile, env=env).allowed)
            makefile.write_text("-include missing.mk\n", encoding="utf-8")
            self.assertTrue(guard._validate_build_file_content(makefile, env=env).allowed)
            makefile.write_text("include *.mk\n", encoding="utf-8")
            self.assertFalse(guard._validate_build_file_content(makefile, env=env).allowed)
            outside_include = root / "outside.mk"
            outside_include.write_text("all:\n\ttrue\n", encoding="utf-8")
            makefile.write_text(f"include {outside_include}\n", encoding="utf-8")
            self.assertFalse(guard._validate_build_file_content(makefile, env=env).allowed)

            self.assertFalse(
                guard._validate_python_module_mode(["-m"], env=env, cwd=run_dir).allowed
            )
            missing_module = guard._validate_python_module_mode(
                ["-m", "missing.module"],
                env=env,
                cwd=run_dir,
            )
            self.assertTrue(missing_module.allowed, missing_module.message)
            self.assertTrue(missing_module.warning)
            self.assertTrue(
                guard._validate_python_module_mode(
                    [
                        "-m",
                        "praxist.plugins.workflow_stages.research_loop.backend.protected_pids",
                    ],
                    env=env,
                    cwd=run_dir,
                ).allowed
            )
            self.assertEqual(
                guard._python_module_file_candidates("missing.module", env=env, cwd=run_dir),
                [],
            )
            self.assertFalse(guard._is_safe_python_module_name("1bad"))

            shell_script = peer_tmp / "script.sh"
            shell_script.write_text(f"rm -rf {protected}\n", encoding="utf-8")
            shell_script.chmod(0o755)
            self.assertFalse(
                guard._validate_script_file(
                    str(shell_script), env=env, cwd=run_dir, depth=0
                ).allowed
            )
            self.assertTrue(
                guard._validate_script_file(
                    str(peer_tmp / "missing.sh"),
                    env=env,
                    cwd=run_dir,
                    depth=0,
                ).allowed
            )
            self.assertFalse(
                guard._validate_script_file(
                    str(task_root / "missing.sh"),
                    env=env,
                    cwd=run_dir,
                    depth=0,
                ).allowed
            )
            self.assertTrue(
                guard._validate_script_file(str(peer_tmp), env=env, cwd=run_dir, depth=0).allowed
            )
            binary_script = peer_tmp / "binary.sh"
            binary_script.write_bytes(b"\x00binary")
            self.assertFalse(
                guard._validate_script_file(
                    str(binary_script), env=env, cwd=run_dir, depth=0
                ).allowed
            )

            python_script = peer_tmp / "script.py"
            python_script.write_text(
                f"import os\nos.unlink({str(protected)!r})\n",
                encoding="utf-8",
            )
            self.assertFalse(
                guard._validate_python_script_file(str(python_script), env=env, cwd=run_dir).allowed
            )
            self.assertTrue(
                guard._validate_python_script_file(
                    str(peer_tmp / "missing.py"),
                    env=env,
                    cwd=run_dir,
                ).allowed
            )
            binary_python = peer_tmp / "binary.py"
            binary_python.write_bytes(b"\x7fELF")
            self.assertFalse(
                guard._validate_python_script_file(str(binary_python), env=env, cwd=run_dir).allowed
            )

            tool = peer_tmp / "tool"
            tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            tool.chmod(0o755)
            self.assertEqual(
                guard._resolve_executable_word("tool", [], env=env, cwd=run_dir),
                tool.resolve(strict=False),
            )
            self.assertIsNone(
                guard._resolve_executable_word("missing-tool", [], env=env, cwd=run_dir)
            )
            self.assertTrue(guard._looks_pathlike_or_protected("shared_findings/keep.json"))
            self.assertFalse(guard._looks_pathlike_or_protected("plainword"))
            self.assertFalse(
                guard._validate_unclassified_path_arg(
                    str(task_file),
                    base="openssl",
                    env=env,
                    cwd=run_dir,
                ).allowed
            )


if __name__ == "__main__":
    unittest.main()
