"""Tests for ``praxist status`` — CLI lifecycle Phases 1 + 2.5.

Phase 1 tests stub ``ps`` output and verify the pattern-matching path.
Phase 2.5 tests additionally seed an isolated registry directory (via
``PRAXIST_STATE_DIR``) and verify the merged source / state / run_id
columns surface registry rows, ps-only rows, and stale rows.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


def _fake_ps_completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["ps"], returncode=returncode, stdout=stdout, stderr="")


class CollectStatusRowsTest(unittest.TestCase):
    """``collect_status_rows`` filters the ps table by the Praxist pattern set."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env_patch = patch.dict(os.environ, {"PRAXIST_STATE_DIR": self.tmp.name}, clear=False)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

    def test_returns_only_matching_praxist_processes(self) -> None:
        from praxist.cli import status

        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   1000       1    01:23 /sbin/launchd\n"
            "   2000       1    00:42 python -u -m praxist.run run --task-path /tmp/task --run-dir /tmp/run\n"
            "   2001    2000    00:30 claude --output-format stream-json\n"
            "   2002    1234    05:00 /usr/bin/sshd\n"
            "   2003    2000    00:10 python /tmp/run/variants/v0/train.py --epochs 3\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        pids = sorted(row.pid for row in rows)
        self.assertEqual(pids, [2000])
        ai_run = next(row for row in rows if row.pid == 2000)
        self.assertEqual(ai_run.ppid, 1)
        self.assertEqual(ai_run.etime, "00:42")
        self.assertEqual(ai_run.run_dir, "/tmp/run")
        self.assertIn("praxist.run", ai_run.command)
        # Non-Praxist processes (launchd, sshd) are filtered out.
        for row in rows:
            self.assertNotIn("launchd", row.command)
            self.assertNotIn("sshd", row.command)

    def test_task_shaped_process_names_do_not_establish_praxist_ownership(self) -> None:
        from praxist.cli import status

        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   2100       1    00:42 python /other/project/train.py --epochs 3\n"
            "   2101       1    00:30 python /other/project/run_benchmark.py\n"
            "   2102       1    00:20 claude --output-format stream-json\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()

        self.assertEqual(rows, [])

    def test_self_and_ancestor_pids_are_excluded(self) -> None:
        """The praxist binary's own pid (and its parent shell) must not appear."""
        from praxist.cli import status

        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   3000       1    00:01 python -m praxist.run run --task-path /t\n"
            "   3001    3000    00:01 python -m praxist.run run --task-path /t\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=3001),
            patch.object(os, "getppid", return_value=3000),
        ):
            rows = status.collect_status_rows()
        # Both 3000 and 3001 are in the self-ancestor chain → excluded.
        self.assertEqual(rows, [])

    def test_returns_empty_when_ps_binary_missing(self) -> None:
        from praxist.cli import status

        with patch("praxist.cli.status.shutil.which", return_value=None):
            rows = status.collect_status_rows()
        self.assertEqual(rows, [])

    def test_returns_empty_when_ps_fails(self) -> None:
        from praxist.cli import status

        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed("", returncode=1),
            ),
        ):
            rows = status.collect_status_rows()
        self.assertEqual(rows, [])


class StatusCliEndToEndTest(unittest.TestCase):
    """``praxist status`` end-to-end through the ``praxist`` dispatcher."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env_patch = patch.dict(os.environ, {"PRAXIST_STATE_DIR": self.tmp.name}, clear=False)
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

    def test_table_output_prints_pid_and_run_dir(self) -> None:
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   4000       1    00:11 python -m praxist.run run --task-path /t --run-dir /runs/r1\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            code, out, _err = self._run(["status"])
        self.assertEqual(code, 0)
        # Header + one matching row.
        self.assertIn("PID", out)
        self.assertIn("4000", out)
        self.assertIn("/runs/r1", out)

    def test_json_output_is_valid_json_with_expected_fields(self) -> None:
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   5000       1    01:11 python -m praxist.run run --task-path /t --run-dir /runs/json\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            code, out, _err = self._run(["status", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 1)
        entry = payload[0]
        self.assertEqual(entry["pid"], 5000)
        self.assertEqual(entry["ppid"], 1)
        self.assertEqual(entry["etime"], "01:11")
        self.assertEqual(entry["run_dir"], "/runs/json")
        self.assertIn("praxist.run", entry["command"])

    def test_empty_table_writes_operator_hint_to_stderr(self) -> None:
        ps_output = "    PID    PPID  ELAPSED COMMAND\n   100  1  00:00 launchd\n"
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            code, out, err = self._run(["status"])
        self.assertEqual(code, 0)
        # No data rows on stdout — only header.
        self.assertIn("PID", out)
        self.assertNotIn("100", out.split("\n", 1)[1] if "\n" in out else "")
        # Hint must go to stderr (output discipline).
        self.assertIn("no Praxist experiment processes found", err)

    def test_probe_and_registry_errors_preserve_unknown_run_and_report_warnings(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(
            **_registry_entry_kwargs(run_id="run_probe_unknown", pid=8123)
        )
        registry.write_entry(entry)
        corrupt = registry.runs_dir(create=True) / "run_corrupt.json"
        corrupt.write_text("{not valid json", encoding="utf-8")

        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                side_effect=OSError("ps blocked\x1b[31m"),
            ),
            patch("praxist.cli.status._pid_is_alive", return_value=True),
        ):
            code, out, err = self._run(["status", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["run_id"], "run_probe_unknown")
        self.assertEqual(payload[0]["source"], "registry")
        self.assertEqual(payload[0]["state"], "unknown")
        self.assertIn("ps blocked", payload[0]["extras"]["probe_error"])
        self.assertIn("process probe failed", err)
        self.assertIn("run_corrupt.json", err)
        self.assertEqual(err.count("praxist status: registry warning:"), 2)
        self.assertNotIn("\x1b", err)


def _registry_entry_kwargs(**overrides: object) -> dict[str, object]:
    base = {
        "schema_version": 1,
        "run_id": "run_2026-05-19_status_demo",
        "pid": 8000,
        "parent_pid": 1,
        "run_dir": "/tmp/status_demo",
        "log_file": "/tmp/status_demo/log",
        "task_path": "/tmp/status_demo_task",
        "model": "claude-opus-4-7",
        "model_provider_ref": "model_provider:anthropic_messages",
        "runtime_ref": "agent_runtime:claude_sdk",
        "command": ("python", "-m", "praxist.run", "run", "--task-path", "/t"),
        "command_prefix": "python -m praxist.run",
        "started_at": "2026-05-19T08:00:00+00:00",
    }
    base.update(overrides)
    return base


class StatusMergeTest(unittest.TestCase):
    """Phase 2.5: registry + ps-scan merge with source/state tagging."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env_patch = patch.dict(os.environ, {"PRAXIST_STATE_DIR": self.tmp.name}, clear=False)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

    def _seed_registry(self, **overrides: object) -> str:
        from praxist.cli import registry

        entry = registry.RegistryEntry(**_registry_entry_kwargs(**overrides))
        registry.write_entry(entry)
        return entry.run_id

    def test_registry_row_enriched_with_task_and_model(self) -> None:
        from praxist.cli import status

        run_id = self._seed_registry()
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       1    00:42 python -m praxist.run run --task-path /t --run-dir /runs/seeded\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        registry_rows = [r for r in rows if r.source == status.SOURCE_REGISTRY]
        self.assertEqual(len(registry_rows), 1)
        row = registry_rows[0]
        self.assertEqual(row.run_id, run_id)
        self.assertEqual(row.task_path, "/tmp/status_demo_task")
        self.assertEqual(row.model, "claude-opus-4-7")
        self.assertEqual(row.state, "running")

    def test_daemonized_registry_row_with_missing_etime_stays_live(self) -> None:
        from praxist.cli import status

        run_id = self._seed_registry(
            command=(
                "python",
                "-m",
                "praxist.run",
                "run",
                "--task-path",
                "/t",
                "--run-dir",
                "/tmp/status_demo",
            ),
        )
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       0 python -m praxist.run run --task-path /t --run-dir /tmp/status_demo\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows(include_peer_health=False)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.run_id, run_id)
        self.assertEqual(row.source, status.SOURCE_REGISTRY)
        self.assertEqual(row.state, "running")
        self.assertEqual(row.ppid, 0)
        self.assertEqual(row.etime, "-")
        self.assertIn("python -m praxist.run", row.command)

    def test_live_process_with_stopped_registry_is_inconsistent(self) -> None:
        from praxist.cli import registry, status

        run_id = self._seed_registry(
            state=registry.STATE_STOPPED,
            extra={"startup_state": registry.STATE_RUNNING},
        )
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       1    00:42 python -m praxist.run run --task-path /t\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()

        row = next(item for item in rows if item.run_id == run_id)
        self.assertEqual(row.state, status.STATE_INCONSISTENT)
        self.assertEqual(
            row.extras,
            {"registry_state": registry.STATE_STOPPED, "process_state": registry.STATE_RUNNING},
        )
        self.assertIn(
            row,
            status._filter_rows(rows, run_id=None, task_path=None, active=True, latest=False),
        )

    def test_peer_health_can_be_disabled_without_filtering_status_rows(self) -> None:
        from praxist.cli import status

        self._seed_registry(run_id="run_target", run_dir="/tmp/run_target", pid=8100)
        self._seed_registry(run_id="run_unrelated", run_dir="/tmp/run_unrelated", pid=8200)

        with (
            patch("praxist.cli.status._read_ps_table", return_value={}),
            patch("praxist.cli.status._LAST_PS_ERROR", ""),
            patch(
                "praxist.cli.status._read_peer_health",
                side_effect=AssertionError("peer artifacts must not be scanned"),
            ) as peer_health,
        ):
            rows = status.collect_status_rows(include_peer_health=False)

        self.assertEqual({row.run_id for row in rows}, {"run_target", "run_unrelated"})
        self.assertTrue(all(row.peers == [] for row in rows))
        self.assertTrue(all(row.peer_health_summary is None for row in rows))
        peer_health.assert_not_called()

    def test_stale_registry_entry_when_pid_gone(self) -> None:
        from praxist.cli import status

        self._seed_registry(pid=99001)
        ps_output = "    PID    PPID  ELAPSED COMMAND\n"
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch("praxist.cli.status._pid_is_alive", return_value=False),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        stale_rows = [r for r in rows if r.source == status.SOURCE_STALE]
        self.assertEqual(len(stale_rows), 1)
        self.assertEqual(stale_rows[0].state, "stale")
        self.assertEqual(stale_rows[0].pid, 99001)

    def test_remote_registry_row_is_aggregated_without_local_pid_validation(self) -> None:
        from praxist.cli import status

        run_dir = Path(self.tmp.name) / "remote_run"
        run_dir.mkdir()
        (run_dir / "orchestrator_status.json").write_text(
            json.dumps(
                {
                    "current_generation": 7,
                    "findings_total": 31,
                    "updated_at": "2026-07-23T20:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        self._seed_registry(
            run_id="run_remote",
            pid=7000,
            run_dir=str(run_dir),
            extra={
                "hostname": "remote-host",
                "host_id": "remote-machine",
                "boot_id": "remote-boot",
            },
        )

        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed("    PID    PPID  ELAPSED COMMAND\n"),
            ),
            patch(
                "praxist.cli.registry.local_host_identity",
                return_value={
                    "hostname": "local-host",
                    "host_id": "local-machine",
                    "boot_id": "local-boot",
                },
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()

        self.assertEqual(len(rows), 1)
        remote = rows[0]
        self.assertEqual(remote.source, status.SOURCE_REMOTE)
        self.assertEqual(remote.state, "remote")
        self.assertEqual(remote.run_id, "run_remote")
        self.assertEqual(remote.generation, 7)
        self.assertEqual(remote.findings_total, 31)
        self.assertEqual(
            remote.extras,
            {"hostname": "remote-host", "boot_id": "remote-boot"},
        )

    def test_ps_only_row_when_process_not_in_registry(self) -> None:
        from praxist.cli import status

        # No registry entry seeded.
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8500       1    00:05 python -m praxist.run run --task-path /t\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        ps_only = [r for r in rows if r.source == status.SOURCE_PS_ONLY]
        self.assertEqual(len(ps_only), 1)
        self.assertEqual(ps_only[0].pid, 8500)
        self.assertIsNone(ps_only[0].run_id)
        self.assertEqual(ps_only[0].state, "running")

    def test_registry_pid_with_mismatched_cmdline_marks_stale(self) -> None:
        """TOCTOU: registry pid is alive but ps shows a different command."""
        from praxist.cli import status

        self._seed_registry(pid=8000)
        ps_output = "    PID    PPID  ELAPSED COMMAND\n   8000       1    00:00 /bin/bash -l\n"
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        # The 8000 row from ps does NOT match a Praxist pattern, so it's also
        # filtered from ps-only. The registry entry is reported as stale
        # because its command_prefix no longer matches.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source, status.SOURCE_STALE)

    def test_registry_command_matches_unquoted_ps_run_dir_with_spaces(self) -> None:
        from praxist.cli import registry, status

        run_dir = "/tmp/status run with spaces"
        entry = registry.RegistryEntry(
            **_registry_entry_kwargs(
                run_dir=run_dir,
                command=(
                    "python",
                    "-m",
                    "praxist.run",
                    "run",
                    "--run-dir",
                    run_dir,
                ),
            )
        )

        self.assertTrue(
            status.registry_command_matches(
                entry,
                "python -m praxist.run run "
                "--run-dir /tmp/status run with spaces --resume-from /tmp/source",
            )
        )
        self.assertFalse(
            status.registry_command_matches(
                entry,
                "python -m praxist.run run "
                "--run-dir /tmp/status run with spaces other --resume-from /tmp/source",
            )
        )
        self.assertFalse(
            status.registry_command_matches(
                entry,
                "python -m praxist.runner run "
                "--run-dir /tmp/status run with spaces --resume-from /tmp/source",
            )
        )
        self.assertFalse(
            status.registry_command_matches(
                entry,
                'python -m praxist.run run --run-dir "unterminated',
            )
        )
        equals_entry = registry.RegistryEntry(
            **_registry_entry_kwargs(
                run_dir=run_dir,
                command=(
                    "python",
                    "-m",
                    "praxist.run",
                    "run",
                    f"--run-dir={run_dir}",
                ),
            )
        )
        self.assertTrue(
            status.registry_command_matches(
                equals_entry,
                "python -m praxist.run run --run-dir='/tmp/status run with spaces'",
            )
        )

    def test_registry_command_matches_resolved_executable_alias_only(self) -> None:
        from praxist.cli import registry, status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "Python.framework" / "python3"
            executable.parent.mkdir()
            executable.write_text("", encoding="utf-8")
            alias = root / "venv" / "bin" / "python"
            alias.parent.mkdir(parents=True)
            alias.symlink_to(executable)
            run_dir = root / "run"
            entry = registry.RegistryEntry(
                **_registry_entry_kwargs(
                    run_dir=str(run_dir),
                    command_prefix=f"{alias} -m praxist.run",
                    command=(
                        str(alias),
                        "-m",
                        "praxist.run",
                        "run",
                        "--run-dir",
                        str(run_dir),
                    ),
                )
            )

            self.assertTrue(
                status.registry_command_matches(
                    entry,
                    f"{executable} -m praxist.run run --run-dir {run_dir}",
                )
            )
            self.assertFalse(
                status.registry_command_matches(
                    entry,
                    f"{executable} -m praxist.runner run --run-dir {run_dir}",
                )
            )
            self.assertFalse(
                status.registry_command_matches(
                    entry,
                    f"{executable} -m praxist.run run --run-dir {root / 'other'}",
                )
            )

    def test_executable_alias_resolution_handles_path_lookup_and_missing_paths(self) -> None:
        from praxist.cli import registry, status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "python3"
            executable.write_text("", encoding="utf-8")
            run_dir = root / "run"
            entry = registry.RegistryEntry(
                **_registry_entry_kwargs(
                    run_dir=str(run_dir),
                    command_prefix="python -m praxist.run",
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
            with patch("praxist.cli.status.shutil.which", return_value=str(executable)):
                self.assertTrue(
                    status.registry_command_matches(
                        entry,
                        f"{executable} -m praxist.run run --run-dir {run_dir}",
                    )
                )
                self.assertFalse(
                    status.registry_command_matches(
                        entry,
                        f"{root / 'other'} -m praxist.run run --run-dir {run_dir}",
                    )
                )

            self.assertTrue(
                status._executable_alias_matches(
                    str(root / "missing" / ".." / "tool"),
                    str(root / "tool"),
                )
            )

    def test_registry_pid_accepts_same_darwin_boot_with_executable_alias(self) -> None:
        from praxist.cli import registry, status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "Python.framework" / "python3"
            executable.parent.mkdir()
            executable.write_text("", encoding="utf-8")
            alias = root / "venv" / "bin" / "python"
            alias.parent.mkdir(parents=True)
            alias.symlink_to(executable)
            run_dir = root / "run"
            entry = registry.RegistryEntry(
                **_registry_entry_kwargs(
                    pid=4242,
                    run_dir=str(run_dir),
                    command_prefix=f"{alias} -m praxist.run",
                    command=(
                        str(alias),
                        "-m",
                        "praxist.run",
                        "run",
                        "--run-dir",
                        str(run_dir),
                    ),
                    extra={
                        "hostname": "darwin-host",
                        "boot_id": "darwin:1777777777:42",
                        "process_start_token": "ps:2026-08-05T00:00:00Z",
                    },
                )
            )
            live = {
                4242: (
                    1,
                    "00:10",
                    f"{executable} -m praxist.run run --run-dir {run_dir}",
                )
            }
            with (
                patch(
                    "praxist.cli.registry.local_host_identity",
                    return_value={
                        "hostname": "darwin-host",
                        "boot_id": "kern.boottime: { sec=1777777777, usec=000043 }",
                    },
                ),
                patch(
                    "praxist.cli.registry.process_start_token",
                    return_value="ps:2026-08-05T00:00:00Z",
                ),
            ):
                self.assertEqual(status._validate_registry_pid(entry, live), live[4242])

    def test_table_includes_source_state_and_run_id_columns(self) -> None:
        from praxist.cli import status

        run_id = self._seed_registry()
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       1    00:42 python -m praxist.run run --task-path /t\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        table = status.format_status_table(rows)
        self.assertIn("SOURCE", table)
        self.assertIn("STATE", table)
        self.assertIn("RUN_ID", table)
        self.assertIn("registry", table)
        self.assertIn(run_id, table)

    def test_json_output_includes_new_fields_for_registry_row(self) -> None:
        from praxist.cli import main

        run_id = self._seed_registry()
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       1    00:42 python -m praxist.run run --task-path /t\n"
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with (
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
                patch(
                    "praxist.cli.status.subprocess.run",
                    return_value=_fake_ps_completed(ps_output),
                ),
                patch.object(os, "getpid", return_value=99998),
                patch.object(os, "getppid", return_value=99999),
            ):
                main(["status", "--json"])
        except SystemExit:
            pass
        payload = json.loads(stdout.getvalue())
        registry_entries = [r for r in payload if r["source"] == "registry"]
        self.assertEqual(len(registry_entries), 1)
        entry = registry_entries[0]
        self.assertEqual(entry["run_id"], run_id)
        self.assertEqual(entry["task_path"], "/tmp/status_demo_task")
        self.assertEqual(entry["model"], "claude-opus-4-7")


class StatusHelpersTest(unittest.TestCase):
    """Direct tests for the smaller helpers in ``praxist.cli.status``."""

    def test_pid_is_alive_returns_false_for_nonpositive_pid(self) -> None:
        from praxist.cli import status

        self.assertFalse(status._pid_is_alive(0))
        self.assertFalse(status._pid_is_alive(-5))

    def test_pid_is_alive_handles_process_lookup_and_permission(self) -> None:
        from praxist.cli import status

        with patch("praxist.cli.status.os.kill", side_effect=ProcessLookupError):
            self.assertFalse(status._pid_is_alive(1234))
        with patch("praxist.cli.status.os.kill", side_effect=PermissionError):
            # EPERM means the PID exists; we cannot signal it but it is alive.
            self.assertTrue(status._pid_is_alive(1234))
        with patch("praxist.cli.status.os.kill", side_effect=OSError):
            self.assertFalse(status._pid_is_alive(1234))
        with patch("praxist.cli.status.os.kill", return_value=None):
            self.assertTrue(status._pid_is_alive(1234))

    def test_truncate_caps_long_strings(self) -> None:
        from praxist.cli import status

        self.assertEqual(status._truncate("short", 10), "short")
        truncated = status._truncate("a" * 200, 80)
        self.assertEqual(len(truncated), 80)
        self.assertTrue(truncated.endswith("…"))

    def test_read_ps_table_returns_empty_on_subprocess_timeout(self) -> None:
        from praxist.cli import status

        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="ps", timeout=10),
            ),
        ):
            self.assertEqual(status._read_ps_table(), {})

    def test_read_ps_table_skips_lines_with_invalid_pids(self) -> None:
        from praxist.cli import status

        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "  abc      1    00:01 not_a_pid_line\n"  # invalid pid
            "  100      1    00:01 launchd\n"
            "  short_line_no_command\n"  # too few fields
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
        ):
            rows = status._read_ps_table()
        # Only the launchd row survives parsing.
        self.assertIn(100, rows)
        self.assertEqual(len(rows), 1)

    def test_validate_registry_pid_when_alive_but_not_in_ps(self) -> None:
        """Permission boundary: registry pid is alive but ``ps`` does not list it."""
        from praxist.cli import registry, status

        entry = registry.RegistryEntry(**_registry_entry_kwargs(pid=99001))
        with patch("praxist.cli.status._pid_is_alive", return_value=True):
            result = status._validate_registry_pid(entry, {})
        self.assertIsNone(result)

    def test_validate_registry_pid_rejects_old_epoch_and_reused_process_token(self) -> None:
        from praxist.cli import registry, status

        entry = registry.RegistryEntry(**_registry_entry_kwargs())
        command = "python -m praxist.run run --task-path /t"
        live = {entry.pid: (1, "00:10", command)}

        with (
            patch("praxist.cli.status.entry_process_epoch_matches", return_value=False),
            patch("praxist.cli.status.process_identity_matches") as identity,
        ):
            self.assertIsNone(status._validate_registry_pid(entry, live))
        identity.assert_not_called()

        with (
            patch("praxist.cli.status.entry_process_epoch_matches", return_value=True),
            patch("praxist.cli.status.process_identity_matches", return_value=False),
        ):
            self.assertIsNone(status._validate_registry_pid(entry, live))

    def test_terminal_registry_state_prefers_lifecycle_and_run_artifacts(self) -> None:
        from praxist.cli import registry, status

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run_summary.json").write_text("[]", encoding="utf-8")
            (run_dir / "run.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            entry = registry.RegistryEntry(**_registry_entry_kwargs(run_dir=str(run_dir)))
            self.assertEqual(status._terminal_registry_state(entry), status.STATE_COMPLETED)

            (run_dir / "run_summary.json").write_text(
                json.dumps({"status": "error"}),
                encoding="utf-8",
            )
            self.assertEqual(status._terminal_registry_state(entry), status.STATE_FAILED)

            startup_failed = registry.RegistryEntry(
                **_registry_entry_kwargs(
                    run_dir=str(run_dir),
                    extra={"startup_state": status.STATE_FAILED},
                )
            )
            self.assertEqual(
                status._terminal_registry_state(startup_failed),
                status.STATE_FAILED,
            )

            stopped = registry.RegistryEntry(
                **_registry_entry_kwargs(
                    run_dir=str(run_dir),
                    state=registry.STATE_STOPPED,
                )
            )
            self.assertEqual(
                status._terminal_registry_state(stopped),
                registry.STATE_STOPPED,
            )

    def test_status_filters_normalize_paths_and_compose_active_latest(self) -> None:
        from praxist.cli import status

        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            task_dir.mkdir()
            task_alias = task_dir / ".." / "task"

            def row(
                pid: int,
                *,
                source: str,
                state: str,
                run_id: str | None,
                task_path: str | None,
                started_at: str,
            ) -> status.StatusRow:
                return status.StatusRow(
                    pid=pid,
                    ppid=1,
                    etime="00:01",
                    command="python -m praxist.run run",
                    run_dir=None,
                    source=source,
                    state=state,
                    run_id=run_id,
                    task_path=task_path,
                    started_at=started_at,
                )

            rows = [
                row(
                    100,
                    source=status.SOURCE_REGISTRY,
                    state="running",
                    run_id="run_old",
                    task_path=str(task_alias),
                    started_at="2026-07-23T10:00:00+00:00",
                ),
                row(
                    200,
                    source=status.SOURCE_REGISTRY,
                    state="starting",
                    run_id="run_new",
                    task_path=str(task_dir),
                    started_at="2026-07-23T11:00:00+00:00",
                ),
                row(
                    300,
                    source=status.SOURCE_PS_ONLY,
                    state="running",
                    run_id=None,
                    task_path=None,
                    started_at="",
                ),
                row(
                    400,
                    source=status.SOURCE_REMOTE,
                    state="remote",
                    run_id="run_remote",
                    task_path=str(task_dir),
                    started_at="2026-07-23T12:00:00+00:00",
                ),
                row(
                    500,
                    source=status.SOURCE_STALE,
                    state="stopped",
                    run_id="run_stopped",
                    task_path=str(task_dir),
                    started_at="2026-07-23T13:00:00+00:00",
                ),
            ]

            by_run_id = status._filter_rows(
                rows,
                run_id="run_remote",
                task_path=None,
                active=False,
                latest=False,
            )
            self.assertEqual([item.pid for item in by_run_id], [400])

            active = status._filter_rows(
                rows,
                run_id=None,
                task_path=None,
                active=True,
                latest=False,
            )
            self.assertEqual([item.pid for item in active], [100, 200, 300])

            latest = status._filter_rows(
                rows,
                run_id=None,
                task_path=str(task_dir),
                active=True,
                latest=True,
            )
            self.assertEqual([item.pid for item in latest], [200])

    def test_self_ancestor_pids_break_on_cycle(self) -> None:
        from praxist.cli import status

        # Pathological ancestor cycle (200 ↔ 300): the loop must terminate.
        rows = {200: (300, "00:01", "p1"), 300: (200, "00:01", "p2")}
        with (
            patch.object(os, "getpid", return_value=100),
            patch.object(os, "getppid", return_value=200),
        ):
            excluded = status._self_ancestor_pids(rows)
        self.assertIn(200, excluded)
        self.assertIn(300, excluded)

    def test_public_aliases_round_trip_to_private_helpers(self) -> None:
        from praxist.cli import status

        with patch("praxist.cli.status._pid_is_alive", return_value=True) as alive_mock:
            self.assertTrue(status.pid_is_alive(42))
            alive_mock.assert_called_once_with(42)
        with (
            patch("praxist.cli.status._read_ps_table", return_value={1: (0, "x", "y")}),
        ):
            self.assertEqual(status.read_ps_table(), {1: (0, "x", "y")})
        with (
            patch("praxist.cli.status._self_ancestor_pids", return_value={7}) as anc_mock,
        ):
            self.assertEqual(status.self_ancestor_pids({}), {7})
            anc_mock.assert_called_once_with({})
        self.assertIsInstance(status.praxist_process_regexes(), list)

    def test_status_peer_health_and_progress_helpers_degrade_cleanly(self) -> None:
        from praxist.cli import status

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            os.mkdir(run_dir)
            with open(
                os.path.join(run_dir, "orchestrator_status.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(["not", "a", "mapping"], fh)
            self.assertEqual(status._read_orchestrator_progress(run_dir), (None, None, None))

            task_dir = os.path.join(tmp, "task")
            os.mkdir(task_dir)
            with open(os.path.join(task_dir, "task.yaml"), "w", encoding="utf-8") as fh:
                fh.write("task_id: demo\n")
            with patch("praxist.cli.status.load_task_spec", side_effect=RuntimeError("bad")):
                self.assertIsNone(status._load_task_spec_for_status(task_dir))

            with patch(
                "praxist.cli.status.collect_peer_memory_health",
                side_effect=RuntimeError("bad"),
            ):
                peer_health = status._read_peer_health(run_dir, task_dir, 3)

        self.assertEqual(peer_health.generation_id, 3)
        self.assertEqual(peer_health.summary, {"red": 0, "yellow": 0, "green": 0})
        self.assertEqual(peer_health.peers, [])
        self.assertEqual(status._short_updated("not-a-timestamp"), "not-a-timestamp")


class StatusProgressColumnsTest(unittest.TestCase):
    """``praxist status`` surfaces orchestrator progress (#165).

    The fields come from ``<run_dir>/orchestrator_status.json``, which
    the research_loop ``OrchestratorStatusWriter`` writes periodically.
    Live, stale, and ps-only rows all benefit when the JSON exists.
    """

    def setUp(self) -> None:
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env_patch = patch.dict(os.environ, {"PRAXIST_STATE_DIR": self.tmp.name}, clear=False)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        self.run_dir = Path(self.tmp.name) / "run_165"
        self.run_dir.mkdir()

    def _write_orchestrator_status(self, **overrides: object) -> None:
        from pathlib import Path

        payload: dict[str, object] = {
            "run_started_at": "2026-05-22T01:00:00+00:00",
            "updated_at": "2026-05-22T01:30:15+00:00",
            "run_dir": str(self.run_dir),
            "task_id": "demo",
            "task_name": "demo",
            "current_generation": 2,
            "max_generations": 5,
            "cohort_size": 5,
            "strategy": "auto",
            "generations_completed": 1,
            "findings_total": 17,
            "variants_total": 4,
            "frontier_candidates": 2,
        }
        payload.update(overrides)
        (Path(self.run_dir) / "orchestrator_status.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def _seed_registry_at_run_dir(self, **overrides: object) -> str:
        from praxist.cli import registry

        kwargs = _registry_entry_kwargs(run_dir=str(self.run_dir))
        kwargs.update(overrides)
        entry = registry.RegistryEntry(**kwargs)
        registry.write_entry(entry)
        return entry.run_id

    def test_live_row_reads_generation_findings_and_updated_at(self) -> None:
        from praxist.cli import status

        self._write_orchestrator_status()
        self._seed_registry_at_run_dir()
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       1    00:42 python -m praxist.run run --task-path /t\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        live = [r for r in rows if r.source == status.SOURCE_REGISTRY]
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].generation, 2)
        self.assertEqual(live[0].findings_total, 17)
        self.assertEqual(live[0].updated_at, "2026-05-22T01:30:15+00:00")

    def test_final_status_wins_and_malformed_final_falls_back(self) -> None:
        from praxist.cli import status

        self._write_orchestrator_status(
            current_generation=2,
            generations_completed=1,
            findings_total=17,
            exit_condition="in_progress",
        )
        final_path = self.run_dir / "orchestrator_status.final.json"
        final_path.write_text(
            json.dumps(
                {
                    "current_generation": 3,
                    "generations_completed": 4,
                    "findings_total": 29,
                    "updated_at": "2026-05-22T02:30:15+00:00",
                    "exit_condition": "completed",
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            status._read_orchestrator_progress(str(self.run_dir)),
            (4, 29, "2026-05-22T02:30:15+00:00"),
        )

        final_path.write_text(
            json.dumps(
                {
                    "current_generation": 3,
                    "generations_completed": "not-an-integer",
                    "findings_total": 29,
                    "updated_at": "2026-05-22T02:30:15+00:00",
                    "exit_condition": "completed",
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(
            status._read_orchestrator_progress(str(self.run_dir)),
            (3, 29, "2026-05-22T02:30:15+00:00"),
        )

        final_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(
            status._read_orchestrator_progress(str(self.run_dir)),
            (2, 17, "2026-05-22T01:30:15+00:00"),
        )

    def test_newer_resumed_periodic_status_wins_over_previous_final(self) -> None:
        from praxist.cli import status

        final_path = self.run_dir / "orchestrator_status.final.json"
        final_path.write_text(
            json.dumps(
                {
                    "generations_completed": 4,
                    "findings_total": 20,
                    "updated_at": "2026-05-22T02:00:00+00:00",
                    "exit_condition": "completed",
                }
            ),
            encoding="utf-8",
        )
        self._write_orchestrator_status(
            current_generation=4,
            generations_completed=4,
            findings_total=23,
            updated_at="2026-05-22T02:10:00+00:00",
            exit_condition="in_progress",
        )
        periodic_path = self.run_dir / "orchestrator_status.json"
        os.utime(final_path, ns=(1_000_000_000, 1_000_000_000))
        os.utime(periodic_path, ns=(2_000_000_000, 2_000_000_000))

        self.assertEqual(
            status._read_orchestrator_progress(str(self.run_dir)),
            (4, 23, "2026-05-22T02:10:00+00:00"),
        )

    def test_missing_orchestrator_status_yields_none_fields(self) -> None:
        """No ``orchestrator_status.json`` (e.g. run hasn't booted yet)
        → the new fields are ``None`` and render as ``-`` in the
        table, not ``0`` — preserving the contract that ``None``
        means *unknown*, not *zero*."""
        from praxist.cli import status

        # Do NOT write orchestrator_status.json this time.
        self._seed_registry_at_run_dir()
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       1    00:42 python -m praxist.run run --task-path /t\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        row = next(r for r in rows if r.source == status.SOURCE_REGISTRY)
        self.assertIsNone(row.generation)
        self.assertIsNone(row.findings_total)
        self.assertIsNone(row.updated_at)

    def test_stale_row_surfaces_last_known_progress(self) -> None:
        """A crashed run still has the last snapshot on disk — the
        operator should see where it got to (#165 acceptance: stale
        rows surface what the orchestrator wrote before death)."""
        from praxist.cli import status

        self._write_orchestrator_status(current_generation=4, findings_total=29)
        self._seed_registry_at_run_dir()
        # Empty ps table — registry entry is stale.
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed("    PID    PPID  ELAPSED COMMAND\n"),
            ),
            patch("praxist.cli.status._pid_is_alive", return_value=False),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        stale = [r for r in rows if r.source == status.SOURCE_STALE]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].generation, 4)
        self.assertEqual(stale[0].findings_total, 29)

    def test_ps_only_row_reads_status_via_extracted_run_dir(self) -> None:
        """For ps-only rows, ``run_dir`` is best-effort regex-extracted
        from the command line — when it succeeds, the progress columns
        get populated too."""
        from praxist.cli import status

        self._write_orchestrator_status()
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            f"   9000       1    00:42 python -m praxist.run run --task-path /t "
            f"--run-dir {self.run_dir}\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        ps_only = [r for r in rows if r.source == status.SOURCE_PS_ONLY]
        self.assertEqual(len(ps_only), 1)
        self.assertEqual(ps_only[0].generation, 2)
        self.assertEqual(ps_only[0].findings_total, 17)

    def test_malformed_json_collapses_to_none(self) -> None:
        """Defence in depth: garbage JSON shouldn't crash the listing."""
        from pathlib import Path

        from praxist.cli import status

        (Path(self.run_dir) / "orchestrator_status.json").write_text(
            "{not really json", encoding="utf-8"
        )
        self._seed_registry_at_run_dir()
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       1    00:42 python -m praxist.run run --task-path /t\n"
        )
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
        ):
            rows = status.collect_status_rows()
        row = next(r for r in rows if r.source == status.SOURCE_REGISTRY)
        self.assertIsNone(row.generation)
        self.assertIsNone(row.findings_total)
        self.assertIsNone(row.updated_at)

    def test_table_format_includes_new_columns(self) -> None:
        from praxist.cli import status

        row = status.StatusRow(
            pid=8000,
            ppid=1,
            etime="00:42",
            command="python -m praxist.run run",
            run_dir="/tmp/r",
            source=status.SOURCE_REGISTRY,
            state="running",
            run_id="run_demo",
            generation=3,
            findings_total=15,
            updated_at="2026-05-22T01:30:15+00:00",
        )
        out = status.format_status_table([row])
        # Headers
        self.assertIn("GEN", out)
        self.assertIn("FINDINGS", out)
        self.assertIn("UPDATED", out)
        # Values
        self.assertIn(" 3 ", out)  # GEN cell rendered as "3" (zero-padded by column width)
        self.assertIn("15", out)
        # Compact timestamp form
        self.assertIn("2026-05-22 01:30:15", out)

    def test_table_renders_dash_when_progress_missing(self) -> None:
        from praxist.cli import status

        row = status.StatusRow(
            pid=8000,
            ppid=1,
            etime="00:42",
            command="python -m praxist.run run",
            run_dir=None,
            source=status.SOURCE_REGISTRY,
            state="running",
            run_id="run_no_progress",
            generation=None,
            findings_total=None,
            updated_at=None,
        )
        out = status.format_status_table([row])
        # All three new columns should render "-" (not "0" / "None").
        lines = out.splitlines()
        self.assertGreaterEqual(len(lines), 2)
        # The data row is the line that contains the run_id.
        data_line = next(line for line in lines if "run_no_progress" in line)
        # Three "-" placeholders in the progress columns. We split on
        # whitespace and check the relevant cells, which are columns
        # 4, 5, 6 (GEN / FINDINGS / UPDATED) in a 0-indexed view.
        cells = data_line.split()
        # cells: PID(8000) SOURCE(registry) STATE(running) AGE(00:42) GEN FINDINGS UPDATED RUN_ID ...
        self.assertEqual(cells[4], "-")
        self.assertEqual(cells[5], "-")
        self.assertEqual(cells[6], "-")

    def test_json_output_includes_progress_fields(self) -> None:
        from praxist.cli import main, status

        self._write_orchestrator_status()
        self._seed_registry_at_run_dir()
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       1    00:42 python -m praxist.run run --task-path /t\n"
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            try:
                main(["status", "--json"])
            except SystemExit as exc:
                self.assertEqual(exc.code, 0)
        payload = json.loads(stdout.getvalue())
        live_rows = [r for r in payload if r["source"] == status.SOURCE_REGISTRY]
        self.assertEqual(len(live_rows), 1)
        self.assertEqual(live_rows[0]["generation"], 2)
        self.assertEqual(live_rows[0]["findings_total"], 17)
        self.assertEqual(live_rows[0]["updated_at"], "2026-05-22T01:30:15+00:00")

    def test_status_json_and_table_include_peer_health(self) -> None:
        from pathlib import Path

        from praxist.cli import main

        task_dir = Path(self.tmp.name) / "task"
        task_dir.mkdir()
        (task_dir / "task.yaml").write_text(
            "\n".join(
                [
                    "task_id: demo",
                    "task_name: Demo",
                    "evaluation:",
                    "  primary_metric: score",
                    "  direction: maximize",
                    "baselines:",
                    "  - name: baseline",
                    "    metric_name: score",
                    "    metric_value: 0.5",
                ]
            ),
            encoding="utf-8",
        )
        self._write_orchestrator_status(current_generation=0, findings_total=1)
        peer_memory = self.run_dir / "gen_0" / "peers" / "gen0_peer0" / "memory"
        peer_memory.mkdir(parents=True)
        (peer_memory / "peer_state.yaml").write_text(
            "\n".join(
                [
                    "peer_id: gen0_peer0",
                    "generation_id: 0",
                    "research_state: evaluating",
                    "active_variant: gen0_peer0_variant",
                    "last_session_id: session_001",
                    "last_session_success: true",
                ]
            ),
            encoding="utf-8",
        )
        result_dir = self.run_dir / "results" / "gen0_peer0_variant"
        result_dir.mkdir(parents=True)
        (result_dir / "result_summary.json").write_text(
            json.dumps({"variant_name": "gen0_peer0_variant", "score": 0.7}),
            encoding="utf-8",
        )
        self._seed_registry_at_run_dir(task_path=str(task_dir))
        ps_output = (
            "    PID    PPID  ELAPSED COMMAND\n"
            "   8000       1    00:42 python -m praxist.run run --task-path /t\n"
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            try:
                main(["status", "--json"])
            except SystemExit as exc:
                self.assertEqual(exc.code, 0)
        payload = json.loads(stdout.getvalue())
        row = next(item for item in payload if item["source"] == "registry")
        self.assertEqual(row["peer_health_summary"], {"red": 0, "yellow": 0, "green": 1})
        self.assertEqual(row["peers"][0]["peer_id"], "gen0_peer0")
        self.assertEqual(row["peers"][0]["health"], "green")

        table_out = io.StringIO()
        with (
            patch("praxist.cli.status.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.status.subprocess.run",
                return_value=_fake_ps_completed(ps_output),
            ),
            patch.object(os, "getpid", return_value=99998),
            patch.object(os, "getppid", return_value=99999),
            redirect_stdout(table_out),
            redirect_stderr(io.StringIO()),
        ):
            try:
                main(["status"])
            except SystemExit as exc:
                self.assertEqual(exc.code, 0)
        self.assertIn("PEERS", table_out.getvalue())
        self.assertIn("R0/Y0/G1", table_out.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
