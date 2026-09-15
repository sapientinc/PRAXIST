"""Tests for ``praxist stop`` — CLI lifecycle Phases 3 + 4."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


def _entry_kwargs(**overrides: object) -> dict[str, object]:
    base = {
        "schema_version": 1,
        "run_id": "run_stop_demo",
        "pid": 5000,
        "parent_pid": 1,
        "run_dir": "/tmp/stop_demo",
        "log_file": "/tmp/stop_demo/log",
        "task_path": "/tmp/task",
        "model": "claude-opus-4-7",
        "model_provider_ref": "model_provider:anthropic_messages",
        "runtime_ref": "agent_runtime:claude_sdk",
        "command": ("python", "-m", "praxist.run", "run", "--task-path", "/t"),
        "command_prefix": "python -m praxist.run",
        "started_at": "2026-05-19T09:00:00+00:00",
    }
    base.update(overrides)
    return base


def _write_entry(**overrides: object) -> str:
    from praxist.cli import registry

    if "run_dir" not in overrides:
        state_dir = Path(os.environ["PRAXIST_STATE_DIR"])
        run_id = str(overrides.get("run_id") or "run_stop_demo")
        run_dir = state_dir / "run_dirs" / run_id
        overrides["run_dir"] = str(run_dir)
        overrides.setdefault("log_file", str(run_dir / "log"))
    entry = registry.RegistryEntry(**_entry_kwargs(**overrides))
    Path(entry.run_dir).mkdir(parents=True, exist_ok=True)
    registry.write_entry(entry)
    return entry.run_id


class _StateDirMixin:
    """Set up an isolated PRAXIST_STATE_DIR for each test."""

    def setUp(self) -> None:  # type: ignore[override]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env_patch = patch.dict(os.environ, {"PRAXIST_STATE_DIR": self.tmp.name}, clear=False)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        self._token_patch = patch(
            "praxist.cli.stop.process_start_token",
            side_effect=lambda pid: f"proc:{pid}",
        )
        self._token_patch.start()
        self.addCleanup(self._token_patch.stop)


class StopRunTest(_StateDirMixin, unittest.TestCase):
    """Selective ``praxist stop <run_id>`` paths."""

    def test_live_pid_signalled_then_marked_stopped(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry()
        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
            5001: (5000, "00:09", "python /t/variants/v0/train.py"),
        }
        killed: list[tuple[int, int]] = []
        alive_pids = {5000, 5001}

        def fake_kill(pid: int, sig: int) -> None:
            killed.append((pid, sig))
            # SIGTERM clears the alive flag on next check (simulate fast exit).
            if pid in alive_pids:
                alive_pids.discard(pid)

        def fake_alive(pid: int) -> bool:
            return pid in alive_pids

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", side_effect=fake_alive),
            patch("praxist.cli.stop.os.kill", side_effect=fake_kill),
            patch("praxist.cli.stop.time.sleep"),
            patch(
                "praxist.cli.stop._utc_now_iso",
                return_value="2026-05-19T09:30:00+00:00",
            ),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.05)

        # Root + descendant signalled.
        self.assertEqual(outcome.matched_pids, [5000])
        self.assertEqual(outcome.descendant_pids, [5001])
        # SIGTERM sent to both.
        sigterms = [p for p, s in killed if s == 15]
        self.assertEqual(sorted(sigterms), [5000, 5001])
        # No SIGKILL needed because both exited.
        self.assertEqual(outcome.killed_pids, [])
        self.assertEqual(outcome.remaining_pids, [])
        # Registry marked stopped.
        on_disk = registry.read_entry(run_id)
        self.assertEqual(on_disk.state, registry.STATE_STOPPED)
        self.assertEqual(on_disk.stopped_at, "2026-05-19T09:30:00+00:00")

    def test_stop_writes_shutdown_fence_before_process_discovery(self) -> None:
        from praxist.cli import stop

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            run_id = _write_entry(run_dir=str(run_dir))

            def snapshot_after_fence():
                self.assertTrue((run_dir / "ORCHESTRATOR_SHUTDOWN").is_file())
                return (
                    {5000: (1, "00:10", "python -m praxist.run run --task-path /t")},
                    {5000: "proc:5000"},
                )

            with (
                patch(
                    "praxist.cli.stop._read_identity_bound_ps_snapshot",
                    side_effect=snapshot_after_fence,
                ),
                patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
                patch("praxist.cli.stop.pid_is_alive", return_value=False),
                patch("praxist.cli.stop.os.kill"),
                patch("praxist.cli.stop._drain_late_run_processes", return_value={}),
            ):
                outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertEqual(outcome.remaining_pids, [])

    def test_stop_refuses_to_signal_when_admission_cannot_close(self) -> None:
        from praxist.cli import registry, stop

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            run_id = _write_entry(run_dir=str(run_dir))
            with (
                patch(
                    "praxist.cli.stop._write_run_shutdown_fence",
                    return_value=False,
                ),
                patch("praxist.cli.stop._read_identity_bound_ps_snapshot") as snapshot,
                patch("praxist.cli.stop.os.kill") as kill,
                self.assertRaisesRegex(stop.StopError, "could not close experiment admission"),
            ):
                stop.stop_run(run_id=run_id, grace_seconds=0.0)

        snapshot.assert_not_called()
        kill.assert_not_called()
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_scheduler_freeze_allows_stop_when_shutdown_file_is_unwritable(self) -> None:
        from praxist.cli import registry, stop

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            endpoint = run_dir / "resource_scheduler" / "endpoint.json"
            endpoint.parent.mkdir(parents=True)
            endpoint.write_text("{}\n", encoding="utf-8")
            run_id = _write_entry(run_dir=str(run_dir))
            scheduler_client = (
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client.freeze_all_for_run"
            )
            with (
                patch(
                    "praxist.cli.stop._write_run_shutdown_fence",
                    return_value=False,
                ),
                patch(scheduler_client, return_value=True),
                patch(
                    "praxist.cli.stop._read_identity_bound_ps_snapshot",
                    return_value=({}, {}),
                ),
                patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
                patch("praxist.cli.stop._drain_late_run_processes", return_value={}),
                patch("praxist.cli.stop.pid_is_alive", return_value=False),
            ):
                outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertEqual(outcome.remaining_pids, [])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)

    def test_stop_drains_process_forked_after_initial_snapshot(self) -> None:
        from praxist.cli import registry, stop

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            run_id = _write_entry(run_dir=str(run_dir))
            alive = {5000}
            signals: list[tuple[int, int]] = []
            late_forked = False

            def fake_kill(pid: int, sig: int) -> None:
                nonlocal late_forked
                signals.append((pid, sig))
                alive.discard(pid)
                if pid == 5000 and sig == 15 and not late_forked:
                    alive.add(6001)
                    late_forked = True

            def discover_late(_run_dir: Path):
                return (
                    ({6001: "proc:6001"} if 6001 in alive else {}),
                    {},
                )

            with (
                patch(
                    "praxist.cli.stop._read_identity_bound_ps_snapshot",
                    return_value=(
                        {5000: (1, "00:10", "python -m praxist.run run --task-path /t")},
                        {5000: "proc:5000"},
                    ),
                ),
                patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
                patch(
                    "praxist.cli.stop._discover_run_owned_processes",
                    side_effect=discover_late,
                ),
                patch("praxist.cli.stop.pid_is_alive", side_effect=lambda pid: pid in alive),
                patch("praxist.cli.stop.os.kill", side_effect=fake_kill),
                patch("praxist.cli.stop.time.sleep"),
            ):
                outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertIn((6001, 15), signals)
        self.assertIn(6001, outcome.descendant_pids)
        self.assertEqual(outcome.remaining_pids, [])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)

    def test_late_drain_escalates_persistent_owned_process(self) -> None:
        from praxist.cli import stop

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            outcome = stop.StopOutcome(run_id="run")
            with (
                patch.object(stop, "_LATE_DRAIN_MAX_SCANS", 3),
                patch.object(stop, "_LATE_DRAIN_SCAN_SECONDS", 0),
                patch.object(stop, "_LATE_DRAIN_TERM_SCANS", 2),
                patch.object(
                    stop,
                    "_discover_run_owned_processes",
                    return_value=({41: "proc:41"}, {}),
                ),
                patch.object(
                    stop,
                    "_live_process_instances",
                    side_effect=[{41}, {41}, {41}, set(), set(), set(), set()],
                ),
                patch.object(stop, "_signal_process_groups") as signal_groups,
                patch.object(stop, "_signal_set") as signal_set,
                patch.object(stop.time, "sleep"),
            ):
                remaining = stop._drain_late_run_processes({"run": run_dir}, outcome)

        self.assertEqual(remaining, {})
        self.assertTrue(
            any(call.args[1] == stop.signal.SIGKILL for call in signal_groups.call_args_list)
        )
        self.assertTrue(
            any(call.args[1] == stop.signal.SIGKILL for call in signal_set.call_args_list)
        )

    def test_late_drain_final_verification_reports_new_survivor(self) -> None:
        from praxist.cli import stop

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            outcome = stop.StopOutcome(run_id="run")
            with (
                patch.object(stop, "_LATE_DRAIN_MAX_SCANS", 1),
                patch.object(stop, "_LATE_DRAIN_SCAN_SECONDS", 0),
                patch.object(
                    stop,
                    "_discover_run_owned_processes",
                    return_value=({42: "proc:42"}, {}),
                ),
                patch.object(
                    stop,
                    "_live_process_instances",
                    side_effect=[{42}, {42}, set(), set(), {42}],
                ),
                patch.object(stop, "_signal_process_groups"),
                patch.object(stop, "_signal_set"),
                patch.object(stop, "pid_is_alive", return_value=False),
                patch.object(stop.time, "sleep"),
            ):
                remaining = stop._drain_late_run_processes({"run": run_dir}, outcome)

        self.assertEqual(remaining, {"run": {42}})
        self.assertEqual(outcome.remaining_pids, [42])

    def test_shutdown_fence_and_process_identity_boundary_fail_closed(self) -> None:
        from praxist.cli import stop

        outcome = stop.StopOutcome(run_id="run")
        with (
            patch.object(Path, "exists", return_value=False),
            patch.object(Path, "write_text", side_effect=OSError("read-only")),
        ):
            self.assertFalse(stop._write_run_shutdown_fence(Path("/run"), outcome, source="test"))
        self.assertIn("read-only", outcome.warnings[0])

        raw = b"PRAXIST_RUN_DIR=bad\0AUTO_RESEARCH_RUN_DIR=/tmp/run\0"
        with (
            patch.object(Path, "read_bytes", return_value=raw),
            patch.object(stop.os, "fsdecode", side_effect=[UnicodeError(), "/tmp/run"]),
        ):
            self.assertEqual(
                stop._run_dir_from_process_environment(123),
                Path("/tmp/run").resolve(),
            )

    def test_pidfd_helpers_use_exact_process_handle_when_available(self) -> None:
        from praxist.cli import stop

        with patch.object(stop.os, "pidfd_open", return_value=17, create=True):
            self.assertEqual(stop._open_process_handle(41), 17)
        with patch.object(stop.os, "pidfd_open", return_value=object(), create=True):
            self.assertIsNone(stop._open_process_handle(41))
        with patch.object(stop.os, "pidfd_open", side_effect=OSError("gone"), create=True):
            self.assertIsNone(stop._open_process_handle(41))

        with patch.object(stop.signal, "pidfd_send_signal", create=True) as sender:
            stop._signal_process_handle(17, stop.signal.SIGTERM)
        sender.assert_called_once_with(17, stop.signal.SIGTERM, None, 0)

    def test_pidfd_helpers_use_libc_fallback_when_stdlib_api_is_missing(self) -> None:
        import ctypes

        from praxist.cli import stop

        class FakeFunction:
            def __init__(self, result: int) -> None:
                self.result = result
                self.argtypes = None
                self.restype = None

            def __call__(self, *_args: object) -> int:
                return self.result

        class FakeLibc:
            def __init__(self, *, open_result: int = 29, signal_result: int = 0) -> None:
                self.pidfd_open = FakeFunction(open_result)
                self.pidfd_send_signal = FakeFunction(signal_result)

        libc = FakeLibc()
        with (
            patch.object(stop.os, "pidfd_open", None, create=True),
            patch.object(stop.sys, "platform", "linux"),
            patch.object(ctypes, "CDLL", return_value=libc),
        ):
            self.assertEqual(stop._open_process_handle(41), 29)

        with (
            patch.object(stop.os, "pidfd_open", None, create=True),
            patch.object(stop.sys, "platform", "darwin"),
        ):
            self.assertIsNone(stop._open_process_handle(41))

        with (
            patch.object(stop.signal, "pidfd_send_signal", None, create=True),
            patch.object(ctypes, "CDLL", return_value=libc),
        ):
            stop._signal_process_handle(29, stop.signal.SIGTERM)

        failing_libc = FakeLibc(signal_result=-1)
        with (
            patch.object(stop.signal, "pidfd_send_signal", None, create=True),
            patch.object(ctypes, "CDLL", return_value=failing_libc),
            patch.object(ctypes, "get_errno", return_value=1),
            self.assertRaises(OSError),
        ):
            stop._signal_process_handle(29, stop.signal.SIGTERM)

    def test_scheduler_state_accepts_cross_platform_process_identity(self) -> None:
        from praxist.cli import stop
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        entry = protected_pids.ProtectedEntry(
            pid=6101,
            pgid=6101,
            pid_start_time="ps:Mon Aug 11 12:00:00 2026",
        )
        with (
            patch.object(protected_pids, "list_all_protected", return_value=[entry]),
            patch.object(protected_pids, "_entry_process_identity_matches", return_value=True),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.getpgid", return_value=6101),
        ):
            observed = stop._verified_scheduler_entries(Path("/run"))

        self.assertEqual(observed, [entry])

    def test_scheduler_state_uses_identity_bound_live_authority(self) -> None:
        from praxist.cli import stop

        ps_rows = {6101: (1, "00:09", "python evaluator.py")}
        with (
            patch("praxist.cli.stop._verified_scheduler_entries", return_value=[]),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client.scheduler_active_process_groups",
                return_value={6100: (6101, "ps:stable")},
            ),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.getpgid", return_value=6100),
            patch("praxist.cli.stop.process_start_token", return_value="ps:stable"),
        ):
            targets, groups = stop._scheduler_owned_state(Path("/run"), ps_rows)

        self.assertEqual(targets, {6101})
        self.assertEqual(groups, {6100: {6101: "ps:stable"}})

    def test_scheduler_state_rejects_reused_rpc_launcher_identity(self) -> None:
        from praxist.cli import stop

        ps_rows = {6101: (1, "00:09", "python evaluator.py")}
        with (
            patch("praxist.cli.stop._verified_scheduler_entries", return_value=[]),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler_client.scheduler_active_process_groups",
                return_value={6100: (6101, "ps:former")},
            ),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.getpgid", return_value=6100),
            patch("praxist.cli.stop.process_start_token", return_value="ps:replacement"),
        ):
            targets, groups = stop._scheduler_owned_state(Path("/run"), ps_rows)

        self.assertEqual(targets, set())
        self.assertEqual(groups, {})

    def test_run_owned_discovery_does_not_bind_identity_after_ownership_check(self) -> None:
        from praxist.cli import stop

        run_dir = Path("/run")
        rows = {6200: (1, "00:01", "python evaluator.py")}
        with (
            patch(
                "praxist.cli.stop._read_identity_bound_ps_snapshot",
                return_value=(rows, {}),
            ),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
            patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
            patch(
                "praxist.cli.stop._run_dir_from_process_environment",
                return_value=run_dir,
            ),
            patch("praxist.cli.stop.process_start_token", return_value="ps:replacement"),
        ):
            tokens, groups = stop._discover_run_owned_processes(run_dir)

        self.assertEqual(tokens, {6200: ""})
        self.assertEqual(groups, {})
        with patch("praxist.cli.stop.pid_is_alive", return_value=True):
            self.assertEqual(stop._live_process_instances(tokens, tokens), {6200})

    def test_stop_reports_unverified_legacy_group_instead_of_false_success(self) -> None:
        from praxist.cli import registry, stop
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            run_id = _write_entry(run_dir=str(run_dir))
            orphan = protected_pids.ProtectedEntry(
                pid=6100,
                pgid=6100,
                pid_start_time="ps:former-launcher",
            )
            with (
                patch(
                    "praxist.cli.stop._read_identity_bound_ps_snapshot",
                    return_value=({}, {}),
                ),
                patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
                patch("praxist.cli.stop._drain_late_run_processes", return_value={}),
                patch.object(protected_pids, "list_all_protected", return_value=[orphan]),
                patch.object(protected_pids, "_is_process_group_alive", return_value=True),
                patch("praxist.cli.stop.pid_is_alive", return_value=False),
            ):
                outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertEqual(outcome.failed_run_ids, [run_id])
        self.assertTrue(any("could not be identity-verified" in w for w in outcome.warnings))
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_live_legacy_launcher_without_start_identity_remains_unresolved(self) -> None:
        from praxist.cli import stop
        from praxist.plugins.workflow_stages.research_loop.backend import protected_pids

        entry = protected_pids.ProtectedEntry(pid=6100, pgid=6100, pid_start_time=None)
        outcome = stop.StopOutcome(run_id="run_missing_identity")
        with (
            patch.object(protected_pids, "list_all_protected", return_value=[entry]),
            patch.object(protected_pids, "_is_process_group_alive", return_value=True),
            patch.object(protected_pids, "_entry_process_identity_matches", return_value=True),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
        ):
            stop._record_unresolved_protected_groups(
                "run_missing_identity",
                Path("/run"),
                outcome,
            )

        self.assertEqual(outcome.failed_run_ids, ["run_missing_identity"])

    def test_process_run_identity_uses_exact_inherited_environment(self) -> None:
        from praxist.cli import stop

        payload = b"PATH=/bin\0PRAXIST_RUN_DIR=/tmp/exact-run\0"
        with patch.object(Path, "read_bytes", return_value=payload):
            self.assertEqual(
                stop._run_dir_from_process_environment(123),
                Path("/tmp/exact-run").resolve(),
            )

    def test_cwd_fallback_is_limited_to_run_owned_peer_workspaces(self) -> None:
        from praxist.cli import stop

        run_dir = Path("/tmp/exact-run").resolve()
        with patch(
            "praxist.cli.stop.os.readlink",
            return_value=str(run_dir / "peer_workspaces" / "gen2_peer3"),
        ):
            self.assertTrue(stop._process_cwd_is_run_workspace(123, run_dir))
        with patch("praxist.cli.stop.os.readlink", return_value=str(run_dir)):
            self.assertFalse(stop._process_cwd_is_run_workspace(123, run_dir))

    def test_stop_does_not_target_independent_monitor_processes(self) -> None:
        from praxist.cli import stop

        run_id = _write_entry()
        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
            7000: (1, "00:01", f"python -m praxist.cli.monitor --run-id {run_id}"),
            8000: (1, "00:01", f"praxist --monitor --run-id {run_id}"),
        }
        alive = {5000, 7000, 8000}
        signalled: list[int] = []

        def fake_kill(pid: int, _sig: int) -> None:
            signalled.append(pid)
            alive.discard(pid)

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", side_effect=lambda pid: pid in alive),
            patch("praxist.cli.stop.os.kill", side_effect=fake_kill),
            patch("praxist.cli.stop.time.sleep"),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertEqual(outcome.matched_pids, [5000])
        self.assertEqual(outcome.monitor_sessions, [])
        self.assertEqual(outcome.monitor_stopped_sessions, [])
        self.assertNotIn(7000, signalled)
        self.assertNotIn(8000, signalled)

    def test_dead_root_still_stops_scheduler_owned_process_group(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(pid=5000)
        ps_rows = {6001: (1, "00:02", "python evaluator.py")}
        alive = {6001}
        group_alive = {6001}
        group_signals: list[tuple[int, int]] = []
        pid_signals: list[tuple[int, int]] = []

        def fake_alive(pid: int) -> bool:
            return pid in alive

        def fake_kill(pid: int, sig: int) -> None:
            pid_signals.append((pid, sig))
            alive.discard(pid)

        def fake_killpg(pgid: int, sig: int) -> None:
            if sig == 0:
                if pgid not in group_alive:
                    raise ProcessLookupError
                return
            group_signals.append((pgid, sig))
            group_alive.discard(pgid)

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", side_effect=fake_alive),
            patch("praxist.cli.stop.os.kill", side_effect=fake_kill),
            patch("praxist.cli.stop.os.killpg", side_effect=fake_killpg),
            patch("praxist.cli.stop.os.getpgid", return_value=6001),
            patch("praxist.cli.stop.process_start_token", return_value="proc:123"),
            patch("praxist.cli.stop.time.sleep"),
            patch(
                "praxist.cli.stop._scheduler_owned_state",
                return_value=({6001}, {6001: {6001: "proc:123"}}),
            ),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertEqual(outcome.matched_pids, [])
        self.assertEqual(outcome.descendant_pids, [6001])
        self.assertIn((6001, 15), pid_signals)
        self.assertNotIn((6001, 15), group_signals)
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)

    def test_scheduler_groups_require_live_matching_leader_identity(self) -> None:
        from praxist.cli import stop
        from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
            ProtectedEntry,
        )

        entry = ProtectedEntry(
            pid=6001,
            pgid=6001,
            pid_start_time=123,
            peer_id="gen0_peer1",
        )
        manifest_module = "praxist.plugins.workflow_stages.research_loop.backend.protected_pids"
        with (
            patch(f"{manifest_module}.list_all_protected", return_value=[entry]),
            patch(f"{manifest_module}._entry_process_identity_matches", return_value=True),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.getpgid", return_value=6001),
            patch(
                "praxist.cli.stop.read_ps_table",
                return_value={6001: (1, "", "")},
            ),
        ):
            self.assertEqual(
                stop._scheduler_owned_targets(Path("/tmp/run"), {6001: (1, "", "")}),
                {6001},
            )
            self.assertEqual(
                stop._scheduler_owned_process_groups(Path("/tmp/run")),
                {6001},
            )

        with (
            patch(f"{manifest_module}.list_all_protected", return_value=[entry]),
            patch(f"{manifest_module}._entry_process_identity_matches", return_value=False),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.os.getpgid") as getpgid,
        ):
            self.assertEqual(
                stop._scheduler_owned_targets(
                    Path("/tmp/run"),
                    {6001: (1, "", "")},
                ),
                set(),
            )
            self.assertEqual(
                stop._scheduler_owned_process_groups(Path("/tmp/run")),
                set(),
            )
            getpgid.assert_not_called()

    def test_scheduler_state_rejects_group_after_recorded_leader_exits(self) -> None:
        from praxist.cli import stop
        from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
            ProtectedEntry,
        )

        entry = ProtectedEntry(
            pid=6001,
            pgid=6001,
            pid_start_time=123,
            peer_id="gen0_peer1",
        )
        manifest_module = "praxist.plugins.workflow_stages.research_loop.backend.protected_pids"
        with (
            patch(f"{manifest_module}.list_all_protected", return_value=[entry]),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.getpgid", return_value=6001),
            patch("praxist.cli.stop.process_start_token", return_value="proc:child"),
        ):
            targets, groups = stop._scheduler_owned_state(
                Path("/tmp/run"),
                {6002: (1, "", "python evaluator.py")},
            )

        self.assertEqual(targets, set())
        self.assertEqual(groups, {})

    def test_scheduler_state_rejects_legacy_member_without_start_token(self) -> None:
        from praxist.cli import stop
        from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
            ProtectedEntry,
        )

        entry = ProtectedEntry(
            pid=6001,
            pgid=6001,
            pid_start_time=None,
            peer_id="gen0_peer1",
        )
        manifest_module = "praxist.plugins.workflow_stages.research_loop.backend.protected_pids"
        with (
            patch(f"{manifest_module}.list_all_protected", return_value=[entry]),
            patch(f"{manifest_module}._entry_process_identity_matches", return_value=True),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.getpgid", return_value=6001),
            patch("praxist.cli.stop.process_start_token", return_value=""),
        ):
            targets, groups = stop._scheduler_owned_state(
                Path("/tmp/run"),
                {6001: (1, "", "python evaluator.py")},
            )

        self.assertEqual(targets, set())
        self.assertEqual(groups, {})

    def test_corrupt_scheduler_manifest_is_treated_as_unverified(self) -> None:
        from praxist.cli import stop

        manifest_module = "praxist.plugins.workflow_stages.research_loop.backend.protected_pids"
        with patch(
            f"{manifest_module}.list_all_protected",
            side_effect=OSError("corrupt manifest"),
        ):
            entries = stop._verified_scheduler_entries(Path("/tmp/run"))

        self.assertEqual(entries, [])

    def test_toctou_mismatch_skips_signal_without_claiming_success(self) -> None:
        """Registry's pid is alive but ``ps`` shows a different command."""
        from praxist.cli import registry, stop

        run_id = _write_entry()
        # ps row claims pid 5000 is `/bin/bash` now (PID was recycled).
        ps_rows = {5000: (1, "00:01", "/bin/bash")}
        killed: list[tuple[int, int]] = []

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill", side_effect=lambda p, s: killed.append((p, s))),
            patch("praxist.cli.stop.time.sleep"),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertEqual(killed, [])  # no signals
        self.assertEqual(outcome.matched_pids, [])
        self.assertTrue(any("TOCTOU" in w for w in outcome.warnings))
        self.assertEqual(outcome.remaining_pids, [5000])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_reused_pid_after_sigterm_is_not_sent_sigkill(self) -> None:
        """Escalation must not signal a new process that reused the target PID."""
        import signal as _sig

        from praxist.cli import registry, stop

        run_id = _write_entry(extra={"process_start_token": "proc:old"})
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        sent: list[tuple[int, int]] = []
        tokens = iter(("proc:old", "proc:old", "proc:old", "proc:new", "proc:new"))

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.process_identity_matches", return_value=True),
            patch("praxist.cli.stop.process_start_token", side_effect=lambda _pid: next(tokens)),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch(
                "praxist.cli.stop.os.kill",
                side_effect=lambda pid, sig: sent.append((pid, sig)),
            ),
            patch("praxist.cli.stop._discover_run_owned_processes", return_value=({}, {})),
            patch("praxist.cli.stop.time.sleep"),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertEqual(sent, [(5000, _sig.SIGTERM)])
        self.assertEqual(outcome.killed_pids, [])
        self.assertEqual(outcome.remaining_pids, [])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)

    def test_ps_target_reused_during_discovery_is_not_signalled(self) -> None:
        from praxist.cli import stop

        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
            5001: (5000, "00:09", "python evaluator.py"),
        }
        sent: list[tuple[int, int]] = []
        root_tokens = iter(("proc:old", "proc:new"))

        def token_for(pid: int) -> str:
            return next(root_tokens) if pid == 5000 else "proc:child"

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch(
                "praxist.cli.stop.process_start_token",
                side_effect=token_for,
            ),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
            patch(
                "praxist.cli.stop.os.kill",
                side_effect=lambda pid, sig: sent.append((pid, sig)),
            ),
            patch("praxist.cli.stop.time.sleep"),
        ):
            outcome = stop.stop_all(
                scope=stop.SCOPE_PS,
                grace_seconds=0.0,
            )

        self.assertEqual(sent, [])
        self.assertEqual(outcome.remaining_pids, [5000])
        self.assertEqual(outcome.descendant_pids, [])

    def test_recorded_root_token_does_not_use_command_fallback_when_unknown(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(extra={"process_start_token": "proc:expected"})
        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
            5001: (5000, "00:09", "python evaluator.py"),
        }
        sent: list[tuple[int, int]] = []
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch(
                "praxist.cli.stop.process_start_token",
                side_effect=lambda pid: "" if pid == 5000 else "proc:child",
            ),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch(
                "praxist.cli.stop.os.kill",
                side_effect=lambda pid, sig: sent.append((pid, sig)),
            ),
            patch("praxist.cli.stop.time.sleep"),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertEqual(sent, [])
        self.assertEqual(outcome.remaining_pids, [5000])
        self.assertEqual(outcome.descendant_pids, [])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_reused_process_group_leader_is_not_sent_later_signal(self) -> None:
        import signal as _sig

        from praxist.cli import stop

        outcome = stop.StopOutcome(run_id="run_group_reuse")
        signals: list[tuple[int, int]] = []
        members = {6001: {6001: "proc:old"}}
        with (
            patch(
                "praxist.cli.stop.read_ps_table",
                return_value={6001: (1, "00:01", "python evaluator.py")},
            ),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch(
                "praxist.cli.stop.process_start_token",
                side_effect=("proc:old", "proc:new", "proc:new"),
            ),
            patch("praxist.cli.stop.os.getpgid", return_value=6001),
            patch(
                "praxist.cli.stop.os.killpg",
                side_effect=lambda pgid, sig: signals.append((pgid, sig)),
            ),
        ):
            stop._signal_process_groups(
                {6001},
                _sig.SIGTERM,
                outcome,
                group_members=members,
            )
            stop._signal_process_groups(
                {6001},
                _sig.SIGKILL,
                outcome,
                group_members=members,
            )

        self.assertEqual(
            signals,
            [(6001, 0)],
        )

    def test_signal_set_uses_identity_bound_process_handle_when_available(self) -> None:
        import signal as _sig

        from praxist.cli import stop

        outcome = stop.StopOutcome(run_id="run_pidfd")
        with (
            patch("praxist.cli.stop._open_process_handle", return_value=77),
            patch("praxist.cli.stop._signal_process_handle") as send_handle,
            patch("praxist.cli.stop.process_start_token", return_value="proc:stable"),
            patch("praxist.cli.stop.os.kill") as numeric_kill,
            patch("praxist.cli.stop.os.close"),
        ):
            stop._signal_set(
                {6001},
                _sig.SIGTERM,
                outcome,
                process_tokens={6001: "proc:stable"},
            )

        send_handle.assert_called_once_with(77, _sig.SIGTERM)
        self.assertNotIn((6001, _sig.SIGTERM), [call.args for call in numeric_kill.mock_calls])

    def test_mixed_identity_process_group_skips_group_signal(self) -> None:
        from praxist.cli import stop

        outcome = stop.StopOutcome(run_id="run_mixed_group")
        group_signals: list[tuple[int, int]] = []
        members = {
            7000: {
                7001: "proc:verified",
                7002: "proc:expected",
            }
        }

        def token_for(pid: int) -> str:
            return "proc:verified" if pid == 7001 else ""

        with (
            patch(
                "praxist.cli.stop.read_ps_table",
                return_value={
                    7001: (1, "00:01", "python evaluator.py"),
                    7002: (1, "00:01", "python evaluator.py"),
                },
            ),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.process_start_token", side_effect=token_for),
            patch("praxist.cli.stop.os.getpgid", return_value=7000),
            patch(
                "praxist.cli.stop.os.killpg",
                side_effect=lambda pgid, sig: group_signals.append((pgid, sig)),
            ),
        ):
            stop._signal_process_groups(
                {7000},
                15,
                outcome,
                group_members=members,
            )

        self.assertEqual(group_signals, [])
        self.assertEqual(outcome.remaining_pids, [7002])
        self.assertTrue(any("skipping the group signal" in item for item in outcome.warnings))

    def test_unknown_member_identity_is_left_running_and_reported(self) -> None:
        from praxist.cli import stop

        outcome = stop.StopOutcome(run_id="run_unknown_member")
        sent: list[tuple[int, int]] = []
        with (
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.process_start_token", return_value=""),
            patch(
                "praxist.cli.stop.os.kill",
                side_effect=lambda pid, sig: sent.append((pid, sig)),
            ),
        ):
            stop._signal_set(
                {6002},
                15,
                outcome,
                process_tokens={6002: "proc:expected"},
            )

        self.assertEqual(sent, [])
        self.assertEqual(outcome.remaining_pids, [6002])
        self.assertTrue(any("cannot verify process identity" in item for item in outcome.warnings))

    def test_unknown_root_identity_uses_registry_command_fallback(self) -> None:
        from praxist.cli import registry, stop

        entry = registry.RegistryEntry(**_entry_kwargs(pid=5000))
        outcome = stop.StopOutcome(run_id=entry.run_id)
        sent: list[tuple[int, int]] = []
        ps_rows = {5000: (1, "00:01", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.process_start_token", return_value=""),
            patch("praxist.cli.stop.process_identity_matches", return_value=None),
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch(
                "praxist.cli.stop.os.kill",
                side_effect=lambda pid, sig: sent.append((pid, sig)),
            ),
        ):
            stop._signal_set(
                {5000},
                15,
                outcome,
                process_tokens={5000: "proc:expected"},
                fallback_entries={5000: entry},
            )

        self.assertEqual(sent, [(5000, 15)])
        self.assertEqual(outcome.remaining_pids, [])

    def test_strong_root_identity_does_not_depend_on_command_text(self) -> None:
        from praxist.cli import registry, stop

        entry = registry.RegistryEntry(**_entry_kwargs(extra={"process_start_token": "proc:live"}))
        outcome = stop.StopOutcome(run_id=entry.run_id)
        with (
            patch("praxist.cli.stop.process_identity_matches", return_value=True),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
        ):
            root = stop._validate_registry_root(
                entry,
                {entry.pid: (1, "00:01", "renamed-controller")},
                outcome,
            )

        self.assertEqual(root, entry.pid)
        self.assertEqual(outcome.remaining_pids, [])

    def test_dead_pid_marks_stopped_without_signalling(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill") as kill_mock,
            patch("praxist.cli.stop.time.sleep"),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)
        kill_mock.assert_not_called()
        self.assertEqual(outcome.matched_pids, [])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)

    def test_epoch_mismatch_preserves_registry_and_reports_failure(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry()
        with (
            patch("praxist.cli.stop.entry_process_epoch_matches", return_value=False),
            patch("praxist.cli.stop.os.kill") as kill_mock,
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        kill_mock.assert_not_called()
        self.assertEqual(outcome.matched_pids, [])
        self.assertEqual(outcome.failed_run_ids, [run_id])
        self.assertTrue(any("state was preserved" in warning for warning in outcome.warnings))
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_non_local_entry_is_rejected_before_process_discovery(self) -> None:
        from praxist.cli import stop

        run_id = _write_entry(extra={"hostname": "remote.example"})
        with (
            patch("praxist.cli.stop.entry_is_local", return_value=False),
            patch("praxist.cli.stop._read_identity_bound_ps_snapshot") as snapshot,
            patch("praxist.cli.stop.os.kill") as kill,
            self.assertRaisesRegex(stop.StopError, "remote.example"),
        ):
            stop.stop_run(run_id=run_id, grace_seconds=0.0)

        snapshot.assert_not_called()
        kill.assert_not_called()

    def test_scheduler_fence_failures_warn_but_do_not_block_stop(self) -> None:
        from praxist.cli import registry, stop

        def prepare_run_dir(name: str) -> Path:
            run_dir = Path(self.tmp.name) / name
            endpoint = run_dir / "resource_scheduler" / "endpoint.json"
            endpoint.parent.mkdir(parents=True)
            endpoint.write_text("{}\n", encoding="utf-8")
            return run_dir

        unavailable_dir = prepare_run_dir("fence_unavailable")
        failing_dir = prepare_run_dir("fence_failing")
        unavailable_id = _write_entry(
            run_id="run_fence_unavailable",
            pid=5100,
            run_dir=str(unavailable_dir),
        )
        failing_id = _write_entry(
            run_id="run_fence_failing",
            pid=5200,
            run_dir=str(failing_dir),
        )

        def freeze(run_dir: Path, reason: str) -> bool:
            self.assertEqual(reason, "praxist_stop")
            if run_dir == unavailable_dir:
                return False
            raise RuntimeError("scheduler unavailable")

        scheduler_client = (
            "praxist.plugins.workflow_stages.research_loop.backend."
            "experiment_scheduler_client.freeze_all_for_run"
        )
        with (
            patch("praxist.cli.stop.entry_process_epoch_matches", return_value=True),
            patch(
                "praxist.cli.stop._read_identity_bound_ps_snapshot",
                return_value=({}, {}),
            ),
            patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch(scheduler_client, side_effect=freeze) as freeze_mock,
            patch("praxist.cli.stop.os.kill") as kill,
        ):
            unavailable = stop.stop_run(run_id=unavailable_id, grace_seconds=0.0)
            failing = stop.stop_run(run_id=failing_id, grace_seconds=0.0)

        self.assertEqual(freeze_mock.call_count, 2)
        self.assertIn("central scheduler stop fence was unavailable", unavailable.warnings)
        self.assertTrue(any("scheduler unavailable" in warning for warning in failing.warnings))
        kill.assert_not_called()
        self.assertEqual(registry.read_entry(unavailable_id).state, registry.STATE_STOPPED)
        self.assertEqual(registry.read_entry(failing_id).state, registry.STATE_STOPPED)

    def test_invalid_grace_is_rejected_before_registry_access(self) -> None:
        from praxist.cli import stop

        with (
            patch("praxist.cli.stop.entry_lock") as lock,
            self.assertRaisesRegex(stop.StopError, "finite non-negative"),
        ):
            stop.stop_run(run_id="never-read", grace_seconds=-0.01)

        lock.assert_not_called()

    def test_registry_lock_error_is_translated_to_stop_error(self) -> None:
        from praxist.cli import registry, stop

        with (
            patch(
                "praxist.cli.stop.entry_lock",
                side_effect=registry.RegistryError("lock unavailable"),
            ),
            self.assertRaisesRegex(stop.StopError, "lock unavailable") as raised,
        ):
            stop.stop_run(run_id="run_locked", grace_seconds=0.0)

        self.assertIsInstance(raised.exception.__cause__, registry.RegistryError)

    def test_grace_expiry_escalates_to_sigkill(self) -> None:
        import signal as _sig

        from praxist.cli import stop

        run_id = _write_entry()
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        killed: list[tuple[int, int]] = []

        # Pid stays alive through TERM grace, dies after KILL.
        kill_count = {"value": 0}

        def fake_alive(pid: int) -> bool:
            return kill_count["value"] < 2

        def fake_kill(pid: int, sig: int) -> None:
            killed.append((pid, sig))
            if sig == _sig.SIGKILL:
                kill_count["value"] = 2

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", side_effect=fake_alive),
            patch("praxist.cli.stop.os.kill", side_effect=fake_kill),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.time.monotonic", side_effect=[0.0, 0.1, 1.0]),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.05)

        # TERM then KILL.
        sigs = [s for _p, s in killed]
        self.assertIn(_sig.SIGTERM, sigs)
        self.assertIn(_sig.SIGKILL, sigs)
        self.assertEqual(outcome.killed_pids, [5000])
        self.assertEqual(outcome.remaining_pids, [])

    def test_dry_run_does_not_signal(self) -> None:
        from praxist.cli import stop

        run_id = _write_entry()
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill") as kill_mock,
            patch("praxist.cli.stop.time.sleep"),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0, dry_run=True)
        kill_mock.assert_not_called()
        self.assertEqual(outcome.matched_pids, [5000])
        self.assertTrue(outcome.dry_run)

    def test_missing_registry_entry_raises_stop_error(self) -> None:
        from praxist.cli import stop

        with self.assertRaises(stop.StopError):
            stop.stop_run(run_id="does_not_exist", grace_seconds=0.0)

    def test_stop_rereads_entry_after_acquiring_lifecycle_lock(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(pid=5000)
        current = registry.read_entry(run_id)
        replacement = registry.RegistryEntry(
            **_entry_kwargs(
                pid=6000,
                run_dir=current.run_dir,
                log_file=current.log_file,
            )
        )

        @contextmanager
        def replace_before_lock_body(_run_id: str):
            registry.write_entry(replacement)
            yield

        ps_rows = {6000: (1, "00:01", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.entry_lock", side_effect=replace_before_lock_body),
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)

        self.assertEqual(outcome.matched_pids, [6000])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)


class StopAllTest(_StateDirMixin, unittest.TestCase):
    """``praxist stop --all`` and scope flags."""

    def _ps_table_with_praxist_processes(self) -> dict[int, tuple[int, str, str]]:
        # 5001 is intentionally a *non-pattern-matching* child of 5000 so
        # the test exercises descendant walking rather than the Praxist-pattern
        # ps-scan path.
        return {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
            5001: (5000, "00:09", "python /tmp/helper_script.py"),
            6000: (1, "00:01", "python -m praxist.run run --task-path /other"),
            7000: (1, "00:30", "/sbin/launchd"),  # unrelated, must not match
        }

    def test_union_scope_matches_registry_and_ps_scan(self) -> None:
        from praxist.cli import registry, stop

        # registry: run with pid 5000
        run_id = _write_entry()
        # ps shows 5000 (registry hit), 6000 (ps-only praxist.run), 7000 (skip)
        ps_rows = self._ps_table_with_praxist_processes()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_UNION, grace_seconds=0.0)

        # Both 5000 (registry root) and 6000 (ps-only) targeted; 7000 skipped.
        self.assertIn(5000, outcome.matched_pids)
        self.assertIn(6000, outcome.matched_pids)
        self.assertNotIn(7000, outcome.matched_pids)
        # Descendant 5001 reached via the ps table.
        self.assertIn(5001, outcome.descendant_pids)
        # Registry entry for 5000 stamped stopped.
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)

    def test_union_does_not_readd_registry_pid_with_rejected_strong_identity(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(extra={"process_start_token": "proc:old"})
        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
            5001: (5000, "00:09", "python evaluator.py"),
        }
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.process_identity_matches", return_value=False),
            patch(
                "praxist.cli.stop.process_start_token",
                side_effect=lambda pid: f"proc:new:{pid}",
            ),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill") as kill,
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_UNION, grace_seconds=0.0)

        kill.assert_not_called()
        self.assertEqual(outcome.matched_pids, [])
        self.assertEqual(outcome.descendant_pids, [])
        self.assertEqual(outcome.remaining_pids, [5000])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_registry_scope_leaves_monitor_running_when_run_root_is_dead(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(pid=99999)
        ps_rows = {6000: (1, "00:09", f"python -m praxist.cli.monitor --run-id {run_id}")}

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch(
                "praxist.cli.stop.pid_is_alive",
                side_effect=lambda pid: pid == 6000,
            ),
            patch("praxist.cli.stop.os.kill") as kill,
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)

        self.assertEqual(outcome.matched_pids, [])
        self.assertEqual(outcome.monitor_sessions, [])
        self.assertEqual(outcome.monitor_stopped_sessions, [])
        kill.assert_not_called()
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)

    def test_registry_only_scope_ignores_ps_only_matches(self) -> None:
        from praxist.cli import stop

        _write_entry()
        ps_rows = self._ps_table_with_praxist_processes()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)

        self.assertIn(5000, outcome.matched_pids)
        self.assertNotIn(6000, outcome.matched_pids)

    def test_ps_scan_only_scope_ignores_registry_entries(self) -> None:
        from praxist.cli import stop

        # The registry entry has pid 5000; but in ps-scan-only mode it
        # should still be matched (pid 5000 also appears in ps as
        # praxist.run). Verify a registry-only pid (one absent
        # from ps) is NOT included.
        _write_entry(run_id="run_registry_only", pid=99999)
        ps_rows = self._ps_table_with_praxist_processes()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_PS, grace_seconds=0.0)

        # 5000 + 6000 from ps-scan; 99999 (registry-only) not present.
        self.assertIn(5000, outcome.matched_pids)
        self.assertIn(6000, outcome.matched_pids)
        self.assertNotIn(99999, outcome.matched_pids)

    def test_ps_scan_does_not_signal_unowned_task_shaped_processes(self) -> None:
        from praxist.cli import stop

        ps_rows = {
            6100: (1, "00:10", "python /other/project/train.py --epochs 3"),
            6101: (1, "00:09", "python /other/project/run_benchmark.py"),
            6102: (1, "00:08", "claude --output-format stream-json"),
        }
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.os.kill") as kill_mock,
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_PS, grace_seconds=0.0)

        self.assertEqual(outcome.matched_pids, [])
        self.assertEqual(outcome.descendant_pids, [])
        kill_mock.assert_not_called()

    def test_dry_run_does_not_signal_or_update_registry(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry()
        ps_rows = self._ps_table_with_praxist_processes()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill") as kill_mock,
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_UNION, grace_seconds=0.0, dry_run=True)
        kill_mock.assert_not_called()
        self.assertTrue(outcome.dry_run)
        # Registry entry remains running.
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_stop_all_rereads_entries_after_acquiring_lifecycle_locks(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(pid=5000)
        current = registry.read_entry(run_id)
        replacement = registry.RegistryEntry(
            **_entry_kwargs(
                pid=6000,
                run_dir=current.run_dir,
                log_file=current.log_file,
            )
        )

        @contextmanager
        def replace_before_lock_body(_run_id: str):
            registry.write_entry(replacement)
            yield

        ps_rows = {6000: (1, "00:01", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.entry_lock", side_effect=replace_before_lock_body),
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)

        self.assertEqual(outcome.matched_pids, [6000])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)

    def test_stop_all_enumerates_after_acquiring_registry_lock(self) -> None:
        from praxist.cli import registry, stop

        run_id = "run_created_before_global_lock_body"
        run_dir = Path(self.tmp.name) / "run_dirs" / run_id
        run_dir.mkdir(parents=True)
        entry = registry.RegistryEntry(
            **_entry_kwargs(
                run_id=run_id,
                pid=6000,
                run_dir=str(run_dir),
                log_file=str(run_dir / "log"),
            )
        )

        @contextmanager
        def create_before_lock_body():
            registry.write_entry(entry)
            yield

        ps_rows = {6000: (1, "00:01", "python -m praxist.run run --task-path /t")}
        with (
            patch(
                "praxist.cli.stop.registry_lock",
                side_effect=create_before_lock_body,
            ),
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)

        self.assertEqual(outcome.matched_pids, [6000])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_STOPPED)

    def test_remote_and_prior_boot_entries_are_handled_conservatively(self) -> None:
        from praxist.cli import registry, stop

        remote_id = _write_entry(
            run_id="run_remote",
            pid=5100,
            extra={"hostname": "remote.example"},
        )
        prior_ok_id = _write_entry(run_id="run_prior_ok", pid=5200)
        prior_fail_id = _write_entry(run_id="run_prior_fail", pid=5300)

        with (
            patch(
                "praxist.cli.stop.entry_is_local",
                side_effect=lambda entry: entry.run_id != remote_id,
            ),
            patch("praxist.cli.stop.entry_process_epoch_matches", return_value=False),
            patch(
                "praxist.cli.stop._read_identity_bound_ps_snapshot",
                return_value=({}, {}),
            ),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
            patch("praxist.cli.stop._scheduler_owned_state") as scheduler_state,
            patch("praxist.cli.stop.os.kill") as kill,
        ):
            outcome = stop.stop_all(
                scope=stop.SCOPE_REGISTRY,
                grace_seconds=0.0,
                now_iso="2026-07-23T12:00:00+00:00",
            )

        scheduler_state.assert_not_called()
        kill.assert_not_called()
        self.assertTrue(any("remote.example" in warning for warning in outcome.warnings))
        self.assertTrue(any("state was preserved" in warning for warning in outcome.warnings))
        self.assertCountEqual(outcome.failed_run_ids, [prior_ok_id, prior_fail_id])
        self.assertEqual(registry.read_entry(remote_id).state, registry.STATE_RUNNING)
        self.assertEqual(registry.read_entry(prior_ok_id).state, registry.STATE_RUNNING)
        self.assertEqual(registry.read_entry(prior_fail_id).state, registry.STATE_RUNNING)

    def test_stop_all_reports_each_scheduler_fence_failure(self) -> None:
        from praxist.cli import registry, stop

        def prepare_run_dir(name: str) -> Path:
            run_dir = Path(self.tmp.name) / name
            endpoint = run_dir / "resource_scheduler" / "endpoint.json"
            endpoint.parent.mkdir(parents=True)
            endpoint.write_text("{}\n", encoding="utf-8")
            return run_dir

        unavailable_dir = prepare_run_dir("all_fence_unavailable")
        failing_dir = prepare_run_dir("all_fence_failing")
        unavailable_id = _write_entry(
            run_id="run_all_fence_unavailable",
            pid=6100,
            run_dir=str(unavailable_dir),
        )
        failing_id = _write_entry(
            run_id="run_all_fence_failing",
            pid=6200,
            run_dir=str(failing_dir),
        )

        def freeze(run_dir: Path, reason: str) -> bool:
            self.assertEqual(reason, "praxist_stop_all")
            if run_dir == unavailable_dir:
                return False
            raise RuntimeError("scheduler RPC failed")

        scheduler_client = (
            "praxist.plugins.workflow_stages.research_loop.backend."
            "experiment_scheduler_client.freeze_all_for_run"
        )
        with (
            patch("praxist.cli.stop.entry_is_local", return_value=True),
            patch("praxist.cli.stop.entry_process_epoch_matches", return_value=True),
            patch(
                "praxist.cli.stop._read_identity_bound_ps_snapshot",
                return_value=({}, {}),
            ),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
            patch(scheduler_client, side_effect=freeze) as freeze_mock,
            patch("praxist.cli.stop.os.kill") as kill,
        ):
            outcome = stop.stop_all(
                scope=stop.SCOPE_REGISTRY,
                grace_seconds=0.0,
            )

        self.assertEqual(freeze_mock.call_count, 2)
        self.assertTrue(
            any(
                unavailable_id in warning and "unavailable" in warning
                for warning in outcome.warnings
            )
        )
        self.assertTrue(
            any(
                failing_id in warning and "scheduler RPC failed" in warning
                for warning in outcome.warnings
            )
        )
        kill.assert_not_called()
        self.assertEqual(registry.read_entry(unavailable_id).state, registry.STATE_STOPPED)
        self.assertEqual(registry.read_entry(failing_id).state, registry.STATE_STOPPED)

    def test_stop_all_skips_run_whose_admission_cannot_close(self) -> None:
        from praxist.cli import registry, stop

        with tempfile.TemporaryDirectory() as td:
            blocked_dir = Path(td) / "blocked"
            fenced_dir = Path(td) / "fenced"
            blocked_dir.mkdir()
            fenced_dir.mkdir()
            blocked_id = _write_entry(
                run_id="run_unfenced",
                pid=6100,
                run_dir=str(blocked_dir),
            )
            fenced_id = _write_entry(
                run_id="run_fenced",
                pid=6200,
                run_dir=str(fenced_dir),
            )
            ps_rows = {
                6100: (1, "00:10", "python -m praxist.run run --task-path /t"),
                6101: (6100, "00:09", "python blocked_eval.py"),
                6200: (1, "00:10", "python -m praxist.run run --task-path /t"),
            }
            alive = {6100, 6101, 6200}
            signals: list[tuple[int, int]] = []

            def signal_process(pid: int, sig: int) -> None:
                signals.append((pid, sig))
                alive.discard(pid)

            def write_fence(run_dir: Path, _outcome, *, source: str) -> bool:
                self.assertEqual(source, "praxist_stop_all")
                return Path(run_dir) == fenced_dir

            with (
                patch(
                    "praxist.cli.stop._write_run_shutdown_fence",
                    side_effect=write_fence,
                ),
                patch(
                    "praxist.cli.stop._read_identity_bound_ps_snapshot",
                    return_value=(ps_rows, {pid: f"proc:{pid}" for pid in ps_rows}),
                ),
                patch("praxist.cli.stop.process_identity_matches", return_value=None),
                patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
                patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
                patch("praxist.cli.stop._drain_late_run_processes", return_value={}),
                patch("praxist.cli.stop.pid_is_alive", side_effect=lambda pid: pid in alive),
                patch("praxist.cli.stop.os.kill", side_effect=signal_process),
                patch("praxist.cli.stop.time.sleep"),
            ):
                outcome = stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)

        self.assertNotIn((6100, 15), signals)
        self.assertNotIn((6101, 15), signals)
        self.assertIn((6200, 15), signals)
        self.assertEqual(outcome.remaining_pids, [6100, 6101])
        self.assertEqual(outcome.failed_run_ids, [blocked_id])
        self.assertEqual(registry.read_entry(blocked_id).state, registry.STATE_RUNNING)
        self.assertEqual(registry.read_entry(fenced_id).state, registry.STATE_STOPPED)

    def test_stop_all_preserves_unfenced_scheduler_work_after_root_exit(self) -> None:
        from praxist.cli import registry, stop

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            run_id = _write_entry(
                run_id="run_unfenced_dead_root",
                pid=6100,
                run_dir=str(run_dir),
            )
            ps_rows = {
                6101: (1, "00:09", "python -m praxist.run run --task-path /replacement"),
            }

            with (
                patch(
                    "praxist.cli.stop._write_run_shutdown_fence",
                    return_value=False,
                ),
                patch(
                    "praxist.cli.stop._read_identity_bound_ps_snapshot",
                    return_value=(ps_rows, {6101: "proc:6101"}),
                ),
                patch("praxist.cli.stop.process_identity_matches", return_value=None),
                patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
                patch(
                    "praxist.cli.stop._scheduler_owned_state",
                    return_value=({6101}, {6101: {6101: "proc:6101"}}),
                ),
                patch("praxist.cli.stop._run_dir_from_process_environment", return_value=None),
                patch("praxist.cli.stop._process_cwd_is_run_workspace", return_value=False),
                patch("praxist.cli.stop.pid_is_alive", side_effect=lambda pid: pid == 6101),
                patch("praxist.cli.stop.os.kill") as kill,
            ):
                outcome = stop.stop_all(scope=stop.SCOPE_UNION, grace_seconds=0.0)

        kill.assert_not_called()
        self.assertEqual(outcome.remaining_pids, [6101])
        self.assertEqual(outcome.failed_run_ids, [run_id])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_stop_all_reports_unfenced_run_even_when_no_process_is_visible(self) -> None:
        from praxist.cli import registry, stop

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            run_id = _write_entry(
                run_id="run_unfenced_not_visible",
                pid=6100,
                run_dir=str(run_dir),
            )
            with (
                patch(
                    "praxist.cli.stop._write_run_shutdown_fence",
                    return_value=False,
                ),
                patch(
                    "praxist.cli.stop._read_identity_bound_ps_snapshot",
                    return_value=({}, {}),
                ),
                patch("praxist.cli.stop.process_identity_matches", return_value=None),
                patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
                patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
                patch("praxist.cli.stop.pid_is_alive", return_value=False),
            ):
                outcome = stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)

        self.assertEqual(outcome.remaining_pids, [])
        self.assertEqual(outcome.failed_run_ids, [run_id])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_union_stop_skips_ambiguous_ps_scan_when_a_run_is_unfenced(self) -> None:
        from praxist.cli import registry, stop

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            run_id = _write_entry(
                run_id="run_unfenced_without_proc",
                pid=6100,
                run_dir=str(run_dir),
            )
            ps_rows = {
                6200: (1, "00:09", "python -m praxist.run run --task-path /unknown"),
            }
            with (
                patch(
                    "praxist.cli.stop._write_run_shutdown_fence",
                    return_value=False,
                ),
                patch(
                    "praxist.cli.stop._read_identity_bound_ps_snapshot",
                    return_value=(ps_rows, {6200: "ps:stable-start"}),
                ),
                patch("praxist.cli.stop.process_identity_matches", return_value=None),
                patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
                patch("praxist.cli.stop._scheduler_owned_state", return_value=(set(), {})),
                patch("praxist.cli.stop._run_dir_from_process_environment", return_value=None),
                patch("praxist.cli.stop._process_cwd_is_run_workspace", return_value=False),
                patch("praxist.cli.stop.pid_is_alive", side_effect=lambda pid: pid == 6200),
                patch("praxist.cli.stop.os.kill") as kill,
            ):
                outcome = stop.stop_all(scope=stop.SCOPE_UNION, grace_seconds=0.0)

        kill.assert_not_called()
        self.assertEqual(outcome.failed_run_ids, [run_id])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)
        self.assertTrue(any("ps scan was skipped" in item for item in outcome.warnings))

    def test_registry_read_race_is_reported_without_signalling(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(run_id="run_disappeared", pid=6300)
        with (
            patch(
                "praxist.cli.stop.read_entry",
                side_effect=registry.RegistryError("entry disappeared"),
            ),
            patch(
                "praxist.cli.stop._read_identity_bound_ps_snapshot",
                return_value=({}, {}),
            ),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
            patch("praxist.cli.stop.os.kill") as kill,
        ):
            outcome = stop.stop_all(
                scope=stop.SCOPE_REGISTRY,
                grace_seconds=0.0,
            )

        kill.assert_not_called()
        self.assertTrue(
            any(
                run_id in warning and "entry disappeared" in warning for warning in outcome.warnings
            )
        )
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_registry_global_lock_error_is_translated_to_stop_error(self) -> None:
        from praxist.cli import registry, stop

        with (
            patch(
                "praxist.cli.stop.registry_lock",
                side_effect=registry.RegistryError("global lock unavailable"),
            ),
            self.assertRaisesRegex(stop.StopError, "global lock unavailable") as raised,
        ):
            stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)

        self.assertIsInstance(raised.exception.__cause__, registry.RegistryError)


class StopCliEndToEndTest(_StateDirMixin, unittest.TestCase):
    """``praxist stop`` end-to-end through the top-level dispatcher."""

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

    def test_run_id_and_all_are_mutually_exclusive(self) -> None:
        code, _out, err = self._run(["stop", "some_id", "--all"])
        self.assertEqual(code, 2)
        self.assertIn("mutually exclusive", err)

    def test_neither_run_id_nor_all_is_rejected(self) -> None:
        code, _out, err = self._run(["stop"])
        self.assertEqual(code, 2)
        self.assertIn("expected", err)

    def test_registry_only_and_ps_scan_only_mutex(self) -> None:
        code, _out, err = self._run(["stop", "--all", "--registry-only", "--ps-scan-only"])
        self.assertEqual(code, 2)
        self.assertIn("--registry-only and --ps-scan-only", err)

    def test_non_finite_grace_is_rejected(self) -> None:
        code, _out, err = self._run(["stop", "some_id", "--grace", "nan"])

        self.assertEqual(code, 2)
        self.assertIn("finite non-negative", err)

    def test_stop_run_id_unknown_returns_nonzero(self) -> None:
        code, _out, err = self._run(["stop", "nonexistent"])
        self.assertEqual(code, 1)
        self.assertIn("praxist stop", err)

    def test_epoch_mismatch_returns_nonzero_json_without_changing_state(self) -> None:
        from praxist.cli import registry

        run_id = _write_entry()
        with (
            patch("praxist.cli.stop.entry_process_epoch_matches", return_value=False),
            patch("praxist.cli.stop.os.kill") as kill,
        ):
            code, out, _err = self._run(["stop", run_id, "--json"])

        payload = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual(payload["failed_run_ids"], [run_id])
        kill.assert_not_called()
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_stop_run_text_summary_to_stderr(self) -> None:
        """Non-JSON output writes the operator narrative to stderr + matched pids to stdout."""
        run_id = _write_entry()
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
        ):
            code, out, err = self._run(["stop", run_id])
        self.assertEqual(code, 0)
        self.assertIn("praxist stop", err)
        self.assertIn("matched roots", err)
        self.assertIn("5000", out)

    def test_stop_all_text_summary_with_no_matches(self) -> None:
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            code, _out, err = self._run(["stop", "--all"])
        self.assertEqual(code, 0)
        self.assertIn("no matching Praxist runs found", err)

    def test_stop_all_returns_nonzero_when_admission_cannot_close(self) -> None:
        from praxist.cli import stop

        outcome = stop.StopOutcome(
            run_id=None,
            failed_run_ids=["run_admission_open"],
        )
        with patch("praxist.cli.stop.stop_all", return_value=outcome):
            code, _out, err = self._run(["stop", "--all"])

        self.assertEqual(code, 1)
        self.assertIn("admission failed", err)
        self.assertIn("run_admission_open", err)

    def test_stop_returns_nonzero_when_pids_remain_alive(self) -> None:
        """If a pid is still alive after SIGKILL the command exits 1."""
        from praxist.cli import stop

        run_id = _write_entry()
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        # alive throughout — simulates a stubborn root (zombie / EPERM).
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.time.monotonic", side_effect=[0.0, 1.0]),
        ):
            outcome = stop.stop_run(run_id=run_id, grace_seconds=0.0)
        self.assertEqual(outcome.remaining_pids, [5000])

    def test_signal_set_records_permission_errors_and_process_lookup(self) -> None:
        """``_signal_set`` warns on EPERM and silently skips ESRCH."""
        from praxist.cli import stop

        outcome = stop.StopOutcome(run_id="x")

        def fake_kill(pid: int, sig: int) -> None:
            if pid == 100:
                raise ProcessLookupError
            if pid == 200:
                raise PermissionError("nope")

        with patch("praxist.cli.stop.os.kill", side_effect=fake_kill):
            stop._signal_set({100, 200, 300}, 15, outcome)
        # 100: ProcessLookupError → silently skipped, no warning.
        self.assertFalse(any("100" in w for w in outcome.warnings))
        # 200: PermissionError → warning recorded.
        self.assertTrue(any("200" in w for w in outcome.warnings))

    def test_signal_process_groups_handles_lookup_and_permission_errors(self) -> None:
        from praxist.cli import stop

        outcome = stop.StopOutcome(run_id="x")
        attempted: list[int] = []

        def fake_killpg(pgid: int, _sig: int) -> None:
            attempted.append(pgid)
            if pgid == 300:
                raise ProcessLookupError
            if pgid == 200:
                raise PermissionError("denied")

        with patch("praxist.cli.stop.os.killpg", side_effect=fake_killpg):
            stop._signal_process_groups({100, 200, 300}, 15, outcome)

        self.assertEqual(attempted, [300, 200, 100])
        self.assertFalse(any("300" in warning for warning in outcome.warnings))
        self.assertTrue(any("200" in warning for warning in outcome.warnings))

    def test_process_group_liveness_treats_permission_as_alive(self) -> None:
        from praxist.cli import stop

        with patch("praxist.cli.stop.os.killpg", side_effect=ProcessLookupError):
            self.assertFalse(stop._process_group_alive(7000))
        with patch("praxist.cli.stop.os.killpg", side_effect=PermissionError):
            self.assertTrue(stop._process_group_alive(7000))

    def test_validate_registry_root_warns_when_pid_alive_but_absent_from_ps(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(pid=99001)
        ps_rows: dict[int, tuple[int, str, str]] = {}
        outcome = stop.StopOutcome(run_id=run_id)
        with patch("praxist.cli.stop.pid_is_alive", return_value=True):
            root = stop._validate_registry_root(registry.read_entry(run_id), ps_rows, outcome)
        self.assertIsNone(root)
        self.assertTrue(any("absent from ps" in w for w in outcome.warnings))

    def test_validate_registry_root_rejects_recycled_pid_for_different_run_dir(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(
            command=(
                "python",
                "-m",
                "praxist.run",
                "run",
                "--run-dir",
                "/tmp/stop_demo",
            )
        )
        ps_rows = {
            5000: (
                1,
                "00:10",
                "python -m praxist.run run --run-dir /tmp/different_run",
            )
        }
        outcome = stop.StopOutcome(run_id=run_id)

        root = stop._validate_registry_root(registry.read_entry(run_id), ps_rows, outcome)

        self.assertIsNone(root)
        self.assertTrue(any("run directory" in warning for warning in outcome.warnings))

    def test_stop_all_kills_survivors_after_sigterm(self) -> None:
        """SIGTERM-then-SIGKILL escalation in the --all path."""
        import signal as _sig

        from praxist.cli import stop

        _write_entry()
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        killed: list[tuple[int, int]] = []
        kill_phase = {"value": 0}

        def fake_alive(pid: int) -> bool:
            # alive through TERM (phase 0/1), dead after KILL (phase 2).
            return kill_phase["value"] < 2

        def fake_kill(pid: int, sig: int) -> None:
            killed.append((pid, sig))
            if sig == _sig.SIGKILL:
                kill_phase["value"] = 2

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", side_effect=fake_alive),
            patch("praxist.cli.stop.os.kill", side_effect=fake_kill),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.time.monotonic", side_effect=[0.0, 1.0]),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_UNION, grace_seconds=0.05)
        self.assertIn(_sig.SIGKILL, [s for _p, s in killed])
        self.assertEqual(outcome.killed_pids, [5000])

    def test_stop_all_skips_stale_registry_entries(self) -> None:
        """stop_all in registry scope silently skips entries whose pid is gone."""
        from praxist.cli import stop

        _write_entry(run_id="run_alive", pid=5000)
        _write_entry(run_id="run_gone", pid=99002)
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)
        self.assertEqual(outcome.matched_pids, [5000])
        self.assertNotIn(99002, outcome.matched_pids)

    def test_stop_all_preserves_unverified_live_pid_as_remaining(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(pid=5000)
        ps_rows = {5000: (1, "00:10", "/usr/bin/unrelated-process")}
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)

        self.assertEqual(outcome.remaining_pids, [5000])
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_stop_all_includes_scheduler_work_for_dead_registry_root(self) -> None:
        from praxist.cli import stop

        _write_entry(run_id="run_dead_root", pid=99002)
        ps_rows = {6001: (1, "00:02", "python evaluator.py")}
        alive = {6001}
        group_alive = {6001}
        group_signals: list[tuple[int, int]] = []
        pid_signals: list[tuple[int, int]] = []

        def fake_alive(pid: int) -> bool:
            return pid in alive

        def fake_kill(pid: int, sig: int) -> None:
            pid_signals.append((pid, sig))
            alive.discard(pid)

        def fake_killpg(pgid: int, sig: int) -> None:
            if sig == 0:
                if pgid not in group_alive:
                    raise ProcessLookupError
                return
            group_signals.append((pgid, sig))
            group_alive.discard(pgid)

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", side_effect=fake_alive),
            patch("praxist.cli.stop.os.kill", side_effect=fake_kill),
            patch("praxist.cli.stop.os.killpg", side_effect=fake_killpg),
            patch("praxist.cli.stop.os.getpgid", return_value=6001),
            patch("praxist.cli.stop.process_start_token", return_value="proc:123"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
            patch(
                "praxist.cli.stop._scheduler_owned_state",
                return_value=({6001}, {6001: {6001: "proc:123"}}),
            ),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_REGISTRY, grace_seconds=0.0)

        self.assertEqual(outcome.matched_pids, [])
        self.assertEqual(outcome.descendant_pids, [6001])
        self.assertIn((6001, 15), pid_signals)
        self.assertNotIn((6001, 15), group_signals)

    def test_stop_all_excludes_self_ancestor_pids_from_ps_scan(self) -> None:
        from praxist.cli import stop

        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
            6000: (1, "00:01", "python -m praxist.run run --task-path /other"),
        }
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value={6000}),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_PS, grace_seconds=0.0)
        self.assertEqual(outcome.matched_pids, [5000])

    def test_gather_targets_handles_cycles_gracefully(self) -> None:
        """A pathological ps table with a parent-child cycle must not loop forever."""
        from praxist.cli import stop

        ps_rows: dict[int, tuple[int, str, str]] = {
            100: (200, "00:01", "cmd_a"),
            200: (100, "00:01", "cmd_b"),  # cycle: 100<->200
        }
        seen = stop._gather_targets(root_pids={100}, ps_rows=ps_rows)
        self.assertEqual(seen, {100, 200})

    def test_gather_targets_dedupes_duplicate_stack_pushes(self) -> None:
        """The seen-check at pop time defends against duplicate pushes.

        Construct a ps table where root B's child is root A — both roots
        push A onto the stack, and the pop loop must reject the second
        copy without re-processing it.
        """
        from praxist.cli import stop

        ps_rows: dict[int, tuple[int, str, str]] = {
            100: (1, "00:01", "cmd_a"),
            200: (1, "00:01", "cmd_b"),
            300: (200, "00:01", "cmd_a_child_of_b"),
        }
        seen = stop._gather_targets(root_pids={100, 200, 300}, ps_rows=ps_rows)
        self.assertEqual(seen, {100, 200, 300})

    def test_await_exit_sleeps_until_targets_clear(self) -> None:
        from praxist.cli import stop

        outcome = stop.StopOutcome(run_id="x")
        alive_calls = {"n": 0}

        def alive(pid: int) -> bool:
            alive_calls["n"] += 1
            return alive_calls["n"] < 3  # stays alive for 2 polls, then dies.

        with (
            patch("praxist.cli.stop.pid_is_alive", side_effect=alive),
            patch("praxist.cli.stop.time.sleep") as sleep_mock,
            patch(
                "praxist.cli.stop.time.monotonic",
                side_effect=[0.0, 0.1, 0.2, 0.3, 1.0],
            ),
        ):
            stop._await_exit({5000}, grace_seconds=0.5, outcome=outcome)
        self.assertGreaterEqual(sleep_mock.call_count, 1)

    def test_stop_all_warning_when_registry_update_fails(self) -> None:
        from praxist.cli import stop

        _write_entry()
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
            patch(
                "praxist.cli.stop.update_state",
                side_effect=OSError("disk full"),
            ),
        ):
            outcome = stop.stop_all(scope=stop.SCOPE_UNION, grace_seconds=0.0)
        self.assertTrue(any("could not mark" in w for w in outcome.warnings))

    def test_stop_exit_code_one_when_pids_remain(self) -> None:
        run_id = _write_entry()
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
            patch("praxist.cli.stop.time.monotonic", side_effect=[0.0, 100.0]),
        ):
            code, _out, err = self._run(["stop", run_id])
        self.assertEqual(code, 1)
        self.assertIn("still alive", err)

    def test_stop_run_dry_run_summary_to_stderr(self) -> None:
        run_id = _write_entry()
        ps_rows = {5000: (1, "00:10", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill") as kill_mock,
            patch("praxist.cli.stop.time.sleep"),
        ):
            code, _out, err = self._run(["stop", run_id, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("(dry-run", err)
        kill_mock.assert_not_called()

    def test_stop_run_with_warning_renders_in_summary(self) -> None:
        run_id = _write_entry()
        # ps row has different cmdline → TOCTOU warning + matched_pids empty.
        ps_rows = {5000: (1, "00:01", "/bin/bash")}
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.time.sleep"),
        ):
            code, _out, err = self._run(["stop", run_id])
        self.assertEqual(code, 1)
        self.assertIn("warning", err)
        self.assertIn("TOCTOU", err)

    def test_stop_all_dry_run_with_json(self) -> None:
        _write_entry()
        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
        }
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
            patch("praxist.cli.stop.os.kill"),
            patch("praxist.cli.stop.self_ancestor_pids", return_value=set()),
        ):
            code, out, _err = self._run(["stop", "--all", "--dry-run", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["dry_run"])
        self.assertIn(5000, payload["matched_pids"])
        self.assertEqual(payload["monitor_sessions"], [])
        self.assertEqual(payload["monitor_stopped_sessions"], [])


class GcStaleEntriesTest(_StateDirMixin, unittest.TestCase):
    """``praxist stop --gc`` cleanup of stale registry entries (#166)."""

    def test_alive_entry_with_matching_prefix_is_kept(self) -> None:
        """Live PID + matching command prefix → entry stays, no warning."""
        from praxist.cli import registry, stop

        run_id = _write_entry()
        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
        }
        with patch("praxist.cli.stop.read_ps_table", return_value=ps_rows):
            outcome = stop.gc_stale_entries()
        self.assertEqual(outcome.removed_run_ids, [])
        self.assertEqual(outcome.kept_run_ids, [run_id])
        self.assertEqual(outcome.warnings, [])
        # File is still on disk.
        self.assertEqual(registry.read_entry(run_id).pid, 5000)

    def test_dead_pid_removes_entry(self) -> None:
        """PID is not in ps and not alive → entry is unlinked."""
        from praxist.cli import registry, stop

        run_id = _write_entry()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
        ):
            outcome = stop.gc_stale_entries()
        self.assertEqual(outcome.removed_run_ids, [run_id])
        self.assertEqual(outcome.kept_run_ids, [])
        # Entry file is gone.
        with self.assertRaises(registry.RegistryError):
            registry.read_entry(run_id)

    def test_prior_boot_entry_is_removed_without_pid_probe(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry()
        with (
            patch("praxist.cli.stop.entry_process_epoch_matches", return_value=False),
            patch("praxist.cli.stop.pid_is_alive") as alive,
        ):
            outcome = stop.gc_stale_entries()

        alive.assert_not_called()
        self.assertEqual(outcome.removed_run_ids, [run_id])
        with self.assertRaises(registry.RegistryError):
            registry.read_entry(run_id)

    def test_recycled_pid_with_foreign_command_removes_entry(self) -> None:
        """PID is alive but the command line no longer starts with the
        prefix we recorded → entry is treated as stale."""
        from praxist.cli import registry, stop

        run_id = _write_entry()
        ps_rows = {
            5000: (1, "00:10", "vim some_file"),  # cmdline drifted; not Praxist
        }
        with patch("praxist.cli.stop.read_ps_table", return_value=ps_rows):
            outcome = stop.gc_stale_entries()
        self.assertEqual(outcome.removed_run_ids, [run_id])
        self.assertEqual(outcome.kept_run_ids, [])
        with self.assertRaises(registry.RegistryError):
            registry.read_entry(run_id)

    def test_recycled_pid_with_matching_command_but_new_start_token_is_removed(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry()
        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
        }
        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.process_identity_matches", return_value=False),
        ):
            outcome = stop.gc_stale_entries()

        self.assertEqual(outcome.removed_run_ids, [run_id])
        with self.assertRaises(registry.RegistryError):
            registry.read_entry(run_id)

    def test_matching_start_token_keeps_entry_when_command_changes(self) -> None:
        from praxist.cli import stop

        run_id = _write_entry(extra={"process_start_token": "proc:live"})
        with (
            patch(
                "praxist.cli.stop.read_ps_table",
                return_value={5000: (1, "00:01", "renamed-controller")},
            ),
            patch("praxist.cli.stop.process_identity_matches", return_value=True),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
        ):
            outcome = stop.gc_stale_entries()

        self.assertEqual(outcome.kept_run_ids, [run_id])
        self.assertEqual(outcome.removed_run_ids, [])

    def test_recycled_pid_with_same_prefix_but_other_run_dir_removes_entry(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(
            command=(
                "python",
                "-m",
                "praxist.run",
                "run",
                "--run-dir",
                "/tmp/stop_demo",
            )
        )
        ps_rows = {
            5000: (
                1,
                "00:10",
                "python -m praxist.run run --run-dir /tmp/different_run",
            )
        }

        with patch("praxist.cli.stop.read_ps_table", return_value=ps_rows):
            outcome = stop.gc_stale_entries()

        self.assertEqual(outcome.removed_run_ids, [run_id])
        with self.assertRaises(registry.RegistryError):
            registry.read_entry(run_id)

    def test_dry_run_lists_without_unlinking(self) -> None:
        """``--dry-run`` reports the would-be-removed ids and leaves the
        registry intact, so the operator can preview before destruction."""
        from praxist.cli import registry, stop

        run_id = _write_entry()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
        ):
            outcome = stop.gc_stale_entries(dry_run=True)
        self.assertEqual(outcome.removed_run_ids, [run_id])
        self.assertTrue(outcome.dry_run)
        # Entry survives the dry-run.
        self.assertEqual(registry.read_entry(run_id).pid, 5000)

    def test_alive_pid_not_in_ps_is_kept_with_warning(self) -> None:
        """PID is alive but ``ps`` doesn't list it (different user /
        container / permission boundary). GC refuses to delete — we
        can't verify ownership, and the file might belong to a run
        we just can't see."""
        from praxist.cli import stop

        run_id = _write_entry()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.pid_is_alive", return_value=True),
        ):
            outcome = stop.gc_stale_entries()
        self.assertEqual(outcome.removed_run_ids, [])
        self.assertEqual(outcome.kept_run_ids, [run_id])
        self.assertEqual(len(outcome.warnings), 1)
        self.assertIn("cannot verify ownership", outcome.warnings[0])

    def test_mixed_registry_partitions_correctly(self) -> None:
        """Three entries — one alive-matching, one dead, one recycled —
        each routes through the correct branch. Live one stays; the
        other two are removed."""
        from praxist.cli import registry, stop

        live = _write_entry(run_id="run_live", pid=5000)
        dead = _write_entry(run_id="run_dead", pid=6000)
        recycled = _write_entry(run_id="run_recycled", pid=7000)
        ps_rows = {
            5000: (1, "00:10", "python -m praxist.run run --task-path /t"),
            7000: (1, "00:05", "vim some_other_file"),
        }

        def _alive(pid: int) -> bool:
            return pid != 6000  # dead

        with (
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
            patch("praxist.cli.stop.pid_is_alive", side_effect=_alive),
        ):
            outcome = stop.gc_stale_entries()
        self.assertEqual(outcome.removed_run_ids, sorted([dead, recycled]))
        self.assertEqual(outcome.kept_run_ids, [live])
        self.assertEqual(registry.read_entry(live).pid, 5000)

    def test_empty_registry_is_a_quiet_no_op(self) -> None:
        from praxist.cli import stop

        with patch("praxist.cli.stop.read_ps_table", return_value={}):
            outcome = stop.gc_stale_entries()
        self.assertEqual(outcome.removed_run_ids, [])
        self.assertEqual(outcome.kept_run_ids, [])
        self.assertEqual(outcome.warnings, [])

    def test_gc_rereads_entry_after_acquiring_lifecycle_lock(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(pid=5000)
        replacement = registry.RegistryEntry(**_entry_kwargs(pid=6000))

        @contextmanager
        def replace_before_lock_body(_run_id: str):
            registry.write_entry(replacement)
            yield

        ps_rows = {6000: (1, "00:01", "python -m praxist.run run --task-path /t")}
        with (
            patch("praxist.cli.stop.entry_lock", side_effect=replace_before_lock_body),
            patch("praxist.cli.stop.read_ps_table", return_value=ps_rows),
        ):
            outcome = stop.gc_stale_entries()

        self.assertEqual(outcome.kept_run_ids, [run_id])
        self.assertEqual(outcome.removed_run_ids, [])
        self.assertEqual(registry.read_entry(run_id).pid, 6000)

    def test_gc_reports_registry_lock_race_and_keeps_entry(self) -> None:
        from praxist.cli import registry, stop

        run_id = _write_entry(run_id="run_gc_locked")
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch(
                "praxist.cli.stop.entry_lock",
                side_effect=registry.RegistryError("entry lock changed"),
            ),
        ):
            outcome = stop.gc_stale_entries()

        self.assertEqual(outcome.removed_run_ids, [])
        self.assertTrue(
            any(
                run_id in warning and "entry lock changed" in warning
                for warning in outcome.warnings
            )
        )
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)

    def test_gc_reports_already_gone_and_unlink_errors(self) -> None:
        from praxist.cli import registry, stop

        already_gone_id = _write_entry(run_id="run_gc_already_gone", pid=7100)
        unlink_error_id = _write_entry(run_id="run_gc_unlink_error", pid=7200)

        def remove_entry(run_id: str) -> bool:
            if run_id == already_gone_id:
                return False
            raise OSError("permission denied")

        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
            patch("praxist.cli.stop.remove_entry", side_effect=remove_entry),
        ):
            outcome = stop.gc_stale_entries()

        self.assertEqual(outcome.removed_run_ids, [])
        self.assertTrue(any("already gone" in warning for warning in outcome.warnings))
        self.assertTrue(any("permission denied" in warning for warning in outcome.warnings))
        self.assertEqual(registry.read_entry(already_gone_id).pid, 7100)
        self.assertEqual(registry.read_entry(unlink_error_id).pid, 7200)


class GcCliDispatchTest(_StateDirMixin, unittest.TestCase):
    """``praxist stop --gc`` end-to-end through the top-level dispatcher."""

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

    def test_gc_alone_runs_through_dispatcher(self) -> None:
        from praxist.cli import registry

        run_id = _write_entry()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
        ):
            code, out, err = self._run(["stop", "--gc"])
        self.assertEqual(code, 0)
        self.assertIn(run_id, out)  # stdout: one run_id per line
        self.assertIn("removed", err)
        with self.assertRaises(registry.RegistryError):
            registry.read_entry(run_id)

    def test_gc_with_dry_run_does_not_unlink(self) -> None:
        from praxist.cli import registry

        run_id = _write_entry()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
        ):
            code, out, err = self._run(["stop", "--gc", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn(run_id, out)
        self.assertIn("dry-run", err)
        # Entry survives dry-run.
        self.assertEqual(registry.read_entry(run_id).pid, 5000)

    def test_gc_with_run_id_is_rejected(self) -> None:
        code, _out, err = self._run(["stop", "some_id", "--gc"])
        self.assertEqual(code, 2)
        self.assertIn("--gc cannot be combined", err)

    def test_gc_with_all_is_rejected(self) -> None:
        code, _out, err = self._run(["stop", "--all", "--gc"])
        self.assertEqual(code, 2)
        self.assertIn("--gc cannot be combined", err)

    def test_gc_with_json_emits_structured_payload(self) -> None:
        _write_entry()
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.pid_is_alive", return_value=False),
        ):
            code, out, _err = self._run(["stop", "--gc", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertFalse(payload["dry_run"])
        self.assertEqual(payload["removed_run_ids"], ["run_stop_demo"])
        self.assertEqual(payload["kept_run_ids"], [])

    def test_gc_keeps_remote_entry_and_summarizes_reason(self) -> None:
        from praxist.cli import registry

        run_id = _write_entry(extra={"hostname": "remote.example"})
        with (
            patch("praxist.cli.stop.read_ps_table", return_value={}),
            patch("praxist.cli.stop.entry_is_local", return_value=False),
        ):
            code, out, err = self._run(["stop", "--gc"])

        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertIn("no stale entries found", err)
        self.assertIn("kept   : 1 live entry", err)
        self.assertIn("remote-host registry entry kept", err)
        self.assertEqual(registry.read_entry(run_id).state, registry.STATE_RUNNING)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
