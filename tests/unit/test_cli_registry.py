"""Tests for ``praxist.cli.registry`` — Phase 2 run registry."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _entry_fields(**overrides: object) -> dict[str, object]:
    base = {
        "schema_version": 1,
        "run_id": "run_2026-05-19_10-00-00_demo",
        "pid": 4242,
        "parent_pid": 1,
        "run_dir": "/tmp/runs/run_demo",
        "log_file": "/tmp/runs/run_demo/logs/launcher.nohup.log",
        "task_path": "/tmp/demo_task",
        "model": "claude-opus-4-7",
        "model_provider_ref": "model_provider:anthropic_messages",
        "runtime_ref": "agent_runtime:claude_sdk",
        "command": (
            "/usr/bin/python",
            "-m",
            "praxist.run",
            "run",
            "--task-path",
            "/tmp/demo_task",
        ),
        "command_prefix": "/usr/bin/python -m praxist.run",
        "started_at": "2026-05-19T10:00:00+00:00",
    }
    base.update(overrides)
    return base


class RegistryBasicsTest(unittest.TestCase):
    """Atomic write, read, list, update_state, and remove paths."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env = patch.dict(os.environ, {"PRAXIST_STATE_DIR": self.tmp.name}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_write_then_read_round_trip(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(**_entry_fields())
        path = registry.write_entry(entry)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "run_2026-05-19_10-00-00_demo.json")
        read_back = registry.read_entry(entry.run_id)
        self.assertEqual(read_back, entry)

    def test_list_entries_returns_sorted_by_started_at(self) -> None:
        from praxist.cli import registry

        first = registry.RegistryEntry(
            **_entry_fields(
                run_id="run_a",
                started_at="2026-05-19T09:00:00+00:00",
            )
        )
        second = registry.RegistryEntry(
            **_entry_fields(
                run_id="run_b",
                started_at="2026-05-19T10:00:00+00:00",
            )
        )
        registry.write_entry(second)
        registry.write_entry(first)
        listed = registry.list_entries()
        self.assertEqual([e.run_id for e in listed], ["run_a", "run_b"])

    def test_update_state_rewrites_only_state_fields(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(**_entry_fields())
        registry.write_entry(entry)
        updated = registry.update_state(
            entry.run_id, registry.STATE_STOPPED, stopped_at="2026-05-19T11:00:00+00:00"
        )
        self.assertEqual(updated.state, registry.STATE_STOPPED)
        self.assertEqual(updated.stopped_at, "2026-05-19T11:00:00+00:00")
        # All other fields preserved.
        for field_name in ("pid", "run_dir", "task_path", "command_prefix"):
            self.assertEqual(getattr(updated, field_name), getattr(entry, field_name))

    def test_list_entries_skips_corrupt_files(self) -> None:
        from praxist.cli import registry

        good = registry.RegistryEntry(**_entry_fields(run_id="run_good"))
        registry.write_entry(good)
        corrupt_path = registry.runs_dir(create=True) / "run_corrupt.json"
        corrupt_path.write_text("not json {{{")
        # Schema mismatch — version 0.
        bad_schema_path = registry.runs_dir(create=True) / "run_v0.json"
        bad_schema_path.write_text(json.dumps({"schema_version": 0}))

        listed = registry.list_entries()
        self.assertEqual([e.run_id for e in listed], ["run_good"])

    def test_iter_entries_with_errors_surfaces_failures(self) -> None:
        from praxist.cli import registry

        good = registry.RegistryEntry(**_entry_fields(run_id="run_ok"))
        registry.write_entry(good)
        bad_path = registry.runs_dir(create=True) / "run_bad.json"
        bad_path.write_text("not json")

        results = list(registry.iter_entries_with_errors())
        # Order is filename alphabetical; "run_bad" < "run_ok".
        errors = [err for _entry, err in results if err is not None]
        oks = [e for e, _err in results if e is not None]
        self.assertEqual(len(errors), 1)
        self.assertIn("run_bad.json", errors[0])
        self.assertEqual([e.run_id for e in oks], ["run_ok"])

    def test_list_entries_with_empty_state_dir(self) -> None:
        from praxist.cli import registry

        # The state_dir exists (env set), but the runs subdir does not.
        self.assertEqual(registry.list_entries(), [])

    def test_invalid_run_id_rejected_by_entry_path(self) -> None:
        from praxist.cli import registry

        with self.assertRaises(ValueError):
            registry.entry_path("")
        with self.assertRaises(ValueError):
            registry.entry_path("../escape")
        with self.assertRaises(ValueError):
            registry.entry_path(".hidden")

    def test_atomic_write_replaces_existing_entry(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(**_entry_fields())
        registry.write_entry(entry)
        new_entry = registry.RegistryEntry(**_entry_fields(pid=9999))
        registry.write_entry(new_entry)
        read_back = registry.read_entry(entry.run_id)
        self.assertEqual(read_back.pid, 9999)

    def test_atomic_create_never_replaces_existing_entry(self) -> None:
        from praxist.cli import registry

        original = registry.RegistryEntry(**_entry_fields(pid=1111))
        registry.create_entry(original)
        competing = registry.RegistryEntry(**_entry_fields(pid=2222))
        with self.assertRaises(registry.RegistryError):
            registry.create_entry(competing)
        self.assertEqual(registry.read_entry(original.run_id).pid, 1111)

    def test_entry_lock_wraps_state_directory_open_failure(self) -> None:
        from praxist.cli import registry

        with (
            patch("praxist.cli.registry.os.open", side_effect=OSError("read-only state dir")),
            self.assertRaises(registry.RegistryError) as raised,
            registry.entry_lock("run_locked"),
        ):
            self.fail("the lock body must not run when the lock file cannot be opened")

        self.assertIn("run_locked", str(raised.exception))
        self.assertIn("read-only state dir", str(raised.exception))

    def test_local_host_identity_skips_unreadable_linux_machine_ids(self) -> None:
        from praxist.cli import registry

        with (
            patch.dict(os.environ, {"PRAXIST_HOST_ID": ""}, clear=False),
            patch("praxist.cli.registry.socket.gethostname", return_value="linux-host"),
            patch.object(
                Path,
                "read_text",
                side_effect=[
                    OSError("/etc/machine-id unavailable"),
                    OSError("dbus machine-id unavailable"),
                    "boot-linux\n",
                ],
            ),
            patch("praxist.cli.registry.os.readlink", return_value="pid:[42]"),
            patch("praxist.cli.registry.shutil.which") as which,
        ):
            identity = registry.local_host_identity()

        self.assertEqual(
            identity,
            {
                "hostname": "linux-host",
                "boot_id": "boot-linux",
                "pid_namespace": "pid:[42]",
            },
        )
        which.assert_not_called()

    def test_local_host_identity_uses_sysctl_and_recovers_from_timeout(self) -> None:
        from praxist.cli import registry

        completed = registry.subprocess.CompletedProcess(
            args=["sysctl"],
            returncode=0,
            stdout="{ sec = 1777777777, usec = 0 }\n",
            stderr="",
        )
        with (
            patch.dict(os.environ, {"PRAXIST_HOST_ID": "operator-node"}, clear=False),
            patch("praxist.cli.registry.socket.gethostname", return_value="darwin-host"),
            patch.object(Path, "read_text", side_effect=OSError("/proc unavailable")),
            patch("praxist.cli.registry.shutil.which", return_value="/usr/sbin/sysctl"),
            patch("praxist.cli.registry.os.readlink", side_effect=OSError("/proc unavailable")),
            patch("praxist.cli.registry.subprocess.run", return_value=completed) as run,
        ):
            identity = registry.local_host_identity()

        self.assertEqual(identity["hostname"], "darwin-host")
        self.assertEqual(
            identity["host_id"],
            hashlib.sha256(b"operator:operator-node").hexdigest(),
        )
        self.assertEqual(identity["boot_id"], "darwin:1777777777")
        run.assert_called_once_with(
            ["/usr/sbin/sysctl", "-n", "kern.boottime"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )

        with (
            patch.dict(os.environ, {"PRAXIST_HOST_ID": "operator-node"}, clear=False),
            patch("praxist.cli.registry.socket.gethostname", return_value="darwin-host"),
            patch.object(Path, "read_text", side_effect=OSError("/proc unavailable")),
            patch("praxist.cli.registry.shutil.which", return_value="/usr/sbin/sysctl"),
            patch(
                "praxist.cli.registry.subprocess.run",
                side_effect=registry.subprocess.TimeoutExpired("sysctl", 2),
            ),
            patch("praxist.cli.registry.os.readlink", side_effect=OSError("/proc unavailable")),
        ):
            timed_out = registry.local_host_identity()

        self.assertNotIn("boot_id", timed_out)
        self.assertNotIn("pid_namespace", timed_out)

    def test_darwin_boot_time_ignores_microsecond_drift(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(
            **_entry_fields(
                extra={
                    "hostname": "darwin-host",
                    "boot_id": "darwin:1777777777:42",
                }
            )
        )
        with patch(
            "praxist.cli.registry.local_host_identity",
            return_value={
                "hostname": "darwin-host",
                "boot_id": "kern.boottime: { sec=1777777777, usec=000043 }",
            },
        ):
            self.assertTrue(registry.entry_process_epoch_matches(entry))

        self.assertEqual(
            registry._normalized_boot_id("{ sec = 1777777777, usec = 42 }"),
            "darwin:1777777777",
        )
        self.assertEqual(
            registry._normalized_boot_id("darwin:1777777777:42"),
            "darwin:1777777777",
        )

    def test_process_start_token_uses_portable_ps_fallback_without_proc(self) -> None:
        from praxist.cli import registry

        with (
            patch.object(Path, "read_text", side_effect=OSError("proc unavailable")),
            patch("praxist.cli.registry.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.registry.subprocess.run",
                return_value=registry.subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="Mon Jul 23 12:34:56 2026\n",
                    stderr="",
                ),
            ) as run,
        ):
            self.assertEqual(
                registry.process_start_token(4242),
                "ps:Mon Jul 23 12:34:56 2026",
            )

        run.assert_called_once_with(
            ["/bin/ps", "-p", "4242", "-o", "lstart="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
            env={"LANG": "C", "LC_ALL": "C"},
        )

    def test_process_start_token_handles_daemonized_dash(self) -> None:
        from praxist.cli import registry

        with (
            patch.object(Path, "read_text", side_effect=OSError("proc unavailable")),
            patch("praxist.cli.registry.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.registry.subprocess.run",
                return_value=registry.subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="-\n",
                ),
            ),
        ):
            self.assertEqual(
                registry.process_start_token(4242),
                "",
            )

    def test_process_start_token_handles_proc_no_ps_and_ps_timeout(self) -> None:
        from praxist.cli import registry

        stat_fields = ["S", *[str(value) for value in range(4, 22)], "987654"]
        proc_stat = f"4242 (worker with spaces) {' '.join(stat_fields)}"
        with patch.object(Path, "read_text", return_value=proc_stat):
            self.assertEqual(registry.process_start_token(4242), "proc:987654")

        self.assertEqual(registry.process_start_token(0), "")
        with (
            patch.object(Path, "read_text", side_effect=OSError("no proc")),
            patch("praxist.cli.registry.shutil.which", return_value=None),
        ):
            self.assertEqual(registry.process_start_token(4242), "")

        with (
            patch.object(Path, "read_text", side_effect=OSError("no proc")),
            patch("praxist.cli.registry.shutil.which", return_value="/bin/ps"),
            patch(
                "praxist.cli.registry.subprocess.run",
                side_effect=registry.subprocess.TimeoutExpired("ps", 2),
            ),
        ):
            self.assertEqual(registry.process_start_token(4242), "")

    def test_process_identity_compares_recorded_process_instance(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(**_entry_fields(extra={"process_start_token": "proc:12345"}))
        with patch("praxist.cli.registry.process_start_token", return_value="proc:12345"):
            self.assertTrue(registry.process_identity_matches(entry))
        with patch("praxist.cli.registry.process_start_token", return_value="proc:54321"):
            self.assertFalse(registry.process_identity_matches(entry))

    def test_host_identity_separates_machine_from_boot_epoch(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(
            **_entry_fields(
                extra={
                    "hostname": "host-a",
                    "host_id": "stable-a",
                    "boot_id": "old-boot",
                    "pid_namespace": "pid:[1]",
                }
            )
        )
        with patch(
            "praxist.cli.registry.local_host_identity",
            return_value={
                "hostname": "host-a",
                "host_id": "stable-a",
                "boot_id": "new-boot",
                "pid_namespace": "pid:[2]",
            },
        ):
            self.assertTrue(registry.entry_is_local(entry))
            self.assertFalse(registry.entry_process_epoch_matches(entry))

    def test_host_identity_fallbacks_use_hostname_and_boot_epoch(self) -> None:
        from praxist.cli import registry

        cases = (
            (
                {"hostname": "same-name"},
                {"hostname": "same-name"},
                True,
            ),
            (
                {"hostname": "old-name", "boot_id": "old-boot"},
                {"hostname": "new-name", "boot_id": "new-boot"},
                False,
            ),
            (
                {"hostname": "old-name", "boot_id": "shared-boot"},
                {"hostname": "new-name", "boot_id": "shared-boot"},
                True,
            ),
        )
        for recorded, local, expected in cases:
            with (
                self.subTest(recorded=recorded, local=local),
                patch("praxist.cli.registry.local_host_identity", return_value=local),
            ):
                entry = registry.RegistryEntry(**_entry_fields(extra=recorded))
                self.assertIs(registry.entry_is_local(entry), expected)

    def test_stable_host_id_distinguishes_same_named_remote_host(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(
            **_entry_fields(extra={"hostname": "shared-name", "host_id": "host-a"})
        )
        with patch(
            "praxist.cli.registry.local_host_identity",
            return_value={"hostname": "shared-name", "host_id": "host-b"},
        ):
            self.assertFalse(registry.entry_is_local(entry))

    def test_boot_id_recovers_container_without_machine_id(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(
            **_entry_fields(
                extra={
                    "hostname": "old-container",
                    "boot_id": "shared-kernel-boot",
                    "pid_namespace": "pid:[10]",
                }
            )
        )
        with patch(
            "praxist.cli.registry.local_host_identity",
            return_value={
                "hostname": "new-container",
                "boot_id": "shared-kernel-boot",
                "pid_namespace": "pid:[10]",
            },
        ):
            self.assertTrue(registry.entry_is_local(entry))

    def test_same_boot_does_not_merge_distinct_container_namespaces(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(
            **_entry_fields(
                extra={
                    "hostname": "container-a",
                    "boot_id": "shared-kernel-boot",
                    "pid_namespace": "pid:[10]",
                }
            )
        )
        with patch(
            "praxist.cli.registry.local_host_identity",
            return_value={
                "hostname": "container-b",
                "boot_id": "shared-kernel-boot",
                "pid_namespace": "pid:[20]",
            },
        ):
            self.assertFalse(registry.entry_is_local(entry))
            self.assertFalse(registry.entry_process_epoch_matches(entry))

    def test_same_host_id_does_not_merge_current_container_namespaces(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(
            **_entry_fields(
                extra={
                    "hostname": "container-a",
                    "host_id": "shared-machine",
                    "boot_id": "same-boot",
                    "pid_namespace": "pid:[10]",
                }
            )
        )
        with patch(
            "praxist.cli.registry.local_host_identity",
            return_value={
                "hostname": "container-b",
                "host_id": "shared-machine",
                "boot_id": "same-boot",
                "pid_namespace": "pid:[20]",
            },
        ):
            self.assertFalse(registry.entry_is_local(entry))
            self.assertFalse(registry.entry_process_epoch_matches(entry))

    def test_remove_entry_returns_true_on_success_false_on_missing(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(**_entry_fields())
        registry.write_entry(entry)
        self.assertTrue(registry.remove_entry(entry.run_id))
        self.assertFalse(registry.remove_entry(entry.run_id))

    def test_read_entry_rejects_filename_mismatch(self) -> None:
        from praxist.cli import registry

        # Write a file whose internal run_id disagrees with its filename.
        path = registry.runs_dir(create=True) / "run_outer.json"
        data = json.dumps(
            {
                **_entry_fields(run_id="run_inner"),
                "command": list(_entry_fields()["command"]),  # type: ignore[index]
            }
        )
        path.write_text(data)
        with self.assertRaises(registry.RegistryError):
            registry.read_entry("run_outer")

    def test_with_state_validation(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(**_entry_fields())
        with self.assertRaises(ValueError):
            entry.with_state("nonsense")
        new = entry.with_state(registry.STATE_STOPPED, stopped_at="t")
        self.assertEqual(new.state, registry.STATE_STOPPED)
        self.assertEqual(new.stopped_at, "t")


class RegistrySchemaValidationTest(unittest.TestCase):
    """Schema validation catches missing fields, type errors, version mismatch."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env = patch.dict(os.environ, {"PRAXIST_STATE_DIR": self.tmp.name}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)

    def _write_raw(self, run_id: str, payload: dict[str, object]) -> Path:
        from praxist.cli import registry

        path = registry.runs_dir(create=True) / f"{run_id}.json"
        path.write_text(json.dumps(payload))
        return path

    def test_missing_field_is_reported(self) -> None:
        from praxist.cli import registry

        partial = _entry_fields()
        del partial["pid"]
        partial["command"] = list(partial["command"])  # type: ignore[index]
        self._write_raw("run_missing", partial)
        with self.assertRaises(registry.RegistryError) as cm:
            registry.read_entry("run_missing")
        self.assertIn("pid", str(cm.exception))

    def test_unknown_state_is_rejected(self) -> None:
        from praxist.cli import registry

        payload = _entry_fields(run_id="run_state")
        payload["command"] = list(payload["command"])  # type: ignore[index]
        payload["state"] = "weird"
        self._write_raw("run_state", payload)
        with self.assertRaises(registry.RegistryError):
            registry.read_entry("run_state")

    def test_command_must_be_list_of_strings(self) -> None:
        from praxist.cli import registry

        payload = _entry_fields(run_id="run_cmd")
        payload["command"] = [1, 2, 3]
        self._write_raw("run_cmd", payload)
        with self.assertRaises(registry.RegistryError):
            registry.read_entry("run_cmd")


class RegistryReadErrorPathsTest(unittest.TestCase):
    """read_entry surfaces missing-file / OS error / JSON parse failures."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._env = patch.dict(os.environ, {"PRAXIST_STATE_DIR": self.tmp.name}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_read_entry_missing_file_raises_registry_error(self) -> None:
        from praxist.cli import registry

        with self.assertRaises(registry.RegistryError) as cm:
            registry.read_entry("nonexistent")
        self.assertIn("no registry entry", str(cm.exception))

    def test_read_entry_invalid_json_raises_registry_error(self) -> None:
        from praxist.cli import registry

        path = registry.runs_dir(create=True) / "run_bad.json"
        path.write_text("{not json")
        with self.assertRaises(registry.RegistryError) as cm:
            registry.read_entry("run_bad")
        self.assertIn("not valid JSON", str(cm.exception))

    def test_read_entry_top_level_must_be_object(self) -> None:
        from praxist.cli import registry

        path = registry.runs_dir(create=True) / "run_array.json"
        path.write_text("[1, 2, 3]")
        with self.assertRaises(registry.RegistryError):
            registry.read_entry("run_array")

    def test_read_entry_unreadable_file_raises_registry_error(self) -> None:
        from praxist.cli import registry

        entry = registry.RegistryEntry(**_entry_fields(run_id="run_oserror"))
        registry.write_entry(entry)
        with (
            patch.object(Path, "read_text", side_effect=OSError("io error")),
            self.assertRaises(registry.RegistryError) as cm,
        ):
            registry.read_entry("run_oserror")
        self.assertIn("could not read", str(cm.exception))

    def test_extra_must_be_dict(self) -> None:
        from praxist.cli import registry

        payload = _entry_fields(run_id="run_extra_bad")
        payload["command"] = list(payload["command"])  # type: ignore[index]
        payload["extra"] = "not a dict"
        path = registry.runs_dir(create=True) / "run_extra_bad.json"
        path.write_text(json.dumps(payload))
        with self.assertRaises(registry.RegistryError):
            registry.read_entry("run_extra_bad")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
