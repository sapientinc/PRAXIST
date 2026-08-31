"""Canonical launcher logs cannot redirect controller output into private files."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from praxist.cli import start


class LauncherLogSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.log = self.root / "output" / "logs" / "launcher.nohup.log"
        self.log.parent.mkdir(parents=True)
        self.private = self.root / "private.json"
        self.private.write_bytes(b"private authority\n")

    def test_spawn_rejects_log_symlink_before_launch(self) -> None:
        self.log.symlink_to(self.private)
        spawn = Mock()
        with self.assertRaises(OSError):
            start._spawn_child(spawn, [sys.executable, "-c", "print('child')"], self.log, {})
        spawn.assert_not_called()
        self.assertEqual(self.private.read_bytes(), b"private authority\n")

    def test_spawn_rejects_swapped_log_directory(self) -> None:
        self.log.parent.parent.chmod(0o777)
        self.log.parent.rmdir()
        self.log.parent.symlink_to(self.root, target_is_directory=True)
        spawn = Mock()
        with (
            patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(self.root / "control")}),
            self.assertRaises(OSError),
        ):
            start._spawn_child(spawn, [sys.executable], self.log, {})
        spawn.assert_not_called()
        self.assertFalse((self.root / self.log.name).exists())

    def test_spawn_preserves_append_log_behavior(self) -> None:
        self.log.write_bytes(b"first launch\n")

        def spawn(command, **kwargs):
            kwargs["stdout"].write(b"resumed launch\n")
            return Mock(pid=2468)

        self.assertEqual(start._spawn_child(spawn, [sys.executable], self.log, {}), 2468)
        self.assertEqual(self.log.read_bytes(), b"first launch\nresumed launch\n")

    @unittest.skipUnless(hasattr(os, "fork"), "daemon launcher requires POSIX")
    def test_daemonized_spawn_rejects_log_symlink(self) -> None:
        self.log.symlink_to(self.private)
        with self.assertRaises(start.StartError):
            start._spawn_daemonized(
                [sys.executable, "-c", "print('child')"], self.log, dict(os.environ)
            )
        self.assertEqual(self.private.read_bytes(), b"private authority\n")
