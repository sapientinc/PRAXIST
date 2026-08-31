"""Artifact writes cannot follow peer-controlled links into controller state."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.core import storage


class StorageWriteSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.unresolved_root = Path(temporary.name)
        self.root = self.unresolved_root.resolve()
        self.output = self.root / "output"
        self.output.mkdir()
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700)
        self.authority = self.private / "startup_config.json"
        self.authority.write_text('"private authority"\n', encoding="utf-8")
        self.authority.chmod(0o600)

    def assert_authority_unchanged(self) -> None:
        self.assertEqual(self.authority.read_text(encoding="utf-8"), '"private authority"\n')

    def test_json_write_ignores_predictable_temporary_symlink(self) -> None:
        destination = self.output / "result.json"
        planted = destination.with_suffix(".json.tmp")
        planted.symlink_to(self.authority)
        storage.write_json(destination, {"result": "captured"})
        self.assert_authority_unchanged()
        self.assertEqual(json.loads(destination.read_text()), {"result": "captured"})
        self.assertTrue(planted.is_symlink())

    def test_json_and_ledger_rewrite_replace_destination_symlink(self) -> None:
        for writer, value in (
            (storage.write_json, {"value": 1}),
            (storage.rewrite_jsonl, [{"value": 1}]),
        ):
            with self.subTest(writer=writer.__name__):
                destination = self.output / writer.__name__
                destination.symlink_to(self.authority)
                writer(destination, value)
                self.assert_authority_unchanged()
                self.assertFalse(destination.is_symlink())
                self.assertEqual(json.loads(destination.read_text()), {"value": 1})

    def test_append_rejects_destination_symlink(self) -> None:
        destination = self.output / "ledger.jsonl"
        destination.symlink_to(self.authority)
        with self.assertRaises(OSError):
            storage.append_jsonl(destination, {"attacker": True})
        self.assert_authority_unchanged()

    def test_append_rejects_hardlinked_file(self) -> None:
        destination = self.output / "ledger.jsonl"
        os.link(self.authority, destination)
        with self.assertRaises(OSError):
            storage.append_jsonl(destination, {"attacker": True})
        self.assert_authority_unchanged()

    def test_writers_reject_symlink_in_output_directory_path(self) -> None:
        self.output.chmod(0o777)
        (self.output / "redirect").symlink_to(self.private, target_is_directory=True)
        for writer, value in (
            (storage.write_json, {"attacker": True}),
            (storage.append_jsonl, {"attacker": True}),
            (storage.rewrite_jsonl, [{"attacker": True}]),
        ):
            with (
                self.subTest(writer=writer.__name__),
                patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(self.private)}),
                self.assertRaises(OSError),
            ):
                writer(self.output / "redirect" / self.authority.name, value)
            self.assert_authority_unchanged()

    def test_failed_rewrite_preserves_original_ledger(self) -> None:
        destination = self.output / "ledger.jsonl"
        destination.write_text('{"original": true}\n', encoding="utf-8")
        with self.assertRaises(TypeError):
            storage.rewrite_jsonl(destination, [{"unsupported": object()}])
        self.assertEqual(destination.read_text(), '{"original": true}\n')

    def test_failed_atomic_replace_preserves_destination_and_removes_temporary(self) -> None:
        destination = self.output / "result.json"
        destination.write_text('"original"\n', encoding="utf-8")
        with (
            patch.object(storage.os, "replace", side_effect=OSError("disk unavailable")),
            self.assertRaisesRegex(OSError, "disk unavailable"),
        ):
            storage.write_json(destination, {"new": True})
        self.assertEqual(destination.read_text(), '"original"\n')
        self.assertEqual(list(self.output.iterdir()), [destination])

    def test_standard_temporary_path_and_missing_parents_remain_supported(self) -> None:
        destination = self.unresolved_root / "new" / "nested" / "data.json"
        storage.write_json(destination, {"title": "évidence", "api_key": "sk-hidden"})
        payload = json.loads(destination.read_text())
        self.assertEqual(payload["title"], "évidence")
        self.assertNotEqual(payload["api_key"], "sk-hidden")
        ledger = destination.with_suffix(".jsonl")
        storage.append_jsonl(ledger, {"order": 1})
        storage.append_jsonl(ledger, {"order": 2})
        self.assertEqual(storage.read_jsonl(ledger), ([{"order": 1}, {"order": 2}], []))
        storage.rewrite_jsonl(ledger, [{"order": 3}])
        self.assertEqual(storage.read_jsonl(ledger), ([{"order": 3}], []))
        storage.rewrite_jsonl(ledger, [])
        self.assertEqual(ledger.read_bytes(), b"")

    def test_operator_directory_alias_remains_supported_outside_controller_mode(self) -> None:
        alias = self.root / "operator-alias"
        alias.symlink_to(self.output, target_is_directory=True)
        with patch.dict(os.environ):
            os.environ.pop("PRAXIST_CONTROLLER_STATE_DIR", None)
            storage.write_json(alias / "result.json", {"result": "normal"})
        self.assertEqual(
            json.loads((self.output / "result.json").read_text()), {"result": "normal"}
        )

    def test_safe_reader_rejects_private_file_symlink_and_hardlink(self) -> None:
        destination = self.output / "peer.json"
        destination.symlink_to(self.authority)
        with self.assertRaises(OSError):
            storage.read_file_bytes(destination)
        destination.unlink()
        os.link(self.authority, destination)
        with self.assertRaises(OSError):
            storage.read_file_bytes(destination)

    def test_safe_reader_rejects_redirected_ancestor_in_controller_mode(self) -> None:
        self.output.chmod(0o777)
        (self.output / "redirect").symlink_to(self.private, target_is_directory=True)
        with (
            patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(self.private)}),
            self.assertRaises(OSError),
        ):
            storage.read_file_bytes(self.output / "redirect" / self.authority.name)

    def test_controller_reader_rejects_nested_directory_links_even_when_root_owned(self) -> None:
        original_stat = os.stat
        original_fstat = os.fstat

        def root_owned(info):
            fields = list(info)
            fields[4] = 0
            return os.stat_result(fields)

        for name in ("redirect", "var"):
            with self.subTest(name=name):
                link = self.output / name
                link.symlink_to(self.private, target_is_directory=True)
                with (
                    patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(self.private)}),
                    patch.object(
                        storage.os,
                        "stat",
                        side_effect=lambda *args, **kwargs: root_owned(
                            original_stat(*args, **kwargs)
                        ),
                    ),
                    patch.object(
                        storage.os,
                        "fstat",
                        side_effect=lambda fd: root_owned(original_fstat(fd)),
                    ),
                    self.assertRaises(OSError),
                ):
                    storage.read_file_bytes(link / self.authority.name)

    def test_actual_macos_system_alias_remains_readable_in_controller_mode(self) -> None:
        for name in ("var", "tmp"):
            target = Path("/private") / name
            alias = Path("/") / name
            if alias.is_symlink() and self.authority.is_relative_to(target):
                source = alias / self.authority.relative_to(target)
                with patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(self.private)}):
                    self.assertEqual(storage.read_file_bytes(source), b'"private authority"\n')
                return
        self.skipTest("temporary fixture does not use a macOS system directory alias")

    def test_safe_reader_does_not_create_missing_directories(self) -> None:
        path = self.output / "missing" / "file.json"
        with self.assertRaises(FileNotFoundError):
            storage.read_file_bytes(path)
        self.assertFalse(path.parent.exists())
        self.assertEqual(storage.read_file_bytes(self.authority), b'"private authority"\n')
