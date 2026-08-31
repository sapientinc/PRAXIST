"""Explicit filesystem roots survive Codex's named-permission boundary."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from praxist.core.runtimes import AgentRuntimeExecutionContext
from praxist.plugins.agent_runtimes.codex_sdk import adapter
from praxist.plugins.agent_runtimes.codex_sdk._sandbox import (
    CodexSandboxSettings,
    sandbox_settings,
)
from tests.unit.test_codex_sdk_adapter import _request, _SdkHarness


def _intent(root: Path) -> dict[str, Any]:
    return {
        "filesystem": "workspace_write",
        "network": "off",
        "approval": "auto",
        "readable_roots": [str(root / "inputs")],
        "writable_roots": [str(root / "work")],
        "denied_paths": [str(root / "inputs" / "private")],
    }


def _profile(settings: CodexSandboxSettings) -> dict[str, Any]:
    return cast(dict[str, Any], settings.config["permissions"])["praxist_task"]


class PermissionProfileMappingTests(unittest.TestCase):
    def test_explicit_roots_use_default_deny_without_legacy_workspace_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            settings = sandbox_settings(
                _request(str(root), runtime_options={"sandbox_intent": _intent(root)})
            )
            self.assertEqual(settings.config.get("default_permissions"), "praxist_task")
            self.assertEqual(
                _profile(settings),
                {
                    "filesystem": {
                        ":root": "deny",
                        ":minimal": "read",
                        str(root / "inputs"): "read",
                        str(root / "work"): "write",
                        str(root / "inputs" / "private"): "deny",
                    },
                    "network": {"enabled": False},
                },
            )
            self.assertNotIn("sandbox_workspace_write", settings.config)

    def test_empty_path_lists_still_activate_default_deny(self):
        settings = sandbox_settings(
            _request("/tmp", runtime_options={"sandbox_intent": {"readable_roots": []}})
        )
        self.assertEqual(settings.config.get("default_permissions"), "praxist_task")
        self.assertEqual(
            _profile(settings)["filesystem"],
            {":root": "deny", ":minimal": "read"},
        )

    def test_read_only_call_downgrades_all_write_roots(self):
        for explicit_flag in (False, True):
            with self.subTest(explicit_flag=explicit_flag), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                intent = _intent(root)
                if not explicit_flag:
                    intent["filesystem"] = "read_only"
                settings = sandbox_settings(
                    _request(
                        str(root),
                        runtime_options={
                            "sandbox_intent": intent,
                            "require_read_only_runtime": explicit_flag,
                        },
                    )
                )
                self.assertEqual(settings.config.get("default_permissions"), "praxist_task")
                filesystem = _profile(settings)["filesystem"]
                self.assertEqual(filesystem[str(root / "work")], "read")
                self.assertNotIn("write", filesystem.values())
                self.assertEqual(settings.sandbox, "read_only")

    def test_denied_ancestor_cannot_be_reopened_by_a_more_specific_grant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            intent = _intent(root)
            intent["readable_roots"].append(str(root / "inputs" / "private" / "readable"))
            intent["writable_roots"].extend(
                [str(root / "inputs" / "private"), str(root / "inputs" / "private" / "write")]
            )
            settings = sandbox_settings(
                _request(str(root), runtime_options={"sandbox_intent": intent})
            )
            self.assertEqual(settings.config.get("default_permissions"), "praxist_task")
            filesystem = _profile(settings)["filesystem"]
            self.assertEqual(filesystem[str(root / "inputs" / "private")], "deny")
            self.assertNotIn(str(root / "inputs" / "private" / "readable"), filesystem)
            self.assertNotIn(str(root / "inputs" / "private" / "write"), filesystem)

    def test_denials_collapse_after_resolving_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            private = root / "private"
            child = private / "child"
            child.mkdir(parents=True)
            alias = root / "alias"
            alias.symlink_to(private, target_is_directory=True)
            intent = _intent(root)
            intent["denied_paths"] = [str(alias / "child"), str(private), str(private)]
            intent["readable_roots"].append(str(alias / "child" / "evidence"))
            settings = sandbox_settings(
                _request(str(root), runtime_options={"sandbox_intent": intent})
            )

            filesystem = _profile(settings)["filesystem"]
            self.assertEqual(filesystem[str(private)], "deny")
            self.assertNotIn(str(child), filesystem)
            self.assertNotIn(str(child / "evidence"), filesystem)
            self.assertEqual(
                [path for path, access in filesystem.items() if access == "deny"],
                [":root", str(private)],
            )

    def test_profile_network_is_independent_of_the_legacy_full_access_limitation(self):
        settings = sandbox_settings(
            _request(
                "/tmp",
                runtime_options={
                    "sandbox_intent": {
                        "filesystem": "full",
                        "network": "off",
                        "writable_roots": [],
                    }
                },
            )
        )
        self.assertFalse(_profile(settings)["network"]["enabled"])

    def test_invalid_path_contract_fails_before_sdk(self):
        for value in (
            "/tmp",
            ["relative/path"],
            [""],
            [None],
            [1],
            ["/tmp/*"],
            ["/tmp/../private"],
            ["//tmp"],
            ["/tmp/line\nbreak"],
            ["/tmp/null\0byte"],
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sandbox_settings(
                    _request("/tmp", runtime_options={"sandbox_intent": {"readable_roots": value}})
                )


class PermissionProfileDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_is_selected_without_a_legacy_sandbox_override(self):
        runtime = adapter.CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", return_value=harness.sdk()),
        ):
            root = Path(tmp).resolve()
            request = _request(str(root), runtime_options={"sandbox_intent": _intent(root)})
            result = await runtime.execute(
                request, AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "offline-fixture"})
            )
        self.assertTrue(result.success, result.error)
        actual = harness.clients[0].thread_calls[0]
        self.assertNotIn("sandbox", actual)
        self.assertEqual(actual["config"]["default_permissions"], "praxist_task")
        self.assertNotIn("sandbox_workspace_write", actual["config"])

    async def test_reused_client_receives_disjoint_profiles_per_thread(self):
        runtime = adapter.CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", return_value=harness.sdk()),
        ):
            root = Path(tmp).resolve()
            for peer in ("one", "two"):
                request = _request(
                    str(root),
                    request_id=peer,
                    runtime_options={"sandbox_intent": _intent(root / peer)},
                )
                result = await runtime.execute(
                    request, AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "offline-fixture"})
                )
                self.assertTrue(result.success, result.error)
            self.assertEqual(len(harness.clients), 1)
            calls = harness.clients[0].thread_calls
            first = calls[0]["config"]["permissions"]["praxist_task"]["filesystem"]
            second = calls[1]["config"]["permissions"]["praxist_task"]["filesystem"]
            self.assertEqual(first[str(root / "one" / "work")], "write")
            self.assertNotIn(str(root / "one" / "work"), second)
            self.assertEqual(second[str(root / "two" / "work")], "write")
            self.assertNotIn(str(root / "two" / "work"), first)

    async def test_legacy_request_still_passes_legacy_sandbox(self):
        runtime = adapter.CodexSdkRuntime()
        self.addAsyncCleanup(runtime.aclose)
        harness = _SdkHarness()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(adapter, "_load_sdk", return_value=harness.sdk()),
        ):
            request = replace(
                _request(tmp),
                runtime_options={
                    "run_dir": tmp,
                    "sandbox_intent": {
                        "filesystem": "workspace_write",
                        "network": "off",
                        "approval": "auto",
                    },
                },
            )
            result = await runtime.execute(
                request, AgentRuntimeExecutionContext(env={"OPENAI_API_KEY": "offline-fixture"})
            )
        self.assertTrue(result.success, result.error)
        actual = harness.clients[0].thread_calls[0]
        self.assertEqual(actual["sandbox"], "sandbox:workspace_write")
        self.assertNotIn("default_permissions", actual["config"])
