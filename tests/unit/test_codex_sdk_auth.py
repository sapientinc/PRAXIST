"""Offline tests for saved ChatGPT authentication used by Codex-native mode."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from praxist.core.credentials import CredentialRef


class ChatGptCredentialDiscoveryTest(unittest.TestCase):
    @staticmethod
    def _completed(
        args: list[str], *, exit_code: int = 0, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, exit_code, stdout=stdout, stderr=stderr)

    def test_chatgpt_login_returns_redacted_home_scoped_reference(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import (
            OPENAI_COMPATIBLE_PROVIDER_REF,
            chatgpt_credential_key_id,
            discover_chatgpt_credential,
            is_chatgpt_subscription_credential,
        )

        with (
            patch.dict(os.environ, {"CODEX_HOME": "/operator/codex"}, clear=False),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.subprocess.run",
                return_value=self._completed(
                    ["test-codex", "login", "status"], stdout="Logged in using ChatGPT\n"
                ),
            ) as run,
        ):
            credential = discover_chatgpt_credential(
                OPENAI_COMPATIBLE_PROVIDER_REF,
                codex_bin="test-codex",
            )

        expected_key = chatgpt_credential_key_id(Path("/operator/codex"))
        self.assertEqual(
            credential,
            CredentialRef(
                scope="model_provider",
                provider="openai_compatible",
                target_ref=OPENAI_COMPATIBLE_PROVIDER_REF,
                key_id=expected_key,
                source="runtime_session",
            ),
        )
        self.assertTrue(is_chatgpt_subscription_credential(credential))
        self.assertEqual(run.call_args.args[0], ["test-codex", "login", "status"])

    def test_subscription_predicate_validates_complete_reference(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import (
            chatgpt_credential_key_id,
            is_chatgpt_subscription_credential,
        )

        key_id = chatgpt_credential_key_id(Path("/operator/codex"))
        valid = CredentialRef(
            scope="model_provider",
            provider="openai_compatible",
            target_ref="model_provider:openai_compatible",
            key_id=key_id,
            source="runtime_session",
        )
        self.assertTrue(is_chatgpt_subscription_credential(valid))
        for field, value in (
            ("scope", "tool_server"),
            ("provider", "openrouter"),
            ("target_ref", "model_provider:openrouter"),
            ("key_id", "openai_compatible:codex_sdk:chatgpt"),
        ):
            payload = {**valid.__dict__, field: value}
            with self.subTest(field=field):
                self.assertFalse(is_chatgpt_subscription_credential(CredentialRef(**payload)))

    def test_probe_uses_non_secret_allowlist(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import (
            SUBSCRIPTION_ENV_KEYS,
            discover_chatgpt_credential,
        )

        polluted = {
            **{key: f"secret-{key}" for key in SUBSCRIPTION_ENV_KEYS},
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "HF_TOKEN": "task-secret",
            "PATH": "/test/bin",
            "HOME": "/test/home",
        }
        with (
            patch.dict(os.environ, polluted, clear=False),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.subprocess.run",
                return_value=self._completed(
                    ["test-codex", "login", "status"], stdout="Logged in using ChatGPT"
                ),
            ) as run,
        ):
            discover_chatgpt_credential("model_provider:openai_compatible", codex_bin="test-codex")

        probe_env = run.call_args.kwargs["env"]
        for key in (
            *SUBSCRIPTION_ENV_KEYS,
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
            "HF_TOKEN",
        ):
            self.assertNotIn(key, probe_env)
        self.assertEqual(probe_env["PATH"], "/test/bin")
        self.assertEqual(probe_env["HOME"], "/test/home")

    def test_unsupported_provider_does_not_probe(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import discover_chatgpt_credential

        with patch("praxist.plugins.agent_runtimes.codex_sdk._auth.subprocess.run") as run:
            credential = discover_chatgpt_credential("model_provider:openrouter")

        self.assertIsNone(credential)
        run.assert_not_called()

    def test_status_must_be_exact_and_errors_remain_redacted(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import discover_chatgpt_credential

        identity = "person@example.invalid"
        token = "private-token-test-value"
        for output in (
            "Not logged in using ChatGPT",
            f"Logged in using API key for {identity}: {token}",
        ):
            with self.subTest(output=output):
                with (
                    patch(
                        "praxist.plugins.agent_runtimes.codex_sdk._auth.subprocess.run",
                        return_value=self._completed(
                            ["test-codex", "login", "status"], stdout=output
                        ),
                    ),
                    self.assertRaisesRegex(RuntimeError, "codex-native") as caught,
                ):
                    discover_chatgpt_credential(
                        "model_provider:openai_compatible", codex_bin="test-codex"
                    )
                self.assertNotIn(identity, str(caught.exception))
                self.assertNotIn(token, str(caught.exception))

    def test_explicit_setup_repairs_api_key_login_with_sdk_pinned_codex(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import ensure_chatgpt_login

        def pinned_binary(_requested: object = None) -> str:
            self.assertNotIn("PRAXIST_CODEX_BIN", os.environ)
            return "/sdk/codex"

        calls = [
            self._completed(
                ["/sdk/codex", "login", "status"],
                stdout="Logged in using an API key\n",
            ),
            self._completed(["/sdk/codex", "login"]),
            self._completed(
                ["/sdk/codex", "login", "status"],
                stdout="Logged in using ChatGPT\n",
            ),
        ]
        with (
            patch.dict(
                os.environ,
                {"PRAXIST_CODEX_BIN": "/operator/custom-codex"},
                clear=False,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.resolve_codex_binary",
                side_effect=pinned_binary,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth._subscription_probe_env",
                return_value={"HOME": "/operator"},
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.subprocess.run",
                side_effect=calls,
            ) as run,
        ):
            login_started = ensure_chatgpt_login(allow_interactive=True)
            restored_override = os.environ["PRAXIST_CODEX_BIN"]

        self.assertTrue(login_started)
        self.assertEqual(restored_override, "/operator/custom-codex")
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ["/sdk/codex", "login", "status"],
                ["/sdk/codex", "login"],
                ["/sdk/codex", "login", "status"],
            ],
        )
        self.assertEqual(run.call_args_list[1].kwargs["env"], {"HOME": "/operator"})
        self.assertNotIn("capture_output", run.call_args_list[1].kwargs)

    def test_explicit_setup_does_not_reopen_an_existing_chatgpt_login(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import ensure_chatgpt_login

        with (
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.resolve_codex_binary",
                return_value="/sdk/codex",
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.subprocess.run",
                return_value=self._completed(
                    ["/sdk/codex", "login", "status"],
                    stdout="Logged in using ChatGPT\n",
                ),
            ) as run,
        ):
            login_started = ensure_chatgpt_login(allow_interactive=True)

        self.assertFalse(login_started)
        run.assert_called_once()

    def test_explicit_setup_never_launches_login_without_a_local_terminal(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import ensure_chatgpt_login

        with (
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.resolve_codex_binary",
                return_value="/sdk/codex",
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.subprocess.run",
                return_value=self._completed(
                    ["/sdk/codex", "login", "status"],
                    stdout="Logged in using an API key\n",
                ),
            ) as run,
            self.assertRaisesRegex(RuntimeError, "local interactive terminal"),
        ):
            ensure_chatgpt_login(allow_interactive=False)

        run.assert_called_once()

    def test_explicit_setup_reports_cancelled_login_without_reprobing(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import ensure_chatgpt_login

        calls = [
            self._completed(
                ["/sdk/codex", "login", "status"],
                stdout="Logged in using an API key\n",
            ),
            self._completed(["/sdk/codex", "login"], exit_code=1),
        ]
        with (
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.resolve_codex_binary",
                return_value="/sdk/codex",
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.subprocess.run",
                side_effect=calls,
            ) as run,
            self.assertRaisesRegex(RuntimeError, "did not complete"),
        ):
            ensure_chatgpt_login(allow_interactive=True)

        self.assertEqual(run.call_count, 2)

    def test_timeout_and_os_error_are_normalized_without_probe_output(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import discover_chatgpt_credential

        for error, message in (
            (subprocess.TimeoutExpired(["codex"], 5, output="private"), "Timed out"),
            (OSError("private executable detail"), "Unable to execute"),
        ):
            with self.subTest(error=type(error).__name__):
                with (
                    patch(
                        "praxist.plugins.agent_runtimes.codex_sdk._auth.subprocess.run",
                        side_effect=error,
                    ),
                    self.assertRaisesRegex(RuntimeError, message) as caught,
                ):
                    discover_chatgpt_credential(
                        "model_provider:openai_compatible", codex_bin="test-codex"
                    )
                self.assertNotIn("private", str(caught.exception))

    def test_default_binary_comes_from_sdk_package(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import resolve_codex_binary

        module = types.ModuleType("codex_cli_bin")
        module.bundled_codex_path = lambda: Path("/sdk/codex")
        with (
            patch.dict(os.environ, {"PRAXIST_CODEX_BIN": ""}, clear=False),
            patch.dict(sys.modules, {"codex_cli_bin": module}),
        ):
            self.assertEqual(resolve_codex_binary(), "/sdk/codex")

    def test_default_binary_reports_missing_sdk_entrypoint(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import resolve_codex_binary

        module = types.ModuleType("codex_cli_bin")
        with (
            patch.dict(os.environ, {"PRAXIST_CODEX_BIN": ""}, clear=False),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.importlib.import_module",
                return_value=module,
            ),
            self.assertRaisesRegex(OSError, "SDK-pinned Codex runtime is not installed"),
        ):
            resolve_codex_binary()

    def test_operator_home_uses_explicit_mapping_without_global_fallback(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import operator_codex_home

        self.assertEqual(
            operator_codex_home({"CODEX_HOME": "/operator/custom", "HOME": "/operator"}),
            Path("/operator/custom"),
        )
        self.assertEqual(operator_codex_home({"HOME": "/operator"}), Path("/operator/.codex"))
        with self.assertRaises(ValueError):
            operator_codex_home({})

    def test_file_account_identity_survives_token_refresh_and_changes_with_account(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import chatgpt_credential_key_id

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            auth_file = home / "auth.json"
            auth_file.write_text(
                '{"auth_mode":"chatgpt","tokens":{"account_id":"account-a",'
                '"access_token":"token-one"}}',
                encoding="utf-8",
            )
            original = chatgpt_credential_key_id(home)
            auth_file.write_text(
                '{"auth_mode":"chatgpt","tokens":{"account_id":"account-a",'
                '"access_token":"token-two"}}',
                encoding="utf-8",
            )
            self.assertEqual(chatgpt_credential_key_id(home), original)
            auth_file.write_text(
                '{"auth_mode":"chatgpt","tokens":{"account_id":"account-b"}}',
                encoding="utf-8",
            )
            self.assertNotEqual(chatgpt_credential_key_id(home), original)

    def test_file_auth_is_staged_privately_without_mutating_operator_home(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import (
            chatgpt_credential_key_id,
            stage_chatgpt_home,
        )

        with tempfile.TemporaryDirectory() as tmp:
            operator_home = Path(tmp) / "operator"
            operator_home.mkdir()
            source = operator_home / "auth.json"
            source.write_text(
                '{"auth_mode":"chatgpt","tokens":{"account_id":"account-a"}}',
                encoding="utf-8",
            )
            before = source.read_bytes()
            expected_key = chatgpt_credential_key_id(operator_home)
            staged = stage_chatgpt_home(operator_home)
            staged_path = staged.path
            try:
                self.assertNotEqual(staged_path, operator_home)
                self.assertEqual(staged.credential_store, "file")
                self.assertEqual(staged.credential_key_id, expected_key)
                self.assertEqual((staged_path / "auth.json").read_bytes(), before)
                self.assertEqual((staged_path / "auth.json").stat().st_mode & 0o777, 0o600)
                source.write_text(
                    '{"auth_mode":"chatgpt","tokens":{"account_id":"account-b"}}',
                    encoding="utf-8",
                )
                self.assertNotEqual(chatgpt_credential_key_id(operator_home), expected_key)
                self.assertEqual((staged_path / "auth.json").read_bytes(), before)
                (staged_path / "auth.json").write_text("runtime-refresh", encoding="utf-8")
                self.assertIn(b"account-b", source.read_bytes())
            finally:
                staged.close()
            self.assertFalse(staged_path.exists())

    def test_keyring_auth_preserves_operator_home_as_lookup_identity(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._auth import stage_chatgpt_home

        with tempfile.TemporaryDirectory() as tmp:
            operator_home = Path(tmp)
            marker = operator_home / "config.toml"
            marker.write_text('model = "operator-default"\n', encoding="utf-8")
            staged = stage_chatgpt_home(operator_home)

            self.assertEqual(staged.path, operator_home)
            self.assertEqual(staged.credential_store, "keyring")
            staged.close()
            self.assertEqual(marker.read_text(encoding="utf-8"), 'model = "operator-default"\n')


class CodexSdkRuntimeCredentialHookTest(unittest.TestCase):
    def test_runtime_hook_delegates_to_sdk_auth_discovery(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk.adapter import CodexSdkRuntime

        credential = CredentialRef(
            scope="model_provider",
            provider="openai_compatible",
            target_ref="model_provider:openai_compatible",
            key_id="openai_compatible:codex_sdk:chatgpt:abc123",
            source="runtime_session",
        )
        with patch(
            "praxist.plugins.agent_runtimes.codex_sdk.adapter.discover_chatgpt_credential",
            return_value=credential,
        ) as discover:
            observed = CodexSdkRuntime().discover_managed_credential(
                "model_provider:openai_compatible"
            )

        self.assertIs(observed, credential)
        discover.assert_called_once_with("model_provider:openai_compatible")


class CodexSdkProcessEnvironmentTest(unittest.TestCase):
    def test_codex_originator_is_preserved_without_forwarding_credentials(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk.adapter import _client_process_env

        source = {
            "CODEX_INTERNAL_ORIGINATOR_OVERRIDE": "Codex Desktop",
            "OPENAI_API_KEY": "openai-secret",
            "PATH": "/test/bin",
            "UNRELATED_SECRET": "other-secret",
        }
        with patch.dict(os.environ, source, clear=True):
            process_env = _client_process_env(
                "openai",
                source,
                Path("/private/codex"),
                subscription=True,
            )

        self.assertEqual(
            process_env["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"],
            "Codex Desktop",
        )
        self.assertEqual(process_env["PATH"], "/test/bin")
        self.assertEqual(process_env["CODEX_HOME"], "/private/codex")
        self.assertEqual(process_env["OPENAI_API_KEY"], "")
        self.assertEqual(process_env["UNRELATED_SECRET"], "")


class ChatGptModelCatalogTest(unittest.TestCase):
    def test_catalog_uses_private_staged_auth_and_closes_resources(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk import adapter

        staged = types.SimpleNamespace(
            path=Path("/private/codex"), credential_store="file", close=Mock()
        )
        response = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(model="gpt-5.6-luna"),
                types.SimpleNamespace(model="gpt-5.4"),
                types.SimpleNamespace(model="gpt-5.6-luna"),
            ]
        )
        client = Mock()
        client.account.return_value = types.SimpleNamespace(
            account=types.SimpleNamespace(root=types.SimpleNamespace(type="chatgpt"))
        )
        client.models.return_value = response
        config = object()
        sdk = {
            "Codex": Mock(return_value=client),
            "CodexConfig": Mock(return_value=config),
        }
        with (
            patch.object(adapter, "verify_chatgpt_login") as verify_login,
            patch.object(adapter, "operator_codex_home", return_value=Path("/operator/codex")),
            patch.object(adapter, "stage_chatgpt_home", return_value=staged),
            patch.object(adapter, "resolve_codex_binary", return_value="/sdk/codex"),
            patch.object(
                adapter, "_client_process_env", return_value={"CODEX_HOME": str(staged.path)}
            ),
            patch.object(adapter, "_load_sdk", return_value=sdk),
        ):
            models = adapter.available_chatgpt_models()

        self.assertEqual(models, ("gpt-5.4", "gpt-5.6-luna"))
        verify_login.assert_called_once_with()
        client.account.assert_called_once_with(refresh_token=False)
        client.models.assert_called_once_with(include_hidden=False)
        self.assertIn(
            'cli_auth_credentials_store="file"',
            sdk["CodexConfig"].call_args.kwargs["config_overrides"],
        )
        client.close.assert_called_once_with()
        staged.close.assert_called_once_with()

    def test_catalog_rejects_non_chatgpt_app_server_account(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk import adapter

        staged = types.SimpleNamespace(
            path=Path("/operator/codex"), credential_store="keyring", close=Mock()
        )
        client = Mock()
        client.account.return_value = types.SimpleNamespace(
            account=types.SimpleNamespace(root=types.SimpleNamespace(type="apiKey"))
        )
        sdk = {
            "Codex": Mock(return_value=client),
            "CodexConfig": Mock(return_value=object()),
        }
        with (
            patch.object(adapter, "verify_chatgpt_login"),
            patch.object(adapter, "operator_codex_home", return_value=staged.path),
            patch.object(adapter, "stage_chatgpt_home", return_value=staged),
            patch.object(adapter, "resolve_codex_binary", return_value="/sdk/codex"),
            patch.object(adapter, "_client_process_env", return_value={}),
            patch.object(adapter, "_load_sdk", return_value=sdk),
            self.assertRaisesRegex(RuntimeError, "model catalog"),
        ):
            adapter.available_chatgpt_models()

        client.models.assert_not_called()
        client.close.assert_called_once_with()
        staged.close.assert_called_once_with()

    def test_selected_model_must_be_present_in_account_catalog(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk import adapter

        with patch.object(
            adapter,
            "available_chatgpt_models",
            return_value=("gpt-5.4", "gpt-5.6-luna"),
        ):
            self.assertEqual(
                adapter.verify_chatgpt_model_available("GPT-5.6-LUNA"),
                "gpt-5.6-luna",
            )
            with self.assertRaisesRegex(RuntimeError, "gpt-unsupported"):
                adapter.verify_chatgpt_model_available("gpt-unsupported")


if __name__ == "__main__":
    unittest.main()
