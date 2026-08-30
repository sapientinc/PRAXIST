"""Tests for Praxist setup/doctor/configure-llm/install-skills CLI commands."""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import praxist
from praxist.cli.examples import ExampleInstallResult
from praxist.cli.setup import SETUP_PROFILES

REPO_ROOT = Path(__file__).resolve().parents[2]


class CliRunnerMixin:
    """Small helper for invoking the top-level CLI dispatcher."""

    def _run(
        self,
        argv: list[str],
        *,
        stdin: str = "",
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        from praxist.cli import main

        stdout, stderr = io.StringIO(), io.StringIO()
        patch_env = env or {}
        try:
            with (
                patch.dict(os.environ, patch_env, clear=False),
                patch("sys.stdin", io.StringIO(stdin)),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                main(argv)
            code = 0
        except SystemExit as exc:
            code = int(exc.code or 0)
        return code, stdout.getvalue(), stderr.getvalue()


class VersionSurfaceTest(CliRunnerMixin, unittest.TestCase):
    """Package version is available through Python and the CLI."""

    def test_package_version_matches_pyproject_when_metadata_installed(self) -> None:
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        # Editable installs expose distribution metadata; a raw source-tree import
        # may fall back to 0+unknown, which is acceptable outside an installed env.
        if praxist.__version__ != "0+unknown":
            self.assertEqual(praxist.__version__, pyproject["project"]["version"])

    def test_top_level_version_flag_reports_package_version(self) -> None:
        code, out, err = self._run(["--version"])
        self.assertEqual(code, 0)
        self.assertIn(praxist.__version__, out + err)


class ConfigureLLMTest(CliRunnerMixin, unittest.TestCase):
    """``praxist configure-llm`` writes a dedicated redacted env file."""

    def test_direct_configuration_clears_stale_setup_profile_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text(
                "PRAXIST_SETUP_PROFILE=codex-native\nPRAXIST_AGENT_SYSTEM=codex_sdk\n",
                encoding="utf-8",
            )
            code, out, err = self._run(
                [
                    "configure-llm",
                    "--provider",
                    "deepseek",
                    "--model",
                    "deepseek-v4-pro[1m]",
                    "--agent-system",
                    "claude_sdk",
                    "--no-api-key",
                    "--config-file",
                    str(config_file),
                ]
            )

            self.assertEqual(code, 0, msg=out + err)
            self.assertNotIn("PRAXIST_SETUP_PROFILE", config_file.read_text(encoding="utf-8"))

    def test_api_key_stdin_writes_0600_env_file_without_echoing_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            secret = "sk-test-secret"
            code, out, err = self._run(
                [
                    "configure-llm",
                    "--provider",
                    "openrouter",
                    "--agent-system",
                    "codex_sdk",
                    "--model",
                    "deepseek/demo",
                    "--api-key-stdin",
                    "--config-file",
                    str(config_file),
                    "--no-project-env",
                ],
                stdin=f"{secret}\n",
            )
            self.assertEqual(code, 0, msg=out + err)
            self.assertNotIn(secret, out + err)
            text = config_file.read_text(encoding="utf-8")
            self.assertIn("export PRAXIST_LLM_PROVIDER=openrouter", text)
            self.assertIn(
                "export PRAXIST_MODEL_PROVIDER_REF=model_provider:openrouter",
                text,
            )
            self.assertIn("export PRAXIST_AGENT_SYSTEM=codex_sdk", text)
            self.assertIn(
                "export PRAXIST_AGENT_RUNTIME_REF=agent_runtime:codex_sdk",
                text,
            )
            self.assertIn("export PRAXIST_MODEL=deepseek/demo", text)
            self.assertIn("export OPENROUTER_API_KEY=sk-test-secret", text)
            mode = stat.S_IMODE(config_file.stat().st_mode)
            self.assertEqual(mode, 0o600)
            self.assertIn("Reading OPENROUTER_API_KEY from stdin", err)

    def test_api_key_stdin_uses_hidden_prompt_when_interactive(self) -> None:
        from praxist.cli.configure_llm import configure_llm

        class FakeTTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            secret = "sk-interactive-secret"
            with (
                patch("sys.stdin", FakeTTY()),
                patch(
                    "praxist.cli.configure_llm.read_masked_secret",
                    return_value=secret,
                ) as secret_prompt,
            ):
                result = configure_llm(
                    provider="openrouter",
                    model=None,
                    agent_system="codex_sdk",
                    api_key_stdin=True,
                    api_key_env=None,
                    no_api_key=False,
                    config_file=config_file,
                    project_config_file=None,
                    dry_run=False,
                )
            self.assertEqual(secret_prompt.call_args.args[0], "Enter OPENROUTER_API_KEY (masked): ")
            self.assertEqual(result["key_status"], "written")
            self.assertIn("OPENROUTER_API_KEY=sk-interactive-secret", config_file.read_text())

    def test_interactive_api_key_cancellation_returns_130(self) -> None:
        from praxist.cli.configure_llm import ConfigureLLMCancelled

        with patch(
            "praxist.cli.configure_llm._read_api_key_from_stdin",
            side_effect=ConfigureLLMCancelled("input was cancelled"),
        ):
            code, out, err = self._run(
                [
                    "configure-llm",
                    "--provider",
                    "deepseek",
                    "--api-key-stdin",
                    "--no-project-env",
                ]
            )
        self.assertEqual(code, 130, msg=out + err)
        self.assertIn("cancelled", err)

    def test_api_key_stdin_normalizes_crlf_and_rejects_controls_and_overflow(self) -> None:
        from praxist.cli.configure_llm import ConfigureLLMError, _read_api_key_from_stdin

        with patch("sys.stdin", io.StringIO("key-value\r\n")):
            self.assertEqual(_read_api_key_from_stdin("TEST_KEY"), "key-value")
        with (
            patch("sys.stdin", io.StringIO("bad\x00key\n")),
            self.assertRaisesRegex(ConfigureLLMError, "control characters"),
        ):
            _read_api_key_from_stdin("TEST_KEY")
        with (
            patch("sys.stdin", io.StringIO("x" * 4097 + "\n")),
            self.assertRaisesRegex(ConfigureLLMError, "safety limit"),
        ):
            _read_api_key_from_stdin("TEST_KEY")

    def test_configure_llm_can_also_write_project_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config" / "env"
            project_env = Path(tmp) / "project" / ".env"
            secret = "sk-project-secret"
            code, out, err = self._run(
                [
                    "configure-llm",
                    "--provider",
                    "openrouter",
                    "--agent-system",
                    "codex_sdk",
                    "--model",
                    "deepseek/demo",
                    "--api-key-stdin",
                    "--config-file",
                    str(config_file),
                    "--project-env-file",
                    str(project_env),
                ],
                stdin=f"{secret}\n",
            )
            self.assertEqual(code, 0, msg=out + err)
            self.assertNotIn(secret, out + err)
            self.assertIn("updated Praxist project env", err)
            self.assertTrue(config_file.is_file())
            self.assertTrue(project_env.is_file())
            text = project_env.read_text(encoding="utf-8")
            self.assertIn("export PRAXIST_LLM_PROVIDER=openrouter", text)
            self.assertIn(
                "export PRAXIST_MODEL_PROVIDER_REF=model_provider:openrouter",
                text,
            )
            self.assertIn("export PRAXIST_AGENT_SYSTEM=codex_sdk", text)
            self.assertIn(
                "export PRAXIST_AGENT_RUNTIME_REF=agent_runtime:codex_sdk",
                text,
            )
            self.assertIn("export PRAXIST_MODEL=deepseek/demo", text)
            self.assertIn("export OPENROUTER_API_KEY=sk-project-secret", text)

    def test_configure_llm_replaces_stale_canonical_runtime_and_provider_refs(self) -> None:
        from praxist.cli import start
        from praxist.cli._setup_common import load_env_file

        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text(
                "\n".join(
                    [
                        "export PRAXIST_AGENT_SYSTEM=claude_sdk",
                        "export PRAXIST_AGENT_RUNTIME_REF=agent_runtime:claude_sdk",
                        "export RUNTIME_REF=agent_runtime:claude_sdk",
                        "export PRAXIST_LLM_PROVIDER=deepseek",
                        "export PRAXIST_MODEL_PROVIDER_REF=model_provider:deepseek_alias",
                        "export MODEL_PROVIDER_REF=model_provider:deepseek_alias",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            result = self._run(
                [
                    "configure-llm",
                    "--provider",
                    "openrouter",
                    "--agent-system",
                    "codex_sdk",
                    "--no-api-key",
                    "--config-file",
                    str(config_file),
                    "--no-project-env",
                ]
            )
            self.assertEqual(result[0], 0, msg=result[1] + result[2])

            with patch.dict(os.environ, {}, clear=True):
                load_env_file(config_file)
                agent, runtime = start._resolve_runtime_selection(None, None)
                provider = start._resolve_provider_ref(None, agent)

            self.assertEqual((agent, runtime), ("codex_sdk", "agent_runtime:codex_sdk"))
            self.assertEqual(provider, "model_provider:openrouter")

    def test_api_key_env_reports_variable_name_not_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            secret = "sk-hidden"
            code, out, err = self._run(
                [
                    "configure-llm",
                    "--provider",
                    "anthropic",
                    "--api-key-env",
                    "ANTHROPIC_API_KEY",
                    "--config-file",
                    str(config_file),
                    "--no-project-env",
                ],
                env={"ANTHROPIC_API_KEY": secret},
            )
            self.assertEqual(code, 0, msg=out + err)
            self.assertNotIn(secret, out + err)
            self.assertIn("ANTHROPIC_API_KEY=written", err)

    def test_json_dry_run_and_source_command_do_not_write_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            project_env = Path(tmp) / ".env"
            code, out, err = self._run(
                [
                    "configure-llm",
                    "--provider",
                    "openai_compatible",
                    "--agent-system",
                    "codex_sdk",
                    "--model",
                    "gpt-test",
                    "--no-api-key",
                    "--config-file",
                    str(config_file),
                    "--project-env-file",
                    str(project_env),
                    "--json",
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 0, msg=out + err)
            payload = json.loads(out)
            self.assertEqual(payload["provider"], "openai")
            self.assertEqual(payload["model"], "gpt-test")
            self.assertEqual(payload["key_status"], "unchanged")
            self.assertEqual(payload["dry_run"], "true")
            self.assertFalse(config_file.exists())
            self.assertFalse(project_env.exists())

            code, out, err = self._run(
                [
                    "configure-llm",
                    "--provider",
                    "openai",
                    "--no-api-key",
                    "--config-file",
                    str(config_file),
                    "--no-project-env",
                    "--print-source-command",
                ]
            )
            self.assertEqual(code, 0, msg=out + err)
            self.assertIn("source with:", err)

    def test_configure_llm_error_paths_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            cases = [
                (
                    [
                        "configure-llm",
                        "--provider",
                        "bad provider",
                        "--no-api-key",
                        "--config-file",
                        str(config_file),
                        "--no-project-env",
                    ],
                    "",
                    "invalid provider",
                ),
                (
                    [
                        "configure-llm",
                        "--provider",
                        "openrouter",
                        "--api-key-stdin",
                        "--config-file",
                        str(config_file),
                        "--no-project-env",
                    ],
                    "\n",
                    "empty key",
                ),
                (
                    [
                        "configure-llm",
                        "--provider",
                        "openrouter",
                        "--api-key-env",
                        "OPENROUTER_API_KEY",
                        "--config-file",
                        str(config_file),
                        "--no-project-env",
                    ],
                    "",
                    "not set or empty",
                ),
            ]
            for argv, stdin, expected in cases:
                with self.subTest(expected=expected):
                    code, out, err = self._run(argv, stdin=stdin, env={"OPENROUTER_API_KEY": ""})
                    self.assertEqual(code, 1)
                    self.assertEqual(out, "")
                    self.assertIn(expected, err)

    def test_configure_llm_rejects_custom_provider_without_guessing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            code, out, err = self._run(
                [
                    "configure-llm",
                    "--provider",
                    "model_provider:custom_research",
                    "--no-api-key",
                    "--config-file",
                    str(config_file),
                    "--no-project-env",
                ]
            )
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertIn("built-in provider profiles only", err)
            self.assertFalse(config_file.exists())

    def test_interactive_api_key_read_failure_is_reported(self) -> None:
        from praxist.cli._terminal_ui import TerminalInteractionError
        from praxist.cli.configure_llm import ConfigureLLMError, _read_api_key_from_stdin

        class FakeTTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        with (
            patch("sys.stdin", FakeTTY()),
            patch(
                "praxist.cli.configure_llm.read_masked_secret",
                side_effect=TerminalInteractionError("terminal unavailable"),
            ),
            self.assertRaises(ConfigureLLMError),
        ):
            _read_api_key_from_stdin("OPENROUTER_API_KEY")


class InstallSkillsTest(CliRunnerMixin, unittest.TestCase):
    """``praxist install-skills`` installs bundled agent skills safely."""

    @staticmethod
    def _make_skill(root: Path, name: str = "praxist-demo") -> Path:
        source = root / "package" / "skills" / name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            f"---\nname: {name}\n---\n# {name}\n",
            encoding="utf-8",
        )
        return source

    @staticmethod
    def _write_skill_manifest(
        target: Path,
        skills: dict[str, dict[str, str]],
    ) -> Path:
        from praxist.cli.install_skills import OWNERSHIP_MANIFEST

        target.mkdir(parents=True, exist_ok=True)
        manifest = target / OWNERSHIP_MANIFEST
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "managed_by": "praxist",
                    "skills": skills,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def test_codex_copy_installs_all_bundled_skills_with_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            code, out, err = self._run(
                ["install-skills", "--target", "codex", "--target-dir", str(target), "--json"]
            )
            self.assertEqual(code, 0, msg=out + err)
            payload = json.loads(out)
            installed = payload["installed"]
            self.assertGreaterEqual(len(installed), 4)
            for entry in installed:
                path = Path(entry["path"])
                self.assertTrue((path / "SKILL.md").is_file())
                self.assertTrue((path / ".praxist-skill.json").is_file())
                marker = json.loads((path / ".praxist-skill.json").read_text())
                self.assertEqual(marker["skill_name"], path.name)

    def test_codex_install_refuses_unmanaged_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            existing = target / "praxist-onboarding"
            existing.mkdir(parents=True)
            (existing / "SKILL.md").write_text("# user-owned\n", encoding="utf-8")
            code, _out, err = self._run(
                ["install-skills", "--target", "codex", "--target-dir", str(target)]
            )
            self.assertEqual(code, 1)
            self.assertIn("not Praxist-managed", err)

    def test_codex_install_can_force_exact_bundled_unmanaged_path(self) -> None:
        from praxist.cli._setup_common import bundled_skill_dirs

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            source = bundled_skill_dirs()[0]
            existing = target / source.name
            existing.mkdir(parents=True)
            (existing / "SKILL.md").write_text("# stale manual copy\n", encoding="utf-8")
            (existing / "user-owned.txt").write_text("remove me\n", encoding="utf-8")
            unrelated = target / "unrelated-user-skill"
            unrelated.mkdir()
            sentinel = unrelated / "SKILL.md"
            sentinel.write_text("# keep me\n", encoding="utf-8")

            code, out, err = self._run(
                [
                    "install-skills",
                    "--target",
                    "codex",
                    "--target-dir",
                    str(target),
                    "--replace",
                    "--force-unmanaged",
                    "--json",
                ]
            )

            self.assertEqual(code, 0, msg=out + err)
            self.assertFalse((existing / "user-owned.txt").exists())
            self.assertEqual(
                (existing / "SKILL.md").read_bytes(),
                (source / "SKILL.md").read_bytes(),
            )
            marker = json.loads((existing / ".praxist-skill.json").read_text())
            self.assertEqual(marker["managed_by"], "praxist")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "# keep me\n")
            backups = json.loads(out)["backups"]
            self.assertEqual(len(backups), 1)
            backup = Path(backups[0])
            self.assertEqual((backup / "user-owned.txt").read_text(), "remove me\n")

    def test_human_install_output_reports_operator_backup(self) -> None:
        result = {
            "installed": [],
            "target_dir": "/tmp/skills",
            "backups": ["/tmp/.operator-skill.praxist-backup-test"],
        }
        with patch("praxist.cli.install_skills.install_codex_skills", return_value=result):
            code, out, err = self._run(["install-skills", "--target", "codex"])

        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("preserved previous skill at", err)
        self.assertIn(result["backups"][0], err)

    def test_codex_install_treats_digestless_copy_as_operator_owned(self) -> None:
        from praxist.cli._setup_common import bundled_skill_dirs

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve() / "skills"
            source = bundled_skill_dirs()[0]
            existing = target / source.name
            existing.mkdir(parents=True)
            shutil.copy2(source / "SKILL.md", existing / "SKILL.md")
            unverified_marker = existing / ".unverified-skill-owner.json"
            unverified_marker.write_text(
                json.dumps(
                    {
                        "managed_by": "another-tool",
                        "package": "another-package",
                        "source": "package-resource",
                        "version": "0.1.3",
                    }
                ),
                encoding="utf-8",
            )

            code, out, err = self._run(
                [
                    "install-skills",
                    "--target",
                    "codex",
                    "--target-dir",
                    str(target),
                    "--replace",
                    "--json",
                ]
            )

            self.assertEqual(code, 1, msg=out + err)
            self.assertIn("not Praxist-managed", err)
            self.assertFalse((existing / ".praxist-skill.json").exists())
            self.assertTrue(unverified_marker.is_file())

    def test_forced_batch_failure_reports_preserved_operator_backup(self) -> None:
        from praxist.cli import install_skills
        from praxist.cli._setup_common import bundled_skill_dirs

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve() / "skills"
            sources = bundled_skill_dirs()[:2]
            for source in sources:
                existing = target / source.name
                existing.mkdir(parents=True)
                (existing / "SKILL.md").write_text("# operator-owned\n", encoding="utf-8")

            original_replace = install_skills._replace_managed_skill
            calls = 0

            def fail_after_first(**kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise install_skills.InstallSkillsError("later publication failed")
                return original_replace(**kwargs)

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=sources,
                ),
                patch(
                    "praxist.cli.install_skills._replace_managed_skill",
                    side_effect=fail_after_first,
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "operator-owned skills remain recoverable at",
                ) as raised,
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=True,
                    dry_run=False,
                    force_unmanaged=True,
                )

            backups = list(target.glob(f".{sources[0].name}.praxist-backup-*"))
            self.assertEqual(len(backups), 1)
            self.assertIn(str(backups[0]), str(raised.exception))
            self.assertEqual(
                (backups[0] / "SKILL.md").read_text(encoding="utf-8"),
                "# operator-owned\n",
            )

    def test_codex_install_can_skip_only_operator_owned_conflicts(self) -> None:
        from praxist.cli._setup_common import bundled_skill_dirs
        from praxist.cli.install_skills import install_codex_skills

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve() / "skills"
            sources = bundled_skill_dirs()
            conflict = target / sources[0].name
            conflict.mkdir(parents=True)
            sentinel = conflict / "SKILL.md"
            sentinel.write_text("# operator-owned\n", encoding="utf-8")

            result = install_codex_skills(
                target_dir=target,
                mode="copy",
                replace=True,
                dry_run=False,
                skip_unmanaged=True,
            )

            self.assertEqual(result["skipped"], [str(conflict)])
            self.assertEqual(sentinel.read_text(), "# operator-owned\n")
            self.assertTrue((target / sources[1].name / ".praxist-skill.json").is_file())
            manifest = json.loads((target / ".praxist-skills.json").read_text(encoding="utf-8"))
            self.assertNotIn(sources[0].name, manifest["skills"])

    def test_force_unmanaged_replaces_symlink_without_touching_its_target(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = self._make_skill(root)
            target = root / "codex-skills"
            target.mkdir()
            external = root / "external-user-skill"
            external.mkdir()
            sentinel = external / "SKILL.md"
            sentinel.write_text("# external remains\n", encoding="utf-8")
            dest = target / source.name
            dest.symlink_to(external, target_is_directory=True)

            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=True,
                    dry_run=False,
                    force_unmanaged=True,
                )

            self.assertFalse(dest.is_symlink())
            self.assertEqual(
                (dest / "SKILL.md").read_bytes(),
                (source / "SKILL.md").read_bytes(),
            )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "# external remains\n")

    def test_install_and_uninstall_preserve_unowned_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            target.mkdir(parents=True)
            manifest = target / ".praxist-skills.json"
            sentinel = b'{"managed_by":"another-tool","skills":{}}\n'
            manifest.write_bytes(sentinel)

            install_code, _out, install_err = self._run(
                ["install-skills", "--target-dir", str(target)]
            )
            uninstall_code, _out, uninstall_err = self._run(
                ["uninstall-skills", "--target-dir", str(target)]
            )

            self.assertEqual(install_code, 1)
            self.assertEqual(uninstall_code, 1)
            self.assertIn("refusing to replace", install_err)
            self.assertIn("refusing to replace", uninstall_err)
            self.assertEqual(manifest.read_bytes(), sentinel)

    def test_install_and_uninstall_preserve_malformed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            target.mkdir(parents=True)
            manifest = target / ".praxist-skills.json"
            sentinel = b"{not-json\n"
            manifest.write_bytes(sentinel)

            install_code, _out, install_err = self._run(
                ["install-skills", "--target-dir", str(target)]
            )
            uninstall_code, _out, uninstall_err = self._run(
                ["uninstall-skills", "--target-dir", str(target)]
            )

            self.assertEqual(install_code, 1)
            self.assertEqual(uninstall_code, 1)
            self.assertIn("invalid", install_err)
            self.assertIn("invalid", uninstall_err)
            self.assertEqual(manifest.read_bytes(), sentinel)

    def test_claude_target_uses_claude_skills_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "claude-skills"
            with patch.dict(os.environ, {"CLAUDE_SKILLS_DIR": str(target)}, clear=False):
                code, out, err = self._run(["install-skills", "--target", "claude", "--json"])

            self.assertEqual(code, 0, msg=out + err)
            payload = json.loads(out)
            self.assertEqual(payload["target"], "claude")
            self.assertEqual(Path(payload["target_dir"]), target.resolve())
            self.assertTrue((target / "praxist-takeover" / "SKILL.md").is_file())

            with patch.dict(os.environ, {"CLAUDE_SKILLS_DIR": str(target)}, clear=False):
                remove_code, remove_out, remove_err = self._run(
                    ["uninstall-skills", "--target", "claude", "--json"]
                )
            self.assertEqual(remove_code, 0, msg=remove_out + remove_err)
            self.assertEqual(json.loads(remove_out)["target"], "claude")
            self.assertFalse((target / "praxist-takeover").exists())

    def test_programmatic_skill_host_rejects_unknown_target(self) -> None:
        from praxist.cli.install_skills import (
            InstallSkillsError,
            install_skills,
            uninstall_skills,
        )

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            with self.assertRaisesRegex(InstallSkillsError, "unsupported skill host"):
                install_skills(
                    target="unknown",
                    target_dir=target,
                    mode="copy",
                    replace=True,
                    dry_run=True,
                )
            with self.assertRaisesRegex(InstallSkillsError, "unsupported skill host"):
                uninstall_skills(target="unknown", target_dir=target, dry_run=True)

    def test_codex_human_dry_run_and_replace_paths(self) -> None:
        from praxist.cli._setup_common import write_skill_marker
        from praxist.cli.install_skills import install_codex_skills

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            code, out, err = self._run(
                [
                    "install-skills",
                    "--target",
                    "codex",
                    "--target-dir",
                    str(target),
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 0, msg=out + err)
            self.assertEqual(out, "")
            self.assertIn("installed", err)
            self.assertFalse(target.exists())

            source = Path(tmp) / "source-skill"
            source.mkdir()
            (source / "SKILL.md").write_text("# Source\n", encoding="utf-8")
            dest = target / source.name
            target.mkdir()
            dest.mkdir()
            (dest / "SKILL.md").write_text("# Old\n", encoding="utf-8")
            write_skill_marker(
                dest,
                source="test",
                skill_name=source.name,
            )
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                result = install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=True,
                    dry_run=False,
                )
            self.assertEqual(result["installed"][0]["name"], "source-skill")
            self.assertTrue((dest / "SKILL.md").is_file())

    def test_install_skills_low_level_error_paths(self) -> None:
        from praxist.cli._setup_common import write_skill_marker
        from praxist.cli.install_skills import (
            InstallSkillsError,
            _is_replaceable,
            install_codex_skills,
        )

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            with self.assertRaises(InstallSkillsError):
                install_codex_skills(
                    target_dir=target,
                    mode="invalid",
                    replace=False,
                    dry_run=False,
                )
            with (
                patch("praxist.cli.install_skills.bundled_skill_dirs", return_value=[]),
                self.assertRaises(InstallSkillsError),
            ):
                install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=False,
                    dry_run=False,
                )
            marker_parent = Path(tmp) / "existing"
            marker_parent.mkdir()
            self.assertFalse(_is_replaceable(marker_parent))
            (marker_parent / ".praxist-skill.json").write_text("{bad", encoding="utf-8")
            self.assertFalse(_is_replaceable(marker_parent))
            (marker_parent / ".praxist-skill.json").write_text(
                json.dumps({"managed_by": "praxist"}),
                encoding="utf-8",
            )
            self.assertFalse(_is_replaceable(marker_parent))
            write_skill_marker(marker_parent, source="test")
            self.assertTrue(_is_replaceable(marker_parent))

    def test_uninstall_preserves_legacy_copy_without_tree_digest(self) -> None:
        from praxist.cli.install_skills import uninstall_codex_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "package" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            (source / "SKILL.md").write_text("# Package\n", encoding="utf-8")
            target = root / "codex-skills"
            dest = target / source.name
            dest.mkdir(parents=True)
            (dest / "SKILL.md").write_text("# Older copy\n", encoding="utf-8")
            (dest / "user-note.txt").write_text("keep me\n", encoding="utf-8")
            (dest / ".praxist-skill.json").write_text(
                json.dumps(
                    {
                        "managed_by": "praxist",
                        "package": "praxist",
                        "skill_name": source.name,
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                result = uninstall_codex_skills(target_dir=target, dry_run=False)

            self.assertEqual(result["removed"], [])
            self.assertEqual(result["refused"], [str(dest)])
            self.assertEqual((dest / "user-note.txt").read_text(), "keep me\n")

    def test_symlink_install_migrates_moved_official_checkout_with_replace(self) -> None:
        from praxist.cli.install_skills import (
            OWNERSHIP_MANIFEST,
            InstallSkillsError,
            install_codex_skills,
            uninstall_codex_skills,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            old_source = root / "old" / "skills" / "praxist-demo"
            new_source = root / "new" / "skills" / "praxist-demo"
            old_source.mkdir(parents=True)
            new_source.mkdir(parents=True)
            for source in (old_source, new_source):
                (source / "SKILL.md").write_text(
                    "---\nname: praxist-demo\n---\n# Demo\n",
                    encoding="utf-8",
                )
            target = root / "codex-skills"
            target.mkdir()
            dest = target / "praxist-demo"
            dest.symlink_to(old_source, target_is_directory=True)
            shutil.rmtree(root / "old")

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[new_source],
                ),
                self.assertRaises(InstallSkillsError),
            ):
                install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )

            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[new_source],
            ):
                install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                    migrate_legacy_symlinks=True,
                )
            self.assertEqual(dest.resolve(strict=True), new_source.resolve(strict=True))
            manifest = json.loads((target / OWNERSHIP_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["skills"]["praxist-demo"]["managed_by"],
                "praxist",
            )

            shutil.rmtree(root / "new")
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[new_source],
            ):
                result = uninstall_codex_skills(target_dir=target, dry_run=False)
            self.assertEqual(result["removed"], [str(dest)])
            self.assertFalse(dest.exists() or dest.is_symlink())
            self.assertFalse((target / OWNERSHIP_MANIFEST).exists())

    def test_manifest_does_not_authorize_a_user_retargeted_symlink(self) -> None:
        from praxist.cli.install_skills import (
            InstallSkillsError,
            install_codex_skills,
            uninstall_codex_skills,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "package" / "skills" / "praxist-demo"
            user_source = root / "user" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            user_source.mkdir(parents=True)
            (source / "SKILL.md").write_text("# Package\n", encoding="utf-8")
            (user_source / "SKILL.md").write_text("# User\n", encoding="utf-8")
            target = root / "codex-skills"
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )
            dest = target / "praxist-demo"
            dest.unlink()
            dest.symlink_to(user_source, target_is_directory=True)

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[source],
                ),
                self.assertRaises(InstallSkillsError),
            ):
                install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                result = uninstall_codex_skills(target_dir=target, dry_run=False)
            self.assertEqual(result["removed"], [])
            self.assertEqual(result["refused"], [str(dest)])
            self.assertEqual(dest.resolve(strict=True), user_source.resolve(strict=True))

    def test_install_refuses_destination_replaced_after_validation(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "package" / "skills" / "praxist-demo"
            user_source = root / "user" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            user_source.mkdir(parents=True)
            (source / "SKILL.md").write_text("# Package\n", encoding="utf-8")
            (user_source / "SKILL.md").write_text("# User\n", encoding="utf-8")
            target = root / "codex-skills"
            target.mkdir()
            dest = target / source.name
            dest.symlink_to(source, target_is_directory=True)
            original = install_skills._replace_managed_skill

            def replace_then_install(**kwargs):
                dest.unlink()
                dest.symlink_to(user_source, target_is_directory=True)
                return original(**kwargs)

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[source],
                ),
                patch(
                    "praxist.cli.install_skills._replace_managed_skill",
                    side_effect=replace_then_install,
                ),
                self.assertRaises(install_skills.InstallSkillsError),
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )

            self.assertEqual(dest.resolve(strict=True), user_source.resolve(strict=True))

    def test_install_preserves_destination_swapped_during_atomic_move(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "package" / "skills" / "praxist-demo"
            user_source = root / "user" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            user_source.mkdir(parents=True)
            (source / "SKILL.md").write_text("# Package\n", encoding="utf-8")
            (user_source / "SKILL.md").write_text("# User\n", encoding="utf-8")
            target = root / "codex-skills"
            target.mkdir()
            dest = target / source.name
            dest.symlink_to(source, target_is_directory=True)
            original_replace = install_skills.os.replace
            swapped = False

            def swap_before_move(src, dst):
                nonlocal swapped
                if Path(src) == dest and not swapped:
                    swapped = True
                    dest.unlink()
                    dest.symlink_to(user_source, target_is_directory=True)
                return original_replace(src, dst)

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[source],
                ),
                patch(
                    "praxist.cli.install_skills.os.replace",
                    side_effect=swap_before_move,
                ),
                self.assertRaises(install_skills.InstallSkillsError),
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )

            self.assertTrue(swapped)
            self.assertEqual(dest.resolve(strict=True), user_source.resolve(strict=True))

    def test_uninstall_refuses_destination_replaced_after_validation(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "package" / "skills" / "praxist-demo"
            user_source = root / "user" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            user_source.mkdir(parents=True)
            (source / "SKILL.md").write_text("# Package\n", encoding="utf-8")
            (user_source / "SKILL.md").write_text("# User\n", encoding="utf-8")
            target = root / "codex-skills"
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )
            dest = target / source.name
            original = install_skills._remove_managed_skill

            def replace_then_remove(path, **kwargs):
                path.unlink()
                path.symlink_to(user_source, target_is_directory=True)
                return original(path, **kwargs)

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[source],
                ),
                patch(
                    "praxist.cli.install_skills._remove_managed_skill",
                    side_effect=replace_then_remove,
                ),
                self.assertRaises(install_skills.InstallSkillsError),
            ):
                install_skills.uninstall_codex_skills(
                    target_dir=target,
                    dry_run=False,
                )

            self.assertEqual(dest.resolve(strict=True), user_source.resolve(strict=True))

    def test_uninstall_preserves_destination_swapped_during_atomic_move(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "package" / "skills" / "praxist-demo"
            user_source = root / "user" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            user_source.mkdir(parents=True)
            (source / "SKILL.md").write_text("# Package\n", encoding="utf-8")
            (user_source / "SKILL.md").write_text("# User\n", encoding="utf-8")
            target = root / "codex-skills"
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )
            dest = target / source.name
            original_replace = install_skills.os.replace
            swapped = False

            def swap_before_move(src, dst):
                nonlocal swapped
                if Path(src) == dest and not swapped:
                    swapped = True
                    dest.unlink()
                    dest.symlink_to(user_source, target_is_directory=True)
                return original_replace(src, dst)

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[source],
                ),
                patch(
                    "praxist.cli.install_skills.os.replace",
                    side_effect=swap_before_move,
                ),
                self.assertRaises(install_skills.InstallSkillsError),
            ):
                install_skills.uninstall_codex_skills(
                    target_dir=target,
                    dry_run=False,
                )

            self.assertTrue(swapped)
            self.assertEqual(dest.resolve(strict=True), user_source.resolve(strict=True))

    def test_install_preserves_manifest_swapped_during_atomic_move(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "package" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            (source / "SKILL.md").write_text("# Package\n", encoding="utf-8")
            target = root / "codex-skills"
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )
            manifest = target / install_skills.OWNERSHIP_MANIFEST
            original_replace = install_skills.os.replace
            user_content = b"user-owned manifest\n"
            swapped = False

            def swap_before_move(src, dst):
                nonlocal swapped
                if Path(src) == manifest and not swapped:
                    swapped = True
                    manifest.unlink()
                    manifest.write_bytes(user_content)
                return original_replace(src, dst)

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[source],
                ),
                patch(
                    "praxist.cli.install_skills.os.replace",
                    side_effect=swap_before_move,
                ),
                self.assertRaises(install_skills.InstallSkillsError),
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )

            self.assertTrue(swapped)
            self.assertEqual(manifest.read_bytes(), user_content)

    def test_uninstall_preserves_manifest_swapped_during_atomic_move(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "package" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            (source / "SKILL.md").write_text("# Package\n", encoding="utf-8")
            target = root / "codex-skills"
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )
            manifest = target / install_skills.OWNERSHIP_MANIFEST
            original_replace = install_skills.os.replace
            user_content = b"user-owned manifest\n"
            swapped = False

            def swap_before_move(src, dst):
                nonlocal swapped
                if Path(src) == manifest and not swapped:
                    swapped = True
                    manifest.unlink()
                    manifest.write_bytes(user_content)
                return original_replace(src, dst)

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[source],
                ),
                patch(
                    "praxist.cli.install_skills.os.replace",
                    side_effect=swap_before_move,
                ),
                self.assertRaises(install_skills.InstallSkillsError),
            ):
                install_skills.uninstall_codex_skills(
                    target_dir=target,
                    dry_run=False,
                )

            self.assertTrue(swapped)
            self.assertEqual(manifest.read_bytes(), user_content)

    def test_copy_publish_does_not_overwrite_concurrent_user_file(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "stage"
            dest = root / "public-skill"
            source.mkdir()
            (source / "SKILL.md").write_text("# Praxist\n", encoding="utf-8")
            original_copy = install_skills._copy_file_exclusive
            injected = False

            def inject_user_file(src, dst):
                nonlocal injected
                if not injected:
                    injected = True
                    Path(dst).write_text("# User\n", encoding="utf-8")
                return original_copy(src, dst)

            with (
                patch(
                    "praxist.cli.install_skills._copy_file_exclusive",
                    side_effect=inject_user_file,
                ),
                self.assertRaises((OSError, shutil.Error)),
            ):
                install_skills._copy_tree_exclusive(source, dest)

            self.assertEqual((dest / "SKILL.md").read_text(), "# User\n")
            self.assertEqual((source / "SKILL.md").read_text(), "# Praxist\n")

    def test_copy_install_does_not_claim_concurrently_added_user_file(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "package" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            (source / "SKILL.md").write_text("# Package\n", encoding="utf-8")
            target = root / "codex-skills"
            dest = target / source.name
            original_copy = install_skills._copy_file_exclusive
            injected = False

            def copy_then_add_user_file(src, dst):
                nonlocal injected
                result = original_copy(src, dst)
                if Path(dst).name == "SKILL.md" and not injected:
                    injected = True
                    (dest / "user-note.txt").write_text("keep me\n", encoding="utf-8")
                return result

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[source],
                ),
                patch(
                    "praxist.cli.install_skills._copy_file_exclusive",
                    side_effect=copy_then_add_user_file,
                ),
                self.assertRaises(install_skills.InstallSkillsError),
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=True,
                    dry_run=False,
                )

            self.assertEqual((dest / "user-note.txt").read_text(), "keep me\n")
            self.assertFalse(install_skills._is_replaceable(dest, expected_name=dest.name))
            self.assertEqual((source / "SKILL.md").read_text(), "# Package\n")

    def test_copy_restore_preserves_quarantine_when_public_path_collides(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            quarantine = root / ".preserved-skill"
            dest = root / "public-skill"
            quarantine.mkdir()
            (quarantine / "SKILL.md").write_text("# Praxist\n", encoding="utf-8")
            original_copy = install_skills._copy_file_exclusive
            injected = False

            def inject_user_file(src, dst):
                nonlocal injected
                if not injected:
                    injected = True
                    Path(dst).write_text("# User\n", encoding="utf-8")
                return original_copy(src, dst)

            with (
                patch(
                    "praxist.cli.install_skills._copy_file_exclusive",
                    side_effect=inject_user_file,
                ),
                self.assertRaises(install_skills.InstallSkillsError),
            ):
                install_skills._restore_quarantined_path(quarantine, dest)

            self.assertEqual((dest / "SKILL.md").read_text(), "# User\n")
            self.assertEqual((quarantine / "SKILL.md").read_text(), "# Praxist\n")

    def test_install_removes_manifest_owned_skill_absent_from_bundle(self) -> None:
        from praxist.cli.install_skills import (
            OWNERSHIP_MANIFEST,
            install_codex_skills,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            current = root / "package" / "skills" / "praxist-current"
            retired = root / "old" / "skills" / "praxist-retired"
            current.mkdir(parents=True)
            retired.mkdir(parents=True)
            (current / "SKILL.md").write_text("# Current\n", encoding="utf-8")
            (retired / "SKILL.md").write_text("# Retired\n", encoding="utf-8")
            target = root / "codex-skills"
            target.mkdir()
            retired_dest = target / retired.name
            retired_dest.symlink_to(retired, target_is_directory=True)
            (target / OWNERSHIP_MANIFEST).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "managed_by": "praxist",
                        "skills": {
                            retired.name: {
                                "managed_by": "praxist",
                                "mode": "symlink",
                                "source": str(retired),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[current],
            ):
                result = install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )

            self.assertEqual(result["removed_stale"], [str(retired_dest)])
            self.assertFalse(retired_dest.exists() or retired_dest.is_symlink())
            self.assertTrue((target / current.name).is_symlink())
            manifest = json.loads((target / OWNERSHIP_MANIFEST).read_text())
            self.assertEqual(set(manifest["skills"]), {current.name})

    def test_uninstall_removes_manifest_owned_skill_absent_from_bundle(self) -> None:
        from praxist.cli.install_skills import (
            OWNERSHIP_MANIFEST,
            uninstall_codex_skills,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            retired = root / "old" / "skills" / "praxist-retired"
            retired.mkdir(parents=True)
            (retired / "SKILL.md").write_text("# Retired\n", encoding="utf-8")
            target = root / "codex-skills"
            target.mkdir()
            retired_dest = target / retired.name
            retired_dest.symlink_to(retired, target_is_directory=True)
            (target / OWNERSHIP_MANIFEST).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "managed_by": "praxist",
                        "skills": {
                            retired.name: {
                                "managed_by": "praxist",
                                "mode": "symlink",
                                "source": str(retired),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[],
            ):
                result = uninstall_codex_skills(target_dir=target, dry_run=False)

            self.assertEqual(result["removed"], [str(retired_dest)])
            self.assertFalse(retired_dest.exists() or retired_dest.is_symlink())
            self.assertFalse((target / OWNERSHIP_MANIFEST).exists())

    def test_uninstall_cli_reports_missing_json_and_refused_human_paths(self) -> None:
        from praxist.cli._setup_common import bundled_skill_dirs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            missing_target = root / "missing"
            code, out, err = self._run(
                [
                    "uninstall-skills",
                    "--target-dir",
                    str(missing_target),
                    "--dry-run",
                    "--json",
                ]
            )

            self.assertEqual(code, 0, msg=out + err)
            self.assertEqual(err, "")
            payload = json.loads(out)
            self.assertEqual(payload["removed"], [])
            self.assertGreaterEqual(len(payload["missing"]), 1)
            self.assertFalse(missing_target.exists())

            target = root / "skills"
            source = bundled_skill_dirs()[0]
            unmanaged = target / source.name
            unmanaged.mkdir(parents=True)
            sentinel = b"# user-owned\n"
            (unmanaged / "SKILL.md").write_bytes(sentinel)

            code, out, err = self._run(
                ["uninstall-skills", "--target-dir", str(target), "--dry-run"]
            )

            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertIn("removed 0 Praxist skill(s)", err)
            self.assertIn(f"refused unmanaged path: {unmanaged}", err)
            self.assertEqual((unmanaged / "SKILL.md").read_bytes(), sentinel)

    def test_skill_lifecycle_preconditions_fail_closed(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "skills"
            target.mkdir()

            with (
                patch(
                    "praxist.cli.install_skills.os.open",
                    side_effect=PermissionError("lock denied"),
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "could not lock skill target directory",
                ),
                install_skills._target_lock(target),
            ):
                self.fail("lock acquisition unexpectedly succeeded")

            with self.assertRaisesRegex(
                install_skills.InstallSkillsError,
                "--migrate-legacy-symlinks requires --replace",
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=False,
                    dry_run=False,
                    migrate_legacy_symlinks=True,
                )

            with self.assertRaisesRegex(
                install_skills.InstallSkillsError,
                "--force-unmanaged requires --replace",
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=False,
                    dry_run=False,
                    force_unmanaged=True,
                )

            with self.assertRaisesRegex(
                install_skills.InstallSkillsError,
                "cannot both skip and force",
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=True,
                    dry_run=False,
                    force_unmanaged=True,
                    skip_unmanaged=True,
                )

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[],
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "no bundled or manifest-owned Praxist skills found",
                ),
            ):
                install_skills.uninstall_codex_skills(
                    target_dir=target,
                    dry_run=False,
                )
            self.assertEqual(list(target.iterdir()), [])

    def test_install_reconciles_missing_stale_entry_and_refuses_unverified_stale_path(
        self,
    ) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            current = self._make_skill(root / "missing-case", "praxist-current")
            target = root / "missing-case" / "codex-skills"
            retired_source = root / "old" / "skills" / "praxist-retired"
            retired_dest = target / retired_source.name
            manifest = self._write_skill_manifest(
                target,
                {
                    retired_source.name: {
                        "managed_by": "praxist",
                        "mode": "symlink",
                        "source": str(retired_source),
                    }
                },
            )

            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[current],
            ):
                result = install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )

            self.assertEqual(result["removed_stale"], [str(retired_dest)])
            self.assertTrue((target / current.name).is_symlink())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["skills"]), {current.name})

            unsafe_root = root / "unsafe-case"
            unsafe_current = self._make_skill(unsafe_root, "praxist-current")
            unsafe_target = unsafe_root / "codex-skills"
            unsafe_retired_source = root / "retired" / "skills" / "praxist-retired"
            unsafe_dest = unsafe_target / unsafe_retired_source.name
            unsafe_dest.mkdir(parents=True)
            sentinel = b"keep user data\n"
            (unsafe_dest / "user.txt").write_bytes(sentinel)
            unsafe_manifest = self._write_skill_manifest(
                unsafe_target,
                {
                    unsafe_retired_source.name: {
                        "managed_by": "praxist",
                        "mode": "symlink",
                        "source": str(unsafe_retired_source),
                    }
                },
            )
            manifest_before = unsafe_manifest.read_bytes()

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[unsafe_current],
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "removed bundled skill has an unverified destination",
                ),
            ):
                install_skills.install_codex_skills(
                    target_dir=unsafe_target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )

            self.assertEqual((unsafe_dest / "user.txt").read_bytes(), sentinel)
            self.assertEqual(unsafe_manifest.read_bytes(), manifest_before)
            self.assertFalse((unsafe_target / unsafe_current.name).exists())

    def test_current_symlink_is_idempotent_and_uninstall_dry_run_preserves_it(
        self,
    ) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = self._make_skill(root)
            target = root / "codex-skills"
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=True,
                    dry_run=False,
                )
                dest = target / source.name
                identity = install_skills._path_identity(dest)
                result = install_skills.install_codex_skills(
                    target_dir=target,
                    mode="symlink",
                    replace=False,
                    dry_run=False,
                )
                dry_run = install_skills.uninstall_codex_skills(
                    target_dir=target,
                    dry_run=True,
                )

            self.assertEqual(result["installed"][0]["path"], str(dest))
            self.assertEqual(install_skills._path_identity(dest), identity)
            self.assertFalse(install_skills._is_replaceable(dest))
            self.assertEqual(dry_run["removed"], [str(dest)])
            self.assertTrue(dest.is_symlink())
            self.assertTrue((target / install_skills.OWNERSHIP_MANIFEST).is_file())

    def test_copy_digest_controls_idempotence_refresh_and_tamper_refusal(self) -> None:
        from praxist.cli import install_skills
        from praxist.cli._setup_common import skill_tree_digest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = self._make_skill(root)
            target = root / "codex-skills"
            with patch(
                "praxist.cli.install_skills.bundled_skill_dirs",
                return_value=[source],
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=True,
                    dry_run=False,
                )
                dest = target / source.name
                original_identity = install_skills._path_identity(dest)
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=False,
                    dry_run=False,
                )
                self.assertEqual(
                    install_skills._path_identity(dest),
                    original_identity,
                )

                (source / "SKILL.md").write_text(
                    "---\nname: praxist-demo\n---\n# refreshed\n",
                    encoding="utf-8",
                )
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=False,
                    dry_run=False,
                )

            self.assertIn("refreshed", (dest / "SKILL.md").read_text(encoding="utf-8"))
            marker = json.loads((dest / ".praxist-skill.json").read_text(encoding="utf-8"))
            self.assertEqual(
                marker["tree_digest"],
                skill_tree_digest(dest),
            )
            self.assertEqual(skill_tree_digest(dest), skill_tree_digest(source))

            expected_identity = install_skills._path_identity(dest)
            (dest / "SKILL.md").write_text("# user changed\n", encoding="utf-8")
            self.assertEqual(install_skills._path_identity(dest), expected_identity)
            with self.assertRaisesRegex(
                install_skills.InstallSkillsError,
                "no longer Praxist-managed",
            ):
                install_skills._assert_managed_path_unchanged(
                    dest,
                    expected_identity=expected_identity,
                    expected_source=source,
                    ownership=None,
                )
            self.assertEqual(
                (dest / "SKILL.md").read_text(encoding="utf-8"),
                "# user changed\n",
            )

            with patch(
                "praxist.cli.install_skills.skill_tree_digest",
                side_effect=OSError("digest read denied"),
            ):
                self.assertFalse(install_skills._is_replaceable(dest))

            copy_mode_link = root / "copy-mode-link"
            copy_mode_link.symlink_to(source, target_is_directory=True)
            self.assertFalse(
                install_skills._installation_is_current(
                    copy_mode_link,
                    source=source,
                    mode="copy",
                )
            )
            malformed = root / "malformed-copy"
            malformed.mkdir()
            (malformed / ".praxist-skill.json").write_text(
                "{not-json",
                encoding="utf-8",
            )
            self.assertFalse(
                install_skills._installation_is_current(
                    malformed,
                    source=source,
                    mode="copy",
                )
            )

    def test_install_refuses_digestless_legacy_copy_marker(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = self._make_skill(root)
            target = root / "codex-skills"
            dest = target / source.name
            dest.mkdir(parents=True)
            sentinel = b"user-owned addition\n"
            (dest / "SKILL.md").write_text("# old copy\n", encoding="utf-8")
            (dest / "user.txt").write_bytes(sentinel)
            (dest / ".praxist-skill.json").write_text(
                json.dumps(
                    {
                        "managed_by": "praxist",
                        "package": "praxist",
                        "skill_name": source.name,
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "praxist.cli.install_skills.bundled_skill_dirs",
                    return_value=[source],
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "not Praxist-managed",
                ),
            ):
                install_skills.install_codex_skills(
                    target_dir=target,
                    mode="copy",
                    replace=True,
                    dry_run=False,
                )

            self.assertEqual((dest / "user.txt").read_bytes(), sentinel)
            self.assertNotIn(
                "name: praxist-demo",
                (dest / "SKILL.md").read_text(encoding="utf-8"),
            )

    def test_manifest_validation_rejects_symlink_traversal_and_inspection_errors(
        self,
    ) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / "symlink-target"
            target.mkdir()
            external = root / "user-manifest.json"
            sentinel = b'{"managed_by":"user"}\n'
            external.write_bytes(sentinel)
            manifest = target / install_skills.OWNERSHIP_MANIFEST
            manifest.symlink_to(external)

            with self.assertRaisesRegex(
                install_skills.InstallSkillsError,
                "not a regular Praxist-managed file",
            ):
                install_skills._read_ownership_manifest(target)
            self.assertTrue(manifest.is_symlink())
            self.assertEqual(external.read_bytes(), sentinel)

            invalid_target = root / "invalid-entry"
            invalid_manifest = self._write_skill_manifest(
                invalid_target,
                {
                    "../escape": {
                        "managed_by": "praxist",
                        "mode": "copy",
                        "source": "/package/skills/praxist-demo",
                    }
                },
            )
            invalid_before = invalid_manifest.read_bytes()
            with self.assertRaisesRegex(
                install_skills.InstallSkillsError,
                "invalid or unowned entry",
            ):
                install_skills._read_ownership_manifest(invalid_target)
            self.assertEqual(invalid_manifest.read_bytes(), invalid_before)
            self.assertFalse((root / "escape").exists())

            inspected = root / "cannot-inspect"
            with (
                patch.object(
                    Path,
                    "lstat",
                    side_effect=PermissionError("inspection denied"),
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "could not inspect skill path",
                ),
            ):
                install_skills._path_identity(inspected)

    def test_legacy_symlink_and_manifest_ownership_checks_are_narrow(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            regular = root / "regular"
            regular.mkdir()
            self.assertFalse(
                install_skills._looks_like_legacy_praxist_symlink(
                    regular,
                    skill_name="praxist-demo",
                )
            )

            wrong_source = root / "checkout" / "not-skills" / "praxist-demo"
            wrong_source.mkdir(parents=True)
            (wrong_source / "SKILL.md").write_text(
                "---\nname: praxist-demo\n---\n",
                encoding="utf-8",
            )
            wrong_link = root / "wrong-link"
            wrong_link.symlink_to(wrong_source, target_is_directory=True)
            self.assertFalse(
                install_skills._looks_like_legacy_praxist_symlink(
                    wrong_link,
                    skill_name="praxist-demo",
                )
            )

            source = root / "checkout" / "skills" / "praxist-demo"
            source.mkdir(parents=True)
            (source / "SKILL.md").write_text(
                "---\nname: praxist-demo\n---\n# Demo\n",
                encoding="utf-8",
            )
            legacy_link = root / "legacy-link"
            legacy_link.symlink_to(source, target_is_directory=True)
            self.assertTrue(
                install_skills._looks_like_legacy_praxist_symlink(
                    legacy_link,
                    skill_name="praxist-demo",
                )
            )

            self.assertFalse(
                install_skills._symlink_matches_ownership(
                    legacy_link,
                    {"managed_by": "another-tool", "source": str(source)},
                )
            )
            self.assertFalse(
                install_skills._symlink_matches_ownership(
                    legacy_link,
                    {"managed_by": "praxist", "source": " "},
                )
            )
            self.assertFalse(
                install_skills._symlink_matches_ownership(
                    regular,
                    {"managed_by": "praxist", "source": str(source)},
                )
            )

            relative_parent = root / "codex-skills"
            relative_parent.mkdir()
            relative_link = relative_parent / source.name
            relative_link.symlink_to(
                Path("..") / "checkout" / "skills" / source.name,
                target_is_directory=True,
            )
            self.assertTrue(
                install_skills._symlink_matches_ownership(
                    relative_link,
                    {"managed_by": "praxist", "source": str(source)},
                )
            )

    def test_manifest_publication_failure_restores_or_preserves_backup(self) -> None:
        from praxist.cli import install_skills

        initial = {
            "praxist-old": {
                "managed_by": "praxist",
                "mode": "symlink",
                "source": "/package/skills/praxist-old",
            }
        }
        updated = {
            "praxist-new": {
                "managed_by": "praxist",
                "mode": "copy",
                "source": "/package/skills/praxist-new",
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / "restored"
            target.mkdir()
            install_skills._write_ownership_manifest(
                target,
                initial,
                expected_identity=None,
            )
            manifest = target / install_skills.OWNERSHIP_MANIFEST
            original = manifest.read_bytes()
            identity = install_skills._path_identity(manifest)

            with (
                patch(
                    "praxist.cli.install_skills.os.link",
                    side_effect=OSError("publish denied"),
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "could not update skill ownership manifest",
                ),
            ):
                install_skills._write_ownership_manifest(
                    target,
                    updated,
                    expected_identity=identity,
                )

            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(
                [path.name for path in target.iterdir()],
                [install_skills.OWNERSHIP_MANIFEST],
            )

            failed_target = root / "restore-failed"
            failed_target.mkdir()
            install_skills._write_ownership_manifest(
                failed_target,
                initial,
                expected_identity=None,
            )
            failed_manifest = failed_target / install_skills.OWNERSHIP_MANIFEST
            failed_original = failed_manifest.read_bytes()
            failed_identity = install_skills._path_identity(failed_manifest)
            with (
                patch(
                    "praxist.cli.install_skills.os.link",
                    side_effect=OSError("publish denied"),
                ),
                patch(
                    "praxist.cli.install_skills._restore_quarantined_path",
                    side_effect=install_skills.InstallSkillsError("restore denied"),
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "publish denied; restore denied",
                ),
            ):
                install_skills._write_ownership_manifest(
                    failed_target,
                    updated,
                    expected_identity=failed_identity,
                )

            self.assertFalse(failed_manifest.exists())
            backups = [path for path in failed_target.iterdir() if "praxist-backup-" in path.name]
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), failed_original)
            self.assertFalse(any(path.name.endswith(".tmp") for path in failed_target.iterdir()))

    def test_manifest_quarantine_detects_race_and_preserves_failed_restore(
        self,
    ) -> None:
        from praxist.cli import install_skills

        skills = {
            "praxist-demo": {
                "managed_by": "praxist",
                "mode": "symlink",
                "source": "/package/skills/praxist-demo",
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            raced_target = root / "raced"
            raced_target.mkdir()
            install_skills._write_ownership_manifest(
                raced_target,
                skills,
                expected_identity=None,
            )
            raced_manifest = raced_target / install_skills.OWNERSHIP_MANIFEST
            old_identity = install_skills._path_identity(raced_manifest)
            replacement = raced_target / "replacement"
            sentinel = b"user-owned concurrent manifest\n"
            replacement.write_bytes(sentinel)
            os.replace(replacement, raced_manifest)

            with self.assertRaisesRegex(
                install_skills.InstallSkillsError,
                "changed during lifecycle operation",
            ):
                install_skills._quarantine_ownership_manifest(
                    raced_manifest,
                    expected_identity=old_identity,
                )
            self.assertEqual(raced_manifest.read_bytes(), sentinel)

            blocked_target = root / "move-blocked"
            blocked_target.mkdir()
            install_skills._write_ownership_manifest(
                blocked_target,
                skills,
                expected_identity=None,
            )
            blocked_manifest = blocked_target / install_skills.OWNERSHIP_MANIFEST
            blocked_original = blocked_manifest.read_bytes()
            blocked_identity = install_skills._path_identity(blocked_manifest)
            with (
                patch(
                    "praxist.cli.install_skills.os.replace",
                    side_effect=PermissionError("quarantine move denied"),
                ),
                self.assertRaisesRegex(
                    PermissionError,
                    "quarantine move denied",
                ),
            ):
                install_skills._quarantine_ownership_manifest(
                    blocked_manifest,
                    expected_identity=blocked_identity,
                )
            self.assertEqual(blocked_manifest.read_bytes(), blocked_original)
            self.assertFalse(
                any("praxist-backup-" in path.name for path in blocked_target.iterdir())
            )

            failed_target = root / "restore-failed"
            failed_target.mkdir()
            install_skills._write_ownership_manifest(
                failed_target,
                skills,
                expected_identity=None,
            )
            failed_manifest = failed_target / install_skills.OWNERSHIP_MANIFEST
            original = failed_manifest.read_bytes()
            identity = install_skills._path_identity(failed_manifest)
            with (
                patch(
                    "praxist.cli.install_skills._read_ownership_manifest_file",
                    side_effect=install_skills.InstallSkillsError("isolated validation failed"),
                ),
                patch(
                    "praxist.cli.install_skills._restore_quarantined_path",
                    side_effect=install_skills.InstallSkillsError("restore denied"),
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "isolated validation failed; restore denied",
                ),
            ):
                install_skills._quarantine_ownership_manifest(
                    failed_manifest,
                    expected_identity=identity,
                )

            self.assertFalse(failed_manifest.exists())
            backups = [path for path in failed_target.iterdir() if "praxist-backup-" in path.name]
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_skill_publication_and_removal_failures_preserve_recoverable_content(
        self,
    ) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = self._make_skill(root)
            target = root / "codex-skills"
            target.mkdir()
            dest = target / source.name
            dest.symlink_to(source, target_is_directory=True)
            identity = install_skills._path_identity(dest)

            with (
                patch(
                    "praxist.cli.install_skills._publish_staged_skill",
                    side_effect=OSError("publish denied"),
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "could not install praxist-demo",
                ),
            ):
                install_skills._replace_managed_skill(
                    source=source,
                    dest=dest,
                    mode="symlink",
                    expected_identity=identity,
                    ownership=None,
                    allow_legacy_symlink=False,
                )

            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.resolve(strict=True), source.resolve(strict=True))
            self.assertFalse(
                any(
                    "praxist-stage-" in path.name or "praxist-backup-" in path.name
                    for path in target.iterdir()
                )
            )

            removable = target / "praxist-removable"
            removable.symlink_to(source, target_is_directory=True)
            removable_identity = install_skills._path_identity(removable)
            with (
                patch(
                    "praxist.cli.install_skills.os.replace",
                    side_effect=PermissionError("remove denied"),
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "could not remove managed skill",
                ),
            ):
                install_skills._remove_managed_skill(
                    removable,
                    expected_identity=removable_identity,
                    expected_source=source,
                    ownership=None,
                )
            self.assertTrue(removable.is_symlink())
            self.assertEqual(
                removable.resolve(strict=True),
                source.resolve(strict=True),
            )

            missing = target / "missing"
            self.assertIsNone(
                install_skills._quarantine_managed_path(
                    missing,
                    expected_identity=None,
                    expected_source=source,
                    ownership=None,
                )
            )

            fragile = target / "praxist-fragile"
            fragile.symlink_to(source, target_is_directory=True)
            fragile_identity = install_skills._path_identity(fragile)
            with (
                patch(
                    "praxist.cli.install_skills._assert_managed_path_unchanged",
                    side_effect=[
                        None,
                        install_skills.InstallSkillsError("isolated validation failed"),
                    ],
                ),
                patch(
                    "praxist.cli.install_skills._restore_quarantined_path",
                    side_effect=install_skills.InstallSkillsError("restore denied"),
                ),
                self.assertRaisesRegex(
                    install_skills.InstallSkillsError,
                    "isolated validation failed; restore denied",
                ),
            ):
                install_skills._quarantine_managed_path(
                    fragile,
                    expected_identity=fragile_identity,
                    expected_source=source,
                    ownership=None,
                )

            self.assertFalse(fragile.exists() or fragile.is_symlink())
            isolated = [
                path
                for path in target.iterdir()
                if path.name.startswith(".praxist-fragile.praxist-quarantine-")
            ]
            self.assertEqual(len(isolated), 1)
            self.assertTrue(isolated[0].is_symlink())
            self.assertEqual(
                isolated[0].resolve(strict=True),
                source.resolve(strict=True),
            )

    def test_staging_helpers_reject_unsupported_paths_without_data_loss(self) -> None:
        from praxist.cli import install_skills

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            invalid_stage = root / "stage-file"
            invalid_stage.write_text("not a staged tree\n", encoding="utf-8")
            invalid_dest = root / "invalid-dest"
            with self.assertRaisesRegex(
                install_skills.InstallSkillsError,
                "invalid staged skill path",
            ):
                install_skills._publish_staged_skill(invalid_stage, invalid_dest)
            self.assertEqual(
                invalid_stage.read_text(encoding="utf-8"),
                "not a staged tree\n",
            )
            self.assertFalse(invalid_dest.exists())

            quarantine = root / "quarantined-special-file"
            os.mkfifo(quarantine)
            restored = root / "restored-special-file"
            with self.assertRaisesRegex(
                install_skills.InstallSkillsError,
                "unsupported quarantined skill path",
            ):
                install_skills._restore_quarantined_path(quarantine, restored)
            self.assertTrue(quarantine.exists())
            self.assertFalse(restored.exists())

            source = root / "stage-tree"
            source.mkdir()
            (source / "SKILL.md").write_text("# complete\n", encoding="utf-8")
            published = root / "published-tree"
            install_skills._copy_tree_exclusive(source, published)
            self.assertEqual(
                (published / "SKILL.md").read_text(encoding="utf-8"),
                "# complete\n",
            )
            self.assertFalse((published / ".praxist-skill.json").exists())

            absent = root / "already-absent"
            install_skills._remove_path(absent)
            self.assertFalse(absent.exists() or absent.is_symlink())


class DoctorAndSetupTest(CliRunnerMixin, unittest.TestCase):
    """Readiness reporting and setup orchestration stay offline."""

    def setUp(self) -> None:
        def example_result(name: str, **_: object) -> ExampleInstallResult:
            return ExampleInstallResult(
                name=name,
                destination=f"/tmp/PraxistExamples/{name}",
                status="installed",
                dry_run=False,
            )

        self.example_install = patch(
            "praxist.cli.setup.materialize_example",
            side_effect=example_result,
        )
        self.example_install.start()
        self.addCleanup(self.example_install.stop)

    def test_doctor_json_reports_version_and_skill_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch(
                    "praxist.cli._setup_common.importlib.util.find_spec",
                    return_value=object(),
                ),
                patch(
                    "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                    return_value="/usr/bin/codex-relay",
                ),
            ):
                code, out, err = self._run(
                    [
                        "doctor",
                        "--json",
                        "--config-file",
                        str(Path(tmp) / "empty-env"),
                    ],
                    env={
                        "CODEX_SKILLS_DIR": str(Path(tmp) / "skills"),
                        "PRAXIST_AGENT_SYSTEM": "codex_sdk",
                        "PRAXIST_LLM_PROVIDER": "openrouter",
                        "PRAXIST_MODEL_PROVIDER_REF": "",
                        "MODEL_PROVIDER_REF": "",
                        "OPENROUTER_API_KEY": "test-openrouter-key",
                    },
                )
            self.assertEqual(code, 0, msg=out + err)
            payload = json.loads(out)
            names = {entry["name"] for entry in payload["checks"]}
            self.assertIn("praxist_package", names)
            self.assertIn("python", names)
            self.assertIn("platform", names)
            self.assertIn("codex_skills", names)
            self.assertEqual(payload["model_provider_ref"], "model_provider:openrouter")

    def test_doctor_reports_unsupported_native_platform_before_runtime(self) -> None:
        from praxist.cli._setup_common import platform_check

        with patch("praxist.cli._setup_common.sys.platform", "win32"):
            check = platform_check()

        self.assertEqual(check.status, "missing")
        self.assertIn("Linux, macOS, or WSL", check.detail)

    def test_doctor_checks_codex_relay_only_for_relay_providers(self) -> None:
        from praxist.cli.doctor import _codex_route_checks

        with patch(
            "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
            return_value=None,
        ):
            deepseek = _codex_route_checks("codex_sdk", "deepseek")
            openrouter = _codex_route_checks("codex_sdk", "openrouter")
            openai = _codex_route_checks("codex_sdk", "openai")
            claude = _codex_route_checks("claude_sdk", "deepseek")

        self.assertEqual(deepseek[0].status, "missing")
        self.assertEqual(openrouter[0].status, "missing")
        self.assertEqual(openai[0].status, "ok")
        self.assertIn("not required", openai[0].detail)
        self.assertEqual(claude, [])

    def test_doctor_uses_start_provider_precedence(self) -> None:
        from praxist.cli.doctor import build_report

        with (
            patch.dict(
                os.environ,
                {
                    "PRAXIST_AGENT_SYSTEM": "codex_sdk",
                    "PRAXIST_LLM_PROVIDER": "openai",
                    "MODEL_PROVIDER_REF": "model_provider:deepseek_alias",
                    "DEEPSEEK_API_KEY": "test-deepseek-key",
                    "OPENAI_API_KEY": "test-openai-key",
                },
                clear=True,
            ),
            patch(
                "praxist.cli._setup_common.importlib.util.find_spec",
                return_value=object(),
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                return_value="/usr/bin/codex-relay",
            ),
        ):
            report = build_report(task_path=None, agent_system="codex_sdk")

        self.assertEqual(
            report["model_provider_ref"],
            "model_provider:deepseek_alias",
        )
        checks = {check["name"]: check for check in report["checks"]}
        self.assertIn("deepseek", checks["provider_auth"]["detail"])
        self.assertIn("deepseek", checks["codex_relay"]["detail"])

    def test_doctor_codex_native_requires_saved_login_and_ignores_relay_config(
        self,
    ) -> None:
        from praxist.cli.doctor import build_report

        def unavailable_saved_login() -> None:
            self.assertNotIn("PRAXIST_CODEX_BIN", os.environ)
            raise RuntimeError("saved login unavailable")

        with (
            patch.dict(
                os.environ,
                {
                    "PRAXIST_AGENT_SYSTEM": "claude_sdk",
                    "PRAXIST_LLM_PROVIDER": "deepseek",
                    "MODEL_PROVIDER_REF": "model_provider:deepseek_alias",
                    "DEEPSEEK_API_KEY": "test-deepseek-key",
                    "OPENAI_API_KEY": "test-openai-key",
                    "PRAXIST_CODEX_BIN": "/opt/ignored-codex",
                },
                clear=True,
            ),
            patch(
                "praxist.cli._setup_common.importlib.util.find_spec",
                return_value=object(),
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.verify_chatgpt_login",
                side_effect=unavailable_saved_login,
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                return_value=None,
            ),
        ):
            report = build_report(task_path=None, codex_native=True)

        self.assertEqual(report["agent_system"], "codex_sdk")
        self.assertEqual(
            report["model_provider_ref"],
            "model_provider:openai_compatible",
        )
        self.assertEqual(report["auth_mode"], "codex-native")
        checks = {check["name"]: check for check in report["checks"]}
        self.assertEqual(checks["provider_auth"]["status"], "missing")
        self.assertEqual(checks["codex_model_catalog"]["status"], "warn")
        self.assertIn("ignored", checks["provider_key"]["detail"])
        self.assertEqual(checks["codex_relay"]["status"], "ok")
        self.assertIn("not required", checks["codex_relay"]["detail"])

    def test_doctor_normal_codex_route_does_not_require_subscription_login(self) -> None:
        from praxist.cli.doctor import _provider_auth_check

        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "",
                    "PRAXIST_CODEX_BIN": "/opt/codex-custom",
                },
                clear=False,
            ),
            patch("praxist.plugins.agent_runtimes.codex_sdk._auth.verify_chatgpt_login") as verify,
        ):
            check = _provider_auth_check("codex_sdk", provider="openai")
            observed_override = os.environ["PRAXIST_CODEX_BIN"]

        self.assertEqual(check.status, "warn")
        self.assertIn("outside explicit Codex-native mode", check.detail)
        self.assertEqual(observed_override, "/opt/codex-custom")
        verify.assert_not_called()

    def test_doctor_codex_native_checks_selected_model_through_app_server(self) -> None:
        from praxist.cli.doctor import build_report

        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text(
                "PRAXIST_AGENT_SYSTEM=claude_sdk\n"
                "PRAXIST_LLM_PROVIDER=deepseek\n"
                "PRAXIST_MODEL=deepseek-v4-pro[1m]\n",
                encoding="utf-8",
            )
            with (
                patch("praxist.plugins.agent_runtimes.codex_sdk._auth.verify_chatgpt_login"),
                patch(
                    "praxist.plugins.agent_runtimes.codex_sdk.adapter."
                    "verify_chatgpt_model_available",
                    return_value="gpt-5.6-luna",
                ) as verify_model,
            ):
                report = build_report(
                    task_path=None,
                    codex_native=True,
                    config_file=config_file,
                )

        checks = {check["name"]: check for check in report["checks"]}
        self.assertEqual(checks["codex_model_catalog"]["status"], "ok")
        self.assertEqual(checks["PRAXIST_MODEL"]["detail"], "gpt-5.6-luna")
        self.assertIn("override", report["diagnostic_scope"])
        self.assertEqual(
            report["persistent_configuration"],
            {
                "agent_system": "claude_sdk",
                "provider": "deepseek",
                "model": "deepseek-v4-pro[1m]",
            },
        )
        verify_model.assert_called_once_with("gpt-5.6-luna")

    def test_doctor_codex_native_fails_when_model_catalog_is_unavailable(self) -> None:
        from praxist.cli.doctor import build_report

        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.touch()
            with (
                patch("praxist.plugins.agent_runtimes.codex_sdk._auth.verify_chatgpt_login"),
                patch(
                    "praxist.plugins.agent_runtimes.codex_sdk.adapter."
                    "verify_chatgpt_model_available",
                    side_effect=RuntimeError("catalog unavailable"),
                ),
            ):
                report = build_report(
                    task_path=None,
                    codex_native=True,
                    config_file=config_file,
                )

        checks = {check["name"]: check for check in report["checks"]}
        self.assertFalse(report["ok"])
        self.assertEqual(checks["codex_model_catalog"]["status"], "missing")
        self.assertIn("catalog unavailable", checks["codex_model_catalog"]["detail"])
        self.assertTrue(any("codex extra" in action for action in report["next_actions"]))

    def test_setup_configures_llm_and_installs_codex_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config" / "env"
            skills_dir = Path(tmp) / "skills"
            with (
                patch(
                    "praxist.cli._setup_common.importlib.util.find_spec",
                    return_value=object(),
                ),
                patch(
                    "praxist.plugins.agent_runtimes.codex_sdk._relay._relay_binary",
                    return_value="/usr/bin/codex-relay",
                ),
            ):
                code, out, err = self._run(
                    [
                        "setup",
                        "--provider",
                        "openrouter",
                        "--agent-system",
                        "codex_sdk",
                        "--api-key-stdin",
                        "--config-file",
                        str(config_file),
                        "--install-skills",
                        "codex",
                        "--json",
                    ],
                    stdin="sk-setup-secret\n",
                    env={"CODEX_SKILLS_DIR": str(skills_dir)},
                )
            self.assertEqual(code, 0, msg=out + err)
            self.assertTrue(config_file.is_file())
            self.assertTrue((skills_dir / "praxist-onboarding" / "SKILL.md").is_file())
            self.assertNotIn("sk-setup-secret", out + err)

    def test_setup_installs_claude_code_skills_without_changing_runtime_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config" / "env"
            skills_dir = Path(tmp) / "claude-skills"
            code, out, err = self._run(
                [
                    "setup",
                    "--agent-system",
                    "claude_sdk",
                    "--config-file",
                    str(config_file),
                    "--install-skills",
                    "claude",
                    "--skip-doctor",
                    "--json",
                ],
                env={"CLAUDE_SKILLS_DIR": str(skills_dir)},
            )

            self.assertEqual(code, 0, msg=out + err)
            payload = json.loads(out)
            operation = next(
                item
                for item in payload["operations"]
                if item["operation"] == "install_claude_skills"
            )
            self.assertEqual(operation["operation"], "install_claude_skills")
            self.assertEqual(operation["result"]["target"], "claude")
            self.assertTrue((skills_dir / "praxist-onboarding" / "SKILL.md").is_file())
            self.assertIn("PRAXIST_AGENT_SYSTEM=claude_sdk", config_file.read_text())

    def test_setup_lists_complete_profiles_without_side_effects(self) -> None:
        code, out, err = self._run(["setup", "--list-profiles"])
        self.assertEqual(code, 0, msg=out + err)
        profiles = {item["profile_id"]: item for item in json.loads(out)}
        self.assertEqual(profiles["codex-native"]["agent_system"], "codex_sdk")
        self.assertFalse(profiles["codex-native"]["requires_api_key"])
        self.assertEqual(profiles["deepseek-api"]["agent_system"], "claude_sdk")
        self.assertTrue(profiles["deepseek-api"]["requires_api_key"])
        self.assertEqual(profiles["orcarouter-api"]["agent_system"], "claude_sdk")
        self.assertTrue(profiles["orcarouter-api"]["requires_api_key"])
        self.assertEqual(profiles["codex-native"]["authentication"], "saved_chatgpt_login")

    def test_codex_managed_status_does_not_treat_defaults_as_a_profile_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text(
                "PRAXIST_AGENT_SYSTEM=codex_sdk\n"
                "PRAXIST_AGENT_RUNTIME_REF=agent_runtime:codex_sdk\n"
                "PRAXIST_LLM_PROVIDER=openai\n"
                "PRAXIST_MODEL_PROVIDER_REF=model_provider:openai_compatible\n"
                "PRAXIST_MODEL=gpt-5.6-luna\n",
                encoding="utf-8",
            )
            with (
                patch("praxist.cli.setup.current_acceptance", return_value=object()),
                patch(
                    "praxist.cli.setup.read_product_usage_status",
                    return_value={"collection_available": False, "status": "unset"},
                ),
            ):
                code, out, err = self._run(
                    ["setup", "--codex-managed", "--config-file", str(config_file)]
                )

            self.assertEqual(code, 0, msg=out + err)
            payload = json.loads(out)
            self.assertFalse(payload["setup_decisions_complete"])
            self.assertEqual(payload["next_required_action"], "choose_profile")
            self.assertEqual(payload["profile"]["state"], "configured_but_not_selected")
            self.assertEqual(payload["profile"]["configured_profile_id"], "codex-native")
            self.assertFalse(payload["product_usage"]["decision_required"])
            self.assertIn("no product-usage authorization", payload["product_usage"]["detail"])

    def test_agent_managed_status_is_the_canonical_alias(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("praxist.cli.setup.current_acceptance", return_value=None),
            patch(
                "praxist.cli.setup.read_product_usage_status",
                return_value={"collection_available": False, "status": "unset"},
            ),
        ):
            config_file = Path(tmp) / "env"
            code, out, err = self._run(
                ["setup", "--agent-managed", "--config-file", str(config_file)]
            )

        self.assertEqual(code, 0, msg=out + err)
        payload = json.loads(out)
        self.assertEqual(payload["next_required_action"], "review_user_agreement")
        self.assertEqual(payload["user_agreement"]["license_version"], "1.0")
        self.assertIn("LICENSE.md", payload["user_agreement"]["license_url"])

    def test_codex_managed_status_reports_confirmed_profile_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text(
                "PRAXIST_SETUP_PROFILE=codex-native\n"
                "PRAXIST_AGENT_SYSTEM=codex_sdk\n"
                "PRAXIST_AGENT_RUNTIME_REF=agent_runtime:codex_sdk\n"
                "PRAXIST_LLM_PROVIDER=openai\n"
                "PRAXIST_MODEL_PROVIDER_REF=model_provider:openai_compatible\n"
                "PRAXIST_MODEL=gpt-5.6-luna\n",
                encoding="utf-8",
            )
            with (
                patch("praxist.cli.setup.current_acceptance", return_value=object()),
                patch(
                    "praxist.cli.setup.read_product_usage_status",
                    return_value={"collection_available": False, "status": "unset"},
                ),
            ):
                code, out, err = self._run(
                    ["setup", "--codex-managed", "--config-file", str(config_file)]
                )

            self.assertEqual(code, 0, msg=out + err)
            payload = json.loads(out)
            self.assertTrue(payload["setup_decisions_complete"])
            self.assertEqual(payload["next_required_action"], "run_doctor_then_finish_setup")
            self.assertTrue(payload["profile"]["selected"])
            self.assertEqual(payload["profile"]["authorization"]["mode"], "saved_chatgpt_login")
            self.assertFalse(payload["profile"]["authorization"]["api_key_required"])

    def test_codex_managed_status_prioritizes_agreement_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "missing-env"
            with (
                patch("praxist.cli.setup.current_acceptance", return_value=None),
                patch(
                    "praxist.cli.setup.read_product_usage_status",
                    return_value={"collection_available": True, "status": "unset"},
                ),
            ):
                code, out, err = self._run(
                    ["setup", "--codex-managed", "--config-file", str(config_file)]
                )
            self.assertEqual(code, 0, msg=out + err)
            self.assertEqual(json.loads(out)["next_required_action"], "review_user_agreement")
            self.assertFalse(config_file.exists())

            code, out, err = self._run(["setup", "--codex-managed", "--profile", "codex-native"])
            self.assertEqual(code, 2, msg=out + err)
            self.assertIn("read-only status command", err)

    def test_setup_profile_applies_one_coherent_configuration(self) -> None:
        profile_result = {
            "provider": "openai",
            "agent_system": "codex_sdk",
            "model": "gpt-5.6-luna",
        }
        with (
            patch("praxist.cli.setup.configure_llm", return_value=profile_result) as configure,
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.ensure_chatgpt_login",
                return_value=True,
            ) as ensure_login,
            patch(
                "praxist.cli.setup.build_report",
                return_value={"ok": True, "checks": [], "next_actions": []},
            ) as report,
        ):
            code, out, err = self._run(["setup", "--profile", "codex-native", "--json"])
        self.assertEqual(code, 0, msg=out + err)
        self.assertEqual(configure.call_args.kwargs["provider"], "openai")
        self.assertEqual(configure.call_args.kwargs["agent_system"], "codex_sdk")
        self.assertTrue(configure.call_args.kwargs["no_api_key"])
        self.assertEqual(configure.call_args.kwargs["setup_profile"], "codex-native")
        self.assertTrue(report.call_args.kwargs["codex_native"])
        self.assertEqual(
            report.call_args.kwargs["model_provider_ref"],
            "model_provider:openai_compatible",
        )
        self.assertEqual(report.call_args.kwargs["model"], "gpt-5.6-luna")
        ensure_login.assert_called_once_with(allow_interactive=True)
        payload = json.loads(out)
        auth = next(
            item
            for item in payload["operations"]
            if item["operation"] == "ensure_codex_chatgpt_login"
        )
        self.assertTrue(auth["result"]["login_started"])

    def test_codex_native_setup_stops_before_writes_when_login_is_unavailable(self) -> None:
        with (
            patch("praxist.cli.setup.interactive_terminal_available", return_value=False),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.ensure_chatgpt_login",
                side_effect=RuntimeError("saved ChatGPT login unavailable"),
            ) as ensure_login,
            patch("praxist.cli.setup.configure_llm") as configure,
        ):
            code, out, err = self._run(["setup", "--profile", "codex-native"])

        self.assertEqual(code, 1, msg=out + err)
        self.assertIn("saved ChatGPT login unavailable", err)
        ensure_login.assert_called_once_with(allow_interactive=False)
        configure.assert_not_called()

    def test_api_profile_preserves_an_existing_key_without_reprompting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text("DEEPSEEK_API_KEY=stored-secret\n", encoding="utf-8")
            with (
                patch("praxist.cli.setup.configure_llm", return_value={}) as configure,
                patch(
                    "praxist.cli.setup.build_report",
                    return_value={"ok": True, "checks": [], "next_actions": []},
                ),
            ):
                code, out, err = self._run(
                    [
                        "setup",
                        "--profile",
                        "deepseek-api",
                        "--config-file",
                        str(config_file),
                        "--json",
                    ]
                )
        self.assertEqual(code, 0, msg=out + err)
        self.assertFalse(configure.call_args.kwargs["api_key_stdin"])
        self.assertTrue(configure.call_args.kwargs["no_api_key"])
        self.assertNotIn("stored-secret", out + err)

    def test_api_profile_requires_tty_when_key_is_missing(self) -> None:
        with (
            patch("praxist.cli.setup._provider_key_available", return_value=False),
            patch("praxist.cli.setup.interactive_terminal_available", return_value=False),
            patch("praxist.cli.setup.configure_llm") as configure,
        ):
            code, out, err = self._run(["setup", "--profile", "deepseek-api"])
        self.assertEqual(code, 2, msg=out + err)
        self.assertIn("local terminal for masked key input", err)
        configure.assert_not_called()

    def test_api_profile_dry_run_does_not_require_key_input(self) -> None:
        with (
            patch("praxist.cli.setup._provider_key_available", return_value=False),
            patch("praxist.cli.setup.interactive_terminal_available", return_value=False),
            patch("praxist.cli.setup.configure_llm", return_value={}) as configure,
            patch(
                "praxist.cli.setup.build_report",
                return_value={"ok": True, "checks": [], "next_actions": []},
            ),
        ):
            code, out, err = self._run(
                ["setup", "--profile", "deepseek-api", "--dry-run", "--json"]
            )
        self.assertEqual(code, 0, msg=out + err)
        self.assertTrue(configure.call_args.kwargs["api_key_stdin"])

    def test_setup_propagates_api_key_cancellation(self) -> None:
        from praxist.cli.configure_llm import ConfigureLLMCancelled

        with (
            patch("praxist.cli.setup._provider_key_available", return_value=False),
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch(
                "praxist.cli.setup.configure_llm",
                side_effect=ConfigureLLMCancelled("input was cancelled"),
            ),
        ):
            code, out, err = self._run(["setup", "--profile", "deepseek-api"])
        self.assertEqual(code, 130, msg=out + err)
        self.assertIn("cancelled", err)

    def test_existing_profile_is_the_interactive_default(self) -> None:
        from praxist.cli.setup import _select_setup_profile

        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text(
                "PRAXIST_AGENT_SYSTEM=codex_sdk\n"
                "PRAXIST_LLM_PROVIDER=openai\n"
                "PRAXIST_MODEL=gpt-5.6-luna\n",
                encoding="utf-8",
            )
            with patch("praxist.cli.setup.select_choice", return_value="codex-native") as select:
                profile = _select_setup_profile(config_file)
        self.assertEqual(profile.profile_id, "codex-native")
        self.assertEqual(select.call_args.kwargs["default"], 0)

    def test_new_install_defaults_to_codex_native_profile(self) -> None:
        from praxist.cli.setup import _configured_profile_default

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_configured_profile_default(Path(tmp) / "missing-env"), 0)

    def test_interactive_setup_uses_privacy_and_profile_selectors(self) -> None:
        profile_result = {
            "provider": "deepseek",
            "agent_system": "claude_sdk",
            "model": "deepseek-v4-pro[1m]",
        }
        with (
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch(
                "praxist.cli.setup.prompt_for_acceptance_if_needed",
                return_value=True,
            ),
            patch("praxist.cli.setup.prompt_for_consent_if_unset", return_value=True) as consent,
            patch(
                "praxist.cli.setup._select_setup_profile",
                return_value=SETUP_PROFILES[1],
            ),
            patch("praxist.cli.setup._provider_key_available", return_value=False),
            patch("praxist.cli.setup.configure_llm", return_value=profile_result) as configure,
            patch(
                "praxist.cli.setup.install_codex_skills",
                return_value={"installed": [], "target_dir": "/tmp/skills"},
            ),
            patch(
                "praxist.cli.setup.build_report",
                return_value={"ok": True, "checks": [], "next_actions": []},
            ),
        ):
            code, out, err = self._run(
                ["setup", "--interactive", "--json"],
                env={"DEEPSEEK_API_KEY": ""},
            )
        self.assertEqual(code, 0, msg=out + err)
        json.loads(out)
        self.assertIsNotNone(consent.call_args.kwargs["output_stream"])
        self.assertTrue(configure.call_args.kwargs["api_key_stdin"])
        self.assertEqual(configure.call_args.kwargs["provider"], "deepseek")

    def test_interactive_setup_requires_choice_before_replacing_user_skills(self) -> None:
        from praxist.cli.install_skills import SkillConflictError

        conflict = Path("/tmp/operator-skill")
        replacement = {
            "installed": [{"name": "praxist-onboarding"}],
            "backups": ["/tmp/.operator-skill.praxist-backup-test"],
        }
        with (
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch(
                "praxist.cli.setup.prompt_for_acceptance_if_needed",
                return_value=True,
            ),
            patch("praxist.cli.setup.prompt_for_consent_if_unset", return_value=True),
            patch(
                "praxist.cli.setup._select_setup_profile",
                return_value=SETUP_PROFILES[0],
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.ensure_chatgpt_login",
                return_value=False,
            ),
            patch("praxist.cli.setup.configure_llm", return_value={}),
            patch(
                "praxist.cli.setup.install_codex_skills",
                side_effect=[SkillConflictError([conflict]), replacement],
            ) as install,
            patch("praxist.cli.setup.select_choice", return_value="backup-replace") as choose,
            patch(
                "praxist.cli.setup.build_report",
                return_value={"ok": True, "checks": [], "next_actions": []},
            ),
        ):
            code, out, err = self._run(["setup", "--interactive", "--json"])

        self.assertEqual(code, 0, msg=out + err)
        self.assertIn(str(conflict), err)
        self.assertEqual(choose.call_count, 1)
        self.assertFalse(install.call_args_list[0].kwargs.get("force_unmanaged", False))
        self.assertTrue(install.call_args_list[1].kwargs["force_unmanaged"])
        payload = json.loads(out)
        skill_operation = next(
            item for item in payload["operations"] if item["operation"] == "install_codex_skills"
        )
        self.assertEqual(skill_operation["result"], replacement)

    def test_interactive_setup_can_cancel_at_skill_conflict(self) -> None:
        from praxist.cli.install_skills import SkillConflictError

        conflict = Path("/tmp/operator-skill")
        with (
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch(
                "praxist.cli.setup.prompt_for_acceptance_if_needed",
                return_value=True,
            ),
            patch("praxist.cli.setup.prompt_for_consent_if_unset", return_value=True),
            patch(
                "praxist.cli.setup._select_setup_profile",
                return_value=SETUP_PROFILES[0],
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.ensure_chatgpt_login",
                return_value=False,
            ),
            patch("praxist.cli.setup.configure_llm", return_value={}),
            patch(
                "praxist.cli.setup.install_codex_skills",
                side_effect=SkillConflictError([conflict]),
            ),
            patch("praxist.cli.setup.select_choice", return_value="cancel"),
        ):
            code, out, err = self._run(["setup", "--interactive"])

        self.assertEqual(code, 130, msg=out + err)
        self.assertIn("skill installation cancelled", err)

    def test_interactive_setup_reports_failed_backup_replacement(self) -> None:
        from praxist.cli.install_skills import InstallSkillsError, SkillConflictError

        conflict = Path("/tmp/operator-skill")
        with (
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch(
                "praxist.cli.setup.prompt_for_acceptance_if_needed",
                return_value=True,
            ),
            patch("praxist.cli.setup.prompt_for_consent_if_unset", return_value=True),
            patch(
                "praxist.cli.setup._select_setup_profile",
                return_value=SETUP_PROFILES[0],
            ),
            patch(
                "praxist.plugins.agent_runtimes.codex_sdk._auth.ensure_chatgpt_login",
                return_value=False,
            ),
            patch("praxist.cli.setup.configure_llm", return_value={}),
            patch(
                "praxist.cli.setup.install_codex_skills",
                side_effect=[
                    SkillConflictError([conflict]),
                    InstallSkillsError("backup replacement failed"),
                ],
            ),
            patch("praxist.cli.setup.select_choice", return_value="backup-replace"),
        ):
            code, out, err = self._run(["setup", "--interactive"])

        self.assertEqual(code, 1, msg=out + err)
        self.assertIn("backup replacement failed", err)

    def test_interactive_setup_stops_when_privacy_selection_is_cancelled(self) -> None:
        with (
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch(
                "praxist.cli.setup.prompt_for_acceptance_if_needed",
                return_value=True,
            ),
            patch("praxist.cli.setup.prompt_for_consent_if_unset", return_value=False),
            patch("praxist.cli.setup._select_setup_profile") as profile,
        ):
            code, out, err = self._run(["setup", "--interactive"])
        self.assertEqual(code, 130, msg=out + err)
        self.assertIn("interactive setup cancelled", err)
        profile.assert_not_called()

    def test_interactive_setup_requests_agreement_before_optional_privacy(self) -> None:
        order: list[str] = []

        def agreement(**_kwargs: object) -> bool:
            order.append("agreement")
            return True

        def privacy(**_kwargs: object) -> bool:
            order.append("privacy")
            return False

        with (
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch(
                "praxist.cli.setup.prompt_for_acceptance_if_needed",
                side_effect=agreement,
            ),
            patch(
                "praxist.cli.setup.prompt_for_consent_if_unset",
                side_effect=privacy,
            ),
        ):
            code, out, err = self._run(["setup", "--interactive"])

        self.assertEqual(code, 130, msg=out + err)
        self.assertEqual(order, ["agreement", "privacy"])

    def test_interactive_setup_stops_before_privacy_when_agreement_is_declined(self) -> None:
        with (
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch(
                "praxist.cli.setup.prompt_for_acceptance_if_needed",
                return_value=False,
            ),
            patch("praxist.cli.setup.prompt_for_consent_if_unset") as privacy,
        ):
            code, out, err = self._run(["setup", "--interactive"])

        self.assertEqual(code, 130, msg=out + err)
        self.assertIn("License and User Agreement were not accepted", err)
        privacy.assert_not_called()

    def test_interactive_setup_dry_run_does_not_persist_consent(self) -> None:
        with (
            patch("praxist.cli.setup.interactive_terminal_available", return_value=True),
            patch("praxist.cli.setup.prompt_for_acceptance_if_needed") as agreement,
            patch("praxist.cli.setup.prompt_for_consent_if_unset") as consent,
            patch(
                "praxist.cli.setup._select_setup_profile",
                return_value=SETUP_PROFILES[0],
            ),
            patch("praxist.cli.setup.configure_llm", return_value={}),
            patch("praxist.cli.setup.install_codex_skills", return_value={}),
            patch(
                "praxist.cli.setup.build_report",
                return_value={"ok": True, "checks": [], "next_actions": []},
            ),
        ):
            code, out, err = self._run(["setup", "--interactive", "--dry-run", "--json"])
        self.assertEqual(code, 0, msg=out + err)
        agreement.assert_not_called()
        consent.assert_not_called()

    def test_interactive_setup_requires_tty_and_rejects_explicit_options(self) -> None:
        code, out, err = self._run(["setup", "--interactive"])
        self.assertEqual(code, 2, msg=out + err)
        self.assertIn("requires a local interactive terminal", err)

        with patch("praxist.cli.setup.interactive_terminal_available", return_value=True):
            code, out, err = self._run(["setup", "--interactive", "--profile", "codex-native"])
        self.assertEqual(code, 2, msg=out + err)
        self.assertIn("alternative setup modes", err)

        code, out, err = self._run(
            ["setup", "--profile", "codex-native", "--api-key-env", "OPENAI_API_KEY"]
        )
        self.assertEqual(code, 2, msg=out + err)
        self.assertIn("cannot be combined", err)

    def test_setup_persists_agent_system_and_doctor_reads_same_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "custom" / "env"
            config_file.parent.mkdir()
            config_file.write_text(
                "export PRAXIST_AGENT_RUNTIME_REF=agent_runtime:claude_sdk\n"
                "export RUNTIME_REF=agent_runtime:claude_sdk\n",
                encoding="utf-8",
            )
            report = {"ok": True, "checks": [], "next_actions": []}
            with patch("praxist.cli.setup.build_report", return_value=report) as build:
                code, out, err = self._run(
                    [
                        "setup",
                        "--agent-system",
                        "codex_sdk",
                        "--config-file",
                        str(config_file),
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, msg=out + err)
            config_text = config_file.read_text()
            self.assertIn("PRAXIST_AGENT_SYSTEM=codex_sdk", config_text)
            self.assertIn(
                "PRAXIST_AGENT_RUNTIME_REF=agent_runtime:codex_sdk",
                config_text,
            )
            self.assertEqual(build.call_args.kwargs["config_file"], config_file.resolve())

    def test_doctor_human_output_and_next_actions(self) -> None:
        from praxist.cli._setup_common import Check
        from praxist.cli.doctor import _next_actions, print_report

        stderr = io.StringIO()
        report = {
            "checks": [
                {"name": "PRAXIST_LLM_PROVIDER", "status": "warn", "detail": "not set"},
                {
                    "name": "provider_key",
                    "status": "missing",
                    "detail": "not set",
                    "variable": "OPENROUTER_API_KEY",
                },
            ],
            "next_actions": ["praxist configure-llm --provider openrouter --api-key-stdin"],
        }
        with redirect_stderr(stderr):
            print_report(report)
        text = stderr.getvalue()
        self.assertIn("Praxist doctor", text)
        self.assertIn("Next action", text)
        with patch.dict(os.environ, {"PRAXIST_LLM_PROVIDER": "openrouter"}, clear=False):
            actions = _next_actions(
                [
                    Check("PRAXIST_LLM_PROVIDER", "warn"),
                    Check("provider_auth", "missing", variable="OPENROUTER_API_KEY"),
                    Check("codex_sdk", "missing"),
                    Check("codex_skills", "warn"),
                ]
            )
        self.assertIn("praxist configure-llm --provider openrouter --api-key-stdin", actions)
        self.assertEqual(
            actions.count("praxist configure-llm --provider openrouter --api-key-stdin"),
            1,
        )
        self.assertIn("install the Praxist codex extra, then rerun: praxist doctor", actions)
        self.assertIn("praxist install-skills --target codex --replace", actions)

    def test_doctor_task_and_skill_status_variants(self) -> None:
        from praxist.cli.doctor import _codex_skill_checks, _task_path_check, build_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            missing_task = root / "missing"
            self.assertEqual(_task_path_check(missing_task).status, "missing")
            no_yaml = root / "task"
            no_yaml.mkdir()
            self.assertEqual(_task_path_check(no_yaml).status, "missing")
            (no_yaml / "task.yaml").write_text("name: demo\n", encoding="utf-8")
            self.assertEqual(_task_path_check(no_yaml).status, "ok")

            source = root / "skill-a"
            source.mkdir()
            (source / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
            target = root / "codex-skills"
            with (
                patch("praxist.cli.doctor.bundled_skill_dirs", return_value=[source]),
                patch("praxist.cli.doctor.default_codex_skills_dir", return_value=target),
            ):
                checks = _codex_skill_checks()
                self.assertEqual(checks[0].status, "warn")
                self.assertIn("missing", checks[0].detail)

            dest = target / "skill-a"
            dest.mkdir(parents=True)
            (dest / "SKILL.md").write_text("# Installed\n", encoding="utf-8")
            with (
                patch("praxist.cli.doctor.bundled_skill_dirs", return_value=[source]),
                patch("praxist.cli.doctor.default_codex_skills_dir", return_value=target),
            ):
                checks = _codex_skill_checks()
                self.assertEqual(checks[0].status, "warn")
                self.assertIn("unmarked", checks[0].detail)
            (dest / ".praxist-skill.json").write_text("{}", encoding="utf-8")
            with (
                patch("praxist.cli.doctor.bundled_skill_dirs", return_value=[source]),
                patch("praxist.cli.doctor.default_codex_skills_dir", return_value=target),
            ):
                self.assertEqual(_codex_skill_checks()[0].status, "warn")

            report = build_report(task_path=no_yaml)
            names = {check["name"] for check in report["checks"]}
            self.assertIn("codex_skills", names)
            self.assertIn("task_path", names)

    def test_doctor_auto_detects_a_managed_claude_skill_install(self) -> None:
        from praxist.cli.doctor import _skill_checks

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / "codex"
            claude = root / "claude"
            claude.mkdir()
            (claude / ".praxist-skills.json").write_text(
                '{"schema_version": 1, "managed_by": "praxist", "skills": {}}\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"CODEX_SKILLS_DIR": str(codex), "CLAUDE_SKILLS_DIR": str(claude)},
                clear=False,
            ):
                checks = _skill_checks("auto")

        self.assertEqual([check.name for check in checks], ["claude_skills"])

    def test_setup_error_and_human_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self._run(["setup", "--install-agent", "codex"])
            self.assertEqual(code, 2)
            self.assertIn("unrecognized arguments", err)

            code, out, err = self._run(["setup", "--model", "gpt-test"])
            self.assertEqual(code, 2)
            self.assertIn("require --provider", err)

            code, out, err = self._run(["setup", "--provider", "bad provider"])
            self.assertEqual(code, 1)
            self.assertIn("invalid provider", err)

            config_file = Path(tmp) / "config" / "env"
            with (
                patch(
                    "praxist.cli.setup.build_report",
                    return_value={"ok": True, "checks": [], "next_actions": []},
                ),
                patch(
                    "praxist.cli.setup.install_codex_skills",
                    return_value={"installed": [], "target_dir": str(Path(tmp) / "skills")},
                ),
            ):
                code, out, err = self._run(
                    [
                        "setup",
                        "--provider",
                        "openrouter",
                        "--api-key-env",
                        "OPENROUTER_API_KEY",
                        "--config-file",
                        str(config_file),
                        "--install-skills",
                        "codex",
                    ],
                    env={"OPENROUTER_API_KEY": "test-key"},
                )
            self.assertEqual(code, 0, msg=out + err)
            self.assertIn("configure_llm: ok", err)
            self.assertIn("install_codex_skills: ok", err)

    def test_setup_human_output_reports_doctor_failure_without_success_claim(self) -> None:
        report = {
            "ok": False,
            "checks": [
                {
                    "name": "codex_model_catalog",
                    "status": "missing",
                    "detail": "catalog unavailable",
                }
            ],
            "next_actions": ["reinstall the Praxist codex extra"],
        }
        with patch("praxist.cli.setup.build_report", return_value=report):
            code, out, err = self._run(["setup"])

        self.assertEqual(code, 1, msg=out + err)
        self.assertIn("setup incomplete", err)
        self.assertIn("codex_model_catalog", err)
        self.assertNotIn("setup complete", err)


class SetupCommonHelpersTest(unittest.TestCase):
    """Direct coverage for shared setup helpers."""

    def test_provider_aliases_and_unknown_provider(self) -> None:
        from praxist.cli._setup_common import (
            provider_key_var,
            provider_plugin_ref,
            provider_short_name,
        )

        self.assertEqual(provider_key_var("model_provider:openai_compatible"), "OPENAI_API_KEY")
        self.assertEqual(provider_key_var("anthropic_messages"), "ANTHROPIC_API_KEY")
        self.assertEqual(provider_plugin_ref("dashscope"), "model_provider:dashscope")
        self.assertEqual(provider_short_name("model_provider:deepseek_alias"), "deepseek")
        with self.assertRaises(ValueError):
            provider_key_var("not-a-provider")

    def test_env_file_parsing_loading_and_preservation(self) -> None:
        from praxist.cli._setup_common import (
            load_env_file,
            load_env_files,
            read_env_file,
            selected_config_file,
            write_env_file,
        )

        with tempfile.TemporaryDirectory() as tmp:
            user_env = Path(tmp) / "user.env"
            project_env = Path(tmp) / "project.env"
            user_env.write_text(
                "\n".join(
                    [
                        "# comment",
                        "export PRAXIST_LLM_PROVIDER=openrouter",
                        "export OPENROUTER_API_KEY=   ",
                        "BROKEN='unterminated",
                        "UNRELATED=value",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            lines, values = read_env_file(user_env)
            self.assertEqual(len(lines), 5)
            self.assertEqual(values["PRAXIST_LLM_PROVIDER"], "openrouter")
            self.assertEqual(values["OPENROUTER_API_KEY"], "")
            self.assertEqual(values["BROKEN"], "unterminated")

            project_env.write_text(
                "\n".join(
                    [
                        "export PRAXIST_LLM_PROVIDER=openai",
                        "export PRAXIST_AGENT_SYSTEM=codex_sdk",
                        "export PRAXIST_MODEL_PROVIDER_REF=model_provider:custom",
                        "export MODEL_PROVIDER_REF=model_provider:legacy",
                        "export PRAXIST_AGENT_RUNTIME_REF=agent_runtime:codex_sdk",
                        "export RUNTIME_REF=agent_runtime:claude_sdk",
                        "export MODEL=legacy-model",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"PRAXIST_MODEL": "explicit-model"}, clear=True):
                loaded = load_env_files([user_env, project_env])
                self.assertEqual(os.environ["PRAXIST_LLM_PROVIDER"], "openai")
                self.assertEqual(os.environ["PRAXIST_AGENT_SYSTEM"], "codex_sdk")
                self.assertEqual(os.environ["PRAXIST_MODEL"], "explicit-model")
                self.assertEqual(
                    os.environ["PRAXIST_MODEL_PROVIDER_REF"],
                    "model_provider:custom",
                )
                self.assertEqual(
                    os.environ["MODEL_PROVIDER_REF"],
                    "model_provider:legacy",
                )
                self.assertEqual(
                    os.environ["PRAXIST_AGENT_RUNTIME_REF"],
                    "agent_runtime:codex_sdk",
                )
                self.assertEqual(
                    os.environ["RUNTIME_REF"],
                    "agent_runtime:claude_sdk",
                )
                self.assertEqual(os.environ["MODEL"], "legacy-model")
                self.assertNotIn("UNRELATED", os.environ)
                self.assertEqual(loaded["PRAXIST_LLM_PROVIDER"], "openai")

            with patch.dict(os.environ, {"PRAXIST_LLM_PROVIDER": "explicit"}, clear=True):
                load_env_file(project_env)
                self.assertEqual(os.environ["PRAXIST_LLM_PROVIDER"], "explicit")
                load_env_file(project_env, override=True)
                self.assertEqual(os.environ["PRAXIST_LLM_PROVIDER"], "openai")

            with patch.dict(
                os.environ,
                {"PRAXIST_CONFIG_FILE": str(project_env)},
                clear=False,
            ):
                self.assertEqual(selected_config_file(), project_env.resolve())
                self.assertEqual(selected_config_file(user_env), user_env.resolve())

            write_env_file(
                user_env,
                {"PRAXIST_LLM_PROVIDER": "deepseek", "PRAXIST_MODEL": "deepseek-chat"},
            )
            text = user_env.read_text(encoding="utf-8")
            self.assertIn("UNRELATED=value", text)
            self.assertIn("export PRAXIST_LLM_PROVIDER=deepseek", text)
            self.assertIn("export PRAXIST_MODEL=deepseek-chat", text)

    def test_write_env_file_preserves_symlink_and_updates_its_target(self) -> None:
        from praxist.cli._setup_common import write_env_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / "dotfiles" / "praxist.env"
            target.parent.mkdir()
            target.write_text(
                "UNRELATED=value\nexport PRAXIST_MODEL=old\n",
                encoding="utf-8",
            )
            config_dir = root / "config"
            config_dir.mkdir()
            link = config_dir / "env"
            link.symlink_to(Path("..") / "dotfiles" / "praxist.env")

            write_env_file(link, {"PRAXIST_MODEL": "new"})

            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), "../dotfiles/praxist.env")
            text = target.read_text(encoding="utf-8")
            self.assertIn("UNRELATED=value", text)
            self.assertIn("export PRAXIST_MODEL=new", text)

    def test_skill_and_cli_helpers(self) -> None:
        from praxist.cli._setup_common import (
            Check,
            cli_checks,
            copy_skill_tree,
            default_claude_skills_dir,
            default_codex_skills_dir,
            praxist_console_path,
            version_checks,
            write_skill_marker,
        )

        self.assertEqual(Check("x", "ok", "d", "VAR").to_dict()["variable"], "VAR")
        with patch.dict(os.environ, {"CODEX_SKILLS_DIR": "/tmp/custom-skills"}, clear=False):
            self.assertEqual(default_codex_skills_dir(), Path("/tmp/custom-skills"))
        with patch.dict(os.environ, {"CLAUDE_SKILLS_DIR": "/tmp/claude-skills"}, clear=False):
            self.assertEqual(default_claude_skills_dir(), Path("/tmp/claude-skills"))
        with patch("praxist.cli._setup_common.shutil.which", return_value=None):
            self.assertIn("not on PATH", praxist_console_path())
            self.assertEqual(version_checks()[1].status, "warn")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("praxist.cli._setup_common.importlib.util.find_spec", return_value=object()),
            patch("praxist.cli._setup_common.importlib.metadata.version", return_value="1.2.3"),
        ):
            checks = cli_checks()
            self.assertEqual(
                [check.name for check in checks],
                ["claude_sdk", "codex_sdk"],
            )
            self.assertEqual([check.status for check in checks], ["ok", "ok"])
        with (
            patch(
                "praxist.cli._setup_common.importlib.util.find_spec",
                side_effect=lambda module: None if module == "mcp" else object(),
            ),
            patch("praxist.cli._setup_common.importlib.metadata.version", return_value="1.2.3"),
        ):
            codex = cli_checks("codex_sdk")
            self.assertEqual(len(codex), 1)
            self.assertEqual(codex[0].status, "missing")
            self.assertIn("mcp is not installed", codex[0].detail)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "source"
            source.mkdir()
            (source / "SKILL.md").write_text("# Skill\n", encoding="utf-8")
            dest = root / "dest"
            dest.mkdir()
            (dest / "old").write_text("old", encoding="utf-8")
            copy_skill_tree(source, dest)
            self.assertTrue((dest / "SKILL.md").is_file())
            write_skill_marker(dest, source="test")
            marker = json.loads((dest / ".praxist-skill.json").read_text())
            self.assertEqual(marker["managed_by"], "praxist")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
