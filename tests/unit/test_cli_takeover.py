"""Tests for the thin agent first-project takeover handoff."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from praxist.cli._terminal_ui import TerminalInteractionCancelled


class TakeoverCliTest(unittest.TestCase):
    def _run(
        self,
        argv: list[str],
        *,
        agreement_accepted: bool = True,
    ) -> tuple[int, str, str]:
        from praxist.cli import main

        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with (
                patch(
                    "praxist.cli.takeover.current_acceptance",
                    return_value=object() if agreement_accepted else None,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                main(argv)
        except SystemExit as exc:
            code = int(exc.code or 0)
        else:
            code = 0
        return code, stdout.getvalue(), stderr.getvalue()

    def test_live_takeover_requires_agreement_before_noninteractive_launch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            code, out, err = self._run(
                ["takeover", "--task-path", raw, "--configured-provider", "--yes"],
                agreement_accepted=False,
            )
        self.assertEqual(code, 1, msg=out + err)
        self.assertIn("License and User Agreement have not been accepted", err)

    def test_alias_uses_codex_native_skill_from_saved_login_profile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "env"
            config.write_text(
                "PRAXIST_AGENT_SYSTEM=codex_sdk\nPRAXIST_LLM_PROVIDER=openai\n",
                encoding="utf-8",
            )
            with (
                patch.dict(os.environ, {"PRAXIST_CONFIG_FILE": str(config)}, clear=True),
                patch(
                    "praxist.cli.takeover._bundled_codex_binary",
                    return_value="/sdk/bin/codex",
                ),
            ):
                code, out, err = self._run(
                    ["--takeover", "--task-path", str(root), "--dry-run", "--json"]
                )
        self.assertEqual(code, 0, msg=out + err)
        payload = json.loads(out)
        self.assertEqual(payload["skill"], "praxist-takeover-codex")
        self.assertEqual(payload["command"][0], "/sdk/bin/codex")
        self.assertFalse(payload["launched"])
        self.assertNotIn("API_KEY", json.dumps(payload))

    def test_legacy_codex_runtime_alias_uses_codex_native_skill(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "env"
            config.write_text(
                "PRAXIST_AGENT_SYSTEM=codex\nPRAXIST_LLM_PROVIDER=openai\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"PRAXIST_CONFIG_FILE": str(config)}, clear=True):
                code, out, err = self._run(
                    ["takeover", "--task-path", str(root), "--dry-run", "--json"]
                )
        self.assertEqual(code, 0, msg=out + err)
        self.assertEqual(json.loads(out)["skill"], "praxist-takeover-codex")

    def test_canonical_refs_select_codex_native_skill(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "env"
            config.write_text(
                "PRAXIST_AGENT_RUNTIME_REF=agent_runtime:codex_sdk\n"
                "PRAXIST_MODEL_PROVIDER_REF=model_provider:openai_compatible\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"PRAXIST_CONFIG_FILE": str(config)}, clear=True):
                code, out, err = self._run(
                    ["takeover", "--task-path", str(root), "--dry-run", "--json"]
                )
        self.assertEqual(code, 0, msg=out + err)
        self.assertEqual(json.loads(out)["skill"], "praxist-takeover-codex")

    def test_task_environment_can_select_codex_native_skill(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "user-env"
            config.write_text(
                "PRAXIST_AGENT_SYSTEM=claude_sdk\nPRAXIST_LLM_PROVIDER=deepseek\n",
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "PRAXIST_AGENT_RUNTIME_REF=agent_runtime:codex_sdk\n"
                "PRAXIST_MODEL_PROVIDER_REF=model_provider:openai_compatible\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"PRAXIST_CONFIG_FILE": str(config)}, clear=True):
                code, out, err = self._run(
                    ["takeover", "--task-path", str(root), "--dry-run", "--json"]
                )
        self.assertEqual(code, 0, msg=out + err)
        self.assertEqual(json.loads(out)["skill"], "praxist-takeover-codex")

    def test_explicit_configured_provider_launches_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch("praxist.cli.takeover.shutil.which", return_value="/usr/bin/codex"),
                patch("praxist.cli.takeover.subprocess.run", return_value=completed) as run,
            ):
                code, out, err = self._run(
                    [
                        "takeover",
                        "--task-path",
                        str(root),
                        "--configured-provider",
                        "--yes",
                    ]
                )
        self.assertEqual(code, 0, msg=out + err)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["/usr/bin/codex", "--yolo"])
        self.assertIn("$praxist-takeover", command[-1])
        self.assertFalse(run.call_args.kwargs.get("shell", False))
        self.assertEqual(run.call_args.kwargs["cwd"], root)

    def test_explicit_claude_operator_uses_claude_skill_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch("praxist.cli.takeover.shutil.which", return_value="/usr/bin/claude"),
                patch("praxist.cli.takeover.subprocess.run", return_value=completed) as run,
            ):
                code, out, err = self._run(
                    [
                        "takeover",
                        "--task-path",
                        str(root),
                        "--operator",
                        "claude",
                        "--configured-provider",
                        "--yes",
                    ]
                )

        self.assertEqual(code, 0, msg=out + err)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["/usr/bin/claude", "--dangerously-skip-permissions"])
        self.assertIn("/praxist-takeover", command[-1])
        self.assertEqual(run.call_args.kwargs["cwd"], root)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_missing_claude_operator_is_reported_without_launch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with (
                patch("praxist.cli.takeover.shutil.which", return_value=None),
                patch("praxist.cli.takeover.subprocess.run") as run,
            ):
                code, out, err = self._run(
                    [
                        "takeover",
                        "--task-path",
                        str(root),
                        "--operator",
                        "claude",
                        "--configured-provider",
                        "--yes",
                    ]
                )

        self.assertEqual(code, 1, msg=out + err)
        self.assertIn("Claude Code is not on PATH", err)
        run.assert_not_called()

    def test_missing_project_is_reported_without_launch(self) -> None:
        code, out, err = self._run(
            ["takeover", "--task-path", "/definitely/missing/project", "--yes"]
        )
        self.assertEqual(code, 1)
        self.assertIn("does not exist", err)

    def test_process_environment_overrides_saved_profile_for_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "env"
            config.write_text(
                "PRAXIST_AGENT_SYSTEM=codex_sdk\nPRAXIST_LLM_PROVIDER=openai\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "PRAXIST_CONFIG_FILE": str(config),
                    "PRAXIST_AGENT_SYSTEM": "claude_sdk",
                    "PRAXIST_LLM_PROVIDER": "deepseek",
                },
                clear=True,
            ):
                code, out, err = self._run(
                    ["takeover", "--task-path", str(root), "--dry-run", "--json"]
                )
        self.assertEqual(code, 0, msg=out + err)
        self.assertEqual(json.loads(out)["skill"], "praxist-takeover")

    def test_json_is_dry_run_only(self) -> None:
        code, out, err = self._run(["takeover", "--json"])
        self.assertEqual(code, 1, msg=out + err)
        self.assertIn("--json requires --dry-run", err)

    def test_text_dry_run_reports_project_and_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            code, out, err = self._run(
                ["takeover", "--task-path", raw, "--configured-provider", "--dry-run"]
            )
        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("Praxist first launch", err)
        self.assertIn("$praxist-takeover", err)

    def test_live_takeover_requires_confirmation_or_terminal(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch("praxist.cli.takeover.interactive_terminal_available", return_value=False),
            patch("praxist.cli.takeover.shutil.which", return_value="/usr/bin/codex"),
        ):
            code, out, err = self._run(["takeover", "--task-path", raw, "--configured-provider"])
        self.assertEqual(code, 1, msg=out + err)
        self.assertIn("local terminal or --yes", err)

    def test_explicit_project_escape_cancels_with_status_130(self) -> None:
        with (
            tempfile.TemporaryDirectory() as raw,
            patch("praxist.cli.takeover._codex_command", return_value=["codex"]),
            patch("praxist.cli.takeover.interactive_terminal_available", return_value=True),
            patch(
                "praxist.cli.takeover.confirm_action",
                side_effect=TerminalInteractionCancelled("back"),
            ),
        ):
            code, out, err = self._run(["takeover", "--task-path", raw])
        self.assertEqual(code, 130, msg=out + err)
        self.assertIn("cancelled", err)

    def test_interactive_selection_has_no_implicit_project_and_accepts_other_path(self) -> None:
        from praxist.cli.takeover import _select_task_path

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with (
                patch("praxist.cli.takeover.interactive_terminal_available", return_value=True),
                patch("praxist.cli.takeover.select_choice", return_value="other") as select,
                patch("praxist.cli.takeover.read_visible_text", return_value=str(root)),
            ):
                selected, prepare = _select_task_path(None)
        self.assertEqual(selected, root.resolve())
        self.assertFalse(prepare)
        self.assertIsNone(select.call_args.kwargs["default"])

    def test_escape_from_path_input_returns_to_project_choices(self) -> None:
        from praxist.cli.takeover import _select_task_path

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with (
                patch("praxist.cli.takeover.interactive_terminal_available", return_value=True),
                patch("praxist.cli.takeover.select_choice", side_effect=["other", "other"]),
                patch(
                    "praxist.cli.takeover.read_visible_text",
                    side_effect=[TerminalInteractionCancelled("back"), str(root)],
                ),
            ):
                selected, prepare = _select_task_path(None)
        self.assertEqual(selected, root.resolve())
        self.assertFalse(prepare)

    def test_invalid_interactive_path_returns_to_project_choices(self) -> None:
        from praxist.cli.takeover import _select_task_path

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with (
                patch("praxist.cli.takeover.interactive_terminal_available", return_value=True),
                patch("praxist.cli.takeover.select_choice", side_effect=["other", "other"]),
                patch(
                    "praxist.cli.takeover.read_visible_text",
                    side_effect=[str(root / "missing"), str(root)],
                ),
                redirect_stderr(io.StringIO()) as error,
            ):
                selected, prepare = _select_task_path(None)
        self.assertEqual(selected, root.resolve())
        self.assertFalse(prepare)
        self.assertIn("Project not selected", error.getvalue())

    def test_interactive_current_finish_and_template_choices(self) -> None:
        from praxist.cli.takeover import _select_task_path

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with (
                patch("praxist.cli.takeover.interactive_terminal_available", return_value=True),
                patch("praxist.cli.takeover.Path.cwd", return_value=root),
                patch("praxist.cli.takeover.select_choice", return_value="current"),
            ):
                selected, prepare = _select_task_path(None)
            self.assertEqual((selected, prepare), (root.resolve(), False))

            with (
                patch("praxist.cli.takeover.interactive_terminal_available", return_value=True),
                patch("praxist.cli.takeover.select_choice", return_value="finish"),
            ):
                self.assertEqual(_select_task_path(None), (None, False))

            with (
                patch("praxist.cli.takeover.interactive_terminal_available", return_value=True),
                patch("praxist.cli.takeover.Path.cwd", return_value=root),
                patch("praxist.cli.takeover.select_choice", return_value="template"),
            ):
                selected, prepare = _select_task_path(None)
            self.assertEqual(selected, (root / "praxist-toy-math-demo").resolve())
            self.assertTrue(prepare)

    def test_noninteractive_selection_uses_current_directory(self) -> None:
        from praxist.cli.takeover import _select_task_path

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with (
                patch("praxist.cli.takeover.interactive_terminal_available", return_value=False),
                patch("praxist.cli.takeover.Path.cwd", return_value=root),
            ):
                self.assertEqual(_select_task_path(None), (root.resolve(), False))

    def test_template_dry_run_has_no_filesystem_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "demo"
            with (
                patch(
                    "praxist.cli.takeover._select_task_path",
                    return_value=(destination, True),
                ),
                patch("praxist.cli.takeover._prepare_template_project") as prepare,
            ):
                code, out, err = self._run(["takeover", "--dry-run", "--json"])
        self.assertEqual(code, 0, msg=out + err)
        prepare.assert_not_called()
        self.assertFalse(destination.exists())

    def test_template_preparation_copies_bundle_and_reuses_task(self) -> None:
        from praxist.cli.takeover import TakeoverError, _prepare_template_project

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            destination = root / "demo"
            prepared = _prepare_template_project(destination)
            self.assertEqual(prepared, destination.resolve())
            self.assertTrue((prepared / "task.yaml").is_file())
            self.assertEqual(_prepare_template_project(destination), prepared)

            conflict = root / "conflict"
            conflict.mkdir()
            with self.assertRaisesRegex(TakeoverError, "already exists"):
                _prepare_template_project(conflict)

    def test_live_template_is_prepared_before_codex_launch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "demo"
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch(
                    "praxist.cli.takeover._select_task_path",
                    return_value=(destination, True),
                ),
                patch(
                    "praxist.cli.takeover._prepare_template_project",
                    return_value=destination,
                ) as prepare,
                patch("praxist.cli.takeover._codex_command") as command,
                patch("praxist.cli.takeover.subprocess.run", return_value=completed) as run,
            ):
                code, out, err = self._run(["takeover", "--configured-provider", "--yes"])
        self.assertEqual(code, 0, msg=out + err)
        prepare.assert_called_once_with(destination)
        command.assert_not_called()
        run.assert_called_once_with(
            [sys.executable, "-m", "praxist", "resolve", str(destination)],
            check=False,
        )

    def test_forced_skills_and_missing_codex_contract(self) -> None:
        from praxist.cli.takeover import TakeoverError, _codex_command, _takeover_skill

        root = Path.cwd()
        self.assertEqual(
            _takeover_skill(task_path=root, force_codex_native=True, force_configured=False),
            "praxist-takeover-codex",
        )
        self.assertEqual(
            _takeover_skill(task_path=root, force_codex_native=False, force_configured=True),
            "praxist-takeover",
        )
        with (
            patch("praxist.cli.takeover.shutil.which", return_value=None),
            self.assertRaisesRegex(TakeoverError, "not on PATH"),
        ):
            _codex_command(root, "praxist-takeover", require_installed=True)

        with (
            patch(
                "praxist.cli.takeover._bundled_codex_binary",
                side_effect=TakeoverError("package-pinned Codex CLI is unavailable"),
            ),
            self.assertRaisesRegex(TakeoverError, "package-pinned"),
        ):
            _codex_command(root, "praxist-takeover-codex", require_installed=True)

    def test_finish_setup_exits_without_opening_codex(self) -> None:
        with (
            patch("praxist.cli.takeover._select_task_path", return_value=(None, False)),
            patch("praxist.cli.takeover.subprocess.run") as run,
        ):
            code, out, err = self._run(["takeover"])
        self.assertEqual(code, 0, msg=out + err)
        self.assertIn("no research project was launched", err)
        run.assert_not_called()

    def test_escape_from_confirmation_returns_to_project_selection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with (
                patch(
                    "praxist.cli.takeover._select_task_path",
                    side_effect=[(root, False), (None, False)],
                ) as select,
                patch("praxist.cli.takeover._codex_command", return_value=["codex"]),
                patch("praxist.cli.takeover.interactive_terminal_available", return_value=True),
                patch(
                    "praxist.cli.takeover.confirm_action",
                    side_effect=TerminalInteractionCancelled("back"),
                ),
                patch("praxist.cli.takeover.subprocess.run") as run,
            ):
                code, out, err = self._run(["takeover"])
        self.assertEqual(code, 0, msg=out + err)
        self.assertEqual(select.call_count, 2)
        self.assertIn("Returning to project selection", err)
        run.assert_not_called()
