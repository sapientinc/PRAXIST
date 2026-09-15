"""Regression tests for the Praxist user-level uninstaller."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.cli import uninstall

REPO_ROOT = Path(__file__).resolve().parents[2]
UNINSTALLER = REPO_ROOT / "praxist-uninstall.sh"


class PraxistUninstallTest(unittest.TestCase):
    def _environment(self, root: Path) -> tuple[dict[str, str], Path, Path, Path]:
        home = root / "home"
        bin_dir = home / "bin"
        skills_dir = home / "skills"
        env = {
            "HOME": str(home),
            "XDG_BIN_HOME": str(bin_dir),
            "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_DATA_HOME": str(home / "data"),
            "XDG_STATE_HOME": str(home / "state"),
            "XDG_CACHE_HOME": str(home / "cache"),
            "CODEX_SKILLS_DIR": str(skills_dir),
            "PRAXIST_CONFIG_FILE": "",
        }
        return env, home, bin_dir, skills_dir

    @staticmethod
    def _create_install(env: dict[str, str], bin_dir: Path) -> Path:
        venv = Path(env["XDG_DATA_HOME"]) / "praxist" / "venv"
        scripts = venv / "bin"
        scripts.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        (venv / uninstall.MANAGED_VENV_MARKER).write_text(
            uninstall.MANAGED_VENV_MARKER_CONTENT,
            encoding="utf-8",
        )
        for name in uninstall.ENTRYPOINT_NAMES:
            target = scripts / name
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            target.chmod(0o755)
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / name).symlink_to(target)
        for key in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
            app_root = Path(env[key]) / "praxist"
            app_root.mkdir(parents=True)
            (app_root / "state.txt").write_text("managed\n", encoding="utf-8")
        return venv

    @staticmethod
    def _no_runs():
        return patch("praxist.cli.status.collect_status_rows", return_value=[])

    @staticmethod
    def _managed_skills(skills_dir: Path, *, dry_run: bool = False) -> dict[str, object]:
        return {
            "removed": [str(skills_dir / "praxist-control")],
            "missing": [],
            "refused": [],
            "dry_run": dry_run,
        }

    def test_default_skill_targets_include_each_managed_agent_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw).resolve()
            codex = root / "codex"
            claude = root / "claude"
            for target in (codex, claude):
                target.mkdir()
                (target / ".praxist-skills.json").write_text("{}\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"CODEX_SKILLS_DIR": str(codex), "CLAUDE_SKILLS_DIR": str(claude)},
                clear=False,
            ):
                targets = uninstall._skill_targets(None)

        self.assertEqual(targets, [("codex", codex.resolve()), ("claude", claude.resolve())])

    def test_full_uninstall_removes_only_managed_user_installation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            env, home, bin_dir, skills_dir = self._environment(root)
            project = root / "research-project"
            run_dir = project / "experiments" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text("{}\n", encoding="utf-8")
            venv = self._create_install(env, bin_dir)

            with (
                patch.dict(os.environ, env, clear=False),
                patch("pathlib.Path.home", return_value=home),
                self._no_runs(),
                patch(
                    "praxist.cli.uninstall.uninstall_codex_skills",
                    return_value=self._managed_skills(skills_dir),
                ),
            ):
                result = uninstall.uninstall_installation(skills_dir=skills_dir)

            self.assertFalse(venv.exists())
            for name in uninstall.ENTRYPOINT_NAMES:
                self.assertFalse((bin_dir / name).exists())
            for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
                self.assertFalse((Path(env[key]) / "praxist").exists())
            self.assertTrue((run_dir / "result.json").is_file())
            self.assertIn(str(skills_dir / "praxist-control"), result["removed"])

    def test_keep_user_data_removes_venv_but_preserves_application_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            env, home, bin_dir, skills_dir = self._environment(root)
            venv = self._create_install(env, bin_dir)

            with (
                patch.dict(os.environ, env, clear=False),
                patch("pathlib.Path.home", return_value=home),
                self._no_runs(),
                patch(
                    "praxist.cli.uninstall.uninstall_codex_skills",
                    return_value=self._managed_skills(skills_dir),
                ),
            ):
                result = uninstall.uninstall_installation(
                    skills_dir=skills_dir,
                    keep_user_data=True,
                )

            self.assertFalse(venv.exists())
            self.assertTrue((Path(env["XDG_CONFIG_HOME"]) / "praxist").is_dir())
            self.assertTrue((Path(env["XDG_DATA_HOME"]) / "praxist").is_dir())
            self.assertTrue(result["keep_user_data"])

    def test_dry_run_reports_without_removing_anything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            env, home, bin_dir, skills_dir = self._environment(root)
            venv = self._create_install(env, bin_dir)

            with (
                patch.dict(os.environ, env, clear=False),
                patch("pathlib.Path.home", return_value=home),
                self._no_runs(),
                patch(
                    "praxist.cli.uninstall.uninstall_codex_skills",
                    return_value=self._managed_skills(skills_dir, dry_run=True),
                ),
            ):
                result = uninstall.uninstall_installation(
                    skills_dir=skills_dir,
                    dry_run=True,
                )

            self.assertTrue(venv.is_dir())
            self.assertTrue((bin_dir / "praxist").is_symlink())
            self.assertTrue((Path(env["XDG_CONFIG_HOME"]) / "praxist").is_dir())
            self.assertTrue(result["dry_run"])

    def test_active_or_uninspectable_runs_block_before_removal(self) -> None:
        active = SimpleNamespace(source="registry", run_id="run-live", pid=123)
        with (
            patch(
                "praxist.cli.status.collect_status_rows",
                return_value=[active],
            ),
            self.assertRaisesRegex(uninstall.UninstallError, "run-live"),
        ):
            uninstall.uninstall_installation()

        def report_error(*, errors: list[str], **_kwargs):
            errors.append("ps unavailable")
            return []

        with (
            patch(
                "praxist.cli.status.collect_status_rows",
                side_effect=report_error,
            ),
            self.assertRaisesRegex(uninstall.UninstallError, "could not verify"),
        ):
            uninstall.uninstall_installation()

    def test_refused_skill_path_aborts_without_touching_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            env, home, bin_dir, skills_dir = self._environment(root)
            self._create_install(env, bin_dir)
            refused = skills_dir / "praxist-control"

            with (
                patch.dict(os.environ, env, clear=False),
                patch("pathlib.Path.home", return_value=home),
                self._no_runs(),
                patch(
                    "praxist.cli.uninstall.uninstall_codex_skills",
                    return_value={"removed": [], "missing": [], "refused": [str(refused)]},
                ),
                self.assertRaisesRegex(uninstall.UninstallError, "unmanaged skill"),
            ):
                uninstall.uninstall_installation(skills_dir=skills_dir)

            self.assertTrue((bin_dir / "praxist").is_symlink())

    def test_skill_verification_error_is_normalized(self) -> None:
        from praxist.cli.install_skills import InstallSkillsError

        with (
            self._no_runs(),
            patch(
                "praxist.cli.uninstall.uninstall_codex_skills",
                side_effect=InstallSkillsError("ownership mismatch"),
            ),
            self.assertRaisesRegex(uninstall.UninstallError, "ownership mismatch"),
        ):
            uninstall.uninstall_installation()

    def test_unowned_entrypoints_and_custom_venv_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            env, home, bin_dir, skills_dir = self._environment(root)
            custom_venv = root / "shared-venv"
            (custom_venv / "bin").mkdir(parents=True)
            (custom_venv / "pyvenv.cfg").write_text("shared\n", encoding="utf-8")
            bin_dir.mkdir(parents=True)
            (bin_dir / "praxist").write_text("user command\n", encoding="utf-8")
            other = root / "other-command"
            other.write_text("user command\n", encoding="utf-8")
            (bin_dir / "praxist-uninstall").symlink_to(other)

            with (
                patch.dict(os.environ, env, clear=False),
                patch("pathlib.Path.home", return_value=home),
                self._no_runs(),
                patch(
                    "praxist.cli.uninstall.uninstall_codex_skills",
                    return_value=self._managed_skills(skills_dir),
                ),
            ):
                result = uninstall.uninstall_installation(
                    venv_dir=custom_venv,
                    bin_dir=bin_dir,
                    skills_dir=skills_dir,
                    keep_user_data=True,
                )

            self.assertTrue(custom_venv.is_dir())
            self.assertTrue((bin_dir / "praxist").is_file())
            self.assertTrue((bin_dir / "praxist-uninstall").is_symlink())
            reasons = {item["reason"] for item in result["preserved"]}
            self.assertIn("entrypoint is not an installer symlink", reasons)
            self.assertIn("entrypoint points outside this install", reasons)
            self.assertIn("virtualenv is not proven installer-managed", reasons)

    def test_custom_venv_symlink_requires_marker_and_never_follows_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            target = root / "target"
            target.mkdir()
            link = root / "venv-link"
            link.symlink_to(target, target_is_directory=True)
            removed: list[str] = []
            missing: list[str] = []
            preserved: list[dict[str, str]] = []

            uninstall._remove_managed_venv(
                link,
                default_venv=root / "default",
                dry_run=False,
                removed=removed,
                missing=missing,
                preserved=preserved,
            )
            self.assertTrue(link.is_symlink())
            self.assertIn("not proven", preserved[0]["reason"])

            (target / uninstall.MANAGED_VENV_MARKER).write_text(
                uninstall.MANAGED_VENV_MARKER_CONTENT,
                encoding="utf-8",
            )
            preserved.clear()
            uninstall._remove_managed_venv(
                link,
                default_venv=root / "default",
                dry_run=False,
                removed=removed,
                missing=missing,
                preserved=preserved,
            )
            self.assertFalse(link.exists() or link.is_symlink())
            self.assertTrue(target.is_dir())
            self.assertIn("not followed", preserved[0]["reason"])

    def test_custom_config_outside_application_roots_is_reported_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            env, home, _bin_dir, skills_dir = self._environment(root)
            custom_config = root / "operator" / "praxist.env"
            custom_config.parent.mkdir()
            custom_config.write_text("keep\n", encoding="utf-8")
            env["PRAXIST_CONFIG_FILE"] = str(custom_config)

            with (
                patch.dict(os.environ, env, clear=False),
                patch("pathlib.Path.home", return_value=home),
                self._no_runs(),
                patch(
                    "praxist.cli.uninstall.uninstall_codex_skills",
                    return_value=self._managed_skills(skills_dir),
                ),
            ):
                result = uninstall.uninstall_installation(
                    skills_dir=skills_dir,
                    keep_user_data=True,
                )

            self.assertTrue(custom_config.is_file())
            self.assertTrue(any(item["path"] == str(custom_config) for item in result["preserved"]))

    def test_cli_json_success_and_normalized_error(self) -> None:
        payload = {
            "dry_run": True,
            "keep_user_data": False,
            "removed": [],
            "missing": [],
            "preserved": [],
            "skills_dir": "/skills",
            "venv_dir": "/venv",
            "bin_dir": "/bin",
        }
        with (
            patch("praxist.cli.uninstall.uninstall_installation", return_value=payload),
            redirect_stdout(stdout := io.StringIO()),
            redirect_stderr(stderr := io.StringIO()),
        ):
            code = uninstall.cmd_uninstall(
                SimpleNamespace(
                    venv_dir=None,
                    bin_dir=None,
                    skills_dir=None,
                    keep_user_data=False,
                    dry_run=True,
                    json_output=True,
                )
            )
        self.assertEqual(code, 0)
        self.assertTrue('"dry_run": true' in stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

        with (
            patch(
                "praxist.cli.uninstall.uninstall_installation",
                side_effect=uninstall.UninstallError("active run"),
            ),
            redirect_stderr(stderr := io.StringIO()),
        ):
            code = uninstall.cmd_uninstall(
                SimpleNamespace(
                    venv_dir=None,
                    bin_dir=None,
                    skills_dir=None,
                    keep_user_data=False,
                    dry_run=False,
                    json_output=False,
                )
            )
        self.assertEqual(code, 1)
        self.assertIn("active run", stderr.getvalue())

    def test_shell_entrypoint_delegates_all_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            root = Path(tmp_raw)
            log = root / "args.log"
            fake_cli = root / "praxist"
            fake_cli.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$*" > "$PRAXIST_TEST_LOG"\n',
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)
            result = subprocess.run(
                ["bash", str(UNINSTALLER), "--dry-run", "--keep-user-data"],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PRAXIST_UNINSTALL_CLI": str(fake_cli),
                    "PRAXIST_TEST_LOG": str(log),
                },
            )
            observed = log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(observed, "uninstall --dry-run --keep-user-data\n")

    def test_unsafe_application_root_and_io_failure_are_normalized(self) -> None:
        with self.assertRaisesRegex(uninstall.UninstallError, "unsafe application root"):
            uninstall._assert_application_root(Path.home())
        with self.assertRaisesRegex(uninstall.UninstallError, "shallow application root"):
            uninstall._assert_application_root(Path("/praxist"))

        with tempfile.TemporaryDirectory() as tmp_raw:
            path = Path(tmp_raw) / "praxist"
            path.mkdir()
            with (
                patch("praxist.cli.uninstall.shutil.rmtree", side_effect=OSError("busy")),
                self.assertRaisesRegex(uninstall.UninstallError, "busy"),
            ):
                uninstall._remove_path(path, dry_run=False, removed=[], missing=[])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
