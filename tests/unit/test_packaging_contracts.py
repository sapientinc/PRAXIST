"""Packaging contracts for PyPI artifacts."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from praxist.testing import fake_workflow_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


class PyprojectPackagingContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    def test_wheel_installs_only_canonical_python_package(self) -> None:
        wheel = self.data["tool"]["hatch"]["build"]["targets"]["wheel"]

        self.assertEqual(wheel.get("packages"), ["praxist"])

    def test_wheel_places_generic_top_level_trees_under_package_resources(self) -> None:
        wheel = self.data["tool"]["hatch"]["build"]["targets"]["wheel"]
        force_include = wheel.get("force-include", {})

        self.assertEqual(
            force_include,
            {
                "docs": "praxist/resources/docs",
                "examples": "praxist/resources/examples",
                "templates": "praxist/resources/templates",
                "scripts": "praxist/resources/scripts",
                "skills": "praxist/resources/skills",
                "tests/fixtures/plugins": "praxist/testing/fixtures/plugins",
            },
        )
        for source, target in force_include.items():
            self.assertTrue((REPO_ROOT / source).is_dir(), f"{source} source tree is missing")
            self.assertTrue(
                target.startswith("praxist/"),
                f"{source} must stay inside the canonical wheel package",
            )

    def test_distribution_declares_the_canonical_license(self) -> None:
        self.assertEqual(self.data["project"]["license"], {"file": "LICENSE.md"})

    def test_documented_oobe_runtime_extras_are_publishable(self) -> None:
        extras = self.data["project"].get("optional-dependencies", {})

        self.assertIn("agents", extras)
        self.assertIn("codex", extras)
        self.assertTrue(extras["agents"])
        self.assertTrue(extras["codex"])

    def test_sdist_keeps_source_checkout_layout_for_reproducibility(self) -> None:
        sdist = self.data["tool"]["hatch"]["build"]["targets"]["sdist"]

        self.assertEqual(sdist.get("exclude"), ["/.venv*"])
        self.assertEqual(
            sdist.get("force-include", {}),
            {
                "docs": "docs",
                "examples": "examples",
                "templates": "templates",
                "scripts": "scripts",
                "skills": "skills",
            },
        )

    def test_templates_and_examples_have_distinct_source_directories(self) -> None:
        self.assertTrue((REPO_ROOT / "templates" / "tasks").is_dir())
        self.assertTrue((REPO_ROOT / "examples" / "rocket_booster_recovery").is_dir())
        self.assertTrue((REPO_ROOT / "examples" / "rocket_booster_recovery_rust").is_dir())

    def test_console_scripts_do_not_expose_generic_script_package_name(self) -> None:
        project_scripts = self.data["project"].get("scripts", {})

        self.assertEqual(
            project_scripts,
            {
                "praxist": "praxist.cli:main",
                "praxist-uninstall": "praxist.cli.uninstall:main",
                "praxist-collector": "praxist.product_usage.app:main",
                "praxist-retention": "praxist.product_usage.retention:main",
            },
        )

    def test_release_checkout_includes_one_click_uninstaller(self) -> None:
        uninstaller = REPO_ROOT / "praxist-uninstall.sh"

        self.assertTrue(uninstaller.is_file())
        self.assertTrue(uninstaller.stat().st_mode & 0o111)

    def test_skill_local_helper_scripts_remain_inside_skill_trees(self) -> None:
        skill_scripts = sorted(
            path
            for path in (REPO_ROOT / "skills").glob("*/scripts/*")
            if path.name != "__pycache__"
        )

        self.assertGreater(len(skill_scripts), 0)
        for script in skill_scripts:
            self.assertTrue(script.is_file(), f"{script.relative_to(REPO_ROOT)} must be a file")
            self.assertEqual(script.parents[2].name, "skills")

    def test_fake_workflow_uses_packaged_fixture_plugins_when_available(self) -> None:
        with TemporaryDirectory(prefix="praxist_packaged_fixtures_") as tmp_raw:
            package_root = Path(tmp_raw).resolve() / "site-packages" / "praxist"
            module_path = package_root / "testing" / "fake_workflow_fixture.py"
            fixture_root = package_root / "testing" / "fixtures" / "plugins"
            fixture_root.mkdir(parents=True)

            with patch.object(fake_workflow_fixture, "__file__", str(module_path)):
                roots = fake_workflow_fixture._fixture_plugin_roots(Path(tmp_raw))

            self.assertIn(fixture_root, roots.bundled)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
