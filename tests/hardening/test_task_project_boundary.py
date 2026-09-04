from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from praxist.core.panel_topology import panel_topology_from_manifest
from praxist.core.registry import ALL_KINDS, PluginLoader, PluginRoots
from praxist.core.replay import verify_run
from praxist.core.role_skills import load_role_skill
from praxist.core.task_project import resolve_task_project, task_project_global_plugin_refs
from praxist.run import _default_run_dir_for_task_project, cmd_run
from praxist.task_spec import load_task_spec


class TaskProjectBoundaryTest(unittest.TestCase):
    def test_tasks_are_not_bundled_plugins(self) -> None:
        self.assertNotIn("task", ALL_KINDS)
        self.assertFalse((Path.cwd() / "praxist" / "plugins" / "tasks").exists())
        self.assertEqual(list((Path.cwd() / "praxist" / "plugins").glob("**/fake*")), [])

        loader = PluginLoader(PluginRoots.defaults(Path.cwd()))
        discovered_kinds = {candidate.identity.kind for candidate in loader.discover().candidates}
        self.assertNotIn("task", discovered_kinds)

    def test_tracked_templates_are_explicit_task_projects(self) -> None:
        for rel in (
            "templates/tasks/template",
            "templates/tasks/toy_math",
            "templates/tasks/sam_optimizer",
            "templates/tasks/machine_learning_template",
        ):
            project = resolve_task_project(Path.cwd() / rel, workspace=Path.cwd())
            self.assertEqual(project.manifest["source"], "external_task_project")
            self.assertTrue(project.manifest["files"])
            refs = {ref.as_string() for ref in task_project_global_plugin_refs(project.descriptor)}
            self.assertIn("workflow_stage:research_loop", refs)
            self.assertNotIn(project.task_ref, refs)

    def test_machine_learning_template_exposes_generic_ml_task_contract(self) -> None:
        task_root = Path.cwd() / "templates" / "tasks" / "machine_learning_template"
        project = resolve_task_project(task_root, workspace=Path.cwd())
        spec = load_task_spec(task_root / "task.yaml")

        self.assertEqual(project.task_ref, "task:machine_learning_template")
        self.assertEqual(spec.prompt_layout.base_template, "prompt_base.jinja2")
        self.assertEqual(spec.prompt_layout.generation_template, "prompt_generation.jinja2")
        self.assertEqual(spec.get_prompt_base_path(Path("fallback")).name, "prompt_base.jinja2")
        self.assertEqual(
            spec.get_prompt_generation_path(Path("fallback")).name,
            "prompt_generation.jinja2",
        )
        self.assertEqual(spec.panel_topology_ref, "panel_topology:machine_learning_template")
        lane_names = {lane["name"] for lane in spec.evaluation.frontier_lanes}
        self.assertEqual(lane_names, {"confirmed", "incubator", "task_candidate", "diagnostic"})
        lane_by_name = {lane["name"]: lane for lane in spec.evaluation.frontier_lanes}
        self.assertEqual(lane_by_name["confirmed"]["require_metrics"], ["task_score"])
        self.assertEqual(
            lane_by_name["confirmed"]["require_truthy_metrics"],
            ["scored_complete"],
        )
        self.assertEqual(lane_by_name["incubator"]["require_metrics"], ["task_score"])
        self.assertEqual(
            lane_by_name["incubator"]["require_truthy_metrics"],
            ["scored_complete"],
        )
        self.assertTrue(lane_by_name["incubator"]["allow_non_promotable"])
        self.assertEqual(lane_by_name["task_candidate"]["require_metrics"], [])

        for role_ref in (
            "task_role:peer_generalist",
            "task_role:starter",
            "task_role:solver",
            "task_role:analyst",
            "task_role:literature_scout",
            "task_role:builder_pi",
            "task_role:skeptic_pi",
            "task_role:portfolio_pi",
            "task_role:external_validity_pi",
            "task_role:chair",
        ):
            role = load_role_skill(role_ref, workspace=Path.cwd(), task_project_path=task_root)
            self.assertTrue(role.skill_markdown.strip(), role_ref)

        topology_manifest = yaml.safe_load(
            (
                task_root
                / ".praxist/plugins/panel_topologies/machine_learning_template/plugin.yaml"
            ).read_text(encoding="utf-8")
        )
        topology = panel_topology_from_manifest(
            "panel_topology:machine_learning_template",
            topology_manifest,
            plugin_path=task_root / ".praxist/plugins/panel_topologies/machine_learning_template",
        )
        self.assertEqual(topology.chair_role_ref, "task_role:chair")
        self.assertEqual(
            topology.peer_role_rotation,
            ("starter", "solver", "solver", "analyst", "peer_generalist"),
        )
        self.assertIn("task_role:builder_pi", topology.role_refs_for_mode("full"))

        all_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(task_root.rglob("*"))
            if path.is_file() and path.suffix in {".md", ".yaml", ".yml", ".jinja2", ".json"}
        )
        for forbidden in (
            "MLE-Bench",
            "official grader",
            "medal",
            "campaign",
            "praxist_mle_train",
        ):
            self.assertNotIn(forbidden, all_text)
        for required in (
            "QD-DIG",
            "Gems",
            "research memory",
            "frontier lanes",
            "lower-admission durable",
            "tool_server:run_report",
            "share_finding",
            "evidence maturity",
            "metrics.scored_complete",
            'frontier_lane":"performance"',
            "lane-routing regression",
            'extra.frontier_lane="task_candidate"',
        ):
            self.assertIn(required, all_text)

        evaluator_help = subprocess.run(
            [
                sys.executable,
                str(task_root / "evaluations" / "primary" / "run.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(evaluator_help.returncode, 0, evaluator_help.stderr)
        self.assertIn("--prediction-artifact", evaluator_help.stdout)

        skill_text = Path("skills/praxist-task-initialization/SKILL.md").read_text(encoding="utf-8")
        self.assertIn("templates/tasks/machine_learning_template", skill_text)

    def test_tracked_template_audit_rules_are_declarative(self) -> None:
        """Keep tutorial task audits editable as text, not Python framework code."""
        templates_root = Path.cwd() / "templates" / "tasks"
        offenders = sorted(
            str(path.relative_to(Path.cwd()))
            for path in templates_root.glob("*/audit_rules/**/*.py")
        )
        self.assertEqual(offenders, [])

        for audit_yaml in sorted(templates_root.glob("*/audit_rules/**/audit.yaml")):
            manifest = yaml.safe_load(audit_yaml.read_text(encoding="utf-8"))
            rel = audit_yaml.relative_to(Path.cwd())
            self.assertEqual(manifest.get("kind"), "task_audit", rel)
            self.assertEqual(manifest.get("mode"), "declarative", rel)
            self.assertNotIn("entrypoint", manifest, rel)
            self.assertNotIn("code", manifest, rel)

    def test_cli_runs_task_project_without_task_plugin_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "toy_math_run"
            args = SimpleNamespace(
                fake=False,
                task="",
                task_path=str(Path.cwd() / "templates" / "tasks" / "toy_math"),
                task_spec="",
                workspace=tmp,
                model="",
                runtime="",
                model_provider="",
                budget_policy="",
                credential_profile="",
                run_dir=str(run_dir),
                resolve_only=True,
                local=True,
                frontier_strategy="auto",
            )
            with patch("sys.stdout"):
                cmd_run(args)

            run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_json["task_ref"], "task:toy_math")
            self.assertTrue((run_dir / "task_project_manifest.json").exists())
            resolution = json.loads(
                (run_dir / "plugin_resolution.json").read_text(encoding="utf-8")
            )
            selected_refs = {
                item["metadata"]["kind"] + ":" + item["metadata"]["name"]
                for item in resolution["selected"]
            }
            self.assertNotIn("task:toy_math", selected_refs)
            self.assertTrue(verify_run(run_dir)["success"])

    def test_task_project_can_own_default_experiment_root(self) -> None:
        project = resolve_task_project(
            Path.cwd() / "templates" / "tasks" / "sam_optimizer",
            workspace=Path.cwd(),
        )

        run_dir = _default_run_dir_for_task_project(
            project,
            workspace=Path.cwd(),
            task_ref=project.task_ref,
        )

        self.assertEqual(run_dir.parent, project.path / "experiments")
        self.assertTrue(run_dir.name.startswith("run_"))
        self.assertTrue(run_dir.name.endswith("_sam_optimizer"))

    def test_sam_template_does_not_bundle_reference_implementation_corpus(self) -> None:
        task_root = Path.cwd() / "templates" / "tasks" / "sam_optimizer"
        project = resolve_task_project(task_root, workspace=Path.cwd())
        asset_manifest = json.loads(
            (task_root / "assets" / "task_assets_manifest.json").read_text(encoding="utf-8")
        )

        self.assertFalse((task_root / "assets" / "reference_implementations").exists())
        self.assertNotIn("reference_implementations", project.descriptor.get("task_assets", {}))
        self.assertNotIn("reference_implementations", asset_manifest)
        self.assertFalse(
            any("reference_implementations" in item["path"] for item in project.manifest["files"])
        )

    def test_task_project_manifest_excludes_local_experiment_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "task"
            task_root.mkdir()
            (task_root / "task.yaml").write_text(
                "task_id: output_skip\n"
                "capabilities: [testing.fake_workflow_fixture]\n"
                "praxist_plugins:\n"
                "  workflow_stage: workflow_stage:research_loop\n"
                "runtime_outputs:\n"
                "  root: experiments\n",
                encoding="utf-8",
            )
            (task_root / "description.md").write_text("task", encoding="utf-8")
            generated = task_root / "experiments" / "run_x" / "results"
            generated.mkdir(parents=True)
            (generated / "metrics.json").write_text("{}", encoding="utf-8")
            asset_fixture = task_root / "assets" / "results" / "fixture.json"
            asset_fixture.parent.mkdir(parents=True)
            asset_fixture.write_text("{}", encoding="utf-8")

            project = resolve_task_project(task_root, workspace=Path.cwd())

        manifest_paths = {item["path"] for item in project.manifest["files"]}
        self.assertIn("description.md", manifest_paths)
        self.assertIn("assets/results/fixture.json", manifest_paths)
        self.assertFalse(any(path.startswith("experiments/") for path in manifest_paths))

    def test_generated_reports_do_not_change_task_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "task"
            task_root.mkdir()
            (task_root / "task.yaml").write_text(
                "task_id: report_identity\n"
                "praxist_plugins:\n"
                "  workflow_stage: workflow_stage:research_loop\n",
                encoding="utf-8",
            )
            docs = task_root / "docs"
            docs.mkdir()
            (docs / "task_contract.md").write_text("canonical task documentation", encoding="utf-8")
            before = resolve_task_project(task_root, workspace=Path.cwd()).manifest

            reports = docs / "praxist_reports"
            reports.mkdir()
            (reports / "run_report.md").write_text("derived report", encoding="utf-8")
            (reports / "run_report.pdf").write_bytes(b"derived pdf")
            after = resolve_task_project(task_root, workspace=Path.cwd()).manifest

        self.assertEqual(after["sha256"], before["sha256"])
        self.assertEqual(after["files"], before["files"])
        manifest_paths = {item["path"] for item in after["files"]}
        self.assertIn("docs/task_contract.md", manifest_paths)
        self.assertFalse(any(path.startswith("docs/praxist_reports/") for path in manifest_paths))

    def test_legacy_bash_launcher_is_absent_from_product_sources(self) -> None:
        legacy_name = "run_auto_" + "research.sh"
        roots = (
            Path(".github"),
            Path("docs"),
            Path("examples"),
            Path("praxist"),
            Path("scripts"),
            Path("skills"),
            Path("templates"),
            Path("tests"),
            Path("AGENTS.md"),
            Path("README.md"),
            Path("mkdocs.yml"),
            Path("pyproject.toml"),
        )
        references: list[str] = []
        for root in roots:
            paths = (root,) if root.is_file() else root.rglob("*")
            for path in paths:
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                if legacy_name in text:
                    references.append(path.as_posix())

        self.assertFalse(Path(legacy_name).exists())
        self.assertEqual(references, [])


if __name__ == "__main__":
    unittest.main()
