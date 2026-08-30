"""Tests for ``praxist resolve`` — task-project resolve-only smoke test.

``praxist resolve`` is a thin delegator: it builds an argparse.Namespace
that ``praxist.run.cmd_run`` already understands and hands off
with ``resolve_only=True`` / ``local=True``. These tests mock
``cmd_run`` to verify the argument hand-off contract.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ResolveCliEndToEndTest(unittest.TestCase):
    """``praxist resolve`` is dispatched and forwards to ``cmd_run``."""

    def _run(
        self,
        argv: list[str],
        *,
        validate_task_project=None,
    ) -> tuple[int, str, str, list]:
        from praxist.cli import main

        captured: list = []

        def fake_cmd_run(args: argparse.Namespace) -> None:
            captured.append(args)

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            tempfile.TemporaryDirectory() as config_home,
            patch.dict(
                os.environ,
                {
                    "XDG_CONFIG_HOME": config_home,
                    "PRAXIST_AGENT_SYSTEM": "",
                    "PRAXIST_AGENT_RUNTIME_REF": "",
                    "RUNTIME_REF": "",
                    "PRAXIST_LLM_PROVIDER": "",
                    "PRAXIST_MODEL": "",
                    "PRAXIST_MODEL_PROVIDER_REF": "",
                    "MODEL_PROVIDER_REF": "",
                    "MODEL": "",
                    "DEEPSEEK_API_KEY": "",
                    "OPENROUTER_API_KEY": "",
                    "ORCAROUTER_API_KEY": "",
                },
                clear=False,
            ),
            patch("praxist.cli.start._resolve_task_path", side_effect=lambda value: Path(value)),
            patch(
                "praxist.cli.start._validate_task_project",
                side_effect=validate_task_project,
            ),
            patch("praxist.run.cmd_run", side_effect=fake_cmd_run),
        ):
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    main(argv)
                code = 0
            except SystemExit as exc:
                code = int(exc.code or 0)
        return code, stdout.getvalue(), stderr.getvalue(), captured

    def test_resolve_surfaces_unwired_baseline_warning(self) -> None:
        from praxist.cli import start

        real_validate = start._validate_task_project
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "task"
            task.mkdir()
            (task / "task.yaml").write_text(
                "task_id: demo\ntask_name: demo\n",
                encoding="utf-8",
            )
            baseline_dir = task / "assets" / "baselines"
            baseline_dir.mkdir(parents=True)
            (baseline_dir / "results.jsonl").write_text(
                json.dumps({"metric_name": "score", "metric_value": 0.75}) + "\n",
                encoding="utf-8",
            )

            code, _out, stderr, captured = self._run(
                ["resolve", str(task)],
                validate_task_project=real_validate,
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(captured), 1)
        self.assertIn("WARNING: task.yaml declares no baselines", stderr)

    def test_resolve_forwards_task_path_with_resolve_only_and_local(self) -> None:
        code, _out, _err, captured = self._run(["resolve", "/path/to/task"])
        self.assertEqual(code, 0)
        self.assertEqual(len(captured), 1)
        ns = captured[0]
        # Resolve-only + local mode are the entire point — must be on.
        self.assertTrue(ns.resolve_only)
        self.assertTrue(ns.local)
        # Task path comes from the positional arg.
        self.assertEqual(ns.task_path, "/path/to/task")
        # Effective defaults are resolved before the hand-off.
        self.assertEqual(ns.workspace, "")
        self.assertEqual(ns.run_dir, "")
        self.assertEqual(ns.runtime, "agent_runtime:claude_sdk")
        self.assertEqual(ns.model_provider, "model_provider:anthropic_messages")
        self.assertEqual(ns.budget_policy, "")
        self.assertEqual(ns.credential_profile, "")
        self.assertEqual(ns.model, "claude-opus-4-7")
        # Resolve-only is not the fake-fixture path.
        self.assertFalse(ns.fake)
        self.assertEqual(ns.task, "")
        self.assertEqual(ns.task_spec, "")
        # Frontier strategy stays at the default; resolve-only doesn't use it.
        self.assertEqual(ns.frontier_strategy, "auto")

    def test_resolve_forwards_optional_overrides(self) -> None:
        code, _out, _err, captured = self._run(
            [
                "resolve",
                "/path/to/task",
                "--workspace",
                "/ws",
                "--run-dir",
                "/runs/r",
                "--runtime",
                "agent_runtime:fake",
                "--model-provider",
                "model_provider:fake_provider",
                "--budget-policy",
                "budget_policy:fake_tiered",
                "--credential-profile",
                "ci",
                "--model",
                "claude-opus-4-7",
            ]
        )
        self.assertEqual(code, 0)
        ns = captured[0]
        self.assertEqual(ns.workspace, "/ws")
        self.assertEqual(ns.run_dir, "/runs/r")
        self.assertEqual(ns.runtime, "agent_runtime:fake")
        self.assertEqual(ns.model_provider, "model_provider:fake_provider")
        self.assertEqual(ns.budget_policy, "budget_policy:fake_tiered")
        self.assertEqual(ns.credential_profile, "ci")
        self.assertEqual(ns.model, "claude-opus-4-7")
        # Resolve-only is still on regardless of overrides.
        self.assertTrue(ns.resolve_only)
        self.assertTrue(ns.local)

    def test_resolve_validates_requested_actual_result_summary(self) -> None:
        with patch("praxist.cli.resolve._validate_result_summary_contract") as validate:
            code, _out, _err, captured = self._run(
                [
                    "resolve",
                    "/path/to/task",
                    "--result-summary",
                    "/path/to/evaluation_summary.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(captured), 1)
        validate.assert_called_once_with(
            Path("/path/to/task"),
            Path("/path/to/evaluation_summary.json"),
        )

    def test_resolve_loads_effective_values_from_selected_config(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text(
                "\n".join(
                    [
                        "export PRAXIST_AGENT_SYSTEM=codex_sdk",
                        "export PRAXIST_LLM_PROVIDER=deepseek",
                        "export PRAXIST_MODEL=deepseek-v4-pro[1m]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            code, _out, _err, captured = self._run(
                ["resolve", "/path/to/task", "--config-file", str(config_file)]
            )
        self.assertEqual(code, 0)
        self.assertEqual(captured[0].runtime, "agent_runtime:codex_sdk")
        self.assertEqual(captured[0].model_provider, "model_provider:deepseek_alias")
        self.assertEqual(captured[0].model, "deepseek-v4-pro[1m]")

    def test_codex_native_overrides_configured_relay_provider(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text(
                "\n".join(
                    [
                        "export PRAXIST_AGENT_SYSTEM=codex_sdk",
                        "export PRAXIST_LLM_PROVIDER=deepseek",
                        "export PRAXIST_MODEL=deepseek-v4-pro[1m]",
                        "export DEEPSEEK_API_KEY=relay-key",
                        "export OPENAI_API_KEY=api-billing-key",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            code, _out, _err, captured = self._run(
                [
                    "resolve",
                    "/path/to/task",
                    "--config-file",
                    str(config_file),
                    "--codex-native",
                    "--model",
                    "gpt-5.6-luna",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(captured[0].runtime, "agent_runtime:codex_sdk")
        self.assertEqual(captured[0].model_provider, "model_provider:openai_compatible")
        self.assertEqual(captured[0].model, "gpt-5.6-luna")

    def test_codex_native_does_not_reuse_provider_specific_model_default(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "env"
            config_file.write_text(
                "\n".join(
                    [
                        "export PRAXIST_AGENT_SYSTEM=claude_sdk",
                        "export PRAXIST_LLM_PROVIDER=deepseek",
                        "export PRAXIST_MODEL=deepseek-v4-pro[1m]",
                        "export DEEPSEEK_API_KEY=relay-key",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            code, _out, _err, captured = self._run(
                [
                    "resolve",
                    "/path/to/task",
                    "--config-file",
                    str(config_file),
                    "--codex-native",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(captured[0].runtime, "agent_runtime:codex_sdk")
        self.assertEqual(captured[0].model_provider, "model_provider:openai_compatible")
        self.assertEqual(captured[0].model, "")

    def test_resolve_without_task_path_exits_nonzero(self) -> None:
        from praxist.cli import main

        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                main(["resolve"])
            code = 0
        except SystemExit as exc:
            code = int(exc.code or 0)
        # The current directory is the default task path and is validated.
        self.assertNotEqual(code, 0)
        self.assertIn("task.yaml", stderr.getvalue())

    def test_resource_scheduler_module_import_does_not_require_fcntl(self) -> None:
        script = """
import builtins
import os

real_import = builtins.__import__
def import_without_fcntl(name, *args, **kwargs):
    if name == "fcntl":
        raise ModuleNotFoundError("simulated native Windows host")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_fcntl
from praxist.plugins.workflow_stages.research_loop.backend import experiment_scheduler
from praxist.plugins.workflow_stages.research_loop.backend import resource_scheduler
from praxist.cli import main
main([
    "resolve",
    "templates/tasks/toy_math",
    "--run-dir", os.environ["PRAXIST_TEST_RUN_DIR"],
    "--runtime", "agent_runtime:claude_sdk",
    "--model-provider", "model_provider:anthropic_messages",
])
"""
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[2],
                env={
                    **os.environ,
                    "XDG_CONFIG_HOME": str(Path(tmp) / "config"),
                    "PRAXIST_TEST_RUN_DIR": str(Path(tmp) / "run"),
                },
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"status": "resolved"', completed.stdout)

    def test_result_summary_preflight_uses_ratio_output_not_stage_labels(self) -> None:
        from praxist.cli.resolve import _validate_result_summary_contract
        from templates.tasks.sam_optimizer.evaluations.pareto_tiered import run as evaluator

        task_spec = SimpleNamespace(
            evaluation=SimpleNamespace(
                maturity_policy={
                    "require_ratio_gate": True,
                    "min_effort_ratio": 0.75,
                    "min_coverage_ratio": 0.80,
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_summary = {"scored_cell_count": 15}
            evaluator.attach_maturity_ratios(metrics_summary, tier="T3")
            summary = {
                "variant_name": "candidate",
                "tiers": [{"tier": "T3", "metrics_summary": metrics_summary}],
            }
            evaluator._write_summary(root, "candidate", summary)
            summary_path = Path(summary["summary_path"])
            with patch("praxist.task_spec.load_task_spec", return_value=task_spec):
                _validate_result_summary_contract(root, summary_path)

            summary_path.write_text(
                json.dumps(
                    {
                        "variant_name": "candidate",
                        "current_aggregate": {"score": 1.0},
                        "evidence_stage": "complete",
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("praxist.task_spec.load_task_spec", return_value=task_spec),
                self.assertRaisesRegex(
                    ValueError,
                    "missing finite effort_ratio, coverage_ratio",
                ),
            ):
                _validate_result_summary_contract(root, summary_path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
