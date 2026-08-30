from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml


def _write_plugin(
    root: Path, *, kind_dir: str, name: str, kind: str, contract_key: str, contract: dict
) -> Path:
    plugin_dir = root / kind_dir / name
    plugin_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "name": name,
        "kind": kind,
        "version": "0.1.0",
        "protocol_version": 1,
        "description": name,
        contract_key: contract,
    }
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return plugin_dir


class StartupHelperContractsTest(unittest.TestCase):
    def test_startup_defaults_env_and_optional_descriptors_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop import provider_env, startup
        from praxist.task_spec import GenerationPolicy, MultiPIConfig, TaskSpec

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "data"
            (data_root / "generic_task").mkdir(parents=True)
            sam_data = workspace / "sam_data"
            sam_data.mkdir()

            self.assertFalse(startup.is_research_loop_plugin_task("task:any"))
            with patch.object(startup, "resolve_task_project", side_effect=RuntimeError("bad")):
                self.assertFalse(startup.is_research_loop_task_project(workspace))
            with patch.object(
                startup,
                "resolve_task_project",
                return_value=SimpleNamespace(
                    descriptor={
                        "praxist_plugins": {"workflow": {"stage": startup.RESEARCH_LOOP_STAGE_REF}}
                    }
                ),
            ):
                self.assertTrue(startup.is_research_loop_task_project(workspace))
            with patch.object(
                startup,
                "resolve_task_project",
                return_value=SimpleNamespace(
                    descriptor={"praxist_plugins": {"workflow": {"stage": "workflow_stage:other"}}}
                ),
            ):
                self.assertFalse(startup.is_research_loop_task_project(workspace))
            self.assertEqual(
                startup.default_runtime_for_task("task:fake_panel"),
                "agent_runtime:fake_runtime",
            )
            self.assertEqual(
                startup.default_model_provider_for_task("task:fake_panel"),
                "model_provider:fake_provider",
            )
            with patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "deepseek-key",
                    "OPENROUTER_API_KEY": "openrouter-key",
                    "ORCAROUTER_API_KEY": "",
                    "ANTHROPIC_API_KEY": "anthropic-key",
                },
                clear=False,
            ):
                self.assertEqual(
                    startup.default_model_provider_for_task("task:x"),
                    "model_provider:deepseek_alias",
                )
            with patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "",
                    "OPENROUTER_API_KEY": "openrouter-key",
                    "ORCAROUTER_API_KEY": "",
                    "ANTHROPIC_API_KEY": "anthropic-key",
                },
                clear=False,
            ):
                self.assertEqual(
                    startup.default_model_provider_for_task("task:x"),
                    "model_provider:openrouter",
                )
            with patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "",
                    "OPENROUTER_API_KEY": "",
                    "ORCAROUTER_API_KEY": "orcarouter-key",
                    "ANTHROPIC_API_KEY": "anthropic-key",
                },
                clear=False,
            ):
                self.assertEqual(
                    startup.default_model_provider_for_task("task:x"),
                    "model_provider:orcarouter",
                )
            with patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "",
                    "OPENROUTER_API_KEY": "",
                    "ORCAROUTER_API_KEY": "",
                    "ANTHROPIC_API_KEY": "anthropic-key",
                },
                clear=False,
            ):
                self.assertEqual(
                    startup.default_model_provider_for_task("task:x"),
                    "model_provider:anthropic_messages",
                )
            self.assertEqual(
                startup.default_model_provider_for_task("task:x", "model_provider:custom"),
                "model_provider:custom",
            )
            self.assertEqual(
                startup.default_budget_policy_for_task("task:fake_panel"),
                "budget_policy:fake_tiered",
            )
            self.assertEqual(
                startup.default_runtime_for_task("task:x", "agent_runtime:custom"),
                "agent_runtime:custom",
            )

            self.assertEqual(
                startup._default_model_for_provider("model_provider:openai_compatible"),
                "gpt-5.2",
            )
            self.assertEqual(
                startup._default_model_for_provider("model_provider:openrouter"),
                "anthropic/claude-opus-4.7",
            )
            self.assertEqual(
                startup._default_model_for_provider("model_provider:anthropic_messages"),
                "claude-opus-4-7",
            )
            self.assertEqual(
                startup._default_model_for_provider("model_provider:fake_provider"),
                "fake-deterministic",
            )
            self.assertEqual(
                startup._default_model_for_provider("model_provider:deepseek_alias"),
                "deepseek-v4-pro[1m]",
            )
            self.assertEqual(startup._positive_int_env({}, "MISSING"), None)
            with self.assertRaises(ValueError):
                startup._positive_int_env({"MAX_GENERATIONS": "0"}, "MAX_GENERATIONS")
            with self.assertRaises(ValueError):
                startup._positive_int_env({"MAX_GENERATIONS": "bad"}, "MAX_GENERATIONS")

            task_spec = TaskSpec(
                task_id="generic_task",
                task_name="Generic Task",
                _task_dir=str(workspace),
                generation_policy=GenerationPolicy(max_generations=1, cohort_size=2),
                multi_pi=MultiPIConfig(enabled=True, chair_peer_budget=2),
            )
            descriptor = {
                "generation_policy": {"max_generations": 1},
                "multi_pi": {"enabled": True},
            }
            overridden, effective_descriptor, seen = startup._apply_internal_env_overrides(
                task_spec,
                descriptor,
                {"MAX_GENERATIONS": "3", "COHORT_SIZE": "4"},
            )
            self.assertEqual(overridden.generation_policy.max_generations, 3)
            self.assertEqual(overridden.generation_policy.cohort_size, 4)
            self.assertEqual(overridden.multi_pi.chair_peer_budget, 4)
            self.assertEqual(effective_descriptor["multi_pi"]["chair_peer_budget"], 4)
            self.assertEqual({item["env"] for item in seen}, {"MAX_GENERATIONS", "COHORT_SIZE"})

            self.assertEqual(
                provider_env.freeze_provider_env(
                    "model_provider:openrouter",
                    {
                        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1/messages",
                        "OPENROUTER_API_KEY": "key",
                    },
                )["ANTHROPIC_BASE_URL"],
                "https://openrouter.ai/api/v1/messages",
            )
            self.assertEqual(
                provider_env.freeze_provider_env(
                    "model_provider:orcarouter",
                    {
                        "ORCAROUTER_BASE_URL": "https://api.orcarouter.ai/v1",
                        "ORCAROUTER_API_KEY": "key",
                    },
                )["ANTHROPIC_BASE_URL"],
                "https://api.orcarouter.ai",
            )
            self.assertEqual(
                provider_env.freeze_provider_env(
                    "model_provider:orcarouter", {"ORCAROUTER_API_KEY": "k"}
                )["ORCAROUTER_API_KEY"],
                "k",
            )
            self.assertEqual(
                provider_env.freeze_provider_env(
                    "model_provider:anthropic_messages", {"ANTHROPIC_API_KEY": "a"}
                )["ANTHROPIC_API_KEY"],
                "a",
            )
            self.assertEqual(
                provider_env.freeze_provider_env(
                    "model_provider:openai_compatible", {"OPENAI_API_KEY": "o"}
                )["OPENAI_API_KEY"],
                "o",
            )
            deepseek_env = provider_env.freeze_provider_env(
                "model_provider:deepseek_alias", {"DEEPSEEK_API_KEY": "d"}
            )
            self.assertEqual(deepseek_env["DEEPSEEK_API_KEY"], "d")
            self.assertEqual(deepseek_env["ANTHROPIC_AUTH_TOKEN"], "d")
            self.assertEqual(
                deepseek_env["ANTHROPIC_BASE_URL"], "https://api.deepseek.com/anthropic"
            )
            self.assertEqual(deepseek_env["ANTHROPIC_MODEL"], "deepseek-v4-pro[1m]")
            self.assertEqual(deepseek_env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "deepseek-v4-flash")
            self.assertEqual(deepseek_env["CLAUDE_CODE_SUBAGENT_MODEL"], "deepseek-v4-flash")
            self.assertEqual(deepseek_env["CLAUDE_CODE_EFFORT_LEVEL"], "max")
            self.assertEqual(
                provider_env.freeze_provider_env("model_provider:fake_provider", {})[
                    "PRAXIST_MODEL_PROVIDER_REF"
                ],
                "model_provider:fake_provider",
            )
            with self.assertRaises(ValueError):
                provider_env.freeze_provider_env("model_provider:unknown", {})

            env = {"PRAXIST_DATASETS_DIR": str(data_root), "PYTHONPATH": "/x"}
            runtime_env = startup._task_runtime_env(
                task_project_path=workspace / "task",
                workspace=workspace,
                task_id="generic_task",
                env=env,
            )
            self.assertEqual(
                runtime_env["PRAXIST_DATA_DIR"], str((data_root / "generic_task").resolve())
            )
            self.assertTrue(runtime_env["PYTHONPATH"].startswith(str(workspace.resolve())))
            self.assertEqual(
                startup._prepend_path_env(str(workspace), runtime_env["PYTHONPATH"]),
                runtime_env["PYTHONPATH"],
            )
            self.assertEqual(
                startup._resolved_task_data_dir(
                    task_id="generic_task",
                    workspace=workspace,
                    env={"PRAXIST_SAM_DATA_DIR": str(sam_data)},
                    data_env_aliases=["PRAXIST_SAM_DATA_DIR"],
                ),
                sam_data.resolve(),
            )
            task_root = workspace / "runtime_task"
            task_root.mkdir()
            evaluator = task_root / "evaluations" / "run.py"
            evaluator.parent.mkdir()
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            venv_python = task_root / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (venv_python.parent / "activate").write_text("", encoding="utf-8")
            runtime_env = startup._task_runtime_env(
                task_project_path=task_root,
                workspace=workspace,
                task_id="generic_task",
                env={"PATH": "/usr/bin", "PYTHONPATH": "/x"},
                task_descriptor={
                    "runtime_environment": {
                        "venv": ".venv",
                        "cwd": "evaluations",
                        "env": {"TASK_MODE": "dogfood"},
                        "data_env_aliases": ["TASK_DATA_ALIAS"],
                        "path_prepend": ["bin"],
                    }
                },
                evaluation_entrypoint="python run.py",
            )
            self.assertEqual(runtime_env["PRAXIST_TASK_VENV"], str((task_root / ".venv").resolve()))
            self.assertEqual(runtime_env["VIRTUAL_ENV"], str((task_root / ".venv").resolve()))
            self.assertEqual(runtime_env["PRAXIST_TASK_PYTHON"], str(venv_python.resolve()))
            self.assertIn(str((task_root / "bin").resolve()), runtime_env["PATH"])
            self.assertIn(
                str((task_root / ".venv").resolve()),
                runtime_env["PRAXIST_TASK_WRITABLE_ROOTS"].split(os.pathsep),
            )
            self.assertEqual(runtime_env["TASK_MODE"], "dogfood")
            self.assertEqual(
                runtime_env["PRAXIST_TASK_RUNTIME_ENV_KEYS"], "TASK_DATA_ALIAS,TASK_MODE"
            )
            self.assertEqual(runtime_env["TASK_DATA_ALIAS"], runtime_env["PRAXIST_DATA_DIR"])
            self.assertIn("source ", runtime_env["PRAXIST_TASK_SHELL_PREFIX"])
            self.assertEqual(
                runtime_env["PRAXIST_EVALUATION_ENTRYPOINT_PATH"],
                str(evaluator.resolve()),
            )

            real_venv = workspace / "real_venv"
            real_python = real_venv / "bin" / "python"
            real_python.parent.mkdir(parents=True)
            real_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            linked_venv = task_root / "linked_venv"
            linked_venv.symlink_to(real_venv, target_is_directory=True)
            runtime_env = startup._task_runtime_env(
                task_project_path=task_root,
                workspace=workspace,
                task_id="generic_task",
                env={"PATH": "/usr/bin"},
                task_descriptor={
                    "runtime_environment": {
                        "venv": "linked_venv",
                        "python": "linked_venv/bin/python",
                    }
                },
            )
            self.assertEqual(
                Path(runtime_env["PRAXIST_TASK_VENV"]).resolve(strict=False),
                (task_root / "linked_venv").resolve(strict=False),
            )
            self.assertEqual(
                Path(runtime_env["VIRTUAL_ENV"]).resolve(strict=False),
                (task_root / "linked_venv").resolve(strict=False),
            )
            self.assertEqual(
                Path(runtime_env["PRAXIST_TASK_PYTHON"]).resolve(strict=False),
                (task_root / "linked_venv" / "bin" / "python").resolve(strict=False),
            )

            self.assertEqual(
                startup._task_execution_cwd(
                    task_project_path=task_root,
                    run_dir=workspace / "run",
                    task_descriptor={"runtime_environment": {"cwd": "task_project"}},
                ),
                task_root.resolve(),
            )
            run_cwd = startup._task_execution_cwd(
                task_project_path=task_root,
                run_dir=workspace / "run_cwd",
                task_descriptor={"runtime_environment": {"cwd": "run_dir"}},
            )
            self.assertEqual(run_cwd, (workspace / "run_cwd").resolve())
            with self.assertRaises(FileNotFoundError):
                startup._task_execution_cwd(
                    task_project_path=task_root,
                    run_dir=workspace / "run",
                    task_descriptor={"runtime_environment": {"cwd": "missing"}},
                )
            with self.assertRaises(ValueError):
                startup._task_runtime_env(
                    task_project_path=task_root,
                    workspace=workspace,
                    task_id="generic_task",
                    env={},
                    task_descriptor={"runtime_environment": {"env": {"1BAD": "x"}}},
                )
            with self.assertRaises(ValueError):
                startup._task_runtime_env(
                    task_project_path=task_root,
                    workspace=workspace,
                    task_id="generic_task",
                    env={},
                    task_descriptor={
                        "runtime_environment": {"env": {"TASK_TOKEN": "sk-" + "a" * 48}}
                    },
                )
            with self.assertRaises(FileNotFoundError):
                startup._task_runtime_env(
                    task_project_path=task_root,
                    workspace=workspace,
                    task_id="generic_task",
                    env={},
                    task_descriptor={"runtime_environment": {"venv": "missing_venv"}},
                )
            not_a_venv = task_root / "not_a_venv"
            not_a_venv.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "venv must be a directory"):
                startup._task_runtime_env(
                    task_project_path=task_root,
                    workspace=workspace,
                    task_id="generic_task",
                    env={},
                    task_descriptor={"runtime_environment": {"venv": "not_a_venv"}},
                )
            bootstrap_env = startup._task_runtime_env(
                task_project_path=task_root,
                workspace=workspace,
                task_id="generic_task",
                env={"PATH": "/usr/bin"},
                task_descriptor={
                    "runtime_environment": {
                        "venv": "missing_bootstrap_venv",
                        "require_paths": False,
                        "writable_roots": ["scratch"],
                    }
                },
            )
            self.assertEqual(
                bootstrap_env["PRAXIST_TASK_VENV"],
                str((task_root / "missing_bootstrap_venv").resolve()),
            )
            self.assertIn(
                str((task_root / "missing_bootstrap_venv").resolve()),
                bootstrap_env["PRAXIST_TASK_WRITABLE_ROOTS"].split(os.pathsep),
            )
            self.assertIn(
                str((task_root / "scratch").resolve()),
                bootstrap_env["PRAXIST_TASK_WRITABLE_ROOTS"].split(os.pathsep),
            )
            with self.assertRaises(FileNotFoundError):
                startup._task_runtime_env(
                    task_project_path=task_root,
                    workspace=workspace,
                    task_id="generic_task",
                    env={},
                    task_descriptor={"runtime_environment": {"python": "missing_python"}},
                )
            not_a_python = task_root / "not_a_python"
            not_a_python.mkdir()
            with self.assertRaisesRegex(ValueError, "python must be a file"):
                startup._task_runtime_env(
                    task_project_path=task_root,
                    workspace=workspace,
                    task_id="generic_task",
                    env={},
                    task_descriptor={"runtime_environment": {"python": "not_a_python"}},
                )
            runtime_env = startup._task_runtime_env(
                task_project_path=task_root,
                workspace=workspace,
                task_id="generic_task",
                env={"PATH": "/usr/bin"},
                task_descriptor={
                    "runtime_environment": {
                        "path_prepend": "bin",
                        "shell_prefix": "source /custom/env &&",
                    }
                },
            )
            self.assertEqual(runtime_env["PRAXIST_TASK_SHELL_PREFIX"], "source /custom/env &&")
            self.assertEqual(
                runtime_env["PRAXIST_RUNNER_PYTHON"],
                os.path.abspath(sys.executable),
            )
            self.assertIn(str((task_root / "bin").resolve()), runtime_env["PATH"])

            descriptor = {
                "praxist_plugins": {
                    "task_ref": "task:demo",
                    "workflow": {"stage": startup.RESEARCH_LOOP_STAGE_REF},
                    "optional_workflow_stages": {
                        "ideation": {"ref": "workflow_stage:ideation_stub", "enabled": False},
                        "review": "workflow_stage:reviewer_stub",
                    },
                    "panel": {
                        "roles": ["task_role:peer", "task_role:builder"],
                        "optional_roles": {
                            "literature": {
                                "role": "task_role:literature",
                                "tool": "tool_server:literature",
                                "enabled": False,
                            },
                            "bad": "ignored",
                        },
                    },
                }
            }
            startup._validate_research_loop_task_eligibility("task:demo", descriptor)
            with self.assertRaises(ValueError):
                startup._validate_research_loop_task_eligibility(
                    "task:demo",
                    {
                        "praxist_plugins": {
                            "task_ref": "task:other",
                            "workflow": {"stage": startup.RESEARCH_LOOP_STAGE_REF},
                        }
                    },
                )
            with self.assertRaises(ValueError):
                startup._validate_research_loop_task_eligibility(
                    "task:demo",
                    {"praxist_plugins": {"workflow": {"stage": "workflow_stage:other"}}},
                )
            self.assertEqual(
                startup._role_refs_from_task_descriptor(descriptor),
                ["task_role:peer", "task_role:builder"],
            )
            disabled = startup._disabled_optional_from_descriptor(descriptor)
            self.assertEqual(len(disabled), 3)
            enabled_descriptor = {
                "praxist_plugins": {
                    "workflow": {"stage": startup.RESEARCH_LOOP_STAGE_REF},
                    "optional_workflow_stages": {
                        "ideation": {"ref": "workflow_stage:ideation_stub", "enabled": True}
                    },
                    "panel": {
                        "optional_roles": {"lit": {"role": "task_role:lit", "enabled": True}}
                    },
                }
            }
            with self.assertRaises(ValueError):
                startup._reject_enabled_optional_stubs(enabled_descriptor)
            self.assertEqual(
                startup._disabled_optional_from_descriptor(enabled_descriptor),
                [],
            )
            self.assertEqual(
                startup._optional_role_entries(
                    {"praxist_plugins": {"panel": {"optional_roles": {"x": {}}}}}
                ),
                [],
            )
            default_tool_refs = {
                "tool_server:evaluation_tools",
                "tool_server:frontier_tools",
                "tool_server:finding_graph_query",
                "tool_server:memory_tools",
                "tool_server:prior_work_tools",
                "tool_server:run_report",
                "tool_server:literature_lookup",
            }
            startup_refs = startup._plugin_refs_from_task_descriptor(
                {"praxist_plugins": {"tools": []}}
            )
            self.assertTrue(default_tool_refs.issubset(set(startup_refs)))
            self.assertEqual(
                startup._plugin_refs_from_task_descriptor(
                    {"praxist_plugins": {"tools": ["tool_server:frontier_tools"]}}
                ).count("tool_server:frontier_tools"),
                1,
            )
            self.assertNotIn(
                "tool_server:evaluation_tools",
                startup._plugin_refs_from_task_descriptor(
                    {"praxist_plugins": {"tools": ["tool_server:frontier_tools"]}}
                ),
            )

            self.assertEqual(startup._dedupe_refs(["a", "b", "a"]), ["a", "b"])
            run_dir = workspace / "run"
            run_dir.mkdir()
            (run_dir / ".gitkeep").write_text("", encoding="utf-8")
            (run_dir / "empty").mkdir()
            startup._ensure_fresh_run_dir(run_dir)
            (run_dir / "run.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                startup._ensure_fresh_run_dir(run_dir)
            blocked = workspace / "blocked_run"
            (blocked / "nonempty").mkdir(parents=True)
            (blocked / "nonempty" / "x").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                startup._ensure_fresh_run_dir(blocked)

            descriptor_path = workspace / "task.yaml"
            descriptor_path.write_text("[1]", encoding="utf-8")
            with self.assertRaises(ValueError):
                startup._read_task_descriptor(descriptor_path)

            selected_path = startup._selected_plugin_path(
                {
                    "selected": [
                        {"metadata": {"kind": "tool_server", "name": "x"}, "path": "/tmp/x"}
                    ]
                },
                SimpleNamespace(kind="tool_server", name="x", as_string=lambda: "tool_server:x"),
            )
            self.assertEqual(selected_path, Path("/tmp/x").resolve())
            with self.assertRaises(ValueError):
                startup._selected_plugin_path(
                    {"selected": []},
                    SimpleNamespace(
                        kind="tool_server", name="x", as_string=lambda: "tool_server:x"
                    ),
                )

    def test_runtime_provider_contracts_use_manifest_metadata(self) -> None:
        from praxist.core.registry import PluginLoader, PluginRoots
        from praxist.plugins.workflow_stages.research_loop import startup

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin_root = root / "plugins"
            _write_plugin(
                plugin_root,
                kind_dir="agent_runtimes",
                name="runtime",
                kind="agent_runtime",
                contract_key="runtime",
                contract={
                    "cache_strategy": "runtime_auto_cache",
                    "compatible_model_providers": ["model_provider:provider"],
                    "usage_reporting": "unknown",
                    "event_schema": "v1",
                },
            )
            _write_plugin(
                plugin_root,
                kind_dir="model_providers",
                name="provider",
                kind="model_provider",
                contract_key="provider",
                contract={"cache_strategy": "provider_explicit_cache", "usage_reporting": "exact"},
            )
            loader = PluginLoader(PluginRoots(bundled=[plugin_root], project=[], user=[]))
            manifest = loader.resolve(
                ["agent_runtime:runtime", "model_provider:provider"],
                loader.discover(),
                enforce_bundled_execution=True,
            )
            registry = loader.load(manifest)
            self.assertEqual(
                startup._cache_strategy_for_runtime_provider(
                    "agent_runtime:runtime",
                    "model_provider:provider",
                    registry,
                ),
                ("runtime_auto_cache", "runtime_auto_cache", None),
            )
            startup._validate_runtime_provider_compatibility(
                "agent_runtime:runtime",
                "model_provider:provider",
                registry,
            )
            with (
                patch.object(
                    startup, "_runtime_contract", return_value={"cache_strategy": "disabled"}
                ),
                patch.object(
                    startup,
                    "_provider_contract",
                    return_value={"cache_strategy": "provider_explicit_cache"},
                ),
            ):
                self.assertEqual(
                    startup._cache_strategy_for_runtime_provider("r", "p"),
                    ("disabled", None, None),
                )
            with (
                patch.object(startup, "_runtime_contract", return_value={}),
                patch.object(
                    startup,
                    "_provider_contract",
                    return_value={
                        "cache_strategy": "provider_explicit_cache",
                        "explicit_cache_strategy": "ephemeral",
                    },
                ),
            ):
                self.assertEqual(
                    startup._cache_strategy_for_runtime_provider("r", "p"),
                    ("provider_explicit_cache", None, "ephemeral"),
                )
            with (
                patch.object(startup, "_runtime_contract", return_value={}),
                patch.object(startup, "_provider_contract", return_value={}),
            ):
                self.assertEqual(
                    startup._cache_strategy_for_runtime_provider("r", "p"),
                    ("provider_default", None, None),
                )
            with self.assertRaises(ValueError):
                startup._validate_runtime_provider_compatibility(
                    "agent_runtime:runtime",
                    "model_provider:other",
                    registry,
                )
            snapshot = startup._runtime_provider_conformance_snapshot(
                "agent_runtime:runtime",
                "model_provider:provider",
                SimpleNamespace(
                    mode="runtime_auto_cache",
                    runtime_cache_strategy="runtime_auto_cache",
                    provider_cache_strategy=None,
                ),
                registry,
            )
            self.assertEqual(snapshot["runtime_usage_reporting"], "unknown")
            self.assertEqual(snapshot["provider_usage_reporting"], "exact")

            bundled_loader = PluginLoader(PluginRoots.defaults(Path(__file__).resolve().parents[2]))
            bundled_manifest = bundled_loader.resolve(
                ["agent_runtime:claude_sdk", "model_provider:deepseek_alias"],
                bundled_loader.discover(),
                enforce_bundled_execution=True,
            )
            bundled_registry = bundled_loader.load(bundled_manifest)
            startup._validate_runtime_provider_compatibility(
                "agent_runtime:claude_sdk",
                "model_provider:deepseek_alias",
                bundled_registry,
            )

            class RoleSkill:
                def __init__(self, role_kind: str, model_profile: str | None) -> None:
                    self.role_kind = role_kind
                    self.default_model_profile_ref = model_profile

            descriptor = {
                "praxist_plugins": {
                    "panel": {
                        "roles": [
                            {"role_ref": "task_role:builder"},
                            {"role": "task_role:peer"},
                        ],
                        "optional_roles": {"lit": {"role": "task_role:lit", "enabled": True}},
                    }
                }
            }
            skills = {
                "task_role:builder": RoleSkill("panel", "expensive"),
                "task_role:peer": RoleSkill("peer", "cheap"),
                "task_role:lit": RoleSkill("panel", None),
            }
            from praxist.plugins.workflow_stages.research_loop import peer_roles

            with patch.object(
                peer_roles, "load_role_skill", side_effect=lambda ref, **_kw: skills[ref]
            ):
                self.assertEqual(
                    startup._model_profile_defaults_from_task_descriptor(
                        descriptor, registry, root
                    ),
                    {
                        "research_loop": "cheap_peer",
                        "task_role:builder": "expensive",
                        "task_role:peer": "cheap",
                    },
                )
                self.assertEqual(
                    startup._peer_role_ref_from_task_descriptor(descriptor, registry, root),
                    "task_role:peer",
                )
                self.assertEqual(
                    startup._peer_role_refs_from_task_descriptor(descriptor, registry, root),
                    ("task_role:peer",),
                )
            with patch.object(peer_roles, "load_role_skill", side_effect=RuntimeError("bad")):
                self.assertEqual(
                    startup._model_profile_defaults_from_task_descriptor(
                        descriptor, registry, root
                    ),
                    {"research_loop": "cheap_peer"},
                )
                with self.assertRaisesRegex(ValueError, "cannot load declared task role"):
                    startup._peer_role_ref_from_task_descriptor(descriptor, registry, root)
            bad_plugin = root / "bad_plugin"
            bad_plugin.mkdir()
            (bad_plugin / "plugin.yaml").write_text("[]", encoding="utf-8")
            self.assertEqual(startup._manifest_contract("x:y", "runtime", None), {})
            self.assertEqual(
                startup._manifest_contract(
                    "x:y",
                    "runtime",
                    SimpleNamespace(
                        descriptor_for_ref=lambda _ref: SimpleNamespace(path=bad_plugin)
                    ),
                ),
                {},
            )
            self.assertEqual(
                startup._manifest_contract(
                    "x:y",
                    "runtime",
                    SimpleNamespace(
                        descriptor_for_ref=lambda _ref: (_ for _ in ()).throw(RuntimeError("bad"))
                    ),
                ),
                {},
            )


if __name__ == "__main__":
    unittest.main()
