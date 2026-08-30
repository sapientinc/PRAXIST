from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from praxist.core.cache import build_cache_policy
from praxist.core.modeling import default_model_profile, provider_for_ref
from praxist.core.protocol import AgentRunRequest, EnvPolicy, ToolPermissionSet
from praxist.core.registry import PluginLoader, PluginRoots
from praxist.core.runtimes import runtime_for_ref
from praxist.core.source_snapshot import SOURCE_PATTERNS


class PluginExecutableClosureTest(unittest.TestCase):
    def test_agent_runtime_code_lives_under_plugin_boundary(self) -> None:
        self.assertFalse((Path.cwd() / "auto_research").exists())
        self.assertIn("praxist/plugins/**/*.py", SOURCE_PATTERNS)
        self.assertNotIn("auto_research/agent_runtimes/**/*.py", SOURCE_PATTERNS)

    def test_research_loop_code_lives_under_workflow_stage_plugin_boundary(self) -> None:
        self.assertFalse((Path.cwd() / "auto_research").exists())
        plugin_dir = Path.cwd() / "praxist" / "plugins" / "workflow_stages" / "research_loop"
        manifest = yaml.safe_load(plugin_dir.joinpath("plugin.yaml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["entrypoint"], "stage:create_stage")
        self.assertIn("stage.py", manifest["code"])
        self.assertIn("startup.py", manifest["code"])
        self.assertIn("c5_materializer.py", manifest["code"])
        self.assertIn("backend/**/*.py", manifest["code"])
        self.assertTrue(plugin_dir.joinpath("stage.py").is_file())
        self.assertTrue(plugin_dir.joinpath("startup.py").is_file())
        self.assertTrue(plugin_dir.joinpath("c5_materializer.py").is_file())
        self.assertTrue(plugin_dir.joinpath("backend", "generation_loop.py").is_file())

    def test_fake_workflow_fixture_lives_under_testing_boundary(self) -> None:
        runner_path = Path.cwd() / "praxist" / "testing" / "fake_workflow_fixture.py"
        self.assertTrue(runner_path.is_file())
        self.assertFalse((Path.cwd() / "praxist" / "plugins" / "tasks").exists())
        self.assertFalse(Path.cwd().joinpath("auto_research").exists())

    def test_core_legacy_bridge_import_shims_are_removed(self) -> None:
        self.assertFalse(Path.cwd().joinpath("praxist", "core", "legacy_migration.py").exists())
        self.assertFalse(Path.cwd().joinpath("praxist", "core", "c5_materializer.py").exists())

    def test_core_dispatchers_do_not_embed_plugin_metadata_tables(self) -> None:
        forbidden = {
            "praxist/core/tool_servers.py": ["TOOL_SERVER_SPECS", "_HANDLERS"],
            "praxist/core/modeling.py": [
                "PROVIDER_API_FORMATS",
                "DEFAULT_MODELS",
                "PROVIDER_CAPABILITIES",
            ],
            "praxist/core/budget.py": ["POLICY_CAPABILITIES"],
            "praxist/core/runtimes.py": ["RUNTIME_CAPABILITIES"],
        }
        for rel_path, markers in forbidden.items():
            text = Path.cwd().joinpath(rel_path).read_text(encoding="utf-8")
            for marker in markers:
                self.assertNotIn(marker, text)

    def test_runtime_plugins_have_executable_entrypoints(self) -> None:
        loader = PluginLoader(PluginRoots.defaults(Path.cwd()))
        manifest = loader.resolve(
            ["agent_runtime:fake_runtime", "agent_runtime:claude_sdk", "agent_runtime:codex_sdk"],
            run_id="run_plugin_executable_closure",
            root_task_ref="test:runtime_closure",
            enforce_bundled_execution=True,
        )
        selected = {
            item["metadata"]["kind"] + ":" + item["metadata"]["name"]: item
            for item in manifest["selected"]
        }
        for runtime_ref in (
            "agent_runtime:fake_runtime",
            "agent_runtime:claude_sdk",
            "agent_runtime:codex_sdk",
        ):
            metadata = selected[runtime_ref]["metadata"]
            self.assertEqual(metadata["entrypoint"], "adapter:create_runtime")
            self.assertIn("adapter.py", metadata["code"])

    def test_runtime_factory_uses_registry_entrypoint_object(self) -> None:
        loader = PluginLoader(PluginRoots.defaults(Path.cwd()))
        manifest = loader.resolve(
            ["agent_runtime:fake_runtime"],
            run_id="run_plugin_executable_closure",
            root_task_ref="test:runtime_closure",
            enforce_bundled_execution=True,
        )
        registry = loader.load(manifest)
        registry_runtime = registry.require("agent_runtime", "fake_runtime")
        runtime = runtime_for_ref("agent_runtime:fake_runtime", registry=registry)
        self.assertIs(runtime, registry_runtime)

        result = runtime.execute_sync(self._agent_request("agent_runtime:fake_runtime"))
        self.assertTrue(result.success)
        self.assertEqual(
            [event.type for event in result.events],
            ["agent_run_started", "assistant_text", "final_result"],
        )

    def test_core_owned_plugins_are_executable_or_declared_contracts(self) -> None:
        loader = PluginLoader(PluginRoots.defaults(Path.cwd()))
        refs = [
            "workflow_stage:research_loop",
            "budget_policy:default_basic",
            "model_provider:openrouter",
            "model_provider:orcarouter",
            "tool_server:evaluation_tools",
            "tool_server:frontier_tools",
            "tool_server:finding_graph_query",
            "tool_server:memory_tools",
            "tool_server:prior_work_tools",
            "tool_server:run_report",
            "graph_maintainer:finding_graph_mvp",
        ]
        manifest = loader.resolve(
            refs,
            run_id="run_plugin_executable_closure",
            root_task_ref="test:plugin_closure",
            enforce_bundled_execution=True,
        )
        registry = loader.load(manifest)
        self.assertTrue(hasattr(registry.require("workflow_stage", "research_loop"), "execute"))
        self.assertTrue(hasattr(registry.require("budget_policy", "default_basic"), "decide"))
        self.assertTrue(hasattr(registry.require("model_provider", "openrouter"), "build_call"))
        self.assertTrue(hasattr(registry.require("model_provider", "orcarouter"), "build_call"))
        self.assertEqual(
            registry.require("tool_server", "memory_tools")["factory"],
            "praxist.plugins.tools.memory_tools.adapter:create_memory_tools_server",
        )
        self.assertTrue(
            hasattr(registry.require("graph_maintainer", "finding_graph_mvp"), "builder_class")
        )

    def test_tool_and_graph_plugins_own_their_executable_code(self) -> None:
        plugin_root = Path.cwd() / "praxist" / "plugins"
        tool_names = [
            "evaluation_tools",
            "frontier_tools",
            "finding_graph_query",
            "memory_tools",
            "prior_work_tools",
            "run_report",
        ]
        for name in tool_names:
            manifest = yaml.safe_load(
                plugin_root.joinpath("tools", name, "plugin.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["entrypoint"], "adapter:create_tool_plugin")
            self.assertIn("adapter.py", manifest["code"])
            self.assertTrue(plugin_root.joinpath("tools", name, "adapter.py").is_file())

        graph_manifest = yaml.safe_load(
            plugin_root.joinpath("graph_maintainers", "finding_graph_mvp", "plugin.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(graph_manifest["entrypoint"], "adapter:create_graph_maintainer")
        for rel_path in ("adapter.py", "atomic_io.py", "engine.py", "cli.py", "viz.py"):
            self.assertIn(rel_path, graph_manifest["code"])
            self.assertTrue(
                plugin_root.joinpath("graph_maintainers", "finding_graph_mvp", rel_path).is_file()
            )

    def test_openrouter_manifest_records_model_contract(self) -> None:
        manifest = yaml.safe_load(
            Path.cwd()
            .joinpath("praxist", "plugins", "model_providers", "openrouter", "plugin.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["provider"]["api_format"], "openrouter")
        self.assertEqual(manifest["provider"]["default_model"], "anthropic/claude-opus-4.7")
        self.assertEqual(
            manifest["provider"]["model_profiles"]["cheap_peer"], "anthropic/claude-sonnet-4.5"
        )

    def test_orcarouter_manifest_records_model_contract(self) -> None:
        manifest = yaml.safe_load(
            Path.cwd()
            .joinpath("praxist", "plugins", "model_providers", "orcarouter", "plugin.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["provider"]["api_format"], "orcarouter")
        self.assertEqual(manifest["provider"]["default_model"], "orcarouter/auto")

    def test_project_agent_runtime_can_execute_through_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin_dir = root / ".praxist" / "plugins" / "agent_runtimes" / "custom_runtime"
            plugin_dir.mkdir(parents=True)
            plugin_dir.joinpath("adapter.py").write_text(
                """from praxist.core.protocol import AgentRunResult


class CustomRuntime:
    runtime_ref = "agent_runtime:custom_runtime"

    def execute_sync(self, request):
        return AgentRunResult(
            success=True,
            events=[],
            text_output_refs=["custom-runtime"],
            tool_uses=[],
            error=None,
            failover_reason="none",
            credential_ref=request.credential_ref,
        )


def create_runtime():
    return CustomRuntime()
""",
                encoding="utf-8",
            )
            plugin_dir.joinpath("plugin.yaml").write_text(
                """schema_version: 1
name: custom_runtime
kind: agent_runtime
version: 0.1.0
protocol_version: 1
stability: v1_stable
description: Project runtime used by plugin executable closure tests.
compatibility:
  praxist_core: ">=0.1.0,<1.0"
  python: ">=3.11"
dependencies: []
capabilities:
  - runtime.custom_test
entrypoint: adapter:create_runtime
code:
  - adapter.py
assets: []
""",
                encoding="utf-8",
            )

            loader = PluginLoader(PluginRoots.defaults(root))
            manifest = loader.resolve(
                ["agent_runtime:custom_runtime"],
                run_id="run_custom_runtime",
                root_task_ref="test:custom_runtime",
            )
            self.assertEqual(manifest["selected"][0]["source"], "project")
            runtime = runtime_for_ref(
                "agent_runtime:custom_runtime", registry=loader.load(manifest)
            )
            result = runtime.execute_sync(self._agent_request("agent_runtime:custom_runtime"))
            self.assertEqual(result.text_output_refs, ["custom-runtime"])

    def _agent_request(self, runtime_ref: str) -> AgentRunRequest:
        profile = default_model_profile("model_provider:fake_provider")
        model_call = provider_for_ref("model_provider:fake_provider").build_call(
            profile, credential_ref=None
        )
        return AgentRunRequest(
            request_id="agent_test",
            run_id="run_test",
            stage_id="research_loop",
            role_ref="role:fake_peer",
            agent_runtime_ref=runtime_ref,
            prompt_ref={"artifact_id": "prompt", "logical_path": "prompts/test.md"},
            system_prompt_ref=None,
            cwd="/tmp",
            model_profile_ref=profile.profile_id,
            model_call=model_call,
            tool_permissions=ToolPermissionSet(),
            tool_servers=[],
            env_policy=EnvPolicy(),
            credential_ref=None,
            credential_mode="single",
            budget_grant_id="grant_test",
            artifact_scope="run",
            timeout_seconds=30,
            cache_policy=build_cache_policy(frozen_prefix_parts={"task": "fake"}),
            runtime_options={},
        )


if __name__ == "__main__":
    unittest.main()
