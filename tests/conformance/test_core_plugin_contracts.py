from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from praxist.core.budget import policy_for_ref
from praxist.core.cache import build_cache_policy, frozen_prefix_hash
from praxist.core.credentials import CredentialFailoverManager, CredentialResolver
from praxist.core.modeling import default_model_profile, provider_for_ref
from praxist.core.protocol import AgentRunRequest, BudgetRequest, EnvPolicy, ToolPermissionSet
from praxist.core.registry import (
    PluginLoader,
    PluginRegistryBuilder,
    PluginRoots,
    selected_plugin_from_dict,
)
from praxist.core.replay import verify_run
from praxist.core.runtimes import event_types_for_conformance, runtime_for_ref
from praxist.core.storage import read_jsonl
from praxist.testing.fake_workflow_fixture import run_fake_workflow_fixture


class CorePluginContractsTest(unittest.TestCase):
    def test_plugin_resolution_prefers_project_over_user_over_bundled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundled = root / "bundled"
            user = root / "user"
            project = root / "project"
            for source_root, description in (
                (bundled, "bundled runtime"),
                (user, "user runtime"),
                (project, "project runtime"),
            ):
                self._write_manifest(
                    source_root, "agent_runtimes", "fake_runtime", "agent_runtime", description
                )

            loader = PluginLoader(PluginRoots(bundled=[bundled], user=[user], project=[project]))
            report = loader.discover()
            self.assertEqual(len(report.candidates), 3)
            manifest = loader.resolve(
                ["agent_runtime:fake_runtime"],
                report,
                run_id="run_test",
                root_task_ref="task:fake_panel",
            )
            selected = manifest["selected"][0]
            self.assertEqual(selected["source"], "project")
            self.assertEqual(len(manifest["shadowed"]), 2)

            registry = loader.load(manifest)
            self.assertEqual(len(registry.list("agent_runtime")), 1)

            builder = PluginRegistryBuilder()
            builder.add(selected_plugin_from_dict(selected))
            builder.freeze()
            with self.assertRaises(RuntimeError):
                builder.add(selected_plugin_from_dict(selected))

    def test_execution_resolution_filters_project_and_user_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundled = root / "bundled"
            user = root / "user"
            project = root / "project"
            for source_root, description in (
                (bundled, "bundled runtime"),
                (user, "user runtime"),
                (project, "project runtime"),
            ):
                self._write_manifest(
                    source_root, "agent_runtimes", "fake_runtime", "agent_runtime", description
                )

            loader = PluginLoader(PluginRoots(bundled=[bundled], user=[user], project=[project]))
            manifest = loader.resolve(
                ["agent_runtime:fake_runtime"],
                run_id="run_test",
                root_task_ref="task:fake_panel",
                enforce_bundled_execution=True,
            )
            self.assertEqual(manifest["execution_source_policy"], "bundled_only")
            self.assertEqual(manifest["selected"][0]["source"], "bundled")
            self.assertEqual(len(manifest["shadowed"]), 2)

    def test_runtime_conformance_event_shape_is_shared(self) -> None:
        # codex_sdk owns a real app-server client; its typed event stream is
        # covered by focused adapter tests, while this check stays offline.
        request = self._agent_request("agent_runtime:fake_runtime")
        event_types = event_types_for_conformance(
            ["agent_runtime:fake_runtime", "agent_runtime:claude_sdk"],
            request,
        )
        self.assertEqual(
            event_types["agent_runtime:fake_runtime"],
            ["agent_run_started", "assistant_text", "final_result"],
        )
        self.assertEqual(
            event_types["agent_runtime:claude_sdk"], event_types["agent_runtime:fake_runtime"]
        )

        from praxist.core.runtimes import runtime_for_ref

        first = runtime_for_ref("agent_runtime:fake_runtime").execute_sync(request).to_dict()
        second = runtime_for_ref("agent_runtime:fake_runtime").execute_sync(request).to_dict()
        self.assertEqual(first, second)

    def test_runtime_factory_requires_selected_registry_descriptor_when_provided(self) -> None:
        loader = PluginLoader(PluginRoots.defaults(Path.cwd()))
        manifest = loader.resolve(
            ["agent_runtime:fake_runtime"], run_id="run_test", root_task_ref="task:fake_panel"
        )
        registry = loader.load(manifest)
        runtime = runtime_for_ref("agent_runtime:fake_runtime", registry=registry)
        self.assertEqual(runtime.runtime_ref, "agent_runtime:fake_runtime")
        with self.assertRaisesRegex(KeyError, "claude_sdk"):
            runtime_for_ref("agent_runtime:claude_sdk", registry=registry)

    def test_model_provider_normalization_and_cache_hash(self) -> None:
        for provider_ref in (
            "model_provider:fake_provider",
            "model_provider:anthropic_messages",
            "model_provider:openai_compatible",
            "model_provider:openrouter",
            "model_provider:deepseek_alias",
            "model_provider:groq_alias",
            "model_provider:mistral_alias",
            "model_provider:xai_alias",
        ):
            profile = default_model_profile(provider_ref)
            call = provider_for_ref(provider_ref).build_call(profile, credential_ref=None)
            self.assertEqual(call.provider_ref, provider_ref)
            self.assertEqual(call.model, profile.model)

        self.assertEqual(
            provider_for_ref("model_provider:openrouter").classify_error({"status": 429}),
            "rate_limited",
        )
        self.assertEqual(
            provider_for_ref("model_provider:deepseek_alias").classify_error(
                {"message": "quota exhausted"}
            ),
            "quota_exhausted",
        )
        self.assertEqual(
            provider_for_ref("model_provider:anthropic_messages").classify_error({"status": 401}),
            "auth_error",
        )
        self.assertEqual(
            provider_for_ref("model_provider:openai_compatible").classify_error(
                {"code": "timeout"}
            ),
            "timeout",
        )

        first = frozen_prefix_hash({"b": "line\r\n", "a": ["x"]})
        second = frozen_prefix_hash({"a": ["x"], "b": "line\n"})
        self.assertEqual(first, second)
        self.assertNotEqual(first, frozen_prefix_hash({"a": ["changed"], "b": "line\n"}))

    def test_credential_robust_mode_fails_over_to_next_key(self) -> None:
        credential_set = CredentialResolver({}).discover(profile="fake_multi_key")
        manager = CredentialFailoverManager(credential_set)
        first = manager.select(
            scope="model_provider",
            provider="fake_provider",
            target_ref="model_provider:fake_provider",
        )
        self.assertIsNotNone(first)
        self.assertTrue(first.key_id.endswith(":A"))

        second = manager.record_failure(first, "quota_exhausted")
        self.assertIsNotNone(second)
        self.assertTrue(second.key_id.endswith(":B"))
        snapshot = json.dumps(manager.snapshot())
        self.assertIn("quota_exhausted", snapshot)
        self.assertNotIn("sk-", snapshot)

    def test_budget_policy_conformance_cases(self) -> None:
        policy = policy_for_ref("budget_policy:fake_tiered")
        cheap = policy.decide(
            BudgetRequest(
                request_id="cheap",
                requester_id="peer",
                experiment_id="cheap_probe",
                model_profile_ref="cheap_peer",
                requested={"tokens": 1000, "wall_clock_seconds": 30},
                expected_value={"confidence": "weak"},
                evidence_refs=[],
                cheaper_alternatives=[],
                abort_conditions=[],
            )
        )
        self.assertEqual(cheap.decision, "grant")
        self.assertIsNotNone(cheap.grant)

        strong = policy.decide(
            BudgetRequest(
                request_id="strong",
                requester_id="peer",
                experiment_id="expensive",
                model_profile_ref="strong_reasoner",
                requested={"tokens": 80_000, "wall_clock_seconds": 3600, "gpu_hours": 4},
                expected_value={"confidence": "strong"},
                evidence_refs=["finding:accepted"],
                cheaper_alternatives=["small_probe"],
                abort_conditions=["no_signal"],
            )
        )
        self.assertEqual(strong.decision, "downscope")
        self.assertEqual(strong.grant.approved["gpu_hours"], 2.0)

        weak = policy.decide(
            BudgetRequest(
                request_id="weak",
                requester_id="peer",
                experiment_id="expensive_weak",
                model_profile_ref="strong_reasoner",
                requested={"tokens": 80_000},
                expected_value={"confidence": "weak"},
                evidence_refs=[],
                cheaper_alternatives=[],
                abort_conditions=[],
            )
        )
        self.assertEqual(weak.decision, "require_review")

        denied = policy.decide(
            BudgetRequest(
                request_id="deny",
                requester_id="peer",
                experiment_id="impossible",
                model_profile_ref=None,
                requested={"tokens": 1},
                expected_value={"impossible": True},
                evidence_refs=[],
                cheaper_alternatives=[],
                abort_conditions=[],
            )
        )
        self.assertEqual(denied.decision, "deny")

    def test_fake_panel_switches_runtime_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for runtime_ref in ("agent_runtime:fake_runtime",):
                run_dir = Path(tmp) / runtime_ref.split(":")[1]
                result = run_fake_workflow_fixture(
                    workspace=Path(tmp),
                    run_dir=run_dir,
                    runtime_ref=runtime_ref,
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    credential_profile="fake_multi_key",
                )
                report = verify_run(Path(result["run_dir"]))
                self.assertTrue(report["success"], report)

                resolution = json.loads(
                    (Path(result["run_dir"]) / "plugin_resolution.json").read_text()
                )
                selected_refs = {
                    item["metadata"]["kind"] + ":" + item["metadata"]["name"]
                    for item in resolution["selected"]
                }
                self.assertIn(runtime_ref, selected_refs)
                self.assertIn("model_provider:fake_provider", selected_refs)

                trajectory, errors = read_jsonl(Path(result["run_dir"]) / "trajectory.jsonl")
                self.assertEqual(errors, [])
                self.assertTrue(any(event["kind"] == "credential.failover" for event in trajectory))
                self.assertTrue(
                    any(event["kind"] == "workflow.stage_skipped" for event in trajectory)
                )

            incompatible_run_dir = Path(tmp) / "claude_sdk_fake_provider"
            with self.assertRaisesRegex(ValueError, "not compatible"):
                run_fake_workflow_fixture(
                    workspace=Path(tmp),
                    run_dir=incompatible_run_dir,
                    runtime_ref="agent_runtime:claude_sdk",
                    model_provider_ref="model_provider:fake_provider",
                    budget_policy_ref="budget_policy:fake_tiered",
                    credential_profile="fake_multi_key",
                )
            self.assertFalse(incompatible_run_dir.exists())

    def _agent_request(self, runtime_ref: str) -> AgentRunRequest:
        profile = default_model_profile("model_provider:fake_provider")
        model_call = provider_for_ref("model_provider:fake_provider").build_call(
            profile, credential_ref=None
        )
        cache_policy = build_cache_policy(frozen_prefix_parts={"task": "fake"})
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
            cache_policy=cache_policy,
            runtime_options={},
        )

    def _write_manifest(
        self, root: Path, dirname: str, name: str, kind: str, description: str
    ) -> None:
        plugin_dir = root / dirname / name
        plugin_dir.mkdir(parents=True)
        plugin_dir.joinpath("plugin.yaml").write_text(
            f"""schema_version: 1
name: {name}
kind: {kind}
version: 0.1.0
protocol_version: 1
stability: v1_stable
description: {description}
compatibility:
  praxist_core: ">=0.1.0,<1.0"
  python: ">=3.11"
dependencies: []
capabilities: []
code: []
assets: []
""",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
