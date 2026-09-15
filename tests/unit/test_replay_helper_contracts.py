from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ReplayHelperContractsTest(unittest.TestCase):
    def test_replay_reference_binding_budget_and_summary_helpers(self) -> None:
        from praxist.core import replay
        from praxist.core.protocol import BudgetDecision, BudgetGrant, BudgetRequest

        errors: list[str] = []
        warnings: list[str] = []
        replay._record_lockable_issue(errors, warnings, "warn", locked=False)
        replay._record_lockable_issue(errors, warnings, "err", locked=True)
        self.assertEqual(warnings, ["warn"])
        self.assertEqual(errors, ["err"])

        self.assertIsNone(replay._budget_policy_ref({"canonical_args": "bad"}))
        self.assertEqual(
            replay._budget_policy_ref({"canonical_args": {"budget_policy": "budget_policy:x"}}),
            "budget_policy:x",
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            errors = []
            self.assertEqual(
                replay._resolve_run_relative_path(run_dir, "artifacts/a.txt", errors, "payload"),
                (run_dir / "artifacts/a.txt").resolve(),
            )
            self.assertIsNone(
                replay._resolve_run_relative_path(run_dir, "../bad", errors, "payload")
            )
            self.assertTrue(any("run-relative" in error for error in errors))

        errors = []
        replay._verify_required_plugin_refs(
            {"workflow_stage:research_loop"},
            {"task_ref": "task:external", "workflow_ref": "workflow_stage:research_loop"},
            {
                "canonical_args": {
                    "task": "task:external",
                    "runtime": "agent_runtime:missing",
                    "model_provider": "model_provider:missing",
                }
            },
            [
                {
                    "kind": "plugin.resolution_started",
                    "payload": {"requested": ["budget_policy:missing"]},
                },
                {"kind": "plugins.resolved", "payload": {"selected": ["task:external"]}},
            ],
            errors,
        )
        self.assertTrue(any("agent_runtime:missing" in error for error in errors))
        self.assertTrue(any("budget_policy:missing" in error for error in errors))

        errors = []
        replay._verify_ref_binding("", None, set(), "label", errors)
        replay._verify_ref_binding(
            "model_provider:other", "model_provider:x", {"model_provider:x"}, "label", errors
        )
        self.assertTrue(any("missing ref" in error for error in errors))
        self.assertTrue(any("not selected" in error for error in errors))
        self.assertTrue(any("mismatch" in error for error in errors))

        self.assertEqual(
            replay._expected_cache_contract({"cache_strategy": "disabled"}, {}),
            ("disabled", None, None),
        )
        self.assertEqual(
            replay._expected_cache_contract(
                {"cache_strategy": "runtime_auto_cache"},
                {"cache_strategy": "provider_explicit_cache"},
            ),
            ("runtime_auto_cache", "runtime_auto_cache", None),
        )
        self.assertEqual(
            replay._expected_cache_contract(
                {},
                {
                    "cache_strategy": "provider_explicit_cache",
                    "explicit_cache_strategy": "anthropic_cache_control",
                },
            ),
            ("provider_explicit_cache", None, "anthropic_cache_control"),
        )
        self.assertEqual(replay._expected_cache_contract({}, {}), ("provider_default", None, None))

        errors = []
        self.assertEqual(
            replay._credential_key_ids_for_provider(None, "model_provider:openrouter", errors),
            set(),
        )
        self.assertTrue(any("credentials_redacted missing" in error for error in errors))
        errors = []
        ids = replay._credential_key_ids_for_provider(
            {
                "credential_profiles": [
                    {"scope": "tool", "provider": "openrouter", "key_id": "tool"},
                    {
                        "scope": "model_provider",
                        "provider": "openrouter",
                        "target_ref": "model_provider:openrouter",
                        "key_id": "key1",
                    },
                    {
                        "scope": "model_provider",
                        "provider": "openrouter",
                        "status": "cooldown",
                        "key_id": "key2",
                    },
                ]
            },
            "model_provider:openrouter",
            errors,
        )
        self.assertEqual(ids, {"key1"})
        errors = []
        replay._verify_model_call_credential(
            {}, "model_provider:openrouter", {"key1"}, "call", errors
        )
        replay._verify_model_call_credential(
            {"credential_ref": {"key_id": "wrong", "target_ref": "model_provider:wrong"}},
            "model_provider:openrouter",
            {"key1"},
            "call",
            errors,
        )
        self.assertTrue(any("missing credential_ref" in error for error in errors))
        self.assertTrue(any("not selected" in error for error in errors))
        self.assertTrue(any("target mismatch" in error for error in errors))

        errors = []
        selected = [
            {
                "metadata": {
                    "kind": "agent_runtime",
                    "name": "r",
                    "dependencies": [{"kind": "model_provider", "name": "p"}],
                }
            },
            {"metadata": {"kind": "model_provider", "name": "p", "dependencies": "bad"}},
            "bad",
        ]
        replay._verify_plugin_dependency_closure(
            selected,
            [{"from": "agent_runtime:r", "to": "model_provider:missing"}],
            {"agent_runtime:r", "model_provider:p"},
            errors,
        )
        self.assertTrue(any("unselected to ref" in error for error in errors))
        self.assertTrue(any("missing dependency edge" in error for error in errors))
        self.assertTrue(any("dependencies is not a list" in error for error in errors))
        self.assertTrue(any("selected item missing metadata" in error for error in errors))

        errors = []
        request = replay._budget_request_from_record(
            {
                "kind": "request",
                "record_id": "r1",
                "request_record": {
                    "request_id": "req",
                    "requester_id": "peer",
                    "experiment_id": "exp",
                    "requested": {"tokens": 1},
                    "expected_value": {"confidence": "low"},
                    "evidence_refs": ["f1"],
                },
            },
            errors,
        )
        self.assertIsInstance(request, BudgetRequest)
        self.assertEqual(errors, [])
        self.assertIsNone(
            replay._budget_request_from_record(
                {"kind": "request", "record_id": "bad", "request_record": {"requested": []}},
                errors,
            )
        )
        self.assertTrue(any("malformed request_record" in error for error in errors))
        replay._verify_budget_decision_against_policy(
            {"record_id": "decision"}, request, None, errors
        )
        self.assertTrue(any("without budget_policy ref" in error for error in errors))

        self.assertTrue(replay._summary_indicates_execution({"generations_completed": "1"}))
        self.assertTrue(
            replay._summary_indicates_execution(
                {"legacy_generation_loop_summary": {"total_duration_seconds": 0}}
            )
        )
        self.assertFalse(
            replay._summary_indicates_execution(
                {"legacy_generation_loop_summary": {"exit_condition": "resolve_only"}}
            )
        )
        self.assertTrue(
            replay._stage_payload_indicates_execution({"result": {"total_duration_seconds": 0}})
        )
        self.assertFalse(
            replay._stage_payload_indicates_execution({"exit_condition": "resolve_only"})
        )
        self.assertFalse(replay._positive_number("bad"))

        errors = []
        replay._check_count("bad", 1, "count", errors)
        replay._check_count(2, 1, "count", errors)
        replay._verify_budget_amounts(
            {"tokens": "bad", "wall_clock_seconds": math.inf, "unknown": 1},
            "grant",
            errors,
        )
        self.assertTrue(any("not an integer" in error for error in errors))
        self.assertTrue(any("mismatch" in error for error in errors))
        self.assertTrue(any("non-numeric" in error for error in errors))
        self.assertTrue(any("invalid amount" in error for error in errors))
        self.assertTrue(any("unsupported unit" in error for error in errors))

        def request_record(request_id: str, requested: dict[str, object]) -> dict[str, object]:
            return {
                "request_id": request_id,
                "requester_id": "peer",
                "experiment_id": f"exp_{request_id}",
                "requested": requested,
                "expected_value": {"confidence": "weak"},
                "evidence_refs": [],
            }

        class FakeBudgetPolicy:
            def decide(self, req: BudgetRequest) -> BudgetDecision:
                return BudgetDecision(
                    decision="grant",
                    reason_codes=["test_policy"],
                    grant=BudgetGrant(
                        grant_id=f"grant_{req.request_id}",
                        approved={"tokens": 1},
                        conditions=[],
                        expires_at_generation=None,
                    ),
                )

        budget_records = [
            {
                "kind": "request",
                "record_id": "r1",
                "requested_budget": {"tokens": 2},
                "request_record": request_record("req1", {"tokens": 2}),
            },
            {
                "kind": "request",
                "record_id": "r1_dup",
                "requested_budget": {"tokens": 2},
                "request_record": request_record("req1", {"tokens": 2}),
            },
            {
                "kind": "decision",
                "record_id": "d1",
                "grant_id": "g1",
                "decision": "deny",
                "requested_budget": {"tokens": 2},
                "granted_budget": {"tokens": 2, "wall_clock_seconds": 2},
                "request_record": request_record("req1", {"tokens": 2}),
                "decision_record": {"decision": "deny"},
            },
            {
                "kind": "decision",
                "record_id": "d1_dup_grant",
                "grant_id": "g1",
                "requested_budget": {"tokens": 2},
                "granted_budget": {"tokens": 2},
                "request_record": request_record("req1", {"tokens": 2}),
                "decision_record": {"decision": "grant"},
            },
            {
                "kind": "decision",
                "record_id": "d_no_prior",
                "grant_id": "g2",
                "requested_budget": {"tokens": 1},
                "granted_budget": {"tokens": "bad"},
                "request_record": request_record("req2", {"tokens": 1}),
                "decision_record": {"decision": "grant"},
            },
            {
                "kind": "decision",
                "record_id": "d_non_dict_grant",
                "grant_id": "g_bad",
                "requested_budget": {"tokens": 1},
                "granted_budget": ["bad"],
                "request_record": request_record("req3", {"tokens": 1}),
            },
            {"kind": "usage_unknown", "record_id": "uu_missing", "grant_id": ""},
            {"kind": "usage_unknown", "record_id": "uu_unknown", "grant_id": "missing"},
            {
                "kind": "usage_unknown",
                "record_id": "uu_bad_units",
                "grant_id": "g1",
                "unknown_units": "bad",
            },
            {
                "kind": "usage_unknown",
                "record_id": "uu1",
                "grant_id": "g1",
                "unknown_units": ["tokens", "gpu_hours"],
            },
            {"kind": "usage", "record_id": "u_missing", "grant_id": ""},
            {"kind": "usage", "record_id": "u_unknown", "grant_id": "missing"},
            {
                "kind": "usage",
                "record_id": "u1",
                "grant_id": "g1",
                "actual_usage": {
                    "tokens": 3,
                    "wall_clock_seconds": "oops",
                    "gpu_hours": 1,
                    "cost_usd": 0.25,
                },
            },
            {
                "kind": "usage",
                "record_id": "u2",
                "grant_id": "g2",
                "actual_usage": {"tokens": 1},
            },
        ]
        errors = []
        warnings = []
        with patch.object(replay, "policy_for_ref", return_value=FakeBudgetPolicy()):
            replay._verify_budget_ledger(
                budget_records,
                "budget_policy:fake",
                errors,
                warnings,
            )
        joined_errors = "\n".join(errors)
        joined_warnings = "\n".join(warnings)
        self.assertIn("duplicate request_id", joined_errors)
        self.assertIn("does not match replayed policy decision", joined_errors)
        self.assertIn("decision field does not match replayed policy", joined_errors)
        self.assertIn("grant_id does not match replayed policy", joined_errors)
        self.assertIn("granted_budget does not match replayed policy", joined_errors)
        self.assertIn("duplicate grant_id", joined_errors)
        self.assertIn("has no prior request record", joined_errors)
        self.assertIn("invalid granted_budget", joined_errors)
        self.assertIn("missing grant_id", joined_errors)
        self.assertIn("references unknown grant_id", joined_errors)
        self.assertIn("unknown_units is not a list", joined_errors)
        self.assertIn("uses unapproved budget unit: gpu_hours", joined_errors)
        self.assertNotIn("uses unapproved budget unit: cost_usd", joined_errors)
        self.assertIn("non-numeric usage", joined_errors)
        self.assertIn("uses unapproved budget unit: gpu_hours", joined_errors)
        self.assertIn("non-numeric approval", joined_errors)
        self.assertIn("usage_unknown for grant g1", joined_warnings)
        self.assertIn("exceeds approved tokens", joined_warnings)

    def test_replay_provenance_artifact_prompt_and_state_helpers(self) -> None:
        from praxist.core import replay
        from praxist.core.prompt_layout import sha256_json, sha256_text
        from praxist.core.storage import sha256_bytes
        from praxist.plugins.workflow_stages.research_loop.backend.tools import local_store

        errors: list[str] = []
        warnings: list[str] = []
        agent_event = {
            "event_id": "evt_agent",
            "kind": "agent.run_finished",
            "payload": {
                "output_summary": {
                    "tool_uses": [
                        {
                            "tool": "mcp__evaluation-tools__share_finding",
                            "input": {
                                "peer_id": "gen0_peer0",
                                "finding_type": "result",
                                "title": "T",
                                "content": "C",
                                "variant_name": "V",
                                "metrics": json.dumps({"score": 1}),
                            },
                        }
                    ]
                }
            },
        }
        finding = {
            "finding_id": "f1",
            "peer_id": "gen0_peer0",
            "finding_type": "result",
            "title": "T",
            "content": "C",
            "variant_name": "V",
            "metrics": {"score": 1},
            "source_event_ids": ["evt_agent"],
        }
        self.assertTrue(replay._agent_event_supports_finding(agent_event, finding))
        replay._verify_output_agent_provenance(
            [agent_event], [finding], [{"finding_id": "f1"}], errors, warnings
        )
        self.assertEqual(errors, [])
        imported_event = {"event_id": "evt_imp", "kind": "finding.imported"}
        replay._verify_output_agent_provenance(
            [imported_event],
            [{"finding_id": "f2", "source_event_ids": ["evt_imp"], "provenance_quality": "weak"}],
            [{"finding_id": "f2", "source_event_ids": ["evt_imp"], "provenance_quality": "weak"}],
            errors,
            warnings,
        )
        self.assertTrue(any("imported legacy" in warning for warning in warnings))
        replay._verify_output_agent_provenance(
            [], [{"finding_id": "f3"}], [{"finding_id": "f3"}], errors, warnings
        )
        self.assertTrue(any("missing agent.run_finished" in error for error in errors))

        self.assertIsNone(replay._parse_metrics("{bad"))
        self.assertEqual(replay._parse_metrics({"x": 1}), {"x": 1})
        self.assertEqual(replay._norm_text(" a\n b "), "a b")
        self.assertEqual(replay._jsonable({"b": object()})["b"].startswith("<object"), True)
        self.assertTrue(
            replay._share_finding_tool_input_matches(
                agent_event["payload"]["output_summary"]["tool_uses"][0],
                finding,
            )
        )
        self.assertFalse(replay._share_finding_tool_input_matches({"tool": "Read"}, finding))
        self.assertFalse(
            replay._share_finding_tool_input_matches(
                {"tool": "share_finding", "input": {"peer_id": "other"}},
                finding,
            )
        )

        errors = []
        replay._verify_frontier_finding_refs(
            [{"finding_id": "f1"}, {"finding_id": ""}, {"finding_id": "missing"}],
            [{"finding_id": "f1"}],
            errors,
        )
        replay._verify_source_event_ids(
            [{"source_event_ids": "bad"}, {"source_event_ids": ["unknown"]}],
            {"known"},
            "ledger",
            errors,
        )
        self.assertTrue(any("missing finding_id" in error for error in errors))
        self.assertTrue(any("unknown finding_id" in error for error in errors))
        self.assertTrue(any("source_event_ids is not a list" in error for error in errors))
        self.assertTrue(any("unknown source_event_id" in error for error in errors))

        artifact_by_id = {
            "a1": {
                "artifact_id": "a1",
                "payload_path": "artifacts/by_id/a1/payload.txt",
                "content_hash": "sha256:" + "0" * 64,
            }
        }
        errors = []
        replay._verify_artifact_references(
            [
                {
                    "source_artifact_ids": ["missing"],
                    "artifact_refs": [
                        {"artifact_id": "a1", "payload_path": "bad", "content_hash": "bad"}
                    ],
                    "evidence_refs": [{"artifact_id": "missing"}],
                },
                {"source_artifact_ids": "bad", "artifact_refs": "bad", "evidence_refs": "bad"},
            ],
            artifact_by_id,
            "records",
            errors,
        )
        self.assertTrue(any("unknown source_artifact_id" in error for error in errors))
        self.assertTrue(any("payload_path mismatch" in error for error in errors))
        self.assertTrue(any("artifact_refs is not a list" in error for error in errors))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            payload_dir = run_dir / "artifacts" / "by_id" / "layout"
            payload_dir.mkdir(parents=True)
            prompt_path = payload_dir / "prompt.txt"
            prompt_text = "hello prompt"
            prompt_path.write_text(prompt_text, encoding="utf-8")
            blocks = [
                {
                    "block_id": "frozen",
                    "partition": "frozen_prefix",
                    "rendered_hash": sha256_text("frozen text"),
                    "dynamic_markers_in_template": [],
                    "dynamic_markers_in_rendered": [],
                },
                {
                    "block_id": "dynamic",
                    "partition": "dynamic_payload",
                    "rendered_hash": sha256_text("dynamic text"),
                },
            ]
            semi_hash = sha256_json([])
            frozen_hash = sha256_json(
                [{"block_id": "frozen", "rendered_hash": blocks[0]["rendered_hash"]}]
            )
            dynamic_hash = sha256_json(
                [{"block_id": "dynamic", "rendered_hash": blocks[1]["rendered_hash"]}]
            )
            layout_hash = sha256_json(
                {
                    "layout_version": "praxist.prompt_layout.v1",
                    "frozen_prefix_hash": frozen_hash,
                    "semi_static_hash": semi_hash,
                    "dynamic_payload_hash": dynamic_hash,
                    "block_hashes": [block["rendered_hash"] for block in blocks],
                }
            )
            manifest = {
                "schema_version": "praxist.prompt_layout.v1",
                "layout_version": "praxist.prompt_layout.v1",
                "blocks": blocks,
                "frozen_prefix_hash": frozen_hash,
                "semi_static_hash": semi_hash,
                "dynamic_payload_hash": dynamic_hash,
                "layout_hash": layout_hash,
                "frozen_audit": {"status": "pass"},
                "cache_mode": "runtime_auto_cache",
                "runtime_cache_strategy": "stable_prefix",
                "rendered_prompt_hash": sha256_text(prompt_text),
                "rendered_prompt_ref": {
                    "artifact_id": "prompt",
                    "payload_path": "artifacts/by_id/layout/prompt.txt",
                    "content_hash": sha256_bytes(prompt_path.read_bytes()),
                },
            }
            manifest_path = payload_dir / "layout.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            artifact_index = [
                {
                    "artifact_id": "layout",
                    "artifact_type": "prompt.layout_manifest",
                    "payload_path": "artifacts/by_id/layout/layout.json",
                    "content_hash": sha256_bytes(manifest_path.read_bytes()),
                }
            ]
            artifact_by_id = {
                "layout": artifact_index[0],
                "prompt": manifest["rendered_prompt_ref"],
            }
            errors = []
            warnings = []
            replay._verify_prompt_layout_artifacts(
                run_dir, artifact_index, artifact_by_id, errors, warnings
            )
            self.assertEqual(errors, [])

            bad_manifest = {**manifest, "schema_version": "bad", "frozen_audit": {"status": "fail"}}
            manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
            errors = []
            replay._verify_prompt_layout_artifacts(
                run_dir, artifact_index, artifact_by_id, errors, []
            )
            self.assertTrue(any("invalid schema_version" in error for error in errors))
            self.assertTrue(any("frozen_audit" in error for error in errors))

            (run_dir / "shared_findings").mkdir()
            (run_dir / "shared_findings" / "a.json").write_text(
                json.dumps({"id": "shared_only"}), encoding="utf-8"
            )
            (run_dir / "frontier").mkdir()
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                json.dumps({"cumulative_top": [{"finding_id": "frontier_only"}]}),
                encoding="utf-8",
            )
            ledger_dir = run_dir / "research_memory" / "ledgers"
            ledger_dir.mkdir(parents=True)
            (ledger_dir / "claims.yaml").write_text("entries:\n  - id: c1\n", encoding="utf-8")
            (run_dir / "graph").mkdir()
            (run_dir / "graph" / "graph.html").write_text("<html></html>", encoding="utf-8")
            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(run_dir)}, clear=False):
                local_store.init_db()
                local_store.insert_finding(
                    {
                        "id": "sqlite_only",
                        "finding_type": "result",
                        "title": "T",
                        "content": "C",
                        "metrics": {},
                        "variant_name": "V",
                        "peer_id": "p",
                        "generation_id": 0,
                    }
                )
            errors = []
            warnings = []
            report = replay._verify_state_surface_recovery(
                run_dir,
                findings=[],
                frontier=[],
                research_memory=[],
                graph_edges=[],
                artifact_index=[],
                errors=errors,
                warnings=warnings,
            )
            self.assertIn("shared_only", report["legacy"]["shared_finding_ids"])
            self.assertTrue(any("canonical findings missing" in error for error in errors))
            self.assertTrue(any("sync lag" in warning for warning in warnings))

    def test_runtime_provider_and_prompt_replay_edges_are_contract_checked(self) -> None:
        from praxist.core import replay
        from praxist.core.prompt_layout import sha256_json, sha256_text
        from praxist.core.storage import sha256_bytes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = root / "runtime"
            provider_dir = root / "provider"
            runtime_dir.mkdir()
            provider_dir.mkdir()
            (runtime_dir / "plugin.yaml").write_text(
                """
runtime:
  compatible_model_providers:
    - model_provider:other_provider
  cache_strategy: runtime_auto_cache
""",
                encoding="utf-8",
            )
            (provider_dir / "plugin.yaml").write_text(
                """
provider:
  cache_strategy: provider_explicit_cache
  explicit_cache_strategy: provider_cache_control
""",
                encoding="utf-8",
            )
            selected = [
                {
                    "metadata": {"kind": "agent_runtime", "name": "fake_runtime"},
                    "path": str(runtime_dir),
                },
                {
                    "metadata": {"kind": "model_provider", "name": "fake_provider"},
                    "path": str(provider_dir),
                },
            ]
            startup = {
                "canonical_args": {
                    "runtime": "agent_runtime:fake_runtime",
                    "model_provider": "model_provider:fake_provider",
                }
            }
            model_profiles = {
                "runtime_ref": "agent_runtime:other",
                "provider_adapters": {"model_provider:other": {}},
                "profiles": {
                    "bad_provider": {
                        "provider_ref": "model_provider:other",
                        "model": "fake/model",
                    },
                    "non_object": "bad",
                },
                "runtime_provider_conformance": {
                    "runtime_ref": "agent_runtime:other",
                    "model_provider_ref": "model_provider:other",
                    "cache_mode": "wrong",
                    "cache_policy_runtime_strategy": "wrong",
                    "cache_policy_provider_strategy": "wrong",
                },
            }
            errors: list[str] = []
            replay._verify_runtime_provider_conformance(
                selected,
                startup,
                model_profiles,
                {
                    "mode": "wrong",
                    "runtime_cache_strategy": "wrong",
                    "provider_cache_strategy": "wrong",
                },
                errors,
            )
            joined = "\n".join(errors)
            self.assertIn("runtime/provider conformance mismatch", joined)
            self.assertIn("cache_policy mode mismatch", joined)
            self.assertIn("runtime_ref mismatch", joined)
            self.assertEqual(
                replay._selected_manifest_contract({"path": "builtin://x"}, "runtime"), {}
            )
            self.assertEqual(
                replay._selected_manifest_contract({"path": str(root / "missing")}, "runtime"), {}
            )

            errors = []
            warnings: list[str] = []
            trajectory = [
                {
                    "kind": "agent.run_started",
                    "actor": {"type": "agent_runtime", "id": "agent_runtime:fake_runtime"},
                    "payload": "bad",
                },
                {
                    "kind": "agent.run_finished",
                    "actor": {"type": "agent_runtime", "id": "agent_runtime:fake_runtime"},
                    "payload": {
                        "request": {
                            "agent_runtime_ref": "agent_runtime:fake_runtime",
                            "model_call": {
                                "provider_ref": "model_provider:fake_provider",
                                "model": "fake-model",
                            },
                            "budget_grant_id": "grant",
                        }
                    },
                },
                {
                    "kind": "model.result",
                    "actor": {"type": "model_provider", "id": "model_provider:fake_provider"},
                    "payload": {
                        "provider_ref": "model_provider:wrong",
                        "model_call": {
                            "provider_ref": "model_provider:fake_provider",
                            "model": "fake-model",
                        },
                    },
                },
                {
                    "kind": "agent.run_started",
                    "actor": {"type": "agent_runtime", "id": "agent_runtime:fake_runtime"},
                    "payload": {},
                },
            ]
            replay._verify_runtime_provider_bindings(
                {"agent_runtime:fake_runtime", "model_provider:fake_provider"},
                startup,
                model_profiles,
                {"credential_profiles": []},
                trajectory,
                errors,
                warnings,
                locked=False,
            )
            self.assertTrue(any("provider adapter is not selected" in error for error in errors))
            self.assertTrue(any("model result provider_ref mismatch" in error for error in errors))
            self.assertTrue(any("missing replayable model_call" in warning for warning in warnings))

            run_dir = root / "run"
            layout_dir = run_dir / "artifacts" / "by_id" / "bad_layout"
            layout_dir.mkdir(parents=True)
            rendered = layout_dir / "rendered.txt"
            rendered.write_text("rendered", encoding="utf-8")
            bad_blocks = [
                "bad",
                {
                    "block_id": "frozen",
                    "partition": "frozen_prefix",
                    "rendered_hash": "bad",
                    "dynamic_markers_in_template": ["generation_id"],
                },
                {"block_id": "unknown", "partition": "bad", "rendered_hash": sha256_text("x")},
            ]
            bad_manifest = {
                "schema_version": "praxist.prompt_layout.v1",
                "layout_version": "wrong",
                "blocks": bad_blocks,
                "frozen_prefix_hash": "bad",
                "semi_static_hash": "bad",
                "dynamic_payload_hash": "bad",
                "layout_hash": "bad",
                "frozen_audit": {"status": "pass"},
                "cache_mode": "provider_explicit_cache",
                "rendered_prompt_hash": sha256_text("different"),
                "rendered_prompt_ref": {
                    "artifact_id": "rendered",
                    "payload_path": "artifacts/by_id/bad_layout/rendered.txt",
                    "content_hash": sha256_bytes(rendered.read_bytes()),
                },
            }
            manifest_path = layout_dir / "layout.json"
            manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
            artifact = {
                "artifact_id": "bad_layout",
                "artifact_type": "prompt.layout_manifest",
                "payload_path": "artifacts/by_id/bad_layout/layout.json",
                "content_hash": sha256_bytes(manifest_path.read_bytes()),
            }
            errors = []
            warnings = []
            replay._verify_prompt_layout_artifacts(
                run_dir,
                [artifact],
                {"bad_layout": artifact, "rendered": bad_manifest["rendered_prompt_ref"]},
                errors,
                warnings,
            )
            prompt_errors = "\n".join(errors)
            self.assertIn("invalid layout_version", prompt_errors)
            self.assertIn("block 1 is not an object", prompt_errors)
            self.assertIn("invalid partition", prompt_errors)
            self.assertIn("invalid rendered_hash", prompt_errors)
            self.assertIn("contains dynamic markers", prompt_errors)
            self.assertIn("provider_explicit_cache missing provider strategy", prompt_errors)
            self.assertIn("rendered_prompt_hash mismatch", prompt_errors)

            errors = []
            replay._verify_prompt_layout_hashes(
                {
                    "frozen_prefix_hash": sha256_json([]),
                    "semi_static_hash": sha256_json([]),
                    "dynamic_payload_hash": sha256_json([]),
                    "layout_hash": "bad",
                },
                [],
                "layout",
                errors,
            )
            self.assertTrue(any("layout_hash mismatch" in error for error in errors))

    def test_replay_edge_helpers_preserve_labels_without_trusting_surfaces(self) -> None:
        from praxist.core import replay
        from praxist.core.storage import sha256_bytes

        errors: list[str] = []
        replay._verify_trajectory(
            [
                {"seq": 2, "kind": "run.started", "parent_event_ids": "bad"},
                {"seq": 2, "event_id": "evt_000002", "parent_event_ids": ["missing"]},
                {"seq": 3, "event_id": "evt_000002", "parent_event_ids": []},
            ],
            errors,
        )
        joined = "\n".join(errors)
        self.assertIn("seq mismatch", joined)
        self.assertIn("missing event_id", joined)
        self.assertIn("parent_event_ids is not a list", joined)
        self.assertIn("unknown parent_event_id", joined)
        self.assertIn("duplicate event_id", joined)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            artifact = {
                "artifact_id": "a1",
                "payload_path": "wrong/location.txt",
                "content_hash": "sha256:" + "0" * 64,
            }
            errors = []
            warnings: list[str] = []
            replay._verify_artifact_tree(
                run_dir,
                {"a1": artifact},
                set(),
                errors,
                warnings,
                locked=False,
            )
            self.assertIn("artifact root missing", "\n".join(errors))

            payload = run_dir / "wrong" / "location.txt"
            payload.parent.mkdir(parents=True)
            payload.write_text("payload", encoding="utf-8")
            artifact_root = run_dir / "artifacts" / "by_id" / "a1"
            artifact_root.mkdir(parents=True)
            (artifact_root / "metadata.json").write_text(
                json.dumps({"artifact_id": "different"}),
                encoding="utf-8",
            )
            (artifact_root / "orphan.txt").write_text("orphan", encoding="utf-8")
            errors = []
            warnings = []
            replay._verify_artifact_tree(
                run_dir,
                {"a1": artifact},
                set(),
                errors,
                warnings,
                locked=False,
            )
            artifact_errors = "\n".join(errors)
            self.assertIn("does not live under its artifact directory", artifact_errors)
            self.assertIn("metadata mismatch", artifact_errors)
            self.assertIn("unindexed artifact file", "\n".join(warnings))

            logs = run_dir / "logs"
            logs.mkdir()
            (logs / "secret.log").write_text("OPENAI_API_KEY=sk-testsecret1234", encoding="utf-8")
            text_dir = run_dir / "notes"
            text_dir.mkdir()
            (text_dir / "secret.md").write_text("Bearer abcdefgh12345678", encoding="utf-8")
            errors = []
            replay._scan_tree(logs, run_dir, errors)
            replay._scan_run_text_tree(run_dir, errors)
            self.assertTrue(any("redaction scan hit" in error for error in errors))

            outside = run_dir.parent / f"{run_dir.name}_outside"
            outside.mkdir()
            link = run_dir / "link"
            link.symlink_to(outside, target_is_directory=True)
            errors = []
            self.assertIsNone(
                replay._resolve_run_relative_path(run_dir, "link/payload.txt", errors, "link")
            )
            self.assertIn("escapes run_dir", "\n".join(errors))

            for text in ("[1]", "runtime: ["):
                plugin_dir = run_dir / f"plugin_{len(text)}"
                plugin_dir.mkdir()
                (plugin_dir / "plugin.yaml").write_text(text, encoding="utf-8")
                self.assertEqual(
                    replay._selected_manifest_contract({"path": str(plugin_dir)}, "runtime"),
                    {},
                )
            provider_dir = run_dir / "provider"
            provider_dir.mkdir()
            (provider_dir / "plugin.yaml").write_text("provider: []", encoding="utf-8")
            self.assertEqual(
                replay._selected_manifest_contract({"path": str(provider_dir)}, "provider"),
                {},
            )

            (run_dir / "trajectory.jsonl").write_text(
                json.dumps({"kind": "agent.run_started"}) + "\n",
                encoding="utf-8",
            )
            errors = []
            warnings = []
            replay._verify_prompt_layout_artifacts(run_dir, [], {}, errors, warnings)
            self.assertIn("no prompt.layout_manifest", "\n".join(warnings))
            errors = []
            replay._verify_prompt_layout_artifacts(
                run_dir,
                [
                    {
                        "artifact_id": "layout_missing_payload",
                        "artifact_type": "prompt.layout_manifest",
                        "payload_path": None,
                    },
                    {
                        "artifact_id": "layout_missing_file",
                        "artifact_type": "prompt.layout_manifest",
                        "payload_path": "artifacts/by_id/missing/layout.json",
                    },
                ],
                {},
                errors,
                [],
            )
            self.assertEqual(errors, [])

            rendered = run_dir / "rendered.txt"
            rendered.write_text("prompt", encoding="utf-8")
            errors = []
            replay._verify_rendered_prompt_hash(
                run_dir,
                {},
                {"payload_path": "rendered.txt"},
                "layout",
                errors,
            )
            replay._verify_rendered_prompt_hash(
                run_dir,
                {"rendered_prompt_hash": "sha256:" + "0" * 64},
                {"payload_path": None},
                "layout",
                errors,
            )
            replay._verify_rendered_prompt_hash(
                run_dir,
                {"rendered_prompt_hash": "sha256:" + "0" * 64},
                {"payload_path": "missing.txt"},
                "layout",
                errors,
            )
            self.assertTrue(any("missing rendered_prompt_hash" in error for error in errors))

            frontier = run_dir / "frontier"
            frontier.mkdir(exist_ok=True)
            (frontier / "frontier_manifest.json").write_text("[1]", encoding="utf-8")
            self.assertEqual(replay._frontier_manifest_ids(run_dir), set())
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "cumulative_top": [1, {"id": "cumulative"}],
                        "generations": {"0": [2, {"finding_id": "gen"}], "1": {"members": []}},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(replay._frontier_manifest_ids(run_dir), {"cumulative", "gen"})

            ledger_dir = run_dir / "research_memory" / "ledgers"
            ledger_dir.mkdir(parents=True, exist_ok=True)
            (ledger_dir / "bad.yaml").write_text("x: [", encoding="utf-8")
            (ledger_dir / "list.yaml").write_text("- item\n", encoding="utf-8")
            self.assertEqual(replay._research_memory_ledger_entry_count(run_dir), 0)

            invalid = run_dir / "invalid.json"
            invalid.write_text("{bad", encoding="utf-8")
            not_object = run_dir / "list.json"
            not_object.write_text("[1]", encoding="utf-8")
            errors = []
            self.assertIsNone(replay._read_json(invalid, errors))
            self.assertIsNone(replay._read_json(not_object, errors))
            self.assertIn("json_decode", "\n".join(errors))
            self.assertIn("not_object", "\n".join(errors))

            payload_path = run_dir / "artifacts" / "by_id" / "a2" / "payload.txt"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text("payload", encoding="utf-8")
            artifact_index = [
                {
                    "artifact_id": "a2",
                    "artifact_type": "payload",
                    "payload_path": "artifacts/by_id/a2/payload.txt",
                    "content_hash": sha256_bytes(payload_path.read_bytes()),
                }
            ]
            self.assertEqual(replay._artifact_type_counts(artifact_index), {"payload": 1})

        errors = []
        replay._verify_plugin_dependency_closure(
            [
                {
                    "metadata": {
                        "kind": "tool_server",
                        "name": "x",
                        "dependencies": [
                            {"required": False},
                            {"kind": 1, "name": "bad"},
                        ],
                    }
                }
            ],
            [{"from": "tool_server:x", "to": "tool_server:x"}],
            {"tool_server:x"},
            errors,
        )
        self.assertIn("malformed dependency", "\n".join(errors))

        self.assertEqual(
            replay._expected_cache_contract({"cache_strategy": "deterministic_no_cache"}, {}),
            ("disabled", None, None),
        )
        self.assertEqual(
            replay._expected_cache_contract({}, {"cache_strategy": "disabled"}),
            ("disabled", None, None),
        )
        errors = []
        self.assertEqual(
            replay._credential_key_ids_for_provider({}, "model_provider:fake_provider", errors),
            set(),
        )
        replay._verify_model_call_credential(
            {},
            "model_provider:fake_provider",
            set(),
            "fake call",
            errors,
        )
        replay._verify_model_call_credential(
            {"credential_ref": "key-1"},
            "model_provider:openrouter",
            {"key-1"},
            "string call",
            errors,
        )
        self.assertEqual(errors, [])

        errors = []
        replay._verify_runtime_provider_conformance([], {"canonical_args": "bad"}, {}, {}, errors)
        replay._verify_runtime_provider_conformance(
            [],
            {"canonical_args": {"runtime": 1, "model_provider": "model_provider:x"}},
            {},
            {},
            errors,
        )
        self.assertEqual(errors, [])

        self.assertFalse(
            replay._agent_event_supports_finding(
                {
                    "payload": {
                        "output_summary": {"tool_uses": [{"tool": "share_finding", "input": []}]}
                    }
                },
                {"finding_id": "f1", "peer_id": "peer"},
            )
        )
        self.assertFalse(
            replay._share_finding_tool_input_matches(
                {
                    "tool": "share_finding",
                    "input": {
                        "peer_id": "p",
                        "title": "different",
                        "metrics": {"score": 2},
                    },
                },
                {"peer_id": "p", "title": "expected", "metrics": {"score": 1}},
            )
        )


if __name__ == "__main__":
    unittest.main()
