from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import json
import os
import sqlite3
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch


def _text_payload(result: dict) -> object:
    return json.loads(result["content"][0]["text"])


class ToolServerFactoryCoverageContractsTest(unittest.TestCase):
    def _fake_sdk_module(self) -> ModuleType:
        fake = ModuleType("claude_agent_sdk")

        def tool(name, description, schema):
            def decorate(fn):
                fn._tool_name = name
                fn._tool_description = description
                fn._tool_schema = schema
                return fn

            return decorate

        def create_sdk_mcp_server(name, tools):
            return {"name": name, "tools": list(tools)}

        fake.tool = tool  # type: ignore[attr-defined]
        fake.create_sdk_mcp_server = create_sdk_mcp_server  # type: ignore[attr-defined]
        return fake

    def _reload_with_fake_sdk(self, module_name: str):
        original_sdk = sys.modules.get("claude_agent_sdk")
        fake_sdk = self._fake_sdk_module()
        sys.modules["claude_agent_sdk"] = fake_sdk
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        self.addCleanup(self._restore_module, module_name, original_sdk)
        return module

    @staticmethod
    def _restore_module(module_name: str, original_sdk) -> None:
        if original_sdk is None:
            sys.modules.pop("claude_agent_sdk", None)
        else:
            sys.modules["claude_agent_sdk"] = original_sdk
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

    def test_tool_server_factories_register_real_handlers_when_sdk_is_available(self) -> None:
        eval_adapter = self._reload_with_fake_sdk("praxist.plugins.tools.evaluation_tools.adapter")
        frontier_adapter = self._reload_with_fake_sdk(
            "praxist.plugins.tools.frontier_tools.adapter"
        )
        graph_adapter = self._reload_with_fake_sdk(
            "praxist.plugins.tools.finding_graph_query.adapter"
        )
        prior_adapter = self._reload_with_fake_sdk("praxist.plugins.tools.prior_work_tools.adapter")
        memory_adapter = self._reload_with_fake_sdk("praxist.plugins.tools.memory_tools.adapter")

        eval_server = eval_adapter.create_evaluation_tools_server()
        self.assertEqual(eval_server["name"], "evaluation-tools")
        self.assertEqual(
            [tool._tool_name for tool in eval_server["tools"]],
            [
                "log_experiment_metrics",
                "share_finding",
                "get_leaderboard",
                "wait_for_file",
                "read_tool_result",
            ],
        )
        self.assertEqual(frontier_adapter.create_frontier_tools_server()["name"], "frontier-tools")
        self.assertEqual(
            graph_adapter.create_finding_graph_query_server()["name"],
            "finding-graph-query",
        )
        self.assertEqual(prior_adapter.create_prior_work_tools_server()["name"], "prior-work-tools")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            memory_server = memory_adapter.create_memory_tools_server(run_dir)
            self.assertEqual(memory_server["name"], "memory-tools")
            handlers = {tool._tool_name: tool for tool in memory_server["tools"]}
            self.assertEqual(
                sorted(handlers),
                [
                    "get_evidence_card",
                    "get_ledger_entry",
                    "list_active_claims",
                    "list_open_objections",
                    "query_coverage_matrix",
                    "query_evidence_cards",
                    "resolve_source_ref",
                ],
            )
            self.assertIn(
                "not found",
                _text_payload(
                    asyncio.run(handlers["get_evidence_card"]({"evidence_id": "missing"}))
                )["error"],
            )
            self.assertEqual(
                _text_payload(asyncio.run(handlers["query_evidence_cards"]({"limit": 1}))),
                [],
            )
            self.assertFalse(
                _text_payload(
                    asyncio.run(
                        handlers["query_coverage_matrix"](
                            {"variant_family": "family", "parameter": "rho"}
                        )
                    )
                )["covered"]
            )
            self.assertEqual(_text_payload(asyncio.run(handlers["list_active_claims"]({}))), [])
            self.assertEqual(_text_payload(asyncio.run(handlers["list_open_objections"]({}))), [])
            self.assertIn(
                "unsupported",
                _text_payload(
                    asyncio.run(
                        handlers["get_ledger_entry"]({"ledger_name": "../bad", "entry_id": "x"})
                    )
                )["error"],
            )
            self.assertIn(
                "no resolvable",
                _text_payload(asyncio.run(handlers["resolve_source_ref"]({"source_ref": {}})))[
                    "error"
                ],
            )


class ResearchMemoryCoverage95ContractsTest(unittest.TestCase):
    def _write_shared_store(self, db_path: Path) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE findings (
                    id TEXT PRIMARY KEY,
                    finding_type TEXT,
                    title TEXT,
                    content TEXT,
                    metrics TEXT,
                    variant_name TEXT,
                    notes TEXT,
                    peer_id TEXT,
                    generation_id INTEGER,
                    timestamp TEXT,
                    extra TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO findings
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "finding-1",
                    "result",
                    "A strong result",
                    "content",
                    json.dumps({"mean_test_accuracy": {"mean": 0.81}, "tier": "T3"}),
                    "VariantA",
                    "notes",
                    "gen1_peer0",
                    1,
                    "2026-05-12T00:00:00",
                    json.dumps({"peer_role": "bridge"}),
                ),
            )

    def test_source_resolver_handles_files_sqlite_and_rejections(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.source_resolver import (
            SourceResolver,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "shared_findings").mkdir()
            (run_dir / "agendas").mkdir()
            (run_dir / "results").mkdir()
            (run_dir / "logs").mkdir()
            (run_dir / "shared_findings" / "finding.json").write_text(
                json.dumps({"id": "file-finding"}), encoding="utf-8"
            )
            (run_dir / "agendas" / "agenda.yaml").write_text("a: 1\n", encoding="utf-8")
            (run_dir / "results" / "result.txt").write_text("plain text", encoding="utf-8")
            db_path = run_dir / "shared_store.db"
            self._write_shared_store(db_path)

            resolver = SourceResolver(run_dir)
            self.assertEqual(
                resolver.resolve({"finding_path": "shared_findings/finding.json"})["content"]["id"],
                "file-finding",
            )
            self.assertEqual(
                resolver.resolve({"agenda_path": "agendas/agenda.yaml"})["content"],
                {"a": 1},
            )
            self.assertEqual(
                resolver.resolve({"result_path": "results/result.txt"})["content"],
                "plain text",
            )
            self.assertIn("rejected", resolver.resolve({"raw_log_path": "/etc/passwd"})["error"])
            self.assertIn("not found", resolver.resolve({"finding_path": "missing.json"})["error"])
            found = resolver.resolve({"finding_id": "finding-1"})
            self.assertEqual(found["kind"], "finding_db")
            self.assertEqual(found["content"]["metrics"]["tier"], "T3")
            self.assertIn(
                "not in shared_store",
                resolver.resolve({"finding_id": "missing"})["error"],
            )

            with patch.object(sqlite3, "connect", side_effect=sqlite3.Error("db")):
                self.assertIn("sqlite read failed", resolver.resolve({"finding_id": "x"})["error"])

    def test_card_builder_handles_paths_metrics_and_db_failure_modes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            card_builder,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            shared = run_dir / "shared_findings"
            shared.mkdir()
            exact = shared / "finding-1_VariantA.json"
            exact.write_text("{}", encoding="utf-8")
            finding = {
                "id": "finding-1",
                "finding_type": "challenge",
                "title": "Kill weak claim",
                "notes": "failed to reproduce",
                "metrics": json.dumps(
                    {
                        "mean_test_accuracy": {"mean": "0.82"},
                        "sharpness_top_eigen": "bad",
                        "promotion_eligible": True,
                        "tier": {"bad": "shape"},
                        "seed_count": 3.0,
                    }
                ),
                "variant_name": "VariantA",
                "peer_id": "gen1_peer0",
                "generation_id": 1,
                "extra": '{"peer_role": "falsifier"}',
                "timestamp": "2026-05-12T00:00:00",
            }
            card = card_builder.build_card_from_finding(finding, run_dir)
            self.assertEqual(
                card["source_ref"]["finding_path"], "shared_findings/finding-1_VariantA.json"
            )
            self.assertTrue(card["quality"]["is_negative"])
            self.assertEqual(card["metrics"]["seed_count"], 3)
            self.assertIn("bad", card["metrics"]["tier"])
            self.assertIsNone(card_builder._safe_get_metric(["bad"], "score"))
            self.assertIsNone(card_builder._safe_get_metric({"score": None}, "score"))
            self.assertIsNone(card_builder._safe_get_metric({"score": {"mean": "nan"}}, "score"))
            self.assertEqual(card_builder._safe_metric_map(["bad"]), {})
            self.assertEqual(
                card_builder._safe_metric_map(
                    {"ok": "1.5", "flag": True, "bad": "x", "nested": {"mean": 2}}
                ),
                {"ok": 1.5, "nested": 2.0},
            )
            self.assertTrue(card_builder._safe_bool(1))
            self.assertFalse(card_builder._safe_bool(0))
            self.assertTrue(card_builder._safe_bool("passed"))
            self.assertFalse(card_builder._safe_bool("failed"))
            self.assertIsNone(card_builder._safe_bool("maybe"))
            categories = card_builder._safe_categorical_map(
                {"peer_role": "top"},
                {"frontier_lane": ["alpha", 2], "promotion_eligible": "promotable"},
                {"extra": {"parent_usage": "repair"}, "peer_role": "nested"},
            )
            self.assertEqual(categories["frontier_lane"], ["alpha", "2"])
            self.assertEqual(categories["parent_usage"], "repair")
            self.assertTrue(categories["promotion_eligible"])
            self.assertTrue(card_builder._detect_negative({"extra": {"is_negative": True}}))
            self.assertTrue(
                card_builder._detect_negative(
                    {"extra": {"peer_role": "falsifier"}, "title": "no improvement"}
                )
            )
            self.assertTrue(card_builder._detect_negative({"extra": {"is_negative": True}}))
            self.assertTrue(
                card_builder._detect_negative(
                    {
                        "finding_type": "result",
                        "title": "neutral title",
                        "content": "implementation ran successfully",
                        "metrics": {"score": 0.4},
                        "extra": {
                            "evidence_valence": "negative",
                            "failure_mode": "primary_metric_regression",
                        },
                    }
                )
            )
            structured_card = card_builder.build_card_from_finding(
                {
                    "id": "finding-structured-neg",
                    "finding_type": "result",
                    "title": "Ablation result",
                    "content": "The implementation completed, but the mechanism was unnecessary.",
                    "metrics": {"score": 0.4},
                    "variant_name": "VariantNegative",
                    "peer_id": "gen2_peer4",
                    "generation_id": 2,
                    "extra": {
                        "target_hypothesis": "H_mechanism",
                        "is_negative": False,
                        "evidence_valence": "negative",
                        "failure_mode": "ablation_no_effect",
                        "disconfirming_claim_ids": ["C_parent"],
                    },
                },
                run_dir,
            )
            self.assertTrue(structured_card["quality"]["is_negative"])
            self.assertEqual(structured_card["quality"]["evidence_valence"], "negative")
            self.assertEqual(structured_card["quality"]["failure_mode"], "ablation_no_effect")
            self.assertEqual(
                structured_card["quality"]["disconfirming_claim_ids"],
                ["C_parent"],
            )
            self.assertEqual(
                structured_card["metrics"]["disconfirming_claim_ids"],
                ["C_parent"],
            )
            self.assertEqual(
                structured_card["claim_relevance"]["challenges"],
                ["C_parent", "H_mechanism"],
            )
            self.assertFalse(card_builder._detect_negative({"extra": "{bad", "title": "neutral"}))
            self.assertIn(
                "_x",
                card_builder._evidence_id("", 1, "gen1_peer0", content_seed="stable"),
            )

            globbed = shared / "finding-2_other.json"
            globbed.write_text("{}", encoding="utf-8")
            glob_card = card_builder.build_card_from_finding(
                {
                    "id": "finding-2",
                    "title": "No improvement",
                    "metrics": "{bad",
                    "extra": "{bad",
                    "generation_id": 2,
                    "peer_id": "gen2_peer0",
                    "timestamp": "",
                },
                run_dir,
            )
            self.assertEqual(
                glob_card["source_ref"]["finding_path"], "shared_findings/finding-2_other.json"
            )
            self.assertTrue(glob_card["quality"]["is_negative"])

            db_path = run_dir / "shared_store.db"
            self._write_shared_store(db_path)
            cards = card_builder.build_cards_from_db(run_dir, db_path=db_path, only_gen=1)
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0]["source_ref"]["generation_id"], 1)
            with patch.object(card_builder.sqlite3, "connect", side_effect=sqlite3.Error("db")):
                self.assertEqual(card_builder.build_cards_from_db(run_dir, db_path=db_path), [])


class StageAndAgentCoverage95ContractsTest(unittest.TestCase):
    def test_stage_budget_usage_paths_and_provider_env_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop import stage

        self.assertEqual(stage.create_stage().ref, "workflow_stage:research_loop")
        self.assertFalse(stage._stage_result_indicates_execution(None))
        self.assertFalse(stage._stage_result_indicates_execution({"generations_completed": "bad"}))
        self.assertTrue(stage._stage_result_indicates_execution({"frontier_summary": [{}]}))
        self.assertTrue(stage._stage_result_indicates_execution({"usage": {"tokens": 1}}))
        self.assertEqual(
            stage._missing_approved_usage_units(
                {"granted_budget": {"tokens": 10, "gpu_hours": "bad"}},
                {"wall_clock_seconds": 1},
            ),
            ["tokens"],
        )
        self.assertEqual(
            stage._missing_approved_usage_units({"granted_budget": "bad"}, {}),
            ["approved budget payload is invalid"],
        )
        self.assertEqual(
            stage._budget_shortfalls(planned={}, approved="bad"),
            ["approved budget payload is invalid"],
        )
        self.assertIn(
            "tokens approved is non-numeric",
            stage._budget_shortfalls(planned={"tokens": 10}, approved={"tokens": "x"}),
        )
        self.assertEqual(stage._first_positive_number({"usage": {"tokens": "2"}}, ("tokens",)), 2.0)
        self.assertIsNone(stage._first_positive_number({"tokens": "bad"}, ("tokens",)))

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "or-key",
                "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1",
                "ANTHROPIC_API_KEY": "anthropic-key",
                "OPENAI_API_KEY": "openai-key",
                "DEEPSEEK_API_KEY": "deepseek-key",
                "GROQ_API_KEY": "groq-key",
                "MISTRAL_API_KEY": "mistral-key",
                "XAI_API_KEY": "xai-key",
            },
            clear=True,
        ):
            self.assertEqual(
                stage._provider_env("model_provider:openrouter")["ANTHROPIC_AUTH_TOKEN"],
                "or-key",
            )
            self.assertEqual(
                stage._provider_env("model_provider:openrouter")["OPENROUTER_API_KEY"],
                "or-key",
            )
            self.assertEqual(
                stage._provider_env("model_provider:anthropic_messages")["ANTHROPIC_API_KEY"],
                "anthropic-key",
            )
            self.assertEqual(
                stage._provider_env("model_provider:openai_compatible")["OPENAI_API_KEY"],
                "openai-key",
            )
            self.assertEqual(
                stage._provider_env("model_provider:groq_alias")["GROQ_API_KEY"],
                "groq-key",
            )
            self.assertEqual(
                stage._provider_env("model_provider:mistral_alias")["MISTRAL_API_KEY"],
                "mistral-key",
            )
            self.assertEqual(
                stage._provider_env("model_provider:xai_alias")["XAI_API_KEY"],
                "xai-key",
            )
            self.assertEqual(
                stage._provider_env("model_provider:deepseek_alias")["DEEPSEEK_API_KEY"],
                "deepseek-key",
            )
            deepseek_env = stage._provider_env("model_provider:deepseek_alias")
            self.assertEqual(
                deepseek_env["ANTHROPIC_BASE_URL"],
                "https://api.deepseek.com/anthropic",
            )
            self.assertEqual(deepseek_env["ANTHROPIC_AUTH_TOKEN"], "deepseek-key")
            self.assertEqual(deepseek_env["ANTHROPIC_MODEL"], "deepseek-v4-pro[1m]")
            self.assertIsNone(stage._provider_env("model_provider:fake_provider")["OPENAI_API_KEY"])
            with self.assertRaises(ValueError):
                stage._provider_env("model_provider:unknown")

    def test_research_loop_stage_execute_resolve_success_and_unknown_usage(self) -> None:
        from praxist.plugins.workflow_stages.research_loop import stage

        class FakeLedger:
            records: list[tuple[str, dict]] = []

            def __init__(self, run_dir, run_id):
                self.run_dir = run_dir
                self.run_id = run_id

            def require_active_grant(self, grant_id):
                if grant_id == "missing":
                    raise ValueError("missing grant")
                return {
                    "request_id": "req",
                    "granted_budget": {
                        "tokens": 300000,
                        "wall_clock_seconds": 4000,
                        "gpu_hours": 1,
                    },
                }

            def append_usage(self, **kwargs):
                self.records.append(("usage", kwargs))

            def append_usage_unknown(self, **kwargs):
                self.records.append(("unknown", kwargs))

        class FakeLoop:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def run(self):
                os.environ["PRAXIST_PROTECTED_CHILD_PATHS"] = "/loop-local-protected-child"
                return {
                    "generations_completed": 1,
                    "tokens": 123,
                    "run_dir": str(self.kwargs["run_dir"]),
                }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            ctx = stage.ResearchLoopStageContext(
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(
                        max_generations=1,
                        cohort_size=1,
                        per_generation_hours=1,
                    ),
                    compute_budget=SimpleNamespace(per_experiment_gpu_hours=0.1),
                ),
                workspace=run_dir,
                run_dir=run_dir,
                local_mode=True,
                model="fake",
                model_provider_ref="model_provider:fake_provider",
                frontier_strategy="auto",
                budget_grant_id="grant",
                provider_env={"CUSTOM_ENV": "value"},
            )
            with (
                patch.object(stage, "BudgetLedger", FakeLedger),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.generation_loop.GenerationLoop",
                    FakeLoop,
                ),
            ):
                result = asyncio.run(stage.ResearchLoopStage().execute(ctx))
                self.assertTrue(result.success)
                self.assertEqual(result.summary["usage_unknown_units"], ["gpu_hours"])
                self.assertIsNone(os.environ.get("CUSTOM_ENV"))
                self.assertIsNone(os.environ.get("PRAXIST_PROTECTED_CHILD_PATHS"))

                resolve_ctx = stage.ResearchLoopStageContext(
                    **{**ctx.__dict__, "resolve_only": True}
                )
                self.assertTrue(asyncio.run(stage.ResearchLoopStage().execute(resolve_ctx)).success)
                missing_ctx = stage.ResearchLoopStageContext(
                    **{**ctx.__dict__, "budget_grant_id": "missing"}
                )
                self.assertFalse(
                    asyncio.run(stage.ResearchLoopStage().execute(missing_ctx)).success
                )

            no_budget_ctx = stage.ResearchLoopStageContext(
                **{**ctx.__dict__, "budget_grant_id": None}
            )
            self.assertFalse(asyncio.run(stage.ResearchLoopStage().execute(no_budget_ctx)).success)

    def test_agent_helper_paths_do_not_depend_on_runtime_side_effects(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop = root / "STOP_SIGNAL"
            checker = agent.StopChecker(max_runtime=999, stop_signal_path=stop)
            self.assertIsNone(checker.check())
            stop.write_text("stop", encoding="utf-8")
            self.assertEqual(checker.check(), agent.StopReason.SYNTHESIS_TRIGGER)

            self.assertEqual(agent._runtime_final_payload(SimpleNamespace(events=[])), {})
            self.assertEqual(agent._float_payload("bad"), 0.0)
            self.assertEqual(agent._int_payload("bad"), 0)
            self.assertIn(
                "Praxist Bootstrap Recovery", agent._with_bootstrap_retry_directive("task")
            )

            loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer0",
                generation_id=0,
                task_prompt="task",
                workspace=root,
                max_runtime_seconds=1,
                logs_dir=root / "logs",
                findings_dir=root / "findings",
                local_mode=True,
                stop_signal_path=stop,
            )
            self.assertEqual(loop._session_event_watch_paths(productive=False), [stop])
            self.assertFalse(loop._is_next_session_event(root / "note.txt", productive=False))
            self.assertFalse(loop._is_next_session_event(root / "shared_store.db-wal"))
            self.assertFalse(loop._is_next_session_event(root / "trace.log"))
            finding_path = loop._findings_dir / "x.json"
            self.assertTrue(loop._is_next_session_event(finding_path))
            self.assertTrue(loop._session_was_productive(None))
            self.assertFalse(loop._session_was_productive(SimpleNamespace(iteration_count="bad")))
            self.assertFalse(loop._session_was_bootstrap_wait(SimpleNamespace(success=False)))
            self.assertFalse(
                loop._session_was_bootstrap_wait(
                    SimpleNamespace(success=True, iteration_count=1, output={})
                )
            )
            self.assertFalse(
                loop._session_was_bootstrap_wait(
                    SimpleNamespace(success=True, iteration_count=0, output="bad")
                )
            )
            self.assertFalse(
                loop._session_was_bootstrap_wait(
                    SimpleNamespace(success=True, iteration_count=0, output={"text_outputs": "bad"})
                )
            )
            self.assertFalse(
                loop._session_was_bootstrap_wait(
                    SimpleNamespace(success=True, iteration_count=0, output={"text_outputs": [""]})
                )
            )

            with patch.dict(os.environ, {"PRAXIST_RUN_DIR": str(root), "PRAXIST_RUN_ID": "run"}):
                writer = agent._legacy_trajectory_writer()
                self.assertIsNotNone(writer)
            with patch.dict(os.environ, {"PRAXIST_RUN_DIR": ""}, clear=False):
                self.assertIsNone(agent._legacy_trajectory_writer())

            for provider_ref, expected_key in (
                ("model_provider:openrouter", "ANTHROPIC_AUTH_TOKEN"),
                ("model_provider:anthropic_messages", "ANTHROPIC_API_KEY"),
                ("model_provider:openai_compatible", "OPENAI_API_KEY"),
                ("model_provider:deepseek_alias", "ANTHROPIC_AUTH_TOKEN"),
                ("model_provider:fake_provider", None),
            ):
                env = {
                    "PRAXIST_MODEL_PROVIDER_REF": provider_ref,
                    "ANTHROPIC_API_KEY": "anthropic",
                    "ANTHROPIC_AUTH_TOKEN": "token",
                    "OPENROUTER_API_KEY": "openrouter",
                    "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
                    "OPENAI_API_KEY": "openai",
                    "DEEPSEEK_API_KEY": "deepseek",
                }
                with patch.dict(os.environ, env, clear=True):
                    scoped = agent._scoped_legacy_provider_env()
                if expected_key is None:
                    self.assertEqual(scoped, {"PRAXIST_MODEL_PROVIDER_REF": provider_ref})
                else:
                    if provider_ref == "model_provider:openrouter":
                        self.assertEqual(scoped["OPENROUTER_API_KEY"], "openrouter")
                    if provider_ref == "model_provider:deepseek_alias":
                        self.assertEqual(scoped[expected_key], "deepseek")
                        self.assertEqual(
                            scoped["ANTHROPIC_BASE_URL"], "https://api.deepseek.com/anthropic"
                        )
                        self.assertEqual(scoped["DEEPSEEK_API_KEY"], "deepseek")
                    else:
                        self.assertEqual(scoped[expected_key], env[expected_key])
                    self.assertEqual(scoped["PRAXIST_MODEL_PROVIDER_REF"], provider_ref)
                    if provider_ref == "model_provider:deepseek_alias":
                        self.assertEqual(scoped["ANTHROPIC_AUTH_TOKEN"], "deepseek")
                        self.assertEqual(
                            scoped["ANTHROPIC_BASE_URL"], "https://api.deepseek.com/anthropic"
                        )

            with patch.dict(
                os.environ,
                {"PRAXIST_MODEL_PROVIDER_REF": "", "ANTHROPIC_API_KEY": "legacy"},
                clear=True,
            ):
                self.assertIn("ANTHROPIC_API_KEY", agent._scoped_legacy_provider_env())

    def test_base_agent_normalizes_runtime_result_and_prompt_helpers(self) -> None:
        from praxist.core.protocol import AgentEvent, AgentRunResult
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jinja2"
            task = root / "task.jinja2"
            gen = root / "gen.jinja2"
            out = root / "prompt.md"
            base.write_text("Base {{ value }}\n", encoding="utf-8")
            task.write_text("Task {{ value }}\n", encoding="utf-8")
            gen.write_text("Gen {{ value }}\n", encoding="utf-8")
            rendered = agent.resolve_prompt(base, task, gen, out, {"value": "V"})
            self.assertIn("Base V", rendered)
            self.assertTrue((root / "prompt_layout.json").exists())

            class FakeRuntime:
                def execute_sync(self, request):
                    return AgentRunResult(
                        success=True,
                        events=[
                            AgentEvent(
                                event_id="evt",
                                run_id=request.run_id,
                                agent_run_id="run",
                                stage_id=request.stage_id,
                                type="message",
                                payload={},
                                artifact_refs=[],
                                credential_refs=[],
                                timestamp_ms=1,
                            ),
                            AgentEvent(
                                event_id="evt2",
                                run_id=request.run_id,
                                agent_run_id="run",
                                stage_id=request.stage_id,
                                type="final_result",
                                payload={
                                    "duration": "2.5",
                                    "iteration_count": "3",
                                    "legacy_output": {"tool_uses": [{"name": "Read"}]},
                                },
                                artifact_refs=[],
                                credential_refs=[],
                                timestamp_ms=2,
                            ),
                        ],
                        text_output_refs=[],
                        tool_uses=[],
                        error=None,
                        failover_reason=None,
                        credential_ref=None,
                    )

            with (
                patch.dict(
                    os.environ,
                    {
                        "PRAXIST_AGENT_RUNTIME_REF": "agent_runtime:fake",
                        "PRAXIST_MODEL_PROVIDER_REF": "model_provider:openrouter",
                        "PRAXIST_RUN_ID": "run",
                        "PRAXIST_STAGE_ID": "stage",
                    },
                    clear=True,
                ),
                patch.object(agent, "runtime_for_ref", return_value=FakeRuntime()),
            ):
                base_agent = agent.BaseAgent(
                    name="agent/name",
                    allowed_tools=["Read"],
                    workspace=root,
                    mcp_servers={"tools": object()},
                    model="fake-model",
                    prompt_layout_manifest={
                        "layout_hash": "lh",
                        "frozen_prefix_hash": "fh",
                        "dynamic_payload_hash": "dh",
                    },
                )
                result = asyncio.run(base_agent.execute("task"))
                self.assertTrue(result.success)
                self.assertEqual(result.duration, 2.5)
                self.assertEqual(result.iteration_count, 3)
                self.assertEqual(result.output["tool_uses"][0]["name"], "Read")
                request = base_agent._build_agent_run_request("task", {})
                self.assertEqual(request.prompt_ref["kind"], "prompt_layout_v1")
                self.assertEqual(request.cache_policy.frozen_prefix_hash, "fh")
                self.assertEqual(
                    agent._legacy_model_provider_ref("deepseek-chat"), "model_provider:openrouter"
                )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    agent._legacy_model_provider_ref("openai/gpt-4o"),
                    "model_provider:openrouter",
                )
                self.assertEqual(
                    agent._legacy_model_provider_ref("deepseek-chat"),
                    "model_provider:deepseek_alias",
                )
                self.assertEqual(
                    agent._legacy_model_provider_ref("gpt-4o"),
                    "model_provider:openai_compatible",
                )
                self.assertEqual(
                    agent._legacy_model_provider_ref("claude-sonnet"),
                    "model_provider:anthropic_messages",
                )

    def test_agent_session_logging_bootstrap_and_s3_paths_are_bounded(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        class TextPart:
            text = "hello secret-free text"

        class ToolPart:
            name = "Bash"
            input = {"cmd": "python train.py", "payload": "x" * 1200}

        class Message:
            content = [TextPart(), ToolPart()]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer0",
                generation_id=0,
                task_prompt="task",
                workspace=root,
                logs_dir=root / "logs",
                findings_dir=root / "findings",
                local_mode=True,
                max_runtime_seconds=1,
            )

            created_callbacks = []

            class FakeBaseAgent:
                def __init__(self, callback):
                    self.callback = callback

                async def execute(self, task):
                    self.callback(Message())
                    return agent.AgentResult(
                        success=True,
                        output={"text_outputs": ["done"], "tool_uses": [{"name": "Bash"}]},
                        duration=1.0,
                        iteration_count=1,
                    )

            def fake_create_agent(_session_id, message_callback=None):
                created_callbacks.append(message_callback)
                return FakeBaseAgent(message_callback)

            with patch.object(loop, "_create_agent", side_effect=fake_create_agent):
                result = asyncio.run(loop._run_session())
            self.assertTrue(result.success)
            log_text = next((root / "logs").glob("session_*.log")).read_text(encoding="utf-8")
            self.assertIn("Tool: Bash", log_text)
            self.assertIn("[truncated]", log_text)
            self.assertEqual(len(created_callbacks), 1)

            wait_result = agent.AgentResult(
                success=True,
                output={"text_outputs": ["What would you like me to do?"]},
                duration=0.0,
                iteration_count=0,
            )
            self.assertTrue(loop._session_was_bootstrap_wait(wait_result))
            self.assertEqual(agent._int_env("MISSING_INT_ENV", 7), 7)
            with patch.dict(os.environ, {"BAD_INT_ENV": "bad"}):
                self.assertEqual(agent._int_env("BAD_INT_ENV", 7), 7)

            uploaded = []

            def fake_upload(**kwargs):
                uploaded.append(kwargs)
                return True

            loop.findings_path.write_text("{}", encoding="utf-8")
            with patch(
                "praxist.infrastructure.s3_utils.upload_file_to_s3",
                side_effect=fake_upload,
            ):
                asyncio.run(loop._sync_to_s3())
            self.assertTrue(any(item["s3_key"].endswith("findings.json") for item in uploaded))


class PIAgentCoverage95ContractsTest(unittest.TestCase):
    def _agent(self, run_dir: Path):
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        prompt = run_dir / "pi_prompt.jinja2"
        prompt.write_text("gen={{ completed_gen_id }}", encoding="utf-8")
        return PIAgent(
            run_dir=run_dir,
            workspace=run_dir,
            cohort_size=5,
            model="fake",
            max_runtime_minutes=1,
            prompt_template_path=prompt,
        )

    def _write_pi_db(self, db_path: Path) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE findings (
                    id TEXT PRIMARY KEY,
                    finding_type TEXT,
                    title TEXT,
                    content TEXT,
                    metrics TEXT,
                    variant_name TEXT,
                    notes TEXT,
                    peer_id TEXT,
                    generation_id INTEGER,
                    timestamp TEXT,
                    extra TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE finding_edges (
                    edge_id TEXT PRIMARY KEY,
                    src_finding_id TEXT,
                    dst_finding_id TEXT,
                    edge_type TEXT,
                    confidence REAL,
                    created_by TEXT,
                    rationale TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO findings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "f0",
                    "result",
                    "prior",
                    "content",
                    json.dumps({"score": 0.8, "extra_metric": "x" * 400}),
                    "VariantA",
                    "notes",
                    "gen0_peer0",
                    0,
                    "2026-05-12T00:00:00",
                    "{bad",
                ),
            )
            conn.execute(
                "INSERT INTO findings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "f1",
                    "insight",
                    "current",
                    "content",
                    json.dumps({"tier": "T2"}),
                    "VariantB",
                    "notes",
                    "gen1_peer0",
                    1,
                    "2026-05-12T00:00:01",
                    json.dumps({"ok": True}),
                ),
            )
            conn.execute(
                "INSERT INTO finding_edges VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("e1", "f0", "f1", "supports", 0.9, "engine", "rationale"),
            )

    def _valid_agenda(self, gen_id: int = 2) -> dict[str, Any]:
        # 5 peers with 4 roles (cycles: exploit, falsifier, bridge, anti_mainline, exploit)
        roles = ["exploit", "falsifier", "bridge", "anti_mainline", "exploit"]
        return {
            "generation": gen_id,
            "cross_peer_hypotheses": [
                {
                    "id": "H1",
                    "claim": "claim",
                    "minimal_test": "test",
                    "kill_condition": "kill",
                    "promote_condition": "promote",
                }
            ],
            "peer_contracts": {
                f"gen{gen_id}_peer{i}": {
                    "role": role,
                    "target_hypothesis": "H1",
                    "success_signal": "signal",
                }
                for i, role in enumerate(roles)
            },
            "mainline_observation": {},
            "bridge_hypothesis": {},
            "anti_mainline_contract": {},
            "falsification_contract": {},
            "success_metrics": {},
        }

    def test_pi_loaders_validators_and_single_pi_recovery_paths(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            agent = self._agent(run_dir)
            db_path = run_dir / "shared_store.db"
            self._write_pi_db(db_path)
            self.assertEqual(len(agent._load_gen_findings(1)), 1)
            self.assertEqual(len(agent._load_gen_edges(1)), 1)
            self.assertEqual(agent._build_findings_summary_for_panel(1)["by_type"], {"insight": 1})
            self.assertEqual(agent._load_prior_findings_summary(1)[0]["metrics"]["score"], 0.8)

            frontier = run_dir / "frontier"
            frontier.mkdir()
            (frontier / "frontier_manifest.json").write_text(
                json.dumps(
                    {
                        "generations": {
                            "1": {
                                "members": [
                                    {
                                        "evidence_stage": "full_T1",
                                        "metrics": {"x": float("nan")},
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            loaded_frontier = agent._load_frontier_summary()
            self.assertIsNone(loaded_frontier[0]["metrics"]["x"])
            self.assertEqual(loaded_frontier[0]["generation_id"], 1)
            self.assertEqual(agent._load_frontier_summary(completed_gen_id=0), [])

            with patch.object(pi_agent.sqlite3, "connect", side_effect=sqlite3.Error("db")):
                self.assertEqual(agent._load_gen_findings(1), [])
                self.assertEqual(agent._load_gen_edges(1), [])
                self.assertEqual(agent._load_prior_findings_summary(1), [])
                self.assertEqual(agent._build_findings_summary_for_panel(1), {})

            agent.agendas_dir.mkdir()
            prior = agent.agendas_dir / pi_agent.AGENDA_FILE_PATTERN.format(1)
            prior.write_text(
                "```yaml\nmainline_observation: bad\ncross_peer_hypotheses: scalar\n"
                "anti_mainline_contract:\n  forbidden_mechanisms: one\n```\n",
                encoding="utf-8",
            )
            summary = agent._load_prior_agendas_summary(2)[0]
            self.assertEqual(summary["mainline_dominant"], [])
            self.assertEqual(summary["anti_mainline_forbidden"], ["one"])

            valid = self._valid_agenda(2)
            self.assertIsNone(agent.validate_agenda(valid, 2))
            self.assertEqual(valid["cross_peer_hypotheses"][0]["id"], "H1")
            self.assertIn("not a dict", agent.validate_agenda([], 2))
            self.assertIn(
                "cannot be parsed",
                agent.validate_agenda({**self._valid_agenda(2), "generation": "gen"}, 2),
            )
            self.assertIn(
                "is not int-coercible",
                agent.validate_agenda({**self._valid_agenda(2), "generation": object()}, 2),
            )
            self.assertIn(
                "non-empty list",
                agent.validate_agenda({**self._valid_agenda(2), "cross_peer_hypotheses": []}, 2),
            )
            self.assertIn(
                "at least one dict",
                agent.validate_agenda({**self._valid_agenda(2), "cross_peer_hypotheses": ["x"]}, 2),
            )
            self.assertIn(
                "peer_contracts must be a dict",
                agent.validate_agenda({**self._valid_agenda(2), "peer_contracts": [1]}, 2),
            )
            bad_peers = self._valid_agenda(2)
            bad_peers["peer_contracts"] = {"gen2_peer0": {"role": "exploit"}}
            self.assertIn("cohort_size=5", agent.validate_agenda(bad_peers, 2))
            bad_contract = self._valid_agenda(2)
            bad_contract["peer_contracts"] = {f"gen2_peer{i}": "bad" for i in range(5)}
            self.assertIn("entries must be dicts", agent.validate_agenda(bad_contract, 2))
            missing_role = self._valid_agenda(2)
            missing_role["peer_contracts"]["gen2_peer3"]["role"] = "exploit"
            self.assertIn("missing required roles", agent.validate_agenda(missing_role, 2))
            placeholder = self._valid_agenda(2)
            placeholder["cross_peer_hypotheses"][0]["claim"] = "<one paragraph>"
            self.assertIn("literal placeholder", agent.validate_agenda(placeholder, 2))
            technical_notation = self._valid_agenda(2)
            technical_notation["cross_peer_hypotheses"][0]["claim"] = (
                "test <lookahead-k> schedule with rho<0.20 on the validation slice"
            )
            self.assertIsNone(agent.validate_agenda(technical_notation, 2))
            wrong_optional = self._valid_agenda(2)
            wrong_optional["success_metrics"] = []
            self.assertIn(
                "success_metrics must be a dict", agent.validate_agenda(wrong_optional, 2)
            )

            load_path = agent.agendas_dir / pi_agent.AGENDA_FILE_PATTERN.format(2)
            load_path.write_text("peer_contracts: []\n", encoding="utf-8")
            self.assertIsNone(pi_agent.load_agenda_for_gen(run_dir, 2))
            load_path.write_text("mainline_observation: []\n", encoding="utf-8")
            self.assertIsNone(pi_agent.load_agenda_for_gen(run_dir, 2))
            load_path.write_text("cross_peer_hypotheses: bad\n", encoding="utf-8")
            self.assertIsNone(pi_agent.load_agenda_for_gen(run_dir, 2))

    def test_pi_run_paths_keep_partial_outputs_and_report_failures(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            agent = self._agent(run_dir)

            async def timeout_invoke(_prompt, out_path, *, request_id):
                del request_id
                out_path.write_text(json.dumps(self._valid_agenda(1)), encoding="utf-8")
                raise TimeoutError

            with (
                patch.object(agent, "_load_gen_findings", return_value=[{"id": "f"}]),
                patch.object(agent, "_load_gen_edges", return_value=[]),
                patch.object(agent, "_load_frontier_summary", return_value=[]),
                patch.object(agent, "_load_prior_agenda", return_value=None),
                patch.object(agent, "_load_prior_agendas_summary", return_value=[]),
                patch.object(agent, "_load_prior_findings_summary", return_value=[]),
                patch.object(agent, "_invoke_synthesizer", side_effect=timeout_invoke),
            ):
                partial = asyncio.run(agent.run(0))
            self.assertFalse(partial.success)
            self.assertIsNone(partial.agenda_path)
            self.assertIn("no agenda", partial.error or "")
            self.assertFalse((agent.agendas_dir / pi_agent.AGENDA_FILE_PATTERN.format(1)).exists())
            self.assertFalse(
                (agent.agendas_dir / pi_agent.AGENDA_FILE_PATTERN.format(1))
                .with_suffix(".yaml.candidate")
                .exists()
            )

            async def none_invoke(_prompt, _out_path, *, request_id):
                del request_id
                return None

            with (
                patch.object(agent, "_load_gen_findings", return_value=[]),
                patch.object(agent, "_load_gen_edges", return_value=[]),
                patch.object(agent, "_load_frontier_summary", return_value=[]),
                patch.object(agent, "_load_prior_agenda", return_value=None),
                patch.object(agent, "_load_prior_agendas_summary", return_value=[]),
                patch.object(agent, "_load_prior_findings_summary", return_value=[]),
                patch.object(agent, "_invoke_synthesizer", side_effect=none_invoke),
            ):
                failed = asyncio.run(agent.run(0))
            self.assertFalse(failed.success)
            self.assertIn("no agenda", failed.error or "")

            async def invalid_invoke(_prompt, out_path, *, request_id):
                del request_id
                out_path.write_text(
                    "generation: 1\ncross_peer_hypotheses: []\npeer_contracts: {}\n",
                    encoding="utf-8",
                )
                return {"generation": 1, "cross_peer_hypotheses": [], "peer_contracts": {}}

            with (
                patch.object(agent, "_load_gen_findings", return_value=[{"id": "f"}]),
                patch.object(agent, "_load_gen_edges", return_value=[]),
                patch.object(agent, "_load_frontier_summary", return_value=[]),
                patch.object(agent, "_load_prior_agenda", return_value=None),
                patch.object(agent, "_load_prior_agendas_summary", return_value=[]),
                patch.object(agent, "_load_prior_findings_summary", return_value=[]),
                patch.object(agent, "_invoke_synthesizer", side_effect=invalid_invoke),
                patch.object(Path, "replace", side_effect=OSError("replace")),
            ):
                invalid = asyncio.run(agent.run(0))
            self.assertFalse(invalid.success)
            self.assertIn("validation:", invalid.error or "")
            rejected = pi_agent._parse_agenda_file(invalid.agenda_path)
            self.assertIsInstance(rejected, dict)
            self.assertEqual(rejected["artifact_semantics"]["status"], "failed")
            self.assertEqual(rejected["artifact_semantics"]["role"], "partial_output")
            self.assertIn("generation: 1", rejected["raw_candidate_text"])

            class FailingPanelConfig:
                panel_mode_default = "full"
                auto_escalate_to_high_stakes = False
                pi_max_runtime_minutes = 1
                chair_max_runtime_minutes = 1
                n_rounds = 1
                round2_max_runtime_minutes = 1
                fallback_to_single_pi_on_panel_failure = False

            panel_agent = pi_agent.PIAgent(
                run_dir=run_dir,
                workspace=run_dir,
                cohort_size=5,
                model="fake",
                use_multi_pi_panel=True,
                multi_pi_config=FailingPanelConfig(),
            )

            async def raise_panel(**_kwargs):
                raise RuntimeError("panel")

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                side_effect=raise_panel,
            ):
                panel_failed = asyncio.run(panel_agent.run(0))
            self.assertFalse(panel_failed.success)
            self.assertIn("no fallback", panel_failed.error or "")


class PanelAndLoopCoverage95ContractsTest(unittest.TestCase):
    def test_chair_parser_and_fallback_contracts_cover_non_llm_recovery(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            chair_arbiter,
        )

        fenced = "intro\n```yaml\nagenda_version: '2.0'\npeer_contracts: {}\n```\ntrailer"
        parsed = chair_arbiter._parse_chair_agenda_text(fenced)
        self.assertIsNotNone(parsed.agenda)
        self.assertEqual(parsed.agenda["agenda_version"], "2.0")
        self.assertEqual(chair_arbiter._strip_yaml_fence(123), "")
        self.assertEqual(chair_arbiter._unique_strings([{"id": "x"}, "x", None, " y "]), ["x", "y"])
        self.assertEqual(chair_arbiter._clip("", fallback="fb"), "fb")
        self.assertIn("…", chair_arbiter._clip("x" * 400, limit=10))

        memos = {
            "builder": {
                "top_claims": [
                    {
                        "id": "C1",
                        "statement": "Claim 1",
                        "supports": [{"finding_id": "f1"}],
                    }
                ],
                "objections_or_warnings": [
                    {
                        "target_claim": "C1",
                        "objection": "risk",
                        "resolving_experiment": "control",
                    }
                ],
                "proposed_peer_contracts": [
                    {"role": "bridge", "target_hypothesis": "H2", "rationale": "bridge"}
                ],
            },
            "portfolio": {
                "top_claims": [{"id": "C2", "statement": "Claim 2"}],
                "proposed_experiments": [{"id": "E1", "description": "preserve idea"}],
            },
            "extra": {"top_claims": [{"id": "C3", "statement": "Claim 3"}]},
        }
        reviews = {
            "skeptic": {
                "singleton_high_upside_idea_to_preserve": {
                    "idea_summary": "rare upside",
                    "source": "private",
                },
                "own_revisions": [
                    {
                        "claim_id": "C1",
                        "boundary_old": "old",
                        "boundary_new": "new",
                        "triggered_by": "review",
                    }
                ],
            },
            "bad": "not-a-dict",
        }
        fallback = chair_arbiter._build_deterministic_fallback_agenda(
            pi_memos=memos,
            cross_reviews=reviews,
            next_gen_id=2,
            completed_gen_id=1,
            panel_mode="full",
            shared_core_id="abc",
            peer_budget=5,
            parse_error="bad yaml",
        )
        self.assertEqual(len(fallback["peer_contracts"]), 5)
        self.assertEqual(fallback["fallback_metadata"]["reason"], "chair_yaml_parse_failed")
        self.assertEqual(fallback["minority_high_upside"][0]["rationale"], "rare upside")
        self.assertEqual(fallback["claim_boundary_updates"][0]["claim_id"], "C1")

    def test_panel_helpers_round2_and_generation_loop_error_paths_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_loop
        from praxist.plugins.workflow_stages.research_loop.backend.multi_pi import (
            legacy_two_round_executor as panel,
        )

        class FakePI:
            def __init__(self, role_name: str, parsed=None, raises: bool = False):
                self.role_name = role_name
                self.parsed = parsed or {}
                self.raises = raises

            async def run(self, **_kwargs):
                if self.raises:
                    raise RuntimeError("round1")
                return SimpleNamespace(parsed=self.parsed)

            async def run_cross_review(self, **_kwargs):
                if self.raises:
                    raise RuntimeError("round2")
                return SimpleNamespace(
                    parsed={
                        "own_revisions": [
                            {"claim_id": "known", "boundary_new": "new"},
                            {"claim_id": "hallucinated", "boundary_new": "bad"},
                        ]
                    }
                )

        memos = asyncio.run(
            panel._run_pi_parallel(
                [FakePI("ok", {"top_claims": []}), FakePI("bad", raises=True)],
                {"shared_core_id": "nothex"},
                {},
                ["decide"],
            )
        )
        self.assertTrue(memos["bad"]["_panic"])
        round2, label_maps = asyncio.run(
            panel._run_round2_parallel(
                [FakePI("ok"), FakePI("skip"), FakePI("bad", raises=True)],
                {
                    "ok": {"top_claims": [{"id": "known"}]},
                    "skip": {"_pi_unavailable": True},
                    "bad": {"top_claims": [{"id": "known"}]},
                },
                round2_max_runtime_minutes=1,
                rng_seed=7,
            )
        )
        self.assertTrue(round2["skip"]["_round2_skipped"])
        self.assertTrue(round2["bad"]["_round2_failed"])
        self.assertEqual(round2["ok"]["own_revisions"][0]["claim_id"], "known")
        self.assertIn("ok", label_maps)
        long_memo = {"top_claims": [{"id": "keep", "statement": "x" * 5000}], "extra": "drop"}
        truncated = panel._truncate_memo_for_round2(long_memo, max_tokens=1)
        self.assertEqual(truncated["top_claims"][0]["id"], "keep")
        self.assertIn("[truncated", truncated["top_claims"][0]["statement"])

        class FakeTaskSpec:
            task_id = "task"
            task_name = "Task"
            _raw = {"task": "raw"}
            generation_policy = SimpleNamespace(
                max_generations=1,
                cohort_size=1,
                promote_top_k=1,
                promote_criterion="top_k",
            )
            evaluation = SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                anchor_metrics=[],
                requires_tier=False,
            )
            baselines = []
            multi_pi = SimpleNamespace(
                enabled=True, panel_mode_default="full", auto_escalate_to_high_stakes=True
            )
            research_memory = None
            pi_agent = SimpleNamespace(enabled=True, max_runtime_minutes=1, strict=False)
            agent = SimpleNamespace(premium_mode=False)

            def get_prompt_task_path(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = generation_loop.GenerationLoop(
                task_spec=FakeTaskSpec(),
                workspace=root,
                run_dir=root / "run",
                local_mode=True,
                model="fake",
            )
            self.assertEqual(loop._strategy_for_gen(0), "explore")
            self.assertEqual(loop._strategy_for_gen(1), "pi_directed")
            loop.frontier_strategy = "mixed"
            self.assertEqual(loop._strategy_for_gen(3), "mixed")
            (loop.run_dir / "orchestrator.lock").write_text("old lock", encoding="utf-8")

            async def fail_generation(_gen_id):
                raise RuntimeError("boom")

            with (
                patch.object(loop, "_run_generation", side_effect=fail_generation),
                patch.object(generation_loop, "configure_runtime_environment"),
                patch.object(generation_loop, "initialize_local_store_if_needed"),
                patch.object(generation_loop, "validate_baseline_cache_for_run"),
                patch.object(generation_loop, "start_sidecars"),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.sidecars.stop_sidecars"
                ),
                self.assertRaises(RuntimeError),
            ):
                asyncio.run(loop.run())
            summary = json.loads((loop.run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["exit_condition"], "error")
            self.assertEqual(summary["error_type"], "RuntimeError")

            failed_start_loop = generation_loop.GenerationLoop(
                task_spec=FakeTaskSpec(),
                workspace=root,
                run_dir=root / "failed-sidecar-start",
                local_mode=True,
                model="fake",
            )
            runtime_closed: list[bool] = []
            runtime_scope = SimpleNamespace(close=lambda: runtime_closed.append(True))
            with (
                patch.object(
                    generation_loop,
                    "enter_orchestrator_runtime_scope",
                    return_value=runtime_scope,
                ),
                patch.object(generation_loop, "configure_runtime_environment"),
                patch.object(generation_loop, "initialize_local_store_if_needed"),
                patch.object(generation_loop, "validate_baseline_cache_for_run"),
                patch.object(
                    generation_loop,
                    "start_sidecars",
                    side_effect=RuntimeError("sidecar unavailable"),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.sidecars.stop_sidecars"
                ) as stop_failed_sidecars,
                self.assertRaisesRegex(RuntimeError, "sidecar unavailable"),
            ):
                asyncio.run(failed_start_loop.run())
            self.assertEqual(runtime_closed, [True])
            stop_failed_sidecars.assert_called_once_with(
                failed_start_loop,
                exit_condition="error",
            )

            failed_setup_loop = generation_loop.GenerationLoop(
                task_spec=FakeTaskSpec(),
                workspace=root,
                run_dir=root / "failed-runtime-setup",
                local_mode=True,
                model="fake",
            )
            setup_runtime_closed: list[bool] = []
            setup_runtime_scope = SimpleNamespace(close=lambda: setup_runtime_closed.append(True))
            with (
                patch.object(
                    generation_loop,
                    "enter_orchestrator_runtime_scope",
                    return_value=setup_runtime_scope,
                ),
                patch.object(
                    generation_loop,
                    "configure_runtime_environment",
                    side_effect=RuntimeError("configuration unavailable"),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.sidecars.stop_sidecars"
                ),
                self.assertRaisesRegex(RuntimeError, "configuration unavailable"),
            ):
                asyncio.run(failed_setup_loop.run())
            self.assertEqual(setup_runtime_closed, [True])

            failed_stop_loop = generation_loop.GenerationLoop(
                task_spec=FakeTaskSpec(),
                workspace=root,
                run_dir=root / "failed-sidecar-stop",
                local_mode=True,
                model="fake",
            )
            stop_runtime_closed: list[bool] = []
            stop_runtime_scope = SimpleNamespace(close=lambda: stop_runtime_closed.append(True))
            with (
                patch.object(
                    generation_loop,
                    "enter_orchestrator_runtime_scope",
                    return_value=stop_runtime_scope,
                ),
                patch.object(failed_stop_loop, "_run_generation", side_effect=RuntimeError("boom")),
                patch.object(generation_loop, "configure_runtime_environment"),
                patch.object(generation_loop, "initialize_local_store_if_needed"),
                patch.object(generation_loop, "validate_baseline_cache_for_run"),
                patch.object(generation_loop, "start_sidecars"),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.sidecars.stop_sidecars",
                    side_effect=KeyboardInterrupt("stop interrupted"),
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "stop interrupted"),
            ):
                asyncio.run(failed_stop_loop.run())
            self.assertEqual(stop_runtime_closed, [True])

            with patch.object(
                loop, "_collect_findings_for_generation", side_effect=RuntimeError("x")
            ):
                snapshot = loop._build_status_snapshot()
            self.assertEqual(snapshot.current_generation, loop._current_generation)

    def test_generation_loop_helper_fallbacks_and_delegates_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_loop

        class FakeTaskSpec:
            task_id = "task"
            task_name = "Task"
            _raw = {
                "task": "raw",
                "risk_violating_frontier": {
                    "enabled": True,
                    "min_primary_metric": "not-a-number",
                },
            }
            generation_policy = SimpleNamespace(
                max_generations=1,
                cohort_size=1,
                promote_top_k=1,
                promote_criterion="top_k",
            )
            evaluation = SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                anchor_metrics=[],
                frontier_lanes=[],
                requires_tier=False,
            )
            baselines = []
            multi_pi = None
            research_memory = None
            pi_agent = SimpleNamespace(enabled=False, max_runtime_minutes=1, strict=False)
            agent = SimpleNamespace(premium_mode=False)

            def get_prompt_task_path(self):
                return None

        with patch(
            "praxist.core.panel_topology.panel_topology_for_ref",
            side_effect=RuntimeError("registry unavailable"),
        ):
            self.assertEqual(
                generation_loop._resolve_topology_peer_info(None, topology_ref="panel_topology:x"),
                ((), {}),
            )
            self.assertEqual(
                generation_loop._resolve_peer_role_rotation(None, topology_ref="panel_topology:x"),
                (),
            )

        with self.assertRaisesRegex(ValueError, "explicit external run_dir"):
            generation_loop.GenerationLoop(
                task_spec=FakeTaskSpec(),
                workspace=Path(tempfile.gettempdir()),
                run_dir=None,
                local_mode=True,
                tool_server_refs=[],
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(generation_loop, "open", create=True, side_effect=OSError("nope")),
                self.assertLogs(generation_loop.logger.name, level="WARNING") as logs,
            ):
                loop = generation_loop.GenerationLoop(
                    task_spec=FakeTaskSpec(),
                    workspace=root,
                    run_dir=root / "run",
                    local_mode=True,
                    tool_server_refs=[],
                    runtime_ref="agent_runtime:codex_sdk",
                )
            self.assertTrue(
                any("failed to snapshot task_spec" in item for item in logs.output),
                logs.output,
            )
            self.assertTrue(
                any("min_primary_metric" in item for item in logs.output),
                logs.output,
            )

            with patch.object(
                generation_loop,
                "build_prompt_context",
                return_value={"context": "ok"},
            ) as build_prompt:
                self.assertEqual(loop._build_prompt_context(0, 0, 1), {"context": "ok"})
            self.assertEqual(
                build_prompt.call_args.kwargs["runtime_ref"], "agent_runtime:codex_sdk"
            )
            self.assertEqual(build_prompt.call_args.kwargs["strategy"], "explore")

            with patch.object(
                generation_loop,
                "persist_prompt_layout_artifacts",
                return_value={"manifest": "stored"},
            ) as persist:
                persisted = loop._persist_prompt_layout_artifacts(
                    prompt_text="prompt",
                    prompt_path=root / "prompt.txt",
                    manifest={"v": 1},
                    manifest_path=root / "manifest.json",
                    peer_id="peer-0",
                    gen_id=0,
                )
            self.assertEqual(persisted, {"manifest": "stored"})
            self.assertEqual(persist.call_args.kwargs["run_dir"], loop.run_dir)

            with patch.object(generation_loop, "run_generation_cohort", return_value=[]) as cohort:
                self.assertEqual(asyncio.run(loop._run_generation(0)), [])
            cohort.assert_called_once_with(loop, 0)
            topology_path = loop.run_dir / "gen_0" / "research_topology.json"
            self.assertTrue(topology_path.exists())
            topology = json.loads(topology_path.read_text(encoding="utf-8"))
            self.assertEqual(topology["nodes"][0]["worker_id"], "gen0_peer0")
            self.assertFalse(loop._check_plateau())

            with patch.object(generation_loop, "update_research_memory_post_gen") as update:
                loop._update_research_memory_post_gen(0, [{"id": "f"}], ["p"])
            self.assertEqual(update.call_args.kwargs["run_dir"], loop.run_dir)
            self.assertIs(update.call_args.kwargs["evaluation"], loop.task_spec.evaluation)

    def test_generation_loop_resume_and_interrupt_paths_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_loop
        from praxist.plugins.workflow_stages.research_loop.backend.resume_state import (
            write_boundary_marker,
        )

        class FakeTaskSpec:
            task_id = "task"
            task_name = "Task"
            _raw = {"task": "raw"}
            generation_policy = SimpleNamespace(
                max_generations=1,
                cohort_size=1,
                promote_top_k=1,
                promote_criterion="top_k",
            )
            evaluation = SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                anchor_metrics=[],
                frontier_lanes=[],
                requires_tier=False,
            )
            baselines = []
            multi_pi = None
            research_memory = None
            pi_agent = SimpleNamespace(enabled=False, max_runtime_minutes=1, strict=False)
            agent = SimpleNamespace(premium_mode=False)

            def get_prompt_task_path(self):
                return None

        async def fake_complete(loop, *, gen_id, **_kwargs):
            write_boundary_marker(
                loop.run_dir,
                gen_id=gen_id,
                promoted_count=0,
                pi_status="test_committed",
            )
            return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = generation_loop.GenerationLoop(
                task_spec=FakeTaskSpec(),
                workspace=root,
                run_dir=root / "resume-run",
                local_mode=True,
                tool_server_refs=[],
                resume=True,
            )
            resume_plan = SimpleNamespace(
                completed_generations=0,
                has_pending_boundary=True,
                pending_boundary_generation=0,
                start_generation=1,
                warnings=["old boundary"],
                to_dict=lambda: {"resume": True},
            )
            recovered = SimpleNamespace(triggered=False, reset_count=0, admitted_count=0)
            with (
                patch.object(
                    generation_loop,
                    "prepare_resume_for_sidecars",
                    return_value=resume_plan,
                ),
                patch.object(generation_loop, "append_resume_event") as append_event,
                patch.object(
                    generation_loop,
                    "repair_inferred_gems_boundary_markers",
                    return_value=[{"generation": 0}],
                ),
                patch.object(
                    generation_loop,
                    "load_generation_results",
                    return_value=[{"generation": 0}],
                ),
                patch.object(
                    generation_loop,
                    "recover_pending_gems_reset_for_resume",
                    return_value=recovered,
                ),
                patch.object(generation_loop, "complete_generation_boundary", fake_complete),
                patch.object(generation_loop, "configure_runtime_environment"),
                patch.object(generation_loop, "initialize_local_store_if_needed"),
                patch.object(generation_loop, "validate_baseline_cache_for_run"),
                patch.object(generation_loop, "start_sidecars"),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.sidecars.stop_sidecars"
                ),
                patch.object(
                    loop,
                    "_collect_findings_for_generation",
                    side_effect=RuntimeError("stats failed"),
                ),
            ):
                summary = asyncio.run(loop.run())
            self.assertEqual(summary["generations_completed"], 1)
            self.assertEqual(summary["last_gen_findings_count"], 0)
            self.assertGreaterEqual(append_event.call_count, 2)

        async def raise_keyboard_interrupt(_gen_id):
            raise KeyboardInterrupt("stop now")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = generation_loop.GenerationLoop(
                task_spec=FakeTaskSpec(),
                workspace=root,
                run_dir=root / "interrupt-run",
                local_mode=True,
                tool_server_refs=[],
            )
            with (
                patch.object(loop, "_run_generation", side_effect=raise_keyboard_interrupt),
                patch.object(generation_loop, "configure_runtime_environment"),
                patch.object(generation_loop, "initialize_local_store_if_needed"),
                patch.object(generation_loop, "validate_baseline_cache_for_run"),
                patch.object(generation_loop, "start_sidecars"),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.sidecars.stop_sidecars"
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                asyncio.run(loop.run())
            summary = json.loads((loop.run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["exit_condition"], "interrupted")
            self.assertEqual(summary["error_type"], "KeyboardInterrupt")


class IngestAndToolCoverage95ContractsTest(unittest.TestCase):
    def test_finding_ingest_error_paths_and_metric_edges_are_stable(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_ingest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_dir = root / "missing"
            self.assertEqual(findings_ingest.ingest_findings_directory(missing_dir), 0)
            self.assertIsNone(findings_ingest._parse_numeric(object()))
            self.assertIsNone(findings_ingest._walk_find({"a": {"b": 1}}, ("b",), depth=7))
            self.assertIsNone(findings_ingest._walk_find_str({"a": {"b": "x"}}, ("b",), depth=7))
            self.assertEqual(
                findings_ingest.infer_dataset(
                    root / "imagenet_result.json",
                    {"dataset_aliases": ["imagenet"]},
                ),
                "imagenet",
            )
            self.assertIsNone(
                findings_ingest.infer_dataset(
                    root / "x.json",
                    {
                        "title": "compare cifar10 and cifar100",
                        "dataset_aliases": {"cifar10": ["cifar10"], "cifar100": ["cifar100"]},
                    },
                )
            )
            self.assertEqual(
                findings_ingest.extract_metrics(
                    {"metrics": {"tier": "T1", "promotion_eligible": False, "score": "0.7"}}
                )["promotion_eligible"],
                False,
            )
            self.assertEqual(
                findings_ingest.extract_metrics(
                    {"nested": {"test_acc": "82%"}, "gap": "12%"},
                    "mnist",
                    primary_metric="test_accuracy",
                )["test_accuracy_mnist"],
                0.82,
            )
            self.assertNotIn(
                "test_accuracy",
                findings_ingest.extract_metrics(
                    {"nested": {"test_acc": "82%"}},
                    "mnist",
                ),
            )
            self.assertEqual(findings_ingest.extract_variant_name({"variant_name": " V "}), "V")
            self.assertEqual(findings_ingest.extract_variant_name({"optimizer": "AdamW"}), "")
            self.assertEqual(
                findings_ingest.derive_finding_id(root / "abc.json", {"id": "not-a-uuid"})[:3],
                "fs_",
            )

            unreadable = root / "unreadable.json"
            unreadable.write_text("{}", encoding="utf-8")
            with patch("builtins.open", side_effect=OSError("read")):
                self.assertIsNone(findings_ingest.parse_finding_file(unreadable))
            bad_json = root / "bad.json"
            bad_json.write_text("{bad", encoding="utf-8")
            self.assertIsNone(findings_ingest.parse_finding_file(bad_json))
            list_json = root / "list.json"
            list_json.write_text("[]", encoding="utf-8")
            self.assertIsNone(findings_ingest.parse_finding_file(list_json))

            findings_dir = root / "findings"
            findings_dir.mkdir()
            (findings_dir / "gen2_peer0_result.json").write_text(
                json.dumps(
                    {
                        "title": "result",
                        "content": "body",
                        "test_accuracy": 0.8,
                        "peer_role": "bridge",
                        "links": [{"target_finding_id": "x"}],
                        "design_dimensions": {"axis": "value"},
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                sys.modules,
                {"praxist.plugins.workflow_stages.research_loop.backend.tools.local_store": None},
            ):
                self.assertEqual(findings_ingest.ingest_findings_directory(findings_dir), 0)

            class FakeConn:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def execute(self, *_args):
                    raise RuntimeError("select")

            inserted = []

            def fake_get_conn(readonly=False):
                return FakeConn()

            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store.init_db",
                    return_value=None,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store._get_conn",
                    side_effect=fake_get_conn,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store.insert_finding",
                    side_effect=lambda row: inserted.append(row),
                ),
            ):
                self.assertEqual(findings_ingest.ingest_findings_directory(findings_dir), 1)
            self.assertEqual(inserted[0]["generation_id"], 2)

    def test_evaluation_tool_wait_and_generation_edge_paths_are_offline(self) -> None:
        from praxist.plugins.tools.evaluation_tools import adapter

        self.assertIsNone(adapter._gen_id_from_peer_id(""))
        self.assertIsNone(adapter._gen_id_from_peer_id("bad"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = root / "ready.txt"
            ready.write_text("prefix NEEDLE suffix", encoding="utf-8")
            with patch.dict(
                os.environ, {"LOCAL_STORE_DIR": str(root), "LOCAL_MODE": "true"}, clear=True
            ):
                payload = _text_payload(
                    asyncio.run(
                        adapter._handle_wait_for_file(
                            {
                                "path": str(ready),
                                "timeout_seconds": 1,
                                "poll_interval_seconds": 2,
                                "contains_text": "NEEDLE",
                                "mode": "all",
                            }
                        )
                    )
                )
            self.assertEqual(payload["status"], "ready")

            with patch.dict(os.environ, {"LOCAL_STORE_DIR": str(root)}, clear=True):
                self.assertTrue(
                    asyncio.run(
                        adapter._handle_wait_for_file(
                            {"path": str(root / "bad\npath"), "timeout_seconds": 1}
                        )
                    )["is_error"]
                )
                self.assertTrue(
                    asyncio.run(
                        adapter._handle_wait_for_file(
                            {
                                "path": str(ready),
                                "timeout_seconds": 1,
                                "contains_text": "x" * 1025,
                            }
                        )
                    )["is_error"]
                )
                timeout = _text_payload(
                    asyncio.run(
                        adapter._handle_wait_for_file(
                            {
                                "path": str(root / "missing.txt"),
                                "timeout_seconds": 1,
                                "poll_interval_seconds": "bad",
                                "min_bytes": "bad",
                                "mode": "bad",
                            }
                        )
                    )
                )
            self.assertEqual(timeout["status"], "timeout")
            with (
                patch.dict(os.environ, {"LOCAL_MODE": "true"}, clear=True),
                patch.object(adapter, "_sqlite_leaderboard", return_value="[]"),
            ):
                self.assertEqual(
                    _text_payload(
                        asyncio.run(
                            adapter._handle_get_leaderboard({"generation": "bad", "top_k": "bad"})
                        )
                    ),
                    [],
                )


class CoreContractCoverage95Test(unittest.TestCase):
    def test_modeling_error_classification_and_provider_edges_are_stable(self) -> None:
        from praxist.core import modeling
        from praxist.core.protocol import ModelProfile

        adapter = modeling.ModelProviderAdapter("model_provider:offline", api_format="fake")
        self.assertEqual(adapter.classify_error({"status": 401}), "auth_error")
        self.assertEqual(adapter.classify_error({"code": "insufficient_quota"}), "quota_exhausted")
        self.assertEqual(adapter.classify_error({"status": 429}), "rate_limited")
        self.assertEqual(adapter.classify_error({"code": "timeout"}), "timeout")
        self.assertEqual(adapter.classify_error({"status": 503}), "provider_unavailable")
        self.assertEqual(adapter.classify_error({"status": 400}), "invalid_request")
        self.assertEqual(adapter.classify_error({"error": "unexpected"}), "runtime_error")
        failed = adapter.normalize_result({"error": "bad key", "model": "m"})
        self.assertFalse(failed.success)
        self.assertEqual(failed.failover_reason, "runtime_error")
        ok = adapter.normalize_result({"text": "hello", "model": "m", "usage": {"tokens": 2}})
        self.assertTrue(ok.success)
        self.assertEqual(ok.usage["tokens"], 2.0)

        profile = ModelProfile(
            profile_id="p",
            provider_ref="model_provider:offline",
            model="m",
            api_format="fake",
            capability_tags=[],
            cost_tier="cheap",
            default_parameters={"temperature": 0},
        )
        call = adapter.build_call(
            profile,
            credential_ref=None,
            runtime_options={"max_tokens": 7},
        )
        self.assertEqual(call.parameters, {"temperature": 0, "max_tokens": 7})

        with patch.object(
            modeling,
            "_provider_contract",
            return_value={"compatible_model_patterns": ["ok-*"]},
        ):
            modeling.validate_model_for_provider("model_provider:x", "", registry=None)
            modeling.validate_model_for_provider("model_provider:x", "ok-model", registry=None)
            with self.assertRaisesRegex(ValueError, "not compatible"):
                modeling.validate_model_for_provider("model_provider:x", "bad-model", registry=None)

        with patch.object(
            modeling,
            "_provider_contract",
            return_value={
                "api_format": "anthropic_messages",
                "model_profiles": {"cheap_peer": "cheap-model"},
            },
        ):
            default = modeling.default_model_profile("model_provider:x")
            self.assertEqual(default.model, "cheap-model")
            self.assertIn("prompt_cache", default.capability_tags)

        class CustomProvider:
            api_format = "custom"

            def build_call(self, *_args, **_kwargs):
                return "call"

            def normalize_result(self, *_args, **_kwargs):
                return "result"

        class FakeRegistry:
            def require(self, kind: str, name: str):
                self.last = (kind, name)
                return CustomProvider()

        with patch.object(modeling, "require_execution_plugin", return_value=object()):
            self.assertIsInstance(
                modeling.provider_for_ref("model_provider:custom", registry=FakeRegistry()),
                CustomProvider,
            )

        with self.assertRaisesRegex(ValueError, "kind model_provider"):
            modeling._load_single_provider_registry("tool_server:x")

    def test_budget_guard_records_usage_and_degrades_accounting_failures(self) -> None:
        from praxist.core import execution_guards

        events: list[dict[str, Any]] = []

        class FakeTrajectoryWriter:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def emit(self, kind: str, **kwargs: Any) -> None:
                events.append({"kind": kind, **kwargs})

        class FakeLedger:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def require_active_grant(self, grant_id: str) -> dict[str, Any]:
                if grant_id == "bad":
                    raise ValueError("bad grant")
                if grant_id == "malformed":
                    return {"granted_budget": "not-a-budget"}
                return {
                    "request_id": "req-1",
                    "granted_budget": {
                        "tokens": 10,
                        "wall_clock_seconds": 1,
                        "gpu_hours": 2,
                    },
                }

            def append_usage(self, **kwargs: Any) -> dict[str, Any]:
                self.usage = kwargs
                return {"record_id": "usage-1"}

            def append_usage_unknown(self, **kwargs: Any) -> dict[str, Any]:
                self.unknown = kwargs
                return {"record_id": "unknown-1"}

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with (
                patch.object(execution_guards, "BudgetLedger", FakeLedger),
                patch.object(execution_guards, "TrajectoryWriter", FakeTrajectoryWriter),
            ):
                guard = execution_guards.BudgetedActionGuard(
                    run_dir=run_dir,
                    run_id="run",
                    stage_id="stage",
                    actor_ref="actor",
                    action_type="tool",
                    budget_grant_id="grant",
                )
                guard.start()
                report = guard.finish(
                    actual_usage={"tokens": 5, "negative": -1, "bad": "nan", "ignored": 9},
                    expected_units=["tokens", "wall_clock_seconds", "gpu_hours"],
                )
                self.assertTrue(report.recorded)
                self.assertEqual(report.usage_record_id, "usage-1")
                self.assertEqual(report.unknown_record_id, "unknown-1")
                self.assertEqual(report.actual_usage["tokens"], 5.0)
                self.assertIn("gpu_hours", report.unknown_units)

                warn = execution_guards.BudgetedActionGuard(
                    run_dir=run_dir,
                    run_id="run",
                    stage_id="stage",
                    actor_ref="actor",
                    action_type="tool",
                    budget_grant_id="bad",
                )
                warn.start()
                self.assertTrue(any(e["kind"] == "resource.action_budget_warning" for e in events))

                strict = execution_guards.BudgetedActionGuard(
                    run_dir=run_dir,
                    run_id="run",
                    stage_id="stage",
                    actor_ref="actor",
                    action_type="tool",
                    budget_grant_id="bad",
                    require_budget_grant=True,
                )
                with self.assertRaises(execution_guards.ResourceBudgetError):
                    strict.start()

                malformed = execution_guards.BudgetedActionGuard(
                    run_dir=run_dir,
                    run_id="run",
                    stage_id="stage",
                    actor_ref="actor",
                    action_type="tool",
                    budget_grant_id="malformed",
                )
                warning = malformed.finish(actual_usage={"tokens": 1})
                self.assertFalse(warning.recorded)
                self.assertIn("invalid granted budget", warning.warning or "")

        missing = execution_guards.BudgetedActionGuard(
            run_dir=None,
            run_id="run",
            stage_id="stage",
            actor_ref="actor",
            action_type="tool",
        ).finish()
        self.assertFalse(missing.recorded)
        self.assertIn("missing", missing.warning or "")

        with self.assertRaises(execution_guards.ResourceBudgetError):
            execution_guards.BudgetedActionGuard(
                run_dir=None,
                run_id="run",
                stage_id="stage",
                actor_ref="actor",
                action_type="tool",
                require_budget_grant=True,
            ).start()

    def test_role_skill_task_contracts_reject_escape_and_load_kb_assets(self) -> None:
        from praxist.core import role_skills

        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "task"
            role_root = task_root / "roles" / "builder"
            kb = role_root / "private_kb"
            kb.mkdir(parents=True)
            (role_root / "role.yaml").write_text(
                "role:\n"
                "  role_id: builder\n"
                "  display_name: Builder\n"
                "  role_kind: pi\n"
                "  tool_scope: [memory]\n"
                "assets:\n"
                "  - private_kb/*.md\n",
                encoding="utf-8",
            )
            (role_root / "skill.md").write_text(
                "# Builder\n\nFixed review questions:\n- What changed?\n- What failed?\n\nStop.",
                encoding="utf-8",
            )
            (kb / "note.md").write_text("private note", encoding="utf-8")
            skill = role_skills.load_role_skill("task_role:builder", task_project_path=task_root)
            self.assertEqual(skill.role_id, "builder")
            self.assertEqual(skill.tool_scope, ("memory",))
            self.assertEqual(skill.fixed_questions, ("What changed?", "What failed?"))
            self.assertEqual([p.name for p in skill.private_kb_paths], ["note.md"])
            self.assertEqual(skill.to_prompt_context()["display_name"], "Builder")

            for bad_ref in ("task_role:", "task_role:../x", "task_role:a/b"):
                with self.assertRaises(ValueError):
                    role_skills.load_role_skill(bad_ref, task_project_path=task_root)
            with (
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(ValueError, "require task_project_path"),
            ):
                role_skills.load_role_skill("task_role:builder")

            missing = task_root / "roles" / "missing_skill"
            missing.mkdir(parents=True)
            (missing / "role.yaml").write_text("role_id: missing_skill\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing skill.md"):
                role_skills.load_role_skill("task_role:missing_skill", task_project_path=task_root)

            bad_kb = task_root / "roles" / "bad_kb"
            bad_kb.mkdir(parents=True)
            (bad_kb / "role.yaml").write_text("private_kb: ../escape.md\n", encoding="utf-8")
            (bad_kb / "skill.md").write_text("skill", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stay inside"):
                role_skills.load_role_skill("task_role:bad_kb", task_project_path=task_root)

            bad_manifest = task_root / "roles" / "bad_manifest"
            bad_manifest.mkdir(parents=True)
            (bad_manifest / "role.yaml").write_text("- not\n- mapping\n", encoding="utf-8")
            (bad_manifest / "skill.md").write_text("skill", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest must be an object"):
                role_skills.load_role_skill("task_role:bad_manifest", task_project_path=task_root)

    def test_research_memory_ledgers_and_auditors_cover_contract_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            context_auditor,
            context_firewall,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.coverage_matrix import (
            CoverageMatrix,
        )

        self.assertFalse(context_auditor._has_source_id({}))
        self.assertTrue(context_auditor._has_source_id({"source_findings": [{"id": "f"}]}))
        self.assertEqual(context_auditor._has_overclaim_language(7), [])
        self.assertIn(
            "solved",
            context_auditor._has_overclaim_language("This is a solved problem"),
        )
        agenda = {
            "consensus_actions": [{"id": "a", "claim_or_hypothesis": "universal improvement"}],
            "cross_peer_hypotheses": [{"id": "h", "supports": ["f1"], "claim": "ok"}],
            "retired_claims": [{"id": "r", "boundary": "only local"}],
            "peer_contracts": {"peer1": {"role": "Bridge", "notes": "no check"}},
        }
        pack = {
            "private_packs": {
                "skeptic": [{"quality": {"is_negative": False}}],
                "builder": [{"quality": {"is_negative": False}}],
            }
        }
        report = context_auditor.audit_agenda(
            agenda,
            pack,
            {"builder": {}},
            completed_gen_id=2,
        )
        self.assertFalse(report.pass_)
        self.assertGreaterEqual(len(report.warnings), 3)
        self.assertIn("negative evidence ratio", "\n".join(report.blocking_issues))

        first = context_auditor.audit_agenda(
            {"peer_contracts": {}},
            {"private_packs": {"builder": [{"quality": {"is_negative": False}}]}},
            {},
            completed_gen_id=0,
        )
        self.assertTrue(any("gen 0" in warning for warning in first.warnings))

        with tempfile.TemporaryDirectory() as tmp:
            matrix = CoverageMatrix(Path(tmp))
            self.assertIsNone(matrix.query_grid("family", "lr"))
            matrix.record_grid_point("family", "lr", 0.1, seed_count=1, source_evidence_id="f1")
            matrix.record_grid_point("family", "lr", 0.2, seed_count=3, source_evidence_id="f2")
            grid = matrix.query_grid("family", "lr")
            self.assertEqual(grid["values_tested"], [0.1, 0.2])
            self.assertEqual(grid["seed_counts"]["0.2"], 3)
            self.assertFalse(matrix.is_bridge_covered("a", "b", "scale", min_points=1))
            matrix.record_bridge_point("b", "a", "scale", {"x": 1}, source_evidence_id="b1")
            matrix.record_bridge_point("a", "b", "scale", {"x": 2}, source_evidence_id="b2")
            self.assertTrue(matrix.is_bridge_covered("a", "b", "scale", min_points=2))
            self.assertEqual(
                matrix.query_bridge("b", "a", "scale")["variant_pair"],
                ["a", "b"],
            )
            self.assertEqual(len(matrix.all()), 2)

        class Pack:
            pack_id = "p"
            built_at = "now"
            panel_mode = "full"
            target_decisions = []
            shared_core = {
                "keep": "x" * 5000,
                "negative_evidence_digest": "n" * 5000,
                "role_performance": "r" * 5000,
                "findings_summary": "f" * 5000,
            }
            private_packs = {
                "builder": [
                    {"interpretation": {"short": "x" * 2000}, "quality": {}},
                    *({"interpretation": {"short": str(i)}, "quality": {}} for i in range(30)),
                ]
            }
            audit = {"ok": True}

        fitted = context_firewall.fit_pack_to_budget(Pack(), "mini")
        self.assertLessEqual(len(fitted["private_packs"]["builder"]), 2)
        fitted_full = context_firewall.fit_pack_to_budget(Pack(), "full")
        self.assertIn("audit", fitted_full)
        self.assertTrue(context_firewall.forbid_raw_history({"x": {"raw_history": "secret"}}))
        self.assertFalse(context_firewall.forbid_raw_history({"x": "summary"}))

    def test_http_and_orchestrator_edges_are_offline_and_deterministic(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import orchestrator_status
        from praxist.plugins.workflow_stages.research_loop.backend.tools import http_utils

        with patch.dict(os.environ, {"SERVER_URL": "http://example.test/"}, clear=True):
            self.assertEqual(http_utils.get_server_url(), "http://example.test")
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "SERVER_URL"),
        ):
            http_utils.get_server_url()
        self.assertEqual(http_utils.validate_safe_identifier(" abc-123_", "name"), "abc-123_")
        for bad in ("", "../x", "a/b", "x y"):
            with self.assertRaises(ValueError):
                http_utils.validate_safe_identifier(bad, "name")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inside = root / "inside.txt"
            inside.write_text("x", encoding="utf-8")
            self.assertEqual(
                http_utils.validate_safe_path(str(inside), "path", str(root)),
                str(inside.resolve()),
            )
            with self.assertRaisesRegex(ValueError, "within"):
                http_utils.validate_safe_path(str(root.parent), "path", str(root))

            now = "2026-01-01T00:00:00+00:00"

            def snapshot() -> orchestrator_status.OrchestratorSnapshot:
                return orchestrator_status.OrchestratorSnapshot(
                    run_started_at=now,
                    updated_at=now,
                    run_dir=str(root),
                    task_id="task",
                    task_name="Task",
                    current_generation=1,
                    max_generations=2,
                    cohort_size=3,
                    strategy="auto",
                    generations_completed=1,
                )

            with self.assertRaises(ValueError):
                orchestrator_status.OrchestratorStatusWriter(root, snapshot, interval_seconds=0)
            writer = orchestrator_status.OrchestratorStatusWriter(
                root, snapshot, interval_seconds=999
            )
            writer.start()
            writer.start()
            writer.stop(exit_condition="completed")
            writer.stop(exit_condition="error")
            self.assertEqual(
                json.loads(writer.final_status_path.read_text(encoding="utf-8"))["exit_condition"],
                "completed",
            )

            failing = orchestrator_status.OrchestratorStatusWriter(
                root / "failing",
                lambda: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
                interval_seconds=999,
            )
            failing._write_once("in_progress")
            status = json.loads(failing.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["current_generation"], -1)
            self.assertIn("snapshot failed", status["last_snapshot_error"])
            failing.stop(exit_condition="error")

        self.assertEqual(
            orchestrator_status.describe_promotion_criteria(2, "score", "loss", "minimize"),
            "promote top-2 findings per generation by score (loss ↓)",
        )
        self.assertEqual(
            orchestrator_status.describe_promotion_blocker(
                variants_with_primary_metric=0,
                variants_above_baseline=0,
                promote_top_k=2,
            ),
            "no variants have reported the primary metric yet",
        )
        self.assertIn(
            "none above baseline",
            orchestrator_status.describe_promotion_blocker(
                variants_with_primary_metric=1,
                variants_above_baseline=0,
                promote_top_k=2,
            ),
        )
        self.assertIn(
            "only 1",
            orchestrator_status.describe_promotion_blocker(
                variants_with_primary_metric=2,
                variants_above_baseline=1,
                promote_top_k=2,
            ),
        )
        self.assertEqual(
            orchestrator_status.describe_promotion_blocker(
                variants_with_primary_metric=2,
                variants_above_baseline=2,
                promote_top_k=2,
            ),
            "",
        )

    def test_agent_helper_contracts_cover_provider_env_and_stop_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        class BadPath:
            def exists(self) -> bool:
                raise OSError("fs")

        checker = agent.StopChecker(max_runtime=999, stop_signal_path=BadPath())
        with patch.object(agent.time, "time", return_value=checker.start_time + 1000):
            self.assertEqual(checker.check(), agent.StopReason.TIMEOUT)
        checker.record_error()
        checker.record_success()
        self.assertEqual(checker.consecutive_errors, 0)

        env = {
            "PRAXIST_MODEL_PROVIDER_REF": "model_provider:openrouter",
            "ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1/",
            "ANTHROPIC_AUTH_TOKEN": "tok",
            "OPENROUTER_API_KEY": "openrouter-native-token",
            "OPENAI_API_KEY": "must-not-leak",
            "PRAXIST_RUN_ID": "run",
        }
        with patch.dict(os.environ, env, clear=True):
            scoped = agent._scoped_legacy_provider_env()
        self.assertEqual(scoped["ANTHROPIC_BASE_URL"], "https://openrouter.ai/api")
        self.assertEqual(scoped["ANTHROPIC_AUTH_TOKEN"], "tok")
        self.assertEqual(scoped["OPENROUTER_API_KEY"], "openrouter-native-token")
        self.assertNotIn("OPENAI_API_KEY", scoped)
        self.assertEqual(scoped["PRAXIST_RUN_ID"], "run")

        with patch.dict(
            os.environ,
            {
                "PRAXIST_MODEL_PROVIDER_REF": "model_provider:fake_provider",
                "ANTHROPIC_API_KEY": "secret",
                "PRAXIST_RUN_DIR": "/tmp/run",
            },
            clear=True,
        ):
            self.assertEqual(
                agent._scoped_legacy_provider_env(),
                {
                    "PRAXIST_MODEL_PROVIDER_REF": "model_provider:fake_provider",
                    "PRAXIST_RUN_DIR": "/tmp/run",
                },
            )

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                agent._legacy_model_provider_ref("deepseek-chat"),
                "model_provider:deepseek_alias",
            )
            self.assertEqual(
                agent._legacy_model_provider_ref("gpt-4.1"),
                "model_provider:openai_compatible",
            )
            self.assertEqual(
                agent._legacy_model_provider_ref("openrouter/model"),
                "model_provider:openrouter",
            )

        self.assertTrue(agent.AutonomousAgentLoop._session_was_productive(None))
        self.assertFalse(
            agent.AutonomousAgentLoop._session_was_productive(
                SimpleNamespace(iteration_count="bad")
            )
        )
        self.assertFalse(agent.AutonomousAgentLoop._session_was_bootstrap_wait(None))
        self.assertFalse(
            agent.AutonomousAgentLoop._session_was_bootstrap_wait(
                SimpleNamespace(success=True, iteration_count="bad", output={})
            )
        )
        self.assertFalse(
            agent.AutonomousAgentLoop._session_was_bootstrap_wait(
                SimpleNamespace(success=True, iteration_count=0, output={"text_outputs": "bad"})
            )
        )
        self.assertEqual(
            agent._runtime_final_payload(
                SimpleNamespace(events=[SimpleNamespace(type="message", payload={"x": 1})])
            ),
            {},
        )

    def test_http_post_get_fallbacks_are_offline(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import http_utils

        class FakeResponse:
            def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
                self.status_code = status_code
                self._payload = payload

            def json(self) -> dict[str, Any]:
                return self._payload

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError(f"http {self.status_code}")

        class FakeAsyncClient:
            def __init__(self, timeout: int) -> None:
                self.timeout = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
                self.last_post = (url, json, headers)
                return FakeResponse(400, {"error": "bad"})

            async def get(self, url: str, params: dict[str, Any] | None, headers: dict[str, str]):
                self.last_get = (url, params, headers)
                return FakeResponse(200, {"ok": True})

        with (
            patch.object(http_utils, "HAS_HTTPX", True),
            patch.object(
                http_utils,
                "httpx",
                SimpleNamespace(AsyncClient=FakeAsyncClient),
                create=True,
            ),
        ):
            self.assertEqual(
                asyncio.run(http_utils.async_http_post("http://x", {"a": 1}, headers={"X": "Y"})),
                {"error": "bad"},
            )
            self.assertEqual(
                asyncio.run(http_utils.async_http_get("http://x", params={"q": "v"})),
                {"ok": True},
            )

        class FakeRequests:
            def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
                return FakeResponse(200, {"posted": kwargs["json"]})

            def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
                return FakeResponse(500, {"error": "server"})

        with (
            patch.object(http_utils, "HAS_HTTPX", False),
            patch.object(http_utils, "HAS_REQUESTS", True),
            patch.object(http_utils, "requests", FakeRequests(), create=True),
        ):
            self.assertEqual(
                asyncio.run(http_utils.async_http_post("http://x", {"a": 2})),
                {"posted": {"a": 2}},
            )
            with self.assertRaisesRegex(RuntimeError, "http 500"):
                asyncio.run(http_utils.async_http_get("http://x"))
        with (
            patch.object(http_utils, "HAS_HTTPX", False),
            patch.object(http_utils, "HAS_REQUESTS", False),
            self.assertRaises(ImportError),
        ):
            asyncio.run(http_utils.async_http_get("http://x"))

    def test_deliver_and_task_spec_edges_are_filesystem_local(self) -> None:
        from praxist import deliver, task_spec

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            self.assertIsNone(deliver.load_run_summary(run_dir))
            self.assertIsNone(deliver.load_frontier_manifest(run_dir))
            self.assertEqual(deliver.load_all_findings(run_dir), [])
            self.assertIn("No metrics recorded", deliver.generate_metrics_table([]))

            shared = run_dir / "shared_findings"
            shared.mkdir()
            (shared / "bad.json").write_text("{bad", encoding="utf-8")
            (shared / "good.json").write_text('{"finding_id":"f1"}', encoding="utf-8")
            self.assertEqual(deliver.load_all_findings(run_dir), [{"finding_id": "f1"}])

            (run_dir / "nested").mkdir()
            (run_dir / "nested" / "metrics_log.jsonl").write_text(
                '{"variant_name":"v","metrics":"{bad","generation_id":1}\n'
                '{"variant_name":"v","metrics":"{\\"score\\": 1}","generation_id":1}\n',
                encoding="utf-8",
            )
            metrics = deliver.load_all_metrics(run_dir)
            table = deliver.generate_metrics_table(metrics, "score")
            self.assertIn("| v | 1 | ? |", table)

            summary = deliver.generate_executive_summary(
                {"task_name": "Task", "task_id": "t", "generations_completed": 1},
                {
                    "primary_metric": "score",
                    "metric_direction": "maximize",
                    "cumulative_top": [
                        {"generation_id": 0, "variant_name": "v", "metric_value": 1, "metrics": {}}
                    ],
                    "generations": {"0": [{"metric_value": 1, "variant_name": "v"}]},
                },
                [{"finding_type": "result", "generation_id": 0}],
                metrics,
            )
            self.assertIn("Best Results", summary)

            frontier = run_dir / "frontier"
            frontier.mkdir()
            tar_path = frontier / "best_snapshot.tar.gz"
            source = root / "source.txt"
            source.write_text("payload", encoding="utf-8")
            with tarfile.open(tar_path, "w:gz") as tf:
                tf.add(source, arcname="safe/source.txt")
            out_dir = root / "deliverables"
            self.assertEqual(deliver.extract_frontier_snapshots(run_dir, out_dir), 1)
            self.assertTrue((out_dir / "code" / "best_snapshot" / "safe" / "source.txt").exists())

            report_dir = root / "report"
            report_dir.mkdir()
            (report_dir / "a.txt").write_text("a", encoding="utf-8")
            (report_dir / "data").mkdir()
            readme = deliver._generate_readme(
                report_dir,
                {"task_name": "Task", "generations_completed": 1},
                {"cumulative_top": [{"metric_value": 1, "variant_name": "v"}]},
                [{"id": "f"}],
                metrics,
            )
            self.assertIn("data/", readme)

            spec_path = root / "task.yaml"
            spec_path.write_text(
                "task_id: t\n"
                "task_name: T\n"
                "description_file: missing.md\n"
                "research_direction: fallback\n"
                "evaluation:\n"
                "  diversity_dimensions: bad\n"
                "  must_explore_axes: bad\n"
                "  anchor_metrics: bad\n"
                "multi_pi: bad\n"
                "research_memory: bad\n"
                "synthesis_trigger:\n"
                "  max_interval_minutes: 1000\n",
                encoding="utf-8",
            )
            loaded = task_spec.load_task_spec(str(spec_path))
            self.assertEqual(loaded.get_description(), "fallback")
            self.assertEqual(loaded.description_path, root / "missing.md")
            self.assertEqual(loaded.get_prompt_task_path(), root / "prompt_task.jinja2")
            self.assertEqual(loaded.evaluation.anchor_metrics, [])
            with self.assertRaises(FileNotFoundError):
                task_spec.load_task_spec(str(root / "missing.yaml"))

    def test_cli_support_edges_do_not_require_real_runtime(self) -> None:
        from praxist import run as cli

        class RunnerObject:
            def __init__(self) -> None:
                self.called = False

            def run(self, **kwargs: Any) -> dict[str, bool]:
                self.called = True
                return {"ran": True, "kwargs": bool(kwargs)}

        project = SimpleNamespace(task_ref="task:x")
        runner = RunnerObject()
        self.assertEqual(
            cli._task_runner_for_capability(project, "testing.fake", lambda _p: runner)(
                workspace=Path(".")
            ),
            {"ran": True, "kwargs": True},
        )
        self.assertEqual(
            cli._task_runner_for_capability(project, "testing.fake", lambda _p: lambda **_: "ok")(),
            "ok",
        )
        with self.assertRaises(TypeError):
            cli._task_runner_for_capability(project, "testing.fake", lambda _p: object())

        with (
            patch(
                "praxist.infrastructure.execute_autonomous.main",
                side_effect=lambda: None,
            ) as peer_main,
            patch.dict(os.environ, {}, clear=True),
        ):
            cli.cmd_peer(
                SimpleNamespace(
                    peer_id="peer",
                    generation_id=3,
                    max_runtime=9,
                    prompt_file="prompt.md",
                    model="m",
                    local=True,
                )
            )
            self.assertEqual(os.environ["PEER_ID"], "peer")
            self.assertEqual(os.environ["GENERATION_ID"], "3")
            self.assertEqual(os.environ["LOCAL_MODE"], "true")
        peer_main.assert_called_once()

        with self.assertRaises(SystemExit) as cm:
            cli.cmd_server(SimpleNamespace())
        self.assertEqual(cm.exception.code, 1)

        out = io.StringIO()
        with (
            patch("praxist.core.replay.dry_run", return_value={"success": True, "dry": True}),
            contextlib.redirect_stdout(out),
        ):
            cli.cmd_replay(
                SimpleNamespace(
                    run_dir="/tmp/run",
                    mode="dry",
                    strict_tail=False,
                    allow_plugin_drift=False,
                    locked=False,
                )
            )
        self.assertIn('"dry": true', out.getvalue())

    def test_cohort_runner_and_generation_boundary_contracts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            cohort_runner,
            generation_boundary,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_dir = root / "gen_0"
            gen_dir.mkdir()
            stale = gen_dir / "STOP_SIGNAL"
            stale.write_text("old", encoding="utf-8")

            class FakePeer:
                def __init__(self, **kwargs: Any) -> None:
                    self.kwargs = kwargs

                async def run(self) -> dict[str, Any]:
                    return {"peer_id": self.kwargs["peer_id"], "success": True}

            class FakeTrigger:
                fired = False

                def __init__(self, **kwargs: Any) -> None:
                    self.kwargs = kwargs

                async def wait_until_fire(self, *, abort_event: asyncio.Event) -> None:
                    raise RuntimeError("trigger")

                async def evaluate_async(self):
                    return SimpleNamespace(fired=True)

                def fire(self, snap: Any) -> None:
                    self.fired = True

                def write_postgen_marker(self, snap: Any) -> None:
                    (gen_dir / "STOP_SIGNAL_POSTGEN").write_text("postgen", encoding="utf-8")

            class FakeLoop:
                run_dir = root
                workspace = root
                base_template = root / "base.j2"
                task_prompt_path = root / "task.j2"
                gen_template = root / "gen.j2"
                findings_dir = root / "findings"
                model = "m"
                local_mode = True
                mcp_servers: list[str] = []
                plugin_registry = None
                _peer_allowed_tools = ["tool"]
                _findings_sync = SimpleNamespace(
                    sync_once=lambda: (_ for _ in ()).throw(RuntimeError("sync"))
                )
                task_spec = SimpleNamespace(
                    generation_policy=SimpleNamespace(
                        cohort_size=2,
                        per_generation_hours=0.001,
                    ),
                    synthesis_trigger=SimpleNamespace(
                        enabled=True,
                        min_findings=1,
                        min_interval_minutes=1,
                        max_interval_minutes=2,
                        min_contributing_peers=1,
                        poll_interval_seconds=1,
                    ),
                    agent=SimpleNamespace(premium_mode=False),
                )

                def _build_prompt_context(
                    self, gen_id: int, i: int, cohort_size: int
                ) -> dict[str, Any]:
                    return {"gen_id": gen_id, "i": i, "cohort_size": cohort_size}

                def _persist_prompt_layout_artifacts(self, **kwargs: Any) -> dict[str, Any]:
                    return kwargs["manifest"]

            with (
                patch.object(cohort_runner, "AutonomousAgentLoop", FakePeer),
                patch.object(
                    cohort_runner,
                    "resolve_prompt_with_layout",
                    return_value=("prompt", {"layout_hash": "h"}),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
            ):
                results = asyncio.run(cohort_runner.run_generation_cohort(FakeLoop(), 0))
            self.assertEqual([r["success"] for r in results], [True, True])
            self.assertTrue((gen_dir / "generation_results.json").exists())

            findings = [{"variant_name": "v", "metrics": {"diversity_overlap_status": "no_data"}}]
            loop = SimpleNamespace(
                _strategy_for_gen=lambda _gen: "exploit",
                frontier=SimpleNamespace(get_summary=lambda: []),
            )
            self.assertIs(
                generation_boundary._annotate_diversity_overlap(loop, gen_id=1, findings=findings),
                findings,
            )

            loop = SimpleNamespace(
                _graph_maintainer=None,
                _strategy_for_gen=lambda _gen: "pi_directed",
                frontier=SimpleNamespace(get_summary=lambda: [{"variant_name": "anchor"}]),
                task_spec=SimpleNamespace(evaluation=SimpleNamespace(diversity_dimensions=[])),
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.frontier.annotate_findings_with_diversity_overlap",
                return_value=[
                    {
                        "variant_name": "clone",
                        "metrics": {
                            "diversity_overlap_status": "clone",
                            "diversity_overlap_count": 4,
                            "diversity_overlap_total": 4,
                            "diversity_overlap_fraction": 1.0,
                        },
                    },
                    {
                        "variant_name": "narrow",
                        "metrics": {
                            "diversity_overlap_status": "narrow",
                            "diversity_overlap_count": 2,
                            "diversity_overlap_total": 4,
                            "diversity_overlap_fraction": 0.5,
                        },
                    },
                ],
            ):
                annotated = generation_boundary._annotate_diversity_overlap(
                    loop, gen_id=1, findings=findings
                )
            self.assertEqual(len(annotated), 2)

            generation_boundary._sync_graph_before_next_generation(loop, gen_id=0)
            graph_loop = SimpleNamespace(
                _graph_maintainer=SimpleNamespace(
                    sync_once_blocking=lambda timeout: {"status": "timeout"}
                )
            )
            generation_boundary._sync_graph_before_next_generation(graph_loop, gen_id=0)
            graph_loop._graph_maintainer = SimpleNamespace(
                sync_once_blocking=lambda timeout: (_ for _ in ()).throw(RuntimeError("graph"))
            )
            generation_boundary._sync_graph_before_next_generation(graph_loop, gen_id=0)

            async def successful_pi(**_kwargs: Any):
                return SimpleNamespace(
                    success=True, next_gen_id=1, agenda_path="a.yaml", duration_seconds=2
                )

            boundary_loop = SimpleNamespace(
                _collect_findings_for_generation=lambda _gen: [
                    {"variant_name": "v", "metrics": {}}
                ],
                _strategy_for_gen=lambda _gen: "explore",
                frontier=SimpleNamespace(
                    get_summary=lambda: [],
                    promote=lambda _gen, _findings: [{"finding_id": "f"}],
                ),
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(max_generations=3),
                    research_memory=SimpleNamespace(enabled=True),
                    evaluation=SimpleNamespace(diversity_dimensions=[]),
                ),
                _update_research_memory_post_gen=lambda **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("memory")
                ),
                _graph_maintainer=None,
            )
            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    boundary_loop,
                    gen_id=0,
                    pi_agent=SimpleNamespace(run=successful_pi),
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )

            async def failed_pi(**_kwargs: Any):
                return SimpleNamespace(success=False, next_gen_id=1, error="bad")

            asyncio.run(
                generation_boundary.complete_generation_boundary(
                    boundary_loop,
                    gen_id=0,
                    pi_agent=SimpleNamespace(run=failed_pi),
                    pi_cfg=SimpleNamespace(strict=False),
                )
            )
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    generation_boundary.complete_generation_boundary(
                        boundary_loop,
                        gen_id=0,
                        pi_agent=SimpleNamespace(run=failed_pi),
                        pi_cfg=SimpleNamespace(strict=True),
                    )
                )

    def test_c5_and_legacy_materializers_cover_import_surfaces(self) -> None:
        from praxist.plugins.workflow_stages.research_loop import (
            c5_materializer,
            legacy_output_materializer,
        )

        class FakeTrajectory:
            def __init__(self) -> None:
                self.events: list[dict[str, Any]] = []

            def emit(self, kind: str, **kwargs: Any) -> dict[str, Any]:
                event = {"event_id": f"evt-{len(self.events)}", "kind": kind, **kwargs}
                self.events.append(event)
                return event

        class FakeArtifacts:
            def __init__(self) -> None:
                self.records: list[dict[str, Any]] = []

            def persist_json(
                self, artifact_type: str, logical_path: str, payload: Any, **kwargs: Any
            ):
                return self._record(artifact_type, logical_path, payload, **kwargs)

            def persist_text(
                self, artifact_type: str, logical_path: str, payload: str, **kwargs: Any
            ):
                return self._record(artifact_type, logical_path, payload, **kwargs)

            def _record(self, artifact_type: str, logical_path: str, payload: Any, **kwargs: Any):
                record = {
                    "artifact_id": f"art-{len(self.records)}",
                    "artifact_type": artifact_type,
                    "logical_path": logical_path,
                    "payload": payload,
                    **kwargs,
                }
                self.records.append(record)
                return record

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            ledger_dir = run_dir / "research_memory" / "ledgers"
            ledger_dir.mkdir(parents=True)
            (ledger_dir / "bad.yaml").write_text(":\n", encoding="utf-8")
            (ledger_dir / "list.yaml").write_text("- bad\n", encoding="utf-8")
            (ledger_dir / "hypothesis_ledger.yaml").write_text(
                "ledger_name: hypothesis_ledger\n"
                "entries:\n"
                "  - id: h1\n"
                "    created_at: t\n"
                "    data:\n"
                "      confidence: 0.8\n"
                "      supports:\n"
                "        - finding_id: f1\n"
                "  - not-a-dict\n",
                encoding="utf-8",
            )
            graph_dir = run_dir / "graph"
            graph_dir.mkdir()
            (graph_dir / "graph_health.json").write_text('{"ok": true}', encoding="utf-8")
            (graph_dir / "unlinked_recent_findings.json").write_text("{bad", encoding="utf-8")
            (graph_dir / "graph.html").write_text("<html></html>", encoding="utf-8")
            conn = sqlite3.connect(run_dir / "shared_store.db")
            conn.execute(
                "CREATE TABLE finding_edges (edge_id TEXT, src_finding_id TEXT, "
                "dst_finding_id TEXT, edge_type TEXT, confidence REAL, created_by TEXT, "
                "created_at TEXT, rationale TEXT, provenance TEXT)"
            )
            conn.execute(
                "INSERT INTO finding_edges VALUES (?,?,?,?,?,?,?,?,?)",
                ("e1", "f1", "f2", "supports", 0.7, "rule", "now", "why", "{bad"),
            )
            conn.commit()
            conn.close()

            prepared = SimpleNamespace(run_dir=run_dir, run_id="run", task_ref="task:x")
            trajectory = FakeTrajectory()
            artifacts = FakeArtifacts()
            counts = c5_materializer.materialize_legacy_c5_views(
                prepared,
                {},
                trajectory=trajectory,
                artifacts=artifacts,
            )
            self.assertEqual(counts["research_memory_record_count"], 1)
            self.assertEqual(counts["graph_edge_count"], 1)
            self.assertEqual(counts["graph_artifact_count"], 2)
            self.assertTrue((run_dir / "memory" / "research_memory.jsonl").exists())
            self.assertTrue((run_dir / "memory" / "graph_edges.jsonl").exists())

            adapter = c5_materializer.LegacyRunDirAdapter(run_dir / "missing")
            self.assertEqual(adapter.collect_research_memory_entries(), ([], {}))
            self.assertEqual(adapter.collect_graph_edges(), [])
            self.assertEqual(adapter.collect_graph_artifacts(), [])
            with patch.object(c5_materializer.sqlite3, "connect", side_effect=sqlite3.Error("db")):
                self.assertEqual(
                    c5_materializer.LegacyRunDirAdapter(run_dir).collect_graph_edges(), []
                )

            self.assertEqual(c5_materializer._entity_type_for_ledger("claim_ledger"), "claim")
            self.assertIsNone(c5_materializer._optional_number(True))
            self.assertIsNone(c5_materializer._optional_number("1"))
            self.assertEqual(c5_materializer._optional_number(2), 2.0)
            self.assertEqual(
                c5_materializer._source_finding_ids_from_refs(
                    [{"id": "a", "finding_ids": ["b", "a"]}]
                ),
                ["a", "b"],
            )
            self.assertTrue(
                c5_materializer._run_relative(run_dir, Path("/definitely/outside")).startswith("/")
            )

            frontier_dir = run_dir / "frontier"
            frontier_dir.mkdir(exist_ok=True)
            (frontier_dir / "frontier_manifest.json").write_text(
                '{"generations":{"bad":[{"id":"fb"}],"1":[{"finding_id":"f1"}]}}',
                encoding="utf-8",
            )
            self.assertEqual(
                [
                    item.get("finding_id") or item.get("id")
                    for item in legacy_output_materializer._collect_legacy_frontier_summary(
                        run_dir, {}
                    )
                ],
                ["f1", "fb"],
            )
            self.assertEqual(
                legacy_output_materializer._collect_legacy_frontier_summary(
                    run_dir, {"frontier_summary": [{"id": "direct"}, "bad"]}
                ),
                [{"id": "direct"}],
            )
            self.assertEqual(
                legacy_output_materializer._finding_from_frontier_entry(
                    {"id": "f", "metric_name": "score", "metric_value": 1, "variant_name": "v"}
                )["metrics"],
                {"score": 1},
            )
            self.assertEqual(
                legacy_output_materializer._producer_ref(
                    SimpleNamespace(peer_role_ref="role:peer"), {"peer_role": "builder"}
                ),
                "role:peer/builder",
            )
            self.assertEqual(legacy_output_materializer._supersedes({"updates": "old"}), ["old"])
            self.assertEqual(
                legacy_output_materializer._compact_legacy_payload({"id": "f", "x": 1}), {"id": "f"}
            )

            matching_event = {
                "event_id": "agent-1",
                "agent_name": "gen0_peer0-session",
                "payload": {
                    "output": {
                        "tool_uses": [
                            {
                                "name": "share_finding",
                                "input": {
                                    "peer_id": "gen0_peer0",
                                    "title": "A title",
                                    "metrics": '{"score": 1}',
                                },
                            }
                        ]
                    }
                },
            }
            finding = {"peer_id": "gen0_peer0", "title": "A   title", "metrics": {"score": 1}}
            self.assertEqual(
                legacy_output_materializer._source_event_ids_for_finding([matching_event], finding),
                ["agent-1"],
            )
            self.assertFalse(
                legacy_output_materializer._share_finding_tool_input_matches(
                    {"name": "other", "input": {}}, finding
                )
            )
            self.assertFalse(
                legacy_output_materializer._share_finding_tool_input_matches(
                    {"name": "share_finding", "input": []}, finding
                )
            )
            self.assertFalse(
                legacy_output_materializer._share_finding_tool_input_matches(
                    {"name": "share_finding", "input": {"peer_id": "other"}}, finding
                )
            )
            self.assertFalse(
                legacy_output_materializer._share_finding_tool_input_matches(
                    {
                        "name": "share_finding",
                        "input": {"peer_id": "gen0_peer0", "title": "different"},
                    },
                    finding,
                )
            )
            self.assertIsNone(legacy_output_materializer._parse_metrics("[1]"))
            weak_ids, quality, warning = (
                legacy_output_materializer._source_event_ids_for_finding_or_import(
                    trajectory,
                    [],
                    {"id": "weak"},
                    reason="weak",
                )
            )
            self.assertEqual(quality, "legacy_weak")
            self.assertEqual(warning, "weak")
            self.assertEqual(len(weak_ids), 1)

    def test_parity_checker_contracts_cover_pass_warn_fail_surfaces(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import parity

        legacy_empty = {
            "finding_ids": set(),
            "frontier_ids": set(),
            "research_memory_entry_count": 0,
            "graph_edge_count": 0,
            "graph_artifact_names": set(),
            "postgen_prompt_paths": [],
            "prompt_texts": {},
            "agenda_paths": [],
        }
        canonical_empty = {
            "finding_ids": set(),
            "frontier_ids": set(),
            "research_memory": [],
            "graph_edges": [],
            "artifact_types": set(),
            "artifact_logical_paths": set(),
            "budget": [],
        }
        self.assertTrue(parity._check_replay({"success": False, "errors": ["bad"]}).failed)
        self.assertTrue(
            parity._check_legacy_findings_materialized(legacy_empty, canonical_empty).warning
        )
        self.assertTrue(parity._check_frontier_materialized(legacy_empty, canonical_empty).warning)
        self.assertTrue(
            parity._check_research_memory_materialized(legacy_empty, canonical_empty).warning
        )
        self.assertTrue(
            parity._check_graph_edges_materialized(legacy_empty, canonical_empty).warning
        )
        self.assertTrue(
            parity._check_graph_artifacts_materialized(legacy_empty, canonical_empty).warning
        )

        legacy = {
            **legacy_empty,
            "finding_ids": {"f1"},
            "frontier_ids": {"f1"},
            "research_memory_entry_count": 2,
            "graph_edge_count": 1,
            "graph_artifact_names": {"graph.html"},
        }
        canonical = {
            **canonical_empty,
            "finding_ids": set(),
            "frontier_ids": set(),
            "research_memory": [{}],
            "graph_edges": [],
            "artifact_types": set(),
        }
        self.assertTrue(parity._check_legacy_findings_materialized(legacy, canonical).failed)
        self.assertTrue(parity._check_frontier_materialized(legacy, canonical).failed)
        self.assertTrue(parity._check_research_memory_materialized(legacy, canonical).failed)
        self.assertTrue(parity._check_graph_edges_materialized(legacy, canonical).failed)
        self.assertTrue(parity._check_graph_artifacts_materialized(legacy, canonical).failed)

        canonical_ok = {
            **canonical_empty,
            "finding_ids": {"f1"},
            "frontier_ids": {"f1"},
            "research_memory": [{}, {}],
            "graph_edges": [{}],
            "artifact_types": {"graph_materialized_artifact"},
            "artifact_logical_paths": {"graph/legacy/graph.html"},
            "budget": [{"kind": "usage", "action_type": "gpu_slot"}],
        }
        self.assertFalse(parity._check_legacy_findings_materialized(legacy, canonical_ok).failed)
        self.assertFalse(parity._check_frontier_materialized(legacy, canonical_ok).failed)
        self.assertFalse(parity._check_research_memory_materialized(legacy, canonical_ok).failed)
        self.assertFalse(parity._check_graph_edges_materialized(legacy, canonical_ok).failed)
        self.assertFalse(parity._check_graph_artifacts_materialized(legacy, canonical_ok).failed)
        self.assertFalse(parity._check_resource_guard_usage(canonical_ok, strict=True).failed)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text('{"task_ref":"task:x"}', encoding="utf-8")
            self.assertFalse(parity._check_task_ref(run_dir).failed)
            (run_dir / "run.json").write_text('{"task_ref":"bad"}', encoding="utf-8")
            self.assertTrue(parity._check_task_ref(run_dir).warning)

            prompt = run_dir / "gen_1" / "gen1_peer0_prompt.md"
            prompt.parent.mkdir(parents=True)
            prompt.write_text(
                "Graph-surfaced context frontier f1 "
                "mcp__finding-graph-query__get_unlinked_recent_findings",
                encoding="utf-8",
            )
            legacy_prompt = {
                **legacy,
                "postgen_prompt_paths": [prompt],
                "prompt_texts": {prompt: prompt.read_text(encoding="utf-8")},
            }
            self.assertFalse(
                parity._check_prompt_guidance_surfaces(
                    legacy_prompt, canonical_ok, strict=True
                ).failed
            )
            prompt.write_text("missing guidance", encoding="utf-8")
            legacy_prompt["prompt_texts"] = {prompt: "missing guidance"}
            self.assertTrue(
                parity._check_prompt_guidance_surfaces(
                    legacy_prompt, canonical_ok, strict=True
                ).failed
            )

            agenda = run_dir / "agendas" / "research_agenda_gen1.yaml"
            agenda.parent.mkdir()
            agenda.write_text("peer_contracts: {}\n", encoding="utf-8")
            self.assertFalse(
                parity._check_panel_agenda_surface(
                    {**legacy_empty, "agenda_paths": [agenda]}, strict=True
                ).failed
            )
            agenda.write_text("peer_contracts: []\n", encoding="utf-8")
            self.assertTrue(
                parity._check_panel_agenda_surface(
                    {**legacy_empty, "agenda_paths": [agenda]}, strict=True
                ).failed
            )

            status_path = run_dir / "orchestrator_status.final.json"
            status_path.write_text(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "task_id": "t",
                        "task_name": "Task",
                        "current_generation": 1,
                        "generations_completed": 1,
                        "findings_total": 1,
                        "frontier_candidates": 0,
                        "exit_condition": "completed",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                parity._check_operator_status_surface(
                    run_dir, legacy, canonical_ok, strict=True
                ).failed
            )
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["frontier_candidates"] = 1
            status_path.write_text(json.dumps(status), encoding="utf-8")
            self.assertFalse(
                parity._check_operator_status_surface(
                    run_dir, legacy, canonical_ok, strict=True
                ).failed
            )

            deliverables = root / "deliverables"
            deliverables.mkdir()
            self.assertTrue(parity._check_deliverables(deliverables, strict=True).failed)
            (deliverables / "data").mkdir()
            for rel in (
                "README.md",
                "executive_summary.md",
                "data/run_summary.json",
                "data/frontier_manifest.json",
            ):
                (deliverables / rel).write_text("x", encoding="utf-8")
            self.assertFalse(parity._check_deliverables(deliverables, strict=True).failed)

            (run_dir / "frontier").mkdir(exist_ok=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                '{"cumulative_top":[{"finding_id":"f1"}]}',
                encoding="utf-8",
            )
            self.assertEqual(parity._legacy_frontier_ids(run_dir), {"f1"})
            self.assertEqual(parity._prompt_generation(prompt), 1)
            self.assertIsNone(parity._prompt_generation(run_dir / "bad" / "x.md"))

    def test_replay_malformed_run_reports_many_independent_faults(self) -> None:
        from praxist.core import replay

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for rel in ("findings", "memory", "logs", "artifacts/by_id/b", "prompt_layouts"):
                (run_dir / rel).mkdir(parents=True, exist_ok=True)
            (run_dir / "effective_task_spec.yaml").write_text("task: x\n", encoding="utf-8")
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "task_ref": "task:x",
                        "workflow_ref": "workflow_stage:research_loop",
                        "workspace_hash": "sha256:" + "0" * 64,
                        "source_hash_algorithm": "bad",
                        "source_file_count": 0,
                        "status": "completed",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "other",
                        "status": "failed",
                        "events": 1,
                        "findings": 99,
                        "frontier_records": 99,
                        "research_memory_records": 99,
                        "graph_edges": 99,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "startup_config.json").write_text(
                json.dumps(
                    {
                        "canonical_args": {
                            "task": "task:x",
                            "runtime": "agent_runtime:missing",
                            "model_provider": "model_provider:missing",
                            "budget_policy": "budget_policy:missing",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "model_profiles.json").write_text(
                json.dumps(
                    {
                        "runtime_ref": "agent_runtime:not_selected",
                        "provider_adapters": {"model_provider:not_selected": "fake"},
                        "profiles": {
                            "p": {
                                "provider_ref": "model_provider:not_selected",
                                "model": "m",
                            }
                        },
                        "runtime_provider_conformance": {
                            "runtime_ref": "bad",
                            "model_provider_ref": "bad",
                            "cache_mode": "bad",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "credentials_redacted.json").write_text(
                '{"credential_profiles": "bad"}',
                encoding="utf-8",
            )
            (run_dir / "cache_policy.json").write_text(
                '{"mode":"bad","runtime_cache_strategy":"bad","provider_cache_strategy":"bad"}',
                encoding="utf-8",
            )
            plugin_path = run_dir / "plugin"
            plugin_path.mkdir()
            selected = [
                {
                    "metadata": {"kind": "workflow_stage", "name": "research_loop"},
                    "path": str(run_dir / "missing_plugin"),
                    "content_hash": "sha256:" + "1" * 64,
                },
                {
                    "metadata": {"kind": "agent_runtime", "name": "bad_hash"},
                    "path": str(plugin_path),
                    "content_hash": "bad",
                },
                {
                    "metadata": {"kind": "model_provider", "name": "bad_manifest"},
                    "path": str(plugin_path),
                    "content_hash": "sha256:" + "2" * 64,
                },
            ]
            (run_dir / "plugin_resolution.json").write_text(
                json.dumps(
                    {
                        "algorithm_version": 1,
                        "run_id": "other",
                        "selected": selected,
                        "dependency_edges": [{"from": "a", "to": "b"}],
                    }
                ),
                encoding="utf-8",
            )
            trajectory = [
                {"event_id": "bad-first", "seq": 2, "kind": "noop", "run_id": "wrong"},
                {"event_id": "start", "seq": 1, "kind": "run.started", "run_id": "run"},
                {
                    "event_id": "agent-start",
                    "seq": 3,
                    "kind": "agent.run_started",
                    "run_id": "run",
                    "payload": {
                        "request": {
                            "agent_runtime_ref": "agent_runtime:not_selected",
                            "model_call": {
                                "provider_ref": "model_provider:not_selected",
                                "model": "m",
                                "credential_ref": {"key_id": "missing"},
                            },
                        }
                    },
                },
                {
                    "event_id": "stage",
                    "seq": 4,
                    "kind": "workflow.stage_succeeded",
                    "run_id": "run",
                    "scope": {"stage_id": "research_loop"},
                    "payload": {
                        "findings": 5,
                        "frontier_records": 5,
                        "research_memory_records": 5,
                        "graph_edges": 5,
                        "result": {"frontier_summary": []},
                    },
                },
                {"event_id": "final-1", "seq": 5, "kind": "run.finalized", "run_id": "run"},
                {"event_id": "final-2", "seq": 6, "kind": "run.finalized", "run_id": "run"},
            ]
            (run_dir / "trajectory.jsonl").write_text(
                "\n".join(json.dumps(item) for item in trajectory) + "\n{bad\n",
                encoding="utf-8",
            )
            payload = run_dir / "artifacts" / "by_id" / "b" / "payload.txt"
            payload.write_text("actual", encoding="utf-8")
            layout_payload = run_dir / "prompt_layouts" / "bad.json"
            layout_payload.write_text("[]", encoding="utf-8")
            artifact_index = [
                {"run_id": "run"},
                {
                    "run_id": "run",
                    "artifact_id": "a",
                    "payload_path": "missing.txt",
                    "content_hash": "bad",
                },
                {
                    "run_id": "run",
                    "artifact_id": "b",
                    "payload_path": "artifacts/by_id/b/payload.txt",
                    "content_hash": "sha256:" + "3" * 64,
                },
                {
                    "run_id": "run",
                    "artifact_id": "layout",
                    "artifact_type": "prompt.layout_manifest",
                    "payload_path": "prompt_layouts/bad.json",
                    "content_hash": replay.sha256_bytes(layout_payload.read_bytes()),
                },
            ]
            (run_dir / "artifact_index.jsonl").write_text(
                "\n".join(json.dumps(item) for item in artifact_index) + "\n",
                encoding="utf-8",
            )
            budget = [
                {
                    "run_id": "run",
                    "kind": "request",
                    "record_id": "r0",
                    "requested_budget": [],
                },
                {
                    "run_id": "run",
                    "kind": "grant",
                    "record_id": "g0",
                    "grant_id": "grant",
                    "granted_budget": {"tokens": 1, "gpu_hours": "bad"},
                },
                {
                    "run_id": "run",
                    "kind": "usage_unknown",
                    "record_id": "u0",
                    "grant_id": "grant",
                    "unknown_units": "bad",
                },
                {
                    "run_id": "run",
                    "kind": "usage",
                    "record_id": "u1",
                    "grant_id": "grant",
                    "actual_usage": {"tokens": 2, "extra": 1},
                },
            ]
            (run_dir / "budget_ledger.jsonl").write_text(
                "\n".join(json.dumps(item) for item in budget) + "\n",
                encoding="utf-8",
            )
            (run_dir / "findings" / "findings.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "finding_id": "f1",
                        "source_event_ids": ["missing"],
                        "evidence_refs": [{"artifact_id": "unknown"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "findings" / "frontier.jsonl").write_text(
                json.dumps({"run_id": "run", "finding_id": "missing"}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "memory" / "research_memory.jsonl").write_text(
                json.dumps({"run_id": "run", "memory_record_id": "m1"}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "memory" / "graph_edges.jsonl").write_text(
                json.dumps({"run_id": "run", "graph_edge_id": "e1"}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "shared_findings").mkdir()
            (run_dir / "shared_findings" / "bad.json").write_text("{bad", encoding="utf-8")
            (run_dir / "shared_findings" / "list.json").write_text("[]", encoding="utf-8")
            (run_dir / "shared_findings" / "good.json").write_text(
                '{"id":"legacy"}', encoding="utf-8"
            )
            (run_dir / "frontier").mkdir(exist_ok=True)
            (run_dir / "frontier" / "frontier_manifest.json").write_text(
                '{"cumulative_top":[{"id":"legacy-frontier"}],"generations":{"1":[{"finding_id":"g1"}]}}',
                encoding="utf-8",
            )
            (run_dir / "research_memory" / "ledgers").mkdir(parents=True)
            (run_dir / "research_memory" / "ledgers" / "bad.yaml").write_text(
                ":\n", encoding="utf-8"
            )
            (run_dir / "research_memory" / "ledgers" / "ok.yaml").write_text(
                "entries:\n- id: m\n- bad\n",
                encoding="utf-8",
            )
            conn = sqlite3.connect(run_dir / "shared_store.db")
            conn.execute("CREATE TABLE findings (id TEXT)")
            conn.execute("INSERT INTO findings VALUES ('sqlite-only')")
            conn.commit()
            conn.close()

            with (
                patch.object(
                    replay, "assert_bundled_execution_manifest", side_effect=ValueError("closure")
                ),
                patch.object(
                    replay, "selected_plugin_from_dict", side_effect=RuntimeError("metadata")
                ),
                patch.object(
                    replay,
                    "build_core_source_snapshot",
                    return_value={"workspace_hash": "sha256:" + "9" * 64},
                ),
                patch.object(replay, "validate_model_for_provider", create=True),
            ):
                report = replay.verify_run(run_dir, locked=False)
            errors = "\n".join(report["errors"])
            warnings = "\n".join(report["warnings"])
            self.assertFalse(report["success"])
            self.assertIn("trajectory expected exactly one run.finalized", errors)
            self.assertIn("artifact hash mismatch", errors)
            self.assertIn("plugin_resolution run_id mismatch", errors)
            self.assertIn("selected plugin metadata unreadable", warnings)
            self.assertIn("state recovery", errors)
            self.assertTrue((run_dir / "replay" / "replay_report.json").exists())

            self.assertEqual(replay._frontier_manifest_ids(run_dir), {"legacy-frontier", "g1"})
            (run_dir / "frontier" / "frontier_manifest.json").write_text("{bad", encoding="utf-8")
            self.assertEqual(replay._frontier_manifest_ids(run_dir), set())
            self.assertGreaterEqual(replay._research_memory_ledger_entry_count(run_dir), 1)

    def test_credentials_failover_and_resolution_edges(self) -> None:
        from praxist.core.credentials import (
            CredentialFailoverManager,
            CredentialRef,
            CredentialResolver,
            CredentialSet,
            find_model_provider_credential,
            provider_name_from_ref,
            require_model_provider_credential,
        )

        resolver = CredentialResolver(
            env={
                "ANTHROPIC_API_KEY": "anthropic-value",
                "OPENROUTER_API_KEY": "openrouter-value",
                "SEMANTIC_SCHOLAR_API_KEY": "literature-value",
            }
        )
        discovered = resolver.discover()
        self.assertEqual(discovered.mode, "robust")
        snapshot = resolver.snapshot(discovered)
        self.assertTrue(snapshot["robust_mode_enabled"])
        self.assertFalse(snapshot["raw_secret_fields_present"])
        self.assertIn("tool_server", snapshot["scopes"])
        self.assertNotIn("anthropic-value", json.dumps(snapshot))
        self.assertEqual(discovered.to_dict()["mode"], "robust")

        fake_set = resolver.discover("fake_multi_key")
        manager = CredentialFailoverManager(fake_set)
        first = manager.select(
            scope="model_provider",
            provider="fake_provider",
            target_ref="model_provider:fake_provider",
        )
        self.assertIsNotNone(first)
        assert first is not None
        second = manager.record_failure(first, "timeout")
        self.assertIsNotNone(second)
        self.assertNotEqual(first.key_id, second.key_id if second else None)
        self.assertEqual(manager.active_ref(first).status, "cooling_down")
        self.assertIsNone(manager.record_failure(first, "unknown_reason"))
        self.assertIsNone(manager.select(scope="wrong"))
        self.assertIsNone(manager.select(scope="model_provider", provider="wrong"))
        self.assertIn("failures", manager.snapshot())

        single_manager = CredentialFailoverManager(
            CredentialSet(mode="single", credentials=[first])
        )
        self.assertIsNone(single_manager.record_failure(first, "timeout"))

        candidate_set = CredentialSet(
            mode="robust",
            credentials=[
                CredentialRef("tool_server", "openrouter", "tool_server:x", "tool", "test"),
                CredentialRef("model_provider", "wrong", "model_provider:wrong", "wrong", "test"),
                CredentialRef(
                    "model_provider",
                    "openrouter",
                    "model_provider:other",
                    "other",
                    "test",
                ),
                CredentialRef(
                    "model_provider",
                    "openrouter",
                    "model_provider:openrouter",
                    "inactive",
                    "test",
                    status="inactive",
                ),
                CredentialRef(
                    "model_provider",
                    "openrouter",
                    "model_provider:openrouter",
                    "active",
                    "test",
                ),
            ],
        )
        self.assertEqual(
            find_model_provider_credential(candidate_set, "model_provider:openrouter").key_id,
            "active",
        )
        self.assertEqual(
            require_model_provider_credential(candidate_set, "model_provider:openrouter").key_id,
            "active",
        )
        self.assertIsNone(
            require_model_provider_credential(
                CredentialSet(mode="single", credentials=[]),
                "model_provider:fake_provider",
            )
        )
        with self.assertRaises(ValueError):
            provider_name_from_ref("tool_server:not_model")
        with self.assertRaises(ValueError):
            require_model_provider_credential(
                CredentialSet(mode="single", credentials=[]),
                "model_provider:openrouter",
            )

    def test_budget_ledger_validation_and_recovery_edges(self) -> None:
        from praxist.core.ledgers import BudgetLedger, _validate_budget_amounts
        from praxist.core.protocol import BudgetDecision, BudgetGrant, BudgetRequest

        request = BudgetRequest(
            request_id="req",
            requester_id="peer",
            experiment_id="exp",
            model_profile_ref="model_provider:fake_provider",
            requested={"tokens": 10.0, "wall_clock_seconds": 2.0},
            expected_value={"summary": "cheap check"},
            evidence_refs=["finding:f1"],
            cheaper_alternatives=["skip"],
            abort_conditions=["no progress"],
        )
        grant = BudgetGrant(
            grant_id="grant",
            approved={"tokens": 1.0, "wall_clock_seconds": 2.0},
            conditions=["record usage"],
            expires_at_generation=1,
        )
        decision = BudgetDecision(decision="grant", reason_codes=["small"], grant=grant)

        with tempfile.TemporaryDirectory() as tmp:
            ledger = BudgetLedger(Path(tmp), "run")
            request_record = ledger.append_request(
                request,
                actor_ref="peer",
                stage_id="stage",
                action_type="experiment",
                reason="request",
                source_event_ids=["e1"],
                artifact_refs=[{"artifact_id": "a1"}],
            )
            self.assertEqual(request_record["kind"], "request")
            decision_record = ledger.append_decision(
                request,
                decision,
                actor_ref="chair",
                stage_id="stage",
                action_type="experiment",
                reason="approve",
            )
            self.assertEqual(decision_record["decision_record"]["grant"]["grant_id"], "grant")
            with self.assertRaises(ValueError):
                ledger.append_decision(
                    request,
                    decision,
                    actor_ref="chair",
                    stage_id="stage",
                    action_type="experiment",
                    reason="duplicate",
                )

            ledger.append_usage(
                request_id="req",
                grant_id="grant",
                actor_ref="peer",
                stage_id="stage",
                action_type="experiment",
                actual_usage={"tokens": 0.75},
                reason="first usage",
            )
            overrun = ledger.append_usage(
                request_id="req",
                grant_id="grant",
                actor_ref="peer",
                stage_id="stage",
                action_type="experiment",
                actual_usage={"tokens": 0.5},
                reason="second usage",
            )
            self.assertTrue(overrun["budget_overrun"])
            self.assertEqual(overrun["overrun_units"], ["tokens"])
            ledger.append_usage_unknown(
                request_id="req",
                grant_id="grant",
                actor_ref="peer",
                stage_id="stage",
                action_type="experiment",
                unknown_units=["tokens", "tokens", ""],
                reason="runtime omitted exact usage",
            )
            self.assertEqual(ledger.require_active_grant("grant")["grant_id"], "grant")
            with self.assertRaises(ValueError):
                ledger.append_usage(
                    request_id="req",
                    grant_id="grant",
                    actor_ref="peer",
                    stage_id="stage",
                    action_type="experiment",
                    actual_usage={"gpu_hours": 1.0},
                    reason="unapproved",
                )
            with self.assertRaises(ValueError):
                ledger.append_usage(
                    request_id="req",
                    grant_id="grant",
                    actor_ref="peer",
                    stage_id="stage",
                    action_type="experiment",
                    actual_usage={"tokens": -1.0},
                    reason="negative",
                )
            with self.assertRaises(ValueError):
                ledger.append_usage_unknown(
                    request_id="req",
                    grant_id="grant",
                    actor_ref="peer",
                    stage_id="stage",
                    action_type="experiment",
                    unknown_units=["gpu_hours"],
                    reason="unapproved unknown",
                )
            with self.assertRaises(ValueError):
                ledger.require_active_grant("missing")

            ledger.path.write_text("{bad json\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                ledger.records()

        for invalid in (
            {"unsupported": 1.0},
            {"tokens": object()},
            {"tokens": float("nan")},
            {"tokens": -0.1},
        ):
            with self.assertRaises(ValueError):
                _validate_budget_amounts(invalid, "invalid")

    def test_research_memory_ledger_retrieval_and_firewall_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory import (
            context_firewall,
            retrieval_policy,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base import (
            LedgerEntry,
            LedgerStore,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers.hypothesis_ledger import (
            HypothesisLedger,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "memory" / "ledger.yaml"
            store = LedgerStore(ledger_path, "contract")
            self.assertEqual(store._read_all()["entries"], [])
            ledger_path.write_text("- not a dict\n", encoding="utf-8")
            self.assertEqual(store._read_all()["entries"], [])
            ledger_path.write_text("entries: not-a-list\n", encoding="utf-8")
            self.assertEqual(store._read_all()["entries"], [])
            ledger_path.write_text(":\n", encoding="utf-8")
            self.assertEqual(store._read_all()["entries"], [])

            created = store.upsert("entry", {"status": "pending"}, created_by="tester")
            self.assertEqual(created.data["status"], "pending")
            updated = store.upsert(
                "entry",
                {"status": "confirmed", "score": 1},
                created_by="tester",
                action="confirm",
            )
            self.assertEqual(updated.data["status"], "confirmed")
            self.assertEqual(len(store), 1)
            self.assertEqual(store.get("entry").data["score"], 1)
            self.assertEqual(
                [entry.id for entry in store.filter(lambda e: e.data["score"] == 1)],
                ["entry"],
            )

            appended = store.append_only("append", {"kind": "new"}, created_by="tester")
            self.assertIsInstance(LedgerEntry.from_dict(appended.to_dict()), LedgerEntry)
            with self.assertRaises(ValueError):
                store.append_only("append", {"kind": "dupe"})
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.research_memory.ledgers._ledger_base.os.replace",
                    side_effect=OSError("replace failed"),
                ),
                self.assertRaises(OSError),
            ):
                store.upsert("broken", {"status": "pending"})
            self.assertFalse(ledger_path.with_suffix(".yaml.tmp").exists())

            hypotheses = HypothesisLedger(Path(tmp))
            hypotheses.upsert(
                "h1",
                "Active hypothesis",
                prediction="will improve",
                status="testing",
                source_findings=["f1"],
            )
            hypotheses.upsert("h2", "Killed hypothesis", status="killed")
            self.assertEqual([entry.id for entry in hypotheses.list_active()], ["h1"])
            self.assertEqual([entry.id for entry in hypotheses.list_killed()], ["h2"])
            self.assertEqual(hypotheses.get("h1").data["prediction"], "will improve")
            self.assertEqual({entry.id for entry in hypotheses.all()}, {"h1", "h2"})
            with self.assertRaises(ValueError):
                hypotheses.upsert("bad", "bad", status="unknown")

        zero_mix = retrieval_policy.RetrievalMix(0, 0, 0, 0, 0)
        self.assertEqual(zero_mix.normalized().support, 0.0)
        self.assertEqual(zero_mix.slot_counts(5)["support"], 1)
        cards = [
            {"evidence_id": "support", "quality": {}, "interpretation": {"short": "ok"}},
            {
                "evidence_id": "negative",
                "quality": {"is_negative": True},
                "interpretation": {"short": "bad"},
            },
            {
                "evidence_id": "retired",
                "quality": {"is_retired": True},
                "interpretation": {"short": "old"},
            },
            {
                "evidence_id": "frontier",
                "source_type": "frontier_delta",
                "quality": {},
                "interpretation": {"short": "delta"},
            },
            {
                "evidence_id": "external",
                "quality": {},
                "interpretation": {"short": "cross_arch sentinel"},
            },
            ["not", "a", "card"],
        ]
        selected = retrieval_policy.select_cards_with_mix(
            cards,
            retrieval_policy.RetrievalMix(),
            total_budget=4,
            high_stakes=True,
        )
        self.assertEqual(len(selected), 4)
        self.assertGreater(retrieval_policy.negative_evidence_ratio(cards), 0.0)
        self.assertEqual(retrieval_policy.negative_evidence_ratio([]), 0.0)
        self.assertEqual(
            retrieval_policy.select_cards_with_mix([], retrieval_policy.RetrievalMix(), 3),
            [],
        )

        self.assertGreaterEqual(context_firewall.estimate_tokens({"unicode": "汉字"}), 1)
        small = {"ok": "short"}
        self.assertIs(context_firewall.shrink_dict(small, 100), small)
        truncated = context_firewall.shrink_dict({"long": "x" * 2000}, 200)
        self.assertIn("[truncated]", truncated["long"])
        list_capped = context_firewall.shrink_dict({"items": ["x" * 20] * 100}, 200)
        self.assertLessEqual(len(list_capped["items"]), 21)
        dropped = context_firewall.shrink_dict(
            {
                "negative_evidence_digest": "n" * 5000,
                "coverage_matrix_digest": "c" * 5000,
            },
            5,
        )
        self.assertEqual(dropped["negative_evidence_digest"], "...[budget truncated]")
        self.assertIn("coverage_matrix_digest", dropped)

        pack = SimpleNamespace(
            pack_id="pack",
            built_at="now",
            panel_mode="full",
            target_decisions=["promote"],
            shared_core={"summary": "s" * 200},
            private_packs={"Builder": [{"interpretation": {"short": "x" * 5000}}]},
            audit={"raw_history": False},
        )
        with patch.dict(
            context_firewall.BUDGETS,
            {"tiny": context_firewall.ModeBudget(5, private_pack_tokens=5, max_cards=5)},
        ):
            fitted = context_firewall.fit_pack_to_budget(pack, "tiny")
        self.assertIn(
            "[truncated for budget]",
            fitted["private_packs"]["Builder"][0]["interpretation"]["short"],
        )
        mini = context_firewall.fit_pack_to_budget(pack, "mini")
        self.assertLessEqual(len(mini["private_packs"]["Builder"]), 2)
        self.assertTrue(context_firewall.forbid_raw_history({"raw_history": "not allowed"}))
        self.assertFalse(context_firewall.forbid_raw_history({"summary": "allowed"}))

    def test_runtime_environment_and_schedule_helper_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            runtime_environment,
        )
        from praxist.plugins.workflow_stages.research_loop.backend.tools import schedule

        task_spec = SimpleNamespace(
            evaluation=SimpleNamespace(
                primary_metric="score",
                direction="maximize",
                requires_tier=True,
                anchor_metrics=[
                    {"name": "score", "direction": "maximize"},
                    {"name": "loss", "direction": "bad"},
                    ("cost", "minimize"),
                    ("fallback", "bad"),
                    {},
                    [],
                ],
            )
        )
        payload = runtime_environment._anchor_metrics_payload(task_spec)
        self.assertEqual(
            payload,
            [
                {"name": "score", "direction": "maximize"},
                {"name": "loss", "direction": "maximize"},
                {"name": "cost", "direction": "minimize"},
                {"name": "fallback", "direction": "maximize"},
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with (
                patch.dict(os.environ, {"BYPASS_GPU_GOVERNOR": "1"}, clear=False),
                patch.object(runtime_environment.os, "replace", side_effect=OSError("pointer")),
            ):
                configured = runtime_environment.configure_runtime_environment(
                    task_spec=task_spec,
                    run_dir=run_dir,
                    findings_dir=run_dir / "shared_findings",
                    local_mode=False,
                )
                self.assertEqual(configured, payload)
                self.assertNotIn("BYPASS_GPU_GOVERNOR", os.environ)
                self.assertEqual(os.environ["REQUIRES_TIER"], "true")
            runtime_environment.initialize_local_store_if_needed(local_mode=False)
            from praxist.plugins.workflow_stages.research_loop.backend.tools import (
                local_store,
            )

            with patch.object(local_store, "init_db", side_effect=RuntimeError("db")):
                runtime_environment.initialize_local_store_if_needed(local_mode=True)

        self.assertEqual(schedule.epoch_fraction(1, 0), 0.0)
        self.assertEqual(schedule.epoch_fraction(-1, 10), 0.0)
        self.assertEqual(schedule.epoch_fraction(12, 10), 1.0)
        self.assertEqual(schedule.epoch_fraction(12, 10, clamp=False), 1.2)
        self.assertAlmostEqual(schedule.linear_schedule(2.0, start=0.0, end=10.0), 10.0)
        self.assertAlmostEqual(schedule.cosine_schedule(2.0, start=0.0, end=10.0), 10.0)
        self.assertAlmostEqual(
            schedule.peaked_schedule(0.1, start=0.0, peak=1.0, peak_at=0.0), 0.9755282581475768
        )
        self.assertAlmostEqual(
            schedule.peaked_schedule(0.9, start=0.0, peak=1.0, peak_at=1.0), 0.9755282581475768
        )
        self.assertAlmostEqual(
            schedule.peaked_schedule(0.25, start=0.0, peak=1.0, peak_at=0.5), 0.5
        )
        self.assertAlmostEqual(
            schedule.peaked_schedule(0.75, start=0.0, peak=1.0, end=0.2, peak_at=0.5), 0.6
        )
        self.assertEqual(
            schedule.warmup_then_schedule(
                0.5,
                warmup_fraction=0,
                warmup_start=0,
                base_start=1,
                base_end=3,
                base_kind="linear",
            ),
            2.0,
        )
        self.assertEqual(
            schedule.warmup_then_schedule(
                0.1,
                warmup_fraction=0.2,
                warmup_start=0,
                base_start=2,
                base_end=4,
            ),
            1.0,
        )
        hits = schedule.scan_for_step_anti_pattern(
            """
total_steps = 100
# max_steps ignored
doc = \"\"\"num_training_steps ignored\"\"\"
value = max_steps + 1
"""
        )
        self.assertEqual([hit[2] for hit in hits], ["total_steps", "max_steps"])

    def test_findings_sync_http_and_filesystem_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import findings_sync

        self.assertEqual(findings_sync._sanitize_filename(""), "unknown")
        self.assertEqual(findings_sync._sanitize_filename("bad / name!!"), "bad_name")
        self.assertEqual(
            findings_sync.finding_filename({"id": "f", "title": "Title With Space"}),
            "f_Title_With_Space.json",
        )
        with tempfile.TemporaryDirectory() as tmp:
            findings_dir = Path(tmp)
            (findings_dir / "source.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(
                findings_sync.save_finding_to_dir(
                    {"id": "f1", "source_filename": "source.json"},
                    findings_dir,
                )
            )
            existing = findings_dir / "f2_Name.json"
            existing.write_text("{}", encoding="utf-8")
            self.assertIsNone(
                findings_sync.save_finding_to_dir(
                    {"id": "f2", "variant_name": "Name"}, findings_dir
                )
            )
            with patch.object(findings_sync, "atomic_write_json", side_effect=OSError("disk")):
                self.assertIsNone(
                    findings_sync.save_finding_to_dir(
                        {"id": "f3", "variant_name": "Name"},
                        findings_dir,
                    )
                )

            sync = findings_sync.FindingsSync(findings_dir, poll_interval=0, local_mode=True)
            sync._thread = SimpleNamespace(is_alive=lambda: True)
            sync.start()
            self.assertIsNotNone(sync._thread)
            self.assertTrue(sync._sync_mutex.acquire(blocking=False))
            try:
                self.assertEqual(sync.sync_once(), 0)
            finally:
                sync._sync_mutex.release()

            fake_ingest = ModuleType(
                "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_ingest"
            )
            fake_ingest.ingest_findings_directory = lambda _path: (_ for _ in ()).throw(
                RuntimeError("ingest")
            )
            with (
                patch.dict(sys.modules, {fake_ingest.__name__: fake_ingest}),
                patch.object(
                    sync, "_fetch_all_findings", return_value=[{"id": "fresh", "variant_name": "A"}]
                ),
            ):
                self.assertEqual(sync._sync_once_locked(), 1)

            with patch.object(sync, "_fetch_from_sqlite", return_value=["sqlite"]):
                self.assertEqual(sync._fetch_all_findings(), ["sqlite"])
            sync.local_mode = False
            with patch.object(sync, "_fetch_from_http", return_value=["http"]):
                self.assertEqual(sync._fetch_all_findings(), ["http"])
            self.assertEqual(sync._fetch_from_http(), [])

            with (
                patch.object(sync, "sync_once", return_value=0),
                patch.object(
                    sync._stop_event,
                    "wait",
                    side_effect=lambda timeout=0: sync._stop_event.set() or True,
                ),
            ):
                sync._run()

            local_sync = findings_sync.FindingsSync(findings_dir, poll_interval=0, local_mode=True)
            with (
                patch.object(local_sync, "sync_once", return_value=0),
                patch.object(
                    findings_sync,
                    "wait_for_filesystem_event",
                    side_effect=RuntimeError("watch"),
                ),
                patch.object(
                    local_sync._stop_event,
                    "wait",
                    side_effect=lambda timeout=0: local_sync._stop_event.set() or True,
                ),
            ):
                local_sync._run()

    def test_http_utils_fallback_and_validation_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.tools import http_utils

        class FakeResponse:
            def __init__(self, status_code: int, payload: object, json_error: bool = False):
                self.status_code = status_code
                self.payload = payload
                self.json_error = json_error
                self.raised = False

            def json(self):
                if self.json_error:
                    raise ValueError("not json")
                return self.payload

            def raise_for_status(self):
                self.raised = True
                if self.status_code >= 400:
                    raise RuntimeError("http")

        fake_requests = SimpleNamespace()
        error_response = FakeResponse(400, {"error": "bad"})
        fake_requests.post = lambda *args, **kwargs: error_response
        fake_requests.get = lambda *args, **kwargs: FakeResponse(200, {"ok": True})
        with (
            patch.object(http_utils, "HAS_HTTPX", False),
            patch.object(http_utils, "HAS_REQUESTS", True),
            patch.object(http_utils, "requests", fake_requests, create=True),
        ):
            self.assertEqual(
                asyncio.run(http_utils.async_http_post("http://x", {"a": 1})),
                {"error": "bad"},
            )
            self.assertEqual(
                asyncio.run(http_utils.async_http_get("http://x", headers={"X": "1"})),
                {"ok": True},
            )

        failing_response = FakeResponse(500, {}, json_error=True)
        fake_requests.post = lambda *args, **kwargs: failing_response
        with (
            patch.object(http_utils, "HAS_HTTPX", False),
            patch.object(http_utils, "HAS_REQUESTS", True),
            patch.object(http_utils, "requests", fake_requests, create=True),
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(http_utils.async_http_post("http://x", {}))
            self.assertTrue(failing_response.raised)

        with (
            patch.object(http_utils, "HAS_HTTPX", False),
            patch.object(http_utils, "HAS_REQUESTS", False),
        ):
            with self.assertRaises(ImportError):
                asyncio.run(http_utils.async_http_post("http://x", {}))
            with self.assertRaises(ImportError):
                asyncio.run(http_utils.async_http_get("http://x"))

        self.assertEqual(http_utils.validate_safe_identifier(" ok-id_1 ", "name"), "ok-id_1")
        for bad in ("", "../x", "x/y", "x\\y", "bad!"):
            with self.assertRaises(ValueError):
                http_utils.validate_safe_identifier(bad, "name")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            child = base / "child"
            child.write_text("x", encoding="utf-8")
            self.assertEqual(
                http_utils.validate_safe_path(str(child), "path", str(base)), str(child.resolve())
            )
            with self.assertRaises(ValueError):
                http_utils.validate_safe_path("", "path", str(base))
            with self.assertRaises(ValueError):
                http_utils.validate_safe_path("/etc/passwd", "path", str(base))

    def test_hook_metrics_parser_is_best_effort(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.hooks import (
            log_tool_usage,
        )

        self.assertIsNone(log_tool_usage.parse_mcp_tool_name("Read")[0])
        self.assertEqual(
            log_tool_usage.parse_mcp_tool_name("mcp__server__tool"), ("server", "tool")
        )
        self.assertEqual(log_tool_usage.parse_mcp_tool_name("mcp__bad"), (None, None))

        with patch.object(sys, "stdin", io.StringIO("")):
            self.assertIsNone(log_tool_usage.read_hook_input())
        with patch.object(sys, "stdin", io.StringIO("{bad")):
            self.assertIsNone(log_tool_usage.read_hook_input())

        class TrackingSink:
            def __init__(self) -> None:
                self.mcp: list[dict[str, Any]] = []
                self.skills: list[dict[str, Any]] = []

            def record_mcp_tool(self, **kwargs) -> None:
                self.mcp.append(kwargs)

            def record_skill(self, **kwargs) -> None:
                self.skills.append(kwargs)

        sink = TrackingSink()
        timing_module = ModuleType(
            "praxist.plugins.workflow_stages.research_loop.backend.telemetry.tool_timing"
        )
        timing_module.get_duration_ms = lambda **kwargs: 123
        usage_module = ModuleType(
            "praxist.plugins.workflow_stages.research_loop.backend.telemetry.usage_tracker"
        )
        usage_module.get_tracker = lambda: sink
        modules = {
            timing_module.__name__: timing_module,
            usage_module.__name__: usage_module,
        }
        with patch.dict(sys.modules, modules):
            with patch.object(
                sys,
                "stdin",
                io.StringIO(
                    json.dumps(
                        {
                            "tool_name": "mcp__eval__share",
                            "tool_input": {"x": 1},
                            "tool_response": '{"error":"bad"}',
                            "session_id": "s",
                        }
                    )
                ),
            ):
                log_tool_usage.main()
            with patch.object(
                sys,
                "stdin",
                io.StringIO(
                    json.dumps(
                        {
                            "tool_name": "Skill",
                            "tool_input": {},
                            "tool_response": "not-json",
                            "session_id": "s",
                        }
                    )
                ),
            ):
                log_tool_usage.main()
        self.assertFalse(sink.mcp[0]["success"])
        self.assertEqual(sink.mcp[0]["metadata"], {"server": "eval", "tool": "share"})
        self.assertTrue(sink.skills[0]["success"])

        usage_module.get_tracker = lambda: (_ for _ in ()).throw(RuntimeError("tracker"))
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                sys,
                "stdin",
                io.StringIO(json.dumps({"tool_name": "Skill", "tool_response": "{}"})),
            ),
        ):
            log_tool_usage.main()

    def test_role_skill_loader_rejects_unsafe_task_shapes(self) -> None:
        from praxist.core import role_skills

        with self.assertRaises(ValueError):
            role_skills.load_role_skill("task_role:builder")
        for bad in ("", "../bad", "a/b", "a\\b"):
            with self.assertRaises(ValueError):
                role_skills._validate_task_role_name(bad)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task"
            roles = task / "roles"
            roles.mkdir(parents=True)
            with self.assertRaises(ValueError):
                role_skills.load_role_skill("task_role:missing", task_project_path=task)

            role = roles / "builder"
            role.mkdir()
            (role / "role.yaml").write_text("role:\n  role_id: builder\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                role_skills.load_role_skill("task_role:builder", task_project_path=task)
            (role / "skill.md").write_text(
                "Fixed review questions:\n- What changed?\nnot a bullet\n",
                encoding="utf-8",
            )
            (role / "private.md").write_text("private", encoding="utf-8")
            (role / "role.yaml").write_text(
                "\n".join(
                    [
                        "role:",
                        "  role_id: builder",
                        "  display_name: Builder",
                        "  private_kb: private.md",
                        "  tool_scope:",
                        "    - finding_graph_query",
                    ]
                ),
                encoding="utf-8",
            )
            loaded = role_skills.load_role_skill(
                "task_role:builder",
                task_project_path=Path("task"),
                workspace=root,
            )
            self.assertEqual(loaded.role_id, "builder")
            self.assertEqual(loaded.fixed_questions, ("What changed?",))
            self.assertEqual(len(loaded.private_kb_paths), 1)
            self.assertIn("private_kb_paths", loaded.to_prompt_context())

            (role / "role.yaml").write_text("- not-an-object\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                role_skills._read_manifest(role / "role.yaml")

            plugin = root / "plugin_role"
            (plugin / "private_kb").mkdir(parents=True)
            (plugin / "private_kb" / "a.md").write_text("a", encoding="utf-8")
            (plugin / "private_kb" / "skip_dir").mkdir()
            manifest = {"assets": ["private_kb/*.md", "docs/*.md"]}
            self.assertEqual(
                [path.name for path in role_skills._private_kb_paths(plugin, manifest, {})],
                ["a.md"],
            )
            with self.assertRaises(ValueError):
                role_skills._private_kb_paths(plugin, {"assets": ["private_kb/../escape.md"]}, {})
            with self.assertRaises(ValueError):
                role_skills._safe_plugin_file(plugin, "../escape.md")
            with self.assertRaises(ValueError):
                role_skills._selected_role_plugin("task:not_role", registry=None, workspace=root)

    def test_finding_graph_cli_commands_use_public_local_store_contracts(self) -> None:
        from praxist.plugins.graph_maintainers.finding_graph_mvp import cli as graph_cli

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            with self.assertRaises(SystemExit):
                graph_cli._setup_env(run_dir)
            run_dir.mkdir()
            saved_env = {
                key: os.environ.get(key) for key in ("LOCAL_STORE_DIR", "LOCAL_FINDINGS_DIR")
            }

            def restore_cli_env() -> None:
                for key, value in saved_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            self.addCleanup(restore_cli_env)

            local_store_module = ModuleType(
                "praxist.plugins.workflow_stages.research_loop.backend.tools.local_store"
            )
            local_store_module.init_db = lambda: None
            local_store_module.get_all_findings = lambda: []
            local_store_module.insert_edges_batch = lambda edges: len(edges)
            local_store_module.count_edges = lambda: 3

            class FakeConn:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def execute(self, _sql):
                    self.deleted = True

            local_store_module._get_conn = lambda: FakeConn()

            engine_module = ModuleType("praxist.plugins.graph_maintainers.finding_graph_mvp.engine")

            class FakeBuilder:
                MIN_CONFIDENCE = 0.5

                def __init__(self, findings):
                    self.findings = findings

                def build_all_edges(self):
                    return [{"confidence": 0.9}, {"confidence": 0.1}]

            engine_module.FindingGraphBuilder = FakeBuilder
            engine_module.write_graph_health = lambda graph_dir: {"graph_dir": str(graph_dir)}

            class FakeMaintainer:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.started = False
                    self.stopped = False

                def start(self):
                    self.started = True

                def stop(self):
                    self.stopped = True

            engine_module.FindingGraphMaintainer = FakeMaintainer

            viz_module = ModuleType("praxist.plugins.graph_maintainers.finding_graph_mvp.viz")
            viz_module.build_viz_payload = lambda: {"nodes": [1, 2], "edges": [1]}

            def render_graph_html(path, payload):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("html", encoding="utf-8")
                return path

            viz_module.render_graph_html = render_graph_html
            from praxist.plugins.graph_maintainers.finding_graph_mvp import (
                engine as actual_engine,
            )
            from praxist.plugins.graph_maintainers.finding_graph_mvp import (
                viz as actual_viz,
            )
            from praxist.plugins.workflow_stages.research_loop.backend.tools import (
                local_store as actual_local_store,
            )

            args = SimpleNamespace(run_dir=run_dir, poll_interval=0, yes=False, output=None)
            with (
                patch.object(actual_local_store, "init_db", local_store_module.init_db),
                patch.object(
                    actual_local_store,
                    "get_all_findings",
                    local_store_module.get_all_findings,
                ),
                patch.object(
                    actual_local_store,
                    "insert_edges_batch",
                    local_store_module.insert_edges_batch,
                ),
                patch.object(actual_local_store, "count_edges", local_store_module.count_edges),
                patch.object(actual_local_store, "_get_conn", local_store_module._get_conn),
                patch.object(actual_engine, "FindingGraphBuilder", FakeBuilder),
                patch.object(actual_engine, "write_graph_health", engine_module.write_graph_health),
                patch.object(actual_engine, "FindingGraphMaintainer", FakeMaintainer),
                patch.object(actual_viz, "build_viz_payload", viz_module.build_viz_payload),
                patch.object(actual_viz, "render_graph_html", viz_module.render_graph_html),
            ):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    graph_cli.cmd_backfill(args)
                self.assertIn("nothing to build", stdout.getvalue())

                actual_local_store.get_all_findings = lambda: [{"id": "f"}]
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    graph_cli.cmd_backfill(args)
                self.assertIn("inserted 1 new edges", stdout.getvalue())

                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    graph_cli.cmd_health(args)
                self.assertIn("graph_dir", stdout.getvalue())

                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    graph_cli.cmd_viz(args)
                self.assertIn("2 nodes", stdout.getvalue())

                with (
                    patch("builtins.input", return_value="n"),
                    contextlib.redirect_stdout(io.StringIO()) as stdout,
                ):
                    graph_cli.cmd_wipe(args)
                self.assertIn("aborted", stdout.getvalue())

                args.yes = True
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    graph_cli.cmd_wipe(args)
                self.assertIn("deleted 3 edges", stdout.getvalue())

                fake_time = ModuleType("time")
                fake_time.sleep = lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt())
                with (
                    patch.dict(sys.modules, {"time": fake_time}),
                    contextlib.redirect_stdout(io.StringIO()) as stdout,
                ):
                    graph_cli.cmd_daemon(args)
                self.assertIn("stopping", stdout.getvalue())

                with (
                    patch.object(
                        sys, "argv", ["graph", "--run-dir", str(run_dir), "--mode", "health"]
                    ),
                    patch.object(graph_cli, "cmd_health") as cmd_health,
                ):
                    graph_cli.main()
                cmd_health.assert_called_once()

    def test_deliver_loaders_and_safe_extract_fallback_edges(self) -> None:
        from praxist import deliver

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "shared_store.db").write_text("not sqlite", encoding="utf-8")
            shared = run_dir / "shared_findings"
            shared.mkdir()
            (shared / "bad.json").write_text("{bad", encoding="utf-8")
            (shared / "good.json").write_text('{"id":"f"}', encoding="utf-8")
            self.assertEqual(deliver.load_all_findings(run_dir), [{"id": "f"}])

            metrics_log = run_dir / "peer" / "metrics_log.jsonl"
            metrics_log.parent.mkdir()
            metrics_log.write_text('{"metric":1}\n{bad\n', encoding="utf-8")
            self.assertEqual(deliver.load_all_metrics(run_dir), [{"metric": 1}])

            fake_tar = SimpleNamespace(
                extract=lambda member, path, filter=None: (
                    (_ for _ in ()).throw(TypeError("old")) if filter else None
                )
            )
            deliver._extract_safe_tar_member(fake_tar, SimpleNamespace(name="x"), run_dir)

            (run_dir / "run_summary.json").write_text('{"task_id":"task"}', encoding="utf-8")
            (run_dir / "frontier").mkdir()
            (run_dir / "frontier" / "frontier_manifest.json").write_text("{}", encoding="utf-8")
            out_root = Path(tmp) / "deliverables"
            packaged = deliver.package_deliverables(str(run_dir), str(out_root), overwrite=True)
            self.assertTrue(packaged.name.startswith("task_deliverables_"))

    def test_pi_agent_error_recovery_contract_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import pi_agent

        class BadString:
            def __str__(self) -> str:
                raise RuntimeError("no str")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            agent = pi_agent.PIAgent(
                run_dir=run_dir,
                workspace=run_dir,
                cohort_size=5,
                model="fake",
                max_runtime_minutes=1,
                strict=False,
            )
            db_path = run_dir / "shared_store.db"
            db_path.write_text("not sqlite", encoding="utf-8")
            with patch.object(pi_agent.sqlite3, "connect", side_effect=sqlite3.Error("db")):
                self.assertEqual(agent._load_gen_findings(1), [])
                self.assertEqual(agent._load_gen_edges(1), [])
                self.assertEqual(agent._load_prior_findings_summary(1), [])
                self.assertEqual(agent._build_findings_summary_for_panel(1), {})

            self.assertIsNone(pi_agent.PIAgent._sanitize_json_value(BadString()))
            self.assertEqual(pi_agent.PIAgent._normalize_role(123), "")
            valid = {
                "generation": 2,
                "cross_peer_hypotheses": [
                    {
                        "id": "H1",
                        "claim": "claim",
                        "minimal_test": "test",
                        "kill_condition": "kill",
                        "promote_condition": "promote",
                    }
                ],
                "peer_contracts": {
                    "gen2_peer0": {"role": "exploit"},
                    "gen2_peer1": {"role": "falsifier"},
                    "gen2_peer2": {"role": "bridge"},
                    "gen2_peer3": {"role": "anti-mainline"},
                    "gen2_peer4": {"role": "exploit"},
                },
                "mainline_observation": {},
                "bridge_hypothesis": {},
                "anti_mainline_contract": {},
                "falsification_contract": {},
                "success_metrics": {},
            }
            self.assertIsNone(agent.validate_agenda(valid, 2))
            bad_agenda = dict(valid)
            bad_agenda["generation"] = object()
            self.assertIn("not int-coercible", agent.validate_agenda(bad_agenda, 2))
            bad_agenda = dict(valid)
            bad_agenda["generation"] = "gen x"
            self.assertIn("cannot be parsed", agent.validate_agenda(bad_agenda, 2))
            bad_agenda = dict(valid)
            bad_agenda["cross_peer_hypotheses"] = [{}]
            self.assertIn("at least one dict", agent.validate_agenda(bad_agenda, 2))
            bad_agenda = dict(valid)
            bad_agenda["peer_contracts"] = ["bad"]
            self.assertIn("must be a dict", agent.validate_agenda(bad_agenda, 2))
            bad_agenda = dict(valid)
            bad_agenda["peer_contracts"] = {"gen2_peer0": {"role": "exploit"}}
            self.assertIn("exactly cohort_size", agent.validate_agenda(bad_agenda, 2))
            bad_agenda = dict(valid)
            contracts = dict(valid["peer_contracts"])
            contracts["gen2_peer4"] = "bad"
            bad_agenda["peer_contracts"] = contracts
            self.assertIn("non-dict peers", agent.validate_agenda(bad_agenda, 2))
            bad_agenda = dict(valid)
            contracts = {key: {"role": "exploit"} for key in valid["peer_contracts"]}
            bad_agenda["peer_contracts"] = contracts
            self.assertIn("missing required roles", agent.validate_agenda(bad_agenda, 2))
            bad_agenda = dict(valid)
            bad_agenda["cross_peer_hypotheses"] = [
                {
                    "id": "H1",
                    "claim": "<one paragraph>",
                    "minimal_test": "test",
                    "kill_condition": "kill",
                    "promote_condition": "promote",
                }
            ]
            self.assertIn("placeholder", agent.validate_agenda(bad_agenda, 2))

            out_path = run_dir / "agendas" / "research_agenda_gen1.yaml"
            out_path.parent.mkdir(parents=True)
            out_path.write_text("x: [", encoding="utf-8")
            with (
                patch.object(agent, "_invoke_synthesizer", side_effect=TimeoutError),
                patch.object(pi_agent.Path, "unlink", side_effect=OSError("unlink")),
                patch.object(pi_agent.Path, "write_text", side_effect=OSError("prompt")),
            ):
                result = asyncio.run(agent.run(0))
            self.assertFalse(result.success)
            self.assertIn("no agenda", result.error)

            panel_agent = pi_agent.PIAgent(
                run_dir=run_dir,
                workspace=run_dir,
                cohort_size=5,
                model="fake",
                use_multi_pi_panel=True,
                multi_pi_config=SimpleNamespace(
                    panel_mode_default="mini",
                    auto_escalate_to_high_stakes=False,
                    pi_max_runtime_minutes=1,
                    chair_max_runtime_minutes=1,
                    n_rounds=1,
                    round2_max_runtime_minutes=1,
                ),
            )

            async def fake_run_panel(**kwargs):
                return SimpleNamespace(success=True, agenda=valid, error=None)

            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel",
                    side_effect=fake_run_panel,
                ),
                patch("builtins.open", side_effect=OSError("write failed")),
            ):
                panel_result = asyncio.run(
                    panel_agent._run_multi_pi_panel(0, run_dir / "agendas" / "panel.yaml")
                )
            self.assertFalse(panel_result.success)
            self.assertIn("write canonical agenda failed", panel_result.error)
