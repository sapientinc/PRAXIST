"""Scoreless prompt and operational health contracts."""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


def _task_spec() -> SimpleNamespace:
    return SimpleNamespace(
        research_loop={"mode": "scoreless"},
        evaluation=SimpleNamespace(diversity_dimensions=[]),
        _raw={},
    )


def _freeze_findings(root: Path, generation_id: int, findings: list[dict]) -> None:
    generation_dir = root / f"gen_{generation_id}"
    generation_dir.mkdir(parents=True, exist_ok=True)
    (generation_dir / "scoreless_evidence.json").write_text(
        json.dumps(
            {
                "mode": "scoreless",
                "generation_id": generation_id,
                "evidence_cutoff_at": 100.0,
                "evidence_source_snapshot_at_cutoff": True,
                "findings": findings,
            }
        ),
        encoding="utf-8",
    )


class ScorelessContextTest(unittest.TestCase):
    def test_scoreless_audit_does_not_apply_a_private_card_challenge_gate(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.context_auditor import (
            audit_agenda,
        )

        report = audit_agenda(
            {
                "peer_contracts": {"peer": {"role": "researcher"}},
                "cross_peer_hypotheses": [{"id": "H1", "claim": "Universally superior evidence"}],
            },
            {"shared_core": {"research_loop_mode": "scoreless"}, "private_packs": {"skeptic": []}},
            {},
            completed_gen_id=1,
            cohort_size=1,
        )

        self.assertFalse(any("no challenge" in warning for warning in report.warnings))
        self.assertTrue(any("lacks source_id" in warning for warning in report.warnings))

    def test_scoreless_context_audit_does_not_invent_a_negative_evidence_ratio(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.context_auditor import (
            audit_agenda,
        )

        agenda = {"peer_contracts": {"peer": {"role": "researcher"}}}
        pack = {
            "shared_core": {"research_loop_mode": "scoreless", "scoreless_evidence": []},
            "private_packs": {"builder": [], "skeptic": []},
        }
        empty_report = audit_agenda(agenda, pack, {}, completed_gen_id=1, cohort_size=1)
        pack["shared_core"]["negative_evidence_digest"] = [{"id": "N1", "summary": "Failed check"}]
        digest_report = audit_agenda(agenda, pack, {}, completed_gen_id=1, cohort_size=1)

        for report in (empty_report, digest_report):
            self.assertIsNone(report.metrics["negative_evidence_ratio"])
            self.assertEqual(report.metrics["negative_evidence_ratio_status"], "not_applicable")
        self.assertTrue(any("not referenced" in issue for issue in digest_report.blocking_issues))

    def test_scoreless_prompt_budget_preserves_row_types_and_truthful_truncation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.context_firewall import (
            estimate_tokens,
            shrink_dict,
        )

        shared = {
            "research_loop_mode": "scoreless",
            "evidence_status": "not_scored",
            "scoreless_evidence": [
                {
                    "id": f"finding-{index}",
                    "finding_type": "challenge" if index >= 28 else "hypothesis",
                    "evidence_manifest": "gen_0/scoreless_evidence.json",
                    "evidence_status": "not_scored",
                    "content": "x" * 1200,
                }
                for index in range(30)
            ],
        }
        original = copy.deepcopy(shared)

        compact = shrink_dict(shared, budget_tokens=600)

        rows = compact["scoreless_evidence"]
        self.assertTrue(rows)
        self.assertTrue(all(isinstance(row, dict) for row in rows))
        self.assertEqual({row["finding_type"] for row in rows}, {"challenge", "hypothesis"})
        self.assertTrue(all(row["content_truncated"] for row in rows))
        metadata = compact["scoreless_evidence_meta"]
        self.assertEqual(metadata["input_count"], 30)
        self.assertEqual(metadata["returned"], len(rows))
        self.assertEqual(metadata["omitted_for_budget"], 30 - len(rows))
        self.assertTrue(metadata["budget_truncated"])
        self.assertLessEqual(estimate_tokens(compact), 600)
        self.assertEqual(shared, original)

    def test_scoreless_panel_pack_retains_narratives_without_ranked_or_live_cards(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
            build_evidence_pack,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            finding_types = ["hypothesis", "insight", "error", "result", "observation"]
            _freeze_findings(
                root,
                0,
                [
                    {"id": kind, "finding_type": kind, "content": f"Frozen {kind} evidence."}
                    for kind in finding_types
                ],
            )
            (root / "frontier").mkdir()
            (root / "frontier" / "frontier_manifest.json").write_text(
                json.dumps(
                    {"cumulative_top": [{"variant_name": "numeric-winner", "metric_value": 0.9}]}
                ),
                encoding="utf-8",
            )
            with sqlite3.connect(root / "shared_store.db") as conn:
                conn.execute(
                    "CREATE TABLE findings (id TEXT, finding_type TEXT, title TEXT, "
                    "content TEXT, metrics TEXT, variant_name TEXT, notes TEXT, peer_id TEXT, "
                    "generation_id INTEGER, timestamp REAL, extra TEXT)"
                )
                conn.execute(
                    "INSERT INTO findings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "late",
                        "result",
                        "late",
                        "Live late card",
                        '{"metric_value": 99}',
                        "late",
                        "",
                        "p",
                        0,
                        200,
                        "{}",
                    ),
                )
            pack = build_evidence_pack(
                root,
                "full",
                0,
                ["plan research"],
                ["builder", "skeptic"],
                findings_summary={
                    "research_loop_mode": "scoreless",
                    "scoreless_evidence": [{"id": "unfrozen", "content": "Unfrozen input"}],
                    "advisory_graph_edges": [
                        {
                            "edge_id": "retained",
                            "src_finding_id": "hypothesis",
                            "dst_finding_id": "error",
                        },
                        {
                            "edge_id": "late",
                            "src_finding_id": "hypothesis",
                            "dst_finding_id": "unfrozen",
                        },
                    ],
                },
            )

        shared = pack.shared_core
        self.assertEqual(shared["research_loop_mode"], "scoreless")
        self.assertEqual(
            {row["finding_type"] for row in shared["scoreless_evidence"]}, set(finding_types)
        )
        self.assertEqual(shared["evidence_status"], "not_scored")
        self.assertNotIn("scoreless_evidence", shared["findings_summary"])
        self.assertEqual([row["edge_id"] for row in shared["advisory_graph_edges"]], ["retained"])
        self.assertEqual(shared["current_frontier"], {})
        self.assertEqual(shared["gems"], {})
        self.assertEqual(shared["role_performance"], {})
        self.assertEqual(pack.all_cards, [])
        self.assertEqual(pack.private_packs, {"builder": [], "skeptic": []})
        self.assertNotIn("Live late card", json.dumps(pack.shared_core))
        self.assertIsNone(pack.audit["negative_evidence_ratio_global"])

    def test_scoreless_findings_sidecar_does_not_extract_a_primary_metric(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import sidecars

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_spec = _task_spec()
            task_spec.evaluation.primary_metric = "legacy_metric"
            loop = SimpleNamespace(
                task_spec=task_spec,
                run_dir=root,
                findings_dir=root / "shared_findings",
                local_mode=True,
                _build_status_snapshot=lambda: {},
            )
            sync = Mock()
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend.tools.findings_sync.FindingsSync",
                    sync,
                ),
                patch(
                    "praxist.plugins.graph_maintainers.finding_graph_mvp.engine.FindingGraphMaintainer"
                ),
                patch.object(sidecars, "OrchestratorStatusWriter"),
            ):
                sidecars.start_sidecars(loop)

        self.assertIsNone(sync.call_args.kwargs["primary_metric"])
        self.assertFalse(sync.call_args.kwargs["materialize_result_artifacts"])

    def test_scoreless_health_recovers_after_a_later_successful_session(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = PeerSessionMemory(
                run_dir=root,
                gen_dir=root / "gen_0",
                peer_id="gen0_peer0",
                generation_id=0,
                findings_dir=root / "shared_findings",
                task_spec=_task_spec(),
            )
            memory.record_session_result(
                session_id="failed",
                result=SimpleNamespace(success=False, error="earlier tool failure"),
            )
            failed = collect_peer_memory_health(
                run_dir=root, generation_id=0, task_spec=_task_spec()
            )
            memory.record_session_result(
                session_id="recovered", result=SimpleNamespace(success=True, error="")
            )
            recovered = collect_peer_memory_health(
                run_dir=root, generation_id=0, task_spec=_task_spec()
            )
            history = memory.ledger_path.read_text(encoding="utf-8")

        self.assertEqual(failed.peers[0].health, "red")
        self.assertEqual(recovered.peers[0].health, "green")
        self.assertIn("earlier tool failure", history)

    def test_status_preserves_scoreless_operational_health(self) -> None:
        from praxist.cli import status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "gen_0" / "peers" / "gen0_peer0" / "memory"
            memory_dir.mkdir(parents=True)
            (memory_dir / "peer_state.yaml").write_text(
                json.dumps({"peer_id": "gen0_peer0", "last_session_success": True}),
                encoding="utf-8",
            )
            with patch.object(status, "_load_task_spec_for_status", return_value=_task_spec()):
                snapshot = status._read_peer_health(str(root), str(root), 0)

        self.assertEqual(snapshot.peers[0].health, "green")
        self.assertEqual(snapshot.peers[0].baseline_status, "not_applicable")
        self.assertIn("research quality not evaluated", snapshot.peers[0].health_reason)

    def test_scoreless_shared_memory_renders_bounded_content_for_non_result_findings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            findings_dir = root / "shared_findings"
            findings_dir.mkdir()
            for kind in ("hypothesis", "challenge", "method", "observation"):
                (findings_dir / f"{kind}.json").write_text(
                    json.dumps(
                        {
                            "id": kind,
                            "finding_type": kind,
                            "title": kind,
                            "content": f"Evidence from {kind}. " + "x" * 3000 + " OMITTED_END",
                        }
                    ),
                    encoding="utf-8",
                )
            memory = PeerSessionMemory(
                run_dir=root,
                gen_dir=root / "gen_0",
                peer_id="gen0_peer0",
                generation_id=0,
                findings_dir=findings_dir,
                task_spec=_task_spec(),
            )
            prompt = memory.build_prompt_block(session_id="session0", session_index=0)

        for kind in ("hypothesis", "challenge", "method", "observation"):
            self.assertIn(f"Evidence from {kind}.", prompt)
        self.assertNotIn("OMITTED_END", prompt)
        self.assertLessEqual(len(prompt), 12_000)

    def test_multi_pi_receives_frozen_scoreless_context_with_bounded_count_label(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _freeze_findings(root, 0, [{"id": "prior", "finding_type": "hypothesis"}])
            _freeze_findings(root, 1, [{"id": "current", "finding_type": "observation"}])
            with sqlite3.connect(root / "shared_store.db") as conn:
                conn.execute(
                    "CREATE TABLE finding_edges (edge_id TEXT, src_finding_id TEXT, "
                    "dst_finding_id TEXT, edge_type TEXT, confidence REAL, created_by TEXT, rationale TEXT)"
                )
                conn.execute(
                    "INSERT INTO finding_edges VALUES ('link', 'current', 'prior', 'supports', 0.5, 'p', 'Connection')"
                )
            pi = PIAgent(
                run_dir=root,
                workspace=root,
                cohort_size=1,
                model="offline",
                task_spec=_task_spec(),
                use_multi_pi_panel=True,
                multi_pi_config=SimpleNamespace(fallback_to_single_pi_on_panel_failure=False),
            )
            panel = AsyncMock(return_value=SimpleNamespace(success=False, error="offline capture"))
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend.multi_pi.run_panel", panel
            ):
                asyncio.run(pi.run(completed_gen_id=1))
            assert panel.await_args is not None
            summary = panel.await_args.kwargs["findings_summary"]

        self.assertEqual(summary["research_loop_mode"], "scoreless")
        self.assertEqual(summary["surfaced_findings_count"], 1)
        self.assertEqual(summary["by_type"], {"observation": 1})
        self.assertEqual(summary["count_scope"], "bounded_current_generation_evidence")
        self.assertEqual([row["edge_id"] for row in summary["advisory_graph_edges"]], ["link"])
        self.assertEqual({row["id"] for row in summary["scoreless_evidence"]}, {"prior", "current"})

    def test_pi_prompt_uses_only_frozen_current_and_prior_evidence(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.pi_agent import PIAgent

        class CapturingPI(PIAgent):
            prompt = ""

            async def _invoke_synthesizer(self, prompt_text, output_path, *, request_id):
                self.prompt = prompt_text
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generation_id, kind in ((0, "hypothesis"), (1, "observation"), (2, "plan")):
                _freeze_findings(
                    root,
                    generation_id,
                    [
                        {
                            "id": f"frozen-{generation_id}",
                            "generation_id": generation_id,
                            "finding_type": kind,
                            "title": kind,
                            "content": "Frozen evidence. " + "x" * 2000,
                        }
                    ],
                )
            with sqlite3.connect(root / "shared_store.db") as conn:
                conn.execute(
                    "CREATE TABLE findings (id TEXT, finding_type TEXT, title TEXT, "
                    "content TEXT, metrics TEXT, variant_name TEXT, notes TEXT, peer_id TEXT, "
                    "generation_id INTEGER, timestamp REAL, extra TEXT)"
                )
                conn.execute(
                    "INSERT INTO findings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "late",
                        "result",
                        "late",
                        "Mutable late evidence",
                        "{}",
                        "late",
                        "",
                        "p",
                        1,
                        200,
                        "{}",
                    ),
                )
                conn.execute(
                    "CREATE TABLE finding_edges (edge_id TEXT, src_finding_id TEXT, "
                    "dst_finding_id TEXT, edge_type TEXT, confidence REAL, created_by TEXT, rationale TEXT)"
                )
                conn.executemany(
                    "INSERT INTO finding_edges VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            "retained-link",
                            "frozen-1",
                            "frozen-0",
                            "supports",
                            0.5,
                            "p",
                            "Advisory connection",
                        ),
                        ("late-link", "frozen-1", "late", "supports", 0.5, "p", "Late connection"),
                    ],
                )
            template_path = root / "capture.jinja2"
            template_path.write_text(
                "{{ {'mode': research_loop_mode, 'current': findings, "
                "'prior': prior_findings_summary, 'evidence': scoreless_evidence, "
                "'frontier': frontier, 'gems': gems_context, 'validation': validation_candidates, "
                "'edges': edges, 'graph_evidence_scope': graph_evidence_scope} | tojson }}",
                encoding="utf-8",
            )
            pi = CapturingPI(
                run_dir=root,
                workspace=root,
                cohort_size=1,
                model="offline",
                task_spec=_task_spec(),
                prompt_template_path=template_path,
            )
            asyncio.run(pi.run(completed_gen_id=1))
            prompt = json.loads(pi.prompt)

        self.assertEqual(prompt["mode"], "scoreless")
        self.assertEqual([row["id"] for row in prompt["current"]], ["frozen-1"])
        self.assertEqual([row["id"] for row in prompt["prior"]], ["frozen-0"])
        self.assertEqual(prompt["prior"][0]["type"], "hypothesis")
        self.assertTrue(prompt["prior"][0]["content"].startswith("Frozen evidence."))
        self.assertTrue(all(len(row["content"]) <= 1200 for row in prompt["evidence"]))
        self.assertEqual({row["id"] for row in prompt["evidence"]}, {"frozen-0", "frozen-1"})
        self.assertEqual(prompt["frontier"], [])
        self.assertEqual(prompt["validation"], [])
        self.assertEqual(prompt["gems"], {})
        self.assertEqual([edge["edge_id"] for edge in prompt["edges"]], ["retained-link"])
        self.assertEqual(prompt["graph_evidence_scope"], "advisory_edges_between_retained_findings")
        self.assertNotIn("Mutable late evidence", pi.prompt)

    def test_peer_context_uses_frozen_all_type_evidence_without_numeric_parents(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import prompt_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            finding_types = ["hypothesis", "error", "observation", "insight", "result"]
            _freeze_findings(
                root,
                0,
                [
                    {
                        "id": f"finding-{kind}",
                        "finding_type": kind,
                        "generation_id": 0,
                        "peer_id": "gen0_peer0",
                        "title": kind,
                        "content": "Evidence to carry forward. " + "x" * 2000,
                    }
                    for kind in finding_types
                ],
            )
            _freeze_findings(root, 2, [{"id": "future", "generation_id": 2}])
            (root / "gen_0" / "generation_boundary.json").write_text(
                json.dumps(
                    {"peer_mix": {"mature_constructive_ratio": 0.0}, "stop_audit": {"score": 1.0}}
                ),
                encoding="utf-8",
            )
            numeric_parent = {"variant_name": "numeric-winner", "metric_value": 0.99}
            frontier = SimpleNamespace(get_summary=lambda: [numeric_parent])

            context = prompt_context.build_prompt_context(
                task_spec=_task_spec(),
                workspace=root,
                run_dir=root,
                results_dir=root / "results",
                variants_dir=root / "variants",
                findings_dir=root / "shared_findings",
                frontier=frontier,
                local_mode=True,
                gen_id=1,
                peer_index=0,
                cohort_size=1,
                strategy="exploit",
                gems_context={"entries": [numeric_parent]},
            )

        self.assertEqual(context["research_loop_mode"], "scoreless")
        self.assertEqual(
            context["notebook_path"],
            str(root / "work" / "notebooks" / "notebook_gen1_peer0.json"),
        )
        evidence = context["scoreless_evidence"]
        self.assertEqual({row["finding_type"] for row in evidence}, set(finding_types))
        self.assertTrue(
            all(row["content"].startswith("Evidence to carry forward.") for row in evidence)
        )
        self.assertTrue(all(len(row["content"]) <= 1200 for row in evidence))
        self.assertEqual(context["frontier_summary"], [])
        self.assertEqual(context["incubator_top_k"], [])
        self.assertEqual(context["validation_candidate_top_k"], [])
        self.assertEqual(context["diagnostic_control_top_k"], [])
        self.assertEqual(context["gems_context"], {})
        self.assertEqual(context["research_loop_control"], {})
        self.assertNotIn("numeric-winner", context["variant_hint"])
        self.assertNotIn("metric", context["variant_hint"].lower())

    def test_scoreless_health_describes_operations_without_numeric_performance(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            states = {
                "gen0_peer0": {"last_session_success": True},
                "gen0_peer1": {"last_session_success": False},
                "gen0_peer2": {},
                "gen0_peer3": {"last_session_success": True, "research_state": "blocked"},
            }
            for peer_id, state in states.items():
                memory_dir = root / "gen_0" / "peers" / peer_id / "memory"
                memory_dir.mkdir(parents=True)
                (memory_dir / "peer_state.yaml").write_text(
                    json.dumps(
                        {
                            "peer_id": peer_id,
                            "recent_result_artifacts": [{"metric_value": 99.0}],
                            **state,
                        }
                    ),
                    encoding="utf-8",
                )
            snapshot = collect_peer_memory_health(
                run_dir=root,
                generation_id=0,
                primary_metric="metric_value",
                baselines=[{"metric_name": "metric_value", "metric_value": 0.1}],
                task_spec=_task_spec(),
            )

        self.assertEqual(
            [peer.health for peer in snapshot.peers], ["green", "red", "yellow", "red"]
        )
        for peer in snapshot.peers:
            self.assertEqual(peer.baseline_status, "not_applicable")
            self.assertEqual(peer.primary_metric, "")
            self.assertIsNone(peer.best_metric_value)
            self.assertIsNone(peer.baseline_metric_value)
        self.assertIn("research quality not evaluated", snapshot.peers[0].health_reason)
        self.assertIn("no completed session", snapshot.peers[2].health_reason)
