from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


class PeerMemoryContractsTest(unittest.TestCase):
    def test_low_level_memory_helpers_cover_file_and_metric_edges(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import (
            peer_memory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            nested = data_root / "nested"
            nested.mkdir(parents=True)
            payload_path = nested / "payload.json"
            payload_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            bad_json = nested / "bad.json"
            bad_json.write_text("{bad", encoding="utf-8")
            large_json = nested / "large.json"
            large_json.write_text(json.dumps({"blob": "x" * 20}), encoding="utf-8")

            self.assertEqual(
                peer_memory.read_bounded_file_under_root_no_follow(
                    payload_path,
                    data_root,
                    max_bytes=100,
                ),
                payload_path.read_bytes(),
            )
            self.assertIsNone(
                peer_memory.read_bounded_file_under_root_no_follow(
                    payload_path,
                    root / "other",
                    max_bytes=100,
                )
            )
            self.assertIsNone(
                peer_memory.read_bounded_file_under_root_no_follow(
                    data_root,
                    data_root,
                    max_bytes=100,
                )
            )
            self.assertIsNone(
                peer_memory.read_bounded_file_under_root_no_follow(
                    large_json,
                    data_root,
                    max_bytes=5,
                )
            )
            self.assertEqual(
                peer_memory._read_json_under_root(payload_path, data_root, {}),
                {"ok": True},
            )
            self.assertEqual(
                peer_memory._read_json_under_root(bad_json, data_root, {"fallback": True}),
                {"fallback": True},
            )
            self.assertEqual(peer_memory._read_json(payload_path, {}), {"ok": True})
            self.assertEqual(peer_memory._read_json(bad_json, {"bad": True}), {"bad": True})

            if hasattr(os, "symlink"):
                linked = nested / "linked.json"
                os.symlink(payload_path, linked)
                self.assertEqual(
                    peer_memory._read_json(linked, {"linked": False}), {"linked": False}
                )

            metrics = peer_memory._metric_subset(
                {
                    "noise": object(),
                    "score": 1.2,
                    "loss_values": [1, 2, 3, 4, 5],
                    "nested_metric": {"accuracy": 0.9, "ignore": object()},
                    "status": None,
                    "rank": {"deep_score": 7},
                },
                max_items=4,
            )
            self.assertEqual(metrics["score"], 1.2)
            self.assertEqual(metrics["loss_values"], ["1", "2", "3", "4"])
            self.assertEqual(metrics["nested_metric"], {"accuracy": 0.9})
            self.assertIn("status", metrics)
            self.assertEqual(peer_memory._metric_subset("not-a-dict"), {})
            self.assertEqual(
                peer_memory._extract_variant_name(
                    root / "results" / "fallback" / "summary.json", {}
                ),
                "fallback",
            )
            self.assertTrue(peer_memory._matches_peer("gen0_peer2_candidate", "gen0_peer2", 0))
            self.assertFalse(peer_memory._matches_peer("gen1_peer2_candidate", "gen0_peer2", 0))
            self.assertTrue(peer_memory._safe_path_component("../bad name").startswith("unsafe-"))
            self.assertEqual(peer_memory._safe_path_component("gen0_peer2"), "gen0_peer2")

            self.assertEqual(peer_memory._safe_result_summary_paths(root), [])
            results = root / "results"
            results.mkdir()
            child = results / "gen0_peer2_variant"
            child.mkdir()
            summary = child / "tiered_eval_summary.json"
            summary.write_text("{}", encoding="utf-8")
            (child / "not_summary.txt").write_text("x", encoding="utf-8")
            if hasattr(os, "symlink"):
                os.symlink(summary, child / "result_summary.json")
            self.assertEqual(peer_memory._safe_result_summary_paths(root), [summary])

        if hasattr(os, "symlink"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                outside = root / "outside"
                outside.mkdir()
                os.symlink(outside, root / "results")
                self.assertEqual(peer_memory._safe_result_summary_paths(root), [])

    def test_memory_block_includes_contract_findings_and_bounded_state(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            MEMORY_HEADER,
            PeerMemoryConfig,
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            peer_id = "gen0_peer2"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            dig_dir = gen_dir / "peers" / peer_id / "dig"
            dig_dir.mkdir(parents=True)
            (dig_dir / "selected_contract.yaml").write_text(
                yaml.safe_dump(
                    {
                        "variant_name": "calibration_probe_v1",
                        "diversity_cell": {
                            "mechanism_family": "calibration",
                            "intervention_surface": "loss",
                            "intent": "repair",
                        },
                        "mechanism_hypothesis": "Sharper confidence should reduce bad actions.",
                        "expected_metric_signature": {
                            "primary": "score improves",
                            "diagnostic": "confidence error falls",
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (findings / "finding_a.json").write_text(
                json.dumps(
                    {
                        "finding_id": "finding_a",
                        "title": "Sibling peer found a useful ablation",
                        "finding_type": "negative_evidence",
                        "producer_ref": "task_role:peer/gen0_peer1",
                    }
                ),
                encoding="utf-8",
            )

            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id=peer_id,
                generation_id=0,
                findings_dir=findings,
                config=PeerMemoryConfig(max_prompt_chars=4000),
            )
            prompt = memory.compose_session_prompt(
                "Base task prompt",
                session_id="session_000",
                session_index=0,
            )

            self.assertIn("Base task prompt", prompt)
            self.assertIn(MEMORY_HEADER, prompt)
            self.assertIn("calibration_probe_v1", prompt)
            self.assertIn("finding_a", prompt)
            self.assertIn("Anti-Anchoring Check", prompt)
            self.assertNotIn("raw transcript", prompt.lower().replace("not a raw transcript", ""))
            self.assertTrue((gen_dir / "peers" / peer_id / "memory/peer_state.yaml").exists())
            self.assertTrue((gen_dir / "peers" / peer_id / "memory/memory_prompt.md").exists())
            self.assertTrue(
                (gen_dir / "peers" / peer_id / "memory/session_prompt_manifest.json").exists()
            )
            self.assertTrue(
                (gen_dir / "peers" / peer_id / "memory/memory_prompt_session_000.md").exists()
            )
            self.assertTrue(
                (
                    gen_dir / "peers" / peer_id / "memory/session_prompt_manifest_session_000.json"
                ).exists()
            )

    def test_session_result_records_handoff_ledger_and_seen_findings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            peer_id = "gen0_peer3"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            (findings / "finding_b.json").write_text(
                json.dumps({"finding_id": "finding_b", "title": "new evidence"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id=peer_id,
                generation_id=0,
                findings_dir=findings,
            )
            memory.compose_session_prompt("task", session_id="session_001", session_index=1)
            result = AgentResult(
                success=True,
                output={"text_outputs": ["implemented and wrote finding"]},
                duration=3.0,
                iteration_count=4,
            )
            memory.record_session_result(
                session_id="session_001",
                result=result,
                log_file=gen_dir / peer_id / "session_001.log",
            )

            memory_dir = gen_dir / "peers" / peer_id / "memory"
            self.assertIn(
                "implemented and wrote finding",
                (memory_dir / "session_handoff.md").read_text(encoding="utf-8"),
            )
            ledger_rows = [
                json.loads(line)
                for line in (memory_dir / "experiment_ledger.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(ledger_rows[-1]["session_id"], "session_001")
            state = yaml.safe_load((memory_dir / "peer_state.yaml").read_text(encoding="utf-8"))
            self.assertTrue(state["last_session_success"])
            seen = json.loads(
                (memory_dir / "seen_shared_findings.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                seen,
                [f"sha256:{hashlib.sha256(b'finding_b').hexdigest()}"],
            )

            self.assertFalse(memory.should_wake_for_shared_finding(findings / "finding_b.json"))
            malformed = findings / "malformed.json"
            malformed.write_text("{not complete", encoding="utf-8")
            self.assertTrue(memory.should_wake_for_shared_finding(malformed))
            self.assertTrue(memory.should_wake_for_shared_finding(findings / "missing.json"))
            (findings / "finding_b.json").write_text("{}", encoding="utf-8")
            self.assertTrue(memory.should_wake_for_shared_finding(findings / "finding_b.json"))
            non_object = findings / "non_object.json"
            non_object.write_text("[]", encoding="utf-8")
            self.assertTrue(memory.should_wake_for_shared_finding(non_object))
            (findings / "finding_b.json").write_text(
                json.dumps({"finding_id": "finding_b", "title": "updated"}),
                encoding="utf-8",
            )
            with patch.object(memory, "_load_seen_finding_keys", side_effect=OSError("read")):
                self.assertTrue(memory.should_wake_for_shared_finding(findings / "finding_b.json"))

    def test_lossless_seen_finding_tracks_content_version(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerMemoryConfig,
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            path = findings / "finding_versioned.json"
            path.write_text(
                json.dumps({"finding_id": "stable-id", "title": "first evidence"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer_versioned",
                generation_id=0,
                findings_dir=findings,
                config=PeerMemoryConfig(track_finding_content_versions=True),
            )
            first_prompt = memory.compose_session_prompt(
                "task",
                session_id="session_first",
                session_index=0,
            )
            self.assertIn("first evidence", first_prompt)
            memory.record_session_result(
                session_id="session_first",
                result=AgentResult(
                    success=True,
                    output={"text_outputs": ["consumed"]},
                    duration=1.0,
                    iteration_count=1,
                ),
            )
            self.assertFalse(memory.should_wake_for_shared_finding(path))
            seen = json.loads(memory.seen_findings_path.read_text(encoding="utf-8"))
            self.assertEqual(len(seen), 2)
            self.assertTrue(any(key.startswith("sha256v2:") and len(key) == 73 for key in seen))
            self.assertIn(
                f"sha256:{hashlib.sha256(b'stable-id').hexdigest()}",
                seen,
            )

            path.write_text(
                json.dumps({"finding_id": "stable-id", "title": "corrected evidence"}),
                encoding="utf-8",
            )
            self.assertTrue(memory.should_wake_for_shared_finding(path))
            second_prompt = memory.compose_session_prompt(
                "task",
                session_id="session_second",
                session_index=1,
            )
            self.assertIn("corrected evidence", second_prompt)

            unknown = findings / "unknown_identity.json"
            unknown.write_text("{}", encoding="utf-8")
            unknown_prompt = memory.compose_session_prompt(
                "task",
                session_id="session_unknown_first",
                session_index=2,
            )
            self.assertIn("unknown_identity", unknown_prompt)
            memory.record_session_result(
                session_id="session_unknown_first",
                result=AgentResult(
                    success=True,
                    output={"text_outputs": ["consumed unknown"]},
                    duration=1.0,
                    iteration_count=1,
                ),
            )
            self.assertTrue(memory.should_wake_for_shared_finding(unknown))
            repeated_unknown_prompt = memory.compose_session_prompt(
                "task",
                session_id="session_unknown_second",
                session_index=3,
            )
            self.assertIn("unknown_identity", repeated_unknown_prompt)

            derived = findings / "derived_result.json"
            derived.write_text(
                json.dumps(
                    {
                        "id": "derived-id",
                        "title": "derived result",
                        "timestamp": "first-materialization",
                        "metrics": {
                            "source_result_sha256": "a" * 64,
                            "auto_materialized_from_result_artifact": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            derived_prompt = memory.compose_session_prompt(
                "task",
                session_id="session_derived_first",
                session_index=4,
            )
            self.assertIn("derived result", derived_prompt)
            memory.record_session_result(
                session_id="session_derived_first",
                result=AgentResult(
                    success=True,
                    output={"text_outputs": ["consumed derived"]},
                    duration=1.0,
                    iteration_count=1,
                ),
            )
            derived.write_text(
                json.dumps(
                    {
                        "id": "derived-id",
                        "title": "derived result",
                        "timestamp": "second-materialization",
                        "metrics": {
                            "source_result_sha256": "a" * 64,
                            "auto_materialized_from_result_artifact": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(memory.should_wake_for_shared_finding(derived))
            derived.write_text(
                json.dumps(
                    {
                        "id": "derived-id",
                        "title": "revised derived interpretation",
                        "timestamp": "interpretation-materialization",
                        "metrics": {
                            "source_result_sha256": "a" * 64,
                            "auto_materialized_from_result_artifact": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(memory.should_wake_for_shared_finding(derived))
            derived.write_text(
                json.dumps(
                    {
                        "id": "derived-id",
                        "title": "updated derived result",
                        "timestamp": "third-materialization",
                        "metrics": {
                            "source_result_sha256": "b" * 64,
                            "auto_materialized_from_result_artifact": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(memory.should_wake_for_shared_finding(derived))

            manual = findings / "manual_reference.json"
            manual.write_text(
                json.dumps(
                    {
                        "id": "manual-id",
                        "title": "manual interpretation",
                        "metrics": {"source_result_sha256": "c" * 64},
                    }
                ),
                encoding="utf-8",
            )
            memory.compose_session_prompt(
                "task",
                session_id="session_manual_first",
                session_index=5,
            )
            memory.record_session_result(
                session_id="session_manual_first",
                result=AgentResult(
                    success=True,
                    output={"text_outputs": ["consumed manual"]},
                    duration=1.0,
                    iteration_count=1,
                ),
            )
            manual.write_text(
                json.dumps(
                    {
                        "id": "manual-id",
                        "title": "revised manual interpretation",
                        "metrics": {"source_result_sha256": "c" * 64},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(memory.should_wake_for_shared_finding(manual))

    def test_lossless_finding_version_normalizes_only_derived_materialization_time(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            _finding_source_result_hash,
            _finding_version_seen_key,
        )

        payload = {
            "id": "derived-id",
            "title": "derived result",
            "source_result_sha256": "A" * 64,
            "artifact_semantics": {
                "role": "derived_view",
                "stage": "result_finding_reference",
                "created_at": "first-materialization",
            },
        }
        first_key = _finding_version_seen_key("derived-id", payload)
        payload["artifact_semantics"]["created_at"] = "second-materialization"
        self.assertEqual(first_key, _finding_version_seen_key("derived-id", payload))

        self.assertIsNone(
            _finding_source_result_hash(
                {
                    "metrics": {"auto_materialized_from_result_artifact": True},
                    "extra": "not-a-mapping",
                }
            )
        )

    def test_session_record_preserves_agent_written_handoff(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer7",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            memory.memory_dir.mkdir(parents=True)
            memory.handoff_path.write_text(
                "# Agent Handoff\n\nDetailed open process and next action.\n",
                encoding="utf-8",
            )
            memory.record_session_result(
                session_id="session_preserve",
                result=AgentResult(
                    success=True,
                    output={"text_outputs": ["auto summary"]},
                    duration=1.0,
                    iteration_count=1,
                ),
            )
            handoff = memory.handoff_path.read_text(encoding="utf-8")
            self.assertIn("Detailed open process and next action.", handoff)
            self.assertIn("PRAXIST_AUTO_SESSION_STATUS", handoff)
            self.assertTrue(memory.auto_handoff_path.exists())

    def test_memory_rejects_symlinked_peers_root(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir(parents=True)
            external = root / "external_peers"
            external.mkdir()
            try:
                (gen_dir / "peers").symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest("filesystem does not support directory symlinks")

            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer4",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )

            with self.assertRaises(OSError):
                memory.compose_session_prompt("task", session_id="session_symlink", session_index=0)

            self.assertFalse((external / "gen0_peer4").exists())

    def test_memory_rejects_symlinked_generation_ancestor(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            external_gen = root / "external_gen"
            external_gen.mkdir()
            try:
                (run_dir / "gen_0").symlink_to(external_gen, target_is_directory=True)
            except OSError:
                self.skipTest("filesystem does not support directory symlinks")
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=run_dir / "gen_0",
                peer_id="gen0_peer4",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )

            with self.assertRaises(OSError):
                memory.compose_session_prompt(
                    "task", session_id="session_symlink_gen", session_index=0
                )

            self.assertFalse((external_gen / "peers").exists())

    def test_atomic_write_does_not_follow_legacy_tmp_symlink(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer5",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            memory.memory_dir.mkdir(parents=True)
            outside = root / "outside.txt"
            outside.write_text("do-not-touch", encoding="utf-8")
            legacy_tmp = memory.memory_dir / ".peer_state.yaml.tmp"
            try:
                legacy_tmp.symlink_to(outside)
            except OSError:
                self.skipTest("filesystem does not support symlinks")

            memory.compose_session_prompt("task", session_id="session_tmp", session_index=0)

            self.assertEqual(outside.read_text(encoding="utf-8"), "do-not-touch")
            self.assertTrue((memory.memory_dir / "peer_state.yaml").exists())
            self.assertTrue(legacy_tmp.is_symlink())

    def test_memory_writes_reject_directory_swap_after_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        for operation in ("replace", "append"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                run_dir = root / "run"
                memory = PeerSessionMemory(
                    run_dir=run_dir,
                    gen_dir=run_dir / "gen_0",
                    peer_id="gen0_peer0",
                    generation_id=0,
                    findings_dir=run_dir / "shared_findings",
                )
                memory.initialize()
                memory.peers_root.chmod(0o777)
                destination = memory.state_path if operation == "replace" else memory.ledger_path
                outside = root / "outside"
                outside_memory = outside / "memory"
                outside_memory.mkdir(parents=True)
                outside_file = outside_memory / destination.name
                original = b"outside authority must remain unchanged\n"
                outside_file.write_bytes(original)
                validate = memory._assert_safe_write_target

                def swap_directory(
                    path: Path,
                    *,
                    original_validate=validate,
                    peer_memory=memory,
                    outside_dir=outside,
                ) -> None:
                    original_validate(path)
                    peer_memory.peer_root.rename(peer_memory.peers_root / "original_peer")
                    peer_memory.peer_root.symlink_to(outside_dir, target_is_directory=True)

                writer = (
                    memory._write_memory_text
                    if operation == "replace"
                    else memory._append_memory_text
                )
                with (
                    patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(root / "control")}),
                    patch.object(memory, "_assert_safe_write_target", side_effect=swap_directory),
                    self.assertRaises(OSError),
                ):
                    try:
                        writer(destination, "untrusted redirected write\n")
                    finally:
                        self.assertEqual(outside_file.read_bytes(), original)

    def test_memory_reads_reject_directory_swap_after_validation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_dir = root / "run"
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=run_dir / "gen_0",
                peer_id="gen0_peer0",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            memory.initialize()
            memory.peers_root.chmod(0o777)
            outside = root / "outside"
            outside_memory = outside / "memory"
            outside_memory.mkdir(parents=True)
            (outside_memory / memory.state_path.name).write_text(
                "private fixture content", encoding="utf-8"
            )
            validate = memory._safe_existing_file

            def swap_after_validation(path, allowed_root):
                valid = validate(path, allowed_root)
                memory.peer_root.rename(memory.peers_root / "original_peer")
                memory.peer_root.symlink_to(outside, target_is_directory=True)
                return valid

            with (
                patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(root / "control")}),
                patch.object(memory, "_safe_existing_file", side_effect=swap_after_validation),
            ):
                self.assertEqual(memory._read_memory_text(memory.state_path), "")

    def test_memory_reads_handoff_with_bounded_tail(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            DEFAULT_MAX_HANDOFF_BYTES,
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer6",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            memory.memory_dir.mkdir(parents=True)
            memory.handoff_path.write_text(
                "x" * (DEFAULT_MAX_HANDOFF_BYTES + 100) + "\nTAIL_MARKER\n",
                encoding="utf-8",
            )

            prompt = memory.compose_session_prompt(
                "task", session_id="session_tail", session_index=0
            )

            self.assertIn("TAIL_MARKER", prompt)

    def test_oversized_shared_finding_is_not_loaded_into_prompt(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            DEFAULT_MAX_EXTERNAL_JSON_BYTES,
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            (findings / "huge.json").write_text(
                '{"finding_id":"huge","title":"' + ("x" * DEFAULT_MAX_EXTERNAL_JSON_BYTES) + '"}',
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer8",
                generation_id=0,
                findings_dir=findings,
            )

            prompt = memory.compose_session_prompt(
                "task", session_id="session_huge", session_index=0
            )

            self.assertIn("New Shared Findings Since Last Session", prompt)
            self.assertNotIn("`huge`", prompt)

    def test_symlinked_shared_findings_root_is_not_loaded_into_prompt(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            external = root / "external_findings"
            external.mkdir()
            (external / "outside.json").write_text(
                json.dumps({"finding_id": "outside", "title": "external"}),
                encoding="utf-8",
            )
            run_dir.mkdir()
            try:
                (run_dir / "shared_findings").symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest("filesystem does not support directory symlinks")
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer11",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )

            prompt = memory.compose_session_prompt(
                "task", session_id="session_findings_link", session_index=0
            )

            self.assertIn("New Shared Findings Since Last Session", prompt)
            self.assertNotIn("`outside`", prompt)

    def test_symlinked_results_directory_is_not_scanned_for_memory_artifacts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            external_result = root / "external_result"
            external_result.mkdir()
            (external_result / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "gen0_peer12_external",
                        "mean_return_pct": 999,
                        "leak_sentinel": "SYMLINKED_RESULT_DIR_SENTINEL",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results").mkdir(parents=True)
            try:
                (run_dir / "results" / "gen0_peer12_external").symlink_to(
                    external_result,
                    target_is_directory=True,
                )
            except OSError:
                self.skipTest("filesystem does not support directory symlinks")
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer12",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )

            prompt = memory.compose_session_prompt(
                "task", session_id="session_results_link", session_index=0
            )

            self.assertNotIn("gen0_peer12_external", prompt)
            self.assertNotIn("SYMLINKED_RESULT_DIR_SENTINEL", prompt)

    def test_symlinked_results_root_is_not_scanned_for_memory_artifacts(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            external_results = root / "external_results"
            external_variant = external_results / "gen0_peer13_external"
            external_variant.mkdir(parents=True)
            (external_variant / "tiered_eval_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "gen0_peer13_external",
                        "mean_return_pct": 1000,
                        "leak_sentinel": "SYMLINKED_RESULTS_ROOT_SENTINEL",
                    }
                ),
                encoding="utf-8",
            )
            run_dir.mkdir()
            try:
                (run_dir / "results").symlink_to(external_results, target_is_directory=True)
            except OSError:
                self.skipTest("filesystem does not support directory symlinks")
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer13",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )

            prompt = memory.compose_session_prompt(
                "task", session_id="session_results_root", session_index=0
            )

            self.assertNotIn("gen0_peer13_external", prompt)
            self.assertNotIn("SYMLINKED_RESULTS_ROOT_SENTINEL", prompt)

    def test_session_id_is_sandboxed_before_per_session_memory_filenames(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer14",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )

            memory.compose_session_prompt(
                "task",
                session_id="../escape/session",
                session_index=0,
            )

            self.assertFalse(
                (memory.memory_dir.parent / "memory_prompt_../escape/session.md").exists()
            )
            self.assertFalse((memory.memory_dir.parent / "escape").exists())
            self.assertTrue(
                any(
                    path.name.startswith("memory_prompt_unsafe-")
                    for path in memory.memory_dir.glob("memory_prompt_*.md")
                )
            )

    def test_no_follow_reader_rejects_symlinked_intermediate_directory(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            read_bounded_file_under_root_no_follow,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe_root = root / "safe"
            external = root / "external"
            safe_root.mkdir()
            external.mkdir()
            (external / "secret.json").write_text('{"leak": true}', encoding="utf-8")
            try:
                (safe_root / "linked").symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest("filesystem does not support directory symlinks")

            data = read_bounded_file_under_root_no_follow(
                safe_root / "linked" / "secret.json",
                safe_root,
                max_bytes=1024,
            )

            self.assertIsNone(data)

    def test_session_prompt_snapshots_are_bounded(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerMemoryConfig,
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer15",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
                config=PeerMemoryConfig(max_session_snapshots=2),
            )
            for idx in range(5):
                memory.compose_session_prompt(
                    "task",
                    session_id=f"session_{idx}",
                    session_index=idx,
                )

            prompt_snapshots = sorted(
                path.name for path in memory.memory_dir.glob("memory_prompt_session_*.md")
            )
            manifest_snapshots = sorted(
                path.name
                for path in memory.memory_dir.glob("session_prompt_manifest_session_*.json")
            )
            self.assertEqual(len(prompt_snapshots), 2)
            self.assertEqual(len(manifest_snapshots), 2)
            self.assertEqual(
                prompt_snapshots, ["memory_prompt_session_3.md", "memory_prompt_session_4.md"]
            )
            self.assertEqual(
                manifest_snapshots,
                [
                    "session_prompt_manifest_session_3.json",
                    "session_prompt_manifest_session_4.json",
                ],
            )

    def test_auto_handoff_replaces_stale_auto_status_without_manual_prefix(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer9",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            first = AgentResult(
                success=True,
                output={"text_outputs": ["first auto summary"]},
                duration=1.0,
                iteration_count=1,
            )
            second = AgentResult(
                success=True,
                output={"text_outputs": ["second auto summary"]},
                duration=1.0,
                iteration_count=1,
            )
            memory.record_session_result(session_id="session_first", result=first)
            memory.record_session_result(session_id="session_second", result=second)
            handoff = memory.handoff_path.read_text(encoding="utf-8")
            self.assertIn("PRAXIST_AUTO_SESSION_STATUS", handoff)
            self.assertIn("second auto summary", handoff)
            self.assertNotIn("first auto summary", handoff)

    def test_seen_findings_are_not_marked_when_runtime_never_returns_result(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            (findings / "finding_c.json").write_text(
                json.dumps({"finding_id": "finding_c", "title": "not consumed"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer10",
                generation_id=0,
                findings_dir=findings,
            )
            memory.compose_session_prompt("task", session_id="session_no_result", session_index=0)
            memory.record_session_result(
                session_id="session_no_result",
                result=None,
                error=RuntimeError("startup failed"),
            )
            self.assertFalse(memory.seen_findings_path.exists())

    def test_seen_findings_only_marks_ids_that_survive_prompt_truncation(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerMemoryConfig,
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            (findings / "finding_truncated.json").write_text(
                json.dumps({"finding_id": "finding_truncated", "title": "not in block"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer14",
                generation_id=0,
                findings_dir=findings,
                config=PeerMemoryConfig(max_prompt_chars=260),
            )
            prompt = memory.compose_session_prompt(
                "task",
                session_id="session_trunc",
                session_index=0,
            )
            self.assertNotIn("finding_truncated", prompt)
            memory.record_session_result(
                session_id="session_trunc",
                result=AgentResult(
                    success=True,
                    output={"text_outputs": ["consumed prompt"]},
                    duration=1.0,
                    iteration_count=1,
                ),
            )
            self.assertFalse(memory.seen_findings_path.exists())

    def test_seen_findings_requires_complete_rendered_finding_line(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerMemoryConfig,
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            (findings / "a.json").write_text(
                json.dumps({"finding_id": "a", "title": "not surfaced"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer18",
                generation_id=0,
                findings_dir=findings,
                config=PeerMemoryConfig(max_prompt_chars=260),
            )
            prompt = memory.compose_session_prompt(
                "task mentions a elsewhere",
                session_id="session_short_id",
                session_index=0,
            )
            self.assertIn("a", prompt)
            self.assertNotIn("not surfaced", prompt)
            memory.record_session_result(
                session_id="session_short_id",
                result=AgentResult(
                    success=True,
                    output={"text_outputs": ["consumed prompt"]},
                    duration=1.0,
                    iteration_count=1,
                ),
            )
            self.assertFalse(memory.seen_findings_path.exists())

    def test_failed_zero_iteration_result_does_not_mark_seen_findings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            (findings / "finding_failed.json").write_text(
                json.dumps({"finding_id": "finding_failed", "title": "not consumed"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer15",
                generation_id=0,
                findings_dir=findings,
            )
            memory.compose_session_prompt("task", session_id="session_failed", session_index=0)
            memory.record_session_result(
                session_id="session_failed",
                result=AgentResult(
                    success=False,
                    output={},
                    duration=0.0,
                    iteration_count=0,
                    error="auth failed",
                ),
            )
            self.assertFalse(memory.seen_findings_path.exists())

    def test_failed_zero_iteration_text_output_does_not_mark_seen_findings(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            (findings / "finding_wait.json").write_text(
                json.dumps({"finding_id": "finding_wait", "title": "not consumed"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer16",
                generation_id=0,
                findings_dir=findings,
            )
            memory.compose_session_prompt("task", session_id="session_wait", session_index=0)
            memory.record_session_result(
                session_id="session_wait",
                result=AgentResult(
                    success=False,
                    output={"text_outputs": ["What would you like me to do?"]},
                    duration=0.0,
                    iteration_count=0,
                    error="bootstrap failure",
                ),
            )
            self.assertFalse(memory.seen_findings_path.exists())

    def test_legacy_raw_seen_finding_ids_are_migrated_to_digest(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            (findings / "legacy_id.json").write_text(
                json.dumps({"finding_id": "legacy_id", "title": "already seen"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer17",
                generation_id=0,
                findings_dir=findings,
            )
            memory.memory_dir.mkdir(parents=True)
            memory.seen_findings_path.write_text(
                json.dumps(["legacy_id"]),
                encoding="utf-8",
            )
            prompt = memory.compose_session_prompt(
                "task",
                session_id="session_legacy_seen",
                session_index=0,
            )
            self.assertNotIn("legacy_id", prompt)
            seen = json.loads(memory.seen_findings_path.read_text(encoding="utf-8"))
            self.assertEqual(
                seen,
                [f"sha256:{hashlib.sha256(b'legacy_id').hexdigest()}"],
            )

    def test_session_memory_redacts_secrets_before_persistence_and_replay(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer11",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            memory.record_session_result(
                session_id="session_secret",
                result=AgentResult(
                    success=True,
                    output={"text_outputs": ["secret sk-abcdefghijklmnopqrstuvwxyz"]},
                    duration=1.0,
                    iteration_count=1,
                ),
            )
            ledger = memory.ledger_path.read_text(encoding="utf-8")
            handoff = memory.handoff_path.read_text(encoding="utf-8")
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", ledger)
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", handoff)
            self.assertIn("<redacted:openai_style_key>", ledger)
            memory.handoff_path.write_text(
                "manual secret sk-abcdefghijklmnopqrstuvwxyz\n",
                encoding="utf-8",
            )
            prompt = memory.compose_session_prompt(
                "task",
                session_id="session_replay",
                session_index=1,
            )
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", prompt)
            self.assertIn("<redacted:openai_style_key>", prompt)

    def test_prompt_memory_redacts_all_structured_sources(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        raw_secret = "sk-abcdefghijklmnopqrstuvwxyz"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            peer_id = "gen0_peer12"
            memory_dir = gen_dir / "peers" / peer_id / "memory"
            dig_dir = gen_dir / "peers" / peer_id / "dig"
            findings = run_dir / "shared_findings"
            memory_dir.mkdir(parents=True)
            dig_dir.mkdir(parents=True)
            findings.mkdir(parents=True)
            (memory_dir / "peer_state.yaml").write_text(
                yaml.safe_dump({"current_hypothesis": f"state {raw_secret}"}),
                encoding="utf-8",
            )
            (memory_dir / "experiment_ledger.jsonl").write_text(
                json.dumps({"summary": f"ledger {raw_secret}"}) + "\n",
                encoding="utf-8",
            )
            (dig_dir / "selected_contract.yaml").write_text(
                yaml.safe_dump({"variant_name": f"contract {raw_secret}"}),
                encoding="utf-8",
            )
            (findings / "finding_secret.json").write_text(
                json.dumps({"finding_id": "finding_secret", "title": f"finding {raw_secret}"}),
                encoding="utf-8",
            )
            result_dir = run_dir / "results/gen0_peer12_result"
            result_dir.mkdir(parents=True)
            (result_dir / "result_summary.json").write_text(
                json.dumps({"score": 0.9, "status": f"result {raw_secret}"}),
                encoding="utf-8",
            )

            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id=peer_id,
                generation_id=0,
                findings_dir=findings,
            )
            prompt = memory.compose_session_prompt(
                "task",
                session_id="session_redact_all",
                session_index=0,
            )
            snapshot = memory.prompt_snapshot_path.read_text(encoding="utf-8")
            self.assertNotIn(raw_secret, prompt)
            self.assertNotIn(raw_secret, snapshot)
            self.assertIn("<redacted:openai_style_key>", prompt)

    def test_memory_reads_skip_symlinked_handoff_and_writes_refuse_symlink_ledger(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import AgentResult
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            secret = root / "outside_secret.txt"
            secret.write_text("secret sk-abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer19",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            memory.memory_dir.mkdir(parents=True)
            try:
                memory.handoff_path.symlink_to(secret)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            prompt = memory.compose_session_prompt(
                "task",
                session_id="session_symlink_read",
                session_index=0,
            )
            self.assertNotIn("outside_secret", prompt)
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", prompt)

            memory.handoff_path.unlink()
            memory.ledger_path.symlink_to(secret)
            with self.assertRaises(OSError):
                memory.record_session_result(
                    session_id="session_symlink_write",
                    result=AgentResult(
                        success=True,
                        output={"text_outputs": ["done"]},
                        duration=1.0,
                        iteration_count=1,
                    ),
                )
            self.assertEqual(
                secret.read_text(encoding="utf-8"), "secret sk-abcdefghijklmnopqrstuvwxyz"
            )

    def test_prompt_build_failure_does_not_mark_unsurfaced_findings_seen(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import peer_memory
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            (findings / "finding_d.json").write_text(
                json.dumps({"finding_id": "finding_d", "title": "not surfaced"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer13",
                generation_id=0,
                findings_dir=findings,
            )

            def fail_snapshot(path: Path, text: str) -> None:
                if path.name == "memory_prompt.md":
                    raise OSError("snapshot failed")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")

            with (
                patch.object(peer_memory, "_atomic_write_text", side_effect=fail_snapshot),
                self.assertRaises(OSError),
            ):
                memory.compose_session_prompt("task", session_id="session_fail", session_index=0)
            memory.record_session_result(
                session_id="session_fail",
                result=None,
                error=RuntimeError("startup failed"),
            )
            self.assertFalse(memory.seen_findings_path.exists())

    def test_agent_loop_injects_memory_into_runtime_prompt(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            MEMORY_HEADER,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            peer_dir = gen_dir / "gen0_peer4"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer4",
                generation_id=0,
                task_prompt="Praxist task body",
                workspace=run_dir,
                max_runtime_seconds=30,
                logs_dir=peer_dir,
                findings_dir=findings,
                local_mode=True,
            )
            prompts: list[str] = []

            class FakeAgent:
                async def execute(self, task: str):
                    prompts.append(task)
                    return agent.AgentResult(
                        success=True,
                        output={"text_outputs": ["done"], "tool_uses": [{"tool": "Read"}]},
                        duration=1.0,
                        iteration_count=1,
                    )

            loop._create_agent = lambda *args, **kwargs: FakeAgent()  # type: ignore[method-assign]
            result = asyncio.run(loop._run_session())

            self.assertTrue(result.success)
            self.assertEqual(len(prompts), 1)
            self.assertIn("Praxist task body", prompts[0])
            self.assertIn(MEMORY_HEADER, prompts[0])
            self.assertTrue((gen_dir / "peers/gen0_peer4/memory/session_handoff.md").exists())

    def test_agent_loop_resolves_memory_dirs_when_logs_dir_is_run_logs(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            logs_dir = run_dir / "logs"
            findings = run_dir / "shared_findings"
            findings.mkdir(parents=True)
            dig = run_dir / "gen_2/peers/gen2_peer0/dig"
            dig.mkdir(parents=True)
            (dig / "selected_contract.yaml").write_text(
                yaml.safe_dump({"variant_name": "gen2_contract_visible"}, sort_keys=False),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"PRAXIST_RUN_DIR": ""}, clear=False):
                loop = agent.AutonomousAgentLoop(
                    peer_id="gen2_peer0",
                    generation_id=2,
                    task_prompt="task",
                    workspace=run_dir,
                    max_runtime_seconds=30,
                    logs_dir=logs_dir,
                    findings_dir=findings,
                    local_mode=True,
                )
            self.assertEqual(loop.run_dir, run_dir)
            self.assertEqual(loop.gen_dir, run_dir / "gen_2")
            prompt = loop._compose_session_task_prompt(session_id="session_000")
            self.assertIn("gen2_contract_visible", prompt)
            self.assertTrue(
                loop.peer_memory.memory_dir.is_relative_to((run_dir / "gen_2/peers").resolve())
            )

    def test_explicit_logs_dir_wins_over_stale_praxist_run_dir_for_memory_dirs(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit_run = root / "explicit_run"
            stale_run = root / "stale_run"
            logs_dir = explicit_run / "logs"
            findings = explicit_run / "shared_findings"
            findings.mkdir(parents=True)
            with patch.dict(os.environ, {"PRAXIST_RUN_DIR": str(stale_run)}, clear=False):
                loop = agent.AutonomousAgentLoop(
                    peer_id="gen3_peer0",
                    generation_id=3,
                    task_prompt="task",
                    workspace=explicit_run,
                    max_runtime_seconds=30,
                    logs_dir=logs_dir,
                    findings_dir=findings,
                    local_mode=True,
                )
            self.assertEqual(loop.run_dir, explicit_run)
            self.assertEqual(loop.gen_dir, explicit_run / "gen_3")
            self.assertTrue(
                loop.peer_memory.memory_dir.is_relative_to((explicit_run / "gen_3/peers").resolve())
            )

    def test_agent_loop_memory_initialization_failure_uses_noop(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(agent, "PeerSessionMemory", side_effect=RuntimeError("boom")):
                loop = agent.AutonomousAgentLoop(
                    peer_id="gen0_peer6",
                    generation_id=0,
                    task_prompt="base prompt",
                    workspace=root,
                    max_runtime_seconds=30,
                    logs_dir=root / "run/gen_0/gen0_peer6",
                    findings_dir=root / "run/shared_findings",
                    local_mode=True,
                )
            self.assertIsInstance(loop.peer_memory, agent.NoOpPeerSessionMemory)
            self.assertTrue(loop.peer_memory.should_wake_for_shared_finding(root / "finding.json"))
            prompts: list[str] = []

            class FakeAgent:
                async def execute(self, task: str):
                    prompts.append(task)
                    return agent.AgentResult(
                        success=True,
                        output={"text_outputs": ["done"], "tool_uses": [{"tool": "Read"}]},
                        duration=1.0,
                        iteration_count=1,
                    )

            loop._create_agent = lambda *args, **kwargs: FakeAgent()  # type: ignore[method-assign]
            result = asyncio.run(loop._run_session())
            self.assertTrue(result.success)
            self.assertEqual(prompts, ["base prompt"])

    def test_memory_failures_are_best_effort_for_agent_session(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import agent

        class BrokenMemory:
            def compose_session_prompt(self, *_args, **_kwargs):
                raise OSError("memory unavailable")

            def record_session_result(self, *_args, **_kwargs):
                raise OSError("disk full")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = agent.AutonomousAgentLoop(
                peer_id="gen0_peer5",
                generation_id=0,
                task_prompt="base prompt",
                workspace=root,
                max_runtime_seconds=30,
                logs_dir=root / "run/gen_0/gen0_peer5",
                findings_dir=root / "run/shared_findings",
                local_mode=True,
            )
            loop.peer_memory = BrokenMemory()  # type: ignore[assignment]
            prompts: list[str] = []

            class FakeAgent:
                async def execute(self, task: str):
                    prompts.append(task)
                    return agent.AgentResult(
                        success=True,
                        output={"text_outputs": ["done"], "tool_uses": [{"tool": "Read"}]},
                        duration=1.0,
                        iteration_count=1,
                    )

            loop._create_agent = lambda *args, **kwargs: FakeAgent()  # type: ignore[method-assign]
            result = asyncio.run(loop._run_session())

            self.assertTrue(result.success)
            self.assertEqual(prompts, ["base prompt"])

    def test_peer_result_matching_uses_generation_and_peer_boundaries(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            _matches_peer,
        )

        self.assertTrue(_matches_peer("gen1_peer1_alpha", "gen1_peer1", 1))
        self.assertTrue(_matches_peer("gen_1_peer_1-alpha", "gen1_peer1", 1))
        self.assertFalse(_matches_peer("gen1_peer10_alpha", "gen1_peer1", 1))
        self.assertFalse(_matches_peer("gen10_peer1_alpha", "gen1_peer1", 1))
        self.assertFalse(_matches_peer("gen2_peer1_alpha", "gen1_peer1", 1))

    def test_unsafe_peer_id_is_sandboxed_under_peers_memory_root(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="../evil",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            peers_root = (gen_dir / "peers").resolve()
            self.assertTrue(memory.memory_dir.is_relative_to(peers_root))
            self.assertNotIn("..", str(memory.memory_dir.relative_to(peers_root)))

    def test_unsafe_peer_id_does_not_read_external_dig_contract(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            outside = run_dir / "outside/dig"
            outside.mkdir(parents=True)
            (outside / "selected_contract.yaml").write_text(
                yaml.safe_dump({"variant_name": "must_not_leak"}, sort_keys=False),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="../../outside",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            prompt = memory.compose_session_prompt(
                "task",
                session_id="session_unsafe",
                session_index=0,
            )
            self.assertNotIn("must_not_leak", prompt)

    def test_unsafe_peer_id_does_not_alias_existing_peer_contract(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            real_dig = gen_dir / "peers/gen0_peer1/dig"
            real_dig.mkdir(parents=True)
            (real_dig / "selected_contract.yaml").write_text(
                yaml.safe_dump({"variant_name": "real_peer_secret"}, sort_keys=False),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="../../gen0_peer1",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            prompt = memory.compose_session_prompt(
                "task",
                session_id="session_alias",
                session_index=0,
            )
            self.assertNotIn("real_peer_secret", prompt)
            self.assertNotEqual(memory.peer_root, real_dig.parent.resolve())
            self.assertTrue(memory.safe_peer_id.startswith("unsafe-"))

    def test_noncanonical_peer_id_does_not_alias_existing_peer_contract(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            real_dig = gen_dir / "peers/gen0_peer1/dig"
            real_dig.mkdir(parents=True)
            (real_dig / "selected_contract.yaml").write_text(
                yaml.safe_dump({"variant_name": "real_peer_secret"}, sort_keys=False),
                encoding="utf-8",
            )
            for peer_id in ("gen0 peer1", "gen0@peer1", "!!!"):
                memory = PeerSessionMemory(
                    run_dir=run_dir,
                    gen_dir=gen_dir,
                    peer_id=peer_id,
                    generation_id=0,
                    findings_dir=run_dir / "shared_findings",
                )
                prompt = memory.compose_session_prompt(
                    "task",
                    session_id=f"session_{memory.safe_peer_id}",
                    session_index=0,
                )
                self.assertNotIn("real_peer_secret", prompt)
                self.assertTrue(memory.safe_peer_id.startswith("unsafe-"), peer_id)

    def test_result_artifact_scan_uses_supported_summary_shapes(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            PeerSessionMemory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            gen_dir = run_dir / "gen_0"
            result_dir = run_dir / "results/gen0_peer8_custom"
            result_dir.mkdir(parents=True)
            (result_dir / "evaluation_summary.json").write_text(
                json.dumps(
                    {
                        "variant_name": "ignored_parent_name",
                        "score": 0.71,
                        "status": "scored_complete",
                    }
                ),
                encoding="utf-8",
            )
            custom = run_dir / "results/custom_gen0_peer8_flat_result_summary.json"
            custom.write_text(
                json.dumps({"score": 0.82, "status": "scored_complete"}),
                encoding="utf-8",
            )
            memory = PeerSessionMemory(
                run_dir=run_dir,
                gen_dir=gen_dir,
                peer_id="gen0_peer8",
                generation_id=0,
                findings_dir=run_dir / "shared_findings",
            )
            artifacts = memory._scan_recent_result_artifacts()
            names = {item["variant_name"] for item in artifacts}
            self.assertIn("gen0_peer8_custom", names)
            self.assertIn("gen0_peer8_flat", names)


class PeerMemoryHealthAggregationTest(unittest.TestCase):
    def _write_state(
        self,
        run_dir: Path,
        peer_id: str,
        data: dict[str, object],
        *,
        generation_id: int = 0,
    ) -> None:
        memory_dir = run_dir / f"gen_{generation_id}" / "peers" / peer_id / "memory"
        memory_dir.mkdir(parents=True)
        payload = {
            "peer_id": peer_id,
            "generation_id": generation_id,
            "research_state": "evaluating",
            "session_count_recorded": 1,
            **data,
        }
        (memory_dir / "peer_state.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

    def _write_result(self, run_dir: Path, peer_id: str, score: float) -> None:
        result_dir = run_dir / "results" / f"{peer_id}_variant"
        result_dir.mkdir(parents=True)
        (result_dir / "result_summary.json").write_text(
            json.dumps({"variant_name": f"{peer_id}_variant", "score": score}),
            encoding="utf-8",
        )

    def test_all_green_when_peers_reach_maximize_baseline(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            for peer_id, score in (("gen0_peer0", 0.71), ("gen0_peer1", 0.66)):
                self._write_state(run_dir, peer_id, {"last_session_success": True})
                self._write_result(run_dir, peer_id, score)

            snapshot = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        self.assertEqual(snapshot.summary, {"red": 0, "yellow": 0, "green": 2})
        self.assertTrue(all(peer.health == "green" for peer in snapshot.peers))
        payload = snapshot.to_dict()
        self.assertEqual(payload["generation_id"], 0)
        self.assertEqual(payload["peers"][0]["health"], "green")

    def test_full_health_reconciles_result_tree_once_for_all_peers(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import peer_memory

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            for peer_id, score in (("gen0_peer0", 0.71), ("gen0_peer1", 0.66)):
                self._write_state(run_dir, peer_id, {"last_session_success": True})
                self._write_result(run_dir, peer_id, score)

            original_scan = peer_memory._safe_result_summary_paths
            with patch.object(
                peer_memory,
                "_safe_result_summary_paths",
                wraps=original_scan,
            ) as result_scan:
                snapshot = peer_memory.collect_peer_memory_health(
                    run_dir=run_dir,
                    generation_id=0,
                    primary_metric="score",
                    direction="maximize",
                    baselines=[{"metric_name": "score", "metric_value": 0.5}],
                )

        result_scan.assert_called_once_with(run_dir)
        self.assertEqual(snapshot.summary, {"red": 0, "yellow": 0, "green": 2})

    def test_below_baseline_is_yellow_and_minimize_uses_lower_values(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._write_state(run_dir, "gen0_peer0", {"last_session_success": True})
            self._write_state(run_dir, "gen0_peer1", {"last_session_success": True})
            self._write_result(run_dir, "gen0_peer0", 0.42)
            self._write_result(run_dir, "gen0_peer1", 0.77)

            max_snapshot = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )
            min_snapshot = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="minimize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        by_id_max = {peer.peer_id: peer for peer in max_snapshot.peers}
        self.assertEqual(by_id_max["gen0_peer0"].health, "yellow")
        self.assertEqual(by_id_max["gen0_peer0"].baseline_status, "below_baseline")
        by_id_min = {peer.peer_id: peer for peer in min_snapshot.peers}
        self.assertEqual(by_id_min["gen0_peer0"].health, "green")
        self.assertEqual(by_id_min["gen0_peer1"].health, "yellow")

    def test_failed_missing_and_malformed_peer_memory_are_red(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._write_state(
                run_dir,
                "gen0_peer0",
                {"last_session_success": False, "last_error": "runtime failed"},
            )
            (run_dir / "gen_0" / "peers" / "gen0_peer1").mkdir(parents=True)
            malformed = run_dir / "gen_0" / "peers" / "gen0_peer2" / "memory"
            malformed.mkdir(parents=True)
            (malformed / "peer_state.yaml").write_text(": not: [yaml", encoding="utf-8")

            snapshot = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        self.assertEqual(snapshot.summary["red"], 3)
        reasons = {peer.peer_id: peer.health_reason for peer in snapshot.peers}
        self.assertEqual(reasons["gen0_peer0"], "last session failed")
        self.assertEqual(reasons["gen0_peer1"], "missing peer_state.yaml")
        self.assertEqual(reasons["gen0_peer2"], "peer_state.yaml malformed")

    def test_missing_generation_returns_empty_warning_snapshot(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = collect_peer_memory_health(
                run_dir=Path(tmp) / "run",
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        self.assertEqual(snapshot.generation_id, 0)
        self.assertEqual(snapshot.summary, {"red": 0, "yellow": 0, "green": 0})
        self.assertEqual(snapshot.peers, [])
        self.assertEqual(snapshot.warnings, ["generation directory unavailable"])

    def test_latest_generation_is_discovered_when_generation_is_omitted(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._write_state(
                run_dir,
                "gen0_peer0",
                {"last_session_success": True},
                generation_id=0,
            )
            self._write_state(
                run_dir,
                "gen2_peer0",
                {"last_session_success": True},
                generation_id=2,
            )
            self._write_result(run_dir, "gen2_peer0", 0.72)

            snapshot = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=None,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        self.assertEqual(snapshot.generation_id, 2)
        self.assertEqual([peer.peer_id for peer in snapshot.peers], ["gen2_peer0"])
        self.assertEqual(snapshot.summary, {"red": 0, "yellow": 0, "green": 1})

    def test_ledger_fallback_sets_last_session_and_warning(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._write_state(run_dir, "gen0_peer0", {})
            ledger = (
                run_dir / "gen_0" / "peers" / "gen0_peer0" / "memory" / "experiment_ledger.jsonl"
            )
            ledger.write_text(
                "\n".join(
                    [
                        "",
                        "{not json}",
                        json.dumps(["not", "a", "row"]),
                        json.dumps({"session_id": "session_ledger", "success": False}),
                    ]
                ),
                encoding="utf-8",
            )

            snapshot = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        peer = snapshot.peers[0]
        self.assertEqual(peer.health, "red")
        self.assertEqual(peer.health_reason, "last session failed")
        self.assertEqual(peer.last_session_id, "session_ledger")
        self.assertFalse(peer.last_session_success)
        self.assertEqual(peer.warnings, ["experiment_ledger.jsonl has malformed rows"])

    def test_state_result_artifacts_can_drive_green_without_results_dir(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._write_state(
                run_dir,
                "gen0_peer0",
                {
                    "last_session_success": True,
                    "active_variant": "",
                    "recent_result_artifacts": [
                        "ignored",
                        {
                            "summary": "cached_variant",
                            "metrics": {"result": {"score": "0.64"}},
                        },
                    ],
                },
            )

            snapshot = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        peer = snapshot.peers[0]
        self.assertEqual(peer.health, "green")
        self.assertEqual(peer.active_variant, "cached_variant")
        self.assertEqual(peer.best_metric_value, 0.64)

    def test_lightweight_health_uses_peer_summary_without_result_tree_scan(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import peer_memory

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._write_state(
                run_dir,
                "gen0_peer0",
                {
                    "last_session_success": True,
                    "recent_result_artifacts": [
                        {
                            "variant_name": "cached_variant",
                            "metrics": {"score": 0.71},
                        }
                    ],
                },
            )
            self._write_result(run_dir, "unrelated_peer", 0.99)

            with patch.object(
                peer_memory,
                "_safe_result_summary_paths",
                side_effect=AssertionError("result tree must not be scanned"),
            ) as result_scan:
                snapshot = peer_memory.collect_peer_memory_health(
                    run_dir=run_dir,
                    generation_id=0,
                    primary_metric="score",
                    direction="maximize",
                    baselines=[{"metric_name": "score", "metric_value": 0.5}],
                    scan_result_artifacts=False,
                )

        result_scan.assert_not_called()
        self.assertEqual(snapshot.summary, {"red": 0, "yellow": 0, "green": 1})
        self.assertEqual(snapshot.peers[0].active_variant, "cached_variant")
        self.assertEqual(snapshot.peers[0].best_metric_value, 0.71)

    def test_yellow_covers_baseline_unavailable_and_missing_metric(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._write_state(run_dir, "gen0_peer0", {"last_session_success": True})
            self._write_result(run_dir, "gen0_peer0", 0.72)

            no_baseline = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[],
            )

        self.assertEqual(no_baseline.peers[0].health, "yellow")
        self.assertEqual(no_baseline.peers[0].baseline_status, "baseline_unavailable")
        self.assertEqual(no_baseline.peers[0].health_reason, "baseline unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._write_state(run_dir, "gen0_peer0", {"last_session_success": True})

            no_metric = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        self.assertEqual(no_metric.peers[0].health, "yellow")
        self.assertEqual(no_metric.peers[0].baseline_status, "no_primary_metric_result")
        self.assertEqual(no_metric.peers[0].health_reason, "no primary metric result yet")

    def test_last_error_and_blocked_research_state_are_red(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            self._write_state(
                run_dir,
                "gen0_peer0",
                {"last_session_success": True, "last_error": "tool timeout"},
            )
            self._write_state(
                run_dir,
                "gen0_peer1",
                {"last_session_success": True, "research_state": "blocked by missing data"},
            )

            snapshot = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        reasons = {peer.peer_id: peer.health_reason for peer in snapshot.peers}
        self.assertEqual(reasons["gen0_peer0"], "last session recorded an error")
        self.assertEqual(reasons["gen0_peer1"], "research_state=blocked by missing data")

    def test_health_reader_rejects_unsafe_and_oversized_memory_files(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.peer_memory import (
            DEFAULT_MAX_LEDGER_FILE_BYTES,
            DEFAULT_MAX_MEMORY_FILE_BYTES,
            collect_peer_memory_health,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            peer0_memory = run_dir / "gen_0" / "peers" / "gen0_peer0" / "memory"
            peer0_memory.mkdir(parents=True)
            (peer0_memory / "peer_state.yaml").mkdir()

            peer1_memory = run_dir / "gen_0" / "peers" / "gen0_peer1" / "memory"
            peer1_memory.mkdir(parents=True)
            (peer1_memory / "peer_state.yaml").write_text(
                "x" * (DEFAULT_MAX_MEMORY_FILE_BYTES + 1),
                encoding="utf-8",
            )

            peer2_memory = run_dir / "gen_0" / "peers" / "gen0_peer2" / "memory"
            peer2_memory.mkdir(parents=True)
            (peer2_memory / "peer_state.yaml").write_text("[]\n", encoding="utf-8")

            self._write_state(run_dir, "gen0_peer3", {"last_session_success": True})
            peer3_ledger = (
                run_dir / "gen_0" / "peers" / "gen0_peer3" / "memory" / "experiment_ledger.jsonl"
            )
            peer3_ledger.mkdir()

            self._write_state(run_dir, "gen0_peer4", {"last_session_success": True})
            peer4_ledger = (
                run_dir / "gen_0" / "peers" / "gen0_peer4" / "memory" / "experiment_ledger.jsonl"
            )
            peer4_ledger.write_text(
                "x" * (DEFAULT_MAX_LEDGER_FILE_BYTES + 1),
                encoding="utf-8",
            )

            bad_generation = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id="not-an-int",  # type: ignore[arg-type]
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )
            snapshot = collect_peer_memory_health(
                run_dir=run_dir,
                generation_id=0,
                primary_metric="score",
                direction="maximize",
                baselines=[{"metric_name": "score", "metric_value": 0.5}],
            )

        self.assertIsNone(bad_generation.generation_id)
        self.assertEqual(bad_generation.warnings, ["generation directory unavailable"])

        by_id = {peer.peer_id: peer for peer in snapshot.peers}
        self.assertEqual(by_id["gen0_peer0"].health_reason, "unsafe peer_state.yaml")
        self.assertEqual(by_id["gen0_peer1"].health_reason, "peer_state.yaml too large")
        self.assertEqual(by_id["gen0_peer2"].health_reason, "peer_state.yaml malformed")
        self.assertEqual(
            by_id["gen0_peer3"].warnings,
            ["unsafe experiment_ledger.jsonl"],
        )
        self.assertEqual(
            by_id["gen0_peer4"].warnings,
            ["experiment_ledger.jsonl too large"],
        )


if __name__ == "__main__":
    unittest.main()
