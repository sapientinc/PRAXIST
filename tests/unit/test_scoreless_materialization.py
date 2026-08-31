from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop import legacy_output_materializer as materializer
from praxist.plugins.workflow_stages.research_loop.backend.resume_state import write_boundary_marker
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import (
    write_scoreless_evidence_manifest,
)


class ScorelessMaterializationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name).resolve()
        self.prepared = SimpleNamespace(
            run_dir=self.run_dir,
            run_id="scoreless-run",
            task_ref="task:scoreless",
            peer_role_ref="task_role:research",
            task_spec=SimpleNamespace(
                research_loop={"mode": "scoreless"},
                evaluation=SimpleNamespace(primary_metric=""),
            ),
        )
        patcher = patch.object(
            materializer,
            "materialize_legacy_c5_views",
            return_value={
                "research_memory_record_count": 0,
                "graph_edge_count": 0,
                "graph_artifact_count": 0,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _freeze(self, findings: list[dict], gen_id: int = 0) -> Path:
        path = write_scoreless_evidence_manifest(
            self.run_dir,
            gen_id=gen_id,
            findings=findings,
            evidence_cutoff_at="2026-08-31T00:00:00Z",
            evidence_source_snapshot={},
        )
        write_boundary_marker(self.run_dir, gen_id=gen_id, promoted_count=0, pi_status="disabled")
        return path

    def _rows(self, path: str) -> list[dict]:
        return [json.loads(line) for line in (self.run_dir / path).read_text().splitlines()]

    def test_frozen_findings_override_mutable_sources_and_have_no_scores(self) -> None:
        frozen = {
            "id": "finding-1",
            "finding_type": "hypothesis",
            "title": "Frozen claim",
            "content": "complete evidence " * 1000,
            "custom": {"nested": ["retained"]},
            "metrics": {"observations": 7},
        }
        self._freeze([frozen])
        live = self.run_dir / "shared_findings"
        live.mkdir()
        (live / "finding-1.json").write_text(
            json.dumps({**frozen, "title": "MUTATED", "content": "replacement"})
        )
        with patch.object(
            materializer, "_collect_legacy_findings", return_value=[{**frozen, "title": "SQL"}]
        ) as mutable_reader:
            counts = materializer._materialize_legacy_outputs(
                self.prepared,
                {"frontier_summary": [{"finding_id": "fake-score", "metric_value": 1}]},
            )
        canonical = self._rows("findings/findings.jsonl")
        self.assertEqual([row["claim"] for row in canonical], ["Frozen claim"])
        self.assertEqual(canonical[0]["scores"], {})
        self.assertEqual(canonical[0]["evidence_status"], "not_scored")
        payload = json.loads(
            (self.run_dir / canonical[0]["evidence_refs"][0]["payload_path"]).read_text()
        )
        self.assertEqual(payload["legacy_finding"]["content"], frozen["content"])
        self.assertEqual(payload["legacy_finding"]["custom"], frozen["custom"])
        self.assertEqual(counts["frontier_count"], 0)
        self.assertEqual(self._rows("findings/frontier.jsonl"), [])
        mutable_reader.assert_not_called()

    def test_committed_manifest_and_delivery_are_indexed_with_source_hashes(self) -> None:
        manifest = self._freeze([])
        delivery = self.run_dir / "delivery.json"
        delivery.write_text('{"answer": [0.25, 0.75]}\n')
        digest = hashlib.sha256(delivery.read_bytes()).hexdigest()
        result = {
            "task_delivery": {
                "status": "completed",
                "artifacts": ["delivery.json"],
                "artifact_hashes": {"delivery.json": digest},
                "summary": {},
            }
        }
        materializer._materialize_legacy_outputs(self.prepared, result)
        self.assertTrue((self.run_dir / "artifact_index.jsonl").is_file())
        by_path = {row["logical_path"]: row for row in self._rows("artifact_index.jsonl")}
        for path in (manifest, delivery):
            relative = str(path.relative_to(self.run_dir))
            self.assertIn(relative, by_path)
            artifact = by_path[relative]
            self.assertEqual(artifact["artifact_status"], "committed")
            self.assertTrue(artifact["runtime_fact_source"])
            payload = json.loads((self.run_dir / artifact["payload_path"]).read_text())
            self.assertEqual(
                payload["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
            self.assertEqual(payload["payload"], json.loads(path.read_text()))
            self.assertEqual(
                artifact["content_hash"],
                "sha256:"
                + hashlib.sha256(
                    (self.run_dir / artifact["payload_path"]).read_bytes()
                ).hexdigest(),
            )

    def test_missing_committed_manifest_fails_without_using_mutable_evidence(self) -> None:
        path = self._freeze([])
        path.unlink()
        with (
            patch.object(materializer, "_collect_legacy_findings") as mutable_reader,
            self.assertRaisesRegex(ValueError, "committed scoreless evidence"),
        ):
            materializer._materialize_legacy_outputs(self.prepared, {})
        mutable_reader.assert_not_called()

    def test_changed_committed_manifest_is_not_reimported_as_frozen_evidence(self) -> None:
        path = self._freeze([{"id": "frozen", "title": "Original"}])
        changed = json.loads(path.read_text())
        changed["findings"][0]["title"] = "Changed after boundary"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "scoreless evidence.*hash"):
            materializer._materialize_legacy_outputs(self.prepared, {})

    def test_delivery_mutation_after_commit_is_rejected(self) -> None:
        self._freeze([])
        delivery = self.run_dir / "delivery.json"
        delivery.write_text('{"answer": "changed"}')
        with self.assertRaisesRegex(ValueError, "hash"):
            materializer._materialize_legacy_outputs(
                self.prepared,
                {
                    "task_delivery": {
                        "status": "completed",
                        "artifacts": ["delivery.json"],
                        "artifact_hashes": {"delivery.json": "0" * 64},
                    }
                },
            )

    def test_uncommitted_generation_and_incomplete_delivery_are_not_canonical(self) -> None:
        self._freeze([])
        write_scoreless_evidence_manifest(
            self.run_dir,
            gen_id=1,
            findings=[{"id": "not-committed", "title": "late"}],
            evidence_cutoff_at="2026-08-31T00:01:00Z",
            evidence_source_snapshot={},
        )
        materializer._materialize_legacy_outputs(
            self.prepared,
            {"task_delivery": {"status": "incomplete", "artifacts": ["missing.json"]}},
        )
        self.assertEqual(self._rows("findings/findings.jsonl"), [])
        self.assertTrue((self.run_dir / "artifact_index.jsonl").is_file())
        self.assertEqual(
            [row["logical_path"] for row in self._rows("artifact_index.jsonl")],
            ["gen_0/scoreless_evidence.json"],
        )

    def test_unsafe_delivery_paths_are_rejected_without_following_them(self) -> None:
        self._freeze([])
        outside = self.run_dir / "outside.json"
        outside.write_text('{"private": "never import"}')
        (self.run_dir / "alias.json").symlink_to(outside)
        os.mkfifo(self.run_dir / "pipe.json")
        (self.run_dir / "alias_dir").symlink_to(self.run_dir, target_is_directory=True)
        for relative in ("alias.json", "alias_dir/outside.json", "../outside.json", "pipe.json"):
            with self.subTest(relative=relative), self.assertRaises((ValueError, OSError)):
                materializer._materialize_legacy_outputs(
                    self.prepared,
                    {
                        "task_delivery": {
                            "status": "completed",
                            "artifacts": [relative],
                            "artifact_hashes": {relative: "0" * 64},
                        }
                    },
                )

    def test_hardlinked_delivery_is_not_read_as_a_committed_output(self) -> None:
        self._freeze([])
        data = b'{"private": "must not import"}'
        (self.run_dir / "source.json").write_bytes(data)
        os.link(self.run_dir / "source.json", self.run_dir / "linked.json")
        with self.assertRaises((ValueError, OSError)):
            materializer._materialize_legacy_outputs(
                self.prepared,
                {
                    "task_delivery": {
                        "status": "completed",
                        "artifacts": ["linked.json"],
                        "artifact_hashes": {"linked.json": hashlib.sha256(data).hexdigest()},
                    }
                },
            )

    def test_noncommitted_or_symlinked_boundary_cannot_authorize_evidence(self) -> None:
        self._freeze([{"id": "finding", "title": "must not be imported"}])
        marker = self.run_dir / "gen_0/generation_boundary.json"
        partial = json.loads(marker.read_text())
        partial["artifact_semantics"]["status"] = "partial"
        partial["artifact_semantics"]["runtime_fact_source"] = False
        marker.write_text(json.dumps(partial))
        with self.assertRaisesRegex(ValueError, "boundary"):
            materializer._materialize_legacy_outputs(self.prepared, {})
        marker.unlink()
        outside = self.run_dir / "other.json"
        outside.write_text("{}")
        marker.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "boundary"):
            materializer._materialize_legacy_outputs(self.prepared, {})

    def test_binary_delivery_is_redacted_before_base64_encoding(self) -> None:
        self._freeze([])
        data = b"\xff binary header\x00 Bearer synthetic_test_token_12345"
        (self.run_dir / "binary.bin").write_bytes(data)
        materializer._materialize_legacy_outputs(
            self.prepared,
            {
                "task_delivery": {
                    "status": "completed",
                    "artifacts": ["binary.bin"],
                    "artifact_hashes": {"binary.bin": hashlib.sha256(data).hexdigest()},
                }
            },
        )
        artifact = next(
            row for row in self._rows("artifact_index.jsonl") if row["logical_path"] == "binary.bin"
        )
        payload = json.loads((self.run_dir / artifact["payload_path"]).read_text())
        self.assertNotIn(b"synthetic_test_token_12345", base64.b64decode(payload["payload"]))
        self.assertIn("bearer_token", payload["source_redaction_hits"])

    def test_text_and_binary_delivery_formats_are_preserved(self) -> None:
        self._freeze([])
        contents = {
            "report.md": b"# Result\n\nEvidence retained.\n",
            "plot.bin": b"\x89\x00\xff\x01",
        }
        for relative, data in contents.items():
            (self.run_dir / relative).write_bytes(data)
        materializer._materialize_legacy_outputs(
            self.prepared,
            {
                "task_delivery": {
                    "status": "completed",
                    "artifacts": list(contents),
                    "artifact_hashes": {
                        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
                    },
                }
            },
        )
        self.assertTrue((self.run_dir / "artifact_index.jsonl").is_file())
        by_path = {row["logical_path"]: row for row in self._rows("artifact_index.jsonl")}
        self.assertEqual(set(by_path), {"gen_0/scoreless_evidence.json", *contents})
        text_payload = json.loads((self.run_dir / by_path["report.md"]["payload_path"]).read_text())
        self.assertEqual(text_payload["payload"], contents["report.md"].decode())
        binary_payload = json.loads(
            (self.run_dir / by_path["plot.bin"]["payload_path"]).read_text()
        )
        self.assertEqual(base64.b64decode(binary_payload["payload"]), contents["plot.bin"])


if __name__ == "__main__":
    unittest.main()
