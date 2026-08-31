"""Filesystem ingestion cannot copy private linked files into shared findings."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend.tools import (
    findings_ingest,
    findings_sync,
    local_store,
)


class FindingsSafeIngestionTest(unittest.TestCase):
    def test_legacy_alias_is_read_only_outside_controller_mode_with_original_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / "canonical.json"
            target.write_text(
                json.dumps({"title": "Public finding", "content": "Allowed narrative"}),
                encoding="utf-8",
            )
            alias = root / "alias.json"
            alias.symlink_to(target)

            with patch.dict(os.environ, {}, clear=True):
                row = findings_ingest.parse_finding_file(alias)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["content"], "Allowed narrative")
            self.assertEqual(row["source_filepath"], str(alias))
            self.assertEqual(row["source_filename"], alias.name)

            with patch.dict(os.environ, {"PRAXIST_CONTROLLER_STATE_DIR": str(root / "private")}):
                self.assertIsNone(findings_ingest.parse_finding_file(alias))

    def test_ingestion_and_background_sync_skip_linked_private_files(self) -> None:
        for entrypoint in ("ingest", "sync"):
            for link_kind in ("leaf", "parent", "hardlink"):
                with (
                    self.subTest(entrypoint=entrypoint, link_kind=link_kind),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp).resolve()
                    run_dir = root / "run"
                    findings_dir = run_dir / "shared_findings"
                    findings_dir.mkdir(parents=True)
                    private_dir = root / "controller_private"
                    private_dir.mkdir()
                    private_file = private_dir / "control.json"
                    private_payload = json.dumps(
                        {
                            "title": "Private control",
                            "content": "SYNTHETIC_PRIVATE_CONTROL_MUST_NOT_BE_INGESTED",
                            "finding_type": "insight",
                            "peer_id": "gen0_peer0",
                        }
                    )
                    private_file.write_text(private_payload, encoding="utf-8")
                    (findings_dir / "public.json").write_text(
                        json.dumps(
                            {
                                "title": "Public finding",
                                "content": "Allowed narrative",
                                "finding_type": "hypothesis",
                                "peer_id": "gen0_peer0",
                            }
                        ),
                        encoding="utf-8",
                    )
                    linked_dir = None
                    if link_kind == "parent":
                        generation_dir = run_dir / "gen_0"
                        generation_dir.mkdir()
                        linked_dir = generation_dir / "shared_findings"
                        linked_dir.symlink_to(private_dir, target_is_directory=True)
                    elif link_kind == "leaf":
                        (findings_dir / "linked.json").symlink_to(private_file)
                    else:
                        os.link(private_file, findings_dir / "linked.json")

                    with patch.dict(
                        os.environ,
                        {
                            "LOCAL_STORE_DIR": str(root / "database"),
                            "PRAXIST_CONTROLLER_STATE_DIR": str(private_dir),
                        },
                    ):
                        if entrypoint == "ingest":
                            findings_ingest.ingest_findings_directory(findings_dir)
                            if linked_dir is not None:
                                findings_ingest.ingest_findings_directory(linked_dir)
                        else:
                            findings_sync.FindingsSync(
                                findings_dir, run_dir=run_dir, local_mode=True
                            ).sync_once()
                        rows = local_store.get_all_findings()

                    self.assertEqual(
                        [(row["title"], row["content"]) for row in rows],
                        [("Public finding", "Allowed narrative")],
                    )
                    self.assertEqual(private_file.read_text(), private_payload)
