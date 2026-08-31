"""The runtime prompt renderer preserves task-selected session navigation."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from praxist.core.prompt_layout import DEFAULT_START_COMMAND, sha256_text
from praxist.plugins.workflow_stages.research_loop.backend.agent import (
    BOOTSTRAP_RETRY_DIRECTIVE,
    AgentResult,
    AutonomousAgentLoop,
    resolve_prompt_with_layout,
)


class TaskStartCommandTest(unittest.TestCase):
    def test_task_start_command_replaces_metric_navigation_and_records_provenance(self) -> None:
        command = (
            "Begin by reading the assigned task, research notebook, and prior retained "
            "findings. Plan your research from that evidence."
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "base.jinja2"
            template.write_text("Your assigned research: {{ assignment }}.", encoding="utf-8")
            default_text, default_manifest = resolve_prompt_with_layout(
                template, None, None, root / "metric.md", {"assignment": "compare evidence"}
            )
            text, manifest = resolve_prompt_with_layout(
                template,
                None,
                None,
                root / "scoreless.md",
                {"assignment": "compare evidence"},
                start_command_text=command,
            )

            self.assertIn("Your assigned research: compare evidence.", text)
            self.assertIn(command, text)
            self.assertNotIn(DEFAULT_START_COMMAND.strip(), text)
            self.assertNotIn("query the frontier", text.casefold())
            self.assertIn(DEFAULT_START_COMMAND.strip(), default_text)
            self.assertEqual((root / "scoreless.md").read_text(encoding="utf-8"), text)
            self.assertEqual(manifest["rendered_prompt_hash"], sha256_text(text))
            self.assertEqual(manifest["frozen_prefix_hash"], default_manifest["frozen_prefix_hash"])
            self.assertNotEqual(
                manifest["dynamic_payload_hash"], default_manifest["dynamic_payload_hash"]
            )

    def test_absent_override_preserves_metric_start_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "base.jinja2"
            template.write_text("Research task", encoding="utf-8")
            text, _ = resolve_prompt_with_layout(
                template, None, None, root / "prompt.md", {}, start_command_text=None
            )
        self.assertIn(DEFAULT_START_COMMAND.strip(), text)

    def test_bootstrap_retry_uses_retained_evidence_only_in_scoreless_mode(self) -> None:
        for mode in ("scoreless", "metric"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first = SimpleNamespace(
                    execute=AsyncMock(
                        return_value=AgentResult(
                            success=True,
                            output={"text_outputs": ["Waiting for your instruction."]},
                            duration=0.1,
                            iteration_count=0,
                        )
                    )
                )
                retry = SimpleNamespace(
                    execute=AsyncMock(
                        return_value=AgentResult(
                            success=True,
                            output={"text_outputs": ["Research completed."]},
                            duration=0.1,
                            iteration_count=1,
                        )
                    )
                )
                loop = AutonomousAgentLoop(
                    peer_id="researcher",
                    generation_id=0,
                    task_prompt="Assigned task",
                    workspace=root,
                    logs_dir=root / "logs",
                    findings_dir=root / "findings",
                    local_mode=True,
                    max_runtime_seconds=10,
                    task_spec=SimpleNamespace(research_loop={"mode": mode}),
                )
                with (
                    patch.object(loop, "_create_agent", side_effect=[first, retry]),
                    patch.object(
                        loop, "_compose_session_task_prompt", return_value="Assigned task"
                    ),
                ):
                    result = asyncio.run(loop._run_session())

                self.assertTrue(result.success)
                first.execute.assert_awaited_once_with(task="Assigned task")
                prompt = retry.execute.await_args.kwargs["task"]
                if mode == "scoreless":
                    self.assertIn("Assigned task", prompt)
                    self.assertIn("notebook", prompt)
                    self.assertIn("retained findings", prompt)
                    self.assertNotIn("frontier", prompt)
                else:
                    self.assertEqual(
                        prompt, "Assigned task\n\n" + BOOTSTRAP_RETRY_DIRECTIVE.strip() + "\n"
                    )


if __name__ == "__main__":
    unittest.main()
