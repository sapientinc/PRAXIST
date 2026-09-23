from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class Issue83GenerationContractsTest(unittest.TestCase):
    def test_cohort_rejects_zero_evaluation_before_writing_results(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import cohort_runner

        class FakeScheduler:
            def __init__(self) -> None:
                self.calls: list[str] = []

            @property
            def settings(self) -> SimpleNamespace:
                return SimpleNamespace(
                    mature_supply_fraction=0,
                    mature_supply_redundancy=0,
                )

            def open_generation(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self.calls.append("open")

            def configure_generation_maturity(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                self.calls.append("maturity")

            def require_generation_evaluation(self, generation_id: int) -> None:
                self.calls.append(f"guard:{generation_id}")
                raise RuntimeError("generation completed without any evaluation jobs")

        class FakeTrigger:
            fired = False
            closing = False
            assessment_started = False
            required_mature_result_peers = 0

            def __init__(self, **kwargs: object) -> None:
                del kwargs

            async def wait_until_fire(self, *, abort_event: asyncio.Event) -> None:
                del abort_event

            async def evaluate_async(self) -> SimpleNamespace:
                return SimpleNamespace(fired=False)

            def fire(self, snapshot: object) -> None:
                del snapshot
                type(self).fired = True

            def write_postgen_marker(self, snapshot: object) -> None:
                del snapshot

            def mature_peer_count(self) -> int:
                return 0

            def mature_result_count(self) -> int:
                return 0

            def mature_peer_ids(self) -> set[str]:
                return set()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scheduler = FakeScheduler()
            loop = SimpleNamespace(
                run_dir=root,
                workspace=root,
                mcp_servers={},
                _peer_allowed_tools=[],
                _findings_sync=None,
                _experiment_scheduler=scheduler,
                task_spec=SimpleNamespace(
                    generation_policy=SimpleNamespace(
                        cohort_size=0,
                        per_generation_hours=0,
                    ),
                    synthesis_trigger=SimpleNamespace(
                        enabled=True,
                        min_findings=0,
                        min_interval_minutes=1,
                        max_interval_minutes=2,
                        min_contributing_peers=0,
                        poll_interval_seconds=1,
                    ),
                    agent=SimpleNamespace(
                        premium_mode=False,
                        reasoning_effort="low",
                    ),
                ),
            )

            with (
                patch.object(cohort_runner, "_PEER_DRAIN_GRACE_SECONDS", 0),
                patch.object(
                    cohort_runner,
                    "_start_generation_deadline_watchdog",
                    return_value=(
                        threading.Event(),
                        SimpleNamespace(join=lambda timeout=None: None),
                    ),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "synthesis_trigger.SynthesisTrigger",
                    FakeTrigger,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client.freeze_generation",
                ),
                self.assertRaisesRegex(RuntimeError, "without any evaluation jobs"),
            ):
                asyncio.run(cohort_runner.run_generation_cohort(loop, 0))

            self.assertEqual(scheduler.calls, ["open", "maturity", "guard:0"])
            self.assertFalse((root / "gen_0" / "generation_results.json").exists())

    def test_generation_loop_rejects_zero_evaluation_before_boundary(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend import generation_loop

        class FakeScheduler:
            def __init__(self) -> None:
                self.checked_generations: list[int] = []

            def require_generation_evaluation(self, generation_id: int) -> None:
                self.checked_generations.append(generation_id)
                raise RuntimeError("generation completed without any evaluation jobs")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scheduler = FakeScheduler()
            loop = object.__new__(generation_loop.GenerationLoop)
            loop.task_spec = SimpleNamespace(
                task_id="task",
                task_name="Task",
                generation_policy=SimpleNamespace(
                    max_generations=1,
                    cohort_size=0,
                ),
                pi_agent=SimpleNamespace(enabled=False),
                agent=SimpleNamespace(
                    premium_mode=False,
                    reasoning_effort="low",
                ),
            )
            loop.workspace = root
            loop.run_dir = root
            loop.findings_dir = root / "findings"
            loop.local_mode = True
            loop.model = "fake"
            loop.mcp_servers = {}
            loop.resume = False
            loop.resume_policy = "completed_generation"
            loop.frontier_strategy = "auto"
            loop.frontier = SimpleNamespace(get_summary=lambda: [])
            loop.gems = SimpleNamespace(enabled=False, load_state=lambda: {})
            loop._state_lock = threading.Lock()
            loop._current_generation = 0
            loop._generations_completed = 0
            loop._experiment_scheduler = scheduler

            async def empty_generation(_generation_id: int) -> list[dict[str, object]]:
                return []

            loop._run_generation = empty_generation  # type: ignore[method-assign]

            def start_sidecars(orchestrator: object, resume_plan: object = None) -> None:
                del resume_plan
                orchestrator._experiment_scheduler = scheduler  # type: ignore[attr-defined]

            with (
                patch.object(
                    generation_loop,
                    "enter_orchestrator_runtime_scope",
                    return_value=SimpleNamespace(close=lambda: None),
                ),
                patch.object(generation_loop, "configure_runtime_environment"),
                patch.object(generation_loop, "initialize_local_store_if_needed"),
                patch.object(generation_loop, "validate_baseline_cache_for_run"),
                patch.object(generation_loop, "start_sidecars", side_effect=start_sidecars),
                patch.object(
                    generation_loop,
                    "evaluate_run_stop_gate",
                    return_value=SimpleNamespace(should_stop=False),
                ),
                patch.object(generation_loop, "complete_generation_boundary") as complete_boundary,
                patch.object(generation_loop, "write_run_summary"),
                patch.object(
                    generation_loop,
                    "canonical_completed_generation_count",
                    return_value=0,
                ),
                patch.object(generation_loop, "close_sidecars_and_runtime"),
                self.assertRaisesRegex(RuntimeError, "without any evaluation jobs"),
            ):
                asyncio.run(loop.run())

            self.assertEqual(scheduler.checked_generations, [0])
            complete_boundary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
