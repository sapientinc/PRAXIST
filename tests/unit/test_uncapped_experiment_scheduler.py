"""Uncapped generations keep admission, recovery, and explicit close semantics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from praxist.plugins.tools.evaluation_tools.adapter import _generation_wait_budget_seconds
from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler import (
    ExperimentSchedulerService,
)
from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client import (
    ExperimentRejected,
)
from praxist.plugins.workflow_stages.research_loop.backend.resource_scheduler import HostSnapshot
from tests.unit.test_central_experiment_scheduler import _GPUAllocator, _settings, _SupplyAllocator


class UncappedExperimentSchedulerTest(unittest.TestCase):
    def test_uncapped_generation_runs_job_and_persists_null_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            service = ExperimentSchedulerService(
                run_dir=root / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            marker = root / "completed"
            service.open_generation(4, deadline=None, cohort_size=1)
            job = service.submit(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('done')",
                        str(marker),
                    ],
                    "peer_id": "gen4_peer0",
                    "generation_id": 4,
                    "experiment_id": "uncapped-admitted-work",
                    "eta_seconds": 86400,
                }
            )
            self.assertEqual(job.state, "queued")
            status = json.loads((service.state_dir / "status.json").read_text())
            self.assertIsNone(status["generation_deadlines"]["4"])
            self.assertIsNone(_generation_wait_budget_seconds([service.run_dir], 4))
            events = [
                json.loads(line)
                for line in (service.state_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertIsNone(
                next(row for row in events if row["event"] == "generation_open")["deadline"]
            )
            json.dumps(status, allow_nan=False)
            service.start()
            try:
                self.assertEqual(service.wait(job.job_id, 5)["job"]["state"], "completed")
                self.assertEqual(marker.read_text(), "done")
                self.assertIsNone(service.status()["generation_deadlines"][4])
            finally:
                service.stop()

    def test_uncapped_generation_cancel_and_close_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary).resolve() / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir, settings=_settings(maximum=1), allocator=_GPUAllocator("")
            )
            service.open_generation(4, deadline=None)
            payload = {
                "command": [sys.executable, "-c", "raise SystemExit(99)"],
                "peer_id": "gen4_peer0",
                "generation_id": 4,
                "eta_seconds": 86400,
            }
            cancelled = service.submit({**payload, "experiment_id": "cancelled"})
            self.assertTrue(service.cancel_queued(cancelled.job_id)["cancelled"])
            closed = service.submit({**payload, "experiment_id": "closed"})
            service.freeze_generation(4, "operator_stop")
            self.assertNotEqual(closed.state, "queued")
            self.assertEqual(service.status()["queued"], 0)
            resumed = ExperimentSchedulerService(
                run_dir=run_dir, settings=_settings(maximum=1), allocator=_GPUAllocator("")
            )
            resumed.start()
            try:
                self.assertIsNone(resumed.status()["generation_deadlines"][4])
                self.assertEqual(resumed.status()["queued"], 0)
                self.assertEqual(resumed.status()["running"], 0)
                with self.assertRaises(ExperimentRejected):
                    resumed.submit({**payload, "experiment_id": "after-close"})
            finally:
                resumed.stop()

    def test_uncapped_generation_issues_supply_until_explicit_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 1
            service = ExperimentSchedulerService(
                run_dir=Path(temporary).resolve() / "run",
                settings=settings,
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(0, deadline=None, cohort_size=1)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            leases = service.status()["resource_supply"]["leases"]
            self.assertEqual([lease["peer_id"] for lease in leases], ["gen0_peer0"])
            service.freeze_generation(0, "operator_stop")
            self.assertEqual(service.status()["resource_supply"]["leases"], [])
            self.assertEqual(service.register_idle_supply("gen0_peer0", 0), {})


if __name__ == "__main__":
    unittest.main()
