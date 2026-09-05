from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from praxist.plugins.workflow_stages.research_loop.backend import protected_pids
from praxist.plugins.workflow_stages.research_loop.backend.experiment_process import (
    process_group_alive,
)
from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler import (
    ExperimentSchedulerService,
    _ActiveJob,
    _RequestHandler,
    _SupplyLease,
)
from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client import (
    ExperimentRejected,
    SchedulerUnavailable,
    freeze_generation,
    rebase_recovered_task_context,
    submit_and_wait,
    task_runtime_context_changed,
)
from praxist.plugins.workflow_stages.research_loop.backend.resource_scheduler import (
    Allocation,
    GPUDevice,
    GPUProcess,
    HostAllocationRegistry,
    HostSnapshot,
    ResourceAllocator,
    SchedulerSettings,
)


class _GPUAllocator:
    def __init__(self, uuid: str, limit: int = 2) -> None:
        self.uuid = uuid
        self.concurrency_limit = limit
        self.snapshot = HostSnapshot(8, 10, 10, 0)
        self.released: list[str] = []
        self.active: set[str] = set()

    def refresh(self, *, queued: bool) -> HostSnapshot:
        return self.snapshot

    def reserve(self, **kwargs) -> Allocation | None:
        if len(self.active) >= self.concurrency_limit:
            return None
        self.active.add(kwargs["allocation_id"])
        return Allocation(
            allocation_id=kwargs["allocation_id"],
            run_id=kwargs["run_id"],
            pid=kwargs["pid"],
            pgid=kwargs["pgid"],
            profile=kwargs["profile"].name,
            gpu_uuids=(self.uuid,) if self.uuid else (),
            gpu_memory_mb=1024,
            gpu_utilization_pct=20,
            started_at=time.time(),
        )

    def bind_process(self, allocation_id: str, *, pid: int, pgid: int) -> bool:
        return allocation_id in self.active

    def recover_allocation(self, **kwargs) -> Allocation:
        self.active.add(kwargs["allocation_id"])
        return Allocation(
            allocation_id=kwargs["allocation_id"],
            run_id=kwargs["run_id"],
            pid=kwargs["pid"],
            pgid=kwargs["pgid"],
            profile=kwargs["profile"].name,
            gpu_uuids=tuple(kwargs["gpu_uuids"]),
            gpu_memory_mb=1024,
            gpu_utilization_pct=20,
            started_at=time.time(),
        )

    def release(self, allocation_id: str) -> None:
        self.released.append(allocation_id)
        self.active.discard(allocation_id)

    def describe_allocation_activity(self, allocation: Allocation) -> dict[str, object]:
        return {"state": "gpu_compute_active" if allocation.gpu_uuids else "non_gpu_allocation"}


class _StaticObserver:
    def snapshot(self) -> HostSnapshot:
        return HostSnapshot(8, 10, 10, 0)


class _BlockedAllocator(_GPUAllocator):
    def reserve(self, **kwargs) -> None:
        return None


class _BindFailAllocator(_GPUAllocator):
    def bind_process(self, allocation_id: str, *, pid: int, pgid: int) -> None:
        raise OSError("binding ledger unavailable")


class _BindingMissingAllocator(_GPUAllocator):
    def bind_process(self, allocation_id: str, *, pid: int, pgid: int) -> bool:
        self.active.discard(allocation_id)
        return False


class _SupplyAllocator(_GPUAllocator):
    def __init__(self, slots: tuple[tuple[str, ...], ...]) -> None:
        super().__init__("", limit=max(1, len(slots)))
        self.slots = slots
        self.claims: dict[str, tuple[str, ...]] = {}

    def claim_supply(self, *, lease_id: str, run_id: str, expires_at: float) -> tuple[str, ...]:
        del run_id, expires_at
        if len(self.claims) >= len(self.slots):
            return ()
        profiles = self.slots[len(self.claims)]
        self.claims[lease_id] = profiles
        return profiles

    def supply_claim_valid(self, lease_id: str) -> bool:
        return lease_id in self.claims

    def release(self, allocation_id: str) -> None:
        self.claims.pop(allocation_id, None)
        super().release(allocation_id)


def _settings(*, maximum: int = 2, initial: int | None = None) -> SchedulerSettings:
    return SchedulerSettings.from_dict(
        {
            "mode": "central",
            "initial_concurrent_experiments": initial or maximum,
            "max_concurrent_experiments": maximum,
            "adjustment_interval_seconds": 2,
            "profiles": {
                "cpu": {"accelerator": "cpu"},
                "gpu": {
                    "accelerator": "gpu",
                    "gpu_memory_gb": 20,
                    "gpu_utilization_pct": 40,
                },
            },
            "default_profile": "cpu",
        }
    )


def _write_active_attempt(run_dir: Path, attempt_id: str) -> Path:
    attempt_dir = run_dir / "resource_scheduler" / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "READY.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "attempt_id": attempt_id,
            }
        ),
        encoding="utf-8",
    )
    (attempt_dir / "GO.json").write_text('{"go":true}\n', encoding="utf-8")
    return attempt_dir


class ExperimentSchedulerTest(unittest.TestCase):
    def test_rpc_handler_ignores_client_disconnect_while_writing_response(self) -> None:
        handler = object.__new__(_RequestHandler)
        handler.server = MagicMock()
        handler.server.service.handle_request.return_value = {"ok": True}
        handler.rfile = MagicMock()
        handler.rfile.readline.return_value = b'{"action":"ping"}\n'
        handler.wfile = MagicMock()
        handler.wfile.write.side_effect = BrokenPipeError("client exited")

        handler.handle()

        handler.server.service.handle_request.assert_called_once_with({"action": "ping"})

    def test_unavailable_accelerator_rejects_only_gpu_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("", limit=2)
            allocator.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                accelerator_probe_state="unavailable",
                accelerator_probe_reason="nvidia-smi is not installed",
            )
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=2),
                allocator=allocator,
            )
            cpu = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "cpu-still-works",
                    "profile": "cpu",
                }
            )
            self.assertEqual(cpu.state, "queued")
            with self.assertRaisesRegex(ExperimentRejected, "accelerator_unavailable"):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer1",
                        "generation_id": 0,
                        "experiment_id": "gpu-cannot-run",
                        "profile": "gpu",
                    }
                )

    def test_transient_accelerator_probe_has_bounded_queue_wait(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("", limit=2)
            allocator.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                accelerator_probe_state="unknown",
                accelerator_probe_reason="driver query timed out",
            )
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=2),
                allocator=allocator,
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "gpu-probe-recovery-window",
                    "profile": "gpu",
                }
            )
            assert service._accelerator_probe_unknown_since is not None
            service._accelerator_probe_unknown_since -= 121
            service._launch_ready_jobs()
            self.assertEqual(job.state, "rejected")
            self.assertIn("accelerator_probe_unknown", job.error)
            scheduler_status = service.status()
            self.assertEqual(scheduler_status["accelerator_probe"]["state"], "unknown")

    def test_same_experiment_can_be_resubmitted_after_accelerator_probe_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _BlockedAllocator("", limit=1)
            allocator.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                accelerator_probe_state="unknown",
                accelerator_probe_reason="driver query timed out",
            )
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            request = {
                "command": [sys.executable, "-c", "pass"],
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "experiment_id": "gpu-probe-retry",
                "profile": "gpu",
            }
            rejected = service.submit(request)
            assert service._accelerator_probe_unknown_since is not None
            service._accelerator_probe_unknown_since -= 121
            service._launch_ready_jobs()
            self.assertEqual(rejected.state, "rejected")

            allocator.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                gpus=(GPUDevice(0, "GPU-a", 40 * 1024, 39 * 1024, 100),),
                accelerator_probe_state="available",
            )
            retried = service.submit(request)

            self.assertNotEqual(retried.job_id, rejected.job_id)
            self.assertEqual(retried.state, "queued")
            self.assertEqual(service._semantic_jobs[retried.semantic_key], retried.job_id)

    def test_probe_rejection_remains_resubmittable_after_scheduler_restart(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            unavailable = _BlockedAllocator("", limit=1)
            unavailable.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                accelerator_probe_state="unknown",
                accelerator_probe_reason="driver query timed out",
            )
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=unavailable,
            )
            request = {
                "command": [sys.executable, "-c", "pass"],
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "experiment_id": "gpu-probe-restart-retry",
                "profile": "gpu",
            }
            rejected = first.submit(request)
            assert first._accelerator_probe_unknown_since is not None
            first._accelerator_probe_unknown_since -= 121
            first._launch_ready_jobs()
            self.assertEqual(rejected.state, "rejected")

            available = _BlockedAllocator("", limit=1)
            available.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                gpus=(GPUDevice(0, "GPU-a", 40 * 1024, 39 * 1024, 100),),
                accelerator_probe_state="available",
            )
            resumed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=available,
            )
            resumed._recover_terminal_events()
            retried = resumed.submit(request)

            self.assertNotEqual(retried.job_id, rejected.job_id)
            self.assertEqual(retried.state, "queued")
            self.assertEqual(resumed._semantic_jobs[retried.semantic_key], retried.job_id)

    def test_old_gpu_job_survives_first_transient_probe_unknown_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _BlockedAllocator("", limit=1)
            allocator.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                gpus=(GPUDevice(0, "GPU-a", 40 * 1024, 39 * 1024, 100),),
                accelerator_probe_state="available",
            )
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "old-job-first-probe-gap",
                    "profile": "gpu",
                }
            )
            job.submitted_at -= 3600
            allocator.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                accelerator_probe_state="unknown",
                accelerator_probe_reason="single inventory timeout",
            )

            service._launch_ready_jobs()

            self.assertEqual(job.state, "queued")
            self.assertIn(job.job_id, service._queue)
            self.assertIsNotNone(service._accelerator_probe_unknown_since)

    def test_structurally_impossible_gpu_profiles_are_rejected_without_queueing(self) -> None:
        gpu = GPUDevice(0, "GPU-a", 40 * 1024, 0, 0)
        for label, profile, expected in (
            (
                "count",
                {
                    "accelerator": "gpu",
                    "gpu_count": 2,
                    "gpu_memory_gb": 10,
                    "gpu_utilization_pct": 20,
                },
                "requests 2 GPUs",
            ),
            (
                "memory",
                {
                    "accelerator": "gpu",
                    "gpu_count": 1,
                    "gpu_memory_gb": 80,
                    "gpu_utilization_pct": 20,
                },
                "requests 80 GiB",
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                settings = SchedulerSettings.from_dict(
                    {
                        "mode": "central",
                        "initial_concurrent_experiments": 1,
                        "max_concurrent_experiments": 1,
                        "profiles": {"gpu": profile},
                        "default_profile": "gpu",
                    }
                )
                allocator = _BlockedAllocator("", limit=1)
                allocator.snapshot = HostSnapshot(8, 10, 10, 0, gpus=(gpu,))
                service = ExperimentSchedulerService(
                    run_dir=Path(td) / "run",
                    settings=settings,
                    allocator=allocator,
                )
                with self.assertRaisesRegex(
                    ExperimentRejected,
                    f"accelerator_profile_unsatisfied: .*{expected}",
                ):
                    service.submit(
                        {
                            "command": [sys.executable, "-c", "pass"],
                            "peer_id": "gen0_peer0",
                            "generation_id": 0,
                            "experiment_id": f"impossible-{label}",
                            "profile": "gpu",
                        }
                    )

    def test_temporarily_busy_compatible_gpu_profile_remains_queued(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _BlockedAllocator("", limit=1)
            allocator.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                gpus=(GPUDevice(0, "GPU-a", 40 * 1024, 39 * 1024, 100),),
                accelerator_probe_state="available",
            )
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "temporarily-busy-gpu",
                    "profile": "gpu",
                }
            )
            job.submitted_at -= 3600
            service._launch_ready_jobs()

            self.assertEqual(job.state, "queued")
            self.assertIn(job.job_id, service._queue)

    def test_existing_gpu_semantic_job_survives_later_probe_outage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("GPU-a", limit=1)
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            request = {
                "command": [sys.executable, "-c", "pass"],
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "experiment_id": "stable-semantic-job",
                "profile": "gpu",
            }
            existing = service.submit(request)
            allocator.snapshot = HostSnapshot(
                8,
                10,
                10,
                0,
                accelerator_probe_state="unavailable",
                accelerator_probe_reason="temporary inventory loss",
            )

            duplicate = service.submit(request)

            self.assertIs(duplicate, existing)
            self.assertEqual(existing.state, "queued")

    def test_same_peer_job_waits_in_queue_until_active_job_drains(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
                max_parallel_runs_per_peer=1,
            )
            first = service.submit(
                {
                    "command": [sys.executable, "-c", "import time; time.sleep(0.8)"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "first",
                    "profile": "cpu",
                }
            )
            second = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "second",
                    "profile": "cpu",
                }
            )
            third = service.submit(
                {
                    "command": [sys.executable, "-c", "import time; time.sleep(0.6)"],
                    "peer_id": "gen0_peer1",
                    "generation_id": 0,
                    "experiment_id": "other-peer",
                    "profile": "cpu",
                }
            )
            service.start()
            try:
                deadline = time.time() + 3
                while (
                    not {first.job_id, third.job_id} <= service._active.keys()
                    and time.time() < deadline
                ):
                    time.sleep(0.02)
                status = service.status()
                self.assertEqual(status["running"], 2)
                self.assertEqual(status["queued"], 1)
                self.assertEqual(status["peer_capacity_blocked"], 1)
                self.assertEqual(second.attempts, 0)
                queued = next(job for job in status["jobs"] if job["state"] == "queued")
                self.assertEqual(queued["queue_blocked_reason"], "per_peer_capacity")
                self.assertEqual(
                    status["running_activity"]["by_resource_phase"],
                    {"non_gpu_allocation": 2},
                )
                running = next(job for job in status["jobs"] if job["state"] == "running")
                self.assertEqual(running["resource_activity"]["state"], "non_gpu_allocation")
                self.assertEqual(service.wait(second.job_id, 5)["job"]["state"], "completed")
            finally:
                service.stop()
            events = (service.state_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("ProtectedPidCapacityError", events)

    def test_durable_command_references_environment_secrets_without_plaintext(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            with patch.dict(os.environ, {"SERVICE_TOKEN": "opaque-secret-value"}, clear=False):
                job = service.submit(
                    {
                        "command": [
                            sys.executable,
                            "task.py",
                            "--auth=opaque-secret-value",
                        ],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "secret-reference",
                        "profile": "cpu",
                        "environment": dict(os.environ),
                    }
                )
                event_text = (service.state_dir / "events.jsonl").read_text(encoding="utf-8")
                self.assertNotIn("opaque-secret-value", event_text)
                self.assertIn("__PRAXIST_ENV_REF_SERVICE_TOKEN__", event_text)

                recovered = ExperimentSchedulerService(
                    run_dir=service.run_dir,
                    settings=_settings(maximum=1),
                    allocator=_GPUAllocator(""),
                )
                recovered._recover_terminal_events()
                self.assertEqual(
                    recovered._jobs[job.job_id].command[-1],
                    "--auth=opaque-secret-value",
                )

    def test_recovered_queued_task_reapplies_task_runtime_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            task_root = root / "task"
            evaluator = task_root / "evaluations" / "run.py"
            task_python = task_root / ".venv" / "bin" / "python"
            evaluator.parent.mkdir(parents=True)
            task_python.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            task_python.write_text("", encoding="utf-8")
            service = ExperimentSchedulerService(
                run_dir=root / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            event = {
                "command": ["python3.8", "evaluations/run.py"],
                "environment_values": {
                    "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                    "PRAXIST_TASK_PYTHON": str(task_python),
                    "PYTHONPATH": "/runner/python313",
                    "PYTHONHOME": "/runner",
                    "PATH": "/usr/bin",
                },
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "recovered-task-boundary",
                "profile": "cpu",
                "cwd": str(root / "run"),
            }

            recovered = service._job_from_event("job", event, state="queued")

        self.assertEqual(recovered.command[0], str(task_python))
        self.assertEqual(recovered.command[1], str(evaluator.resolve()))
        self.assertNotIn("PYTHONPATH", recovered.environment)
        self.assertNotIn("PYTHONHOME", recovered.environment)
        self.assertEqual(
            recovered.environment["PATH"].split(os.pathsep)[0],
            str(task_python.parent),
        )

    def test_recovered_job_resolves_relative_cwd_against_task_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            work_dir = task_root / "work"
            work_dir.mkdir(parents=True)
            service = ExperimentSchedulerService(
                run_dir=root / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            event = {
                "command": [sys.executable, "-c", "pass"],
                "environment_values": {"PRAXIST_TASK_PROJECT_PATH": str(task_root)},
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "relative-recovery-cwd",
                "profile": "cpu",
                "cwd": "work",
            }

            recovered = service._job_from_event("job", event, state="queued")

        self.assertEqual(recovered.cwd, str(work_dir.resolve()))

    def test_recovered_job_rebases_task_owned_paths_after_checkout_moves(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            old_root = root / "old" / "task"
            new_root = root / "new" / "task"
            old_workspace = root / "old"
            new_workspace = root / "new"
            new_evaluator = new_root / "evaluations" / "run.py"
            new_python = new_root / ".venv" / "bin" / "python"
            new_work = new_root / "work"
            new_evaluator.parent.mkdir(parents=True)
            new_python.parent.mkdir(parents=True)
            new_work.mkdir(parents=True)
            new_evaluator.write_text("print('ok')\n", encoding="utf-8")
            new_python.write_text("", encoding="utf-8")
            service = ExperimentSchedulerService(
                run_dir=root / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            old_python = old_root / ".venv" / "bin" / "python"
            event = {
                "command": [str(old_python), str(old_root / "evaluations" / "run.py")],
                "environment_values": {
                    "PRAXIST_TASK_PROJECT_PATH": str(old_root),
                    "PRAXIST_WORKSPACE_ROOT": str(old_workspace),
                    "PRAXIST_DATASETS_DIR": str(old_workspace / "datasets"),
                    "PRAXIST_TASK_RUNTIME_ENV_KEYS": "TASK_DATA_ALIAS",
                    "TASK_DATA_ALIAS": str(old_workspace / "task_data"),
                    "EXTERNAL_DATA": "/datasets/shared",
                    "PRAXIST_TASK_PYTHON": str(old_python),
                    "PRAXIST_TASK_VENV": str(old_root / ".venv"),
                    "VIRTUAL_ENV": str(old_root / ".venv"),
                    "PATH": f"{old_root / '.venv' / 'bin'}:/usr/bin",
                },
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "relocated-task",
                "profile": "cpu",
                "cwd": str(old_root / "work"),
            }
            with patch.dict(
                os.environ,
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(new_root),
                    "PRAXIST_WORKSPACE_ROOT": str(new_workspace),
                    "PRAXIST_DATASETS_DIR": str(new_workspace / "datasets"),
                    "PRAXIST_TASK_RUNTIME_ENV_KEYS": "TASK_DATA_ALIAS",
                    "TASK_DATA_ALIAS": str(new_workspace / "task_data"),
                    "PRAXIST_TASK_PYTHON": str(new_python),
                    "PRAXIST_TASK_VENV": str(new_root / ".venv"),
                    "VIRTUAL_ENV": str(new_root / ".venv"),
                    "PATH": f"{new_root / '.venv' / 'bin'}:/usr/bin",
                },
                clear=True,
            ):
                recovered = service._job_from_event("job", event, state="queued")

        self.assertEqual(recovered.command, [str(new_python), str(new_evaluator)])
        self.assertEqual(recovered.cwd, str(new_work))
        self.assertEqual(recovered.environment["PRAXIST_TASK_PROJECT_PATH"], str(new_root))
        self.assertEqual(recovered.environment["PRAXIST_WORKSPACE_ROOT"], str(new_workspace))
        self.assertEqual(
            recovered.environment["PRAXIST_DATASETS_DIR"], str(new_workspace / "datasets")
        )
        self.assertEqual(recovered.environment["TASK_DATA_ALIAS"], str(new_workspace / "task_data"))
        self.assertEqual(recovered.environment["EXTERNAL_DATA"], "/datasets/shared")
        self.assertEqual(recovered.environment["PRAXIST_TASK_PYTHON"], str(new_python))
        self.assertNotIn(str(old_root), json.dumps(recovered.environment, sort_keys=True))

    def test_recovered_task_relocation_does_not_rewrite_path_prefix_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_root = root / "task"
            new_root = root / "moved" / "task"
            prefix_collision = f"{old_root}_backup/results"

            command, environment, cwd = rebase_recovered_task_context(
                ["python", "evaluate.py", f"--external={prefix_collision}"],
                {
                    "PRAXIST_TASK_PROJECT_PATH": str(old_root),
                    "EXTERNAL_DATA": prefix_collision,
                },
                str(old_root / "work"),
                current_environment={"PRAXIST_TASK_PROJECT_PATH": str(new_root)},
            )

        self.assertEqual(command[-1], f"--external={prefix_collision}")
        self.assertEqual(environment["EXTERNAL_DATA"], prefix_collision)
        self.assertEqual(cwd, str(new_root / "work"))

    def test_recovered_task_rebases_runtime_locations_when_task_root_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            evaluator = task_root / "evaluations" / "run.py"
            old_python = root / "venvs" / "old" / "bin" / "python"
            new_python = root / "venvs" / "new" / "bin" / "python"
            old_data = root / "data-old"
            new_data = root / "data-new"
            evaluator.parent.mkdir(parents=True)
            evaluator.write_text("print('ok')\n", encoding="utf-8")
            service = ExperimentSchedulerService(
                run_dir=root / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            persisted_environment = {
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                "PRAXIST_TASK_PYTHON": str(old_python),
                "PRAXIST_DATASETS_DIR": str(old_data),
            }
            current_environment = {
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                "PRAXIST_TASK_PYTHON": str(new_python),
                "PRAXIST_DATASETS_DIR": str(new_data),
            }
            event = {
                "command": [str(old_python), str(evaluator)],
                "environment_values": persisted_environment,
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "same-root-new-runtime",
                "profile": "cpu",
                "cwd": str(task_root),
            }

            self.assertTrue(
                task_runtime_context_changed(persisted_environment, current_environment)
            )
            with patch.dict(os.environ, current_environment, clear=True):
                recovered = service._job_from_event("job", event, state="queued")

        self.assertEqual(recovered.command[0], str(new_python))
        self.assertEqual(recovered.environment["PRAXIST_TASK_PYTHON"], str(new_python))
        self.assertEqual(recovered.environment["PRAXIST_DATASETS_DIR"], str(new_data))

    def test_recovered_missing_cwd_is_rejected_before_relaunch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_root = root / "task"
            task_root.mkdir()
            service = ExperimentSchedulerService(
                run_dir=root / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            event = {
                "command": [sys.executable, "-c", "pass"],
                "environment_values": {"PRAXIST_TASK_PROJECT_PATH": str(task_root)},
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "missing-recovery-cwd",
                "profile": "cpu",
                "cwd": str(root / "removed-work"),
            }
            recovered = service._job_from_event("job", event, state="queued")

            reason = service._recovery_rejection_reason(recovered)

        self.assertEqual(reason, "recovery_cwd_unavailable")

    def test_recovered_shell_task_runtime_normalization_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_python = root / "task" / ".venv" / "bin" / "python"
            service = ExperimentSchedulerService(
                run_dir=root / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            submitted = service.submit(
                {
                    "command": ["bash", "-lc", "python evaluations/run.py"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "shell-recovery",
                    "environment": {
                        "PRAXIST_TASK_PYTHON": str(task_python),
                        "PATH": "/usr/bin",
                    },
                }
            )
            event = {
                **service._event_identity(submitted, include_request=True),
                "event": "submitted",
            }

            recovered = service._job_from_event(submitted.job_id, event, state="queued")

        self.assertEqual(recovered.command[2].count("PRAXIST_TASK_PYTHON="), 1)
        self.assertEqual(recovered.command[2].count("unset PYTHONPATH"), 1)
        self.assertEqual(recovered.command[2].count("unset PYTHONHOME"), 1)

    def test_task_python_binding_normalizes_only_supported_launch_forms(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            task_python = str(Path(td) / "task-python")
            environment = {"PRAXIST_TASK_PYTHON": task_python}
            cases = (
                (["python", "task.py"], [task_python, "task.py"]),
                (["python3", "task.py"], [task_python, "task.py"]),
                (["/usr/bin/env", "python", "task.py"], ["/usr/bin/env", task_python, "task.py"]),
                (
                    ["/usr/bin/env", "--", "python", "task.py"],
                    ["/usr/bin/env", "--", task_python, "task.py"],
                ),
                (
                    ["/usr/bin/env", "-u", "HOME", "python3", "task.py"],
                    ["/usr/bin/env", "-u", "HOME", task_python, "task.py"],
                ),
                (
                    ["/usr/bin/env", "-i", "MODE=test", "python3", "task.py"],
                    ["/usr/bin/env", "-i", "MODE=test", task_python, "task.py"],
                ),
                (["/usr/bin/env", "--"], ["/usr/bin/env", "--"]),
                ([sys.executable, "task.py"], [sys.executable, "task.py"]),
                (["python3.11", "task.py"], [task_python, "task.py"]),
                (
                    ["/usr/bin/env", "python3.11", "task.py"],
                    ["/usr/bin/env", task_python, "task.py"],
                ),
                (["/usr/bin/python3", "task.py"], ["/usr/bin/python3", "task.py"]),
            )
            for index, (command, expected) in enumerate(cases):
                with self.subTest(command=command):
                    job = service.submit(
                        {
                            "command": command,
                            "peer_id": "gen0_peer0",
                            "generation_id": 0,
                            "experiment_id": f"python-binding-{index}",
                            "environment": environment,
                        }
                    )
                    self.assertEqual(job.command, expected)

            shell = service.submit(
                {
                    "command": ["/bin/bash", "-lc", "python3 -c 'pass'"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "python-binding-shell",
                    "environment": environment,
                }
            )
            self.assertEqual(shell.command[:2], ["/bin/bash", "-lc"])
            self.assertIn("unset PYTHONPATH", shell.command[2])
            self.assertIn("unset PYTHONHOME", shell.command[2])
            self.assertIn(f"PRAXIST_TASK_PYTHON={task_python}", shell.command[2])
            self.assertIn('python3() { command "$PRAXIST_TASK_PYTHON" "$@"; }', shell.command[2])
            self.assertTrue(shell.command[2].endswith(f"{task_python} -c 'pass'"))

            script_invocation = service.submit(
                {
                    "command": ["/bin/bash", "task.sh", "-c", "python3 task.py"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "python-binding-shell-script",
                    "environment": environment,
                }
            )
            self.assertEqual(
                script_invocation.command,
                ["/bin/bash", "task.sh", "-c", "python3 task.py"],
            )

            shell_with_option = service.submit(
                {
                    "command": ["/bin/bash", "-O", "extglob", "-lc", "python task.py"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "python-binding-shell-option",
                    "environment": environment,
                }
            )
            self.assertIn("PRAXIST_TASK_PYTHON", shell_with_option.command[-1])

            incomplete_shell_option = service.submit(
                {
                    "command": ["/bin/bash", "-O"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "python-binding-incomplete-shell-option",
                    "environment": environment,
                }
            )
            self.assertEqual(incomplete_shell_option.command, ["/bin/bash", "-O"])

    def test_task_child_drops_runner_python_paths_unless_task_declares_them(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            task_root = Path(td).resolve() / "task"
            task_root.mkdir()
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            base_environment = {
                "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                "PRAXIST_TASK_PYTHON": str(task_root / ".venv" / "bin" / "python"),
                "PYTHONPATH": "/runner/site-packages",
                "PYTHONHOME": "/runner/python",
            }
            isolated = service.submit(
                {
                    "command": ["python", "evaluations/run.py"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "isolated-python-env",
                    "environment": base_environment,
                }
            )
            self.assertNotIn("PYTHONPATH", isolated.environment)
            self.assertNotIn("PYTHONHOME", isolated.environment)
            self.assertEqual(
                isolated.environment["PATH"].split(os.pathsep)[0],
                str(task_root / ".venv" / "bin"),
            )
            self.assertEqual(isolated.cwd, str(task_root.resolve()))

            task_owned = service.submit(
                {
                    "command": ["python", "evaluations/run.py"],
                    "peer_id": "gen0_peer1",
                    "generation_id": 0,
                    "experiment_id": "task-owned-python-env",
                    "environment": {
                        **base_environment,
                        "PRAXIST_TASK_RUNTIME_ENV_KEYS": "PYTHONHOME,PYTHONPATH",
                    },
                }
            )
            self.assertEqual(task_owned.environment["PYTHONPATH"], "/runner/site-packages")
            self.assertEqual(task_owned.environment["PYTHONHOME"], "/runner/python")

    def test_launch_barrier_ignores_task_pythonpath_but_task_process_receives_it(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            task_root = root / "task"
            shadow_dir = task_root / "pythonpath"
            output = task_root / "observed-pythonpath.txt"
            shadow_dir.mkdir(parents=True)
            (shadow_dir / "json.py").write_text(
                "raise RuntimeError('task PYTHONPATH reached scheduler barrier')\n",
                encoding="utf-8",
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            service.start()
            try:
                with patch.dict(
                    os.environ,
                    {
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_TASK_PYTHON": sys.executable,
                        "PRAXIST_TASK_RUNTIME_ENV_KEYS": "PYTHONPATH",
                        "PYTHONPATH": str(shadow_dir),
                        "TASK_OUTPUT": str(output),
                    },
                ):
                    code = submit_and_wait(
                        [
                            sys.executable,
                            "-c",
                            (
                                "import os,pathlib;"
                                "pathlib.Path(os.environ['TASK_OUTPUT']).write_text("
                                "os.environ['PYTHONPATH'])"
                            ),
                        ],
                        peer_id="gen0_peer0",
                        experiment_id="task-owned-pythonpath",
                        run_dir=run_dir,
                    )
            finally:
                service.stop()

            self.assertEqual(code, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), str(shadow_dir))

    def test_scheduler_rejects_missing_explicit_experiment_cwd_at_admission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            with self.assertRaisesRegex(ExperimentRejected, "cwd is not an existing directory"):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "missing-cwd",
                        "cwd": str(Path(td) / "missing"),
                    }
                )

    def test_failure_classification_keeps_resource_and_permission_distinct(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler import (
            _classify_failure,
        )

        self.assertEqual(_classify_failure(1, "resource exhausted"), "resource")
        self.assertEqual(_classify_failure(1, "permission denied"), "permission")

    def test_task_python_binding_reaches_bash_login_shell_process_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            marker = root / "task-python-used"
            wrapper_dir = root / "task environment"
            wrapper_dir.mkdir()
            task_python = wrapper_dir / "python"
            task_python.write_text(
                f"#!{sys.executable}\n"
                "import os, pathlib, sys\n"
                "pathlib.Path(os.environ['WRAPPER_MARKER']).write_text('used')\n"
                "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n",
                encoding="utf-8",
            )
            task_python.chmod(0o755)
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = service.submit(
                {
                    "command": ["/bin/bash", "-lc", "python3 -c 'pass'"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "bash-login-task-python",
                    "environment": {
                        **os.environ,
                        "PRAXIST_TASK_PYTHON": str(task_python),
                        "WRAPPER_MARKER": str(marker),
                    },
                }
            )
            service.start()
            try:
                result = service.wait(job.job_id, 5)["job"]
            finally:
                service.stop()
            self.assertEqual(result["state"], "completed", result)
            self.assertEqual(marker.read_text(encoding="utf-8"), "used")

    def test_failed_job_status_retains_redacted_bounded_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            secret_sample = Path(td) / "failure-sample.txt"
            secret_sample.write_text("API_KEY=supersecretvalue", encoding="utf-8")
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = service.submit(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "import pathlib,sys;print('x'*6000);"
                            "print(pathlib.Path(sys.argv[1]).read_text());"
                            "print('ModuleNotFoundError: missing_widget');sys.exit(2)"
                        ),
                        str(secret_sample),
                    ],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "explained-failure",
                }
            )
            service.start()
            try:
                result = service.wait(job.job_id, 5)["job"]
                status = service.status()
            finally:
                service.stop()
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["failure_category"], "environment")
            self.assertEqual(result["failure_signature"], "ModuleNotFoundError: missing_widget")
            self.assertEqual(result["error"], result["failure_signature"])
            self.assertLessEqual(len(result["failure_log_tail"]), 4096)
            self.assertIn("<redacted:named_secret>", result["failure_log_tail"])
            self.assertNotIn("supersecretvalue", result["failure_log_tail"])
            self.assertEqual(status["failure_counts_by_category"], {"environment": 1})
            self.assertEqual(status["jobs_total"], 1)
            self.assertEqual(status["jobs_retained"], 1)
            self.assertEqual(status["jobs_omitted"], 0)

            events_path = run_dir / "resource_scheduler" / "events.jsonl"
            events_text = events_path.read_text(encoding="utf-8")
            self.assertNotIn("supersecretvalue", events_text)
            terminal = [
                json.loads(line)
                for line in events_text.splitlines()
                if json.loads(line).get("event") == "completed"
            ][-1]
            self.assertEqual(terminal["failure_category"], "environment")
            self.assertEqual(terminal["failure_signature"], result["failure_signature"])

            recovered = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            recovered._recover_terminal_events()
            recovered_job = recovered.status()["jobs"][0]
            self.assertEqual(recovered_job["failure_category"], "environment")
            self.assertEqual(recovered_job["failure_signature"], result["failure_signature"])

    def test_status_reports_total_and_retained_job_counts_separately(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            for index in range(3):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": f"gen0_peer{index}",
                        "generation_id": 0,
                        "experiment_id": f"retention-{index}",
                    }
                )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler._JOB_DETAIL_LIMIT",
                2,
            ):
                status = service.status()
            self.assertEqual(status["jobs_total"], 3)
            self.assertEqual(status["jobs_retained"], 2)
            self.assertEqual(status["jobs_omitted"], 1)
            self.assertEqual(len(status["jobs"]), 2)

    def test_unbound_command_secret_is_rejected_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            for index, option in enumerate(
                (
                    "--token",
                    "--access-token",
                    "--client-secret",
                    "--private-key",
                    "--connection-string",
                )
            ):
                service = ExperimentSchedulerService(
                    run_dir=Path(td) / f"run-{index}",
                    settings=_settings(maximum=1),
                    allocator=_GPUAllocator(""),
                )
                with (
                    self.subTest(option=option),
                    self.assertRaisesRegex(ExperimentRejected, "credential-bearing"),
                ):
                    service.submit(
                        {
                            "command": [sys.executable, "task.py", option, "opaque-value"],
                            "peer_id": "gen0_peer0",
                            "generation_id": 0,
                            "experiment_id": f"unsafe-command-secret-{index}",
                            "profile": "cpu",
                            "environment": {},
                        }
                    )
                self.assertFalse((service.state_dir / "events.jsonl").exists())

    def test_second_scheduler_cannot_take_over_live_run_owner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("", limit=1),
            )
            second = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            first.start()
            try:
                endpoint = first.endpoint
                self.assertTrue(endpoint.exists())
                with self.assertRaisesRegex(RuntimeError, "already owns run"):
                    second.start()
                self.assertTrue(endpoint.exists())
                self.assertEqual(first.handle_request({"action": "ping"})["run_id"], run_dir.name)
            finally:
                first.stop()

    def test_durable_submitted_event_recovers_acknowledged_queue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            marker = Path(td) / "executed"
            interrupted = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            submitted = interrupted.submit(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('once')",
                        str(marker),
                    ],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "durable-queue-entry",
                    "profile": "cpu",
                }
            )
            events = (interrupted.state_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn('"event": "submitted"', events)

            resumed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            resumed.start()
            try:
                duplicate = resumed.submit(
                    {
                        "command": [sys.executable, "-c", "raise SystemExit(99)"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "durable-queue-entry",
                        "profile": "cpu",
                    }
                )
                self.assertEqual(duplicate.job_id, submitted.job_id)
                result = resumed.wait(submitted.job_id, 5)["job"]
            finally:
                resumed.stop()
            self.assertEqual(result["state"], "completed")
            self.assertEqual(marker.read_text(), "once")

    def test_cancelled_queue_entry_is_not_resurrected_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            marker = Path(td) / "must-not-run"
            interrupted = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            submitted = interrupted.submit(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('bad')",
                        str(marker),
                    ],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "cancel-before-launch",
                    "profile": "cpu",
                }
            )
            self.assertTrue(interrupted.cancel_queued(submitted.job_id)["cancelled"])

            resumed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            resumed.start()
            try:
                self.assertEqual(resumed.status()["queued"], 0)
                self.assertFalse(marker.exists())
                replacement = resumed.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "cancel-before-launch",
                        "profile": "cpu",
                    }
                )
                self.assertNotEqual(replacement.job_id, submitted.job_id)
                self.assertEqual(resumed.wait(replacement.job_id, 5)["job"]["state"], "completed")
            finally:
                resumed.stop()

    def test_assessment_and_freeze_fences_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            interrupted = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            interrupted.open_generation(3, deadline=time.time() + 600, cohort_size=2)
            interrupted.begin_assessment(3, "evidence-ready")
            interrupted.freeze_generation(4, "closed")

            resumed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            resumed.start()
            try:
                self.assertIn(3, resumed.status()["assessment_generations"])
                self.assertIn(4, resumed.status()["frozen_generations"])
                with self.assertRaisesRegex(ExperimentRejected, "only mature work"):
                    resumed.submit(
                        {
                            "command": [sys.executable, "-c", "pass"],
                            "peer_id": "gen3_peer0",
                            "generation_id": 3,
                            "experiment_id": "ordinary-after-restart",
                            "profile": "cpu",
                        }
                    )
                with self.assertRaisesRegex(ExperimentRejected, "is closing"):
                    resumed.submit(
                        {
                            "command": [sys.executable, "-c", "pass"],
                            "peer_id": "gen4_peer0",
                            "generation_id": 4,
                            "experiment_id": "frozen-after-restart",
                            "profile": "cpu",
                        }
                    )
            finally:
                resumed.stop()

    def test_launch_intent_finds_delayed_ready_process_without_requeue(self) -> None:
        if not Path("/proc").is_dir():
            self.skipTest("/proc filesystem required for process cmdline inspection")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            interrupted = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = interrupted.submit(
                {
                    "command": [sys.executable, "-c", "raise SystemExit(99)"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "delayed-ready",
                    "profile": "cpu",
                }
            )
            job.attempts = 1
            attempt_dir = interrupted.state_dir / "attempts" / f"{job.job_id}-a1"
            attempt_dir.mkdir(parents=True)
            interrupted._append_event(
                {
                    "event": "launch_intent",
                    **interrupted._event_identity(job, include_request=True),
                    "allocation_id": "delayed-allocation",
                    "gpu_uuids": [],
                    "attempt_id": f"{job.job_id}-a1",
                    "attempt_dir": str(attempt_dir),
                },
                required=True,
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time;time.sleep(30)",
                    str(attempt_dir / "READY.json"),
                    f"{job.job_id}-a1",
                ],
                start_new_session=True,
            )
            resumed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            try:
                resumed.start()
                self.assertIn(job.job_id, resumed._active)
                self.assertNotIn(job.job_id, resumed._queue)
                self.assertTrue((attempt_dir / "GO.json").exists())
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
                self.assertEqual(resumed.wait(job.job_id, 5)["job"]["state"], "drained_unknown")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                resumed.stop()

    def test_new_attempt_does_not_inherit_previous_attempt_pid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "attempt-reset",
                    "profile": "cpu",
                }
            )
            attempt_dir = service.state_dir / "attempts" / f"{job.job_id}-a2"
            attempt_dir.mkdir(parents=True)
            service._append_event(
                {
                    "event": "process_started",
                    **service._event_identity(job, include_request=True),
                    "attempt": 1,
                    "pid": 111,
                    "pgid": 111,
                    "pid_start_time": 1,
                },
                required=True,
            )
            job.attempts = 2
            service._append_event(
                {
                    "event": "launch_intent",
                    **service._event_identity(job, include_request=True),
                    "allocation_id": "attempt-2-allocation",
                    "attempt_id": f"{job.job_id}-a2",
                    "attempt_dir": str(attempt_dir),
                    "gpu_uuids": [],
                },
                required=True,
            )
            with (
                patch.object(service, "_read_ready_process", return_value={}) as read_ready,
                patch.object(service, "_find_attempt_process", return_value={}) as find_process,
            ):
                service._recover_terminal_events()
            read_ready.assert_called_once()
            find_process.assert_called_once()
            self.assertIn(job.job_id, service._queue)

    def test_replay_restores_submitted_retry_terminal_and_generation_controls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
            )
            common = {
                "generation_id": 2,
                "profile": "cpu",
                "work_class": "ordinary",
                "command": [sys.executable, "-c", "pass"],
                "environment_values": {},
                "attempt": 0,
                "submitted_at": time.time(),
            }
            events = [
                {
                    "event": "generation_open",
                    "generation_id": 2,
                    "deadline": time.time() + 600,
                    "cohort_size": 2,
                },
                {
                    "event": "submitted",
                    **common,
                    "job_id": "submitted-job",
                    "peer_id": "gen2_peer0",
                    "experiment_id": "submitted-science",
                },
                {
                    "event": "retry_queued",
                    **common,
                    "job_id": "retry-job",
                    "peer_id": "gen2_peer1",
                    "experiment_id": "retry-science",
                    "attempt": 1,
                },
                {
                    "event": "completed",
                    **common,
                    "job_id": "completed-job",
                    "peer_id": "gen2_peer2",
                    "experiment_id": "completed-science",
                    "state": "completed",
                    "exit_code": 0,
                },
            ]
            (service.state_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            service._recover_terminal_events()
            self.assertEqual(set(service._queue), {"submitted-job", "retry-job"})
            self.assertEqual(service._jobs["completed-job"].state, "completed")
            self.assertEqual(service._generation_cohort_sizes[2], 2)

            service.begin_assessment(2)
            service.freeze_generation(2)
            service.freeze_all()
            replayed = ExperimentSchedulerService(
                run_dir=service.run_dir,
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
            )
            replayed._recover_terminal_events()
            self.assertIn(2, replayed._assessment_generations)
            self.assertIn(2, replayed._frozen_generations)
            self.assertTrue(replayed._admission_closed)

    def test_resume_clears_rerun_signal_before_scheduler_recovers_submitted_job(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.generation_resume import (
            prepare_resume_for_sidecars,
        )

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
            )
            submitted = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "resume-after-close",
                    "profile": "cpu",
                    "eta_seconds": 10,
                }
            )
            gen_dir = run_dir / "gen_0"
            gen_dir.mkdir()
            (gen_dir / "CLOSING_SIGNAL").write_text("stale\n", encoding="utf-8")
            service._append_event(
                {"event": "generation_frozen", "generation_id": 0, "reason": "closing"},
                required=True,
            )
            service._append_event(
                {"event": "admission_closed", "reason": "scheduler_stopped"},
                required=True,
            )

            plan = prepare_resume_for_sidecars(
                run_dir,
                max_generations=2,
                pi_enabled=True,
                policy="completed_generation",
            )
            replayed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
                recovery_rerun_generation=plan.start_generation,
            )
            replayed._recover_terminal_events()

            self.assertFalse(plan.has_pending_boundary)
            self.assertFalse((gen_dir / "CLOSING_SIGNAL").exists())
            self.assertEqual(replayed._jobs[submitted.job_id].state, "queued")
            self.assertIn(submitted.job_id, replayed._queue)
            self.assertTrue(replayed._admission_closed)
            self.assertIn(0, replayed._frozen_generations)
            replayed.open_generation(0, deadline=time.time() + 60, cohort_size=1)
            self.assertFalse(replayed._admission_closed)
            self.assertNotIn(0, replayed._frozen_generations)

    def test_resume_rerun_releases_semantic_identity_rejected_by_prior_close(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
            )
            first = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "closed-before-launch",
                    "profile": "cpu",
                }
            )
            service.freeze_generation(0, "interrupted")
            self.assertEqual(first.state, "rejected")

            replayed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
                recovery_rerun_generation=0,
            )
            replayed._recover_terminal_events()
            replayed.open_generation(0, deadline=time.time() + 60, cohort_size=1)
            replacement = replayed.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "closed-before-launch",
                    "profile": "cpu",
                }
            )

            self.assertNotEqual(replacement.job_id, first.job_id)
            self.assertEqual(replacement.state, "queued")

    def test_resume_rerun_accepts_prior_close_control_error_spellings(self) -> None:
        control_errors = (
            "admission_closed",
            "admission_closed:operator_stop",
            "generation_closing",
            "generation_closing:boundary",
            "generation_assessment",
            "generation_assessment:quorum_reached",
        )
        for error in control_errors:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as td:
                run_dir = Path(td) / "run"
                service = ExperimentSchedulerService(
                    run_dir=run_dir,
                    settings=_settings(maximum=2),
                    allocator=_GPUAllocator("", limit=2),
                )
                first = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "closed-control-spelling",
                        "profile": "cpu",
                    }
                )
                service._finish_without_launch(first, error)

                replayed = ExperimentSchedulerService(
                    run_dir=run_dir,
                    settings=_settings(maximum=2),
                    allocator=_GPUAllocator("", limit=2),
                    recovery_rerun_generation=0,
                )
                replayed._recover_terminal_events()
                replayed.open_generation(0, deadline=time.time() + 60, cohort_size=1)
                replacement = replayed.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "closed-control-spelling",
                        "profile": "cpu",
                    }
                )

                self.assertNotEqual(replacement.job_id, first.job_id)
                self.assertEqual(replacement.state, "queued")

    def test_resume_rerun_preserves_unrelated_terminal_semantic_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
            )
            first = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "unrelated-rejection",
                    "profile": "cpu",
                }
            )
            service._finish_without_launch(first, "task_policy_rejected")

            replayed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
                recovery_rerun_generation=0,
            )
            replayed._recover_terminal_events()
            replayed.open_generation(0, deadline=time.time() + 60, cohort_size=1)
            duplicate = replayed.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "unrelated-rejection",
                    "profile": "cpu",
                }
            )

            self.assertEqual(duplicate.job_id, first.job_id)
            self.assertEqual(duplicate.state, "rejected")

    def test_resume_rerun_never_releases_completed_or_failed_identity_by_error_text(self) -> None:
        for state, exit_code in (("completed", 0), ("failed", 1)):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as td:
                run_dir = Path(td) / "run"
                service = ExperimentSchedulerService(
                    run_dir=run_dir,
                    settings=_settings(maximum=2),
                    allocator=_GPUAllocator("", limit=2),
                )
                first = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "terminal-with-close-text",
                        "profile": "cpu",
                    }
                )
                service._append_event(
                    {
                        "event": "completed",
                        **service._event_identity(first),
                        "state": state,
                        "exit_code": exit_code,
                        "error": "generation_closing",
                    },
                    required=True,
                )

                replayed = ExperimentSchedulerService(
                    run_dir=run_dir,
                    settings=_settings(maximum=2),
                    allocator=_GPUAllocator("", limit=2),
                    recovery_rerun_generation=0,
                )
                replayed._recover_terminal_events()
                replayed.open_generation(0, deadline=time.time() + 60, cohort_size=1)
                duplicate = replayed.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "terminal-with-close-text",
                        "profile": "cpu",
                    }
                )

                self.assertEqual(duplicate.job_id, first.job_id)
                self.assertEqual(duplicate.state, state)

    def test_recovery_and_timing_helpers_cover_conservative_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("GPU-a", limit=2),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen1_peer0",
                    "generation_id": 1,
                    "experiment_id": "helper-boundaries",
                    "profile": "cpu",
                    "eta_seconds": 10,
                }
            )
            self.assertEqual(service._recovery_rejection_reason(job), "")
            service._admission_closed = True
            self.assertEqual(service._recovery_rejection_reason(job), "recovery_admission_closed")
            service._admission_closed = False
            service._frozen_generations.add(1)
            self.assertEqual(service._recovery_rejection_reason(job), "recovery_generation_frozen")
            service._frozen_generations.clear()
            service._assessment_generations.add(1)
            self.assertEqual(
                service._recovery_rejection_reason(job), "recovery_generation_assessment"
            )
            job.work_class = "mature"
            self.assertEqual(service._recovery_rejection_reason(job), "")

            with patch.object(Path, "read_bytes", side_effect=OSError("gone")):
                self.assertEqual(service._read_process_environment(1), {})
            with patch.object(Path, "read_bytes", return_value=b"X" * (1024 * 1024 + 1)):
                self.assertEqual(service._read_process_environment(1), {})
            with patch.object(
                Path,
                "read_bytes",
                return_value=b"GOOD=value\0missing-separator\0BAD=\xff\0",
            ):
                self.assertEqual(service._read_process_environment(1), {"GOOD": "value"})

            service.allocator.snapshot = HostSnapshot(
                8,
                5,
                5,
                0,
                gpus=(GPUDevice(0, "GPU-a", 40960, 0, 0),),
                gpu_processes=(GPUProcess(22, "GPU-a", 1024, 25),),
            )
            self.assertEqual(
                service._recovered_gpu_uuids(
                    {"PRAXIST_ASSIGNED_GPU_UUIDS": "GPU-a"}, pid=1, pgid=1
                ),
                ("GPU-a",),
            )
            self.assertEqual(
                service._recovered_gpu_uuids({"CUDA_VISIBLE_DEVICES": "0"}, pid=1, pgid=1),
                ("GPU-a",),
            )
            with patch.object(os, "getpgid", return_value=7):
                self.assertEqual(
                    service._recovered_gpu_uuids({}, pid=1, pgid=7),
                    ("GPU-a",),
                )
            with patch.object(os, "getpgid", side_effect=OSError("gone")):
                self.assertEqual(
                    service._recovered_gpu_uuids({}, pid=22, pgid=7),
                    ("GPU-a",),
                )

            self.assertEqual(
                service._recovered_profile({"PRAXIST_RESOURCE_PROFILE": "gpu"}, ("GPU-a",)).name,
                "gpu",
            )
            self.assertEqual(service._recovered_profile({}, ()).name, "cpu")
            service.settings.profiles = {"gpu": service.settings.profiles["gpu"]}
            service.settings.default_profile = "gpu"
            self.assertEqual(service._recovered_profile({}, ()).name, "unclassified_recovery")

            self.assertFalse(
                service._event_process_matches({"pid": 0, "pgid": 0}, attempt_dir=None)
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler._pid_start_time",
                return_value=5,
            ):
                self.assertTrue(
                    service._event_process_matches(
                        {"pid": 1, "pgid": 1, "pid_start_time": 5}, attempt_dir=None
                    )
                )
                self.assertFalse(
                    service._event_process_matches(
                        {"pid": 1, "pgid": 1, "pid_start_time": "bad"}, attempt_dir=None
                    )
                )
            with (
                patch.object(Path, "is_dir", return_value=False),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler._pid_start_time",
                    return_value="ps:stable-start",
                ),
            ):
                self.assertTrue(
                    service._event_process_matches(
                        {"pid": 1, "pgid": 1, "pid_start_time": "ps:stable-start"},
                        attempt_dir=Path(td),
                    )
                )
                self.assertFalse(
                    service._event_process_matches({"pid": 1, "pgid": 1}, attempt_dir=Path(td))
                )
            self.assertFalse(
                service._event_process_matches({"pid": 1, "pgid": 1}, attempt_dir=None)
            )
            with patch.object(Path, "read_bytes", side_effect=OSError("gone")):
                self.assertFalse(
                    service._event_process_matches({"pid": 1, "pgid": 1}, attempt_dir=Path(td))
                )

            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                service._restore_durable_command(["__PRAXIST_ENV_REF_TOKEN__"], {})
            service._queue.remove(job.job_id)
            job.supply_claim_id = "claim"
            service._finish_without_launch(job, "done")
            self.assertEqual(job.state, "rejected")

            job.state = "running"
            job.work_class = "ordinary"
            service.settings.deadline_admission = True
            service._generation_deadlines[1] = time.time() - 1
            self.assertFalse(service._fits_generation(job))
            job.eta_seconds = 0
            service._observe_runtime(job, started_at=time.time())
            self.assertEqual(service._completion_probability(job, deadline=time.time() + 10), 0)
            job.eta_seconds = 10
            service._observe_runtime(job, started_at=time.time() - 5)
            service._observe_runtime(job, started_at=time.time() - 20)
            self.assertGreater(service._completion_probability(job, deadline=time.time() + 10), 0)

    def test_scheduler_defensive_lifecycle_and_supply_branches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            allocator = _GPUAllocator("", limit=2)
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=2),
                allocator=allocator,
            )
            service._acquire_owner_lock()
            with self.assertRaisesRegex(RuntimeError, "already started"):
                service._acquire_owner_lock()
            service._release_owner_lock()
            service._release_owner_lock()

            queued = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "stop-queued",
                }
            )
            service._acquire_owner_lock()
            service.stop()
            self.assertEqual(queued.state, "rejected")

            failing_allocator = MagicMock()
            failing = ExperimentSchedulerService(
                run_dir=Path(td) / "failing",
                settings=_settings(maximum=1),
                allocator=failing_allocator,
            )
            with (
                patch.object(failing, "_recover_terminal_events", side_effect=RuntimeError("boom")),
                self.assertRaisesRegex(RuntimeError, "boom"),
            ):
                failing.start()
            failing_allocator.close.assert_called_once()
            self.assertIsNone(failing._owner_lock_fd)

            service = ExperimentSchedulerService(
                run_dir=Path(td) / "supply",
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
            )
            service.settings.supply_idle_samples = 2
            service.open_generation(1, deadline=time.time() + 600, cohort_size=2)
            service.configure_generation_maturity(
                1,
                cohort_size=2,
                mature_target=1,
                count_callback=lambda: (_ for _ in ()).throw(RuntimeError("unknown")),
            )
            service._refresh_mature_counts(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            self.assertIsNone(service._mature_completed[1])
            self.assertEqual(service.generation_advice("bad", 1), {})
            self.assertEqual(service.generation_advice("gen1_peer0", 99), {})
            service._frozen_generations.add(1)
            service.begin_assessment(1)
            service._frozen_generations.clear()

            service.register_idle_supply("gen1_peer0", 1)
            first = HostSnapshot(8, 5, 5, 0, observed_at=2.0)
            service._reconcile_supply(first)
            self.assertEqual(service._supply_leases, {})
            service._reconcile_supply(first)
            second = HostSnapshot(8, 5, 5, 0, observed_at=3.0)
            service._reconcile_supply(second)
            self.assertEqual(len(service._supply_leases), 1)
            lease_id = next(iter(service._supply_leases))
            self.assertEqual(service.register_idle_supply("gen1_peer0", 1)["lease_id"], lease_id)
            self.assertEqual(service.get_supply_lease("bad", 1, lease_id), {})
            self.assertEqual(service.get_supply_lease("gen1_peer0", 1, "missing"), {})
            service.release_supply_lease(lease_id, "wrong-peer")
            self.assertIn(lease_id, service._supply_leases)
            service.unregister_idle_supply("gen1_peer0", 1)
            self.assertNotIn(lease_id, service._supply_leases)

            service.register_idle_supply("gen1_peer1", 1)
            service.allocator.has_supply_headroom = lambda *_args, **_kwargs: False
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=4.0))
            self.assertEqual(service._supply_idle_samples, 0)
            service.freeze_generation(1)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=5.0))
            self.assertNotIn("gen1_peer1", service._idle_supply_waiters)

    def test_replay_handles_invalid_credentials_empty_requests_and_closed_generations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("", limit=2),
            )
            base = {
                "generation_id": 0,
                "profile": "cpu",
                "work_class": "ordinary",
                "environment_values": {},
                "attempt": 0,
                "submitted_at": time.time(),
            }
            events = [
                {
                    "event": "submitted",
                    **base,
                    "job_id": "valid",
                    "peer_id": "gen0_peer0",
                    "experiment_id": "valid",
                    "command": [sys.executable, "-c", "pass"],
                },
                {
                    "event": "submitted",
                    **base,
                    "job_id": "empty",
                    "peer_id": "gen0_peer1",
                    "experiment_id": "empty",
                    "command": [],
                },
                {
                    "event": "retry_queued",
                    **base,
                    "job_id": "changed-secret",
                    "peer_id": "gen0_peer2",
                    "experiment_id": "changed-secret",
                    "command": [sys.executable, "-c", "pass"],
                    "environment_sensitive_hashes": {"SERVICE_TOKEN": "not-current"},
                },
                {
                    "event": "retry_queued",
                    **base,
                    "job_id": "valid-retry",
                    "peer_id": "gen0_peer3",
                    "experiment_id": "valid-retry",
                    "command": [sys.executable, "-c", "pass"],
                    "attempt": 1,
                },
            ]
            (service.state_dir / "events.jsonl").write_text(
                "malformed\n" + "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"SERVICE_TOKEN": "current"}, clear=True):
                service._recover_terminal_events()
            self.assertEqual(set(service._queue), {"valid", "valid-retry"})
            self.assertNotIn("empty", service._jobs)
            self.assertEqual(service._jobs["changed-secret"].state, "failed")

            frozen = ExperimentSchedulerService(
                run_dir=Path(td) / "frozen",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("", limit=1),
            )
            frozen._frozen_generations.add(4)
            frozen_event = {
                "event": "submitted",
                "job_id": "frozen-job",
                "generation_id": 4,
                "peer_id": "gen4_peer0",
                "experiment_id": "frozen-job",
                "profile": "cpu",
                "work_class": "ordinary",
                "command": [sys.executable, "-c", "pass"],
                "environment_values": {},
            }
            (frozen.state_dir / "events.jsonl").write_text(
                json.dumps(frozen_event) + "\n", encoding="utf-8"
            )
            frozen._recover_terminal_events()
            self.assertEqual(frozen._jobs["frozen-job"].state, "rejected")

    def test_attempt_discovery_and_identity_validation_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            attempt_dir = Path(td) / "attempt"
            attempt_dir.mkdir()
            ready_path = attempt_dir / "READY.json"
            payload = json.dumps(
                {"pid": 123, "pgid": 123, "attempt_id": "attempt-a1", "pid_start_time": 5}
            )
            command = b"python\0" + str(ready_path).encode() + b"\0attempt-a1\0"
            with (
                patch.object(Path, "read_text", return_value=payload),
                patch.object(Path, "read_bytes", return_value=command),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler._pid_start_time",
                    return_value=5,
                ),
            ):
                self.assertEqual(
                    ExperimentSchedulerService._read_ready_process(attempt_dir, "attempt-a1"),
                    {"pid": 123, "pgid": 123},
                )

            portable_payload = json.dumps(
                {
                    "pid": 123,
                    "pgid": 123,
                    "attempt_id": "attempt-a1",
                    "pid_start_time": "ps:stable-start",
                }
            )
            with (
                patch.object(Path, "read_text", return_value=portable_payload),
                patch.object(Path, "is_dir", return_value=False),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler._pid_start_time",
                    return_value="ps:stable-start",
                ),
            ):
                self.assertEqual(
                    ExperimentSchedulerService._read_ready_process(attempt_dir, "attempt-a1"),
                    {"pid": 123, "pgid": 123},
                )

            for invalid_portable in (
                {**json.loads(portable_payload), "attempt_id": "other"},
                {**json.loads(portable_payload), "pid_start_time": None},
            ):
                with (
                    self.subTest(portable_payload=invalid_portable),
                    patch.object(Path, "read_text", return_value=json.dumps(invalid_portable)),
                    patch.object(Path, "is_dir", return_value=False),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler._pid_start_time",
                        return_value="ps:stable-start",
                    ),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler.time.monotonic",
                        side_effect=[0.0, 0.0, 3.0],
                    ),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler.time.sleep"
                    ),
                ):
                    self.assertEqual(
                        ExperimentSchedulerService._read_ready_process(attempt_dir, "attempt-a1"),
                        {},
                    )

            invalid_payloads = [
                ({**json.loads(payload), "attempt_id": "other"}, command),
                (json.loads(payload), b"python\0" + str(ready_path).encode() + b"\0"),
                ({**json.loads(payload), "pid_start_time": 6}, command),
                (json.loads(payload), b"python\0wrong\0"),
            ]
            for invalid_payload, invalid_command in invalid_payloads:
                with (
                    self.subTest(payload=invalid_payload),
                    patch.object(Path, "read_text", return_value=json.dumps(invalid_payload)),
                    patch.object(Path, "read_bytes", return_value=invalid_command),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler._pid_start_time",
                        return_value=5,
                    ),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler.time.monotonic",
                        side_effect=[0.0, 0.0, 3.0],
                    ),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler.time.sleep"
                    ),
                ):
                    self.assertEqual(
                        ExperimentSchedulerService._read_ready_process(attempt_dir, "attempt-a1"),
                        {},
                    )

            fake_proc = Path(td) / "123"
            fake_proc.mkdir()
            (fake_proc / "cmdline").write_bytes(command)
            non_pid = Path(td) / "not-a-pid"
            non_pid.mkdir()
            with (
                patch.object(Path, "iterdir", return_value=[non_pid, fake_proc]),
                patch.object(os, "getpgid", return_value=456),
            ):
                self.assertEqual(
                    ExperimentSchedulerService._find_attempt_process(attempt_dir, "attempt-a1"),
                    {"pid": 123, "pgid": 456},
                )
            with patch.object(Path, "iterdir", side_effect=OSError("no proc")):
                self.assertEqual(
                    ExperimentSchedulerService._find_attempt_process(attempt_dir, "attempt-a1"),
                    {},
                )
            with patch.object(Path, "read_bytes", return_value=command):
                self.assertTrue(
                    ExperimentSchedulerService._event_process_matches(
                        {"pid": 123, "pgid": 123, "attempt_id": "attempt-a1"},
                        attempt_dir=attempt_dir,
                    )
                )

    def test_recovery_does_not_terminate_reused_process_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            event = {
                "event": "launched",
                "job_id": "stale-process",
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "stale-process",
                "profile": "cpu",
                "work_class": "ordinary",
                "command": ["task"],
                "environment_values": {},
                "attempt": 1,
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "pid_start_time": -1,
                "allocation_id": "stale-allocation",
            }
            (service.state_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n", encoding="utf-8"
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.process_group_alive",
                    return_value=True,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.terminate_process_group"
                ) as terminate,
            ):
                service._recover_terminal_events()
            terminate.assert_not_called()
            self.assertEqual(service._jobs["stale-process"].state, "drained_unknown")

    def test_stop_retains_owner_until_active_process_is_reaped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            first.start()
            submitted = first.submit(
                {
                    "command": [sys.executable, "-c", "import time;time.sleep(30)"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "drain-before-handoff",
                    "profile": "cpu",
                }
            )
            deadline = time.time() + 5
            while submitted.job_id not in first._active and time.time() < deadline:
                time.sleep(0.02)
            process = first._active[submitted.job_id].process
            self.assertIsNotNone(process)

            stop_thread = threading.Thread(target=first.stop)
            stop_thread.start()
            deadline = time.time() + 3
            while not first._stopping and time.time() < deadline:
                time.sleep(0.01)
            contender = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "already owns run"):
                    contender.start()
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
                stop_thread.join(timeout=5)
                self.assertFalse(stop_thread.is_alive())

                successor = ExperimentSchedulerService(
                    run_dir=run_dir,
                    settings=_settings(maximum=1),
                    allocator=_GPUAllocator(""),
                )
                successor.start()
                successor.stop()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                stop_thread.join(timeout=5)

    def test_recovered_terminal_waiter_is_not_blocked_by_allocation_release_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("")
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "recovered-release-retry",
                }
            )
            service._queue.remove(job.job_id)
            job.state = "running"
            job.pgid = 123
            allocation = Allocation(
                allocation_id="recovered-allocation",
                run_id=service.run_id,
                pid=123,
                pgid=123,
                profile="cpu",
                gpu_uuids=(),
                gpu_memory_mb=0,
                gpu_utilization_pct=0,
                started_at=time.time(),
            )
            service._active[job.job_id] = _ActiveJob(job, None, allocation, None, None)
            result: dict[str, object] = {}
            waiter = threading.Thread(
                target=lambda: result.update(service.wait(job.job_id, 2.0)),
                daemon=True,
            )
            waiter.start()
            time.sleep(0.02)
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.process_group_alive",
                    return_value=False,
                ),
                patch.object(allocator, "release", side_effect=OSError("registry unavailable")),
            ):
                service._reap()
            waiter.join(timeout=0.5)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(result["job"]["state"], "drained_unknown")  # type: ignore[index]
            self.assertIn(job.job_id, service._active)

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler.process_group_alive",
                return_value=False,
            ):
                service._reap()
            self.assertNotIn(job.job_id, service._active)
            events = [
                json.loads(line)
                for line in (service.state_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                sum(
                    event["event"] == "completed" and event.get("job_id") == job.job_id
                    for event in events
                ),
                1,
            )

    def test_run_dir_is_resolved_before_child_uses_different_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            previous = Path.cwd()
            try:
                os.chdir(td)
                service = ExperimentSchedulerService(
                    run_dir=Path("relative-run"), settings=_settings(maximum=1)
                )
            finally:
                os.chdir(previous)
            self.assertTrue(service.run_dir.is_absolute())
            self.assertEqual(service.run_dir, Path(td).resolve() / "relative-run")

    def test_unknown_explicit_profile_is_rejected_without_default_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            with self.assertRaisesRegex(ExperimentRejected, "unknown resource profile"):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "profile-typo",
                        "profile": "gup",
                    }
                )

    def test_scheduler_endpoint_enters_scoped_agent_runtime_environment(self) -> None:
        from praxist.plugins.workflow_stages.research_loop.backend.agent import (
            _legacy_runtime_env_keys,
        )

        self.assertIn("PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT", _legacy_runtime_env_keys())

    def test_final_launcher_overrides_inherited_gpu_zero_with_assigned_uuid(self) -> None:
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
        ):
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=2),
                allocator=_GPUAllocator("GPU-exact-uuid"),
            )
            service.start()
            output = Path(td) / "env.json"
            try:
                code = submit_and_wait(
                    [
                        sys.executable,
                        "-c",
                        "import json,os,sys;json.dump({k:os.environ.get(k) for k in "
                        "['CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PRAXIST_ASSIGNED_GPU_UUIDS']},open(sys.argv[1],'w'))",
                        str(output),
                    ],
                    peer_id="gen0_peer0",
                    experiment_id="semantic-variant-a",
                    profile="gpu",
                    run_dir=run_dir,
                )
            finally:
                service.stop()
            self.assertEqual(code, 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["CUDA_VISIBLE_DEVICES"], "GPU-exact-uuid")
            self.assertEqual(payload["NVIDIA_VISIBLE_DEVICES"], "GPU-exact-uuid")
            self.assertEqual(payload["PRAXIST_ASSIGNED_GPU_UUIDS"], "GPU-exact-uuid")

    def test_final_launcher_preserves_submitter_environment_and_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td).resolve() / "run"
            work_dir = Path(td).resolve() / "peer-workspace"
            work_dir.mkdir()
            output = Path(td).resolve() / "context.json"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("", limit=1),
            )
            service.start()
            previous = Path.cwd()
            try:
                os.chdir(work_dir)
                with patch.dict(os.environ, {"TASK_SCOPED_VALUE": "kept"}, clear=False):
                    code = submit_and_wait(
                        [
                            sys.executable,
                            "-c",
                            "import json,os,sys;json.dump({'cwd':os.getcwd(),"
                            "'value':os.getenv('TASK_SCOPED_VALUE')},open(sys.argv[1],'w'))",
                            str(output),
                        ],
                        peer_id="gen0_peer0",
                        experiment_id="context-preserved",
                        run_dir=run_dir,
                        cwd=Path("."),
                    )
            finally:
                os.chdir(previous)
                service.stop()
            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(output.read_text()),
                {"cwd": str(work_dir), "value": "kept"},
            )

    def test_relative_evaluator_runs_from_task_root_with_isolated_python_environment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            run_dir = root / "run"
            task_root = root / "task"
            caller_cwd = run_dir / "gen_0" / "peer_workspace"
            evaluator = task_root / "evaluations" / "public" / "run.py"
            output = root / "evaluation-context.json"
            evaluator.parent.mkdir(parents=True)
            caller_cwd.mkdir(parents=True)
            evaluator.write_text(
                "import json, os, pathlib, sys\n"
                "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
                "'cwd': os.getcwd(), 'pythonpath': os.getenv('PYTHONPATH')}))\n",
                encoding="utf-8",
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("", limit=1),
            )
            service.start()
            previous = Path.cwd()
            try:
                os.chdir(caller_cwd)
                with patch.dict(
                    os.environ,
                    {
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_TASK_PYTHON": sys.executable,
                        "PYTHONPATH": "/runner/site-packages",
                    },
                    clear=False,
                ):
                    code = submit_and_wait(
                        ["python", "evaluations/public/run.py", str(output)],
                        peer_id="gen0_peer0",
                        experiment_id="relative-task-evaluator",
                        run_dir=run_dir,
                    )
            finally:
                os.chdir(previous)
                service.stop()

            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"cwd": str(caller_cwd), "pythonpath": None},
            )

    def test_explicit_empty_environment_does_not_inherit_scheduler_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            output = Path(td) / "environment.json"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            service.start()
            try:
                job = service.submit(
                    {
                        "command": [
                            sys.executable,
                            "-c",
                            "import json,os,sys;json.dump({'home':os.getenv('HOME'),"
                            "'secret':os.getenv('OPENAI_API_KEY')},open(sys.argv[1],'w'))",
                            str(output),
                        ],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "empty-environment",
                        "environment": {},
                    }
                )
                result = service.wait(job.job_id, 5)["job"]
            finally:
                service.stop()
            self.assertEqual(result["state"], "completed")
            self.assertEqual(json.loads(output.read_text()), {"home": None, "secret": None})

    def test_semantic_identity_deduplicates_command_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            first = service.submit(
                {
                    "command": ["python", "a.py", "--output", "one"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "same-science",
                }
            )
            second = service.submit(
                {
                    "command": ["python", "a.py", "--output", "two", "--verbose"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "same-science",
                }
            )
            self.assertEqual(first.job_id, second.job_id)
            self.assertEqual(len(service._queue), 1)

    def test_close_atomically_rejects_queued_work_but_drains_running(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1, initial=1),
                allocator=_GPUAllocator("", limit=1),
            )
            service.start()
            try:
                first = service.submit(
                    {
                        "command": [sys.executable, "-c", "import time;time.sleep(2)"],
                        "peer_id": "gen2_peer0",
                        "generation_id": 2,
                        "experiment_id": "already-running",
                    }
                )
                deadline = time.time() + 3
                while service._jobs[first.job_id].state != "running" and time.time() < deadline:
                    time.sleep(0.02)
                second = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen2_peer1",
                        "generation_id": 2,
                        "experiment_id": "queued-late",
                    }
                )
                freeze_generation(2, "mature_quorum")
                second_result = service.wait(second.job_id, 2)["job"]
                first_result = service.wait(first.job_id, 5)["job"]
                self.assertEqual(second_result["state"], "rejected")
                self.assertEqual(first_result["state"], "completed")
                same = service.submit(
                    {
                        "command": [sys.executable, "-c", "raise SystemExit(99)"],
                        "peer_id": "gen2_peer0",
                        "generation_id": 2,
                        "experiment_id": "already-running",
                    }
                )
                self.assertEqual(same.job_id, first.job_id)
                with self.assertRaises(ExperimentRejected):
                    service.submit(
                        {
                            "command": [sys.executable, "-c", "pass"],
                            "peer_id": "gen2_peer2",
                            "generation_id": 2,
                            "experiment_id": "after-close",
                        }
                    )
            finally:
                service.stop()

    def test_only_exit_75_retries_and_keeps_one_semantic_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            marker = Path(td) / "attempt"
            code = (
                "import pathlib,sys;p=pathlib.Path(sys.argv[1]);"
                "seen=p.exists();p.write_text('x');sys.exit(0 if seen else 75)"
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            service.start()
            try:
                result = submit_and_wait(
                    [sys.executable, "-c", code, str(marker)],
                    peer_id="gen0_peer0",
                    experiment_id="retry-me",
                    run_dir=run_dir,
                )
                status = service.status()
            finally:
                service.stop()
            self.assertEqual(result, 0)
            self.assertEqual(len(status["jobs"]), 1)
            self.assertEqual(status["jobs"][0]["attempts"], 2)
            self.assertEqual(status["jobs"][0]["failure_category"], "")
            self.assertEqual(status["failure_counts_by_category"], {})
            events = [
                json.loads(line)
                for line in (run_dir / "resource_scheduler" / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                [
                    "submitted",
                    "launch_intent",
                    "process_started",
                    "launched",
                    "retry_queued",
                    "launch_intent",
                    "process_started",
                    "launched",
                    "completed",
                    "admission_closed",
                ],
            )
            retry_event = next(event for event in events if event["event"] == "retry_queued")
            self.assertEqual(retry_event["failure_category"], "infrastructure")

    def test_corrected_terminal_request_requires_explicit_retry_and_keeps_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            service.start()
            command = [
                sys.executable,
                "-c",
                "import os,sys;sys.exit(0 if os.getenv('PRAXIST_TEST_FIXED') == '1' else 9)",
            ]
            try:
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("PRAXIST_TEST_FIXED", None)
                    self.assertEqual(
                        submit_and_wait(
                            command,
                            peer_id="gen0_peer0",
                            experiment_id="corrected-contract",
                            run_dir=run_dir,
                        ),
                        9,
                    )

                duplicate_response = service.handle_request(
                    {
                        "action": "submit",
                        "command": command,
                        "environment": dict(os.environ),
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "corrected-contract",
                    }
                )
                self.assertTrue(duplicate_response["existing_terminal_job"])
                self.assertTrue(duplicate_response["retry_requires_explicit_request"])

                with patch.dict(os.environ, {"PRAXIST_TEST_FIXED": "1"}):
                    with self.assertRaisesRegex(ExperimentRejected, "--retry-terminal"):
                        submit_and_wait(
                            command,
                            peer_id="gen0_peer0",
                            experiment_id="corrected-contract",
                            run_dir=run_dir,
                        )
                    self.assertEqual(
                        submit_and_wait(
                            command,
                            peer_id="gen0_peer0",
                            experiment_id="corrected-contract",
                            run_dir=run_dir,
                            retry_terminal=True,
                        ),
                        0,
                    )
                status = service.status()
            finally:
                service.stop()

            self.assertEqual(len(status["jobs"]), 1)
            self.assertEqual(status["jobs"][0]["state"], "completed")
            self.assertEqual(status["jobs"][0]["attempts"], 2)
            events = [
                json.loads(line)
                for line in (run_dir / "resource_scheduler" / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            explicit_retry = next(
                event
                for event in events
                if event.get("event") == "retry_queued"
                and event.get("retry_reason") == "explicit_terminal_retry"
            )
            self.assertEqual(explicit_retry["prior_terminal_state"], "failed")
            self.assertEqual(sum(event.get("event") == "submitted" for event in events), 1)

    def test_explicit_terminal_retry_rejects_nonretryable_states_and_wrong_peer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            request = {
                "command": [sys.executable, "-c", "pass"],
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "experiment_id": "stable-science",
            }
            with self.assertRaisesRegex(ExperimentRejected, "retained failed or rejected"):
                service.submit({**request, "retry_terminal": True})
            queued = service.submit(request)
            duplicate = service.submit({**request, "retry_terminal": True})
            self.assertIs(duplicate, queued)
            with self.assertRaisesRegex(ExperimentRejected, "current state is 'queued'"):
                service.submit(
                    {
                        **request,
                        "command": [sys.executable, "-c", "raise SystemExit(2)"],
                        "retry_terminal": True,
                    }
                )

            service._queue.remove(queued.job_id)
            service._finish_without_launch(queued, "task_contract_rejected")
            with self.assertRaisesRegex(ExperimentRejected, "inherited runtime identity"):
                service.submit(
                    {
                        **request,
                        "environment": {
                            **os.environ,
                            "PRAXIST_PEER_ID": "gen0_peer1",
                            "PEER_ID": "gen0_peer1",
                        },
                        "retry_terminal": True,
                    }
                )
            with self.assertRaisesRegex(ExperimentRejected, "peer that owns"):
                service.submit(
                    {
                        **request,
                        "peer_id": "gen0_peer1",
                        "retry_terminal": True,
                    }
                )
            retried = service.submit({**request, "retry_terminal": True})
            self.assertEqual(retried.job_id, queued.job_id)
            self.assertEqual(retried.state, "queued")
            self.assertEqual(service._queue, [queued.job_id])
            self.assertIs(service.submit({**request, "retry_terminal": True}), retried)

            retried.state = "completed"
            self.assertIs(service.submit({**request, "retry_terminal": True}), retried)

    def test_admission_timeout_after_attempt_preserves_identity_and_attempt_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            request = {
                "command": [sys.executable, "-c", "pass"],
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "experiment_id": "attempted-admission-timeout",
            }
            attempted = service.submit(request)
            attempted.attempts = 1

            self.assertTrue(service.cancel_queued(attempted.job_id)["cancelled"])
            self.assertEqual(attempted.state, "rejected")
            self.assertIs(service.submit(request), attempted)

            retried = service.submit({**request, "retry_terminal": True})
            self.assertEqual(retried.job_id, attempted.job_id)
            self.assertEqual(retried.attempts, 1)
            self.assertEqual(retried.state, "queued")

    def test_submit_response_does_not_mislabel_new_terminal_job_as_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(run_dir=Path(td) / "run", settings=_settings())
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "fresh-terminal-response",
                }
            )
            job.state = "failed"
            with patch.object(service, "_submit", return_value=(job, False)):
                response = service.handle_request({"action": "submit"})

            self.assertFalse(response["existing_terminal_job"])
            self.assertFalse(response["retry_requires_explicit_request"])

    def test_explicit_retry_recovery_replaces_stale_terminal_runtime_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            settings = _settings(maximum=1)
            settings.infrastructure_retries = 0
            first = ExperimentSchedulerService(run_dir=run_dir, settings=settings)
            request = {
                "command": [sys.executable, "-c", "pass"],
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "experiment_id": "retry-replay-reset",
            }
            failed = first.submit(request)
            first._queue.remove(failed.job_id)
            failed.attempts = 1
            failed.pid = 1234
            failed.pgid = 1234
            failed.log_path = "/old/attempt.log"
            failed.error = "old failure"
            first._retry_or_finish(failed, 9)
            self.assertEqual(failed.state, "failed")

            retried = first.submit({**request, "retry_terminal": True})
            self.assertEqual(retried.job_id, failed.job_id)

            resumed = ExperimentSchedulerService(run_dir=run_dir, settings=settings)
            resumed._recover_terminal_events()
            recovered = resumed._jobs[failed.job_id]
            self.assertEqual(recovered.state, "queued")
            self.assertEqual(recovered.pid, 0)
            self.assertEqual(recovered.pgid, 0)
            self.assertEqual(recovered.log_path, "")
            self.assertEqual(recovered.error, "")

    def test_launch_failure_retries_once_without_duplicate_queue_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            service.start()
            try:
                code = submit_and_wait(
                    [str(Path(td) / "does-not-exist")],
                    peer_id="gen0_peer0",
                    experiment_id="missing-launcher",
                    run_dir=run_dir,
                )
                status = service.status()
            finally:
                service.stop()
            self.assertEqual(code, 75)
            self.assertEqual(status["queued"], 0)
            self.assertEqual(status["failed"], 1)
            self.assertEqual(status["jobs"][0]["attempts"], 2)

    def test_attempt_directory_failure_releases_reservation_without_limbo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("")
            settings = _settings(maximum=1)
            settings.infrastructure_retries = 0
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=settings, allocator=allocator
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "mkdir-failure",
                }
            )
            with patch.object(Path, "mkdir", side_effect=OSError("disk unavailable")):
                self.assertTrue(service._launch(job))
            self.assertEqual(job.state, "failed")
            self.assertEqual(service._queue, [])
            self.assertEqual(service._active, {})
            self.assertEqual(len(allocator.released), 1)

    def test_log_open_failure_releases_reservation_without_limbo(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("")
            settings = _settings(maximum=1)
            settings.infrastructure_retries = 0
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=settings, allocator=allocator
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "log-failure",
                }
            )
            real_open = open

            def fail_log(path, *args, **kwargs):
                if str(path).endswith(".log"):
                    raise OSError("log unavailable")
                return real_open(path, *args, **kwargs)

            with patch("builtins.open", side_effect=fail_log):
                self.assertTrue(service._launch(job))
            self.assertEqual(job.state, "failed")
            self.assertEqual(service._queue, [])
            self.assertEqual(service._active, {})
            self.assertEqual(len(allocator.released), 1)

    def test_manifest_failure_never_releases_barrier_or_starts_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("")
            settings = _settings(maximum=1)
            settings.infrastructure_retries = 0
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=settings, allocator=allocator
            )
            child_pid_file = Path(td) / "child.pid"
            parent_code = "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('started')"
            job = service.submit(
                {
                    "command": [sys.executable, "-c", parent_code, str(child_pid_file)],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "manifest-failure-tree",
                }
            )

            def fail_manifest(_job) -> None:
                raise OSError("manifest unavailable")

            with patch.object(service, "_register_protected", side_effect=fail_manifest):
                self.assertTrue(service._launch(job))
            self.assertFalse(child_pid_file.exists())
            self.assertEqual(job.state, "failed")
            self.assertEqual(service._queue, [])
            self.assertEqual(service._active, {})
            self.assertEqual(len(allocator.released), 1)

    def test_binding_failure_terminates_started_process_and_releases_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _BindFailAllocator("")
            settings = _settings(maximum=1)
            settings.infrastructure_retries = 0
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=settings, allocator=allocator
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "import time;time.sleep(30)"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "binding-failure",
                }
            )
            self.assertTrue(service._launch(job))
            self.assertEqual(job.state, "failed")
            self.assertEqual(service._queue, [])
            self.assertEqual(service._active, {})
            self.assertEqual(len(allocator.released), 1)

    def test_retry_queued_event_recovers_after_scheduler_restart(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            marker = Path(td) / "recovered"
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = first.submit(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('ran')",
                        str(marker),
                    ],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "recover-retry",
                    "environment": {"TASK_RECOVERY_VALUE": "preserved"},
                }
            )
            job.command = [
                sys.executable,
                "-c",
                "import os,pathlib,sys;pathlib.Path(sys.argv[1]).write_text(os.environ['TASK_RECOVERY_VALUE'])",
                str(marker),
            ]
            first._queue.remove(job.job_id)
            job.attempts = 1
            first._retry_or_finish(job, 75)

            second = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            second.start()
            try:
                duplicate = second.submit(
                    {
                        "command": [sys.executable, "-c", "raise SystemExit(99)"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "recover-retry",
                    }
                )
                result = second.wait(duplicate.job_id, 5)["job"]
            finally:
                second.stop()
            self.assertEqual(result["state"], "completed")
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(marker.read_text(), "preserved")

    def test_terminal_and_retry_state_wait_for_required_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "durable-transition",
                }
            )
            service._queue.remove(job.job_id)
            job.state = "running"
            job.attempts = 1

            with (
                patch.object(service, "_append_event", side_effect=OSError("disk unavailable")),
                self.assertRaisesRegex(OSError, "disk unavailable"),
            ):
                service._retry_or_finish(job, 75)
            self.assertEqual(job.state, "running")
            self.assertNotIn(job.job_id, service._queue)

            service.settings.infrastructure_retries = 0
            with (
                patch.object(service, "_append_event", side_effect=OSError("disk unavailable")),
                self.assertRaisesRegex(OSError, "disk unavailable"),
            ):
                service._retry_or_finish(job, 1)
            self.assertEqual(job.state, "running")
            self.assertIsNone(job.exit_code)

    def test_assessment_does_not_retry_non_mature_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "assessment-retry",
                    "work_class": "ordinary",
                }
            )
            service._queue.remove(job.job_id)
            job.state = "running"
            job.attempts = 1
            service.begin_assessment(0)
            service._retry_or_finish(job, 75)
            self.assertEqual(job.state, "failed")
            self.assertNotIn(job.job_id, service._queue)

    def test_legacy_event_and_manifest_pair_recovers_live_job(self) -> None:
        if not Path("/proc").is_dir():
            self.skipTest("/proc filesystem required for process environment inspection")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            process_env = {
                **os.environ,
                "PRAXIST_EXPERIMENT_ID": "legacy-live",
                "PRAXIST_EXPERIMENT_ATTEMPT_ID": "legacy-job-a1",
            }
            process = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                start_new_session=True,
                env=process_env,
            )
            path = protected_pids._manifest_path("gen1_peer0", run_dir)
            protected_pids._write_manifest(
                path,
                [
                    protected_pids.ProtectedEntry(
                        pid=process.pid,
                        pgid=process.pid,
                        peer_id="gen1_peer0",
                        tag="legacy-live",
                    )
                ],
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            event = {
                "event": "launched",
                "job_id": "legacy-job",
                "generation_id": 1,
                "peer_id": "gen1_peer0",
                "experiment_id": "legacy-live",
                "profile": "cpu",
                "work_class": "ordinary",
                "command": [sys.executable, "-c", "pass"],
                "environment_values": {},
                "attempt": 1,
                "attempt_id": "legacy-job-a1",
                "pid": process.pid,
                "pgid": process.pid,
                "allocation_id": "legacy-allocation",
            }
            (service.state_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n", encoding="utf-8"
            )
            try:
                service._recover_terminal_events()
                self.assertIn("legacy-job", service._active)
                persisted = protected_pids._read_manifest(path)
                self.assertEqual(len(persisted), 1)
                self.assertIsNotNone(persisted[0].pid_start_time)
            finally:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)

    def test_eventless_legacy_manifest_reserves_semantic_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            process = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                start_new_session=True,
            )
            path = protected_pids._manifest_path("gen2_peer0", run_dir)
            protected_pids._write_manifest(
                path,
                [
                    protected_pids.ProtectedEntry(
                        pid=process.pid,
                        pgid=process.pid,
                        peer_id="gen2_peer0",
                        tag="eventless-legacy",
                    )
                ],
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("", limit=1),
            )
            try:
                service._recover_active_process_groups()
                recovered = next(iter(service._active.values())).job
                duplicate = service.submit(
                    {
                        "command": [sys.executable, "-c", "raise SystemExit(99)"],
                        "peer_id": "gen2_peer0",
                        "generation_id": 2,
                        "experiment_id": "eventless-legacy",
                    }
                )
                self.assertEqual(duplicate.job_id, recovered.job_id)
                self.assertEqual(service._queue, [])
                unrelated = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen2_peer1",
                        "generation_id": 2,
                        "experiment_id": "unrelated-new-work",
                    }
                )
                service._launch_ready_jobs()
                self.assertEqual(unrelated.state, "queued")
                self.assertIn(unrelated.job_id, service._queue)
            finally:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)

    def test_manifest_recovery_keeps_descendants_after_launcher_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            launcher = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import subprocess,sys; "
                    "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'], "
                    "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
                ],
                start_new_session=True,
            )
            launcher_start = protected_pids._pid_start_time(launcher.pid)
            launcher.wait(timeout=5)
            path = protected_pids._manifest_path("gen4_peer0", run_dir)
            protected_pids._write_manifest(
                path,
                [
                    protected_pids.ProtectedEntry(
                        pid=launcher.pid,
                        pgid=launcher.pid,
                        pid_start_time=launcher_start,
                        peer_id="gen4_peer0",
                        tag="descendant-live",
                    )
                ],
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("", limit=1),
            )
            try:
                service._recover_active_process_groups()
                self.assertEqual(len(service._active), 1)
                recovered = next(iter(service._active.values())).job
                self.assertEqual(recovered.experiment_id, "descendant-live")
                self.assertEqual(recovered.pgid, launcher.pid)
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(launcher.pid, signal.SIGKILL)

    def test_assessment_keeps_idle_waiter_for_future_maturity_debt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=2)
            settings.supply_idle_samples = 1
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=settings,
                allocator=_GPUAllocator("", limit=2),
            )
            completed = [1]
            service.open_generation(0, deadline=time.time() + 600, cohort_size=2)
            service.configure_generation_maturity(
                0,
                cohort_size=2,
                mature_target=1,
                count_callback=lambda: completed[0],
            )
            service.begin_assessment(0)
            service.register_idle_supply("gen0_peer0", 0)
            first = HostSnapshot(8, 5, 5, 0, observed_at=1.0)
            service._refresh_mature_counts(first)
            service._reconcile_supply(first)
            self.assertIn("gen0_peer0", service._idle_supply_waiters)
            self.assertEqual(service._supply_leases, {})

            completed[0] = 0
            second = HostSnapshot(8, 5, 5, 0, observed_at=2.0)
            service._refresh_mature_counts(second)
            service._reconcile_supply(second)
            self.assertNotIn("gen0_peer0", service._idle_supply_waiters)
            self.assertEqual(len(service._supply_leases), 1)

            lease_id = next(iter(service._supply_leases))
            service.release_supply_lease(lease_id, "gen0_peer0", declined=True)
            service.register_idle_supply("gen0_peer0", 0)
            third = HostSnapshot(8, 5, 5, 0, observed_at=3.0)
            service._reconcile_supply(third)
            self.assertEqual(service._supply_leases, {})

            service._declined_supply_peers[("gen0_peer0", 0, "mature")] = (0.0, 1)
            fourth = HostSnapshot(8, 5, 5, 0, observed_at=4.0)
            service._reconcile_supply(fourth)
            self.assertEqual(len(service._supply_leases), 1)

    def test_maturity_refresh_does_not_depend_on_a_new_host_sample(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("", limit=1),
            )
            completed = {"count": 0}
            service.open_generation(0, deadline=time.time() + 600, cohort_size=1)
            service.configure_generation_maturity(
                0,
                cohort_size=1,
                mature_target=1,
                count_callback=lambda: completed["count"],
            )
            snapshot = HostSnapshot(8, 5, 5, 0, observed_at=1.0)
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler.time.monotonic",
                side_effect=(10.0, 16.0),
            ):
                service._refresh_mature_counts(snapshot)
                completed["count"] = 1
                service._refresh_mature_counts(snapshot)
            self.assertEqual(
                service.status()["resource_supply"]["maturity"]["0"]["completed"],
                1,
            )

    def test_launch_intent_recovers_waiting_barrier_without_duplicate_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td).resolve() / "run"
            marker = Path(td).resolve() / "executed"
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = first.submit(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        "import os,pathlib,sys; "
                        "from praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler_client import scheduler_attempt_is_active; "
                        "active=scheduler_attempt_is_active("
                        "pathlib.Path(os.environ['PRAXIST_RUN_DIR']), "
                        "os.environ['PRAXIST_EXPERIMENT_ATTEMPT_ID'], os.getpgrp()); "
                        "pathlib.Path(sys.argv[1]).write_text('active' if active else 'inactive')",
                        str(marker),
                    ],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "crash-after-popen",
                    "environment": dict(os.environ),
                }
            )
            first._queue.remove(job.job_id)
            job.attempts = 1
            attempt_id = f"{job.job_id}-a1"
            attempt_dir = first.state_dir / "attempts" / attempt_id
            attempt_dir.mkdir(parents=True)
            job.environment.update(
                {
                    "PRAXIST_RUN_DIR": str(run_dir),
                    "AUTO_RESEARCH_RUN_DIR": str(run_dir),
                    "PRAXIST_EXPERIMENT_ATTEMPT_ID": attempt_id,
                    "PRAXIST_EXPERIMENT_ATTEMPT_DIR": str(attempt_dir),
                }
            )
            ready = attempt_dir / "READY.json"
            go = attempt_dir / "GO.json"
            first._append_event(
                {
                    "event": "launch_intent",
                    **first._event_identity(job, include_request=True),
                    "allocation_id": "recovered-allocation",
                    "gpu_uuids": [],
                    "attempt_id": attempt_id,
                    "attempt_dir": str(attempt_dir),
                },
                required=True,
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-E",
                    "-S",
                    str(
                        Path(
                            "praxist/plugins/workflow_stages/research_loop/backend/experiment_exec.py"
                        ).resolve()
                    ),
                    str(ready),
                    str(go),
                    attempt_id,
                    *job.command,
                ],
                env={**os.environ, **job.environment},
                start_new_session=True,
            )
            deadline = time.time() + 3
            while not ready.exists() and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists())
            self.assertFalse(marker.exists())

            second = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            second.start()
            try:
                recovered = second.submit(
                    {
                        "command": [sys.executable, "-c", "raise SystemExit(99)"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "crash-after-popen",
                    }
                )
                process.wait(timeout=5)
                result = second.wait(recovered.job_id, 5)["job"]
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                second.stop()
            self.assertEqual(result["state"], "drained_unknown")
            self.assertEqual(second.status()["queued"], 0)
            self.assertEqual(marker.read_text(), "active")

    def test_recovery_requeues_waiting_barrier_after_task_checkout_moves(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            run_dir = root / "run"
            old_task = root / "old" / "task"
            new_task = root / "new" / "task"
            old_evaluator = old_task / "evaluations" / "run.py"
            new_evaluator = new_task / "evaluations" / "run.py"
            marker = root / "executed"
            old_evaluator.parent.mkdir(parents=True)
            new_evaluator.parent.mkdir(parents=True)
            old_evaluator.write_text(
                "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('old')\n",
                encoding="utf-8",
            )
            new_evaluator.write_text(
                "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('new')\n",
                encoding="utf-8",
            )
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = first.submit(
                {
                    "command": [sys.executable, str(old_evaluator), str(marker)],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "relocated-before-go",
                    "profile": "cpu",
                    "cwd": str(old_task),
                    "environment": {"PRAXIST_TASK_PROJECT_PATH": str(old_task)},
                }
            )
            first._queue.remove(job.job_id)
            job.attempts = 1
            attempt_id = f"{job.job_id}-a1"
            attempt_dir = first.state_dir / "attempts" / attempt_id
            attempt_dir.mkdir(parents=True)
            ready = attempt_dir / "READY.json"
            go = attempt_dir / "GO.json"
            first._append_event(
                {
                    "event": "launch_intent",
                    **first._event_identity(job, include_request=True),
                    "allocation_id": "relocated-barrier-allocation",
                    "gpu_uuids": [],
                    "attempt_id": attempt_id,
                    "attempt_dir": str(attempt_dir),
                },
                required=True,
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-E",
                    "-S",
                    str(
                        Path(
                            "praxist/plugins/workflow_stages/research_loop/backend/experiment_exec.py"
                        ).resolve()
                    ),
                    str(ready),
                    str(go),
                    attempt_id,
                    *job.command,
                ],
                cwd=old_task,
                env={**os.environ, **job.environment},
                start_new_session=True,
            )
            deadline = time.time() + 3
            while not ready.exists() and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists())

            second = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            try:
                with patch.dict(
                    os.environ,
                    {"PRAXIST_TASK_PROJECT_PATH": str(new_task)},
                    clear=False,
                ):
                    second.start()
                    result = second.wait(job.job_id, 10)["job"]
                process.wait(timeout=5)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                second.stop()

            self.assertEqual(result["state"], "completed")
            self.assertEqual(marker.read_text(encoding="utf-8"), "new")
            events = (second.state_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn('"recovery_reason": "task_context_relocated_before_launch"', events)

    def test_resume_uses_latest_launched_state_instead_of_requeueing_old_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = first.submit(
                {
                    "command": [sys.executable, "-c", "import time;time.sleep(30)"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "live-second-attempt",
                }
            )
            first._queue.remove(job.job_id)
            job.attempts = 1
            first._retry_or_finish(job, 75)
            self.assertTrue(first._launch(job))
            process = first._active[job.job_id].process
            assert process is not None

            second = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            second.start()
            try:
                self.assertEqual(second.status()["queued"], 0)
                self.assertEqual(second.status()["running"], 1)
                duplicate = second.submit(
                    {
                        "command": [sys.executable, "-c", "raise SystemExit(99)"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "live-second-attempt",
                    }
                )
                self.assertEqual(duplicate.semantic_key, job.semantic_key)
            finally:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                second.stop()
                first._active[job.job_id].log_handle.close()

    def test_resume_preserves_identity_when_process_finished_before_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            marker = Path(td) / "runs"
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = first.submit(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('once')",
                        str(marker),
                    ],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "finished-before-reap",
                }
            )
            self.assertTrue(first._launch(job))
            process = first._active[job.job_id].process
            assert process is not None
            process.wait(timeout=5)
            first._active[job.job_id].log_handle.close()

            second = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            second.start()
            try:
                duplicate = second.submit(
                    {
                        "command": [sys.executable, "-c", "raise SystemExit(99)"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "finished-before-reap",
                    }
                )
                recovered = second.wait(duplicate.job_id, 1)["job"]
            finally:
                second.stop()
            self.assertEqual(recovered["state"], "drained_unknown")
            self.assertEqual(marker.read_text(), "once")

    def test_deadline_admission_rejects_late_full_work_without_killing_active_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            service.open_generation(4, deadline=time.time() + 5)
            with self.assertRaises(ExperimentRejected):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen4_peer0",
                        "generation_id": 4,
                        "experiment_id": "full-too-late",
                        "work_class": "mature",
                        "eta_seconds": 60,
                    }
                )

    def test_assessment_uses_bounded_completion_probability_for_mature_topup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            service.open_generation(4, deadline=time.time() + 30)
            service.begin_assessment(4, "maturity_debt")

            admitted = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen4_peer0",
                    "generation_id": 4,
                    "experiment_id": "probable-mature-topup",
                    "work_class": "mature",
                    "eta_seconds": 50,
                }
            )
            self.assertEqual(admitted.state, "queued")
            with self.assertRaises(ExperimentRejected):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen4_peer1",
                        "generation_id": 4,
                        "experiment_id": "improbable-mature-topup",
                        "work_class": "mature",
                        "eta_seconds": 60,
                    }
                )

    def test_successful_wall_time_calibrates_assessment_completion_probability(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "calibration",
                    "eta_seconds": 100,
                }
            )
            service._observe_runtime(job, started_at=time.time() - 50)
            self.assertAlmostEqual(service._runtime_log_ratios[-1], math.log(0.5), places=2)

    def test_gpu_retry_keeps_the_same_profile_and_never_falls_back_to_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            marker = Path(td) / "attempt"
            observed = Path(td) / "observed.jsonl"
            code = (
                "import json,os,pathlib,sys;"
                "p=pathlib.Path(sys.argv[1]);o=pathlib.Path(sys.argv[2]);"
                "f=o.open('a');f.write(json.dumps({'profile':os.getenv('PRAXIST_RESOURCE_PROFILE'),"
                "'gpu':os.getenv('CUDA_VISIBLE_DEVICES')})+'\\n');f.close();"
                "seen=p.exists();p.write_text('x');sys.exit(0 if seen else 75)"
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("GPU-fixed"),
            )
            service.start()
            try:
                result = submit_and_wait(
                    [sys.executable, "-c", code, str(marker), str(observed)],
                    peer_id="gen0_peer0",
                    experiment_id="gpu-stable-retry",
                    profile="gpu",
                    run_dir=run_dir,
                )
            finally:
                service.stop()
            rows = [json.loads(line) for line in observed.read_text().splitlines()]
            self.assertEqual(result, 0)
            self.assertEqual(rows, [{"profile": "gpu", "gpu": "GPU-fixed"}] * 2)

    def test_protected_pid_launch_is_a_facade_in_central_mode(self) -> None:
        with (
            patch.dict(os.environ, {"PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT": "/tmp/fake"}),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client.submit_and_wait",
                return_value=0,
            ) as submit,
            patch.object(protected_pids.subprocess, "Popen") as popen,
        ):
            result = protected_pids.launch_command(
                ["python", "train.py"],
                peer_id="gen0_peer0",
                tag="variant-a",
                resource_profile="gpu",
                work_class="mature",
                retry_terminal=True,
            )
        self.assertEqual(result, 0)
        submit.assert_called_once()
        self.assertTrue(submit.call_args.kwargs["retry_terminal"])
        popen.assert_not_called()

    def test_protected_pid_terminal_retry_uses_inherited_peer_identity(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT": "/tmp/fake",
                    "PRAXIST_PEER_ID": "gen0_peer1",
                    "PEER_ID": "gen0_peer1",
                },
            ),
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client.submit_and_wait",
            ) as submit,
            self.assertRaisesRegex(ValueError, "inherited runtime identity"),
        ):
            protected_pids.launch_command(
                ["python", "train.py"],
                peer_id="gen0_peer0",
                tag="variant-a",
                retry_terminal=True,
            )
        submit.assert_not_called()

    def test_protected_pid_identity_uses_ps_fallback_without_proc(self) -> None:
        with (
            patch.object(Path, "read_text", side_effect=OSError("no procfs")),
            patch("praxist.cli.registry.process_start_token", return_value="ps:stable-start"),
        ):
            observed = protected_pids._pid_start_time(4321)

        self.assertEqual(observed, "ps:stable-start")
        entry = protected_pids.ProtectedEntry(pid=4321, pid_start_time=observed)
        with patch.object(
            protected_pids,
            "_pid_start_time",
            return_value="ps:stable-start",
        ):
            self.assertTrue(protected_pids._entry_process_identity_matches(entry))
        with patch.object(protected_pids, "_pid_start_time", return_value=None):
            self.assertFalse(protected_pids._entry_process_identity_matches(entry))

    def test_run_owned_endpoint_prevents_peer_from_downgrading_to_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(run_dir=run_dir, settings=_settings(maximum=1))
            scheduler_dir = run_dir / "resource_scheduler"
            scheduler_dir.mkdir(parents=True, exist_ok=True)
            (scheduler_dir / "endpoint.json").write_text(
                json.dumps({"endpoint": str(service.endpoint)}),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "PRAXIST_EXPERIMENT_SCHEDULER_CONFIG": '{"mode":"legacy"}',
                        "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT": "",
                        "PRAXIST_EXPERIMENT_ATTEMPT_ID": "forged-attempt",
                        "PRAXIST_EXPERIMENT_ATTEMPT_DIR": str(
                            scheduler_dir / "attempts" / "forged-attempt"
                        ),
                    },
                    clear=False,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client.submit_and_wait",
                    return_value=0,
                ) as submit,
                patch.object(protected_pids.subprocess, "Popen") as popen,
            ):
                result = protected_pids.launch_command(
                    ["python", "train.py"],
                    peer_id="gen4_peer5",
                    tag="variant-e3",
                    run_dir=run_dir,
                )

        self.assertEqual(result, 0)
        self.assertEqual(
            submit.call_args.kwargs["scheduler_endpoint"],
            str(service.endpoint),
        )
        popen.assert_not_called()

    def test_run_owned_endpoint_rejects_redirected_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            scheduler_dir = run_dir / "resource_scheduler"
            scheduler_dir.mkdir(parents=True, exist_ok=True)
            (scheduler_dir / "endpoint.json").write_text(
                json.dumps({"endpoint": "/tmp/redirected.sock"}),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "PROTECTED_PIDS_DIR": str(run_dir / "protected_pids"),
                        "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT": "",
                    },
                    clear=False,
                ),
                patch.object(protected_pids.subprocess, "Popen") as popen,
                self.assertRaisesRegex(SchedulerUnavailable, "does not match its run"),
            ):
                protected_pids.launch_command(
                    ["python", "train.py"],
                    peer_id="gen0_peer0",
                    tag="redirected-endpoint",
                    run_dir=run_dir,
                )
        popen.assert_not_called()

    def test_explicit_run_dir_cannot_cross_inherited_run_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            inherited_run = Path(td) / "run-a"
            explicit_run = Path(td) / "run-b"
            with (
                patch.dict(
                    os.environ,
                    {
                        "PRAXIST_RUN_DIR": str(inherited_run),
                        "AUTO_RESEARCH_RUN_DIR": str(inherited_run),
                        "PROTECTED_PIDS_DIR": str(explicit_run / "protected_pids"),
                    },
                    clear=False,
                ),
                patch.object(protected_pids.subprocess, "Popen") as popen,
                self.assertRaisesRegex(ValueError, "does not match the inherited run"),
            ):
                protected_pids.launch_command(
                    ["python", "train.py"],
                    peer_id="gen0_peer0",
                    tag="wrong-run",
                    run_dir=explicit_run,
                )
        popen.assert_not_called()

    def test_attempt_validation_rejects_inconsistent_run_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            other_run = root / "other-run"
            with (
                patch.dict(
                    os.environ,
                    {
                        "PRAXIST_RUN_DIR": str(run_dir),
                        "AUTO_RESEARCH_RUN_DIR": str(other_run),
                    },
                    clear=False,
                ),
                self.assertRaisesRegex(ValueError, "run boundaries disagree"),
            ):
                protected_pids._run_dir_from_runtime_env()

            attempt_id = "attempt-1"
            attempts_root = run_dir / "resource_scheduler" / "attempts"
            attempt_dir = attempts_root / attempt_id
            attempt_dir.mkdir(parents=True)
            environment = {
                "PRAXIST_EXPERIMENT_ATTEMPT_ID": attempt_id,
                "PRAXIST_EXPERIMENT_ATTEMPT_DIR": str(root / "outside" / attempt_id),
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertFalse(protected_pids._inside_active_scheduler_attempt(run_dir))

            environment["PRAXIST_EXPERIMENT_ATTEMPT_DIR"] = str(attempt_dir)
            (attempt_dir / "READY.json").write_text(
                json.dumps({"attempt_id": "other-attempt", "pgid": os.getpgrp()}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, environment, clear=False):
                self.assertFalse(protected_pids._inside_active_scheduler_attempt(run_dir))

            (attempt_dir / "READY.json").write_text(
                json.dumps({"attempt_id": attempt_id, "pgid": os.getpgrp()}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, environment, clear=False):
                self.assertFalse(protected_pids._inside_active_scheduler_attempt(run_dir))

            with patch.object(Path, "exists", side_effect=OSError("unavailable")):
                self.assertIsNone(protected_pids._run_shutdown_signal(run_dir))

    def test_forged_attempt_files_cannot_bypass_live_scheduler_authority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(run_dir=run_dir, settings=_settings(maximum=1))
            scheduler_dir = run_dir / "resource_scheduler"
            scheduler_dir.mkdir(parents=True, exist_ok=True)
            (scheduler_dir / "endpoint.json").write_text(
                json.dumps({"endpoint": str(service.endpoint)}),
                encoding="utf-8",
            )
            attempt_dir = _write_active_attempt(run_dir, "forged-attempt")
            with (
                patch.dict(
                    os.environ,
                    {
                        "PRAXIST_EXPERIMENT_ATTEMPT_ID": "forged-attempt",
                        "PRAXIST_EXPERIMENT_ATTEMPT_DIR": str(attempt_dir),
                    },
                    clear=False,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client.scheduler_attempt_is_active",
                    return_value=False,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client.submit_and_wait",
                    return_value=0,
                ) as submit,
                patch.object(protected_pids.subprocess, "run") as direct_run,
            ):
                result = protected_pids.launch_command(
                    ["python", "train.py"],
                    peer_id="gen0_peer0",
                    tag="forged-nested",
                    run_dir=run_dir,
                )

        self.assertEqual(result, 0)
        submit.assert_called_once()
        direct_run.assert_not_called()

    def test_legacy_capacity_wait_rechecks_run_shutdown_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            shutdown = run_dir / "ORCHESTRATOR_SHUTDOWN"

            def write_shutdown(_seconds: float) -> None:
                shutdown.write_text("source=praxist_stop\n", encoding="utf-8")

            with (
                patch.dict(
                    os.environ,
                    {
                        "PRAXIST_EXPERIMENT_SCHEDULER_CONFIG": '{"mode":"legacy"}',
                        "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT": "",
                        "PRAXIST_EXPERIMENT_ATTEMPT_ID": "",
                    },
                    clear=False,
                ),
                patch.object(
                    protected_pids,
                    "_manifest_lock",
                    return_value=contextlib.nullcontext(),
                ),
                patch.object(protected_pids, "_read_manifest", return_value=[]),
                patch.object(
                    protected_pids,
                    "_check_peer_capacity",
                    side_effect=protected_pids.ProtectedPidCapacityError("busy"),
                ),
                patch.object(protected_pids.time, "sleep", side_effect=write_shutdown),
                patch.object(protected_pids.subprocess, "Popen") as popen,
                self.assertRaises(protected_pids.GenerationClosingLaunchError),
            ):
                protected_pids.launch_command(
                    ["python", "train.py"],
                    peer_id="gen0_peer0",
                    tag="late",
                    run_dir=run_dir,
                    wait_timeout_seconds=10,
                )

        popen.assert_not_called()

    def test_central_launch_never_falls_back_when_endpoint_is_missing(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "PRAXIST_EXPERIMENT_SCHEDULER_CONFIG": '{"mode":"central"}',
                    "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT": "",
                },
                clear=False,
            ),
            patch.object(protected_pids.subprocess, "Popen") as popen,
            self.assertRaisesRegex(SchedulerUnavailable, "endpoint is unavailable"),
        ):
            protected_pids.launch_command(
                ["python", "train.py"],
                peer_id="gen0_peer0",
                tag="variant-a",
            )
        popen.assert_not_called()

        with (
            patch.dict(
                os.environ,
                {
                    "PRAXIST_EXPERIMENT_SCHEDULER_CONFIG": '{"mode":"  "}',
                    "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT": "",
                },
                clear=False,
            ),
            patch.object(protected_pids.subprocess, "Popen") as blank_mode_popen,
            self.assertRaisesRegex(SchedulerUnavailable, "endpoint is unavailable"),
        ):
            protected_pids.launch_command(
                ["python", "train.py"],
                peer_id="gen0_peer0",
                tag="variant-b",
            )
        blank_mode_popen.assert_not_called()

    def test_scheduler_validates_only_its_live_attempt_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "attempt-authority",
                }
            )
            job.attempts = 2
            job.pgid = 4321
            job.pid = 4321
            job.state = "running"
            process = MagicMock()
            process.pid = 4321
            process.poll.return_value = None
            attempt_dir = service.state_dir / "attempts" / f"{job.job_id}-a2"
            attempt_dir.mkdir(parents=True)
            service._active[job.job_id] = _ActiveJob(
                job,
                process,
                None,
                None,
                attempt_dir,
                12345,
            )

            with (
                patch.object(protected_pids, "_pid_start_time", return_value=12345),
                patch.object(os, "getpgid", return_value=4321),
            ):
                accepted = service.handle_request(
                    {
                        "action": "validate_attempt",
                        "attempt_id": f"{job.job_id}-a2",
                        "pgid": 4321,
                    }
                )
                wrong_attempt = service.handle_request(
                    {
                        "action": "validate_attempt",
                        "attempt_id": f"{job.job_id}-a1",
                        "pgid": 4321,
                    }
                )
                wrong_group = service.handle_request(
                    {
                        "action": "validate_attempt",
                        "attempt_id": f"{job.job_id}-a2",
                        "pgid": 9876,
                    }
                )
                active_groups = service.handle_request({"action": "active_process_groups"})
                service._active[job.job_id].pid_start_time = 54321
                reused_identity = service.handle_request(
                    {
                        "action": "validate_attempt",
                        "attempt_id": f"{job.job_id}-a2",
                        "pgid": 4321,
                    }
                )
                service._active[job.job_id].pid_start_time = 12345
                process.poll.return_value = 0
                stale_attempt = service.handle_request(
                    {
                        "action": "validate_attempt",
                        "attempt_id": f"{job.job_id}-a2",
                        "pgid": 4321,
                    }
                )

        self.assertTrue(accepted["active"])
        self.assertFalse(wrong_attempt["active"])
        self.assertFalse(wrong_group["active"])
        self.assertFalse(reused_identity["active"])
        self.assertFalse(stale_attempt["active"])
        self.assertEqual(
            active_groups["groups"],
            [{"pgid": 4321, "pid": 4321, "pid_start_time": 12345}],
        )

    def test_scheduler_does_not_publish_exited_launcher_as_group_authority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "stale-group-authority",
                }
            )
            job.pid = job.pgid = 4321
            job.state = "running"
            process = MagicMock()
            process.pid = 4321
            process.poll.return_value = 0
            service._active[job.job_id] = _ActiveJob(job, process, None, None, None)

            active_groups = service.handle_request({"action": "active_process_groups"})

        self.assertEqual(active_groups["groups"], [])

    def test_scheduler_validates_verified_recovered_attempt_without_popen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "recovered-attempt-authority",
                }
            )
            process = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                start_new_session=True,
            )
            job.attempts = 1
            job.pid = job.pgid = process.pid
            job.state = "running"
            attempt_id = f"{job.job_id}-a1"
            attempt_dir = run_dir / "resource_scheduler" / "attempts" / attempt_id
            attempt_dir.mkdir(parents=True)
            service._active[job.job_id] = _ActiveJob(
                job,
                None,
                None,
                None,
                attempt_dir,
                protected_pids._pid_start_time(process.pid),
            )
            try:
                self.assertTrue(service.attempt_is_active(attempt_id, process.pid))
                self.assertFalse(service.attempt_is_active(f"{job.job_id}-a2", process.pid))
            finally:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            self.assertFalse(service.attempt_is_active(attempt_id, process.pid))

    def test_scheduler_validates_live_descendant_after_launcher_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            child_pid_path = root / "child.pid"
            launcher = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,subprocess,sys; "
                    "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                    "pathlib.Path(sys.argv[1]).write_text(str(child.pid))",
                    str(child_pid_path),
                ],
                start_new_session=True,
            )
            launcher_start = protected_pids._pid_start_time(launcher.pid)
            launcher.wait(timeout=5)
            self.assertGreater(int(child_pid_path.read_text(encoding="utf-8")), 1)
            run_dir = root / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "descendant-attempt-authority",
                }
            )
            job.attempts = 1
            job.pid = job.pgid = launcher.pid
            job.state = "running"
            attempt_id = f"{job.job_id}-a1"
            attempt_dir = run_dir / "resource_scheduler" / "attempts" / attempt_id
            attempt_dir.mkdir(parents=True)
            service._active[job.job_id] = _ActiveJob(
                job,
                launcher,
                None,
                None,
                attempt_dir,
                launcher_start,
            )
            try:
                self.assertTrue(service.attempt_is_active(attempt_id, launcher.pid))
                service._active[job.job_id].process = None
                self.assertTrue(service.attempt_is_active(attempt_id, launcher.pid))
            finally:
                os.killpg(launcher.pid, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while process_group_alive(launcher.pid) and time.monotonic() < deadline:
                    time.sleep(0.02)
            self.assertFalse(service.attempt_is_active(attempt_id, launcher.pid))

    def test_nested_task_child_stays_inside_existing_scheduler_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            attempt_dir = _write_active_attempt(run_dir, "attempt-1")
            output = Path(td) / "nested"
            with (
                patch.dict(
                    os.environ,
                    {
                        "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT": "",
                        "PRAXIST_EXPERIMENT_ATTEMPT_ID": "attempt-1",
                        "PRAXIST_EXPERIMENT_ATTEMPT_DIR": str(attempt_dir),
                    },
                    clear=False,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client.submit_and_wait"
                ) as submit,
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client.scheduler_attempt_is_active",
                    return_value=True,
                ),
            ):
                code = protected_pids.launch_command(
                    [
                        sys.executable,
                        "-c",
                        "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text('ok')",
                        str(output),
                    ],
                    peer_id="gen0_peer0",
                    tag="nested",
                    run_dir=run_dir,
                )
            observed = output.read_text()
        self.assertEqual(code, 0)
        self.assertEqual(observed, "ok")
        submit.assert_not_called()

    def test_nested_task_child_reuses_the_shared_task_runtime_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            task_root = root / "task"
            run_dir = root / "run"
            caller_cwd = run_dir / "attempt"
            attempt_dir = _write_active_attempt(run_dir, "attempt-1")
            evaluator = task_root / "evaluations" / "run.py"
            output = root / "nested-context.json"
            evaluator.parent.mkdir(parents=True)
            caller_cwd.mkdir(parents=True)
            evaluator.write_text(
                "import json, os, pathlib, sys\n"
                "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
                "'cwd': os.getcwd(), 'pythonpath': os.getenv('PYTHONPATH'), "
                "'pythonhome': os.getenv('PYTHONHOME')}))\n",
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT": "/tmp/not-contacted",
                        "PRAXIST_EXPERIMENT_ATTEMPT_ID": "attempt-1",
                        "PRAXIST_EXPERIMENT_ATTEMPT_DIR": str(attempt_dir),
                        "PRAXIST_TASK_PROJECT_PATH": str(task_root),
                        "PRAXIST_TASK_PYTHON": sys.executable,
                        "PYTHONPATH": "/runner/python313",
                        "PYTHONHOME": "/runner/python313",
                    },
                    clear=True,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client.submit_and_wait"
                ) as submit,
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler_client.scheduler_attempt_is_active",
                    return_value=True,
                ),
            ):
                code = protected_pids.launch_command(
                    ["bash", "-lc", f"python3.11 evaluations/run.py {output}"],
                    peer_id="gen0_peer0",
                    tag="nested-task-boundary",
                    run_dir=run_dir,
                    cwd=caller_cwd,
                )

            observed = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            observed,
            {"cwd": str(caller_cwd), "pythonpath": None, "pythonhome": None},
        )
        submit.assert_not_called()

    def test_filesystem_close_signal_is_scheduler_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            gen_dir = run_dir / "gen_5"
            gen_dir.mkdir(parents=True)
            (gen_dir / "CLOSING_SIGNAL").write_text("closing")
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("", limit=1),
            )
            with self.assertRaises(ExperimentRejected):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen5_peer0",
                        "generation_id": 5,
                        "experiment_id": "after-signal",
                    }
                )

    def test_resume_adopts_live_process_group_and_deduplicates_it(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            process = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                start_new_session=True,
            )
            path = protected_pids._manifest_path("gen3_peer2", run_dir)
            protected_pids._write_manifest(
                path,
                [
                    protected_pids.ProtectedEntry(
                        pid=process.pid,
                        pgid=process.pid,
                        peer_id="gen3_peer2",
                        tag="same-science",
                        pid_start_time=protected_pids._pid_start_time(process.pid),
                    )
                ],
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            service.start()
            try:
                recovered = next(iter(service._jobs.values()))
                duplicate = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen3_peer2",
                        "generation_id": 3,
                        "experiment_id": "same-science",
                    }
                )
                self.assertEqual(duplicate.job_id, recovered.job_id)
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
                result = service.wait(recovered.job_id, 5)["job"]
                self.assertEqual(result["state"], "drained_unknown")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                service.stop()

            second_service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            second_service.start()
            try:
                duplicate = second_service.submit(
                    {
                        "command": [sys.executable, "-c", "raise SystemExit(99)"],
                        "peer_id": "gen3_peer2",
                        "generation_id": 3,
                        "experiment_id": "same-science",
                    }
                )
                persisted = second_service.wait(duplicate.job_id, 1)["job"]
            finally:
                second_service.stop()
            self.assertEqual(persisted["state"], "drained_unknown")
            self.assertIsNone(persisted["exit_code"])

    def test_manifest_adoption_survives_repeated_restart_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            process = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                start_new_session=True,
            )
            path = protected_pids._manifest_path("gen2_peer1", run_dir)
            protected_pids._write_manifest(
                path,
                [
                    protected_pids.ProtectedEntry(
                        pid=process.pid,
                        pgid=process.pid,
                        peer_id="gen2_peer1",
                        tag="stable-identity",
                        pid_start_time=protected_pids._pid_start_time(process.pid),
                    )
                ],
            )
            first_allocator = _GPUAllocator("")
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=first_allocator,
            )
            first._acquire_owner_lock()
            try:
                first._recover_active_process_groups()
                adopted = next(iter(first._active.values()))
                self.assertIsNotNone(adopted.allocation)
                self.assertEqual(adopted.job.profile, "cpu")
            finally:
                first._release_owner_lock()

            second_allocator = _GPUAllocator("")
            second = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=second_allocator,
            )
            try:
                with (
                    patch.dict(os.environ, {"PRAXIST_TASK_PYTHON": sys.executable}),
                    patch(
                        "praxist.plugins.workflow_stages.research_loop.backend."
                        "experiment_scheduler.terminate_process_group"
                    ) as terminate,
                ):
                    second.start()
                terminate.assert_not_called()
                recovered = next(iter(second._active.values()))
                self.assertIsNotNone(recovered.allocation)
                self.assertIn(recovered.allocation.allocation_id, second_allocator.active)
                events = (second.state_dir / "events.jsonl").read_text(encoding="utf-8")
                self.assertIn('"recovery_reason": "protected_manifest"', events)

                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
                result = second.wait(recovered.job.job_id, 5)["job"]
                self.assertEqual(result["state"], "drained_unknown")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                second.stop()

    def test_event_and_manifest_recovery_share_one_host_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            registry = HostAllocationRegistry(root / "host-allocations.json")
            settings = _settings(maximum=1)
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=ResourceAllocator(
                    settings,
                    observer=_StaticObserver(),
                    registry=registry,
                ),
            )
            process = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                start_new_session=True,
            )
            try:
                submitted = first.submit(
                    {
                        "command": [sys.executable, "-c", "import time;time.sleep(30)"],
                        "peer_id": "gen1_peer0",
                        "generation_id": 1,
                        "experiment_id": "single-allocation",
                        "profile": "cpu",
                    }
                )
                submitted.attempts = 1
                submitted.state = "running"
                submitted.pid = process.pid
                submitted.pgid = process.pid
                allocation_id = f"{first.resource_owner_id}:{submitted.job_id}:1"
                allocation = first.allocator.recover_allocation(
                    allocation_id=allocation_id,
                    run_id=first.resource_owner_id,
                    pid=process.pid,
                    pgid=process.pid,
                    profile=first.settings.profiles["cpu"],
                    gpu_uuids=(),
                    require_admission=False,
                )
                self.assertIsNotNone(allocation)
                first._register_protected(submitted)
                first._append_event(
                    {
                        "event": "launched",
                        **first._event_identity(submitted, include_request=True),
                        "pid": process.pid,
                        "pgid": process.pid,
                        "gpu_uuids": [],
                        "allocation_id": allocation_id,
                    },
                    required=True,
                )

                second = ExperimentSchedulerService(
                    run_dir=run_dir,
                    settings=settings,
                    allocator=ResourceAllocator(
                        settings,
                        observer=_StaticObserver(),
                        registry=registry,
                    ),
                )
                second.start()
                with registry.locked() as rows:
                    allocations = [row for row in rows if row.get("record_type") == "allocation"]
                self.assertEqual(len(allocations), 1)
                self.assertEqual(allocations[0]["pgid"], process.pid)
                recovered_job_id = next(iter(second._active))

                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
                result = second.wait(recovered_job_id, 5)["job"]
                self.assertEqual(result["state"], "drained_unknown")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                if "second" in locals():
                    second.stop()

    def test_resume_restores_terminal_semantic_identity_without_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            marker = Path(td) / "runs"
            code = "import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.write_text(p.read_text()+'x' if p.exists() else 'x')"
            first_service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            first_service.start()
            try:
                self.assertEqual(
                    submit_and_wait(
                        [sys.executable, "-c", code, str(marker)],
                        peer_id="gen1_peer0",
                        experiment_id="terminal-science",
                        run_dir=run_dir,
                    ),
                    0,
                )
            finally:
                first_service.stop()
            second_service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            second_service.start()
            try:
                self.assertEqual(
                    submit_and_wait(
                        [sys.executable, "-c", "raise SystemExit(99)"],
                        peer_id="gen1_peer0",
                        experiment_id="terminal-science",
                        run_dir=run_dir,
                    ),
                    0,
                )
            finally:
                second_service.stop()
            self.assertEqual(marker.read_text(), "x")

    def test_idle_capacity_issues_only_directed_profile_aware_leases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            settings = _settings(maximum=4)
            settings.supply_idle_samples = 2
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=_SupplyAllocator((("cpu", "gpu"), ("cpu",), ("cpu",))),
            )
            service.open_generation(0, deadline=time.time() + 600)
            for peer_index in range(6):
                service.register_idle_supply(f"gen0_peer{peer_index}", 0)
            first = HostSnapshot(16, 5, 5, 0, observed_at=1.0)
            second = HostSnapshot(16, 5, 5, 0, observed_at=2.0)
            service._reconcile_supply(first)
            self.assertEqual(service.status()["resource_supply"]["leases"], [])
            service._reconcile_supply(second)

            supply = service.status()["resource_supply"]
            self.assertEqual(len(supply["leases"]), 3)
            self.assertEqual(supply["idle_waiters"], 3)
            self.assertEqual(supply["leases"][0]["generation_id"], 0)
            self.assertEqual(supply["leases"][0]["admissible_profiles"], ("cpu", "gpu"))
            self.assertNotIn("gpu_uuids", supply["leases"][0])
            paths = sorted((run_dir / "gen_0" / "resource_supply").glob("*.json"))
            self.assertEqual(len(paths), 3)
            self.assertEqual(
                {path.stem for path in paths},
                {"gen0_peer0", "gen0_peer1", "gen0_peer2"},
            )
            locator = json.loads(paths[0].read_text())
            self.assertNotIn("admissible_profiles", locator)
            canonical = service.get_supply_lease(
                locator["peer_id"], locator["generation_id"], locator["lease_id"]
            )
            self.assertIn("admissible_profiles", canonical)

    def test_supply_registration_rejects_noncanonical_or_oversized_peer_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(0, deadline=time.time() + 600)
            for peer_id in ("../peer", "gen1_peer0", "gen0_peer" + "9" * 200):
                self.assertEqual(service.register_idle_supply(peer_id, 0), {})
            self.assertEqual(service.status()["resource_supply"]["idle_waiters"], 0)

    def test_supply_cleanup_and_write_do_not_follow_symlink_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            external = Path(td) / "external"
            external.mkdir()
            victim = external / "victim.json"
            victim.write_text("keep")
            supply_dir = run_dir / "gen_0" / "resource_supply"
            supply_dir.parent.mkdir(parents=True)
            supply_dir.symlink_to(external, target_is_directory=True)
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 2
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service._remove_stale_supply_files()
            self.assertEqual(victim.read_text(), "keep")
            service.open_generation(0, deadline=time.time() + 600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            with self.assertRaises(OSError):
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            self.assertEqual(victim.read_text(), "keep")

    def test_supply_owner_identity_uses_absolute_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            first = ExperimentSchedulerService(
                run_dir=Path(td) / "task-a" / "run",
                settings=_settings(maximum=1),
            )
            second = ExperimentSchedulerService(
                run_dir=Path(td) / "task-b" / "run",
                settings=_settings(maximum=1),
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertNotEqual(first.supply_owner_id, second.supply_owner_id)

    def test_supply_lease_ids_are_unique_across_run_services(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 2
            services = [
                ExperimentSchedulerService(
                    run_dir=Path(td) / run_name,
                    settings=settings,
                    allocator=_SupplyAllocator((("cpu",),)),
                )
                for run_name in ("run-a", "run-b")
            ]
            lease_ids: list[str] = []
            for index, service in enumerate(services):
                service.open_generation(0, deadline=time.time() + 600)
                service.register_idle_supply(f"gen0_peer{index}", 0)
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
                lease_ids.append(service.status()["resource_supply"]["leases"][0]["lease_id"])
            self.assertEqual(len(set(lease_ids)), 2)

    def test_supply_lease_is_consumed_by_one_new_semantic_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 2
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(0, deadline=time.time() + 600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            lease = service.status()["resource_supply"]["leases"][0]
            path = run_dir / "gen_0" / "resource_supply" / "gen0_peer0.json"
            self.assertTrue(path.exists())

            service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "planned-follow-up",
                    "supply_lease_id": lease["lease_id"],
                }
            )
            self.assertEqual(service.status()["resource_supply"]["leases"], [])
            self.assertFalse(path.exists())
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["consumed"], 1)
            self.assertEqual(stats["conversion_rate"], 1.0)
            self.assertEqual(stats["by_priority"]["frontier_followup"]["conversion_rate"], 1.0)
            second = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "ordinary-second-follow-up",
                    "supply_lease_id": lease["lease_id"],
                }
            )
            self.assertEqual(second.supply_claim_id, "")
            self.assertEqual(service.status()["resource_supply"]["stats"]["reuse_ignored"], 1)
            events = [
                json.loads(line)
                for line in (service.state_dir / "events.jsonl").read_text().splitlines()
            ]
            supply_events = [event["event"] for event in events if "supply" in event["event"]]
            self.assertEqual(
                supply_events,
                ["supply_granted", "supply_consumed", "supply_reuse_ignored"],
            )

    def test_expired_lease_submission_is_stale_not_consumed_or_reused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 2
            service.open_generation(0, deadline=time.time() + 3600, cohort_size=1)
            service.configure_generation_maturity(
                0,
                cohort_size=1,
                mature_target=1,
                count_callback=lambda: 0,
            )
            service._refresh_mature_counts(HostSnapshot(8, 5, 5, 0, observed_at=0.5))
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            lease = service.status()["resource_supply"]["leases"][0]

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler.time.time",
                return_value=float(lease["expires_at"]) + 1.0,
            ):
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=3.0))

            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "late-existing-plan",
                    "work_class": "mature",
                    "supply_lease_id": lease["lease_id"],
                }
            )
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(job.supply_claim_id, "")
            self.assertEqual(stats["consumed"], 0)
            self.assertEqual(stats["reuse_ignored"], 0)
            self.assertEqual(stats["stale_submission"], 1)
            self.assertEqual(stats["expired"], 1)
            self.assertEqual(stats["by_priority"]["mature"]["conversion_rate"], 0.0)

    def test_expired_active_lease_submission_is_stale_and_expires_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 2
            service.open_generation(0, deadline=time.time() + 3600, cohort_size=1)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            lease = service.status()["resource_supply"]["leases"][0]

            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler.time.time",
                return_value=float(lease["expires_at"]) + 1.0,
            ):
                job = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "late-existing-plan",
                        "supply_lease_id": lease["lease_id"],
                    }
                )

            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(job.supply_claim_id, "")
            self.assertEqual(stats["consumed"], 0)
            self.assertEqual(stats["expired"], 1)
            self.assertEqual(stats["stale_submission"], 1)
            self.assertEqual(stats["reuse_ignored"], 0)
            self.assertEqual(stats["outstanding"], 0)
            self.assertEqual(stats["by_priority"]["frontier_followup"]["expired"], 1)
            self.assertEqual(
                stats["by_priority"]["frontier_followup"]["stale_submission"],
                1,
            )

    def test_release_after_response_window_expires_without_decline_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler.time.time",
                return_value=float(lease["expires_at"]) + 1.0,
            ):
                service.release_supply_lease(
                    lease["lease_id"],
                    "gen0_peer0",
                    declined=True,
                )
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["expired"], 1)
            self.assertEqual(stats["declined"], 0)
            self.assertEqual(service._declined_supply_peers, {})

    def test_supply_transition_event_failure_keeps_or_releases_lease_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            with (
                patch.object(service, "_append_event", side_effect=OSError("disk full")),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            self.assertEqual(service.status()["resource_supply"]["leases"], [])
            self.assertEqual(service.status()["resource_supply"]["stats"]["granted"], 0)

            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            lease = service.status()["resource_supply"]["leases"][0]
            with (
                patch.object(service, "_append_event", side_effect=OSError("disk full")),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                service.release_supply_lease(
                    lease["lease_id"],
                    "gen0_peer0",
                    declined=True,
                )
            self.assertEqual(len(service.status()["resource_supply"]["leases"]), 1)
            self.assertEqual(service.status()["resource_supply"]["stats"]["declined"], 0)
            service.release_supply_lease(lease["lease_id"], "gen0_peer0", declined=True)
            self.assertEqual(service.status()["resource_supply"]["stats"]["declined"], 1)

    def test_signal_and_terminal_audit_failure_release_unpublished_supply_claim(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            original_append = service._append_event
            original_release = allocator.release
            release_attempts = {"count": 0}

            def fail_supply_revoked(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_revoked":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            def fail_release_once(allocation_id: str) -> None:
                release_attempts["count"] += 1
                if release_attempts["count"] == 1:
                    raise OSError("registry unavailable")
                original_release(allocation_id)

            with (
                patch.object(service, "_append_event", side_effect=fail_supply_revoked),
                patch.object(
                    service,
                    "_write_supply_signal",
                    side_effect=OSError("signal unavailable"),
                ),
                patch.object(allocator, "release", side_effect=fail_release_once),
                self.assertRaisesRegex(OSError, "signal unavailable"),
            ):
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))

            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["granted"], 1)
            self.assertEqual(stats["revoked"], 0)
            self.assertEqual(stats["outstanding"], 1)
            self.assertTrue(service.status()["resource_supply"]["leases"][0]["release_pending"])
            self.assertIn(next(iter(allocator.claims)), service._pending_supply_releases)
            service._idle_supply_waiters.clear()
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["revoked"], 1)
            self.assertEqual(stats["outstanding"], 0)
            self.assertEqual(allocator.claims, {})

    def test_grant_audit_and_release_failure_remains_reconcilable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            original_append = service._append_event

            def fail_grant(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_granted":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with (
                patch.object(service, "_append_event", side_effect=fail_grant),
                patch.object(allocator, "release", side_effect=OSError("registry unavailable")),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            supply = service.status()["resource_supply"]
            self.assertEqual(len(supply["leases"]), 1)
            self.assertTrue(supply["leases"][0]["release_pending"])
            lease_id = supply["leases"][0]["lease_id"]
            self.assertIn(lease_id, allocator.claims)

            service._idle_supply_waiters.clear()
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            self.assertEqual(service.status()["resource_supply"]["leases"], [])
            self.assertNotIn(lease_id, allocator.claims)
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["granted"], 0)
            self.assertEqual(stats["revoked"], 0)
            events = [
                json.loads(line)
                for line in (service.state_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertFalse(
                any(
                    event["event"] in {"supply_granted", "supply_revoked"}
                    and event.get("lease_id") == lease_id
                    for event in events
                )
            )

    def test_supply_audit_failure_cannot_duplicate_an_accepted_semantic_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            original_append = service._append_event

            def fail_supply_consumed(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_consumed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            request = {
                "command": [sys.executable, "-c", "pass"],
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "experiment_id": "one-semantic-job",
                "supply_lease_id": lease["lease_id"],
            }
            with patch.object(service, "_append_event", side_effect=fail_supply_consumed):
                first = service.submit(request)
                duplicate = service.submit(request)
                second = service.submit(
                    {
                        **request,
                        "experiment_id": "different-semantic-job",
                    }
                )
            self.assertEqual(duplicate.job_id, first.job_id)
            self.assertEqual(service._queue, [first.job_id, second.job_id])
            self.assertEqual(first.supply_claim_id, lease["lease_id"])
            self.assertEqual(second.supply_claim_id, "")
            self.assertEqual(
                set(service._semantic_jobs.values()),
                {first.job_id, second.job_id},
            )
            service.release_supply_lease(
                lease["lease_id"],
                "gen0_peer0",
                declined=True,
            )
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["consumed"], 1)
            self.assertEqual(stats["declined"], 0)
            self.assertEqual(stats["revoked"], 0)
            self.assertEqual(stats["reuse_ignored"], 1)

            resumed = ExperimentSchedulerService(
                run_dir=service.run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            resumed._recover_terminal_events()
            self.assertEqual(len(resumed._queue), 2)
            self.assertEqual(len(resumed._semantic_jobs), 2)

    def test_terminal_job_retains_pending_consumption_until_audit_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600, cohort_size=2)
            service.configure_generation_maturity(
                0,
                cohort_size=2,
                mature_target=2,
                count_callback=lambda: 0,
                identity_callback=set,
                distinct_peer_commitments=True,
            )
            service._refresh_mature_counts(HostSnapshot(8, 5, 5, 0, observed_at=0.5))
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            original_append = service._append_event

            def fail_consumed(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_consumed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            request = {
                "command": [sys.executable, "-c", "pass"],
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "experiment_id": "terminal-pending-consumption",
                "supply_lease_id": lease["lease_id"],
                "work_class": "mature",
            }
            with patch.object(service, "_append_event", side_effect=fail_consumed):
                job = service.submit(request)
                job.state = "completed"
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            self.assertEqual(job.supply_claim_id, lease["lease_id"])
            self.assertEqual(
                service.get_supply_lease("gen0_peer0", 0, lease["lease_id"]),
                {},
            )
            self.assertIn(lease["lease_id"], allocator.claims)
            maturity = service.status()["resource_supply"]["maturity"]["0"]
            self.assertEqual(maturity["priority_leases"], 0)
            self.assertEqual(maturity["needed_inflight"], 2)

            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=3.0))
            self.assertEqual(job.supply_claim_id, "")
            self.assertNotIn(lease["lease_id"], allocator.claims)
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["consumed"], 1)
            self.assertEqual(stats["revoked"], 0)

    def test_freeze_consumes_accepted_pending_supply_before_queue_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            original_append = service._append_event

            def fail_supply_consumed(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_consumed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with patch.object(service, "_append_event", side_effect=fail_supply_consumed):
                job = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "accepted-before-freeze",
                        "supply_lease_id": lease["lease_id"],
                    }
                )
            with patch.object(
                service,
                "_unlink_supply_signal",
                side_effect=OSError("unlink unavailable"),
            ):
                service.freeze_generation(0, "test boundary")
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["consumed"], 1)
            self.assertEqual(stats["revoked"], 0)
            self.assertEqual(service._jobs[job.job_id].state, "rejected")
            self.assertEqual(service._supply_leases, {})
            self.assertNotIn(lease["lease_id"], allocator.claims)

    def test_rejection_completes_when_supply_audit_and_signal_cleanup_both_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            original_append = service._append_event

            def fail_supply_consumed(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_consumed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with patch.object(service, "_append_event", side_effect=fail_supply_consumed):
                job = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "rejected-after-double-cleanup-failure",
                        "supply_lease_id": lease["lease_id"],
                    }
                )
                with patch.object(
                    service,
                    "_unlink_supply_signal",
                    side_effect=OSError("unlink unavailable"),
                ):
                    service._queue.remove(job.job_id)
                    service._finish_without_launch(job, "test rejection")

            self.assertEqual(job.state, "rejected")
            self.assertEqual(job.supply_claim_id, "")
            self.assertEqual(service._supply_leases, {})
            self.assertNotIn(lease["lease_id"], allocator.claims)

    def test_unregister_cannot_revoke_supply_bound_to_an_accepted_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            original_append = service._append_event

            def fail_consumed_once(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_consumed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with patch.object(service, "_append_event", side_effect=fail_consumed_once):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "accepted-before-unregister",
                        "supply_lease_id": lease["lease_id"],
                    }
                )
            service.unregister_idle_supply("gen0_peer0", 0)
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["consumed"], 1)
            self.assertEqual(stats["revoked"], 0)

    def test_assessment_cannot_revoke_supply_bound_to_a_running_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            original_append = service._append_event

            def fail_consumed_once(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_consumed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with patch.object(service, "_append_event", side_effect=fail_consumed_once):
                job = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "accepted-before-assessment",
                        "supply_lease_id": lease["lease_id"],
                    }
                )
            service._queue.remove(job.job_id)
            job.state = "running"
            service.begin_assessment(0)
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["consumed"], 1)
            self.assertEqual(stats["revoked"], 0)
            self.assertEqual(job.supply_claim_id, "")

    def test_required_event_fsync_error_is_not_acknowledged_as_durable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.os.fsync",
                    side_effect=OSError("uncertain fsync"),
                ),
                self.assertRaisesRegex(OSError, "uncertain fsync"),
            ):
                service.release_supply_lease(lease["lease_id"], "gen0_peer0", declined=True)
            self.assertEqual(service.status()["resource_supply"]["stats"]["declined"], 0)
            self.assertEqual(len(service.status()["resource_supply"]["leases"]), 1)
            events = [
                json.loads(line)
                for line in (service.state_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertFalse(any(event["event"] == "supply_declined" for event in events))
            service.release_supply_lease(lease["lease_id"], "gen0_peer0", declined=True)
            self.assertEqual(service.status()["resource_supply"]["stats"]["declined"], 1)

    def test_failed_submission_fsync_cannot_recover_as_a_ghost_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            request = {
                "command": [sys.executable, "-c", "pass"],
                "peer_id": "gen0_peer0",
                "generation_id": 0,
                "experiment_id": "single-semantic-retry",
            }
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.os.fsync",
                    side_effect=(OSError("uncertain fsync"), None),
                ),
                self.assertRaisesRegex(OSError, "uncertain fsync"),
            ):
                service.submit(request)
            self.assertEqual(service._jobs, {})
            self.assertFalse(
                any(
                    json.loads(line).get("event") == "submitted"
                    for line in (service.state_dir / "events.jsonl").read_text().splitlines()
                )
            )

            accepted = service.submit(request)
            resumed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            resumed._recover_terminal_events()
            self.assertEqual(resumed._queue, [accepted.job_id])
            self.assertEqual(list(resumed._semantic_jobs.values()), [accepted.job_id])

    def test_partial_event_tail_is_removed_before_the_next_submission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            original_write = os.write
            write_attempts = {"count": 0}

            def partial_then_fail(fd: int, data: bytes | memoryview) -> int:
                write_attempts["count"] += 1
                if write_attempts["count"] == 1:
                    return original_write(fd, data[:12])
                raise OSError("write interrupted")

            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.os.write",
                    side_effect=partial_then_fail,
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.os.ftruncate",
                    side_effect=OSError("rollback unavailable"),
                ),
                self.assertRaisesRegex(OSError, "write interrupted"),
            ):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "partial-write",
                    }
                )

            accepted = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "accepted-after-partial-write",
                }
            )
            resumed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            resumed._recover_terminal_events()
            self.assertEqual(resumed._queue, [accepted.job_id])

    def test_unrecoverable_event_rollback_is_not_acknowledged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.os.fsync",
                    side_effect=OSError("fsync unavailable"),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.os.ftruncate",
                    side_effect=OSError("rollback unavailable"),
                ),
                self.assertRaisesRegex(OSError, "fsync unavailable"),
            ):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "uncertain-but-owned",
                    }
                )
            self.assertEqual(service._queue, [])
            self.assertEqual(service._jobs, {})

    def test_event_descriptor_close_failure_does_not_reject_a_durable_submission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            with patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "experiment_scheduler.os.close",
                side_effect=OSError("close unavailable"),
            ):
                accepted = service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "durable-before-close",
                    }
                )
            self.assertEqual(service._queue, [accepted.job_id])

    def test_retry_audit_failure_keeps_running_job_owned_until_retry_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "retry-remains-owned",
                }
            )
            service._queue.remove(job.job_id)
            job.attempts = 1
            job.state = "running"
            original_append = service._append_event

            def fail_retry(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "retry_queued":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with (
                patch.object(service, "_append_event", side_effect=fail_retry),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service._retry_or_finish(job, 75)
            self.assertEqual(job.state, "running")
            self.assertEqual(service._queue, [])
            self.assertEqual(service._semantic_jobs[job.semantic_key], job.job_id)

    def test_launch_failure_retry_audit_failure_restores_queued_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "launch-failure-retry-audit",
                }
            )
            original_append = service._append_event

            def fail_retry(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "retry_queued":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with (
                patch.object(service, "_append_event", side_effect=fail_retry),
                patch.object(Path, "mkdir", side_effect=OSError("attempt directory unavailable")),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service._launch(job)
            self.assertEqual(job.state, "queued")
            self.assertEqual(service._queue, [job.job_id])
            self.assertNotIn(job.job_id, service._active)

    def test_post_start_launch_failure_retry_audit_failure_restores_queue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "post-start-retry-audit",
                }
            )
            original_append = service._append_event

            def fail_retry(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "retry_queued":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with (
                patch.object(service, "_append_event", side_effect=fail_retry),
                patch.object(
                    service,
                    "_register_protected",
                    side_effect=OSError("manifest unavailable"),
                ),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service._launch(job)
            self.assertEqual(job.state, "queued")
            self.assertEqual(service._queue, [job.job_id])
            self.assertNotIn(job.job_id, service._active)

    def test_assessment_rechecks_ordinary_queue_after_rejection_audit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            service.open_generation(0, deadline=time.time() + 600)
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "ordinary-before-assessment",
                    "work_class": "ordinary",
                }
            )
            original_append = service._append_event

            def fail_completed(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "completed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with (
                patch.object(service, "_append_event", side_effect=fail_completed),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service.begin_assessment(0)
            self.assertEqual(job.state, "queued")
            self.assertEqual(service._queue, [job.job_id])
            with (
                patch.object(service, "_append_event", side_effect=fail_completed),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service._launch_ready_jobs()
            self.assertEqual(job.state, "queued")
            self.assertNotIn(job.job_id, service._active)

    def test_terminal_audit_failure_does_not_expose_undurable_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.infrastructure_retries = 0
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "notify-terminal-waiter",
                }
            )
            service._queue.remove(job.job_id)
            job.state = "running"
            result: dict[str, object] = {}
            waiter = threading.Thread(
                target=lambda: result.update(service.wait(job.job_id, 0.1)),
                daemon=True,
            )
            waiter.start()
            time.sleep(0.02)
            with (
                patch.object(service, "_append_event", side_effect=OSError("audit unavailable")),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service._retry_or_finish(job, 1)
            waiter.join(timeout=0.5)
            self.assertFalse(waiter.is_alive())
            self.assertTrue(result["timeout"])
            self.assertEqual(result["job"]["state"], "running")  # type: ignore[index]

    def test_queued_rejection_audit_failure_remains_queued_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "cancel-remains-observable",
                }
            )
            result: dict[str, object] = {}
            waiter = threading.Thread(
                target=lambda: result.update(service.wait(job.job_id, 0.1)),
                daemon=True,
            )
            waiter.start()
            time.sleep(0.02)
            original_append = service._append_event

            def fail_completed(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "completed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with (
                patch.object(service, "_append_event", side_effect=fail_completed),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service.cancel_queued(job.job_id)
            waiter.join(timeout=0.5)
            self.assertFalse(waiter.is_alive())
            self.assertTrue(result["timeout"])
            self.assertEqual(job.state, "queued")
            self.assertEqual(result["job"]["state"], "queued")  # type: ignore[index]
            self.assertEqual(service._queue, [job.job_id])

            resumed = ExperimentSchedulerService(
                run_dir=service.run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            resumed._recover_terminal_events()
            self.assertEqual(resumed._queue, [job.job_id])
            self.assertEqual(resumed._jobs[job.job_id].state, "queued")

    def test_terminal_supply_release_failure_remains_visible_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            original_release = allocator.release
            attempts = {"count": 0}

            def fail_once(allocation_id: str) -> None:
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise OSError("registry unavailable")
                original_release(allocation_id)

            with patch.object(allocator, "release", side_effect=fail_once):
                service.release_supply_lease(
                    lease["lease_id"],
                    "gen0_peer0",
                    declined=True,
                )
                first = service.status()["resource_supply"]
                self.assertEqual(first["stats"]["declined"], 0)
                self.assertEqual(first["stats"]["outstanding"], 1)
                self.assertTrue(first["leases"][0]["release_pending"])
                self.assertIn(lease["lease_id"], allocator.claims)
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))

            final = service.status()["resource_supply"]
            self.assertEqual(final["stats"]["declined"], 1)
            self.assertEqual(final["stats"]["outstanding"], 0)
            self.assertNotIn(lease["lease_id"], allocator.claims)

    def test_rejected_job_supply_claim_release_retries_during_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "terminal-claim-retry",
                    "supply_lease_id": lease["lease_id"],
                }
            )
            with patch.object(allocator, "release", side_effect=OSError("registry unavailable")):
                service._queue.remove(job.job_id)
                service._finish_without_launch(job, "task_contract_rejected")
            self.assertEqual(job.state, "rejected")
            self.assertTrue(job.public()["supply_release_pending"])
            self.assertIn(lease["lease_id"], allocator.claims)
            with self.assertRaisesRegex(ExperimentRejected, "resource release is still pending"):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "terminal-claim-retry",
                        "retry_terminal": True,
                    }
                )

            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            self.assertFalse(job.public()["supply_release_pending"])
            self.assertNotIn(lease["lease_id"], allocator.claims)
            retried = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "terminal-claim-retry",
                    "retry_terminal": True,
                }
            )
            self.assertEqual(retried.job_id, job.job_id)
            self.assertEqual(retried.state, "queued")

    def test_signal_unlink_failure_cannot_strand_terminal_supply_claim(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            with patch.object(
                service,
                "_unlink_supply_signal",
                side_effect=OSError("unlink unavailable"),
            ):
                service.release_supply_lease(
                    lease["lease_id"],
                    "gen0_peer0",
                    declined=True,
                )
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["declined"], 1)
            self.assertEqual(stats["outstanding"], 0)
            self.assertNotIn(lease["lease_id"], allocator.claims)
            self.assertIn(("gen0_peer0", 0, "frontier_followup"), service._declined_supply_peers)

    def test_stop_continues_when_supply_revocation_audit_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",), ("cpu",)))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=2),
                allocator=allocator,
            )
            service._acquire_owner_lock()
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer1",
                    "generation_id": 0,
                    "experiment_id": "queued-before-stop",
                }
            )
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            original_append = service._append_event

            def fail_supply_revoke(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_revoked":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with (
                patch.object(service, "_append_event", side_effect=fail_supply_revoke),
                patch.object(
                    service,
                    "_unlink_supply_signal",
                    side_effect=OSError("unlink unavailable"),
                ),
            ):
                service.stop()
                service._launch_ready_jobs()
            self.assertEqual(service._queue, [])
            self.assertEqual(service._supply_leases, {})
            self.assertEqual(service.status()["running"], 0)
            self.assertEqual(service._jobs[job.job_id].state, "rejected")
            self.assertEqual(allocator.claims, {})

    def test_stop_closes_transport_even_when_queued_terminal_audit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_BlockedAllocator(""),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "stop-audit-failure",
                }
            )
            service.start()
            original_append = service._append_event

            def fail_completed(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "completed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with (
                patch.object(service, "_append_event", side_effect=fail_completed),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service.stop()
            self.assertFalse(service.endpoint.exists())
            self.assertFalse((service.state_dir / "endpoint.json").exists())
            self.assertNotIn("PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT", os.environ)
            self.assertEqual(job.state, "queued")

            resumed = ExperimentSchedulerService(
                run_dir=service.run_dir,
                settings=_settings(maximum=1),
                allocator=_BlockedAllocator(""),
            )
            resumed._recover_terminal_events()
            self.assertEqual(resumed._queue, [])
            self.assertEqual(resumed._jobs[job.job_id].state, "rejected")
            resumed.open_generation(0, deadline=time.time() + 600)
            replacement = resumed.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "replacement-after-explicit-resume",
                }
            )
            self.assertEqual(resumed._jobs[job.job_id].state, "rejected")
            self.assertEqual(resumed._queue, [replacement.job_id])

    def test_stop_fence_audit_failure_retains_owner_until_retry_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_BlockedAllocator(""),
            )
            service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "queued-before-fence-failure",
                }
            )
            service.start()
            original_append = service._append_event

            failures = {"remaining": 2}

            def fail_stop_fence(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "admission_closed" and failures["remaining"]:
                    failures["remaining"] -= 1
                    raise OSError("fence unavailable")
                original_append(payload, required=required)

            with patch.object(service, "_append_event", side_effect=fail_stop_fence):
                with self.assertRaisesRegex(OSError, "fence unavailable"):
                    service.stop()
                deadline = time.time() + 3
                while service._owner_lock_fd is not None and time.time() < deadline:
                    time.sleep(0.05)

            self.assertFalse(service.endpoint.exists())
            self.assertFalse((service.state_dir / "endpoint.json").exists())
            self.assertNotIn("PRAXIST_EXPERIMENT_SCHEDULER_ENDPOINT", os.environ)
            self.assertIsNone(service._owner_lock_fd)
            service._worker_thread.join(timeout=3)
            self.assertFalse(service._worker_thread.is_alive())
            events = [
                json.loads(line)
                for line in (service.state_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any(event["event"] == "admission_closed" for event in events))

            successor = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_BlockedAllocator(""),
            )
            successor.start()
            successor.stop()

    def test_stop_retries_pending_host_claim_before_releasing_owner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.start()
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease_id = service.status()["resource_supply"]["leases"][0]["lease_id"]
            original_release = allocator.release
            release_calls = {"count": 0}

            def fail_release_once(allocation_id: str) -> None:
                release_calls["count"] += 1
                if release_calls["count"] == 1:
                    raise OSError("registry unavailable")
                original_release(allocation_id)

            clear_calls = {"count": 0}

            def fail_clear_once(owner_id: str) -> None:
                clear_calls["count"] += 1
                if clear_calls["count"] == 1:
                    raise OSError("registry unavailable")
                allocator.claims.clear()

            with (
                patch.object(allocator, "release", side_effect=fail_release_once),
                patch.object(
                    allocator,
                    "clear_supply",
                    side_effect=fail_clear_once,
                    create=True,
                ),
            ):
                service.stop()

            self.assertGreaterEqual(release_calls["count"], 2)
            self.assertNotIn(lease_id, allocator.claims)
            self.assertIsNone(service._owner_lock_fd)

    def test_stop_releases_queued_job_claim_after_terminal_audit_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            allocator.active.add("block-launch")
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service.start()
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            lease = service.status()["resource_supply"]["leases"][0]
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "queued-with-consumed-claim",
                    "supply_lease_id": lease["lease_id"],
                }
            )
            self.assertEqual(job.supply_claim_id, lease["lease_id"])
            original_append = service._append_event
            completed_failures = {"remaining": 1}

            def fail_completed_once(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "completed" and completed_failures["remaining"]:
                    completed_failures["remaining"] -= 1
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            clear_failures = {"remaining": 1}

            def fail_clear_once(owner_id: str) -> None:
                del owner_id
                if clear_failures["remaining"]:
                    clear_failures["remaining"] -= 1
                    raise OSError("registry unavailable")
                allocator.claims.clear()

            with (
                patch.object(service, "_append_event", side_effect=fail_completed_once),
                patch.object(
                    allocator,
                    "clear_supply",
                    side_effect=fail_clear_once,
                    create=True,
                ),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service.stop()

            deadline = time.time() + 3
            while service._owner_lock_fd is not None and time.time() < deadline:
                time.sleep(0.05)
            self.assertNotIn(lease["lease_id"], allocator.claims)
            self.assertEqual(job.supply_claim_id, "")
            self.assertIsNone(service._owner_lock_fd)

    def test_rpc_returns_structured_error_for_malformed_request(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_BlockedAllocator(""),
            )
            service.start()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(str(service.endpoint))
                    client.sendall(b"{not-json}\n")
                    response = json.loads(client.makefile("rb").readline())
                self.assertFalse(response["ok"])
                self.assertIn("JSONDecodeError", response["error"])
            finally:
                service.stop()

    def test_supply_lease_expiry_release_and_pending_submission_edges(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",), ("cpu",), ("cpu",)))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=3),
                allocator=allocator,
            )
            service.open_generation(0, deadline=time.time() + 3600)
            now = time.time()

            expired_reregister = _SupplyLease(
                "expired-reregister",
                "gen0_peer0",
                0,
                ("cpu",),
                "frontier_followup",
                now - 20,
                now - 10,
            )
            service._supply_leases[expired_reregister.lease_id] = expired_reregister
            allocator.claims[expired_reregister.lease_id] = ("cpu",)
            self.assertEqual(service.register_idle_supply("gen0_peer0", 0), {})
            self.assertNotIn(expired_reregister.lease_id, allocator.claims)

            expired_fetch = _SupplyLease(
                "expired-fetch",
                "gen0_peer1",
                0,
                ("cpu",),
                "frontier_followup",
                now - 20,
                now - 10,
            )
            service._supply_leases[expired_fetch.lease_id] = expired_fetch
            allocator.claims[expired_fetch.lease_id] = ("cpu",)
            self.assertEqual(service.get_supply_lease("gen0_peer1", 0, expired_fetch.lease_id), {})
            self.assertNotIn(expired_fetch.lease_id, allocator.claims)

            released = _SupplyLease(
                "ordinary-release",
                "gen0_peer2",
                0,
                ("cpu",),
                "frontier_followup",
                now,
                now + 600,
            )
            service._supply_leases[released.lease_id] = released
            allocator.claims[released.lease_id] = ("cpu",)
            service.release_supply_lease(released.lease_id, "gen0_peer2")
            self.assertNotIn(released.lease_id, allocator.claims)

            service._pending_supply_releases["pending-release"] = (
                "revoked",
                "cleanup_pending",
                None,
                True,
            )
            submitted = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer3",
                    "generation_id": 0,
                    "experiment_id": "stale-pending-lease",
                    "supply_lease_id": "pending-release",
                }
            )
            self.assertEqual(submitted.supply_claim_id, "")
            self.assertEqual(service.status()["resource_supply"]["stats"]["stale_submission"], 1)

    def test_stop_compacts_all_pending_supply_release_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",), ("cpu",), ("cpu",)))
            allocator.clear_supply = MagicMock(side_effect=lambda _owner: allocator.claims.clear())
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=3),
                allocator=allocator,
            )
            now = time.time()
            leases = {
                lease_id: _SupplyLease(
                    lease_id,
                    f"gen0_peer{index}",
                    0,
                    ("cpu",),
                    "frontier_followup",
                    now,
                    now + 600,
                )
                for index, lease_id in enumerate(("grant-failed", "audit-pending", "recorded"))
            }
            service._supply_leases.update(leases)
            allocator.claims.update({lease_id: ("cpu",) for lease_id in leases})
            service._pending_supply_releases.update(
                {
                    "missing": ("revoked", "cleanup", None, True),
                    "grant-failed": ("revoked", "grant_audit_failed", None, False),
                    "audit-pending": (
                        "revoked",
                        "cleanup",
                        {"cleanup_source": "test"},
                        False,
                    ),
                    "recorded": ("revoked", "cleanup", None, True),
                }
            )
            service._acquire_owner_lock()
            with patch.object(service, "_revoke_supply_locked"):
                service.stop()
            self.assertEqual(service._pending_supply_releases, {})
            self.assertEqual(service._supply_leases, {})
            self.assertEqual(allocator.claims, {})
            events = [
                json.loads(line)
                for line in (service.state_dir / "events.jsonl").read_text().splitlines()
            ]
            audit_event = next(
                event
                for event in events
                if event.get("event") == "supply_revoked"
                and event.get("lease_id") == "audit-pending"
            )
            self.assertEqual(audit_event["cleanup_source"], "test")

    def test_stop_clears_host_claim_when_supply_terminal_cleanup_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _SupplyAllocator((("cpu",),))
            allocator.clear_supply = MagicMock(side_effect=lambda _owner: allocator.claims.clear())
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service._acquire_owner_lock()
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            original_append = service._append_event

            def fail_revoked(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_revoked":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with (
                patch.object(service, "_append_event", side_effect=fail_revoked),
                patch.object(
                    service,
                    "_unlink_supply_signal",
                    side_effect=OSError("unlink unavailable"),
                ),
                patch.object(allocator, "release", side_effect=OSError("registry unavailable")),
            ):
                service.stop()

            self.assertEqual(allocator.claims, {})
            allocator.clear_supply.assert_called_once_with(service.supply_owner_id)
            pending = service.status()["resource_supply"]["leases"]
            self.assertEqual(len(pending), 1)
            self.assertTrue(pending[0]["release_pending"])

    def test_supply_ledger_recovers_stats_tombstone_and_dangling_grant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            first = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            events = [
                {
                    "event": "supply_granted",
                    "lease_id": "consumed-lease",
                    "priority": "mature",
                },
                {
                    "event": "supply_consumed",
                    "lease_id": "consumed-lease",
                    "priority": "mature",
                },
                {
                    "event": "supply_reuse_ignored",
                    "lease_id": "consumed-lease",
                },
                {
                    "event": "supply_granted",
                    "lease_id": "dangling-lease",
                    "priority": "frontier_followup",
                },
                {
                    "event": "supply_granted",
                    "lease_id": "accepted-before-terminal",
                    "priority": "mature",
                    "recorded_at": 40.0,
                },
                {
                    "event": "submitted",
                    "recorded_at": 90.0,
                    "environment_values": {
                        "PRAXIST_RESOURCE_SUPPLY_LEASE_ID": "accepted-before-terminal"
                    },
                },
            ]
            (first.state_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            first._recover_terminal_events()
            stats = first.status()["resource_supply"]["stats"]
            self.assertEqual(stats["granted"], 3)
            self.assertEqual(stats["consumed"], 2)
            self.assertEqual(stats["revoked"], 1)
            self.assertEqual(stats["reuse_ignored"], 1)
            self.assertIn("consumed-lease", first._consumed_supply_leases)
            self.assertIn("accepted-before-terminal", first._consumed_supply_leases)
            first.open_generation(1, deadline=time.time() + 600)
            self.assertIn("consumed-lease", first._consumed_supply_leases)

            resumed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            resumed._recover_terminal_events()
            self.assertEqual(resumed.status()["resource_supply"]["stats"], stats)

    def test_supply_ledger_recovers_explicit_terminal_retry_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            now = time.time()
            service._recover_supply_events(
                [
                    {
                        "event": "supply_granted",
                        "lease_id": "retry-accepted-lease",
                        "priority": "ordinary",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "admissible_profiles": ["cpu"],
                        "issued_at": now - 10,
                        "expires_at": now + 60,
                    },
                    {
                        "event": "retry_queued",
                        "retry_reason": "explicit_terminal_retry",
                        "supply_lease_id": "retry-accepted-lease",
                        "supply_lease_eligible": True,
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "profile": "cpu",
                        "work_class": "ordinary",
                        "recorded_at": now,
                    },
                ]
            )

            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["granted"], 1)
            self.assertEqual(stats["consumed"], 1)
            self.assertEqual(stats["revoked"], 0)
            self.assertIn("retry-accepted-lease", service._consumed_supply_leases)

    def test_restart_does_not_consume_an_ineligible_mature_supply_submission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600, cohort_size=1)
            service.configure_generation_maturity(
                0,
                cohort_size=1,
                mature_target=1,
                count_callback=lambda: 0,
            )
            service.register_idle_supply("gen0_peer0", 0)
            snapshot = HostSnapshot(8, 5, 5, 0, observed_at=1.0)
            service._refresh_mature_counts(snapshot)
            service._reconcile_supply(snapshot)
            lease = service.status()["resource_supply"]["leases"][0]
            self.assertEqual(lease["priority"], "mature")
            original_append = service._append_event

            def fail_supply_revoked(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_revoked":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with patch.object(service, "_append_event", side_effect=fail_supply_revoked):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "ordinary-work-on-mature-supply",
                        "work_class": "ordinary",
                        "supply_lease_id": lease["lease_id"],
                        "environment": {"PRAXIST_RESOURCE_SUPPLY_LEASE_ID": lease["lease_id"]},
                    }
                )
            pending = service.status()["resource_supply"]["leases"]
            self.assertEqual(len(pending), 1)
            self.assertTrue(pending[0]["release_pending"])
            self.assertEqual(
                service.get_supply_lease("gen0_peer0", 0, lease["lease_id"]),
                {},
            )

            resumed = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            resumed._recover_terminal_events()
            stats = resumed.status()["resource_supply"]["stats"]
            self.assertEqual(stats["consumed"], 0)
            self.assertEqual(stats["revoked"], 1)
            self.assertEqual(stats["conversion_rate"], 0.0)

    def test_pending_consumption_counts_mature_job_and_lease_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=3),
                allocator=_SupplyAllocator((("cpu",), ("cpu",), ("cpu",))),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600, cohort_size=3)
            service.configure_generation_maturity(
                0,
                cohort_size=3,
                mature_target=3,
                count_callback=lambda: 0,
            )
            service.register_idle_supply("gen0_peer0", 0)
            snapshot = HostSnapshot(8, 5, 5, 0, observed_at=1.0)
            service._refresh_mature_counts(snapshot)
            service._reconcile_supply(snapshot)
            lease = service.status()["resource_supply"]["leases"][0]
            original_append = service._append_event

            def fail_supply_consumed(payload: dict[str, object], *, required: bool = False) -> None:
                if payload.get("event") == "supply_consumed":
                    raise OSError("audit unavailable")
                original_append(payload, required=required)

            with patch.object(service, "_append_event", side_effect=fail_supply_consumed):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "mature-pending-consumption",
                        "work_class": "mature",
                        "supply_lease_id": lease["lease_id"],
                    }
                )
            maturity = service.status()["resource_supply"]["maturity"]["0"]
            self.assertEqual(maturity["inflight_jobs"], 1)
            self.assertEqual(maturity["priority_leases"], 0)
            self.assertEqual(maturity["needed_inflight"], 2)

    def test_reused_consumed_lease_keeps_its_priority_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600, cohort_size=1)
            service.configure_generation_maturity(
                0,
                cohort_size=1,
                mature_target=1,
                count_callback=lambda: 0,
            )
            service.register_idle_supply("gen0_peer0", 0)
            snapshot = HostSnapshot(8, 5, 5, 0, observed_at=1.0)
            service._refresh_mature_counts(snapshot)
            service._reconcile_supply(snapshot)
            lease = service.status()["resource_supply"]["leases"][0]
            service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "consume-mature-lease",
                    "work_class": "mature",
                    "supply_lease_id": lease["lease_id"],
                }
            )
            service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "reuse-mature-lease",
                    "work_class": "mature",
                    "supply_lease_id": lease["lease_id"],
                }
            )
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["reuse_ignored"], 1)
            self.assertEqual(stats["by_priority"]["mature"]["reuse_ignored"], 1)

    def test_restart_classifies_elapsed_unanswered_supply_as_expired(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service._append_event(
                {
                    "event": "supply_granted",
                    "lease_id": "elapsed-before-restart",
                    "priority": "mature",
                    "expires_at": time.time() - 400,
                },
                required=True,
            )
            service._recover_terminal_events()
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["expired"], 1)
            self.assertEqual(stats["revoked"], 0)

    def test_live_process_owns_semantic_identity_over_older_terminal_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            common = {
                "generation_id": 0,
                "peer_id": "gen0_peer0",
                "experiment_id": "same-semantic-experiment",
                "profile": "cpu",
                "work_class": "ordinary",
                "command": [sys.executable, "-c", "pass"],
                "environment_values": {},
                "attempt": 0,
            }
            events = [
                {"event": "submitted", "job_id": "old-job", **common},
                {
                    "event": "completed",
                    "job_id": "old-job",
                    **common,
                    "state": "completed",
                    "exit_code": 0,
                    "recorded_at": 1.0,
                },
                {"event": "submitted", "job_id": "live-job", **common},
                {
                    "event": "launched",
                    "job_id": "live-job",
                    **common,
                    "pid": 123,
                    "pgid": 123,
                    "allocation_id": "live-allocation",
                    "recorded_at": 2.0,
                },
            ]
            (service.state_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with (
                patch.object(service, "_event_process_matches", return_value=True),
                patch.object(service, "_register_protected"),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.process_group_alive",
                    return_value=True,
                ),
            ):
                service._recover_terminal_events()
            semantic = next(iter(service._semantic_jobs))
            self.assertEqual(service._semantic_jobs[semantic], "live-job")
            self.assertIn("live-job", service._active)
            self.assertNotIn("old-job", service._jobs)

    def test_semantic_duplicate_does_not_consume_supply_lease(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            settings = _settings(maximum=2)
            settings.supply_idle_samples = 2
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=_SupplyAllocator((("cpu",), ("cpu",))),
            )
            original = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "existing-work",
                }
            )
            service.open_generation(0, deadline=time.time() + 600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            lease = service.status()["resource_supply"]["leases"][0]
            duplicate = service.submit(
                {
                    "command": [sys.executable, "-c", "raise SystemExit(99)"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "existing-work",
                    "supply_lease_id": lease["lease_id"],
                }
            )
            self.assertEqual(duplicate.job_id, original.job_id)
            self.assertEqual(len(service.status()["resource_supply"]["leases"]), 1)

    def test_closing_revokes_supply_without_disturbing_running_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 2
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(4, deadline=time.time() + 600)
            service.register_idle_supply("gen4_peer0", 4)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            self.assertEqual(len(service.status()["resource_supply"]["leases"]), 1)
            service.freeze_generation(4, "generation_closing")
            supply = service.status()["resource_supply"]
            self.assertEqual(supply["leases"], [])
            self.assertEqual(supply["idle_waiters"], 0)
            self.assertFalse((run_dir / "gen_4" / "resource_supply" / "gen4_peer0.json").exists())

    def test_invalid_host_claim_withdraws_published_supply_lease(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 2
            allocator = _SupplyAllocator((("cpu",),))
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=allocator,
            )
            service.open_generation(1, deadline=time.time() + 600)
            service.register_idle_supply("gen1_peer0", 1)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            lease = service.status()["resource_supply"]["leases"][0]
            path = run_dir / "gen_1" / "resource_supply" / "gen1_peer0.json"
            allocator.claims.pop(lease["lease_id"])
            service._reconcile_supply(HostSnapshot(8, 95, 5, 0, observed_at=3.0))
            supply = service.status()["resource_supply"]
            self.assertEqual(supply["leases"], [])
            self.assertEqual(supply["stats"]["revoked"], 1)
            self.assertFalse(path.exists())

    def test_pressure_resample_keeps_supply_lease_until_submission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 2
            allocator = ResourceAllocator(
                settings,
                observer=_StaticObserver(),
                registry=HostAllocationRegistry(Path(td) / "registry"),
            )
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=allocator,
            )
            service.open_generation(1, deadline=time.time() + 600)
            service.register_idle_supply("gen1_peer0", 1)
            low = HostSnapshot(8, 5, 5, 0, observed_at=1.0)
            service._reconcile_supply(low)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            lease = service.status()["resource_supply"]["leases"][0]

            allocator._last_snapshot = HostSnapshot(8, 99, 5, 0, observed_at=3.0)
            service._reconcile_supply(allocator.snapshot)
            self.assertEqual(
                service.get_supply_lease("gen1_peer0", 1, lease["lease_id"])["lease_id"],
                lease["lease_id"],
            )

            service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "experiment_id": "planned-followup",
                    "generation_id": 1,
                    "peer_id": "gen1_peer0",
                    "profile": "cpu",
                    "supply_lease_id": lease["lease_id"],
                }
            )
            supply = service.status()["resource_supply"]
            self.assertEqual(supply["stats"]["consumed"], 1)
            self.assertEqual(supply["leases"], [])

    def test_unproductive_peer_unregisters_waiter_and_published_lease(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 2
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(2, deadline=time.time() + 600)
            service.register_idle_supply("gen2_peer0", 2)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            lease_id = service.status()["resource_supply"]["leases"][0]["lease_id"]
            service.unregister_idle_supply("gen2_peer0", 2)
            supply = service.status()["resource_supply"]
            self.assertEqual(supply["leases"], [])
            self.assertEqual(supply["stats"]["consumed"], 0)
            self.assertEqual(supply["stats"]["revoked"], 1)
            self.assertNotIn(lease_id, service._consumed_supply_leases)
            service.open_generation(3, deadline=time.time() + 600)
            self.assertEqual(service._consumed_supply_leases, {})

    def test_declined_supply_can_later_receive_changed_mature_priority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 2
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=settings,
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(0, deadline=time.time() + 600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            lease_id = service.status()["resource_supply"]["leases"][0]["lease_id"]
            service.release_supply_lease(lease_id, "gen0_peer0", declined=True)
            self.assertEqual(service.status()["resource_supply"]["stats"]["declined"], 1)
            service.register_idle_supply("gen0_peer0", 0)
            self.assertEqual(service.status()["resource_supply"]["idle_waiters"], 1)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=3.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=4.0))
            self.assertEqual(service.status()["resource_supply"]["leases"], [])

            service._assessment_generations.add(0)
            service._generation_mature_targets[0] = 1
            service._generation_cohort_sizes[0] = 1
            service._mature_completed[0] = 0
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=5.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=6.0))
            leases = service.status()["resource_supply"]["leases"]
            self.assertEqual(len(leases), 1)
            self.assertEqual(leases[0]["priority"], "mature")
            service.open_generation(1, deadline=time.time() + 600)
            service.register_idle_supply("gen1_peer0", 1)
            self.assertEqual(service.status()["resource_supply"]["idle_waiters"], 1)

    def test_repeated_same_priority_declines_back_off_and_submission_resets_them(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.settings.supply_idle_samples = 1
            service.open_generation(0, deadline=time.time() + 3600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            first = service.status()["resource_supply"]["leases"][0]
            service.release_supply_lease(first["lease_id"], "gen0_peer0", declined=True)
            key = ("gen0_peer0", 0, "frontier_followup")
            first_expiry, first_count = service._declined_supply_peers[key]
            self.assertEqual(first_count, 1)

            service._declined_supply_peers[key] = (0.0, first_count)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            second = service.status()["resource_supply"]["leases"][0]
            service.release_supply_lease(second["lease_id"], "gen0_peer0", declined=True)
            second_expiry, second_count = service._declined_supply_peers[key]
            self.assertEqual(second_count, 2)
            self.assertGreater(second_expiry - time.time(), first_expiry - time.time())

            service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "newly-justified-plan",
                }
            )
            self.assertNotIn(key, service._declined_supply_peers)
            stats = service.status()["resource_supply"]["stats"]
            self.assertEqual(stats["declined"], 2)
            self.assertEqual(stats["outstanding"], 0)
            events = [
                json.loads(line)
                for line in (service.state_dir / "events.jsonl").read_text().splitlines()
            ]
            declines = [event for event in events if event["event"] == "supply_declined"]
            self.assertEqual([event["cooldown_seconds"] for event in declines], [60.0, 120.0])

    def test_failed_submission_does_not_reset_decline_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            key = ("gen0_peer0", 0, "frontier_followup")
            expected = (time.time() + 300, 3)
            service._declined_supply_peers[key] = expected
            with (
                patch.object(service, "_append_event", side_effect=OSError("audit unavailable")),
                self.assertRaisesRegex(OSError, "audit unavailable"),
            ):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "submission-not-durable",
                    }
                )
            self.assertEqual(service._declined_supply_peers[key], expected)

    def test_scheduler_start_removes_stale_supply_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            stale = run_dir / "gen_2" / "resource_supply" / "gen2_peer1.json"
            stale.parent.mkdir(parents=True)
            stale.write_text("{}")
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_GPUAllocator(""),
            )
            service.start()
            try:
                self.assertFalse(stale.exists())
            finally:
                service.stop()

    def test_scheduler_stop_materializes_empty_supply_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 2
            service = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=settings,
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service._acquire_owner_lock()
            service.open_generation(0, deadline=time.time() + 600)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0, observed_at=2.0))
            service._write_snapshot(force=True)
            self.assertEqual(
                len(
                    json.loads((service.state_dir / "status.json").read_text())["resource_supply"][
                        "leases"
                    ]
                ),
                1,
            )
            service.stop()
            final_status = json.loads((service.state_dir / "status.json").read_text())
            self.assertEqual(final_status["resource_supply"]["leases"], [])
            self.assertEqual(final_status["resource_supply"]["idle_waiters"], 0)

    def test_mature_debt_issues_three_bounded_priority_leases_per_missing_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=12)
            settings.supply_idle_samples = 2
            allocator = _SupplyAllocator(tuple(("cpu",) for _ in range(12)))
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=settings,
                allocator=allocator,
            )
            service.open_generation(0, deadline=time.time() + 600, cohort_size=12)
            completed = {"count": 0}
            service.configure_generation_maturity(
                0,
                cohort_size=12,
                mature_target=3,
                count_callback=lambda: completed["count"],
            )
            for index in range(12):
                service.register_idle_supply(f"gen0_peer{index}", 0)
            first = HostSnapshot(16, 5, 5, 0, observed_at=1.0)
            second = HostSnapshot(16, 5, 5, 0, observed_at=2.0)
            service._refresh_mature_counts(first)
            service._reconcile_supply(first)
            service._refresh_mature_counts(second)
            service._reconcile_supply(second)
            supply = service.status()["resource_supply"]
            self.assertEqual(len(supply["leases"]), 12)
            self.assertEqual(
                sum(lease["priority"] == "mature" for lease in supply["leases"]),
                9,
            )
            self.assertEqual(
                sum(lease["priority"] == "frontier_followup" for lease in supply["leases"]),
                3,
            )
            self.assertEqual(supply["maturity"]["0"]["debt"], 3)
            self.assertEqual(supply["maturity"]["0"]["target_inflight"], 9)

            completed["count"] = 3
            third = HostSnapshot(16, 5, 5, 0, observed_at=3.0)
            service._refresh_mature_counts(third)
            service._reconcile_supply(third)
            remaining = service.status()["resource_supply"]["leases"]
            self.assertEqual(len(remaining), 3)
            self.assertEqual({lease["priority"] for lease in remaining}, {"frontier_followup"})

    def test_first_wave_preserves_exploration_and_assigns_direct_mature_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=12),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(4, deadline=time.time() + 600, cohort_size=12)
            service.configure_generation_maturity(
                4,
                cohort_size=12,
                mature_target=3,
                count_callback=lambda: 0,
            )
            self.assertEqual(
                service.generation_advice("gen4_peer0", 4)["first_wave"], "direct_mature"
            )
            self.assertEqual(
                service.generation_advice("gen4_peer2", 4)["first_wave"], "direct_mature"
            )
            self.assertEqual(service.generation_advice("gen4_peer3", 4)["first_wave"], "explore")

    def test_first_wave_ranks_surviving_sparse_peer_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=3),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(
                4,
                deadline=time.time() + 600,
                cohort_size=3,
                peer_ids=("gen4_peer1", "gen4_peer2", "gen4_peer3"),
            )
            service.configure_generation_maturity(
                4,
                cohort_size=3,
                mature_target=1,
                count_callback=lambda: 0,
            )
            self.assertEqual(
                service.generation_advice("gen4_peer1", 4)["first_wave"], "direct_mature"
            )
            self.assertEqual(service.generation_advice("gen4_peer2", 4)["first_wave"], "explore")

    def test_stop_without_owner_cannot_mutate_live_run_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            owner = ExperimentSchedulerService(
                run_dir=run_dir,
                settings=_settings(maximum=1),
                allocator=_BlockedAllocator(""),
            )
            owner.start()
            try:
                events_path = owner.state_dir / "events.jsonl"
                events_before = events_path.read_bytes() if events_path.exists() else b""
                non_owner = ExperimentSchedulerService(
                    run_dir=run_dir,
                    settings=_settings(maximum=1),
                    allocator=_BlockedAllocator(""),
                )
                non_owner.stop()
                self.assertEqual(
                    events_path.read_bytes() if events_path.exists() else b"",
                    events_before,
                )
                self.assertFalse(owner._admission_closed)
            finally:
                owner.stop()

    def test_open_generation_retires_previous_maturity_callback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=2)
            )
            service.open_generation(0, deadline=time.time() + 600, cohort_size=4)

            def callback() -> int:
                return 0

            service.configure_generation_maturity(
                0,
                cohort_size=4,
                mature_target=1,
                count_callback=callback,
            )
            service.open_generation(1, deadline=time.time() + 600, cohort_size=4)
            self.assertEqual(service._mature_count_callbacks, {})
            self.assertNotIn(0, service._generation_mature_targets)
            self.assertNotIn(0, service._mature_completed)

    def test_hard_quorum_commitments_are_distinct_uncompleted_peers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=10)
            )
            service.open_generation(0, deadline=time.time() + 600, cohort_size=4)
            completed = {"gen0_peer0"}
            service.configure_generation_maturity(
                0,
                cohort_size=4,
                mature_target=2,
                count_callback=lambda: len(completed),
                identity_callback=lambda: set(completed),
                distinct_peer_commitments=True,
            )
            for peer_id in ("gen0_peer0", "gen0_peer1"):
                for index in range(3):
                    service.submit(
                        {
                            "command": [sys.executable, "-c", "pass"],
                            "peer_id": peer_id,
                            "generation_id": 0,
                            "experiment_id": f"{peer_id}-{index}",
                            "work_class": "mature",
                        }
                    )
            service._refresh_mature_counts(HostSnapshot(8, 5, 5, 0, observed_at=1.0))
            maturity = service.status()["resource_supply"]["maturity"]["0"]
            self.assertEqual(maturity["completed"], 1)
            self.assertEqual(maturity["inflight_jobs"], 1)
            self.assertEqual(maturity["target_inflight"], 3)
            self.assertEqual(maturity["needed_inflight"], 2)

    def test_completed_mature_peer_can_receive_frontier_followup_supply(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 1
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=settings,
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(0, deadline=time.time() + 600, cohort_size=1)
            service.configure_generation_maturity(
                0,
                cohort_size=1,
                mature_target=1,
                count_callback=lambda: 1,
                identity_callback=lambda: {"gen0_peer0"},
                distinct_peer_commitments=True,
            )
            snapshot = HostSnapshot(8, 5, 5, 0, observed_at=1.0)
            service._refresh_mature_counts(snapshot)
            service.register_idle_supply("gen0_peer0", 0)
            service._reconcile_supply(snapshot)
            leases = service.status()["resource_supply"]["leases"]
            self.assertEqual(len(leases), 1)
            self.assertEqual(leases[0]["priority"], "frontier_followup")
            self.assertEqual(service.status()["resource_supply"]["idle_waiters"], 0)

    def test_completed_mature_waiter_survives_debt_then_receives_followup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=2)
            settings.supply_idle_samples = 1
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=settings,
                allocator=_SupplyAllocator((("cpu",), ("cpu",))),
            )
            completed = {"gen0_peer0"}
            service.open_generation(0, deadline=time.time() + 600, cohort_size=2)
            service.configure_generation_maturity(
                0,
                cohort_size=2,
                mature_target=2,
                count_callback=lambda: len(completed),
                identity_callback=lambda: set(completed),
                distinct_peer_commitments=True,
            )
            for peer_id in ("gen0_peer0", "gen0_peer1"):
                service.register_idle_supply(peer_id, 0)
            first = HostSnapshot(8, 5, 5, 0, observed_at=1.0)
            service._refresh_mature_counts(first)
            service._reconcile_supply(first)
            self.assertIn("gen0_peer0", service._idle_supply_waiters)
            self.assertEqual(
                service.status()["resource_supply"]["leases"][0]["peer_id"],
                "gen0_peer1",
            )

            completed.add("gen0_peer1")
            second = HostSnapshot(8, 5, 5, 0, observed_at=2.0)
            service._refresh_mature_counts(second)
            service._reconcile_supply(second)
            leases = service.status()["resource_supply"]["leases"]
            self.assertEqual(len(leases), 1)
            self.assertEqual(leases[0]["peer_id"], "gen0_peer0")
            self.assertEqual(leases[0]["priority"], "frontier_followup")

    def test_assessment_rejects_ordinary_but_keeps_mature_topups_open(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=2),
                allocator=_GPUAllocator(""),
            )
            service.open_generation(0, deadline=time.time() + 600, cohort_size=4)
            service.begin_assessment(0, "evidence_ready")
            with self.assertRaisesRegex(ExperimentRejected, "only mature work"):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "experiment_id": "late-scout",
                        "work_class": "scout",
                    }
                )
            mature = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer1",
                    "generation_id": 0,
                    "experiment_id": "mature-topup",
                    "work_class": "mature",
                    "eta_seconds": 60,
                }
            )
            self.assertEqual(mature.state, "queued")
            with self.assertRaisesRegex(ExperimentRejected, "ETA does not fit"):
                service.submit(
                    {
                        "command": [sys.executable, "-c", "pass"],
                        "peer_id": "gen0_peer2",
                        "generation_id": 0,
                        "experiment_id": "unknown-duration-topup",
                        "work_class": "mature",
                    }
                )

    def test_same_basename_runs_have_distinct_resource_controller_owners(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            first = ExperimentSchedulerService(
                run_dir=Path(td) / "task-a" / "run",
                settings=_settings(maximum=1),
            )
            second = ExperimentSchedulerService(
                run_dir=Path(td) / "task-b" / "run",
                settings=_settings(maximum=1),
            )
            self.assertNotEqual(first.resource_owner_id, second.resource_owner_id)

    def test_removed_persisted_profile_fails_resume_instead_of_using_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            with self.assertRaisesRegex(RuntimeError, "no longer declared"):
                service._require_profile("removed-gpu-profile")

    def test_start_failure_closes_rpc_server_and_releases_owner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.threading.Thread.start",
                    side_effect=OSError("thread unavailable"),
                ),
                self.assertRaisesRegex(OSError, "thread unavailable"),
            ):
                service.start()
            self.assertIsNone(service._server)
            self.assertIsNone(service._owner_lock_fd)
            self.assertFalse(service.endpoint.exists())

    def test_endpoint_publication_failure_never_starts_scheduler_threads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.atomic_write_json",
                    side_effect=OSError("endpoint publication unavailable"),
                ),
                self.assertRaisesRegex(OSError, "endpoint publication unavailable"),
            ):
                service.start()
            self.assertFalse(service._server_thread.is_alive())
            self.assertFalse(service._worker_thread.is_alive())
            self.assertIsNone(service._server)
            self.assertIsNone(service._owner_lock_fd)
            self.assertFalse(service.endpoint.exists())
            self.assertFalse((service.state_dir / "endpoint.json").exists())

    def test_post_thread_start_failure_quiesces_worker_before_releasing_owner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.logger.info",
                    side_effect=KeyboardInterrupt("startup interrupted"),
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "startup interrupted"),
            ):
                service.start()
            self.assertFalse(service._server_thread.is_alive())
            self.assertFalse(service._worker_thread.is_alive())
            self.assertIsNone(service._server)
            self.assertIsNone(service._owner_lock_fd)
            self.assertFalse(service.endpoint.exists())
            self.assertFalse((service.state_dir / "endpoint.json").exists())

    def test_start_failure_retains_owner_for_recovered_active_processes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            drained = threading.Event()

            def recover_active() -> None:
                service._active["recovered-live-job"] = MagicMock()

            def drain_active() -> None:
                service._active.clear()
                service._finalize_owner_resources()
                drained.set()

            with (
                patch.object(service, "_recover_active_process_groups", side_effect=recover_active),
                patch.object(service, "_worker", side_effect=drain_active),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler._UnixServer",
                    side_effect=OSError("socket unavailable"),
                ),
                self.assertRaisesRegex(OSError, "socket unavailable"),
            ):
                service.start()
            self.assertTrue(drained.wait(timeout=2))
            self.assertIsNone(service._owner_lock_fd)
            self.assertIsNone(service._server_thread)
            self.assertFalse(service._worker_thread.is_alive())

    def test_owner_metadata_failure_releases_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            first = ExperimentSchedulerService(run_dir=run_dir, settings=_settings(maximum=1))
            with (
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.os.ftruncate",
                    side_effect=OSError("owner metadata unavailable"),
                ),
                self.assertRaisesRegex(OSError, "owner metadata unavailable"),
            ):
                first._acquire_owner_lock()
            self.assertIsNone(first._owner_lock_fd)

            second = ExperimentSchedulerService(run_dir=run_dir, settings=_settings(maximum=1))
            second._acquire_owner_lock()
            self.assertIsNotNone(second._owner_lock_fd)
            second._release_owner_lock()

    def test_start_cleanup_unlink_failure_does_not_strand_owner_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            with (
                patch.object(Path, "unlink", side_effect=PermissionError("unlink unavailable")),
                self.assertRaisesRegex(PermissionError, "unlink unavailable"),
            ):
                service.start()
            self.assertIsNone(service._owner_lock_fd)

    def test_stop_retains_owner_until_transport_cleanup_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            service._acquire_owner_lock()
            server = MagicMock()
            server.server_close.side_effect = OSError("server close unavailable")
            service._server = server
            with (
                self.assertRaisesRegex(OSError, "server close unavailable"),
            ):
                service.stop()
            self.assertIsNotNone(service._owner_lock_fd)
            self.assertFalse(service._transport_cleanup_complete)

            server.server_close.side_effect = None
            service.stop()
            self.assertIsNone(service._owner_lock_fd)
            self.assertTrue(service._transport_cleanup_complete)

    def test_failed_drainer_never_releases_owner_while_recovered_work_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )

            def recover_active() -> None:
                service._active["recovered-live-job"] = MagicMock()

            with (
                patch.object(service, "_recover_active_process_groups", side_effect=recover_active),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler._UnixServer",
                    side_effect=OSError("socket unavailable"),
                ),
                patch(
                    "praxist.plugins.workflow_stages.research_loop.backend."
                    "experiment_scheduler.threading.Thread.start",
                    side_effect=OSError("drainer unavailable"),
                ),
                self.assertRaisesRegex(OSError, "socket unavailable"),
            ):
                service.start()

            service.stop()
            self.assertIsNotNone(service._owner_lock_fd)
            service._active.clear()
            service.stop()
            self.assertIsNone(service._owner_lock_fd)

    def test_allocator_cleanup_failure_retains_owner_for_stop_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("")
            allocator.close = MagicMock(side_effect=[OSError("controller unavailable"), None])
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )
            service._acquire_owner_lock()

            with self.assertRaisesRegex(RuntimeError, "cleanup remains pending"):
                service.stop()
            self.assertIsNotNone(service._owner_lock_fd)

            service.stop()
            self.assertIsNone(service._owner_lock_fd)
            self.assertEqual(allocator.close.call_count, 2)

    def test_worker_keeps_scheduler_observable_after_iteration_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            allocator = _GPUAllocator("")
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=allocator,
            )

            def fail_once(*, queued: bool) -> HostSnapshot:
                del queued
                service._stopping = True
                service._transport_cleanup_complete = True
                raise RuntimeError("host observation failed")

            allocator.refresh = fail_once  # type: ignore[method-assign]
            with (
                patch.object(service._condition, "wait", return_value=None),
                patch.object(service, "_finalize_owner_resources"),
            ):
                service._worker()
            self.assertEqual(service._worker_error, "RuntimeError: host observation failed")

    def test_reconcile_retries_pending_submission_audit_without_losing_lease(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_SupplyAllocator((("cpu",),)),
            )
            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "pending-audit",
                }
            )
            lease = _SupplyLease(
                lease_id="supply-pending-audit",
                peer_id=job.peer_id,
                generation_id=job.generation_id,
                admissible_profiles=("cpu",),
                priority="mature",
                issued_at=time.time(),
                expires_at=time.time() + 60,
            )
            service._supply_leases[lease.lease_id] = lease
            job.supply_claim_id = lease.lease_id
            with patch.object(
                service,
                "_consume_pending_supply_locked",
                side_effect=OSError("audit unavailable"),
            ):
                service._reconcile_supply(HostSnapshot(8, 5, 5, 0))
            self.assertIn(lease.lease_id, service._supply_leases)
            self.assertEqual(job.supply_claim_id, lease.lease_id)

    def test_reconcile_drops_idle_waiter_after_generation_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=1)
            settings.supply_idle_samples = 1
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=settings,
                allocator=_SupplyAllocator((("cpu",),)),
            )
            service.open_generation(0, deadline=time.time() - 1, cohort_size=1)
            service._idle_supply_waiters["gen0_peer0"] = (0, time.time())
            service._reconcile_supply(HostSnapshot(8, 5, 5, 0))
            self.assertNotIn("gen0_peer0", service._idle_supply_waiters)
            self.assertEqual(service.status()["resource_supply"]["leases"], [])

    def test_supply_replay_handles_stale_late_and_legacy_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run", settings=_settings(maximum=1)
            )
            now = time.time()
            service._recover_supply_events(
                [
                    {"event": "supply_stale_submission", "priority": "mature"},
                    {
                        "event": "supply_consumed",
                        "lease_id": "terminal-without-grant",
                        "priority": "mature",
                    },
                    {
                        "event": "supply_granted",
                        "lease_id": "late",
                        "priority": "mature",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "admissible_profiles": ["cpu"],
                        "issued_at": now - 20,
                        "expires_at": now - 10,
                    },
                    {
                        "event": "submitted",
                        "supply_lease_id": "late",
                        "peer_id": "gen0_peer0",
                        "generation_id": 0,
                        "profile": "cpu",
                        "work_class": "mature",
                        "recorded_at": now,
                    },
                    {
                        "event": "supply_granted",
                        "lease_id": "legacy-accepted",
                        "priority": "mature",
                        "peer_id": "gen0_peer1",
                        "generation_id": 0,
                        "admissible_profiles": ["cpu"],
                        "issued_at": now,
                        "expires_at": now + 60,
                    },
                    {
                        "event": "submitted",
                        "supply_lease_id": "legacy-accepted",
                        "peer_id": "gen0_peer1",
                        "generation_id": 0,
                        "profile": "cpu",
                        "work_class": "mature",
                        "recorded_at": now + 1,
                    },
                ]
            )
            status = service.status()["resource_supply"]["stats"]
            self.assertEqual(status["stale_submission"], 1)
            self.assertEqual(status["expired"], 1)
            self.assertEqual(status["consumed"], 1)
            self.assertIn("legacy-accepted", service._consumed_supply_leases)

    def test_recovery_helpers_preserve_live_identity_and_truncate_partial_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            service = ExperimentSchedulerService(
                run_dir=Path(td) / "run",
                settings=_settings(maximum=1),
                allocator=_GPUAllocator("GPU-0"),
            )
            service.allocator.snapshot = HostSnapshot(
                8,
                5,
                5,
                0,
                gpus=(GPUDevice(0, "GPU-0", 10000, 100, 2),),
                gpu_processes=(GPUProcess(999999, "GPU-0", 100, 2),),
            )
            with patch("os.getpgid", side_effect=OSError("process vanished")):
                self.assertEqual(
                    service._recovered_gpu_uuids({}, pid=999999, pgid=42),
                    ("GPU-0",),
                )
            self.assertEqual(service._recovered_profile({}, ("GPU-0",)).name, "gpu")

            job = service.submit(
                {
                    "command": [sys.executable, "-c", "pass"],
                    "peer_id": "gen0_peer0",
                    "generation_id": 0,
                    "experiment_id": "legacy-live",
                }
            )
            service._queue.clear()
            service._jobs.clear()
            service._semantic_jobs.clear()
            job.pid = os.getpid()
            job.pgid = os.getpgid(0)
            service._adopt_unaccounted_live(job, {}, reason="legacy_process")
            self.assertIn(job.job_id, service._active)

            audit = Path(td) / "partial.jsonl"
            audit.write_bytes(b"complete\n" + b"x" * 70000)
            fd = os.open(audit, os.O_RDWR)
            try:
                service._truncate_incomplete_event_tail(fd)
            finally:
                os.close(fd)
            self.assertEqual(audit.read_bytes(), b"complete\n")
            self.assertFalse(service._event_line_present(Path(td) / "missing", b"event\n"))


if __name__ == "__main__":
    unittest.main()
