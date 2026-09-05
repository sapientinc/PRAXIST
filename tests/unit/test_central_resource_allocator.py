from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from praxist.plugins.workflow_stages.research_loop.backend import resource_scheduler
from praxist.plugins.workflow_stages.research_loop.backend.resource_scheduler import (
    Allocation,
    GPUDevice,
    GPUProcess,
    HostAllocationRegistry,
    HostObserver,
    HostSnapshot,
    ResourceAllocator,
    SchedulerSettings,
)
from praxist.task_spec import load_task_spec


class _Observer:
    def __init__(self, snapshots: list[HostSnapshot]) -> None:
        self.snapshots = list(snapshots)

    def snapshot(self) -> HostSnapshot:
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]


class HostAllocationRegistryPlatformTest(unittest.TestCase):
    def test_missing_posix_locking_fails_only_when_registry_is_used(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        registry = HostAllocationRegistry(Path(temporary.name))
        with (
            patch(
                "praxist.plugins.workflow_stages.research_loop.backend."
                "resource_scheduler._posix_file_locking",
                side_effect=RuntimeError(
                    "Praxist central resource scheduling requires POSIX file locking"
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "requires POSIX file locking"),
            registry.locked(),
        ):
            self.fail("registry must not enter an unlocked critical section")


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


class ResourceAllocatorTest(unittest.TestCase):
    def test_host_observer_parses_gpu_inventory_processes_and_pressure(self) -> None:
        inventory = SimpleNamespace(
            returncode=0,
            stdout="0, GPU-a, 40960, 1024, 75\nbad\n1, GPU-b, x, 0, 0\n",
        )
        compute = SimpleNamespace(
            returncode=0,
            stdout="123, GPU-a, 2048\nbad\nnotpid, GPU-a, 1\n",
        )
        pmon = SimpleNamespace(
            returncode=0,
            stdout="# gpu pid type sm mem enc dec command\n0 123 C 64 0 0 0 task\nbad\n",
        )
        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend."
            "resource_scheduler.subprocess.run",
            side_effect=[inventory, compute, pmon],
        ):
            gpus = HostObserver._gpus()
            processes, memory_seen, utilization_seen = HostObserver._gpu_processes(tuple(gpus))
        self.assertEqual([gpu.uuid for gpu in gpus], ["GPU-a"])
        self.assertEqual(processes[0].pid, 123)
        self.assertEqual(processes[0].memory_used_mb, 2048)
        self.assertEqual(processes[0].utilization_pct, 64.0)
        self.assertTrue(memory_seen)
        self.assertTrue(utilization_seen)

        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend."
            "resource_scheduler.subprocess.run",
            side_effect=OSError("missing driver"),
        ):
            self.assertEqual(HostObserver._gpus(), [])
            self.assertEqual(HostObserver._gpu_processes(tuple()), ((), False, False))
        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend."
            "resource_scheduler.subprocess.run",
            side_effect=[compute, OSError("pmon unavailable")],
        ):
            processes, memory_seen, utilization_seen = HostObserver._gpu_processes(tuple(gpus))
        self.assertEqual(len(processes), 1)
        self.assertTrue(memory_seen)
        self.assertFalse(utilization_seen)
        unmatched_pmon = SimpleNamespace(
            returncode=0,
            stdout="# gpu pid type sm mem enc dec command\n0 999 C 64 0 0 0 other\n",
        )
        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend."
            "resource_scheduler.subprocess.run",
            side_effect=[compute, unmatched_pmon],
        ):
            processes, memory_seen, utilization_seen = HostObserver._gpu_processes(tuple(gpus))
        self.assertIsNone(processes[0].utilization_pct)
        self.assertTrue(memory_seen)
        self.assertFalse(utilization_seen)

        observer = HostObserver()
        with patch(
            "praxist.plugins.workflow_stages.research_loop.backend."
            "resource_scheduler.Path.read_text",
            side_effect=[
                "cpu  10 0 10 80 0\n",
                "cpu  30 0 20 100 0\n",
                "MemTotal: 1000 kB\nMemAvailable: 250 kB\n",
                "some avg10=12.5 avg60=0 avg300=0 total=0\n",
            ],
        ):
            observer._cpu_utilization()
            self.assertGreater(observer._cpu_utilization(), 0)
            self.assertEqual(observer._memory_utilization(), 75.0)
            self.assertEqual(observer._io_pressure(), 12.5)

    def test_central_config_rejects_all_malformed_profile_shapes(self) -> None:
        invalid = [
            ({"mode": "central", "profiles": []}, "profiles must be an object"),
            ({"mode": "central", "max_concurrent_experiments": True}, "must be an integer"),
            (
                {"mode": "central", "profiles": {1: {"accelerator": "cpu"}}},
                "names must be strings",
            ),
            (
                {"mode": "central", "profiles": {"": {"accelerator": "cpu"}}},
                "names must be non-empty",
            ),
            (
                {
                    "mode": "central",
                    "profiles": {"cpu": {"accelerator": "cpu", "unknown": 1}},
                },
                "unknown field",
            ),
            (
                {
                    "mode": "central",
                    "profiles": {"cpu": {"accelerator": "cpu", "pressure_domains": "cpu"}},
                },
                "must be a list",
            ),
            (
                {
                    "mode": "central",
                    "profiles": {"gpu": {"accelerator": "gpu", "gpu_memory_gb": False}},
                },
                "positive finite number",
            ),
        ]
        for raw, message in invalid:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                SchedulerSettings.from_dict(raw)

        self.assertEqual(resource_scheduler._as_int("bad", 7), 7)
        self.assertEqual(resource_scheduler._as_int(object(), 8), 8)
        self.assertEqual(resource_scheduler._as_float(object(), 3.0), 3.0)
        self.assertEqual(resource_scheduler._as_float("bad", 4.0), 4.0)
        self.assertEqual(resource_scheduler._as_float("inf", 5.0), 5.0)
        self.assertTrue(resource_scheduler._as_bool("yes", False))
        self.assertFalse(resource_scheduler._as_bool("off", True))
        self.assertTrue(resource_scheduler._as_bool("unknown", True))
        self.assertIsNone(resource_scheduler._optional_positive_float(0))
        self.assertEqual(resource_scheduler._optional_positive_float(9, maximum=3), 3)

        legacy = SchedulerSettings.from_dict(
            {
                "mode": "legacy",
                "max_concurrent_experiments": True,
                "profiles": {
                    "bad": "ignored",
                    "": {"accelerator": "cpu"},
                    "cpu": {"accelerator": "cpu", "pressure_domains": "bad"},
                },
                "supply_signal_enabled": "unknown",
            }
        )
        self.assertFalse(legacy.enabled)

    def test_process_group_liveness_handles_expiry_reuse_and_permissions(self) -> None:
        now = time.time()
        self.assertFalse(
            resource_scheduler._process_group_alive(
                {
                    "record_type": "supply",
                    "expires_at": now - 1,
                    "pid": os.getpid(),
                }
            )
        )
        with patch.object(resource_scheduler, "_pid_start_time", return_value=2):
            self.assertFalse(
                resource_scheduler._process_group_alive(
                    {
                        "record_type": "controller",
                        "pid": os.getpid(),
                        "pid_start_time": 1,
                    }
                )
            )
            self.assertFalse(
                resource_scheduler._process_group_alive(
                    {
                        "record_type": "allocation",
                        "pid": os.getpid(),
                        "pgid": os.getpgrp(),
                        "pid_start_time": 1,
                    }
                )
            )
        self.assertFalse(
            resource_scheduler._process_group_alive(
                {
                    "record_type": "allocation",
                    "pid": os.getpid(),
                    "pending": True,
                    "started_at": now - 120,
                }
            )
        )
        with patch.object(os, "kill", side_effect=PermissionError):
            self.assertTrue(
                resource_scheduler._process_group_alive(
                    {"record_type": "allocation", "pid": 123, "pgid": 0}
                )
            )
        with (
            patch.object(Path, "read_text", side_effect=OSError("gone")),
            patch("praxist.cli.registry.process_start_token", return_value=""),
        ):
            self.assertIsNone(resource_scheduler._pid_start_time(999999))

        with (
            patch.object(Path, "read_text", side_effect=OSError("no procfs")),
            patch("praxist.cli.registry.process_start_token", return_value="ps:stable-start"),
        ):
            self.assertEqual(resource_scheduler._pid_start_time(4321), "ps:stable-start")

    def test_legacy_config_and_driver_failures_fall_back_without_crashing(self) -> None:
        legacy = SchedulerSettings.from_dict(
            {
                "mode": "legacy",
                "initial_concurrent_experiments": object(),
                "adjustment_interval_seconds": True,
                "cpu_low_pct": object(),
                "profiles": {
                    "gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": object(),
                        "gpu_utilization_pct": True,
                    }
                },
            }
        )
        self.assertFalse(legacy.enabled)
        self.assertEqual(legacy.initial_concurrency, 1)
        self.assertIsNone(legacy.profiles["gpu"].gpu_memory_gb)
        self.assertIsNone(legacy.profiles["gpu"].gpu_utilization_pct)

        with self.assertRaisesRegex(ValueError, "finite number"):
            SchedulerSettings.from_dict({"mode": "central", "adjustment_interval_seconds": True})
        with self.assertRaisesRegex(ValueError, "finite number"):
            SchedulerSettings.from_dict(
                {"mode": "central", "adjustment_interval_seconds": object()}
            )
        with self.assertRaisesRegex(ValueError, "positive finite number"):
            SchedulerSettings.from_dict(
                {
                    "mode": "central",
                    "profiles": {"gpu": {"accelerator": "gpu", "gpu_memory_gb": object()}},
                }
            )

        failed = SimpleNamespace(returncode=1, stdout="driver unavailable")
        with patch.object(resource_scheduler.subprocess, "run", return_value=failed):
            self.assertEqual(HostObserver._gpu_processes(tuple()), ((), False, False))

    def test_supply_and_recovery_capacity_boundaries_are_conservative(self) -> None:
        snapshot = HostSnapshot(
            16,
            10,
            10,
            0,
            (GPUDevice(0, "GPU-a", 80 * 1024, 0, 0),),
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 1,
                "max_concurrent_experiments": 1,
                "profiles": {
                    "cpu": {"accelerator": "cpu"},
                    "gpu_known": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 10,
                        "gpu_utilization_pct": 20,
                    },
                    "gpu_unknown": {"accelerator": "gpu"},
                },
                "default_profile": "cpu",
            }
        )
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=registry,
            )
            self.assertFalse(allocator.supply_claim_valid("missing"))
            self.assertEqual(
                allocator.claim_supply(
                    lease_id="unknown-gpu-supply",
                    run_id="supply:run",
                    expires_at=time.time() + 60,
                ),
                ("cpu", "gpu_known", "gpu_unknown"),
            )
            self.assertTrue(allocator.supply_claim_valid("unknown-gpu-supply"))
            allocator.clear_supply("supply:run")

            first = allocator.reserve(
                allocation_id="first",
                run_id="run-a",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("cpu"),
            )
            self.assertIsNotNone(first)
            self.assertIsNone(
                allocator.reserve(
                    allocation_id="same-run-overflow",
                    run_id="run-a",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=settings.profile("cpu"),
                )
            )
            self.assertIsNone(
                allocator.reserve(
                    allocation_id="host-overflow",
                    run_id="run-b",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=settings.profile("cpu"),
                )
            )
            self.assertIsNone(
                allocator.recover_allocation(
                    allocation_id="cpu-with-gpu",
                    run_id="run-a",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=settings.profile("cpu"),
                    gpu_uuids=("GPU-a",),
                    require_admission=False,
                )
            )
            self.assertIsNone(
                allocator.recover_allocation(
                    allocation_id="first",
                    run_id="run-a",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=settings.profile("gpu_known"),
                    gpu_uuids=("GPU-a",),
                    require_admission=False,
                )
            )

        known = settings.profile("gpu_known")
        unknown = settings.profile("gpu_unknown")
        cpu = settings.profile("cpu")
        self.assertFalse(ResourceAllocator._profile_fits_supply_claim(known, cpu))
        self.assertTrue(ResourceAllocator._profile_fits_supply_claim(known, unknown))
        self.assertFalse(ResourceAllocator._profile_fits_supply_claim(unknown, known))
        self.assertIsNone(
            ResourceAllocator._choose_gpus(
                known,
                snapshot,
                [],
                required_uuids=("GPU-missing",),
            )
        )
        busy_snapshot = HostSnapshot(
            16,
            10,
            10,
            0,
            (GPUDevice(0, "GPU-a", 80 * 1024, 512, 0),),
        )
        self.assertIsNone(ResourceAllocator._choose_gpus(unknown, busy_snapshot, []))

        process_snapshot = HostSnapshot(
            16,
            10,
            10,
            0,
            (GPUDevice(0, "GPU-a", 80 * 1024, 0, 0),),
            (GPUProcess(123, "GPU-a", 1, 1),),
            True,
            True,
        )
        with patch.object(os, "getpgid", side_effect=ProcessLookupError):
            memory, utilization = ResourceAllocator._attributed_gpu_load(
                process_snapshot,
                [{"record_type": "allocation", "pgid": 123}],
            )
        self.assertEqual(memory, {})
        self.assertEqual(utilization, {})

    def test_explicit_invalid_accelerator_or_default_profile_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown resource scheduler mode"):
            SchedulerSettings.from_dict({"mode": "centrla"})
        with self.assertRaisesRegex(ValueError, "must be an object"):
            SchedulerSettings.from_dict({"mode": "central", "profiles": {"gpu": "not-a-map"}})
        with self.assertRaisesRegex(ValueError, "invalid accelerator"):
            SchedulerSettings.from_dict(
                {
                    "mode": "central",
                    "profiles": {"train": {"accelerator": "gup"}},
                    "default_profile": "train",
                }
            )
        with self.assertRaisesRegex(ValueError, "is not declared"):
            SchedulerSettings.from_dict(
                {
                    "mode": "central",
                    "profiles": {"train": {"accelerator": "gpu"}},
                    "default_profile": "missing",
                }
            )
        with self.assertRaisesRegex(ValueError, "supply_signal_enabled must be a boolean"):
            SchedulerSettings.from_dict({"mode": "central", "supply_signal_enabled": "flase"})
        with self.assertRaisesRegex(ValueError, "unknown central resource scheduler field"):
            SchedulerSettings.from_dict({"mode": "central", "supply_signla_enabled": False})
        with self.assertRaisesRegex(ValueError, "unknown pressure domain"):
            SchedulerSettings.from_dict(
                {
                    "mode": "central",
                    "profiles": {
                        "default": {
                            "accelerator": "cpu",
                            "pressure_domains": ["cpu", "memroy"],
                        }
                    },
                }
            )

    def test_task_load_rejects_malformed_central_scheduler_policy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            task_yaml = Path(td) / "task.yaml"
            task_yaml.write_text(
                "compute_budget:\n"
                "  resource_scheduler:\n"
                "    mode: central\n"
                "    supply_signal_enabled: flase\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "supply_signal_enabled must be a boolean"):
                load_task_spec(task_yaml)

            task_yaml.write_text(
                "compute_budget:\n  resource_scheduler: central\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "resource_scheduler must be an object"):
                load_task_spec(task_yaml)

    def test_high_declared_cpu_or_memory_pressure_blocks_new_admission(self) -> None:
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 8,
                "max_concurrent_experiments": 8,
                "profiles": {
                    "cpu": {
                        "accelerator": "cpu",
                        "pressure_domains": ["cpu", "memory"],
                    }
                },
            }
        )
        for snapshot in (
            HostSnapshot(64, 99, 20, 0),
            HostSnapshot(64, 20, 99, 0),
        ):
            with tempfile.TemporaryDirectory() as td:
                allocator = ResourceAllocator(
                    settings,
                    observer=_Observer([snapshot]),
                    registry=HostAllocationRegistry(Path(td)),
                )
                self.assertIsNone(
                    allocator.reserve(
                        allocation_id="blocked",
                        run_id="run",
                        pid=0,
                        pgid=0,
                        profile=settings.profiles["cpu"],
                    )
                )

    def test_final_reservation_refreshes_host_pressure(self) -> None:
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 4,
                "max_concurrent_experiments": 4,
                "profiles": {
                    "cpu": {
                        "accelerator": "cpu",
                        "pressure_domains": ["cpu", "memory"],
                    }
                },
            }
        )
        observer = _Observer(
            [
                HostSnapshot(64, 10, 10, 0),
                HostSnapshot(64, 99, 10, 0),
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            allocator = ResourceAllocator(
                settings,
                observer=observer,
                registry=HostAllocationRegistry(Path(td)),
            )
            self.assertEqual(allocator.snapshot.cpu_utilization_pct, 10)
            self.assertIsNone(
                allocator.reserve(
                    allocation_id="blocked-after-refresh",
                    run_id="run",
                    pid=0,
                    pgid=0,
                    profile=settings.profiles["cpu"],
                )
            )
            self.assertEqual(allocator.snapshot.cpu_utilization_pct, 99)

    def test_sam_template_never_rewrites_central_uuid_assignment(self) -> None:
        path = Path(
            "templates/tasks/sam_optimizer/assets/harness/benchmark/run_benchmark.py"
        ).resolve()
        spec = importlib.util.spec_from_file_location("sam_central_assignment_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = SimpleNamespace(parallel_datasets="1", optimizer="sam", tier="T1")
        with (
            patch.dict(
                os.environ,
                {"PRAXIST_ASSIGNED_GPU_UUIDS": "GPU-exact", "CUDA_VISIBLE_DEVICES": "GPU-exact"},
                clear=True,
            ),
            patch.object(module, "pick_and_acquire_gpu") as pick,
            patch.object(module, "_main_after_acquire") as run,
            patch.object(module, "_reclaim_stale_slots"),
        ):
            module._main_after_parse(args, ["dataset"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-exact")
            self.assertEqual(os.environ["NVIDIA_VISIBLE_DEVICES"], "GPU-exact")
        pick.assert_not_called()
        run.assert_called_once_with(args, -1)
        self.assertIn(
            "refusing to publish CPU results under the GPU protocol",
            path.read_text(encoding="utf-8"),
        )

    def test_sam_template_validates_uuid_mig_and_conflicting_masks(self) -> None:
        path = Path(
            "templates/tasks/sam_optimizer/assets/harness/benchmark/run_benchmark.py"
        ).resolve()
        spec = importlib.util.spec_from_file_location("sam_assignment_contract_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for assigned in ("GPU-a,GPU-b", "MIG-GPU-a/1/0"):
            env = {"PRAXIST_ASSIGNED_GPU_UUIDS": assigned}
            self.assertEqual(module._apply_scheduler_assignment(env), assigned)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], assigned)
            self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], assigned)

        with self.assertRaisesRegex(RuntimeError, "accelerator binding mismatch"):
            module._apply_scheduler_assignment(
                {"PRAXIST_ASSIGNED_GPU_UUIDS": "GPU-a", "CUDA_VISIBLE_DEVICES": "0"}
            )
        standalone = {"CUDA_VISIBLE_DEVICES": "2"}
        self.assertEqual(module._apply_scheduler_assignment(standalone), "")
        self.assertEqual(standalone["CUDA_VISIBLE_DEVICES"], "2")

    def test_recovery_atomically_rebinds_or_recreates_exact_gpu_allocation(self) -> None:
        snapshot = HostSnapshot(
            16,
            10,
            10,
            0,
            (GPUDevice(0, "GPU-a", 80_000, 0, 0),),
        )
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=2)
            registry = HostAllocationRegistry(Path(td))
            allocator = ResourceAllocator(
                settings, observer=_Observer([snapshot]), registry=registry
            )
            profile = settings.profile("gpu")
            recovered = allocator.recover_allocation(
                allocation_id="lost-pending-row",
                run_id="run",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=profile,
                gpu_uuids=("GPU-a",),
            )
            self.assertIsNotNone(recovered)
            rows = json.loads(registry.path.read_text())
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["pending"])
            self.assertEqual(rows[0]["gpu_uuids"], ["GPU-a"])
            unavailable = allocator.recover_allocation(
                allocation_id="wrong-device",
                run_id="run",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=profile,
                gpu_uuids=("GPU-missing",),
            )
            self.assertIsNone(unavailable)

    def test_recovery_preempts_conflicting_speculative_gpu_supply(self) -> None:
        snapshot = HostSnapshot(
            16,
            10,
            10,
            0,
            (GPUDevice(0, "GPU-a", 80_000, 0, 0),),
        )
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=2)
            supplier = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            recovering = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            self.assertTrue(
                supplier.claim_supply(
                    lease_id="speculative",
                    run_id="supply:other",
                    expires_at=time.time() + 60,
                )
            )
            allocation = recovering.recover_allocation(
                allocation_id="live-process",
                run_id="run:current",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("gpu"),
                gpu_uuids=("GPU-a",),
            )
            self.assertIsNotNone(allocation)
            rows = json.loads(supplier.registry.path.read_text())
            self.assertEqual([row["allocation_id"] for row in rows], ["live-process"])

    def test_dead_controller_is_not_kept_alive_by_unrelated_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            registry.path.write_text(
                json.dumps(
                    [
                        {
                            "record_type": "controller",
                            "run_id": "dead",
                            "pid": 999_999_999,
                            "pgid": os.getpgrp(),
                            "concurrency_limit": 1,
                        }
                    ]
                )
            )
            with registry.locked() as rows:
                self.assertEqual(rows, [])

    def test_controller_pid_reuse_is_rejected_by_process_start_time(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            registry.path.write_text(
                json.dumps(
                    [
                        {
                            "record_type": "controller",
                            "run_id": "reused",
                            "pid": os.getpid(),
                            "pgid": os.getpgrp(),
                            "pid_start_time": -1,
                            "concurrency_limit": 1,
                        }
                    ]
                )
            )
            with registry.locked() as rows:
                self.assertEqual(rows, [])

    def test_abandoned_pending_reservation_expires_while_controller_lives(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            registry.path.write_text(
                json.dumps(
                    [
                        {
                            "record_type": "allocation",
                            "allocation_id": "abandoned",
                            "pending": True,
                            "pid": os.getpid(),
                            "pgid": os.getpgrp(),
                            "started_at": time.time() - 120,
                        }
                    ]
                )
            )
            with registry.locked() as rows:
                self.assertEqual(rows, [])

    def test_task_template_exposes_central_scheduler_contract(self) -> None:
        spec = load_task_spec("templates/tasks/toy_math/task.yaml")
        settings = SchedulerSettings.from_dict(spec.compute_budget.resource_scheduler)
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.default_profile, "cpu_eval")

    def test_task_spec_rejects_non_object_resource_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "description.md").write_text("generic\n")
            (root / "task.yaml").write_text(
                """
task_id: invalid_scheduler_shape
task_name: Invalid Scheduler Shape
description_file: description.md
research_direction: Generic research.
compute_budget:
  resource_scheduler: central
""".lstrip()
            )
            with self.assertRaisesRegex(ValueError, "resource_scheduler must be an object"):
                load_task_spec(root / "task.yaml")

    def test_gpu_profile_cannot_disable_assignment_with_zero_count(self) -> None:
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "profiles": {"gpu": {"accelerator": "gpu", "gpu_count": 0}},
            }
        )
        self.assertEqual(settings.profile("gpu").gpu_count, 1)

    def test_gpu_packing_uses_uuid_and_live_memory_utilization(self) -> None:
        snapshot = HostSnapshot(
            cpu_count=16,
            cpu_utilization_pct=20,
            memory_utilization_pct=20,
            io_pressure_pct=0,
            gpus=(
                GPUDevice(0, "GPU-a", 80 * 1024, 0, 0),
                GPUDevice(1, "GPU-b", 80 * 1024, 0, 0),
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            settings = _settings(maximum=8)
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            profile = settings.profile("gpu")
            first = allocator.reserve(
                allocation_id="a1",
                run_id="r",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=profile,
            )
            second = allocator.reserve(
                allocation_id="a2",
                run_id="r",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=profile,
            )
            third = allocator.reserve(
                allocation_id="a3",
                run_id="r",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=profile,
            )
            self.assertEqual(first.gpu_uuids, ("GPU-a",))
            self.assertEqual(second.gpu_uuids, ("GPU-b",))
            self.assertEqual(third.gpu_uuids, ("GPU-a",))

    def test_external_gpu_load_and_pending_reservations_are_both_counted(self) -> None:
        snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            (GPUDevice(0, "GPU-a", 100 * 1024, 30 * 1024, 30),),
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 8,
                "max_concurrent_experiments": 8,
                "profiles": {
                    "gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 30,
                        "gpu_utilization_pct": 30,
                    }
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            results = [
                allocator.reserve(
                    allocation_id=f"a{i}",
                    run_id="r",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=settings.profile("gpu"),
                )
                for i in range(3)
            ]
            self.assertIsNotNone(results[0])
            self.assertIsNotNone(results[1])
            self.assertIsNone(results[2])

    def test_live_gpu_load_does_not_double_count_settled_peak_reservations(self) -> None:
        snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            (GPUDevice(0, "GPU-a", 100 * 1024, 40 * 1024, 40),),
            (GPUProcess(os.getpid(), "GPU-a", 40 * 1024, 40),),
            True,
            True,
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 8,
                "max_concurrent_experiments": 8,
                "profiles": {
                    "gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 30,
                        "gpu_utilization_pct": 30,
                    }
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            registry.path.write_text(
                json.dumps(
                    [
                        {
                            "record_type": "allocation",
                            "allocation_id": f"settled-{index}",
                            "pending": False,
                            "run_id": "other",
                            "pid": os.getpid(),
                            "pgid": os.getpgrp(),
                            "gpu_uuids": ["GPU-a"],
                            "gpu_memory_mb": 30 * 1024,
                            "gpu_utilization_pct": 30,
                            "started_at": snapshot.observed_at - 30,
                        }
                        for index in range(2)
                    ]
                )
            )
            allocator = ResourceAllocator(
                settings, observer=_Observer([snapshot]), registry=registry
            )
            admitted = allocator.reserve(
                allocation_id="new",
                run_id="run",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("gpu"),
            )
            self.assertIsNotNone(admitted)

    def test_low_observed_compute_does_not_discard_settled_peak_reservation(self) -> None:
        observed = HostSnapshot(
            16,
            20,
            20,
            0,
            (GPUDevice(0, "GPU-a", 100 * 1024, 5 * 1024, 5),),
            (GPUProcess(os.getpid(), "GPU-a", 5 * 1024, 5),),
            True,
            True,
        )
        profile = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "profiles": {
                    "gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 40,
                        "gpu_utilization_pct": 40,
                    }
                },
            }
        ).profile("gpu")
        active = [
            {
                "record_type": "allocation",
                "allocation_id": "settled",
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "gpu_uuids": ["GPU-a"],
                "gpu_memory_mb": 50 * 1024,
                "gpu_utilization_pct": 70,
                "started_at": observed.observed_at - 30,
            }
        ]
        self.assertIsNone(ResourceAllocator._choose_gpus(profile, observed, active))

        no_compute_attribution = HostSnapshot(
            16,
            20,
            20,
            0,
            observed.gpus,
            observed.gpu_processes,
            True,
            False,
            observed_at=observed.observed_at,
        )
        active[0]["gpu_memory_mb"] = 50 * 1024
        self.assertIsNone(ResourceAllocator._choose_gpus(profile, no_compute_attribution, active))

    def test_allocation_activity_separates_lifecycle_from_resource_phase(self) -> None:
        snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            (GPUDevice(0, "GPU-a", 100 * 1024, 1024, 5),),
            (GPUProcess(os.getpid(), "GPU-a", 1024, 5),),
            True,
            True,
        )
        allocator = ResourceAllocator(_settings(), observer=_Observer([snapshot]))
        allocation = Allocation(
            allocation_id="active",
            run_id="run",
            pid=os.getpid(),
            pgid=os.getpgrp(),
            profile="gpu",
            gpu_uuids=("GPU-a",),
            gpu_memory_mb=20 * 1024,
            gpu_utilization_pct=40,
            started_at=time.time(),
        )
        activity = allocator.describe_allocation_activity(allocation)
        self.assertEqual(activity["state"], "gpu_compute_active")
        self.assertEqual(activity["gpu_processes"], 1)
        self.assertEqual(activity["gpu_memory_mb"], 1024)

        allocator._last_snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            snapshot.gpus,
            (GPUProcess(os.getpid(), "GPU-a", 1024, 0),),
            True,
            True,
        )
        self.assertEqual(
            allocator.describe_allocation_activity(allocation)["state"],
            "gpu_context_idle",
        )
        allocator._last_snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            snapshot.gpus,
            (GPUProcess(os.getpid(), "GPU-a", 1024, None),),
            True,
            False,
        )
        self.assertEqual(
            allocator.describe_allocation_activity(allocation)["state"],
            "gpu_context_present",
        )
        allocator._last_snapshot = HostSnapshot(16, 20, 20, 0, snapshot.gpus, (), True, True)
        self.assertEqual(
            allocator.describe_allocation_activity(allocation)["state"],
            "no_gpu_process_observed",
        )
        allocator._last_snapshot = HostSnapshot(16, 20, 20, 0, snapshot.gpus)
        self.assertEqual(
            allocator.describe_allocation_activity(allocation)["state"],
            "unknown",
        )
        allocator._last_snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            snapshot.gpus,
            (
                GPUProcess(os.getpid(), "GPU-other", 1024, 5),
                GPUProcess(999_999_999, "GPU-a", 1024, 5),
            ),
            True,
            True,
        )
        with patch.object(os, "getpgid", side_effect=ProcessLookupError):
            unavailable = allocator.describe_allocation_activity(allocation)
        self.assertEqual(unavailable["state"], "gpu_process_attribution_unavailable")
        self.assertEqual(unavailable["attribution"], "unavailable")
        self.assertIsNone(unavailable["gpu_processes"])
        self.assertIsNone(unavailable["gpu_memory_mb"])
        self.assertIsNone(unavailable["gpu_utilization_pct"])

        invisible_pid = 999_999_999
        allocator._last_snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            snapshot.gpus,
            (
                GPUProcess(os.getpid(), "GPU-a", 1024, 5),
                GPUProcess(invisible_pid, "GPU-a", 1024, 5),
            ),
            True,
            True,
        )

        def process_group(pid: int) -> int:
            if pid == invisible_pid:
                raise ProcessLookupError
            return allocation.pgid

        with patch.object(os, "getpgid", side_effect=process_group):
            partial = allocator.describe_allocation_activity(allocation)
        self.assertEqual(partial["state"], "gpu_compute_active")
        self.assertEqual(partial["attribution"], "partial")
        self.assertEqual(partial["gpu_processes"], 1)
        cpu_allocation = Allocation(
            allocation_id="cpu",
            run_id="run",
            pid=os.getpid(),
            pgid=os.getpgrp(),
            profile="cpu",
            gpu_uuids=(),
            gpu_memory_mb=0,
            gpu_utilization_pct=0,
            started_at=time.time(),
        )
        self.assertEqual(
            allocator.describe_allocation_activity(cpu_allocation)["state"],
            "non_gpu_allocation",
        )

    def test_external_gpu_load_is_added_to_settled_peak_reservations(self) -> None:
        snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            (GPUDevice(0, "GPU-a", 100 * 1024, 40 * 1024, 40),),
            (GPUProcess(os.getpid(), "GPU-a", 10 * 1024, 10),),
            True,
            True,
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 8,
                "max_concurrent_experiments": 8,
                "profiles": {
                    "gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 30,
                        "gpu_utilization_pct": 30,
                    }
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            registry.path.write_text(
                json.dumps(
                    [
                        {
                            "record_type": "allocation",
                            "allocation_id": "settled",
                            "pending": False,
                            "run_id": "other",
                            "pid": os.getpid(),
                            "pgid": os.getpgrp(),
                            "gpu_uuids": ["GPU-a"],
                            "gpu_memory_mb": 60 * 1024,
                            "gpu_utilization_pct": 60,
                            "started_at": snapshot.observed_at - 30,
                        }
                    ]
                )
            )
            allocator = ResourceAllocator(
                settings, observer=_Observer([snapshot]), registry=registry
            )
            self.assertIsNone(
                allocator.reserve(
                    allocation_id="unsafe-new",
                    run_id="run",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=settings.profile("gpu"),
                )
            )

    def test_unresolvable_gpu_pid_keeps_conservative_admission_accounting(self) -> None:
        snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            (GPUDevice(0, "GPU-a", 100 * 1024, 40 * 1024, 40),),
            (GPUProcess(999_999_999, "GPU-a", 40 * 1024, 40),),
            True,
            True,
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "profiles": {
                    "gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 30,
                        "gpu_utilization_pct": 30,
                    }
                },
            }
        )
        active = [
            {
                "record_type": "allocation",
                "pgid": os.getpgrp(),
                "gpu_uuids": ["GPU-a"],
                "gpu_memory_mb": 40 * 1024,
                "gpu_utilization_pct": 40,
                "started_at": snapshot.observed_at - 30,
            }
        ]
        with patch.object(os, "getpgid", side_effect=ProcessLookupError):
            chosen = ResourceAllocator._choose_gpus(
                settings.profile("gpu"),
                snapshot,
                active,
            )
        self.assertIsNone(chosen)

    def test_settled_peak_reservations_enforce_vram_and_compute_caps(self) -> None:
        snapshot = HostSnapshot(
            16,
            20,
            20,
            0,
            (GPUDevice(0, "GPU-a", 100 * 1024, 20 * 1024, 20),),
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 8,
                "max_concurrent_experiments": 8,
                "profiles": {
                    "gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 30,
                        "gpu_utilization_pct": 30,
                    }
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            registry.path.write_text(
                json.dumps(
                    [
                        {
                            "record_type": "allocation",
                            "allocation_id": "settled",
                            "pending": False,
                            "run_id": "other",
                            "pid": os.getpid(),
                            "pgid": os.getpgrp(),
                            "gpu_uuids": ["GPU-a"],
                            "gpu_memory_mb": 70 * 1024,
                            "gpu_utilization_pct": 70,
                            "started_at": snapshot.observed_at - 30,
                        }
                    ]
                )
            )
            allocator = ResourceAllocator(
                settings, observer=_Observer([snapshot]), registry=registry
            )
            self.assertIsNone(
                allocator.reserve(
                    allocation_id="new",
                    run_id="run",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=settings.profile("gpu"),
                )
            )

    def test_unknown_gpu_demand_is_exclusive_without_blocking_cpu(self) -> None:
        snapshot = HostSnapshot(16, 10, 10, 0, (GPUDevice(0, "GPU-a", 1000, 0, 0),))
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 3,
                "max_concurrent_experiments": 3,
                "profiles": {
                    "gpu": {"accelerator": "gpu"},
                    "measured_gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 0.2,
                        "gpu_utilization_pct": 20,
                    },
                    "cpu": {"accelerator": "cpu"},
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            gpu = allocator.reserve(
                allocation_id="gpu1",
                run_id="r",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("gpu"),
            )
            blocked = allocator.reserve(
                allocation_id="gpu2",
                run_id="r",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("gpu"),
            )
            measured_blocked = allocator.reserve(
                allocation_id="gpu-measured",
                run_id="r",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("measured_gpu"),
            )
            cpu = allocator.reserve(
                allocation_id="cpu1",
                run_id="r",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("cpu"),
            )
            self.assertIsNotNone(gpu)
            self.assertIsNone(blocked)
            self.assertIsNone(measured_blocked)
            self.assertIsNotNone(cpu)

    def test_cpu_pressure_changes_only_total_concurrency(self) -> None:
        low = HostSnapshot(64, 10, 10, 0)
        high = HostSnapshot(64, 99, 10, 0)
        settings = _settings(maximum=8, initial=4)
        allocator = ResourceAllocator(settings, observer=_Observer([low, high]))
        allocator._last_adjustment = -100
        allocator.refresh(queued=True)
        self.assertEqual(allocator.concurrency_limit, 3)
        self.assertFalse(hasattr(settings.profile("cpu"), "cpu_cores"))

    def test_independent_runs_share_one_host_gpu_ledger(self) -> None:
        snapshot = HostSnapshot(16, 10, 10, 0, (GPUDevice(0, "GPU-a", 80_000, 0, 0),))
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 2,
                "max_concurrent_experiments": 2,
                "profiles": {"gpu": {"accelerator": "gpu"}},
            }
        )
        with tempfile.TemporaryDirectory() as td:
            registry_a = HostAllocationRegistry(Path(td))
            registry_b = HostAllocationRegistry(Path(td))
            allocator_a = ResourceAllocator(
                settings, observer=_Observer([snapshot]), registry=registry_a
            )
            allocator_b = ResourceAllocator(
                settings, observer=_Observer([snapshot]), registry=registry_b
            )
            first = allocator_a.reserve(
                allocation_id="run-a-job",
                run_id="run-a",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("gpu"),
            )
            second = allocator_b.reserve(
                allocation_id="run-b-job",
                run_id="run-b",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("gpu"),
            )
            self.assertIsNotNone(first)
            self.assertIsNone(second)

    def test_one_run_limit_does_not_throttle_an_independent_cpu_run(self) -> None:
        snapshot = HostSnapshot(64, 20, 20, 0)
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            low = ResourceAllocator(
                _settings(maximum=2, initial=2),
                observer=_Observer([snapshot]),
                registry=registry,
            )
            high = ResourceAllocator(
                _settings(maximum=8, initial=8),
                observer=_Observer([snapshot]),
                registry=registry,
            )
            low.set_owner("low")
            high.set_owner("high")
            try:
                first = high.reserve(
                    allocation_id="one",
                    run_id="high",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=high.settings.profile("cpu"),
                )
                second = high.reserve(
                    allocation_id="two",
                    run_id="high",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=high.settings.profile("cpu"),
                )
                third = high.reserve(
                    allocation_id="three",
                    run_id="high",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=high.settings.profile("cpu"),
                )
            finally:
                low.close()
                high.close()
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertIsNotNone(third)

    def test_supply_claims_keep_profile_specific_capacity_visible(self) -> None:
        snapshot = HostSnapshot(
            64,
            10,
            10,
            0,
            (
                GPUDevice(0, "GPU-a", 80 * 1024, 0, 0),
                GPUDevice(1, "GPU-b", 80 * 1024, 0, 0),
            ),
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 4,
                "max_concurrent_experiments": 4,
                "profiles": {
                    "cpu": {
                        "accelerator": "cpu",
                        "pressure_domains": ["cpu", "memory"],
                    },
                    "gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 10,
                        "gpu_utilization_pct": 60,
                        "pressure_domains": ["gpu"],
                    },
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            claims = [
                allocator.claim_supply(
                    lease_id=f"lease-{index}",
                    run_id="run",
                    expires_at=time.time() + 60,
                )
                for index in range(4)
            ]
            self.assertEqual(claims, [("cpu", "gpu"), ("cpu", "gpu"), ("cpu",), ("cpu",)])

    def test_supply_claims_exclude_only_the_pressured_resource_profile(self) -> None:
        snapshot = HostSnapshot(64, 10, 10, 80)
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 3,
                "max_concurrent_experiments": 3,
                "io_pressure_high_pct": 30,
                "profiles": {
                    "cpu": {"accelerator": "cpu", "pressure_domains": ["cpu"]},
                    "io": {"accelerator": "cpu", "pressure_domains": ["io"]},
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            self.assertEqual(
                [
                    allocator.claim_supply(
                        lease_id=f"io-lease-{index}",
                        run_id="run",
                        expires_at=time.time() + 60,
                    )
                    for index in range(3)
                ],
                [("cpu",), ("cpu",), ("cpu",)],
            )

    def test_supply_uses_open_slots_until_high_pressure(self) -> None:
        snapshot = HostSnapshot(64, 80, 10, 0)
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 2,
                "max_concurrent_experiments": 4,
                "cpu_low_pct": 65,
                "cpu_high_pct": 92,
                "adjustment_interval_seconds": 2,
                "profiles": {"cpu": {"accelerator": "cpu", "pressure_domains": ["cpu"]}},
            }
        )
        with tempfile.TemporaryDirectory() as td:
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            allocator._last_adjustment = -100
            allocator.refresh(queued=True)
            self.assertEqual(allocator.concurrency_limit, 2)
            self.assertEqual(
                [
                    allocator.claim_supply(
                        lease_id=f"cpu-lease-{index}",
                        run_id="run",
                        expires_at=time.time() + 60,
                    )
                    for index in range(2)
                ],
                [("cpu",), ("cpu",)],
            )

    def test_supply_settings_are_bounded_and_enabled_by_default(self) -> None:
        defaults = SchedulerSettings.from_dict({"mode": "central"})
        low = SchedulerSettings.from_dict(
            {"mode": "central", "supply_signal_enabled": False, "supply_idle_samples": 0}
        )
        high = SchedulerSettings.from_dict({"mode": "central", "supply_idle_samples": 999})
        short_lease = SchedulerSettings.from_dict({"mode": "central", "supply_lease_seconds": 1})
        long_lease = SchedulerSettings.from_dict({"mode": "central", "supply_lease_seconds": 99999})
        self.assertTrue(defaults.supply_signal_enabled)
        self.assertEqual(defaults.supply_idle_samples, 3)
        self.assertEqual(defaults.supply_lease_seconds, 600)
        self.assertFalse(low.supply_signal_enabled)
        self.assertEqual(low.supply_idle_samples, 2)
        self.assertEqual(high.supply_idle_samples, 12)
        self.assertEqual(short_lease.supply_lease_seconds, 180)
        self.assertEqual(long_lease.supply_lease_seconds, 3600)

    def test_supply_claims_respect_existing_host_allocations_per_profile(self) -> None:
        snapshot = HostSnapshot(
            32,
            10,
            10,
            0,
            (GPUDevice(0, "GPU-a", 80 * 1024, 0, 0),),
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 3,
                "max_concurrent_experiments": 3,
                "profiles": {
                    "cpu": {"accelerator": "cpu", "pressure_domains": ["cpu"]},
                    "gpu": {"accelerator": "gpu", "pressure_domains": ["gpu"]},
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            registry.path.write_text(
                json.dumps(
                    [
                        {
                            "record_type": "allocation",
                            "allocation_id": "other-run-gpu",
                            "pending": False,
                            "run_id": "other",
                            "pid": os.getpid(),
                            "pgid": os.getpgrp(),
                            "gpu_uuids": ["GPU-a"],
                            "gpu_memory_mb": 0,
                            "gpu_utilization_pct": 0,
                            "started_at": snapshot.observed_at - 20,
                        }
                    ]
                )
            )
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=registry,
            )
            self.assertEqual(
                [
                    allocator.claim_supply(
                        lease_id=f"existing-lease-{index}",
                        run_id="run",
                        expires_at=time.time() + 60,
                    )
                    for index in range(2)
                ],
                [("cpu",), ("cpu",)],
            )

    def test_supply_claims_are_atomic_across_independent_runs(self) -> None:
        snapshot = HostSnapshot(16, 10, 10, 0)
        settings = _settings(maximum=2)
        with tempfile.TemporaryDirectory() as td:
            registry_a = HostAllocationRegistry(Path(td))
            registry_b = HostAllocationRegistry(Path(td))
            first = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=registry_a,
            )
            second = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=registry_b,
            )
            results = [
                first.claim_supply(lease_id="run-a-1", run_id="run-a", expires_at=time.time() + 60),
                second.claim_supply(
                    lease_id="run-b-1", run_id="run-b", expires_at=time.time() + 60
                ),
                first.claim_supply(lease_id="run-a-2", run_id="run-a", expires_at=time.time() + 60),
                first.claim_supply(lease_id="run-a-3", run_id="run-a", expires_at=time.time() + 60),
            ]
            self.assertEqual(results, [("cpu",), ("cpu",), (), ()])

    def test_per_run_overflow_revokes_own_supply_not_another_runs_claim(self) -> None:
        snapshot = HostSnapshot(16, 10, 10, 0)
        with tempfile.TemporaryDirectory() as td:
            first_settings = _settings(maximum=1)
            second_settings = _settings(maximum=2)
            first = ResourceAllocator(
                first_settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            second = ResourceAllocator(
                second_settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            first.set_owner("run:a")
            second.set_owner("run:b")
            self.assertTrue(
                second.claim_supply(
                    lease_id="b-supply",
                    run_id="supply:b",
                    expires_at=time.time() + 60,
                )
            )
            self.assertTrue(
                first.claim_supply(
                    lease_id="a-supply",
                    run_id="supply:a",
                    expires_at=time.time() + 60,
                )
            )
            allocation = first.reserve(
                allocation_id="a-real",
                run_id="run:a",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=first_settings.profile("cpu"),
            )
            self.assertIsNotNone(allocation)
            rows = json.loads(first.registry.path.read_text())
            ids = {row.get("allocation_id") for row in rows}
            self.assertIn("b-supply", ids)
            self.assertIn("a-real", ids)
            self.assertNotIn("a-supply", ids)

    def test_waiting_recovery_rechecks_host_capacity(self) -> None:
        snapshot = HostSnapshot(16, 10, 10, 0)
        settings = _settings(maximum=1)
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=registry,
            )
            allocator.set_owner("run:current")
            self.assertIsNotNone(
                allocator.reserve(
                    allocation_id="existing",
                    run_id="run:other",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=settings.profile("cpu"),
                )
            )
            self.assertIsNone(
                allocator.recover_allocation(
                    allocation_id="waiting",
                    run_id="run:current",
                    pid=os.getpid(),
                    pgid=os.getpgrp(),
                    profile=settings.profile("cpu"),
                    gpu_uuids=(),
                    require_admission=True,
                )
            )

    def test_supply_claim_promotes_to_pending_allocation_without_a_gap(self) -> None:
        snapshot = HostSnapshot(16, 10, 10, 0)
        settings = _settings(maximum=1)
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=registry,
            )
            self.assertEqual(
                allocator.claim_supply(
                    lease_id="claim",
                    run_id="run",
                    expires_at=time.time() + 60,
                ),
                ("cpu",),
            )
            allocation = allocator.reserve(
                allocation_id="real-allocation",
                run_id="run",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("cpu"),
                supply_claim_id="claim",
            )
            self.assertIsNotNone(allocation)
            rows = json.loads(registry.path.read_text())
            self.assertEqual([row["allocation_id"] for row in rows], ["real-allocation"])
            self.assertEqual(rows[0]["record_type"], "allocation")

    def test_cpu_work_preempts_speculative_supply_at_host_limit(self) -> None:
        snapshot = HostSnapshot(16, 10, 10, 0)
        settings = _settings(maximum=1)
        with tempfile.TemporaryDirectory() as td:
            registry_a = HostAllocationRegistry(Path(td))
            registry_b = HostAllocationRegistry(Path(td))
            supplier = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=registry_a,
            )
            worker = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=registry_b,
            )
            self.assertEqual(
                supplier.claim_supply(
                    lease_id="speculative",
                    run_id="run-a",
                    expires_at=time.time() + 60,
                ),
                ("cpu",),
            )
            allocation = worker.reserve(
                allocation_id="submitted-work",
                run_id="run-b",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("cpu"),
            )
            self.assertIsNotNone(allocation)
            rows = json.loads(registry_a.path.read_text())
            self.assertEqual(
                [row["allocation_id"] for row in rows],
                ["submitted-work"],
            )
            self.assertFalse(supplier.supply_claim_valid("speculative"))

    def test_real_work_preserves_nonconflicting_supply_claims_when_capacity_remains(self) -> None:
        snapshot = HostSnapshot(16, 10, 10, 0)
        settings = _settings(maximum=3)
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=registry,
            )
            for index in range(2):
                self.assertEqual(
                    allocator.claim_supply(
                        lease_id=f"claim-{index}",
                        run_id=f"run-{index}",
                        expires_at=time.time() + 60,
                    ),
                    ("cpu",),
                )
            allocation = allocator.reserve(
                allocation_id="actual",
                run_id="actual-run",
                pid=os.getpid(),
                pgid=os.getpgrp(),
                profile=settings.profile("cpu"),
            )
            self.assertIsNotNone(allocation)
            rows = json.loads(registry.path.read_text())
            self.assertEqual(
                {row["allocation_id"] for row in rows},
                {"claim-0", "claim-1", "actual"},
            )

    def test_supply_claim_remains_stable_when_profile_pressure_rises(self) -> None:
        low = HostSnapshot(16, 10, 10, 0)
        high = HostSnapshot(16, 99, 10, 0)
        settings = _settings(maximum=1)
        with tempfile.TemporaryDirectory() as td:
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([low]),
                registry=HostAllocationRegistry(Path(td)),
            )
            self.assertEqual(
                allocator.claim_supply(
                    lease_id="pressure-lease",
                    run_id="run",
                    expires_at=time.time() + 60,
                ),
                ("cpu",),
            )
            allocator._last_snapshot = high
            self.assertTrue(allocator.supply_claim_valid("pressure-lease"))

    def test_supply_claim_remains_stable_across_advertised_profile_pressure(self) -> None:
        low = HostSnapshot(
            16,
            10,
            10,
            0,
            (GPUDevice(0, "GPU-a", 80 * 1024, 0, 0),),
        )
        high_cpu = HostSnapshot(
            16,
            99,
            10,
            0,
            (GPUDevice(0, "GPU-a", 80 * 1024, 0, 0),),
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 1,
                "max_concurrent_experiments": 1,
                "profiles": {
                    "cpu": {"accelerator": "cpu", "pressure_domains": ["cpu"]},
                    "gpu": {
                        "accelerator": "gpu",
                        "gpu_memory_gb": 10,
                        "gpu_utilization_pct": 20,
                        "pressure_domains": ["gpu"],
                    },
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([low]),
                registry=HostAllocationRegistry(Path(td)),
            )
            self.assertEqual(
                allocator.claim_supply(
                    lease_id="mixed-pressure",
                    run_id="run",
                    expires_at=time.time() + 60,
                ),
                ("cpu", "gpu"),
            )
            allocator._last_snapshot = high_cpu
            self.assertTrue(allocator.supply_claim_valid("mixed-pressure"))

    def test_dead_supply_owner_is_not_kept_alive_by_shared_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            registry = HostAllocationRegistry(Path(td))
            registry.path.write_text(
                json.dumps(
                    [
                        {
                            "record_type": "supply",
                            "allocation_id": "dead-claim",
                            "run_id": "dead",
                            "pid": 999_999_999,
                            "pgid": os.getpgrp(),
                            "expires_at": time.time() + 60,
                        }
                    ]
                )
            )
            with registry.locked() as rows:
                self.assertEqual(rows, [])

    def test_supply_claims_do_not_advertise_incompatible_gpu_envelopes(self) -> None:
        snapshot = HostSnapshot(
            16,
            10,
            10,
            0,
            (
                GPUDevice(0, "GPU-a", 80 * 1024, 0, 0),
                GPUDevice(1, "GPU-b", 80 * 1024, 0, 0),
            ),
        )
        settings = SchedulerSettings.from_dict(
            {
                "mode": "central",
                "initial_concurrent_experiments": 2,
                "max_concurrent_experiments": 2,
                "profiles": {
                    "cpu": {"accelerator": "cpu"},
                    "heavy_one": {
                        "accelerator": "gpu",
                        "gpu_count": 1,
                        "gpu_memory_gb": 70,
                        "gpu_utilization_pct": 20,
                        "pressure_domains": ["gpu"],
                    },
                    "wide": {
                        "accelerator": "gpu",
                        "gpu_count": 2,
                        "gpu_memory_gb": 20,
                        "gpu_utilization_pct": 60,
                        "pressure_domains": ["gpu"],
                    },
                },
            }
        )
        with tempfile.TemporaryDirectory() as td:
            allocator = ResourceAllocator(
                settings,
                observer=_Observer([snapshot]),
                registry=HostAllocationRegistry(Path(td)),
            )
            first = allocator.claim_supply(
                lease_id="mixed-1", run_id="run", expires_at=time.time() + 60
            )
            second = allocator.claim_supply(
                lease_id="mixed-2", run_id="run", expires_at=time.time() + 60
            )
            self.assertEqual(first, ("cpu", "wide"))
            self.assertEqual(second, ("cpu",))
