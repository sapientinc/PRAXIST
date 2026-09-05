"""Small, host-aware admission controller for task experiments.

The scheduler deliberately owns only resource admission.  Scientific maturity,
frontier promotion, and generation policy remain research-loop concerns.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _posix_file_locking() -> Any:
    """Load POSIX locking only when the host allocation ledger is used."""

    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError(
            "Praxist central resource scheduling requires POSIX file locking; "
            "use Linux, macOS, or WSL for research runs"
        ) from exc
    return fcntl


_SCHEDULER_KEYS = {
    "mode",
    "initial_concurrent_experiments",
    "min_concurrent_experiments",
    "max_concurrent_experiments",
    "cpu_low_pct",
    "cpu_high_pct",
    "memory_high_pct",
    "io_pressure_high_pct",
    "adjustment_interval_seconds",
    "exploration_reserve",
    "infrastructure_retries",
    "deadline_admission",
    "supply_signal_enabled",
    "supply_idle_samples",
    "supply_lease_seconds",
    "mature_supply_fraction",
    "mature_supply_redundancy",
    "mature_assessment_min_completion_probability",
    "default_profile",
    "profiles",
}
_PROFILE_KEYS = {
    "accelerator",
    "gpu_count",
    "gpu_memory_gb",
    "gpu_utilization_pct",
    "pressure_domains",
}
_PRESSURE_DOMAINS = {"cpu", "memory", "io", "gpu"}


@dataclass(frozen=True)
class GPUDevice:
    """Observed capacity and load for one physical GPU."""

    index: int
    uuid: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_pct: float


@dataclass(frozen=True)
class GPUProcess:
    """Driver-observed process load used to separate Praxist and external work."""

    pid: int
    uuid: str
    memory_used_mb: int
    utilization_pct: float | None = None


@dataclass(frozen=True)
class HostSnapshot:
    """One host pressure sample used for admission decisions."""

    cpu_count: int
    cpu_utilization_pct: float
    memory_utilization_pct: float
    io_pressure_pct: float
    gpus: tuple[GPUDevice, ...] = ()
    gpu_processes: tuple[GPUProcess, ...] = ()
    gpu_process_memory_observed: bool = False
    gpu_process_utilization_observed: bool = False
    observed_at: float = field(default_factory=time.time)
    accelerator_probe_state: str = "unknown"
    accelerator_probe_reason: str = ""


@dataclass(frozen=True)
class ResourceProfile:
    """Task-declared resource envelope for one experiment class."""

    name: str
    accelerator: str = "cpu"
    gpu_count: int = 0
    gpu_memory_gb: float | None = None
    gpu_utilization_pct: float | None = None
    pressure_domains: tuple[str, ...] = ("cpu", "memory")

    @property
    def needs_gpu(self) -> bool:
        return self.accelerator == "gpu" and self.gpu_count > 0


@dataclass
class SchedulerSettings:
    """Validated central scheduler policy derived from the task specification."""

    enabled: bool = False
    initial_concurrency: int = 1
    min_concurrency: int = 1
    max_concurrency: int = 1
    cpu_low_pct: float = 65.0
    cpu_high_pct: float = 92.0
    memory_high_pct: float = 95.0
    io_pressure_high_pct: float = 35.0
    adjustment_interval_seconds: float = 20.0
    exploration_reserve: int = 1
    infrastructure_retries: int = 1
    deadline_admission: bool = True
    supply_signal_enabled: bool = True
    supply_idle_samples: int = 3
    supply_lease_seconds: int = 600
    mature_supply_fraction: float = 0.25
    mature_supply_redundancy: float = 3.0
    mature_assessment_min_completion_probability: float = 0.25
    default_profile: str = "default"
    profiles: dict[str, ResourceProfile] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SchedulerSettings:
        raw = raw if isinstance(raw, dict) else {}
        mode = str(raw.get("mode", "legacy")).strip().lower()
        if mode not in {"legacy", "central"}:
            raise ValueError(
                f"unknown resource scheduler mode {mode!r}; expected 'legacy' or 'central'"
            )
        if mode == "central":
            unknown = sorted(set(raw) - _SCHEDULER_KEYS)
            if unknown:
                raise ValueError(
                    "unknown central resource scheduler field(s): " + ", ".join(unknown)
                )
        maximum = max(1, _config_int(raw, "max_concurrent_experiments", 1, mode=mode))
        minimum = min(
            maximum,
            max(1, _config_int(raw, "min_concurrent_experiments", 1, mode=mode)),
        )
        initial = min(
            maximum,
            max(
                minimum,
                _config_int(raw, "initial_concurrent_experiments", minimum, mode=mode),
            ),
        )
        profiles_raw = raw.get("profiles")
        profiles: dict[str, ResourceProfile] = {}
        if profiles_raw is not None and not isinstance(profiles_raw, dict):
            raise ValueError("resource scheduler profiles must be an object")
        if isinstance(profiles_raw, dict):
            for name, value in profiles_raw.items():
                if not isinstance(value, dict):
                    if mode == "central":
                        raise ValueError(f"resource profile {name!r} must be an object")
                    continue
                if mode == "central":
                    unknown_profile = sorted(set(value) - _PROFILE_KEYS)
                    if unknown_profile:
                        raise ValueError(
                            f"resource profile {name!r} has unknown field(s): "
                            + ", ".join(unknown_profile)
                        )
                if mode == "central" and not isinstance(name, str):
                    raise ValueError("resource profile names must be strings")
                profile_name = str(name).strip()
                if not profile_name:
                    if mode == "central":
                        raise ValueError("resource profile names must be non-empty")
                    continue
                accelerator = str(value.get("accelerator", "cpu")).strip().lower()
                if accelerator not in {"cpu", "gpu"}:
                    raise ValueError(
                        f"resource profile {profile_name!r} has invalid accelerator "
                        f"{accelerator!r}; expected 'cpu' or 'gpu'"
                    )
                gpu_count = max(
                    1 if accelerator == "gpu" else 0,
                    _config_int(
                        value,
                        "gpu_count",
                        1 if accelerator == "gpu" else 0,
                        mode=mode,
                        prefix=f"resource profile {profile_name}",
                    ),
                )
                domains = value.get("pressure_domains", ["cpu", "memory"])
                if not isinstance(domains, list):
                    if mode == "central":
                        raise ValueError(
                            f"resource profile {profile_name!r} pressure_domains must be a list"
                        )
                    domains = ["cpu", "memory"]
                normalized_domains = tuple(
                    dict.fromkeys(
                        str(item).strip().lower() for item in domains if str(item).strip()
                    )
                )
                if mode == "central":
                    unknown_domains = sorted(set(normalized_domains) - _PRESSURE_DOMAINS)
                    if unknown_domains:
                        raise ValueError(
                            f"resource profile {profile_name!r} has unknown pressure domain(s): "
                            + ", ".join(unknown_domains)
                        )
                profiles[profile_name] = ResourceProfile(
                    name=profile_name,
                    accelerator=accelerator,
                    gpu_count=gpu_count if accelerator == "gpu" else 0,
                    gpu_memory_gb=_config_optional_positive_float(
                        value,
                        "gpu_memory_gb",
                        mode=mode,
                        prefix=f"resource profile {profile_name}",
                    ),
                    gpu_utilization_pct=_config_optional_positive_float(
                        value,
                        "gpu_utilization_pct",
                        mode=mode,
                        prefix=f"resource profile {profile_name}",
                        maximum=100.0,
                    ),
                    pressure_domains=normalized_domains,
                )
        if not profiles:
            profiles["default"] = ResourceProfile(name="default")
        explicit_default = raw.get("default_profile")
        default_profile = (
            str(explicit_default).strip() if explicit_default is not None else next(iter(profiles))
        )
        if mode == "central" and explicit_default is not None and not default_profile:
            raise ValueError("default resource profile must be non-empty")
        default_profile = default_profile or next(iter(profiles))
        if default_profile not in profiles:
            raise ValueError(f"default resource profile {default_profile!r} is not declared")
        return cls(
            enabled=mode == "central",
            initial_concurrency=initial,
            min_concurrency=minimum,
            max_concurrency=maximum,
            cpu_low_pct=_config_bounded_float(raw, "cpu_low_pct", 65.0, 0.0, 100.0, mode=mode),
            cpu_high_pct=_config_bounded_float(raw, "cpu_high_pct", 92.0, 0.0, 100.0, mode=mode),
            memory_high_pct=_config_bounded_float(
                raw, "memory_high_pct", 95.0, 0.0, 100.0, mode=mode
            ),
            io_pressure_high_pct=_config_bounded_float(
                raw, "io_pressure_high_pct", 35.0, 0.0, 100.0, mode=mode
            ),
            adjustment_interval_seconds=max(
                2.0,
                _config_float(raw, "adjustment_interval_seconds", 20.0, mode=mode),
            ),
            exploration_reserve=max(0, _config_int(raw, "exploration_reserve", 1, mode=mode)),
            infrastructure_retries=max(0, _config_int(raw, "infrastructure_retries", 1, mode=mode)),
            deadline_admission=_config_bool(raw, "deadline_admission", True, mode=mode),
            supply_signal_enabled=_config_bool(raw, "supply_signal_enabled", True, mode=mode),
            supply_idle_samples=min(
                12, max(2, _config_int(raw, "supply_idle_samples", 3, mode=mode))
            ),
            supply_lease_seconds=min(
                3600,
                max(180, _config_int(raw, "supply_lease_seconds", 600, mode=mode)),
            ),
            mature_supply_fraction=_config_bounded_float(
                raw,
                "mature_supply_fraction",
                0.25,
                0.0,
                1.0,
                mode=mode,
            ),
            mature_supply_redundancy=_config_bounded_float(
                raw,
                "mature_supply_redundancy",
                3.0,
                0.0,
                10.0,
                mode=mode,
            ),
            mature_assessment_min_completion_probability=_config_bounded_float(
                raw,
                "mature_assessment_min_completion_probability",
                0.25,
                0.25,
                1.0,
                mode=mode,
            ),
            default_profile=default_profile,
            profiles=profiles,
        )

    def profile(self, name: str | None) -> ResourceProfile:
        return self.profiles.get(str(name or ""), self.profiles[self.default_profile])


def _as_int(value: object, default: int) -> int:
    try:
        if isinstance(value, (str, bytes, int, float)):
            return int(value)
    except (TypeError, ValueError):
        pass
    return default


def _config_int(
    raw: dict[str, Any],
    key: str,
    default: int,
    *,
    mode: str,
    prefix: str = "resource scheduler",
) -> int:
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool):
        if mode == "central":
            raise ValueError(f"{prefix} {key} must be an integer")
        return default
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.strip())
    except (TypeError, ValueError):
        pass
    if mode == "central":
        raise ValueError(f"{prefix} {key} must be an integer")
    return default


def _as_float(value: object, default: float) -> float:
    try:
        if not isinstance(value, (str, bytes, int, float)):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _config_float(raw: dict[str, Any], key: str, default: float, *, mode: str) -> float:
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool):
        if mode == "central":
            raise ValueError(f"resource scheduler {key} must be a finite number")
        return default
    try:
        result = float(value) if isinstance(value, (str, int, float)) else float("nan")
    except (TypeError, ValueError):
        result = float("nan")
    if math.isfinite(result):
        return result
    if mode == "central":
        raise ValueError(f"resource scheduler {key} must be a finite number")
    return default


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _config_bool(raw: dict[str, Any], key: str, default: bool, *, mode: str) -> bool:
    if key not in raw:
        return default
    value = raw[key]
    parsed = _as_bool(value, default)
    if isinstance(value, bool):
        return parsed
    if isinstance(value, str) and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "0",
        "false",
        "no",
        "off",
    }:
        return parsed
    if mode == "central":
        raise ValueError(f"resource scheduler {key} must be a boolean")
    return default


def _bounded_float(value: object, default: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, _as_float(value, default)))


def _config_bounded_float(
    raw: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
    *,
    mode: str,
) -> float:
    return min(
        maximum,
        max(minimum, _config_float(raw, key, default, mode=mode)),
    )


def _optional_positive_float(value: object, maximum: float | None = None) -> float | None:
    if value is None:
        return None
    parsed = _as_float(value, -1.0)
    if parsed <= 0:
        return None
    return min(parsed, maximum) if maximum is not None else parsed


def _config_optional_positive_float(
    raw: dict[str, Any],
    key: str,
    *,
    mode: str,
    prefix: str,
    maximum: float | None = None,
) -> float | None:
    if key not in raw:
        return None
    value = raw[key]
    if isinstance(value, bool):
        if mode == "central":
            raise ValueError(f"{prefix} {key} must be a positive finite number")
        return None
    try:
        parsed = float(value) if isinstance(value, (str, int, float)) else float("nan")
    except (TypeError, ValueError):
        parsed = float("nan")
    if not math.isfinite(parsed) or parsed <= 0:
        if mode == "central":
            raise ValueError(f"{prefix} {key} must be a positive finite number")
        return None
    return min(parsed, maximum) if maximum is not None else parsed


class HostObserver:
    """Read Linux host pressure and accelerator state without extra dependencies."""

    def __init__(self) -> None:
        self._last_cpu: tuple[int, int] | None = None

    def snapshot(self) -> HostSnapshot:
        gpu_rows, probe_state, probe_reason = self._query_gpus()
        gpus = tuple(gpu_rows)
        gpu_processes, memory_observed, utilization_observed = self._gpu_processes(gpus)
        return HostSnapshot(
            cpu_count=max(1, os.cpu_count() or 1),
            cpu_utilization_pct=self._cpu_utilization(),
            memory_utilization_pct=self._memory_utilization(),
            io_pressure_pct=self._io_pressure(),
            gpus=gpus,
            gpu_processes=gpu_processes,
            gpu_process_memory_observed=memory_observed,
            gpu_process_utilization_observed=utilization_observed,
            accelerator_probe_state=probe_state,
            accelerator_probe_reason=probe_reason,
        )

    def _cpu_utilization(self) -> float:
        try:
            fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
            values = [int(item) for item in fields]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            previous = self._last_cpu
            self._last_cpu = (idle, total)
            if previous is not None and total > previous[1]:
                busy_delta = (total - previous[1]) - (idle - previous[0])
                return _bounded_float(100.0 * busy_delta / (total - previous[1]), 0.0, 0, 100)
        except (OSError, ValueError, IndexError, ZeroDivisionError):
            pass
        try:
            return min(100.0, os.getloadavg()[0] * 100.0 / max(1, os.cpu_count() or 1))
        except OSError:
            return 0.0

    @staticmethod
    def _memory_utilization() -> float:
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0])
            total = values["MemTotal"]
            available = values.get("MemAvailable", values.get("MemFree", 0))
            return 100.0 * max(0, total - available) / total
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _io_pressure() -> float:
        try:
            text = Path("/proc/pressure/io").read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.startswith("some "):
                    for token in line.split():
                        if token.startswith("avg10="):
                            return _bounded_float(token.split("=", 1)[1], 0.0, 0.0, 100.0)
        except OSError:
            pass
        return 0.0

    @staticmethod
    def _gpus() -> list[GPUDevice]:
        devices, _state, _reason = HostObserver._query_gpus()
        return devices

    @staticmethod
    def _query_gpus() -> tuple[list[GPUDevice], str, str]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        except FileNotFoundError:
            if shutil.which("rocm-smi"):
                return (
                    [],
                    "unsupported",
                    "ROCm device tooling detected; NVIDIA placement unavailable",
                )
            return [], "unavailable", "nvidia-smi is not installed"
        except subprocess.TimeoutExpired:
            return [], "unknown", "nvidia-smi inventory probe timed out"
        except OSError as exc:
            return [], "unknown", f"nvidia-smi inventory probe failed: {exc}"
        if result.returncode != 0:
            if shutil.which("rocm-smi"):
                return (
                    [],
                    "unsupported",
                    "ROCm device tooling detected; NVIDIA placement unavailable",
                )
            detail = (
                str(getattr(result, "stderr", "") or getattr(result, "stdout", "") or "")
                .strip()
                .splitlines()
            )
            reason = detail[0] if detail else f"nvidia-smi exited {result.returncode}"
            return [], "unknown", reason
        devices: list[GPUDevice] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                continue
            try:
                devices.append(
                    GPUDevice(
                        index=int(parts[0]),
                        uuid=parts[1],
                        memory_total_mb=int(float(parts[2])),
                        memory_used_mb=int(float(parts[3])),
                        utilization_pct=float(parts[4]),
                    )
                )
            except ValueError:
                continue
        if devices:
            return devices, "available", ""
        return [], "unavailable", "nvidia-smi reported no usable devices"

    @staticmethod
    def _gpu_processes(
        gpus: tuple[GPUDevice, ...],
    ) -> tuple[tuple[GPUProcess, ...], bool, bool]:
        command = [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return (), False, False
        if result.returncode != 0:
            return (), False, False
        rows: dict[tuple[int, str], dict[str, float | int | str | None]] = {}
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                pid = int(parts[0])
                memory = int(float(parts[2]))
            except ValueError:
                continue
            rows[(pid, parts[1])] = {
                "pid": pid,
                "uuid": parts[1],
                "memory_used_mb": memory,
                "utilization_pct": None,
            }

        utilization_sampled = False
        index_to_uuid = {device.index: device.uuid for device in gpus}
        try:
            pmon = subprocess.run(
                ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pmon = None
        if pmon is not None and pmon.returncode == 0:
            utilization_sampled = True
            for line in pmon.stdout.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    index = int(parts[0])
                    pid = int(parts[1])
                    utilization = float(parts[3])
                except ValueError:
                    continue
                key = (pid, index_to_uuid.get(index, ""))
                if key in rows:
                    rows[key]["utilization_pct"] = utilization
        utilization_observed = utilization_sampled and all(
            row.get("utilization_pct") is not None for row in rows.values()
        )
        return (
            tuple(
                GPUProcess(
                    pid=_as_int(row["pid"], 0),
                    uuid=str(row["uuid"]),
                    memory_used_mb=_as_int(row["memory_used_mb"], 0),
                    utilization_pct=(
                        None
                        if row.get("utilization_pct") is None
                        else float(row["utilization_pct"])
                    ),
                )
                for row in rows.values()
            ),
            True,
            utilization_observed,
        )


@dataclass(frozen=True)
class Allocation:
    """Durable host reservation bound to one experiment process group."""

    allocation_id: str
    run_id: str
    pid: int
    pgid: int
    profile: str
    gpu_uuids: tuple[str, ...]
    gpu_memory_mb: int
    gpu_utilization_pct: float
    started_at: float
    gpu_demand_unknown: bool = False


class HostAllocationRegistry:
    """Atomic host-local allocation ledger shared by independent Praxist runs."""

    def __init__(self, root: Path | None = None) -> None:
        uid = getattr(os, "getuid", lambda: 0)()
        self.root = root or Path(f"/tmp/praxist-resources-{uid}")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "allocations.json"
        self.lock_path = self.root / "allocations.lock"

    @contextlib.contextmanager
    def locked(self):
        file_locking = _posix_file_locking()
        fd = os.open(
            self.lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            file_locking.flock(fd, file_locking.LOCK_EX)
            allocations = self._read_and_prune()
            yield allocations
            self._write(allocations)
        finally:
            file_locking.flock(fd, file_locking.LOCK_UN)
            os.close(fd)

    def _read_and_prune(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = []
        if not isinstance(raw, list):
            raw = []
        return [item for item in raw if isinstance(item, dict) and _process_group_alive(item)]

    def _write(self, allocations: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(allocations, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)

    def update_limit(self, *, run_id: str, pid: int, pgid: int, limit: int) -> None:
        with self.locked() as rows:
            rows[:] = [
                row
                for row in rows
                if not (row.get("record_type") == "controller" and row.get("run_id") == run_id)
            ]
            rows.append(
                {
                    "record_type": "controller",
                    "run_id": run_id,
                    "pid": int(pid),
                    "pgid": int(pgid),
                    "pid_start_time": _pid_start_time(int(pid)),
                    "concurrency_limit": max(1, int(limit)),
                    "updated_at": time.time(),
                }
            )

    def remove_limit(self, run_id: str) -> None:
        with self.locked() as rows:
            rows[:] = [
                row
                for row in rows
                if not (row.get("record_type") == "controller" and row.get("run_id") == run_id)
            ]

    def remove_supply(self, run_id: str) -> None:
        with self.locked() as rows:
            rows[:] = [
                row
                for row in rows
                if not (row.get("record_type") == "supply" and row.get("run_id") == run_id)
            ]


def _process_group_alive(item: dict[str, Any]) -> bool:
    if (
        item.get("record_type") == "supply"
        and _as_float(item.get("expires_at"), 0.0) <= time.time()
    ):
        return False
    pgid = _as_int(item.get("pgid"), 0)
    pid = _as_int(item.get("pid"), 0)
    if item.get("record_type") in {"controller", "supply"}:
        pgid = 0
        recorded_start = item.get("pid_start_time")
        if recorded_start is not None and recorded_start != _pid_start_time(pid):
            return False
    elif item.get("pid_start_time") is not None:
        current_start = _pid_start_time(pid)
        if current_start is not None and current_start != item.get("pid_start_time"):
            return False
    if item.get("pending") and time.time() - _as_float(item.get("started_at"), 0.0) > 60.0:
        return False
    for value, group in ((pgid, True), (pid, False)):
        if value <= 1:
            continue
        try:
            os.killpg(value, 0) if group else os.kill(value, 0)
            return True
        except ProcessLookupError:
            continue
        except PermissionError:
            return True
    return False


def _pid_start_time(pid: int) -> int | str | None:
    try:
        suffix = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        return int(suffix.split()[19])
    except (OSError, ValueError, IndexError):
        from praxist.cli.registry import process_start_token

        return process_start_token(pid) or None


def _resource_owner_group(run_id: str) -> set[str]:
    """Return the controller and speculative-supply identities for one run."""

    text = str(run_id)
    suffix = text.split(":", 1)[1] if text.startswith(("run:", "supply:")) else text
    return {text, f"run:{suffix}", f"supply:{suffix}"}


class ResourceAllocator:
    """Choose admission and GPU placement from live pressure plus reservations."""

    def __init__(
        self,
        settings: SchedulerSettings,
        *,
        observer: HostObserver | None = None,
        registry: HostAllocationRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.observer = observer or HostObserver()
        self.registry = registry or HostAllocationRegistry()
        self.concurrency_limit = settings.initial_concurrency
        self._last_adjustment = 0.0
        self._last_snapshot = self.observer.snapshot()
        self._run_id = ""

    def set_owner(self, run_id: str) -> None:
        self._run_id = str(run_id)
        self._publish_limit()

    def close(self) -> None:
        if self._run_id:
            self.registry.remove_limit(self._run_id)

    @property
    def snapshot(self) -> HostSnapshot:
        return self._last_snapshot

    def describe_allocation_activity(self, allocation: Allocation) -> dict[str, Any]:
        """Describe the observed resource phase without changing admission state."""

        if not allocation.gpu_uuids:
            return {
                "state": "non_gpu_allocation",
                "attribution": "not_applicable",
                "gpu_processes": 0,
                "gpu_memory_mb": 0,
                "gpu_utilization_pct": None,
                "observed_at": self._last_snapshot.observed_at,
            }
        snapshot = self._last_snapshot
        if not (snapshot.gpu_process_memory_observed or snapshot.gpu_process_utilization_observed):
            return {
                "state": "unknown",
                "attribution": "unavailable",
                "gpu_processes": 0,
                "gpu_memory_mb": None,
                "gpu_utilization_pct": None,
                "observed_at": snapshot.observed_at,
            }

        assigned = set(allocation.gpu_uuids)
        matched: list[GPUProcess] = []
        mapping_errors = 0
        for process in snapshot.gpu_processes:
            if process.uuid not in assigned:
                continue
            try:
                if os.getpgid(process.pid) == allocation.pgid:
                    matched.append(process)
            except (OSError, ProcessLookupError):
                mapping_errors += 1
        if not matched:
            return {
                "state": (
                    "gpu_process_attribution_unavailable"
                    if mapping_errors
                    else "no_gpu_process_observed"
                ),
                "attribution": "unavailable" if mapping_errors else "complete",
                "gpu_processes": None if mapping_errors else 0,
                "gpu_memory_mb": (
                    None if mapping_errors or not snapshot.gpu_process_memory_observed else 0
                ),
                "gpu_utilization_pct": (
                    None if mapping_errors or not snapshot.gpu_process_utilization_observed else 0.0
                ),
                "observed_at": snapshot.observed_at,
            }

        observed_utilization = [
            process.utilization_pct for process in matched if process.utilization_pct is not None
        ]
        utilization = (
            sum(observed_utilization)
            if snapshot.gpu_process_utilization_observed and observed_utilization
            else None
        )
        if utilization is None:
            state = "gpu_context_present"
        elif utilization > 0:
            state = "gpu_compute_active"
        else:
            state = "gpu_context_idle"
        return {
            "state": state,
            "attribution": "partial" if mapping_errors else "complete",
            "gpu_processes": len(matched),
            "gpu_memory_mb": (
                sum(process.memory_used_mb for process in matched)
                if snapshot.gpu_process_memory_observed
                else None
            ),
            "gpu_utilization_pct": utilization,
            "observed_at": snapshot.observed_at,
        }

    def refresh(self, *, queued: bool) -> HostSnapshot:
        now = time.monotonic()
        if now - self._last_adjustment < self.settings.adjustment_interval_seconds:
            return self._last_snapshot
        self._last_adjustment = now
        self._last_snapshot = self.observer.snapshot()
        snap = self._last_snapshot
        overloaded = (
            snap.cpu_utilization_pct >= self.settings.cpu_high_pct
            or snap.memory_utilization_pct >= self.settings.memory_high_pct
        )
        underloaded = (
            queued
            and snap.cpu_utilization_pct <= self.settings.cpu_low_pct
            and snap.memory_utilization_pct < self.settings.memory_high_pct - 5.0
        )
        if overloaded:
            self.concurrency_limit = max(self.settings.min_concurrency, self.concurrency_limit - 1)
        elif underloaded:
            self.concurrency_limit = min(self.settings.max_concurrency, self.concurrency_limit + 1)
        self._publish_limit()
        return snap

    def has_supply_headroom(
        self,
        snapshot: HostSnapshot | None = None,
        *,
        domains: set[str] | None = None,
    ) -> bool:
        """Return whether observed task pressure permits requesting more work."""

        snap = snapshot or self._last_snapshot
        relevant = domains or {
            domain
            for profile in self.settings.profiles.values()
            for domain in profile.pressure_domains
        }
        if "cpu" in relevant and snap.cpu_utilization_pct >= self.settings.cpu_high_pct:
            return False
        if "memory" in relevant and snap.memory_utilization_pct >= self.settings.memory_high_pct:
            return False
        return not ("io" in relevant and snap.io_pressure_pct >= self.settings.io_pressure_high_pct)

    def claim_supply(self, *, lease_id: str, run_id: str, expires_at: float) -> tuple[str, ...]:
        """Atomically claim one host-wide supply slot and return safe profile choices."""

        snapshot = self._last_snapshot
        with self.registry.locked() as active:
            allocations = [row for row in active if row.get("record_type") != "controller"]
            owner_group = _resource_owner_group(run_id)
            own_commitments = sum(row.get("run_id") in owner_group for row in allocations)
            if own_commitments >= self.concurrency_limit:
                return ()
            if len(allocations) >= self._host_concurrency_limit(active):
                return ()
            candidates: list[tuple[ResourceProfile, list[str]]] = []
            for profile in self.settings.profiles.values():
                if not self.has_supply_headroom(
                    snapshot,
                    domains=set(profile.pressure_domains),
                ):
                    continue
                gpu_uuids = self._choose_gpus(profile, snapshot, allocations)
                if profile.needs_gpu and gpu_uuids is None:
                    continue
                candidates.append((profile, list(gpu_uuids or ())))
            if not candidates:
                return ()

            reserved_profile, gpu_uuids = max(
                candidates,
                key=lambda item: self._supply_profile_priority(item[0]),
            )
            admissible = tuple(
                sorted(
                    profile.name
                    for profile, _gpu_uuids in candidates
                    if self._profile_fits_supply_claim(profile, reserved_profile)
                )
            )
            gpu_memory_mb = int((reserved_profile.gpu_memory_gb or 0.0) * 1024)
            gpu_utilization_pct = float(reserved_profile.gpu_utilization_pct or 0.0)
            if reserved_profile.needs_gpu and (
                reserved_profile.gpu_memory_gb is None
                or reserved_profile.gpu_utilization_pct is None
            ):
                devices = {device.uuid: device for device in snapshot.gpus}
                gpu_memory_mb = max(
                    (int(devices[uuid].memory_total_mb * 0.95) for uuid in gpu_uuids),
                    default=0,
                )
                gpu_utilization_pct = 100.0
            active.append(
                {
                    "record_type": "supply",
                    "allocation_id": lease_id,
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "pgid": os.getpgrp(),
                    "pid_start_time": _pid_start_time(os.getpid()),
                    "profile": reserved_profile.name,
                    "admissible_profiles": list(admissible),
                    "gpu_uuids": gpu_uuids,
                    "gpu_memory_mb": gpu_memory_mb,
                    "gpu_utilization_pct": gpu_utilization_pct,
                    "started_at": time.time(),
                    "expires_at": float(expires_at),
                }
            )
            return admissible

    def supply_claim_valid(self, lease_id: str) -> bool:
        """Return whether the bounded host-ledger claim is still active."""

        with self.registry.locked() as active:
            return any(
                row.get("record_type") == "supply" and row.get("allocation_id") == lease_id
                for row in active
            )

    def clear_supply(self, run_id: str) -> None:
        self.registry.remove_supply(run_id)

    @staticmethod
    def _supply_profile_priority(profile: ResourceProfile) -> tuple[float, ...]:
        unknown_gpu = profile.needs_gpu and (
            profile.gpu_memory_gb is None or profile.gpu_utilization_pct is None
        )
        return (
            float(profile.needs_gpu),
            float(unknown_gpu),
            float(profile.gpu_count),
            float(profile.gpu_memory_gb or 0.0),
            float(profile.gpu_utilization_pct or 0.0),
        )

    @staticmethod
    def _profile_fits_supply_claim(
        profile: ResourceProfile,
        reserved: ResourceProfile,
    ) -> bool:
        if not profile.needs_gpu:
            return True
        if not reserved.needs_gpu or profile.gpu_count > reserved.gpu_count:
            return False
        reserved_unknown = reserved.gpu_memory_gb is None or reserved.gpu_utilization_pct is None
        profile_unknown = profile.gpu_memory_gb is None or profile.gpu_utilization_pct is None
        if reserved_unknown:
            return True
        if profile_unknown:
            return False
        assert profile.gpu_memory_gb is not None
        assert reserved.gpu_memory_gb is not None
        assert profile.gpu_utilization_pct is not None
        assert reserved.gpu_utilization_pct is not None
        return bool(
            profile.gpu_memory_gb <= reserved.gpu_memory_gb
            and profile.gpu_utilization_pct <= reserved.gpu_utilization_pct
        )

    def _publish_limit(self) -> None:
        if not self._run_id:
            return
        self.registry.update_limit(
            run_id=self._run_id,
            pid=os.getpid(),
            pgid=os.getpgrp(),
            limit=self.concurrency_limit,
        )

    def reserve(
        self,
        *,
        allocation_id: str,
        run_id: str,
        pid: int,
        pgid: int,
        profile: ResourceProfile,
        supply_claim_id: str = "",
    ) -> Allocation | None:
        snapshot = self.observer.snapshot()
        self._last_snapshot = snapshot
        if not self.has_supply_headroom(
            snapshot,
            domains=set(profile.pressure_domains),
        ):
            return None
        with self.registry.locked() as active:
            # Real queued work always outranks speculative supply promises.
            allocations = [row for row in active if row.get("record_type") == "allocation"]
            supply_rows = [row for row in active if row.get("record_type") == "supply"]
            own_allocations = sum(row.get("run_id") == run_id for row in allocations)
            if own_allocations >= self.concurrency_limit:
                return None
            host_limit = self._host_concurrency_limit(active)
            if len(allocations) >= host_limit:
                return None
            gpu_uuids = self._choose_gpus(profile, snapshot, allocations)
            if profile.needs_gpu and gpu_uuids is None:
                return None
            allocation = Allocation(
                allocation_id=allocation_id,
                run_id=run_id,
                pid=pid,
                pgid=pgid,
                profile=profile.name,
                gpu_uuids=tuple(gpu_uuids or ()),
                gpu_memory_mb=int((profile.gpu_memory_gb or 0.0) * 1024),
                gpu_utilization_pct=float(profile.gpu_utilization_pct or 0.0),
                started_at=time.time(),
                gpu_demand_unknown=bool(
                    profile.needs_gpu
                    and (profile.gpu_memory_gb is None or profile.gpu_utilization_pct is None)
                ),
            )
            remove_supply_ids = {
                str(row.get("allocation_id", ""))
                for row in supply_rows
                if str(row.get("allocation_id", "")) == supply_claim_id
                or bool(set(row.get("gpu_uuids", []) or []) & set(allocation.gpu_uuids))
            }
            remaining_supply = [
                row
                for row in supply_rows
                if str(row.get("allocation_id", "")) not in remove_supply_ids
            ]
            own_remaining_supply = sum(
                row.get("run_id") in _resource_owner_group(run_id) for row in remaining_supply
            )
            overflow = max(
                0,
                own_allocations + 1 + own_remaining_supply - self.concurrency_limit,
            )
            for row in sorted(
                (
                    row
                    for row in remaining_supply
                    if row.get("run_id") in _resource_owner_group(run_id)
                ),
                key=lambda item: _as_float(item.get("started_at"), 0.0),
            )[:overflow]:
                remove_supply_ids.add(str(row.get("allocation_id", "")))
            remaining_after_run_limit = [
                row
                for row in remaining_supply
                if str(row.get("allocation_id", "")) not in remove_supply_ids
            ]
            host_overflow = max(
                0,
                len(allocations) + 1 + len(remaining_after_run_limit) - host_limit,
            )
            for row in sorted(
                remaining_after_run_limit,
                key=lambda item: _as_float(item.get("started_at"), 0.0),
            )[:host_overflow]:
                remove_supply_ids.add(str(row.get("allocation_id", "")))
            active[:] = [
                row
                for row in active
                if not (
                    row.get("record_type") == "supply"
                    and str(row.get("allocation_id", "")) in remove_supply_ids
                )
            ]
            active.append(
                {
                    "record_type": "allocation",
                    "pending": True,
                    "pid_start_time": _pid_start_time(pid),
                    **asdict(allocation),
                }
            )
            return allocation

    def release(self, allocation_id: str) -> None:
        with self.registry.locked() as active:
            active[:] = [item for item in active if item.get("allocation_id") != allocation_id]

    def bind_process(self, allocation_id: str, *, pid: int, pgid: int) -> bool:
        """Replace the short pending reservation owner with the launched process."""

        with self.registry.locked() as active:
            for item in active:
                if item.get("allocation_id") == allocation_id:
                    item["pid"] = int(pid)
                    item["pgid"] = int(pgid)
                    item["pid_start_time"] = _pid_start_time(int(pid))
                    item["pending"] = False
                    return True
        return False

    def recover_allocation(
        self,
        *,
        allocation_id: str,
        run_id: str,
        pid: int,
        pgid: int,
        profile: ResourceProfile,
        gpu_uuids: tuple[str, ...],
        require_admission: bool = True,
    ) -> Allocation | None:
        """Rebind an already-approved launch before releasing its barrier."""

        snapshot = self._last_snapshot
        if profile.needs_gpu:
            available = {device.uuid for device in snapshot.gpus}
            if len(gpu_uuids) != profile.gpu_count or not set(gpu_uuids) <= available:
                return None
        elif gpu_uuids:
            return None
        with self.registry.locked() as active:
            existing = next(
                (row for row in active if row.get("allocation_id") == allocation_id), None
            )
            allocations = [
                row
                for row in active
                if row.get("record_type") == "allocation"
                and row.get("allocation_id") != allocation_id
            ]
            if require_admission:
                if (
                    sum(row.get("run_id") == run_id for row in allocations)
                    >= self.concurrency_limit
                ):
                    return None
                if len(allocations) >= self._host_concurrency_limit(active):
                    return None
                chosen = self._choose_gpus(profile, snapshot, allocations, required_uuids=gpu_uuids)
                if profile.needs_gpu and tuple(chosen or ()) != gpu_uuids:
                    return None
            elif existing is not None and (
                str(existing.get("profile", "")) != profile.name
                or tuple(existing.get("gpu_uuids", []) or ()) != gpu_uuids
            ):
                return None
            active[:] = [
                row
                for row in active
                if not (
                    row.get("record_type") == "supply"
                    and bool(set(row.get("gpu_uuids", []) or []) & set(gpu_uuids))
                )
            ]
            allocation = Allocation(
                allocation_id=allocation_id,
                run_id=run_id,
                pid=pid,
                pgid=pgid,
                profile=profile.name,
                gpu_uuids=gpu_uuids,
                gpu_memory_mb=int((profile.gpu_memory_gb or 0.0) * 1024),
                gpu_utilization_pct=float(profile.gpu_utilization_pct or 0.0),
                started_at=time.time(),
                gpu_demand_unknown=bool(
                    profile.needs_gpu
                    and (profile.gpu_memory_gb is None or profile.gpu_utilization_pct is None)
                ),
            )
            active[:] = [
                row
                for row in active
                if row.get("allocation_id") != allocation_id
                and not (
                    not require_admission
                    and row.get("record_type") == "allocation"
                    and row.get("run_id") == run_id
                    and (int(row.get("pid", 0) or 0) == pid or int(row.get("pgid", 0) or 0) == pgid)
                )
            ]
            active.append(
                {
                    "record_type": "allocation",
                    "pending": False,
                    "pid_start_time": _pid_start_time(pid),
                    **asdict(allocation),
                }
            )
        return allocation

    def _host_concurrency_limit(self, rows: list[dict[str, Any]]) -> int:
        limits = [
            _as_int(row.get("concurrency_limit"), 0)
            for row in rows
            if row.get("record_type") == "controller"
        ]
        return max([self.concurrency_limit, *limits])

    @staticmethod
    def _choose_gpus(
        profile: ResourceProfile,
        snapshot: HostSnapshot,
        active: list[dict[str, Any]],
        *,
        required_uuids: tuple[str, ...] = (),
    ) -> list[str] | None:
        if not profile.needs_gpu:
            return []
        attributed_memory, attributed_utilization = ResourceAllocator._attributed_gpu_load(
            snapshot,
            active,
        )
        reservations: dict[str, tuple[int, float, int, int, float, int]] = {}
        for item in active:
            for uuid in item.get("gpu_uuids", []) or []:
                settled_memory, settled_util, count, pending_memory, pending_util, unknown = (
                    reservations.get(str(uuid), (0, 0.0, 0, 0, 0.0, 0))
                )
                memory = _as_int(item.get("gpu_memory_mb"), 0)
                util = _as_float(item.get("gpu_utilization_pct"), 0.0)
                demand_unknown = bool(
                    item.get("gpu_demand_unknown", False)
                    or (memory <= 0 and util <= 0 and item.get("gpu_uuids"))
                )
                # Recent reservations have not reached the physical snapshot.
                # Settled reservations may already be represented by the live
                # driver sample, so they must not be blindly added to it.
                pending = _as_float(item.get("started_at"), 0.0) > snapshot.observed_at - 10.0
                reservations[str(uuid)] = (
                    settled_memory + (0 if pending else memory),
                    settled_util + (0.0 if pending else util),
                    count + 1,
                    pending_memory + (memory if pending else 0),
                    pending_util + (util if pending else 0.0),
                    unknown + int(demand_unknown),
                )
        request_memory = int((profile.gpu_memory_gb or 0.0) * 1024)
        request_util = float(profile.gpu_utilization_pct or 0.0)
        unknown_request = profile.gpu_memory_gb is None or profile.gpu_utilization_pct is None
        candidates: list[tuple[tuple[float, ...], GPUDevice]] = []
        required = set(required_uuids)
        for device in snapshot.gpus:
            if required and device.uuid not in required:
                continue
            settled_memory, settled_util, count, pending_memory, pending_util, unknown = (
                reservations.get(device.uuid, (0, 0.0, 0, 0, 0.0, 0))
            )
            if unknown:
                continue
            if unknown_request:
                if count or device.memory_used_mb > 256 or device.utilization_pct > 5.0:
                    continue
            else:
                # The driver sample contains Praxist work that has reached the GPU
                # plus any unrelated host load.  The durable reservation is an
                # upper envelope for Praxist work.  ``max`` preserves whichever is
                # larger without double-counting the same settled process;
                # pending launches are then added because they cannot yet be in
                # the sample.
                if attributed_memory is None:
                    effective_memory = device.memory_used_mb + settled_memory + pending_memory
                else:
                    observed_praxist_memory = attributed_memory.get(device.uuid, 0)
                    external_memory = max(0, device.memory_used_mb - observed_praxist_memory)
                    effective_memory = (
                        external_memory
                        + max(observed_praxist_memory, settled_memory)
                        + pending_memory
                    )
                if attributed_utilization is None:
                    effective_util = device.utilization_pct + settled_util + pending_util
                else:
                    observed_praxist_util = attributed_utilization.get(device.uuid, 0.0)
                    external_util = max(0.0, device.utilization_pct - observed_praxist_util)
                    effective_util = (
                        external_util + max(observed_praxist_util, settled_util) + pending_util
                    )
                if effective_memory + request_memory > device.memory_total_mb * 0.95:
                    continue
                if effective_util + request_util > 100.0:
                    continue
            observed_score = max(
                device.memory_used_mb / max(1, device.memory_total_mb),
                device.utilization_pct / 100.0,
            )
            reserved_score = max(
                (settled_memory + pending_memory) / max(1, device.memory_total_mb),
                (settled_util + pending_util) / 100.0,
            )
            if attributed_memory is None:
                effective_memory = device.memory_used_mb + settled_memory + pending_memory
            else:
                observed_praxist_memory = attributed_memory.get(device.uuid, 0)
                effective_memory = (
                    max(0, device.memory_used_mb - observed_praxist_memory)
                    + max(observed_praxist_memory, settled_memory)
                    + pending_memory
                )
            if attributed_utilization is None:
                effective_util = device.utilization_pct + settled_util + pending_util
            else:
                observed_praxist_util = attributed_utilization.get(device.uuid, 0.0)
                effective_util = (
                    max(0.0, device.utilization_pct - observed_praxist_util)
                    + max(observed_praxist_util, settled_util)
                    + pending_util
                )
            # Balance by the tighter of compute and memory headroom.  The two
            # dimensions remain independent hard admission checks above; this
            # scalar is used only to choose among already-feasible devices.
            key = (
                max(
                    effective_memory / max(1, device.memory_total_mb * 0.95),
                    effective_util / 100.0,
                ),
                observed_score,
                reserved_score,
                float(device.index),
            )
            candidates.append((key, device))
        candidates.sort(key=lambda item: item[0])
        if required and {item[1].uuid for item in candidates} != required:
            return None
        if len(candidates) < profile.gpu_count:
            return None
        if required:
            return list(required_uuids)
        return [item[1].uuid for item in candidates[: profile.gpu_count]]

    @staticmethod
    def _attributed_gpu_load(
        snapshot: HostSnapshot,
        active: list[dict[str, Any]],
    ) -> tuple[dict[str, int] | None, dict[str, float] | None]:
        memory = {} if snapshot.gpu_process_memory_observed else None
        utilization = {} if snapshot.gpu_process_utilization_observed else None
        if memory is None and utilization is None:
            return None, None
        active_pgids = {
            _as_int(row.get("pgid"), 0) for row in active if row.get("record_type") == "allocation"
        }
        active_pgids.discard(0)
        for process in snapshot.gpu_processes:
            try:
                if os.getpgid(process.pid) not in active_pgids:
                    continue
            except (OSError, ProcessLookupError):
                continue
            if memory is not None:
                memory[process.uuid] = memory.get(process.uuid, 0) + process.memory_used_mb
            if utilization is not None and process.utilization_pct is not None:
                utilization[process.uuid] = (
                    utilization.get(process.uuid, 0.0) + process.utilization_pct
                )
        return memory, utilization
