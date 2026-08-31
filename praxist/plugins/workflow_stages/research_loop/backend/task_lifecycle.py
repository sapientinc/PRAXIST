"""Replayable task-owned lifecycle phases within the research-loop lifetime.

The caller must hold the orchestrator lock while using this helper. Callbacks
are trusted asynchronous Python code running in Praxist's interpreter; they must
cooperate with cancellation. They do not replace the AgentRuntime boundary.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import stat
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from praxist.core.protocol import AgentRunResult

from .tools.atomic_io import atomic_write_json

Phase = Literal["initial", "review", "finalize"]
AgentRole = Literal["research", "pi", "review", "final"]
AgentCallback = Callable[..., Awaitable[AgentRunResult]]
SCHEMA_VERSION = "praxist.task_lifecycle.v1"


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _digest(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _positive_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _earliest_deadline(*deadlines: float | None) -> float | None:
    return min((deadline for deadline in deadlines if deadline is not None), default=None)


def _contained_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not relative:
        raise ValueError("path must be relative and contained in its root")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("path must identify a file inside its root")
    return resolved


def _open_directory(path: Path) -> int:
    # Pin every ancestor, not only the final directory: the output tree can be
    # writable by peers while the lifecycle controller holds stronger access.
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@dataclass(frozen=True)
class TaskLifecycleContext:
    """Task callback inputs and the controller-owned agent execution boundary.

    Attributes:
        task_path: Explicit task project root.
        run_dir: Run artifact root.
        phase: Initial preparation, generation review, or terminal delivery.
        deadline_at: Persisted run deadline as Unix epoch seconds, or None if uncapped.
        phase_deadline_at: Persisted phase deadline, or None if uncapped.
        config: Task-owned lifecycle configuration, copied from startup.
        findings: Frozen phase input findings, unaffected by later collection.
        generation_id: Generation being reviewed, otherwise None.
    """

    task_path: Path
    run_dir: Path
    phase: Phase
    deadline_at: float | None
    phase_deadline_at: float | None
    config: dict[str, Any]
    findings: tuple[dict[str, Any], ...]
    _run_agent: AgentCallback = field(repr=False)
    _clock: Callable[[], float] = field(repr=False)
    generation_id: int | None = None

    async def run_agent(
        self,
        prompt: str,
        *,
        role: AgentRole = "research",
        allowed_tools: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> AgentRunResult:
        """Execute through the supplied runtime callback within the phase budget.

        Args:
            prompt: Task-authored prompt.
            role: Controller model-routing role.
            allowed_tools: Optional runtime tool allowlist.
            timeout_seconds: Optional tighter timeout than the phase remainder.

        Returns:
            The normalized runtime result.

        Raises:
            TimeoutError: The phase or requested call budget is exhausted.
            ValueError: The role or requested timeout is invalid.
        """
        if role not in {"research", "pi", "review", "final"}:
            raise ValueError("unsupported lifecycle agent role")
        remaining = (
            None if self.phase_deadline_at is None else self.phase_deadline_at - self._clock()
        )
        if timeout_seconds is not None:
            requested = _positive_seconds(timeout_seconds, "timeout_seconds")
            remaining = requested if remaining is None else min(remaining, requested)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("lifecycle phase deadline exceeded")
        execution = self._run_agent(
            prompt,
            role=role,
            allowed_tools=allowed_tools,
            timeout_seconds=remaining,
        )
        result = (
            await execution
            if remaining is None
            else await asyncio.wait_for(execution, timeout=remaining)
        )
        if self.phase_deadline_at is not None and self._clock() >= self.phase_deadline_at:
            raise TimeoutError("lifecycle phase deadline exceeded")
        return result


class TaskLifecycle:
    """Persist deadlines and commit optional task-owned lifecycle deliveries.

    Args:
        task_spec: Resolved task specification with research-loop configuration.
        task_path: Explicit task project root containing the callback module.
        run_dir: Artifact root for the current run.
        run_agent: Asynchronous execution callback supplied by the controller.
        clock: Wall clock used to compare persisted epoch deadlines.
        state_dir: Optional controller-private authoritative checkpoint directory.
            When supplied, the public run directory receives observation copies
            only and is never read for lifecycle authority.
    """

    def __init__(
        self,
        task_spec: Any,
        task_path: Path,
        run_dir: Path,
        run_agent: AgentCallback,
        *,
        clock: Callable[[], float] = time.time,
        state_dir: Path | None = None,
    ) -> None:
        self.task_path = Path(task_path).resolve()
        self.run_dir = Path(run_dir).resolve()
        self._clock = clock
        self._run_agent = run_agent
        raw = getattr(task_spec, "_raw", {}) or {}
        loop_config = getattr(task_spec, "research_loop", None) or raw.get("research_loop", {})
        self._scoreless = loop_config.get("mode") == "scoreless"
        self._config = _json_copy(loop_config.get("lifecycle") or {})
        self.enabled = self._scoreless and bool(self._config)
        self._public_path = self.run_dir / "lifecycle" / "state.json"
        self._state_path = (
            (Path(state_dir).resolve() / "state.json") if state_dir else self._public_path
        )
        self._state: dict[str, Any] = {}
        self._callback_path: Path | None = None
        self._callback_name = ""
        self._initial_seconds: float | None = None
        self._finalization_seconds: float | None = None
        self._deadline_required = False
        if not self._scoreless:
            return
        policy = getattr(task_spec, "run_lifecycle", None)
        hours = getattr(policy, "max_wall_clock_hours", None)
        total = None if hours is None else _positive_seconds(hours, "max_wall_clock_hours") * 3600
        self._deadline_required = total is not None
        if total is not None and not math.isfinite(total):
            raise ValueError("max_wall_clock_hours must produce a finite deadline")
        if self.enabled:
            entrypoint = self._config.get("entrypoint", "")
            if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
                raise ValueError("lifecycle entrypoint must be a task Python file:function")
            file_name, self._callback_name = entrypoint.split(":")
            if not file_name.endswith(".py") or not self._callback_name.isidentifier():
                raise ValueError("lifecycle entrypoint must be a task Python file:function")
            self._callback_path = _contained_file(self.task_path, file_name)
            default_seconds = None if total is None else 1800
            initial = self._config.get("initial_seconds", default_seconds)
            finalization = self._config.get("finalization_seconds", default_seconds)
            self._initial_seconds = (
                None if initial is None else _positive_seconds(initial, "initial_seconds")
            )
            self._finalization_seconds = (
                None
                if finalization is None
                else _positive_seconds(finalization, "finalization_seconds")
            )
            reserves = (self._initial_seconds or 0) + (self._finalization_seconds or 0)
            if total is not None and reserves >= total:
                raise ValueError("initial and finalization reserves must leave time for research")
            if not isinstance(self._config.get("config", {}), dict):
                raise ValueError("lifecycle config must be a mapping")
            if not isinstance(self._config.get("after_generation", False), bool):
                raise ValueError("lifecycle after_generation must be a boolean")
        if state_dir:
            self._state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._state_path.parent.chmod(0o700)
        if self._state_path.exists():
            self._state = json.loads(self._state_path.read_text(encoding="utf-8"))
            if self._state.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("invalid lifecycle checkpoint schema")
            if self._state.get("task_path") != str(self.task_path) or self._state.get(
                "run_dir"
            ) != str(self.run_dir):
                raise ValueError("lifecycle checkpoint belongs to a different task or run")
            if self._state.get("config_digest") != _digest(self._config):
                raise ValueError("lifecycle configuration changed after the run started")
            self._validate_clock_state()
        else:
            now = self._clock()
            self._state = {
                "schema_version": SCHEMA_VERSION,
                "task_path": str(self.task_path),
                "run_dir": str(self.run_dir),
                "started_at": now,
                "deadline_at": None if total is None else now + total,
                "config_digest": _digest(self._config),
                "phases": {},
                "events": [],
            }
            self._persist()

    @property
    def started_at(self) -> float | None:
        """Return the original run start, retained across resume."""
        return self._state.get("started_at")

    @property
    def deadline_at(self) -> float | None:
        """Return the original absolute run deadline, if one was configured."""
        return self._state.get("deadline_at")

    @property
    def research_deadline_at(self) -> float | None:
        """Return the research cutoff with terminal-delivery time reserved."""
        deadline = self.deadline_at
        return None if deadline is None else deadline - (self._finalization_seconds or 0)

    @property
    def initial_completed(self) -> bool:
        """Whether initial delivery has a committed checkpoint."""
        return self._phase_completed("initial")

    @property
    def finalization_completed(self) -> bool:
        """Whether terminal delivery has a committed checkpoint."""
        return self._phase_completed("finalize")

    @property
    def finalization_started(self) -> bool:
        """Whether frozen finalization inputs exist and research must not resume."""
        return "finalize" in self._state.get("phases", {})

    @property
    def after_generation(self) -> bool:
        """Whether task-owned reviews are requested after generation boundaries."""
        return self.enabled and self._config.get("after_generation", False)

    @property
    def research_completed(self) -> bool:
        """Whether a committed generation review declared research complete."""
        return any(
            name.startswith("review_gen_")
            and record.get("status") == "committed"
            and record.get("result", {}).get("summary", {}).get("research_complete") is True
            for name, record in self._state.get("phases", {}).items()
        )

    def remaining_seconds(self, *, finalization: bool = False) -> float | None:
        """Return the nonnegative budget before the research or total deadline."""
        deadline = self.deadline_at if finalization else self.research_deadline_at
        return None if deadline is None else max(0.0, deadline - self._clock())

    def final_artifact_hashes(self) -> dict[str, str]:
        """Return controller-committed hashes for terminal artifact materialization.

        Returns:
            Original content hashes, or an empty mapping before finalization commits.
        """
        record = self._state.get("phases", {}).get("finalize", {})
        if record.get("status") != "committed":
            return {}
        return dict(record.get("artifact_hashes") or {})

    async def run_phase(
        self,
        phase: Phase,
        findings: Iterable[dict[str, Any]] = (),
        *,
        generation_id: int | None = None,
    ) -> dict[str, Any]:
        """Run or resume one task phase without replacing its original inputs.

        Args:
            phase: Initial preparation, generation review, or terminal delivery.
            findings: Findings frozen on the first attempt for this phase.
            generation_id: Nonnegative generation number required for reviews.

        Returns:
            A task result with completed/incomplete status, artifacts and summary.
            Ordinary callback failures become incomplete results, with partial
            files retained. Cancellation is recorded and then propagated.

        Raises:
            ValueError: The phase is invalid or committed evidence was changed.
        """
        if phase not in {"initial", "review", "finalize"}:
            raise ValueError("unsupported lifecycle phase")
        if phase == "review":
            if (
                isinstance(generation_id, bool)
                or not isinstance(generation_id, int)
                or generation_id < 0
            ):
                raise ValueError("review requires a nonnegative integer generation_id")
        elif generation_id is not None:
            raise ValueError("generation_id is only valid for a review phase")
        if not self.enabled:
            return {"status": "completed", "artifacts": [], "summary": {"lifecycle": "disabled"}}
        if phase == "review" and not self.after_generation:
            return {"status": "completed", "artifacts": [], "summary": {"review": "disabled"}}
        phase_key = f"review_gen_{generation_id}" if phase == "review" else phase
        phases = self._state["phases"]
        record: dict[str, Any] | None = phases.get(phase_key)
        if record is not None and record["status"] == "committed":
            if self._artifact_hashes(record["result"]["artifacts"]) != record["artifact_hashes"]:
                raise ValueError("committed lifecycle artifact content changed")
            return copy.deepcopy(record["result"])
        if phase == "review" and self.finalization_started:
            return {
                "status": "incomplete",
                "artifacts": [],
                "summary": {"reason": "finalization_started"},
            }
        if record is None:
            now = self._clock()
            deadline = self.deadline_at
            if phase == "initial":
                started = self.started_at
                assert started is not None
                phase_deadline = _earliest_deadline(
                    None if self._initial_seconds is None else started + self._initial_seconds,
                    self.research_deadline_at,
                )
            elif phase == "review":
                phase_deadline = self.research_deadline_at
            else:
                phase_deadline = _earliest_deadline(
                    None
                    if self._finalization_seconds is None
                    else now + self._finalization_seconds,
                    deadline,
                )
            inputs = {
                "config": self._config.get("config", {}),
                "findings": _json_copy(list(findings)),
            }
            if phase == "review":
                inputs["generation_id"] = generation_id
            record = {
                "status": "pending",
                "phase_deadline_at": phase_deadline,
                "inputs": inputs,
                "input_digest": _digest(inputs),
                "attempts": 0,
            }
            phases[phase_key] = record
        if _digest(record["inputs"]) != record["input_digest"]:
            raise ValueError("lifecycle frozen input digest changed")
        record["attempts"] += 1
        self._record_transition(phase_key, record, "running")
        phase_deadline = record["phase_deadline_at"]
        remaining = None if phase_deadline is None else phase_deadline - self._clock()
        if remaining is not None and remaining <= 0:
            return self._incomplete(phase_key, record, "deadline_exceeded")
        deadline = self.deadline_at
        context = TaskLifecycleContext(
            task_path=self.task_path,
            run_dir=self.run_dir,
            phase=phase,
            deadline_at=deadline,
            phase_deadline_at=record["phase_deadline_at"],
            config=copy.deepcopy(record["inputs"]["config"]),
            findings=tuple(copy.deepcopy(record["inputs"]["findings"])),
            _run_agent=self._run_agent,
            _clock=self._clock,
            generation_id=generation_id,
        )
        try:
            with self._callback() as callback:
                remaining = None if phase_deadline is None else phase_deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    return self._incomplete(phase_key, record, "deadline_exceeded")
                result = (
                    await callback(context)
                    if remaining is None
                    else await asyncio.wait_for(callback(context), timeout=remaining)
                )
            if phase_deadline is not None and self._clock() >= phase_deadline:
                return self._incomplete(phase_key, record, "deadline_exceeded")
            try:
                result, artifact_hashes = self._validate_result(result, phase=phase)
            except (ValueError, TypeError, OSError):
                return self._incomplete(phase_key, record, "invalid_result")
            if phase_deadline is not None and self._clock() >= phase_deadline:
                return self._incomplete(phase_key, record, "deadline_exceeded")
            record["result"] = result
            record["artifact_hashes"] = artifact_hashes
            status = "committed" if result["status"] == "completed" else "incomplete"
            self._record_transition(phase_key, record, status)
            return copy.deepcopy(result)
        except asyncio.CancelledError:
            # A failed checkpoint must not turn operator cancellation into an
            # ordinary error. The previous running record remains resumable.
            with suppress(Exception):
                self._incomplete(phase_key, record, "cancelled")
            raise
        except TimeoutError:
            return self._incomplete(phase_key, record, "deadline_exceeded")
        except Exception as exc:
            return self._incomplete(
                phase_key, record, "callback_failed", error_type=type(exc).__name__
            )

    def _phase_completed(self, phase: Phase) -> bool:
        return self._state.get("phases", {}).get(phase, {}).get("status") == "committed"

    def _validate_clock_state(self) -> None:
        started = self._state.get("started_at")
        deadline = self._state.get("deadline_at")
        if not isinstance(started, (float, int)) or not math.isfinite(started):
            raise ValueError("invalid lifecycle persisted start")
        if deadline is not None and (
            not isinstance(deadline, (float, int))
            or not math.isfinite(deadline)
            or deadline <= started
        ):
            raise ValueError("invalid lifecycle persisted deadline")
        if self._deadline_required and deadline is None:
            raise ValueError("lifecycle checkpoint has no deadline")
        if not isinstance(self._state.get("phases"), dict) or not isinstance(
            self._state.get("events"), list
        ):
            raise ValueError("invalid lifecycle persisted phases")

    def _persist(self) -> None:
        atomic_write_json(self._state_path, self._state)
        if self._state_path != self._public_path:
            # The public copy is observation only. Directory-relative operations
            # prevent an analyst-owned symlink from redirecting controller writes.
            with suppress(OSError):
                self._write_public_mirror()

    def _write_public_mirror(self) -> None:
        projection = copy.deepcopy(self._state)
        for phase in projection["phases"].values():
            phase.pop("inputs", None)
        root_fd = _open_directory(self.run_dir)
        try:
            with suppress(FileExistsError):
                os.mkdir("lifecycle", mode=0o755, dir_fd=root_fd)
            directory_fd = os.open(
                "lifecycle", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
            )
            try:
                temporary = f".state.{uuid.uuid4().hex}.tmp"
                fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o644,
                    dir_fd=directory_fd,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as stream:
                        json.dump(projection, stream, allow_nan=False, sort_keys=True, indent=2)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(
                        temporary, "state.json", src_dir_fd=directory_fd, dst_dir_fd=directory_fd
                    )
                    os.fsync(directory_fd)
                finally:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary, dir_fd=directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            os.close(root_fd)

    def _record_transition(self, phase: str, record: dict[str, Any], status: str) -> None:
        record["status"] = status
        event = {
            "phase": "review" if phase.startswith("review_gen_") else phase,
            "status": status,
            "at": self._clock(),
            "attempt": record["attempts"],
            "input_digest": record["input_digest"],
        }
        if phase.startswith("review_gen_"):
            event["generation_id"] = record["inputs"]["generation_id"]
        self._state["events"].append(event)
        self._persist()

    def _incomplete(
        self, phase: str, record: dict[str, Any], reason: str, **summary: Any
    ) -> dict[str, Any]:
        result = {"status": "incomplete", "artifacts": [], "summary": {"reason": reason, **summary}}
        record["result"] = result
        self._record_transition(phase, record, "incomplete")
        return result

    def _artifact_hashes(self, artifacts: list[str]) -> dict[str, str]:
        result = {}
        for relative in artifacts:
            if not isinstance(relative, str):
                raise ValueError("artifact path must be a string")
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
                raise ValueError("invalid lifecycle artifact path")
            directory_fd = _open_directory(self.run_dir)
            try:
                for component in candidate.parts[:-1]:
                    child_fd = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    os.close(directory_fd)
                    directory_fd = child_fd
                # Nonblocking open lets us reject FIFOs before they can block the
                # controller. fstat validates the exact object subsequently read.
                artifact_fd = os.open(
                    candidate.parts[-1],
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=directory_fd,
                )
                with os.fdopen(artifact_fd, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ValueError("lifecycle artifact must be a regular file")
                    result[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
            finally:
                os.close(directory_fd)
        return result

    def _validate_result(
        self, result: Any, *, phase: Phase
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not isinstance(result, Mapping) or result.get("status") not in {
            "completed",
            "incomplete",
        }:
            raise ValueError("invalid lifecycle result status")
        if not isinstance(result.get("artifacts"), list) or not isinstance(
            result.get("summary"), dict
        ):
            raise ValueError("lifecycle result requires artifacts and summary")
        if phase == "review" and not isinstance(
            result["summary"].get("research_complete", False), bool
        ):
            raise ValueError("review research_complete must be a boolean")
        data = _json_copy(dict(result))
        return data, self._artifact_hashes(data["artifacts"])

    @contextmanager
    def _callback(self) -> Iterator[Callable[[TaskLifecycleContext], Awaitable[Any]]]:
        path = self._callback_path
        assert path is not None
        name = "_praxist_task_lifecycle_" + hashlib.sha256(str(path).encode()).hexdigest()
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError("cannot load lifecycle entrypoint")
        module = importlib.util.module_from_spec(spec)
        previous = dict(sys.modules)
        inserted_paths = [str(self.task_path)]
        if path.parent != self.task_path:
            inserted_paths.insert(0, str(path.parent))
        local_names = {
            child.stem if child.is_file() else child.name
            for directory in inserted_paths
            for child in Path(directory).iterdir()
            if (child.is_dir() or child.suffix == ".py")
            and child.stem.isidentifier()
            and child.name != "__pycache__"
        }
        shadowed = {
            key: value for key, value in previous.items() if key.split(".", 1)[0] in local_names
        }
        for key in shadowed:
            sys.modules.pop(key, None)
        for directory in reversed(inserted_paths):
            sys.path.insert(0, directory)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            callback = getattr(module, self._callback_name, None)
            if not inspect.iscoroutinefunction(callback):
                raise ValueError("lifecycle entrypoint must be an async function")
            yield callback
        finally:
            for directory in inserted_paths:
                for index, entry in enumerate(sys.path):
                    if entry is directory:
                        sys.path.pop(index)
                        break
            for module_name, loaded in tuple(sys.modules.items()):
                if (
                    module_name == name
                    or module_name.split(".", 1)[0] in local_names
                    or self._task_local_module(loaded)
                ):
                    if module_name in previous:
                        sys.modules[module_name] = previous[module_name]
                    else:
                        sys.modules.pop(module_name, None)
            sys.modules.update(shadowed)

    def _task_local_module(self, module: ModuleType) -> bool:
        file_name = getattr(module, "__file__", None)
        return isinstance(file_name, str) and Path(file_name).resolve().is_relative_to(
            self.task_path
        )
