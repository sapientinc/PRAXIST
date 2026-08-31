"""Trusted task runtime preparation before research-loop workspace setup."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any


def prepare_task_runtime(
    *,
    task_descriptor: Mapping[str, Any],
    task_path: Path,
    run_dir: Path,
    resume: bool,
) -> None:
    """Invoke an optional synchronous task-owned environment preparation hook.

    The caller must validate fresh-run eligibility or private resume selection
    before calling. Preparation precedes cwd/store setup and the research clock;
    it must not launch agents or an alternate research process. The task source
    and its dependencies must be trusted and unavailable for peer mutation.

    Args:
        task_descriptor: Effective startup-selected task descriptor.
        task_path: Explicit task root containing the entrypoint file.
        run_dir: Destination for the run's generated files.
        resume: Whether startup is resuming existing research state.

    Raises:
        ValueError: The descriptor, entrypoint, or configuration is invalid.
        TypeError: The handler is not synchronous or returns a non-None result.
        Exception: A task preparation failure; startup must not continue.
    """
    runtime = task_descriptor.get("runtime_environment", {})
    if runtime is None:
        return
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime_environment must be a mapping")
    config = runtime.get("prepare_config", {})
    if not isinstance(config, dict):
        raise ValueError("runtime_environment.prepare_config must be a mapping")
    if "prepare_entrypoint" not in runtime:
        if config:
            raise ValueError("runtime_environment.prepare_config requires prepare_entrypoint")
        return
    entrypoint = runtime["prepare_entrypoint"]
    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
        raise ValueError("prepare_entrypoint must name a task-relative Python file:function")
    filename, name = entrypoint.split(":")
    relative = Path(filename)
    root = Path(task_path).resolve()
    source = (root / relative).resolve()
    if (
        not filename
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix != ".py"
        or not name.isidentifier()
        or not source.is_relative_to(root)
        or not source.is_file()
    ):
        raise ValueError("prepare_entrypoint must identify a Python file inside the task root")
    try:
        detached_config = json.loads(json.dumps(config, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError("runtime_environment.prepare_config must be JSON-compatible") from exc
    with _task_callback(root, source, name) as callback:
        result = callback(
            task_path=root,
            run_dir=Path(run_dir).resolve(),
            resume=bool(resume),
            config=detached_config,
        )
        if isinstance(result, (asyncio.Future, concurrent.futures.Future)):
            result.cancel()
            raise TypeError("runtime preparation must not return deferred work")
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("runtime preparation must not return an awaitable")
        if inspect.isgenerator(result) or inspect.isasyncgen(result):
            if inspect.isgenerator(result):
                result.close()
            raise TypeError("runtime preparation must not return deferred work")
        if result is not None:
            raise TypeError("runtime preparation must complete synchronously and return None")


@contextmanager
def _task_callback(root: Path, source: Path, name: str) -> Iterator[Callable[..., Any]]:
    module_name = "_praxist_runtime_prepare_" + hashlib.sha256(str(source).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load runtime preparation entrypoint")
    module = importlib.util.module_from_spec(spec)
    previous = dict(sys.modules)
    inserted = [str(root)]
    if source.parent != root:
        inserted.insert(0, str(source.parent))
    local_names = {
        child.stem if child.is_file() else child.name
        for directory in inserted
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
    for directory in reversed(inserted):
        sys.path.insert(0, directory)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        callback = getattr(module, name, None)
        if (
            not inspect.isfunction(callback)
            or inspect.iscoroutinefunction(callback)
            or inspect.isgeneratorfunction(callback)
            or inspect.isasyncgenfunction(callback)
        ):
            raise TypeError("runtime preparation entrypoint must be a synchronous function")
        yield callback
    finally:
        for directory in inserted:
            for index, entry in enumerate(sys.path):
                if entry is directory:
                    sys.path.pop(index)
                    break
        for key, loaded in tuple(sys.modules.items()):
            if (
                key == module_name
                or key.split(".", 1)[0] in local_names
                or _inside_task(loaded, root)
            ):
                if key in previous:
                    sys.modules[key] = previous[key]
                else:
                    sys.modules.pop(key, None)
        sys.modules.update(shadowed)


def _inside_task(module: ModuleType, root: Path) -> bool:
    source = getattr(module, "__file__", None)
    return isinstance(source, str) and Path(source).resolve().is_relative_to(root)
