"""Startup/finalization bridge for the plugin-local research_loop backend."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from praxist import __version__
from praxist.core.budget import policy_for_ref
from praxist.core.cache import build_cache_policy
from praxist.core.credentials import (
    CredentialFailoverManager,
    CredentialResolver,
    CredentialSet,
)
from praxist.core.ledgers import BudgetLedger
from praxist.core.modeling import (
    model_profiles_snapshot,
    normalize_model_for_provider,
    provider_default_model,
    validate_model_for_provider,
)
from praxist.core.protocol import BudgetRequest
from praxist.core.redaction import scan_text
from praxist.core.registry import (
    PluginLoader,
    PluginRef,
    PluginRoots,
    assert_bundled_execution_manifest,
)
from praxist.core.run_config import DEFAULT_AGENT_MODEL
from praxist.core.runtimes import resolve_model_credential_for_runtime
from praxist.core.source_snapshot import build_core_source_snapshot
from praxist.core.storage import (
    ensure_run_dirs,
    output_ledger_hashes,
    utc_now,
    write_json,
)
from praxist.core.task_project import (
    TaskProject,
    resolve_task_project,
    task_project_global_plugin_refs,
    write_task_project_manifest,
)
from praxist.core.tool_servers import (
    effective_research_tool_server_refs_from_task_descriptor,
)
from praxist.core.trajectory import TrajectoryWriter
from praxist.core.workflow import emit_disabled_optional_events
from praxist.plugins.workflow_stages.research_loop.backend.run_summary import (
    write_run_summary,
)
from praxist.plugins.workflow_stages.research_loop.legacy_output_materializer import (
    _materialize_legacy_outputs,
)
from praxist.plugins.workflow_stages.research_loop.peer_roles import (
    model_profile_defaults_from_task_descriptor,
    peer_role_ref_from_task_descriptor,
    peer_role_refs_from_task_descriptor,
    role_refs_from_task_descriptor,
)
from praxist.plugins.workflow_stages.research_loop.provider_env import (
    freeze_provider_env,
)
from praxist.plugins.workflow_stages.research_loop.stage import planned_research_loop_usage
from praxist.task_spec import TaskSpec, load_task_spec

from .backend import resume_state, run_report, runtime_environment
from .backend.resume_state import ensure_resumable_run_dir, validate_resume_startup_identity

RESEARCH_LOOP_STAGE_REF = "workflow_stage:research_loop"
_model_profile_defaults_from_task_descriptor = model_profile_defaults_from_task_descriptor
_peer_role_ref_from_task_descriptor = peer_role_ref_from_task_descriptor
_peer_role_refs_from_task_descriptor = peer_role_refs_from_task_descriptor
_role_refs_from_task_descriptor = role_refs_from_task_descriptor


def _cache_strategy_for_runtime_provider(
    runtime_ref: str,
    model_provider_ref: str,
    registry: Any | None = None,
) -> tuple[str, str | None, str | None]:
    runtime_contract = _runtime_contract(runtime_ref, registry)
    provider_contract = _provider_contract(model_provider_ref, registry)
    runtime_strategy = str(runtime_contract.get("cache_strategy") or "")
    provider_strategy = str(provider_contract.get("cache_strategy") or "")
    if runtime_strategy in {"disabled", "deterministic_no_cache"}:
        return "disabled", None, None
    if provider_strategy in {"disabled", "deterministic_no_cache"}:
        return "disabled", None, None
    if runtime_strategy == "runtime_auto_cache":
        return "runtime_auto_cache", runtime_strategy, None
    if provider_strategy == "provider_explicit_cache":
        explicit = str(
            provider_contract.get("explicit_cache_strategy") or "provider_explicit_cache"
        )
        return "provider_explicit_cache", None, explicit
    return "provider_default", None, None


@dataclass
class ResearchLoopPluginRun:
    """Prepared research-loop run bundle."""

    task_ref: str
    run_id: str
    run_dir: Path
    task_spec: TaskSpec
    task_project_path: Path
    task_descriptor: dict[str, Any]
    task_project_manifest: dict[str, Any]
    task_execution_cwd: Path
    task_runtime_env: dict[str, str]
    runtime_ref: str
    model_provider_ref: str
    budget_policy_ref: str
    registry: Any
    tool_server_refs: tuple[str, ...]
    peer_role_ref: str
    credential_set: CredentialSet
    credential_manager: CredentialFailoverManager
    model_provider_credential_key_id: str | None
    provider_env: dict[str, str | None]
    resolution_manifest: dict[str, Any]
    startup_config: dict[str, Any]
    trajectory: TrajectoryWriter
    stage_budget_grant_id: str | None
    peer_role_refs: tuple[str, ...] = ()


def is_research_loop_task_project(task_path: str | Path, workspace: Path | None = None) -> bool:
    """Return whether a task project can run with the research_loop workflow stage."""
    try:
        project = resolve_task_project(task_path, workspace=workspace or Path.cwd())
    except Exception:
        return False
    return _task_workflow_stage_ref(project.descriptor) == RESEARCH_LOOP_STAGE_REF


def is_research_loop_plugin_task(task_ref: str, workspace: Path | None = None) -> bool:
    """Compatibility shim for external task projects."""
    return False


def default_runtime_for_task(task_ref: str, runtime_ref: str | None = None) -> str:
    """Return the default agent runtime reference for a task descriptor."""
    if runtime_ref:
        return runtime_ref
    if task_ref == "task:fake_panel":
        return "agent_runtime:fake_runtime"
    return "agent_runtime:claude_sdk"


def default_model_provider_for_task(task_ref: str, model_provider_ref: str | None = None) -> str:
    """Return the default model provider reference for a task descriptor."""
    if model_provider_ref:
        return model_provider_ref
    if task_ref == "task:fake_panel":
        return "model_provider:fake_provider"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "model_provider:deepseek_alias"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "model_provider:openrouter"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "model_provider:anthropic_messages"
    return "model_provider:openrouter"


def default_budget_policy_for_task(task_ref: str, budget_policy_ref: str | None = None) -> str:
    """Return the default budget policy reference for a task descriptor."""
    if budget_policy_ref:
        return budget_policy_ref
    if task_ref == "task:fake_panel":
        return "budget_policy:fake_tiered"
    return "budget_policy:default_basic"


def _default_model_for_provider(model_provider_ref: str) -> str:
    """Return the default model name for ``model_provider_ref``.

    Reads the plugin's yaml ``provider.default_model`` field as the
    single source of truth.  Operators who want to change the
    default for a provider edit one place — the plugin's
    ``plugin.yaml`` — rather than hunting down a hardcoded branch
    here that drifts out of sync with the yaml manifest (#144).

    Falls back to the package default when provider metadata is unavailable.
    """
    return provider_default_model(model_provider_ref) or DEFAULT_AGENT_MODEL


def _apply_internal_env_overrides(
    task_spec: TaskSpec,
    task_descriptor: dict[str, Any],
    env: Mapping[str, str],
) -> tuple[TaskSpec, dict[str, Any], list[dict[str, Any]]]:
    effective_descriptor = deepcopy(task_descriptor)
    generation_policy = task_spec.generation_policy
    overrides_seen: list[dict[str, Any]] = []
    raw_generation_policy = dict(effective_descriptor.get("generation_policy") or {})

    for env_name in ("PRAXIST_MAX_GENERATIONS", "MAX_GENERATIONS"):
        max_generations = _positive_int_env(env, env_name)
        if max_generations is not None:
            generation_policy = replace(generation_policy, max_generations=max_generations)
            raw_generation_policy["max_generations"] = max_generations
            overrides_seen.append(
                {
                    "env": env_name,
                    "path": "generation_policy.max_generations",
                    "value": max_generations,
                }
            )
            break

    for env_name in ("PRAXIST_COHORT_SIZE", "COHORT_SIZE"):
        cohort_size = _positive_int_env(env, env_name)
        if cohort_size is not None:
            generation_policy = replace(generation_policy, cohort_size=cohort_size)
            raw_generation_policy["cohort_size"] = cohort_size
            overrides_seen.append(
                {
                    "env": env_name,
                    "path": "generation_policy.cohort_size",
                    "value": cohort_size,
                }
            )
            break

    if not overrides_seen:
        return task_spec, effective_descriptor, []

    effective_descriptor["generation_policy"] = raw_generation_policy
    multi_pi = task_spec.multi_pi
    if getattr(multi_pi, "enabled", False):
        multi_pi = replace(multi_pi, chair_peer_budget=generation_policy.cohort_size)
        multi_pi_raw = dict(effective_descriptor.get("multi_pi") or {})
        multi_pi_raw["chair_peer_budget"] = generation_policy.cohort_size
        effective_descriptor["multi_pi"] = multi_pi_raw
    return (
        replace(
            task_spec,
            generation_policy=generation_policy,
            multi_pi=multi_pi,
            _raw=effective_descriptor,
        ),
        effective_descriptor,
        overrides_seen,
    )


def _positive_int_env(env: Mapping[str, str], name: str) -> int | None:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(str(raw))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _task_runtime_env(
    *,
    task_project_path: Path,
    workspace: Path,
    task_id: str,
    env: Mapping[str, str],
    task_descriptor: dict[str, Any] | None = None,
    evaluation_entrypoint: str = "",
) -> dict[str, str]:
    workspace, task_project_path = workspace.resolve(), task_project_path.resolve()
    task_descriptor = task_descriptor or {}
    runtime_cfg = _runtime_environment_config(task_descriptor)
    runtime_env: dict[str, str] = {
        "PRAXIST_TASK_PROJECT_PATH": str(task_project_path),
        "PRAXIST_WORKSPACE_ROOT": str(workspace),
        "PYTHONPATH": _prepend_path_env(str(workspace), env.get("PYTHONPATH", "")),
    }
    runtime_env.update(
        runtime_environment.task_process_identity_env(
            evaluation_entrypoint=evaluation_entrypoint,
            task_dir=task_project_path,
            runtime_cwd=runtime_cfg.get("cwd"),
        )
    )

    data_root = env.get("PRAXIST_DATASETS_DIR") or env.get("PRAXIST_DATA_ROOT")
    if data_root:
        runtime_env.setdefault("PRAXIST_DATASETS_DIR", str(Path(data_root).expanduser()))
    else:
        workspace_data_root = workspace / "data"
        if workspace_data_root.exists():
            runtime_env["PRAXIST_DATASETS_DIR"] = str(workspace_data_root)

    data_env_aliases = _runtime_data_env_aliases(runtime_cfg)
    task_data_dir = _resolved_task_data_dir(
        task_id=task_id,
        workspace=workspace,
        env=env,
        data_env_aliases=data_env_aliases,
    )
    if task_data_dir is not None:
        runtime_env.setdefault("PRAXIST_DATA_DIR", str(task_data_dir))

    custom_env_keys: list[str] = []
    if task_data_dir is not None:
        for env_key in data_env_aliases:
            runtime_env.setdefault(env_key, str(task_data_dir))
            custom_env_keys.append(env_key)

    custom_env = runtime_cfg.get("env")
    if isinstance(custom_env, dict):
        for key, value in custom_env.items():
            env_key = str(key).strip()
            if not _is_safe_env_key(env_key):
                raise ValueError(f"runtime_environment.env contains invalid key: {env_key!r}")
            env_value = str(value)
            hits = scan_text(env_value)
            if hits:
                raise ValueError(
                    f"runtime_environment.env.{env_key} looks like raw secret material: "
                    f"{','.join(sorted(set(hits)))}"
                )
            runtime_env[env_key] = env_value
            custom_env_keys.append(env_key)

    venv_path = _runtime_path(
        runtime_cfg.get("venv") or runtime_cfg.get("virtualenv"),
        task_project_path=task_project_path,
        field_name="runtime_environment.venv",
    )
    python_path = _runtime_path(
        runtime_cfg.get("python") or runtime_cfg.get("python_executable"),
        task_project_path=task_project_path,
        field_name="runtime_environment.python",
    )
    require_paths = bool(runtime_cfg.get("require_paths", True))
    writable_roots: list[Path] = []

    if venv_path is not None:
        if require_paths and not venv_path.exists():
            raise FileNotFoundError(f"runtime_environment.venv not found: {venv_path}")
        if require_paths and not venv_path.is_dir():
            raise ValueError(f"runtime_environment.venv must be a directory: {venv_path}")
        writable_roots.append(venv_path)
        runtime_env["PRAXIST_TASK_VENV"] = str(venv_path)
        runtime_env["VIRTUAL_ENV"] = str(venv_path)
        runtime_env["PATH"] = _prepend_path_env(
            str(venv_path / "bin"), env.get("PATH", os.environ.get("PATH", ""))
        )
        if python_path is None:
            python_path = venv_path / "bin" / "python"

    if python_path is not None:
        if require_paths and not python_path.exists():
            raise FileNotFoundError(f"runtime_environment.python not found: {python_path}")
        if require_paths and not python_path.is_file():
            raise ValueError(f"runtime_environment.python must be a file: {python_path}")
        runtime_env["PRAXIST_TASK_PYTHON"] = str(python_path)
        runtime_env["PATH"] = _prepend_path_env(
            str(python_path.parent),
            runtime_env.get("PATH") or env.get("PATH", os.environ.get("PATH", "")),
        )

    writable_roots.extend(
        _runtime_paths(
            runtime_cfg.get("writable_roots") or runtime_cfg.get("task_writable_roots"),
            task_project_path=task_project_path,
            field_name="runtime_environment.writable_roots",
        )
    )
    if writable_roots:
        runtime_env["PRAXIST_TASK_WRITABLE_ROOTS"] = os.pathsep.join(
            str(path) for path in dict.fromkeys(writable_roots)
        )

    path_prepend = runtime_cfg.get("path_prepend")
    if isinstance(path_prepend, str):
        path_prepend = [path_prepend]
    if isinstance(path_prepend, list):
        for item in reversed(path_prepend):
            item_path = _runtime_path(
                item,
                task_project_path=task_project_path,
                field_name="runtime_environment.path_prepend",
            )
            if item_path is not None:
                runtime_env["PATH"] = _prepend_path_env(
                    str(item_path),
                    runtime_env.get("PATH") or env.get("PATH", os.environ.get("PATH", "")),
                )

    shell_prefix = runtime_cfg.get("shell_prefix")
    if shell_prefix:
        runtime_env["PRAXIST_TASK_SHELL_PREFIX"] = str(shell_prefix)
    elif venv_path is not None:
        runtime_env["PRAXIST_TASK_SHELL_PREFIX"] = (
            f"source {shlex.quote(str(venv_path / 'bin' / 'activate'))} &&"
        )

    if custom_env_keys:
        runtime_env["PRAXIST_TASK_RUNTIME_ENV_KEYS"] = ",".join(sorted(custom_env_keys))

    return runtime_env


def _runtime_environment_config(descriptor: dict[str, Any]) -> dict[str, Any]:
    raw = descriptor.get("runtime_environment") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _runtime_data_env_aliases(runtime_cfg: Mapping[str, Any]) -> list[str]:
    raw = runtime_cfg.get("data_env_aliases") or runtime_cfg.get("data_dir_env_aliases") or []
    raw_items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    keys = (key for item in raw_items if (key := str(item).strip()) and _is_safe_env_key(key))
    return list(dict.fromkeys(keys))


def _task_execution_cwd(
    *,
    task_project_path: Path,
    run_dir: Path,
    task_descriptor: dict[str, Any],
) -> Path:
    cfg = _runtime_environment_config(task_descriptor)
    raw = str(cfg.get("cwd") or "task_project").strip()
    if raw in {"", ".", "task_project"}:
        cwd = task_project_path
    elif raw == "run_dir":
        cwd = run_dir
    else:
        cwd = _runtime_path(
            raw,
            task_project_path=task_project_path,
            field_name="runtime_environment.cwd",
        )
        if cwd is None:
            cwd = task_project_path
    if not cwd.exists():
        if raw == "run_dir":
            cwd.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"runtime_environment.cwd not found: {cwd}")
    if not cwd.is_dir():
        raise ValueError(f"runtime_environment.cwd is not a directory: {cwd}")
    return cwd.resolve()


def _runtime_path(value: Any, *, task_project_path: Path, field_name: str) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = task_project_path / path
    return Path(os.path.abspath(path))


def _runtime_paths(value: Any, *, task_project_path: Path, field_name: str) -> list[Path]:
    if value is None:
        return []
    values = [value] if isinstance(value, (str, os.PathLike)) else value
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a path or list of paths")
    out: list[Path] = []
    for item in values:
        path = _runtime_path(item, task_project_path=task_project_path, field_name=field_name)
        if path is not None:
            out.append(path)
    return out


def _is_safe_env_key(value: str) -> bool:
    if not value or value[0].isdigit():
        return False
    return all(ch.isalnum() or ch == "_" for ch in value)


def _resolved_task_data_dir(
    *,
    task_id: str,
    workspace: Path,
    env: Mapping[str, str],
    data_env_aliases: list[str] | tuple[str, ...] = (),
) -> Path | None:
    direct_keys = [*data_env_aliases, f"PRAXIST_{task_id.upper()}_DATA_DIR", "PRAXIST_DATA_DIR"]
    for key in direct_keys:
        raw = str(env.get(key) or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()

    for key in ("PRAXIST_DATASETS_DIR", "PRAXIST_DATA_ROOT"):
        raw = str(env.get(key) or "").strip()
        if raw:
            return (Path(raw).expanduser() / task_id).resolve()

    candidate = (workspace / "data" / task_id).resolve()
    if candidate.exists():
        return candidate
    return None


def _prepend_path_env(path: str, existing: str) -> str:
    parts = [part for part in existing.split(os.pathsep) if part]
    try:
        duplicate = path in parts or str(Path(path).expanduser().resolve()) in {
            str(Path(part).expanduser().resolve()) for part in parts
        }
    except OSError:
        duplicate = path in parts
    return existing if duplicate else os.pathsep.join([path, *parts]) if parts else path


def prepare_research_loop_plugin_run(
    *,
    task_ref: str | None = None,
    task_project_path: str | Path | None = None,
    task_project: TaskProject | None = None,
    workspace: Path,
    run_dir: Path,
    runtime_ref: str,
    model_provider_ref: str,
    budget_policy_ref: str,
    model: str,
    local_mode: bool,
    frontier_strategy: str,
    credential_profile: str | None = None,
    command: str = "",
    deprecated_args_seen: list[str] | None = None,
    resolve_only: bool = False,
    resume: bool = False,
    resume_policy: str = "completed_generation",
) -> ResearchLoopPluginRun:
    """Resolve task, plugins, credentials, budget, and startup artifacts."""
    workspace = Path(workspace)
    run_dir = Path(run_dir).expanduser().resolve()
    run_id = run_dir.name
    _ensure_safe_run_id(run_id)

    if task_project is None and task_project_path is None:
        raise ValueError("research_loop startup requires task_project or task_project_path")
    project = task_project or resolve_task_project(task_project_path or "", workspace=workspace)
    task_ref = task_ref or project.task_ref
    task_project_path = project.path
    task_descriptor = project.descriptor
    _ensure_run_dir_not_in_system_repo(run_dir)
    _ensure_fresh_run_dir(run_dir, resume=resume)
    _validate_research_loop_task_eligibility(task_ref, task_descriptor)
    _reject_enabled_optional_stubs(task_descriptor)
    disabled_optional = _disabled_optional_from_descriptor(task_descriptor)
    loader = PluginLoader(PluginRoots.defaults(workspace, task_path=task_project_path))
    discovery = loader.discover()
    plugin_refs = _plugin_refs_from_task_descriptor(task_descriptor)
    plugin_refs.extend([runtime_ref, model_provider_ref, budget_policy_ref])
    resolution_manifest = loader.resolve(
        _dedupe_refs(plugin_refs),
        discovery,
        run_id=run_id,
        root_task_ref=task_ref,
        disabled_optional=disabled_optional,
        enforce_bundled_execution=True,
    )
    assert_bundled_execution_manifest(resolution_manifest)
    registry = loader.load(resolution_manifest)
    _validate_runtime_provider_compatibility(runtime_ref, model_provider_ref, registry)

    task_spec = load_task_spec(str(project.descriptor_path))
    task_spec, effective_task_descriptor, env_overrides_seen = _apply_internal_env_overrides(
        task_spec,
        task_descriptor,
        os.environ,
    )
    tool_server_refs = effective_research_tool_server_refs_from_task_descriptor(
        effective_task_descriptor
    )
    peer_role_refs = _peer_role_refs_from_task_descriptor(
        effective_task_descriptor,
        registry,
        task_project_path=task_project_path,
    )
    peer_role_ref = peer_role_refs[0] if peer_role_refs else "role:peer"
    effective_model = normalize_model_for_provider(
        model_provider_ref,
        model or _default_model_for_provider(model_provider_ref),
        registry,
    )
    validate_model_for_provider(model_provider_ref, effective_model, registry)
    resolver = CredentialResolver()
    credential_set, model_provider_credential = resolve_model_credential_for_runtime(
        resolver.discover(profile=credential_profile),
        runtime_ref=runtime_ref,
        model_provider_ref=model_provider_ref,
        registry=registry,
        resolve_only=resolve_only,
    )
    task_execution_cwd = _task_execution_cwd(
        task_project_path=task_project_path,
        run_dir=run_dir,
        task_descriptor=effective_task_descriptor,
    )
    task_runtime_env = _task_runtime_env(
        task_project_path=task_project_path,
        workspace=workspace,
        task_id=project.task_id,
        env=os.environ,
        task_descriptor=effective_task_descriptor,
        evaluation_entrypoint=task_spec.toolchain.eval_entrypoint,
    )
    provider_env = freeze_provider_env(model_provider_ref, os.environ)
    provider_env.update(task_runtime_env)
    credential_manager = CredentialFailoverManager(credential_set)
    cache_mode, runtime_cache_strategy, provider_cache_strategy = (
        _cache_strategy_for_runtime_provider(
            runtime_ref,
            model_provider_ref,
            registry,
        )
    )
    cache_policy = build_cache_policy(
        mode=cache_mode,
        frozen_prefix_parts={
            "task_ref": task_ref,
            "workflow_stage": "workflow_stage:research_loop",
            "panel_topology": task_descriptor.get("praxist_plugins", {})
            .get("panel", {})
            .get("topology"),
            "roles": task_descriptor.get("praxist_plugins", {}).get("panel", {}).get("roles", []),
            "runtime_ref": runtime_ref,
            "model_provider_ref": model_provider_ref,
        },
        cache_breakpoints=["system_prompt", "task_brief", "role_skill"],
        runtime_cache_strategy=runtime_cache_strategy,
        provider_cache_strategy=provider_cache_strategy,
    )

    startup_config = {
        "schema_version": "praxist.startup.v1",
        "command": command or f"python -m praxist.run run --task-path {task_project_path}",
        "canonical_args": {
            "task": task_ref,
            "task_path": str(task_project_path),
            "runtime": runtime_ref,
            "model_provider": model_provider_ref,
            "budget_policy": budget_policy_ref,
            "model": effective_model,
            "frontier_strategy": frontier_strategy,
            "run_dir": str(run_dir),
        },
        "deprecated_args_seen": deprecated_args_seen or [],
        "env_overrides_seen": env_overrides_seen,
        "runtime_environment": {
            "task_execution_cwd": str(task_execution_cwd),
            "task_runtime_env_keys": sorted(task_runtime_env),
            "task_python": task_runtime_env.get("PRAXIST_TASK_PYTHON"),
            "task_venv": task_runtime_env.get("PRAXIST_TASK_VENV"),
            "task_shell_prefix": task_runtime_env.get("PRAXIST_TASK_SHELL_PREFIX"),
        },
        "plugin_roots": _plugin_roots_payload(
            PluginRoots.defaults(workspace, task_path=task_project_path)
        ),
        "tool_server_refs": list(tool_server_refs),
        "local_mode": local_mode,
        "detached": False,
        "resume": {
            "enabled": bool(resume),
            "policy": resume_policy,
            "run_dir": str(run_dir) if resume else "",
        },
        "resume_identity": {
            "task_project_manifest_sha256": project.manifest["sha256"],
            "effective_task_descriptor_sha256": hashlib.sha256(
                json.dumps(
                    effective_task_descriptor, sort_keys=True, default=str, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "local_mode": bool(local_mode),
        },
    }
    previous_run_metadata = (
        validate_resume_startup_identity(
            run_dir,
            startup_config,
            candidate_task_project_manifest=project.manifest,
        )
        if resume
        else {}
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_dirs(run_dir)
    _touch_required_jsonl(run_dir)

    trajectory = TrajectoryWriter(run_dir, run_id)
    source_snapshot = build_core_source_snapshot()
    startup_time = utc_now()
    run_metadata = {
        "schema_version": "praxist.run.v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "task_ref": task_ref,
        "workflow_ref": "workflow_stage:research_loop",
        "status": "running",
        "created_at": previous_run_metadata.get("created_at") or startup_time,
        "started_at": startup_time,
        "finalized_at": None,
        "praxist_version": __version__,
        "git_commit": source_snapshot["git_commit"],
        "workspace_hash": source_snapshot["workspace_hash"],
        "source_hash_algorithm": source_snapshot["source_hash_algorithm"],
        "source_file_count": source_snapshot["source_file_count"],
        "source_patterns": source_snapshot["source_patterns"],
        "task_project": {
            "path": str(task_project_path),
            "manifest_sha256": project.manifest["sha256"],
            "file_count": len(project.manifest["files"]),
        },
        "schema_versions": {
            "trajectory": "praxist.trajectory.v1",
            "artifact": "praxist.artifact.v1",
            "credentials": "praxist.credentials.v1",
            "cache_policy": "praxist.cache_policy.v1",
        },
    }
    if resume:
        run_metadata["resumed_at"] = startup_time
        run_metadata["resume_count"] = int(previous_run_metadata.get("resume_count") or 0) + 1
        run_metadata["previous_status"] = previous_run_metadata.get("status")
    write_json(run_dir / "run.json", run_metadata)
    write_json(run_dir / "startup_config.json", startup_config)
    write_task_project_manifest(run_dir, project)
    write_json(run_dir / "plugin_resolution.json", resolution_manifest)
    write_json(
        run_dir / "credentials_redacted.json",
        _credential_snapshot(resolver, credential_set, credential_manager),
    )
    model_profiles = model_profiles_snapshot(
        provider_ref=model_provider_ref,
        runtime_ref=runtime_ref,
        credential_mode=credential_set.mode,
        cache_policy=cache_policy,
        selected_model=effective_model,
        registry=registry,
    )
    model_profiles["selected_defaults"] = _model_profile_defaults_from_task_descriptor(
        effective_task_descriptor,
        registry,
        task_project_path=task_project_path,
    )
    model_profiles["runtime_provider_conformance"] = _runtime_provider_conformance_snapshot(
        runtime_ref,
        model_provider_ref,
        cache_policy,
        registry,
    )
    write_json(run_dir / "model_profiles.json", model_profiles)
    write_json(
        run_dir / "cache_policy.json",
        {"schema_version": "praxist.cache_policy.v1", **cache_policy.to_dict()},
    )
    (run_dir / "effective_task_spec.yaml").write_text(
        yaml.safe_dump(effective_task_descriptor, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    trajectory.emit(
        "run.started",
        actor={"type": "core", "id": "startup"},
        payload={"task_ref": task_ref},
    )
    if resume:
        trajectory.emit(
            "run.resumed",
            actor={"type": "core", "id": "startup"},
            payload={"task_ref": task_ref, "resume_policy": resume_policy},
        )
    trajectory.emit(
        "startup.parsed", actor={"type": "core", "id": "startup"}, payload=startup_config
    )
    trajectory.emit(
        "task.resolved",
        actor={"type": "core", "id": "startup"},
        payload={
            "task_ref": task_ref,
            "task_project_path": str(task_project_path),
            "task_project_manifest_sha256": project.manifest["sha256"],
        },
    )
    trajectory.emit(
        "plugins.resolved",
        actor={"type": "core", "id": "registry"},
        payload={
            "selected": [
                item["metadata"]["kind"] + ":" + item["metadata"]["name"]
                for item in resolution_manifest["selected"]
            ]
        },
    )
    trajectory.emit(
        "registry.frozen",
        actor={"type": "core", "id": "registry"},
        payload={"plugin_count": len(resolution_manifest["selected"])},
    )
    trajectory.emit(
        "tool_servers.resolved",
        actor={"type": "core", "id": "registry"},
        payload={"tool_server_refs": list(tool_server_refs)},
    )
    trajectory.emit(
        "workflow.stage_started",
        scope={"stage_id": "research_loop"},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={"implementation_backend": "GenerationLoop", "task_ref": task_ref},
    )
    emit_disabled_optional_events(
        trajectory,
        stages=[item for item in disabled_optional if "stage_id" in item and "ref" in item],
        tools=[item for item in disabled_optional if "tool_ref" in item],
    )
    stage_budget_grant_id = _grant_stage_budget(
        run_dir=run_dir,
        run_id=run_id,
        task_ref=task_ref,
        task_spec=task_spec,
        budget_policy_ref=budget_policy_ref,
        trajectory=trajectory,
        registry=registry,
    )

    return ResearchLoopPluginRun(
        task_ref=task_ref,
        run_id=run_id,
        run_dir=run_dir,
        task_spec=task_spec,
        task_project_path=task_project_path,
        task_descriptor=task_descriptor,
        task_project_manifest=project.manifest,
        task_execution_cwd=task_execution_cwd,
        task_runtime_env=task_runtime_env,
        runtime_ref=runtime_ref,
        model_provider_ref=model_provider_ref,
        budget_policy_ref=budget_policy_ref,
        registry=registry,
        tool_server_refs=tool_server_refs,
        peer_role_ref=peer_role_ref,
        peer_role_refs=peer_role_refs or (peer_role_ref,),
        credential_set=credential_set,
        credential_manager=credential_manager,
        model_provider_credential_key_id=(
            model_provider_credential.key_id if model_provider_credential is not None else None
        ),
        provider_env=provider_env,
        resolution_manifest=resolution_manifest,
        startup_config=startup_config,
        trajectory=trajectory,
        stage_budget_grant_id=stage_budget_grant_id,
    )


def finalize_research_loop_plugin_run(
    prepared: ResearchLoopPluginRun,
    *,
    success: bool,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    exit_code: int | None = None,
) -> None:
    """Write terminal run summary, materialized views, and trajectory events."""
    status = "succeeded" if success else "failed"
    materialization_error = None
    try:
        canonical_outputs = _materialize_legacy_outputs(prepared, result or {})
    except Exception as exc:  # noqa: BLE001 - finalization must preserve the run summary.
        materialization_error = str(exc)
        canonical_outputs = {
            "finding_count": 0,
            "frontier_count": 0,
            "gems_count": 0,
            "research_memory_record_count": 0,
            "graph_edge_count": 0,
            "graph_artifact_count": 0,
        }
    trajectory = TrajectoryWriter(prepared.run_dir, prepared.run_id)
    trajectory.emit(
        "workflow.stage_succeeded" if success else "workflow.stage_failed",
        scope={"stage_id": "research_loop"},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={
            "implementation_backend": "GenerationLoop",
            "result": result or {},
            "error": error,
            "findings": canonical_outputs["finding_count"],
            "frontier_records": canonical_outputs["frontier_count"],
            "gems": canonical_outputs.get("gems_count", 0),
            "research_memory_records": canonical_outputs.get("research_memory_record_count", 0),
            "graph_edges": canonical_outputs.get("graph_edge_count", 0),
            "graph_artifacts": canonical_outputs.get("graph_artifact_count", 0),
            "materialization_error": materialization_error,
        },
    )
    normalized_result = result or {}
    summary = {
        "schema_version": "praxist.run_summary.v1",
        "run_id": prepared.run_id,
        "status": status,
        "exit_code": 0 if success else (exit_code if exit_code is not None else 1),
        "task_id": prepared.task_spec.task_id,
        "task_name": prepared.task_spec.task_name,
        "generations_completed": resume_state.reported_completed_generations(
            normalized_result,
            prepared.run_dir,
        ),
        "max_generations": normalized_result.get(
            "max_generations",
            prepared.task_spec.generation_policy.max_generations,
        ),
        "exit_condition": normalized_result.get("exit_condition", status),
        "run_dir": str(prepared.run_dir),
        "frontier_summary": normalized_result.get("frontier_summary", []),
        "gems": normalized_result.get("gems", {}),
        "gems_count": _gems_count_from_result(normalized_result),
        "frontier_records": canonical_outputs["frontier_count"],
        "gem_records": canonical_outputs.get("gems_count", 0),
        "research_memory_records": canonical_outputs.get("research_memory_record_count", 0),
        "graph_edges": canonical_outputs.get("graph_edge_count", 0),
        "graph_artifacts": canonical_outputs.get("graph_artifact_count", 0),
        "stage_summary": {"research_loop": status},
        "legacy_generation_loop_summary": result or {},
        "task_ref": prepared.task_ref,
        "runtime_ref": prepared.runtime_ref,
        "model_provider_ref": prepared.model_provider_ref,
        "budget_policy_ref": prepared.budget_policy_ref,
        "credential_mode": prepared.credential_set.mode,
        "credential_failover": prepared.credential_manager.snapshot(),
        "output_hashes": output_ledger_hashes(prepared.run_dir),
        "error": error,
        "materialization_error": materialization_error,
    }
    summary = write_run_summary(prepared.run_dir / "run_summary.json", summary)
    write_json(
        prepared.run_dir / "credentials_redacted.json",
        {
            **json.loads(
                (prepared.run_dir / "credentials_redacted.json").read_text(encoding="utf-8")
            ),
            "failover": prepared.credential_manager.snapshot(),
        },
    )
    trajectory = TrajectoryWriter(prepared.run_dir, prepared.run_id)
    trajectory.emit(
        "run.finalized",
        actor={"type": "core", "id": "startup"},
        payload=summary,
    )
    run_json = json.loads((prepared.run_dir / "run.json").read_text(encoding="utf-8"))
    run_json["status"] = status
    run_json["finalized_at"] = utc_now()
    write_json(prepared.run_dir / "run.json", run_json)
    run_report.generate_terminal_report_safely(prepared, summary)


def _gems_count_from_result(result: dict[str, Any]) -> int:
    gems = result.get("gems") if isinstance(result, dict) else None
    if not isinstance(gems, dict):
        return 0
    for key in ("gems", "entries"):
        value = gems.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _plugin_refs_from_task_descriptor(descriptor: dict[str, Any]) -> list[str]:
    refs = [ref.as_string() for ref in task_project_global_plugin_refs(descriptor)]
    refs.extend(effective_research_tool_server_refs_from_task_descriptor(descriptor))
    return _dedupe_refs(refs)


def _task_workflow_stage_ref(descriptor: dict[str, Any]) -> str | None:
    plugins = descriptor.get("praxist_plugins") or {}
    workflow = plugins.get("workflow") or {}
    stage_ref = workflow.get("stage") if isinstance(workflow, dict) else None
    return str(stage_ref) if stage_ref else None


def _validate_research_loop_task_eligibility(task_ref: str, descriptor: dict[str, Any]) -> None:
    declared_task_ref = (descriptor.get("praxist_plugins") or {}).get("task_ref")
    if declared_task_ref and str(declared_task_ref) != task_ref:
        raise ValueError(
            f"task descriptor task_ref mismatch: expected {task_ref}, got {declared_task_ref}"
        )
    stage_ref = _task_workflow_stage_ref(descriptor)
    if stage_ref != RESEARCH_LOOP_STAGE_REF:
        raise ValueError(
            f"{task_ref} is not eligible for research_loop startup: "
            f"praxist_plugins.workflow.stage must be {RESEARCH_LOOP_STAGE_REF}, got {stage_ref!r}"
        )


def _validate_runtime_provider_compatibility(
    runtime_ref: str,
    model_provider_ref: str,
    registry: Any | None,
) -> None:
    runtime_contract = _runtime_contract(runtime_ref, registry)
    compatible = [
        str(item) for item in runtime_contract.get("compatible_model_providers") or [] if item
    ]
    if compatible and model_provider_ref not in compatible:
        raise ValueError(
            f"{runtime_ref} is not compatible with {model_provider_ref}; "
            f"compatible providers: {', '.join(sorted(compatible))}"
        )


def _runtime_provider_conformance_snapshot(
    runtime_ref: str,
    model_provider_ref: str,
    cache_policy: Any,
    registry: Any | None,
) -> dict[str, Any]:
    runtime_contract = _runtime_contract(runtime_ref, registry)
    provider_contract = _provider_contract(model_provider_ref, registry)
    return {
        "schema_version": "praxist.runtime_provider_conformance.v1",
        "runtime_ref": runtime_ref,
        "model_provider_ref": model_provider_ref,
        "runtime_cache_strategy": runtime_contract.get("cache_strategy"),
        "provider_cache_strategy": provider_contract.get("cache_strategy"),
        "cache_mode": cache_policy.mode,
        "cache_policy_runtime_strategy": cache_policy.runtime_cache_strategy,
        "cache_policy_provider_strategy": cache_policy.provider_cache_strategy,
        "runtime_usage_reporting": runtime_contract.get("usage_reporting"),
        "provider_usage_reporting": provider_contract.get("usage_reporting"),
        "event_schema": runtime_contract.get("event_schema"),
    }


def _runtime_contract(runtime_ref: str, registry: Any | None) -> dict[str, Any]:
    return _manifest_contract(runtime_ref, "runtime", registry)


def _provider_contract(model_provider_ref: str, registry: Any | None) -> dict[str, Any]:
    return _manifest_contract(model_provider_ref, "provider", registry)


def _manifest_contract(ref: str, key: str, registry: Any | None) -> dict[str, Any]:
    if registry is None:
        return {}
    try:
        selected = registry.descriptor_for_ref(ref)
        value = (
            yaml.safe_load((Path(selected.path) / "plugin.yaml").read_text(encoding="utf-8")) or {}
        )
    except Exception:
        return {}
    if not isinstance(value, dict):
        return {}
    contract = value.get(key)
    return dict(contract) if isinstance(contract, dict) else {}


def _disabled_optional_from_descriptor(descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    disabled = []
    for stage_id, config in _optional_workflow_stage_entries(descriptor):
        if bool(config.get("enabled", False)):
            continue
        disabled.append(
            {
                "stage_id": str(stage_id),
                "ref": str(config["ref"]),
                "enabled": False,
                "reason": "optional_stage_disabled",
            }
        )
    for role_id, config in _optional_role_entries(descriptor):
        if bool(config.get("enabled", False)):
            continue
        disabled.append(
            {
                "stage_id": "research_loop",
                "role_id": str(role_id),
                "role_ref": str(config["role"]),
                "tool_ref": str(config.get("tool_server_ref") or ""),
                "enabled": False,
                "reason": "optional_role_disabled",
            }
        )
    return disabled


def _reject_enabled_optional_stubs(descriptor: dict[str, Any]) -> None:
    enabled = []
    enabled.extend(
        f"workflow stage {stage_id}"
        for stage_id, config in _optional_workflow_stage_entries(descriptor)
        if bool(config.get("enabled", False))
    )
    enabled.extend(
        f"research_loop optional role {role_id}"
        for role_id, config in _optional_role_entries(descriptor)
        if bool(config.get("enabled", False))
    )
    if enabled:
        raise ValueError(
            "optional Step 10 stubs are enabled but no implementation is configured: "
            + ", ".join(enabled)
        )


def _optional_workflow_stage_entries(
    descriptor: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    raw = descriptor.get("praxist_plugins", {}).get("optional_workflow_stages") or {}
    entries = []
    if isinstance(raw, dict):
        for stage_id, value in raw.items():
            if isinstance(value, dict):
                ref = value.get("ref") or value.get("stage") or value.get("workflow_stage")
                enabled = bool(value.get("enabled", False))
            else:
                ref = value
                enabled = False
            if ref:
                entries.append((str(stage_id), {"ref": str(ref), "enabled": enabled}))
    return entries


def _optional_role_entries(descriptor: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    panel = descriptor.get("praxist_plugins", {}).get("panel") or {}
    raw = panel.get("optional_roles") or {}
    entries = []
    if isinstance(raw, dict):
        for role_id, value in raw.items():
            if not isinstance(value, dict):
                continue
            role_ref = value.get("role") or value.get("role_ref")
            if not role_ref:
                continue
            entries.append(
                (
                    str(role_id),
                    {
                        "role": str(role_ref),
                        "tool_server_ref": str(
                            value.get("tool_server_ref") or value.get("tool") or ""
                        ),
                        "enabled": bool(value.get("enabled", False)),
                    },
                )
            )
    return entries


def _read_task_descriptor(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"task descriptor must be an object: {path}")
    return value


def _selected_plugin_path(manifest: dict[str, Any], ref: PluginRef) -> Path:
    for selected in manifest.get("selected", []):
        metadata = selected.get("metadata", {})
        if metadata.get("kind") == ref.kind and metadata.get("name") == ref.name:
            return Path(str(selected["path"])).expanduser().resolve()
    raise ValueError(f"Selected plugin missing from manifest: {ref.as_string()}")


def _dedupe_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
    return out


def _touch_required_jsonl(run_dir: Path) -> None:
    for rel in (
        "artifact_index.jsonl",
        "budget_ledger.jsonl",
        "findings/findings.jsonl",
        "findings/frontier.jsonl",
        "memory/research_memory.jsonl",
        "memory/graph_edges.jsonl",
    ):
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


PRAXIST_RUN_ARTIFACT_REL_PATHS = (
    "run.json",
    "startup_config.json",
    "trajectory.jsonl",
    "budget_ledger.jsonl",
    "artifact_index.jsonl",
    "run_summary.json",
    "plugin_resolution.json",
    "effective_task_spec.yaml",
    "model_profiles.json",
    "cache_policy.json",
    "task_project_manifest.json",
    "credentials_redacted.json",
    "findings",
    "frontier",
    "agendas",
    "memory",
    "gems",
    "gen_0",
)


def _ensure_fresh_run_dir(run_dir: Path, *, resume: bool = False) -> None:
    if resume:
        ensure_resumable_run_dir(run_dir)
        return
    if not run_dir.exists():
        return
    has_artifacts = any((run_dir / rel).exists() for rel in PRAXIST_RUN_ARTIFACT_REL_PATHS) or any(
        child.name.startswith("gen_") for child in run_dir.iterdir()
    )
    if has_artifacts:
        raise ValueError(
            f"run_dir already contains Praxist run artifacts: {run_dir}. "
            "Use --resume or --resume-from to continue this run, or choose a fresh run directory."
        )
    blocking_paths = [path for path in run_dir.iterdir() if not _is_ignorable_precreated_path(path)]
    if blocking_paths:
        raise ValueError(
            f"run_dir already exists and is not empty: {run_dir}. Choose a fresh run directory."
        )


def _is_ignorable_precreated_path(path: Path) -> bool:
    if path.is_file():
        return path.name in {".DS_Store", ".gitkeep"}
    if path.is_dir():
        if path.name == "logs":
            return all(child.name in {".gitkeep", "launcher.nohup.log"} for child in path.iterdir())
        if path.name in {
            "findings",
            "frontier",
            "agendas",
            "memory",
            "gems",
        } or path.name.startswith("gen_"):
            return False
        return not any(path.iterdir())
    return False


def _ensure_run_dir_not_in_system_repo(run_dir: Path) -> None:
    system_root = Path(__file__).resolve().parents[4]
    resolved = Path(run_dir).expanduser().resolve()
    try:
        resolved.relative_to(system_root.resolve())
    except ValueError:
        return
    raise ValueError(
        f"run_dir must live outside the Praxist source checkout: {resolved}. "
        "Use a task-local experiments directory or another explicit external path."
    )


def _ensure_safe_run_id(run_id: str) -> None:
    hits = scan_text(run_id)
    if hits:
        raise ValueError(f"run_id contains secret-looking content: {','.join(sorted(set(hits)))}")


def _grant_stage_budget(
    *,
    run_dir: Path,
    run_id: str,
    task_ref: str,
    task_spec: TaskSpec,
    budget_policy_ref: str,
    trajectory: TrajectoryWriter,
    registry: Any | None = None,
) -> str | None:
    requested = planned_research_loop_usage(task_spec)
    request = BudgetRequest(
        request_id="budget_request_research_loop_start",
        requester_id="workflow_stage:research_loop",
        experiment_id=f"{task_ref}/research_loop",
        model_profile_ref="",
        requested=requested,
        expected_value={
            "confidence": "strong",
            "value": "stage_execution",
            "requires_full_stage_budget": True,
        },
        evidence_refs=[task_ref],
        cheaper_alternatives=[],
        abort_conditions=["stage_startup_failed"],
    )
    decision = policy_for_ref(budget_policy_ref, registry=registry).decide(request)
    ledger = BudgetLedger(run_dir, run_id)
    if decision.grant and decision.grant.grant_id in ledger.active_grants():
        return decision.grant.grant_id
    ledger.append_request(
        request,
        actor_ref="workflow_stage:research_loop",
        stage_id="research_loop",
        action_type="stage_start",
        reason="legacy_research_loop_stage_budget_request",
    )
    ledger.append_decision(
        request,
        decision,
        actor_ref=budget_policy_ref,
        stage_id="research_loop",
        action_type="stage_start",
        reason="legacy_research_loop_stage_budget_decision",
    )
    trajectory.emit(
        "budget.requested",
        scope={"stage_id": "research_loop"},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={"request_id": request.request_id, "requested": request.requested},
    )
    trajectory.emit(
        "budget.granted" if decision.grant else "budget.review_required",
        scope={
            "stage_id": "research_loop",
            "grant_id": decision.grant.grant_id if decision.grant else "",
        },
        actor={"type": "budget_policy", "id": budget_policy_ref},
        payload={"request_id": request.request_id, "decision": decision.to_dict()},
    )
    return decision.grant.grant_id if decision.grant else None


def _credential_snapshot(
    resolver: CredentialResolver,
    credential_set: CredentialSet,
    credential_manager: CredentialFailoverManager,
) -> dict[str, Any]:
    snapshot = resolver.snapshot(credential_set)
    snapshot["failover"] = credential_manager.snapshot()
    return snapshot


def _plugin_roots_payload(plugin_roots: PluginRoots) -> dict[str, list[str]]:
    return {
        "bundled": [str(path) for path in plugin_roots.bundled],
        "project": [str(path) for path in plugin_roots.project],
        "user": [str(path) for path in plugin_roots.user],
        "task_project": [str(path) for path in plugin_roots.task_project],
    }
