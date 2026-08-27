"""Fake workflow fixture for core-plugin conformance runs."""

from __future__ import annotations

import shutil
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
    find_model_provider_credential,
    require_model_provider_credential,
)
from praxist.core.modeling import (
    default_model_profile,
    model_profiles_snapshot,
    provider_for_ref,
)
from praxist.core.protocol import AgentRunRequest, BudgetRequest, EnvPolicy, ToolPermissionSet
from praxist.core.registry import PluginLoader, PluginRoots, assert_bundled_execution_manifest
from praxist.core.runtimes import runtime_for_ref
from praxist.core.source_snapshot import build_core_source_snapshot
from praxist.core.storage import (
    ArtifactWriter,
    append_jsonl,
    ensure_run_dirs,
    new_run_id,
    output_ledger_hashes,
    utc_now,
    write_json,
)
from praxist.core.task_project import (
    TaskProject,
    task_project_global_plugin_refs,
    write_task_project_manifest,
)
from praxist.core.trajectory import TrajectoryWriter
from praxist.core.workflow import (
    disabled_optional_stages,
    disabled_optional_tools,
    emit_disabled_optional_events,
)
from praxist.plugins.workflow_stages.research_loop.lifecycle import (
    record_generation_finished_safely,
)


def run_fake_workflow_fixture(
    *,
    workspace: Path,
    task_ref: str = "task:fake_panel",
    task_project: TaskProject | None = None,
    run_dir: Path | None = None,
    runtime_ref: str = "agent_runtime:fake_runtime",
    model_provider_ref: str = "model_provider:fake_provider",
    budget_policy_ref: str = "budget_policy:fake_tiered",
    credential_profile: str | None = None,
    resolve_only: bool = False,
    run_lifecycle_observer: Any | None = None,
) -> dict[str, Any]:
    """Run the fake workflow fixture used by conformance tests and task templates."""
    if task_project is not None:
        task_ref = task_project.task_ref
    task_slug = task_ref.split(":", 1)[1]
    run_id = new_run_id(task_slug)
    if run_dir is None:
        run_dir = workspace / "runs" / run_id
    resolver = CredentialResolver()
    credential_set = resolver.discover(profile=credential_profile)
    if resolve_only:
        # Resolve-only is a no-LLM-call smoke test; allow it to run
        # without a provider credential. See issue #86 / startup.py
        # for the matching short-circuit on the real workflow path.
        find_model_provider_credential(credential_set, model_provider_ref)
    else:
        require_model_provider_credential(credential_set, model_provider_ref)
    credential_manager = CredentialFailoverManager(credential_set)
    _claim_run_dir(run_dir)
    ensure_run_dirs(run_dir)
    _touch_run_ledgers(run_dir)

    trajectory = TrajectoryWriter(run_dir, run_id)
    artifacts = ArtifactWriter(run_dir, trajectory)
    disabled_optional = [*disabled_optional_stages(), *disabled_optional_tools()]
    cache_mode, runtime_cache_strategy, provider_cache_strategy = (
        _cache_strategy_for_runtime_provider(
            runtime_ref,
            model_provider_ref,
        )
    )
    cache_policy = build_cache_policy(
        mode=cache_mode,
        frozen_prefix_parts={
            "task_ref": task_ref,
            "panel_topology": "panel_topology:fake_two_round",
            "roles": ["role:fake_peer", "role:fake_pi", "role:fake_chair"],
            "protocol": "gate_b",
        },
        cache_breakpoints=["system_prompt", "task_brief"],
        runtime_cache_strategy=runtime_cache_strategy,
        provider_cache_strategy=provider_cache_strategy,
    )

    source_snapshot = build_core_source_snapshot()
    run_metadata: dict[str, Any] = {
        "schema_version": "praxist.run.v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "task_ref": task_ref,
        "workflow_ref": "workflow_stage:research_loop",
        "status": "running",
        "created_at": utc_now(),
        "started_at": utc_now(),
        "finalized_at": None,
        "praxist_version": __version__,
        "git_commit": source_snapshot["git_commit"],
        "workspace_hash": source_snapshot["workspace_hash"],
        "source_hash_algorithm": source_snapshot["source_hash_algorithm"],
        "source_file_count": source_snapshot["source_file_count"],
        "source_patterns": source_snapshot["source_patterns"],
        "task_project": (
            {
                "path": str(task_project.path),
                "manifest_sha256": task_project.manifest["sha256"],
                "file_count": len(task_project.manifest["files"]),
            }
            if task_project is not None
            else None
        ),
        "schema_versions": {
            "trajectory": "praxist.trajectory.v1",
            "artifact": "praxist.artifact.v1",
            "credentials": "praxist.credentials.v1",
            "cache_policy": "praxist.cache_policy.v1",
        },
    }
    write_json(run_dir / "run.json", run_metadata)
    trajectory.emit(
        "run.started", actor={"type": "core", "id": "startup"}, payload={"task_ref": task_ref}
    )

    plugin_roots = _fixture_plugin_roots(workspace)
    loader = PluginLoader(plugin_roots)
    plugin_refs = _plugin_refs(
        task_ref, runtime_ref, model_provider_ref, budget_policy_ref, task_project
    )
    trajectory.emit(
        "plugin.discovery_started",
        actor={"type": "core", "id": "registry"},
        payload={"roots": _plugin_roots_payload(plugin_roots)},
    )
    discovery = loader.discover()
    trajectory.emit(
        "plugin.discovery_finished",
        actor={"type": "core", "id": "registry"},
        payload={
            "candidate_count": len(discovery.candidates),
            "warning_count": len(discovery.warnings),
        },
    )
    trajectory.emit(
        "plugin.resolution_started",
        actor={"type": "core", "id": "registry"},
        payload={"root_task_ref": task_ref, "requested": plugin_refs},
    )
    resolution_manifest = loader.resolve(
        plugin_refs,
        discovery,
        run_id=run_id,
        root_task_ref=task_ref,
        disabled_optional=disabled_optional,
        enforce_bundled_execution=True,
    )
    assert_bundled_execution_manifest(resolution_manifest)
    registry = loader.load(resolution_manifest)
    try:
        _validate_runtime_provider_compatibility(runtime_ref, model_provider_ref, registry)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    startup_config = {
        "schema_version": "praxist.startup.v1",
        "command": (
            f"python -m praxist.run run --task-path {task_project.path}"
            if task_project is not None
            else f"python -m praxist.run run --task {task_ref}"
        ),
        "canonical_args": {
            "task": task_ref,
            "task_path": str(task_project.path) if task_project is not None else None,
            "runtime": runtime_ref,
            "model_provider": model_provider_ref,
            "budget_policy": budget_policy_ref,
        },
        "legacy_args_seen": [],
        "env_overrides_seen": [],
        "plugin_roots": _plugin_roots_payload(plugin_roots),
        "local_mode": True,
        "detached": False,
    }
    write_json(run_dir / "startup_config.json", startup_config)
    if task_project is not None:
        write_task_project_manifest(run_dir, task_project)
    write_json(
        run_dir / "credentials_redacted.json",
        _credential_snapshot(resolver, credential_set, credential_manager),
    )
    write_json(
        run_dir / "model_profiles.json",
        model_profiles_snapshot(
            provider_ref=model_provider_ref,
            runtime_ref=runtime_ref,
            credential_mode=credential_set.mode,
            cache_policy=cache_policy,
            registry=registry,
        ),
    )
    write_json(
        run_dir / "cache_policy.json",
        {"schema_version": "praxist.cache_policy.v1", **cache_policy.to_dict()},
    )
    (run_dir / "effective_task_spec.yaml").write_text(
        _effective_task_spec(
            task_ref, runtime_ref, model_provider_ref, budget_policy_ref, task_project
        ),
        encoding="utf-8",
    )

    write_json(run_dir / "plugin_resolution.json", resolution_manifest)

    trajectory.emit(
        "startup.parsed", actor={"type": "core", "id": "startup"}, payload=startup_config
    )
    trajectory.emit(
        "task.resolved",
        actor={"type": "core", "id": "startup"},
        payload={
            "task_ref": task_ref,
            "task_project_path": str(task_project.path) if task_project is not None else None,
            "task_project_manifest_sha256": (
                task_project.manifest["sha256"] if task_project is not None else None
            ),
        },
    )
    trajectory.emit(
        "plugin.resolution_finished",
        actor={"type": "core", "id": "registry"},
        payload={
            "selected_count": len(resolution_manifest["selected"]),
            "shadowed_count": len(resolution_manifest["shadowed"]),
        },
    )
    trajectory.emit(
        "plugins.resolved",
        actor={"type": "core", "id": "registry"},
        payload={"selected": plugin_refs},
    )
    trajectory.emit(
        "registry.frozen",
        actor={"type": "core", "id": "registry"},
        payload={
            "plugin_count": len(resolution_manifest["selected"]),
            "agent_runtime_count": len(registry.list("agent_runtime")),
            "model_provider_count": len(registry.list("model_provider")),
        },
    )

    stage_event = trajectory.emit(
        "workflow.stage_started",
        scope={"stage_id": "research_loop"},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={"topology": "panel_topology:fake_two_round"},
    )
    emit_disabled_optional_events(trajectory)

    grant_id = _budget(
        run_dir, trajectory, run_id, stage_event["event_id"], budget_policy_ref, registry, task_ref
    )

    if resolve_only:
        trajectory.emit(
            "workflow.stage_succeeded",
            scope={"stage_id": "research_loop"},
            actor={"type": "workflow_stage", "id": "research_loop"},
            payload={"findings": 0, "frontier_records": 0, "exit_condition": "resolve_only"},
        )
        run_metadata["status"] = "succeeded"
        run_metadata["finalized_at"] = utc_now()
        write_json(run_dir / "run.json", run_metadata)
        summary = {
            "schema_version": "praxist.run_summary.v1",
            "run_id": run_id,
            "status": "succeeded",
            "exit_code": 0,
            "stage_summary": {"research_loop": "succeeded"},
            "finding_summary": {"drafts": 0, "accepted": 0, "retry_corrections": 0},
            "frontier_records": 0,
            "credential_mode": credential_set.mode,
            "runtime_ref": runtime_ref,
            "model_provider_ref": model_provider_ref,
            "budget_policy_ref": budget_policy_ref,
            "cache_frozen_prefix_hash": cache_policy.frozen_prefix_hash,
            "registry_plugin_count": len(resolution_manifest["selected"]),
            "credential_failover": credential_manager.snapshot(),
            "exit_condition": "resolve_only",
            "output_hashes": output_ledger_hashes(run_dir),
        }
        write_json(run_dir / "run_summary.json", summary)
        trajectory.emit("run.finalized", actor={"type": "core", "id": "startup"}, payload=summary)
        return {"run_id": run_id, "run_dir": str(run_dir), "status": "resolved"}

    accepted_finding_id = None
    retry_source_id = None
    finding_event_ids: list[str] = []
    for idx in range(3):
        finding_id, event_id = _peer_finding(
            run_dir,
            trajectory,
            artifacts,
            run_id,
            idx,
            runtime_ref=runtime_ref,
            model_provider_ref=model_provider_ref,
            credential_set=credential_set,
            credential_manager=credential_manager,
            cache_policy=cache_policy,
            budget_grant_id=grant_id,
            registry=registry,
            task_ref=task_ref,
        )
        finding_event_ids.append(event_id)
        if idx == 0:
            accepted_finding_id = finding_id
            _audit(trajectory, idx, finding_id, "pass", "info", False)
        elif idx == 1:
            _audit(trajectory, idx, finding_id, "fail", "blocking", True)
        else:
            retry_source_id = finding_id
            _audit(trajectory, idx, finding_id, "warning", "warning", False, retry=True)

    retry_finding_id, retry_event_id = _retry_finding(
        run_dir,
        trajectory,
        artifacts,
        run_id,
        retry_source_id or "finding_fake_peer_2",
        runtime_ref=runtime_ref,
        model_provider_ref=model_provider_ref,
        credential_manager=credential_manager,
        cache_policy=cache_policy,
        budget_grant_id=grant_id,
        registry=registry,
        task_ref=task_ref,
    )
    finding_event_ids.append(retry_event_id)
    _audit(trajectory, 3, retry_finding_id, "warning", "warning", False)

    chair_artifact = artifacts.persist_json(
        "report",
        "reports/fake_chair_agenda.json",
        {
            "agenda_id": "agenda_fake_panel",
            "accepted_finding_id": accepted_finding_id,
            "retry_finding_id": retry_finding_id,
            "verdict": "promote accepted deterministic finding",
        },
        schema_ref="core:chair_agenda.v1",
        producer={"stage_id": "research_loop", "role_ref": "role:fake_chair"},
        source_event_ids=finding_event_ids,
    )
    frontier_record = {
        "schema_version": "praxist.frontier.v1",
        "frontier_record_id": "frontier_fake_001",
        "run_id": run_id,
        "finding_id": accepted_finding_id,
        "action": "promoted",
        "baseline_ref": "baseline:fake",
        "metric_name": "deterministic_score",
        "metric_value": 1.0,
        "promotion_reason": "fake chair accepted deterministic finding",
        "decided_by": "role:fake_chair",
        "source_event_ids": finding_event_ids,
        "source_artifact_ids": [chair_artifact["artifact_id"]],
        "artifact_refs": [chair_artifact],
        "created_at": utc_now(),
    }
    append_jsonl(run_dir / "findings" / "frontier.jsonl", frontier_record)
    trajectory.emit(
        "frontier.promoted",
        scope={"stage_id": "research_loop"},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={"finding_id": accepted_finding_id, "frontier_record_id": "frontier_fake_001"},
        artifact_refs=[chair_artifact],
    )
    append_jsonl(
        run_dir / "budget_ledger.jsonl",
        {
            "schema_version": "praxist.budget_ledger.v1",
            "record_id": "budget_usage_001",
            "run_id": run_id,
            "timestamp": utc_now(),
            "kind": "usage",
            "request_id": "budget_request_001",
            "grant_id": grant_id,
            "actor_ref": "workflow_stage:research_loop",
            "stage_id": "research_loop",
            "action_type": "stage_start",
            "requested_budget": {},
            "decision": None,
            "granted_budget": None,
            "actual_usage": {"tokens": 900, "wall_clock_seconds": 1},
            "reason": "fake_usage_recorded",
            "source_event_ids": [],
            "artifact_refs": [],
        },
    )
    trajectory.emit(
        "budget.usage_recorded",
        scope={"stage_id": "research_loop", "grant_id": grant_id},
        actor={"type": "budget_policy", "id": budget_policy_ref},
        payload={"actual_usage": {"tokens": 900, "wall_clock_seconds": 1}},
    )
    trajectory.emit(
        "workflow.stage_succeeded",
        scope={"stage_id": "research_loop"},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={"findings": 4, "frontier_records": 1},
    )

    run_metadata["status"] = "succeeded"
    run_metadata["finalized_at"] = utc_now()
    write_json(run_dir / "run.json", run_metadata)
    write_json(
        run_dir / "credentials_redacted.json",
        _credential_snapshot(resolver, credential_set, credential_manager),
    )
    summary = {
        "schema_version": "praxist.run_summary.v1",
        "run_id": run_id,
        "status": "succeeded",
        "exit_code": 0,
        "stage_summary": {"research_loop": "succeeded"},
        "finding_summary": {"drafts": 3, "accepted": 1, "retry_corrections": 1},
        "frontier_records": 1,
        "credential_mode": credential_set.mode,
        "runtime_ref": runtime_ref,
        "model_provider_ref": model_provider_ref,
        "budget_policy_ref": budget_policy_ref,
        "cache_frozen_prefix_hash": cache_policy.frozen_prefix_hash,
        "registry_plugin_count": len(resolution_manifest["selected"]),
        "credential_failover": credential_manager.snapshot(),
        "output_hashes": output_ledger_hashes(run_dir),
    }
    write_json(run_dir / "run_summary.json", summary)
    trajectory.emit("run.finalized", actor={"type": "core", "id": "startup"}, payload=summary)
    record_generation_finished_safely(
        run_lifecycle_observer,
        generation_ordinal=0,
        planned_peer_count=3,
        results=[{"peer_id": f"gen0_peer{peer_index}", "success": True} for peer_index in range(3)],
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "status": "succeeded"}


def _claim_run_dir(run_dir: Path) -> None:
    """Claim a fresh run directory, including the shell made by ``praxist start``."""

    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return
    blocking: list[Path] = []
    for path in run_dir.iterdir():
        if path.is_file() and path.name in {".DS_Store", ".gitkeep"}:
            continue
        if (
            path.is_dir()
            and path.name == "logs"
            and all(
                child.is_file() and child.name in {".gitkeep", "launcher.nohup.log"}
                for child in path.iterdir()
            )
        ):
            continue
        blocking.append(path)
    if blocking:
        names = ", ".join(sorted(path.name for path in blocking))
        raise FileExistsError(f"run_dir already contains run artifacts: {run_dir} ({names})")


def _plugin_refs(
    task_ref: str,
    runtime_ref: str,
    model_provider_ref: str,
    budget_policy_ref: str,
    task_project: TaskProject | None,
) -> list[str]:
    refs = [runtime_ref, model_provider_ref, budget_policy_ref]
    if task_project is not None:
        refs.extend(
            ref.as_string() for ref in task_project_global_plugin_refs(task_project.descriptor)
        )
    else:
        refs.extend(
            [
                "workflow_stage:research_loop",
                "panel_topology:fake_two_round",
                "role:fake_peer",
                "role:fake_pi",
                "role:fake_chair",
                "audit_rule:fake_panel_audit",
                "evaluation:fake_pareto",
            ]
        )
    return _dedupe_refs(refs)


def _cache_strategy_for_runtime_provider(
    runtime_ref: str,
    model_provider_ref: str,
) -> tuple[str, str | None, str | None]:
    if (
        runtime_ref == "agent_runtime:fake_runtime"
        or model_provider_ref == "model_provider:fake_provider"
    ):
        return "disabled", None, None
    if runtime_ref == "agent_runtime:claude_sdk":
        return "runtime_auto_cache", "runtime_auto_cache", None
    if model_provider_ref == "model_provider:anthropic_messages":
        return "provider_explicit_cache", None, "anthropic_messages_cache_control"
    return "provider_default", None, None


def _validate_runtime_provider_compatibility(
    runtime_ref: str,
    model_provider_ref: str,
    registry: Any,
) -> None:
    contract = _manifest_contract(runtime_ref, "runtime", registry)
    compatible = [str(item) for item in contract.get("compatible_model_providers") or [] if item]
    if compatible and model_provider_ref not in compatible:
        raise ValueError(
            f"{runtime_ref} is not compatible with {model_provider_ref}; "
            f"compatible providers: {', '.join(sorted(compatible))}"
        )


def _manifest_contract(ref: str, key: str, registry: Any) -> dict[str, Any]:
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


def _touch_run_ledgers(run_dir: Path) -> None:
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


def _plugin_roots_payload(plugin_roots: PluginRoots) -> dict[str, list[str]]:
    return {
        "bundled": [str(path) for path in plugin_roots.bundled],
        "project": [str(path) for path in plugin_roots.project],
        "user": [str(path) for path in plugin_roots.user],
    }


def _fixture_plugin_roots(workspace: Path) -> PluginRoots:
    roots = PluginRoots.defaults(workspace)
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = next(
        (
            candidate
            for candidate in (
                Path(__file__).resolve().parent / "fixtures" / "plugins",
                repo_root / "tests" / "fixtures" / "plugins",
            )
            if candidate.is_dir()
        ),
        None,
    )
    if fixture_root is None:
        return roots
    if fixture_root.exists() and fixture_root not in roots.bundled:
        return PluginRoots(
            bundled=[*roots.bundled, fixture_root],
            project=roots.project,
            user=roots.user,
            task_project=roots.task_project,
        )
    return roots


def _credential_snapshot(
    resolver: CredentialResolver,
    credential_set: CredentialSet,
    credential_manager: CredentialFailoverManager,
) -> dict[str, Any]:
    snapshot = resolver.snapshot(credential_set)
    snapshot["failover"] = credential_manager.snapshot()
    return snapshot


def _effective_task_spec(
    task_ref: str,
    runtime_ref: str,
    model_provider_ref: str,
    budget_policy_ref: str,
    task_project: TaskProject | None,
) -> str:
    if task_project is not None:
        return task_project.descriptor_path.read_text(encoding="utf-8")
    return f"""task: {task_ref}
workflow:
  stages:
    research_loop:
      enabled: true
    ideation:
      enabled: false
    paper_writing:
      enabled: false
    reviewer:
      enabled: false
panel:
  topology: panel_topology:fake_two_round
  roles:
    - role:fake_peer
    - role:fake_pi
    - role:fake_chair
  optional_roles:
    literature_scout:
      role: task_role:literature_scout
      tool_server_ref: tool_server:literature_lookup
      enabled: false
runtime:
  agent_runtime: {runtime_ref}
model_profiles:
  cheap_peer:
    provider_ref: {model_provider_ref}
    model: fake-deterministic
budget:
  policy: {budget_policy_ref}
"""


def _dedupe_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
    return out


def _budget(
    run_dir: Path,
    trajectory: TrajectoryWriter,
    run_id: str,
    parent_event_id: str,
    budget_policy_ref: str,
    registry: Any,
    task_ref: str,
) -> str:
    budget_request = BudgetRequest(
        request_id="budget_request_001",
        requester_id="workflow_stage:research_loop",
        experiment_id="fake_panel_stage_start",
        model_profile_ref="cheap_peer",
        requested={"tokens": 1000, "wall_clock_seconds": 30},
        expected_value={"confidence": "strong", "value": "gate_b_smoke"},
        evidence_refs=[task_ref],
        cheaper_alternatives=[],
        abort_conditions=["stage_timeout"],
    )
    decision = policy_for_ref(budget_policy_ref, registry=registry).decide(budget_request)
    grant_id = decision.grant.grant_id if decision.grant else ""
    request_record = {
        "schema_version": "praxist.budget_ledger.v1",
        "record_id": "budget_request_001",
        "run_id": run_id,
        "timestamp": utc_now(),
        "kind": "request",
        "request_id": "budget_request_001",
        "grant_id": None,
        "actor_ref": "workflow_stage:research_loop",
        "stage_id": "research_loop",
        "action_type": "stage_start",
        "request_record": {
            "request_id": budget_request.request_id,
            "requester_id": budget_request.requester_id,
            "experiment_id": budget_request.experiment_id,
            "model_profile_ref": budget_request.model_profile_ref,
            "requested": dict(budget_request.requested),
            "expected_value": dict(budget_request.expected_value),
            "evidence_refs": list(budget_request.evidence_refs),
            "cheaper_alternatives": list(budget_request.cheaper_alternatives),
            "abort_conditions": list(budget_request.abort_conditions),
        },
        "requested_budget": budget_request.requested,
        "decision": None,
        "granted_budget": None,
        "actual_usage": None,
        "reason": "fake_panel_budget_request",
        "source_event_ids": [parent_event_id],
        "artifact_refs": [],
    }
    append_jsonl(run_dir / "budget_ledger.jsonl", request_record)
    trajectory.emit(
        "budget.requested",
        scope={"stage_id": "research_loop"},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={
            "request_id": "budget_request_001",
            "requested": request_record["requested_budget"],
        },
    )
    grant_record = dict(request_record)
    grant_record.update(
        {
            "record_id": "budget_grant_001",
            "kind": "decision",
            "grant_id": grant_id,
            "decision": decision.decision,
            "granted_budget": decision.grant.approved if decision.grant else None,
            "reason": ",".join(decision.reason_codes),
            "decision_record": decision.to_dict(),
        }
    )
    append_jsonl(run_dir / "budget_ledger.jsonl", grant_record)
    trajectory.emit(
        "budget.granted" if decision.grant else "budget.review_required",
        scope={"stage_id": "research_loop", "grant_id": grant_id},
        actor={"type": "budget_policy", "id": budget_policy_ref},
        payload={
            "request_id": "budget_request_001",
            "grant_id": grant_id,
            "decision": decision.to_dict(),
        },
    )
    return grant_id


def _peer_finding(
    run_dir: Path,
    trajectory: TrajectoryWriter,
    artifacts: ArtifactWriter,
    run_id: str,
    peer_index: int,
    *,
    runtime_ref: str,
    model_provider_ref: str,
    credential_set: CredentialSet,
    credential_manager: CredentialFailoverManager,
    cache_policy: Any,
    budget_grant_id: str,
    registry: Any,
    task_ref: str,
) -> tuple[str, str]:
    agent_run_id = f"fake_peer_{peer_index}"
    provider_name = model_provider_ref.split(":", 1)[1]
    credential = credential_manager.select(
        scope="model_provider", provider=provider_name, target_ref=model_provider_ref
    )
    if (
        peer_index == 0
        and credential is not None
        and credential_set.mode == "robust"
        and credential.key_id.endswith(":A")
    ):
        failed_credential = credential
        next_credential = credential_manager.record_failure(failed_credential, "quota_exhausted")
        if next_credential is not None:
            trajectory.emit(
                "credential.failover",
                scope={"stage_id": "research_loop", "agent_run_id": agent_run_id},
                actor={"type": "core", "id": "credential_resolver"},
                severity="warning",
                payload={
                    "from_key_id": failed_credential.key_id,
                    "to_key_id": next_credential.key_id,
                    "reason": "quota_exhausted",
                    "credential_mode": credential_set.mode,
                },
            )
            credential = next_credential
    profile = default_model_profile(model_provider_ref, registry=registry)
    model_call = provider_for_ref(model_provider_ref, registry=registry).build_call(
        profile,
        credential_ref=credential,
        runtime_options={"script_id": "fake_peer_success_v1", "peer_index": peer_index},
    )
    request = AgentRunRequest(
        request_id=agent_run_id,
        run_id=run_id,
        stage_id="research_loop",
        role_ref="role:fake_peer",
        agent_runtime_ref=runtime_ref,
        prompt_ref={
            "artifact_id": f"prompt_{agent_run_id}",
            "logical_path": f"prompts/{agent_run_id}.md",
        },
        system_prompt_ref={
            "artifact_id": "system_prompt_fake_panel",
            "logical_path": "prompts/system.md",
        },
        cwd=str(run_dir),
        model_profile_ref=profile.profile_id,
        model_call=model_call,
        tool_permissions=ToolPermissionSet(),
        tool_servers=[],
        env_policy=EnvPolicy(scoped_credential_refs=[credential] if credential else []),
        credential_ref=credential,
        credential_mode=credential_set.mode,
        budget_grant_id=budget_grant_id,
        artifact_scope="run",
        timeout_seconds=30,
        cache_policy=cache_policy,
        runtime_options={"script_id": "fake_peer_success_v1", "peer_index": peer_index},
    )
    runtime_result = runtime_for_ref(runtime_ref, registry=registry).execute_sync(request)
    trajectory.emit(
        "agent.run_started",
        scope={
            "stage_id": "research_loop",
            "role_ref": "role:fake_peer",
            "agent_run_id": agent_run_id,
        },
        actor={"type": "agent_runtime", "id": runtime_ref},
        payload={"peer_index": peer_index, "request": request.to_dict()},
    )
    trajectory.emit(
        "model.call_started",
        scope={"stage_id": "research_loop", "agent_run_id": agent_run_id},
        actor={"type": "model_provider", "id": model_provider_ref},
        payload={"model_profile_ref": "cheap_peer", "model_call": model_call.to_dict()},
    )
    model_result = provider_for_ref(model_provider_ref, registry=registry).normalize_result(
        {
            "model": profile.model,
            "text": f"deterministic fake finding draft {peer_index}",
            "usage": {"tokens": 100 + peer_index},
        }
    )
    trajectory.emit(
        "model.call_finished",
        scope={"stage_id": "research_loop", "agent_run_id": agent_run_id},
        actor={"type": "model_provider", "id": model_provider_ref},
        payload=model_result.to_dict(),
    )
    for runtime_event in runtime_result.events:
        if runtime_event.type == "assistant_text":
            trajectory.emit(
                "agent.event",
                scope={"stage_id": "research_loop", "agent_run_id": agent_run_id},
                actor={"type": "agent_runtime", "id": runtime_ref},
                payload=runtime_event.to_dict(),
            )
    model_artifact = artifacts.persist_json(
        "model_io",
        f"model_io/fake_peer_{peer_index}.json",
        {
            "text": model_result.text,
            "usage": model_result.usage,
            "runtime_ref": runtime_ref,
            "model_call": model_call.to_dict(),
            "cache_policy": cache_policy.to_dict(),
        },
        schema_ref="core:model_io.v1",
        producer={"stage_id": "research_loop", "agent_run_id": agent_run_id},
    )
    finding_id = f"finding_fake_peer_{peer_index}"
    finding_artifact = artifacts.persist_json(
        "finding",
        f"findings/{finding_id}.json",
        {
            "finding_id": finding_id,
            "claim": f"deterministic fake finding {peer_index}",
            "score": 1.0 - peer_index * 0.1,
            "source_model_io": model_artifact["artifact_id"],
        },
        schema_ref="core:finding.v1",
        producer={"stage_id": "research_loop", "agent_run_id": agent_run_id},
        source_artifact_ids=[model_artifact["artifact_id"]],
    )
    agent_finished = trajectory.emit(
        "agent.run_finished",
        scope={"stage_id": "research_loop", "agent_run_id": agent_run_id},
        actor={"type": "agent_runtime", "id": runtime_ref},
        payload={
            "success": runtime_result.success,
            "finding_id": finding_id,
            "failover_reason": runtime_result.failover_reason,
            "agent_runtime_ref": runtime_ref,
            "model_call": model_call.to_dict(),
            "budget_grant_id": budget_grant_id,
        },
    )
    record = {
        "schema_version": "praxist.finding.v1",
        "finding_id": finding_id,
        "run_id": run_id,
        "status": "draft",
        "claim": f"deterministic fake finding {peer_index}",
        "task_ref": task_ref,
        "stage_id": "research_loop",
        "producer_ref": f"role:fake_peer/{agent_run_id}",
        "evidence_refs": [model_artifact],
        "metric_refs": [],
        "scores": {"deterministic_score": 1.0 - peer_index * 0.1},
        "supersedes": [],
        "source_event_ids": [agent_finished["event_id"]],
        "created_at": utc_now(),
    }
    append_jsonl(run_dir / "findings" / "findings.jsonl", record)
    event = trajectory.emit(
        "finding.created",
        scope={"stage_id": "research_loop", "agent_run_id": agent_run_id},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={"finding_id": finding_id, "status": "draft"},
        artifact_refs=[finding_artifact],
    )
    return finding_id, event["event_id"]


def _retry_finding(
    run_dir: Path,
    trajectory: TrajectoryWriter,
    artifacts: ArtifactWriter,
    run_id: str,
    source_finding_id: str,
    *,
    runtime_ref: str,
    model_provider_ref: str,
    credential_manager: CredentialFailoverManager,
    cache_policy: Any,
    budget_grant_id: str,
    registry: Any,
    task_ref: str,
) -> tuple[str, str]:
    finding_id = "finding_fake_retry_001"
    provider_name = model_provider_ref.split(":", 1)[1]
    credential = credential_manager.select(
        scope="model_provider", provider=provider_name, target_ref=model_provider_ref
    )
    profile = default_model_profile(model_provider_ref, registry=registry)
    model_call = provider_for_ref(model_provider_ref, registry=registry).build_call(
        profile,
        credential_ref=credential,
        runtime_options={"script_id": "fake_retry_success_v1"},
    )
    agent_finished = trajectory.emit(
        "agent.run_finished",
        scope={"stage_id": "research_loop", "agent_run_id": "fake_retry_peer"},
        actor={"type": "agent_runtime", "id": runtime_ref},
        payload={
            "success": True,
            "finding_id": finding_id,
            "retry_of": source_finding_id,
            "agent_runtime_ref": runtime_ref,
            "model_call": model_call.to_dict(),
            "budget_grant_id": budget_grant_id,
        },
    )
    artifact = artifacts.persist_json(
        "finding",
        "findings/finding_fake_retry_001.json",
        {
            "finding_id": finding_id,
            "claim": "corrected deterministic fake finding after retry",
            "score": 0.95,
            "retry_of": source_finding_id,
        },
        schema_ref="core:finding.v1",
        producer={"stage_id": "research_loop", "agent_run_id": "fake_retry_peer"},
    )
    append_jsonl(
        run_dir / "findings" / "findings.jsonl",
        {
            "schema_version": "praxist.finding.v1",
            "finding_id": finding_id,
            "run_id": run_id,
            "status": "candidate",
            "claim": "corrected deterministic fake finding after retry",
            "task_ref": task_ref,
            "stage_id": "research_loop",
            "producer_ref": "role:fake_peer/fake_retry_peer",
            "evidence_refs": [artifact],
            "metric_refs": [],
            "scores": {"deterministic_score": 0.95},
            "supersedes": [source_finding_id],
            "source_event_ids": [agent_finished["event_id"]],
            "created_at": utc_now(),
        },
    )
    event = trajectory.emit(
        "finding.created",
        scope={"stage_id": "research_loop", "agent_run_id": "fake_retry_peer"},
        actor={"type": "workflow_stage", "id": "research_loop"},
        payload={"finding_id": finding_id, "status": "candidate", "retry_of": source_finding_id},
        artifact_refs=[artifact],
    )
    return finding_id, event["event_id"]


def _audit(
    trajectory: TrajectoryWriter,
    audit_index: int,
    finding_id: str,
    status: str,
    severity: str,
    blocking: bool,
    *,
    retry: bool = False,
) -> None:
    trajectory.emit(
        "audit.verdict_recorded",
        scope={"stage_id": "research_loop", "finding_id": finding_id},
        actor={"type": "plugin", "id": "role:fake_pi"},
        severity="warning" if severity in {"warning", "blocking"} else "info",
        payload={
            "audit_id": f"fake_audit_{audit_index}",
            "status": status,
            "severity": severity,
            "blocking": blocking,
            "retry_requested": retry,
        },
    )
    if retry:
        trajectory.emit(
            "audit.retry_requested",
            scope={"stage_id": "research_loop", "finding_id": finding_id},
            actor={"type": "plugin", "id": "role:fake_pi"},
            severity="warning",
            payload={"reason": "fake_retry_path", "finding_id": finding_id},
        )


class FakeWorkflowFixtureTaskRunner:
    """Task runner class that exposes the fake workflow fixture to task projects."""

    def __init__(self, task_project: TaskProject | None = None) -> None:
        self.task_project = task_project
        self.task_ref = task_project.task_ref if task_project is not None else "task:fake_panel"

    def run(
        self,
        *,
        run_lifecycle_observer=None,
        **kwargs,
    ):
        if self.task_project is not None:
            kwargs.setdefault("task_project", self.task_project)

        kwargs["run_lifecycle_observer"] = run_lifecycle_observer
        return run_fake_workflow_fixture(**kwargs)


def create_task_runner(task_project: TaskProject | None = None) -> FakeWorkflowFixtureTaskRunner:
    """Task project entrypoint that constructs the fake workflow fixture runner."""
    return FakeWorkflowFixtureTaskRunner(task_project)
