"""Inspect / verify / dry-run support for run directories.

Replay is an internal explainability and consistency mechanism for fast Praxist
runs. The default verifier should help humans and downstream modules understand
what happened after the fact; locked mode is reserved for benchmark/release
artifacts that need strict drift failures.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from praxist.core.budget import ALLOWED_BUDGET_UNITS, policy_for_ref
from praxist.core.controller_state import read_private_startup_config
from praxist.core.prompt_layout import sha256_json, sha256_text
from praxist.core.protocol import BudgetRequest
from praxist.core.redaction import scan_file
from praxist.core.registry import (
    assert_bundled_execution_manifest,
    compute_plugin_content_hash,
    read_plugin_metadata,
    selected_plugin_from_dict,
)
from praxist.core.source_snapshot import build_core_source_snapshot
from praxist.core.storage import (
    OUTPUT_LEDGER_RELS,
    output_ledger_hashes,
    read_jsonl,
    sha256_bytes,
    utc_now,
    write_json,
)

REQUIRED_TOP_LEVEL = (
    "run.json",
    "run_summary.json",
    "startup_config.json",
    "effective_task_spec.yaml",
    "plugin_resolution.json",
    "model_profiles.json",
    "credentials_redacted.json",
    "cache_policy.json",
    "trajectory.jsonl",
    "budget_ledger.jsonl",
    "artifact_index.jsonl",
)

SCAN_FILES = (
    "run.json",
    "startup_config.json",
    "effective_task_spec.yaml",
    "model_profiles.json",
    "run_summary.json",
    "plugin_resolution.json",
    "credentials_redacted.json",
    "cache_policy.json",
    "trajectory.jsonl",
    "budget_ledger.jsonl",
    "artifact_index.jsonl",
    "findings/findings.jsonl",
    "findings/frontier.jsonl",
    "memory/research_memory.jsonl",
    "memory/graph_edges.jsonl",
)

TEXT_SCAN_SUFFIXES = {
    ".csv",
    ".err",
    ".html",
    ".jinja2",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".out",
    ".py",
    ".sh",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def inspect_run(run_dir: Path) -> dict[str, Any]:
    """Inspect a run directory and return counts, artifacts, and replay-facing diagnostics."""
    trajectory, trajectory_errors = read_jsonl(run_dir / "trajectory.jsonl")
    findings, finding_errors = read_jsonl(run_dir / "findings" / "findings.jsonl")
    frontier, frontier_errors = read_jsonl(run_dir / "findings" / "frontier.jsonl")
    research_memory, memory_errors = read_jsonl(run_dir / "memory" / "research_memory.jsonl")
    graph_edges, graph_edge_errors = read_jsonl(run_dir / "memory" / "graph_edges.jsonl")
    stages: dict[str, str] = {}
    event_counts: dict[str, int] = {}
    for event in trajectory:
        kind = str(event.get("kind", "unknown"))
        event_counts[kind] = event_counts.get(kind, 0) + 1
        if kind.startswith("workflow.stage_"):
            stage_id = str((event.get("scope") or {}).get("stage_id", "unknown"))
            stages[stage_id] = kind.rsplit("_", 1)[-1]
    return {
        "run_dir": str(run_dir),
        "events": len(trajectory),
        "event_counts": event_counts,
        "stages": stages,
        "findings": len(findings),
        "frontier_records": len(frontier),
        "research_memory_records": len(research_memory),
        "graph_edges": len(graph_edges),
        "errors": trajectory_errors
        + finding_errors
        + frontier_errors
        + memory_errors
        + graph_edge_errors,
    }


def verify_run(
    run_dir: Path,
    *,
    strict_tail: bool = False,
    allow_plugin_drift: bool = False,
    locked: bool = False,
) -> dict[str, Any]:
    """Verify replay invariants for run lifecycle, redaction, plugin provenance, usage, and materialized state."""
    errors: list[str] = []
    warnings: list[str] = []
    run_dir = Path(run_dir)

    for rel in REQUIRED_TOP_LEVEL:
        if not (run_dir / rel).exists():
            errors.append(f"missing required file: {rel}")

    run_json = _read_json(run_dir / "run.json", errors)
    run_summary = _read_json(run_dir / "run_summary.json", errors)
    startup_config = _read_json(run_dir / "startup_config.json", errors)
    try:
        private_startup = read_private_startup_config(run_dir)
    except ValueError as exc:
        errors.append(f"private startup authority is unavailable for budget replay: {exc}")
        private_startup = {}
    if private_startup is not None:
        startup_config = private_startup
    plugin_resolution = _read_json(run_dir / "plugin_resolution.json", errors)
    model_profiles = _read_json(run_dir / "model_profiles.json", errors)
    credentials_redacted = _read_json(run_dir / "credentials_redacted.json", errors)
    cache_policy = _read_json(run_dir / "cache_policy.json", errors)
    expected_run_id = run_json.get("run_id") if run_json else None
    if run_json and not isinstance(expected_run_id, str):
        errors.append("run.json missing required run_id")

    trajectory, trajectory_errors = read_jsonl(run_dir / "trajectory.jsonl")
    if trajectory_errors:
        errors.extend(trajectory_errors)
    _verify_trajectory(trajectory, errors)
    if trajectory and trajectory[0].get("kind") != "run.started":
        errors.append("trajectory first event is not run.started")
    started_count = sum(1 for event in trajectory if event.get("kind") == "run.started")
    finalized_count = sum(1 for event in trajectory if event.get("kind") == "run.finalized")
    if started_count != 1:
        errors.append(f"trajectory expected exactly one run.started event, found {started_count}")
    if finalized_count == 0:
        errors.append("trajectory has no run.finalized event")
    elif finalized_count != 1:
        errors.append(
            f"trajectory expected exactly one run.finalized event, found {finalized_count}"
        )
    elif trajectory and trajectory[-1].get("kind") != "run.finalized":
        errors.append("trajectory has events after run.finalized")

    artifact_index, artifact_errors = read_jsonl(run_dir / "artifact_index.jsonl")
    if artifact_errors:
        errors.extend(artifact_errors)
    budget_ledger, budget_errors = read_jsonl(run_dir / "budget_ledger.jsonl")
    if budget_errors:
        errors.extend(budget_errors)
    authorization = (startup_config or {}).get("budget_authorization")
    _verify_budget_ledger(
        budget_ledger,
        _budget_policy_ref(startup_config),
        errors,
        warnings,
        allow_uncapped=isinstance(authorization, dict) and authorization.get("uncapped") is True,
    )
    findings, finding_errors = read_jsonl(run_dir / "findings" / "findings.jsonl")
    frontier, frontier_errors = read_jsonl(run_dir / "findings" / "frontier.jsonl")
    research_memory, memory_errors = read_jsonl(run_dir / "memory" / "research_memory.jsonl")
    graph_edges, graph_edge_errors = read_jsonl(run_dir / "memory" / "graph_edges.jsonl")
    errors.extend(finding_errors + frontier_errors + memory_errors + graph_edge_errors)
    if isinstance(expected_run_id, str):
        _verify_record_run_ids(trajectory, expected_run_id, "trajectory", errors)
        _verify_record_run_ids(artifact_index, expected_run_id, "artifact_index", errors)
        _verify_record_run_ids(budget_ledger, expected_run_id, "budget_ledger", errors)
        _verify_record_run_ids(findings, expected_run_id, "findings", errors)
        _verify_record_run_ids(frontier, expected_run_id, "frontier", errors)
        _verify_record_run_ids(research_memory, expected_run_id, "research_memory", errors)
        _verify_record_run_ids(graph_edges, expected_run_id, "graph_edges", errors)
    artifact_by_id: dict[str, dict[str, Any]] = {}
    indexed_artifact_paths: set[Path] = set()
    for artifact in artifact_index:
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            errors.append("artifact record missing artifact_id")
        elif artifact_id in artifact_by_id:
            errors.append(f"artifact duplicate artifact_id: {artifact_id}")
        else:
            artifact_by_id[artifact_id] = artifact
        payload_rel = artifact.get("payload_path")
        expected_hash = artifact.get("content_hash")
        if not isinstance(payload_rel, str):
            errors.append(f"artifact {artifact.get('artifact_id')} missing payload_path")
            continue
        if not _is_sha256_digest(expected_hash):
            errors.append(
                f"artifact {artifact.get('artifact_id')} has invalid content_hash: {expected_hash}"
            )
            continue
        payload_path = _resolve_run_relative_path(
            run_dir,
            payload_rel,
            errors,
            f"artifact {artifact.get('artifact_id')} payload_path",
        )
        if payload_path is None:
            continue
        if not payload_path.exists():
            errors.append(f"artifact payload missing: {payload_rel}")
            continue
        actual_hash = sha256_bytes(payload_path.read_bytes())
        if actual_hash != expected_hash:
            errors.append(f"artifact hash mismatch: {payload_rel}")
        indexed_artifact_paths.add(payload_path.resolve())

    for rel in SCAN_FILES:
        path = run_dir / rel
        if path.exists():
            hits = scan_file(path)
            if hits:
                errors.append(f"redaction scan hit in {rel}: {','.join(sorted(set(hits)))}")
    _scan_tree(run_dir / "logs", run_dir, errors)
    _scan_run_text_tree(run_dir, errors)
    _verify_artifact_tree(
        run_dir, artifact_by_id, indexed_artifact_paths, errors, warnings, locked=locked
    )
    for artifact in artifact_index:
        payload_rel = artifact.get("payload_path")
        if isinstance(payload_rel, str):
            path = _resolve_run_relative_path(
                run_dir,
                payload_rel,
                errors,
                f"artifact {artifact.get('artifact_id')} payload_path",
            )
            if path is not None and path.exists():
                hits = scan_file(path)
                if hits:
                    errors.append(
                        f"redaction scan hit in {payload_rel}: {','.join(sorted(set(hits)))}"
                    )
    known_event_ids = {str(event.get("event_id")) for event in trajectory if event.get("event_id")}
    _verify_source_event_ids(artifact_index, known_event_ids, "artifact_index", errors)
    _verify_source_event_ids(budget_ledger, known_event_ids, "budget_ledger", errors)
    _verify_source_event_ids(findings, known_event_ids, "findings", errors)
    _verify_source_event_ids(frontier, known_event_ids, "frontier", errors)
    _verify_source_event_ids(research_memory, known_event_ids, "research_memory", errors)
    _verify_source_event_ids(graph_edges, known_event_ids, "graph_edges", errors)
    _verify_output_agent_provenance(trajectory, findings, frontier, errors, warnings)
    _verify_artifact_references(artifact_index, artifact_by_id, "artifact_index", errors)
    _verify_artifact_references(trajectory, artifact_by_id, "trajectory", errors)
    _verify_artifact_references(budget_ledger, artifact_by_id, "budget_ledger", errors)
    _verify_artifact_references(findings, artifact_by_id, "findings", errors)
    _verify_artifact_references(frontier, artifact_by_id, "frontier", errors)
    _verify_artifact_references(research_memory, artifact_by_id, "research_memory", errors)
    _verify_artifact_references(graph_edges, artifact_by_id, "graph_edges", errors)
    _verify_prompt_layout_artifacts(run_dir, artifact_index, artifact_by_id, errors, warnings)
    _verify_required_budget_usage(
        budget_ledger, run_summary, trajectory, findings, frontier, errors, warnings
    )
    _verify_run_summary_consistency(run_json, run_summary, trajectory, errors)
    _verify_frontier_finding_refs(frontier, findings, errors)
    _verify_output_counts(
        run_summary, trajectory, findings, frontier, research_memory, graph_edges, errors
    )
    _verify_output_hashes(
        run_dir, run_summary, trajectory, findings, frontier, research_memory, graph_edges, errors
    )
    state_recovery = _verify_state_surface_recovery(
        run_dir,
        findings,
        frontier,
        research_memory,
        graph_edges,
        artifact_index,
        errors,
        warnings,
    )

    if plugin_resolution:
        if plugin_resolution.get("algorithm_version") != 2:
            errors.append("plugin_resolution algorithm_version must be 2")
        if isinstance(expected_run_id, str) and plugin_resolution.get("run_id") != expected_run_id:
            errors.append(
                f"plugin_resolution run_id mismatch: expected {expected_run_id}, got {plugin_resolution.get('run_id')}"
            )
        selected_plugins = plugin_resolution.get("selected")
        if not isinstance(selected_plugins, list) or not selected_plugins:
            errors.append("plugin_resolution selected must be a non-empty list")
            selected_plugins = []
        selected_refs = {
            item.get("metadata", {}).get("kind") + ":" + item.get("metadata", {}).get("name")
            for item in selected_plugins
            if isinstance(item, dict)
            and isinstance(item.get("metadata"), dict)
            and isinstance(item["metadata"].get("kind"), str)
            and isinstance(item["metadata"].get("name"), str)
        }
        try:
            assert_bundled_execution_manifest(plugin_resolution)
        except ValueError as exc:
            errors.append(str(exc))
        _verify_plugin_dependency_closure(
            selected_plugins, plugin_resolution.get("dependency_edges"), selected_refs, errors
        )
        _verify_required_plugin_refs(selected_refs, run_json, startup_config, trajectory, errors)
        _verify_runtime_provider_bindings(
            selected_refs,
            startup_config,
            model_profiles,
            credentials_redacted,
            trajectory,
            errors,
            warnings,
            locked=locked,
        )
        _verify_runtime_provider_conformance(
            selected_plugins,
            startup_config,
            model_profiles,
            cache_policy,
            errors,
        )
        for selected in selected_plugins:
            if not isinstance(selected, dict):
                errors.append("plugin_resolution selected item is not an object")
                continue
            path = str(selected.get("path", ""))
            content_hash = str(selected.get("content_hash", ""))
            drift_is_error = locked and not allow_plugin_drift
            if path.startswith("builtin://"):
                _record_lockable_issue(
                    errors,
                    warnings,
                    f"selected plugin path is not replay-verifiable: {path}",
                    locked=drift_is_error,
                )
                continue
            plugin_path = Path(path)
            if not plugin_path.exists():
                _record_lockable_issue(
                    errors,
                    warnings,
                    f"selected plugin path missing: {path}",
                    locked=drift_is_error,
                )
                continue
            if not _is_sha256_digest(content_hash):
                errors.append(f"selected plugin {path} has invalid content_hash: {content_hash}")
                continue
            try:
                selected_plugin = selected_plugin_from_dict(selected)
                disk_metadata = read_plugin_metadata(plugin_path)
            except Exception as exc:  # noqa: BLE001 - replay must report all manifest issues.
                _record_lockable_issue(
                    errors,
                    warnings,
                    f"selected plugin metadata unreadable: {path}: {exc}",
                    locked=drift_is_error,
                )
                continue
            if selected_plugin.metadata.to_dict() != disk_metadata.to_dict():
                _record_lockable_issue(
                    errors,
                    warnings,
                    f"plugin metadata drift: {path}",
                    locked=drift_is_error,
                )
            actual_hash = compute_plugin_content_hash(plugin_path, disk_metadata)
            if actual_hash != content_hash:
                _record_lockable_issue(
                    errors,
                    warnings,
                    f"plugin hash drift: {path} expected {content_hash} got {actual_hash}",
                    locked=drift_is_error,
                )

    if run_json:
        expected_core_hash = run_json.get("workspace_hash")
        if not isinstance(expected_core_hash, str) or not expected_core_hash:
            _record_lockable_issue(
                errors,
                warnings,
                "run.json missing required workspace_hash",
                locked=locked,
            )
        else:
            current_core_hash = build_core_source_snapshot().get("workspace_hash")
            if current_core_hash != expected_core_hash:
                _record_lockable_issue(
                    errors,
                    warnings,
                    f"core source hash drift: expected {expected_core_hash} got {current_core_hash}",
                    locked=locked,
                )
        if run_json.get("source_hash_algorithm") != "sha256":
            _record_lockable_issue(
                errors,
                warnings,
                "run.json missing required source_hash_algorithm=sha256",
                locked=locked,
            )
        if (
            not isinstance(run_json.get("source_file_count"), int)
            or run_json.get("source_file_count") <= 0
        ):
            _record_lockable_issue(
                errors,
                warnings,
                "run.json missing required positive source_file_count",
                locked=locked,
            )
        if "git_commit" not in run_json:
            _record_lockable_issue(
                errors,
                warnings,
                "run.json missing required git_commit field",
                locked=locked,
            )

    report = {
        "schema_version": "praxist.replay_report.v1",
        "mode": "verify",
        "verification_level": "locked" if locked else "internal",
        "locked": locked,
        "allow_plugin_drift": allow_plugin_drift,
        "strict_tail": strict_tail,
        "run_dir": str(run_dir),
        "verified_at": utc_now(),
        "success": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": inspect_run(run_dir),
        "state_recovery": state_recovery,
    }
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    write_json(replay_dir / "state_recovery_report.json", state_recovery)
    write_json(replay_dir / "replay_report.json", report)
    return report


def _verify_trajectory(trajectory: list[dict[str, Any]], errors: list[str]) -> None:
    seen_event_ids: set[str] = set()
    previous_event_ids: set[str] = set()
    for index, event in enumerate(trajectory, start=1):
        seq = event.get("seq")
        if seq != index:
            errors.append(f"trajectory seq mismatch at record {index}: expected {index}, got {seq}")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            errors.append(f"trajectory record {index} missing event_id")
        elif event_id in seen_event_ids:
            errors.append(f"trajectory duplicate event_id: {event_id}")
        else:
            seen_event_ids.add(event_id)
            expected_event_id = f"evt_{index:06d}"
            if event_id != expected_event_id:
                errors.append(
                    f"trajectory event_id mismatch at record {index}: expected {expected_event_id}, got {event_id}"
                )
        parent_event_ids = event.get("parent_event_ids") or []
        if not isinstance(parent_event_ids, list):
            errors.append(f"trajectory {event_id or index} parent_event_ids is not a list")
        else:
            for parent_id in parent_event_ids:
                if parent_id not in previous_event_ids:
                    errors.append(
                        f"trajectory {event_id or index} references unknown parent_event_id: {parent_id}"
                    )
        if isinstance(event_id, str):
            previous_event_ids.add(event_id)


def _verify_budget_ledger(
    records: list[dict[str, Any]],
    budget_policy_ref: str | None,
    errors: list[str],
    warnings: list[str],
    *,
    allow_uncapped: bool = False,
) -> None:
    grants: dict[str, dict[str, Any]] = {}
    usage_totals: dict[str, dict[str, float]] = {}
    seen_usage_grants: set[str] = set()
    unknown_units_by_grant: dict[str, set[str]] = {}
    requests: dict[str, BudgetRequest] = {}
    for record in records:
        kind = record.get("kind")
        grant_id = record.get("grant_id")
        if kind in {"request", "decision"}:
            requested = record.get("requested_budget") or {}
            if not isinstance(requested, dict):
                errors.append(
                    f"budget {kind} {record.get('record_id')} has invalid requested_budget"
                )
            else:
                _verify_budget_amounts(
                    requested, f"{kind} {record.get('record_id')} requested_budget", errors
                )
            request = _budget_request_from_record(record, errors)
            if request is not None:
                if dict(request.requested) != requested:
                    errors.append(
                        f"budget {kind} {record.get('record_id')} requested_budget does not match request_record"
                    )
                if kind == "request":
                    if request.request_id in requests:
                        errors.append(f"budget duplicate request_id: {request.request_id}")
                    requests[request.request_id] = request
                else:
                    prior_request = requests.get(request.request_id)
                    if prior_request is None:
                        errors.append(
                            f"budget decision {record.get('record_id')} has no prior request record: {request.request_id}"
                        )
                    elif prior_request != request:
                        errors.append(
                            f"budget decision {record.get('record_id')} request_record does not match prior request"
                        )
                    _verify_budget_decision_against_policy(
                        record, request, budget_policy_ref, errors, allow_uncapped=allow_uncapped
                    )
        if (
            kind == "decision"
            and isinstance(grant_id, str)
            and record.get("granted_budget") is not None
        ):
            if grant_id in grants:
                errors.append(f"budget duplicate grant_id: {grant_id}")
                continue
            approved = record.get("granted_budget") or {}
            if not isinstance(approved, dict):
                errors.append(f"budget grant {grant_id} has invalid granted_budget")
                continue
            _verify_budget_amounts(approved, f"grant {grant_id}", errors)
            grants[grant_id] = record
            usage_totals.setdefault(grant_id, {})
            continue
        if kind == "usage_unknown":
            grant_id = record.get("grant_id")
            if not isinstance(grant_id, str) or not grant_id:
                errors.append(f"budget usage_unknown {record.get('record_id')} missing grant_id")
                continue
            grant = grants.get(grant_id)
            if grant is None:
                errors.append(
                    f"budget usage_unknown {record.get('record_id')} references unknown grant_id: {grant_id}"
                )
                continue
            approved = grant.get("granted_budget") or {}
            if not isinstance(approved, dict):
                errors.append(
                    f"budget usage_unknown {record.get('record_id')} has invalid budget payload"
                )
                continue
            raw_units = record.get("unknown_units") or []
            if not isinstance(raw_units, list):
                errors.append(
                    f"budget usage_unknown {record.get('record_id')} unknown_units is not a list"
                )
                continue
            seen_usage_grants.add(grant_id)
            unknown_units = unknown_units_by_grant.setdefault(grant_id, set())
            for raw_unit in raw_units:
                unit = str(raw_unit)
                if unit not in approved:
                    errors.append(
                        f"budget usage_unknown {record.get('record_id')} uses unapproved budget unit: {unit}"
                    )
                    continue
                unknown_units.add(unit)
            if raw_units:
                warnings.append(
                    f"budget usage_unknown for grant {grant_id}: "
                    + ", ".join(sorted({str(unit) for unit in raw_units}))
                )
            continue
        if kind != "usage":
            continue
        grant_id = record.get("grant_id")
        if not isinstance(grant_id, str) or not grant_id:
            errors.append(f"budget usage {record.get('record_id')} missing grant_id")
            continue
        grant = grants.get(grant_id)
        if grant is None:
            errors.append(
                f"budget usage {record.get('record_id')} references unknown grant_id: {grant_id}"
            )
            continue
        approved = grant.get("granted_budget") or {}
        actual_usage = record.get("actual_usage") or {}
        if not isinstance(approved, dict) or not isinstance(actual_usage, dict):
            errors.append(f"budget usage {record.get('record_id')} has invalid budget payload")
            continue
        seen_usage_grants.add(grant_id)
        totals = usage_totals.setdefault(grant_id, {})
        for unit, raw_amount in actual_usage.items():
            try:
                amount = float(raw_amount)
            except (TypeError, ValueError):
                errors.append(
                    f"budget usage {record.get('record_id')} has non-numeric usage for {unit}"
                )
                continue
            if not math.isfinite(amount) or amount < 0:
                errors.append(
                    f"budget usage {record.get('record_id')} has invalid usage for {unit}: {raw_amount}"
                )
                continue
            if unit not in approved:
                errors.append(
                    f"budget usage {record.get('record_id')} uses unapproved budget unit: {unit}"
                )
                continue
            if approved[unit] is None:
                totals[unit] = totals.get(unit, 0.0) + amount
                continue
            try:
                approved_amount = float(approved[unit])
            except (TypeError, ValueError):
                errors.append(f"budget grant {grant_id} has non-numeric approval for {unit}")
                continue
            totals[unit] = totals.get(unit, 0.0) + amount
            if totals[unit] > approved_amount:
                warnings.append(
                    f"budget usage for grant {grant_id} exceeds approved {unit}: "
                    f"{totals[unit]} > {approved_amount}"
                )
    for grant_id in sorted(seen_usage_grants):
        grant = grants.get(grant_id) or {}
        approved = grant.get("granted_budget") or {}
        totals = usage_totals.get(grant_id) or {}
        unknown_units = unknown_units_by_grant.get(grant_id, set())
        if not isinstance(approved, dict):
            continue
        for unit, raw_approved in approved.items():
            if raw_approved is None:
                if unit not in totals and unit not in unknown_units:
                    warnings.append(
                        f"budget usage for grant {grant_id} missing uncapped unit and usage_unknown: {unit}"
                    )
                continue
            try:
                approved_amount = float(raw_approved)
            except (TypeError, ValueError):
                continue
            if approved_amount > 0 and unit not in totals and unit not in unknown_units:
                warnings.append(
                    f"budget usage for grant {grant_id} missing approved unit and usage_unknown: {unit}"
                )


def _scan_tree(root: Path, run_dir: Path, errors: list[str]) -> None:
    if not root.exists():
        return
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        hits = scan_file(path)
        if hits:
            rel = path.relative_to(run_dir)
            errors.append(f"redaction scan hit in {rel}: {','.join(sorted(set(hits)))}")


def _scan_run_text_tree(run_dir: Path, errors: list[str]) -> None:
    if not run_dir.exists():
        return
    scanned_explicit = {Path(rel) for rel in SCAN_FILES}
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        rel = path.relative_to(run_dir)
        if rel in scanned_explicit:
            continue
        if path.suffix and path.suffix.lower() not in TEXT_SCAN_SUFFIXES:
            continue
        hits = scan_file(path)
        if hits:
            errors.append(f"redaction scan hit in {rel}: {','.join(sorted(set(hits)))}")


def _verify_artifact_tree(
    run_dir: Path,
    artifact_by_id: dict[str, dict[str, Any]],
    indexed_payload_paths: set[Path],
    errors: list[str],
    warnings: list[str],
    *,
    locked: bool,
) -> None:
    artifact_root = run_dir / "artifacts" / "by_id"
    if not artifact_root.exists():
        if artifact_by_id:
            errors.append("artifact root missing: artifacts/by_id")
        return
    indexed_paths = set(indexed_payload_paths)
    for artifact_id, record in artifact_by_id.items():
        payload_rel = record.get("payload_path")
        if isinstance(payload_rel, str):
            payload_path = _resolve_run_relative_path(
                run_dir,
                payload_rel,
                errors,
                f"artifact {artifact_id} payload_path",
            )
            if payload_path is not None:
                expected_prefix = Path("artifacts") / "by_id" / artifact_id
                if Path(payload_rel).parts[:3] != expected_prefix.parts:
                    errors.append(
                        f"artifact {artifact_id} payload_path does not live under its artifact directory"
                    )
                indexed_paths.add(payload_path.resolve())
        metadata_path = (artifact_root / artifact_id / "metadata.json").resolve()
        indexed_paths.add(metadata_path)
        if not metadata_path.exists():
            errors.append(f"artifact metadata missing: artifacts/by_id/{artifact_id}/metadata.json")
            continue
        metadata = _read_json(metadata_path, errors)
        if metadata is not None and metadata != record:
            errors.append(f"artifact metadata mismatch: {artifact_id}")

    for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
        resolved = path.resolve()
        if resolved not in indexed_paths:
            rel = path.relative_to(run_dir)
            _record_lockable_issue(
                errors,
                warnings,
                f"unindexed artifact file: {rel}",
                locked=locked,
            )


def _record_lockable_issue(
    errors: list[str],
    warnings: list[str],
    message: str,
    *,
    locked: bool,
) -> None:
    if locked:
        errors.append(message)
    else:
        warnings.append(message)


def _verify_record_run_ids(
    records: list[dict[str, Any]],
    expected_run_id: str,
    label: str,
    errors: list[str],
) -> None:
    for index, record in enumerate(records, start=1):
        run_id = record.get("run_id")
        if run_id != expected_run_id:
            errors.append(
                f"{label}:{index} run_id mismatch: expected {expected_run_id}, got {run_id}"
            )


def _budget_policy_ref(startup_config: dict[str, Any] | None) -> str | None:
    canonical_args = (startup_config or {}).get("canonical_args") or {}
    if not isinstance(canonical_args, dict):
        return None
    value = canonical_args.get("budget_policy")
    return value if isinstance(value, str) and value else None


def _resolve_run_relative_path(
    run_dir: Path, payload_rel: str, errors: list[str], label: str
) -> Path | None:
    payload_path = Path(payload_rel)
    if payload_path.is_absolute() or ".." in payload_path.parts:
        errors.append(f"{label} must be a run-relative confined path: {payload_rel}")
        return None
    run_root = run_dir.resolve()
    resolved = (run_dir / payload_path).resolve()
    if not resolved.is_relative_to(run_root):
        errors.append(f"{label} escapes run_dir: {payload_rel}")
        return None
    return resolved


def _verify_required_plugin_refs(
    selected_refs: set[str],
    run_json: dict[str, Any] | None,
    startup_config: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    errors: list[str],
) -> None:
    required_refs = []
    if run_json:
        for key in ("task_ref", "workflow_ref"):
            value = run_json.get(key)
            if isinstance(value, str) and value:
                required_refs.append(value)
    canonical_args = (startup_config or {}).get("canonical_args") or {}
    if isinstance(canonical_args, dict):
        for key in ("task", "runtime", "model_provider", "budget_policy"):
            value = canonical_args.get(key)
            if isinstance(value, str) and value:
                required_refs.append(value)
    for event in trajectory:
        if event.get("kind") == "plugin.resolution_started":
            requested = (event.get("payload") or {}).get("requested") or []
            if isinstance(requested, list):
                required_refs.extend(str(ref) for ref in requested)
        if event.get("kind") == "plugins.resolved":
            selected = (event.get("payload") or {}).get("selected") or []
            if isinstance(selected, list):
                required_refs.extend(str(ref) for ref in selected)
    for ref in sorted(set(required_refs)):
        if ref.startswith("task:"):
            continue
        if ref not in selected_refs:
            errors.append(f"plugin_resolution missing required selected ref: {ref}")


def _verify_runtime_provider_bindings(
    selected_refs: set[str],
    startup_config: dict[str, Any] | None,
    model_profiles: dict[str, Any] | None,
    credentials_redacted: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    *,
    locked: bool,
) -> None:
    canonical_args = (startup_config or {}).get("canonical_args") or {}
    runtime_ref = canonical_args.get("runtime") if isinstance(canonical_args, dict) else None
    provider_ref = (
        canonical_args.get("model_provider") if isinstance(canonical_args, dict) else None
    )
    if not isinstance(runtime_ref, str) or not runtime_ref:
        runtime_ref = None
    if not isinstance(provider_ref, str) or not provider_ref:
        provider_ref = None
    credential_key_ids = _credential_key_ids_for_provider(
        credentials_redacted, provider_ref, errors
    )

    if model_profiles:
        profile_runtime_ref = model_profiles.get("runtime_ref")
        if runtime_ref and profile_runtime_ref != runtime_ref:
            errors.append(
                f"model_profiles runtime_ref mismatch: expected {runtime_ref}, got {profile_runtime_ref}"
            )
        if isinstance(profile_runtime_ref, str) and profile_runtime_ref not in selected_refs:
            errors.append(f"model_profiles runtime_ref is not selected: {profile_runtime_ref}")
        provider_adapters = model_profiles.get("provider_adapters") or {}
        if isinstance(provider_adapters, dict):
            for adapter_ref in provider_adapters:
                if adapter_ref not in selected_refs:
                    errors.append(f"model_profiles provider adapter is not selected: {adapter_ref}")
                if provider_ref and adapter_ref != provider_ref:
                    errors.append(
                        f"model_profiles provider adapter mismatch: expected {provider_ref}, got {adapter_ref}"
                    )
        profiles = model_profiles.get("profiles") or {}
        if isinstance(profiles, dict):
            for profile_id, profile in profiles.items():
                if not isinstance(profile, dict):
                    continue
                profile_provider_ref = profile.get("provider_ref")
                profile_model = profile.get("model")
                if isinstance(profile_provider_ref, str):
                    if profile_provider_ref not in selected_refs:
                        errors.append(
                            f"model profile {profile_id} provider_ref is not selected: {profile_provider_ref}"
                        )
                    if provider_ref and profile_provider_ref != provider_ref:
                        errors.append(
                            f"model profile {profile_id} provider_ref mismatch: expected {provider_ref}, got {profile_provider_ref}"
                        )
                    _verify_model_provider_compatibility(
                        profile_provider_ref, profile_model, f"model profile {profile_id}", errors
                    )

    for index, event in enumerate(trajectory, start=1):
        actor = event.get("actor") or {}
        actor_type = actor.get("type") if isinstance(actor, dict) else None
        actor_id = actor.get("id") if isinstance(actor, dict) else None
        if actor_type == "agent_runtime":
            _verify_ref_binding(
                actor_id,
                runtime_ref,
                selected_refs,
                f"trajectory:{index} agent_runtime actor",
                errors,
            )
        if actor_type == "model_provider":
            _verify_ref_binding(
                actor_id,
                provider_ref,
                selected_refs,
                f"trajectory:{index} model_provider actor",
                errors,
            )
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        request = payload.get("request")
        checked_model_call = False
        if isinstance(request, dict):
            _verify_ref_binding(
                request.get("agent_runtime_ref"),
                runtime_ref,
                selected_refs,
                f"trajectory:{index} AgentRunRequest.agent_runtime_ref",
                errors,
            )
            model_call = request.get("model_call")
            if isinstance(model_call, dict):
                checked_model_call = True
                _verify_ref_binding(
                    model_call.get("provider_ref"),
                    provider_ref,
                    selected_refs,
                    f"trajectory:{index} AgentRunRequest.model_call.provider_ref",
                    errors,
                )
                _verify_model_provider_compatibility(
                    model_call.get("provider_ref"),
                    model_call.get("model"),
                    f"trajectory:{index} AgentRunRequest.model_call",
                    errors,
                )
                _verify_model_call_credential(
                    model_call,
                    provider_ref,
                    credential_key_ids,
                    f"trajectory:{index} AgentRunRequest.model_call",
                    errors,
                )
        model_call = payload.get("model_call")
        if isinstance(model_call, dict):
            checked_model_call = True
            _verify_ref_binding(
                model_call.get("provider_ref"),
                provider_ref,
                selected_refs,
                f"trajectory:{index} model_call.provider_ref",
                errors,
            )
            _verify_model_provider_compatibility(
                model_call.get("provider_ref"),
                model_call.get("model"),
                f"trajectory:{index} model_call",
                errors,
            )
            _verify_model_call_credential(
                model_call,
                provider_ref,
                credential_key_ids,
                f"trajectory:{index} model_call",
                errors,
            )
        if (
            event.get("kind") in {"agent.run_started", "agent.run_finished"}
            and actor_type == "agent_runtime"
        ):
            if not checked_model_call:
                _record_lockable_issue(
                    errors,
                    warnings,
                    f"trajectory:{index} {event.get('kind')} missing replayable model_call",
                    locked=locked,
                )
            request_budget_grant_id = (
                request.get("budget_grant_id") if isinstance(request, dict) else None
            )
            if "budget_grant_id" not in payload and not request_budget_grant_id:
                _record_lockable_issue(
                    errors,
                    warnings,
                    f"trajectory:{index} {event.get('kind')} missing budget_grant_id",
                    locked=locked,
                )
        if "provider_ref" in payload:
            _verify_ref_binding(
                payload.get("provider_ref"),
                provider_ref,
                selected_refs,
                f"trajectory:{index} model result provider_ref",
                errors,
            )


def _verify_runtime_provider_conformance(
    selected_plugins: list[Any],
    startup_config: dict[str, Any] | None,
    model_profiles: dict[str, Any] | None,
    cache_policy: dict[str, Any] | None,
    errors: list[str],
) -> None:
    canonical_args = (startup_config or {}).get("canonical_args") or {}
    if not isinstance(canonical_args, dict):
        return
    runtime_ref = canonical_args.get("runtime")
    provider_ref = canonical_args.get("model_provider")
    if not isinstance(runtime_ref, str) or not isinstance(provider_ref, str):
        return
    selected_by_ref = {
        f"{item.get('metadata', {}).get('kind')}:{item.get('metadata', {}).get('name')}": item
        for item in selected_plugins
        if isinstance(item, dict) and isinstance(item.get("metadata"), dict)
    }
    runtime_contract = _selected_manifest_contract(selected_by_ref.get(runtime_ref), "runtime")
    provider_contract = _selected_manifest_contract(selected_by_ref.get(provider_ref), "provider")
    compatible = [
        str(item) for item in runtime_contract.get("compatible_model_providers") or [] if item
    ]
    if compatible and provider_ref not in compatible:
        errors.append(
            f"runtime/provider conformance mismatch: {runtime_ref} is not compatible with {provider_ref}"
        )
    expected_mode, expected_runtime_strategy, expected_provider_strategy = _expected_cache_contract(
        runtime_contract,
        provider_contract,
    )
    if cache_policy:
        if cache_policy.get("mode") != expected_mode:
            errors.append(
                "cache_policy mode mismatch for runtime/provider: "
                f"expected {expected_mode}, got {cache_policy.get('mode')}"
            )
        if cache_policy.get("runtime_cache_strategy") != expected_runtime_strategy:
            errors.append(
                "cache_policy runtime_cache_strategy mismatch: "
                f"expected {expected_runtime_strategy}, got {cache_policy.get('runtime_cache_strategy')}"
            )
        if cache_policy.get("provider_cache_strategy") != expected_provider_strategy:
            errors.append(
                "cache_policy provider_cache_strategy mismatch: "
                f"expected {expected_provider_strategy}, got {cache_policy.get('provider_cache_strategy')}"
            )
    conformance = (model_profiles or {}).get("runtime_provider_conformance") or {}
    if isinstance(conformance, dict) and conformance:
        if conformance.get("runtime_ref") != runtime_ref:
            errors.append("model_profiles runtime_provider_conformance runtime_ref mismatch")
        if conformance.get("model_provider_ref") != provider_ref:
            errors.append("model_profiles runtime_provider_conformance model_provider_ref mismatch")
        if conformance.get("cache_mode") != expected_mode:
            errors.append("model_profiles runtime_provider_conformance cache_mode mismatch")
        if conformance.get("cache_policy_runtime_strategy") != expected_runtime_strategy:
            errors.append("model_profiles runtime_provider_conformance runtime cache mismatch")
        if conformance.get("cache_policy_provider_strategy") != expected_provider_strategy:
            errors.append("model_profiles runtime_provider_conformance provider cache mismatch")


def _selected_manifest_contract(selected: Any, key: str) -> dict[str, Any]:
    if not isinstance(selected, dict):
        return {}
    path = selected.get("path")
    if not isinstance(path, str) or path.startswith("builtin://"):
        return {}
    try:
        value = yaml.safe_load((Path(path) / "plugin.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(value, dict):
        return {}
    contract = value.get(key)
    return dict(contract) if isinstance(contract, dict) else {}


def _expected_cache_contract(
    runtime_contract: dict[str, Any],
    provider_contract: dict[str, Any],
) -> tuple[str, str | None, str | None]:
    runtime_strategy = str(runtime_contract.get("cache_strategy") or "")
    provider_strategy = str(provider_contract.get("cache_strategy") or "")
    if runtime_strategy in {"disabled", "deterministic_no_cache"}:
        return "disabled", None, None
    if provider_strategy in {"disabled", "deterministic_no_cache"}:
        return "disabled", None, None
    if runtime_strategy == "runtime_auto_cache":
        return "runtime_auto_cache", runtime_strategy, None
    if provider_strategy == "provider_explicit_cache":
        return (
            "provider_explicit_cache",
            None,
            str(provider_contract.get("explicit_cache_strategy") or "provider_explicit_cache"),
        )
    return "provider_default", None, None


def _verify_ref_binding(
    value: Any,
    expected: str | None,
    selected_refs: set[str],
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} missing ref")
        return
    if value not in selected_refs:
        errors.append(f"{label} is not selected: {value}")
    if expected and value != expected:
        errors.append(f"{label} mismatch: expected {expected}, got {value}")


def _verify_model_provider_compatibility(
    provider_ref: Any,
    model: Any,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(provider_ref, str) or not isinstance(model, str) or not model:
        return
    try:
        from praxist.core.modeling import validate_model_for_provider

        validate_model_for_provider(provider_ref, model)
    except ValueError as exc:
        errors.append(f"{label} model/provider incompatible: {exc}")


def _credential_key_ids_for_provider(
    credentials_redacted: dict[str, Any] | None,
    provider_ref: str | None,
    errors: list[str],
) -> set[str]:
    if not provider_ref or provider_ref == "model_provider:fake_provider":
        return set()
    provider_name = provider_ref.split(":", 1)[1] if ":" in provider_ref else provider_ref
    if not credentials_redacted:
        errors.append(f"credentials_redacted missing for selected model provider: {provider_ref}")
        return set()
    profiles = credentials_redacted.get("credential_profiles")
    if not isinstance(profiles, list):
        errors.append("credentials_redacted credential_profiles is not a list")
        return set()
    key_ids: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        if profile.get("scope") != "model_provider":
            continue
        if profile.get("provider") != provider_name:
            continue
        target_ref = profile.get("target_ref")
        if target_ref not in (None, provider_ref):
            continue
        if profile.get("status", "active") != "active":
            continue
        key_id = profile.get("key_id")
        if isinstance(key_id, str) and key_id:
            key_ids.add(key_id)
    if not key_ids:
        errors.append(
            f"credentials_redacted missing active credential for selected model provider: {provider_ref}"
        )
    return key_ids


def _verify_model_call_credential(
    model_call: dict[str, Any],
    provider_ref: str | None,
    credential_key_ids: set[str],
    label: str,
    errors: list[str],
) -> None:
    credential_ref = model_call.get("credential_ref")
    credential_key_id = _credential_key_id(credential_ref)
    if (
        provider_ref == "model_provider:fake_provider"
        and not credential_key_ids
        and not credential_key_id
    ):
        return
    if not credential_key_id:
        errors.append(f"{label} missing credential_ref")
        return
    if credential_key_ids and credential_key_id not in credential_key_ids:
        errors.append(
            f"{label} credential_ref is not selected for provider {provider_ref}: {credential_key_id}"
        )
    if isinstance(credential_ref, dict):
        target_ref = credential_ref.get("target_ref")
        if provider_ref and target_ref not in (None, provider_ref):
            errors.append(
                f"{label} credential_ref target mismatch: expected {provider_ref}, got {target_ref}"
            )


def _credential_key_id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        key_id = value.get("key_id")
        if isinstance(key_id, str) and key_id:
            return key_id
    return None


def _verify_plugin_dependency_closure(
    selected_plugins: list[Any],
    dependency_edges: Any,
    selected_refs: set[str],
    errors: list[str],
) -> None:
    edges = dependency_edges if isinstance(dependency_edges, list) else []
    edge_pairs = {
        (edge.get("from"), edge.get("to"))
        for edge in edges
        if isinstance(edge, dict)
        and isinstance(edge.get("from"), str)
        and isinstance(edge.get("to"), str)
    }
    for edge_from, edge_to in edge_pairs:
        if edge_from not in selected_refs:
            errors.append(f"plugin_resolution dependency edge has unselected from ref: {edge_from}")
        if edge_to not in selected_refs:
            errors.append(f"plugin_resolution dependency edge has unselected to ref: {edge_to}")

    for item in selected_plugins:
        if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
            errors.append("plugin_resolution selected item missing metadata")
            continue
        metadata = item["metadata"]
        ref = f"{metadata.get('kind')}:{metadata.get('name')}"
        dependencies = metadata.get("dependencies") or []
        if not isinstance(dependencies, list):
            errors.append(f"plugin_resolution {ref} dependencies is not a list")
            continue
        for dep in dependencies:
            if not isinstance(dep, dict) or dep.get("required", True) is False:
                continue
            dep_kind = dep.get("kind")
            dep_name = dep.get("name")
            if not isinstance(dep_kind, str) or not isinstance(dep_name, str):
                errors.append(f"plugin_resolution {ref} has malformed dependency")
                continue
            dep_ref = f"{dep_kind}:{dep_name}"
            if dep_ref not in selected_refs:
                errors.append(f"plugin_resolution missing required dependency {dep_ref} for {ref}")
            if (ref, dep_ref) not in edge_pairs:
                errors.append(f"plugin_resolution missing dependency edge {ref} -> {dep_ref}")


def _budget_request_from_record(record: dict[str, Any], errors: list[str]) -> BudgetRequest | None:
    request_record = record.get("request_record")
    if not isinstance(request_record, dict):
        errors.append(
            f"budget {record.get('kind')} {record.get('record_id')} missing request_record"
        )
        return None
    try:
        requested = request_record.get("requested")
        if not isinstance(requested, dict):
            raise ValueError("requested must be an object")
        return BudgetRequest(
            request_id=str(request_record["request_id"]),
            requester_id=str(request_record["requester_id"]),
            experiment_id=str(request_record["experiment_id"]),
            model_profile_ref=(
                str(request_record["model_profile_ref"])
                if request_record.get("model_profile_ref") is not None
                else None
            ),
            requested=dict(requested),
            expected_value=dict(request_record.get("expected_value") or {}),
            evidence_refs=[str(item) for item in request_record.get("evidence_refs") or []],
            cheaper_alternatives=[
                str(item) for item in request_record.get("cheaper_alternatives") or []
            ],
            abort_conditions=[str(item) for item in request_record.get("abort_conditions") or []],
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(
            f"budget {record.get('kind')} {record.get('record_id')} malformed request_record: {exc}"
        )
        return None


def _verify_budget_decision_against_policy(
    record: dict[str, Any],
    request: BudgetRequest,
    budget_policy_ref: str | None,
    errors: list[str],
    *,
    allow_uncapped: bool = False,
) -> None:
    if not budget_policy_ref:
        errors.append(
            f"budget decision {record.get('record_id')} cannot be replayed without budget_policy ref"
        )
        return
    try:
        policy = policy_for_ref(budget_policy_ref)
        decision = (
            policy.decide(request, allow_uncapped=True)
            if allow_uncapped and any(amount is None for amount in request.requested.values())
            else policy.decide(request)
        )
        expected_decision = decision.to_dict()
    except Exception as exc:  # noqa: BLE001 - replay should surface policy load errors.
        errors.append(f"budget decision {record.get('record_id')} policy replay failed: {exc}")
        return
    actual_decision = record.get("decision_record")
    if actual_decision != expected_decision:
        errors.append(
            f"budget decision {record.get('record_id')} does not match replayed policy decision"
        )
    if record.get("decision") != expected_decision.get("decision"):
        errors.append(
            f"budget decision {record.get('record_id')} decision field does not match replayed policy"
        )
    expected_grant = expected_decision.get("grant")
    if expected_grant is None:
        if record.get("grant_id") is not None or record.get("granted_budget") is not None:
            errors.append(
                f"budget decision {record.get('record_id')} has grant fields but policy did not grant"
            )
        return
    if record.get("grant_id") != expected_grant.get("grant_id"):
        errors.append(
            f"budget decision {record.get('record_id')} grant_id does not match replayed policy"
        )
    if record.get("granted_budget") != expected_grant.get("approved"):
        errors.append(
            f"budget decision {record.get('record_id')} granted_budget does not match replayed policy"
        )


def _verify_required_budget_usage(
    records: list[dict[str, Any]],
    run_summary: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not run_summary:
        summary_execution = False
    else:
        summary_execution = _summary_indicates_execution(run_summary)
    output_execution = bool(findings or frontier)
    stage_execution = any(
        event.get("kind") == "workflow.stage_succeeded"
        and (event.get("scope") or {}).get("stage_id") == "research_loop"
        and _stage_payload_indicates_execution(event.get("payload"))
        for event in trajectory
    )
    if not (summary_execution or output_execution or stage_execution):
        return
    has_grant = any(
        record.get("kind") == "decision" and record.get("granted_budget") for record in records
    )
    has_usage = any(record.get("kind") in {"usage", "usage_unknown"} for record in records)
    if has_grant and not has_usage:
        warnings.append(
            "budget ledger missing usage or usage_unknown record for executed successful run"
        )


def _summary_indicates_execution(run_summary: dict[str, Any]) -> bool:
    if _positive_number(run_summary.get("generations_completed")):
        return True
    if _positive_number(run_summary.get("frontier_records")):
        return True
    finding_summary = run_summary.get("finding_summary") or {}
    if isinstance(finding_summary, dict):
        for key in ("drafts", "accepted", "retry_corrections"):
            if _positive_number(finding_summary.get(key)):
                return True
    legacy_summary = run_summary.get("legacy_generation_loop_summary") or {}
    if isinstance(legacy_summary, dict):
        if legacy_summary.get("exit_condition") == "resolve_only":
            return False
        if _positive_number(legacy_summary.get("generations_completed")):
            return True
        if "total_duration_seconds" in legacy_summary:
            return True
    return False


def _stage_payload_indicates_execution(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("exit_condition") == "resolve_only":
        return False
    if _positive_number(payload.get("findings")) or _positive_number(
        payload.get("frontier_records")
    ):
        return True
    result = payload.get("result") or {}
    if isinstance(result, dict):
        if result.get("exit_condition") == "resolve_only":
            return False
        if _positive_number(result.get("generations_completed")):
            return True
        if "total_duration_seconds" in result:
            return True
    return False


def _verify_run_summary_consistency(
    run_json: dict[str, Any] | None,
    run_summary: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if not run_summary:
        return
    expected_run_id = (run_json or {}).get("run_id")
    if isinstance(expected_run_id, str) and run_summary.get("run_id") != expected_run_id:
        errors.append(
            f"run_summary run_id mismatch: expected {expected_run_id}, got {run_summary.get('run_id')}"
        )
    run_status = (run_json or {}).get("status")
    summary_status = run_summary.get("status")
    if isinstance(run_status, str) and run_status and summary_status != run_status:
        errors.append(
            f"run_summary status mismatch: run.json has {run_status}, run_summary has {summary_status}"
        )
    finalized_payloads = [
        event.get("payload")
        for event in trajectory
        if event.get("kind") == "run.finalized" and isinstance(event.get("payload"), dict)
    ]
    if not finalized_payloads:
        return
    finalized_payload = finalized_payloads[-1]
    for key in ("run_id", "status", "exit_condition", "exit_code", "output_hashes"):
        if (key in run_summary or key in finalized_payload) and finalized_payload.get(
            key
        ) != run_summary.get(key):
            errors.append(
                f"run_summary {key} mismatch: run.finalized has {finalized_payload.get(key)}, "
                f"run_summary has {run_summary.get(key)}"
            )


def _verify_output_counts(
    run_summary: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    research_memory: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    errors: list[str],
) -> None:
    finding_count = len(findings)
    frontier_count = len(frontier)
    memory_count = len(research_memory)
    graph_edge_count = len(graph_edges)
    if run_summary:
        _check_count(
            run_summary.get("frontier_records"),
            frontier_count,
            "run_summary frontier_records",
            errors,
        )
        _check_count(
            run_summary.get("research_memory_records"),
            memory_count,
            "run_summary research_memory_records",
            errors,
        )
        _check_count(
            run_summary.get("graph_edges"), graph_edge_count, "run_summary graph_edges", errors
        )
        finding_summary = run_summary.get("finding_summary") or {}
        if isinstance(finding_summary, dict):
            draft_count = sum(1 for finding in findings if finding.get("status") == "draft")
            retry_count = sum(1 for finding in findings if finding.get("supersedes"))
            _check_count(
                finding_summary.get("drafts"),
                draft_count,
                "run_summary finding_summary.drafts",
                errors,
            )
            _check_count(
                finding_summary.get("retry_corrections"),
                retry_count,
                "run_summary finding_summary.retry_corrections",
                errors,
            )
            _check_count(
                finding_summary.get("accepted"),
                frontier_count,
                "run_summary finding_summary.accepted",
                errors,
            )

    for index, event in enumerate(trajectory, start=1):
        if event.get("kind") != "workflow.stage_succeeded":
            continue
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        _check_count(
            payload.get("findings"), finding_count, f"trajectory:{index} stage findings", errors
        )
        _check_count(
            payload.get("frontier_records"),
            frontier_count,
            f"trajectory:{index} stage frontier_records",
            errors,
        )
        _check_count(
            payload.get("research_memory_records"),
            memory_count,
            f"trajectory:{index} stage research_memory_records",
            errors,
        )
        _check_count(
            payload.get("graph_edges"),
            graph_edge_count,
            f"trajectory:{index} stage graph_edges",
            errors,
        )
        result = payload.get("result") or {}
        if isinstance(result, dict):
            frontier_summary = result.get("frontier_summary")
            if isinstance(frontier_summary, list):
                _check_count(
                    len(frontier_summary),
                    frontier_count,
                    f"trajectory:{index} result frontier_summary",
                    errors,
                )


def _verify_output_hashes(
    run_dir: Path,
    run_summary: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    research_memory: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if not run_summary:
        return
    needs_hashes = (
        run_summary.get("status") == "succeeded"
        or bool(findings)
        or bool(frontier)
        or bool(research_memory)
        or bool(graph_edges)
        or any(
            event.get("kind") == "workflow.stage_succeeded"
            and (event.get("scope") or {}).get("stage_id") == "research_loop"
            for event in trajectory
        )
    )
    expected = run_summary.get("output_hashes")
    if not needs_hashes and expected is None:
        return
    if not isinstance(expected, dict):
        errors.append("run_summary missing output_hashes for canonical output ledgers")
        return
    actual = output_ledger_hashes(run_dir)
    for rel in OUTPUT_LEDGER_RELS:
        expected_hash = expected.get(rel)
        if not _is_sha256_digest(expected_hash):
            errors.append(
                f"run_summary output_hashes has invalid digest for {rel}: {expected_hash}"
            )
            continue
        if expected_hash != actual.get(rel):
            errors.append(f"run_summary output_hashes mismatch for {rel}")
    for rel in expected:
        if rel not in OUTPUT_LEDGER_RELS:
            errors.append(f"run_summary output_hashes contains unexpected ledger: {rel}")


def _check_count(value: Any, actual: int, label: str, errors: list[str]) -> None:
    if value is None:
        return
    try:
        expected = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} is not an integer count: {value}")
        return
    if expected != actual:
        errors.append(f"{label} mismatch: expected {expected}, found {actual}")


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _verify_budget_amounts(values: dict[str, Any], label: str, errors: list[str]) -> None:
    for unit, raw_value in values.items():
        if unit not in ALLOWED_BUDGET_UNITS:
            errors.append(f"budget {label} has unsupported unit: {unit}")
            continue
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"budget {label} has non-numeric amount for {unit}: {raw_value}")
            continue
        if not math.isfinite(value) or value < 0:
            errors.append(f"budget {label} has invalid amount for {unit}: {raw_value}")


def _verify_source_event_ids(
    records: list[dict[str, Any]],
    known_event_ids: set[str],
    label: str,
    errors: list[str],
) -> None:
    for index, record in enumerate(records, start=1):
        source_event_ids = record.get("source_event_ids") or []
        if not isinstance(source_event_ids, list):
            errors.append(f"{label}:{index} source_event_ids is not a list")
            continue
        for event_id in source_event_ids:
            if event_id not in known_event_ids:
                errors.append(f"{label}:{index} references unknown source_event_id: {event_id}")


def _verify_output_agent_provenance(
    trajectory: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> None:
    event_by_id = {
        event.get("event_id"): event
        for event in trajectory
        if isinstance(event.get("event_id"), str)
    }

    def has_agent_finished(record: dict[str, Any]) -> bool:
        source_ids = record.get("source_event_ids") or []
        if not isinstance(source_ids, list):
            return False
        for event_id in source_ids:
            event = event_by_id.get(event_id)
            if not isinstance(event, dict) or event.get("kind") != "agent.run_finished":
                continue
            if _agent_event_supports_finding(event, record):
                return True
        return False

    def has_imported_provenance(record: dict[str, Any]) -> bool:
        quality = str(record.get("provenance_quality") or "")
        source_ids = record.get("source_event_ids") or []
        if not isinstance(source_ids, list):
            return False
        imported_event = any(
            isinstance(event_id, str)
            and isinstance(event_by_id.get(event_id), dict)
            and event_by_id[event_id].get("kind")
            in {"legacy.output_imported", "finding.imported", "frontier.imported"}
            for event_id in source_ids
        )
        return imported_event and quality in {"legacy_weak", "imported", "weak"}

    finding_provenance: dict[str, str] = {}
    for index, record in enumerate(findings, start=1):
        finding_id = record.get("finding_id")
        provenance_state = ""
        if has_agent_finished(record):
            provenance_state = "agent"
        elif has_imported_provenance(record):
            provenance_state = "imported"
            warnings.append(f"findings:{index} uses imported legacy provenance")
        if isinstance(finding_id, str) and finding_id:
            finding_provenance[finding_id] = provenance_state
        if not provenance_state:
            errors.append(f"findings:{index} missing agent.run_finished provenance")
    for index, record in enumerate(frontier, start=1):
        finding_id = record.get("finding_id")
        if not isinstance(finding_id, str):
            continue
        if finding_provenance.get(finding_id) in {"agent", "imported"}:
            continue
        if has_agent_finished(record):
            continue
        if has_imported_provenance(record):
            warnings.append(
                f"frontier:{index} uses imported legacy provenance for finding: {finding_id}"
            )
            continue
        if finding_id:
            errors.append(
                f"frontier:{index} promoted finding missing agent.run_finished provenance: {finding_id}"
            )


def _agent_event_supports_finding(event: dict[str, Any], record: dict[str, Any]) -> bool:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    finding_id = str(record.get("finding_id") or "")
    if str(payload.get("finding_id") or "") == finding_id:
        return True
    legacy_payload = record.get("legacy_payload")
    finding_payload = legacy_payload if isinstance(legacy_payload, dict) else record
    if "id" not in finding_payload and finding_id:
        finding_payload = {**finding_payload, "id": finding_id}
    for tool_use in _tool_uses_from_agent_payload(payload):
        if _share_finding_tool_input_matches(tool_use, finding_payload):
            return True
    return False


def _tool_uses_from_agent_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for key in ("output_summary", "runtime_output", "output"):
        value = payload.get(key)
        if isinstance(value, dict):
            summaries.append(value)
    tool_uses = []
    for summary in summaries:
        raw = summary.get("tool_uses")
        if isinstance(raw, list):
            tool_uses.extend(item for item in raw if isinstance(item, dict))
    return tool_uses


def _share_finding_tool_input_matches(tool_use: dict[str, Any], finding: dict[str, Any]) -> bool:
    tool_name = str(tool_use.get("tool") or tool_use.get("name") or "")
    if "share_finding" not in tool_name:
        return False
    tool_input = tool_use.get("input") or {}
    if not isinstance(tool_input, dict):
        return False
    if str(tool_input.get("peer_id") or "") != str(finding.get("peer_id") or ""):
        return False
    for field in ("finding_type", "title", "content", "variant_name"):
        expected = finding.get(field)
        observed = tool_input.get(field)
        if (
            expected not in (None, "")
            and observed not in (None, "")
            and _norm_text(expected) != _norm_text(observed)
        ):
            return False
    expected_metrics = finding.get("metrics")
    observed_metrics = _parse_metrics(tool_input.get("metrics"))
    return not (
        isinstance(expected_metrics, dict)
        and observed_metrics is not None
        and _jsonable(expected_metrics) != _jsonable(observed_metrics)
    )


def _parse_metrics(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _norm_text(value: Any) -> str:
    return " ".join(str(value).split())


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _verify_frontier_finding_refs(
    frontier: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    errors: list[str],
) -> None:
    finding_ids = {
        finding.get("finding_id")
        for finding in findings
        if isinstance(finding.get("finding_id"), str) and finding.get("finding_id")
    }
    for index, record in enumerate(frontier, start=1):
        finding_id = record.get("finding_id")
        if not isinstance(finding_id, str) or not finding_id:
            errors.append(f"frontier:{index} missing finding_id")
        elif finding_id not in finding_ids:
            errors.append(f"frontier:{index} references unknown finding_id: {finding_id}")


def _verify_artifact_references(
    records: list[dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
    label: str,
    errors: list[str],
) -> None:
    for index, record in enumerate(records, start=1):
        source_artifact_ids = record.get("source_artifact_ids") or []
        if not isinstance(source_artifact_ids, list):
            errors.append(f"{label}:{index} source_artifact_ids is not a list")
        else:
            for artifact_id in source_artifact_ids:
                if artifact_id not in artifact_by_id:
                    errors.append(
                        f"{label}:{index} references unknown source_artifact_id: {artifact_id}"
                    )

        artifact_refs = record.get("artifact_refs") or []
        if not isinstance(artifact_refs, list):
            errors.append(f"{label}:{index} artifact_refs is not a list")
        else:
            _verify_artifact_ref_list(
                artifact_refs, artifact_by_id, f"{label}:{index} artifact_refs", errors
            )

        evidence_refs = record.get("evidence_refs") or []
        if not isinstance(evidence_refs, list):
            errors.append(f"{label}:{index} evidence_refs is not a list")
        else:
            artifact_evidence_refs = [
                item for item in evidence_refs if isinstance(item, dict) and "artifact_id" in item
            ]
            _verify_artifact_ref_list(
                artifact_evidence_refs,
                artifact_by_id,
                f"{label}:{index} evidence_refs",
                errors,
            )


def _verify_artifact_ref_list(
    artifact_refs: list[Any],
    artifact_by_id: dict[str, dict[str, Any]],
    label: str,
    errors: list[str],
) -> None:
    for ref_index, artifact_ref in enumerate(artifact_refs, start=1):
        if not isinstance(artifact_ref, dict):
            errors.append(f"{label}[{ref_index}] is not an object")
            continue
        artifact_id = artifact_ref.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            errors.append(f"{label}[{ref_index}] missing artifact_id")
            continue
        indexed = artifact_by_id.get(artifact_id)
        if indexed is None:
            errors.append(f"{label}[{ref_index}] references unknown artifact_id: {artifact_id}")
            continue
        for key in ("payload_path", "content_hash"):
            ref_value = artifact_ref.get(key)
            if ref_value != indexed.get(key):
                errors.append(
                    f"{label}[{ref_index}] {key} mismatch for {artifact_id}: "
                    f"expected {indexed.get(key)}, got {ref_value}"
                )


def _verify_prompt_layout_artifacts(
    run_dir: Path,
    artifact_index: list[dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> None:
    layout_artifacts = [
        item for item in artifact_index if item.get("artifact_type") == "prompt.layout_manifest"
    ]
    if not layout_artifacts:
        if any(
            item.get("kind") == "agent.run_started"
            for item in read_jsonl(run_dir / "trajectory.jsonl")[0]
        ):
            warnings.append("no prompt.layout_manifest artifacts found for agent run")
        return
    for artifact in layout_artifacts:
        payload_rel = artifact.get("payload_path")
        if not isinstance(payload_rel, str):
            continue
        payload_path = _resolve_run_relative_path(
            run_dir,
            payload_rel,
            errors,
            f"prompt layout {artifact.get('artifact_id')} payload_path",
        )
        if payload_path is None or not payload_path.exists():
            continue
        payload = _read_json(payload_path, errors)
        if not isinstance(payload, dict):
            errors.append(f"prompt layout {artifact.get('artifact_id')} payload is not an object")
            continue
        if payload.get("schema_version") != "praxist.prompt_layout.v1":
            errors.append(f"prompt layout {artifact.get('artifact_id')} has invalid schema_version")
        if payload.get("layout_version") != "praxist.prompt_layout.v1":
            errors.append(f"prompt layout {artifact.get('artifact_id')} has invalid layout_version")
        blocks = payload.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            errors.append(f"prompt layout {artifact.get('artifact_id')} has no blocks")
            continue
        _verify_prompt_layout_hashes(payload, blocks, artifact.get("artifact_id"), errors)
        audit = payload.get("frozen_audit") or {}
        if not isinstance(audit, dict) or audit.get("status") != "pass":
            errors.append(f"prompt layout {artifact.get('artifact_id')} frozen_audit did not pass")
        for index, block in enumerate(blocks, start=1):
            if not isinstance(block, dict):
                errors.append(
                    f"prompt layout {artifact.get('artifact_id')} block {index} is not an object"
                )
                continue
            if block.get("partition") not in {
                "frozen_prefix",
                "semi_static_run_context",
                "dynamic_payload",
            }:
                errors.append(
                    f"prompt layout {artifact.get('artifact_id')} block {index} has invalid partition: "
                    f"{block.get('partition')}"
                )
            if not _is_sha256_digest(block.get("rendered_hash")):
                errors.append(
                    f"prompt layout {artifact.get('artifact_id')} block {index} has invalid rendered_hash: "
                    f"{block.get('rendered_hash')}"
                )
            if block.get("partition") == "frozen_prefix":
                markers = list(block.get("dynamic_markers_in_template") or []) + list(
                    block.get("dynamic_markers_in_rendered") or []
                )
                if markers:
                    errors.append(
                        f"prompt layout {artifact.get('artifact_id')} frozen block "
                        f"{block.get('block_id')} contains dynamic markers: {sorted(set(map(str, markers)))}"
                    )
        if payload.get("cache_mode") == "provider_explicit_cache" and not payload.get(
            "provider_cache_strategy"
        ):
            errors.append(
                f"prompt layout {artifact.get('artifact_id')} provider_explicit_cache missing provider strategy"
            )
        if payload.get("cache_mode") == "runtime_auto_cache" and not payload.get(
            "runtime_cache_strategy"
        ):
            errors.append(
                f"prompt layout {artifact.get('artifact_id')} runtime_auto_cache missing runtime strategy"
            )
        rendered_ref = payload.get("rendered_prompt_ref")
        if isinstance(rendered_ref, dict):
            _verify_artifact_ref_list(
                [rendered_ref],
                artifact_by_id,
                f"prompt layout {artifact.get('artifact_id')} rendered_prompt_ref",
                errors,
            )
            _verify_rendered_prompt_hash(
                run_dir, payload, rendered_ref, artifact.get("artifact_id"), errors
            )


def _verify_prompt_layout_hashes(
    payload: dict[str, Any],
    blocks: list[Any],
    artifact_id: Any,
    errors: list[str],
) -> None:
    normalized_blocks = [block for block in blocks if isinstance(block, dict)]
    for partition, key in (
        ("frozen_prefix", "frozen_prefix_hash"),
        ("semi_static_run_context", "semi_static_hash"),
        ("dynamic_payload", "dynamic_payload_hash"),
    ):
        expected = sha256_json(
            [
                {
                    "block_id": block.get("block_id"),
                    "rendered_hash": block.get("rendered_hash"),
                }
                for block in normalized_blocks
                if block.get("partition") == partition
            ]
        )
        if payload.get(key) != expected:
            errors.append(
                f"prompt layout {artifact_id} {key} mismatch: expected {expected}, got {payload.get(key)}"
            )
    expected_layout_hash = sha256_json(
        {
            "layout_version": "praxist.prompt_layout.v1",
            "frozen_prefix_hash": payload.get("frozen_prefix_hash"),
            "semi_static_hash": payload.get("semi_static_hash"),
            "dynamic_payload_hash": payload.get("dynamic_payload_hash"),
            "block_hashes": [block.get("rendered_hash") for block in normalized_blocks],
        }
    )
    if payload.get("layout_hash") != expected_layout_hash:
        errors.append(
            f"prompt layout {artifact_id} layout_hash mismatch: "
            f"expected {expected_layout_hash}, got {payload.get('layout_hash')}"
        )


def _verify_rendered_prompt_hash(
    run_dir: Path,
    payload: dict[str, Any],
    rendered_ref: dict[str, Any],
    artifact_id: Any,
    errors: list[str],
) -> None:
    expected_hash = payload.get("rendered_prompt_hash")
    if not isinstance(expected_hash, str):
        errors.append(f"prompt layout {artifact_id} missing rendered_prompt_hash")
        return
    payload_rel = rendered_ref.get("payload_path")
    if not isinstance(payload_rel, str):
        return
    rendered_path = _resolve_run_relative_path(
        run_dir,
        payload_rel,
        errors,
        f"prompt layout {artifact_id} rendered_prompt_ref payload_path",
    )
    if rendered_path is None or not rendered_path.exists():
        return
    actual_hash = sha256_text(rendered_path.read_text(encoding="utf-8", errors="ignore"))
    if actual_hash != expected_hash:
        errors.append(
            f"prompt layout {artifact_id} rendered_prompt_hash mismatch: "
            f"expected {expected_hash}, got {actual_hash}"
        )


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in value[7:])
    )


def _verify_state_surface_recovery(
    run_dir: Path,
    findings: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    research_memory: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    artifact_index: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    legacy = _legacy_state_surfaces(run_dir)
    canonical = {
        "finding_ids": sorted(_record_ids(findings, "finding_id")),
        "frontier_finding_ids": sorted(_record_ids(frontier, "finding_id")),
        "research_memory_records": len(research_memory),
        "graph_edge_ids": sorted(_record_ids(graph_edges, "graph_edge_id")),
        "artifact_types": _artifact_type_counts(artifact_index),
    }

    missing_shared = sorted(set(legacy["shared_finding_ids"]) - set(canonical["finding_ids"]))
    missing_sqlite = sorted(set(legacy["sqlite_finding_ids"]) - set(canonical["finding_ids"]))
    missing_frontier = sorted(
        set(legacy["frontier_finding_ids"]) - set(canonical["frontier_finding_ids"])
    )
    missing_edges = sorted(set(legacy["sqlite_graph_edge_ids"]) - set(canonical["graph_edge_ids"]))

    if missing_shared:
        errors.append(
            "state recovery: canonical findings missing shared_findings ids: "
            + ", ".join(missing_shared[:10])
        )
    if missing_sqlite:
        errors.append(
            "state recovery: canonical findings missing SQLite finding ids: "
            + ", ".join(missing_sqlite[:10])
        )
    if missing_frontier:
        errors.append(
            "state recovery: canonical frontier missing legacy frontier ids: "
            + ", ".join(missing_frontier[:10])
        )
    if missing_edges:
        errors.append(
            "state recovery: canonical graph_edges missing SQLite edge ids: "
            + ", ".join(missing_edges[:10])
        )
    if legacy["research_memory_entry_count"] > len(research_memory):
        errors.append(
            "state recovery: canonical research_memory records fewer than legacy YAML entries: "
            f"{len(research_memory)} < {legacy['research_memory_entry_count']}"
        )
    graph_artifact_count = int(canonical["artifact_types"].get("graph_materialized_artifact", 0))
    if legacy["graph_artifact_count"] > graph_artifact_count:
        errors.append(
            "state recovery: canonical graph artifacts fewer than legacy graph artifacts: "
            f"{graph_artifact_count} < {legacy['graph_artifact_count']}"
        )

    shared_only = sorted(set(legacy["shared_finding_ids"]) - set(legacy["sqlite_finding_ids"]))
    sqlite_only = sorted(set(legacy["sqlite_finding_ids"]) - set(legacy["shared_finding_ids"]))
    if shared_only:
        warnings.append(
            "state recovery: shared_findings has ids not present in SQLite "
            f"(sync lag or partial run): {', '.join(shared_only[:10])}"
        )
    if sqlite_only:
        warnings.append(
            "state recovery: SQLite has ids not present in shared_findings "
            f"(sync lag or partial run): {', '.join(sqlite_only[:10])}"
        )

    return {
        "schema_version": "praxist.state_recovery.v1",
        "legacy": legacy,
        "canonical": canonical,
        "missing": {
            "shared_findings_not_in_canonical": missing_shared,
            "sqlite_findings_not_in_canonical": missing_sqlite,
            "frontier_manifest_not_in_canonical": missing_frontier,
            "sqlite_graph_edges_not_in_canonical": missing_edges,
        },
    }


def _legacy_state_surfaces(run_dir: Path) -> dict[str, Any]:
    shared_finding_ids = _shared_finding_ids(run_dir)
    sqlite_finding_ids, sqlite_graph_edge_ids = _sqlite_state_ids(run_dir)
    frontier_ids = _frontier_manifest_ids(run_dir)
    memory_entry_count = _research_memory_ledger_entry_count(run_dir)
    graph_artifact_count = sum(
        1 for path in _legacy_graph_artifact_paths(run_dir) if path.is_file()
    )
    return {
        "shared_finding_ids": sorted(shared_finding_ids),
        "sqlite_finding_ids": sorted(sqlite_finding_ids),
        "frontier_finding_ids": sorted(frontier_ids),
        "sqlite_graph_edge_ids": sorted(sqlite_graph_edge_ids),
        "research_memory_entry_count": memory_entry_count,
        "graph_artifact_count": graph_artifact_count,
    }


def _shared_finding_ids(run_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in sorted((run_dir / "shared_findings").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        finding_id = value.get("finding_id") or value.get("id")
        if isinstance(finding_id, str) and finding_id:
            ids.add(finding_id)
    return ids


def _sqlite_state_ids(run_dir: Path) -> tuple[set[str], set[str]]:
    db_path = run_dir / "shared_store.db"
    if not db_path.exists():
        return set(), set()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return set(), set()
    try:
        finding_ids = {
            str(row["id"])
            for row in conn.execute("SELECT id FROM findings").fetchall()
            if row["id"]
        }
    except sqlite3.Error:
        finding_ids = set()
    try:
        graph_edge_ids = {
            str(row["edge_id"])
            for row in conn.execute("SELECT edge_id FROM finding_edges").fetchall()
            if row["edge_id"]
        }
    except sqlite3.Error:
        graph_edge_ids = set()
    finally:
        conn.close()
    return finding_ids, graph_edge_ids


def _frontier_manifest_ids(run_dir: Path) -> set[str]:
    path = run_dir / "frontier" / "frontier_manifest.json"
    if not path.exists():
        return set()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(manifest, dict):
        return set()
    ids: set[str] = set()

    def add_entry(entry: Any) -> None:
        if not isinstance(entry, dict):
            return
        finding_id = entry.get("finding_id") or entry.get("id")
        if isinstance(finding_id, str) and finding_id:
            ids.add(finding_id)

    cumulative = manifest.get("cumulative_top")
    if isinstance(cumulative, list):
        for item in cumulative:
            add_entry(item)
    generations = manifest.get("generations")
    if isinstance(generations, dict):
        for value in generations.values():
            if isinstance(value, list):
                for item in value:
                    add_entry(item)
    return ids


def _research_memory_ledger_entry_count(run_dir: Path) -> int:
    count = 0
    ledger_dir = run_dir / "research_memory" / "ledgers"
    if not ledger_dir.exists():
        return 0
    for path in sorted(ledger_dir.glob("*.yaml")):
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(value, dict):
            continue
        entries = value.get("entries")
        if isinstance(entries, list):
            count += sum(1 for item in entries if isinstance(item, dict))
    return count


def _legacy_graph_artifact_paths(run_dir: Path) -> list[Path]:
    graph_dir = run_dir / "graph"
    return [
        graph_dir / name
        for name in (
            "graph_health.json",
            "unlinked_recent_findings.json",
            "graph.html",
            "graph_live.html",
        )
    ]


def _record_ids(records: list[dict[str, Any]], key: str) -> set[str]:
    return {
        str(record.get(key))
        for record in records
        if isinstance(record.get(key), str) and record.get(key)
    }


def _artifact_type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        artifact_type = record.get("artifact_type")
        if isinstance(artifact_type, str) and artifact_type:
            counts[artifact_type] = counts.get(artifact_type, 0) + 1
    return counts


def dry_run(
    run_dir: Path,
    *,
    strict_tail: bool = False,
    allow_plugin_drift: bool = False,
    locked: bool = False,
) -> dict[str, Any]:
    """Return a non-mutating replay plan that summarizes what verify would inspect."""
    report = verify_run(
        run_dir,
        strict_tail=strict_tail,
        allow_plugin_drift=allow_plugin_drift,
        locked=locked,
    )
    summary = report["summary"]
    if summary.get("findings", 0) < 3:
        report["success"] = False
        report.setdefault("errors", []).append("dry-run expected at least 3 fake findings")
    if summary.get("frontier_records", 0) < 1:
        report["success"] = False
        report.setdefault("errors", []).append("dry-run expected at least 1 frontier record")
    report["mode"] = "dry-run"
    write_json(run_dir / "replay" / "replay_report.json", report)
    return report


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{path.name}:json_decode:{exc.msg}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.name}:not_object")
        return None
    return value
