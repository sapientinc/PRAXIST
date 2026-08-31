"""Legacy output materialization for research-loop finalization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from praxist.core.storage import ArtifactWriter, rewrite_jsonl, utc_now
from praxist.core.trajectory import TrajectoryWriter
from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    is_committed_runtime_fact_source,
)
from praxist.plugins.workflow_stages.research_loop.backend.scoreless import is_scoreless
from praxist.plugins.workflow_stages.research_loop.c5_materializer import (
    materialize_legacy_c5_views,
)
from praxist.plugins.workflow_stages.research_loop.scoreless_materializer import (
    archive_committed_scoreless_sources,
    collect_committed_scoreless_sources,
)


def _materialize_legacy_outputs(prepared: Any, result: dict[str, Any]) -> dict[str, int]:
    scoreless = is_scoreless(prepared.task_spec)
    snapshots = []
    if scoreless:
        findings, snapshots = collect_committed_scoreless_sources(prepared.run_dir, result)
        frontier_summary = []
        gems_summary = []
    else:
        findings = _collect_legacy_findings(prepared.run_dir)
        frontier_summary = _collect_legacy_frontier_summary(prepared.run_dir, result)
        gems_summary = _collect_legacy_gems(prepared.run_dir, result)
    canonical_findings: dict[str, dict[str, Any]] = {}
    agent_events = _legacy_agent_events(prepared.run_dir)
    trajectory = TrajectoryWriter(prepared.run_dir, prepared.run_id)
    artifacts = ArtifactWriter(prepared.run_dir, trajectory)
    archive_committed_scoreless_sources(artifacts, snapshots)
    finding_artifacts: dict[str, dict[str, Any]] = {}
    for finding in findings:
        finding_id = _finding_id(finding)
        if not finding_id:
            continue
        source_event_ids, provenance_quality, provenance_warning = (
            _source_event_ids_for_finding_or_import(
                trajectory,
                agent_events,
                finding,
                reason="legacy_finding_without_strong_agent_provenance",
            )
        )
        artifact = artifacts.persist_json(
            "finding",
            f"findings/legacy/{finding_id}.json",
            {
                "legacy_finding": finding,
                "finding_id": finding_id,
                "provenance_quality": provenance_quality,
                "provenance_warning": provenance_warning,
            },
            schema_ref="core:legacy_finding.v1",
            producer={"stage_id": "research_loop", "role_ref": prepared.peer_role_ref},
            source_event_ids=source_event_ids,
        )
        finding_artifacts[finding_id] = artifact
        canonical_findings[finding_id] = _canonical_finding_record(
            prepared,
            finding,
            finding_id,
            artifact,
            source_event_ids,
            provenance_quality=provenance_quality,
            provenance_warning=provenance_warning,
        )
    for entry in frontier_summary:
        finding_id = _frontier_finding_id(entry)
        if finding_id and finding_id not in canonical_findings:
            finding = _finding_from_frontier_entry(entry)
            source_event_ids, provenance_quality, provenance_warning = (
                _source_event_ids_for_finding_or_import(
                    trajectory,
                    agent_events,
                    finding,
                    reason="legacy_frontier_without_strong_agent_provenance",
                )
            )
            artifact = artifacts.persist_json(
                "finding",
                f"findings/legacy/{finding_id}.json",
                {
                    "legacy_finding": finding,
                    "finding_id": finding_id,
                    "provenance_quality": provenance_quality,
                    "provenance_warning": provenance_warning,
                },
                schema_ref="core:legacy_finding.v1",
                producer={"stage_id": "research_loop", "role_ref": prepared.peer_role_ref},
                source_event_ids=source_event_ids,
            )
            finding_artifacts[finding_id] = artifact
            canonical_findings[finding_id] = _canonical_finding_record(
                prepared,
                finding,
                finding_id,
                artifact,
                source_event_ids,
                provenance_quality=provenance_quality,
                provenance_warning=provenance_warning,
            )
    for gem in gems_summary:
        finding_id = str(gem.get("gem_finding_id") or gem.get("finding_id") or "")
        if not finding_id or finding_id in canonical_findings:
            continue
        finding = _finding_from_gem_entry(gem)
        source_event_ids, provenance_quality, provenance_warning = (
            _source_event_ids_for_finding_or_import(
                trajectory,
                agent_events,
                finding,
                reason="legacy_gem_without_shared_finding_file",
            )
        )
        artifact = artifacts.persist_json(
            "finding",
            f"findings/legacy/{finding_id}.json",
            {
                "legacy_finding": finding,
                "finding_id": finding_id,
                "provenance_quality": provenance_quality,
                "provenance_warning": provenance_warning,
            },
            schema_ref="core:legacy_finding.v1",
            producer={"stage_id": "research_loop", "role_ref": prepared.peer_role_ref},
            source_event_ids=source_event_ids,
        )
        finding_artifacts[finding_id] = artifact
        canonical_findings[finding_id] = _canonical_finding_record(
            prepared,
            finding,
            finding_id,
            artifact,
            source_event_ids,
            provenance_quality=provenance_quality,
            provenance_warning=provenance_warning,
        )

    _rewrite_jsonl(
        prepared.run_dir / "findings" / "findings.jsonl",
        sorted(canonical_findings.values(), key=lambda item: str(item.get("finding_id", ""))),
    )
    frontier_records = []
    for index, entry in enumerate(frontier_summary, start=1):
        finding_id = _frontier_finding_id(entry)
        if not finding_id:
            continue
        source_artifact_ids = []
        finding_artifact = finding_artifacts.get(finding_id)
        if finding_artifact:
            source_artifact_ids.append(finding_artifact["artifact_id"])
        finding_record = canonical_findings.get(finding_id)
        source_event_ids = (
            list(finding_record.get("source_event_ids") or [])
            if isinstance(finding_record, dict)
            else []
        )
        provenance_quality = (
            str(finding_record.get("provenance_quality") or "agent")
            if isinstance(finding_record, dict)
            else "agent"
        )
        provenance_warning = (
            str(finding_record.get("provenance_warning") or "")
            if isinstance(finding_record, dict) and finding_record.get("provenance_warning")
            else None
        )
        if not source_event_ids:
            source_event_ids, provenance_quality, provenance_warning = (
                _source_event_ids_for_finding_or_import(
                    trajectory,
                    agent_events,
                    _finding_from_frontier_entry(entry),
                    reason="legacy_frontier_without_any_materialized_finding_provenance",
                )
            )
        frontier_artifact = artifacts.persist_json(
            "frontier_record",
            f"frontier/legacy/frontier_{index:06d}.json",
            {
                "legacy_frontier_entry": entry,
                "finding_id": finding_id,
                "provenance_quality": provenance_quality,
                "provenance_warning": provenance_warning,
            },
            schema_ref="core:legacy_frontier.v1",
            producer={"stage_id": "research_loop", "role_ref": "workflow_stage:research_loop"},
            source_event_ids=source_event_ids,
            source_artifact_ids=source_artifact_ids,
        )
        frontier_records.append(
            _canonical_frontier_record(
                prepared,
                entry,
                index,
                frontier_artifact,
                source_event_ids,
                source_artifact_ids,
                provenance_quality=provenance_quality,
                provenance_warning=provenance_warning,
            )
        )
    _rewrite_jsonl(prepared.run_dir / "findings" / "frontier.jsonl", frontier_records)
    _rewrite_jsonl(
        prepared.run_dir / "findings" / "gems.jsonl",
        [
            _canonical_gem_record(
                prepared,
                gem,
                canonical_findings.get(str(gem.get("gem_finding_id") or "")),
            )
            for gem in gems_summary
            if isinstance(gem, dict) and (gem.get("gem_finding_id") or gem.get("finding_id"))
        ],
    )
    c5_counts = materialize_legacy_c5_views(
        prepared,
        result,
        trajectory=trajectory,
        artifacts=artifacts,
    )
    return {
        "finding_count": len(canonical_findings),
        "frontier_count": len(frontier_records),
        "gems_count": len(gems_summary),
        **c5_counts,
    }


def _collect_legacy_findings(run_dir: Path) -> list[dict[str, Any]]:
    findings: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "shared_findings").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        finding_id = _finding_id(value)
        if finding_id:
            findings[finding_id] = value
    previous_store = None
    try:
        previous_store = os.environ.get("LOCAL_STORE_DIR")
        os.environ["LOCAL_STORE_DIR"] = str(run_dir)
        from praxist.plugins.workflow_stages.research_loop.backend.tools.local_store import (
            get_all_findings,
            init_db,
        )

        init_db()
        for value in get_all_findings():
            if not isinstance(value, dict):
                continue
            finding_id = _finding_id(value)
            if finding_id:
                findings[finding_id] = value
    except Exception:
        pass
    finally:
        if previous_store is None:
            os.environ.pop("LOCAL_STORE_DIR", None)
        else:
            os.environ["LOCAL_STORE_DIR"] = previous_store
    return list(findings.values())


def _collect_legacy_frontier_summary(run_dir: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    summary = result.get("frontier_summary")
    if isinstance(summary, list):
        return [item for item in summary if isinstance(item, dict)]
    manifest_path = run_dir / "frontier" / "frontier_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not is_committed_runtime_fact_source(manifest, legacy_ok=True):
        return []
    cumulative = manifest.get("cumulative_top")
    if isinstance(cumulative, list):
        return [item for item in cumulative if isinstance(item, dict)]
    generations = manifest.get("generations")
    if not isinstance(generations, dict):
        return []
    entries: list[dict[str, Any]] = []
    for key in sorted(
        generations, key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value))
    ):
        value = generations.get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
    return entries


def _collect_legacy_gems(run_dir: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    gems: dict[str, dict[str, Any]] = {}

    def _add_many(raw: Any) -> None:
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, dict):
                continue
            gem_id = str(
                item.get("gem_finding_id") or item.get("finding_id") or item.get("id") or ""
            )
            if gem_id:
                gems[gem_id] = dict(item)

    result_gems = result.get("gems") if isinstance(result, dict) else None
    if isinstance(result_gems, dict):
        _add_many(result_gems.get("gems"))
        _add_many(result_gems.get("entries"))
    manifest_path = run_dir / "frontier" / "frontier_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    if isinstance(manifest, dict) and is_committed_runtime_fact_source(manifest, legacy_ok=True):
        manifest_gems = manifest.get("gems")
        if isinstance(manifest_gems, dict):
            _add_many(manifest_gems.get("entries"))
    state_path = run_dir / "gems" / "gems_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    if isinstance(state, dict) and is_committed_runtime_fact_source(state, legacy_ok=True):
        _add_many(state.get("gems"))
    return list(gems.values())


def _finding_from_gem_entry(entry: dict[str, Any]) -> dict[str, Any]:
    finding_id = str(
        entry.get("gem_finding_id") or entry.get("finding_id") or entry.get("id") or ""
    )
    metrics = {
        "is_gem_finding": True,
        "frontier_lane": entry.get("frontier_lane", ""),
        "gem_metric_name": entry.get("metric_name", ""),
        "gem_metric_value": entry.get("metric_value"),
        "source_finding_id": entry.get("source_finding_id", ""),
        "source_generation_id": entry.get("source_generation_id"),
    }
    return {
        "id": finding_id,
        "finding_type": "result",
        "title": f"GEM: {entry.get('variant_name') or finding_id}",
        "content": (
            "Durable Gems finding imported from the Gems state/manifest. "
            f"Variant: {entry.get('variant_name') or finding_id}; "
            f"lane: {entry.get('frontier_lane') or '(unspecified)'}."
        ),
        "metrics": metrics,
        "peer_id": "gems_agent",
        "variant_name": entry.get("variant_name", ""),
        "timestamp": entry.get("admitted_at") or utc_now(),
    }


def _canonical_gem_record(
    prepared: Any,
    gem: dict[str, Any],
    finding_record: dict[str, Any] | None,
) -> dict[str, Any]:
    gem_id = str(gem.get("gem_finding_id") or gem.get("finding_id") or gem.get("id") or "")
    return {
        "schema_version": "praxist.gem.v1",
        "gem_finding_id": gem_id,
        "run_id": prepared.run_id,
        "task_ref": prepared.task_ref,
        "variant_name": str(gem.get("variant_name") or ""),
        "frontier_lane": str(gem.get("frontier_lane") or ""),
        "metric_name": str(gem.get("metric_name") or ""),
        "metric_value": gem.get("metric_value"),
        "source_finding_id": str(gem.get("source_finding_id") or ""),
        "source_generation_id": gem.get("source_generation_id"),
        "gem_variant_ref": str(gem.get("gem_variant_ref") or ""),
        "finding_path": str(gem.get("finding_path") or ""),
        "canonical_finding_ref": finding_record,
        "created_at": str(gem.get("admitted_at") or utc_now()),
    }


def _canonical_finding_record(
    prepared: Any,
    finding: dict[str, Any],
    finding_id: str,
    artifact: dict[str, Any],
    source_event_ids: list[str],
    *,
    provenance_quality: str = "agent",
    provenance_warning: str | None = None,
) -> dict[str, Any]:
    raw_metrics = finding.get("metrics")
    metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
    scores = {str(key): value for key, value in metrics.items() if isinstance(value, (int, float))}
    claim = finding.get("title") or finding.get("content") or finding.get("claim") or finding_id
    record = {
        "schema_version": "praxist.finding.v1",
        "finding_id": finding_id,
        "run_id": prepared.run_id,
        "status": str(finding.get("status") or "draft"),
        "claim": str(claim),
        "task_ref": prepared.task_ref,
        "stage_id": "research_loop",
        "producer_ref": _producer_ref(prepared, finding),
        "evidence_refs": [artifact],
        "metric_refs": [],
        "scores": scores,
        "supersedes": _supersedes(finding),
        "source_event_ids": source_event_ids,
        "created_at": str(finding.get("timestamp") or utc_now()),
        "legacy_payload": _compact_legacy_payload(finding),
    }
    record["provenance_quality"] = provenance_quality
    if is_scoreless(prepared.task_spec):
        record["scores"] = {}
        record["evidence_status"] = "not_scored"
    if provenance_warning:
        record["provenance_warning"] = provenance_warning
    return record


def _canonical_frontier_record(
    prepared: Any,
    entry: dict[str, Any],
    index: int,
    artifact: dict[str, Any],
    source_event_ids: list[str],
    source_artifact_ids: list[str],
    *,
    provenance_quality: str = "agent",
    provenance_warning: str | None = None,
) -> dict[str, Any]:
    finding_id = _frontier_finding_id(entry)
    metric_name = str(entry.get("metric_name") or prepared.task_spec.evaluation.primary_metric)
    metric_value = entry.get("metric_value")
    raw_metrics = entry.get("metrics")
    metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
    if metric_value is None and metric_name in metrics:
        metric_value = metrics.get(metric_name)
    record = {
        "schema_version": "praxist.frontier.v1",
        "frontier_record_id": f"frontier_legacy_{index:06d}",
        "run_id": prepared.run_id,
        "finding_id": finding_id,
        "action": "promoted",
        "baseline_ref": "legacy_frontier",
        "metric_name": metric_name,
        "metric_value": metric_value,
        "promotion_reason": str(
            entry.get("promotion_reason") or "legacy GenerationLoop frontier promotion"
        ),
        "decided_by": "workflow_stage:research_loop",
        "source_event_ids": source_event_ids,
        "source_artifact_ids": source_artifact_ids,
        "artifact_refs": [artifact],
        "created_at": str(entry.get("promoted_at") or utc_now()),
        "legacy_payload": _compact_legacy_payload(entry),
    }
    record["provenance_quality"] = provenance_quality
    if provenance_warning:
        record["provenance_warning"] = provenance_warning
    return record


def _finding_from_frontier_entry(entry: dict[str, Any]) -> dict[str, Any]:
    finding_id = _frontier_finding_id(entry)
    raw_metrics = entry.get("metrics")
    metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
    metric_name = entry.get("metric_name")
    if metric_name and entry.get("metric_value") is not None:
        metrics = {**metrics, str(metric_name): entry.get("metric_value")}
    return {
        "id": finding_id,
        "title": entry.get("variant_name") or finding_id,
        "metrics": metrics,
        "peer_id": entry.get("peer_id") or entry.get("agent_name") or "",
        "variant_name": entry.get("variant_name", ""),
        "timestamp": entry.get("promoted_at") or utc_now(),
    }


def _finding_id(finding: dict[str, Any]) -> str:
    value = finding.get("id") or finding.get("finding_id")
    return str(value) if value else ""


def _frontier_finding_id(entry: dict[str, Any]) -> str:
    value = entry.get("finding_id") or entry.get("id")
    return str(value) if value else ""


def _producer_ref(prepared: Any, finding: dict[str, Any]) -> str:
    peer_id = finding.get("peer_id")
    if peer_id:
        return f"{prepared.peer_role_ref}/{peer_id}"
    role = finding.get("peer_role")
    if role:
        return f"{prepared.peer_role_ref}/{role}"
    return f"{prepared.peer_role_ref}/legacy"


def _supersedes(finding: dict[str, Any]) -> list[str]:
    raw = finding.get("supersedes") or finding.get("updates") or finding.get("retry_of")
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if raw:
        return [str(raw)]
    return []


def _compact_legacy_payload(value: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "id",
        "finding_id",
        "finding_type",
        "title",
        "variant_name",
        "metrics",
        "generation_id",
        "peer_id",
        "peer_role",
        "tier",
        "promotion_eligible",
    )
    return {key: value[key] for key in allowed if key in value}


def _legacy_agent_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "trajectory.jsonl"
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "agent.run_finished":
            event_id = record.get("event_id")
            scope = record.get("scope") or {}
            agent_name = scope.get("agent_name") or scope.get("agent_run_id") or ""
            if isinstance(event_id, str) and event_id:
                events.append(
                    {
                        "event_id": event_id,
                        "agent_name": str(agent_name),
                        "payload": record.get("payload")
                        if isinstance(record.get("payload"), dict)
                        else {},
                    }
                )
    return events


def _source_event_ids_for_finding_or_import(
    trajectory: TrajectoryWriter,
    agent_events: list[dict[str, Any]],
    finding: dict[str, Any],
    *,
    reason: str,
) -> tuple[list[str], str, str | None]:
    source_event_ids = _source_event_ids_for_finding(agent_events, finding)
    if source_event_ids:
        return source_event_ids, "agent", None
    finding_id = _finding_id(finding)
    import_event = trajectory.emit(
        "legacy.output_imported",
        severity="warning",
        scope={"stage_id": "research_loop"},
        actor={"type": "core", "id": "legacy_materializer"},
        payload={
            "finding_id": finding_id,
            "peer_id": str(finding.get("peer_id") or ""),
            "reason": reason,
            "provenance_quality": "legacy_weak",
        },
    )
    return [import_event["event_id"]], "legacy_weak", reason


def _source_event_ids_for_finding(
    agent_events: list[dict[str, Any]], finding: dict[str, Any]
) -> list[str]:
    peer_id = str(finding.get("peer_id") or "")
    if not peer_id:
        return []
    matched = []
    for event in agent_events:
        agent_name = event.get("agent_name", "")
        if (
            agent_name == peer_id or agent_name.startswith(f"{peer_id}-")
        ) and _agent_event_produced_finding(event, finding):
            matched.append(event["event_id"])
    return matched


def _agent_event_produced_finding(event: dict[str, Any], finding: dict[str, Any]) -> bool:
    raw_payload = event.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    for tool_use in _tool_uses_from_agent_payload(payload):
        if _share_finding_tool_input_matches(tool_use, finding):
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


def _rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    rewrite_jsonl(path, records)
