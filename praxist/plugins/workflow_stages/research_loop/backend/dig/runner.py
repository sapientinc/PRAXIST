"""DIG-Lite planner runner for research-loop peers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

from .config import DIGLiteConfig
from .prompts import (
    build_baseline_map_prompt,
    build_candidate_generation_prompt,
    build_candidate_review_prompt,
    build_contract_prompt,
)
from .schema import (
    BaselineMechanismMap,
    CandidatePool,
    CandidateReviews,
    SelectedContract,
)
from .selection import select_quality_diverse_candidate
from .validator import (
    DIGValidationContext,
    validate_candidate_pool,
    validate_reviews,
    validate_selected_contract,
    validate_selected_contract_matches_candidate,
)

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```\s*(?:yaml|yml|json)?\s*\n(.*?)\n\s*```", re.DOTALL | re.I)
_READ_ONLY_PLANNER_TOOLS = ("Read", "Grep", "Glob")
_READ_ONLY_PLANNER_TOOL_SET = set(_READ_ONLY_PLANNER_TOOLS)

_EXPECTED_TOP_LEVEL_KEYS = {
    "baseline_map": ("task_objective", "baseline_core_path", "intervention_surfaces"),
    "candidate_pool": ("candidates",),
    "candidate_reviews": ("reviews",),
    "selected_contract": (
        "variant_name",
        "diversity_cell",
        "mechanism_hypothesis",
    ),
}


def _dig_output_schema(label: str) -> dict[str, Any] | None:
    """Return a permissive JSON schema for one DIG planner phase."""

    object_schema: dict[str, Any] = {"type": "object", "additionalProperties": True}
    string_array = {"type": "array", "items": {"type": "string"}}
    if label == "baseline_map":
        return {
            "type": "object",
            "required": list(_EXPECTED_TOP_LEVEL_KEYS["baseline_map"]),
            "additionalProperties": True,
            "properties": {
                "task_objective": object_schema,
                "baseline_core_path": {"type": "array", "items": object_schema},
                "intervention_surfaces": {"type": "array", "items": object_schema},
                "forbidden_surfaces": {"type": "array", "items": object_schema},
            },
        }
    if label == "candidate_pool":
        return {
            "type": "object",
            "required": list(_EXPECTED_TOP_LEVEL_KEYS["candidate_pool"]),
            "additionalProperties": True,
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "candidate_id",
                            "mechanism_family",
                            "intervention_surface",
                            "intent",
                            "implementation_sketch",
                            "diversity_signature",
                        ],
                        "additionalProperties": True,
                    },
                },
            },
        }
    if label == "candidate_reviews":
        return {
            "type": "object",
            "required": list(_EXPECTED_TOP_LEVEL_KEYS["candidate_reviews"]),
            "additionalProperties": True,
            "properties": {
                "reviews": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["candidate_id", "scores", "fatal_flaws"],
                        "additionalProperties": True,
                    },
                },
            },
        }
    if label == "selected_contract":
        return {
            "type": "object",
            "required": list(_EXPECTED_TOP_LEVEL_KEYS["selected_contract"]),
            "additionalProperties": True,
            "properties": {
                "selected_candidate_id": {"type": "string"},
                "variant_name": {"type": "string"},
                "diversity_cell": object_schema,
                "mechanism_hypothesis": {"type": "string"},
                "why_selected": {"type": "string"},
                "rejected_alternatives": {"type": "array", "items": object_schema},
                "files_to_modify": string_array,
                "allowed_changes": string_array,
                "forbidden_changes": string_array,
                "implementation_plan": {"type": "array", "items": object_schema},
                "expected_metric_signature": object_schema,
                "ablation_hooks": string_array,
                "fail_fast_checks": string_array,
            },
        }
    if label.endswith("_repair"):
        return _dig_output_schema(label[: -len("_repair")])
    return None


def _build_repair_prompt(*, label: str, raw_output: str, error: str) -> str:
    expected_keys = _EXPECTED_TOP_LEVEL_KEYS.get(label, ())
    keys_text = ", ".join(expected_keys) if expected_keys else "the requested top-level keys"
    return (
        "Your previous DIG-Lite planner response could not be parsed as the required "
        f"{label} structured object. Repair the serialization only.\n"
        "Return strict JSON only: no markdown fence, no commentary, no tool plan.\n"
        "Preserve the original meaning and do not invent empirical results. If a "
        "field is missing, use an empty string, empty array, or empty object as "
        "appropriate so downstream validation can decide whether it is acceptable.\n"
        f"The repaired object must include top-level keys: {keys_text}.\n\n"
        f"Parse error:\n{error}\n\n"
        f"Raw response to repair:\n{raw_output}"
    )


def _canonicalize_selected_contract_identity(
    contract_raw: dict[str, Any], selected_candidate: Any
) -> dict[str, Any]:
    """Lock model-authored contract metadata to the deterministic QD selection."""

    aligned = dict(contract_raw)
    signature = selected_candidate.diversity_signature
    aligned["selected_candidate_id"] = selected_candidate.candidate_id
    aligned["diversity_cell"] = {
        "mechanism_family": signature.mechanism_family,
        "intervention_surface": signature.intervention_surface,
        "intent": signature.intent,
    }
    for field in ("semantic_family", "parent_lineage", "novelty_axis"):
        candidate_value = str(getattr(selected_candidate, field, "") or "").strip()
        if candidate_value:
            aligned[field] = candidate_value
    return aligned


@dataclass
class DIGLiteResult:
    """Successful DIG planner output returned to cohort launch code."""

    dig_dir: Path
    selected_contract: SelectedContract
    selected_contract_path: Path
    qd_selection: dict[str, Any]
    candidate_pool: CandidatePool | None = None
    candidate_reviews: CandidateReviews | None = None
    validation_context: DIGValidationContext | None = None

    def to_prompt_context(self, run_dir: Path) -> dict[str, Any]:
        try:
            rel_path = self.selected_contract_path.relative_to(run_dir)
            path_text = str(rel_path)
        except ValueError:
            path_text = str(self.selected_contract_path)
        return {
            "enabled": True,
            "dig_dir": str(self.dig_dir),
            "selected_contract_path": path_text,
            "selected_contract": self.selected_contract.to_dict(),
            "qd_selection": self.qd_selection,
        }


def _strip_yaml_fence(text: str) -> str:
    text = (text or "").strip()
    matches = _FENCE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text


def _looks_like_label_payload(payload: dict[str, Any], label: str) -> bool:
    """Return True when a parsed mapping has the expected DIG top-level shape."""

    expected = _EXPECTED_TOP_LEVEL_KEYS.get(label)
    if not expected:
        return True
    return all(key in payload for key in expected)


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    """Extract JSON object candidates from noisy model text.

    Claude-compatible runtimes may emit plan/status text before the final JSON
    object, or several assistant text blocks may be concatenated together.  The
    DIG validator remains strict, but the parser should first recover the final
    structured object from that transport noise.
    """

    candidates: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
    return candidates


def _yaml_mapping_candidates(text: str) -> list[dict[str, Any]]:
    """Extract simple YAML mapping candidates from fenced/noisy text."""

    candidates: list[dict[str, Any]] = []
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        blocks = [text]
    for block in blocks:
        cleaned = block.strip()
        if not cleaned:
            continue
        try:
            parsed = yaml.safe_load(cleaned)
        except yaml.YAMLError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
    return candidates


def _parse_yaml_mapping(text: str, *, label: str) -> dict[str, Any]:
    raw_text = text or ""
    cleaned = _strip_yaml_fence(raw_text)
    try:
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed_json = None
    if isinstance(parsed_json, dict) and _looks_like_label_payload(parsed_json, label):
        return parsed_json

    for candidate in reversed(_json_object_candidates(raw_text)):
        if _looks_like_label_payload(candidate, label):
            return candidate

    for candidate in reversed(_yaml_mapping_candidates(raw_text)):
        if _looks_like_label_payload(candidate, label):
            return candidate

    try:
        parsed = yaml.safe_load(cleaned)
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} output was not valid JSON/YAML: {exc}") from exc
    if not isinstance(parsed, dict) or not _looks_like_label_payload(parsed, label):
        raise ValueError(f"{label} output must be a JSON/YAML mapping")
    return parsed


def _agent_text_output(output: dict[str, Any]) -> str:
    if not isinstance(output, dict):
        return ""
    text_outputs = output.get("text_outputs")
    if isinstance(text_outputs, list):
        return "\n\n".join(str(item) for item in text_outputs if item is not None)
    text = output.get("text")
    if isinstance(text, str):
        return text
    return ""


class DIGPhaseTimeoutError(TimeoutError):
    """Raised when one DIG planner phase exhausts its phase budget."""


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    return payload


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checkpoint_manifest_path(dig_dir: Path) -> Path:
    return dig_dir / "dig_checkpoint_manifest.json"


def _load_checkpoint_manifest(dig_dir: Path) -> dict[str, Any]:
    path = _checkpoint_manifest_path(dig_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _record_checkpoint(dig_dir: Path, *, phase: str, prompt: str, artifact: Path) -> None:
    manifest = _load_checkpoint_manifest(dig_dir)
    phases = manifest.get("phases")
    if not isinstance(phases, dict):
        phases = {}
    artifact_bytes = artifact.read_bytes()
    phases[phase] = {
        "prompt_sha256": _sha256_text(prompt),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "artifact_size": len(artifact_bytes),
        "artifact": artifact.name,
        "written_at": time.time(),
    }
    manifest.update({"schema_version": 1, "phases": phases})
    _checkpoint_manifest_path(dig_dir).write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _checkpoint_matches(dig_dir: Path, *, phase: str, prompt: str, artifact: Path) -> bool:
    if not artifact.exists():
        return False
    phases = _load_checkpoint_manifest(dig_dir).get("phases")
    if not isinstance(phases, dict):
        return False
    entry = phases.get(phase)
    if not isinstance(entry, dict):
        return False
    try:
        artifact_bytes = artifact.read_bytes()
    except OSError:
        return False
    return (
        entry.get("artifact") == artifact.name
        and entry.get("prompt_sha256") == _sha256_text(prompt)
        and entry.get("artifact_sha256") == hashlib.sha256(artifact_bytes).hexdigest()
        and entry.get("artifact_size") == len(artifact_bytes)
    )


def _write_stage_status(
    dig_dir: Path,
    *,
    phase: str,
    status: str,
    attempt: Any = None,
    duration_seconds: float | None = None,
    reused: bool = False,
    detail: str = "",
) -> None:
    path = dig_dir / "dig_stage_status.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        existing = {}
    events = existing.get("events")
    if not isinstance(events, list):
        events = []
    event: dict[str, Any] = {
        "phase": phase,
        "status": status,
        "reused": reused,
        "written_at": time.time(),
    }
    if attempt is not None:
        event["attempt"] = attempt
    if duration_seconds is not None:
        event["duration_seconds"] = round(float(duration_seconds), 3)
    if detail:
        event["detail"] = detail
    events.append(event)
    existing.update(
        {
            "last_phase": phase,
            "last_status": status,
            "events": events[-80:],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, default=str) + "\n", encoding="utf-8")


def _clear_final_artifacts(dig_dir: Path) -> None:
    """Remove final implementation-unlocking artifacts before a fresh attempt."""

    for name in ("qd_selection.yaml", "selected_contract.yaml", "dig_summary.md"):
        path = dig_dir / name
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("could not clear stale DIG final artifact %s: %s", path, exc)


def _write_summary(
    path: Path,
    *,
    baseline_map: BaselineMechanismMap,
    candidate_pool: CandidatePool,
    reviews: CandidateReviews,
    qd_selection: dict[str, Any],
    selected_contract: SelectedContract,
) -> None:
    lines = [
        "# DIG-Lite Summary",
        "",
        f"- Selected variant: `{selected_contract.variant_name}`",
        (
            "- Diversity cell: "
            f"`{selected_contract.diversity_cell.mechanism_family}` / "
            f"`{selected_contract.diversity_cell.intervention_surface}` / "
            f"`{selected_contract.diversity_cell.intent}`"
        ),
        f"- Candidate count: {len(candidate_pool.candidates)}",
        f"- Review count: {len(reviews.reviews)}",
        f"- Selected candidate id: `{qd_selection.get('selected_candidate_id', '')}`",
        "",
        "## Mechanism Hypothesis",
        "",
        selected_contract.mechanism_hypothesis,
        "",
        "## Expected Metric Signature",
        "",
        f"- Primary: {selected_contract.expected_metric_signature.primary}",
        f"- Secondary or safety: {selected_contract.expected_metric_signature.secondary_or_safety}",
        f"- Diagnostic: {selected_contract.expected_metric_signature.diagnostic}",
        "",
        "## Baseline Map Coverage",
        "",
        f"- Baseline core paths: {len(baseline_map.baseline_core_path)}",
        f"- Intervention surfaces: {len(baseline_map.intervention_surfaces)}",
        f"- Forbidden surfaces: {len(baseline_map.forbidden_surfaces)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_only_planner_tools(config: DIGLiteConfig) -> list[str]:
    """Return the safe planner tool subset, ignoring permissive task config."""

    requested = [str(tool) for tool in config.planner_allowed_tools or []]
    safe = [tool for tool in dict.fromkeys(requested) if tool in _READ_ONLY_PLANNER_TOOL_SET]
    return safe or list(_READ_ONLY_PLANNER_TOOLS)


def _peer_lane_from_context(ctx: dict[str, Any]) -> dict[str, Any]:
    peer_id = str(ctx.get("peer_id") or "")
    agenda = ctx.get("research_agenda")
    if not isinstance(agenda, dict):
        return {}
    contracts = agenda.get("peer_contracts")
    if not isinstance(contracts, dict):
        return {}
    contract = contracts.get(peer_id)
    if not isinstance(contract, dict):
        return {}
    lane: dict[str, Any] = {}
    for src_key, dst_key in (
        ("mechanism_family_preferences", "mechanism_family_preferences"),
        ("intervention_surface_preferences", "intervention_surface_preferences"),
        ("intent_preference", "intent_preference"),
    ):
        if src_key in contract:
            lane[dst_key] = contract[src_key]
    role = str(contract.get("role") or "").strip()
    if role and "intent_preference" not in lane:
        role_to_intent = {
            "exploit": "exploit",
            "falsifier": "falsify",
            "bridge": "bridge",
            "anti_mainline": "anti_mainline",
        }
        if role in role_to_intent:
            lane["intent_preference"] = role_to_intent[role]
    return lane


def _known_signatures_from_context(ctx: dict[str, Any]) -> set[tuple[str, str, str]]:
    signatures: set[tuple[str, str, str]] = set()

    def add_from_obj(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        cell = obj.get("diversity_cell")
        if isinstance(cell, dict):
            triplet = (
                str(cell.get("mechanism_family") or ""),
                str(cell.get("intervention_surface") or ""),
                str(cell.get("intent") or ""),
            )
            if all(triplet):
                signatures.add(triplet)
        triplet = (
            str(obj.get("mechanism_family") or ""),
            str(obj.get("intervention_surface") or ""),
            str(obj.get("intent") or ""),
        )
        if all(triplet):
            signatures.add(triplet)

    for entry in ctx.get("frontier_summary") or []:
        add_from_obj(entry)
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        add_from_obj(metrics)
    for entry in ctx.get("validation_candidates") or []:
        add_from_obj(entry)
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        add_from_obj(metrics)

    gems = (
        (ctx.get("gems_context") or {}).get("gems")
        if isinstance(ctx.get("gems_context"), dict)
        else []
    )
    for gem in gems or []:
        add_from_obj(gem)

    agenda = ctx.get("research_agenda")
    peer_id = str(ctx.get("peer_id") or "")
    contracts = agenda.get("peer_contracts") if isinstance(agenda, dict) else {}
    if isinstance(contracts, dict):
        for other_peer, contract in contracts.items():
            if other_peer == peer_id:
                continue
            add_from_obj(contract)
    sibling_roster = agenda.get("sibling_roster") if isinstance(agenda, dict) else []
    if isinstance(sibling_roster, list):
        for sibling in sibling_roster:
            add_from_obj(sibling)
    return signatures


def _known_texts_from_context(ctx: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    def add_text(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            texts.append(text)

    for entry in ctx.get("frontier_summary") or []:
        if not isinstance(entry, dict):
            continue
        add_text(entry.get("variant_name"))
        add_text(entry.get("title"))
        add_text(entry.get("content"))
    for entry in ctx.get("validation_candidates") or []:
        if not isinstance(entry, dict):
            continue
        add_text(entry.get("variant_name"))
        add_text(entry.get("title"))
        add_text(entry.get("content"))
        add_text(entry.get("mechanism_hypothesis"))
        add_text(entry.get("recommended_next_step"))
    gems_context = ctx.get("gems_context") or {}
    if isinstance(gems_context, dict):
        for gem in gems_context.get("gems") or []:
            if isinstance(gem, dict):
                add_text(gem.get("variant_name"))
                add_text(gem.get("mechanism_hypothesis"))
    agenda = ctx.get("research_agenda")
    contracts = agenda.get("peer_contracts") if isinstance(agenda, dict) else {}
    if isinstance(contracts, dict):
        for contract in contracts.values():
            if isinstance(contract, dict):
                add_text(contract.get("target_hypothesis"))
                add_text(contract.get("success_signal"))
    sibling_roster = agenda.get("sibling_roster") if isinstance(agenda, dict) else []
    if isinstance(sibling_roster, list):
        for sibling in sibling_roster:
            if isinstance(sibling, dict):
                add_text(sibling.get("target_hypothesis"))
                add_text(sibling.get("mechanism_family"))
                add_text(sibling.get("intervention_surface"))
    return texts


def build_validation_context(ctx: dict[str, Any], config: DIGLiteConfig) -> DIGValidationContext:
    """Build validator context from prompt context, lanes, and known mechanisms."""

    disallowed = [
        "evaluator",
        "metrics",
        "data_split",
        "split",
    ]
    disallowed.extend(config.disallowed_file_rules)
    task_spec = ctx.get("task_spec")
    raw = getattr(task_spec, "_raw", {}) if task_spec is not None else {}
    if isinstance(raw, dict):
        for key in ("disallowed_file_rules", "forbidden_file_rules"):
            values = raw.get(key) or []
            if isinstance(values, list):
                disallowed.extend(str(item) for item in values)
        dig_raw = raw.get("dig_lite")
        if isinstance(dig_raw, dict):
            for key in ("disallowed_file_rules", "forbidden_file_rules"):
                values = dig_raw.get(key) or []
                if isinstance(values, list):
                    disallowed.extend(str(item) for item in values)
    return DIGValidationContext(
        peer_lane=_peer_lane_from_context(ctx),
        selection_policy=dict(ctx.get("dig_selection_policy") or {}),
        disallowed_file_rules=disallowed,
        known_diversity_signatures=_known_signatures_from_context(ctx),
        known_mechanism_texts=_known_texts_from_context(ctx),
        duplicate_threshold=config.diversity.duplicate_threshold,
    )


async def _execute_planner_agent(
    *,
    prompt: str,
    label: str,
    agent_factory: Callable[[str], BaseAgent],
    timeout_seconds: float | None,
) -> str:
    agent = agent_factory(label)
    try:
        result = await asyncio.wait_for(agent.execute(prompt), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise DIGPhaseTimeoutError(
            f"DIG planner phase {label} exceeded {timeout_seconds}s timeout"
        ) from exc
    if not result.success:
        raise RuntimeError(f"DIG planner call {label} failed: {result.error or result.output}")
    return _agent_text_output(result.output)


async def _planner_call(
    *,
    prompt: str,
    label: str,
    agent_factory: Callable[[str], BaseAgent],
    timeout_seconds: float | None,
    set_runtime_timeout_seconds: Callable[[float | None], None] | None = None,
) -> dict[str, Any]:
    phase_deadline = (
        time.monotonic() + max(1.0, float(timeout_seconds)) if timeout_seconds is not None else None
    )

    def phase_remaining_seconds() -> float | None:
        if phase_deadline is None:
            return None
        remaining = phase_deadline - time.monotonic()
        if remaining <= 0:
            raise DIGPhaseTimeoutError(
                f"DIG planner phase {label} exhausted its {timeout_seconds}s budget"
            )
        return max(1.0, remaining)

    subcall_timeout = phase_remaining_seconds()
    if set_runtime_timeout_seconds is not None:
        set_runtime_timeout_seconds(subcall_timeout)
    try:
        raw_output = await _execute_planner_agent(
            prompt=prompt,
            label=label,
            agent_factory=agent_factory,
            timeout_seconds=subcall_timeout,
        )
    finally:
        if set_runtime_timeout_seconds is not None:
            set_runtime_timeout_seconds(None)
    try:
        return _parse_yaml_mapping(raw_output, label=label)
    except ValueError as parse_error:
        repair_label = f"{label}_repair"
        repair_timeout = phase_remaining_seconds()
        if set_runtime_timeout_seconds is not None:
            set_runtime_timeout_seconds(repair_timeout)
        try:
            repair_output = await _execute_planner_agent(
                prompt=_build_repair_prompt(
                    label=label,
                    raw_output=raw_output,
                    error=str(parse_error),
                ),
                label=repair_label,
                agent_factory=agent_factory,
                timeout_seconds=repair_timeout,
            )
        finally:
            if set_runtime_timeout_seconds is not None:
                set_runtime_timeout_seconds(None)
        return _parse_yaml_mapping(repair_output, label=label)


async def run_dig_lite(
    *,
    ctx: dict[str, Any],
    config: DIGLiteConfig,
    dig_dir: Path,
    workspace: Path,
    model: str,
    mcp_servers: dict[str, Any],
    plugin_registry: Any | None,
    premium_mode: bool = False,
    agent_factory: Callable[[str], BaseAgent] | None = None,
    quality_diversity_enabled: bool = True,
    reasoning_effort: str = "max",
) -> DIGLiteResult:
    """Run the read-only DIG-Lite planner and persist its artifacts."""

    dig_dir.mkdir(parents=True, exist_ok=True)
    _clear_final_artifacts(dig_dir)
    per_call_timeout_seconds = (
        max(30.0, float(config.planner_max_runtime_minutes) * 60.0)
        if config.planner_max_runtime_minutes is not None
        else None
    )
    total_deadline: float | None = None
    remaining_budget = ctx.get("dig_remaining_budget_seconds")
    try:
        if remaining_budget is not None:
            total_deadline = time.monotonic() + max(1.0, float(remaining_budget))
    except (TypeError, ValueError):
        total_deadline = None

    def remaining_timeout_seconds() -> float | None:
        timeout = per_call_timeout_seconds
        if total_deadline is not None:
            remaining = total_deadline - time.monotonic()
            timeout = min(timeout, remaining) if timeout is not None else remaining
        if timeout is None:
            return None
        if timeout <= 0:
            raise TimeoutError("DIG-Lite total planner budget exhausted.")
        return max(1.0, timeout)

    planner_runtime_ref = str(ctx.get("agent_runtime_ref") or "")
    codex_planner = planner_runtime_ref == "agent_runtime:codex_sdk"
    runtime_timeout_override_seconds: float | None = None

    def set_runtime_timeout_seconds(value: float | None) -> None:
        nonlocal runtime_timeout_override_seconds
        runtime_timeout_override_seconds = value

    def default_agent_factory(label: str) -> BaseAgent:
        runtime_timeout = (
            runtime_timeout_override_seconds
            if runtime_timeout_override_seconds is not None
            else remaining_timeout_seconds()
        )
        return BaseAgent(
            name=f"{ctx.get('peer_id', 'peer')}-dig-{label}",
            execution_role="research",
            allowed_tools=_read_only_planner_tools(config),
            workspace=workspace,
            mcp_servers={},
            model=model,
            permission_mode="default",
            plugin_registry=plugin_registry,
            premium_mode=premium_mode,
            reasoning_effort=reasoning_effort,
            runtime_timeout_seconds=(
                max(1, int(runtime_timeout)) if runtime_timeout is not None else None
            ),
            # claude_sdk can satisfy the stronger no-shell planner contract.
            # codex_sdk's read/search surface is shell-backed, so require the
            # weaker read-only runtime contract instead.
            require_no_shell_runtime=not codex_planner,
            require_read_only_runtime=codex_planner,
            runtime_sandbox_intent={
                "filesystem": "read_only",
                "network": "on",
                "approval": "auto",
            },
            runtime_env_overrides={
                "PRAXIST_PEER_ID": str(ctx.get("peer_id") or ""),
                "PEER_ID": str(ctx.get("peer_id") or ""),
                "GENERATION_ID": str(ctx.get("gen_id") or ""),
            },
            runtime_output_schema=_dig_output_schema(label),
        )

    make_agent = agent_factory or default_agent_factory
    validation_ctx = build_validation_context(ctx, config)

    attempt = ctx.get("dig_attempt")

    baseline_path = dig_dir / "baseline_mechanism_map.yaml"
    baseline_prompt = build_baseline_map_prompt(ctx, config)
    try:
        baseline_raw = _read_yaml_mapping(baseline_path)
        if not _looks_like_label_payload(baseline_raw, "baseline_map") or not _checkpoint_matches(
            dig_dir,
            phase="baseline_map",
            prompt=baseline_prompt,
            artifact=baseline_path,
        ):
            raise ValueError("baseline checkpoint does not match current DIG input")
        baseline_map = BaselineMechanismMap.from_dict(baseline_raw)
        _write_stage_status(
            dig_dir,
            phase="baseline_map",
            status="reused",
            attempt=attempt,
            reused=True,
        )
    except Exception as exc:  # noqa: BLE001 - stale partial artifact should be regenerated.
        if baseline_path.exists():
            logger.info("regenerating invalid DIG baseline map %s: %s", baseline_path, exc)
        started = time.monotonic()
        _write_stage_status(
            dig_dir,
            phase="baseline_map",
            status="started",
            attempt=attempt,
        )
        baseline_map_raw = await _planner_call(
            prompt=baseline_prompt,
            label="baseline_map",
            agent_factory=make_agent,
            timeout_seconds=remaining_timeout_seconds(),
            set_runtime_timeout_seconds=set_runtime_timeout_seconds,
        )
        baseline_map = BaselineMechanismMap.from_dict(baseline_map_raw)
        _write_yaml(baseline_path, baseline_map)
        _record_checkpoint(
            dig_dir,
            phase="baseline_map",
            prompt=baseline_prompt,
            artifact=baseline_path,
        )
        _write_stage_status(
            dig_dir,
            phase="baseline_map",
            status="generated",
            attempt=attempt,
            duration_seconds=time.monotonic() - started,
        )

    candidate_pool_path = dig_dir / "candidate_pool.yaml"
    candidate_pool_prompt = build_candidate_generation_prompt(ctx, baseline_map.to_dict(), config)
    try:
        candidate_pool_raw = _read_yaml_mapping(candidate_pool_path)
        if not _looks_like_label_payload(
            candidate_pool_raw, "candidate_pool"
        ) or not _checkpoint_matches(
            dig_dir,
            phase="candidate_pool",
            prompt=candidate_pool_prompt,
            artifact=candidate_pool_path,
        ):
            raise ValueError("candidate pool checkpoint does not match current DIG input")
        candidate_pool = CandidatePool.from_dict(candidate_pool_raw)
        validate_candidate_pool(candidate_pool, config)
        _write_stage_status(
            dig_dir,
            phase="candidate_pool",
            status="reused",
            attempt=attempt,
            reused=True,
        )
    except Exception as exc:  # noqa: BLE001 - stale partial artifact should be regenerated.
        if candidate_pool_path.exists():
            logger.info("regenerating invalid DIG candidate pool %s: %s", candidate_pool_path, exc)
        started = time.monotonic()
        _write_stage_status(
            dig_dir,
            phase="candidate_pool",
            status="started",
            attempt=attempt,
        )
        candidate_pool_raw = await _planner_call(
            prompt=candidate_pool_prompt,
            label="candidate_pool",
            agent_factory=make_agent,
            timeout_seconds=remaining_timeout_seconds(),
            set_runtime_timeout_seconds=set_runtime_timeout_seconds,
        )
        candidate_pool = CandidatePool.from_dict(candidate_pool_raw)
        validate_candidate_pool(candidate_pool, config)
        _write_yaml(candidate_pool_path, candidate_pool)
        _record_checkpoint(
            dig_dir,
            phase="candidate_pool",
            prompt=candidate_pool_prompt,
            artifact=candidate_pool_path,
        )
        _write_stage_status(
            dig_dir,
            phase="candidate_pool",
            status="generated",
            attempt=attempt,
            duration_seconds=time.monotonic() - started,
        )

    reviews_path = dig_dir / "candidate_reviews.yaml"
    reviews_prompt = build_candidate_review_prompt(
        ctx,
        baseline_map.to_dict(),
        candidate_pool.to_dict(),
        config,
    )
    try:
        reviews_raw = _read_yaml_mapping(reviews_path)
        if not _looks_like_label_payload(
            reviews_raw, "candidate_reviews"
        ) or not _checkpoint_matches(
            dig_dir,
            phase="candidate_reviews",
            prompt=reviews_prompt,
            artifact=reviews_path,
        ):
            raise ValueError("candidate reviews checkpoint does not match current DIG input")
        reviews = CandidateReviews.from_dict(reviews_raw)
        validate_reviews(candidate_pool, reviews)
        _write_stage_status(
            dig_dir,
            phase="candidate_reviews",
            status="reused",
            attempt=attempt,
            reused=True,
        )
    except Exception as exc:  # noqa: BLE001 - stale partial artifact should be regenerated.
        if reviews_path.exists():
            logger.info("regenerating invalid DIG candidate reviews %s: %s", reviews_path, exc)
        started = time.monotonic()
        _write_stage_status(
            dig_dir,
            phase="candidate_reviews",
            status="started",
            attempt=attempt,
        )
        reviews_raw = await _planner_call(
            prompt=reviews_prompt,
            label="candidate_reviews",
            agent_factory=make_agent,
            timeout_seconds=remaining_timeout_seconds(),
            set_runtime_timeout_seconds=set_runtime_timeout_seconds,
        )
        reviews = CandidateReviews.from_dict(reviews_raw)
        validate_reviews(candidate_pool, reviews)
        _write_yaml(reviews_path, reviews)
        _record_checkpoint(
            dig_dir,
            phase="candidate_reviews",
            prompt=reviews_prompt,
            artifact=reviews_path,
        )
        _write_stage_status(
            dig_dir,
            phase="candidate_reviews",
            status="generated",
            attempt=attempt,
            duration_seconds=time.monotonic() - started,
        )

    selected_candidate, _, qd_selection = select_quality_diverse_candidate(
        candidate_pool,
        reviews,
        validation_ctx,
        config,
        quality_diversity_enabled=quality_diversity_enabled,
    )
    qd_payload = qd_selection.to_dict()
    _write_yaml(dig_dir / "qd_selection.yaml", qd_payload)
    _write_stage_status(
        dig_dir,
        phase="qd_selection",
        status="generated",
        attempt=attempt,
    )
    contract_validation_ctx = DIGValidationContext(
        peer_lane=validation_ctx.peer_lane,
        selection_policy=validation_ctx.selection_policy,
        allow_adjacent_lane_selected=not bool(qd_payload.get("eligible_candidates"))
        or not any(
            item.get("candidate_id") == qd_payload.get("selected_candidate_id")
            and bool(item.get("lane_fit"))
            for item in qd_payload.get("eligible_candidates", [])
            if isinstance(item, dict)
        ),
        disallowed_file_rules=validation_ctx.disallowed_file_rules,
        known_diversity_signatures=validation_ctx.known_diversity_signatures,
        known_mechanism_texts=validation_ctx.known_mechanism_texts,
        duplicate_threshold=validation_ctx.duplicate_threshold,
    )

    started = time.monotonic()
    _write_stage_status(
        dig_dir,
        phase="selected_contract",
        status="started",
        attempt=attempt,
    )
    contract_raw = await _planner_call(
        prompt=build_contract_prompt(
            ctx,
            baseline_map.to_dict(),
            candidate_pool.to_dict(),
            reviews.to_dict(),
            qd_payload,
            config,
        ),
        label="selected_contract",
        agent_factory=make_agent,
        timeout_seconds=remaining_timeout_seconds(),
        set_runtime_timeout_seconds=set_runtime_timeout_seconds,
    )
    contract_raw = _canonicalize_selected_contract_identity(contract_raw, selected_candidate)
    selected_contract = SelectedContract.from_dict(contract_raw)
    validate_selected_contract(
        selected_contract,
        contract_validation_ctx,
        config,
        quality_diversity_enabled=quality_diversity_enabled,
    )
    validate_selected_contract_matches_candidate(selected_contract, selected_candidate)
    selected_contract_path = dig_dir / "selected_contract.yaml"
    _write_yaml(selected_contract_path, selected_contract)
    _write_stage_status(
        dig_dir,
        phase="selected_contract",
        status="generated",
        attempt=attempt,
        duration_seconds=time.monotonic() - started,
    )
    _write_summary(
        dig_dir / "dig_summary.md",
        baseline_map=baseline_map,
        candidate_pool=candidate_pool,
        reviews=reviews,
        qd_selection=qd_payload,
        selected_contract=selected_contract,
    )
    try:
        (dig_dir / "dig_failure_summary.json").unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("could not clear stale DIG failure summary %s: %s", dig_dir, exc)

    logger.info(
        "DIG-Lite selected %s for %s (%s/%s/%s)",
        selected_contract.variant_name,
        ctx.get("peer_id"),
        selected_contract.diversity_cell.mechanism_family,
        selected_contract.diversity_cell.intervention_surface,
        selected_contract.diversity_cell.intent,
    )
    return DIGLiteResult(
        dig_dir=dig_dir,
        selected_contract=selected_contract,
        selected_contract_path=selected_contract_path,
        qd_selection=qd_payload,
        candidate_pool=candidate_pool,
        candidate_reviews=reviews,
        validation_context=contract_validation_ctx,
    )
