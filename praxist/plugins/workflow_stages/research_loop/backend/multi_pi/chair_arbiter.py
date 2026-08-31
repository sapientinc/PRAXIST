"""Chair arbiter — merges PI memos into a final research_agenda.

Hard constraints (enforced via prompt + post-validator):
  - Cannot read raw history (only memos + shared_core_digest + evidence_index).
  - Cannot introduce new scientific claims; every claim must originate
    from a PI memo or evidence card.
  - Must convert blocking objections to experiments OR archive_reason.
  - Stays within the configured peer_budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.core.role_skills import RoleSkill, load_role_skill
from praxist.plugins.workflow_stages.research_loop.backend.effective_config import (
    has_effective_config_metadata,
)

logger = logging.getLogger(__name__)

# R1#10 fix: more permissive fence matcher — allow optional inline comment
# after the opening fence (common LLM behavior: "```yaml # comment").
_FENCE_RE = re.compile(
    r"```\s*(?:yaml|yml)?\s*(?:#[^\n]*)?\s*\n(.*?)\n\s*```",
    re.DOTALL | re.IGNORECASE,
)
_ANY_FENCE_RE = re.compile(r"```\s*[^\n]*\n(.*?)\n\s*```", re.DOTALL)
# Strip BOM variants. Use explicit unicode escapes to avoid embedding
# raw control characters or null bytes in source.
_BOM_VARIANTS = ("\ufeff", "\ufffe")
# Above this line count we skip the O(N\u00b2) Strategy C in
# `_strip_trailing_prose` (Chair-audit-R3#1 fix).
_BOTH_ENDS_LINE_CAP = 100
_AGENDA_ANCHOR_RE = re.compile(
    r"^\s*(agenda_version|mainline_observation|cross_peer_hypotheses|peer_contracts)\s*:",
    re.MULTILINE,
)
_AGENDA_MARKER_KEYS = frozenset(
    {
        "agenda_version",
        "mainline_observation",
        "cross_peer_hypotheses",
        "bridge_hypothesis",
        "anti_mainline_contract",
        "falsification_contract",
        "peer_contracts",
    }
)
_PANEL_ROLE_ORDER = ("builder", "skeptic", "portfolio", "external_validity")
_PEER_ROLE_ROTATION = ("exploit", "falsifier", "bridge", "anti_mainline", "theorist")
_RESEARCH_METADATA_KEYS = (
    "bottleneck_target",
    "evidence_stage",
    "tradeoff_class",
    "primary_tradeoff",
    "next_step_intent",
    "parent_candidate",
    "parent_usage",
)


def _fallback_research_metadata(
    role: str = "", *, research_loop_mode: str = "metric"
) -> dict[str, str]:
    """Conservative preserve/repair/pivot labels for deterministic fallback agendas."""

    if research_loop_mode == "scoreless":
        return {
            "bottleneck_target": "source_or_explanation_gap",
            "evidence_stage": "open_question",
            "tradeoff_class": "incomplete_evidence",
            "primary_tradeoff": "evidence_vs_research_effort",
            "next_step_intent": "preserve_and_validate",
            "parent_candidate": "",
            "parent_usage": "",
        }
    norm = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
    if norm == "falsifier":
        return {
            "bottleneck_target": "negative_control_or_falsification",
            "evidence_stage": "scout",
            "tradeoff_class": "negative_falsifier",
            "primary_tradeoff": "custom",
            "next_step_intent": "ablate_or_falsify",
            "parent_candidate": "",
            "parent_usage": "falsify",
        }
    if norm == "bridge":
        return {
            "bottleneck_target": "process_or_evidence_gap",
            "evidence_stage": "scout",
            "tradeoff_class": "incomplete_evidence",
            "primary_tradeoff": "custom",
            "next_step_intent": "combine_with_other_mechanism",
            "parent_candidate": "",
            "parent_usage": "compare",
        }
    if norm == "anti_mainline":
        return {
            "bottleneck_target": "process_or_evidence_gap",
            "evidence_stage": "scout",
            "tradeoff_class": "incomplete_evidence",
            "primary_tradeoff": "custom",
            "next_step_intent": "pivot_to_distinct_surface",
            "parent_candidate": "",
            "parent_usage": "none",
        }
    if norm == "theorist":
        return {
            "bottleneck_target": "process_or_evidence_gap",
            "evidence_stage": "smoke",
            "tradeoff_class": "incomplete_evidence",
            "primary_tradeoff": "custom",
            "next_step_intent": "ablate_or_falsify",
            "parent_candidate": "",
            "parent_usage": "none",
        }
    return {
        "bottleneck_target": "process_or_evidence_gap",
        "evidence_stage": "scout",
        "tradeoff_class": "incomplete_evidence",
        "primary_tradeoff": "custom",
        "next_step_intent": "preserve_and_validate",
        "parent_candidate": "",
        "parent_usage": "none",
    }


def _research_metadata_from_source(
    source: Any, role: str = "", *, research_loop_mode: str = "metric"
) -> dict[str, str]:
    """Return fallback metadata with any source-provided labels preserved."""

    metadata = _fallback_research_metadata(role, research_loop_mode=research_loop_mode)
    if not isinstance(source, dict):
        return metadata

    containers: list[dict[str, Any]] = [source]
    for key in ("metrics", "extra"):
        value = source.get(key)
        if not isinstance(value, dict):
            continue
        containers.append(value)
        nested_extra = value.get("extra")
        if isinstance(nested_extra, dict):
            containers.append(nested_extra)

    for key in _RESEARCH_METADATA_KEYS:
        for container in containers:
            value = container.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                metadata[key] = text
                break
    return metadata


def _strip_yaml_fence(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip any BOM variant from start
    for bom in _BOM_VARIANTS:
        if text.startswith(bom):
            text = text[len(bom) :]
            break
    matches = _FENCE_RE.findall(text)
    if matches:
        return matches[-1]
    return text


def _unique_strings(values: list[Any], *, limit: int | None = None) -> list[str]:
    """Return non-empty strings with stable order and duplicate removal."""

    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = (
                value.get("finding_id")
                or value.get("evidence_id")
                or value.get("id")
                or value.get("claim_id")
            )
        text = str(value).strip() if value is not None else ""
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _clip(value: Any, *, limit: int = 360, fallback: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return fallback
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _iter_chair_yaml_candidates(text: str) -> list[str]:
    """Return possible YAML bodies from a Chair response.

    The production failure that motivated this helper was not a bad agenda
    object; it was a response where visible prose reached the parser. Try
    language-specific fences, generic fences, and finally the full text.
    Candidate order is last-fence-first because LLMs sometimes quote the
    schema before emitting the actual document.
    """

    if not isinstance(text, str):
        return []
    candidates: list[str] = []
    for regex in (_FENCE_RE, _ANY_FENCE_RE):
        for match in reversed(list(regex.finditer(text))):
            candidates.append(match.group(1))
    candidates.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


@dataclass(frozen=True)
class _AgendaParseResult:
    agenda: dict[str, Any] | None
    cleaned_text: str
    error: str | None = None


def _parse_chair_agenda_text(text: str) -> _AgendaParseResult:
    """Parse Chair output into an agenda mapping with recovery from prose.

    This is stricter than plain ``yaml.safe_load``: the accepted mapping
    must contain at least one agenda marker key. That prevents a prose
    preamble like ``"I will synthesize: ..."`` from being accepted as a
    one-key YAML mapping.
    """

    import yaml

    first_error: str | None = None
    for candidate in _iter_chair_yaml_candidates(text):
        cleaned = _strip_trailing_prose(candidate)
        try:
            parsed = yaml.safe_load(cleaned) or {}
        except Exception as exc:  # noqa: BLE001 - surface parser details to fallback metadata.
            first_error = first_error or str(exc)
            continue
        if isinstance(parsed, dict) and any(k in parsed for k in _AGENDA_MARKER_KEYS):
            return _AgendaParseResult(agenda=parsed, cleaned_text=cleaned)
        if isinstance(parsed, dict):
            first_error = first_error or "yaml mapping did not contain agenda marker keys"
        else:
            first_error = first_error or f"chair output parsed as {type(parsed).__name__}"
    return _AgendaParseResult(agenda=None, cleaned_text="", error=first_error or "no yaml content")


def _strip_trailing_prose(text: str) -> str:
    """Extract an agenda-like YAML slice from text that may have prose around it.

    Chair output is stricter than ordinary PI memos: a natural-language
    preamble can itself be valid YAML, for example ``"I will do X: Y"``.
    A prior Gen0→Gen1 panel hit exactly that shape: the raw response contained
    a valid agenda after ``agenda_version:``, but the previous parser accepted
    the preamble as the first dict-yielding prefix and validator then saw an
    empty agenda. Therefore this helper only accepts YAML mappings that contain
    agenda marker keys, and it preferentially starts extraction at agenda
    anchors.

    Strategy A — anchored starts, then pop trailing lines: handles
    "Preamble ... agenda_version: ... Closing".
    Strategy B — pop trailing lines from whole text: handles fenced/extracted
    "yaml ... Done." when the YAML body starts at line 0.
    Strategy C — pop leading lines: handles "Preamble ... yaml".
    Strategy D — pop both ends, O(N²): handles "Preamble ... yaml ... Closing".

    Strategy C is gated by `_BOTH_ENDS_LINE_CAP` to avoid pathological
    O(N²) cost on long malformed outputs (Chair-audit-R3#1 fix). Above the
    cap we return the original text and let the caller surface a clean
    "yaml parse" error — same behavior as before this helper existed.

    Returns the original text if nothing parses to a dict.
    """
    import yaml

    if not isinstance(text, str) or not text.strip():
        return text

    def _load_dict(s: str) -> dict[str, Any] | None:
        try:
            parsed = yaml.safe_load(s)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _is_agenda_like(s: str) -> bool:
        parsed = _load_dict(s)
        return parsed is not None and any(k in parsed for k in _AGENDA_MARKER_KEYS)

    stripped = text.lstrip()
    if _AGENDA_ANCHOR_RE.match(stripped) and _is_agenda_like(stripped):
        return stripped
    lines = text.rstrip().split("\n")
    n = len(lines)
    for match in _AGENDA_ANCHOR_RE.finditer(text):
        start_line = text[: match.start()].count("\n")
        for end in range(n, start_line, -1):
            candidate = "\n".join(lines[start_line:end])
            if _is_agenda_like(candidate):
                return candidate.lstrip()
    for end in range(n - 1, 0, -1):
        candidate = "\n".join(lines[:end])
        if _is_agenda_like(candidate):
            return candidate.lstrip()
    for start in range(1, n):
        candidate = "\n".join(lines[start:])
        if _is_agenda_like(candidate):
            return candidate.lstrip()
    if n <= _BOTH_ENDS_LINE_CAP:
        for start in range(1, n):
            for end in range(n, start, -1):
                candidate = "\n".join(lines[start:end])
                if _is_agenda_like(candidate):
                    return candidate.lstrip()
    return text


def _iter_memo_claims(pi_memos: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    claims: list[tuple[str, dict[str, Any]]] = []
    for role in _PANEL_ROLE_ORDER:
        memo = pi_memos.get(role)
        if not isinstance(memo, dict) or memo.get("_pi_unavailable"):
            continue
        for claim in memo.get("top_claims") or []:
            if isinstance(claim, dict) and (claim.get("statement") or claim.get("id")):
                claims.append((role, claim))
    for role, memo in pi_memos.items():
        if role in _PANEL_ROLE_ORDER or not isinstance(memo, dict) or memo.get("_pi_unavailable"):
            continue
        for claim in memo.get("top_claims") or []:
            if isinstance(claim, dict) and (claim.get("statement") or claim.get("id")):
                claims.append((role, claim))
    return claims


def _memo_summary(role: str, memo: dict[str, Any]) -> str:
    claims = [c for c in memo.get("top_claims") or [] if isinstance(c, dict)]
    objections = [o for o in memo.get("objections_or_warnings") or [] if isinstance(o, dict)]
    if claims:
        first = _clip(claims[0].get("statement") or claims[0].get("id"), limit=240)
    else:
        first = f"{role} provided no active top claim."
    if objections:
        objection = _clip(
            objections[0].get("objection") or objections[0].get("resolving_experiment"),
            limit=220,
        )
        return f"{first} Main caution: {objection}"
    return first


def _claim_supports(role: str, claim: dict[str, Any]) -> list[dict[str, str]]:
    raw_refs = []
    for key in ("supports", "source_findings", "source_evidence", "challenges"):
        value = claim.get(key)
        if isinstance(value, list):
            raw_refs.extend(value)
        elif isinstance(value, str):
            raw_refs.append(value)
    refs = _unique_strings(raw_refs, limit=3)
    if not refs:
        claim_id = str(claim.get("id") or "claim")
        refs = [f"PI_MEMO::{role}::{claim_id}"]
    return [{"finding_id": ref, "peer": "pi_panel", "role": role} for ref in refs]


def _claim_minimal_test(
    role: str,
    claim: dict[str, Any],
    pi_memos: dict[str, dict[str, Any]],
    *,
    research_loop_mode: str = "metric",
) -> str:
    claim_id = str(claim.get("id") or "")
    memo = pi_memos.get(role, {}) if isinstance(pi_memos.get(role), dict) else {}
    for objection in memo.get("objections_or_warnings") or []:
        if not isinstance(objection, dict):
            continue
        if claim_id and str(objection.get("target_claim") or "") not in ("", claim_id):
            continue
        experiment = objection.get("resolving_experiment")
        if experiment:
            return _clip(experiment, limit=520)
    for experiment in memo.get("proposed_experiments") or []:
        if isinstance(experiment, dict) and experiment.get("description"):
            return _clip(experiment.get("description"), limit=520)
    if research_loop_mode == "scoreless":
        return (
            f"Inspect the sources and alternative explanations for {claim_id or 'this open inquiry'}; "
            "record what remains uncertain and which next inquiry could resolve it."
        )
    return (
        f"Run the smallest task-valid control or replication that can decide "
        f"{claim_id or 'this PI claim'} under matched resources, then publish "
        "the raw metrics and failure mode."
    )


def _first_objection(pi_memos: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    for role in ("skeptic", "external_validity", "portfolio", "builder"):
        memo = pi_memos.get(role)
        if not isinstance(memo, dict) or memo.get("_pi_unavailable"):
            continue
        for objection in memo.get("objections_or_warnings") or []:
            if isinstance(objection, dict):
                return role, objection
    return None


def _first_bridge_contract(
    pi_memos: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    for role in _PANEL_ROLE_ORDER:
        memo = pi_memos.get(role)
        if not isinstance(memo, dict) or memo.get("_pi_unavailable"):
            continue
        for contract in memo.get("proposed_peer_contracts") or []:
            if not isinstance(contract, dict):
                continue
            if str(contract.get("role") or "").strip().lower().replace("-", "_") == "bridge":
                return role, contract
    return None


def _minority_ideas(
    pi_memos: dict[str, dict[str, Any]],
    cross_reviews: dict[str, dict[str, Any]],
    *,
    next_gen_id: int,
    research_loop_mode: str = "metric",
) -> list[dict[str, Any]]:
    ideas: list[dict[str, Any]] = []
    for role, review in cross_reviews.items():
        if not isinstance(review, dict):
            continue
        idea = review.get("singleton_high_upside_idea_to_preserve")
        if not isinstance(idea, dict):
            continue
        summary = _clip(idea.get("idea_summary"), limit=360)
        if not summary:
            continue
        ideas.append(
            {
                "idea_id": f"M_g{next_gen_id}_{len(ideas) + 1:02d}",
                "originating_pi": role,
                "private_kb_source": _clip(idea.get("source"), limit=120, fallback="round2"),
                "rationale": summary,
                "protected_budget": _clip(
                    idea.get("protected_budget_recommendation"), limit=180, fallback="1 peer"
                ),
                "success_condition": (
                    "Preserve source-backed observations, contradictions, and useful open questions."
                    if research_loop_mode == "scoreless"
                    else "Produce a task-valid result or falsifier finding that changes the next "
                    "frontier or claim boundary."
                ),
                "stop_condition": (
                    "Record unresolved limitations when the allotted inquiry budget is exhausted."
                    if research_loop_mode == "scoreless"
                    else "Retire after one generation if no measurable signal is produced."
                ),
            }
        )
    if ideas:
        return ideas
    portfolio = pi_memos.get("portfolio")
    if isinstance(portfolio, dict):
        for experiment in portfolio.get("proposed_experiments") or []:
            if not isinstance(experiment, dict) or not experiment.get("description"):
                continue
            ideas.append(
                {
                    "idea_id": f"M_g{next_gen_id}_01",
                    "originating_pi": "portfolio",
                    "private_kb_source": _clip(
                        experiment.get("id"), limit=120, fallback="portfolio"
                    ),
                    "rationale": _clip(experiment.get("description"), limit=360),
                    "protected_budget": "1 peer if primary contracts leave capacity",
                    "success_condition": (
                        "Investigate the proposed inquiry and retain useful partial evidence."
                        if research_loop_mode == "scoreless"
                        else "Advance to the next task-valid evaluation tier or produce a clear "
                        "negative finding."
                    ),
                    "stop_condition": (
                        "Record unresolved limitations when the allotted inquiry budget is exhausted."
                        if research_loop_mode == "scoreless"
                        else "Retire after one generation without a measurable signal."
                    ),
                }
            )
            break
    return ideas


def _claim_boundary_updates_from_round2(
    cross_reviews: dict[str, dict[str, Any]],
    *,
    research_loop_mode: str = "metric",
) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for role, review in cross_reviews.items():
        if not isinstance(review, dict):
            continue
        for revision in review.get("own_revisions") or []:
            if not isinstance(revision, dict) or not revision.get("claim_id"):
                continue
            old = _clip(
                revision.get("boundary_old")
                or revision.get("old_language")
                or f"Prior {role} claim boundary",
                limit=420,
            )
            new = _clip(
                revision.get("boundary_new")
                or revision.get("new_language")
                or f"Bounded {role} claim after cross-review",
                limit=420,
            )
            triggered = _clip(revision.get("triggered_by"), limit=260, fallback="Round 2 review")
            updates.append(
                {
                    "claim_id": str(revision["claim_id"]),
                    "old_language": old,
                    "new_language": new,
                    "reason": (
                        "Round 2 self-revision after peer cross-review"
                        f" by {role}; triggered by: {triggered}"
                    ),
                    "required_validation_before_upgrade": [
                        f"Investigate {triggered}; revisit the claim only when cited evidence "
                        "addresses the stated objection."
                        if research_loop_mode == "scoreless"
                        else f"Run the concrete resolving experiment implied by {triggered}; "
                        "upgrade only if the measured result removes the stated objection."
                    ],
                }
            )
    return updates


def _build_scoreless_fallback_agenda(
    *,
    pi_memos: dict[str, dict[str, Any]],
    cross_reviews: dict[str, dict[str, Any]],
    next_gen_id: int,
    completed_gen_id: int,
    panel_mode: str,
    shared_core_id: str,
    peer_budget: int,
    parse_error: str,
    peer_role_rotation: tuple[str, ...],
) -> dict[str, Any]:
    """Preserve memo evidence and allocate inquiries without inventing score gates."""
    claims = _iter_memo_claims(pi_memos)
    metadata = _fallback_research_metadata(research_loop_mode="scoreless")
    boundary_updates = _claim_boundary_updates_from_round2(
        cross_reviews, research_loop_mode="scoreless"
    )
    minority_ideas = _minority_ideas(
        pi_memos, cross_reviews, next_gen_id=next_gen_id, research_loop_mode="scoreless"
    )
    hypotheses = []
    inquiries = (
        "Which available sources support the provisional explanation?",
        "Which contradictory observation or alternative explanation needs investigation?",
        "Which source limitation should the next inquiry resolve?",
    )
    for index in range(max(3, min(5, len(claims)))):
        role, claim = claims[index % len(claims)] if claims else ("panel", {})
        statement = str(claim.get("statement") or claim.get("id") or "")
        is_followup = index >= len(claims)
        hypotheses.append(
            {
                "id": f"H_g{next_gen_id}_{index + 1:02d}",
                "source_claim_id": str(claim.get("id") or ""),
                "claim": (
                    f"Open inquiry: {inquiries[index % len(inquiries)]} {statement}".strip()
                    if is_followup
                    else statement
                ),
                **_research_metadata_from_source(claim, role, research_loop_mode="scoreless"),
                "source_findings": _claim_supports(role, claim) if claim else [],
                "minimal_test": _claim_minimal_test(
                    role, claim, pi_memos, research_loop_mode="scoreless"
                ),
                "kill_condition": "Bound or reject the explanation when cited evidence contradicts it.",
                "promote_condition": "Revisit the provisional claim when cited evidence resolves its limitations.",
            }
        )
        # Peer prompt slicing retains each hypothesis but may omit top-level
        # revisions keyed by the original memo claim rather than its new id.
        hypothesis = hypotheses[-1]
        for field in ("boundary", "confidence"):
            if field in claim:
                hypothesis[field] = claim[field]
        revisions = [
            update
            for update in boundary_updates
            if update["claim_id"] == hypothesis["source_claim_id"]
        ]
        if revisions:
            hypothesis["boundary"] = revisions[-1]["new_language"]
            hypothesis["claim_boundary_updates"] = revisions
        review = cross_reviews.get(role)
        if isinstance(review, dict):
            for revision in review.get("own_revisions") or []:
                if (
                    isinstance(revision, dict)
                    and revision.get("claim_id") == hypothesis["source_claim_id"]
                    and "confidence_new" in revision
                ):
                    hypothesis["confidence"] = revision["confidence_new"]
    first = hypotheses[0]
    objection = _first_objection(pi_memos)
    resolving_inquiry = (
        str(
            objection[1].get("resolving_experiment")
            or objection[1].get("objection")
            or first["minimal_test"]
        )
        if objection
        else first["minimal_test"]
    )
    rotation = peer_role_rotation or _PEER_ROLE_ROTATION
    contracts = {}
    for index in range(peer_budget):
        role = rotation[index % len(rotation)]
        target = hypotheses[index % len(hypotheses)]
        contracts[f"gen{next_gen_id}_peer{index}"] = {
            "role": role,
            "target_hypothesis": target["id"],
            **metadata,
            "source_findings": target["source_findings"],
            "next_inquiry": resolving_inquiry if role == "falsifier" else target["minimal_test"],
            "forbidden_actions": [
                "Do not invent sources or present unresolved claims as established facts."
            ],
            "success_signal": "Publish cited observations, contradictions, or useful partial findings with uncertainty.",
            "coverage_check": "Identify which open question this inquiry addresses and any remaining source gaps.",
            "source": (
                "DISSENT_TO_EXPERIMENT"
                if role == "falsifier"
                else "minority_high_upside"
                if role == "anti_mainline" and minority_ideas
                else "consensus_actions"
            ),
        }
    summaries = {
        f"{role}_summary": _memo_summary(role, memo)
        for role, memo in pi_memos.items()
        if isinstance(memo, dict) and not memo.get("_pi_unavailable")
    }
    summaries["chair_summary"] = (
        "Chair output could not be parsed. Preserve provisional memo claims and dissent; "
        "the assigned inquiries do not imply measured performance or established correctness."
    )
    return {
        "agenda_version": "2.0",
        "research_loop_mode": "scoreless",
        "evidence_status": "not_scored",
        "generation": next_gen_id,
        "synthesized_from_gen": completed_gen_id,
        "synthesized_at": datetime.now(UTC).isoformat(),
        "panel_mode": panel_mode,
        "shared_core_id": shared_core_id,
        "panel_summary": summaries,
        "mainline_observation": {
            "provisional_explanations": [h["claim"] for h in hypotheses],
            "main_risk": resolving_inquiry,
        },
        "cross_peer_hypotheses": hypotheses,
        "bridge_hypothesis": {
            "id": f"B_g{next_gen_id}_01",
            **metadata,
            "source_findings": first["source_findings"] + hypotheses[1]["source_findings"],
            "experiment_matrix": [first["minimal_test"], hypotheses[1]["minimal_test"]],
            "expected_pareto_movement": "Clarify agreement or conflict between provisional explanations.",
            "coverage_check": "Record shared sources and independent evidence before combining explanations.",
        },
        "anti_mainline_contract": {
            **metadata,
            "target_axes": [
                "alternative explanations",
                "contradictory evidence",
                "source limitations",
            ],
            "forbidden_mechanisms": [
                "Treating consensus as correctness",
                "Inventing sources",
                "Discarding unresolved useful evidence",
            ],
            "seed_findings": [entry["finding_id"] for entry in first["source_findings"]],
            "success_condition": "Investigate an alternative explanation and retain its supporting or conflicting evidence.",
        },
        "falsification_contract": {
            "target_hypothesis": first["id"],
            **metadata,
            "required_controls": [resolving_inquiry],
            "decision_rule": "State whether the cited evidence supports, contradicts, or leaves the claim unresolved.",
        },
        "consensus_actions": [
            {
                "action_id": f"A_g{next_gen_id}_01",
                "claim_or_hypothesis": first["id"],
                **metadata,
                "assigned_role": rotation[0],
                "minimal_experiment": first["minimal_test"],
                "supports": [entry["finding_id"] for entry in first["source_findings"]],
                "promote_condition": first["promote_condition"],
                "kill_condition": first["kill_condition"],
            }
        ],
        "DISSENT_TO_EXPERIMENT": [
            {
                "dissent_id": f"D_g{next_gen_id}_01",
                "disputed_claim": str(objection[1].get("target_claim") or first["id"]),
                "resolving_experiment": resolving_inquiry,
                "assigned_peer_role": "falsifier",
                **metadata,
                "decision_rule": "Preserve the objection until cited evidence resolves it; record an unresolved outcome when necessary.",
            }
        ]
        if objection
        else [],
        "minority_high_upside": minority_ideas,
        "claim_boundary_updates": boundary_updates,
        "peer_contracts": contracts,
        "success_metrics": {
            "required": [
                "cite_retained_sources_and_label_uncertainty",
                "preserve_useful_partial_findings",
                "record_contradictions_and_next_inquiries",
            ]
        },
        "dissent_ledger_ref": f"research_memory/dissent_ledger_gen{next_gen_id}.yaml",
        "fallback_metadata": {
            "reason": "chair_yaml_parse_failed",
            "parse_error": _clip(parse_error, limit=500),
        },
    }


def _build_deterministic_fallback_agenda(
    *,
    pi_memos: dict[str, dict[str, Any]],
    cross_reviews: dict[str, dict[str, Any]],
    next_gen_id: int,
    completed_gen_id: int,
    panel_mode: str,
    shared_core_id: str,
    peer_budget: int,
    parse_error: str,
    peer_role_rotation: tuple[str, ...] = (),
    research_loop_mode: str = "metric",
) -> dict[str, Any]:
    """Construct a conservative agenda when Chair emits non-parseable text.

    The fallback is intentionally simple. It does not invent mechanisms; it
    carries forward PI memo claims as hypotheses, routes the strongest
    objection into a falsifier contract, and keeps all peer targets inside
    known agenda IDs so the validator can still audit the generation.
    """

    if research_loop_mode == "scoreless":
        return _build_scoreless_fallback_agenda(
            pi_memos=pi_memos,
            cross_reviews=cross_reviews,
            next_gen_id=next_gen_id,
            completed_gen_id=completed_gen_id,
            panel_mode=panel_mode,
            shared_core_id=shared_core_id,
            peer_budget=peer_budget,
            parse_error=parse_error,
            peer_role_rotation=peer_role_rotation,
        )
    claims: list[tuple[str, dict[str, Any]]] = _iter_memo_claims(pi_memos)
    if not claims:
        claims = [
            (
                "panel",
                {
                    "id": f"C_fallback_g{next_gen_id}_01",
                    "statement": (
                        "The panel did not produce parseable claims; run a conservative "
                        "replication and evidence-gathering generation."
                    ),
                    "supports": [f"PI_PANEL::{completed_gen_id}_to_{next_gen_id}"],
                },
            )
        ]
    while len(claims) < 3:
        role, claim = claims[-1]
        idx = len(claims) + 1
        claims.append(
            (
                role,
                {
                    "id": f"{claim.get('id', 'C_fallback')}_followup_{idx}",
                    "statement": (
                        f"Resolve a separate follow-up uncertainty around {claim.get('id', role)} "
                        "with a matched-control experiment."
                    ),
                    "supports": [claim.get("id", f"PI_MEMO::{role}")],
                    **_research_metadata_from_source(claim, "hypothesis"),
                },
            )
        )

    hypotheses: list[dict[str, Any]] = []
    for idx, (role, claim) in enumerate(claims[: max(3, min(5, len(claims)))], start=1):
        claim_id = str(claim.get("id") or f"C_{role}_{idx}")
        claim_metadata = _research_metadata_from_source(claim, "hypothesis")
        hypotheses.append(
            {
                "id": f"H_g{next_gen_id}_{idx:02d}",
                "source_id": f"{role}:{claim_id}",
                "claim": _clip(
                    claim.get("statement") or claim_id,
                    limit=620,
                    fallback=f"Evaluate PI claim {claim_id}.",
                ),
                **claim_metadata,
                "source_findings": _claim_supports(role, claim),
                "minimal_test": _claim_minimal_test(role, claim, pi_memos),
                "kill_condition": (
                    "Downgrade if the matched control or replication explains the observed "
                    "effect, or if the result fails the task's next valid evaluation gate."
                ),
                "promote_condition": (
                    "Promote only if the claim survives matched controls with reproducible "
                    "task-valid metrics and no unaddressed blocking objection."
                ),
                "inspired_by_private_kb": bool(claim.get("inspired_by_private_kb", False)),
            }
        )

    first_objection = _first_objection(pi_memos)
    if first_objection is not None:
        _, objection = first_objection
        resolving_experiment = _clip(
            objection.get("resolving_experiment")
            or objection.get("objection")
            or "Run the PI-requested falsifier under matched resources.",
            limit=520,
        )
        disputed_claim = str(objection.get("target_claim") or hypotheses[0]["id"])
    else:
        resolving_experiment = "Run a matched-control replication of the highest-confidence claim."
        disputed_claim = hypotheses[0]["id"]

    bridge = _first_bridge_contract(pi_memos)
    if bridge is not None:
        bridge_role, bridge_contract = bridge
        bridge_target = _clip(
            bridge_contract.get("target_hypothesis") or bridge_contract.get("rationale"),
            limit=260,
            fallback=hypotheses[1]["id"],
        )
        bridge_rationale = _clip(
            bridge_contract.get("rationale"), limit=320, fallback="PI-requested bridge contract"
        )
    else:
        bridge_role = "panel"
        bridge_contract = hypotheses[1]
        bridge_target = hypotheses[1]["id"]
        bridge_rationale = (
            "Bridge the two strongest PI-supported hypotheses with explicit coverage reporting."
        )

    active_roles = [
        role
        for role in _PANEL_ROLE_ORDER
        if isinstance(pi_memos.get(role), dict) and not pi_memos[role].get("_pi_unavailable")
    ]
    hypothesis_by_id = {str(h.get("id")): h for h in hypotheses if h.get("id")}
    first_supports = [sf["finding_id"] for sf in hypotheses[0].get("source_findings", [])]
    second_supports = [sf["finding_id"] for sf in hypotheses[1].get("source_findings", [])]
    # Custom panel topologies may supply their own peer-role vocabulary so
    # fallback agendas don't emit role names the task project has no role
    # files for. Unknown role names fall through to the generic ``else``
    # branch below, which still produces a valid (if content-generic)
    # contract. See issue #84.
    rotation = peer_role_rotation or _PEER_ROLE_ROTATION
    peer_contracts: dict[str, dict[str, Any]] = {}
    for i in range(peer_budget):
        role = rotation[i % len(rotation)]
        if role == "bridge":
            target = f"B_g{next_gen_id}_01"
            success = (
                "Publish a bridge result or a clear negative coverage finding with axis "
                "movement recorded."
            )
            source = "fallback_bridge_hypothesis"
            extra = {
                "coverage_check": (
                    "Report which mechanism or metric axis the bridge tests and whether it "
                    "duplicates an already bounded family."
                )
            }
            metadata_source = bridge_contract
        elif role == "anti_mainline":
            target = "anti_mainline_contract"
            success = "Produce a non-mainline result or falsifier that changes portfolio coverage."
            source = "fallback_anti_mainline_contract"
            extra = {}
            metadata_source = {}
        elif role == "falsifier":
            target = hypotheses[0]["id"]
            success = "Resolve the leading blocking objection with a matched-control decision."
            source = "fallback_dissent"
            extra = {}
            metadata_source = first_objection[1] if first_objection is not None else hypotheses[0]
        else:
            target = hypotheses[min(i, len(hypotheses) - 1)]["id"]
            success = (
                "Produce a task-valid result, negative finding, or boundary update grounded "
                "in raw artifacts."
            )
            source = "fallback_hypotheses"
            extra = {}
            metadata_source = hypothesis_by_id.get(target, {})
        peer_contracts[f"gen{next_gen_id}_peer{i}"] = {
            "role": role,
            "target_hypothesis": target,
            **_research_metadata_from_source(metadata_source, role),
            "forbidden_actions": [
                "Do not claim dominance without the matched control requested by the panel.",
                "Do not ignore negative evidence or missing provenance.",
            ],
            "success_signal": success,
            "source": source,
            **extra,
        }

    summaries = {}
    for role in _PANEL_ROLE_ORDER:
        memo = pi_memos.get(role)
        if isinstance(memo, dict) and not memo.get("_pi_unavailable"):
            summaries[f"{role}_summary"] = _memo_summary(role, memo)
    summaries["chair_summary"] = (
        "Deterministic fallback agenda constructed because the Chair runtime emitted "
        "non-parseable YAML. The fallback preserves PI memo claims as hypotheses, "
        "routes the strongest objection to a falsifier, and keeps a bridge plus "
        "anti-mainline slot so the next generation remains scientifically useful."
    )
    success_metric_requirements = (
        [
            "at_least_1_contract_for_each_topology_role",
            "at_least_3_cross_peer_hypotheses_or_actions_tested",
            "at_least_1_non_mainline_or_falsification_attempt_completed",
        ]
        if peer_role_rotation
        else [
            "at_least_3_cross_peer_hypotheses_tested",
            "at_least_1_bridge_experiment_completed",
            "at_least_1_dissent_resolution_peer",
            "at_least_1_anti_mainline_result_or_negative_finding",
        ]
    )

    return {
        "agenda_version": "2.0",
        "generation": next_gen_id,
        "synthesized_from_gen": completed_gen_id,
        "synthesized_at": datetime.now(UTC).isoformat(),
        "panel_mode": panel_mode,
        "shared_core_id": shared_core_id,
        "panel_summary": summaries,
        "mainline_observation": {
            "current_dominant_mechanisms": [
                _clip(claims[0][1].get("statement") or claims[0][1].get("id"), limit=180),
                _clip(claims[1][1].get("statement") or claims[1][1].get("id"), limit=180),
            ],
            "main_risk": resolving_experiment,
            "key_tradeoff": (
                "Balance result-producing peers against the panel's requested matched "
                "controls so apparent frontier movement is not confused with attribution."
            ),
        },
        "cross_peer_hypotheses": hypotheses,
        "bridge_hypothesis": {
            "id": f"B_g{next_gen_id}_01",
            "source_anchor_A": {
                "variant": first_supports[0] if first_supports else hypotheses[0]["source_id"],
                "extracted_mechanism": hypotheses[0]["id"],
            },
            "source_anchor_B": {
                "variant": second_supports[0] if second_supports else hypotheses[1]["source_id"],
                "extracted_mechanism": hypotheses[1]["id"],
            },
            **_research_metadata_from_source(bridge_contract, "bridge"),
            "parent_usage": "compare",
            "experiment_matrix": [bridge_target, bridge_rationale],
            "expected_pareto_movement": (
                "Advance or falsify a PI-requested cross-axis combination under matched controls."
            ),
            "coverage_check": (
                f"Derived from {bridge_role} bridge recommendation; require explicit reporting "
                "of covered metric axes and duplicated mechanism families."
            ),
        },
        "anti_mainline_contract": {
            "forbidden_mechanisms": [
                "minor parameter-only edits to the current frontier without a matched control",
                "reusing the dominant mechanism family without a new falsifiable axis",
                "claiming a result from a single unsupported metric surface",
            ],
            "target_axes": [
                "primary task metric",
                "generalization or stability axis",
                "resource efficiency axis",
            ],
            **_fallback_research_metadata("anti_mainline"),
            "seed_findings": _unique_strings(first_supports + second_supports, limit=4),
            "success_condition": (
                "Find an independently motivated axis movement or publish a negative result "
                "that narrows the search space."
            ),
        },
        "falsification_contract": {
            "target_hypothesis": hypotheses[0]["id"],
            **_research_metadata_from_source(
                first_objection[1] if first_objection is not None else hypotheses[0], "falsifier"
            ),
            "required_controls": [
                resolving_experiment,
                "matched-resource baseline or ablation",
                "fresh validation split, seed pool, or task-equivalent replication when available",
            ],
            "decision_rule": (
                "Keep, downgrade, or kill the target claim according to the matched-control result."
            ),
        },
        "consensus_actions": [
            {
                "action_id": f"A_g{next_gen_id}_01",
                "claim_or_hypothesis": hypotheses[0]["id"],
                "supporting_pis": active_roles or ["panel"],
                "blocking_objections": [disputed_claim] if disputed_claim else [],
                "assigned_role": "exploit",
                **_research_metadata_from_source(hypotheses[0], "exploit"),
                "minimal_experiment": hypotheses[0]["minimal_test"],
                "promote_condition": hypotheses[0]["promote_condition"],
                "kill_condition": hypotheses[0]["kill_condition"],
                "supports": [sf["finding_id"] for sf in hypotheses[0]["source_findings"]],
            }
        ],
        "DISSENT_TO_EXPERIMENT": [
            {
                "dissent_id": f"D_g{next_gen_id}_01",
                "disputed_claim": disputed_claim,
                "pi_positions": {
                    role: _memo_summary(role, memo)
                    for role, memo in pi_memos.items()
                    if isinstance(memo, dict) and not memo.get("_pi_unavailable")
                },
                "resolving_experiment": resolving_experiment,
                "assigned_peer_role": "falsifier",
                **_research_metadata_from_source(
                    first_objection[1] if first_objection is not None else hypotheses[0],
                    "falsifier",
                ),
                "decision_rule": {
                    "keep_if": "target claim beats matched controls under task-valid metrics",
                    "downgrade_if": "effect is small, noisy, or control-explainable",
                    "kill_if": (
                        "matched control reproduces the claimed effect or the experiment "
                        "fails the validity gate"
                    ),
                },
            }
        ],
        "minority_high_upside": _minority_ideas(pi_memos, cross_reviews, next_gen_id=next_gen_id),
        "claim_boundary_updates": _claim_boundary_updates_from_round2(cross_reviews),
        "peer_contracts": peer_contracts,
        "success_metrics": {"required": success_metric_requirements},
        "dissent_ledger_ref": f"research_memory/dissent_ledger_gen{next_gen_id}.yaml",
        "fallback_metadata": {
            "reason": "chair_yaml_parse_failed",
            "parse_error": _clip(parse_error, limit=500),
        },
    }


@dataclass
class ChairResult:
    """Parsed Chair synthesis output for the next generation agenda."""

    success: bool
    agenda: dict[str, Any]
    raw_text: str
    error: str | None = None


class ChairArbiter:
    """Runs the Chair role and normalizes its agenda arbitration result."""

    def __init__(
        self,
        run_dir: Path,
        workspace: Path,
        model: str,
        max_runtime_minutes: int | None = 8,
        peer_budget: int = 5,
        mcp_servers: dict[str, Any] | None = None,
        stop_check_fn: Callable[[], bool] | None = None,
        # 2026-05-07: passed to BaseAgent so Chair can activate adaptive
        # thinking + max effort when task_spec says so.
        premium_mode: bool = False,
        # Optional plugin-supplied prompts directory. When set, ``chair.jinja2``
        # is searched here first; the framework's bundled ``multi_pi/prompts/``
        # remains a fallback. Wired in by ``LegacyTwoRoundExecutor`` from
        # ``PanelTopologySpec.prompts_dir``.
        prompts_dir: Path | None = None,
        # Optional plugin-supplied peer-role rotation used by the
        # deterministic fallback agenda. Empty tuple defers to the bundled
        # ``_PEER_ROLE_ROTATION`` constant. See issue #84.
        peer_role_rotation: tuple[str, ...] = (),
        role_ref: str | None = None,
        task_project_path: Path | None = None,
        plugin_registry: Any | None = None,
        reasoning_effort: str = "max",
    ):
        self.run_dir = Path(run_dir)
        self.workspace = Path(workspace)
        self.model = model
        self.max_runtime_minutes = max_runtime_minutes
        self.peer_budget = peer_budget
        self.mcp_servers = mcp_servers or {}
        self.stop_check_fn = stop_check_fn
        self.premium_mode = premium_mode
        self.reasoning_effort = reasoning_effort
        self.prompts_dir: Path | None = Path(prompts_dir) if prompts_dir is not None else None
        self.peer_role_rotation: tuple[str, ...] = tuple(peer_role_rotation)
        self.role_ref = role_ref or "task_role:chair"
        self.task_project_path: Path | None = (
            Path(task_project_path) if task_project_path is not None else None
        )
        self.plugin_registry = plugin_registry
        self._role_skill: RoleSkill | None | bool = None

    def skill(self) -> RoleSkill | None:
        """Load the task-local Chair RoleSkill when one is configured.

        Generic Chair code owns schema and safety constraints. The task-local
        RoleSkill owns domain-specific allocation policy, so it must be
        available to Chair synthesis in the same way PI RoleSkills are.
        """

        if self._role_skill is False:
            return None
        if isinstance(self._role_skill, RoleSkill):
            return self._role_skill
        try:
            self._role_skill = load_role_skill(
                self.role_ref,
                workspace=self.workspace,
                task_project_path=self.task_project_path,
            )
            return self._role_skill
        except Exception as exc:  # noqa: BLE001 - legacy non-task panels may omit Chair RoleSkill.
            logger.warning(
                "ChairArbiter: failed to load Chair RoleSkill %s: %s",
                self.role_ref,
                exc,
            )
            self._role_skill = False
            return None

    def render_prompt(
        self,
        shared_core_digest: dict[str, Any],
        pi_memos: dict[str, dict[str, Any]],
        cross_reviews: dict[str, dict[str, Any]],
        confidence_revisions: dict[str, Any],
        next_gen_id: int,
        completed_gen_id: int,
        panel_mode: str,
        shared_core_id: str,
        per_pi_label_maps: dict[str, dict[str, str]] | None = None,
        validation_feedback: tuple[str, ...] = (),
        validation_candidate: dict[str, Any] | None = None,
    ) -> str:
        from jinja2 import Environment, FileSystemLoader

        from praxist.plugins.workflow_stages.research_loop.backend import multi_pi as mp_pkg

        # When ``self.prompts_dir`` is provided by the panel topology plugin
        # (see ``PanelTopologySpec.prompts_dir``), ``chair.jinja2`` is searched
        # there first; the framework's bundled ``multi_pi/prompts/`` directory
        # remains a fallback so plugins may override only the templates they
        # need. When ``prompts_dir`` is ``None``, the search path collapses to
        # the bundled directory.
        default_prompts_dir = Path(mp_pkg.__file__).parent / "prompts"
        search_paths: list[str] = []
        if self.prompts_dir is not None:
            search_paths.append(str(self.prompts_dir))
        search_paths.append(str(default_prompts_dir))
        env = Environment(
            loader=FileSystemLoader(search_paths),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        tpl = env.get_template("chair.jinja2")
        role_skill = self.skill()
        prompt = tpl.render(
            role_skill=role_skill,
            max_runtime_minutes=self.max_runtime_minutes,
            shared_core_digest=shared_core_digest,
            pi_memos=pi_memos,
            cross_reviews=cross_reviews or {},
            confidence_revisions=confidence_revisions or {},
            next_gen_id=next_gen_id,
            completed_gen_id=completed_gen_id,
            panel_mode=panel_mode,
            shared_core_id=shared_core_id,
            peer_budget=self.peer_budget,
            peer_role_rotation=self.peer_role_rotation,
            per_pi_label_maps=per_pi_label_maps or {},
            effective_config_provenance_available=has_effective_config_metadata(shared_core_digest),
        )
        feedback = [str(item).strip() for item in validation_feedback if str(item).strip()]
        if not feedback:
            return prompt
        rendered_feedback = "\n".join(f"- {item[:500]}" for item in feedback[:20])
        candidate_text = json.dumps(
            validation_candidate or {},
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        return (
            prompt.rstrip()
            + "\n\nMACHINE VALIDATION CORRECTION (one bounded retry):\n"
            + rendered_feedback
            + "\nReturn a complete replacement agenda that fixes every listed contract "
            "violation while preserving the supplied evidence boundaries. Do not add "
            "unsupported research claims. Correct the rejected candidate below rather "
            "than starting from an unrelated plan.\n\nREJECTED AGENDA CANDIDATE:\n"
            + candidate_text
            + "\n"
        )

    async def run(
        self,
        shared_core_digest: dict[str, Any],
        pi_memos: dict[str, dict[str, Any]],
        cross_reviews: dict[str, dict[str, Any]],
        confidence_revisions: dict[str, Any],
        next_gen_id: int,
        completed_gen_id: int,
        panel_mode: str,
        shared_core_id: str,
        per_pi_label_maps: dict[str, dict[str, str]] | None = None,
        validation_feedback: tuple[str, ...] = (),
        validation_candidate: dict[str, Any] | None = None,
    ) -> ChairResult:
        try:
            prompt = self.render_prompt(
                shared_core_digest=shared_core_digest,
                pi_memos=pi_memos,
                cross_reviews=cross_reviews,
                confidence_revisions=confidence_revisions,
                next_gen_id=next_gen_id,
                completed_gen_id=completed_gen_id,
                panel_mode=panel_mode,
                shared_core_id=shared_core_id,
                per_pi_label_maps=per_pi_label_maps,
                validation_feedback=validation_feedback,
                validation_candidate=validation_candidate,
            )
            from praxist.plugins.workflow_stages.research_loop.backend.agent import BaseAgent

            # Chair invariant: synthesize from memos + shared_core_digest in
            # the prompt only. Raw filesystem tools can bypass evidence
            # cutoffs by reading ledgers/findings/frontier artifacts directly,
            # so Chair receives no tools and no MCP servers.
            allowed_tools: list[str] = []
            agent = BaseAgent(
                name="chair_arbiter",
                execution_role="pi",
                allowed_tools=allowed_tools,
                workspace=self.workspace,
                mcp_servers={},  # ← hard-disable MCP; allowed_tools alone is not enough
                model=self.model,
                stop_check_fn=self.stop_check_fn,
                premium_mode=self.premium_mode,
                reasoning_effort=self.reasoning_effort,
                plugin_registry=self.plugin_registry,
            )
            result = await asyncio.wait_for(
                agent.execute(task=prompt),
                timeout=(
                    self.max_runtime_minutes * 60 if self.max_runtime_minutes is not None else None
                ),
            )
            if not result.success:
                return ChairResult(
                    success=False,
                    agenda={},
                    raw_text="",
                    error=result.error or "chair agent failure",
                )
            output_dict = result.output if isinstance(result.output, dict) else {}
            _items = output_dict.get("text_outputs", []) or []
            text = "\n".join(str(x) for x in _items if x is not None)
            parse_result = _parse_chair_agenda_text(text)
            if parse_result.agenda is None:
                fallback = _build_deterministic_fallback_agenda(
                    pi_memos=pi_memos,
                    cross_reviews=cross_reviews,
                    next_gen_id=next_gen_id,
                    completed_gen_id=completed_gen_id,
                    panel_mode=panel_mode,
                    shared_core_id=shared_core_id,
                    peer_budget=self.peer_budget,
                    parse_error=parse_result.error or "unknown parse failure",
                    peer_role_rotation=self.peer_role_rotation,
                    research_loop_mode=shared_core_digest.get("research_loop_mode", "metric"),
                )
                logger.warning(
                    "ChairArbiter: chair output was not parseable as agenda YAML; "
                    "using deterministic fallback agenda for gen %s (error=%s)",
                    next_gen_id,
                    parse_result.error,
                )
                return ChairResult(
                    success=True,
                    agenda=fallback,
                    raw_text=text,
                    error=f"fallback_agenda_from_chair_yaml_parse_failure: {parse_result.error}",
                )
            agenda = parse_result.agenda
            if not isinstance(agenda, dict):
                fallback = _build_deterministic_fallback_agenda(
                    pi_memos=pi_memos,
                    cross_reviews=cross_reviews,
                    next_gen_id=next_gen_id,
                    completed_gen_id=completed_gen_id,
                    panel_mode=panel_mode,
                    shared_core_id=shared_core_id,
                    peer_budget=self.peer_budget,
                    parse_error="chair output not a yaml mapping",
                    peer_role_rotation=self.peer_role_rotation,
                    research_loop_mode=shared_core_digest.get("research_loop_mode", "metric"),
                )
                logger.warning(
                    "ChairArbiter: chair output parsed as non-mapping; "
                    "using deterministic fallback agenda for gen %s",
                    next_gen_id,
                )
                return ChairResult(
                    success=True,
                    agenda=fallback,
                    raw_text=text,
                    error="fallback_agenda_from_non_mapping_chair_output",
                )
            return ChairResult(success=True, agenda=agenda, raw_text=text)
        except TimeoutError:
            return ChairResult(
                success=False,
                agenda={},
                raw_text="",
                error="chair timeout",
            )
        except Exception as e:
            logger.exception("ChairArbiter: unexpected failure")
            return ChairResult(
                success=False,
                agenda={},
                raw_text="",
                error=str(e),
            )
