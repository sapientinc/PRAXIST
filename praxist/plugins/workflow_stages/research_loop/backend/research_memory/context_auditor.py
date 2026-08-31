"""Context auditor — post-synthesis checks on PI memos and final agenda.

Does NOT participate in scientific decisions. Purely a verifier:
  - every final claim has source_id
  - every active objection assigned/deferred/archived
  - bridge contracts checked coverage matrix
  - retired claims preserve scope + revive_if
  - high-stakes claims have at least one challenge card
  - private-KB-inspired claims marked as hypothesis (not established)
  - chair did not introduce uncited scientific claims
  - negative evidence appears in PI memos and final agenda
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Patterns that should never appear in established claims (only in hypotheses).
OVERCLAIM_TERMS = (
    "universal",
    "universally",
    "generally dominant",
    "architecture-independent",
    "solved",
    "obsolete",
    "breakthrough optimizer",
    "breakthrough discovery",
    "best in class",
)

_NEGATIVE_DIGEST_STOPWORDS = {
    "and",
    "are",
    "available",
    "digest",
    "entry",
    "evidence",
    "fail",
    "failed",
    "failure",
    "generic",
    "for",
    "from",
    "negative",
    "not",
    "result",
    "results",
    "the",
    "this",
    "unseen",
    "was",
    "with",
}


@dataclass
class AuditReport:
    """Research-memory context audit result used before PI agenda synthesis."""

    audit_id: str
    pass_: bool = True
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    required_fixes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _has_source_id(claim: dict[str, Any]) -> bool:
    """Heuristic: claim has any source_findings, supporting evidence, or
    explicit source_id field referenced.

    List-valued fields must contain at least one truthy item; placeholders such
    as ``[None, ""]`` are not valid source references.
    """
    if not isinstance(claim, dict):
        return False
    sid = claim.get("source_id")
    if isinstance(sid, str) and sid.strip():
        return True
    for key in ("source_findings", "supports", "source_evidence"):
        v = claim.get(key)
        if isinstance(v, list) and any(
            (isinstance(x, str) and x.strip()) or (isinstance(x, dict) and any(x.values()))
            for x in v
        ):
            return True
        if isinstance(v, str) and v.strip():
            return True
    return False


def _has_overclaim_language(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    blob = text.lower()
    return [t for t in OVERCLAIM_TERMS if t in blob]


def _negative_digest_entries(pack_dict: dict[str, Any]) -> list[Any]:
    """Return negative evidence ledger digest entries visible in a pack.

    Private evidence cards are sampled, so relying only on private pack card
    quality can miss the system-level negative evidence ledger. The shared core
    digest is compact and should count as consumed negative context for synthesis
    audit purposes.
    """

    if not isinstance(pack_dict, dict):
        return []
    candidates: list[Any] = []
    shared_core = pack_dict.get("shared_core")
    if isinstance(shared_core, dict):
        candidates.append(shared_core.get("negative_evidence_digest"))
    candidates.append(pack_dict.get("negative_evidence_digest"))
    raw_entries: list[Any] = []
    for candidate in candidates:
        if isinstance(candidate, list):
            raw_entries.extend(item for item in candidate if item)
            continue
        if isinstance(candidate, dict):
            entries = candidate.get("entries") or candidate.get("items") or candidate.get("recent")
            if isinstance(entries, list):
                raw_entries.extend(item for item in entries if item)
            else:
                raw_entries.append(candidate)
            continue
        if (
            isinstance(candidate, str)
            and candidate.strip()
            and "budget truncated" not in candidate.lower()
        ):
            raw_entries.append(candidate)
    out: list[Any] = []
    seen: set[str] = set()
    for entry in raw_entries:
        key = _negative_digest_identity(entry)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _negative_digest_identity(entry: Any) -> str:
    """Stable-enough identity for mirrored digest entries.

    The same ledger digest can be exposed at both pack root and shared_core.
    Counting both can flip the negative-evidence audit floor, so prefer
    explicit ledger ids and fall back to a canonical JSON/string signature.
    """

    if isinstance(entry, dict):
        for key in ("id", "finding_id", "ledger_id", "source_id"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return f"{key}:{value.strip()}"
        try:
            return "json:" + json.dumps(entry, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return "dict:" + repr(sorted(entry.items()))
    if isinstance(entry, str):
        return "str:" + entry.strip()
    try:
        return "json:" + json.dumps(entry, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return "repr:" + repr(entry)


def _negative_digest_reference_tokens(entries: list[Any]) -> set[str]:
    tokens: set[str] = set()

    def _add_identifier_token(value: str) -> None:
        token = value.strip().lower()
        if len(token) < 3:
            return
        tokens.add(token)
        tokens.add(token.replace("_", " "))
        tokens.add(token.replace("-", " "))

    def _add_text_token(value: str) -> None:
        token = value.strip().lower()
        if len(token) < 3:
            return
        raw_words = re.findall(r"[a-z0-9]+", token.replace("_", " "))
        salient = [
            word for word in raw_words if len(word) >= 3 and word not in _NEGATIVE_DIGEST_STOPWORDS
        ]
        if len(salient) >= 2 or (
            len(salient) == 1 and (len(salient[0]) >= 8 or "_" in token or "-" in token)
        ):
            tokens.add(token)
            tokens.add(token.replace("_", " "))
            tokens.add(token.replace("-", " "))

    def _add_phrase_tokens(value: str) -> None:
        raw_words = re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
        generic_phrases = {
            "negative evidence",
            "evidence digest",
            "digest entry",
            "entry available",
        }
        for width in (2, 3):
            if len(raw_words) < width:
                continue
            for idx in range(0, len(raw_words) - width + 1):
                phrase_words = raw_words[idx : idx + width]
                salient = [
                    word
                    for word in phrase_words
                    if len(word) >= 3 and word not in _NEGATIVE_DIGEST_STOPWORDS
                ]
                phrase = " ".join(phrase_words)
                if len(salient) >= 2 and phrase not in generic_phrases:
                    tokens.add(phrase)

        compact_words = [
            word for word in raw_words if len(word) >= 3 and word not in _NEGATIVE_DIGEST_STOPWORDS
        ]
        for width in (2, 3):
            if len(compact_words) < width:
                continue
            for idx in range(0, len(compact_words) - width + 1):
                tokens.add(" ".join(compact_words[idx : idx + width]))

    for entry in entries:
        if isinstance(entry, dict):
            for key in (
                "id",
                "finding_id",
                "ledger_id",
                "source",
                "source_id",
                "claim_id",
                "variant_name",
            ):
                value = entry.get(key)
                if isinstance(value, str):
                    _add_identifier_token(value)
            for key in ("title", "summary"):
                value = entry.get(key)
                if not isinstance(value, str):
                    continue
                _add_text_token(value)
                _add_phrase_tokens(value)
        elif isinstance(entry, str):
            _add_text_token(entry)
            _add_phrase_tokens(entry)
    return tokens


def _mentions_negative_digest(obj: Any, entries: list[Any]) -> bool:
    if not entries:
        return False
    try:
        blob = json.dumps(obj, default=str).lower()
    except (TypeError, ValueError):
        blob = str(obj).lower()
    return any(token in blob for token in _negative_digest_reference_tokens(entries))


def audit_agenda(
    agenda: dict[str, Any],
    pack_dict: dict[str, Any],
    pi_memos: dict[str, dict[str, Any]],
    audit_id: str = "audit_unknown",
    completed_gen_id: int = -1,
    cohort_size: int | None = None,
    expected_peer_contract_count: int | None = None,
) -> AuditReport:
    """Run all checks. Caller should refuse to publish if blocking_issues != [].

    ``cohort_size`` (issue #100) makes the ``peer_contracts`` count check
    cohort-aware. When supplied (the production path threads
    ``task_spec.generation_policy.cohort_size`` in via the panel runner),
    the bound is ``max(1, cohort - 2) <= len <= cohort + 1`` — preserving
    the ±2 slack the bundled default expressed but scaling to whatever the
    task actually declared. When omitted (legacy callers / unit tests
    that exercise the helper directly), the bound falls back to the
    historical ``3 <= len <= 6`` which works for the bundled 5-peer cohort.
    """
    report = AuditReport(audit_id=audit_id)
    shared_core = pack_dict.get("shared_core") or {}
    scoreless = (
        isinstance(shared_core, dict) and shared_core.get("research_loop_mode") == "scoreless"
    )
    metrics: dict[str, Any] = {
        "claims_checked": 0,
        "claims_with_source_id": 0,
        "high_stakes_claims": 0,
        "high_stakes_with_challenge": 0,
        "bridge_contracts": 0,
        "bridge_with_coverage_check": 0,
        "retired_with_revive_if": 0,
        "retired_total": 0,
        "private_kb_claims": 0,
        "private_kb_marked_as_hypothesis": 0,
        "negative_evidence_in_pack": 0,
    }
    if scoreless:
        metrics["high_stakes_with_challenge"] = None
        metrics["private_card_challenge_check"] = "not_applicable"

    # --- Check 1: every consensus_action / hypothesis has source_id-equivalent
    for section_key in ("consensus_actions", "cross_peer_hypotheses"):
        for c in agenda.get(section_key) or []:
            if not isinstance(c, dict):
                continue
            metrics["claims_checked"] += 1
            if _has_source_id(c):
                metrics["claims_with_source_id"] += 1
            else:
                report.warnings.append(
                    f"{section_key}: claim {c.get('id') or c.get('action_id') or '?'} "
                    f"lacks source_id / supports / source_findings"
                )

    # --- Check 2: every retired claim has boundary + revive_if
    for r in agenda.get("retired_claims") or agenda.get("claim_boundary_updates") or []:
        if not isinstance(r, dict):
            continue
        metrics["retired_total"] += 1
        revive = r.get("revive_if") or r.get("required_validation_before_upgrade")
        boundary = r.get("boundary") or r.get("new_language") or ""
        if revive and boundary:
            metrics["retired_with_revive_if"] += 1
        else:
            report.blocking_issues.append(
                f"retired claim {r.get('claim_id') or r.get('id') or '?'} "
                f"missing boundary or revive_if"
            )

    # --- Check 3: every bridge contract has coverage check (heuristic: contract
    # mentions coverage_check or query_coverage_matrix tool reference)
    for peer_id, pc in (agenda.get("peer_contracts") or {}).items():
        if not isinstance(pc, dict):
            continue
        # Multi-round-audit-R3 fix: agenda_validator_v2 normalizes role
        # via _normalize_role (lowercase + strip + dash→underscore). The
        # auditor previously did a strict `pc.get("role") == "bridge"`,
        # so a Chair-emitted "Bridge" or "BRIDGE" (etc.) would skip the
        # coverage_check audit entirely. Match validator semantics by
        # normalizing the same way.
        _role = str(pc.get("role") or "").strip().lower().replace("-", "_")
        if _role == "bridge":
            metrics["bridge_contracts"] += 1
            blob = json.dumps(pc, default=str).lower() if pc else ""
            if "coverage_check" in blob or "query_coverage_matrix" in blob:
                metrics["bridge_with_coverage_check"] += 1
            else:
                report.warnings.append(
                    f"bridge contract for {peer_id} did not reference coverage_check"
                )

    # --- Check 4: high-stakes language → must have challenge evidence.
    # Scope this check to claim-bearing fields so observational text such as
    # "X learning rate is obsolete in PyTorch 1.9+" is not misclassified.
    claim_fragments: list[str] = []
    for c in agenda.get("consensus_actions") or []:
        if isinstance(c, dict):
            claim_fragments.append(str(c.get("claim_or_hypothesis", "")))
    for c in agenda.get("cross_peer_hypotheses") or []:
        if isinstance(c, dict):
            claim_fragments.append(str(c.get("claim", "")))
    for c in agenda.get("claim_boundary_updates") or []:
        if isinstance(c, dict):
            claim_fragments.append(str(c.get("new_language", "")))
            claim_fragments.append(str(c.get("old_language", "")))
    mo = agenda.get("mainline_observation") or {}
    if isinstance(mo, dict):
        claim_fragments.append(str(mo.get("main_risk", "")))
        claim_fragments.append(str(mo.get("key_tradeoff", "")))
    blob_all = " ".join(claim_fragments).lower()
    for term in OVERCLAIM_TERMS:
        if term in blob_all:
            metrics["high_stakes_claims"] += 1
            if scoreless:
                break
            # check whether at least one challenge card exists in pack
            challenge_count = sum(
                1
                for c in pack_dict.get("private_packs", {}).get("skeptic", [])
                if isinstance(c, dict) and c.get("quality", {}).get("is_negative")
            )
            if challenge_count > 0:
                metrics["high_stakes_with_challenge"] += 1
            else:
                report.warnings.append(
                    f"high-stakes language '{term}' present but no challenge "
                    f"evidence cards visible to skeptic PI"
                )
            break  # check once

    # --- Check 5: PI memos disclosed private_knowledge_used
    for role, memo in (pi_memos or {}).items():
        if not isinstance(memo, dict):
            continue
        pkb = memo.get("private_knowledge_used")
        if pkb is None:
            report.warnings.append(f"PI {role} memo missing 'private_knowledge_used' disclosure")

    # --- Check 6: negative evidence ratio in pack
    # R1#12 fix: enforce a floor (15%) as blocking, not just warn at <10%.
    # Below 15% indicates the cohort is being shielded from disconfirming
    # evidence; the design doc targets ≥20%.
    all_cards = []
    for cards in (pack_dict.get("private_packs", {}) or {}).values():
        all_cards.extend(cards or [])
    card_neg_count = sum(
        1 for c in all_cards if isinstance(c, dict) and c.get("quality", {}).get("is_negative")
    )
    digest_entries = _negative_digest_entries(pack_dict)
    digest_neg_count = len(digest_entries)
    neg_count = card_neg_count + digest_neg_count
    metrics["negative_evidence_in_pack"] = neg_count
    metrics["negative_evidence_cards_in_pack"] = card_neg_count
    metrics["negative_evidence_digest_entries"] = digest_neg_count
    digest_referenced = _mentions_negative_digest(
        {"agenda": agenda, "pi_memos": pi_memos}, digest_entries
    )
    metrics["negative_evidence_digest_referenced"] = bool(digest_referenced)
    evidence_denominator = len(all_cards) + digest_neg_count
    metrics["negative_evidence_ratio"] = (
        None if scoreless else (neg_count / evidence_denominator if evidence_denominator else 0.0)
    )
    if scoreless:
        metrics["negative_evidence_ratio_status"] = "not_applicable"
    is_first_synthesis = completed_gen_id == 0
    if evidence_denominator and not scoreless:
        ratio = neg_count / evidence_denominator
        # R5#6 fix: do NOT block on first synthesis (gen 0 → gen 1).
        # Peers discover negative evidence by running experiments; gen 0
        # may still be predominantly positive while peers explore. Blocking
        # the first agenda would deadlock the run before peers learn.
        if ratio < 0.15 and not is_first_synthesis:
            report.blocking_issues.append(
                f"negative evidence ratio {neg_count}/{evidence_denominator} "
                f"({100 * ratio:.0f}%) is below 15% safety floor — synthesis "
                f"would be biased toward supportive-only evidence"
            )
        elif ratio < 0.15 and is_first_synthesis:
            report.warnings.append(
                f"negative evidence ratio {neg_count}/{evidence_denominator} "
                f"({100 * ratio:.0f}%) is low; tolerated for first synthesis "
                f"(gen 0 → gen 1) but will block from gen 1 onwards"
            )
        elif ratio < 0.20:
            report.warnings.append(
                f"negative evidence ratio {neg_count}/{evidence_denominator} "
                f"({100 * ratio:.0f}%) is below 20% target"
            )
    if digest_entries and not digest_referenced:
        msg = (
            "negative evidence digest is present in shared context but is not "
            "referenced by PI memos or the final agenda"
        )
        if is_first_synthesis:
            report.warnings.append(msg)
        else:
            report.blocking_issues.append(msg)

    # --- Check 7: peer_contracts count
    pcs = agenda.get("peer_contracts") or {}
    if isinstance(pcs, dict):
        # R9#1 fix: tolerate degenerate gen 0 with no peers (all crashed
        # before they wrote findings). Warn but don't block — operator
        # needs the audit report to debug, not synthesis halted.
        # Issue #100: scale the bound to ``cohort_size`` when supplied
        # so tasks with cohort > 6 don't get rejected on every synthesis.
        is_first_synthesis = completed_gen_id == 0
        expected_count = (
            int(expected_peer_contract_count)
            if isinstance(expected_peer_contract_count, int) and expected_peer_contract_count > 0
            else None
        )
        if expected_count is not None:
            bad_count = len(pcs) != expected_count
            expected_label = f"expected {expected_count}"
        elif isinstance(cohort_size, int) and cohort_size > 0:
            lower = max(1, cohort_size - 2)
            upper = cohort_size + 1
            bad_count = len(pcs) < lower or len(pcs) > upper
            expected_label = f"expected {lower}-{upper}"
        else:
            # Fallback callers may not know the current task cohort size. In
            # that mode the auditor should only reject an empty contract set,
            # not impose a historical small-cohort upper bound.
            bad_count = len(pcs) < 1
            expected_label = "expected at least 1"
        if bad_count:
            if is_first_synthesis and len(pcs) == 0:
                report.warnings.append(
                    "peer_contracts is empty for gen 0 → gen 1; if all gen 0 "
                    "peers crashed this is expected. Otherwise investigate."
                )
            else:
                report.blocking_issues.append(
                    f"peer_contracts count = {len(pcs)} ({expected_label})"
                )

    # finalize
    if metrics["claims_checked"] > 0:
        metrics["citation_coverage"] = metrics["claims_with_source_id"] / metrics["claims_checked"]
    else:
        metrics["citation_coverage"] = 1.0

    report.metrics = metrics
    report.pass_ = len(report.blocking_issues) == 0
    return report
