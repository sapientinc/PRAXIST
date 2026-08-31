"""Context firewall — enforce per-mode token / card budgets on PI prompts.

The firewall exposes two operations:

  fit_pack_to_budget(pack, mode_budget) -> dict (json-safe; size ≤ budget)
  forbid_raw_history(pack) -> bool (returns True if pack contains raw history blobs)

Token budget is enforced by a coarse char-count proxy (~4 chars/token).
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


CHARS_PER_TOKEN = 4  # rough proxy


@dataclass
class ModeBudget:
    """Prompt-context budget split for shared, role-private, and optional evidence sections."""

    shared_core_tokens: int
    private_pack_tokens: int = 0
    max_cards: int = 40
    chair_can_read_raw: bool = False


BUDGETS = {
    "mini": ModeBudget(shared_core_tokens=25_000, private_pack_tokens=0, max_cards=20),
    "full": ModeBudget(shared_core_tokens=50_000, private_pack_tokens=15_000, max_cards=40),
    "high_stakes": ModeBudget(shared_core_tokens=60_000, private_pack_tokens=25_000, max_cards=50),
}


def estimate_tokens(obj: Any) -> int:
    """Estimate tokens via JSON byte count.

    R2#7 fix: use byte length (UTF-8) rather than character length.
    A CJK character is len()==1 but ~3 bytes; SentencePiece-style
    tokenizers approximate 4 bytes/token, not 4 chars/token.
    """
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        s = str(obj)
    try:
        n_bytes = len(s.encode("utf-8"))
    except Exception:
        n_bytes = len(s)
    return max(1, n_bytes // CHARS_PER_TOKEN)


def _shrink_scoreless_context(d: dict[str, Any], budget_tokens: int) -> dict[str, Any]:
    """Keep bounded evidence structured and label every budget-induced omission."""
    out = copy.deepcopy(d)
    rows = [row for row in out.get("scoreless_evidence", []) if isinstance(row, dict)]
    input_count = len(rows)
    out["scoreless_evidence"] = rows
    metadata = {
        "input_count": input_count,
        "returned": input_count,
        "omitted_for_budget": 0,
        "input_scope": "bounded_retained_evidence",
        "budget_truncated": False,
        "budget_exceeded": False,
    }
    out["scoreless_evidence_meta"] = metadata
    if estimate_tokens(out) <= budget_tokens:
        return out

    for row in rows:
        for key in ("content", "title", "summary"):
            value = row.get(key)
            if isinstance(value, str) and len(value) > 500:
                row[key] = value[:500]
                row[f"{key}_truncated"] = True
                metadata["budget_truncated"] = True
    if estimate_tokens(out) > budget_tokens:
        base = {
            key: value
            for key, value in out.items()
            if key not in {"research_loop_mode", "scoreless_evidence", "scoreless_evidence_meta"}
        }
        out = shrink_dict(base, max(1, budget_tokens // 2))
        out.update(
            {
                "research_loop_mode": "scoreless",
                "scoreless_evidence": rows,
                "scoreless_evidence_meta": metadata,
            }
        )
    while rows and estimate_tokens(out) > budget_tokens:
        counts = Counter(str(row.get("finding_type") or "unknown") for row in rows)
        # Retain at least one row per type while the budget permits it.
        drop_index = next(
            (
                index
                for index in range(len(rows) - 1, -1, -1)
                if counts[str(rows[index].get("finding_type") or "unknown")] > 1
            ),
            len(rows) - 1,
        )
        rows.pop(drop_index)
        metadata.update(
            {
                "returned": len(rows),
                "omitted_for_budget": input_count - len(rows),
                "budget_truncated": True,
            }
        )
    metadata["budget_exceeded"] = estimate_tokens(out) > budget_tokens
    return out


def shrink_dict(d: dict[str, Any], budget_tokens: int) -> dict[str, Any]:
    """Try to shrink a dict to fit within token budget by:
    1. truncating string values
    2. truncating list values
    3. dropping low-priority keys
    """
    if d.get("research_loop_mode") == "scoreless":
        return _shrink_scoreless_context(d, budget_tokens)
    if estimate_tokens(d) <= budget_tokens:
        return d

    out = {k: v for k, v in d.items()}

    # Pass 1: truncate strings to 500 chars max
    def _trunc_str(v: Any) -> Any:
        if isinstance(v, str) and len(v) > 500:
            return v[:500] + "...[truncated]"
        if isinstance(v, list):
            return [_trunc_str(x) for x in v]
        if isinstance(v, dict):
            return {k: _trunc_str(x) for k, x in v.items()}
        return v

    out = {k: _trunc_str(v) for k, v in out.items()}
    if estimate_tokens(out) <= budget_tokens:
        return out

    # Pass 2: cap list lengths
    def _cap_list(v: Any, max_n: int, path: tuple[str, ...] = ()) -> Any:
        # Durable Gems are intentionally compact. Keep their fields stable, but
        # respect the global active Gems cap instead of allowing stale states to
        # expand PI/Chair context indefinitely.
        if path == ("shared_core", "gems", "entries") or path == ("gems", "entries"):
            if isinstance(v, list):
                return [
                    {
                        key: _trunc_str(value)
                        for key, value in item.items()
                        if key
                        in {
                            "gem_finding_id",
                            "variant_name",
                            "frontier_lane",
                            "metric_name",
                            "metric_value",
                            "source_finding_id",
                            "source_generation_id",
                            "bottleneck_target",
                            "evidence_stage",
                            "tradeoff_class",
                            "primary_tradeoff",
                            "next_step_intent",
                            "parent_candidate",
                            "parent_usage",
                            "gem_variant_ref",
                            "finding_path",
                            "admission_metrics",
                        }
                    }
                    if isinstance(item, dict)
                    else item
                    for item in v[:4]
                ]
            return v
        if isinstance(v, list) and len(v) > max_n:
            return v[:max_n] + [f"...[+{len(v) - max_n} more]"]
        if isinstance(v, dict):
            return {k: _cap_list(x, max_n, (*path, str(k))) for k, x in v.items()}
        return v

    out = {k: _cap_list(v, 20, (str(k),)) for k, v in out.items()}
    if estimate_tokens(out) <= budget_tokens:
        return out

    # Pass 3: drop low-priority keys.
    # R7#6 fix: removed coverage_matrix_digest from drop list. Bridge
    # contracts MUST consult it; dropping silently led to chair inventing
    # bridge targets without coverage awareness.
    LOW_PRIORITY = (
        "negative_evidence_digest",
        "role_performance",
        "findings_summary",
    )
    for key in LOW_PRIORITY:
        if estimate_tokens(out) <= budget_tokens:
            break
        if key in out:
            out[key] = "...[budget truncated]"

    return out


def fit_pack_to_budget(pack, mode: str) -> dict[str, Any]:
    """Apply mode-specific budget caps to an EvidencePack.

    Returns a json-safe dict ready to render into the PI prompt.
    """
    budget = BUDGETS.get(mode, BUDGETS["full"])
    out: dict[str, Any] = {
        "pack_id": pack.pack_id,
        "built_at": pack.built_at,
        "panel_mode": pack.panel_mode,
        "target_decisions": pack.target_decisions,
    }
    if isinstance(getattr(pack, "audit", None), dict) and isinstance(
        pack.audit.get("artifact_semantics"), dict
    ):
        out["artifact_semantics"] = dict(pack.audit["artifact_semantics"])
    out["shared_core"] = shrink_dict(pack.shared_core, budget.shared_core_tokens)

    # Cap private pack cards
    out["private_packs"] = {}
    for role, cards in (pack.private_packs or {}).items():
        capped = list(cards[: budget.max_cards])
        # R9#5 fix: mini mode declares private_pack_tokens=0 to mean "no
        # role-specific pack" — collapse to ≤2 cards as a thumbnail
        # summary instead of returning the full max_cards pack.
        if budget.private_pack_tokens == 0:
            capped = capped[:2]
        # Per-pack token budget — but never empty out the pack. PI with
        # zero role-specific evidence is worse than one card slightly over
        # budget. (R1#4 fix.)
        if budget.private_pack_tokens > 0:
            # First, if a single card is itself oversized, shrink its
            # interpretation field rather than dropping the only signal.
            while len(capped) > 1 and estimate_tokens(capped) > budget.private_pack_tokens:
                capped.pop()
            # If still over budget with one card, truncate that card's
            # interpretation rather than emptying the pack.
            if capped and estimate_tokens(capped) > budget.private_pack_tokens:
                only = dict(capped[0])
                interp = only.get("interpretation") or {}
                if isinstance(interp, dict) and isinstance(interp.get("short"), str):
                    interp["short"] = interp["short"][:300] + "...[truncated for budget]"
                    only["interpretation"] = interp
                capped = [only]
        out["private_packs"][role] = capped

    out["audit"] = pack.audit
    return out


def forbid_raw_history(pack_dict: dict[str, Any]) -> bool:
    """Return True if pack_dict contains a key resembling raw history."""
    BAD_KEYS = ("raw_findings_full_text", "prior_pi_memo_full_text", "raw_history")
    blob = json.dumps(pack_dict, default=str)
    return any(k in blob for k in BAD_KEYS)
