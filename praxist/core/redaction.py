"""Redaction helpers for run metadata, trajectory, and artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}")),
    ("openrouter_key", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}\b")),
    ("orcarouter_key", re.compile(r"\bsk-orca-[A-Za-z0-9]{16,}\b")),
    ("openai_style_key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9][A-Za-z0-9._-]{7,}\b")),
    ("github_token", re.compile(r"\bghp_[A-Za-z0-9_]{12,}\b")),
    (
        "named_secret",
        re.compile(
            r"(?i)\b[A-Z0-9_]*(api[_-]?key|auth[_-]?token|access[_-]?token|secret|password)\b"
            r"\s*[:=]\s*['\"]?[^'\"\s,}]{6,}"
        ),
    ),
    (
        "named_secret",
        re.compile(r"(?i)\b(api\s*key|key)\s*`[A-Za-z0-9._~+/=-]{16,}`"),
    ),
    (
        "named_secret",
        re.compile(
            r"(?i)['\"]?\b(api[_-]?key|auth[_-]?token|access[_-]?token|secret|password)\b['\"]?"
            r"\s*[:=]\s*['\"]?[^'\"\s,}]{6,}"
        ),
    ),
)

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)^(api[_-]?key|auth[_-]?token|access[_-]?token|secret|password)$"
)


def redact_text(text: str) -> tuple[str, list[str]]:
    """Return redacted text plus the classes of matches found."""
    hits: list[str] = []
    out = text
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(out):
            hits.append(label)
            out = pattern.sub(f"<redacted:{label}>", out)
    return out, hits


def redact_json(value: JSONValue) -> tuple[JSONValue, list[str]]:
    """Recursively redact JSON-compatible data."""
    hits: list[str] = []

    def visit(obj: JSONValue) -> JSONValue:
        if isinstance(obj, str):
            redacted, found = redact_text(obj)
            hits.extend(found)
            return redacted
        if isinstance(obj, list):
            return [visit(item) for item in obj]
        if isinstance(obj, dict):
            out: dict[str, JSONValue] = {}
            for key, item in obj.items():
                redacted_key, key_hits = redact_text(str(key))
                hits.extend(key_hits)
                if _SENSITIVE_KEY_RE.match(str(key)):
                    hits.append("named_secret")
                    out[redacted_key] = "<redacted:named_secret>"
                else:
                    out[redacted_key] = visit(item)
            return out
        return obj

    return visit(value), hits


def scan_text(text: str) -> list[str]:
    """Return redaction classes present in text."""
    hits: list[str] = []
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            hits.append(label)
    return hits


def scan_file(path: Path) -> list[str]:
    """Scan a UTF-8-ish text file for raw secret patterns."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ["unreadable"]
    return scan_text(text)


def dumps_redacted(value: Any, **kwargs: Any) -> str:
    """JSON dump after recursive redaction."""
    redacted, _ = redact_json(value)
    return json.dumps(redacted, ensure_ascii=False, **kwargs)
