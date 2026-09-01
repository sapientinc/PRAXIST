"""Cloudflare Workers AI endpoint helpers.

Workers AI exposes one OpenAI-compatible surface at
``/client/v4/accounts/{account_id}/ai/v1`` and no Anthropic Messages route,
so every Praxist path that talks to it reuses the OpenAI-compatible client
and only varies the base URL and bearer credential.

The account id is part of the URL rather than the credential, so it is
resolved here once and shared by the CLI registry and the Codex relay.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

CLOUDFLARE_PROVIDER = "cloudflare"
"""Short provider name persisted as ``PRAXIST_LLM_PROVIDER``."""

CLOUDFLARE_PROVIDER_REF = "model_provider:cloudflare"
"""Canonical model-provider plugin ref for Workers AI."""

CLOUDFLARE_API_FORMAT = "cloudflare_workers_ai"
"""OpenAI-compatible wire format that keeps full ``@cf/vendor/model`` ids."""

CLOUDFLARE_KEY_VAR = "CLOUDFLARE_API_KEY"
"""Bearer credential variable for the Workers AI REST endpoint."""

CLOUDFLARE_ACCOUNT_VAR = "CLOUDFLARE_ACCOUNT_ID"
"""Account id variable interpolated into the Workers AI base URL."""

CLOUDFLARE_BASE_URL_VAR = "CLOUDFLARE_BASE_URL"
"""Explicit base-URL override that wins over account-id interpolation."""

CLOUDFLARE_BASE_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
"""OpenAI-compatible Workers AI base URL; ``/chat/completions`` hangs off it."""

CLOUDFLARE_DEFAULT_MODEL = "@cf/deepseek-ai/deepseek-v4-flash-0731"
"""Cheap/fast DeepSeek tier served by Workers AI."""

CLOUDFLARE_STRONG_MODEL = "@cf/deepseek-ai/deepseek-v4-pro-0813"
"""Strong-reasoner DeepSeek tier served by Workers AI."""


def workers_ai_base_url(
    env: Mapping[str, str] | None = None,
    *,
    account_id: str | None = None,
) -> str:
    """Return the Workers AI OpenAI-compatible base URL for this host.

    Resolution order: an explicit ``account_id`` argument, then
    ``CLOUDFLARE_BASE_URL`` as a whole-URL override, then
    ``CLOUDFLARE_ACCOUNT_ID`` from the environment.
    """
    source = os.environ if env is None else env
    explicit = (account_id or "").strip()
    if explicit:
        return CLOUDFLARE_BASE_URL_TEMPLATE.format(account_id=explicit)
    override = str(source.get(CLOUDFLARE_BASE_URL_VAR, "")).strip().rstrip("/")
    if override:
        return override
    resolved = str(source.get(CLOUDFLARE_ACCOUNT_VAR, "")).strip()
    if not resolved:
        raise ValueError(
            f"{CLOUDFLARE_ACCOUNT_VAR} is not set; Workers AI base URLs are account-scoped. "
            f"Export {CLOUDFLARE_ACCOUNT_VAR} or set {CLOUDFLARE_BASE_URL_VAR} to a full base URL."
        )
    return CLOUDFLARE_BASE_URL_TEMPLATE.format(account_id=resolved)


__all__ = [
    "CLOUDFLARE_ACCOUNT_VAR",
    "CLOUDFLARE_API_FORMAT",
    "CLOUDFLARE_BASE_URL_TEMPLATE",
    "CLOUDFLARE_BASE_URL_VAR",
    "CLOUDFLARE_DEFAULT_MODEL",
    "CLOUDFLARE_KEY_VAR",
    "CLOUDFLARE_PROVIDER",
    "CLOUDFLARE_PROVIDER_REF",
    "CLOUDFLARE_STRONG_MODEL",
    "workers_ai_base_url",
]
