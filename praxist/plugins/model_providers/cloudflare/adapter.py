"""Executable Cloudflare Workers AI model provider plugin."""

from __future__ import annotations

from praxist.core.cloudflare import CLOUDFLARE_API_FORMAT
from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    """Manifest entrypoint for the Cloudflare Workers AI provider."""
    return ModelProviderAdapter("model_provider:cloudflare", api_format=CLOUDFLARE_API_FORMAT)
