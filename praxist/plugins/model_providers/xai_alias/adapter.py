"""Executable xAI OpenAI-compatible model provider plugin."""

from __future__ import annotations

from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    """Manifest entrypoint for the xAI OpenAI-compatible provider alias."""
    return ModelProviderAdapter("model_provider:xai_alias", api_format="openai_compatible")
