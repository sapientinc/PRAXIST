"""Executable Mistral OpenAI-compatible model provider plugin."""

from __future__ import annotations

from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    """Manifest entrypoint for the Mistral OpenAI-compatible provider alias."""
    return ModelProviderAdapter("model_provider:mistral_alias", api_format="openai_compatible")
