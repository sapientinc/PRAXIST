"""Executable OrcaRouter model provider plugin."""

from __future__ import annotations

from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    """Manifest entrypoint for the OrcaRouter model-provider adapter."""
    return ModelProviderAdapter("model_provider:orcarouter", api_format="orcarouter")
