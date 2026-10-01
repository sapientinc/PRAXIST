"""Executable Cheaper Inference model provider plugin."""

from __future__ import annotations

from praxist.core.modeling import ModelProviderAdapter


def create_provider() -> ModelProviderAdapter:
    """Manifest entrypoint for the Cheaper Inference model-provider adapter."""
    return ModelProviderAdapter("model_provider:cheaperinference", api_format="cheaperinference")
