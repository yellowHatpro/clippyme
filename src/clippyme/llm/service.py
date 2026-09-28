"""Environment-driven LLM provider selection."""

from __future__ import annotations

import os
from collections.abc import Mapping

from clippyme.llm.base import LLMProvider, LLMProviderError
from clippyme.llm.gemini import GeminiProvider
from clippyme.llm.openrouter import OpenRouterProvider

SUPPORTED_PROVIDERS = frozenset({"gemini", "openrouter"})


def get_provider_name(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    name = (source.get("LLM_PROVIDER") or "gemini").strip().lower()
    if name not in SUPPORTED_PROVIDERS:
        raise LLMProviderError(
            f"Unsupported LLM_PROVIDER: {name!r}",
            provider=name or "unknown",
        )
    return name


def _positive_int(value: str | None, default: int) -> int:
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def _positive_float(value: str | None, default: float) -> float:
    try:
        return max(1.0, float(value or default))
    except (TypeError, ValueError):
        return default


def create_provider(env: Mapping[str, str] | None = None) -> LLMProvider:
    source = os.environ if env is None else env
    name = get_provider_name(source)
    if name == "openrouter":
        return OpenRouterProvider(
            api_key=source.get("OPENROUTER_API_KEY", ""),
            fallback_models=source.get("OPENROUTER_FALLBACK_MODELS"),
            timeout=_positive_float(source.get("OPENROUTER_TIMEOUT_SECONDS"), 120.0),
            max_retries=_positive_int(source.get("OPENROUTER_MAX_RETRIES"), 3),
        )
    return GeminiProvider(
        api_key=source.get("GEMINI_API_KEY", ""),
        fallback_models=source.get("GEMINI_FALLBACK_MODELS"),
        max_retries=_positive_int(source.get("GEMINI_MAX_RETRIES"), 3),
    )


def configured_model(env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    if get_provider_name(source) == "openrouter":
        return source.get("OPENROUTER_MODEL") or "openrouter/free"
    return source.get("GEMINI_MODEL") or "gemini-3.5-flash"
