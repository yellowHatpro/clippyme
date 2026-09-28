from __future__ import annotations

import pytest

from clippyme.llm.base import LLMProviderError
from clippyme.llm.openrouter import OpenRouterProvider
from clippyme.llm.service import configured_model, create_provider, get_provider_name


def test_provider_selection_defaults_to_gemini():
    assert get_provider_name({}) == "gemini"
    assert configured_model({}) == "gemini-3.5-flash"


def test_openrouter_selection_and_model():
    env = {
        "LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "secret",
        "OPENROUTER_MODEL": "vendor/model:free",
    }
    provider = create_provider(env)
    assert isinstance(provider, OpenRouterProvider)
    assert configured_model(env) == "vendor/model:free"


def test_unknown_provider_fails_cleanly():
    with pytest.raises(LLMProviderError, match="Unsupported"):
        create_provider({"LLM_PROVIDER": "unknown"})


def test_openrouter_requires_key():
    with pytest.raises(LLMProviderError, match="OPENROUTER_API_KEY"):
        create_provider({"LLM_PROVIDER": "openrouter"})
