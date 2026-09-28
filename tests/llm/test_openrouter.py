from __future__ import annotations

import pytest
import requests

from clippyme.llm.base import LLMProviderError
from clippyme.llm.openrouter import OpenRouterProvider


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_normalizes_chat_completion_and_usage():
    session = FakeSession([FakeResponse(200, {
        "id": "gen-1",
        "model": "nvidia/example:free",
        "choices": [{"message": {"content": '{"shorts": []}'}}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45, "cost": 0},
    })])
    provider = OpenRouterProvider(api_key="secret", session=session, sleep=lambda _: None)

    result = provider.generate(prompt="find clips", model="openrouter/free")

    assert result.content == '{"shorts": []}'
    assert result.provider == "openrouter"
    assert result.model == "nvidia/example:free"
    assert result.input_tokens == 123
    assert result.output_tokens == 45
    assert result.estimated_cost == 0.0
    url, request = session.calls[0]
    assert url.endswith("/chat/completions")
    assert request["json"]["model"] == "openrouter/free"
    assert request["headers"]["Authorization"] == "Bearer secret"


def test_429_retries_then_uses_only_configured_fallback():
    session = FakeSession([
        FakeResponse(429, {"error": {"message": "rate limited"}}),
        FakeResponse(429, {"error": {"message": "still limited"}}),
        FakeResponse(200, {
            "model": "vendor/fallback:free",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }),
    ])
    waits = []
    provider = OpenRouterProvider(
        api_key="secret",
        fallback_models="vendor/fallback:free",
        max_retries=2,
        session=session,
        sleep=waits.append,
    )

    result = provider.generate(prompt="hello", model="openrouter/free")

    assert result.model == "vendor/fallback:free"
    assert [call[1]["json"]["model"] for call in session.calls] == [
        "openrouter/free", "openrouter/free", "vendor/fallback:free"
    ]
    assert waits == [1]


def test_permanent_error_fails_without_fallback_or_key_leak():
    session = FakeSession([FakeResponse(401, {"error": {"message": "invalid key"}})])
    provider = OpenRouterProvider(
        api_key="super-secret-key",
        fallback_models="paid/model",
        session=session,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate(prompt="hello", model="openrouter/free")

    assert exc_info.value.status_code == 401
    assert "super-secret-key" not in str(exc_info.value)
    assert len(session.calls) == 1


def test_network_errors_are_bounded():
    session = FakeSession([
        requests.Timeout("contains-super-secret-key"),
        requests.Timeout("contains-super-secret-key"),
    ])
    provider = OpenRouterProvider(
        api_key="super-secret-key",
        max_retries=2,
        session=session,
        sleep=lambda _: None,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate(prompt="hello", model="openrouter/free")

    assert "super-secret-key" not in str(exc_info.value)
    assert len(session.calls) == 2


def test_malformed_completion_is_rejected():
    provider = OpenRouterProvider(
        api_key="secret",
        session=FakeSession([FakeResponse(200, {"choices": []})]),
    )
    with pytest.raises(LLMProviderError, match="malformed"):
        provider.generate(prompt="hello", model="openrouter/free")
