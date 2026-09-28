"""OpenRouter provider using its OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import requests

from clippyme.llm.base import LLMProviderError, LLMResponse

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_RETRYABLE_STATUS = frozenset({404, 408, 409, 425, 429, 500, 502, 503, 504})


def parse_model_list(value: str | Sequence[str] | None) -> list[str]:
    if not value:
        return []
    values = value.split(",") if isinstance(value, str) else value
    return [str(item).strip() for item in values if str(item).strip()]


class OpenRouterProvider:
    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        fallback_models: str | Sequence[str] | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise LLMProviderError(
                "OPENROUTER_API_KEY is not configured",
                provider=self.name,
            )
        self._api_key = api_key
        self._fallback_models = parse_model_list(fallback_models)
        self._timeout = max(1.0, min(float(timeout), 600.0))
        self._max_retries = max(1, min(int(max_retries), 10))
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._sleep = sleep

    def _models(self, primary: str) -> list[str]:
        models: list[str] = []
        for model in [primary, *self._fallback_models]:
            if model and model not in models:
                models.append(model)
        return models

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = error.get("message") if isinstance(error, dict) else None
            if message:
                return str(message)[:500]
        except (ValueError, TypeError):
            pass
        return f"HTTP {response.status_code}"

    @staticmethod
    def _normalize(payload: Mapping[str, Any], requested_model: str) -> LLMResponse:
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                "OpenRouter returned a malformed completion response",
                provider="openrouter",
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError(
                "OpenRouter returned an empty completion",
                provider="openrouter",
            )
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        raw_cost = usage.get("cost")
        try:
            cost = float(raw_cost) if raw_cost is not None else None
        except (TypeError, ValueError):
            cost = None
        model = str(payload.get("model") or requested_model)
        if cost is None and (requested_model == "openrouter/free" or model.endswith(":free")):
            cost = 0.0
        metadata = {
            key: payload[key]
            for key in ("id", "created", "openrouter_metadata")
            if key in payload
        }
        return LLMResponse(
            content=content,
            model=model,
            provider="openrouter",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
            raw_metadata=metadata,
        )

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        response_schema: dict | None = None,
    ) -> LLMResponse:
        models = self._models(model)
        if not models:
            raise LLMProviderError("No OpenRouter model configured", provider=self.name)

        last_error: LLMProviderError | None = None
        for model_name in models:
            body: dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if response_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "clippyme_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
            for attempt in range(self._max_retries):
                try:
                    response = self._session.post(
                        f"{self._base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                            "X-Title": "ClippyMe",
                        },
                        json=body,
                        timeout=self._timeout,
                    )
                except requests.RequestException as exc:
                    last_error = LLMProviderError(
                        f"OpenRouter request failed: {type(exc).__name__}",
                        provider=self.name,
                        retryable=True,
                    )
                else:
                    if response.ok:
                        try:
                            payload = response.json()
                        except (requests.JSONDecodeError, json.JSONDecodeError, ValueError) as exc:
                            raise LLMProviderError(
                                "OpenRouter returned invalid JSON",
                                provider=self.name,
                            ) from exc
                        return self._normalize(payload, model_name)

                    status = response.status_code
                    retryable = status in _RETRYABLE_STATUS
                    safe_message = self._error_message(response).replace(
                        self._api_key, "***REDACTED***"
                    )
                    last_error = LLMProviderError(
                        f"OpenRouter {safe_message}",
                        provider=self.name,
                        status_code=status,
                        retryable=retryable,
                        rate_limited=status == 429,
                    )
                    if not retryable:
                        raise last_error

                if attempt + 1 < self._max_retries:
                    self._sleep(min(2 ** attempt, 8))
            # The configured model exhausted its bounded attempts. Move only
            # to the next explicitly configured fallback model.
        if last_error is None:  # defensive; models/attempts are bounded above
            last_error = LLMProviderError("OpenRouter request failed", provider=self.name)
        raise last_error
