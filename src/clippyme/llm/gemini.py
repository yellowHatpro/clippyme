"""Gemini adapter preserving ClippyMe's existing retry and pricing behavior."""

from __future__ import annotations

from collections.abc import Sequence

from clippyme.llm.base import LLMProviderError, LLMResponse
from clippyme.pipeline.gemini_request import (
    build_model_chain,
    compute_gemini_cost,
    generate_with_model_fallback,
    is_rate_limit_error,
)


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        fallback_models: str | Sequence[str] | None = None,
        max_retries: int = 3,
        client=None,
    ) -> None:
        if not api_key:
            raise LLMProviderError("GEMINI_API_KEY is not configured", provider=self.name)
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        if isinstance(fallback_models, Sequence) and not isinstance(fallback_models, str):
            fallback_models = ",".join(str(item) for item in fallback_models)
        self._fallback_models = fallback_models
        self._max_retries = max(1, min(int(max_retries), 10))

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        response_schema: dict | None = None,
    ) -> LLMResponse:
        del response_schema  # Existing Gemini path relies on prompt + repair parser.
        chain = build_model_chain(model, self._fallback_models)
        try:
            response, used_model = generate_with_model_fallback(
                self._client,
                prompt,
                chain,
                max_attempts=self._max_retries,
            )
        except Exception as exc:
            raise LLMProviderError(
                f"Gemini request failed: {type(exc).__name__}",
                provider=self.name,
                retryable=True,
                rate_limited=is_rate_limit_error(exc),
            ) from exc

        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        cost = compute_gemini_cost(input_tokens, output_tokens, used_model)
        return LLMResponse(
            content=getattr(response, "text", "") or "",
            model=used_model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost["total_cost"],
            raw_metadata={"model": used_model},
        )
