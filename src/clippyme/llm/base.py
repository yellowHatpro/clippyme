"""Small provider contract and normalized response/error types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class LLMProviderError(RuntimeError):
    """Safe provider failure suitable for logs and pipeline fallbacks."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
        rate_limited: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.rate_limited = rate_limited


class LLMProvider(Protocol):
    name: str

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        response_schema: dict | None = None,
    ) -> LLMResponse:
        ...
