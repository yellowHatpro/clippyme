from __future__ import annotations

from types import SimpleNamespace

from clippyme.llm.gemini import GeminiProvider


class FakeModels:
    def generate_content(self, **kwargs):
        assert kwargs["model"] == "gemini-2.5-flash"
        return SimpleNamespace(
            text="answer",
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=25,
            ),
        )


def test_gemini_adapter_normalizes_existing_sdk_response():
    client = SimpleNamespace(models=FakeModels())
    provider = GeminiProvider(api_key="secret", client=client, fallback_models="")

    result = provider.generate(prompt="hello", model="gemini-2.5-flash")

    assert result.content == "answer"
    assert result.provider == "gemini"
    assert result.model == "gemini-2.5-flash"
    assert result.input_tokens == 100
    assert result.output_tokens == 25
    assert result.estimated_cost is not None
