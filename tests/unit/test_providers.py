"""
Unit tests for Phase 4 provider implementations.

These mock the OpenAI SDK client boundary (AsyncOpenAI.responses /
.embeddings / .moderations) rather than making real network calls —
this environment has no network access to api.openai.com, and the
configured model names (gpt-5.6-terra etc.) are the product owner's
own naming, not real OpenAI models, so a live call would fail
regardless of network access. The mock response shapes below were
verified against the actual installed openai==3.3.1 SDK types
(Response.output_text, Response.usage, Moderation.categories, etc.)
rather than guessed — see the docstrings in the provider files for
exactly what was inspected.

These tests validate OUR wrapper logic — request construction,
response parsing, error translation — not OpenAI's API itself.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.exceptions import ProviderError
from app.embeddings.openai_embeddings import OpenAIEmbeddingProvider
from app.llm.base import LLMMessage
from app.llm.openai_provider import OpenAIProvider
from app.moderation.openai_moderation import OpenAIModerationProvider


def _fake_response(text: str, *, model: str = "gpt-5.6-terra", response_id: str = "resp_123"):
    return SimpleNamespace(
        output_text=text,
        usage=SimpleNamespace(input_tokens=42, output_tokens=17),
        model=model,
        id=response_id,
    )


class TestOpenAIProviderGenerate:
    async def test_generate_returns_text_and_usage(self) -> None:
        provider = OpenAIProvider(api_key="sk-fake")
        fake_response = _fake_response("Hello there!")

        with patch.object(
            provider._client.responses, "create", new=AsyncMock(return_value=fake_response)
        ) as mock_create:
            result = await provider.generate(
                [LLMMessage(role="user", content="Hi")], model="gpt-5.6-terra"
            )

        assert result.text == "Hello there!"
        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 17
        assert result.model == "gpt-5.6-terra"
        assert result.raw_response_id == "resp_123"

        # Confirm the input was translated into the simple role/content
        # shape confirmed against EasyInputMessageParam.
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["input"] == [{"role": "user", "content": "Hi"}]
        assert call_kwargs["model"] == "gpt-5.6-terra"

    async def test_generate_passes_optional_params(self) -> None:
        provider = OpenAIProvider(api_key="sk-fake")
        fake_response = _fake_response("ok")

        with patch.object(
            provider._client.responses, "create", new=AsyncMock(return_value=fake_response)
        ) as mock_create:
            await provider.generate(
                [LLMMessage(role="user", content="Hi")],
                model="gpt-5.6-terra",
                temperature=0.7,
                max_output_tokens=500,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_output_tokens"] == 500

    async def test_generate_wraps_api_errors(self) -> None:
        from openai import APIError

        provider = OpenAIProvider(api_key="sk-fake")

        fake_request = SimpleNamespace(method="POST", url="https://api.openai.com/v1/responses")
        api_error = APIError("boom", request=fake_request, body=None)

        with patch.object(
            provider._client.responses, "create", new=AsyncMock(side_effect=api_error)
        ):
            with pytest.raises(ProviderError):
                await provider.generate(
                    [LLMMessage(role="user", content="Hi")], model="gpt-5.6-terra"
                )


class TestOpenAIProviderStructured:
    async def test_generate_structured_parses_json(self) -> None:
        provider = OpenAIProvider(api_key="sk-fake")
        fake_response = _fake_response(
            '{"should_store": true, "memory_type": "preference", "key": "favorite_cuisine", '
            '"value": "Japanese food", "confidence": 0.95, "reason": "Stable preference"}'
        )

        with patch.object(
            provider._client.responses, "create", new=AsyncMock(return_value=fake_response)
        ) as mock_create:
            result = await provider.generate_structured(
                [LLMMessage(role="user", content="I love Japanese food")],
                model="gpt-5.6-luna",
                schema_name="memory_extraction",
                json_schema={"type": "object", "properties": {}},
            )

        assert result.data["should_store"] is True
        assert result.data["key"] == "favorite_cuisine"

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["text"]["format"]["type"] == "json_schema"
        assert call_kwargs["text"]["format"]["name"] == "memory_extraction"
        assert call_kwargs["text"]["format"]["strict"] is True

    async def test_generate_structured_raises_on_invalid_json(self) -> None:
        provider = OpenAIProvider(api_key="sk-fake")
        fake_response = _fake_response("not valid json {{{")

        with patch.object(
            provider._client.responses, "create", new=AsyncMock(return_value=fake_response)
        ):
            with pytest.raises(ProviderError):
                await provider.generate_structured(
                    [LLMMessage(role="user", content="test")],
                    model="gpt-5.6-luna",
                    schema_name="test_schema",
                    json_schema={"type": "object"},
                )


class TestOpenAIProviderStream:
    async def test_generate_stream_yields_only_text_deltas(self) -> None:
        provider = OpenAIProvider(api_key="sk-fake")

        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="Hel"),
            SimpleNamespace(type="response.output_text.delta", delta="lo"),
            # Reasoning content must NEVER be yielded (spec Section 63).
            SimpleNamespace(type="response.reasoning_text.delta", delta="secret thoughts"),
            SimpleNamespace(type="response.output_text.delta", delta="!"),
            SimpleNamespace(type="response.completed"),
        ]

        async def fake_stream():
            for e in events:
                yield e

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(return_value=fake_stream()),
        ):
            chunks = [
                chunk
                async for chunk in provider.generate_stream(
                    [LLMMessage(role="user", content="Hi")], model="gpt-5.6-terra"
                )
            ]

        assert chunks == ["Hel", "lo", "!"]
        assert "secret thoughts" not in chunks


class TestOpenAIEmbeddingProvider:
    async def test_embed_text_returns_vector(self) -> None:
        provider = OpenAIEmbeddingProvider(api_key="sk-fake")
        fake_response = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
            model="text-embedding-3-small",
        )

        with patch.object(
            provider._client.embeddings, "create", new=AsyncMock(return_value=fake_response)
        ) as mock_create:
            result = await provider.embed_text("hello", model="text-embedding-3-small")

        assert result.vector == [0.1, 0.2, 0.3]
        assert result.dimensions == 3
        assert result.model == "text-embedding-3-small"
        mock_create.assert_called_once_with(input="hello", model="text-embedding-3-small")


class TestOpenAIModerationProvider:
    async def test_moderate_text_returns_flagged_result(self) -> None:
        provider = OpenAIModerationProvider(api_key="sk-fake")

        fake_categories = SimpleNamespace(model_dump=lambda: {"violence": False, "sexual": False})
        fake_scores = SimpleNamespace(model_dump=lambda: {"violence": 0.001, "sexual": 0.002})
        fake_result = SimpleNamespace(
            flagged=False, categories=fake_categories, category_scores=fake_scores
        )
        fake_response = SimpleNamespace(
            results=[fake_result], model_dump=lambda: {"results": ["..."]}
        )

        with patch.object(
            provider._client.moderations, "create", new=AsyncMock(return_value=fake_response)
        ) as mock_create:
            result = await provider.moderate_text("hello", model="omni-moderation-latest")

        assert result.flagged is False
        assert result.categories["violence"] is False
        mock_create.assert_called_once_with(input="hello", model="omni-moderation-latest")

    async def test_moderate_multimodal_builds_correct_input_items(self) -> None:
        provider = OpenAIModerationProvider(api_key="sk-fake")

        fake_categories = SimpleNamespace(model_dump=lambda: {})
        fake_scores = SimpleNamespace(model_dump=lambda: {})
        fake_result = SimpleNamespace(
            flagged=True, categories=fake_categories, category_scores=fake_scores
        )
        fake_response = SimpleNamespace(results=[fake_result], model_dump=lambda: {})

        with patch.object(
            provider._client.moderations, "create", new=AsyncMock(return_value=fake_response)
        ) as mock_create:
            result = await provider.moderate_multimodal(
                text="check this",
                image_url="https://example.com/img.png",
                model="omni-moderation-latest",
            )

        assert result.flagged is True
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["input"] == [
            {"type": "text", "text": "check this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]

    async def test_moderate_multimodal_requires_at_least_one_input(self) -> None:
        from app.core.exceptions import ValidationError

        provider = OpenAIModerationProvider(api_key="sk-fake")

        with pytest.raises(ValidationError):
            await provider.moderate_multimodal(model="omni-moderation-latest")