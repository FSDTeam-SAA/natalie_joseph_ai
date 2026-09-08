"""Focused unit tests for the xAI Responses API provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import APIError

from app.core.exceptions import ProviderError
from app.llm.base import LLMMessage
from app.llm.xai_provider import DEFAULT_XAI_BASE_URL, XAIProvider


def _fake_response(
    text: object,
    *,
    model: str = "grok-4.6",
    response_id: str = "resp_xai_123",
    usage: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=text,
        usage=usage or SimpleNamespace(input_tokens=23, output_tokens=11),
        model=model,
        id=response_id,
    )


def _api_error(message: str = "sensitive upstream detail") -> APIError:
    request = SimpleNamespace(method="POST", url=f"{DEFAULT_XAI_BASE_URL}/responses")
    return APIError(message, request=request, body=None)


class TestXAIProviderConfiguration:
    def test_configures_openai_client_for_xai(self) -> None:
        with patch("app.llm.xai_provider.AsyncOpenAI") as client_class:
            XAIProvider(
                api_key="xai-fake",
                base_url="https://xai.example/v1",
                timeout_seconds=12.5,
                max_retries=4,
            )

        client_class.assert_called_once_with(
            api_key="xai-fake",
            base_url="https://xai.example/v1",
            timeout=12.5,
            max_retries=4,
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"api_key": " "}, "api_key must not be empty"),
            ({"api_key": "key", "base_url": " "}, "base_url must not be empty"),
            (
                {"api_key": "key", "timeout_seconds": 0},
                "timeout_seconds must be greater than zero",
            ),
            ({"api_key": "key", "max_retries": -1}, "max_retries must not be negative"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            XAIProvider(**kwargs)  # type: ignore[arg-type]


class TestXAIProviderGenerate:
    async def test_returns_text_usage_and_response_metadata(self) -> None:
        provider = XAIProvider(api_key="xai-fake")

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(
                return_value=_fake_response(
                    "Hello from Grok",
                    usage=SimpleNamespace(
                        input_tokens=23,
                        output_tokens=11,
                        input_tokens_details=SimpleNamespace(cached_tokens=7),
                        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
                        cost_in_usd_ticks=1234,
                    ),
                )
            ),
        ) as create:
            result = await provider.generate(
                [
                    LLMMessage(role="system", content="You are Lina."),
                    LLMMessage(role="user", content="Hello"),
                ],
                model="grok-4.6",
                temperature=0.4,
                max_output_tokens=256,
                cache_key="conversation:123",
                reasoning_effort="low",
            )

        assert result.text == "Hello from Grok"
        assert result.usage.input_tokens == 23
        assert result.usage.output_tokens == 11
        assert result.usage.cached_input_tokens == 7
        assert result.usage.reasoning_tokens == 3
        assert result.usage.cost_usd_ticks == 1234
        assert result.model == "grok-4.6"
        assert result.raw_response_id == "resp_xai_123"
        assert create.call_args.kwargs == {
            "model": "grok-4.6",
            "input": [
                {"role": "system", "content": "You are Lina."},
                {"role": "user", "content": "Hello"},
            ],
            "temperature": 0.4,
            "max_output_tokens": 256,
            "prompt_cache_key": "conversation:123",
            "reasoning": {"effort": "low"},
        }

    async def test_omits_unspecified_optional_parameters(self) -> None:
        provider = XAIProvider(api_key="xai-fake")

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(return_value=_fake_response("ok")),
        ) as create:
            await provider.generate(
                [LLMMessage(role="user", content="Hello")], model="grok-4.6"
            )

        assert "temperature" not in create.call_args.kwargs
        assert "max_output_tokens" not in create.call_args.kwargs

    async def test_sanitizes_api_errors(self) -> None:
        provider = XAIProvider(api_key="xai-fake")

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(side_effect=_api_error()),
        ):
            with pytest.raises(ProviderError) as raised:
                await provider.generate(
                    [LLMMessage(role="user", content="Hello")], model="grok-4.6"
                )

        assert str(raised.value) == "xAI generation request failed."
        assert "sensitive" not in str(raised.value)

    async def test_sanitizes_malformed_responses(self) -> None:
        provider = XAIProvider(api_key="xai-fake")

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(return_value=_fake_response(None)),
        ):
            with pytest.raises(
                ProviderError, match=r"^xAI returned an invalid generation response\.$"
            ):
                await provider.generate(
                    [LLMMessage(role="user", content="Hello")], model="grok-4.6"
                )


class TestXAIProviderStream:
    async def test_yields_only_user_visible_text_deltas(self) -> None:
        provider = XAIProvider(api_key="xai-fake")
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="Hel"),
            SimpleNamespace(type="response.reasoning_text.delta", delta="private reasoning"),
            SimpleNamespace(type="response.output_text.delta", delta="lo"),
            SimpleNamespace(type="response.output_text.delta", delta=""),
            SimpleNamespace(type="response.completed"),
        ]

        async def fake_stream():
            for event in events:
                yield event

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(return_value=fake_stream()),
        ) as create:
            chunks = [
                chunk
                async for chunk in provider.generate_stream(
                    [LLMMessage(role="user", content="Hello")],
                    model="grok-4.6",
                    max_output_tokens=64,
                    cache_key="conversation:stream",
                    reasoning_effort="low",
                )
            ]

        assert chunks == ["Hel", "lo"]
        assert "private reasoning" not in chunks
        assert create.call_args.kwargs["stream"] is True
        assert create.call_args.kwargs["max_output_tokens"] == 64
        assert create.call_args.kwargs["prompt_cache_key"] == "conversation:stream"
        assert create.call_args.kwargs["reasoning"] == {"effort": "low"}

    async def test_sanitizes_stream_api_errors(self) -> None:
        provider = XAIProvider(api_key="xai-fake")

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(side_effect=_api_error()),
        ):
            with pytest.raises(ProviderError) as raised:
                _ = [
                    chunk
                    async for chunk in provider.generate_stream(
                        [LLMMessage(role="user", content="Hello")], model="grok-4.6"
                    )
                ]

        assert str(raised.value) == "xAI streaming request failed."
        assert "sensitive" not in str(raised.value)


class TestXAIProviderStructured:
    async def test_requests_strict_schema_and_parses_object(self) -> None:
        provider = XAIProvider(api_key="xai-fake")
        schema = {
            "type": "object",
            "properties": {"remember": {"type": "boolean"}},
            "required": ["remember"],
            "additionalProperties": False,
        }

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(return_value=_fake_response('{"remember": true}')),
        ) as create:
            result = await provider.generate_structured(
                [LLMMessage(role="user", content="Remember that I like tea")],
                model="grok-4.6",
                schema_name="memory_decision",
                json_schema=schema,
                temperature=0.1,
            )

        assert result.data == {"remember": True}
        assert result.usage.input_tokens == 23
        assert create.call_args.kwargs["text"] == {
            "format": {
                "type": "json_schema",
                "name": "memory_decision",
                "schema": schema,
                "strict": True,
            }
        }
        assert create.call_args.kwargs["temperature"] == 0.1

    @pytest.mark.parametrize("output", ["not json", "[]", '"string"'])
    async def test_rejects_invalid_or_non_object_json(self, output: str) -> None:
        provider = XAIProvider(api_key="xai-fake")

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(return_value=_fake_response(output)),
        ):
            with pytest.raises(ProviderError) as raised:
                await provider.generate_structured(
                    [LLMMessage(role="user", content="test")],
                    model="grok-4.6",
                    schema_name="test_schema",
                    json_schema={"type": "object"},
                )

        assert output not in str(raised.value)

    async def test_sanitizes_structured_api_errors(self) -> None:
        provider = XAIProvider(api_key="xai-fake")

        with patch.object(
            provider._client.responses,
            "create",
            new=AsyncMock(side_effect=_api_error()),
        ):
            with pytest.raises(ProviderError) as raised:
                await provider.generate_structured(
                    [LLMMessage(role="user", content="test")],
                    model="grok-4.6",
                    schema_name="test_schema",
                    json_schema={"type": "object"},
                )

        assert str(raised.value) == "xAI structured generation request failed."
        assert "sensitive" not in str(raised.value)
