"""
OpenAI implementation of LLMProvider, using the OpenAI Responses API
per spec Section 6 ("Use the OpenAI Responses API for model
interaction").

Verified against openai-python 3.3.1's actual SDK surface (installed
and inspected directly) rather than assumed from memory:
  - client.responses.create(input=[{"role":..., "content":...}, ...], model=...)
  - response.output_text — convenience property aggregating all
    output_text content blocks
  - response.usage.input_tokens / .output_tokens
  - Structured output via text={"format": {"type": "json_schema", ...}}
  - Streaming via stream=True, yielding ResponseStreamEvent objects;
    text chunks arrive as events with type "response.output_text.delta"
    and a .delta string attribute (ResponseTextDeltaEvent)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import APIError, AsyncOpenAI

from app.core.exceptions import ProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResult, LLMStructuredResult, LLMUsage


class OpenAIProvider(LLMProvider):
    def __init__(
        self, api_key: str, *, timeout_seconds: float = 45.0, max_retries: int = 2
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )

    @staticmethod
    def _to_input(messages: list[LLMMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        cache_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        try:
            kwargs: dict[str, Any] = {"model": model, "input": self._to_input(messages)}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens
            if cache_key is not None:
                kwargs["prompt_cache_key"] = cache_key
            if reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": reasoning_effort}

            response = await self._client.responses.create(**kwargs)
        except APIError as exc:
            raise ProviderError("OpenAI generation failed.") from exc

        return LLMResult(
            text=response.output_text,
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens if response.usage else 0,
                output_tokens=response.usage.output_tokens if response.usage else 0,
                cached_input_tokens=(
                    getattr(
                        getattr(response.usage, "input_tokens_details", None),
                        "cached_tokens",
                        0,
                    )
                    if response.usage
                    else 0
                ),
                reasoning_tokens=(
                    getattr(
                        getattr(response.usage, "output_tokens_details", None),
                        "reasoning_tokens",
                        0,
                    )
                    if response.usage
                    else 0
                ),
            ),
            model=response.model,
            raw_response_id=response.id,
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        cache_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[str]:
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "input": self._to_input(messages),
                "stream": True,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens
            if cache_key is not None:
                kwargs["prompt_cache_key"] = cache_key
            if reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": reasoning_effort}

            stream = await self._client.responses.create(**kwargs)

            async for event in stream:
                # Only forward user-visible output text deltas — per
                # spec Section 19/63, never stream reasoning content,
                # chain-of-thought, or other internal event types.
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta
        except APIError as exc:
            raise ProviderError("OpenAI streaming generation failed.") from exc

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        schema_name: str,
        json_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> LLMStructuredResult:
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "input": self._to_input(messages),
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": json_schema,
                        "strict": True,
                    }
                },
            }
            if temperature is not None:
                kwargs["temperature"] = temperature

            response = await self._client.responses.create(**kwargs)
        except APIError as exc:
            raise ProviderError("OpenAI structured generation failed.") from exc

        try:
            data = json.loads(response.output_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProviderError(
                "OpenAI structured generation returned invalid JSON."
            ) from exc

        return LLMStructuredResult(
            data=data,
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens if response.usage else 0,
                output_tokens=response.usage.output_tokens if response.usage else 0,
                cached_input_tokens=(
                    getattr(
                        getattr(response.usage, "input_tokens_details", None),
                        "cached_tokens",
                        0,
                    )
                    if response.usage
                    else 0
                ),
                reasoning_tokens=(
                    getattr(
                        getattr(response.usage, "output_tokens_details", None),
                        "reasoning_tokens",
                        0,
                    )
                    if response.usage
                    else 0
                ),
            ),
            model=response.model,
            raw_response_id=response.id,
        )
