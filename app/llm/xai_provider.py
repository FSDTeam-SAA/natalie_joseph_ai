"""xAI implementation of :class:`LLMProvider` using the Responses API.

xAI exposes an OpenAI-compatible Responses endpoint at ``https://api.x.ai/v1``.
Keeping this adapter behind the application's provider interface lets chat and
memory code use Grok without depending on xAI- or OpenAI-specific response types.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import APIError, AsyncOpenAI

from app.core.exceptions import ProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResult, LLMStructuredResult, LLMUsage

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})


class XAIProvider(LLMProvider):
    """Generate user-visible responses with xAI's OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_XAI_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @staticmethod
    def _to_input(messages: list[LLMMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    @staticmethod
    def _usage(response: Any) -> LLMUsage:
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        return LLMUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
            cost_usd_ticks=getattr(usage, "cost_in_usd_ticks", None),
        )

    @staticmethod
    def _output_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise TypeError("response output_text is empty or not a string")
        return text

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
        kwargs: dict[str, Any] = {"model": model, "input": self._to_input(messages)}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if cache_key is not None:
            kwargs["prompt_cache_key"] = cache_key
        if reasoning_effort is not None:
            if reasoning_effort not in _REASONING_EFFORTS:
                raise ValueError("Unsupported xAI reasoning effort.")
            kwargs["reasoning"] = {"effort": reasoning_effort}

        try:
            response = await self._client.responses.create(**kwargs)
        except APIError as exc:
            # Provider exception text can contain request details. Preserve it only
            # as the internal cause; the application-facing message stays stable.
            raise ProviderError("xAI generation request failed.") from exc

        try:
            text = self._output_text(response)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProviderError("xAI returned an invalid generation response.") from exc

        return LLMResult(
            text=text,
            usage=self._usage(response),
            model=getattr(response, "model", None) or model,
            raw_response_id=getattr(response, "id", None),
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
            if reasoning_effort not in _REASONING_EFFORTS:
                raise ValueError("Unsupported xAI reasoning effort.")
            kwargs["reasoning"] = {"effort": reasoning_effort}

        try:
            stream = await self._client.responses.create(**kwargs)
            async for event in stream:
                # xAI models can emit reasoning events. The provider contract permits
                # only user-visible output text to leave this boundary.
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str) and delta:
                        yield delta
        except APIError as exc:
            raise ProviderError("xAI streaming request failed.") from exc

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        schema_name: str,
        json_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> LLMStructuredResult:
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

        try:
            response = await self._client.responses.create(**kwargs)
        except APIError as exc:
            raise ProviderError("xAI structured generation request failed.") from exc

        try:
            data = json.loads(self._output_text(response))
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            raise ProviderError("xAI structured generation returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise ProviderError("xAI structured generation returned a non-object JSON value.")

        return LLMStructuredResult(
            data=data,
            usage=self._usage(response),
            model=getattr(response, "model", None) or model,
            raw_response_id=getattr(response, "id", None),
        )
