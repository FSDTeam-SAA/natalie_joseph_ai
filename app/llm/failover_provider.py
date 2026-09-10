"""Fail over user-facing generation from xAI to OpenAI when xAI is unavailable."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.core.exceptions import ProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResult, LLMStructuredResult


class FailoverLLMProvider(LLMProvider):
    """Use the preferred provider, then retry once through the fallback provider.

    A fallback is attempted only for :class:`ProviderError`, which is the
    provider boundary's normalized failure type. Application validation and
    application validation errors therefore never cause a second model request.
    """

    def __init__(
        self,
        *,
        primary: LLMProvider | None,
        fallback: LLMProvider | None,
        fallback_model: str,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.fallback_model = fallback_model

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
        kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "cache_key": cache_key,
            "reasoning_effort": reasoning_effort,
        }
        if self.primary is not None:
            try:
                return await self.primary.generate(messages, model=model, **kwargs)
            except ProviderError:
                if self.fallback is None:
                    raise

        if self.fallback is None:
            raise ProviderError("No conversation-generation provider is configured.")
        return await self.fallback.generate(messages, model=self.fallback_model, **kwargs)

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
        kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "cache_key": cache_key,
            "reasoning_effort": reasoning_effort,
        }
        if self.primary is not None:
            emitted = False
            try:
                async for chunk in self.primary.generate_stream(messages, model=model, **kwargs):
                    emitted = True
                    yield chunk
                return
            except ProviderError:
                # A response already sent to a client cannot safely be replaced.
                if emitted or self.fallback is None:
                    raise

        if self.fallback is None:
            raise ProviderError("No conversation-generation provider is configured.")
        async for chunk in self.fallback.generate_stream(
            messages, model=self.fallback_model, **kwargs
        ):
            yield chunk

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        schema_name: str,
        json_schema: dict,
        temperature: float | None = None,
    ) -> LLMStructuredResult:
        kwargs = {
            "schema_name": schema_name,
            "json_schema": json_schema,
            "temperature": temperature,
        }
        if self.primary is not None:
            try:
                return await self.primary.generate_structured(messages, model=model, **kwargs)
            except ProviderError:
                if self.fallback is None:
                    raise

        if self.fallback is None:
            raise ProviderError("No conversation-generation provider is configured.")
        return await self.fallback.generate_structured(
            messages, model=self.fallback_model, **kwargs
        )
