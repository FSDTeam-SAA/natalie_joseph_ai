"""Unit tests for the user-facing LLM provider failover adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

from app.core.exceptions import ProviderError
from app.llm.base import LLMMessage, LLMResult, LLMUsage
from app.llm.failover_provider import FailoverLLMProvider


def _result(model: str) -> LLMResult:
    return LLMResult(text="Hello", usage=LLMUsage(1, 1), model=model)


async def test_uses_openai_when_xai_is_not_configured() -> None:
    fallback = AsyncMock()
    fallback.generate.return_value = _result("gpt-test")
    provider = FailoverLLMProvider(
        primary=None, fallback=fallback, fallback_model="gpt-test"
    )

    result = await provider.generate([LLMMessage(role="user", content="Hi")], model="grok-test")

    assert result.model == "gpt-test"
    fallback.generate.assert_awaited_once()
    assert fallback.generate.await_args.kwargs["model"] == "gpt-test"


async def test_retries_with_openai_when_xai_fails() -> None:
    primary = AsyncMock()
    primary.generate.side_effect = ProviderError("xAI generation request failed.")
    fallback = AsyncMock()
    fallback.generate.return_value = _result("gpt-test")
    provider = FailoverLLMProvider(
        primary=primary, fallback=fallback, fallback_model="gpt-test"
    )

    result = await provider.generate([LLMMessage(role="user", content="Hi")], model="grok-test")

    assert result.model == "gpt-test"
    primary.generate.assert_awaited_once()
    fallback.generate.assert_awaited_once()
    assert fallback.generate.await_args.kwargs["model"] == "gpt-test"
