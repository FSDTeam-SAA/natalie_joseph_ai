"""
LLMProvider interface (spec Section 6 / 36).

Provider-agnostic: the rest of the application depends on this
interface and the dataclasses below, never on OpenAI SDK types
directly. A future provider (e.g. Gemini) implements this same
interface without any other code changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "developer"]


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResult:
    """Result of a non-streaming generate() call."""

    text: str
    usage: LLMUsage
    model: str
    raw_response_id: str | None = None


@dataclass(frozen=True)
class LLMStructuredResult:
    """Result of a generate_structured() call."""

    data: dict[str, Any]
    usage: LLMUsage
    model: str
    raw_response_id: str | None = None


class LLMProvider(ABC):
    """
    Per spec Section 63: internal reasoning must never be surfaced.
    Implementations must ensure generate_stream() yields only
    user-visible output text — never reasoning/thinking content, even
    if the underlying model produces it.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        """Non-streaming generation. Used by the non-streaming chat endpoint."""
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming generation. Yields incremental user-visible text
        chunks only (spec Section 19/63) — never reasoning content,
        chain-of-thought, or internal metadata.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        schema_name: str,
        json_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> LLMStructuredResult:
        """
        Structured generation constrained to the given JSON schema.
        Used for memory extraction (Phase 7) and emotion/context
        classification (Phase 6), per spec Sections 13 and 17.
        """
        raise NotImplementedError