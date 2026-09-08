"""
OpenAI implementation of ModerationProvider.

Verified against openai-python 3.3.1's real SDK surface:
  client.moderations.create(input=..., model=...) -> ModerationCreateResponse
  response.results[0] -> Moderation(flagged, categories, category_scores, ...)

  Multimodal input items:
    {"type": "text", "text": "..."}
    {"type": "image_url", "image_url": {"url": "..."}}
  (url may be a hosted https URL or a base64 data URL)
"""

from __future__ import annotations

from typing import Any

from openai import APIError, AsyncOpenAI

from app.core.exceptions import ProviderError, ValidationError
from app.moderation.base import ModerationProvider, ModerationResult


class OpenAIModerationProvider(ModerationProvider):
    def __init__(
        self, api_key: str, *, timeout_seconds: float = 45.0, max_retries: int = 2
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )

    @staticmethod
    def _to_result(raw: Any) -> ModerationResult:
        result = raw.results[0]
        return ModerationResult(
            flagged=result.flagged,
            categories=result.categories.model_dump(),
            category_scores=result.category_scores.model_dump(),
            raw_response=raw.model_dump(),
        )

    async def moderate_text(self, text: str, *, model: str) -> ModerationResult:
        try:
            response = await self._client.moderations.create(input=text, model=model)
        except APIError as exc:
            raise ProviderError("OpenAI text moderation failed.") from exc

        return self._to_result(response)

    async def moderate_multimodal(
        self, *, text: str | None = None, image_url: str | None = None, model: str
    ) -> ModerationResult:
        if text is None and image_url is None:
            raise ValidationError(
                "moderate_multimodal() requires at least one of text or image_url."
            )

        input_items: list[dict[str, Any]] = []
        if text is not None:
            input_items.append({"type": "text", "text": text})
        if image_url is not None:
            input_items.append({"type": "image_url", "image_url": {"url": image_url}})

        try:
            response = await self._client.moderations.create(input=input_items, model=model)
        except APIError as exc:
            raise ProviderError("OpenAI multimodal moderation failed.") from exc

        return self._to_result(response)
