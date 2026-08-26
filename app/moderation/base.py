"""
ModerationProvider interface.

Spec discrepancy note: Section 6 specifies two methods —
`moderate_text()` and `moderate_multimodal()`. Section 36 later
describes the same class with a single `moderate()` method. This
implementation follows Section 6 (the more detailed, specific
definition) since it distinguishes text-only from multimodal
moderation, which matters given the spec's own Section 26 requirement
to cover "sexual content involving minors" — image moderation is a
distinct capability from text moderation and callers should not be
able to accidentally skip it by calling a single generic method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModerationResult:
    flagged: bool
    categories: dict[str, bool]
    category_scores: dict[str, float]
    raw_response: dict[str, Any]


class ModerationProvider(ABC):
    @abstractmethod
    async def moderate_text(self, text: str, *, model: str) -> ModerationResult:
        raise NotImplementedError

    @abstractmethod
    async def moderate_multimodal(
        self, *, text: str | None = None, image_url: str | None = None, model: str
    ) -> ModerationResult:
        """
        At least one of `text` or `image_url` must be provided.
        `image_url` may be a data URL (base64) or a hosted https URL,
        per OpenAI's multimodal moderation input format.
        """
        raise NotImplementedError