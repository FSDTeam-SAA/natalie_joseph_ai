"""
Companion service.

Per spec Section 9: the engine loads companion configuration; this
service is the one place that knows how the JSONB config blocks map
onto the frontend-safe response shape. If the JSON config structure
ever changes, only this file needs to change — not the API routes.
"""

from __future__ import annotations

import uuid

from app.core.exceptions import NotFoundError
from app.db.models.companion import Companion
from app.repositories.companion_repository import CompanionRepository
from app.schemas.companion import CompanionDetail, CompanionSummary


class CompanionService:
    def __init__(self, companion_repo: CompanionRepository) -> None:
        self.companion_repo = companion_repo

    @staticmethod
    def _to_summary(companion: Companion) -> CompanionSummary:
        personality = companion.personality_config or {}
        background = companion.background_config or {}
        return CompanionSummary(
            id=companion.id,
            slug=companion.slug,
            name=companion.name,
            title=personality.get("title"),
            traits=personality.get("traits", []),
            location=background.get("location"),
            occupation=background.get("occupation"),
        )

    @staticmethod
    def _to_detail(companion: Companion) -> CompanionDetail:
        personality = companion.personality_config or {}
        communication = companion.communication_config or {}
        background = companion.background_config or {}
        interests = companion.interest_config or {}
        visual = companion.visual_config or {}

        return CompanionDetail(
            id=companion.id,
            slug=companion.slug,
            name=companion.name,
            title=personality.get("title"),
            about=personality.get("about"),
            essence=personality.get("essence"),
            traits=personality.get("traits", []),
            location=background.get("location"),
            occupation=background.get("occupation"),
            lifestyle=background.get("lifestyle", []),
            communication_style_traits=communication.get("style_traits", []),
            interests=interests.get("interests", []),
            aesthetic_keywords=visual.get("aesthetic_keywords", []),
            what_you_experience=communication.get("what_you_experience", []),
        )

    async def list_active_companions(self) -> list[CompanionSummary]:
        companions = await self.companion_repo.list_active()
        return [self._to_summary(c) for c in companions]

    async def get_companion_detail(self, companion_id: uuid.UUID) -> CompanionDetail:
        companion = await self.companion_repo.get_by_id(companion_id)
        if companion is None or not companion.active:
            raise NotFoundError(f"Companion {companion_id} not found.")
        return self._to_detail(companion)

    async def get_companion_detail_by_slug(self, slug: str) -> CompanionDetail:
        companion = await self.companion_repo.get_by_slug(slug)
        if companion is None or not companion.active:
            raise NotFoundError(f"Companion '{slug}' not found.")
        return self._to_detail(companion)