from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db.models.relationship_context import RelationshipContext
from app.repositories.base_repository import BaseRepository


class RelationshipRepository(BaseRepository[RelationshipContext]):
    model = RelationshipContext

    async def get_for_user_and_companion(
        self, user_id: uuid.UUID, companion_id: uuid.UUID
    ) -> RelationshipContext | None:
        result = await self.session.execute(
            select(RelationshipContext).where(
                RelationshipContext.user_id == user_id,
                RelationshipContext.companion_id == companion_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_or_create(
        self, user_id: uuid.UUID, companion_id: uuid.UUID
    ) -> RelationshipContext:
        existing = await self.get_for_user_and_companion(user_id, companion_id)
        if existing is not None:
            return existing

        context = RelationshipContext(user_id=user_id, companion_id=companion_id)
        return await self.add(context)