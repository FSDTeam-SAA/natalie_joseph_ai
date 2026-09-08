from __future__ import annotations

import hashlib
import uuid

from sqlalchemy import select, text

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

        digest = hashlib.blake2b(
            f"{user_id}:{companion_id}".encode(), digest_size=8
        ).digest()
        lock_key = int.from_bytes(digest, byteorder="big", signed=True)
        await self.session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )
        # Another transaction may have created the row while this request waited.
        existing = await self.get_for_user_and_companion(user_id, companion_id)
        if existing is not None:
            return existing

        context = RelationshipContext(user_id=user_id, companion_id=companion_id)
        return await self.add(context)
