"""
Memory repository.

Per spec Section 14 ("Never retrieve another user's memories" /
"Tenant/user isolation must exist at the database query layer") and
Section 57: `search_similar` takes user_id as a required parameter and
always filters on it — there is no vector-search method that omits
this filter. Do not add one; user isolation must not depend on every
future caller remembering to filter correctly.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db.models.memory import Memory, MemoryType
from app.repositories.base_repository import BaseRepository


class MemoryRepository(BaseRepository[Memory]):
    model = Memory

    async def search_similar(
        self,
        *,
        user_id: uuid.UUID,
        query_embedding: list[float],
        companion_id: uuid.UUID | None = None,
        top_k: int,
        min_confidence: float = 0.0,
    ) -> list[Memory]:
        """
        Cosine-distance nearest-neighbor search (uses the ivfflat index
        created in the initial migration), scoped to the given user_id
        and, optionally, companion_id, excluding inactive memories.

        Ordering by cosine_distance ascending == most similar first.
        `min_confidence` lets callers apply the confidence threshold at
        the query layer (spec Section 14, "consider confidence") rather
        than filtering in application code after the fact.
        """
        stmt = (
            select(Memory)
            .where(
                Memory.user_id == user_id,
                Memory.active.is_(True),
                Memory.confidence >= min_confidence,
            )
            .order_by(Memory.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        if companion_id is not None:
            stmt = stmt.where(Memory.companion_id == companion_id)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id_for_user(
        self, memory_id: uuid.UUID, user_id: uuid.UUID
    ) -> Memory | None:
        result = await self.session.execute(
            select(Memory).where(Memory.id == memory_id, Memory.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def find_existing_by_key(
        self, *, user_id: uuid.UUID, companion_id: uuid.UUID, key: str
    ) -> Memory | None:
        """
        Used by the dedup/update flow (spec Section 15) to find a prior
        memory with the same key before deciding whether to update it
        in place rather than inserting a duplicate.
        """
        result = await self.session.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.companion_id == companion_id,
                Memory.key == key,
                Memory.active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def list_active_for_user(
        self, user_id: uuid.UUID, *, memory_type: MemoryType | None = None
    ) -> list[Memory]:
        stmt = select(Memory).where(Memory.user_id == user_id, Memory.active.is_(True))
        if memory_type is not None:
            stmt = stmt.where(Memory.memory_type == memory_type)
        result = await self.session.execute(stmt.order_by(Memory.created_at.desc()))
        return list(result.scalars().all())

    async def deactivate(self, memory: Memory) -> Memory:
        memory.active = False
        await self.session.flush()
        return memory