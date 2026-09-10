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

import hashlib
import uuid

from sqlalchemy import select, text

from app.db.models.memory import Memory, MemoryType
from app.repositories.base_repository import BaseRepository


class MemoryRepository(BaseRepository[Memory]):
    model = Memory

    async def acquire_key_lock(
        self, *, user_id: uuid.UUID, companion_id: uuid.UUID, key: str
    ) -> None:
        """Serialize key-based memory upserts across background workers."""
        digest = hashlib.blake2b(
            f"{user_id}:{companion_id}:{key}".encode(), digest_size=8
        ).digest()
        lock_key = int.from_bytes(digest, byteorder="big", signed=True)
        await self.session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )

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
            )
        )
        return result.scalar_one_or_none()

    async def list_active_by_keys(
        self, *, user_id: uuid.UUID, companion_id: uuid.UUID, keys: set[str]
    ) -> list[Memory]:
        """Return exact-key memories without depending on vector similarity.

        Used for small, high-value profile facts such as a user's preferred
        name. These facts must remain available even when embeddings are
        temporarily unavailable or a wording change makes semantic matching
        too weak.
        """
        if not keys:
            return []
        result = await self.session.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.companion_id == companion_id,
                Memory.active.is_(True),
                Memory.key.in_(keys),
            )
        )
        return list(result.scalars().all())

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
