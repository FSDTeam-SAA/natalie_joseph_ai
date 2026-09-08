"""
Conversation repository.

Per spec Section 22 ("Conversation ownership must be verified") and
Section 57 ("never allow ... model access to another user's history"):
every lookup here is scoped by user_id at the query layer. There is no
method that fetches a conversation by id alone — callers must always
supply the requesting user's id, so isolation can't be bypassed by a
future caller forgetting to check ownership after the fact.
"""

from __future__ import annotations

import hashlib
import uuid

from sqlalchemy import exists, select, text

from app.db.models.conversation import Conversation
from app.repositories.base_repository import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    model = Conversation

    async def acquire_creation_lock(self, key: str) -> None:
        """Serialize first-contact conversation creation across API workers."""
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        lock_key = int.from_bytes(digest, byteorder="big", signed=True)
        await self.session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )

    async def get_by_id_for_user(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> Conversation | None:
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_user(
        self, user_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[Conversation]:
        result = await self.session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_by_external_id_for_user(
        self, external_conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> Conversation | None:
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.external_conversation_id == external_conversation_id,
                Conversation.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def external_id_exists(self, external_conversation_id: uuid.UUID) -> bool:
        result = await self.session.execute(
            select(
                exists().where(
                    Conversation.external_conversation_id == external_conversation_id
                )
            )
        )
        return bool(result.scalar_one())

    async def get_latest_for_user_and_companion(
        self, user_id: uuid.UUID, companion_id: uuid.UUID
    ) -> Conversation | None:
        result = await self.session.execute(
            select(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.companion_id == companion_id,
            )
            .order_by(Conversation.updated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def update_summary(self, conversation: Conversation, summary: str) -> Conversation:
        conversation.summary = summary
        await self.session.flush()
        return conversation
