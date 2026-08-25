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

import uuid

from sqlalchemy import select

from app.db.models.conversation import Conversation
from app.repositories.base_repository import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    model = Conversation

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

    async def update_summary(self, conversation: Conversation, summary: str) -> Conversation:
        conversation.summary = summary
        await self.session.flush()
        return conversation