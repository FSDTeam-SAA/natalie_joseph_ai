"""
Message repository.

Per spec Section 11 ("Do not send unlimited conversation history —
use recent turns") and Section 48 ("avoid unlimited message history"),
`get_recent_for_conversation` always applies a limit — there is no
"get all messages" method here by design.

Ordering: uses Message.sequence (a strictly increasing IDENTITY
column), not created_at. Wall-clock timestamps are not reliable for
ordering messages inserted within the same transaction — see the
`sequence` column's docstring in app/db/models/message.py for the
real bug this fixes.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select

from app.db.models.message import Message
from app.repositories.base_repository import BaseRepository


class MessageRepository(BaseRepository[Message]):
    model = Message

    async def get_recent_for_conversation(
        self, conversation_id: uuid.UUID, *, limit: int
    ) -> list[Message]:
        """
        Returns the most recent `limit` messages in chronological
        (oldest-first) order, ready to drop directly into a prompt.
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence.desc())
            .limit(limit)
        )
        messages = list(result.scalars().all())
        messages.reverse()
        return messages

    async def get_all_for_conversation_ordered(
        self, conversation_id: uuid.UUID
    ) -> list[Message]:
        """
        Unbounded, chronological — intended only for the background
        summarization job (Phase 8), which needs the full history to
        summarize, not for request-path prompt assembly.
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence.asc())
        )
        return list(result.scalars().all())

    async def count_for_conversation(self, conversation_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(Message).where(
                Message.conversation_id == conversation_id
            )
        )
        return int(result.scalar_one())