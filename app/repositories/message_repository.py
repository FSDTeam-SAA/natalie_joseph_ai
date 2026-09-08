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

import hashlib
import uuid

from sqlalchemy import func, select, text

from app.db.models.ai_event import AIEvent
from app.db.models.conversation import Conversation
from app.db.models.message import Message, MessageRole
from app.db.models.user import User
from app.repositories.base_repository import BaseRepository


class MessageRepository(BaseRepository[Message]):
    model = Message

    async def get_ai_event_by_request_id(
        self,
        *,
        request_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        event_type: str,
    ) -> AIEvent | None:
        result = await self.session.execute(
            select(AIEvent)
            .where(
                AIEvent.request_id == request_id,
                AIEvent.user_id == user_id,
                AIEvent.conversation_id == conversation_id,
                AIEvent.event_type == event_type,
            )
            .order_by(AIEvent.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

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

    async def count_for_conversation(
        self, conversation_id: uuid.UUID, *, role: MessageRole | None = None
    ) -> int:
        statement = select(func.count()).select_from(Message).where(
            Message.conversation_id == conversation_id
        )
        if role is not None:
            statement = statement.where(Message.role == role)
        result = await self.session.execute(statement)
        return int(result.scalar_one())

    async def count_for_user_and_companion(
        self,
        *,
        user_id: uuid.UUID,
        companion_id: uuid.UUID,
        role: MessageRole | None = None,
    ) -> int:
        """Count relationship-wide turns across every conversation thread."""
        statement = (
            select(func.count(Message.id))
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Conversation.user_id == user_id,
                Conversation.companion_id == companion_id,
            )
        )
        if role is not None:
            statement = statement.where(Message.role == role)
        result = await self.session.execute(statement)
        return int(result.scalar_one())

    async def acquire_idempotency_lock(self, *, scope: str, request_id: uuid.UUID) -> None:
        """Serialize a paid operation across workers until transaction end."""
        digest = hashlib.blake2b(
            f"{scope}:{request_id}".encode(), digest_size=8
        ).digest()
        lock_key = int.from_bytes(digest, byteorder="big", signed=True)
        await self.session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )

    async def get_by_media_asset(
        self, *, conversation_id: uuid.UUID, media_asset_id: uuid.UUID
    ) -> Message | None:
        result = await self.session.execute(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.media_asset_id == media_asset_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_user_by_request_id(
        self, *, conversation_id: uuid.UUID, request_id: uuid.UUID
    ) -> Message | None:
        result = await self.session.execute(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.role == MessageRole.user,
                Message.client_request_id == request_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_user_by_request_id_for_external_user(
        self,
        *,
        conversation_id: uuid.UUID,
        request_id: uuid.UUID,
        external_user_id: uuid.UUID,
    ) -> Message | None:
        """Find a routed request without disclosing another user's conversation."""
        result = await self.session.execute(
            select(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .join(User, User.id == Conversation.user_id)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == MessageRole.user,
                Message.client_request_id == request_id,
                User.external_user_id == external_user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_request_id(
        self, *, conversation_id: uuid.UUID, request_id: uuid.UUID
    ) -> Message | None:
        result = await self.session.execute(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.client_request_id == request_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_first_assistant_after(
        self, *, conversation_id: uuid.UUID, sequence: int
    ) -> Message | None:
        result = await self.session.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == MessageRole.assistant,
                Message.sequence > sequence,
            )
            .order_by(Message.sequence.asc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_by_id_for_conversation(
        self, *, message_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Message | None:
        result = await self.session.execute(
            select(Message).where(
                Message.id == message_id,
                Message.conversation_id == conversation_id,
            )
        )
        return result.scalar_one_or_none()
