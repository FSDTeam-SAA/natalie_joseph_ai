"""
Conversation service.

Every method here requires an AuthContext and enforces ownership at
the repository query layer (spec Section 22/57) — there is no method
that fetches a conversation without a user_id filter.
"""

from __future__ import annotations

import uuid

from app.core.exceptions import NotFoundError, ValidationError
from app.core.security import AuthContext
from app.db.models.message import MessageRole
from app.repositories.companion_repository import CompanionRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.user_repository import UserRepository
from app.schemas.conversation import ConversationResponse, MessageResponse


class ConversationService:
    def __init__(
        self,
        user_repo: UserRepository,
        companion_repo: CompanionRepository,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
    ) -> None:
        self.user_repo = user_repo
        self.companion_repo = companion_repo
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo

    async def create_conversation(
        self,
        auth: AuthContext,
        *,
        companion_id: uuid.UUID,
        external_conversation_id: uuid.UUID | None,
    ) -> ConversationResponse:
        companion = await self.companion_repo.get_by_id(companion_id)
        if companion is None or not companion.active:
            raise ValidationError(f"Companion {companion_id} does not exist or is inactive.")

        user = await self.user_repo.get_or_create_by_external_user_id(auth.user_id)

        from app.db.models.conversation import Conversation

        conversation = await self.conversation_repo.add(
            Conversation(
                user_id=user.id,
                companion_id=companion.id,
                external_conversation_id=external_conversation_id,
            )
        )
        await self.conversation_repo.session.commit()
        return ConversationResponse.model_validate(conversation)

    async def get_conversation(
        self, auth: AuthContext, conversation_id: uuid.UUID
    ) -> ConversationResponse:
        user = await self.user_repo.get_or_create_by_external_user_id(auth.user_id)
        conversation = await self.conversation_repo.get_by_id_for_user(
            conversation_id, user.id
        )
        if conversation is None:
            raise NotFoundError(f"Conversation {conversation_id} not found.")
        return ConversationResponse.model_validate(conversation)

    async def list_messages(
        self, auth: AuthContext, conversation_id: uuid.UUID, *, limit: int = 50
    ) -> list[MessageResponse]:
        user = await self.user_repo.get_or_create_by_external_user_id(auth.user_id)
        conversation = await self.conversation_repo.get_by_id_for_user(
            conversation_id, user.id
        )
        if conversation is None:
            raise NotFoundError(f"Conversation {conversation_id} not found.")

        messages = await self.message_repo.get_recent_for_conversation(
            conversation_id, limit=limit
        )
        # Per spec Section 8: never expose system messages to the client.
        visible = [m for m in messages if m.role != MessageRole.system]
        return [MessageResponse.model_validate(m) for m in visible]

    async def delete_conversation(self, auth: AuthContext, conversation_id: uuid.UUID) -> None:
        user = await self.user_repo.get_or_create_by_external_user_id(auth.user_id)
        conversation = await self.conversation_repo.get_by_id_for_user(
            conversation_id, user.id
        )
        if conversation is None:
            raise NotFoundError(f"Conversation {conversation_id} not found.")

        await self.conversation_repo.delete(conversation)
        await self.conversation_repo.session.commit()