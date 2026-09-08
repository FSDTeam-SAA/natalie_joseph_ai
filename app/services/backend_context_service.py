"""Translate trusted-main-backend identifiers into AI-service context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import NotFoundError, ValidationError
from app.core.security import AuthContext, require_trusted_backend
from app.db.models.conversation import Conversation
from app.db.models.user import User
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.user_repository import UserRepository
from app.services.companion_service import CompanionProfile, CompanionService

_EXTERNAL_CONVERSATION_NAMESPACE = uuid.UUID("122eb12e-a9d4-4a03-b05c-4cceaff07e58")


@dataclass(frozen=True, slots=True)
class BackendConversationContext:
    user: User
    companion: CompanionProfile
    conversation: Conversation


def _stable_external_conversation_id(value: str) -> uuid.UUID:
    """Keep the local schema UUID-based while accepting UUID/cuid/string IDs."""
    normalized = value.strip()
    if not normalized or len(normalized) > 200 or any(ord(char) < 32 for char in normalized):
        raise ValidationError("external_conversation_id is invalid.")
    try:
        return uuid.UUID(normalized)
    except ValueError:
        return uuid.uuid5(_EXTERNAL_CONVERSATION_NAMESPACE, normalized)


class BackendContextService:
    def __init__(
        self,
        *,
        user_repo: UserRepository,
        companion_service: CompanionService,
        conversation_repo: ConversationRepository,
        settings: Settings,
    ) -> None:
        self.user_repo = user_repo
        self.companion_service = companion_service
        self.conversation_repo = conversation_repo
        self.settings = settings

    async def resolve_companion(
        self,
        auth: AuthContext,
        *,
        companion_reference: str,
    ) -> CompanionProfile:
        """Resolve a backend-owned ID without creating user/conversation state."""
        require_trusted_backend(auth)
        reference = self.settings.BACKEND_COMPANION_ID_MAP.get(
            companion_reference, companion_reference
        )
        try:
            companion = await self.companion_service.get_active_profile_by_reference(reference)
        except NotFoundError as exc:
            raise NotFoundError(
                "The backend companion ID is not mapped to an active companion."
            ) from exc
        return companion

    async def resolve(
        self,
        auth: AuthContext,
        *,
        companion_reference: str,
        external_conversation_id: str | None,
    ) -> BackendConversationContext:
        companion = await self.resolve_companion(
            auth,
            companion_reference=companion_reference,
        )

        user = await self.user_repo.get_or_create_by_external_user_id(auth.user_id)
        conversation: Conversation | None = None
        stable_external_id: uuid.UUID | None = None
        if external_conversation_id is not None:
            stable_external_id = _stable_external_conversation_id(external_conversation_id)
            await self.conversation_repo.acquire_creation_lock(
                f"external-conversation:{stable_external_id}"
            )
            conversation = await self.conversation_repo.get_by_external_id_for_user(
                stable_external_id, user.id
            )
            if conversation is None and await self.conversation_repo.external_id_exists(
                stable_external_id
            ):
                raise ValidationError("The external conversation belongs to another user.")
        else:
            await self.conversation_repo.acquire_creation_lock(
                f"default-conversation:{user.id}:{companion.id}"
            )
            conversation = await self.conversation_repo.get_latest_for_user_and_companion(
                user.id, companion.id
            )

        if conversation is not None and conversation.companion_id != companion.id:
            raise ValidationError(
                "The external conversation is already associated with another companion."
            )
        if conversation is None:
            # The supplied NestJS API currently has no conversation ID in its
            # SendMessageDto. In that shape, use one durable conversation per
            # user/companion; callers can opt into multiple threads by sending
            # an explicit external_conversation_id.
            if stable_external_id is None:
                stable_external_id = uuid.uuid5(
                    _EXTERNAL_CONVERSATION_NAMESPACE,
                    f"{auth.user_id}:{companion.id}",
                )
            conversation = await self.conversation_repo.add(
                Conversation(
                    external_conversation_id=stable_external_id,
                    user_id=user.id,
                    companion_id=companion.id,
                )
            )
            await self.conversation_repo.session.commit()

        return BackendConversationContext(
            user=user,
            companion=companion,
            conversation=conversation,
        )
