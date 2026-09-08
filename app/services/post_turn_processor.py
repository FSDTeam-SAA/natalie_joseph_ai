"""Background post-turn work with database sessions owned by the task itself."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMProvider
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.relationship_repository import RelationshipRepository
from app.services.continuity_service import ContinuityService
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class PostTurnProcessor:
    """Run memory extraction and continuity work outside the request session.

    FastAPI 0.106 through 0.117 closes dependencies yielded by a path operation
    before its background tasks run. This processor keeps only a session factory;
    each task creates and closes its own sessions after the chat response commits.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self.session_factory = session_factory
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.settings = settings

    async def process(
        self,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        companion_id: uuid.UUID,
        user_message: str,
        assistant_message: str,
        source_message_id: uuid.UUID,
    ) -> None:
        try:
            async with self.session_factory() as session:
                memory_service = MemoryService(
                    memory_repo=MemoryRepository(session),
                    embedding_provider=self.embedding_provider,
                    llm_provider=self.llm_provider,
                    settings=self.settings,
                )
                await memory_service.extract_and_store(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    companion_id=companion_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    source_message_id=source_message_id,
                )
        except Exception:  # noqa: BLE001 - post-response work is best effort
            logger.exception(
                "post_turn_memory_failed",
                extra={"conversation_id": str(conversation_id)},
            )

        # Use a second transaction so a memory failure cannot suppress relationship
        # progression or a due rolling summary.
        try:
            async with self.session_factory() as session:
                continuity_service = ContinuityService(
                    conversation_repo=ConversationRepository(session),
                    message_repo=MessageRepository(session),
                    relationship_repo=RelationshipRepository(session),
                    llm_provider=self.llm_provider,
                    settings=self.settings,
                )
                await continuity_service.update_after_turn(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    companion_id=companion_id,
                )
        except Exception:  # noqa: BLE001 - post-response work is best effort
            logger.exception(
                "post_turn_continuity_failed",
                extra={"conversation_id": str(conversation_id)},
            )
