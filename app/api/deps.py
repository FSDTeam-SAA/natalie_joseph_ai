"""
Shared service-construction dependencies for the API layer.
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.embeddings.base import EmbeddingProvider
from app.embeddings.openai_embeddings import OpenAIEmbeddingProvider
from app.llm.base import LLMProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.prompts.builder import PromptBuilder
from app.moderation.base import ModerationProvider
from app.moderation.openai_moderation import OpenAIModerationProvider
from app.repositories.companion_repository import CompanionRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.relationship_repository import RelationshipRepository
from app.repositories.user_repository import UserRepository
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.memory_service import MemoryService


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    return OpenAIProvider(api_key=settings.OPENAI_API_KEY)


def get_embedding_provider(settings: Settings = Depends(get_settings)) -> EmbeddingProvider:
    return OpenAIEmbeddingProvider(api_key=settings.OPENAI_API_KEY)


def get_moderation_provider(settings: Settings = Depends(get_settings)) -> ModerationProvider:
    return OpenAIModerationProvider(api_key=settings.OPENAI_API_KEY)


def get_prompt_builder(settings: Settings = Depends(get_settings)) -> PromptBuilder:
    return PromptBuilder.from_settings(settings)


def get_memory_service(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> MemoryService:
    return MemoryService(
        memory_repo=MemoryRepository(db),
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        settings=settings,
    )


def get_conversation_service(
    db: AsyncSession = Depends(get_db_session),
) -> ConversationService:
    return ConversationService(
        user_repo=UserRepository(db),
        companion_repo=CompanionRepository(db),
        conversation_repo=ConversationRepository(db),
        message_repo=MessageRepository(db),
    )


def get_chat_service(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    moderation_provider: ModerationProvider = Depends(get_moderation_provider),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
    memory_service: MemoryService = Depends(get_memory_service),
) -> ChatService:
    return ChatService(
        user_repo=UserRepository(db),
        companion_repo=CompanionRepository(db),
        conversation_repo=ConversationRepository(db),
        message_repo=MessageRepository(db),
        relationship_repo=RelationshipRepository(db),
        llm_provider=llm_provider,
        moderation_provider=moderation_provider,
        prompt_builder=prompt_builder,
        memory_service=memory_service,
        settings=settings,
    )