"""
Shared service-construction dependencies for the API layer.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import MediaStorageBackend, Settings, get_settings
from app.core.exceptions import ServiceUnavailableError
from app.core.rate_limit import RateLimiter
from app.core.security import AuthContext, get_current_auth_context
from app.db.session import get_db_session
from app.embeddings.base import EmbeddingProvider
from app.embeddings.openai_embeddings import OpenAIEmbeddingProvider
from app.images.base import ImageProvider
from app.images.failover import FailoverImageProvider
from app.images.openai_images import OpenAIImageProvider
from app.images.xai_images import XAIImageProvider
from app.llm.base import LLMProvider
from app.llm.failover_provider import FailoverLLMProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.prompts.builder import PromptBuilder
from app.llm.xai_provider import XAIProvider
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.media_repository import MediaRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.relationship_repository import RelationshipRepository
from app.repositories.story_event_repository import StoryEventRepository
from app.repositories.user_repository import UserRepository
from app.services.backend_context_service import BackendContextService
from app.services.chat_routing_service import ChatRoutingService
from app.services.chat_service import ChatService
from app.services.companion_service import CompanionService
from app.services.conversation_service import ConversationService
from app.services.image_service import ImageService
from app.services.media_service import MediaService
from app.services.memory_service import MemoryService
from app.services.post_turn_processor import PostTurnProcessor
from app.services.proactive_service import ProactiveService
from app.services.story_service import StoryService
from app.services.user_data_service import UserDataService
from app.services.voice_service import VoiceService
from app.storage.base import MediaStorage
from app.storage.cloudinary import CloudinaryMediaStorage
from app.storage.local import LocalMediaStorage
from app.voice.base import VoiceProvider
from app.voice.elevenlabs_provider import ElevenLabsVoiceProvider


@lru_cache
def _cached_chat_provider(
    xai_api_key: str,
    xai_base_url: str,
    openai_api_key: str,
    openai_chat_model: str,
    timeout_seconds: float,
    max_retries: int,
) -> LLMProvider:
    return FailoverLLMProvider(
        primary=(
            XAIProvider(
                api_key=xai_api_key,
                base_url=xai_base_url,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            if xai_api_key
            else None
        ),
        fallback=(
            OpenAIProvider(
                api_key=openai_api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            if openai_api_key
            else None
        ),
        fallback_model=openai_chat_model,
    )


def get_chat_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    if not settings.XAI_API_KEY and not settings.OPENAI_API_KEY:
        raise ServiceUnavailableError("No conversation-generation provider is configured.")
    return _cached_chat_provider(
        settings.XAI_API_KEY,
        settings.XAI_BASE_URL,
        settings.OPENAI_API_KEY,
        settings.OPENAI_CHAT_MODEL,
        settings.PROVIDER_TIMEOUT_SECONDS,
        settings.PROVIDER_MAX_RETRIES,
    )


@lru_cache
def _cached_background_provider(
    api_key: str, timeout_seconds: float, max_retries: int
) -> LLMProvider:
    return OpenAIProvider(
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def get_background_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    if not settings.OPENAI_API_KEY:
        raise ServiceUnavailableError("OpenAI background processing is not configured.")
    return _cached_background_provider(
        settings.OPENAI_API_KEY,
        settings.PROVIDER_TIMEOUT_SECONDS,
        settings.PROVIDER_MAX_RETRIES,
    )


# Backward-compatible dependency name for integrations that overrode it.
get_llm_provider = get_chat_llm_provider


@lru_cache
def _cached_embedding_provider(
    api_key: str, timeout_seconds: float, max_retries: int
) -> EmbeddingProvider:
    return OpenAIEmbeddingProvider(
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def get_embedding_provider(settings: Settings = Depends(get_settings)) -> EmbeddingProvider:
    if not settings.OPENAI_API_KEY:
        raise ServiceUnavailableError("OpenAI embeddings are not configured.")
    return _cached_embedding_provider(
        settings.OPENAI_API_KEY,
        settings.PROVIDER_TIMEOUT_SECONDS,
        settings.PROVIDER_MAX_RETRIES,
    )


def get_prompt_builder(settings: Settings = Depends(get_settings)) -> PromptBuilder:
    return PromptBuilder.from_settings(settings)


def get_memory_service(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_background_llm_provider),
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
    settings: Settings = Depends(get_settings),
) -> ConversationService:
    return ConversationService(
        user_repo=UserRepository(db),
        companion_service=CompanionService(settings),
        conversation_repo=ConversationRepository(db),
        message_repo=MessageRepository(db),
        settings=settings,
        media_repo=MediaRepository(db),
        storage=get_media_storage(settings),
    )


def get_post_turn_processor(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_background_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> PostTurnProcessor:
    if db.bind is None:
        raise RuntimeError("The database session has no engine binding.")
    # Retain the engine/connection binding, never the request-scoped session.
    session_factory = async_sessionmaker(bind=db.bind, expire_on_commit=False)
    return PostTurnProcessor(
        session_factory=session_factory,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        settings=settings,
    )


def get_chat_service(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_chat_llm_provider),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
    memory_service: MemoryService = Depends(get_memory_service),
    post_turn_processor: PostTurnProcessor = Depends(get_post_turn_processor),
) -> ChatService:
    return ChatService(
        user_repo=UserRepository(db),
        companion_service=CompanionService(settings),
        conversation_repo=ConversationRepository(db),
        message_repo=MessageRepository(db),
        relationship_repo=RelationshipRepository(db),
        llm_provider=llm_provider,
        prompt_builder=prompt_builder,
        memory_service=memory_service,
        settings=settings,
        post_turn_processor=post_turn_processor,
        story_repo=StoryEventRepository(db),
    )


@lru_cache
def _cached_storage(
    backend: MediaStorageBackend,
    root: str,
    max_bytes: int,
    cloud_name: str,
    api_key: str,
    api_secret: str,
    folder: str,
) -> MediaStorage:
    if backend == MediaStorageBackend.cloudinary:
        return CloudinaryMediaStorage(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            folder=folder,
            max_bytes=max_bytes,
        )
    return LocalMediaStorage(root, max_bytes=max_bytes)


def get_media_storage(settings: Settings = Depends(get_settings)) -> MediaStorage:
    return _cached_storage(
        settings.MEDIA_STORAGE_BACKEND,
        settings.MEDIA_STORAGE_ROOT,
        settings.MAX_GENERATED_MEDIA_BYTES,
        settings.CLOUDINARY_CLOUD_NAME,
        settings.CLOUDINARY_API_KEY,
        settings.CLOUDINARY_API_SECRET,
        settings.CLOUDINARY_FOLDER,
    )


@lru_cache
def _cached_image_provider(
    xai_api_key: str,
    xai_base_url: str,
    openai_api_key: str,
    openai_image_model: str,
    timeout_seconds: float,
    max_retries: int,
) -> ImageProvider:
    return FailoverImageProvider(
        primary=(
            XAIImageProvider(
                api_key=xai_api_key,
                base_url=xai_base_url,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            if xai_api_key
            else None
        ),
        fallback=(
            OpenAIImageProvider(
                api_key=openai_api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            if openai_api_key
            else None
        ),
        fallback_model=openai_image_model,
    )


def get_image_provider(settings: Settings = Depends(get_settings)) -> ImageProvider:
    if not settings.XAI_API_KEY and not settings.OPENAI_API_KEY:
        raise ServiceUnavailableError("No image-generation provider is configured.")
    return _cached_image_provider(
        settings.XAI_API_KEY,
        settings.XAI_BASE_URL,
        settings.OPENAI_API_KEY,
        settings.OPENAI_IMAGE_MODEL,
        settings.PROVIDER_TIMEOUT_SECONDS,
        settings.PROVIDER_MAX_RETRIES,
    )


def get_image_service(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    image_provider: ImageProvider = Depends(get_image_provider),
    storage: MediaStorage = Depends(get_media_storage),
) -> ImageService:
    return ImageService(
        user_repo=UserRepository(db),
        companion_service=CompanionService(settings),
        conversation_repo=ConversationRepository(db),
        message_repo=MessageRepository(db),
        media_repo=MediaRepository(db),
        image_provider=image_provider,
        storage=storage,
        settings=settings,
        story_repo=StoryEventRepository(db),
    )


def get_chat_routing_service(
    db: AsyncSession = Depends(get_db_session),
    chat_service: ChatService = Depends(get_chat_service),
    image_service: ImageService = Depends(get_image_service),
) -> ChatRoutingService:
    """Compose chat and image generation without coupling either domain service."""
    return ChatRoutingService(
        chat_service=chat_service,
        image_service=image_service,
        message_repo=MessageRepository(db),
    )


def get_media_service(
    db: AsyncSession = Depends(get_db_session),
    storage: MediaStorage = Depends(get_media_storage),
) -> MediaService:
    return MediaService(
        user_repo=UserRepository(db),
        media_repo=MediaRepository(db),
        storage=storage,
    )


@lru_cache
def _cached_voice_provider(
    api_key: str, base_url: str, timeout_seconds: float, max_retries: int
) -> VoiceProvider:
    return ElevenLabsVoiceProvider(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def get_voice_provider(settings: Settings = Depends(get_settings)) -> VoiceProvider:
    if not settings.ELEVENLABS_API_KEY:
        raise ServiceUnavailableError("ElevenLabs voice processing is not configured.")
    return _cached_voice_provider(
        settings.ELEVENLABS_API_KEY,
        settings.ELEVENLABS_BASE_URL,
        settings.PROVIDER_TIMEOUT_SECONDS,
        settings.PROVIDER_MAX_RETRIES,
    )


def get_voice_service(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    chat_service: ChatService = Depends(get_chat_service),
    chat_routing_service: ChatRoutingService = Depends(get_chat_routing_service),
    voice_provider: VoiceProvider = Depends(get_voice_provider),
    storage: MediaStorage = Depends(get_media_storage),
) -> VoiceService:
    return VoiceService(
        user_repo=UserRepository(db),
        companion_service=CompanionService(settings),
        conversation_repo=ConversationRepository(db),
        message_repo=MessageRepository(db),
        media_repo=MediaRepository(db),
        chat_service=chat_service,
        chat_routing_service=chat_routing_service,
        voice_provider=voice_provider,
        storage=storage,
        settings=settings,
    )


def get_optional_voice_service(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    storage: MediaStorage = Depends(get_media_storage),
    chat_service: ChatService = Depends(get_chat_service),
    chat_routing_service: ChatRoutingService = Depends(get_chat_routing_service),
) -> VoiceService | None:
    """Build synthesis-only support without constructing the full chat pipeline."""
    if not settings.ELEVENLABS_API_KEY:
        return None
    voice_provider = _cached_voice_provider(
        settings.ELEVENLABS_API_KEY,
        settings.ELEVENLABS_BASE_URL,
        settings.PROVIDER_TIMEOUT_SECONDS,
        settings.PROVIDER_MAX_RETRIES,
    )
    return VoiceService(
        user_repo=UserRepository(db),
        companion_service=CompanionService(settings),
        conversation_repo=ConversationRepository(db),
        message_repo=MessageRepository(db),
        media_repo=MediaRepository(db),
        chat_service=chat_service,
        chat_routing_service=chat_routing_service,
        voice_provider=voice_provider,
        storage=storage,
        settings=settings,
    )


def get_user_data_service(
    db: AsyncSession = Depends(get_db_session),
    storage: MediaStorage = Depends(get_media_storage),
) -> UserDataService:
    return UserDataService(
        user_repo=UserRepository(db),
        media_repo=MediaRepository(db),
        storage=storage,
    )


def get_backend_context_service(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> BackendContextService:
    return BackendContextService(
        user_repo=UserRepository(db),
        companion_service=CompanionService(settings),
        conversation_repo=ConversationRepository(db),
        settings=settings,
    )


def get_proactive_service(
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_chat_llm_provider),
    prompt_builder: PromptBuilder = Depends(get_prompt_builder),
    memory_service: MemoryService = Depends(get_memory_service),
) -> ProactiveService:
    return ProactiveService(
        message_repo=MessageRepository(db),
        relationship_repo=RelationshipRepository(db),
        llm_provider=llm_provider,
        prompt_builder=prompt_builder,
        memory_service=memory_service,
        settings=settings,
        story_repo=StoryEventRepository(db),
    )


def get_story_service(
    db: AsyncSession = Depends(get_db_session),
    backend_context_service: BackendContextService = Depends(get_backend_context_service),
) -> StoryService:
    return StoryService(
        backend_context_service=backend_context_service,
        story_repo=StoryEventRepository(db),
    )


@lru_cache
def _cached_rate_limiter(
    redis_url: str,
    enabled: bool,
    fail_closed: bool,
    per_user_per_minute: int,
    per_ip_per_minute: int,
    burst_per_ten_seconds: int,
) -> RateLimiter:
    return RateLimiter(
        redis_url=redis_url,
        enabled=enabled,
        fail_closed=fail_closed,
        per_user_per_minute=per_user_per_minute,
        per_ip_per_minute=per_ip_per_minute,
        burst_per_ten_seconds=burst_per_ten_seconds,
    )


def get_rate_limiter(settings: Settings = Depends(get_settings)) -> RateLimiter:
    return _cached_rate_limiter(
        settings.REDIS_URL,
        settings.ENABLE_RATE_LIMITING,
        settings.RATE_LIMIT_FAIL_CLOSED,
        settings.RATE_LIMIT_PER_USER_PER_MINUTE,
        settings.RATE_LIMIT_PER_IP_PER_MINUTE,
        settings.RATE_LIMIT_CONVERSATION_BURST,
    )


async def enforce_ai_rate_limit(
    request: Request,
    auth: AuthContext = Depends(get_current_auth_context),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    await limiter.check(
        auth=auth,
        client_ip=request.client.host if request.client else None,
    )
