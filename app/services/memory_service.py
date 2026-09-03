"""
Memory service — Phase 7.

Two responsibilities, kept in one service since they share the same
repositories/providers:

  1. retrieve_relevant() — synchronous, called from ChatService before
     generating a reply. Embeds the current message, does a cosine
     nearest-neighbor search scoped to (user, companion) via
     MemoryRepository.search_similar(), and applies
     settings.MEMORY_MIN_SCORE as a similarity-score threshold.

  2. extract_and_store() — asynchronous, scheduled as a FastAPI
     BackgroundTask by ChatService after the response has already been
     built and persisted. Uses the background model
     (OPENAI_BACKGROUND_MODEL) via generate_structured() to pull
     candidate facts out of the latest turn, then dedups against
     existing memories by (user, companion, key) — updating in place
     rather than inserting duplicates.

Design notes / deliberate interim choices:

  - No Celery/Redis yet (Phase 8) — extraction runs as a FastAPI
    BackgroundTask instead of a queued job. This executes in-process,
    in the same worker, reusing the SAME request-scoped DB session
    (FastAPI/Starlette guarantee: yield-dependency teardown for
    get_db_session() runs *after* background tasks complete, so the
    session is still open when this runs). This has real limitations
    Phase 8 should address: a worker crash/restart between the
    response being sent and the background task finishing silently
    drops that extraction — no retry, no durability, no queue.
    Acceptable for now; not acceptable long-term at scale.

  - MEMORY_MIN_SCORE (settings) is treated here as a COSINE-SIMILARITY
    threshold (score = 1 - cosine_distance), NOT the same thing as the
    `min_confidence` parameter MemoryRepository.search_similar() already
    supports (which filters on the stored Memory.confidence column —
    the *extraction-time* confidence a memory was created with, a
    completely different axis from retrieval-time relevance). The
    setting's name is ambiguous in isolation; this is the interpretation
    used here, applied as a post-filter in Python since search_similar()
    intentionally returns Memory rows, not raw distances. Flagging this
    explicitly rather than silently picking a meaning — same convention
    as the Memory.privacy_class note below.

  - Memory.privacy_class is left unset (None) here. Its docstring
    (Phase 2) already flags that concrete values need to be defined
    "before Phase 7" — that decision was never made, so nothing here
    invents values for it. Every memory this phase creates has
    privacy_class=None until that decision happens.

  - Retrieval degrades gracefully: any failure (embedding API error, DB
    error) is caught and logged, returning an empty list, so a
    memory-layer outage never breaks the chat endpoint — consistent
    with how Phase 6 sections skip cleanly when they have nothing to
    contribute.

  - Extraction failures are caught and logged, never raised — this runs
    after the user already has their response; an exception here must
    never surface to the user or affect the request/response cycle.
"""

from __future__ import annotations

import logging
import math
import uuid

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.db.models.memory import Memory
from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMMessage, LLMProvider
from app.llm.prompts.schemas.memory_extraction import (
    MEMORY_EXTRACTION_JSON_SCHEMA,
    MEMORY_EXTRACTION_SCHEMA_NAME,
    ExtractedMemoryCandidate,
    parse_extraction_result,
)
from app.repositories.memory_repository import MemoryRepository

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable, memory-worthy facts about the USER (never the "
    "AI companion) from a short excerpt of a conversation. Only extract "
    "things worth remembering across future conversations: stated "
    "preferences, facts about their life, interests, goals, or "
    "significant events they mention. Do not extract things the "
    "companion said about itself. Do not fabricate anything not actually "
    "stated or clearly implied by the user. If nothing in this excerpt is "
    "worth remembering, return an empty list. Reuse the same short 'key' "
    "label across calls for the same underlying fact (e.g. always "
    "'pet_name' for the user's pet's name) so it gets updated instead of "
    "duplicated."
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryService:
    def __init__(
        self,
        *,
        memory_repo: MemoryRepository,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self.memory_repo = memory_repo
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.settings = settings

    async def retrieve_relevant(
        self, *, user_id: uuid.UUID, companion_id: uuid.UUID, query_text: str
    ) -> list[str]:
        try:
            embedding_result = await self.embedding_provider.embed_text(
                query_text, model=self.settings.OPENAI_EMBEDDING_MODEL
            )
            candidates = await self.memory_repo.search_similar(
                user_id=user_id,
                query_embedding=embedding_result.vector,
                companion_id=companion_id,
                top_k=self.settings.MEMORY_TOP_K,
            )
        except Exception:  # noqa: BLE001 - retrieval must never break chat
            logger.exception("memory_retrieval_failed", extra={"user_id": str(user_id)})
            return []

        relevant: list[str] = []
        for memory in candidates:
            score = _cosine_similarity(embedding_result.vector, memory.embedding)
            if score >= self.settings.MEMORY_MIN_SCORE:
                relevant.append(f"{memory.key}: {memory.value}")
        return relevant

    async def extract_and_store(
        self,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        companion_id: uuid.UUID,
        user_message: str,
        assistant_message: str,
    ) -> None:
        try:
            extraction_result = await self.llm_provider.generate_structured(
                [
                    LLMMessage(role="system", content=_EXTRACTION_SYSTEM_PROMPT),
                    LLMMessage(
                        role="user",
                        content=(
                            f"User said: {user_message}\n\n"
                            f"Companion replied: {assistant_message}"
                        ),
                    ),
                ],
                model=self.settings.OPENAI_BACKGROUND_MODEL,
                schema_name=MEMORY_EXTRACTION_SCHEMA_NAME,
                json_schema=MEMORY_EXTRACTION_JSON_SCHEMA,
            )
            candidates = parse_extraction_result(extraction_result.data)
        except (ProviderError, ValidationError):
            logger.exception(
                "memory_extraction_failed", extra={"conversation_id": str(conversation_id)}
            )
            return
        except Exception:  # noqa: BLE001 - never let extraction break anything
            logger.exception(
                "memory_extraction_unexpected_error",
                extra={"conversation_id": str(conversation_id)},
            )
            return

        for candidate in candidates:
            await self._upsert_memory(user_id=user_id, companion_id=companion_id, candidate=candidate)

        await self.memory_repo.session.commit()

    async def _upsert_memory(
        self,
        *,
        user_id: uuid.UUID,
        companion_id: uuid.UUID,
        candidate: ExtractedMemoryCandidate,
    ) -> None:
        try:
            embedding_result = await self.embedding_provider.embed_text(
                candidate.value, model=self.settings.OPENAI_EMBEDDING_MODEL
            )
        except Exception:  # noqa: BLE001
            logger.exception("memory_embedding_failed", extra={"key": candidate.key})
            return

        existing = await self.memory_repo.find_existing_by_key(
            user_id=user_id, companion_id=companion_id, key=candidate.key
        )
        if existing is not None:
            existing.value = candidate.value
            existing.memory_type = candidate.memory_type
            existing.confidence = candidate.confidence
            existing.embedding = embedding_result.vector
            await self.memory_repo.session.flush()
        else:
            await self.memory_repo.add(
                Memory(
                    user_id=user_id,
                    companion_id=companion_id,
                    memory_type=candidate.memory_type,
                    key=candidate.key,
                    value=candidate.value,
                    embedding=embedding_result.vector,
                    confidence=candidate.confidence,
                    privacy_class=None,
                    source_message_id=None,
                    active=True,
                )
            )