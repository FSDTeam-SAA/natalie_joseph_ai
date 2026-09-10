"""
End-to-end Phase 7 test: does a chat request actually result in a
persisted Memory row (via the FastAPI BackgroundTask), and does a
later chat request actually retrieve it into the prompt?

This also verifies that the post-turn processor opens an independent
database session inside the background task. FastAPI 0.106 through
0.117 closes yielded request dependencies before background work, so
reusing the request session would be unsafe and this real-database test
would catch that regression.

Real Postgres+pgvector round-trip; provider calls mocked at the SDK
boundary, same convention as test_chat_and_conversations.py.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.models.memory import Memory
from app.db.session import get_db_session
from app.llm.base import LLMUsage
from app.llm.prompts.builder import PromptBuilder
from app.main import app

_settings = get_settings()

DEV_HEADERS = {"Authorization": f"Bearer {_settings.LOCAL_API_KEY}"}

# A fixed, non-trivial vector reused for every embed_text call in this
# file. Using the SAME vector for both extraction-time storage and
# retrieval-time querying makes cosine similarity deterministically
# 1.0 -- well above MEMORY_MIN_SCORE -- without depending on real
# OpenAI embeddings (unavailable in this environment).
_FIXED_VECTOR = [0.1] * 1536


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    connect_args: dict = {"ssl": _settings.DATABASE_SSL_MODE}
    if _settings.DATABASE_DISABLE_STATEMENT_CACHE:
        connect_args["statement_cache_size"] = 0

    engine = create_async_engine(
        _settings.DATABASE_URL, poolclass=NullPool, connect_args=connect_args
    )
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def _override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = _override_get_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.pop(get_db_session, None)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_reader() -> AsyncGenerator[AsyncSession, None]:
    """A second, independent session used only to read back what the
    app wrote, to be sure we're checking real committed state rather
    than an in-memory object the app already held a reference to."""
    connect_args: dict = {"ssl": _settings.DATABASE_SSL_MODE}
    if _settings.DATABASE_DISABLE_STATEMENT_CACHE:
        connect_args["statement_cache_size"] = 0
    engine = create_async_engine(
        _settings.DATABASE_URL, poolclass=NullPool, connect_args=connect_args
    )
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


async def _get_elena_id(client: AsyncClient) -> str:
    resp = await client.get("/api/v1/companions")
    return next(c["id"] for c in resp.json() if c["slug"] == "elena")


def _fake_embedding():
    return SimpleNamespace(vector=_FIXED_VECTOR, model="text-embedding-3-small", dimensions=1536)


def _fake_extraction_with_one_memory():
    return SimpleNamespace(
        data={
            "memories": [
                {
                    "memory_type": "fact",
                    "key": "pet_name",
                    "value": "User has a dog named Rex",
                    "confidence": 0.9,
                }
            ]
        },
        usage=LLMUsage(input_tokens=20, output_tokens=10),
        model="gpt-5.6-luna",
        raw_response_id="resp_extract",
    )


def _fake_extraction_empty():
    return SimpleNamespace(
        data={"memories": []},
        usage=LLMUsage(input_tokens=20, output_tokens=10),
        model="gpt-5.6-luna",
        raw_response_id="resp_extract_empty",
    )


class TestMemoryExtractionAndRetrievalEndToEnd:
    async def test_extraction_persists_memory_and_retrieval_picks_it_up(
        self, async_client: AsyncClient, db_reader: AsyncSession
    ) -> None:
        elena_id = await _get_elena_id(async_client)
        debug_user_id = str(uuid.uuid4())
        headers = {**DEV_HEADERS, "X-Debug-User-Id": debug_user_id}

        conv_resp = await async_client.post(
            "/api/v1/conversations", json={"companion_id": elena_id}, headers=headers
        )
        conversation_id = conv_resp.json()["id"]

        from app.embeddings.openai_embeddings import OpenAIEmbeddingProvider
        from app.llm.openai_provider import OpenAIProvider
        from app.llm.xai_provider import XAIProvider

        # --- Turn 1: mention the dog. Extraction should store a memory. ---
        with (
            patch.object(
                XAIProvider,
                "generate",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        text="Rex sounds adorable! Tell me more about him.",
                        usage=LLMUsage(input_tokens=50, output_tokens=20),
                        model="grok-4.6",
                        raw_response_id="resp_turn1",
                    )
                ),
            ),
            patch.object(
                OpenAIEmbeddingProvider, "embed_text", new=AsyncMock(return_value=_fake_embedding())
            ),
            patch.object(
                OpenAIProvider,
                "generate_structured",
                new=AsyncMock(return_value=_fake_extraction_with_one_memory()),
            ),
        ):
            turn1_resp = await async_client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": conversation_id,
                    "companion_id": elena_id,
                    "message": "My dog's name is Rex.",
                    "idempotency_key": str(uuid.uuid4()),
                },
                headers=headers,
            )
        assert turn1_resp.status_code == 200

        # --- Verify a real Memory row landed in the DB, via a totally
        # independent session/connection -- proving the background
        # task actually committed, not just that no exception fired.
        # Scoped to this test's own user (this test DB persists across
        # runs, so an unscoped key search could match other tests'
        # rows too). ---
        from app.repositories.user_repository import UserRepository

        stored_user = await UserRepository(db_reader).get_by_external_user_id(
            uuid.UUID(debug_user_id)
        )
        assert stored_user is not None

        result = await db_reader.execute(
            select(Memory).where(
                Memory.user_id == stored_user.id,
                Memory.companion_id == uuid.UUID(elena_id),
                Memory.key == "pet_name",
            )
        )
        stored = result.scalar_one_or_none()
        assert stored is not None, (
            "No Memory row found after chat request completed -- the "
            "BackgroundTask did not persist its independent transaction."
        )
        assert stored.value == "User has a dog named Rex"
        assert stored.confidence == 0.9
        assert stored.companion_id == uuid.UUID(elena_id)

        # --- Turn 2: ask something else. Retrieval should surface the
        # stored memory into PromptContext (spied on PromptBuilder,
        # since the system prompt is never exposed via the API). ---
        captured_contexts = []
        original_build = PromptBuilder.build_system_prompt

        def _spy(self, context):
            captured_contexts.append(context)
            return original_build(self, context)

        with (
            patch.object(
                XAIProvider,
                "generate",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        text="Sure, happy to chat!",
                        usage=LLMUsage(input_tokens=40, output_tokens=15),
                        model="grok-4.6",
                        raw_response_id="resp_turn2",
                    )
                ),
            ),
            patch.object(
                OpenAIEmbeddingProvider, "embed_text", new=AsyncMock(return_value=_fake_embedding())
            ),
            patch.object(
                OpenAIProvider,
                "generate_structured",
                new=AsyncMock(return_value=_fake_extraction_empty()),
            ),
            patch.object(PromptBuilder, "build_system_prompt", new=_spy),
        ):
            turn2_resp = await async_client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": conversation_id,
                    "companion_id": elena_id,
                    "message": "What should I do this weekend?",
                    "idempotency_key": str(uuid.uuid4()),
                },
                headers=headers,
            )
        assert turn2_resp.status_code == 200

        assert len(captured_contexts) == 1
        assert "pet_name: User has a dog named Rex" in captured_contexts[0].retrieved_memories

    async def test_extraction_updates_existing_memory_key_instead_of_duplicating(
        self, async_client: AsyncClient, db_reader: AsyncSession
    ) -> None:
        elena_id = await _get_elena_id(async_client)
        debug_user_id = str(uuid.uuid4())
        headers = {**DEV_HEADERS, "X-Debug-User-Id": debug_user_id}

        conv_resp = await async_client.post(
            "/api/v1/conversations", json={"companion_id": elena_id}, headers=headers
        )
        conversation_id = conv_resp.json()["id"]

        from app.embeddings.openai_embeddings import OpenAIEmbeddingProvider
        from app.llm.openai_provider import OpenAIProvider
        from app.llm.xai_provider import XAIProvider

        def _common_patches():
            return [
                patch.object(
                    XAIProvider,
                    "generate",
                    new=AsyncMock(
                        return_value=SimpleNamespace(
                            text="Got it!",
                            usage=LLMUsage(input_tokens=30, output_tokens=10),
                            model="grok-4.6",
                            raw_response_id="resp",
                        )
                    ),
                ),
                patch.object(
                    OpenAIEmbeddingProvider,
                    "embed_text",
                    new=AsyncMock(return_value=_fake_embedding()),
                ),
            ]

        with ExitStack() as stack:
            for p in _common_patches():
                stack.enter_context(p)
            stack.enter_context(
                patch.object(
                    OpenAIProvider,
                    "generate_structured",
                    new=AsyncMock(return_value=_fake_extraction_with_one_memory()),
                )
            )
            await async_client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": conversation_id,
                    "companion_id": elena_id,
                    "message": "My dog's name is Rex.",
                    "idempotency_key": str(uuid.uuid4()),
                },
                headers=headers,
            )

        updated_extraction = SimpleNamespace(
            data={
                "memories": [
                    {
                        "memory_type": "fact",
                        "key": "pet_name",
                        "value": "User's dog Rex just turned 2 years old",
                        "confidence": 0.95,
                    }
                ]
            },
            usage=LLMUsage(input_tokens=20, output_tokens=10),
            model="gpt-5.6-luna",
            raw_response_id="resp_extract_update",
        )
        with ExitStack() as stack:
            for p in _common_patches():
                stack.enter_context(p)
            stack.enter_context(
                patch.object(
                    OpenAIProvider,
                    "generate_structured",
                    new=AsyncMock(return_value=updated_extraction),
                )
            )
            await async_client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": conversation_id,
                    "companion_id": elena_id,
                    "message": "Rex just turned 2!",
                    "idempotency_key": str(uuid.uuid4()),
                },
                headers=headers,
            )

        # Scoped to this test's own user (via the same external debug
        # id used in the headers above) and companion — a global
        # `key == "pet_name"` search would also match rows created by
        # other tests/runs sharing this persistent test database.
        from app.repositories.user_repository import UserRepository

        user = await UserRepository(db_reader).get_by_external_user_id(uuid.UUID(debug_user_id))
        assert user is not None

        result = await db_reader.execute(
            select(Memory).where(
                Memory.user_id == user.id,
                Memory.companion_id == uuid.UUID(elena_id),
                Memory.key == "pet_name",
            )
        )
        rows = result.scalars().all()
        assert len(rows) == 1, (
            "Expected the existing 'pet_name' memory to be updated, not duplicated"
        )
        assert rows[0].value == "User's dog Rex just turned 2 years old"
        assert rows[0].confidence == 0.95
