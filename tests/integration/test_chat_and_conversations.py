"""
Integration tests for dev-mode auth, conversation endpoints, and the
chat endpoint (Phase 5 base + Phase 7 memory wiring).

Provider calls are mocked at the provider boundary since integration
tests must not call xAI or OpenAI. Everything else — auth, database writes, ownership
checks and message persistence — runs for real
against the real (schema-isolated) test database.

Phase 7 note: test_send_message_happy_path now also mocks
OpenAIEmbeddingProvider.embed_text (memory retrieval runs on every
chat request) and OpenAIProvider.generate_structured (memory
extraction runs as a background task after the response). Both are
mocked to return "nothing to do" (an empty-vector query result / an
empty memories list) since this test's concern is the chat flow
itself, not memory behavior — that's covered separately in
test_memory_service.py and test_memory_extraction_e2e below.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.session import get_db_session
from app.llm.base import LLMUsage
from app.main import app

_settings = get_settings()

DEV_HEADERS = {"Authorization": f"Bearer {_settings.LOCAL_API_KEY}"}


def _fake_generate_response(text: str):
    return SimpleNamespace(
        output_text=text,
        usage=LLMUsage(input_tokens=50, output_tokens=20),
        model="grok-4.6",
        id=f"resp_{uuid.uuid4().hex[:8]}",
    )


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


async def _get_elena_id(client: AsyncClient) -> str:
    resp = await client.get("/api/v1/companions")
    return next(c["id"] for c in resp.json() if c["slug"] == "elena")


class TestDevAuth:
    async def test_missing_auth_header_rejected(self, async_client: AsyncClient) -> None:
        resp = await async_client.post(
            "/api/v1/conversations", json={"companion_id": str(uuid.uuid4())}
        )
        assert resp.status_code == 401

    async def test_wrong_api_key_rejected(self, async_client: AsyncClient) -> None:
        resp = await async_client.post(
            "/api/v1/conversations",
            json={"companion_id": str(uuid.uuid4())},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    async def test_valid_key_with_default_test_user(self, async_client: AsyncClient) -> None:
        elena_id = await _get_elena_id(async_client)
        resp = await async_client.post(
            "/api/v1/conversations",
            json={"companion_id": elena_id},
            headers=DEV_HEADERS,
        )
        assert resp.status_code == 201


class TestConversationEndpoints:
    async def test_create_get_and_list_messages(self, async_client: AsyncClient) -> None:
        elena_id = await _get_elena_id(async_client)
        debug_user = str(uuid.uuid4())
        headers = {**DEV_HEADERS, "X-Debug-User-Id": debug_user}

        create_resp = await async_client.post(
            "/api/v1/conversations", json={"companion_id": elena_id}, headers=headers
        )
        assert create_resp.status_code == 201
        conversation_id = create_resp.json()["id"]

        get_resp = await async_client.get(
            f"/api/v1/conversations/{conversation_id}", headers=headers
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["companion_id"] == elena_id

        messages_resp = await async_client.get(
            f"/api/v1/conversations/{conversation_id}/messages", headers=headers
        )
        assert messages_resp.status_code == 200
        assert messages_resp.json() == []

    async def test_ownership_isolation_across_debug_users(self, async_client: AsyncClient) -> None:
        elena_id = await _get_elena_id(async_client)
        owner_headers = {**DEV_HEADERS, "X-Debug-User-Id": str(uuid.uuid4())}
        intruder_headers = {**DEV_HEADERS, "X-Debug-User-Id": str(uuid.uuid4())}

        create_resp = await async_client.post(
            "/api/v1/conversations", json={"companion_id": elena_id}, headers=owner_headers
        )
        conversation_id = create_resp.json()["id"]

        intruder_resp = await async_client.get(
            f"/api/v1/conversations/{conversation_id}", headers=intruder_headers
        )
        assert intruder_resp.status_code == 404

    async def test_delete_conversation(self, async_client: AsyncClient) -> None:
        elena_id = await _get_elena_id(async_client)
        headers = {**DEV_HEADERS, "X-Debug-User-Id": str(uuid.uuid4())}

        create_resp = await async_client.post(
            "/api/v1/conversations", json={"companion_id": elena_id}, headers=headers
        )
        conversation_id = create_resp.json()["id"]

        delete_resp = await async_client.delete(
            f"/api/v1/conversations/{conversation_id}", headers=headers
        )
        assert delete_resp.status_code == 204

        get_resp = await async_client.get(
            f"/api/v1/conversations/{conversation_id}", headers=headers
        )
        assert get_resp.status_code == 404


class TestChatEndpoint:
    async def test_send_message_happy_path(self, async_client: AsyncClient) -> None:
        elena_id = await _get_elena_id(async_client)
        headers = {**DEV_HEADERS, "X-Debug-User-Id": str(uuid.uuid4())}

        conv_resp = await async_client.post(
            "/api/v1/conversations", json={"companion_id": elena_id}, headers=headers
        )
        conversation_id = conv_resp.json()["id"]

        from app.embeddings.openai_embeddings import OpenAIEmbeddingProvider
        from app.llm.openai_provider import OpenAIProvider
        from app.llm.xai_provider import XAIProvider

        with (
            patch.object(
                XAIProvider,
                "generate",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        text="Hey there! My day was great, thanks for asking!",
                        usage=LLMUsage(input_tokens=50, output_tokens=20),
                        model="grok-4.6",
                        raw_response_id="resp_test123",
                    )
                ),
            ),
            # Phase 7: retrieve_relevant() always embeds the incoming
            # message before generation. Dimension must be 1536 to
            # match the memories.embedding column, or pgvector errors
            # at the SQL layer regardless of row count.
            patch.object(
                OpenAIEmbeddingProvider,
                "embed_text",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        vector=[0.0] * 1536,
                        model="text-embedding-3-small",
                        dimensions=1536,
                    )
                ),
            ),
            # Phase 7: extract_and_store() runs as a background task
            # after the response is built. Returning an empty list
            # keeps this test focused on the chat flow, not memory
            # extraction behavior.
            patch.object(
                OpenAIProvider,
                "generate_structured",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        data={"memories": []},
                        usage=LLMUsage(input_tokens=30, output_tokens=10),
                        model="gpt-5.6-luna",
                        raw_response_id="resp_extract_test",
                    )
                ),
            ),
        ):
            chat_resp = await async_client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": conversation_id,
                    "companion_id": elena_id,
                    "message": "Hey Elena, how was your day?",
                    "idempotency_key": str(uuid.uuid4()),
                },
                headers=headers,
            )

        assert chat_resp.status_code == 200
        data = chat_resp.json()
        assert data["response"] == "Hey there! My day was great, thanks for asking!"
        assert data["companion_id"] == elena_id
        assert data["conversation_id"] == conversation_id
        assert data["usage"]["input_tokens"] == 50
        assert data["usage"]["output_tokens"] == 20

        # Confirm both messages were actually persisted.
        messages_resp = await async_client.get(
            f"/api/v1/conversations/{conversation_id}/messages", headers=headers
        )
        messages = messages_resp.json()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hey Elena, how was your day?"
        assert messages[1]["role"] == "assistant"

    async def test_wrong_companion_for_conversation_rejected(
        self, async_client: AsyncClient
    ) -> None:
        companions_resp = await async_client.get("/api/v1/companions")
        companions = companions_resp.json()
        elena_id = next(c["id"] for c in companions if c["slug"] == "elena")
        luna_id = next(c["id"] for c in companions if c["slug"] == "luna")

        headers = {**DEV_HEADERS, "X-Debug-User-Id": str(uuid.uuid4())}
        conv_resp = await async_client.post(
            "/api/v1/conversations", json={"companion_id": elena_id}, headers=headers
        )
        conversation_id = conv_resp.json()["id"]

        chat_resp = await async_client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "companion_id": luna_id,  # mismatched on purpose
                "message": "hello",
                "idempotency_key": str(uuid.uuid4()),
            },
            headers=headers,
        )
        assert chat_resp.status_code == 422
