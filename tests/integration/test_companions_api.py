"""
Integration tests for Phase 3 companion endpoints.

Assumes the five companions have already been seeded (run
`python -m app.scripts.seed_companions` before running these tests,
same as the repository tests assume a migrated database).

Uses httpx.AsyncClient + ASGITransport, run inside pytest-asyncio
async tests, with the app's DB dependency overridden to use a
per-test engine (same pattern as tests/integration/conftest.py's `db`
fixture). This avoids a known issue where Starlette's synchronous
TestClient opens a new event loop per call, which conflicts with the
app's shared connection pool across multiple calls within one test.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.session import get_db_session
from app.main import app

EXPECTED_SLUGS = {"elena", "chloe", "thalia", "lina", "luna"}

_settings = get_settings()


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


async def test_list_companions_returns_all_five(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/companions")
    assert response.status_code == 200

    data = response.json()
    slugs = {c["slug"] for c in data}
    assert slugs == EXPECTED_SLUGS

    # Frontend-safe fields only — no internal config keys should leak.
    for companion in data:
        assert set(companion.keys()) == {
            "id", "slug", "name", "title", "traits", "location", "occupation"
        }


async def test_get_companion_detail_by_id(async_client: AsyncClient) -> None:
    listing = (await async_client.get("/api/v1/companions")).json()
    elena_id = next(c["id"] for c in listing if c["slug"] == "elena")

    response = await async_client.get(f"/api/v1/companions/{elena_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["slug"] == "elena"
    assert data["name"] == "Elena"
    assert "Dubai" in data["location"]
    assert len(data["interests"]) > 0
    assert len(data["lifestyle"]) > 0

    # Internal config fields must never be exposed.
    forbidden_keys = {
        "personality_config", "communication_config", "background_config",
        "interest_config", "visual_config", "system_prompt",
    }
    assert forbidden_keys.isdisjoint(data.keys())


async def test_get_companion_detail_404_for_unknown_id(async_client: AsyncClient) -> None:
    fake_id = uuid.uuid4()
    response = await async_client.get(f"/api/v1/companions/{fake_id}")
    assert response.status_code == 404

    body = response.json()
    assert body["error"]["code"] == "NOT_FOUND"
    assert "request_id" in body["error"]
