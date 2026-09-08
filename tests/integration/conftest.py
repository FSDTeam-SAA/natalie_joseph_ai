"""
Shared test fixtures for integration tests.

Each test function gets its own engine (NullPool — no connection
reuse across tests) bound to that test's own event loop. This avoids
asyncpg's "attached to a different loop" errors that occur when a
module-level pooled engine is reused across the separate event loops
pytest-asyncio creates per test function by default. Production code
(app/db/session.py) is untouched — this is purely a test-time fixture.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

_TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()
if _TEST_DATABASE_URL:
    # Integration tests must never inherit DATABASE_URL from a developer's .env.
    os.environ["DATABASE_URL"] = _TEST_DATABASE_URL
    os.environ["DATABASE_SSL_MODE"] = os.getenv("TEST_DATABASE_SSL_MODE", "disable")
    os.environ["AUTH_MODE"] = "local_api_key"
    os.environ["LOCAL_API_KEY"] = "integration-test-only-key"
    # Database integration tests exercise application behavior, not Redis.
    # Keeping this off makes the suite deterministic and prevents accidental
    # access to a developer's configured Redis instance.
    os.environ["ENABLE_RATE_LIMITING"] = "false"
    get_settings.cache_clear()

_settings = get_settings()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/integration/" not in item.path.as_posix():
            continue
        item.add_marker(pytest.mark.integration)
        if not _TEST_DATABASE_URL:
            item.add_marker(
                pytest.mark.skip(
                    reason="Set TEST_DATABASE_URL to an isolated migrated test database."
                )
            )


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    connect_args: dict = {"ssl": _settings.DATABASE_SSL_MODE}
    if _settings.DATABASE_DISABLE_STATEMENT_CACHE:
        connect_args["statement_cache_size"] = 0

    engine = create_async_engine(
        _settings.DATABASE_URL,
        poolclass=NullPool,
        connect_args=connect_args,
    )
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()

    await engine.dispose()
