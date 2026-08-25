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

from collections.abc import AsyncGenerator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

_settings = get_settings()


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