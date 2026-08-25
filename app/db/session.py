"""
Async database session management.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

_settings = get_settings()

# connect_args tuned for asyncpg against hosted Postgres providers
# (e.g. Supabase). DATABASE_SSL_MODE and DATABASE_DISABLE_STATEMENT_CACHE
# are both configurable via env — nothing provider-specific is hard-coded.
_connect_args: dict = {"ssl": _settings.DATABASE_SSL_MODE}
if _settings.DATABASE_DISABLE_STATEMENT_CACHE:
    # Required when connecting through a PgBouncer transaction-mode
    # pooler, which cannot support asyncpg's prepared statement cache.
    _connect_args["statement_cache_size"] = 0

engine: AsyncEngine = create_async_engine(
    _settings.DATABASE_URL,
    pool_size=_settings.DATABASE_POOL_SIZE,
    max_overflow=_settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    echo=False,
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a scoped async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
