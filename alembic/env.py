from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
import sqlalchemy as sa
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.db.base import ELYSIA_SCHEMA, Base

# Import all models so Base.metadata is fully populated before
# autogenerate runs.
from app.db.models import (  # noqa: F401
    AIEvent,
    Companion,
    Conversation,
    Memory,
    Message,
    RelationshipContext,
    SafetyEvent,
    User,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=ELYSIA_SCHEMA,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    # Alembic creates its own version-tracking table in ELYSIA_SCHEMA
    # before running any migration, so the schema must already exist
    # at this point — the CREATE SCHEMA statement inside the migration
    # file itself runs too late for that. Committed explicitly and
    # separately from Alembic's own migration transaction below, since
    # SQLAlchemy 2.x connections use autobegin/rollback-on-close
    # semantics and this must not depend on Alembic's transaction
    # handling.
    connection.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {ELYSIA_SCHEMA}"))
    connection.commit()

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=ELYSIA_SCHEMA,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connect_args: dict = {"ssl": settings.DATABASE_SSL_MODE}
    if settings.DATABASE_DISABLE_STATEMENT_CACHE:
        connect_args["statement_cache_size"] = 0

    connectable = create_async_engine(
        settings.DATABASE_URL,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())