from __future__ import annotations

import ssl
from logging.config import fileConfig

import sqlalchemy as sa
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Connection, make_url

from alembic import context
from app.core.config import get_settings
from app.db.base import ELYSIA_SCHEMA, Base

# Import all models so Base.metadata is fully populated before
# autogenerate runs.
from app.db.models import (  # noqa: F401
    AIEvent,
    Companion,
    Conversation,
    MediaAsset,
    Memory,
    Message,
    RelationshipContext,
    StoryEvent,
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
        context.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {ELYSIA_SCHEMA}"))
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


def run_migrations_online() -> None:
    """Run Alembic with a synchronous driver, independent of greenlet.

    The application intentionally uses ``asyncpg``.  Alembic, however, runs
    synchronous migration callbacks; adapting an AsyncEngine needs greenlet,
    whose native Windows extension may be blocked by application-control
    policies.  pg8000 is a pure-Python PostgreSQL driver, so migrations can
    run without changing the application's runtime database driver.
    """
    migration_url = make_url(settings.DATABASE_URL).set(
        drivername="postgresql+pg8000"
    )
    # A configured ``require`` setting is used by hosted PostgreSQL providers.
    # ``prefer`` deliberately remains unencrypted for local Docker Postgres,
    # whose default image does not enable TLS.
    connect_args: dict = {}
    if settings.DATABASE_SSL_MODE == "require":
        connect_args["ssl_context"] = ssl.create_default_context()

    connectable = create_engine(
        migration_url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
