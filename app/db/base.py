"""
Declarative base for all ORM models.

All Meet Elysia tables live in a dedicated Postgres schema (`elysia`),
not `public`. This is set once here, at the metadata level, so every
model automatically lives in the right schema without needing
per-model configuration — and so this service can share a single
Supabase database with the main backend's existing `public` schema
without any risk of table-name collisions (see incident: `users` in
`public` was already in use by the main backend's Prisma-managed
schema).

The schema name is intentionally a plain constant, not an env var —
changing it would require a migration anyway, so it's not meant to be
configurable per-deployment.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, MetaData
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

ELYSIA_SCHEMA = "elysia"


class Base(DeclarativeBase):
    """Shared declarative base. All ORM models inherit from this."""

    metadata = MetaData(schema=ELYSIA_SCHEMA)


class UUIDPrimaryKeyMixin:
    """Standard UUID primary key used by all tables per spec (UUIDs confirmed)."""

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """Standard created_at / updated_at columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )