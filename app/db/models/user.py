"""
users table.

Per spec Section 8 and confirmed backend contract: `external_user_id`
is the UUID the main backend uses to identify the user. Our internal
`id` is a separate UUID primary key so this service is never forced to
use the main backend's ID as its own primary key (keeps this service's
schema independent, per spec Section 1 — "AI service must remain
independent from the frontend/main backend").
"""

from __future__ import annotations

import uuid

from sqlalchemy import Index, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    external_user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        unique=True,
        doc="The user UUID issued/owned by the main backend (confirmed format: UUID).",
    )

    # Optional personalization fields. Not defined with strict enums in
    # the spec — kept as free-form strings per Section 66 (do not
    # fabricate constraints not given).
    locale: Mapped[str | None] = mapped_column(String(35), nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index("ix_users_external_user_id", "external_user_id", unique=True),
    )