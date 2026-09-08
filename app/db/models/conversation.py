"""
conversations table.

Per spec Section 8: ensure user isolation. Enforced here via a
non-nullable FK to users.id plus an index on user_id, so every
repository query can (and must) filter by owning user_id at the query
layer — never relying on the LLM for isolation (Section 14/57).
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Conversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "conversations"

    external_conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        unique=True,
        doc="Conversation UUID as known to the main backend, if provided "
        "at creation time. Nullable because the AI service may originate "
        "conversations itself (e.g. POST /api/v1/conversations).",
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    companion_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Rolling conversation summary maintained by the background "
        "rolling summary maintained after chat turns. Null until the first "
        "summary run.",
    )

    __table_args__ = (
        Index("ix_conversations_user_id", "user_id"),
        Index("ix_conversations_companion_id", "companion_id"),
        Index("ix_conversations_created_at", "created_at"),
    )
