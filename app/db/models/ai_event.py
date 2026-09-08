"""
ai_events table (spec Section 39-40: observability, cost tracking).

Per Section 39: "Do not store unnecessary raw private conversation
data in logs." This table stores metadata about AI calls (latency,
tokens, model, event type) — never message content.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class AIEvent(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "ai_events"

    request_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    companion_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Not enumerated in spec — open string, e.g. 'chat_completion', "
        "'memory_extraction', 'summarization'. Left flexible per Section 66.",
    )
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    companion_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Column name `metadata` is reserved on the SQLAlchemy declarative
    # Base, so the Python attribute is `event_metadata` while the actual
    # DB column stays named `metadata` per the spec's table definition.
    event_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_ai_events_user_id", "user_id"),
        Index("ix_ai_events_companion_id", "companion_id"),
        Index("ix_ai_events_conversation_id", "conversation_id"),
        Index("ix_ai_events_created_at", "created_at"),
        Index("ix_ai_events_request_id", "request_id"),
    )
