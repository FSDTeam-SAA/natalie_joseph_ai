"""
messages table.

Per spec Section 8: roles are exactly user / assistant / system, and
system prompts must never be exposed to the client (enforced at the
API/schema layer in later phases — this model just stores the role
faithfully so that enforcement is possible).
"""

from __future__ import annotations

import enum
import uuid

from sqlalchemy import BigInteger, Enum as SAEnum, Identity
from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin
from sqlalchemy import DateTime
from datetime import datetime, timezone


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Strictly increasing, gap-tolerant ordering key. created_at alone
    # is NOT sufficient to order messages correctly: multiple messages
    # inserted in the same transaction (e.g. a user message and the
    # assistant's reply, written together in chat_service) can receive
    # identical or ambiguously-close wall-clock timestamps, especially
    # under real network latency (observed in practice against a
    # hosted Postgres instance — two same-transaction inserts sorted
    # in the wrong order under `ORDER BY created_at`). `sequence` uses
    # a Postgres IDENTITY column, which guarantees a strictly
    # increasing value per row regardless of timing. All ordering
    # queries in MessageRepository use this column, not created_at.
    sequence: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        nullable=False,
        unique=True,
    )

    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole, name="message_role", native_enum=False, validate_strings=True),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Metadata for cost tracking / observability (spec Section 39-40).
    model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        doc="The specific model identifier used to generate this message "
        "(assistant messages only). Sourced from config, never hard-coded.",
    )
    prompt_version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        doc="Value of PROMPT_VERSION at generation time (assistant messages only).",
    )
    companion_version: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Snapshot of companions.version at generation time, for "
        "persona-consistency auditing.",
    )
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_created_at", "created_at"),
        Index("ix_messages_conversation_id_sequence", "conversation_id", "sequence"),
    )