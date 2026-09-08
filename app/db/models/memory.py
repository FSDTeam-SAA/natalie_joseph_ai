"""
memories table — long-term semantic memory (spec Section 12-15).

Embedding dimension note: OPENAI_EMBEDDING_MODEL is currently
configured as "text-embedding-3-small", which produces 1536-dimension
vectors. The vector column width below is fixed at migration time
(pgvector requires a fixed dimension per column). If the embedding
model is ever changed to one with a different output dimension, this
column (and its index) must be migrated accordingly — it is not
dynamically read from config. This is flagged here deliberately per
Section 66 rather than silently assumed.
"""

from __future__ import annotations

import enum
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Float, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

EMBEDDING_DIMENSIONS = 1536  # tied to OPENAI_EMBEDDING_MODEL=text-embedding-3-small


class MemoryType(str, enum.Enum):
    preference = "preference"
    fact = "fact"
    interest = "interest"
    goal = "goal"
    relationship_context = "relationship_context"
    conversation_preference = "conversation_preference"
    important_event = "important_event"


class Memory(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "memories"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    companion_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
    )

    memory_type: Mapped[MemoryType] = mapped_column(
        SAEnum(MemoryType, name="memory_type", native_enum=False, validate_strings=True),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    privacy_class: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        doc="Not enumerated in the source spec — stored as an open string. "
        "No privacy taxonomy is assumed; owner/companion scoping is enforced "
        "independently at the repository layer.",
    )

    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=True,
    )

    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        UniqueConstraint("user_id", "companion_id", "key", name="uq_memory_user_companion_key"),
        Index("ix_memories_user_id", "user_id"),
        Index("ix_memories_companion_id", "companion_id"),
        Index("ix_memories_active", "active"),
        # Composite index for the most common retrieval filter (Section 14:
        # same user, optionally same companion, active only).
        Index("ix_memories_user_companion_active", "user_id", "companion_id", "active"),
        # Vector similarity index is created explicitly in the Alembic
        # migration (ivfflat/hnsw options aren't expressible via this
        # declarative Index() call the same way) — see migration file.
    )
