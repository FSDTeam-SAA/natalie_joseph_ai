"""
relationship_context table (spec Section 32).

Explicitly NOT an attachment/dependency score — only the fields listed
in the spec are present. `preferred_tone`, `topic_preferences`, and
`interaction_preferences` are not given fixed enumerations in the
source documents, so they're stored as open string/JSONB rather than
constrained to invented values (Section 66).
"""

from __future__ import annotations

import enum
import uuid

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class FamiliarityLevel(str, enum.Enum):
    new = "new"
    familiar = "familiar"
    established = "established"


class ConversationDepth(str, enum.Enum):
    light = "light"
    medium = "medium"
    deep = "deep"


class RelationshipContext(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "relationship_context"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    companion_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
    )

    familiarity_level: Mapped[FamiliarityLevel] = mapped_column(
        SAEnum(
            FamiliarityLevel,
            name="familiarity_level",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=FamiliarityLevel.new,
    )
    preferred_tone: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        doc="Not enumerated in spec — open string, e.g. 'warm', 'playful'.",
    )
    conversation_depth: Mapped[ConversationDepth] = mapped_column(
        SAEnum(
            ConversationDepth,
            name="conversation_depth",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
        default=ConversationDepth.light,
    )
    topic_preferences: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    interaction_preferences: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "user_id", "companion_id", name="uq_relationship_context_user_companion"
        ),
        Index("ix_relationship_context_user_id", "user_id"),
        Index("ix_relationship_context_companion_id", "companion_id"),
    )
