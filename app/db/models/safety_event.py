"""
safety_events table (spec Section 26, 47: moderation audit trail).

`direction` is explicitly enumerated in the spec (input/output).
`category`, `severity`, and `action` are referenced conceptually
(harassment, hate, self-harm, etc. as moderation categories; some
action taken) but no fixed value set is given in the source documents,
so they are stored as open strings rather than invented enums —
the concrete taxonomy should be defined alongside `safety_policy.json`
in Phase 9, not guessed here.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class SafetyDirection(str, enum.Enum):
    input = "input"
    output = "output"


class SafetyEvent(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "safety_events"

    request_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )

    direction: Mapped[SafetyDirection] = mapped_column(
        SAEnum(
            SafetyDirection,
            name="safety_direction",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
    )
    category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        doc="Open string — concrete taxonomy (harassment, hate, self_harm, "
        "prompt_injection, etc.) defined in Phase 9 safety_policy.json, not here.",
    )
    severity: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        doc="Not enumerated in spec — TODO-CONFIRM scale (e.g. low/medium/high) "
        "in Phase 9 rather than assumed.",
    )
    action: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        doc="Not enumerated in spec — e.g. 'blocked', 'allowed', 'flagged'. "
        "TODO-CONFIRM concrete action set in Phase 9.",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_safety_events_user_id", "user_id"),
        Index("ix_safety_events_conversation_id", "conversation_id"),
        Index("ix_safety_events_created_at", "created_at"),
        Index("ix_safety_events_request_id", "request_id"),
    )