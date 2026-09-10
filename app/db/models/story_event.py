"""Companion-world events shared across conversations and media generation."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class StoryEvent(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A trusted-backend event in a companion's ongoing public storyline.

    The main backend/social pipeline owns publishing and scheduling. This table
    gives the AI layer a durable, provider-neutral view of those events so a
    companion can discuss the same fictional activity in chat, proactive
    messages, and generated imagery.
    """

    __tablename__ = "story_events"

    companion_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
    )
    external_event_id: Mapped[str] = mapped_column(String(200), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="backend")
    happened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "companion_id",
            "external_event_id",
            name="uq_story_event_companion_external",
        ),
        Index("ix_story_events_companion_happened_at", "companion_id", "happened_at"),
    )

    @property
    def prompt_fact(self) -> str:
        """A timestamped fact passed into every provider-neutral prompt flow."""
        return f"{self.happened_at.isoformat()} [{self.source}]: {self.summary}"
