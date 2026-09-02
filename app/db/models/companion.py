"""
companions table.

Per spec Section 2: the five companion IDs must be stable
(elena, chloe, thalia, lina, luna). That stability requirement is
implemented via `slug` (a short, stable, human-readable string), while
`id` remains a normal UUID primary key used for foreign keys elsewhere
— this keeps FK columns consistent with every other table while still
giving the application a stable, spec-mandated lookup key.

Character content (identity, personality, communication style,
background, interests, visual profile) is stored as JSONB per Section
9 ("Character personality must be data-driven") and Section 62 (no
duplicated character configuration, no hard-coded prompts). The
authoritative source for the JSON content itself is
`config/companions/*.json`, loaded by the seed script (Phase 3) — this
model only defines the storage shape.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Companion(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "companions"

    slug: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        doc="Stable companion identifier, e.g. 'elena'. Must be one of the five fixed IDs.",
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        doc="Incremented whenever this companion's configuration changes; "
        "recorded on messages as companion_version for consistency tracking.",
    )

    # JSONB blocks map directly onto the companion spec document
    # sections: identity/about, lifestyle+communication_style,
    # background, interests, visual_profile. Exact key shape is
    # defined by config/companions/*.json, not hard-coded here.
    personality_config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    communication_config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    background_config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    interest_config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    visual_config: Mapped[dict] = mapped_column(JSONB, nullable=False)

    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        Index("ix_companions_slug", "slug", unique=True),
        Index("ix_companions_active", "active"),
    )