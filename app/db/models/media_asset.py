"""Private generated-media metadata.

The bytes themselves are owned by a storage adapter. Database rows retain
ownership and correlation data so retrieval is always authorized before the
storage key is used. Storage keys are never returned to clients.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class MediaKind(str, enum.Enum):
    audio = "audio"
    image = "image"


class MediaAsset(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "media_assets"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    companion_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=False
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[MediaKind] = mapped_column(
        SAEnum(MediaKind, name="media_kind", native_enum=False, validate_strings=True),
        nullable=False,
    )
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    idempotency_key: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    asset_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "kind", "idempotency_key", name="uq_media_user_kind_idempotency"
        ),
        Index("ix_media_assets_user_id", "user_id"),
        Index("ix_media_assets_conversation_id", "conversation_id"),
        Index("ix_media_assets_companion_id", "companion_id"),
        Index("ix_media_assets_created_at", "created_at"),
    )
