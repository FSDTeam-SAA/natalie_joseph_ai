from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db.models.media_asset import MediaAsset, MediaKind
from app.repositories.base_repository import BaseRepository


class MediaRepository(BaseRepository[MediaAsset]):
    model = MediaAsset

    async def get_by_id_for_user(
        self, media_id: uuid.UUID, user_id: uuid.UUID
    ) -> MediaAsset | None:
        result = await self.session.execute(
            select(MediaAsset).where(MediaAsset.id == media_id, MediaAsset.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def list_for_user(self, user_id: uuid.UUID) -> list[MediaAsset]:
        result = await self.session.execute(
            select(MediaAsset).where(MediaAsset.user_id == user_id)
        )
        return list(result.scalars().all())

    async def list_for_conversation(self, conversation_id: uuid.UUID) -> list[MediaAsset]:
        result = await self.session.execute(
            select(MediaAsset).where(MediaAsset.conversation_id == conversation_id)
        )
        return list(result.scalars().all())

    async def get_by_idempotency_key(
        self,
        *,
        user_id: uuid.UUID,
        kind: MediaKind,
        idempotency_key: uuid.UUID,
    ) -> MediaAsset | None:
        result = await self.session.execute(
            select(MediaAsset).where(
                MediaAsset.user_id == user_id,
                MediaAsset.kind == kind,
                MediaAsset.idempotency_key == idempotency_key,
            )
        )
        return result.scalar_one_or_none()
