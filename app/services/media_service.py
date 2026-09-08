from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.exceptions import NotFoundError
from app.core.security import AuthContext
from app.repositories.media_repository import MediaRepository
from app.repositories.user_repository import UserRepository
from app.storage.base import MediaStorage


@dataclass(frozen=True)
class MediaContent:
    data: bytes
    mime_type: str
    filename: str


class MediaService:
    def __init__(
        self,
        *,
        user_repo: UserRepository,
        media_repo: MediaRepository,
        storage: MediaStorage,
    ) -> None:
        self.user_repo = user_repo
        self.media_repo = media_repo
        self.storage = storage

    async def get_for_user(self, auth: AuthContext, media_id: uuid.UUID) -> MediaContent:
        user = await self.user_repo.get_by_external_user_id(auth.user_id)
        if user is None:
            raise NotFoundError("Media file not found.")
        asset = await self.media_repo.get_by_id_for_user(media_id, user.id)
        if asset is None:
            # Do not reveal whether an asset owned by another user exists.
            raise NotFoundError("Media file not found.")
        data = await self.storage.get(asset.storage_key)
        return MediaContent(data=data, mime_type=asset.mime_type, filename=f"{asset.id}")
