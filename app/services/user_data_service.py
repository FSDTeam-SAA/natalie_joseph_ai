from __future__ import annotations

from app.core.security import AuthContext
from app.repositories.media_repository import MediaRepository
from app.repositories.user_repository import UserRepository
from app.storage.base import MediaStorage


class UserDataService:
    """Delete AI-owned user data when requested by the canonical backend."""

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

    async def delete_current_user_data(self, auth: AuthContext) -> None:
        user = await self.user_repo.get_by_external_user_id(auth.user_id)
        if user is None:
            return
        assets = await self.media_repo.list_for_user(user.id)
        for asset in assets:
            await self.storage.delete(asset.storage_key)
        await self.user_repo.delete(user)
        await self.user_repo.session.commit()
