from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db.models.user import User
from app.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    async def get_by_external_user_id(self, external_user_id: uuid.UUID) -> User | None:
        result = await self.session.execute(
            select(User).where(User.external_user_id == external_user_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create_by_external_user_id(
        self,
        external_user_id: uuid.UUID,
        *,
        locale: str | None = None,
        timezone: str | None = None,
    ) -> User:
        """
        The main backend authenticates the user and passes external_user_id
        (spec Section 58); this service creates a local row on first
        contact rather than requiring a separate provisioning step.
        """
        existing = await self.get_by_external_user_id(external_user_id)
        if existing is not None:
            return existing

        user = User(external_user_id=external_user_id, locale=locale, timezone=timezone)
        return await self.add(user)