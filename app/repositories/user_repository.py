from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

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

        statement = (
            insert(User)
            .values(
                external_user_id=external_user_id,
                locale=locale,
                timezone=timezone,
            )
            .on_conflict_do_nothing(index_elements=[User.external_user_id])
            .returning(User)
        )
        result = await self.session.execute(statement)
        created = result.scalar_one_or_none()
        if created is not None:
            return created

        # Another worker won the first-contact race. PostgreSQL waits for that
        # transaction at the unique index, so the committed row is now visible.
        existing = await self.get_by_external_user_id(external_user_id)
        if existing is None:  # defensive: the unique row should always be visible
            raise RuntimeError("User provisioning conflict did not resolve.")
        return existing
