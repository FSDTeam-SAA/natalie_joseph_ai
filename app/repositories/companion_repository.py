from __future__ import annotations

from sqlalchemy import select

from app.db.models.companion import Companion
from app.repositories.base_repository import BaseRepository


class CompanionRepository(BaseRepository[Companion]):
    model = Companion

    async def get_by_slug(self, slug: str) -> Companion | None:
        result = await self.session.execute(select(Companion).where(Companion.slug == slug))
        return result.scalar_one_or_none()

    async def list_active(self) -> list[Companion]:
        result = await self.session.execute(
            select(Companion).where(Companion.active.is_(True)).order_by(Companion.name)
        )
        return list(result.scalars().all())