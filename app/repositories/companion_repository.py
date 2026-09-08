from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db.models.companion import Companion
from app.repositories.base_repository import BaseRepository


class CompanionRepository(BaseRepository[Companion]):
    model = Companion

    async def get_by_slug(self, slug: str) -> Companion | None:
        result = await self.session.execute(select(Companion).where(Companion.slug == slug))
        return result.scalar_one_or_none()

    async def resolve_backend_reference(
        self, reference: str, *, id_map: dict[str, str]
    ) -> Companion | None:
        """Resolve an ID owned by the main backend without exposing DB IDs.

        Deployments configure backend-ID -> stable-slug mappings. Stable slugs
        and this service's UUIDs are also accepted to make local integration and
        staged migrations straightforward.
        """
        mapped = id_map.get(reference, reference).strip().lower()
        if not mapped:
            return None
        try:
            internal_id = uuid.UUID(mapped)
        except ValueError:
            return await self.get_by_slug(mapped)
        return await self.get_by_id(internal_id)

    async def list_active(self) -> list[Companion]:
        result = await self.session.execute(
            select(Companion).where(Companion.active.is_(True)).order_by(Companion.name)
        )
        return list(result.scalars().all())
