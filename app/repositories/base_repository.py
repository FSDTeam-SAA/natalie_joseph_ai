"""
Base repository.

Per spec Section 62 (repository pattern) and Section 57 (never allow
global/unrestricted access): this base class only provides generic
CRUD helpers. It does NOT provide a generic "get by id with no owner
check" method for tables that carry user-owned data — those repos
implement scoped queries explicitly (e.g. get_by_id_for_user) so
isolation can never be accidentally bypassed by using the generic
method.
"""

from __future__ import annotations

import uuid
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, instance: ModelT) -> ModelT:
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def get_by_id(self, id_: uuid.UUID) -> ModelT | None:
        """
        Unscoped lookup by primary key. Only safe for tables with no
        per-user ownership concept (e.g. companions). Do not use this
        for user-owned rows — use each repository's explicit
        ownership-scoped method instead.
        """
        result = await self.session.execute(select(self.model).where(self.model.id == id_))
        return result.scalar_one_or_none()

    async def delete(self, instance: ModelT) -> None:
        await self.session.delete(instance)
        await self.session.flush()