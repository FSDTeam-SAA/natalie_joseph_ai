from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.models.story_event import StoryEvent
from app.repositories.base_repository import BaseRepository


class StoryEventRepository(BaseRepository[StoryEvent]):
    model = StoryEvent

    async def upsert(
        self,
        *,
        companion_id: uuid.UUID,
        external_event_id: str,
        summary: str,
        source: str,
        happened_at: datetime,
        event_metadata: dict | None,
    ) -> StoryEvent:
        """Idempotently accept new events and trusted corrections."""
        statement = (
            insert(StoryEvent)
            .values(
                companion_id=companion_id,
                external_event_id=external_event_id,
                summary=summary,
                source=source,
                happened_at=happened_at,
                event_metadata=event_metadata,
            )
            .on_conflict_do_update(
                constraint="uq_story_event_companion_external",
                set_={
                    "summary": summary,
                    "source": source,
                    "happened_at": happened_at,
                    "metadata": event_metadata,
                    "updated_at": datetime.now(UTC),
                },
            )
            .returning(StoryEvent)
        )
        result = await self.session.execute(statement)
        return result.scalar_one()

    async def list_recent(self, companion_id: uuid.UUID, *, limit: int = 8) -> list[StoryEvent]:
        result = await self.session.execute(
            select(StoryEvent)
            .where(StoryEvent.companion_id == companion_id)
            .order_by(StoryEvent.happened_at.desc(), StoryEvent.id.desc())
            .limit(limit)
        )
        # Prompt order is chronological even though the query efficiently reads newest first.
        return list(reversed(result.scalars().all()))
