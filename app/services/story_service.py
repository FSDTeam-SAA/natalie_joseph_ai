"""Ingest durable companion-world context from the trusted main backend."""

from __future__ import annotations

from app.core.security import AuthContext, require_feature, require_trusted_backend
from app.db.models.story_event import StoryEvent
from app.repositories.story_event_repository import StoryEventRepository
from app.schemas.backend import BackendStoryEventRequest
from app.services.backend_context_service import BackendContextService


class StoryService:
    def __init__(
        self,
        *,
        backend_context_service: BackendContextService,
        story_repo: StoryEventRepository,
    ) -> None:
        self.backend_context_service = backend_context_service
        self.story_repo = story_repo

    async def ingest(
        self,
        auth: AuthContext,
        *,
        companion_reference: str,
        request: BackendStoryEventRequest,
    ) -> StoryEvent:
        require_trusted_backend(auth)
        require_feature(auth, "story_context")
        companion = await self.backend_context_service.resolve_companion(
            auth,
            companion_reference=companion_reference,
        )
        event = await self.story_repo.upsert(
            companion_id=companion.id,
            external_event_id=request.external_event_id,
            summary=request.summary,
            source=request.source,
            happened_at=request.happened_at,
            event_metadata=request.metadata,
        )
        await self.story_repo.session.commit()
        return event
