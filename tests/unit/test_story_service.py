from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.exceptions import AuthorizationError
from app.core.security import AuthContext
from app.schemas.backend import BackendStoryEventRequest
from app.services.story_service import StoryService


def _auth(*, features: frozenset[str], trusted: bool = True) -> AuthContext:
    return AuthContext(
        user_id=uuid.uuid4(),
        adult_eligible=False,
        entitled=False,
        features=features,
        raw_claims={"source": "trusted_backend" if trusted else "jwt"},
    )


def _request() -> BackendStoryEventRequest:
    return BackendStoryEventRequest(
        external_event_id="instagram-post-42",
        summary="Luna had brunch with friends in Copenhagen.",
        source="instagram",
        happened_at=datetime(2026, 9, 6, 10, 30, tzinfo=UTC),
        metadata={"post_id": "42"},
    )


@pytest.mark.asyncio
async def test_ingest_resolves_companion_upserts_and_commits() -> None:
    companion = SimpleNamespace(id=uuid.uuid4())
    event = SimpleNamespace(id=uuid.uuid4())
    backend_context = SimpleNamespace(resolve_companion=AsyncMock(return_value=companion))
    story_repo = SimpleNamespace(
        upsert=AsyncMock(return_value=event),
        session=SimpleNamespace(commit=AsyncMock()),
    )
    service = StoryService(
        backend_context_service=backend_context,
        story_repo=story_repo,
    )
    request = _request()

    result = await service.ingest(
        _auth(features=frozenset({"story_context"})),
        companion_reference="backend-luna-id",
        request=request,
    )

    assert result is event
    backend_context.resolve_companion.assert_awaited_once()
    story_repo.upsert.assert_awaited_once_with(
        companion_id=companion.id,
        external_event_id=request.external_event_id,
        summary=request.summary,
        source=request.source,
        happened_at=request.happened_at,
        event_metadata=request.metadata,
    )
    story_repo.session.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth",
    [
        _auth(features=frozenset(), trusted=False),
        _auth(features=frozenset({"chat"})),
    ],
)
async def test_ingest_requires_trusted_backend_and_story_scope(auth: AuthContext) -> None:
    service = StoryService(
        backend_context_service=Mock(),
        story_repo=Mock(),
    )

    with pytest.raises(AuthorizationError):
        await service.ingest(auth, companion_reference="luna", request=_request())


def test_story_event_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone"):
        BackendStoryEventRequest(
            external_event_id="event-1",
            summary="A story event",
            happened_at=datetime(2026, 9, 6, 10, 30),
        )
