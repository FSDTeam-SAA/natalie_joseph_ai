from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.post_turn_processor import PostTurnProcessor


class _SessionContext:
    def __init__(self, session: object) -> None:
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class _SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[object] = []

    def __call__(self) -> _SessionContext:
        session = SimpleNamespace()
        self.sessions.append(session)
        return _SessionContext(session)


async def test_post_turn_work_uses_two_task_owned_sessions() -> None:
    factory = _SessionFactory()
    processor = PostTurnProcessor(
        session_factory=factory,  # type: ignore[arg-type]
        embedding_provider=SimpleNamespace(),
        llm_provider=SimpleNamespace(),
        settings=SimpleNamespace(),
    )

    with (
        patch(
            "app.services.post_turn_processor.MemoryService.extract_and_store",
            new=AsyncMock(),
        ) as extract,
        patch(
            "app.services.post_turn_processor.ContinuityService.update_after_turn",
            new=AsyncMock(),
        ) as update,
    ):
        await processor.process(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            companion_id=uuid.uuid4(),
            user_message="hello",
            assistant_message="hi",
            source_message_id=uuid.uuid4(),
        )

    assert len(factory.sessions) == 2
    assert factory.sessions[0] is not factory.sessions[1]
    extract.assert_awaited_once()
    update.assert_awaited_once()
