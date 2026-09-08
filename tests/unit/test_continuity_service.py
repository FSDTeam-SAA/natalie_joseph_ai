from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.core.config import Settings
from app.db.models.message import MessageRole
from app.db.models.relationship_context import ConversationDepth, FamiliarityLevel
from app.llm.base import LLMResult, LLMUsage
from app.services.continuity_service import ContinuityService


async def test_relationship_progresses_and_summary_is_rolled_forward() -> None:
    conversation_id = uuid.uuid4()
    user_id = uuid.uuid4()
    companion_id = uuid.uuid4()
    relationship = SimpleNamespace(
        familiarity_level=FamiliarityLevel.new,
        conversation_depth=ConversationDepth.light,
    )
    conversation = SimpleNamespace(summary="Earlier summary")
    messages = [
        SimpleNamespace(role=MessageRole.user, content="My birthday is May 4"),
        SimpleNamespace(role=MessageRole.assistant, content="I'll remember that."),
    ]
    session = SimpleNamespace(add=Mock(), commit=AsyncMock(), rollback=AsyncMock())
    message_repo = SimpleNamespace(
        session=session,
        count_for_user_and_companion=AsyncMock(return_value=100),
        count_for_conversation=AsyncMock(return_value=30),
        get_recent_for_conversation=AsyncMock(return_value=messages),
    )
    llm = SimpleNamespace(
        generate=AsyncMock(
            return_value=LLMResult(
                text="The user shared a May 4 birthday.",
                usage=LLMUsage(input_tokens=20, output_tokens=8),
                model="background-model",
            )
        )
    )
    service = ContinuityService(
        conversation_repo=SimpleNamespace(
            get_by_id_for_user=AsyncMock(return_value=conversation)
        ),
        message_repo=message_repo,
        relationship_repo=SimpleNamespace(
            get_or_create=AsyncMock(return_value=relationship)
        ),
        llm_provider=llm,
        settings=Settings(
            _env_file=None,
            SUMMARY_TRIGGER_MESSAGE_COUNT=30,
            RELATIONSHIP_FAMILIAR_AFTER_MESSAGES=20,
            RELATIONSHIP_ESTABLISHED_AFTER_MESSAGES=100,
            RELATIONSHIP_DEEP_AFTER_MESSAGES=200,
        ),
    )

    await service.update_after_turn(
        conversation_id=conversation_id,
        user_id=user_id,
        companion_id=companion_id,
    )

    assert relationship.familiarity_level == FamiliarityLevel.established
    assert relationship.conversation_depth == ConversationDepth.medium
    assert conversation.summary == "The user shared a May 4 birthday."
    message_repo.count_for_user_and_companion.assert_awaited_once_with(
        user_id=user_id,
        companion_id=companion_id,
        role=MessageRole.user,
    )
    llm.generate.assert_awaited_once()
    assert "Earlier summary" in llm.generate.await_args.args[0][1].content
    session.commit.assert_awaited_once()


async def test_summary_generation_is_skipped_before_threshold() -> None:
    relationship = SimpleNamespace(
        familiarity_level=FamiliarityLevel.new,
        conversation_depth=ConversationDepth.light,
    )
    session = SimpleNamespace(add=Mock(), commit=AsyncMock(), rollback=AsyncMock())
    llm = SimpleNamespace(generate=AsyncMock())
    service = ContinuityService(
        conversation_repo=SimpleNamespace(get_by_id_for_user=AsyncMock()),
        message_repo=SimpleNamespace(
            session=session,
            count_for_user_and_companion=AsyncMock(return_value=1),
            count_for_conversation=AsyncMock(return_value=2),
        ),
        relationship_repo=SimpleNamespace(
            get_or_create=AsyncMock(return_value=relationship)
        ),
        llm_provider=llm,
        settings=Settings(_env_file=None, SUMMARY_TRIGGER_MESSAGE_COUNT=30),
    )
    await service.update_after_turn(
        conversation_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        companion_id=uuid.uuid4(),
    )
    llm.generate.assert_not_awaited()
