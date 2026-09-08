from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.core.config import Settings
from app.core.security import AuthContext
from app.db.models.message import MessageRole
from app.llm.base import LLMResult, LLMUsage
from app.moderation.base import ModerationResult
from app.services.proactive_service import ProactiveService


async def test_proactive_message_uses_persona_history_memory_and_is_assistant_only() -> None:
    user = SimpleNamespace(id=uuid.uuid4(), locale=None, timezone=None)
    companion = SimpleNamespace(
        id=uuid.uuid4(),
        name="Lina",
        version=1,
        personality_config={},
        communication_config={},
        background_config={},
        interest_config={},
    )
    conversation = SimpleNamespace(id=uuid.uuid4(), summary="The user likes morning walks.")
    context = SimpleNamespace(user=user, companion=companion, conversation=conversation)
    persisted: list[object] = []

    async def refresh(message):
        message.id = uuid.uuid4()
        message.created_at = datetime.now(UTC)

    session = SimpleNamespace(
        add=lambda item: persisted.append(item),
        add_all=lambda items: persisted.extend(items),
        commit=AsyncMock(),
        refresh=AsyncMock(side_effect=refresh),
    )
    llm = SimpleNamespace(
        generate=AsyncMock(
            return_value=LLMResult(
                text="I was thinking of you—how is your morning going?",
                usage=LLMUsage(input_tokens=40, output_tokens=12),
                model="grok-4.6",
            )
        )
    )
    service = ProactiveService(
        message_repo=SimpleNamespace(
            session=session,
            acquire_idempotency_lock=AsyncMock(),
            get_by_request_id=AsyncMock(return_value=None),
            get_recent_for_conversation=AsyncMock(
                return_value=[
                    SimpleNamespace(role=MessageRole.user, content="I enjoy morning walks")
                ]
            ),
        ),
        relationship_repo=SimpleNamespace(
            get_or_create=AsyncMock(return_value=SimpleNamespace())
        ),
        llm_provider=llm,
        moderation_provider=SimpleNamespace(
            moderate_text=AsyncMock(
                side_effect=[
                    ModerationResult(False, {}, {}, {}),
                    ModerationResult(False, {}, {}, {}),
                ]
            )
        ),
        prompt_builder=SimpleNamespace(build_system_prompt=Mock(return_value="persona prompt")),
        memory_service=SimpleNamespace(
            retrieve_relevant=AsyncMock(return_value=["routine: morning walks"])
        ),
        settings=Settings(_env_file=None, XAI_API_KEY="test"),
    )
    auth = AuthContext(
        user_id=uuid.uuid4(),
        adult_eligible=False,
        entitled=False,
        raw_claims={"source": "trusted_backend"},
        features=frozenset({"proactive"}),
    )

    response = await service.generate(
        auth,
        context=context,
        reason="scheduled morning check-in",
        idempotency_key=uuid.uuid4(),
    )

    generated_messages = llm.generate.await_args.args[0]
    assert generated_messages[-1].role == "developer"
    assert "scheduled morning check-in" in generated_messages[-1].content
    stored_messages = [item for item in persisted if hasattr(item, "role")]
    assert [item.role for item in stored_messages] == [MessageRole.system, MessageRole.assistant]
    assert response.response.startswith("I was thinking")
    session.commit.assert_awaited_once()
