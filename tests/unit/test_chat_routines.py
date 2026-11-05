"""Exercise real catalogue adaptation, routine resolution and prompt assembly together."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import BackgroundTasks

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.core.security import AuthContext
from app.db.models.message import MessageRole, MessageType
from app.llm.base import LLMResult, LLMStructuredResult, LLMUsage
from app.llm.prompts.builder import PromptBuilder
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.services.companion_service import CompanionService
from app.services.routine_service import resolve_current_activity

_NOW = datetime(2026, 11, 2, 8, tzinfo=UTC)


def _service(monkeypatch, name="Chloé", timezone="Europe/London"):
    settings = Settings(_env_file=None, COMPANION_CATALOGUE_BASE_URL="https://catalogue.test")
    profile = {
        "id": str(uuid.uuid4()),
        "name": name,
        "status": True,
        "routine": {
            "timezone": timezone,
            "weekly": [
                {"day": 1, "start": "06:00", "end": "12:00", "activity": "Morning gym"},
                {"day": 1, "start": "12:00", "end": "18:00", "activity": "Café"},
                {"day": 7, "start": "23:00", "end": "06:00", "activity": "Sleep"},
            ],
        },
    }
    companion_id = uuid.UUID(profile["id"])
    user = SimpleNamespace(id=uuid.uuid4(), locale=None, timezone="Asia/Dhaka")
    conversation = SimpleNamespace(
        id=uuid.uuid4(), companion_id=companion_id, summary="Previously she was at dinner."
    )
    catalogue = CompanionService(
        settings,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"success": True, "data": profile})
        ),
    )
    persisted = []

    async def refresh(message):
        message.id = uuid.uuid4()
        message.created_at = _NOW
        # Simulate the column default normally populated by a database flush.
        if message.message_type is None:
            message.message_type = MessageType.text

    messages = SimpleNamespace(
        session=SimpleNamespace(
            add=persisted.append, commit=AsyncMock(), refresh=AsyncMock(side_effect=refresh)
        ),
        acquire_idempotency_lock=AsyncMock(),
        get_user_by_request_id=AsyncMock(return_value=None),
        get_first_assistant_after=AsyncMock(),
        get_recent_for_conversation=AsyncMock(
            return_value=[SimpleNamespace(role=MessageRole.assistant, content="I'm at dinner.")]
        ),
    )
    llm = SimpleNamespace(
        generate_structured=AsyncMock(
            return_value=LLMStructuredResult(
                data={"consistent": True, "reason": "none"},
                model="test-review",
                usage=LLMUsage(input_tokens=20, output_tokens=8),
            )
        ),
        generate=AsyncMock(
            return_value=LLMResult(
                text="A natural reply.",
                model="test",
                usage=LLMUsage(input_tokens=10, output_tokens=4),
            )
        ),
    )
    monkeypatch.setattr(
        "app.services.chat_service.resolve_current_activity",
        lambda routine: resolve_current_activity(routine, now=_NOW),
    )
    service = ChatService(
        user_repo=SimpleNamespace(get_or_create_by_external_user_id=AsyncMock(return_value=user)),
        companion_service=catalogue,
        conversation_repo=SimpleNamespace(get_by_id_for_user=AsyncMock(return_value=conversation)),
        message_repo=messages,
        relationship_repo=SimpleNamespace(get_or_create=AsyncMock(return_value=None)),
        llm_provider=llm,
        prompt_builder=PromptBuilder(prompt_version="test"),
        memory_service=SimpleNamespace(retrieve_relevant=AsyncMock(return_value=[])),
        settings=settings,
    )
    auth = AuthContext(user_id=uuid.uuid4(), adult_eligible=False, entitled=True, raw_claims={})

    def request():
        return ChatRequest(
            companion_id=companion_id,
            conversation_id=conversation.id,
            message="What are you doing?",
            idempotency_key=uuid.uuid4(),
        )

    return service, SimpleNamespace(
        profile=profile,
        llm=llm,
        messages=messages,
        auth=auth,
        request=request,
        persisted=persisted,
    )


def _routine_data(prompt):
    return json.loads(prompt.split("<routine_data>")[1].split("</routine_data>")[0])


@pytest.mark.parametrize("channel", ["text", "voice"])
@pytest.mark.parametrize(
    ("name", "timezone", "expected"),
    [
        ("Elena", "Asia/Dubai", "Café"),
        ("Chloé", "Europe/London", "Morning gym"),
        ("Thalia", "America/New_York", "Sleep"),
        ("Lina", "Europe/Paris", "Morning gym"),
        ("Luna", "Asia/Tokyo", "Café"),
    ],
)
async def test_each_companion_and_response_channel_uses_its_own_local_routine(
    monkeypatch, name, timezone, expected, channel
) -> None:
    service, deps = _service(monkeypatch, name, timezone)
    await service.send_message(
        deps.auth, deps.request(), BackgroundTasks(), response_channel=channel
    )
    prompt = deps.llm.generate.await_args.args[0][0].content
    assert f"You are {name}" in prompt
    assert _routine_data(prompt)["activity"] == expected
    assert _routine_data(prompt)["timezone"] == timezone
    assert "precedence over conflicting older messages" in prompt
    event = next(
        item for item in deps.persisted if getattr(item, "event_type", None) == "chat_completion"
    )
    assert event.event_metadata["routine_context"]["activity"] == expected
    assert event.event_metadata["routine_validation"]["checks"][0]["consistent"] is True
    recent_directive = deps.llm.generate.await_args.args[0][-2]
    assert recent_directive.role == "developer"
    assert _routine_data(recent_directive.content)["activity"] == expected


async def test_backend_edit_and_removal_affect_the_next_new_reply(monkeypatch) -> None:
    service, deps = _service(monkeypatch)
    await service.send_message(deps.auth, deps.request(), BackgroundTasks())
    assert (
        _routine_data(deps.llm.generate.await_args.args[0][0].content)["activity"] == "Morning gym"
    )
    deps.profile["routine"]["weekly"][0]["activity"] = "At home"
    await service.send_message(deps.auth, deps.request(), BackgroundTasks())
    assert _routine_data(deps.llm.generate.await_args.args[0][0].content)["activity"] == "At home"
    deps.profile.pop("routine")
    await service.send_message(deps.auth, deps.request(), BackgroundTasks())
    assert "<routine_data>" not in deps.llm.generate.await_args.args[0][0].content


async def test_missing_invalid_and_unmatched_routines_keep_the_original_prompt(monkeypatch) -> None:
    service, deps = _service(monkeypatch)
    prompts = []
    for routine in [
        None,
        {"timezone": "Invalid/Zone", "weekly": []},
        {"timezone": "Europe/London", "weekly": []},
        {
            "timezone": "Europe/London",
            "weekly": [{"day": 2, "start": "06:00", "end": "12:00", "activity": "Gym"}],
        },
    ]:
        deps.profile["routine"] = routine
        await service.send_message(deps.auth, deps.request(), BackgroundTasks())
        prompts.append(deps.llm.generate.await_args.args[0][0].content)
    assert len(set(prompts)) == 1
    assert "<routine_data>" not in prompts[0]
    deps.llm.generate_structured.assert_not_awaited()


async def test_completed_retry_returns_the_original_reply_without_resolving_again(monkeypatch):
    service, deps = _service(monkeypatch)
    request = deps.request()
    response = await service.send_message(deps.auth, request, BackgroundTasks())
    deps.messages.get_user_by_request_id.return_value = SimpleNamespace(
        content=request.message, sequence=1, message_type=MessageType.text
    )
    deps.messages.get_first_assistant_after.return_value = deps.persisted[1]
    deps.profile["routine"]["weekly"][0]["activity"] = "Updated activity"
    resolver = Mock(side_effect=AssertionError("Retries must replay the stored response"))
    monkeypatch.setattr("app.services.chat_service.resolve_current_activity", resolver)
    replay = await service.send_message(deps.auth, request, BackgroundTasks())
    assert replay == response
    deps.llm.generate.assert_awaited_once()
    deps.llm.generate_structured.assert_awaited_once()
    resolver.assert_not_called()


async def test_rejected_draft_never_reaches_messages_or_background_memory(monkeypatch):
    service, deps = _service(monkeypatch)
    deps.llm.generate.side_effect = [
        LLMResult(
            text="I'm at dinner.", model="test", usage=LLMUsage(input_tokens=10, output_tokens=4)
        ),
        LLMResult(
            text="I'm at the gym.", model="test", usage=LLMUsage(input_tokens=11, output_tokens=5)
        ),
    ]
    deps.llm.generate_structured.side_effect = [
        LLMStructuredResult(
            data={"consistent": False, "reason": "activity"},
            model="review",
            usage=LLMUsage(input_tokens=20, output_tokens=8),
        ),
        LLMStructuredResult(
            data={"consistent": True, "reason": "none"},
            model="review",
            usage=LLMUsage(input_tokens=20, output_tokens=8),
        ),
    ]
    service.post_turn_processor = SimpleNamespace(process=AsyncMock())
    tasks = BackgroundTasks()
    response = await service.send_message(
        deps.auth, deps.request(), tasks, response_channel="voice"
    )
    assert response.response == "I'm at the gym."
    assert response.usage.input_tokens == 21
    assert all(getattr(item, "content", None) != "I'm at dinner." for item in deps.persisted)
    assert tasks.tasks[0].kwargs["assistant_message"] == "I'm at the gym."


async def test_unverified_reply_is_not_persisted(monkeypatch):
    service, deps = _service(monkeypatch)
    deps.llm.generate_structured.return_value = LLMStructuredResult(
        data={"consistent": False, "reason": "activity"},
        model="review",
        usage=LLMUsage(input_tokens=20, output_tokens=8),
    )
    with pytest.raises(ProviderError, match="routine-consistent reply"):
        await service.send_message(deps.auth, deps.request(), BackgroundTasks())
    assert deps.persisted == []
    deps.messages.session.commit.assert_not_awaited()
