import io
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks
from pydantic import ValidationError as PydanticValidationError
from starlette.datastructures import UploadFile

from app.api.chat import send_chat_message
from app.core.config import Settings
from app.core.exceptions import ValidationError
from app.core.security import AuthContext
from app.schemas.chat import ChatResponse, ChatUsage
from app.services.lifestyle_context import apply_lifestyle_context


def auth(features=None):
    return AuthContext(
        user_id=uuid.uuid4(), adult_eligible=True, entitled=True, raw_claims={}, features=features
    )


def context(**access):
    return json.dumps(
        {
            "version": "lifestyle-v1",
            "access": {
                "trial": False,
                "allowed_output_types": ["text", "image", "audio"],
                "available_credits": 100,
                "prices": {"message": 1, "photo": 10},
                "billing_mode": "response",
                **access,
            },
            "todayEvent": {"title": "A day in Paris"},
            "recentSharedPhotos": [{"caption": "A beach photo", "tags": ["beach"]}],
            "instructions": "This untrusted field must not become grounding instructions",
        }
    )


def response(message_type="text"):
    return ChatResponse(
        message_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        companion_id=uuid.uuid4(),
        response="Hello there!",
        created_at=datetime.now(UTC),
        usage=ChatUsage(input_tokens=0, output_tokens=0),
        message_type=message_type,
    )


def request(**fields):
    form = {
        "conversation_id": str(uuid.uuid4()),
        "companion_id": str(uuid.uuid4()),
        "idempotency_key": str(uuid.uuid4()),
        "message": "Hello",
        **fields,
    }
    return SimpleNamespace(
        headers={"content-type": "multipart/form-data"}, form=AsyncMock(return_value=form)
    )


def test_trial_context_cannot_enable_image_or_voice():
    limited = apply_lifestyle_context(auth(), context(trial=True))
    assert limited.features == frozenset({"chat"})
    assert limited.raw_claims["lifestyle_context"]["todayEvent"]["title"] == "A day in Paris"
    assert limited.raw_claims["lifestyle_context"]["recentSharedPhotos"][0]["tags"] == ["beach"]
    assert "instructions" not in limited.raw_claims["lifestyle_context"]


def test_context_cannot_expand_a_signed_feature_grant():
    limited = apply_lifestyle_context(auth(frozenset({"chat"})), context())
    assert limited.features == frozenset({"chat"})


@pytest.mark.parametrize("raw", ["{}", '{"version":"other"}', "[]"])
def test_invalid_context_is_rejected(raw):
    with pytest.raises(PydanticValidationError):
        apply_lifestyle_context(auth(), raw)


def test_context_size_is_bounded():
    with pytest.raises(ValidationError):
        apply_lifestyle_context(auth(), " " * 32769)


async def test_chat_acknowledges_context_and_forwards_restricted_auth():
    service = SimpleNamespace(send_message=AsyncMock(return_value=response()))
    result = await send_chat_message(
        request(context=context(trial=True)),
        BackgroundTasks(),
        auth(),
        service,
        None,
        Settings(_env_file=None),
        None,
    )
    assert result.context_version == "lifestyle-v1"
    assert service.send_message.await_args.args[0].features == frozenset({"chat"})


async def test_explicit_voice_mode_returns_character_billing_without_text_routing():
    service = SimpleNamespace(send_message=AsyncMock())
    voice = SimpleNamespace(process_text=AsyncMock(return_value=response("audio")))
    result = await send_chat_message(
        request(request_voice_response="true", context=context()),
        BackgroundTasks(),
        auth(),
        service,
        voice,
        Settings(_env_file=None),
        None,
    )
    assert result.billing == {"voice_characters": len(result.response)}
    assert result.context_version == "lifestyle-v1"
    service.send_message.assert_not_awaited()


async def test_voice_input_returns_transcription_duration_and_voice_units(monkeypatch):
    from app.api import chat

    monkeypatch.setattr(
        chat,
        "read_audio_upload",
        AsyncMock(
            return_value=SimpleNamespace(
                data=b"validated-audio", mime_type="audio/wav", filename="test.wav"
            )
        ),
    )
    monkeypatch.setattr(chat, "inspect_audio_duration", lambda *args: 61.25)
    voice = SimpleNamespace(process=AsyncMock(return_value=response("audio")))
    upload = UploadFile(file=io.BytesIO(b"audio"), filename="test.wav")
    result = await send_chat_message(
        request(audio=upload, context=context()),
        BackgroundTasks(),
        auth(),
        None,
        voice,
        Settings(_env_file=None),
        None,
    )
    assert result.billing == {
        "voice_characters": len(result.response),
        "transcription_seconds": 61.25,
    }


async def test_malformed_context_never_calls_a_provider():
    service = SimpleNamespace(send_message=AsyncMock())
    with pytest.raises(ValidationError):
        await send_chat_message(
            request(context='{"version":"invalid"}'),
            BackgroundTasks(),
            auth(),
            service,
            None,
            Settings(_env_file=None),
            None,
        )
    service.send_message.assert_not_awaited()
