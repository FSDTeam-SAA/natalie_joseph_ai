from __future__ import annotations

import uuid
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.api.backend import create_backend_proactive_message, create_backend_voice_input
from app.core.config import Settings
from app.core.exceptions import AuthorizationError, ServiceUnavailableError, ValidationError
from app.core.security import AuthContext
from app.schemas.backend import BackendProactiveRequest
from app.schemas.chat import ChatResponse, ChatUsage


def _auth() -> AuthContext:
    return AuthContext(
        user_id=uuid.uuid4(),
        adult_eligible=False,
        entitled=True,
        raw_claims={"source": "trusted_backend"},
        features=frozenset({"proactive"}),
    )


def _chat() -> ChatResponse:
    return ChatResponse(
        message_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        companion_id=uuid.uuid4(),
        response="Thinking of you.",
        created_at=datetime.now(UTC),
        usage=ChatUsage(input_tokens=4, output_tokens=3),
    )


async def test_text_proactive_does_not_require_voice_configuration() -> None:
    chat = _chat()
    context = SimpleNamespace(
        user=SimpleNamespace(id=uuid.uuid4()),
        companion=SimpleNamespace(id=chat.companion_id),
        conversation=SimpleNamespace(id=chat.conversation_id),
    )
    context_service = SimpleNamespace(resolve=AsyncMock(return_value=context))
    proactive_service = SimpleNamespace(generate=AsyncMock(return_value=chat))

    result = await create_backend_proactive_message(
        companion_reference="backend-lina",
        body=BackendProactiveRequest(
            idempotency_key=uuid.uuid4(),
            reason="scheduled check-in",
            response_mode="text",
        ),
        auth=_auth(),
        context_service=context_service,
        proactive_service=proactive_service,
        voice_service=None,
        _rate_limit=None,
    )

    assert result is chat
    proactive_service.generate.assert_awaited_once()


async def test_voice_proactive_fails_before_creating_text_when_voice_is_unconfigured() -> None:
    context_service = SimpleNamespace(resolve=AsyncMock())
    proactive_service = SimpleNamespace(generate=AsyncMock())

    with pytest.raises(ServiceUnavailableError, match="not configured"):
        await create_backend_proactive_message(
            companion_reference="backend-lina",
            body=BackendProactiveRequest(
                idempotency_key=uuid.uuid4(),
                reason="scheduled check-in",
                response_mode="voice",
            ),
            auth=_auth(),
            context_service=context_service,
            proactive_service=proactive_service,
            voice_service=None,
            _rate_limit=None,
        )

    context_service.resolve.assert_not_awaited()
    proactive_service.generate.assert_not_awaited()


async def test_voice_proactive_checks_voice_grant_before_creating_text() -> None:
    context_service = SimpleNamespace(resolve=AsyncMock())
    proactive_service = SimpleNamespace(generate=AsyncMock())
    voice_service = SimpleNamespace()

    with pytest.raises(AuthorizationError, match="voice_output"):
        await create_backend_proactive_message(
            companion_reference="backend-lina",
            body=BackendProactiveRequest(
                idempotency_key=uuid.uuid4(),
                reason="scheduled check-in",
                response_mode="voice",
            ),
            auth=_auth(),
            context_service=context_service,
            proactive_service=proactive_service,
            voice_service=voice_service,
            _rate_limit=None,
        )

    context_service.resolve.assert_not_awaited()
    proactive_service.generate.assert_not_awaited()


async def test_backend_voice_input_resolves_external_refs_and_forwards_multipart_audio() -> None:
    auth = AuthContext(
        user_id=uuid.uuid4(),
        adult_eligible=False,
        entitled=True,
        raw_claims={"source": "trusted_backend"},
        features=frozenset({"chat", "voice_input"}),
    )
    context = SimpleNamespace(
        companion=SimpleNamespace(id=uuid.uuid4()),
        conversation=SimpleNamespace(id=uuid.uuid4()),
    )
    context_service = SimpleNamespace(resolve=AsyncMock(return_value=context))
    expected = object()
    voice_service = SimpleNamespace(process=AsyncMock(return_value=expected))
    audio = UploadFile(
        BytesIO(b"encoded-audio"),
        filename="message.webm",
        headers=Headers({"content-type": "audio/webm; codecs=opus"}),
    )
    request_id = uuid.uuid4()
    background_tasks = SimpleNamespace(add_task=AsyncMock())

    result = await create_backend_voice_input(
        companion_reference="backend-lina-id",
        idempotency_key=request_id,
        audio=audio,
        background_tasks=background_tasks,
        external_conversation_id="backend-thread-42",
        request_voice_response=False,
        auth=auth,
        context_service=context_service,
        voice_service=voice_service,
        settings=Settings(_env_file=None, MAX_AUDIO_UPLOAD_BYTES=1024),
        _rate_limit=None,
    )

    assert result is expected
    context_service.resolve.assert_awaited_once_with(
        auth,
        companion_reference="backend-lina-id",
        external_conversation_id="backend-thread-42",
    )
    voice_service.process.assert_awaited_once_with(
        auth,
        conversation_id=context.conversation.id,
        companion_id=context.companion.id,
        audio=b"encoded-audio",
        mime_type="audio/webm",
        filename="message.webm",
        request_voice_response=False,
        idempotency_key=request_id,
        background_tasks=background_tasks,
    )


async def test_backend_voice_input_reads_only_one_byte_past_configured_limit() -> None:
    context = SimpleNamespace(
        companion=SimpleNamespace(id=uuid.uuid4()),
        conversation=SimpleNamespace(id=uuid.uuid4()),
    )
    context_service = SimpleNamespace(resolve=AsyncMock(return_value=context))
    voice_service = SimpleNamespace(process=AsyncMock())
    audio = UploadFile(
        BytesIO(b"12345"),
        filename="message.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )

    with pytest.raises(ValidationError, match="exceeds"):
        await create_backend_voice_input(
            companion_reference="backend-lina-id",
            idempotency_key=uuid.uuid4(),
            audio=audio,
            background_tasks=SimpleNamespace(add_task=AsyncMock()),
            external_conversation_id=None,
            request_voice_response=False,
            auth=_auth(),
            context_service=context_service,
            voice_service=voice_service,
            settings=Settings(_env_file=None, MAX_AUDIO_UPLOAD_BYTES=4),
            _rate_limit=None,
        )

    voice_service.process.assert_not_awaited()
