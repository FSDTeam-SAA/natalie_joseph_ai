from __future__ import annotations

import hashlib
import struct
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderError, ValidationError
from app.core.security import AuthContext
from app.db.models.message import MessageRole, MessageType
from app.schemas.chat import ChatResponse, ChatUsage
from app.services.voice_service import VoiceService
from app.storage.base import StoredObject
from app.voice.base import SynthesisResult, TranscriptionResult, VoiceUsage


def _wav_audio(*, duration_seconds: float = 0.1, sample_rate: int = 8000) -> bytes:
    data_size = max(2, int(duration_seconds * sample_rate) * 2)
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    payload = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    payload += b"data" + struct.pack("<I", data_size) + bytes(data_size)
    return b"RIFF" + struct.pack("<I", len(payload) + 4) + b"WAVE" + payload


def _auth(*features: str) -> AuthContext:
    return AuthContext(
        user_id=uuid.uuid4(),
        adult_eligible=False,
        entitled=False,
        raw_claims={},
        features=frozenset(features),
    )


def _service() -> tuple[VoiceService, SimpleNamespace]:
    session = SimpleNamespace(
        add=Mock(),
        commit=AsyncMock(),
        rollback=AsyncMock(),
    )
    user = SimpleNamespace(id=uuid.uuid4())
    companion = SimpleNamespace(
        id=uuid.uuid4(),
        slug="lina",
        active=True,
        voice_config={"voice_id": "voice-lina", "settings": {"stability": 0.5}},
    )
    conversation = SimpleNamespace(
        id=uuid.uuid4(), user_id=user.id, companion_id=companion.id
    )
    assistant = SimpleNamespace(id=uuid.uuid4(), media_asset_id=None)
    chat = ChatResponse(
        message_id=assistant.id,
        conversation_id=conversation.id,
        companion_id=companion.id,
        response="A warm hello",
        created_at=datetime.now(UTC),
        usage=ChatUsage(input_tokens=10, output_tokens=4),
    )
    dependencies = SimpleNamespace(
        session=session,
        user=user,
        companion=companion,
        conversation=conversation,
        assistant=assistant,
        chat=chat,
        user_repo=SimpleNamespace(get_or_create_by_external_user_id=AsyncMock(return_value=user)),
        companion_service=SimpleNamespace(get_active_profile=AsyncMock(return_value=companion)),
        conversation_repo=SimpleNamespace(get_by_id_for_user=AsyncMock(return_value=conversation)),
        message_repo=SimpleNamespace(
            session=session,
            acquire_idempotency_lock=AsyncMock(),
            get_ai_event_by_request_id=AsyncMock(return_value=None),
            get_by_id_for_conversation=AsyncMock(return_value=assistant),
            get_user_by_request_id=AsyncMock(return_value=None),
            get_first_assistant_after=AsyncMock(),
        ),
        media_repo=SimpleNamespace(
            get_by_id_for_user=AsyncMock(return_value=None),
            add=AsyncMock(),
        ),
        chat_service=SimpleNamespace(send_message=AsyncMock(return_value=chat)),
        voice_provider=SimpleNamespace(
            transcribe=AsyncMock(
                return_value=TranscriptionResult(
                    text="Hello from audio",
                    model="scribe_v2",
                    usage=VoiceUsage(request_id="stt-request"),
                    language_code="en",
                )
            ),
            synthesize=AsyncMock(
                return_value=SynthesisResult(
                    audio=b"ID3mp3 bytes",
                    mime_type="audio/mpeg",
                    model="eleven_flash_v2_5",
                    usage=VoiceUsage(character_cost=12, request_id="tts-request"),
                )
            ),
        ),
        storage=SimpleNamespace(
            put=AsyncMock(
                return_value=StoredObject(key="0" * 32 + ".mp3", byte_size=9, sha256="a" * 64)
            ),
            delete=AsyncMock(),
        ),
    )

    async def add_asset(asset):
        asset.id = uuid.uuid4()
        return asset

    dependencies.media_repo.add.side_effect = add_asset
    service = VoiceService(
        user_repo=dependencies.user_repo,
        companion_service=dependencies.companion_service,
        conversation_repo=dependencies.conversation_repo,
        message_repo=dependencies.message_repo,
        media_repo=dependencies.media_repo,
        chat_service=dependencies.chat_service,
        voice_provider=dependencies.voice_provider,
        storage=dependencies.storage,
        settings=Settings(
            _env_file=None,
            ENABLE_VOICE_GENERATION=True,
            ENABLE_VOICE_INPUT=True,
            ELEVENLABS_API_KEY="test",
            COMPANION_VOICE_ID_MAP={"lina": "voice-lina"},
        ),
    )
    return service, dependencies


async def test_text_message_uses_chat_pipeline_and_generates_companion_audio() -> None:
    service, deps = _service()
    response = await service.process_text(
        _auth("chat", "voice_output"),
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        message="Please answer with voice",
        idempotency_key=uuid.uuid4(),
        background_tasks=SimpleNamespace(add_task=Mock()),
    )

    deps.chat_service.send_message.assert_awaited_once()
    deps.voice_provider.synthesize.assert_awaited_once()
    assert deps.voice_provider.synthesize.await_args.kwargs["voice_id"] == "voice-lina"
    assert response.message_type == "audio"
    assert response.media is not None
    assert response.media.mime_type == "audio/mpeg"


async def test_completed_voice_retry_reuses_media_without_second_tts() -> None:
    service, deps = _service()
    deps.assistant.media_asset_id = uuid.uuid4()
    existing = SimpleNamespace(
        id=deps.assistant.media_asset_id,
        kind=SimpleNamespace(value="audio"),
        mime_type="audio/mpeg",
        byte_size=123,
        asset_metadata={},
    )
    deps.media_repo.get_by_id_for_user.return_value = existing

    response = await service.process_text(
        _auth("chat", "voice_output"),
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        message="Please answer with voice",
        idempotency_key=uuid.uuid4(),
        background_tasks=SimpleNamespace(add_task=Mock()),
    )

    deps.voice_provider.synthesize.assert_not_awaited()
    assert response.media is not None and response.media.id == existing.id


async def test_voice_provider_failure_does_not_write_media() -> None:
    service, deps = _service()
    deps.voice_provider.synthesize.side_effect = ProviderError("upstream unavailable")

    with pytest.raises(ProviderError, match="upstream unavailable"):
        await service.process_text(
            _auth("chat", "voice_output"),
            conversation_id=deps.conversation.id,
            companion_id=deps.companion.id,
            message="Voice please",
            idempotency_key=uuid.uuid4(),
            background_tasks=SimpleNamespace(add_task=Mock()),
        )

    deps.storage.put.assert_not_awaited()
    deps.media_repo.add.assert_not_awaited()


async def test_invalid_provider_audio_is_rejected_before_storage() -> None:
    service, deps = _service()
    deps.voice_provider.synthesize.return_value = SynthesisResult(
        audio=b"this is not MPEG audio",
        mime_type="audio/mpeg",
        model="eleven_flash_v2_5",
        usage=VoiceUsage(character_cost=12, request_id="tts-request"),
    )

    with pytest.raises(ProviderError, match="invalid audio data"):
        await service.process_text(
            _auth("chat", "voice_output"),
            conversation_id=deps.conversation.id,
            companion_id=deps.companion.id,
            message="Voice please",
            idempotency_key=uuid.uuid4(),
            background_tasks=SimpleNamespace(add_task=Mock()),
        )

    deps.storage.put.assert_not_awaited()
    deps.media_repo.add.assert_not_awaited()


async def test_voice_upload_rejects_mime_spoof_before_stt() -> None:
    service, deps = _service()
    with pytest.raises(ValidationError, match="does not match"):
        await service.process(
            _auth("chat", "voice_input"),
            conversation_id=deps.conversation.id,
            companion_id=deps.companion.id,
            audio=b"not a wave file",
            mime_type="audio/wav",
            filename="message.wav",
            request_voice_response=False,
            idempotency_key=uuid.uuid4(),
            background_tasks=SimpleNamespace(add_task=Mock()),
        )
    deps.voice_provider.transcribe.assert_not_awaited()


async def test_voice_upload_transcribes_then_uses_the_shared_chat_pipeline() -> None:
    service, deps = _service()
    response = await service.process(
        _auth("chat", "voice_input"),
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        audio=_wav_audio(),
        mime_type="audio/wav",
        filename="message.wav",
        request_voice_response=False,
        idempotency_key=uuid.uuid4(),
        background_tasks=SimpleNamespace(add_task=Mock()),
    )

    deps.voice_provider.transcribe.assert_awaited_once()
    deps.chat_service.send_message.assert_awaited_once()
    assert (
        deps.chat_service.send_message.await_args.kwargs["user_message_type"]
        == MessageType.audio
    )
    assert response.transcript == "Hello from audio"
    assert response.media is None


async def test_completed_audio_retry_skips_second_transcription_and_chat_call() -> None:
    service, deps = _service()
    user_message = SimpleNamespace(
        message_type=MessageType.audio,
        content="Original transcript",
        sequence=10,
    )
    assistant = SimpleNamespace(
        id=uuid.uuid4(),
        role=MessageRole.assistant,
        content="Existing response",
        created_at=datetime.now(UTC),
        input_tokens=12,
        output_tokens=5,
        message_type=MessageType.text,
        media_asset_id=None,
    )
    deps.message_repo.get_user_by_request_id.return_value = user_message
    deps.message_repo.get_first_assistant_after.return_value = assistant

    response = await service.process(
        _auth("chat", "voice_input"),
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        audio=_wav_audio(),
        mime_type="audio/wav",
        filename="message.wav",
        request_voice_response=False,
        idempotency_key=uuid.uuid4(),
        background_tasks=SimpleNamespace(add_task=Mock()),
    )

    deps.voice_provider.transcribe.assert_not_awaited()
    deps.chat_service.send_message.assert_not_awaited()
    assert response.response == "Existing response"
    assert response.transcript == "Original transcript"


async def test_voice_upload_rejects_overlong_audio_before_context_or_stt() -> None:
    service, deps = _service()
    service.settings.MAX_AUDIO_DURATION_SECONDS = 0.05

    with pytest.raises(ValidationError, match="duration exceeds"):
        await service.process(
            _auth("chat", "voice_input"),
            conversation_id=deps.conversation.id,
            companion_id=deps.companion.id,
            audio=_wav_audio(duration_seconds=0.1),
            mime_type="audio/wav",
            filename="message.wav",
            request_voice_response=False,
            idempotency_key=uuid.uuid4(),
            background_tasks=SimpleNamespace(add_task=Mock()),
        )

    deps.user_repo.get_or_create_by_external_user_id.assert_not_awaited()
    deps.voice_provider.transcribe.assert_not_awaited()


@pytest.mark.parametrize(
    ("transcript", "error"),
    [("   ", "transcribable speech"), ("x" * 4001, "exceeds the supported")],
)
async def test_invalid_transcript_is_terminal_and_never_reaches_chat(
    transcript: str, error: str
) -> None:
    service, deps = _service()
    deps.voice_provider.transcribe.return_value = TranscriptionResult(
        text=transcript,
        model="scribe_v2",
        usage=VoiceUsage(request_id="stt-request"),
    )

    with pytest.raises(ValidationError, match=error):
        await service.process(
            _auth("chat", "voice_input"),
            conversation_id=deps.conversation.id,
            companion_id=deps.companion.id,
            audio=_wav_audio(),
            mime_type="audio/wav",
            filename="message.wav",
            request_voice_response=False,
            idempotency_key=uuid.uuid4(),
            background_tasks=SimpleNamespace(add_task=Mock()),
        )

    deps.chat_service.send_message.assert_not_awaited()
    attempt = next(
        call.args[0]
        for call in deps.session.add.call_args_list
        if getattr(call.args[0], "event_type", None) == "voice_transcription_attempt"
    )
    assert attempt.event_metadata["status"] in {"empty_transcript", "oversized_transcript"}
    assert "transcript" not in attempt.event_metadata


async def test_existing_stt_attempt_checks_fingerprint_and_never_repurchases() -> None:
    service, deps = _service()
    audio = _wav_audio()
    previous = SimpleNamespace(
        event_metadata={"status": "started", "input_sha256": "different"}
    )
    deps.message_repo.get_ai_event_by_request_id.return_value = previous

    with pytest.raises(ValidationError, match="different audio upload"):
        await service.process(
            _auth("chat", "voice_input"),
            conversation_id=deps.conversation.id,
            companion_id=deps.companion.id,
            audio=audio,
            mime_type="audio/wav",
            filename="message.wav",
            request_voice_response=False,
            idempotency_key=uuid.uuid4(),
            background_tasks=SimpleNamespace(add_task=Mock()),
        )

    deps.voice_provider.transcribe.assert_not_awaited()


async def test_terminal_tts_attempt_is_not_purchased_again() -> None:
    service, deps = _service()
    previous = SimpleNamespace(
        event_metadata={
            "status": "invalid_output",
            "input_sha256": hashlib.sha256(deps.chat.response.encode("utf-8")).hexdigest(),
        }
    )
    deps.message_repo.get_ai_event_by_request_id.return_value = previous

    with pytest.raises(ProviderError, match="use a new idempotency key"):
        await service.process_text(
            _auth("chat", "voice_output"),
            conversation_id=deps.conversation.id,
            companion_id=deps.companion.id,
            message="Voice please",
            idempotency_key=uuid.uuid4(),
            background_tasks=SimpleNamespace(add_task=Mock()),
        )

    deps.voice_provider.synthesize.assert_not_awaited()
