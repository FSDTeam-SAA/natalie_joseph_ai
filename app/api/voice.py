from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from app.api.audio_upload import read_audio_upload
from app.api.deps import enforce_ai_rate_limit, get_voice_service
from app.core.config import Settings, get_settings
from app.core.security import AuthContext, get_current_auth_context
from app.schemas.voice import VoiceChatResponse
from app.services.voice_service import VoiceService

router = APIRouter(tags=["AI Voice"])


@router.post(
    "/chat/voice",
    response_model=VoiceChatResponse,
    summary="Send a voice message through the normal chat pipeline",
    description=(
        "Validates and transcribes user audio, then uses the same companion, "
        "history, memory, Grok, and moderation path as text chat. TTS is only "
        "generated when request_voice_response=true and the backend grants it."
    ),
)
async def send_voice_message(
    conversation_id: Annotated[uuid.UUID, Form()],
    companion_id: Annotated[uuid.UUID, Form()],
    idempotency_key: Annotated[uuid.UUID, Form()],
    audio: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    request_voice_response: Annotated[bool, Form()] = False,
    auth: AuthContext = Depends(get_current_auth_context),
    service: VoiceService = Depends(get_voice_service),
    settings: Settings = Depends(get_settings),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> VoiceChatResponse:
    upload = await read_audio_upload(audio, max_bytes=settings.MAX_AUDIO_UPLOAD_BYTES)
    return await service.process(
        auth,
        conversation_id=conversation_id,
        companion_id=companion_id,
        audio=upload.data,
        mime_type=upload.mime_type,
        filename=upload.filename,
        request_voice_response=request_voice_response,
        idempotency_key=idempotency_key,
        background_tasks=background_tasks,
    )
