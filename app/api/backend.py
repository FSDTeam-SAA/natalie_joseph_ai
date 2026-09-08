"""Private orchestration contract called by the canonical NestJS backend."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from app.api.audio_upload import read_audio_upload
from app.api.deps import (
    enforce_ai_rate_limit,
    get_backend_context_service,
    get_chat_routing_service,
    get_image_service,
    get_optional_voice_service,
    get_proactive_service,
    get_story_service,
    get_voice_service,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError
from app.core.security import AuthContext, get_current_auth_context, require_feature
from app.schemas.backend import (
    BackendChatRequest,
    BackendImageRequest,
    BackendProactiveRequest,
    BackendStoryEventRequest,
    BackendStoryEventResponse,
    BackendVoiceRequest,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.media import ImageGenerationRequest, ImageGenerationResponse
from app.schemas.voice import VoiceChatResponse
from app.services.backend_context_service import BackendContextService
from app.services.chat_routing_service import ChatRoutingService
from app.services.image_service import ImageService
from app.services.proactive_service import ProactiveService
from app.services.story_service import StoryService
from app.services.voice_service import VoiceService

router = APIRouter(prefix="/internal", tags=["Trusted Backend Integration"])


@router.put(
    "/companions/{companion_reference}/story-events",
    response_model=BackendStoryEventResponse,
)
async def upsert_backend_story_event(
    companion_reference: str,
    body: BackendStoryEventRequest,
    auth: AuthContext = Depends(get_current_auth_context),
    story_service: StoryService = Depends(get_story_service),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> BackendStoryEventResponse:
    event = await story_service.ingest(
        auth,
        companion_reference=companion_reference,
        request=body,
    )
    return BackendStoryEventResponse.model_validate(event)


@router.post("/companions/{companion_reference}/messages", response_model=ChatResponse)
async def create_backend_chat_message(
    companion_reference: str,
    body: BackendChatRequest,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(get_current_auth_context),
    context_service: BackendContextService = Depends(get_backend_context_service),
    chat_service: ChatRoutingService = Depends(get_chat_routing_service),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> ChatResponse:
    context = await context_service.resolve(
        auth,
        companion_reference=companion_reference,
        external_conversation_id=body.external_conversation_id,
    )
    return await chat_service.send_message(
        auth,
        ChatRequest(
            conversation_id=context.conversation.id,
            companion_id=context.companion.id,
            message=body.message,
            idempotency_key=body.idempotency_key,
        ),
        background_tasks,
    )


@router.post("/companions/{companion_reference}/voice", response_model=VoiceChatResponse)
async def create_backend_voice_response(
    companion_reference: str,
    body: BackendVoiceRequest,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(get_current_auth_context),
    context_service: BackendContextService = Depends(get_backend_context_service),
    voice_service: VoiceService = Depends(get_voice_service),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> VoiceChatResponse:
    context = await context_service.resolve(
        auth,
        companion_reference=companion_reference,
        external_conversation_id=body.external_conversation_id,
    )
    return await voice_service.process_text(
        auth,
        conversation_id=context.conversation.id,
        companion_id=context.companion.id,
        message=body.message,
        idempotency_key=body.idempotency_key,
        background_tasks=background_tasks,
    )


@router.post(
    "/companions/{companion_reference}/voice-input",
    response_model=VoiceChatResponse,
    summary="Transcribe a trusted-backend voice message and run the shared chat pipeline",
)
async def create_backend_voice_input(
    companion_reference: str,
    idempotency_key: Annotated[uuid.UUID, Form()],
    audio: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    external_conversation_id: Annotated[str | None, Form()] = None,
    request_voice_response: Annotated[bool, Form()] = False,
    auth: AuthContext = Depends(get_current_auth_context),
    context_service: BackendContextService = Depends(get_backend_context_service),
    voice_service: VoiceService = Depends(get_voice_service),
    settings: Settings = Depends(get_settings),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> VoiceChatResponse:
    context = await context_service.resolve(
        auth,
        companion_reference=companion_reference,
        external_conversation_id=external_conversation_id,
    )
    upload = await read_audio_upload(audio, max_bytes=settings.MAX_AUDIO_UPLOAD_BYTES)
    return await voice_service.process(
        auth,
        conversation_id=context.conversation.id,
        companion_id=context.companion.id,
        audio=upload.data,
        mime_type=upload.mime_type,
        filename=upload.filename,
        request_voice_response=request_voice_response,
        idempotency_key=idempotency_key,
        background_tasks=background_tasks,
    )


@router.post("/companions/{companion_reference}/images", response_model=ImageGenerationResponse)
async def create_backend_image(
    companion_reference: str,
    body: BackendImageRequest,
    auth: AuthContext = Depends(get_current_auth_context),
    context_service: BackendContextService = Depends(get_backend_context_service),
    image_service: ImageService = Depends(get_image_service),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> ImageGenerationResponse:
    context = await context_service.resolve(
        auth,
        companion_reference=companion_reference,
        external_conversation_id=body.external_conversation_id,
    )
    return await image_service.generate(
        auth,
        ImageGenerationRequest(
            conversation_id=context.conversation.id,
            companion_id=context.companion.id,
            prompt=body.prompt,
            trigger=body.trigger,
            idempotency_key=body.idempotency_key,
        ),
    )


@router.post(
    "/companions/{companion_reference}/proactive",
    response_model=ChatResponse | VoiceChatResponse,
)
async def create_backend_proactive_message(
    companion_reference: str,
    body: BackendProactiveRequest,
    auth: AuthContext = Depends(get_current_auth_context),
    context_service: BackendContextService = Depends(get_backend_context_service),
    proactive_service: ProactiveService = Depends(get_proactive_service),
    voice_service: VoiceService | None = Depends(get_optional_voice_service),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> ChatResponse | VoiceChatResponse:
    if body.response_mode == "voice" and voice_service is None:
        raise ServiceUnavailableError("ElevenLabs voice processing is not configured.")
    # Check both grants before the paid Grok call.  Voice mode consumes the
    # proactive allowance *and* the separate voice-output allowance; failing
    # here avoids generating text that cannot be delivered.
    if body.response_mode == "voice":
        require_feature(auth, "voice_output")
    context = await context_service.resolve(
        auth,
        companion_reference=companion_reference,
        external_conversation_id=body.external_conversation_id,
    )
    chat = await proactive_service.generate(
        auth,
        context=context,
        reason=body.reason,
        idempotency_key=body.idempotency_key,
    )
    if body.response_mode == "voice":
        assert voice_service is not None  # narrowed by the preflight check above
        return await voice_service.synthesize_existing_response(
            auth,
            chat=chat,
            conversation_id=context.conversation.id,
            companion_id=context.companion.id,
            idempotency_key=body.idempotency_key,
        )
    return chat
