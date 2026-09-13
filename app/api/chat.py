"""Unified authenticated chat endpoint for automatic text, voice, and image responses."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import ValidationError as PydanticValidationError
from starlette.datastructures import UploadFile

from app.api.audio_upload import read_audio_upload
from app.api.deps import (
    enforce_ai_rate_limit,
    get_chat_routing_service,
    get_optional_voice_service,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError, ValidationError
from app.core.security import AuthContext, get_current_auth_context
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_routing_service import ChatRoutingService, VoiceResponseClassifier
from app.services.voice_service import VoiceService

router = APIRouter(tags=["Chat"])
voice_response_classifier = VoiceResponseClassifier()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Unified text, voice, and image chat",
    description="Accepts multipart form data with optional audio. Explicit photo "
    "requests return images; voice input returns a spoken reply. Voice IDs stay private.",
    responses={
        422: {"description": "Invalid companion/conversation combination"},
        404: {"description": "Conversation not found"},
    },
    # The handler parses the raw multipart form so a text message and optional
    # audio upload can use the same endpoint.
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["conversation_id", "companion_id", "idempotency_key"],
                        "properties": {
                            "conversation_id": {"type": "string", "format": "uuid"},
                            "companion_id": {"type": "string", "format": "uuid"},
                            "idempotency_key": {"type": "string", "format": "uuid"},
                            "message": {
                                "type": "string",
                                "description": "Optional when audio is supplied.",
                            },
                            "audio": {"type": "string", "format": "binary"},
                        },
                    }
                },
            },
        }
    },
)
async def send_chat_message(
    request: Request,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(get_current_auth_context),
    service: ChatRoutingService = Depends(get_chat_routing_service),
    voice_service: VoiceService | None = Depends(get_optional_voice_service),
    settings: Settings = Depends(get_settings),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> ChatResponse:
    content_type = request.headers.get("content-type", "").lower()
    audio: UploadFile | None = None
    try:
        if not content_type.startswith("multipart/form-data"):
            raise ValidationError("Use multipart/form-data.")
        form = await request.form()
        audio_value = form.get("audio")
        # Swagger UI submits an empty optional binary form field as an empty
        # string when “Send empty value” is selected. Treat it as omitted.
        if audio_value == "":
            audio_value = None
        if audio_value is not None and not isinstance(audio_value, UploadFile):
            raise ValidationError("audio must be a file upload.")
        audio = audio_value
        body = ChatRequest.model_validate(
            {
                "conversation_id": form.get("conversation_id"),
                "companion_id": form.get("companion_id"),
                "message": form.get("message") or "[voice message]",
                "idempotency_key": form.get("idempotency_key"),
            }
        )
    except PydanticValidationError as exc:
        # These are caller-supplied identifiers and modes, so identifying the
        # invalid fields is safe and makes Swagger/API integrations actionable.
        fields = ", ".join(
            str(error["loc"][-1]) for error in exc.errors() if error.get("loc")
        )
        message = f"Invalid or missing field(s): {fields or 'request body'}."
        raise ValidationError(message) from exc
    except (ValueError, TypeError) as exc:
        raise ValidationError("The request payload is invalid.") from exc

    if audio is None:
        if voice_response_classifier.is_voice_request(body.message):
            if voice_service is None:
                raise ServiceUnavailableError("ElevenLabs voice processing is not configured.")
            return await voice_service.process_text(
                auth,
                conversation_id=body.conversation_id,
                companion_id=body.companion_id,
                message=body.message,
                idempotency_key=body.idempotency_key,
                background_tasks=background_tasks,
            )
        return await service.send_message(auth, body, background_tasks)

    if voice_service is None:
        raise ServiceUnavailableError("ElevenLabs voice processing is not configured.")
    upload = await read_audio_upload(audio, max_bytes=settings.MAX_AUDIO_UPLOAD_BYTES)
    return await voice_service.process(
        auth,
        conversation_id=body.conversation_id,
        companion_id=body.companion_id,
        audio=upload.data,
        mime_type=upload.mime_type,
        filename=upload.filename,
        request_voice_response=True,
        idempotency_key=body.idempotency_key,
        background_tasks=background_tasks,
    )
