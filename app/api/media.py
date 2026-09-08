from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Response

from app.api.deps import enforce_ai_rate_limit, get_image_service, get_media_service
from app.core.security import AuthContext, get_current_auth_context
from app.schemas.media import ImageGenerationRequest, ImageGenerationResponse
from app.services.image_service import ImageService
from app.services.media_service import MediaService

router = APIRouter(tags=["AI Media"])


@router.post(
    "/images/generate",
    response_model=ImageGenerationResponse,
    summary="Generate a reference-consistent companion image",
    description=(
        "The trusted backend must pre-authorize the image feature. "
        "Use trigger=contextual for a backend-scheduled spontaneous image; "
        "this AI service never schedules or bills it autonomously."
    ),
)
async def generate_companion_image(
    body: ImageGenerationRequest,
    auth: AuthContext = Depends(get_current_auth_context),
    service: ImageService = Depends(get_image_service),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> ImageGenerationResponse:
    return await service.generate(auth, body)


@router.get(
    "/media/{media_id}",
    summary="Retrieve private generated media",
    responses={404: {"description": "Media does not exist or belongs to another user"}},
)
async def retrieve_media(
    media_id: uuid.UUID,
    auth: AuthContext = Depends(get_current_auth_context),
    service: MediaService = Depends(get_media_service),
) -> Response:
    content = await service.get_for_user(auth, media_id)
    return Response(
        content=content.data,
        media_type=content.mime_type,
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f'inline; filename="{content.filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )
