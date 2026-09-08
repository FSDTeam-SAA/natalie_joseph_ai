"""Authenticated text-chat endpoint."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.deps import enforce_ai_rate_limit, get_chat_routing_service
from app.core.security import AuthContext, get_current_auth_context
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_routing_service import ChatRoutingService

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a text or natural-language image request",
    description="Moderated Grok chat with persona, history, long-term memory, "
    "relationship progression, rolling summary, and companion-story context. "
    "Explicit requests for a companion photo are routed through the authorized, "
    "moderated image pipeline and returned as an image ChatResponse.",
    responses={
        422: {"description": "Invalid companion/conversation combination"},
        404: {"description": "Conversation not found"},
        400: {"description": "Message blocked by content moderation"},
    },
)
async def send_chat_message(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(get_current_auth_context),
    service: ChatRoutingService = Depends(get_chat_routing_service),
    _rate_limit: None = Depends(enforce_ai_rate_limit),
) -> ChatResponse:
    return await service.send_message(auth, body, background_tasks)
