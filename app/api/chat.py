"""
Chat endpoint (spec Section 20 — non-streaming only; streaming is
Phase 10).
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.deps import get_chat_service
from app.core.security import AuthContext, get_current_auth_context
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message (non-streaming)",
    description="Phase 7 scope: moderated chat with long-term memory "
    "retrieval and basic relationship context. Conversation summary "
    "injection and full policy-engine safety controls are added in "
    "later phases.",
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
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await service.send_message(auth, body, background_tasks)