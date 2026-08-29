"""
Conversation endpoints (spec Section 22).

All endpoints require authentication; ownership is enforced by the
service/repository layer, never assumed from the request path alone.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status

from app.api.deps import get_conversation_service
from app.core.security import AuthContext, get_current_auth_context
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationResponse,
    MessageResponse,
)
from app.services.conversation_service import ConversationService

router = APIRouter(tags=["Conversations"])


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
async def create_conversation(
    body: ConversationCreateRequest,
    auth: AuthContext = Depends(get_current_auth_context),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    return await service.create_conversation(
        auth,
        companion_id=body.companion_id,
        external_conversation_id=body.external_conversation_id,
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get a conversation",
    responses={404: {"description": "Conversation not found"}},
)
async def get_conversation(
    conversation_id: uuid.UUID,
    auth: AuthContext = Depends(get_current_auth_context),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    return await service.get_conversation(auth, conversation_id)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
    summary="List messages in a conversation",
    description="Returns the most recent messages, oldest first. System "
    "messages are never included in the response.",
    responses={404: {"description": "Conversation not found"}},
)
async def list_messages(
    conversation_id: uuid.UUID,
    auth: AuthContext = Depends(get_current_auth_context),
    service: ConversationService = Depends(get_conversation_service),
) -> list[MessageResponse]:
    return await service.list_messages(auth, conversation_id)


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete a conversation",
    responses={404: {"description": "Conversation not found"}},
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    auth: AuthContext = Depends(get_current_auth_context),
    service: ConversationService = Depends(get_conversation_service),
) -> None:
    await service.delete_conversation(auth, conversation_id)
    return None