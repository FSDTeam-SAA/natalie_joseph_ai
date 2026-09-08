"""
Conversation API schemas.

Per spec Section 8: message roles include 'system', but system
messages must never be exposed to the client. MessageResponse is used
only for user/assistant messages — the service layer filters system
messages out before they ever reach this schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConversationCreateRequest(BaseModel):
    companion_id: uuid.UUID
    external_conversation_id: uuid.UUID | None = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    companion_id: uuid.UUID
    summary: str | None = None
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    message_type: str = "text"
    media_id: uuid.UUID | None = None
    media_url: str | None = None
    created_at: datetime
