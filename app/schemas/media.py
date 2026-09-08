from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MediaDescriptor(BaseModel):
    id: uuid.UUID
    kind: Literal["audio", "image"]
    url: str
    mime_type: str
    byte_size: int


class ImageGenerationRequest(BaseModel):
    conversation_id: uuid.UUID
    companion_id: uuid.UUID
    prompt: str = Field(min_length=1, max_length=2000)
    trigger: Literal["user_requested", "contextual"] = "user_requested"
    idempotency_key: uuid.UUID


class ImageGenerationResponse(BaseModel):
    message_id: uuid.UUID
    conversation_id: uuid.UUID
    companion_id: uuid.UUID
    caption: str
    media: MediaDescriptor
    created_at: datetime
    provider: str
    model: str
