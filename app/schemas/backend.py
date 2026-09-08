from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BackendContextRequest(BaseModel):
    external_conversation_id: str | None = Field(default=None, min_length=1, max_length=200)
    idempotency_key: uuid.UUID


class BackendChatRequest(BackendContextRequest):
    message: str = Field(min_length=1, max_length=4000)


class BackendImageRequest(BackendContextRequest):
    prompt: str = Field(min_length=1, max_length=2000)
    trigger: Literal["user_requested", "contextual"] = "user_requested"


class BackendVoiceRequest(BackendChatRequest):
    pass


class BackendProactiveRequest(BackendContextRequest):
    reason: str = Field(min_length=1, max_length=2000)
    response_mode: Literal["text", "voice"] = "text"


class BackendStoryEventRequest(BaseModel):
    external_event_id: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)
    source: str = Field(default="backend", min_length=1, max_length=50, pattern=r"^[a-z0-9_-]+$")
    happened_at: datetime
    metadata: dict[str, str | int | float | bool | None] | None = None

    @field_validator("happened_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("happened_at must include a timezone offset")
        return value


class BackendStoryEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    companion_id: uuid.UUID
    external_event_id: str
    summary: str
    source: str
    happened_at: datetime
    metadata: dict[str, str | int | float | bool | None] | None = Field(
        validation_alias="event_metadata"
    )
    created_at: datetime
    updated_at: datetime
