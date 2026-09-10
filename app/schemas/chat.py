"""
Chat API schemas.

Deviation from the literal request shape shown in the original spec
document (Section 18), noted deliberately rather than silently:
the spec's example request body includes "user_id" directly. This
implementation deliberately omits user_id from the request body —
user identity comes from the authenticated AuthContext
(get_current_auth_context), never from client-supplied data. Trusting
a body-supplied user_id would let any caller impersonate any other
user, which directly contradicts spec Section 3 ("deciding which
user's memories can be retrieved" is explicitly listed as something
the backend, not client input, must control) and Section 57 (never
allow unrestricted / client-controlled access to another user's data).

The response shape matches spec Section 21 exactly: message_id,
conversation_id, companion_id, response, created_at, usage. No system
prompts, chain-of-thought, or provider secrets.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.media import MediaDescriptor


class ChatRequest(BaseModel):
    conversation_id: uuid.UUID
    companion_id: uuid.UUID
    message: str = Field(min_length=1, max_length=4000)
    idempotency_key: uuid.UUID


class ChatUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    message_id: uuid.UUID
    conversation_id: uuid.UUID
    companion_id: uuid.UUID
    response: str
    created_at: datetime
    usage: ChatUsage
    message_type: str = "text"
    media: MediaDescriptor | None = None
    transcript: str | None = None
