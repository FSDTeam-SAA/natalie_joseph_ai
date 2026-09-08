from __future__ import annotations

from app.schemas.chat import ChatResponse


class VoiceChatResponse(ChatResponse):
    """Chat response with the accepted STT transcript and optional audio media."""

