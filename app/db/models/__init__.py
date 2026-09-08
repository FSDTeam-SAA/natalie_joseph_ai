"""
Import every model here so `Base.metadata` is fully populated before
Alembic autogenerate runs, and so `from app.db.models import User` etc.
works cleanly elsewhere in the app.
"""

from app.db.models.ai_event import AIEvent
from app.db.models.companion import Companion
from app.db.models.conversation import Conversation
from app.db.models.media_asset import MediaAsset, MediaKind
from app.db.models.memory import Memory, MemoryType
from app.db.models.message import Message, MessageRole, MessageType
from app.db.models.relationship_context import (
    ConversationDepth,
    FamiliarityLevel,
    RelationshipContext,
)
from app.db.models.safety_event import SafetyDirection, SafetyEvent
from app.db.models.story_event import StoryEvent
from app.db.models.user import User

__all__ = [
    "AIEvent",
    "Companion",
    "Conversation",
    "ConversationDepth",
    "FamiliarityLevel",
    "Memory",
    "MemoryType",
    "Message",
    "MessageRole",
    "MessageType",
    "MediaAsset",
    "MediaKind",
    "RelationshipContext",
    "SafetyDirection",
    "SafetyEvent",
    "StoryEvent",
    "User",
]
