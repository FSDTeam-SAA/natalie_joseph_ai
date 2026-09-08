"""
PromptContext — the single data bundle threaded through PromptBuilder.

Per spec Section 34, the modular prompt system assembles context from
several sources in a fixed order. This dataclass exists so that order
is enforced in one place (PromptBuilder + sections.py) rather than
scattered across call sites. Long-term memory retrieval, rolling
conversation summaries, and companion story events all populate this
same bundle without changing PromptBuilder's public shape.

Optional context fields default to None / empty so specialized flows can
reuse the same builder without manufacturing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.security import AuthContext
from app.db.models.relationship_context import RelationshipContext
from app.db.models.user import User
from app.services.companion_service import CompanionProfile


@dataclass(frozen=True)
class PromptContext:
    companion: CompanionProfile
    auth: AuthContext
    user: User

    # Fetched via RelationshipRepository
    # .get_or_create() by ChatService. None only if a caller explicitly
    # omits it (e.g. a future non-conversational use of PromptBuilder).
    relationship_context: RelationshipContext | None = None

    retrieved_memories: list[str] = field(default_factory=list)
    conversation_summary: str | None = None
    story_events: list[str] = field(default_factory=list)
