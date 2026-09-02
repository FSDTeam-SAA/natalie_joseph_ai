"""
PromptContext — the single data bundle threaded through PromptBuilder.

Per spec Section 34, the modular prompt system assembles context from
several sources in a fixed order. This dataclass exists so that order
is enforced in one place (PromptBuilder + sections.py) rather than
scattered across call sites, and so later phases (long-term memory
retrieval — Phase 7, conversation summarization — Phase 8) can
populate their fields here without changing PromptBuilder's public
shape or ChatService's call site again.

Fields for not-yet-implemented phases are typed and default to
None / empty so they are always safe to leave unset:
  - retrieved_memories: populated by the memory engine (Phase 7).
  - conversation_summary: populated by the background summarization
    job (Phase 8) via `conversations.summary` — that column already
    exists (Phase 2, currently always NULL) and is threaded through
    here now so summarization can start writing to it later without
    any PromptBuilder/ChatService changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.security import AuthContext
from app.db.models.companion import Companion
from app.db.models.relationship_context import RelationshipContext
from app.db.models.user import User


@dataclass(frozen=True)
class PromptContext:
    companion: Companion
    auth: AuthContext
    user: User

    # Basic-level wiring, Phase 6: fetched via RelationshipRepository
    # .get_or_create() by ChatService. None only if a caller explicitly
    # omits it (e.g. a future non-conversational use of PromptBuilder).
    relationship_context: RelationshipContext | None = None

    # --- Extension points for later phases. Present now so those
    # phases only need to populate a field, not touch PromptBuilder's
    # signature or the section assembly order. ---
    retrieved_memories: list[str] = field(default_factory=list)  # Phase 7
    conversation_summary: str | None = None  # Phase 8
