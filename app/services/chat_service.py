"""
Chat service — Phase 6 scope.

Implements the spec Section 18 flow. As of Phase 6:
  - System prompt assembly now goes through the modular PromptBuilder
    (app/llm/prompts/), replacing the Phase 5 interim single-function
    prompt. Assembly order and section skipping are documented in
    app/llm/prompts/sections.py.
  - relationship_context is now fetched (or created, on first contact)
    per user+companion and fed into the prompt at a basic level (spec
    Section 32) — this is a read, not a write; nothing here updates
    familiarity/depth/tone yet.
  - conversation.summary is threaded into PromptContext, but nothing
    populates it yet — it stays None until the Phase 8 background
    summarization job exists.

Still explicitly later phases:
  - NO long-term memory retrieval (Phase 7) — PromptContext accepts
    retrieved_memories, but nothing supplies it yet.
  - NO conversation summary generation (Phase 8) — see above.
  - NO safety policy engine / injection defenses / dependency
    safeguard enforcement beyond basic OpenAI moderation (Phase 9).

What IS implemented, deliberately, rather than deferred:
  - Real input and output moderation via the OpenAIModerationProvider
    built in Phase 4 — shipping a companion chat endpoint with zero
    moderation, even temporarily, was judged not acceptable.
  - Basic safety_events logging when moderation blocks something,
    since the table already exists (Phase 2) and the data is already
    on hand at the point moderation runs.
  - Basic ai_events logging for token/latency observability, for the
    same reason — full cost-dollar tracking (CostTracker) remains a
    later phase, but the raw event row is cheap to write now.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from app.core.config import Settings
from app.core.exceptions import ModerationBlockedError, NotFoundError, ValidationError
from app.core.security import AuthContext
from app.db.models.ai_event import AIEvent
from app.db.models.message import Message, MessageRole
from app.db.models.safety_event import SafetyDirection, SafetyEvent
from app.llm.base import LLMMessage, LLMProvider
from app.llm.prompts.builder import PromptBuilder
from app.llm.prompts.context import PromptContext
from app.moderation.base import ModerationProvider
from app.repositories.companion_repository import CompanionRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.relationship_repository import RelationshipRepository
from app.repositories.user_repository import UserRepository
from app.schemas.chat import ChatRequest, ChatResponse, ChatUsage

# Fallback text shown to the user when the model's own output is
# flagged by moderation — the flagged text itself is never returned
# or persisted as the assistant's message.
OUTPUT_MODERATION_FALLBACK = (
    "I want to be thoughtful about how I respond to that — could we talk about "
    "something else, or rephrase what you're looking for?"
)


class ChatService:
    def __init__(
        self,
        *,
        user_repo: UserRepository,
        companion_repo: CompanionRepository,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        relationship_repo: RelationshipRepository,
        llm_provider: LLMProvider,
        moderation_provider: ModerationProvider,
        prompt_builder: PromptBuilder,
        settings: Settings,
    ) -> None:
        self.user_repo = user_repo
        self.companion_repo = companion_repo
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.relationship_repo = relationship_repo
        self.llm_provider = llm_provider
        self.moderation_provider = moderation_provider
        self.prompt_builder = prompt_builder
        self.settings = settings

    async def _log_safety_event(
        self,
        *,
        request_id: uuid.UUID,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        direction: SafetyDirection,
        moderation_result,
    ) -> None:
        flagged_categories = [
            cat for cat, flagged in moderation_result.categories.items() if flagged
        ]
        event = SafetyEvent(
            request_id=request_id,
            user_id=user_id,
            conversation_id=conversation_id,
            direction=direction,
            category=", ".join(flagged_categories) if flagged_categories else None,
            severity=None,  # TODO-CONFIRM scale in Phase 9
            action="blocked",
        )
        self.message_repo.session.add(event)
        await self.message_repo.session.flush()

    async def send_message(self, auth: AuthContext, request: ChatRequest) -> ChatResponse:
        request_id = uuid.uuid4()

        user = await self.user_repo.get_or_create_by_external_user_id(auth.user_id)

        companion = await self.companion_repo.get_by_id(request.companion_id)
        if companion is None or not companion.active:
            raise ValidationError(f"Companion {request.companion_id} does not exist or is inactive.")

        conversation = await self.conversation_repo.get_by_id_for_user(
            request.conversation_id, user.id
        )
        if conversation is None:
            raise NotFoundError(f"Conversation {request.conversation_id} not found.")
        if conversation.companion_id != companion.id:
            raise ValidationError(
                "The given companion_id does not match this conversation's companion."
            )

        # --- Input moderation (spec Section 26) ---
        input_moderation = await self.moderation_provider.moderate_text(
            request.message, model=self.settings.OPENAI_MODERATION_MODEL
        )
        if input_moderation.flagged:
            await self._log_safety_event(
                request_id=request_id,
                user_id=user.id,
                conversation_id=conversation.id,
                direction=SafetyDirection.input,
                moderation_result=input_moderation,
            )
            await self.message_repo.session.commit()
            raise ModerationBlockedError(
                "Your message couldn't be processed. Please rephrase and try again."
            )

        # --- Basic relationship_context wiring (spec Section 32) ---
        # Fetch-or-create is idempotent; the row is flushed into this
        # same session/transaction and committed with everything else
        # below, so a blocked (moderation-flagged) message never
        # creates one — this only runs on the path that will actually
        # generate a reply.
        relationship_context = await self.relationship_repo.get_or_create(
            user.id, companion.id
        )

        # --- Build system prompt (PromptBuilder, Phase 6) + recent history ---
        prompt_context = PromptContext(
            companion=companion,
            auth=auth,
            user=user,
            relationship_context=relationship_context,
            conversation_summary=conversation.summary,
        )
        system_prompt = self.prompt_builder.build_system_prompt(prompt_context)
        recent_messages = await self.message_repo.get_recent_for_conversation(
            conversation.id, limit=self.settings.RECENT_MESSAGE_LIMIT
        )

        llm_messages = [LLMMessage(role="system", content=system_prompt)]
        for m in recent_messages:
            if m.role == MessageRole.system:
                continue
            llm_messages.append(LLMMessage(role=m.role.value, content=m.content))
        llm_messages.append(LLMMessage(role="user", content=request.message))

        # --- Generate ---
        start = time.perf_counter()
        result = await self.llm_provider.generate(
            llm_messages, model=self.settings.OPENAI_CHAT_MODEL
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # --- Output moderation (spec Section 26) ---
        output_moderation = await self.moderation_provider.moderate_text(
            result.text, model=self.settings.OPENAI_MODERATION_MODEL
        )
        final_text = result.text
        if output_moderation.flagged:
            await self._log_safety_event(
                request_id=request_id,
                user_id=user.id,
                conversation_id=conversation.id,
                direction=SafetyDirection.output,
                moderation_result=output_moderation,
            )
            final_text = OUTPUT_MODERATION_FALLBACK

        # --- Persist user + assistant messages ---
        user_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.user,
            content=request.message,
        )
        self.message_repo.session.add(user_message)

        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.assistant,
            content=final_text,
            model=result.model,
            prompt_version=self.settings.PROMPT_VERSION,
            companion_version=companion.version,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            latency_ms=latency_ms,
        )
        self.message_repo.session.add(assistant_message)

        # --- ai_events observability row (spec Section 39) ---
        ai_event = AIEvent(
            request_id=request_id,
            user_id=user.id,
            companion_id=companion.id,
            conversation_id=conversation.id,
            event_type="chat_completion",
            model=result.model,
            prompt_version=self.settings.PROMPT_VERSION,
            companion_version=companion.version,
            latency_ms=latency_ms,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )
        self.message_repo.session.add(ai_event)

        await self.message_repo.session.commit()
        await self.message_repo.session.refresh(assistant_message)

        return ChatResponse(
            message_id=assistant_message.id,
            conversation_id=conversation.id,
            companion_id=companion.id,
            response=final_text,
            created_at=assistant_message.created_at,
            usage=ChatUsage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            ),
        )