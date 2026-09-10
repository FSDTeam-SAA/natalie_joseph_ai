"""Grok chat with persona, memory, story, and relationship continuity.

Only the trusted backend decides age and feature eligibility. User facts are
retrieved synchronously for the current response; extraction, rolling summaries,
and relationship progression run after the response is committed.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from fastapi import BackgroundTasks

from app.core.config import Settings
from app.core.exceptions import NotFoundError, ValidationError
from app.core.security import AuthContext, require_feature
from app.db.models.ai_event import AIEvent
from app.db.models.message import Message, MessageRole, MessageType
from app.llm.base import LLMMessage, LLMProvider
from app.llm.prompts.builder import PromptBuilder
from app.llm.prompts.context import PromptContext
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.relationship_repository import RelationshipRepository
from app.repositories.story_event_repository import StoryEventRepository
from app.repositories.user_repository import UserRepository
from app.schemas.chat import ChatRequest, ChatResponse, ChatUsage
from app.services.companion_service import CompanionService
from app.services.memory_service import MemoryService
from app.services.post_turn_processor import PostTurnProcessor


class ChatService:
    def __init__(
        self,
        *,
        user_repo: UserRepository,
        companion_service: CompanionService,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        relationship_repo: RelationshipRepository,
        llm_provider: LLMProvider,
        prompt_builder: PromptBuilder,
        memory_service: MemoryService,
        settings: Settings,
        post_turn_processor: PostTurnProcessor | None = None,
        story_repo: StoryEventRepository | None = None,
    ) -> None:
        self.user_repo = user_repo
        self.companion_service = companion_service
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.relationship_repo = relationship_repo
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder
        self.memory_service = memory_service
        self.settings = settings
        self.post_turn_processor = post_turn_processor
        self.story_repo = story_repo

    async def send_message(
        self,
        auth: AuthContext,
        request: ChatRequest,
        background_tasks: BackgroundTasks,
        *,
        user_message_type: MessageType = MessageType.text,
    ) -> ChatResponse:
        require_feature(auth, "chat")
        request_id = uuid.uuid4()

        user = await self.user_repo.get_or_create_by_external_user_id(auth.user_id)

        try:
            companion = await self.companion_service.get_active_profile(request.companion_id)
        except NotFoundError as exc:
            raise ValidationError(
                f"Companion {request.companion_id} does not exist or is inactive."
            ) from exc

        conversation = await self.conversation_repo.get_by_id_for_user(
            request.conversation_id, user.id
        )
        if conversation is None:
            raise NotFoundError(f"Conversation {request.conversation_id} not found.")
        if conversation.companion_id != companion.id:
            raise ValidationError(
                "The given companion_id does not match this conversation's companion."
            )

        if request.idempotency_key is not None:
            await self.message_repo.acquire_idempotency_lock(
                scope="chat", request_id=request.idempotency_key
            )
            existing_user_message = await self.message_repo.get_user_by_request_id(
                conversation_id=conversation.id,
                request_id=request.idempotency_key,
            )
            if existing_user_message is not None:
                if existing_user_message.content != request.message:
                    raise ValidationError(
                        "The idempotency_key was already used with different message content."
                    )
                existing_assistant = await self.message_repo.get_first_assistant_after(
                    conversation_id=conversation.id,
                    sequence=existing_user_message.sequence,
                )
                if existing_assistant is None:
                    raise ValidationError(
                        "The previous request with this idempotency key is incomplete."
                    )
                response = ChatResponse(
                    message_id=existing_assistant.id,
                    conversation_id=conversation.id,
                    companion_id=companion.id,
                    response=existing_assistant.content,
                    created_at=existing_assistant.created_at,
                    usage=ChatUsage(
                        input_tokens=existing_assistant.input_tokens or 0,
                        output_tokens=existing_assistant.output_tokens or 0,
                    ),
                    message_type=existing_assistant.message_type.value,
                    transcript=(
                        existing_user_message.content
                        if existing_user_message.message_type == MessageType.audio
                        else None
                    ),
                )
                await self.message_repo.session.commit()
                return response

        # --- Basic relationship_context wiring (spec Section 32) ---
        # Fetch-or-create is idempotent; the row is flushed into this
        # same session/transaction and committed with everything else
        # below, so it runs only on the path that will generate a reply.
        relationship_context = await self.relationship_repo.get_or_create(
            user.id, companion.id
        )

        # --- Long-term memory retrieval (Phase 7) ---
        # Synchronous and on the critical path deliberately — unlike
        # extraction below, retrieved memories directly shape this
        # turn's prompt, so they can't be deferred to a background
        # task. Degrades to an empty list on any failure (embedding
        # API error, DB error) rather than breaking the chat request —
        # see MemoryService.retrieve_relevant()'s docstring.
        retrieved_memories = await self.memory_service.retrieve_relevant(
            user_id=user.id, companion_id=companion.id, query_text=request.message
        )
        story_events = (
            await self.story_repo.list_recent(companion.id)
            if self.story_repo is not None
            else []
        )

        # --- Build system prompt (PromptBuilder, Phase 6) + recent history ---
        prompt_context = PromptContext(
            companion=companion,
            auth=auth,
            user=user,
            relationship_context=relationship_context,
            retrieved_memories=retrieved_memories,
            conversation_summary=conversation.summary,
            story_events=[event.prompt_fact for event in story_events],
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
            llm_messages,
            model=self.settings.XAI_MODEL,
            max_output_tokens=self.settings.CHAT_MAX_OUTPUT_TOKENS,
            cache_key=f"conversation:{conversation.id}",
            reasoning_effort=self.settings.XAI_REASONING_EFFORT,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        final_text = result.text

        # --- Persist user + assistant messages ---
        user_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.user,
            content=request.message,
            message_type=user_message_type,
            client_request_id=request.idempotency_key,
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
            event_metadata=result.usage.as_metadata(),
        )
        self.message_repo.session.add(ai_event)

        # Keep "latest conversation" selection accurate for the trusted-backend adapter.
        conversation.updated_at = datetime.now(UTC)

        await self.message_repo.session.commit()
        await self.message_repo.session.refresh(assistant_message)

        # Post-response work gets fresh DB sessions inside the background task.
        # Uses final_text (what the user actually saw), so a moderated fallback
        # never becomes a source of apparent facts about the user.
        if self.post_turn_processor is not None:
            background_tasks.add_task(
                self.post_turn_processor.process,
                conversation_id=conversation.id,
                user_id=user.id,
                companion_id=companion.id,
                user_message=request.message,
                assistant_message=final_text,
                source_message_id=user_message.id,
            )

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
            message_type=MessageType.text.value,
            transcript=request.message if user_message_type == MessageType.audio else None,
        )
