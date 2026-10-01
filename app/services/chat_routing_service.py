"""Natural-language routing for text chat and companion image requests.

This layer deliberately contains no generation, persistence, or
entitlement business logic.  It only recognizes conservative, explicit image
requests and delegates them to :class:`ImageService`; every other message goes
through the existing :class:`ChatService` pipeline.
"""

from __future__ import annotations

import re

from fastapi import BackgroundTasks

from app.core.exceptions import ValidationError
from app.core.security import AuthContext, require_feature
from app.db.models.message import MessageType
from app.repositories.message_repository import MessageRepository
from app.schemas.chat import ChatRequest, ChatResponse, ChatUsage
from app.schemas.media import ImageGenerationRequest, ImageGenerationResponse
from app.services.chat_service import ChatService
from app.services.image_service import ImageService


class ImageRequestClassifier:
    """Conservatively recognize an explicit request for a companion image.

    A false negative merely leaves the request in normal conversation.  A
    false positive can spend image credits, so this classifier intentionally
    requires request language rather than routing every mention of a photo.
    Image safety decisions remain in ``ImageService``.
    """

    _IMAGE_NOUN = (
        r"(?:photos?|photographs?|pictures?|images?|selfies?|pics?|snapshots?|"
        r"portraits?|headshots?|nudes?)"
    )
    _ACTION = r"(?:send|share|show|generate|create|make|take|give|draw)"
    _CLAUSE = r"[^.!?]{0,120}"

    _REQUEST_PATTERNS = (
        re.compile(
            rf"(?:^|[.!?]\s+)(?:hey\s+\w+[,.]?\s+)?(?:please\s+)?"
            rf"{_ACTION}\b{_CLAUSE}\b{_IMAGE_NOUN}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:can|could|would|will)\s+you\s+{_ACTION}\b"
            rf"{_CLAUSE}\b{_IMAGE_NOUN}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:can|could|may)\s+i\s+(?:see|have|get)\b"
            rf"{_CLAUSE}\b{_IMAGE_NOUN}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:i\s+(?:want|would\s+like|would\s+love)|"
            rf"i['\u2019]d\s+(?:like|love))\b{_CLAUSE}\b{_IMAGE_NOUN}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?:^|[.!?]\s+)(?:a|an|another|one|some|your)?\s*"
            rf"{_IMAGE_NOUN}\b{_CLAUSE}(?:\bplease\b|pls\b|[!?]\s*$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:show\s+me|let\s+me\s+see)\s+(?:you|yourself)"
            r"(?:\s+(?:at|in|on|wearing|with|as|naked|nude)\b[^.!?]*)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:show\s+me|let\s+me\s+see)\s+(?:what\s+you(?:['\u2019]re|\s+are)"
            r"\s+wearing|your\s+(?:outfit|look)|how\s+you\s+look)\b",
            re.IGNORECASE,
        ),
        # A common concise form omits "photo" but clearly asks to see the
        # companion in a particular outfit, e.g. "give me a saree wearing
        # look". Require both a direct request and the visual "wearing look"
        # phrase so ordinary discussion of fashion remains chat.
        re.compile(
            r"\b(?:give|show|send)\s+me\b[^.!?]{0,100}"
            r"\bwearing\s+(?:a\s+|an\s+|the\s+)?[\w-]+\s+look\b"
            r"|\b(?:give|show|send)\s+me\b[^.!?]{0,100}"
            r"\b[\w-]+\s+wearing\s+look\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bwhat\s+(?:do|would)\s+you\s+look\s+like\b", re.IGNORECASE),
        # Common Bangla/Banglish request forms put the action after the
        # image noun, unlike English. Keep this explicit and conservative.
        re.compile(
            r"(?:ছবি|ফটো|সেলফি)[^.!?।]{0,100}"
            r"(?:পাঠাও|পাঠিয়ে\s+দাও|দেখাও|বানাও|তৈরি\s+করো)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:chh?obi|photo|selfie|pic)\b[^.!?]{0,100}"
            r"\b(?:pathao|pathiye\s+dao|dekhao|banao|toiri\s+koro)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:কি\s+পরেছ(?:ো)?|আজ\s+কি\s+পরেছ(?:ো)?)[^.!?।]{0,80}দেখাও",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:ajke?\s+)?ki\s+porech(?:o|ho)\b[^.!?]{0,80}\bdekhao\b",
            re.IGNORECASE,
        ),
    )
    _NEGATED_REQUEST = re.compile(
        rf"\b(?:do\s+not|don['\u2019]?t|never|no\s+need\s+to|stop)\b"
        rf"{_CLAUSE}(?:\b{_ACTION}\b|\b{_IMAGE_NOUN}\b)",
        re.IGNORECASE,
    )
    _CAPABILITY_DISCUSSION = re.compile(
        rf"(?:\b(?:are\s+you\s+able\s+to|do\s+you\s+support|"
        rf"(?:tell|show)\s+me\s+(?:how|whether)|explain\s+how|"
        rf"(?:talk|chat|discuss|ask|learn|know)\b)\b{_CLAUSE}"
        rf"\b{_IMAGE_NOUN}\b|"
        rf"\b(?:how|why)\s+(?:do|would|can)\s+(?:i|you)\b{_CLAUSE}"
        rf"\b{_IMAGE_NOUN}\b)",
        re.IGNORECASE,
    )
    _GENERIC_CAPABILITY_QUESTION = re.compile(
        rf"^\s*(?:can|could)\s+you\s+{_ACTION}\s+"
        rf"(?:any\s+)?(?:photos|photographs|pictures|images|selfies|pics)\s*\?\s*$",
        re.IGNORECASE,
    )

    def is_image_request(self, text: str) -> bool:
        normalized = " ".join(text.strip().split())
        if not normalized:
            return False
        if self._NEGATED_REQUEST.search(normalized):
            return False
        if self._CAPABILITY_DISCUSSION.search(normalized):
            return False
        if self._GENERIC_CAPABILITY_QUESTION.fullmatch(normalized):
            return False
        return any(pattern.search(normalized) for pattern in self._REQUEST_PATTERNS)


class VoiceResponseClassifier:
    """Recognize explicit requests for a spoken companion reply."""

    _PATTERN = re.compile(
        r"\b(?:send|share|record|give|reply with)\b[^.!?]{0,80}"
        r"\b(?:a )?(?:voice note|voice message|audio message|audio|voice)\b"
        r"|\b(?:can|could|may)\s+i\s+hear\s+your\s+voice\b",
        re.IGNORECASE,
    )

    def is_voice_request(self, text: str) -> bool:
        return bool(self._PATTERN.search(" ".join(text.strip().split())))


class ChatRoutingService:
    """Route one ordinary chat request through the correct existing service."""

    def __init__(
        self,
        *,
        chat_service: ChatService,
        image_service: ImageService,
        message_repo: MessageRepository,
        image_classifier: ImageRequestClassifier | None = None,
    ) -> None:
        self.chat_service = chat_service
        self.image_service = image_service
        self.message_repo = message_repo
        self.image_classifier = image_classifier or ImageRequestClassifier()

    @staticmethod
    def _as_chat_response(image: ImageGenerationResponse) -> ChatResponse:
        return ChatResponse(
            message_id=image.message_id,
            conversation_id=image.conversation_id,
            companion_id=image.companion_id,
            # Image clients render media.url directly. An empty response keeps
            # the generated-image caption out of chat surfaces such as WhatsApp
            # while retaining the complete media descriptor.
            response="",
            created_at=image.created_at,
            usage=ChatUsage(input_tokens=0, output_tokens=0),
            message_type=MessageType.image.value,
            media=image.media,
        )

    async def send_message(
        self,
        auth: AuthContext,
        request: ChatRequest,
        background_tasks: BackgroundTasks,
        *,
        user_message_type: MessageType = MessageType.text,
    ) -> ChatResponse:
        # Without this check, routing an image before ChatService would let a
        # caller use the ordinary chat endpoint without a chat grant.
        require_feature(auth, "chat")

        is_image_request = self.image_classifier.is_image_request(request.message)
        # ChatService and ImageService have their own idempotency domains. This
        # shared route lock and modality preflight prevents one /chat key from
        # being reused to buy both an image and a text generation.
        await self.message_repo.acquire_idempotency_lock(
            scope="chat_route", request_id=request.idempotency_key
        )
        existing_user_message = await self.message_repo.get_user_by_request_id_for_external_user(
            conversation_id=request.conversation_id,
            request_id=request.idempotency_key,
            external_user_id=auth.user_id,
        )
        if existing_user_message is not None:
            if existing_user_message.content != request.message:
                raise ValidationError(
                    "The idempotency_key was already used with different message content."
                )
            existing_assistant = await self.message_repo.get_first_assistant_after(
                conversation_id=request.conversation_id,
                sequence=existing_user_message.sequence,
            )
            if existing_assistant is None:
                raise ValidationError(
                    "The previous request with this idempotency key is incomplete."
                )
            previous_was_image = existing_assistant.message_type == MessageType.image
            if previous_was_image != is_image_request:
                raise ValidationError(
                    "The idempotency_key was already used for a different response modality."
                )

        if is_image_request:
            image = await self.image_service.generate(
                auth,
                ImageGenerationRequest(
                    conversation_id=request.conversation_id,
                    companion_id=request.companion_id,
                    prompt=request.message,
                    trigger="user_requested",
                    idempotency_key=request.idempotency_key,
                    user_message_type=user_message_type.value,
                ),
            )
            return self._as_chat_response(image)

        if user_message_type is MessageType.text:
            return await self.chat_service.send_message(auth, request, background_tasks)
        return await self.chat_service.send_message(
            auth, request, background_tasks, user_message_type=user_message_type
        )
