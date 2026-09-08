from __future__ import annotations

import hashlib
import logging
import time
import uuid

from app.core.config import Settings
from app.core.exceptions import NotFoundError, ProviderError, ValidationError
from app.core.security import AuthContext, require_feature
from app.db.models.ai_event import AIEvent
from app.db.models.conversation import Conversation
from app.db.models.media_asset import MediaAsset, MediaKind
from app.db.models.message import MessageRole, MessageType
from app.db.models.user import User
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.media_repository import MediaRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.user_repository import UserRepository
from app.schemas.chat import ChatRequest, ChatResponse, ChatUsage
from app.schemas.media import MediaDescriptor
from app.schemas.voice import VoiceChatResponse
from app.services.chat_service import ChatService
from app.services.companion_service import CompanionProfile, CompanionService
from app.storage.base import MediaStorage, StoredObject
from app.voice.audio_inspection import AudioInspectionError, inspect_audio_duration
from app.voice.base import VoiceProvider

logger = logging.getLogger(__name__)

_SUPPORTED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp4",
    "audio/m4a",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
}


def _validate_audio_signature(data: bytes, mime_type: str) -> None:
    canonical = "audio/wav" if mime_type == "audio/x-wav" else mime_type
    valid = False
    if canonical == "audio/wav":
        valid = len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    elif canonical == "audio/mpeg":
        valid = data.startswith(b"ID3") or (
            len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
        )
    elif canonical in {"audio/mp4", "audio/m4a"}:
        valid = len(data) >= 12 and data[4:8] == b"ftyp"
    elif canonical == "audio/ogg":
        valid = data.startswith(b"OggS")
    elif canonical == "audio/webm":
        valid = data.startswith(b"\x1a\x45\xdf\xa3")
    if not valid:
        raise ValidationError("The uploaded file content does not match its audio type.")


def _validate_synthesized_audio(data: bytes, mime_type: str) -> None:
    if not data:
        raise ProviderError("The voice provider returned empty audio data.")
    if mime_type in _SUPPORTED_AUDIO_TYPES:
        try:
            _validate_audio_signature(data, mime_type)
        except ValidationError as exc:
            raise ProviderError("The voice provider returned invalid audio data.") from exc
        return
    # Headerless PCM/A-law/µ-law payloads have no magic signature to inspect.
    if mime_type not in {"audio/basic", "audio/l16"}:
        raise ProviderError("The voice provider returned an unsupported audio type.")


class VoiceService:
    def __init__(
        self,
        *,
        user_repo: UserRepository,
        companion_service: CompanionService,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        media_repo: MediaRepository,
        chat_service: ChatService | None,
        voice_provider: VoiceProvider,
        storage: MediaStorage,
        settings: Settings,
    ) -> None:
        self.user_repo = user_repo
        self.companion_service = companion_service
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.media_repo = media_repo
        self.chat_service = chat_service
        self.voice_provider = voice_provider
        self.storage = storage
        self.settings = settings

    def _chat_service(self) -> ChatService:
        if self.chat_service is None:
            raise ProviderError("The chat pipeline is unavailable for this voice operation.")
        return self.chat_service

    async def _finish_attempt(self, attempt: AIEvent, status: str, **metadata: object) -> None:
        attempt.event_metadata = {
            **(attempt.event_metadata or {}),
            "status": status,
            **metadata,
        }
        self.message_repo.session.add(attempt)
        await self.message_repo.session.commit()

    async def _finish_attempt_without_masking_error(
        self, attempt: AIEvent, status: str, **metadata: object
    ) -> None:
        """Best-effort terminal audit while preserving the original provider/storage error."""
        try:
            await self._finish_attempt(attempt, status, **metadata)
        except Exception:
            await self.message_repo.session.rollback()
            logger.exception("voice_attempt_audit_failed", extra={"status": status})

    def _media_descriptor(self, asset: MediaAsset) -> MediaDescriptor:
        return MediaDescriptor(
            id=asset.id,
            kind=asset.kind.value,
            url=f"{self.settings.MEDIA_URL_PREFIX.rstrip('/')}/{asset.id}",
            mime_type=asset.mime_type,
            byte_size=asset.byte_size,
        )

    async def _load_context(
        self,
        auth: AuthContext,
        *,
        conversation_id: uuid.UUID,
        companion_id: uuid.UUID,
    ) -> tuple[User, CompanionProfile, Conversation]:
        user = await self.user_repo.get_or_create_by_external_user_id(auth.user_id)
        try:
            companion = await self.companion_service.get_active_profile(companion_id)
        except NotFoundError as exc:
            raise ValidationError("Companion does not exist or is inactive.") from exc
        conversation = await self.conversation_repo.get_by_id_for_user(conversation_id, user.id)
        if conversation is None:
            raise NotFoundError("Conversation not found.")
        if conversation.companion_id != companion.id:
            raise ValidationError("The companion does not match this conversation.")
        return user, companion, conversation

    def _voice_config(self, companion: CompanionProfile) -> tuple[str, dict]:
        if not self.settings.ENABLE_VOICE_GENERATION:
            raise ProviderError("Voice generation is disabled for this deployment.")
        voice_config = companion.voice_config or {}
        voice_id = self.settings.COMPANION_VOICE_ID_MAP.get(companion.slug)
        voice_settings = voice_config.get("settings") or {}
        if not isinstance(voice_id, str) or not voice_id.strip():
            raise ProviderError("This companion has no ElevenLabs voice ID configured.")
        if not isinstance(voice_settings, dict):
            raise ValidationError("The companion voice settings are invalid.")
        return voice_id.strip(), voice_settings

    async def _synthesize_chat_response(
        self,
        *,
        chat: ChatResponse,
        user: User,
        companion: CompanionProfile,
        conversation: Conversation,
        voice_id: str,
        voice_settings: dict,
        idempotency_key: uuid.UUID,
        transcript: str | None,
    ) -> VoiceChatResponse:
        await self.message_repo.acquire_idempotency_lock(
            scope="voice", request_id=idempotency_key
        )
        assistant_message = await self.message_repo.get_by_id_for_conversation(
            message_id=chat.message_id, conversation_id=conversation.id
        )
        if assistant_message is None:
            raise ProviderError("The generated chat message could not be loaded.")

        # Return stored audio on retries so ElevenLabs is never charged twice.
        if assistant_message.media_asset_id is not None:
            existing_asset = await self.media_repo.get_by_id_for_user(
                assistant_message.media_asset_id, user.id
            )
            if existing_asset is not None:
                response = VoiceChatResponse(
                    **chat.model_dump(exclude={"media", "transcript", "message_type"}),
                    message_type=MessageType.audio.value,
                    transcript=transcript,
                    media=self._media_descriptor(existing_asset),
                )
                await self.message_repo.session.commit()
                return response

        if len(chat.response) > self.settings.MAX_TTS_CHARACTERS:
            raise ProviderError("The response is too long for voice generation.")
        input_sha256 = hashlib.sha256(chat.response.encode("utf-8")).hexdigest()
        previous_attempt = await self.message_repo.get_ai_event_by_request_id(
            request_id=idempotency_key,
            user_id=user.id,
            conversation_id=conversation.id,
            event_type="voice_synthesis_attempt",
        )
        if previous_attempt is not None:
            previous_fingerprint = (previous_attempt.event_metadata or {}).get("input_sha256")
            if previous_fingerprint != input_sha256:
                raise ValidationError(
                    "The idempotency key was already used for a different voice response."
                )
            raise ProviderError(
                "The previous voice synthesis attempt did not produce retrievable media; "
                "use a new idempotency key."
            )

        attempt = AIEvent(
            request_id=idempotency_key,
            user_id=user.id,
            companion_id=companion.id,
            conversation_id=conversation.id,
            event_type="voice_synthesis_attempt",
            model=self.settings.ELEVENLABS_TTS_MODEL,
            event_metadata={
                "status": "started",
                "input_sha256": input_sha256,
                "output_format": self.settings.ELEVENLABS_OUTPUT_FORMAT,
            },
        )
        self.message_repo.session.add(attempt)
        await self.message_repo.session.commit()
        # The durable attempt commit releases the transaction-scoped lock. Take
        # it again before the paid call; followers will observe the attempt and
        # return without calling ElevenLabs.
        await self.message_repo.acquire_idempotency_lock(
            scope="voice", request_id=idempotency_key
        )

        synthesis_started = time.perf_counter()
        try:
            synthesis = await self.voice_provider.synthesize(
                chat.response,
                voice_id=voice_id,
                model=self.settings.ELEVENLABS_TTS_MODEL,
                output_format=self.settings.ELEVENLABS_OUTPUT_FORMAT,
                voice_settings=voice_settings,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - synthesis_started) * 1000
            await self._finish_attempt_without_masking_error(
                attempt,
                "provider_failed",
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise
        synthesis_latency_ms = (time.perf_counter() - synthesis_started) * 1000
        try:
            _validate_synthesized_audio(synthesis.audio, synthesis.mime_type)
        except ProviderError:
            await self._finish_attempt_without_masking_error(
                attempt,
                "invalid_output",
                latency_ms=synthesis_latency_ms,
                mime_type=synthesis.mime_type,
                provider_request_id=synthesis.usage.request_id,
            )
            raise
        if len(synthesis.audio) > self.settings.MAX_GENERATED_MEDIA_BYTES:
            await self._finish_attempt_without_masking_error(
                attempt,
                "oversized_output",
                latency_ms=synthesis_latency_ms,
                byte_size=len(synthesis.audio),
                provider_request_id=synthesis.usage.request_id,
            )
            raise ProviderError("The generated audio exceeds the configured size limit.")

        stored: StoredObject | None = None
        try:
            stored = await self.storage.put(synthesis.audio, mime_type=synthesis.mime_type)
            asset = await self.media_repo.add(
                MediaAsset(
                    user_id=user.id,
                    companion_id=companion.id,
                    conversation_id=conversation.id,
                    kind=MediaKind.audio,
                    storage_key=stored.key,
                    mime_type=synthesis.mime_type,
                    byte_size=stored.byte_size,
                    sha256=stored.sha256,
                    provider="elevenlabs",
                    model=synthesis.model,
                    idempotency_key=idempotency_key,
                    asset_metadata={
                        "character_cost": synthesis.usage.character_cost,
                        "provider_request_id": synthesis.usage.request_id,
                    },
                )
            )
            assistant_message.message_type = MessageType.audio
            assistant_message.media_asset_id = asset.id
            attempt.latency_ms = synthesis_latency_ms
            attempt.event_metadata = {
                **(attempt.event_metadata or {}),
                "status": "completed",
                "media_id": str(asset.id),
                "byte_size": stored.byte_size,
                "character_cost": synthesis.usage.character_cost,
                "provider_request_id": synthesis.usage.request_id,
            }
            self.message_repo.session.add(attempt)
            self.message_repo.session.add(
                AIEvent(
                    request_id=idempotency_key,
                    user_id=user.id,
                    companion_id=companion.id,
                    conversation_id=conversation.id,
                    event_type="voice_synthesis",
                    model=synthesis.model,
                    latency_ms=synthesis_latency_ms,
                    event_metadata={
                        "media_id": str(asset.id),
                        "byte_size": stored.byte_size,
                        "character_cost": synthesis.usage.character_cost,
                        "provider_request_id": synthesis.usage.request_id,
                    },
                )
            )
            await self.message_repo.session.commit()
        except Exception as exc:
            await self.message_repo.session.rollback()
            if stored is not None:
                try:
                    await self.storage.delete(stored.key)
                except Exception:
                    logger.exception("voice_media_cleanup_failed", extra={"key": stored.key})
            persisted_attempt = await self.message_repo.get_ai_event_by_request_id(
                request_id=idempotency_key,
                user_id=user.id,
                conversation_id=conversation.id,
                event_type="voice_synthesis_attempt",
            )
            if persisted_attempt is not None:
                await self._finish_attempt_without_masking_error(
                    persisted_attempt,
                    "persistence_failed",
                    latency_ms=synthesis_latency_ms,
                    error_type=type(exc).__name__,
                    provider_request_id=synthesis.usage.request_id,
                )
            else:
                logger.error(
                    "voice_synthesis_attempt_missing_after_rollback",
                    extra={"request_id": str(idempotency_key)},
                )
            raise

        return VoiceChatResponse(
            **chat.model_dump(exclude={"media", "transcript", "message_type"}),
            message_type=MessageType.audio.value,
            transcript=transcript,
            media=self._media_descriptor(asset),
        )

    async def _get_existing_audio_turn(
        self,
        *,
        user: User,
        companion: CompanionProfile,
        conversation: Conversation,
        idempotency_key: uuid.UUID,
        request_voice_response: bool,
        voice_id: str | None,
        voice_settings: dict,
    ) -> VoiceChatResponse | None:
        """Replay a completed voice turn before purchasing another transcription."""
        user_message = await self.message_repo.get_user_by_request_id(
            conversation_id=conversation.id,
            request_id=idempotency_key,
        )
        if user_message is None:
            return None
        if user_message.message_type != MessageType.audio:
            raise ValidationError("The idempotency key was already used for a text request.")
        assistant = await self.message_repo.get_first_assistant_after(
            conversation_id=conversation.id,
            sequence=user_message.sequence,
        )
        if assistant is None or assistant.role != MessageRole.assistant:
            raise ProviderError("The previous voice operation is incomplete.")

        chat = ChatResponse(
            message_id=assistant.id,
            conversation_id=conversation.id,
            companion_id=companion.id,
            response=assistant.content,
            created_at=assistant.created_at,
            usage=ChatUsage(
                input_tokens=assistant.input_tokens or 0,
                output_tokens=assistant.output_tokens or 0,
            ),
            message_type=assistant.message_type.value,
            transcript=user_message.content,
        )
        if assistant.media_asset_id is not None:
            asset = await self.media_repo.get_by_id_for_user(assistant.media_asset_id, user.id)
            if asset is None:
                raise ProviderError("The previous voice response media is unavailable.")
            await self.message_repo.session.commit()
            return VoiceChatResponse(
                **chat.model_dump(exclude={"media", "transcript", "message_type"}),
                message_type=MessageType.audio.value,
                transcript=user_message.content,
                media=self._media_descriptor(asset),
            )
        if request_voice_response:
            return await self._synthesize_chat_response(
                chat=chat,
                user=user,
                companion=companion,
                conversation=conversation,
                voice_id=voice_id or "",
                voice_settings=voice_settings,
                idempotency_key=idempotency_key,
                transcript=user_message.content,
            )
        await self.message_repo.session.commit()
        return VoiceChatResponse(
            **chat.model_dump(exclude={"media", "transcript", "message_type"}),
            message_type=MessageType.text.value,
            transcript=user_message.content,
            media=None,
        )

    async def process_text(
        self,
        auth: AuthContext,
        *,
        conversation_id: uuid.UUID,
        companion_id: uuid.UUID,
        message: str,
        idempotency_key: uuid.UUID,
        background_tasks,
    ) -> VoiceChatResponse:
        """Generate a normal chat turn and synthesize its companion reply."""
        require_feature(auth, "voice_output")
        user, companion, conversation = await self._load_context(
            auth, conversation_id=conversation_id, companion_id=companion_id
        )
        voice_id, voice_settings = self._voice_config(companion)
        chat = await self._chat_service().send_message(
            auth,
            ChatRequest(
                conversation_id=conversation_id,
                companion_id=companion_id,
                message=message,
                idempotency_key=idempotency_key,
            ),
            background_tasks,
        )
        return await self._synthesize_chat_response(
            chat=chat,
            user=user,
            companion=companion,
            conversation=conversation,
            voice_id=voice_id,
            voice_settings=voice_settings,
            idempotency_key=idempotency_key,
            transcript=None,
        )

    async def synthesize_existing_response(
        self,
        auth: AuthContext,
        *,
        chat: ChatResponse,
        conversation_id: uuid.UUID,
        companion_id: uuid.UUID,
        idempotency_key: uuid.UUID,
    ) -> VoiceChatResponse:
        """Attach speech to an already-persisted proactive/text response."""
        require_feature(auth, "voice_output")
        user, companion, conversation = await self._load_context(
            auth, conversation_id=conversation_id, companion_id=companion_id
        )
        voice_id, voice_settings = self._voice_config(companion)
        return await self._synthesize_chat_response(
            chat=chat,
            user=user,
            companion=companion,
            conversation=conversation,
            voice_id=voice_id,
            voice_settings=voice_settings,
            idempotency_key=idempotency_key,
            transcript=None,
        )

    async def process(
        self,
        auth: AuthContext,
        *,
        conversation_id: uuid.UUID,
        companion_id: uuid.UUID,
        audio: bytes,
        mime_type: str,
        filename: str,
        request_voice_response: bool,
        idempotency_key: uuid.UUID,
        background_tasks,
    ) -> VoiceChatResponse:
        require_feature(auth, "voice_input")
        if not self.settings.ENABLE_VOICE_INPUT:
            raise ProviderError("Voice input is disabled for this deployment.")
        if mime_type not in _SUPPORTED_AUDIO_TYPES:
            raise ValidationError("Unsupported audio type.")
        if not audio or len(audio) > self.settings.MAX_AUDIO_UPLOAD_BYTES:
            raise ValidationError("Audio upload is empty or exceeds the configured size limit.")
        _validate_audio_signature(audio, mime_type)
        try:
            duration_seconds = inspect_audio_duration(audio, mime_type)
        except AudioInspectionError as exc:
            raise ValidationError("The audio duration could not be determined safely.") from exc
        if duration_seconds > self.settings.MAX_AUDIO_DURATION_SECONDS:
            raise ValidationError("Audio duration exceeds the configured limit.")

        user, companion, conversation = await self._load_context(
            auth, conversation_id=conversation_id, companion_id=companion_id
        )

        voice_id: str | None = None
        voice_settings: dict = {}
        if request_voice_response:
            require_feature(auth, "voice_output")
            voice_id, voice_settings = self._voice_config(companion)

        await self.message_repo.acquire_idempotency_lock(
            scope="voice_input", request_id=idempotency_key
        )
        input_sha256 = hashlib.sha256(audio).hexdigest()
        previous_attempt = await self.message_repo.get_ai_event_by_request_id(
            request_id=idempotency_key,
            user_id=user.id,
            conversation_id=conversation.id,
            event_type="voice_transcription_attempt",
        )
        if previous_attempt is not None:
            previous_fingerprint = (previous_attempt.event_metadata or {}).get("input_sha256")
            if previous_fingerprint != input_sha256:
                raise ValidationError(
                    "The idempotency key was already used for a different audio upload."
                )
        existing = await self._get_existing_audio_turn(
            user=user,
            companion=companion,
            conversation=conversation,
            idempotency_key=idempotency_key,
            request_voice_response=request_voice_response,
            voice_id=voice_id,
            voice_settings=voice_settings,
        )
        if existing is not None:
            return existing
        if previous_attempt is not None:
            raise ProviderError(
                "The previous transcription attempt is incomplete or terminal; "
                "use a new idempotency key."
            )

        attempt = AIEvent(
            request_id=idempotency_key,
            user_id=user.id,
            companion_id=companion.id,
            conversation_id=conversation.id,
            event_type="voice_transcription_attempt",
            model=self.settings.ELEVENLABS_STT_MODEL,
            event_metadata={
                "status": "started",
                "input_sha256": input_sha256,
                "byte_size": len(audio),
                "mime_type": mime_type,
                "duration_seconds": duration_seconds,
            },
        )
        self.message_repo.session.add(attempt)
        await self.message_repo.session.commit()
        await self.message_repo.acquire_idempotency_lock(
            scope="voice_input", request_id=idempotency_key
        )

        started = time.perf_counter()
        try:
            transcription = await self.voice_provider.transcribe(
                audio,
                mime_type=mime_type,
                model=self.settings.ELEVENLABS_STT_MODEL,
                filename=filename,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            await self._finish_attempt_without_masking_error(
                attempt,
                "provider_failed",
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise
        transcription_latency_ms = (time.perf_counter() - started) * 1000
        transcript = transcription.text.strip()
        if not transcript:
            await self._finish_attempt_without_masking_error(
                attempt,
                "empty_transcript",
                latency_ms=transcription_latency_ms,
                provider_request_id=transcription.usage.request_id,
            )
            raise ValidationError("The audio did not contain transcribable speech.")
        if len(transcript) > self.settings.MAX_TRANSCRIPT_CHARACTERS:
            await self._finish_attempt_without_masking_error(
                attempt,
                "oversized_transcript",
                latency_ms=transcription_latency_ms,
                transcript_characters=len(transcript),
                provider_request_id=transcription.usage.request_id,
            )
            raise ValidationError("The audio transcript exceeds the supported message length.")

        await self._finish_attempt(
            attempt,
            "transcribed",
            latency_ms=transcription_latency_ms,
            language_code=transcription.language_code,
            provider_request_id=transcription.usage.request_id,
        )

        chat = await self._chat_service().send_message(
            auth,
            ChatRequest(
                conversation_id=conversation_id,
                companion_id=companion_id,
                message=transcript,
                idempotency_key=idempotency_key,
            ),
            background_tasks,
            user_message_type=MessageType.audio,
        )
        self.message_repo.session.add(
            AIEvent(
                request_id=idempotency_key,
                user_id=user.id,
                companion_id=companion.id,
                conversation_id=conversation.id,
                event_type="voice_transcription",
                model=transcription.model,
                latency_ms=transcription_latency_ms,
                event_metadata={
                    "language_code": transcription.language_code,
                    "request_id": transcription.usage.request_id,
                },
            )
        )

        if not request_voice_response:
            await self.message_repo.session.commit()
            return VoiceChatResponse(
                **chat.model_dump(exclude={"media", "transcript", "message_type"}),
                message_type=MessageType.text.value,
                transcript=transcript,
                media=None,
            )

        return await self._synthesize_chat_response(
            chat=chat,
            user=user,
            companion=companion,
            conversation=conversation,
            voice_id=voice_id or "",
            voice_settings=voice_settings,
            idempotency_key=idempotency_key,
            transcript=transcript,
        )
