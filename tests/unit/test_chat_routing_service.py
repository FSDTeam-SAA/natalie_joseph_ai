from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import BackgroundTasks
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import AuthorizationError, ValidationError
from app.core.security import AuthContext
from app.db.models.message import MessageType
from app.schemas.chat import ChatRequest, ChatResponse, ChatUsage
from app.schemas.media import ImageGenerationResponse, MediaDescriptor
from app.services.chat_routing_service import ChatRoutingService, ImageRequestClassifier


def _auth(*features: str) -> AuthContext:
    return AuthContext(
        user_id=uuid.uuid4(),
        adult_eligible=False,
        entitled=True,
        raw_claims={"source": "trusted_backend"},
        features=frozenset(features),
    )


def _request(message: str) -> ChatRequest:
    return ChatRequest(
        conversation_id=uuid.uuid4(),
        companion_id=uuid.uuid4(),
        message=message,
        idempotency_key=uuid.uuid4(),
    )


def _message_repo(*, existing_user=None, existing_assistant=None):
    return SimpleNamespace(
        acquire_idempotency_lock=AsyncMock(),
        get_user_by_request_id_for_external_user=AsyncMock(return_value=existing_user),
        get_first_assistant_after=AsyncMock(return_value=existing_assistant),
    )


@pytest.mark.parametrize(
    "message",
    [
        "Send me a photo of you at the beach, please.",
        "Could you share a cozy selfie from your apartment?",
        "I would love a picture of you in that red dress.",
        "That sounds fun! Show me yourself at the concert.",
        "What do you look like?",
        "Please generate a portrait of yourself in Paris.",
        "Show me yourself naked.",
        "Show me what you're wearing tonight.",
        "Send me a picture from your evening.",
        "তোমার একটা ছবি পাঠাও।",
        "আমাকে একটা সেলফি দেখাও।",
        "Tomar ekta chobi pathao.",
        "Ajke ki porecho dekhao.",
    ],
)
def test_classifier_routes_explicit_natural_language_image_requests(message: str) -> None:
    assert ImageRequestClassifier().is_image_request(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "The photo you sent yesterday made me smile.",
        "Don't send me a photo right now.",
        "Can you generate images?",
        "Tell me how image generation works.",
        "How do I create an image?",
        "Show me how image generation works.",
        "I would like to discuss your photo.",
        "Describe what you look like.",
        "ছবি নিয়ে কথা বলি।",
        "Send me a voice note.",
        "What did you do today?",
    ],
)
def test_classifier_leaves_mentions_negations_and_capability_questions_in_chat(
    message: str,
) -> None:
    assert ImageRequestClassifier().is_image_request(message) is False


async def test_explicit_image_request_delegates_with_same_idempotency_key() -> None:
    request = _request("Can you send me a selfie from the cafe?")
    media = MediaDescriptor(
        id=uuid.uuid4(),
        kind="image",
        url=f"/api/v1/media/{uuid.uuid4()}",
        mime_type="image/png",
        byte_size=1234,
    )
    generated = ImageGenerationResponse(
        message_id=uuid.uuid4(),
        conversation_id=request.conversation_id,
        companion_id=request.companion_id,
        caption="Lina shared an AI-generated image.",
        media=media,
        created_at=datetime.now(UTC),
        provider="openai",
        model="gpt-image-2",
    )
    chat_service = SimpleNamespace(send_message=AsyncMock())
    image_service = SimpleNamespace(generate=AsyncMock(return_value=generated))
    message_repo = _message_repo()
    service = ChatRoutingService(
        chat_service=chat_service,
        image_service=image_service,
        message_repo=message_repo,
    )

    response = await service.send_message(
        _auth("chat", "image"), request, BackgroundTasks()
    )

    chat_service.send_message.assert_not_awaited()
    image_service.generate.assert_awaited_once()
    delegated = image_service.generate.await_args.args[1]
    assert delegated.conversation_id == request.conversation_id
    assert delegated.companion_id == request.companion_id
    assert delegated.prompt == request.message
    assert delegated.trigger == "user_requested"
    assert delegated.idempotency_key == request.idempotency_key
    message_repo.acquire_idempotency_lock.assert_awaited_once_with(
        scope="chat_route", request_id=request.idempotency_key
    )
    assert response == ChatResponse(
        message_id=generated.message_id,
        conversation_id=generated.conversation_id,
        companion_id=generated.companion_id,
        response=generated.caption,
        created_at=generated.created_at,
        usage=ChatUsage(input_tokens=0, output_tokens=0),
        message_type="image",
        media=media,
    )


async def test_ordinary_message_uses_existing_chat_pipeline_unchanged() -> None:
    request = _request("The photo from yesterday made me smile. How was your day?")
    expected = ChatResponse(
        message_id=uuid.uuid4(),
        conversation_id=request.conversation_id,
        companion_id=request.companion_id,
        response="I'm glad it did.",
        created_at=datetime.now(UTC),
        usage=ChatUsage(input_tokens=12, output_tokens=5),
    )
    chat_service = SimpleNamespace(send_message=AsyncMock(return_value=expected))
    image_service = SimpleNamespace(generate=AsyncMock())
    background_tasks = BackgroundTasks()
    auth = _auth("chat", "image")
    service = ChatRoutingService(
        chat_service=chat_service,
        image_service=image_service,
        message_repo=_message_repo(),
    )

    response = await service.send_message(auth, request, background_tasks)

    assert response is expected
    chat_service.send_message.assert_awaited_once_with(auth, request, background_tasks)
    image_service.generate.assert_not_awaited()


async def test_chat_grant_is_required_before_either_pipeline_runs() -> None:
    request = _request("Send me a selfie.")
    chat_service = SimpleNamespace(send_message=AsyncMock())
    image_service = SimpleNamespace(generate=AsyncMock())
    message_repo = _message_repo()
    service = ChatRoutingService(
        chat_service=chat_service,
        image_service=image_service,
        message_repo=message_repo,
    )

    with pytest.raises(AuthorizationError, match="'chat'"):
        await service.send_message(_auth("image"), request, BackgroundTasks())

    chat_service.send_message.assert_not_awaited()
    image_service.generate.assert_not_awaited()
    message_repo.acquire_idempotency_lock.assert_not_awaited()


async def test_image_entitlement_error_is_not_retried_as_text_chat() -> None:
    request = _request("Send me a selfie.")
    denied = AuthorizationError("The trusted backend did not authorize the 'image' AI feature.")
    chat_service = SimpleNamespace(send_message=AsyncMock())
    image_service = SimpleNamespace(generate=AsyncMock(side_effect=denied))
    service = ChatRoutingService(
        chat_service=chat_service,
        image_service=image_service,
        message_repo=_message_repo(),
    )

    with pytest.raises(AuthorizationError) as raised:
        await service.send_message(_auth("chat"), request, BackgroundTasks())

    assert raised.value is denied
    image_service.generate.assert_awaited_once()
    chat_service.send_message.assert_not_awaited()


@pytest.mark.parametrize(
    ("original_message", "retry_message", "previous_type"),
    [
        ("Send me a selfie.", "How are you?", MessageType.image),
        ("How are you?", "Send me a selfie.", MessageType.text),
    ],
)
async def test_cross_modal_changed_content_is_rejected_before_generation(
    original_message: str,
    retry_message: str,
    previous_type: MessageType,
) -> None:
    request = _request(retry_message)
    existing_user = SimpleNamespace(content=original_message, sequence=20)
    existing_assistant = SimpleNamespace(message_type=previous_type)
    message_repo = _message_repo(
        existing_user=existing_user,
        existing_assistant=existing_assistant,
    )
    chat_service = SimpleNamespace(send_message=AsyncMock())
    image_service = SimpleNamespace(generate=AsyncMock())
    service = ChatRoutingService(
        chat_service=chat_service,
        image_service=image_service,
        message_repo=message_repo,
    )

    with pytest.raises(ValidationError, match="different message content"):
        await service.send_message(_auth("chat", "image"), request, BackgroundTasks())

    chat_service.send_message.assert_not_awaited()
    image_service.generate.assert_not_awaited()
    message_repo.get_first_assistant_after.assert_not_awaited()


@pytest.mark.parametrize(
    ("current_is_image", "previous_type"),
    [
        (False, MessageType.image),
        (True, MessageType.text),
    ],
)
async def test_same_payload_cannot_change_modality_across_classifier_versions(
    current_is_image: bool,
    previous_type: MessageType,
) -> None:
    request = _request("An intentionally ambiguous but unchanged request")
    existing_user = SimpleNamespace(content=request.message, sequence=30)
    existing_assistant = SimpleNamespace(message_type=previous_type)
    message_repo = _message_repo(
        existing_user=existing_user,
        existing_assistant=existing_assistant,
    )
    classifier = SimpleNamespace(is_image_request=Mock(return_value=current_is_image))
    chat_service = SimpleNamespace(send_message=AsyncMock())
    image_service = SimpleNamespace(generate=AsyncMock())
    service = ChatRoutingService(
        chat_service=chat_service,
        image_service=image_service,
        message_repo=message_repo,
        image_classifier=classifier,
    )

    with pytest.raises(ValidationError, match="different response modality"):
        await service.send_message(_auth("chat", "image"), request, BackgroundTasks())

    chat_service.send_message.assert_not_awaited()
    image_service.generate.assert_not_awaited()


async def test_existing_image_request_replays_with_media_descriptor() -> None:
    request = _request("Send me a picture from your evening.")
    media = MediaDescriptor(
        id=uuid.uuid4(),
        kind="image",
        url=f"/api/v1/media/{uuid.uuid4()}",
        mime_type="image/png",
        byte_size=4321,
    )
    replayed = ImageGenerationResponse(
        message_id=uuid.uuid4(),
        conversation_id=request.conversation_id,
        companion_id=request.companion_id,
        caption="Elena shared an AI-generated image.",
        media=media,
        created_at=datetime.now(UTC),
        provider="openai",
        model="gpt-image-2",
    )
    message_repo = _message_repo(
        existing_user=SimpleNamespace(content=request.message, sequence=40),
        existing_assistant=SimpleNamespace(message_type=MessageType.image),
    )
    chat_service = SimpleNamespace(send_message=AsyncMock())
    image_service = SimpleNamespace(generate=AsyncMock(return_value=replayed))
    service = ChatRoutingService(
        chat_service=chat_service,
        image_service=image_service,
        message_repo=message_repo,
    )

    response = await service.send_message(
        _auth("chat", "image"), request, BackgroundTasks()
    )

    assert response.message_type == "image"
    assert response.media == media
    assert response.message_id == replayed.message_id
    image_service.generate.assert_awaited_once()
    chat_service.send_message.assert_not_awaited()


def test_chat_request_requires_an_idempotency_key() -> None:
    with pytest.raises(PydanticValidationError):
        ChatRequest(
            conversation_id=uuid.uuid4(),
            companion_id=uuid.uuid4(),
            message="Hello",
        )
