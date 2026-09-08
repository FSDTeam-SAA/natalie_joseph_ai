from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import ModerationBlockedError, ProviderError, ValidationError
from app.core.security import AuthContext
from app.images.base import ImageGenerationResult
from app.moderation.base import ModerationResult
from app.schemas.media import ImageGenerationRequest
from app.services.image_service import ImageService
from app.storage.base import StoredObject


def _moderation(flagged: bool = False) -> ModerationResult:
    return ModerationResult(
        flagged=flagged,
        categories={"sexual/minors": flagged, "violence": False},
        category_scores={},
        raw_response={},
    )


def _auth(*, trusted: bool = False) -> AuthContext:
    return AuthContext(
        user_id=uuid.uuid4(),
        adult_eligible=True,
        entitled=False,
        raw_claims={"source": "trusted_backend"} if trusted else {},
        features=frozenset({"image"}),
    )


def _service(tmp_path) -> tuple[ImageService, SimpleNamespace]:
    reference = tmp_path / "lina-reference.png"
    reference.write_bytes(b"\x89PNG\r\n\x1a\n" + b"reference")
    user = SimpleNamespace(id=uuid.uuid4())
    companion = SimpleNamespace(
        id=uuid.uuid4(),
        name="Lina",
        active=True,
        version=2,
        visual_config={
            "reference_images": [reference.name],
            "physical_identity": "Use the approved fictional identity.",
            "aesthetic_keywords": ["Paris", "warm"],
            "generation_instructions": "Keep her appearance consistent.",
        },
    )
    conversation = SimpleNamespace(
        id=uuid.uuid4(), companion_id=companion.id, summary=None
    )
    added: list[object] = []

    async def refresh(instance):
        if getattr(instance, "id", None) is None:
            instance.id = uuid.uuid4()
        if getattr(instance, "created_at", None) is None:
            instance.created_at = datetime.now(UTC)

    session = SimpleNamespace(
        add=lambda item: added.append(item),
        commit=AsyncMock(),
        rollback=AsyncMock(),
        refresh=AsyncMock(side_effect=refresh),
    )
    media_repo = SimpleNamespace(
        get_by_idempotency_key=AsyncMock(return_value=None),
        add=AsyncMock(),
    )

    async def add_asset(asset):
        asset.id = uuid.uuid4()
        return asset

    media_repo.add.side_effect = add_asset
    deps = SimpleNamespace(
        user=user,
        companion=companion,
        conversation=conversation,
        added=added,
        session=session,
        media_repo=media_repo,
        message_repo=SimpleNamespace(
            session=session,
            acquire_idempotency_lock=AsyncMock(),
            get_recent_for_conversation=AsyncMock(return_value=[]),
            get_by_media_asset=AsyncMock(),
            get_ai_event_by_request_id=AsyncMock(return_value=None),
        ),
        image_provider=SimpleNamespace(
            generate=AsyncMock(
                return_value=ImageGenerationResult(
                    data=b"\x89PNG\r\n\x1a\n" + b"generated",
                    mime_type="image/png",
                    model="gpt-image-2",
                    usage={"total_tokens": 25},
                )
            )
        ),
        moderation_provider=SimpleNamespace(
            moderate_text=AsyncMock(return_value=_moderation()),
            moderate_multimodal=AsyncMock(return_value=_moderation()),
        ),
        storage=SimpleNamespace(
            put=AsyncMock(
                return_value=StoredObject(
                    key="1" * 32 + ".png", byte_size=17, sha256="b" * 64
                )
            ),
            delete=AsyncMock(),
        ),
    )
    service = ImageService(
        user_repo=SimpleNamespace(
            get_or_create_by_external_user_id=AsyncMock(return_value=user)
        ),
        companion_repo=SimpleNamespace(get_by_id=AsyncMock(return_value=companion)),
        conversation_repo=SimpleNamespace(
            get_by_id_for_user=AsyncMock(return_value=conversation)
        ),
        message_repo=deps.message_repo,
        media_repo=media_repo,
        image_provider=deps.image_provider,
        moderation_provider=deps.moderation_provider,
        storage=deps.storage,
        settings=Settings(
            _env_file=None,
            OPENAI_API_KEY="test",
            ENABLE_IMAGE_GENERATION=True,
            COMPANION_ASSET_ROOT=str(tmp_path),
        ),
    )
    return service, deps


async def test_image_request_uses_reference_moderates_stores_and_returns_private_url(
    tmp_path,
) -> None:
    service, deps = _service(tmp_path)
    request = ImageGenerationRequest(
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        prompt="Share a picture from your Paris evening",
        idempotency_key=uuid.uuid4(),
    )

    response = await service.generate(_auth(), request)

    references = deps.image_provider.generate.await_args.kwargs["reference_images"]
    assert len(references) == 1
    assert references[0].filename == "lina-reference.png"
    assert deps.moderation_provider.moderate_text.await_count == 2
    deps.moderation_provider.moderate_multimodal.assert_awaited_once()
    deps.storage.put.assert_awaited_once()
    assert response.media.url.endswith(str(response.media.id))
    assert "storage" not in response.media.url
    assert response.provider == "openai"


async def test_contextual_image_requires_trusted_backend(tmp_path) -> None:
    service, deps = _service(tmp_path)
    request = ImageGenerationRequest(
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        prompt="A contextual evening check-in",
        trigger="contextual",
        idempotency_key=uuid.uuid4(),
    )
    with pytest.raises(ValidationError, match="trusted backend"):
        await service.generate(_auth(trusted=False), request)
    deps.image_provider.generate.assert_not_awaited()


async def test_generated_image_is_not_stored_when_output_moderation_blocks(tmp_path) -> None:
    service, deps = _service(tmp_path)
    deps.moderation_provider.moderate_multimodal.return_value = _moderation(True)
    request = ImageGenerationRequest(
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        prompt="Safe request that produces unsafe provider output",
        idempotency_key=uuid.uuid4(),
    )

    with pytest.raises(ModerationBlockedError):
        await service.generate(_auth(), request)

    deps.storage.put.assert_not_awaited()
    deps.session.commit.assert_awaited_once()  # safety audit event


async def test_missing_reference_fails_before_paid_generation(tmp_path) -> None:
    service, deps = _service(tmp_path)
    deps.companion.visual_config["reference_images"] = []
    request = ImageGenerationRequest(
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        prompt="A portrait",
        idempotency_key=uuid.uuid4(),
    )
    with pytest.raises(ProviderError, match="no reference image"):
        await service.generate(_auth(), request)
    deps.image_provider.generate.assert_not_awaited()


async def test_invalid_provider_image_is_rejected_before_storage(tmp_path) -> None:
    service, deps = _service(tmp_path)
    deps.image_provider.generate.return_value = ImageGenerationResult(
        data=b"this is not a PNG",
        mime_type="image/png",
        model="gpt-image-2",
        usage={},
    )
    request = ImageGenerationRequest(
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        prompt="A portrait",
        idempotency_key=uuid.uuid4(),
    )

    with pytest.raises(ProviderError, match="invalid image data"):
        await service.generate(_auth(), request)

    deps.moderation_provider.moderate_multimodal.assert_not_awaited()
    deps.storage.put.assert_not_awaited()


async def test_completed_image_key_cannot_be_reused_for_a_different_prompt(tmp_path) -> None:
    service, deps = _service(tmp_path)
    idempotency_key = uuid.uuid4()
    first = ImageGenerationRequest(
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        prompt="A Paris evening portrait",
        idempotency_key=idempotency_key,
    )
    await service.generate(_auth(), first)
    existing_asset = deps.media_repo.add.await_args.args[0]
    deps.media_repo.get_by_idempotency_key.return_value = existing_asset

    with pytest.raises(ValidationError, match="different image request data"):
        await service.generate(
            _auth(),
            ImageGenerationRequest(
                conversation_id=deps.conversation.id,
                companion_id=deps.companion.id,
                prompt="A completely different scene",
                idempotency_key=idempotency_key,
            ),
        )

    assert deps.image_provider.generate.await_count == 1


async def test_failed_paid_image_attempt_is_not_purchased_again(tmp_path) -> None:
    service, deps = _service(tmp_path)
    deps.message_repo.get_ai_event_by_request_id.return_value = SimpleNamespace(
        event_metadata={"status": "invalid_output"}
    )
    request = ImageGenerationRequest(
        conversation_id=deps.conversation.id,
        companion_id=deps.companion.id,
        prompt="A portrait",
        idempotency_key=uuid.uuid4(),
    )

    with pytest.raises(ProviderError, match="previous image generation"):
        await service.generate(_auth(), request)

    deps.image_provider.generate.assert_not_awaited()
