"""Unit tests for preferred xAI image generation and its GPT fallback."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest

from app.core.exceptions import ProviderError
from app.images.base import ImageGenerationResult, ReferenceImage
from app.images.failover import FailoverImageProvider
from app.images.xai_images import XAIImageProvider


def test_xai_response_decoding_preserves_mime_type_and_usage() -> None:
    result = XAIImageProvider._decode_response(
        {
            "data": [
                {
                    "b64_json": base64.b64encode(b"jpeg bytes").decode("ascii"),
                    "mime_type": "image/jpeg",
                    "revised_prompt": "safe prompt",
                }
            ],
            "usage": {"cost_in_usd_ticks": 400000000},
        },
        model="grok-imagine-image-2.0",
    )

    assert result.data == b"jpeg bytes"
    assert result.provider == "xai"
    assert result.mime_type == "image/jpeg"
    assert result.usage == {"cost_in_usd_ticks": 400000000}


async def test_xai_provider_keeps_every_reference_it_receives() -> None:
    provider = XAIImageProvider(
        api_key="test-key",
        base_url="https://xai.example/v1",
        timeout_seconds=5,
        max_retries=1,
    )
    references = [
        ReferenceImage(data=b"image", filename=f"reference-{index}.png", mime_type="image/png")
        for index in range(3)
    ]

    # The service owns the five-image limit; this adapter must not silently
    # discard references that it receives.
    captured_body: dict = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {"b64_json": base64.b64encode(b"generated").decode("ascii")}
                ]
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, _endpoint: str, *, json: dict) -> _Response:
            captured_body.update(json)
            return _Response()

    with patch("app.images.xai_images.httpx.AsyncClient", return_value=_Client()):
        await provider.generate(
            prompt="portrait",
            model="grok-imagine-image-2.0",
            reference_images=references,
            size="1024x1024",
            quality="medium",
        )

    assert len(captured_body["images"]) == 3


async def test_falls_back_to_openai_when_xai_generation_fails() -> None:
    xai = AsyncMock()
    xai.generate.side_effect = ProviderError("xAI image generation failed.")
    openai = AsyncMock()
    openai.generate.return_value = ImageGenerationResult(
        data=b"png bytes", mime_type="image/png", model="gpt-image-2"
    )
    provider = FailoverImageProvider(
        primary=xai, fallback=openai, fallback_model="gpt-image-2"
    )

    result = await provider.generate(
        prompt="portrait",
        model="grok-imagine-image-2.0",
        reference_images=[],
        size="1024x1024",
        quality="medium",
    )

    xai.generate.assert_awaited_once()
    openai.generate.assert_awaited_once()
    assert openai.generate.await_args.kwargs["model"] == "gpt-image-2"
    assert result.model == "gpt-image-2"


async def test_uses_openai_directly_when_xai_is_not_configured() -> None:
    openai = AsyncMock()
    openai.generate.return_value = ImageGenerationResult(
        data=b"png bytes", mime_type="image/png", model="gpt-image-2"
    )
    provider = FailoverImageProvider(
        primary=None, fallback=openai, fallback_model="gpt-image-2"
    )

    result = await provider.generate(
        prompt="portrait",
        model="gpt-image-2",
        reference_images=[],
        size="1024x1024",
        quality="medium",
    )

    openai.generate.assert_awaited_once()
    assert result.provider == "openai"


def test_rejects_no_image_providers() -> None:
    with pytest.raises(ValueError, match="At least one"):
        FailoverImageProvider(primary=None, fallback=None, fallback_model="gpt-image-2")
