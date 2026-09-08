"""Focused unit tests for the OpenAI image-generation adapter."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import APIError

from app.core.exceptions import ProviderError
from app.images.base import ReferenceImage
from app.images.openai_images import OpenAIImageProvider


def _response(encoded: str | None, *, revised_prompt: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=encoded, revised_prompt=revised_prompt)]
    )


def _api_error(message: str = "sensitive upstream detail") -> APIError:
    request = SimpleNamespace(method="POST", url="https://api.openai.com/v1/images")
    return APIError(message, request=request, body={"api_key": "secret-key"})


def _provider() -> OpenAIImageProvider:
    return OpenAIImageProvider(
        api_key="test-openai-key",
        timeout_seconds=5,
        max_retries=1,
    )


class TestOpenAIImageProviderGeneration:
    async def test_reference_images_are_passed_to_images_edit(self) -> None:
        provider = _provider()
        references = [
            ReferenceImage(
                data=b"first reference",
                filename="lina-front.png",
                mime_type="image/png",
            ),
            ReferenceImage(
                data=b"second reference",
                filename="lina-profile.jpg",
                mime_type="image/jpeg",
            ),
        ]
        generated = b"generated png bytes"

        with (
            patch.object(
                provider._client.images,
                "edit",
                new=AsyncMock(
                    return_value=_response(
                        base64.b64encode(generated).decode("ascii"),
                        revised_prompt="A revised safe prompt",
                    )
                ),
            ) as edit,
            patch.object(provider._client.images, "generate", new=AsyncMock()) as generate,
        ):
            result = await provider.generate(
                prompt="Lina walking through Paris",
                model="gpt-image-2",
                reference_images=references,
                size="1024x1024",
                quality="high",
            )

        edit.assert_awaited_once_with(
            model="gpt-image-2",
            image=[
                ("lina-front.png", b"first reference", "image/png"),
                ("lina-profile.jpg", b"second reference", "image/jpeg"),
            ],
            prompt="Lina walking through Paris",
            size="1024x1024",
            quality="high",
            output_format="png",
        )
        generate.assert_not_awaited()
        assert result.data == generated
        assert result.mime_type == "image/png"
        assert result.model == "gpt-image-2"
        assert result.revised_prompt == "A revised safe prompt"

    async def test_no_references_uses_text_only_generate_path(self) -> None:
        provider = _provider()
        generated = b"text-only generated image"

        with (
            patch.object(
                provider._client.images,
                "generate",
                new=AsyncMock(
                    return_value=_response(base64.b64encode(generated).decode("ascii"))
                ),
            ) as generate,
            patch.object(provider._client.images, "edit", new=AsyncMock()) as edit,
        ):
            result = await provider.generate(
                prompt="A fictional evening scene",
                model="gpt-image-2",
                reference_images=[],
                size="1536x1024",
                quality="medium",
            )

        generate.assert_awaited_once_with(
            model="gpt-image-2",
            prompt="A fictional evening scene",
            size="1536x1024",
            quality="medium",
            output_format="png",
        )
        edit.assert_not_awaited()
        assert result.data == generated


class TestOpenAIImageProviderValidation:
    @pytest.mark.parametrize(
        ("response", "message"),
        [
            (SimpleNamespace(data=[]), "returned no image data"),
            (_response(None), "returned no image data"),
            (_response("not-valid-base64%%%"), "returned invalid image data"),
            (_response(base64.b64encode(b"").decode("ascii")), "returned no image data"),
        ],
    )
    async def test_rejects_missing_invalid_or_empty_provider_output(
        self, response: SimpleNamespace, message: str
    ) -> None:
        provider = _provider()

        with patch.object(
            provider._client.images,
            "generate",
            new=AsyncMock(return_value=response),
        ):
            with pytest.raises(ProviderError, match=message):
                await provider.generate(
                    prompt="test",
                    model="gpt-image-2",
                    reference_images=[],
                    size="1024x1024",
                    quality="high",
                )

    async def test_api_failure_is_sanitized(self) -> None:
        provider = _provider()

        with patch.object(
            provider._client.images,
            "generate",
            new=AsyncMock(side_effect=_api_error()),
        ):
            with pytest.raises(ProviderError) as raised:
                await provider.generate(
                    prompt="test",
                    model="gpt-image-2",
                    reference_images=[],
                    size="1024x1024",
                    quality="high",
                )

        assert str(raised.value) == "OpenAI image generation failed."
        assert "sensitive" not in str(raised.value)
        assert "secret-key" not in str(raised.value)


def test_constructor_rejects_an_empty_api_key() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        OpenAIImageProvider(api_key=" ", timeout_seconds=5, max_retries=1)
