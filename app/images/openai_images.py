"""OpenAI GPT Image adapter with high-fidelity reference-image support."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from openai import APIError, AsyncOpenAI

from app.core.exceptions import ProviderError
from app.images.base import ImageGenerationResult, ImageProvider, ReferenceImage


class OpenAIImageProvider(ImageProvider):
    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is required for image generation.")
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @staticmethod
    def _decode_response(response: Any, *, model: str) -> ImageGenerationResult:
        data_items = getattr(response, "data", None)
        first = data_items[0] if data_items else None
        encoded = getattr(first, "b64_json", None)
        if not isinstance(encoded, str) or not encoded:
            raise ProviderError("The image provider returned no image data.")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProviderError("The image provider returned invalid image data.") from exc
        if not image_bytes:
            raise ProviderError("The image provider returned an empty image.")
        raw_usage = getattr(response, "usage", None)
        if hasattr(raw_usage, "model_dump"):
            usage = raw_usage.model_dump(exclude_none=True)
        elif isinstance(raw_usage, dict):
            usage = raw_usage
        else:
            usage = {}
        return ImageGenerationResult(
            data=image_bytes,
            mime_type="image/png",
            model=model,
            revised_prompt=getattr(first, "revised_prompt", None),
            usage=usage,
        )

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        reference_images: list[ReferenceImage],
        size: str,
        quality: str,
    ) -> ImageGenerationResult:
        try:
            if reference_images:
                # GPT Image 2 automatically processes reference inputs at high
                # fidelity. Passing input_fidelity is intentionally omitted for
                # that model because the API does not accept an override.
                image_files = [
                    (image.filename, image.data, image.mime_type)
                    for image in reference_images
                ]
                response = await self._client.images.edit(
                    model=model,
                    image=image_files,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    output_format="png",
                )
            else:
                response = await self._client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    output_format="png",
                )
        except APIError as exc:
            raise ProviderError("OpenAI image generation failed.") from exc
        return self._decode_response(response, model=model)
