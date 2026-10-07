"""xAI Grok Imagine image-generation adapter."""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

import httpx

from app.core.exceptions import ProviderError
from app.images.base import ImageGenerationResult, ImageProvider, ReferenceImage

logger = logging.getLogger(__name__)


class XAIImageProvider(ImageProvider):
    """Generate and edit images through xAI's JSON-based Images API."""

    def __init__(
        self, *, api_key: str, base_url: str, timeout_seconds: float, max_retries: int
    ) -> None:
        if not api_key.strip():
            raise ValueError("xAI API key is required for image generation.")
        if not base_url.strip():
            raise ValueError("xAI base URL is required for image generation.")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout_seconds

    @staticmethod
    def _aspect_ratio(size: str) -> str | None:
        return {"1024x1024": "1:1", "1024x1536": "2:3", "1536x1024": "3:2"}.get(size)

    @staticmethod
    def _decode_response(payload: dict[str, Any], *, model: str) -> ImageGenerationResult:
        data_items = payload.get("data")
        first = data_items[0] if isinstance(data_items, list) and data_items else None
        encoded = first.get("b64_json") if isinstance(first, dict) else None
        if not isinstance(encoded, str) or not encoded:
            raise ProviderError("The xAI image provider returned no image data.")
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProviderError("The xAI image provider returned invalid image data.") from exc
        if not image_bytes:
            raise ProviderError("The xAI image provider returned an empty image.")
        usage = payload.get("usage")
        return ImageGenerationResult(
            data=image_bytes,
            mime_type=(first.get("mime_type") if isinstance(first, dict) else None) or "image/jpeg",
            model=model,
            provider="xai",
            revised_prompt=first.get("revised_prompt") if isinstance(first, dict) else None,
            usage=usage if isinstance(usage, dict) else {},
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
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json",
            "quality": quality if quality in {"low", "medium", "auto"} else "medium",
        }
        if aspect_ratio := self._aspect_ratio(size):
            body["aspect_ratio"] = aspect_ratio
        endpoint = "images/generations"
        if reference_images:
            endpoint = "images/edits"
            images = [
                {
                    "url": (
                        f"data:{image.mime_type};base64,"
                        f"{base64.b64encode(image.data).decode('ascii')}"
                    ),
                    "type": "image_url",
                }
            for image in reference_images
            ]
            if len(images) == 1:
                body["image"] = images[0]
            else:
                body["images"] = images
        try:
            async with httpx.AsyncClient(
                base_url=f"{self._base_url}/", headers=self._headers, timeout=self._timeout
            ) as client:
                response = await client.post(endpoint, json=body)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("xai_image_generation_failed", exc_info=True)
            raise ProviderError("xAI image generation failed.") from exc
        if not isinstance(payload, dict):
            raise ProviderError("The xAI image provider returned an invalid response.")
        return self._decode_response(payload, model=model)
