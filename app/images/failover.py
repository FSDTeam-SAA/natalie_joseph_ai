"""Fail over image generation from xAI to OpenAI when necessary."""

from __future__ import annotations

import logging

from app.core.exceptions import ProviderError
from app.images.base import ImageGenerationResult, ImageProvider, ReferenceImage

logger = logging.getLogger(__name__)


class FailoverImageProvider(ImageProvider):
    """Prefer xAI, with OpenAI retained as an independent fallback."""

    def __init__(
        self,
        *,
        primary: ImageProvider | None,
        fallback: ImageProvider | None,
        fallback_model: str,
    ) -> None:
        if primary is None and fallback is None:
            raise ValueError("At least one image provider is required.")
        self._primary = primary
        self._fallback = fallback
        self._fallback_model = fallback_model

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        reference_images: list[ReferenceImage],
        size: str,
        quality: str,
    ) -> ImageGenerationResult:
        if self._primary is not None:
            try:
                return await self._primary.generate(
                    prompt=prompt,
                    model=model,
                    reference_images=reference_images,
                    size=size,
                    quality=quality,
                )
            except ProviderError:
                if self._fallback is None:
                    raise
                logger.warning("xai_image_generation_failed_using_openai_fallback")
        assert self._fallback is not None
        return await self._fallback.generate(
            prompt=prompt,
            model=self._fallback_model,
            reference_images=reference_images,
            size=size,
            quality=quality,
        )
