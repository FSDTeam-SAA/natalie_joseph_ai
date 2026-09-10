from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReferenceImage:
    data: bytes
    filename: str
    mime_type: str


@dataclass(frozen=True)
class ImageGenerationResult:
    data: bytes
    mime_type: str
    model: str
    provider: str = "openai"
    revised_prompt: str | None = None
    usage: dict[str, int | float | str] = field(default_factory=dict)


class ImageProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        reference_images: list[ReferenceImage],
        size: str,
        quality: str,
    ) -> ImageGenerationResult:
        raise NotImplementedError
