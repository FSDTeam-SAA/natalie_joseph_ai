"""EmbeddingProvider interface (spec Section 6 / 36)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model: str
    dimensions: int


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_text(self, text: str, *, model: str) -> EmbeddingResult:
        raise NotImplementedError