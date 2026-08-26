"""
OpenAI implementation of EmbeddingProvider.

Verified against openai-python 3.3.1's real SDK surface:
  client.embeddings.create(input=str, model=str) -> CreateEmbeddingResponse
  response.data[0].embedding -> list[float]

Dimension note: the memories.embedding column (Phase 2) is fixed at
1536 dimensions to match OPENAI_EMBEDDING_MODEL="text-embedding-3-small".
If that model config value is ever changed to a model producing a
different dimension, the database column and its ivfflat index must
be migrated accordingly — this provider does not enforce or validate
that; it fails naturally when pgvector rejects the mismatched size.
"""

from __future__ import annotations

from openai import AsyncOpenAI, APIError

from app.core.exceptions import ProviderError
from app.embeddings.base import EmbeddingProvider, EmbeddingResult


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)

    async def embed_text(self, text: str, *, model: str) -> EmbeddingResult:
        try:
            response = await self._client.embeddings.create(input=text, model=model)
        except APIError as exc:
            raise ProviderError(f"OpenAI embed_text() failed: {exc}") from exc

        embedding = response.data[0].embedding
        return EmbeddingResult(
            vector=embedding,
            model=response.model,
            dimensions=len(embedding),
        )