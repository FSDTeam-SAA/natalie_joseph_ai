from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredObject:
    key: str
    byte_size: int
    sha256: str


class MediaStorage(ABC):
    @abstractmethod
    async def put(self, data: bytes, *, mime_type: str) -> StoredObject:
        raise NotImplementedError

    @abstractmethod
    async def get(self, key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        raise NotImplementedError
