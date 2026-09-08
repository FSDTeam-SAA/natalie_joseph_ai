"""Filesystem media storage for development and single-instance deployments."""

from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from pathlib import Path

from app.core.exceptions import NotFoundError, ProviderError, ValidationError
from app.storage.base import MediaStorage, StoredObject

_EXTENSIONS = {
    "audio/basic": ".au",
    "audio/l16": ".pcm",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_KEY_PATTERN = re.compile(r"^[0-9a-f]{32}\.(?:au|pcm|mp3|m4a|ogg|wav|jpg|png|webp)$")


class LocalMediaStorage(MediaStorage):
    def __init__(self, root: str | Path, *, max_bytes: int) -> None:
        self._root = Path(root).resolve()
        self._max_bytes = max_bytes

    def _path_for_key(self, key: str) -> Path:
        if not _KEY_PATTERN.fullmatch(key):
            raise ValidationError("Invalid media storage key.")
        path = (self._root / key).resolve()
        if path.parent != self._root:
            raise ValidationError("Invalid media storage path.")
        return path

    async def put(self, data: bytes, *, mime_type: str) -> StoredObject:
        extension = _EXTENSIONS.get(mime_type)
        if extension is None:
            raise ValidationError(f"Unsupported generated media type: {mime_type}.")
        if not data:
            raise ProviderError("The media provider returned an empty file.")
        if len(data) > self._max_bytes:
            raise ProviderError("The generated media file exceeds the configured size limit.")

        key = f"{uuid.uuid4().hex}{extension}"
        path = self._path_for_key(key)
        digest = hashlib.sha256(data).hexdigest()

        def _write() -> None:
            self._root.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(data)

        try:
            await asyncio.to_thread(_write)
        except OSError as exc:
            raise ProviderError("Media storage is temporarily unavailable.") from exc
        return StoredObject(key=key, byte_size=len(data), sha256=digest)

    async def get(self, key: str) -> bytes:
        path = self._path_for_key(key)
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise NotFoundError("Media file not found.") from exc
        except OSError as exc:
            raise ProviderError("Media storage is temporarily unavailable.") from exc
        if len(data) > self._max_bytes:
            raise ProviderError("Stored media exceeds the configured size limit.")
        return data

    async def delete(self, key: str) -> None:
        path = self._path_for_key(key)

        def _delete() -> None:
            path.unlink(missing_ok=True)

        try:
            await asyncio.to_thread(_delete)
        except OSError as exc:
            raise ProviderError("Media storage cleanup failed.") from exc
