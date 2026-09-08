"""Focused unit tests for private local media storage."""

from __future__ import annotations

import hashlib
import re

import pytest

from app.core.exceptions import NotFoundError, ProviderError, ValidationError
from app.storage.local import LocalMediaStorage


class TestLocalMediaStorageValidation:
    @pytest.mark.parametrize(
        "key",
        [
            "../outside.png",
            "..\\outside.png",
            "/absolute.png",
            "subdirectory/asset.png",
            "0" * 32 + ".png/../outside.png",
            "not-a-private-key.png",
        ],
    )
    async def test_rejects_path_traversal_and_non_storage_keys(
        self, tmp_path, key: str
    ) -> None:
        storage = LocalMediaStorage(tmp_path, max_bytes=1024)

        with pytest.raises(ValidationError, match="Invalid media storage key"):
            await storage.get(key)
        with pytest.raises(ValidationError, match="Invalid media storage key"):
            await storage.delete(key)

    async def test_rejects_unsupported_media_type(self, tmp_path) -> None:
        storage = LocalMediaStorage(tmp_path, max_bytes=1024)

        with pytest.raises(ValidationError, match="Unsupported generated media type"):
            await storage.put(b"payload", mime_type="text/html")

    async def test_rejects_empty_and_oversized_files(self, tmp_path) -> None:
        storage = LocalMediaStorage(tmp_path, max_bytes=4)

        with pytest.raises(ProviderError, match="empty file"):
            await storage.put(b"", mime_type="image/png")
        with pytest.raises(ProviderError, match="exceeds the configured size limit"):
            await storage.put(b"12345", mime_type="image/png")

    async def test_rejects_an_oversized_file_when_reading(self, tmp_path) -> None:
        storage = LocalMediaStorage(tmp_path, max_bytes=4)
        key = "0" * 32 + ".png"
        (tmp_path / key).write_bytes(b"12345")

        with pytest.raises(ProviderError, match="Stored media exceeds"):
            await storage.get(key)


class TestLocalMediaStorageLifecycle:
    async def test_put_uses_private_random_key_then_gets_and_deletes(self, tmp_path) -> None:
        storage = LocalMediaStorage(tmp_path, max_bytes=1024)
        payload = b"private generated image bytes"

        first = await storage.put(payload, mime_type="image/png")
        second = await storage.put(payload, mime_type="image/png")

        assert re.fullmatch(r"[0-9a-f]{32}\.png", first.key)
        assert re.fullmatch(r"[0-9a-f]{32}\.png", second.key)
        assert first.key != second.key
        assert first.byte_size == len(payload)
        assert first.sha256 == hashlib.sha256(payload).hexdigest()
        assert await storage.get(first.key) == payload
        assert (tmp_path / first.key).is_file()

        await storage.delete(first.key)
        assert not (tmp_path / first.key).exists()
        with pytest.raises(NotFoundError, match="Media file not found"):
            await storage.get(first.key)

        # Deletion is deliberately idempotent so cleanup retries are safe.
        await storage.delete(first.key)

    @pytest.mark.parametrize(
        ("mime_type", "extension"),
        [
            ("audio/basic", ".au"),
            ("audio/l16", ".pcm"),
            ("audio/mpeg", ".mp3"),
            ("audio/mp4", ".m4a"),
            ("audio/ogg", ".ogg"),
            ("audio/wav", ".wav"),
            ("image/jpeg", ".jpg"),
            ("image/png", ".png"),
            ("image/webp", ".webp"),
        ],
    )
    async def test_supported_media_types_receive_private_extension(
        self, tmp_path, mime_type: str, extension: str
    ) -> None:
        storage = LocalMediaStorage(tmp_path, max_bytes=1024)

        stored = await storage.put(b"payload", mime_type=mime_type)

        assert stored.key.endswith(extension)
        assert re.fullmatch(r"[0-9a-f]{32}" + re.escape(extension), stored.key)
