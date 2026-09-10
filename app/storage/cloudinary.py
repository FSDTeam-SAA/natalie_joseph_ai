"""Cloudinary public-media storage adapter."""

from __future__ import annotations

import hashlib
import time
import uuid

import httpx

from app.core.exceptions import NotFoundError, ProviderError, ValidationError
from app.storage.base import MediaStorage, StoredObject


class CloudinaryMediaStorage(MediaStorage):
    """Store generated media as permanent public Cloudinary delivery URLs."""

    def __init__(
        self, *, cloud_name: str, api_key: str, api_secret: str, folder: str, max_bytes: int
    ) -> None:
        if not cloud_name or not api_key or not api_secret:
            raise ValueError("Cloudinary cloud name, API key, and API secret are required.")
        self.cloud_name = cloud_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.folder = folder.strip("/")
        self.max_bytes = max_bytes

    @staticmethod
    def _resource_type(mime_type: str) -> str:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("audio/"):
            return "video"  # Cloudinary delivers audio through its video resource type.
        raise ValidationError(f"Unsupported generated media type: {mime_type}.")

    def _signature(self, params: dict[str, str]) -> str:
        encoded = "&".join(f"{key}={params[key]}" for key in sorted(params))
        return hashlib.sha1(f"{encoded}{self.api_secret}".encode()).hexdigest()

    async def put(self, data: bytes, *, mime_type: str) -> StoredObject:
        if not data:
            raise ProviderError("The media provider returned an empty file.")
        if len(data) > self.max_bytes:
            raise ProviderError("The generated media file exceeds the configured size limit.")
        resource_type = self._resource_type(mime_type)
        timestamp = str(int(time.time()))
        public_id = f"{self.folder}/{uuid.uuid4().hex}" if self.folder else uuid.uuid4().hex
        signed = {"public_id": public_id, "timestamp": timestamp}
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"https://api.cloudinary.com/v1_1/{self.cloud_name}/{resource_type}/upload",
                    data={
                        **signed,
                        "api_key": self.api_key,
                        "signature": self._signature(signed),
                    },
                    files={"file": ("generated-media", data, mime_type)},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Cloudinary media upload failed.") from exc
        url = payload.get("secure_url")
        returned_public_id = payload.get("public_id")
        version = payload.get("version")
        if not isinstance(url, str) or not isinstance(returned_public_id, str) or not version:
            raise ProviderError("Cloudinary returned an invalid media upload response.")
        # Preserve enough information to retrieve/delete the object later.
        key = f"{resource_type}:{returned_public_id}:{version}"
        return StoredObject(
            key=key,
            byte_size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            public_url=url,
        )

    def _url_for_key(self, key: str) -> str:
        try:
            resource_type, public_id, version = key.split(":", 2)
        except ValueError as exc:
            raise ValidationError("Invalid Cloudinary media key.") from exc
        if resource_type not in {"image", "video"} or not public_id or not version.isdigit():
            raise ValidationError("Invalid Cloudinary media key.")
        return f"https://res.cloudinary.com/{self.cloud_name}/{resource_type}/upload/v{version}/{public_id}"

    async def get(self, key: str) -> bytes:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(self._url_for_key(key))
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise NotFoundError("Media file not found.") from exc
            raise ProviderError("Cloudinary media retrieval failed.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("Cloudinary media retrieval failed.") from exc
        data = bytes(response.content)
        if len(data) > self.max_bytes:
            raise ProviderError("Stored media exceeds the configured size limit.")
        return data

    async def delete(self, key: str) -> None:
        try:
            resource_type, public_id, _version = key.split(":", 2)
            timestamp = str(int(time.time()))
            signed = {"public_id": public_id, "timestamp": timestamp}
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"https://api.cloudinary.com/v1_1/{self.cloud_name}/{resource_type}/destroy",
                    data={**signed, "api_key": self.api_key, "signature": self._signature(signed)},
                )
                response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Cloudinary media cleanup failed.") from exc
