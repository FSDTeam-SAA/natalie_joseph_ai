"""Shared bounded multipart-audio reader for public and backend adapters."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import UploadFile

from app.core.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class AudioUpload:
    data: bytes
    mime_type: str
    filename: str


async def read_audio_upload(audio: UploadFile, *, max_bytes: int) -> AudioUpload:
    """Read at most one byte over the limit and always close the spool file."""
    mime_type = (audio.content_type or "").partition(";")[0].strip().lower()
    filename = audio.filename or "audio"
    try:
        data = await audio.read(max_bytes + 1)
    finally:
        await audio.close()
    if len(data) > max_bytes:
        raise ValidationError("Audio upload exceeds the configured size limit.")
    return AudioUpload(data=data, mime_type=mime_type, filename=filename)
