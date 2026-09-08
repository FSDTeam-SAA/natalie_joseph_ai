"""Provider-neutral speech transcription and synthesis contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VoiceUsage:
    """Optional billing and request metadata exposed by a voice provider."""

    character_cost: int | None = None
    request_id: str | None = None
    trace_id: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Text produced by speech-to-text."""

    text: str
    model: str
    usage: VoiceUsage
    language_code: str | None = None
    language_probability: float | None = None


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Audio produced by text-to-speech."""

    audio: bytes
    mime_type: str
    model: str
    usage: VoiceUsage

    @property
    def audio_bytes(self) -> bytes:
        """Explicit alias for callers that name byte payloads with a suffix."""

        return self.audio


class VoiceProvider(ABC):
    """Provider-independent interface for user audio and companion speech."""

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str,
        model: str,
        filename: str = "audio",
    ) -> TranscriptionResult:
        """Transcribe one non-empty encoded audio payload."""

        raise NotImplementedError

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        model: str,
        output_format: str = "mp3_44100_128",
        voice_settings: dict[str, Any] | None = None,
    ) -> SynthesisResult:
        """Synthesize one non-empty text payload with the selected voice."""

        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """Release resources held by the provider."""

        raise NotImplementedError
