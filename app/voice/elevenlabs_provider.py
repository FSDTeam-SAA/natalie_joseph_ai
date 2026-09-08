"""ElevenLabs voice provider implemented against the async HTTP REST API."""

from __future__ import annotations

import math
from types import TracebackType
from typing import Any
from urllib.parse import quote

import httpx

from app.core.exceptions import ProviderError
from app.voice.base import SynthesisResult, TranscriptionResult, VoiceProvider, VoiceUsage

_DEFAULT_BASE_URL = "https://api.elevenlabs.io"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

_OUTPUT_MIME_TYPES = {
    "alaw": "audio/basic",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "pcm": "audio/l16",
    "ulaw": "audio/basic",
    "wav": "audio/wav",
}


class ElevenLabsVoiceProvider(VoiceProvider):
    """Speech-to-text and text-to-speech through ElevenLabs."""

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 2,
    ) -> None:
        api_key = _required_string(api_key, "api_key")
        base_url = _required_string(base_url, "base_url").rstrip("/")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        try:
            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers={"xi-api-key": api_key},
                timeout=timeout_seconds,
                transport=httpx.AsyncHTTPTransport(retries=max_retries),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("base_url must be a valid HTTP(S) URL") from exc

    async def __aenter__(self) -> ElevenLabsVoiceProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str,
        model: str,
        filename: str = "audio",
    ) -> TranscriptionResult:
        if not isinstance(audio, bytes) or not audio:
            raise ValueError("audio must be non-empty bytes")
        mime_type = _required_string(mime_type, "mime_type")
        model = _required_string(model, "model")
        filename = _required_string(filename, "filename")

        try:
            response = await self._client.post(
                "/v1/speech-to-text",
                data={"model_id": model},
                files={"file": (filename, audio, mime_type)},
            )
            response.raise_for_status()
        except httpx.HTTPError:
            raise ProviderError("ElevenLabs speech-to-text request failed") from None

        try:
            payload = response.json()
        except ValueError:
            raise ProviderError("ElevenLabs speech-to-text returned an invalid response") from None

        if not isinstance(payload, dict):
            raise ProviderError("ElevenLabs speech-to-text returned an invalid response")

        transcript = payload.get("text")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ProviderError("ElevenLabs speech-to-text returned an empty transcript")

        language_code = payload.get("language_code")
        language_probability = payload.get("language_probability")

        return TranscriptionResult(
            text=transcript.strip(),
            model=model,
            usage=_usage_from_headers(response.headers),
            language_code=language_code if isinstance(language_code, str) else None,
            language_probability=(
                float(language_probability)
                if isinstance(language_probability, int | float)
                and not isinstance(language_probability, bool)
                else None
            ),
        )

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        model: str,
        output_format: str = _DEFAULT_OUTPUT_FORMAT,
        voice_settings: dict[str, Any] | None = None,
    ) -> SynthesisResult:
        text = _required_string(text, "text")
        voice_id = _required_string(voice_id, "voice_id")
        model = _required_string(model, "model")
        output_format = _required_string(output_format, "output_format")

        encoded_voice_id = quote(voice_id, safe="")
        try:
            body: dict[str, Any] = {"text": text, "model_id": model}
            if voice_settings:
                body["voice_settings"] = voice_settings
            response = await self._client.post(
                f"/v1/text-to-speech/{encoded_voice_id}",
                params={"output_format": output_format},
                json=body,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            raise ProviderError("ElevenLabs text-to-speech request failed") from None

        audio = bytes(response.content)
        if not audio:
            raise ProviderError("ElevenLabs text-to-speech returned empty audio")

        mime_type = _response_mime_type(response.headers, output_format)
        if mime_type in {"application/json", "text/html", "text/plain"}:
            raise ProviderError("ElevenLabs text-to-speech returned an invalid response")

        return SynthesisResult(
            audio=audio,
            mime_type=mime_type,
            model=model,
            usage=_usage_from_headers(response.headers),
        )


def _required_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _usage_from_headers(headers: httpx.Headers) -> VoiceUsage:
    raw_character_cost = headers.get("character-cost")
    character_cost: int | None = None
    if raw_character_cost is not None:
        try:
            parsed_cost = int(raw_character_cost)
        except ValueError:
            pass
        else:
            if parsed_cost >= 0:
                character_cost = parsed_cost

    return VoiceUsage(
        character_cost=character_cost,
        request_id=headers.get("request-id"),
        trace_id=headers.get("x-trace-id"),
    )


def _response_mime_type(headers: httpx.Headers, output_format: str) -> str:
    format_family = output_format.partition("_")[0].lower()
    inferred = _OUTPUT_MIME_TYPES.get(format_family, "application/octet-stream")
    content_type = headers.get("content-type")
    if content_type:
        normalized = content_type.partition(";")[0].strip().lower()
        # Binary response endpoints commonly return this generic header. The
        # explicitly requested ElevenLabs output format is more informative.
        if normalized != "application/octet-stream":
            return normalized
    return inferred
