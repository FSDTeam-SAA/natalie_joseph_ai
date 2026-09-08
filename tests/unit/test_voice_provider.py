"""Focused unit tests for the ElevenLabs REST provider boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.exceptions import ProviderError
from app.voice.elevenlabs_provider import ElevenLabsVoiceProvider


def _response(
    *,
    status_code: int = 200,
    content: bytes = b"",
    json: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    url: str = "https://voice.test/v1/result",
) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=content if json is None else None,
        json=json,
        headers=headers,
        request=httpx.Request("POST", url),
    )


class TestElevenLabsTranscription:
    async def test_transcribe_posts_multipart_and_returns_metadata(self) -> None:
        response = _response(
            json={"text": " Hello there ", "language_code": "en", "language_probability": 0.98},
            headers={"request-id": "req_stt", "x-trace-id": "trace_stt"},
        )

        async with ElevenLabsVoiceProvider(
            api_key="test-key", base_url="https://voice.test", timeout_seconds=5
        ) as provider:
            with patch.object(
                provider._client, "post", new=AsyncMock(return_value=response)
            ) as mock_post:
                result = await provider.transcribe(
                    b"encoded audio",
                    filename="message.webm",
                    mime_type="audio/webm",
                    model="scribe_v2",
                )

        assert result.text == "Hello there"
        assert result.model == "scribe_v2"
        assert result.language_code == "en"
        assert result.language_probability == 0.98
        assert result.usage.request_id == "req_stt"
        assert result.usage.trace_id == "trace_stt"

        assert mock_post.call_args.args == ("/v1/speech-to-text",)
        assert mock_post.call_args.kwargs["data"] == {"model_id": "scribe_v2"}
        assert mock_post.call_args.kwargs["files"] == {
            "file": ("message.webm", b"encoded audio", "audio/webm")
        }

    async def test_transcribe_rejects_an_empty_provider_result(self) -> None:
        response = _response(json={"text": "   "})

        async with ElevenLabsVoiceProvider("test-key", "https://voice.test") as provider:
            with patch.object(provider._client, "post", new=AsyncMock(return_value=response)):
                with pytest.raises(ProviderError, match="empty transcript"):
                    await provider.transcribe(
                        b"encoded audio", mime_type="audio/ogg", model="scribe_v2"
                    )

    async def test_transcribe_validates_audio_before_request(self) -> None:
        async with ElevenLabsVoiceProvider("test-key", "https://voice.test") as provider:
            with patch.object(provider._client, "post", new=AsyncMock()) as mock_post:
                with pytest.raises(ValueError, match="audio must be non-empty bytes"):
                    await provider.transcribe(b"", mime_type="audio/ogg", model="scribe_v2")

        mock_post.assert_not_awaited()


class TestElevenLabsSynthesis:
    async def test_synthesize_posts_json_and_returns_audio_usage(self) -> None:
        response = _response(
            content=b"fake mp3 bytes",
            headers={
                "content-type": "audio/mpeg; charset=binary",
                "character-cost": "14",
                "request-id": "req_tts",
                "x-trace-id": "trace_tts",
            },
        )

        async with ElevenLabsVoiceProvider("test-key", "https://voice.test") as provider:
            with patch.object(
                provider._client, "post", new=AsyncMock(return_value=response)
            ) as mock_post:
                result = await provider.synthesize(
                    "Hello there",
                    voice_id="voice/id",
                    model="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )

        assert result.audio == b"fake mp3 bytes"
        assert result.audio_bytes == b"fake mp3 bytes"
        assert result.mime_type == "audio/mpeg"
        assert result.model == "eleven_multilingual_v2"
        assert result.usage.character_cost == 14
        assert result.usage.request_id == "req_tts"
        assert result.usage.trace_id == "trace_tts"

        assert mock_post.call_args.args == ("/v1/text-to-speech/voice%2Fid",)
        assert mock_post.call_args.kwargs["params"] == {"output_format": "mp3_44100_128"}
        assert mock_post.call_args.kwargs["json"] == {
            "text": "Hello there",
            "model_id": "eleven_multilingual_v2",
        }

    async def test_synthesize_infers_mime_type_when_header_is_absent(self) -> None:
        response = _response(content=b"wave bytes")

        async with ElevenLabsVoiceProvider("test-key", "https://voice.test") as provider:
            with patch.object(provider._client, "post", new=AsyncMock(return_value=response)):
                result = await provider.synthesize(
                    "Hello", voice_id="voice", model="model", output_format="wav_44100"
                )

        assert result.mime_type == "audio/wav"

    async def test_synthesize_uses_requested_pcm_type_for_generic_binary_header(self) -> None:
        response = _response(
            content=b"raw pcm samples",
            headers={"content-type": "application/octet-stream"},
        )

        async with ElevenLabsVoiceProvider("test-key", "https://voice.test") as provider:
            with patch.object(provider._client, "post", new=AsyncMock(return_value=response)):
                result = await provider.synthesize(
                    "Hello", voice_id="voice", model="model", output_format="pcm_44100"
                )

        assert result.mime_type == "audio/l16"

    async def test_synthesize_rejects_empty_audio(self) -> None:
        response = _response(headers={"content-type": "audio/mpeg"})

        async with ElevenLabsVoiceProvider("test-key", "https://voice.test") as provider:
            with patch.object(provider._client, "post", new=AsyncMock(return_value=response)):
                with pytest.raises(ProviderError, match="empty audio"):
                    await provider.synthesize("Hello", voice_id="voice", model="model")


class TestElevenLabsErrors:
    async def test_http_error_is_sanitized(self) -> None:
        response = _response(
            status_code=401,
            json={"detail": "secret provider response and test-key"},
        )

        async with ElevenLabsVoiceProvider("test-key", "https://voice.test") as provider:
            with patch.object(provider._client, "post", new=AsyncMock(return_value=response)):
                with pytest.raises(ProviderError) as exc_info:
                    await provider.synthesize("Hello", voice_id="voice", model="model")

        assert str(exc_info.value) == "ElevenLabs text-to-speech request failed"
        assert "secret" not in str(exc_info.value)
        assert "test-key" not in str(exc_info.value)

    async def test_transport_error_is_sanitized(self) -> None:
        request = httpx.Request("POST", "https://voice.test/v1/speech-to-text")
        transport_error = httpx.ReadTimeout(
            "provider timed out with sensitive data", request=request
        )

        async with ElevenLabsVoiceProvider("test-key", "https://voice.test") as provider:
            with patch.object(provider._client, "post", new=AsyncMock(side_effect=transport_error)):
                with pytest.raises(ProviderError) as exc_info:
                    await provider.transcribe(b"audio", mime_type="audio/wav", model="scribe_v2")

        assert str(exc_info.value) == "ElevenLabs speech-to-text request failed"
        assert "sensitive" not in str(exc_info.value)


def test_constructor_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="api_key"):
        ElevenLabsVoiceProvider(" ")

    with pytest.raises(ValueError, match="timeout_seconds"):
        ElevenLabsVoiceProvider("test-key", timeout_seconds=0)
