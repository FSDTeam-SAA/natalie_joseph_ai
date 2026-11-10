"""Bounded live-provider smoke test using synthetic content, without application DB writes.

This makes up to three reviewed text requests, one image, one short TTS and its STT.
Provider usage may be billed. No customer credits, Telegram messages or media uploads.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.images.xai_images import XAIImageProvider
from app.llm.base import LLMMessage
from app.llm.openai_provider import OpenAIProvider
from app.llm.xai_provider import XAIProvider
from app.services.routine_response_guard import RoutineResponseGuard
from app.services.image_service import _validate_generated_image
from app.services.companion_service import CompanionService
from app.voice.elevenlabs_provider import ElevenLabsVoiceProvider
from tests.unit.test_routine_response_guard import _context

logging.disable(logging.CRITICAL)

async def main():
    settings = Settings()
    results = []

    async def measure(label, operation):
        started = time.perf_counter()
        try:
            details = await operation()
            row = {"test": label, "ok": True, "elapsed_ms": round((time.perf_counter() - started) * 1000), **details}
        except Exception as error:
            cause = error.__cause__ or error
            row = {"test": label, "ok": False, "elapsed_ms": round((time.perf_counter() - started) * 1000), "error_type": type(error).__name__, "provider_status": getattr(cause, "status_code", None)}
        results.append(row)
        print(json.dumps(row), flush=True)

    async def texts():
        provider = XAIProvider(settings.XAI_API_KEY, base_url=settings.XAI_BASE_URL, timeout_seconds=40, max_retries=0)
        reviewer = OpenAIProvider(settings.OPENAI_API_KEY, timeout_seconds=30, max_retries=0)
        guard = RoutineResponseGuard(reviewer=reviewer, reviewer_model=settings.OPENAI_BACKGROUND_MODEL)
        for index in range(3):
            async def generate():
                context = _context()
                question = "What are you doing now? Reply in one short English sentence."
                answer = await guard.generate(provider, [LLMMessage(role="system", content="You are a fictional adult companion. The current fictional routine is Beach club, in Dubai in the afternoon. Stay consistent with it."), LLMMessage(role="user", content=question)], context=context, user_message=question, model=settings.XAI_MODEL, max_output_tokens=128, cache_key="synthetic-smoke", reasoning_effort=settings.XAI_REASONING_EFFORT)
                return {"reply": answer.result.text, "routine_validation": answer.metadata.get("routine_validation")}
            await measure(f"reviewed_text_{index + 1}", generate)
        await provider._client.close()
        await reviewer._client.close()

    async def image():
        provider = XAIImageProvider(api_key=settings.XAI_API_KEY, base_url=settings.XAI_BASE_URL, timeout_seconds=60, max_retries=0)
        result = await provider.generate(prompt="A ceramic coffee cup on an empty sunny cafe table, realistic still life, no people, no text.", model=settings.XAI_IMAGE_MODEL, reference_images=[], size="1024x1024", quality="medium")
        _validate_generated_image(result.data, result.mime_type)
        return {"bytes": len(result.data), "mime_type": result.mime_type, "scope": "provider generation and application signature validation only; no companion reference, routine or storage"}

    async def voice():
        voices = list(settings.COMPANION_VOICE_ID_MAP.values())
        if not voices:
            catalogue = CompanionService(settings)
            summaries = await catalogue.list_active_companions({})
            for summary in summaries:
                profile = await catalogue.get_active_profile(summary.id)
                if profile.voice_config.get("voice_id"):
                    voices.append(profile.voice_config["voice_id"])
                    break
        if not voices:
            raise RuntimeError("No configured voice in map or companion catalogue")
        async with ElevenLabsVoiceProvider(settings.ELEVENLABS_API_KEY, base_url=settings.ELEVENLABS_BASE_URL, timeout_seconds=30, max_retries=0) as provider:
            started = time.perf_counter()
            result = await provider.synthesize("Hello, this is a short voice connection test.", voice_id=voices[0], model=settings.ELEVENLABS_TTS_MODEL, output_format=settings.ELEVENLABS_OUTPUT_FORMAT)
            synthesis_ms = round((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            transcript = await provider.transcribe(result.audio, mime_type=result.mime_type, model=settings.ELEVENLABS_STT_MODEL, filename="smoke.mp3")
            return {"synthesis_ms": synthesis_ms, "transcription_ms": round((time.perf_counter() - started) * 1000), "bytes": len(result.audio), "transcript": transcript.text}

    if "--voice-only" in sys.argv:
        await measure("voice_provider_roundtrip", voice)
    else:
        await asyncio.gather(texts(), measure("image_provider", image), measure("voice_provider_roundtrip", voice))
    report = {"tested_at_utc": datetime.now(UTC).isoformat(), "scope": "Live providers with synthetic inputs. Not end-to-end, no Telegram delivery, application database, media storage or user billing.", "results": results}
    target = Path("../VOICE_SMOKE_RESULTS.json" if "--voice-only" in sys.argv else "../PROVIDER_SMOKE_RESULTS.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")

if __name__ == "__main__":
    asyncio.run(main())
