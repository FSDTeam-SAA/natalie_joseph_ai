"""
Health check endpoints.

GET /health        -> basic liveness, no dependency checks
GET /health/ready   -> readiness, checks DB + Redis + config presence

Never exposes secrets or internal configuration values.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import redis.asyncio as redis_asyncio
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models.companion import Companion
from app.db.session import get_db_session

router = APIRouter(tags=["Health"])
_EXPECTED_COMPANIONS = {"elena", "chloe", "thalia", "lina", "luna"}


def _expected_migration_head() -> str:
    config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    return ScriptDirectory.from_config(AlembicConfig(str(config_path))).get_current_head()


def _media_storage_check(settings: Settings) -> str:
    try:
        root = Path(settings.MEDIA_STORAGE_ROOT).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(prefix=".elysia-health-", dir=root):
            pass
    except OSError as exc:
        return f"error: {type(exc).__name__}"
    return "ok"


def _activation_checks(companions: list[Companion], settings: Settings) -> dict[str, str]:
    active_by_slug = {companion.slug: companion for companion in companions}
    checks = {
        "canonical_companions": (
            "ok" if set(active_by_slug) == _EXPECTED_COMPANIONS else "missing"
        )
    }

    if settings.ENABLE_VOICE_GENERATION:
        voice_ready = all(
            isinstance((companion.voice_config or {}).get("voice_id"), str)
            and bool((companion.voice_config or {}).get("voice_id", "").strip())
            for companion in active_by_slug.values()
        ) and set(active_by_slug) == _EXPECTED_COMPANIONS
        checks["companion_voice_configuration"] = "ok" if voice_ready else "missing"
    else:
        checks["companion_voice_configuration"] = "disabled"

    if settings.ENABLE_IMAGE_GENERATION:
        root = Path(settings.COMPANION_ASSET_ROOT).resolve()
        image_ready = set(active_by_slug) == _EXPECTED_COMPANIONS
        for companion in active_by_slug.values():
            references = (companion.visual_config or {}).get("reference_images")
            if not isinstance(references, list) or not references:
                image_ready = False
                break
            for relative in references:
                if not isinstance(relative, str):
                    image_ready = False
                    break
                candidate = (root / relative).resolve()
                if not candidate.is_relative_to(root) or not candidate.is_file():
                    image_ready = False
                    break
            if not image_ready:
                break
        checks["companion_reference_assets"] = "ok" if image_ready else "missing"
    else:
        checks["companion_reference_assets"] = "disabled"
    return checks


@router.get("/health", summary="Basic liveness check")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness check (DB, Redis, config)")
async def health_ready(
    response: Response,
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    checks: dict[str, str] = {}

    # Database check
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {type(exc).__name__}"

    if checks["database"] == "ok":
        try:
            revision_result = await db.execute(
                text("SELECT version_num FROM elysia.alembic_version")
            )
            current_revision = revision_result.scalar_one_or_none()
            checks["database_migrations"] = (
                "ok" if current_revision == _expected_migration_head() else "outdated"
            )
            result = await db.execute(select(Companion).where(Companion.active.is_(True)))
            checks.update(_activation_checks(list(result.scalars().all()), settings))
        except Exception as exc:  # noqa: BLE001
            checks["database_migrations"] = f"error: {type(exc).__name__}"
            checks["canonical_companions"] = f"error: {type(exc).__name__}"
    else:
        checks["database_migrations"] = "unavailable"
        checks["canonical_companions"] = "unavailable"

    # Redis check
    if settings.ENABLE_RATE_LIMITING:
        client = None
        try:
            client = redis_asyncio.from_url(settings.REDIS_URL)
            await client.ping()
            checks["redis"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["redis"] = f"error: {type(exc).__name__}"
        finally:
            if client is not None:
                await client.aclose()
    else:
        checks["redis"] = "disabled"

    # Configuration presence (booleans only — never expose actual secret values)
    checks["openai_api_key_configured"] = "ok" if settings.OPENAI_API_KEY else "missing"
    checks["xai_api_key_configured"] = "ok" if settings.XAI_API_KEY else "missing"
    if settings.ENABLE_VOICE_INPUT or settings.ENABLE_VOICE_GENERATION:
        checks["elevenlabs_api_key_configured"] = (
            "ok" if settings.ELEVENLABS_API_KEY else "missing"
        )
    else:
        checks["elevenlabs_api_key_configured"] = "disabled"

    auth_secret_configured = {
        "jwt": bool(settings.JWT_SECRET),
        "local_api_key": bool(settings.LOCAL_API_KEY),
        "internal_service_token": bool(settings.INTERNAL_SERVICE_TOKEN),
    }[settings.AUTH_MODE.value]
    checks["auth_mode"] = settings.AUTH_MODE.value
    checks["auth_credential_configured"] = "ok" if auth_secret_configured else "missing"
    media_enabled = settings.ENABLE_VOICE_GENERATION or settings.ENABLE_IMAGE_GENERATION
    checks["private_media_storage"] = (
        _media_storage_check(settings) if media_enabled else "disabled"
    )

    required = {
        "database",
        "database_migrations",
        "openai_api_key_configured",
        "xai_api_key_configured",
        "auth_credential_configured",
        "canonical_companions",
    }
    if settings.ENABLE_RATE_LIMITING:
        required.add("redis")
    if settings.ENABLE_VOICE_INPUT or settings.ENABLE_VOICE_GENERATION:
        required.add("elevenlabs_api_key_configured")
    if settings.ENABLE_VOICE_GENERATION:
        required.add("companion_voice_configuration")
    if settings.ENABLE_IMAGE_GENERATION:
        required.add("companion_reference_assets")
    if media_enabled:
        required.add("private_media_storage")
    overall = "ok" if all(checks[key] == "ok" for key in required) else "degraded"
    if overall != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"status": overall, "checks": checks}
