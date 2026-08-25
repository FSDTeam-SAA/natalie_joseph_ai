"""
Health check endpoints.

GET /health        -> basic liveness, no dependency checks
GET /health/ready   -> readiness, checks DB + Redis + config presence

Never exposes secrets or internal configuration values.
"""

from __future__ import annotations

import redis.asyncio as redis_asyncio
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db_session

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Basic liveness check")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness check (DB, Redis, config)")
async def health_ready(
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

    # Redis check
    try:
        client = redis_asyncio.from_url(settings.REDIS_URL)
        await client.ping()
        await client.aclose()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {type(exc).__name__}"

    # Configuration presence (booleans only — never expose actual secret values)
    checks["openai_api_key_configured"] = "ok" if settings.OPENAI_API_KEY else "missing"
    checks["auth_mode"] = settings.AUTH_MODE.value

    overall = "ok" if all(v == "ok" for k, v in checks.items() if k in {"database", "redis"}) else "degraded"

    return {"status": overall, "checks": checks}
