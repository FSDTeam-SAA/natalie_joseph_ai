"""
Companion endpoints (spec Section 23).

GET /api/v1/companions            -> list of active companions (summary)
GET /api/v1/companions/{id}       -> full companion profile (detail)

No authentication dependency yet — companion data is not user-specific
and isn't sensitive (it's the public-facing character catalog), so
these are intentionally left open in Phase 3. Real auth is wired onto
user-specific endpoints starting Phase 5.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.repositories.companion_repository import CompanionRepository
from app.schemas.companion import CompanionDetail, CompanionSummary
from app.services.companion_service import CompanionService

router = APIRouter(tags=["Companions"])


def get_companion_service(db: AsyncSession = Depends(get_db_session)) -> CompanionService:
    return CompanionService(CompanionRepository(db))


@router.get(
    "/companions",
    response_model=list[CompanionSummary],
    summary="List active companions",
    description="Returns the public-facing summary for every active companion. "
    "Does not expose internal configuration, prompts, or safety rules.",
)
async def list_companions(
    service: CompanionService = Depends(get_companion_service),
) -> list[CompanionSummary]:
    return await service.list_active_companions()


@router.get(
    "/companions/{companion_id}",
    response_model=CompanionDetail,
    summary="Get a companion's full profile",
    description="Returns the public-facing detail profile for a single companion. "
    "Raises 404 if the companion does not exist or is inactive.",
    responses={404: {"description": "Companion not found"}},
)
async def get_companion(
    companion_id: uuid.UUID,
    service: CompanionService = Depends(get_companion_service),
) -> CompanionDetail:
    return await service.get_companion_detail(companion_id)