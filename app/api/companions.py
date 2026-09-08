"""Public, non-sensitive companion catalogue endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query

from app.core.config import Settings, get_settings
from app.schemas.companion import CompanionDetail, CompanionSummary
from app.services.companion_service import CompanionService

router = APIRouter(tags=["Companions"])


def get_companion_service(settings: Settings = Depends(get_settings)) -> CompanionService:
    return CompanionService(settings)


@router.get(
    "/companions",
    response_model=list[CompanionSummary],
    summary="List active companions",
    description="Returns public companion profiles from the companion catalogue service.",
)
async def list_companions(
    service: CompanionService = Depends(get_companion_service),
    sort_order: str | None = Query(default=None, alias="sortOrder"),
    sort_by: str | None = Query(default=None, alias="sortBy"),
    limit: int | None = Query(default=None, ge=1),
    page: int | None = Query(default=None, ge=1),
    personality_trait: str | None = Query(default=None, alias="personalityTrait"),
    interest: str | None = None,
    status: bool | None = None,
    location: str | None = None,
    profession: str | None = None,
    search_term: str | None = Query(default=None, alias="searchTerm"),
) -> list[CompanionSummary]:
    filters = {
        key: value
        for key, value in {
            "sortOrder": sort_order,
            "sortBy": sort_by,
            "limit": limit,
            "page": page,
            "personalityTrait": personality_trait,
            "interest": interest,
            "status": status,
            "location": location,
            "profession": profession,
            "searchTerm": search_term,
        }.items()
        if value is not None
    }
    return await service.list_active_companions(filters)


@router.get(
    "/companions/{companion_id}",
    response_model=CompanionDetail,
    summary="Get a companion's full profile",
    description="Returns one public companion profile from the companion catalogue service.",
    responses={404: {"description": "Companion not found"}},
)
async def get_companion(
    companion_id: uuid.UUID,
    service: CompanionService = Depends(get_companion_service),
) -> CompanionDetail:
    return await service.get_companion_detail(companion_id)
