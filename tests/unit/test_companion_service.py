"""Tests for companion-catalogue response handling."""

from __future__ import annotations

import uuid
from datetime import datetime

import httpx
import pytest

from app.api.companions import get_companion_service
from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.main import app
from app.services.companion_service import CompanionService
from app.services.routine_service import resolve_current_activity

LIVE_PROFILE = {
    "id": "2ace0895-0137-4fa9-bce0-b9892258a82c",
    "name": "Elena",
    "title": "The Social & Magnetic Companion",
    "status": True,
    "interests": ["Luxury Travel"],
    "profileImage": "https://example.test/elena.jpg",
    "personality": {
        "traits": ["Social", "Playful"],
        "about": "Elena is charismatic.",
        "essence": "Elena represents confidence.",
    },
    "communicationStyle": {
        "styleTraits": ["Confident and feminine"],
        "whatYouExperience": ["Exciting conversations"],
    },
    "background": {
        "location": "Dubai, United Arab Emirates",
        "occupation": "Luxury Real Estate Advisor",
        "lifestyle": ["Enjoys a vibrant lifestyle"],
    },
    "adultInteraction": {
        "style": "playful and confident",
        "dynamic": "teasing",
        "initiative": "occasionally takes the lead",
    },
    "visualProfile": {
        "aestheticKeywords": ["Dubai sunsets"],
        "physicalIdentity": "A fictional adult companion with a distinctive elegant look.",
        "generationInstructions": "Use a cinematic, intimate evening atmosphere.",
    },
    "voice": {
        "provider": "elevenlabs",
        "voiceId": "elena-voice-id",
        "settings": {"stability": 0.5},
    },
}


@pytest.mark.asyncio
async def test_detail_with_malformed_catalogue_profile_is_provider_error() -> None:
    """External profile schema errors must not surface as an API 500."""

    service = CompanionService(
        Settings(COMPANION_CATALOGUE_BASE_URL="https://catalogue.test/api/v1"),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"success": True, "data": {}})
        ),
    )

    with pytest.raises(ProviderError, match="catalogue returned invalid data"):
        await service.get_companion_detail(uuid.uuid4())


@pytest.mark.asyncio
async def test_detail_endpoint_returns_502_for_malformed_catalogue_profile() -> None:
    service = CompanionService(
        Settings(COMPANION_CATALOGUE_BASE_URL="https://catalogue.test/api/v1"),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"success": True, "data": {}})
        ),
    )
    app.dependency_overrides[get_companion_service] = lambda: service

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get(f"/api/v1/companions/{uuid.uuid4()}")
    finally:
        app.dependency_overrides.pop(get_companion_service, None)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "PROVIDER_ERROR"


@pytest.mark.asyncio
async def test_detail_maps_live_catalogue_profile_shape() -> None:
    service = CompanionService(
        Settings(COMPANION_CATALOGUE_BASE_URL="https://catalogue.test/api/v1"),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"success": True, "data": LIVE_PROFILE})
        ),
    )

    detail = await service.get_companion_detail(uuid.UUID(LIVE_PROFILE["id"]))

    assert detail.name == "Elena"
    assert detail.traits == ["Social", "Playful"]
    assert detail.location == "Dubai, United Arab Emirates"
    assert detail.occupation == "Luxury Real Estate Advisor"
    assert detail.communication_style_traits == ["Confident and feminine"]
    assert detail.aesthetic_keywords == ["Dubai sunsets"]
    assert detail.voice_available is True

    profile = service._to_profile(LIVE_PROFILE)
    assert profile.voice_config == {
        "voice_id": "elena-voice-id",
        "settings": {"stability": 0.5},
        "description": [],
    }
    assert profile.communication_config["adult_interaction"]["dynamic"] == "teasing"
    assert profile.visual_config["physical_identity"].startswith("A fictional adult")
    assert profile.visual_config["generation_instructions"].startswith("Use a cinematic")


@pytest.mark.parametrize("name", ["Elena", "Chloé", "Thalia", "Lina", "Luna"])
async def test_admin_routine_updates_and_removal_are_loaded_for_each_companion(name) -> None:
    companion_id = uuid.uuid4()
    routine = {
        "timezone": "Europe/London",
        "weekly": [{"day": 1, "start": "12:00", "end": "18:00", "activity": "Café"}],
    }
    profile = dict(LIVE_PROFILE, id=str(companion_id), name=name, routine=routine, version=3)
    profile["personality"] = dict(profile["personality"], sharedTraits=["Curious"])
    profile["communicationStyle"] = dict(profile["communicationStyle"], topicsSheEnjoys="Art")
    paths = []

    def handle(request):
        paths.append(request.url.path)
        # Slug resolution must not mistake a summary card for the full profile.
        data = (
            [{"id": str(companion_id), "name": name}]
            if request.url.path.endswith("/companions")
            else profile
        )
        return httpx.Response(200, json={"success": True, "data": data})

    service = CompanionService(
        Settings(_env_file=None, COMPANION_CATALOGUE_BASE_URL="https://catalogue.test/api/v1"),
        transport=httpx.MockTransport(handle),
    )
    current = await service.get_active_profile_by_reference(name.lower())
    now = datetime.fromisoformat("2026-11-02T14:00:00+00:00")
    assert resolve_current_activity(current.routine, now=now).activity == "Café"
    assert current.version == 3
    assert current.background_config["occupation"] == "Luxury Real Estate Advisor"
    assert current.personality_config["shared_traits"] == ["Curious"]
    assert current.communication_config["topics_she_enjoys"] == "Art"
    assert paths == ["/api/v1/companions", f"/api/v1/companions/{companion_id}"]

    routine["weekly"][0]["activity"] = "Gym"
    updated = await service.get_active_profile(companion_id)
    assert resolve_current_activity(updated.routine, now=now).activity == "Gym"
    # Already loaded profiles remain independent of subsequent backend edits.
    assert resolve_current_activity(current.routine, now=now).activity == "Café"
    profile["routine"] = None
    assert (await service.get_active_profile(companion_id)).routine is None


@pytest.mark.parametrize("routine", [None, {}, {"timezone": "Invalid/Zone", "weekly": []}])
def test_optional_bad_routine_does_not_break_the_profile(routine) -> None:
    service = CompanionService(Settings(_env_file=None))
    profile = service._to_profile(dict(LIVE_PROFILE, routine=routine))
    assert profile.name == "Elena"
    assert profile.routine is None
