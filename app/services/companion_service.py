"""
Companion catalogue client.

The main companion backend owns public companion profiles. This service
adapts that API's response to this application's established frontend schema;
it deliberately does not read the local companions table.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import httpx

from app.core.config import Settings
from app.core.exceptions import NotFoundError, ProviderError
from app.schemas.companion import CompanionDetail, CompanionSummary


@dataclass(frozen=True, slots=True)
class CompanionProfile:
    """A companion profile loaded from the authoritative remote catalogue."""

    id: uuid.UUID
    slug: str
    name: str
    active: bool
    version: int
    personality_config: dict
    communication_config: dict
    background_config: dict
    interest_config: dict
    visual_config: dict
    voice_config: dict


class CompanionService:
    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        self.transport = transport

    @staticmethod
    def _slug(profile: dict) -> str:
        # The upstream API has no stable slug. Its UUID remains the canonical
        # identifier; this is retained only for compatibility with existing UI.
        return "-".join(str(profile["name"]).lower().split())

    @classmethod
    def _to_summary(cls, profile: dict) -> CompanionSummary:
        return CompanionSummary(
            id=profile["id"],
            slug=cls._slug(profile),
            name=profile["name"],
            title=profile.get("title"),
            traits=profile.get("traits", []),
            location=profile.get("location"),
            occupation=profile.get("profession"),
        )

    @classmethod
    def _to_detail(cls, profile: dict) -> CompanionDetail:
        return CompanionDetail(
            id=profile["id"],
            slug=cls._slug(profile),
            name=profile["name"],
            title=profile.get("title"),
            about=profile.get("bio"),
            essence=profile.get("backstory"),
            traits=profile.get("traits", []),
            location=profile.get("location"),
            occupation=profile.get("profession"),
            lifestyle=[profile["lifestyle"]] if profile.get("lifestyle") else [],
            communication_style_traits=(
                [profile["communicationStyle"]] if profile.get("communicationStyle") else []
            ),
            interests=profile.get("interests", []),
            voice_available=bool(profile.get("voiceDescription")),
            image_available=bool(profile.get("profileImage")),
        )

    def _to_profile(self, profile: dict) -> CompanionProfile:
        try:
            companion_id = uuid.UUID(str(profile["id"]))
            name = str(profile["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("The companion catalogue returned invalid data.") from exc
        lifestyle = profile.get("lifestyle")
        communication_style = profile.get("communicationStyle")
        return CompanionProfile(
            id=companion_id,
            slug=self._slug(profile),
            name=name,
            active=bool(profile.get("status", False)),
            version=1,
            personality_config={
                "title": profile.get("title"),
                "about": profile.get("bio"),
                "essence": profile.get("backstory"),
                "traits": profile.get("traits", []),
            },
            communication_config={
                "style_traits": [communication_style] if communication_style else [],
                "what_you_experience": [],
            },
            background_config={
                "location": profile.get("location"),
                "lifestyle": [lifestyle] if lifestyle else [],
            },
            interest_config={"interests": profile.get("interests", [])},
            visual_config={
                "reference_images": profile.get("galleryImages", []),
                "reference_image": profile.get("profileImage"),
                "aesthetic_keywords": [],
            },
            voice_config={"description": profile.get("voiceDescription", [])},
        )

    async def _get(self, path: str, params: dict[str, str | int | bool] | None = None) -> dict:
        url = f"{self.settings.COMPANION_CATALOGUE_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.COMPANION_CATALOGUE_TIMEOUT_SECONDS,
                transport=self.transport,
            ) as client:
                response = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise ProviderError("The companion catalogue is currently unavailable.") from exc

        if response.status_code == 404:
            raise NotFoundError("Companion not found.")
        if response.is_error:
            raise ProviderError("The companion catalogue returned an error.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("The companion catalogue returned invalid data.") from exc
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ProviderError("The companion catalogue returned an invalid response.")
        return payload

    async def list_active_companions(
        self, filters: dict[str, str | int | bool]
    ) -> list[CompanionSummary]:
        payload = await self._get("companions", filters)
        profiles = payload.get("data")
        if not isinstance(profiles, list):
            raise ProviderError("The companion catalogue returned invalid data.")
        return [self._to_summary(profile) for profile in profiles]

    async def get_companion_detail(self, companion_id: uuid.UUID) -> CompanionDetail:
        payload = await self._get(f"companions/{companion_id}")
        profile = payload.get("data")
        if not isinstance(profile, dict):
            raise ProviderError("The companion catalogue returned invalid data.")
        return self._to_detail(profile)

    async def get_active_profile(self, companion_id: uuid.UUID) -> CompanionProfile:
        payload = await self._get(f"companions/{companion_id}")
        profile = payload.get("data")
        if not isinstance(profile, dict):
            raise ProviderError("The companion catalogue returned invalid data.")
        companion = self._to_profile(profile)
        if not companion.active:
            raise NotFoundError(f"Companion {companion_id} is inactive.")
        return companion

    async def get_active_profile_by_reference(self, reference: str) -> CompanionProfile:
        """Resolve an external UUID or the UI-compatible slug from the catalogue."""
        try:
            return await self.get_active_profile(uuid.UUID(reference))
        except ValueError:
            pass
        payload = await self._get("companions")
        profiles = payload.get("data")
        if not isinstance(profiles, list):
            raise ProviderError("The companion catalogue returned invalid data.")
        for profile in profiles:
            if isinstance(profile, dict) and self._slug(profile) == reference.strip().lower():
                companion = self._to_profile(profile)
                if companion.active:
                    return companion
                break
        raise NotFoundError("Companion not found.")
