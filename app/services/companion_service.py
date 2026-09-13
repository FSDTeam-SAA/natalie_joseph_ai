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

    @staticmethod
    def _section(profile: dict, key: str) -> dict:
        value = profile.get(key)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _strings(value: object) -> list[str]:
        return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []

    @classmethod
    def _communication(cls, profile: dict) -> tuple[list[str], list[str]]:
        """Support both the live structured style and the legacy flat style."""
        style = profile.get("communicationStyle")
        if isinstance(style, dict):
            return (
                cls._strings(style.get("styleTraits")),
                cls._strings(style.get("whatYouExperience")),
            )
        return ([style] if isinstance(style, str) else [], [])

    @staticmethod
    def _reference_image_urls(profile: dict) -> list[str]:
        """Collect every usable remote image supplied by the companion profile."""
        raw_profile_image = profile.get("profileImage")
        profile_image = (
            raw_profile_image.strip()
            if isinstance(raw_profile_image, str) and raw_profile_image.strip()
            else None
        )
        raw_gallery_images = profile.get("galleryImages", [])
        gallery_images = raw_gallery_images if isinstance(raw_gallery_images, list) else []

        images: list[str] = []
        for image in ([profile_image] if profile_image else []) + gallery_images:
            if isinstance(image, str) and image.strip() and image.strip() not in images:
                images.append(image.strip())
        return images

    @classmethod
    def _to_summary(cls, profile: dict) -> CompanionSummary:
        personality = cls._section(profile, "personality")
        background = cls._section(profile, "background")
        return CompanionSummary(
            id=profile["id"],
            slug=cls._slug(profile),
            name=profile["name"],
            title=profile.get("title"),
            traits=cls._strings(personality.get("traits", profile.get("traits"))),
            location=background.get("location", profile.get("location")),
            occupation=background.get("occupation", profile.get("profession")),
        )

    @classmethod
    def _to_detail(cls, profile: dict) -> CompanionDetail:
        personality = cls._section(profile, "personality")
        background = cls._section(profile, "background")
        visual = cls._section(profile, "visualProfile")
        voice = cls._section(profile, "voice")
        style_traits, what_you_experience = cls._communication(profile)
        return CompanionDetail(
            id=profile["id"],
            slug=cls._slug(profile),
            name=profile["name"],
            title=profile.get("title"),
            about=personality.get("about", profile.get("bio")),
            essence=personality.get("essence", profile.get("backstory")),
            traits=cls._strings(personality.get("traits", profile.get("traits"))),
            location=background.get("location", profile.get("location")),
            occupation=background.get("occupation", profile.get("profession")),
            lifestyle=cls._strings(background.get("lifestyle", profile.get("lifestyle"))),
            communication_style_traits=style_traits,
            interests=cls._strings(profile.get("interests")),
            aesthetic_keywords=cls._strings(
                visual.get("aestheticKeywords", profile.get("aestheticKeywords"))
            ),
            what_you_experience=what_you_experience,
            voice_available=bool(
                isinstance(voice.get("voiceId"), str) and voice["voiceId"].strip()
            ),
            image_available=bool(profile.get("profileImage")),
        )

    def _to_profile(self, profile: dict) -> CompanionProfile:
        try:
            companion_id = uuid.UUID(str(profile["id"]))
            name = str(profile["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("The companion catalogue returned invalid data.") from exc
        personality = self._section(profile, "personality")
        background = self._section(profile, "background")
        visual = self._section(profile, "visualProfile")
        voice = self._section(profile, "voice")
        style_traits, what_you_experience = self._communication(profile)
        adult_interaction = self._section(profile, "adultInteraction")
        if not adult_interaction:
            adult_interaction = self._section(personality, "adultInteraction")
        slug = self._slug(profile)
        # The companion catalogue is the source of truth for character
        # identity. Its profile image is used first, followed by gallery images;
        # local deployment assets must not override these remote references.
        reference_images = self._reference_image_urls(profile)
        return CompanionProfile(
            id=companion_id,
            slug=slug,
            name=name,
            active=bool(profile.get("status", False)),
            version=1,
            personality_config={
                "title": profile.get("title"),
                "about": personality.get("about", profile.get("bio")),
                "essence": personality.get("essence", profile.get("backstory")),
                "traits": self._strings(personality.get("traits", profile.get("traits"))),
            },
            communication_config={
                "style_traits": style_traits,
                "what_you_experience": what_you_experience,
                "adult_interaction": adult_interaction,
            },
            background_config={
                "location": background.get("location", profile.get("location")),
                "lifestyle": self._strings(background.get("lifestyle", profile.get("lifestyle"))),
            },
            interest_config={"interests": self._strings(profile.get("interests"))},
            visual_config={
                "reference_images": reference_images,
                "reference_image": profile.get("profileImage"),
                "aesthetic_keywords": self._strings(
                    visual.get("aestheticKeywords", profile.get("aestheticKeywords"))
                ),
                "physical_identity": visual.get("physicalIdentity", ""),
                "generation_instructions": visual.get("generationInstructions", ""),
            },
            voice_config={
                "voice_id": voice.get("voiceId"),
                "settings": voice.get("settings")
                if isinstance(voice.get("settings"), dict)
                else {},
                "description": self._strings(profile.get("voiceDescription")),
            },
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
        try:
            return self._to_detail(profile)
        except (KeyError, TypeError, ValueError) as exc:
            # Catalogue data is external input.  Missing required fields or a
            # schema mismatch must remain an upstream error rather than escape
            # FastAPI's error boundary as a generic 500.
            raise ProviderError("The companion catalogue returned invalid data.") from exc

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
