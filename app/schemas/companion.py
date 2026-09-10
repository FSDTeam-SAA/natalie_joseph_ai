"""
Companion API schemas.

Per spec Section 23: companion endpoints must return only
frontend-safe data. Explicitly excluded from these schemas:
  - hidden system prompt / prompt construction details
  - internal safety rules
  - provider configuration (model names, API details)
  - internal memory instructions

Only the character-facing content (identity, personality, background,
interests, visual aesthetic) is exposed — the same content a frontend
would show on a companion's profile card.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict


class CompanionSummary(BaseModel):
    """Used for GET /api/v1/companions — the list view."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    slug: str
    name: str
    title: str | None = None
    traits: list[str] = []
    location: str | None = None
    occupation: str | None = None


class CompanionDetail(BaseModel):
    """Used for GET /api/v1/companions/{id} — the full profile view."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    slug: str
    name: str
    title: str | None = None
    about: str | None = None
    essence: str | None = None
    traits: list[str] = []
    location: str | None = None
    occupation: str | None = None
    lifestyle: list[str] = []
    communication_style_traits: list[str] = []
    interests: list[str] = []
    aesthetic_keywords: list[str] = []
    what_you_experience: list[str] = []
    voice_available: bool = False
    image_available: bool = False
