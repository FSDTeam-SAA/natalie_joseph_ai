"""Validate the backend context contract and apply only restrictive output grants."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import ValidationError
from app.core.security import AuthContext


class LifestyleAccess(BaseModel):
    model_config = ConfigDict(strict=True)
    trial: bool
    allowed_output_types: list[Literal["text", "image", "audio"]] = Field(min_length=1)
    available_credits: int = Field(ge=0)
    prices: dict[str, int]
    billing_mode: Literal["legacy", "response"]
    library_reuse: bool = False


class LifestyleContext(BaseModel):
    model_config = ConfigDict(strict=True)
    version: Literal["lifestyle-v1"]
    access: LifestyleAccess


def apply_lifestyle_context(auth: AuthContext, raw: str) -> AuthContext:
    if len(raw) > 32768:
        raise ValidationError("The lifestyle context is too large.")
    data = json.loads(raw)
    context = LifestyleContext.model_validate(data)
    outputs = set(context.access.allowed_output_types)
    if context.access.trial:
        outputs &= {"text"}
    allowed = {"chat"} if "text" in outputs else set()
    if "image" in outputs:
        allowed.add("image")
    if "audio" in outputs:
        allowed.update({"voice_input", "voice_output"})
    # Form data can narrow an authenticated grant, never create a new grant.
    features = frozenset(feature for feature in allowed if auth.allows(feature))
    # Companion facts and this caller's delivery context; all values remain data.
    grounding = {
        key: data[key]
        for key in (
            "timezone",
            "localDate",
            "localTime",
            "currentActivity",
            "todayEvent",
            "recentEvents",
            "dailyContent",
            "recentSharedPhotos",
        )
        if key in data
    }
    return replace(
        auth,
        features=features,
        raw_claims={
            **auth.raw_claims,
            "lifestyle_context": grounding,
            "library_reuse": context.access.library_reuse,
        },
    )
