"""
Seed the five companions from config/companions/*.json.

Per spec Section 43: "The seed script must load JSON configurations.
Do not duplicate character definitions in multiple files." This
script is the only place that reads the JSON files and writes
Companion rows — character content itself lives only in
config/companions/*.json.

Idempotent: running this multiple times updates existing companions
(matched by slug) rather than creating duplicates, so it's safe to
re-run after editing a JSON config file.

Usage:
    python -m app.scripts.seed_companions
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.db.models.companion import Companion
from app.db.session import AsyncSessionLocal
from app.repositories.companion_repository import CompanionRepository

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "companions"

# Per spec Section 2: the five companion IDs must be stable.
# NOTE: "anastacia" was renamed to "lina" per product owner decision
# after initial spec delivery — the stable set below reflects that
# rename, not the original spec document's literal wording.
EXPECTED_SLUGS = {"elena", "chloe", "thalia", "lina", "luna"}


def _load_companion_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_config_blocks(raw: dict) -> dict:
    """
    Map a companion JSON file's structure onto the five JSONB columns
    defined on the Companion model. This is the single place that
    translates config/companions/*.json into the DB schema — if the
    JSON file structure changes, only this function needs to change.
    """
    personality = raw.get("personality", {})
    communication = raw.get("communication_style", {})

    personality_config = {
        "title": raw.get("title"),
        "title_variant_note": raw.get("title_variant_note"),
        "traits": personality.get("traits", []),
        "about": personality.get("about"),
        "essence": personality.get("essence"),
        "shared_traits": personality.get("shared_traits", []),
    }
    communication_config = {
        "style_traits": communication.get("style_traits", []),
        "topics_she_enjoys": communication.get("topics_she_enjoys"),
        "what_you_experience": communication.get("what_you_experience", []),
    }
    background_config = raw.get("background", {})
    interest_config = {"interests": raw.get("interests", [])}
    visual_config = raw.get("visual_profile", {})

    return {
        "personality_config": personality_config,
        "communication_config": communication_config,
        "background_config": background_config,
        "interest_config": interest_config,
        "visual_config": visual_config,
    }


async def seed_companions() -> None:
    json_files = sorted(CONFIG_DIR.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No companion config files found in {CONFIG_DIR}")

    found_slugs = set()

    async with AsyncSessionLocal() as session:
        repo = CompanionRepository(session)

        for path in json_files:
            raw = _load_companion_json(path)
            slug = raw["id"]
            found_slugs.add(slug)

            config_blocks = _build_config_blocks(raw)
            existing = await repo.get_by_slug(slug)

            if existing is not None:
                existing.name = raw["name"]
                existing.version = raw.get("version", existing.version)
                existing.personality_config = config_blocks["personality_config"]
                existing.communication_config = config_blocks["communication_config"]
                existing.background_config = config_blocks["background_config"]
                existing.interest_config = config_blocks["interest_config"]
                existing.visual_config = config_blocks["visual_config"]
                existing.active = True
                print(f"Updated companion: {slug}")
            else:
                companion = Companion(
                    slug=slug,
                    name=raw["name"],
                    version=raw.get("version", 1),
                    active=True,
                    **config_blocks,
                )
                await repo.add(companion)
                print(f"Created companion: {slug}")

        await session.commit()

    missing = EXPECTED_SLUGS - found_slugs
    if missing:
        print(
            f"WARNING: expected companion slugs not found in {CONFIG_DIR}: "
            f"{sorted(missing)}"
        )
    unexpected = found_slugs - EXPECTED_SLUGS
    if unexpected:
        print(
            f"WARNING: unexpected companion slugs found (not in the stable "
            f"five per spec Section 2): {sorted(unexpected)}"
        )

    print(f"Seeding complete. {len(found_slugs)} companion(s) processed.")


if __name__ == "__main__":
    asyncio.run(seed_companions())