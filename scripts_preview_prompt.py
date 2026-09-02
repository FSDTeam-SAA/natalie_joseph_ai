"""
Manual Phase 6 verification script — prints the ACTUAL assembled system
prompt for a real companion + real user, straight from your database.

This exists because the system prompt is intentionally never exposed via
the API (see app/schemas/chat.py's docstring — no system prompts, ever,
in any response). This script is the supported way to eyeball prompt
content directly without needing a live OpenAI call.

USAGE (PowerShell, from the project root, venv activated):
    python scripts_preview_prompt.py elena
    python scripts_preview_prompt.py lina --adult-eligible false

Run it after you've applied the elena.json rename and re-seeded.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.security import AuthContext
from app.llm.prompts.builder import PromptBuilder
from app.llm.prompts.context import PromptContext
from app.repositories.companion_repository import CompanionRepository
from app.repositories.relationship_repository import RelationshipRepository
from app.repositories.user_repository import UserRepository


async def main(slug: str, adult_eligible: bool) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with session_factory() as db:
        companion = await CompanionRepository(db).get_by_slug(slug)
        if companion is None:
            print(f"No companion found with slug '{slug}'. Did seeding run?")
            return

        user = await UserRepository(db).get_or_create_by_external_user_id(uuid.uuid4())
        relationship = await RelationshipRepository(db).get_or_create(user.id, companion.id)
        await db.commit()

        auth = AuthContext(
            user_id=user.id, adult_eligible=adult_eligible, entitled=False, raw_claims={}
        )
        context = PromptContext(
            companion=companion, auth=auth, user=user, relationship_context=relationship
        )

        builder = PromptBuilder.from_settings(settings)
        prompt = builder.build_system_prompt(context)

        print("=" * 80)
        print(f"Companion: {companion.name} ({companion.slug})")
        print(f"adult_eligible: {adult_eligible}")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        print(f"Prompt length: {len(prompt)} characters")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview a Phase 6 assembled system prompt.")
    parser.add_argument("slug", help="Companion slug, e.g. elena, chloe, thalia, lina, luna")
    parser.add_argument(
        "--adult-eligible",
        default="true",
        choices=["true", "false"],
        help="Simulate adult_eligible=true/false (default: true)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.slug, args.adult_eligible.lower() == "true"))