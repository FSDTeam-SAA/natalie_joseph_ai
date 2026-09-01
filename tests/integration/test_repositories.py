"""
Integration tests for the Phase 2 data layer.

These run against a real Postgres+pgvector database (DATABASE_URL from
the environment) — not mocks — because the whole point of Phase 2 is
correct schema + working queries, including vector similarity search
and FK/ownership behavior that a mocked session can't validate.
"""

from __future__ import annotations

import random
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.companion import Companion
from app.db.models.conversation import Conversation
from app.db.models.memory import MemoryType
from app.db.models.message import MessageRole
from app.repositories.companion_repository import CompanionRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.relationship_repository import RelationshipRepository
from app.repositories.user_repository import UserRepository


def _fake_embedding(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(1536)]


async def test_user_get_or_create_is_idempotent(db: AsyncSession) -> None:
    repo = UserRepository(db)
    external_id = uuid.uuid4()

    first = await repo.get_or_create_by_external_user_id(external_id, locale="en-US")
    second = await repo.get_or_create_by_external_user_id(external_id, locale="en-US")

    assert first.id == second.id


async def test_companion_slug_lookup(db: AsyncSession) -> None:
    repo = CompanionRepository(db)
    companion = Companion(
        slug=f"test-elena-{uuid.uuid4().hex[:8]}",
        name="Elena",
        version=1,
        personality_config={},
        communication_config={},
        background_config={},
        interest_config={},
        visual_config={},
        active=True,
    )
    await repo.add(companion)

    found = await repo.get_by_slug(companion.slug)
    assert found is not None
    assert found.id == companion.id


async def test_conversation_ownership_isolation(db: AsyncSession) -> None:
    user_repo = UserRepository(db)
    companion_repo = CompanionRepository(db)
    conv_repo = ConversationRepository(db)

    owner = await user_repo.get_or_create_by_external_user_id(uuid.uuid4())
    intruder = await user_repo.get_or_create_by_external_user_id(uuid.uuid4())
    companion = await companion_repo.add(
        Companion(
            slug=f"test-luna-{uuid.uuid4().hex[:8]}",
            name="Luna",
            version=1,
            personality_config={},
            communication_config={},
            background_config={},
            interest_config={},
            visual_config={},
        )
    )

    conversation = await conv_repo.add(
        Conversation(user_id=owner.id, companion_id=companion.id)
    )

    # Owner can fetch it.
    found = await conv_repo.get_by_id_for_user(conversation.id, owner.id)
    assert found is not None

    # A different user cannot — this is the isolation guarantee from
    # spec Section 57, enforced at the query layer.
    not_found = await conv_repo.get_by_id_for_user(conversation.id, intruder.id)
    assert not_found is None


async def test_message_recent_limit_and_ordering(db: AsyncSession) -> None:
    user_repo = UserRepository(db)
    companion_repo = CompanionRepository(db)
    conv_repo = ConversationRepository(db)
    msg_repo = MessageRepository(db)

    user = await user_repo.get_or_create_by_external_user_id(uuid.uuid4())
    companion = await companion_repo.add(
        Companion(
            slug=f"test-chloe-{uuid.uuid4().hex[:8]}",
            name="Chloe",
            version=1,
            personality_config={},
            communication_config={},
            background_config={},
            interest_config={},
            visual_config={},
        )
    )
    conversation = await conv_repo.add(Conversation(user_id=user.id, companion_id=companion.id))

    for i in range(5):
        from app.db.models.message import Message

        await msg_repo.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.user if i % 2 == 0 else MessageRole.assistant,
                content=f"message {i}",
            )
        )

    recent = await msg_repo.get_recent_for_conversation(conversation.id, limit=3)
    assert len(recent) == 3
    # Oldest-first ordering, and it's the LAST 3 of the 5 inserted.
    assert [m.content for m in recent] == ["message 2", "message 3", "message 4"]


async def test_memory_vector_search_respects_user_isolation(db: AsyncSession) -> None:
    user_repo = UserRepository(db)
    companion_repo = CompanionRepository(db)
    memory_repo = MemoryRepository(db)

    from app.db.models.memory import Memory

    user_a = await user_repo.get_or_create_by_external_user_id(uuid.uuid4())
    user_b = await user_repo.get_or_create_by_external_user_id(uuid.uuid4())
    companion = await companion_repo.add(
        Companion(
            slug=f"test-thalia-{uuid.uuid4().hex[:8]}",
            name="Thalia",
            version=1,
            personality_config={},
            communication_config={},
            background_config={},
            interest_config={},
            visual_config={},
        )
    )

    query_vec = _fake_embedding(seed=1)

    # User A has a memory very close to the query vector.
    await memory_repo.add(
        Memory(
            user_id=user_a.id,
            companion_id=companion.id,
            memory_type=MemoryType.preference,
            key="favorite_cuisine",
            value="Japanese food",
            embedding=query_vec,
            confidence=0.95,
            active=True,
        )
    )
    # User B has a memory with the SAME embedding — if isolation were
    # broken, this would leak into user A's search results.
    await memory_repo.add(
        Memory(
            user_id=user_b.id,
            companion_id=companion.id,
            memory_type=MemoryType.preference,
            key="favorite_cuisine",
            value="Italian food",
            embedding=query_vec,
            confidence=0.95,
            active=True,
        )
    )

    results = await memory_repo.search_similar(
        user_id=user_a.id, query_embedding=query_vec, top_k=5
    )

    assert len(results) == 1
    assert results[0].value == "Japanese food"
    assert all(r.user_id == user_a.id for r in results)


async def test_memory_deactivate_excludes_from_search(db: AsyncSession) -> None:
    user_repo = UserRepository(db)
    companion_repo = CompanionRepository(db)
    memory_repo = MemoryRepository(db)

    from app.db.models.memory import Memory

    user = await user_repo.get_or_create_by_external_user_id(uuid.uuid4())
    companion = await companion_repo.add(
        Companion(
            slug=f"test-anastacia-{uuid.uuid4().hex[:8]}",
            name="Anastacia",
            version=1,
            personality_config={},
            communication_config={},
            background_config={},
            interest_config={},
            visual_config={},
        )
    )
    vec = _fake_embedding(seed=2)

    memory = await memory_repo.add(
        Memory(
            user_id=user.id,
            companion_id=companion.id,
            memory_type=MemoryType.fact,
            key="old_preference",
            value="Used to like Paris",
            embedding=vec,
            confidence=0.9,
            active=True,
        )
    )

    await memory_repo.deactivate(memory)

    results = await memory_repo.search_similar(user_id=user.id, query_embedding=vec, top_k=5)
    assert results == []


async def test_relationship_context_get_or_create(db: AsyncSession) -> None:
    user_repo = UserRepository(db)
    companion_repo = CompanionRepository(db)
    rel_repo = RelationshipRepository(db)

    user = await user_repo.get_or_create_by_external_user_id(uuid.uuid4())
    companion = await companion_repo.add(
        Companion(
            slug=f"test-luna2-{uuid.uuid4().hex[:8]}",
            name="Luna",
            version=1,
            personality_config={},
            communication_config={},
            background_config={},
            interest_config={},
            visual_config={},
        )
    )

    first = await rel_repo.get_or_create(user.id, companion.id)
    second = await rel_repo.get_or_create(user.id, companion.id)

    assert first.id == second.id
    assert first.familiarity_level.value == "new"


async def test_message_ordering_survives_identical_timestamps(db: AsyncSession) -> None:
    """
    Regression test for a real bug found during Phase 5 manual testing
    against a hosted Postgres instance: a user message and its
    assistant reply, inserted in the same transaction, sometimes
    received identical (or ambiguously ordered) created_at timestamps,
    causing ORDER BY created_at to return them in the wrong order.

    This test deliberately forces identical created_at values —
    reproducing the exact failure condition — and confirms ordering by
    `sequence` still returns correct chronological order regardless.
    """
    from datetime import datetime, timezone

    from app.db.models.companion import Companion
    from app.db.models.conversation import Conversation
    from app.db.models.message import Message, MessageRole

    user_repo = UserRepository(db)
    companion_repo = CompanionRepository(db)
    conv_repo = ConversationRepository(db)
    msg_repo = MessageRepository(db)

    user = await user_repo.get_or_create_by_external_user_id(uuid.uuid4())
    companion = await companion_repo.add(
        Companion(
            slug=f"test-seqfix-{uuid.uuid4().hex[:8]}",
            name="SeqFixTest",
            version=1,
            personality_config={},
            communication_config={},
            background_config={},
            interest_config={},
            visual_config={},
        )
    )
    conversation = await conv_repo.add(Conversation(user_id=user.id, companion_id=companion.id))

    identical_timestamp = datetime.now(timezone.utc)

    # Insert assistant FIRST but with the exact same timestamp as the
    # user message inserted second — if ordering relied on created_at
    # alone, this would very plausibly come back in the wrong order,
    # which is exactly the bug this test guards against.
    assistant_msg = Message(
        conversation_id=conversation.id,
        role=MessageRole.assistant,
        content="assistant reply",
        created_at=identical_timestamp,
    )
    await msg_repo.add(assistant_msg)

    user_msg = Message(
        conversation_id=conversation.id,
        role=MessageRole.user,
        content="user message",
        created_at=identical_timestamp,
    )
    await msg_repo.add(user_msg)

    ordered = await msg_repo.get_recent_for_conversation(conversation.id, limit=10)

    assert len(ordered) == 2
    # Correct order is INSERTION order (user's turn happened first in
    # this test's intent — assistant was added first here specifically
    # to prove sequence, not insertion coincidence, drives the result).
    # What actually matters: ordering is deterministic and matches
    # `sequence`, not accidentally correct due to timestamp luck.
    assert ordered[0].sequence < ordered[1].sequence
    assert ordered[0].id == assistant_msg.id
    assert ordered[1].id == user_msg.id