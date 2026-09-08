"""
Unit tests for Phase 7's MemoryService, fully isolated from the
database and any real provider — MemoryRepository, EmbeddingProvider,
and LLMProvider are all mocks/fakes here. End-to-end behavior against
a real DB + real HTTP flow is covered separately in
tests/integration/test_memory_extraction_e2e.py.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.db.models.memory import MemoryType
from app.llm.base import LLMUsage
from app.services.memory_service import MemoryService, _cosine_similarity


def _fake_settings(**overrides) -> SimpleNamespace:
    defaults = dict(
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
        OPENAI_BACKGROUND_MODEL="gpt-5.6-luna",
        MEMORY_TOP_K=5,
        MEMORY_MIN_SCORE=0.75,
        MEMORY_MIN_CONFIDENCE=0.6,
        PROMPT_VERSION="test-v1",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_memory(*, key: str, value: str, embedding: list[float]) -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), key=key, value=value, embedding=embedding)


def _make_service(**overrides) -> tuple[MemoryService, dict]:
    mocks = dict(
        memory_repo=AsyncMock(),
        embedding_provider=AsyncMock(),
        llm_provider=AsyncMock(),
    )
    mocks["memory_repo"].session = SimpleNamespace(
        add=Mock(),
        commit=AsyncMock(),
        flush=AsyncMock(),
    )
    mocks.update(overrides)
    service = MemoryService(settings=_fake_settings(), **mocks)
    return service, mocks


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_does_not_raise(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestRetrieveRelevant:
    async def test_filters_by_min_score_and_formats_output(self) -> None:
        service, mocks = _make_service()
        query_vector = [1.0, 0.0, 0.0]
        mocks["embedding_provider"].embed_text.return_value = SimpleNamespace(
            vector=query_vector, model="text-embedding-3-small", dimensions=3
        )
        close_memory = _fake_memory(key="pet_name", value="Rex", embedding=[1.0, 0.0, 0.0])
        far_memory = _fake_memory(key="unrelated", value="noise", embedding=[0.0, 1.0, 0.0])
        mocks["memory_repo"].search_similar.return_value = [close_memory, far_memory]

        result = await service.retrieve_relevant(
            user_id=uuid.uuid4(), companion_id=uuid.uuid4(), query_text="tell me about my dog"
        )

        assert result == ["pet_name: Rex"]

    async def test_search_scoped_to_user_and_companion(self) -> None:
        service, mocks = _make_service()
        mocks["embedding_provider"].embed_text.return_value = SimpleNamespace(
            vector=[1.0, 0.0], model="x", dimensions=2
        )
        mocks["memory_repo"].search_similar.return_value = []
        user_id, companion_id = uuid.uuid4(), uuid.uuid4()

        await service.retrieve_relevant(user_id=user_id, companion_id=companion_id, query_text="hi")

        _, kwargs = mocks["memory_repo"].search_similar.call_args
        assert kwargs["user_id"] == user_id
        assert kwargs["companion_id"] == companion_id
        assert kwargs["top_k"] == 5
        assert kwargs["min_confidence"] == 0.6

    async def test_embedding_failure_degrades_to_empty_list(self) -> None:
        service, mocks = _make_service()
        mocks["embedding_provider"].embed_text.side_effect = RuntimeError("network down")

        result = await service.retrieve_relevant(
            user_id=uuid.uuid4(), companion_id=uuid.uuid4(), query_text="hi"
        )
        assert result == []

    async def test_search_failure_degrades_to_empty_list(self) -> None:
        service, mocks = _make_service()
        mocks["embedding_provider"].embed_text.return_value = SimpleNamespace(
            vector=[1.0, 0.0], model="x", dimensions=2
        )
        mocks["memory_repo"].search_similar.side_effect = RuntimeError("db down")

        result = await service.retrieve_relevant(
            user_id=uuid.uuid4(), companion_id=uuid.uuid4(), query_text="hi"
        )
        assert result == []


class TestExtractAndStore:
    async def test_inserts_new_memory_when_key_not_found(self) -> None:
        service, mocks = _make_service()
        mocks["llm_provider"].generate_structured.return_value = SimpleNamespace(
            data={
                "memories": [
                    {
                        "memory_type": "fact",
                        "key": "pet_name",
                        "value": "User has a dog named Rex",
                        "confidence": 0.9,
                    }
                ]
            },
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            model="gpt-5.6-luna",
            raw_response_id="resp_1",
        )
        mocks["embedding_provider"].embed_text.return_value = SimpleNamespace(
            vector=[0.1] * 3, model="text-embedding-3-small", dimensions=3
        )
        mocks["memory_repo"].find_existing_by_key.return_value = None

        await service.extract_and_store(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            companion_id=uuid.uuid4(),
            user_message="My dog's name is Rex.",
            assistant_message="Rex sounds adorable!",
        )

        mocks["memory_repo"].add.assert_awaited_once()
        inserted = mocks["memory_repo"].add.call_args[0][0]
        assert inserted.key == "pet_name"
        assert inserted.value == "User has a dog named Rex"
        assert inserted.memory_type == MemoryType.fact
        assert inserted.confidence == 0.9
        assert inserted.privacy_class is None
        mocks["memory_repo"].session.commit.assert_awaited_once()

    async def test_updates_existing_memory_in_place_rather_than_duplicating(self) -> None:
        service, mocks = _make_service()
        mocks["llm_provider"].generate_structured.return_value = SimpleNamespace(
            data={
                "memories": [
                    {
                        "memory_type": "fact",
                        "key": "pet_name",
                        "value": "User's dog Rex is now 2 years old",
                        "confidence": 0.95,
                    }
                ]
            },
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            model="gpt-5.6-luna",
            raw_response_id="resp_2",
        )
        mocks["embedding_provider"].embed_text.return_value = SimpleNamespace(
            vector=[0.2] * 3, model="text-embedding-3-small", dimensions=3
        )
        existing = SimpleNamespace(
            key="pet_name", value="old value", memory_type=MemoryType.fact,
            confidence=0.5, embedding=[0.0] * 3, active=False,
        )
        mocks["memory_repo"].find_existing_by_key.return_value = existing

        await service.extract_and_store(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            companion_id=uuid.uuid4(),
            user_message="Rex just turned 2.",
            assistant_message="Happy birthday Rex!",
        )

        mocks["memory_repo"].add.assert_not_awaited()
        assert existing.value == "User's dog Rex is now 2 years old"
        assert existing.confidence == 0.95
        assert existing.active is True
        mocks["memory_repo"].acquire_key_lock.assert_awaited_once()
        mocks["memory_repo"].session.flush.assert_awaited_once()
        mocks["memory_repo"].session.commit.assert_awaited_once()

    async def test_empty_memories_list_is_a_no_op(self) -> None:
        service, mocks = _make_service()
        mocks["llm_provider"].generate_structured.return_value = SimpleNamespace(
            data={"memories": []},
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            model="gpt-5.6-luna",
            raw_response_id="resp_3",
        )

        await service.extract_and_store(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            companion_id=uuid.uuid4(),
            user_message="just saying hi",
            assistant_message="hi there!",
        )

        mocks["memory_repo"].add.assert_not_awaited()
        mocks["memory_repo"].session.commit.assert_awaited_once()

    async def test_malformed_llm_output_is_swallowed_not_raised(self) -> None:
        service, mocks = _make_service()
        mocks["llm_provider"].generate_structured.return_value = SimpleNamespace(
            data={"memories": [{"key": "missing_required_fields"}]},
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            model="gpt-5.6-luna",
            raw_response_id="resp_4",
        )

        # Must not raise -- extraction failures are never allowed to
        # propagate, since this always runs after the user's response
        # has already been sent.
        await service.extract_and_store(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            companion_id=uuid.uuid4(),
            user_message="hi",
            assistant_message="hi",
        )
        mocks["memory_repo"].add.assert_not_awaited()

    async def test_llm_provider_error_is_swallowed_not_raised(self) -> None:
        service, mocks = _make_service()
        mocks["llm_provider"].generate_structured.side_effect = RuntimeError("api down")

        await service.extract_and_store(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            companion_id=uuid.uuid4(),
            user_message="hi",
            assistant_message="hi",
        )
        mocks["memory_repo"].add.assert_not_awaited()

    async def test_embedding_failure_during_upsert_skips_that_candidate(self) -> None:
        service, mocks = _make_service()
        mocks["llm_provider"].generate_structured.return_value = SimpleNamespace(
            data={
                "memories": [
                    {
                        "memory_type": "fact",
                        "key": "pet_name",
                        "value": "User has a dog named Rex",
                        "confidence": 0.9,
                    }
                ]
            },
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            model="gpt-5.6-luna",
            raw_response_id="resp_5",
        )
        mocks["embedding_provider"].embed_text.side_effect = RuntimeError("embedding api down")

        await service.extract_and_store(
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            companion_id=uuid.uuid4(),
            user_message="My dog's name is Rex.",
            assistant_message="Rex sounds adorable!",
        )

        mocks["memory_repo"].add.assert_not_awaited()
        mocks["memory_repo"].find_existing_by_key.assert_not_awaited()
