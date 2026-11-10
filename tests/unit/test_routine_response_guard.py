from __future__ import annotations

import json
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import ProviderError
from app.llm.base import LLMMessage, LLMResult, LLMStructuredResult, LLMUsage
from app.llm.prompts.context import PromptContext
from app.services.routine_response_guard import RoutineResponseGuard
from app.services.routine_service import parse_companion_routine, resolve_current_activity


def _context():
    routine = parse_companion_routine(
        {
            "timezone": "Asia/Dubai",
            "weekly": [{"day": 3, "start": "12:00", "end": "18:00", "activity": "Beach club"}],
        }
    )
    activity = resolve_current_activity(
        routine, now=datetime.fromisoformat("2026-11-04T09:05:23.840246+00:00")
    )
    return PromptContext(
        companion=SimpleNamespace(id=uuid.uuid4()), auth=None, user=None, routine_activity=activity
    )


def _result(text):
    return LLMResult(
        text=text,
        model="chat",
        usage=LLMUsage(
            input_tokens=20,
            output_tokens=10,
            cached_input_tokens=2,
            reasoning_tokens=3,
            cost_usd_ticks=100,
        ),
    )


def _review(consistent, reason="none"):
    return LLMStructuredResult(
        data={"consistent": consistent, "reason": reason},
        model="review",
        usage=LLMUsage(input_tokens=30, output_tokens=8),
    )


async def _generate(guard, provider, context, messages=None):
    return await guard.generate(
        provider,
        messages or [LLMMessage(role="user", content="What are you doing now?")],
        context=context,
        user_message="What are you doing now?",
        model="chat",
        max_output_tokens=350,
        cache_key="conversation:test",
        reasoning_effort="low",
    )


async def test_consistent_reply_keeps_snapshot_and_review_usage_separate():
    provider = SimpleNamespace(generate=AsyncMock(return_value=_result("I'm at a beach club.")))
    reviewer = SimpleNamespace(generate_structured=AsyncMock(return_value=_review(True)))
    guard = RoutineResponseGuard(reviewer=reviewer, reviewer_model="review")
    result = await _generate(guard, provider, _context())
    assert result.result.text == "I'm at a beach club."
    assert result.result.usage.input_tokens == 20
    assert result.metadata["routine_context"]["local_time_of_day"] == "afternoon"
    assert result.metadata["routine_context"]["activity"] == "Beach club"
    check = result.metadata["routine_validation"]["checks"][0]
    assert check["input_tokens"] == 30
    assert result.metadata["routine_validation"]["repaired"] is False
    provider.generate.assert_awaited_once()
    review_options = reviewer.generate_structured.await_args.kwargs
    assert review_options["max_output_tokens"] == 512
    assert review_options["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    "reason", ["activity", "local_time", "identity_disclaimer", "language", "attribution"]
)
async def test_wrong_draft_is_repaired_once_and_both_generations_are_accounted(reason):
    wrong = "Winding down for a Dubai evening, thinking about what to wear tonight."
    provider = SimpleNamespace(
        generate=AsyncMock(
            side_effect=[_result(wrong), _result("I'm at a beach club in Dubai this afternoon.")]
        )
    )
    reviewer = SimpleNamespace(
        generate_structured=AsyncMock(side_effect=[_review(False, reason), _review(True)])
    )
    guard = RoutineResponseGuard(reviewer=reviewer, reviewer_model="review")
    original = [LLMMessage(role="user", content="What are you doing now?")]
    result = await _generate(guard, provider, _context(), original)
    assert result.result.text == "I'm at a beach club in Dubai this afternoon."
    assert result.result.usage == LLMUsage(
        input_tokens=40,
        output_tokens=20,
        cached_input_tokens=4,
        reasoning_tokens=6,
        cost_usd_ticks=200,
    )
    assert len(result.metadata["routine_validation"]["checks"]) == 2
    assert result.metadata["routine_validation"]["repaired"] is True
    repair = provider.generate.await_args_list[1].args[0]
    assert repair[-1].role == "developer"
    assert reason in repair[-1].content
    assert "Beach club" in repair[-1].content
    assert wrong not in repair[-1].content
    assert len(original) == 1
    payloads = [
        json.loads(c.args[0][1].content) for c in reviewer.generate_structured.await_args_list
    ]
    assert payloads[0]["routine"] == payloads[1]["routine"]


async def test_repeated_contradiction_is_blocked_without_an_unbounded_retry():
    provider = SimpleNamespace(generate=AsyncMock(return_value=_result("I'm at dinner.")))
    reviewer = SimpleNamespace(
        generate_structured=AsyncMock(return_value=_review(False, "activity"))
    )
    guard = RoutineResponseGuard(reviewer=reviewer, reviewer_model="review")
    with pytest.raises(ProviderError, match="routine-consistent reply"):
        await _generate(guard, provider, _context())
    assert provider.generate.await_count == 2
    assert reviewer.generate_structured.await_count == 2


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"consistent": "true", "reason": "none"},
        {"consistent": True, "reason": "activity"},
        {"consistent": False, "reason": "none"},
    ],
)
async def test_malformed_reviews_cannot_approve_a_reply(data):
    provider = SimpleNamespace(generate=AsyncMock(return_value=_result("A draft")))
    reviewer = SimpleNamespace(
        generate_structured=AsyncMock(
            return_value=LLMStructuredResult(
                data=data, model="review", usage=LLMUsage(input_tokens=2, output_tokens=1)
            )
        )
    )
    with pytest.raises(ProviderError, match="could not be verified"):
        await _generate(
            RoutineResponseGuard(reviewer=reviewer, reviewer_model="review"), provider, _context()
        )
    provider.generate.assert_awaited_once()


async def test_reviewer_outage_does_not_release_an_unverified_reply():
    provider = SimpleNamespace(generate=AsyncMock(return_value=_result("A draft")))
    reviewer = SimpleNamespace(generate_structured=AsyncMock(side_effect=ProviderError("offline")))
    with pytest.raises(ProviderError, match="could not be verified"):
        await _generate(
            RoutineResponseGuard(reviewer=reviewer, reviewer_model="review"), provider, _context()
        )


async def test_no_routine_uses_exactly_the_original_generation_without_review():
    draft = _result("An ordinary reply")
    provider = SimpleNamespace(generate=AsyncMock(return_value=draft))
    reviewer = SimpleNamespace(
        generate_structured=AsyncMock(side_effect=AssertionError("no review"))
    )
    context = PromptContext(companion=None, auth=None, user=None)
    result = await _generate(
        RoutineResponseGuard(reviewer=reviewer, reviewer_model="review"), provider, context
    )
    assert result.result is draft
    assert result.metadata == {}
    reviewer.generate_structured.assert_not_awaited()
