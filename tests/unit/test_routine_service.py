from __future__ import annotations

from datetime import datetime

import pytest

from app.services.routine_service import parse_companion_routine, resolve_current_activity


def _slot(day=1, start="06:00", end="12:00", activity="Coffee at home, gym") -> dict:
    return dict(day=day, start=start, end=end, activity=activity)


@pytest.mark.parametrize(
    ("instant", "expected", "start", "end"),
    [
        ("2026-11-02T01:59:59+00:00", "Sleep", "2026-11-01T23:00", "2026-11-02T06:00"),
        (
            "2026-11-02T02:00:00+00:00",
            "Coffee at home, gym",
            "2026-11-02T06:00",
            "2026-11-02T12:00",
        ),
        (
            "2026-11-02T07:59:59+00:00",
            "Coffee at home, gym",
            "2026-11-02T06:00",
            "2026-11-02T12:00",
        ),
        ("2026-11-02T08:00:00+00:00", "Café", "2026-11-02T12:00", "2026-11-02T18:00"),
        ("2026-11-02T14:00:00+00:00", None, None, None),
        ("2026-11-02T19:00:00+00:00", "Monday sleep", "2026-11-02T23:00", "2026-11-03T06:00"),
        ("2026-11-02T22:00:00+00:00", "Monday sleep", "2026-11-02T23:00", "2026-11-03T06:00"),
        ("2026-11-03T02:00:00+00:00", None, None, None),
    ],
)
def test_boundaries_gaps_and_overnight_week_rollover(instant, expected, start, end) -> None:
    routine = parse_companion_routine(
        {
            "timezone": "Asia/Dubai",
            "weekly": [
                _slot(1, "23:00", "06:00", "Monday sleep"),
                _slot(7, "23:00", "06:00", "Sleep"),
                _slot(1, "12:00", "18:00", "Café"),
                _slot(),
            ],
        }
    )
    assert routine is not None
    result = resolve_current_activity(routine, now=datetime.fromisoformat(instant))
    if expected is None:
        assert result is None
    else:
        assert result.activity == expected
        assert result.starts_at.isoformat() == start + ":00+04:00"
        assert result.ends_at.isoformat() == end + ":00+04:00"


@pytest.mark.parametrize(
    ("timezone", "instant", "weekday", "local_hour"),
    [
        ("Asia/Dubai", "2026-11-01T21:00:00+00:00", 1, 1),
        ("America/New_York", "2026-11-02T02:00:00+00:00", 7, 21),
        ("Europe/London", "2026-03-29T00:30:00+00:00", 7, 0),
        ("Europe/London", "2026-03-29T01:30:00+00:00", 7, 2),
        ("Europe/London", "2026-10-25T00:30:00+00:00", 7, 1),
        ("Europe/London", "2026-10-25T01:30:00+00:00", 7, 1),
    ],
)
def test_companion_local_day_and_dst(timezone, instant, weekday, local_hour) -> None:
    routine = parse_companion_routine(
        {"timezone": timezone, "weekly": [_slot(weekday, "00:00", "23:59", "At home")]}
    )
    result = resolve_current_activity(routine, now=datetime.fromisoformat(instant))
    assert result.activity == "At home"
    assert result.local_now.isoweekday() == weekday
    assert result.local_now.hour == local_hour
    assert result.as_prompt_data()["timezone"] == timezone


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        "not a routine",
        {"timezone": "Invalid/Zone", "weekly": [_slot()]},
        {"timezone": "../Dubai", "weekly": [_slot()]},
        *[
            {"timezone": "Asia/Dubai", "weekly": [_slot(**override)]}
            for override in [
                {"day": 0},
                {"day": 8},
                {"day": True},
                {"day": "1"},
                {"start": "6:00"},
                {"end": "24:00"},
                {"start": "06:60"},
                {"activity": " "},
                {"activity": None},
                {"end": "06:00"},
            ]
        ],
        {"timezone": "Asia/Dubai", "weekly": [_slot(), _slot()]},
        {
            "timezone": "Asia/Dubai",
            "weekly": [_slot(1, "23:00", "06:00"), _slot(2, "05:00", "07:00")],
        },
        {
            "timezone": "Asia/Dubai",
            "weekly": [_slot(7, "23:00", "06:00"), _slot(1, "05:00", "07:00")],
        },
    ],
)
def test_invalid_or_missing_routines_fall_back_without_crashing(raw) -> None:
    assert parse_companion_routine(raw) is None
    assert resolve_current_activity(parse_companion_routine(raw)) is None


def test_empty_week_and_naive_clock() -> None:
    empty = parse_companion_routine({"timezone": "Asia/Dubai", "weekly": []})
    assert resolve_current_activity(empty) is None
    routine = parse_companion_routine({"timezone": "Asia/Dubai", "weekly": [_slot()]})
    with pytest.raises(ValueError, match="timezone offset"):
        resolve_current_activity(routine, now=datetime(2026, 11, 2, 8))
