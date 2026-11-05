"""Validate backend-owned weekly routines and resolve companion-local activities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

_TIME_PATTERN = r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$"
_DAY_MINUTES = 24 * 60
_WEEK_MINUTES = 7 * _DAY_MINUTES


def _minutes(value: str) -> int:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


class RoutineSlot(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: int = Field(strict=True, ge=1, le=7)
    start: str = Field(pattern=_TIME_PATTERN)
    end: str = Field(pattern=_TIME_PATTERN)
    activity: str = Field(min_length=1, max_length=2000)

    @field_validator("activity")
    @classmethod
    def nonblank_activity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("activity must not be blank")
        return value.strip()


class CompanionRoutine(BaseModel):
    model_config = ConfigDict(frozen=True)

    timezone: str = Field(min_length=1, max_length=100)
    weekly: tuple[RoutineSlot, ...] = Field(max_length=_WEEK_MINUTES)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("timezone must be an available IANA timezone") from exc
        return value

    @model_validator(mode="after")
    def nonoverlapping_slots(self) -> CompanionRoutine:
        # Split Sunday overnight intervals at the week boundary so they are
        # checked against Monday's early slots as well. Gaps are allowed.
        intervals: list[tuple[int, int]] = []
        for slot in self.weekly:
            start_minute, end_minute = _minutes(slot.start), _minutes(slot.end)
            if start_minute == end_minute:
                raise ValueError("a routine slot must have different start and end times")
            start = (slot.day - 1) * _DAY_MINUTES + start_minute
            duration = (end_minute - start_minute) % _DAY_MINUTES
            end = start + duration
            intervals.append((start, min(end, _WEEK_MINUTES)))
            if end > _WEEK_MINUTES:
                intervals.append((0, end - _WEEK_MINUTES))
        intervals.sort()
        if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:], strict=False)):
            raise ValueError("routine slots must not overlap")
        return self


@dataclass(frozen=True, slots=True)
class RoutineActivity:
    timezone: str
    local_now: datetime
    starts_at: datetime
    ends_at: datetime
    activity: str

    def as_prompt_data(self) -> dict[str, str]:
        hour = self.local_now.hour
        daypart = (
            "morning"
            if 6 <= hour < 12
            else "afternoon"
            if 12 <= hour < 18
            else "evening"
            if 18 <= hour < 22
            else "night"
        )
        return {
            "timezone": self.timezone,
            "local_datetime": self.local_now.isoformat(),
            "local_weekday": self.local_now.strftime("%A"),
            "local_time_of_day": daypart,
            "starts_at": self.starts_at.isoformat(),
            "ends_at": self.ends_at.isoformat(),
            "activity": self.activity,
        }


def parse_companion_routine(value: object) -> CompanionRoutine | None:
    """Missing or malformed optional routines must never disable existing chat."""
    if value is None:
        return None
    try:
        return CompanionRoutine.model_validate(value)
    except ValidationError:
        # Avoid logging profile contents or arbitrary activity strings.
        logger.warning("invalid_companion_routine_ignored")
        return None


def resolve_current_activity(
    routine: CompanionRoutine | None, *, now: datetime | None = None
) -> RoutineActivity | None:
    """Use ISO weekdays and [start, end) local wall-clock slots, including overnight."""
    if routine is None or not routine.weekly:
        return None
    instant = now if now is not None else datetime.now(UTC)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("now must include a timezone offset")
    zone = ZoneInfo(routine.timezone)
    local_now = instant.astimezone(zone)
    for start_date in (local_now.date(), local_now.date() - timedelta(days=1)):
        for slot in routine.weekly:
            if slot.day != start_date.isoweekday():
                continue
            starts_at = datetime.combine(start_date, time.fromisoformat(slot.start), zone)
            end_date = start_date + timedelta(days=int(slot.end < slot.start))
            ends_at = datetime.combine(end_date, time.fromisoformat(slot.end), zone)
            if starts_at <= local_now < ends_at:
                return RoutineActivity(
                    timezone=routine.timezone,
                    local_now=local_now,
                    starts_at=starts_at,
                    ends_at=ends_at,
                    activity=slot.activity,
                )
    return None
