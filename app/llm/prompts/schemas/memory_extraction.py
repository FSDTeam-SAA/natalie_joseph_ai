"""
JSON schema + validation for Phase 7 memory extraction.

Used with LLMProvider.generate_structured() against
settings.OPENAI_BACKGROUND_MODEL. This is the first thing to occupy
app/llm/prompts/schemas/, which existed as empty scaffolding since
Phase 5.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.db.models.memory import MemoryType

MEMORY_EXTRACTION_SCHEMA_NAME = "memory_extraction"

# OpenAI strict structured-output schema (per generate_structured()'s
# strict=True): every property must be listed in "required", and
# additionalProperties must be false at every object level, including
# nested ones.
MEMORY_EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": [t.value for t in MemoryType],
                    },
                    "key": {
                        "type": "string",
                        "description": (
                            "Short, stable, machine-friendly label for this fact "
                            "(e.g. 'pet_name', 'favorite_color', 'job_title'). "
                            "Reuse the same key across turns for the same "
                            "underlying fact so it gets updated instead of "
                            "duplicated."
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "The fact itself, in plain natural language.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": (
                            "0.0-1.0 confidence that this is accurate and worth "
                            "remembering long-term."
                        ),
                    },
                },
                "required": ["memory_type", "key", "value", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}


class ExtractedMemoryCandidate(BaseModel):
    memory_type: MemoryType
    key: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryExtractionResult(BaseModel):
    memories: list[ExtractedMemoryCandidate]


def parse_extraction_result(data: dict[str, Any]) -> list[ExtractedMemoryCandidate]:
    """
    Defensive re-validation on top of OpenAI's strict schema enforcement.
    Strict mode constrains shape, not semantic validity (e.g. confidence
    could technically arrive as 1.5 or a memory_type outside the enum
    if the provider ever misbehaves) — Pydantic re-checks that here
    rather than letting a bad row reach the database.

    Raises pydantic.ValidationError on failure. Callers must handle
    this: a malformed extraction result must never crash the request
    that triggered it (extraction always runs after the user's reply
    has already been generated and persisted).
    """
    return MemoryExtractionResult.model_validate(data).memories