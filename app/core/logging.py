"""
Structured logging configuration.

Per spec Section 39, logs must include request_id, endpoint, a safe
user identifier, companion_id, conversation_id, latency, model, token
counts, provider error type, and safety action — but must NEVER dump
full user conversation content into normal application logs.

This module configures stdlib logging to emit JSON lines so it is
consumable by any log aggregator. Call `configure_logging()` once at
app startup.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Structured extra fields (request_id, user_id, companion_id, etc.)
        # are attached via `logger.info(..., extra={"request_id": ...})`
        # and surfaced here without ever including raw message content
        # unless a caller explicitly and intentionally logs it (which
        # application code must not do for user conversation text).
        reserved = set(logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys())
        for key, value in record.__dict__.items():
            if key not in reserved and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Avoid duplicate handlers on reload
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Quiet noisy third-party loggers by default
    logging.getLogger("uvicorn.access").setLevel("WARNING")
