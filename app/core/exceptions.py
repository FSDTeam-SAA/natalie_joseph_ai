"""
Application-level exceptions.

Per spec Section 37, all API errors are converted (via handlers
registered in main.py) into the consistent envelope:

{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "request_id": "..."
  }
}

Stack traces are never exposed in API responses.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors."""

    code: str = "APP_ERROR"
    status_code: int = 500

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class ValidationError(AppError):
    code = "VALIDATION_ERROR"
    status_code = 422


class AuthenticationError(AppError):
    code = "AUTHENTICATION_ERROR"
    status_code = 401


class AuthorizationError(AppError):
    code = "AUTHORIZATION_ERROR"
    status_code = 403


class NotFoundError(AppError):
    code = "NOT_FOUND"
    status_code = 404


class ProviderError(AppError):
    code = "PROVIDER_ERROR"
    status_code = 502


class ServiceUnavailableError(AppError):
    code = "SERVICE_UNAVAILABLE"
    status_code = 503


class RateLimitError(AppError):
    code = "RATE_LIMIT_EXCEEDED"
    status_code = 429


class ConversationError(AppError):
    code = "CONVERSATION_ERROR"
    status_code = 400


class MemoryError_(AppError):
    """Named MemoryError_ to avoid shadowing the Python builtin MemoryError."""

    code = "MEMORY_ERROR"
    status_code = 400
