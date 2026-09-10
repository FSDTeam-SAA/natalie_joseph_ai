"""Meet Elysia AI Companion Engine application entrypoint."""

from __future__ import annotations

import logging
import uuid
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import backend, chat, companions, conversations, health, media, users, voice
from app.core.config import AppEnv, AuthMode, get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.LOG_LEVEL)

if settings.APP_ENV == AppEnv.production and settings.AUTH_MODE == AuthMode.local_api_key:
    raise RuntimeError(
        "Refusing to start: AUTH_MODE=local_api_key is a development-only "
        "auth mode and must never run with APP_ENV=production. Set "
        "AUTH_MODE=internal_service_token or jwt for production."
    )

app = FastAPI(
    title="Meet Elysia — AI Companion Engine",
    description=(
        "AI backend service for the Meet Elysia companion platform. "
        "Owns conversation orchestration, companion personas, memory, "
        "voice/image generation and story context. Does not own "
        "authentication UI, payments, or the frontend."
    ),
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, _exc: RequestValidationError
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "The request payload is invalid.",
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    try:
        message = HTTPStatus(exc.status_code).phrase
    except ValueError:
        message = "HTTP error"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": message,
                "request_id": request_id,
            }
        },
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never expose stack traces or internal details in the response body.
    request_id = getattr(request.state, "request_id", None)
    logger.exception("unhandled_request_error", extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred.",
                "request_id": request_id,
            }
        },
    )


app.include_router(health.router)
app.include_router(companions.router, prefix=settings.API_V1_PREFIX)
app.include_router(conversations.router, prefix=settings.API_V1_PREFIX)
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)
app.include_router(media.router, prefix=settings.API_V1_PREFIX)
app.include_router(voice.router, prefix=settings.API_V1_PREFIX)
app.include_router(users.router, prefix=settings.API_V1_PREFIX)
app.include_router(backend.router, prefix=settings.API_V1_PREFIX)
