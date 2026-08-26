"""
Meet Elysia — AI Companion Engine
Application entrypoint.

Phase 1 scope: app bootstrap, config, logging, error handling, health
checks only. Business routers (chat, companions, conversations,
memories, users) are added in later phases as their services are
implemented — they are intentionally NOT wired in here yet so this
service never claims functionality it doesn't have.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import companions, health
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.LOG_LEVEL)

app = FastAPI(
    title="Meet Elysia — AI Companion Engine",
    description=(
        "AI backend service for the Meet Elysia companion platform. "
        "Owns conversation orchestration, companion personas, memory, "
        "context assembly, and safety moderation. Does not own "
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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never expose stack traces or internal details in the response body.
    request_id = getattr(request.state, "request_id", None)
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

# Business routers — registered here as they're delivered in later phases:
# app.include_router(conversations.router, prefix=settings.API_V1_PREFIX, tags=["Conversations"])  # Phase 5
# app.include_router(chat.router, prefix=settings.API_V1_PREFIX, tags=["Chat"])                # Phase 5/10
# app.include_router(memories.router, prefix=settings.API_V1_PREFIX, tags=["Memories"])        # Phase 7
# app.include_router(users.router, prefix=settings.API_V1_PREFIX, tags=["Users"])              # Phase 11