"""
Application health tests: confirm the FastAPI app boots and the basic
liveness endpoint responds. Does NOT require Postgres/Redis to be
running (that's what /health/ready is for, tested in integration).
"""

import json
from types import SimpleNamespace

from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from app.api.health import _activation_checks
from app.core.config import Settings
from app.main import app, request_validation_error_handler


def test_health_liveness() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_framework_404_uses_the_consistent_error_envelope() -> None:
    client = TestClient(app)
    response = client.get("/definitely-not-a-route")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "HTTP_404"
    assert response.json()["error"]["message"] == "Not Found"
    assert response.json()["error"]["request_id"]


def test_readiness_detects_missing_enabled_companion_assets(tmp_path) -> None:
    companions = [
        SimpleNamespace(
            slug=slug,
            voice_config={"voice_id": None},
            visual_config={"reference_images": []},
        )
        for slug in ("elena", "chloe", "thalia", "lina", "luna")
    ]
    checks = _activation_checks(
        companions,
        Settings(
            _env_file=None,
            ENABLE_VOICE_GENERATION=True,
            ENABLE_IMAGE_GENERATION=True,
            COMPANION_ASSET_ROOT=str(tmp_path),
        ),
    )

    assert checks["canonical_companions"] == "ok"
    assert checks["companion_voice_configuration"] == "missing"
    assert checks["companion_reference_assets"] == "missing"


async def test_validation_envelope_does_not_echo_private_input() -> None:
    request = SimpleNamespace(state=SimpleNamespace(request_id="request-123"))
    error = RequestValidationError(
        [
            {
                "type": "value_error",
                "loc": ("body", "message"),
                "msg": "invalid",
                "input": "private conversation text",
                "ctx": {"error": ValueError("invalid")},
            }
        ]
    )

    response = await request_validation_error_handler(request, error)
    body = json.loads(response.body)

    assert response.status_code == 422
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["request_id"] == "request-123"
    assert b"private conversation text" not in response.body
