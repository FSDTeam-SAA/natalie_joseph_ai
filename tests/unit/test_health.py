"""
Smoke test for Phase 1: confirms the FastAPI app boots and the basic
liveness endpoint responds. Does NOT require Postgres/Redis to be
running (that's what /health/ready is for, tested in integration).
"""

from fastapi.testclient import TestClient

from app.main import app


def test_health_liveness() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
