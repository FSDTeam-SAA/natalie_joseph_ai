from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.exceptions import RateLimitError, ServiceUnavailableError
from app.core.rate_limit import RateLimiter
from app.core.security import AuthContext


def _limiter(*, fail_closed: bool = True) -> RateLimiter:
    limiter = RateLimiter(
        redis_url="redis://localhost:6379/15",
        enabled=True,
        fail_closed=fail_closed,
        per_user_per_minute=20,
        per_ip_per_minute=60,
        burst_per_ten_seconds=5,
    )
    limiter.client = AsyncMock()
    return limiter


def _auth(*, trusted: bool = False) -> AuthContext:
    return AuthContext(
        user_id=uuid.uuid4(),
        adult_eligible=False,
        entitled=True,
        raw_claims={"source": "trusted_backend"} if trusted else {},
    )


async def test_direct_request_checks_user_burst_and_ip_limits() -> None:
    limiter = _limiter()
    limiter.client.eval.side_effect = [1, 1, 1]
    await limiter.check(auth=_auth(), client_ip="203.0.113.4")
    assert limiter.client.eval.await_count == 3


async def test_trusted_backend_skips_shared_proxy_ip_limit() -> None:
    limiter = _limiter()
    limiter.client.eval.side_effect = [1, 1]
    await limiter.check(auth=_auth(trusted=True), client_ip="10.0.0.2")
    assert limiter.client.eval.await_count == 2


async def test_burst_limit_is_rejected() -> None:
    limiter = _limiter()
    limiter.client.eval.side_effect = [2, 6, 1]
    with pytest.raises(RateLimitError):
        await limiter.check(auth=_auth(), client_ip="203.0.113.4")


async def test_redis_failure_fails_closed_without_leaking_details() -> None:
    limiter = _limiter(fail_closed=True)
    limiter.client.eval.side_effect = RedisConnectionError("secret infrastructure detail")
    with pytest.raises(ServiceUnavailableError) as raised:
        await limiter.check(auth=_auth(), client_ip=None)
    assert "secret" not in str(raised.value)


async def test_redis_failure_can_fail_open_when_explicitly_configured() -> None:
    limiter = _limiter(fail_closed=False)
    limiter.client.eval.side_effect = RedisConnectionError("down")
    await limiter.check(auth=_auth(), client_ip=None)
