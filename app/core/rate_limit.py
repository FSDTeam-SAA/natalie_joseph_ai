"""Redis-backed fixed-window guards for provider-backed API calls."""

from __future__ import annotations

import hashlib
import logging
import time

import redis.asyncio as redis_asyncio
from redis.exceptions import RedisError

from app.core.exceptions import RateLimitError, ServiceUnavailableError
from app.core.security import AuthContext

logger = logging.getLogger(__name__)

_INCREMENT_SCRIPT = """
local value = redis.call('INCR', KEYS[1])
if value == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return value
"""


class RateLimiter:
    def __init__(
        self,
        *,
        redis_url: str,
        enabled: bool,
        fail_closed: bool,
        per_user_per_minute: int,
        per_ip_per_minute: int,
        burst_per_ten_seconds: int,
    ) -> None:
        self.enabled = enabled
        self.fail_closed = fail_closed
        self.per_user_per_minute = per_user_per_minute
        self.per_ip_per_minute = per_ip_per_minute
        self.burst_per_ten_seconds = burst_per_ten_seconds
        self.client = redis_asyncio.from_url(redis_url, decode_responses=True)

    @staticmethod
    def _subject(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]

    async def _increment(self, key: str, *, ttl: int) -> int:
        value = await self.client.eval(_INCREMENT_SCRIPT, 1, key, ttl)
        return int(value)

    async def check(
        self,
        *,
        auth: AuthContext,
        client_ip: str | None,
    ) -> None:
        if not self.enabled:
            return
        now = int(time.time())
        user_subject = self._subject(str(auth.user_id))
        try:
            per_minute = await self._increment(
                f"elysia:limit:user:{user_subject}:{now // 60}", ttl=70
            )
            burst = await self._increment(
                f"elysia:limit:burst:{user_subject}:{now // 10}", ttl=15
            )
            ip_count = 0
            if client_ip and auth.raw_claims.get("source") != "trusted_backend":
                ip_subject = self._subject(client_ip)
                ip_count = await self._increment(
                    f"elysia:limit:ip:{ip_subject}:{now // 60}", ttl=70
                )
        except RedisError:
            logger.exception("rate_limit_backend_unavailable")
            if self.fail_closed:
                raise ServiceUnavailableError(
                    "Request limiting is temporarily unavailable; please retry shortly."
                ) from None
            return

        if (
            per_minute > self.per_user_per_minute
            or burst > self.burst_per_ten_seconds
            or ip_count > self.per_ip_per_minute
        ):
            raise RateLimitError("Too many AI requests. Please retry shortly.")
