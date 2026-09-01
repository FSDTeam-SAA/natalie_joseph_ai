"""
Authentication abstraction.

STATUS:
  - JWT verification: IMPLEMENTED (JWTAuthProvider) based on the
    backend team's actual NestJS login source, confirmed 2026-08-29.
    Two things still fail closed pending backend follow-up — see the
    "STILL OPEN" list in app/core/config.py's Auth section:
      - adult_eligible: no such claim exists in their tokens yet, so
        every user is treated as NOT adult-eligible until they add it
      - entitled: same — defaults to False until added
  - Local API key mode (dev/testing only): IMPLEMENTED
    (LocalAPIKeyAuthProvider) — still useful for testing flows that
    need adult_eligible=True before the backend adds that claim, since
    JWTAuthProvider cannot produce that today no matter what token you
    give it.

Confirmed from backend team's NestJS source (2026-08-29):
  - Algorithm: HS256 (implied — jwtService.sign() with a plain secret
    and no `algorithm` option)
  - User ID claim: "id"
  - No `iss`/`aud` claims are set — not verified here since none exist
  - Secret: their ACCESS_TOKEN_SECRET value, set as JWT_SECRET

Still open (not guessed — see config.py for full detail):
  - Exact type/format of `user.id` from their Prisma schema (UUID?
    cuid? autoincrement int?) — JWTAuthProvider currently requires it
    to parse as a UUID and raises a clear AuthenticationError if not,
    rather than silently coercing or guessing a conversion.
  - adult_eligible / entitled claims — absent from their tokens today.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt as jose_jwt

from app.core.config import AppEnv, AuthMode, Settings, get_settings
from app.core.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

# Using a proper OpenAPI security scheme (HTTPBearer) instead of a
# plain Header(alias="authorization") parameter. This is not just
# stylistic — Swagger UI has a confirmed bug where a header parameter
# literally named "Authorization" (declared as a plain header, not a
# security scheme) is silently dropped from the outgoing "Try it out"
# request even when the field displays the correct value. Using
# HTTPBearer gives Swagger a dedicated "Authorize" button (padlock
# icon) that reliably attaches the header on every subsequent request,
# and is also the standard/correct FastAPI pattern for bearer auth.
bearer_scheme = HTTPBearer(auto_error=False)



@dataclass(frozen=True)
class AuthContext:
    """
    The verified identity/authorization context for an incoming request.

    This is the ONLY source of truth the rest of the application may
    use for user identity, adult eligibility, and entitlement status.
    The LLM never makes these determinations (spec Section 3).
    """

    user_id: UUID
    adult_eligible: bool
    entitled: bool
    raw_claims: dict


class AuthProvider(ABC):
    """Abstract interface."""

    @abstractmethod
    async def authenticate(self, headers: dict[str, str]) -> AuthContext:
        """
        Validate the incoming request's credentials and return an
        AuthContext.

        Takes the full header mapping (not just Authorization) because
        the dev-mode LocalAPIKeyAuthProvider needs additional debug
        headers to simulate a per-user identity that a real JWT would
        otherwise carry. Real JWT verification only ever reads the
        Authorization header from this mapping.

        Must raise AuthenticationError on any failure (missing/invalid
        credential, expired token, wrong issuer/audience, etc.) —
        never fail open.
        """
        raise NotImplementedError


class NotConfiguredAuthProvider(AuthProvider):
    """
    Placeholder provider for auth modes with no implementation yet
    (currently: internal_service_token). Always raises — this
    intentionally blocks any request from being treated as
    authenticated so the service fails closed rather than silently
    trusting unverified input.
    """

    async def authenticate(self, headers: dict[str, str]) -> AuthContext:
        raise AuthenticationError(
            "This auth mode is not yet configured. See app/core/security.py.",
        )


class JWTAuthProvider(AuthProvider):
    """
    Verifies JWTs issued by the main backend's NestJS auth service.

    Based on their actual login service source (confirmed 2026-08-29):
    HS256, secret-based, claims are `{ id, role, email }` — no `sub`,
    no `iss`, no `aud`, no `adult_eligible`, no `entitled`.

    adult_eligible and entitled are NOT present in their current
    tokens. Per spec Section 28 ("fail closed" for missing eligibility
    data), this provider does not guess or default these to True —
    every request is treated as NOT adult-eligible and NOT entitled
    until the backend team adds those claims. This is logged once per
    request at DEBUG level (not WARNING/ERROR) since it is expected,
    known, ongoing behavior until the backend change lands — not a
    per-request anomaly worth alerting on.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.JWT_SECRET:
            raise RuntimeError(
                "AUTH_MODE=jwt but JWT_SECRET is not set. Set it to the backend "
                "team's ACCESS_TOKEN_SECRET value."
            )
        self._settings = settings

    async def authenticate(self, headers: dict[str, str]) -> AuthContext:
        authorization = headers.get("authorization")
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("Missing or malformed Authorization header.")

        token = authorization.removeprefix("Bearer ").strip()

        try:
            claims = jose_jwt.decode(
                token,
                self._settings.JWT_SECRET,
                algorithms=[self._settings.JWT_ALGORITHM],
                # No issuer/audience verification: the backend's token
                # issuance code does not set iss/aud claims, so there is
                # nothing to check them against (confirmed 2026-08-29).
            )
        except JWTError as exc:
            raise AuthenticationError(f"Invalid or expired token: {exc}") from exc

        raw_user_id = claims.get(self._settings.JWT_USER_ID_CLAIM)
        if raw_user_id is None:
            raise AuthenticationError(
                f"Token is missing the expected '{self._settings.JWT_USER_ID_CLAIM}' claim."
            )

        try:
            user_id = UUID(str(raw_user_id))
        except ValueError as exc:
            raise AuthenticationError(
                f"The '{self._settings.JWT_USER_ID_CLAIM}' claim value "
                f"({raw_user_id!r}) is not a valid UUID. This service's user "
                "records require a UUID external_user_id — confirm with the "
                "backend team what format their user IDs actually use "
                "(UUID, cuid, autoincrement integer, etc.) if this keeps "
                "happening."
            ) from exc

        adult_eligible_raw = claims.get(self._settings.JWT_ADULT_ELIGIBLE_CLAIM)
        entitled_raw = claims.get(self._settings.JWT_ENTITLED_CLAIM)

        if adult_eligible_raw is None or entitled_raw is None:
            logger.debug(
                "JWT for user_id=%s is missing adult_eligible and/or entitled "
                "claims (backend has not added these yet) — failing closed: "
                "adult_eligible=False, entitled=False.",
                user_id,
            )

        return AuthContext(
            user_id=user_id,
            adult_eligible=bool(adult_eligible_raw) if adult_eligible_raw is not None else False,
            entitled=bool(entitled_raw) if entitled_raw is not None else False,
            raw_claims=claims,
        )


class LocalAPIKeyAuthProvider(AuthProvider):
    """
    TEMPORARY / DEV-ONLY. Do not use in production.

    Authenticates a request with a single shared static API key
    (settings.LOCAL_API_KEY), then reads the acting user's identity
    and adult-eligibility from debug headers supplied directly by the
    caller — since a static shared key carries no per-user identity
    the way a real JWT would.

    Required header:
        Authorization: Bearer <LOCAL_API_KEY>

    Optional debug headers (only meaningful in this auth mode):
        X-Debug-User-Id: <uuid>            (defaults to a fixed test UUID)
        X-Debug-Adult-Eligible: true|false (defaults to false — fail closed)
        X-Debug-Entitled: true|false       (defaults to true, for convenience)

    Because X-Debug-Adult-Eligible defaults to false, testing intimate/
    romantic conversation flows requires deliberately setting that
    header to true — the same fail-closed principle as the real JWT
    path applies here too, just enforced via a header instead of a claim.
    """

    _DEFAULT_TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")

    def __init__(self, settings: Settings) -> None:
        if settings.APP_ENV == AppEnv.production:
            raise RuntimeError(
                "LocalAPIKeyAuthProvider must never be used when APP_ENV=production. "
                "Set AUTH_MODE=jwt for production deployments."
            )
        if not settings.LOCAL_API_KEY:
            raise RuntimeError(
                "AUTH_MODE=local_api_key but LOCAL_API_KEY is not set in the environment."
            )
        self._settings = settings

    async def authenticate(self, headers: dict[str, str]) -> AuthContext:
        authorization = headers.get("authorization")
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("Missing or malformed Authorization header.")

        provided_key = authorization.removeprefix("Bearer ").strip()
        if provided_key != self._settings.LOCAL_API_KEY:
            raise AuthenticationError("Invalid API key.")

        debug_user_id_header = headers.get("x-debug-user-id")
        try:
            user_id = UUID(debug_user_id_header) if debug_user_id_header else self._DEFAULT_TEST_USER_ID
        except ValueError as exc:
            raise AuthenticationError("X-Debug-User-Id must be a valid UUID.") from exc

        adult_eligible = headers.get("x-debug-adult-eligible", "false").strip().lower() == "true"
        entitled = headers.get("x-debug-entitled", "true").strip().lower() == "true"

        return AuthContext(
            user_id=user_id,
            adult_eligible=adult_eligible,
            entitled=entitled,
            raw_claims={"source": "local_api_key_dev_mode"},
        )


def get_auth_provider(settings: Settings = Depends(get_settings)) -> AuthProvider:
    if settings.AUTH_MODE == AuthMode.local_api_key:
        return LocalAPIKeyAuthProvider(settings)
    if settings.AUTH_MODE == AuthMode.jwt:
        return JWTAuthProvider(settings)
    # internal_service_token falls back to the fail-closed placeholder
    # until implemented.
    return NotConfiguredAuthProvider()


async def get_current_auth_context(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_debug_user_id: str | None = Header(default=None),
    x_debug_adult_eligible: str | None = Header(default=None),
    x_debug_entitled: str | None = Header(default=None),
    auth_provider: AuthProvider = Depends(get_auth_provider),
) -> AuthContext:
    """
    FastAPI dependency. Endpoints that require an authenticated user
    depend on this, never on AuthProvider directly.

    The bearer token is now extracted via the HTTPBearer security
    scheme (see bearer_scheme above) rather than a raw Header —
    this fixes a Swagger UI bug where a plain header parameter named
    "authorization" was silently dropped from outgoing requests.
    """
    headers: dict[str, str] = {}
    if credentials is not None:
        headers["authorization"] = f"{credentials.scheme} {credentials.credentials}"
    if x_debug_user_id is not None:
        headers["x-debug-user-id"] = x_debug_user_id
    if x_debug_adult_eligible is not None:
        headers["x-debug-adult-eligible"] = x_debug_adult_eligible
    if x_debug_entitled is not None:
        headers["x-debug-entitled"] = x_debug_entitled

    return await auth_provider.authenticate(headers)


# ============================================================================
# STATUS AS OF 2026-08-29 — what's real vs. still pending
# ============================================================================
#
# JWTAuthProvider above is REAL and will correctly verify signatures,
# reject expired/tampered tokens, and extract user_id from any token
# actually issued by the backend's login service shown to us.
#
# Two things still block full production behavior — both fail closed,
# not fabricated:
#
#   1. adult_eligible / entitled claims don't exist in their tokens
#      yet. Every authenticated user is currently treated as
#      NOT adult-eligible and NOT entitled, regardless of their real
#      status, until the backend team adds these claims to their
#      jwtService.sign() payload.
#
#   2. `user.id` format is unconfirmed. JWTAuthProvider requires it to
#      parse as a UUID and raises a clear, descriptive
#      AuthenticationError if it doesn't — it does not silently coerce
#      or guess. If real tokens start failing with that error, the
#      fix is either (a) get the backend team to confirm/adjust their
#      ID format, or (b) if their IDs are genuinely not UUIDs (e.g.
#      autoincrement integers), change `users.external_user_id` in
#      this service's schema from UUID to a plain string type — that
#      would need a migration, not just a config change.
#
# TO ACTIVATE JWT AUTH FOR REAL TESTING RIGHT NOW:
#   1. Set JWT_SECRET in .env to the backend team's actual
#      ACCESS_TOKEN_SECRET value (get this from them securely — not
#      pasted in plaintext chat/email).
#   2. Set AUTH_MODE=jwt in .env.
#   3. Get a real access token from their /login endpoint and use it
#      as `Authorization: Bearer <token>` in requests to this service.
#
# Until the backend adds adult_eligible/entitled, you can still test
# the romantic/intimate conversation path using
# AUTH_MODE=local_api_key with X-Debug-Adult-Eligible: true — real JWT
# auth cannot produce that today no matter what token you present.