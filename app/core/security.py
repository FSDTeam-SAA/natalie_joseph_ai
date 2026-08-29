"""
Authentication abstraction.

STATUS:
  - JWT verification (production): STILL AN INTERFACE ONLY, pending
    backend team confirmation — see TODO-CONFIRM list below and in
    app/core/config.py.
  - Local API key mode (dev/testing only): IMPLEMENTED in this file
    (LocalAPIKeyAuthProvider) as a temporary stand-in so Phase 5+ can
    be built and tested end-to-end before JWT details arrive. See the
    "SWITCHING TO REAL JWT" section at the bottom of this file for the
    exact steps to take once the backend team responds.

Per spec Section 30 and 66 ("do not fabricate functionality"), JWT
verification logic requires the following confirmed from the backend
team before it can be written:

  - JWT signing algorithm (HS256 vs RS256/ES256)
  - Public key / JWKS URL, or shared secret
  - Expected issuer (`iss`) and audience (`aud`)
  - Exact claim name for the internal user UUID
  - Exact claim name for the adult-eligibility flag
  - Exact claim name for the entitlement/subscription flag

Confirmed so far (2026-08-24): AUTH_MODE=jwt (for production),
user_id / conversation_id / companion_id are UUIDs. The claim names
and signing details are still open.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header

from app.core.config import AppEnv, AuthMode, Settings, get_settings
from app.core.exceptions import AuthenticationError


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
    Placeholder provider used until JWT verification details are
    confirmed. Always raises — this intentionally blocks any request
    from being treated as authenticated so the service fails closed
    rather than silently trusting unverified input.
    """

    async def authenticate(self, headers: dict[str, str]) -> AuthContext:
        raise AuthenticationError(
            "JWT verification is not yet configured. "
            "See app/core/security.py TODO-CONFIRM items.",
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
    # AuthMode.jwt and AuthMode.internal_service_token both fall back
    # to the fail-closed placeholder until implemented.
    return NotConfiguredAuthProvider()


async def get_current_auth_context(
    authorization: str | None = Header(default=None),
    x_debug_user_id: str | None = Header(default=None),
    x_debug_adult_eligible: str | None = Header(default=None),
    x_debug_entitled: str | None = Header(default=None),
    auth_provider: AuthProvider = Depends(get_auth_provider),
) -> AuthContext:
    """
    FastAPI dependency. Endpoints that require an authenticated user
    depend on this, never on AuthProvider directly.
    """
    headers: dict[str, str] = {}
    if authorization is not None:
        headers["authorization"] = authorization
    if x_debug_user_id is not None:
        headers["x-debug-user-id"] = x_debug_user_id
    if x_debug_adult_eligible is not None:
        headers["x-debug-adult-eligible"] = x_debug_adult_eligible
    if x_debug_entitled is not None:
        headers["x-debug-entitled"] = x_debug_entitled

    return await auth_provider.authenticate(headers)


# ============================================================================
# SWITCHING TO REAL JWT — instructions for when the backend team responds
# ============================================================================
#
# Once you have all six answers (algorithm, key/JWKS, issuer, audience,
# user-id claim name, adult-eligible claim name, entitled claim name):
#
# 1. Fill in the real values in your .env:
#      JWT_ALGORITHM=...
#      JWT_PUBLIC_KEY=...          (or JWT_SECRET=... if HS256)
#      JWT_ISSUER=...
#      JWT_AUDIENCE=...
#      JWT_USER_ID_CLAIM=...
#      JWT_ADULT_ELIGIBLE_CLAIM=...
#      JWT_ENTITLED_CLAIM=...
#      AUTH_MODE=jwt
#
# 2. In THIS file, add a new class (e.g. JWTAuthProvider(AuthProvider))
#    that:
#      - extracts the Bearer token from headers["authorization"]
#      - verifies signature/issuer/audience using python-jose
#        (already in requirements.txt) with settings.JWT_ALGORITHM /
#        JWT_PUBLIC_KEY or JWT_SECRET / JWT_ISSUER / JWT_AUDIENCE
#      - raises AuthenticationError on any verification failure
#      - reads settings.JWT_USER_ID_CLAIM / JWT_ADULT_ELIGIBLE_CLAIM /
#        JWT_ENTITLED_CLAIM out of the verified claims and returns an
#        AuthContext
#
# 3. In get_auth_provider() above, change the AuthMode.jwt branch from
#    NotConfiguredAuthProvider() to JWTAuthProvider(settings).
#
# 4. Nothing else changes. Every endpoint, service, and repository
#    already depends only on AuthContext (via get_current_auth_context),
#    never on how it was produced — so conversation/chat logic, memory
#    isolation, and every test built on LocalAPIKeyAuthProvider keeps
#    working unmodified once real JWT is wired in.
#
# 5. Before deploying to production, confirm AUTH_MODE=jwt in that
#    environment's .env — LocalAPIKeyAuthProvider actively refuses to
#    run when APP_ENV=production, but AUTH_MODE must still be switched
#    manually so the app picks JWTAuthProvider instead.