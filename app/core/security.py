"""Fail-closed authentication and backend-supplied AI authorization context.

The recommended production path is a private service-to-service bearer token:
the main backend authenticates the end user and supplies its age, entitlement,
and feature decisions. Direct JWT verification remains available for deployments
whose tokens already contain the complete decision contract. A development-only
API key mode supports isolated local testing.
"""

from __future__ import annotations

import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt

from app.core.config import AppEnv, AuthMode, Settings, get_settings
from app.core.exceptions import AuthenticationError, AuthorizationError

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
    features: frozenset[str] | None = None

    def allows(self, feature: str) -> bool:
        """Apply a decision made by the external backend.

        A populated feature set is authoritative and least-privilege. The
        legacy `entitled` boolean remains a compatibility fallback until the
        backend starts sending feature-scoped grants.
        """
        if self.features is not None:
            return feature in self.features
        return self.entitled


def require_feature(auth: AuthContext, feature: str) -> None:
    if not auth.allows(feature):
        raise AuthorizationError(
            f"The trusted backend did not authorize the '{feature}' AI feature."
        )


def require_trusted_backend(auth: AuthContext) -> None:
    """Restrict orchestration/scheduling routes to service-to-service calls."""
    if auth.raw_claims.get("source") != "trusted_backend":
        raise AuthorizationError("This operation may only be initiated by the trusted backend.")


def _strict_bool(value: object, *, claim_name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if type(value) is not bool:
        raise AuthenticationError(f"The '{claim_name}' claim must be a JSON boolean.")
    return value


def _feature_set(value: object, *, claim_name: str) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AuthenticationError(f"The '{claim_name}' claim must be an array of strings.")
    return frozenset(item.strip() for item in value if item.strip())


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
    """Defensive fallback for an unsupported auth mode; always fails closed."""

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
            decode_options = {
                "require_exp": self._settings.JWT_REQUIRE_EXP,
                "verify_aud": bool(self._settings.JWT_AUDIENCE),
                "verify_iss": bool(self._settings.JWT_ISSUER),
            }
            claims = jose_jwt.decode(
                token,
                self._settings.JWT_SECRET,
                algorithms=[self._settings.JWT_ALGORITHM],
                audience=self._settings.JWT_AUDIENCE or None,
                issuer=self._settings.JWT_ISSUER or None,
                options=decode_options,
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
            adult_eligible=_strict_bool(
                adult_eligible_raw,
                claim_name=self._settings.JWT_ADULT_ELIGIBLE_CLAIM,
            ),
            entitled=_strict_bool(
                entitled_raw,
                claim_name=self._settings.JWT_ENTITLED_CLAIM,
            ),
            raw_claims=claims,
            features=_feature_set(
                claims.get(self._settings.JWT_AI_FEATURES_CLAIM),
                claim_name=self._settings.JWT_AI_FEATURES_CLAIM,
            ),
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
                "Use internal_service_token or jwt for production deployments."
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
        if not secrets.compare_digest(provided_key, self._settings.LOCAL_API_KEY):
            raise AuthenticationError("Invalid API key.")

        debug_user_id_header = headers.get("x-debug-user-id")
        try:
            user_id = (
                UUID(debug_user_id_header)
                if debug_user_id_header
                else self._DEFAULT_TEST_USER_ID
            )
        except ValueError as exc:
            raise AuthenticationError("X-Debug-User-Id must be a valid UUID.") from exc

        adult_eligible = headers.get("x-debug-adult-eligible", "false").strip().lower() == "true"
        entitled = headers.get("x-debug-entitled", "true").strip().lower() == "true"

        return AuthContext(
            user_id=user_id,
            adult_eligible=adult_eligible,
            entitled=entitled,
            raw_claims={"source": "local_api_key_dev_mode"},
            features=(
                frozenset(
                    feature.strip()
                    for feature in headers["x-debug-ai-features"].split(",")
                    if feature.strip()
                )
                if "x-debug-ai-features" in headers
                else None
            ),
        )


class InternalServiceTokenAuthProvider(AuthProvider):
    """Authenticate the trusted main backend, which supplies user decisions.

    This mode intentionally does not reproduce end-user login or billing. The
    main backend authenticates the user and computes access, then calls this
    private service with a shared service credential and explicit user/scopes.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.INTERNAL_SERVICE_TOKEN:
            raise RuntimeError(
                "AUTH_MODE=internal_service_token but INTERNAL_SERVICE_TOKEN is not set."
            )
        self._settings = settings

    async def authenticate(self, headers: dict[str, str]) -> AuthContext:
        authorization = headers.get("authorization")
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("Missing or malformed Authorization header.")
        provided = authorization.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(provided, self._settings.INTERNAL_SERVICE_TOKEN):
            raise AuthenticationError("Invalid internal service credential.")

        raw_user_id = headers.get("x-backend-user-id")
        if not raw_user_id:
            raise AuthenticationError("Missing X-Backend-User-Id header.")
        try:
            user_id = UUID(raw_user_id)
        except ValueError as exc:
            raise AuthenticationError("X-Backend-User-Id must be a valid UUID.") from exc

        adult = _strict_bool_header(
            headers.get("x-backend-adult-eligible"),
            header_name="X-Backend-Adult-Eligible",
        )
        entitled = _strict_bool_header(
            headers.get("x-backend-entitled"),
            header_name="X-Backend-Entitled",
        )
        features = (
            frozenset(
                feature.strip()
                for feature in headers["x-backend-ai-features"].split(",")
                if feature.strip()
            )
            if "x-backend-ai-features" in headers
            else None
        )
        return AuthContext(
            user_id=user_id,
            adult_eligible=adult,
            entitled=entitled,
            features=features,
            raw_claims={"source": "trusted_backend"},
        )


def _strict_bool_header(value: str | None, *, header_name: str) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise AuthenticationError(f"{header_name} must be 'true' or 'false'.")


def get_auth_provider(settings: Settings = Depends(get_settings)) -> AuthProvider:
    if settings.AUTH_MODE == AuthMode.local_api_key:
        return LocalAPIKeyAuthProvider(settings)
    if settings.AUTH_MODE == AuthMode.jwt:
        return JWTAuthProvider(settings)
    if settings.AUTH_MODE == AuthMode.internal_service_token:
        return InternalServiceTokenAuthProvider(settings)
    return NotConfiguredAuthProvider()


async def get_current_auth_context(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_debug_user_id: str | None = Header(default=None),
    x_debug_adult_eligible: str | None = Header(default=None),
    x_debug_entitled: str | None = Header(default=None),
    x_debug_ai_features: str | None = Header(default=None),
    x_backend_user_id: str | None = Header(default=None),
    x_backend_adult_eligible: str | None = Header(default=None),
    x_backend_entitled: str | None = Header(default=None),
    x_backend_ai_features: str | None = Header(default=None),
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
    if x_debug_ai_features is not None:
        headers["x-debug-ai-features"] = x_debug_ai_features
    if x_backend_user_id is not None:
        headers["x-backend-user-id"] = x_backend_user_id
    if x_backend_adult_eligible is not None:
        headers["x-backend-adult-eligible"] = x_backend_adult_eligible
    if x_backend_entitled is not None:
        headers["x-backend-entitled"] = x_backend_entitled
    if x_backend_ai_features is not None:
        headers["x-backend-ai-features"] = x_backend_ai_features

    return await auth_provider.authenticate(headers)
