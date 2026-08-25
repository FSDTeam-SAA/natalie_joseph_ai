"""
Authentication abstraction.

STATUS: INTERFACE ONLY. Not implemented yet.

Per spec Section 30 and 66 ("do not fabricate functionality"), this
module defines the `AuthProvider` interface and the `AuthContext`
shape the rest of the application will depend on, but the actual JWT
verification logic (Phase 5) requires the following confirmed from the
backend team before it can be written:

  - JWT signing algorithm (HS256 vs RS256/ES256)
  - Public key / JWKS URL, or shared secret
  - Expected issuer (`iss`) and audience (`aud`)
  - Exact claim name for the internal user UUID
  - Exact claim name for the adult-eligibility flag
  - Exact claim name for the entitlement/subscription flag

Confirmed so far (2026-08-24): AUTH_MODE=jwt, and user_id /
conversation_id / companion_id are UUIDs. The claim names and signing
details are still open — see `app/core/config.py` TODO-CONFIRM notes.

Until those are confirmed, any endpoint depending on `AuthProvider`
must not be treated as security-complete.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID


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
    """Abstract interface. Concrete implementation pending Phase 5."""

    @abstractmethod
    async def authenticate(self, authorization_header: str | None) -> AuthContext:
        """
        Validate the incoming credential and return an AuthContext.

        Must raise `app.core.exceptions.AuthenticationError` on any
        failure (missing header, invalid signature, expired token,
        wrong issuer/audience, etc.) — never fail open.
        """
        raise NotImplementedError


class NotConfiguredAuthProvider(AuthProvider):
    """
    Placeholder provider used until JWT verification details are
    confirmed. Always raises — this intentionally blocks any request
    from being treated as authenticated so the service fails closed
    rather than silently trusting unverified input.
    """

    async def authenticate(self, authorization_header: str | None) -> AuthContext:
        from app.core.exceptions import AuthenticationError

        raise AuthenticationError(
            "JWT verification is not yet configured. "
            "See app/core/security.py TODO-CONFIRM items.",
        )
