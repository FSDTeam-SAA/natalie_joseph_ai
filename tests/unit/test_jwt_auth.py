"""
Unit tests for JWTAuthProvider.

These sign real JWTs with python-jose (the same library used for
verification) rather than mocking anything — this is pure crypto/logic
that needs no network access, so there's no reason to mock it.

Token shapes here match the backend team's confirmed NestJS
implementation exactly: HS256, payload `{ id, role, email }`, no
iss/aud, no adult_eligible/entitled (those are tested as absent,
since that's the real current state of their tokens).
"""

from __future__ import annotations

import time
import uuid

import pytest
from jose import jwt as jose_jwt

from app.core.config import AuthMode, Settings
from app.core.exceptions import AuthenticationError
from app.core.security import InternalServiceTokenAuthProvider, JWTAuthProvider

TEST_SECRET = "test-access-token-secret"


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        AUTH_MODE=AuthMode.jwt,
        JWT_ALGORITHM="HS256",
        JWT_SECRET=TEST_SECRET,
        JWT_ISSUER="",
        JWT_AUDIENCE="",
        JWT_USER_ID_CLAIM="id",
        JWT_ADULT_ELIGIBLE_CLAIM="adult_eligible",
        JWT_ENTITLED_CLAIM="entitled",
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/meet_elysia",
        OPENAI_API_KEY="sk-fake",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _sign(
    payload: dict,
    *,
    secret: str = TEST_SECRET,
    algorithm: str = "HS256",
    include_exp: bool = True,
) -> str:
    claims = dict(payload)
    if include_exp and "exp" not in claims:
        claims["exp"] = int(time.time()) + 300
    return jose_jwt.encode(claims, secret, algorithm=algorithm)


class TestJWTAuthProviderHappyPath:
    async def test_valid_token_extracts_user_id(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        user_id = str(uuid.uuid4())
        token = _sign({"id": user_id, "role": "user", "email": "test@example.com"})

        auth = await provider.authenticate({"authorization": f"Bearer {token}"})

        assert str(auth.user_id) == user_id
        assert auth.raw_claims["email"] == "test@example.com"

    async def test_missing_adult_eligible_and_entitled_fails_closed(self) -> None:
        """
        This is the critical current-state test: the backend's real
        tokens don't carry these claims at all, so both must default
        to False — never True — until the backend adds them.
        """
        provider = JWTAuthProvider(_make_settings())
        token = _sign({"id": str(uuid.uuid4()), "role": "user", "email": "a@b.com"})

        auth = await provider.authenticate({"authorization": f"Bearer {token}"})

        assert auth.adult_eligible is False
        assert auth.entitled is False

    async def test_explicit_true_claims_are_honored_if_backend_adds_them_later(self) -> None:
        """
        Forward-looking: once the backend adds these claims, this
        provider must actually respect them, not just always return
        False. This test guards against someone hardcoding False
        instead of implementing real fail-closed-on-absence logic.
        """
        provider = JWTAuthProvider(_make_settings())
        token = _sign(
            {
                "id": str(uuid.uuid4()),
                "role": "user",
                "email": "a@b.com",
                "adult_eligible": True,
                "entitled": True,
            }
        )

        auth = await provider.authenticate({"authorization": f"Bearer {token}"})

        assert auth.adult_eligible is True
        assert auth.entitled is True

    async def test_feature_scopes_are_extracted(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        token = _sign(
            {
                "id": str(uuid.uuid4()),
                "ai_features": ["voice_input", "image"],
            }
        )

        auth = await provider.authenticate({"authorization": f"Bearer {token}"})

        assert auth.features == frozenset({"voice_input", "image"})
        assert auth.allows("image") is True
        assert auth.allows("voice_output") is False

    async def test_explicit_empty_feature_list_denies_entitled_user(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        token = _sign(
            {
                "id": str(uuid.uuid4()),
                "entitled": True,
                "ai_features": [],
            }
        )

        auth = await provider.authenticate({"authorization": f"Bearer {token}"})

        assert auth.features == frozenset()
        assert auth.allows("chat") is False

    async def test_absent_feature_list_uses_legacy_entitlement_fallback(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        token = _sign({"id": str(uuid.uuid4()), "entitled": True})

        auth = await provider.authenticate({"authorization": f"Bearer {token}"})

        assert auth.features is None
        assert auth.allows("chat") is True


class TestJWTAuthProviderRejections:
    async def test_missing_authorization_header_rejected(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        with pytest.raises(AuthenticationError):
            await provider.authenticate({})

    async def test_malformed_authorization_header_rejected(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        with pytest.raises(AuthenticationError):
            await provider.authenticate({"authorization": "NotBearer abc123"})

    async def test_wrong_secret_rejected(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        token = _sign({"id": str(uuid.uuid4())}, secret="wrong-secret")

        with pytest.raises(AuthenticationError):
            await provider.authenticate({"authorization": f"Bearer {token}"})

    async def test_expired_token_rejected(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        token = _sign({"id": str(uuid.uuid4()), "exp": int(time.time()) - 60})

        with pytest.raises(AuthenticationError):
            await provider.authenticate({"authorization": f"Bearer {token}"})

    async def test_missing_exp_rejected(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        token = _sign({"id": str(uuid.uuid4())}, include_exp=False)

        with pytest.raises(AuthenticationError):
            await provider.authenticate({"authorization": f"Bearer {token}"})

    async def test_string_false_privilege_claim_is_rejected(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        token = _sign({"id": str(uuid.uuid4()), "adult_eligible": "false"})

        with pytest.raises(AuthenticationError, match="JSON boolean"):
            await provider.authenticate({"authorization": f"Bearer {token}"})

    async def test_missing_user_id_claim_rejected(self) -> None:
        provider = JWTAuthProvider(_make_settings())
        token = _sign({"role": "user", "email": "a@b.com"})  # no "id" claim

        with pytest.raises(AuthenticationError, match="missing the expected"):
            await provider.authenticate({"authorization": f"Bearer {token}"})

    async def test_non_uuid_user_id_rejected_with_clear_message(self) -> None:
        """
        Guards the exact open question flagged to the backend team:
        if their real user.id turns out not to be a UUID (e.g. an
        autoincrement integer), this must fail with a clear,
        actionable error — not silently coerce or crash obscurely.
        """
        provider = JWTAuthProvider(_make_settings())
        token = _sign({"id": "12345", "role": "user"})  # not a UUID

        with pytest.raises(AuthenticationError, match="not a valid UUID"):
            await provider.authenticate({"authorization": f"Bearer {token}"})

    async def test_missing_secret_raises_at_construction(self) -> None:
        with pytest.raises(RuntimeError, match="JWT_SECRET is not set"):
            JWTAuthProvider(_make_settings(JWT_SECRET=""))


class TestInternalServiceTokenAuthProvider:
    async def test_trusted_backend_context(self) -> None:
        settings = _make_settings(
            AUTH_MODE=AuthMode.internal_service_token,
            INTERNAL_SERVICE_TOKEN="backend-secret",
        )
        provider = InternalServiceTokenAuthProvider(settings)
        user_id = uuid.uuid4()

        auth = await provider.authenticate(
            {
                "authorization": "Bearer backend-secret",
                "x-backend-user-id": str(user_id),
                "x-backend-adult-eligible": "false",
                "x-backend-entitled": "true",
                "x-backend-ai-features": "voice_input,image",
            }
        )

        assert auth.user_id == user_id
        assert auth.adult_eligible is False
        assert auth.entitled is True
        assert auth.features == frozenset({"voice_input", "image"})

    async def test_explicit_empty_backend_features_are_authoritative(self) -> None:
        settings = _make_settings(
            AUTH_MODE=AuthMode.internal_service_token,
            INTERNAL_SERVICE_TOKEN="backend-secret",
        )
        provider = InternalServiceTokenAuthProvider(settings)

        auth = await provider.authenticate(
            {
                "authorization": "Bearer backend-secret",
                "x-backend-user-id": str(uuid.uuid4()),
                "x-backend-entitled": "true",
                "x-backend-ai-features": "",
            }
        )

        assert auth.features == frozenset()
        assert auth.allows("chat") is False

    async def test_invalid_backend_boolean_fails_closed(self) -> None:
        settings = _make_settings(
            AUTH_MODE=AuthMode.internal_service_token,
            INTERNAL_SERVICE_TOKEN="backend-secret",
        )
        provider = InternalServiceTokenAuthProvider(settings)

        with pytest.raises(AuthenticationError):
            await provider.authenticate(
                {
                    "authorization": "Bearer backend-secret",
                    "x-backend-user-id": str(uuid.uuid4()),
                    "x-backend-entitled": "yes",
                }
            )
