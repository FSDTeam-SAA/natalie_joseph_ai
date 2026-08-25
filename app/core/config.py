"""
Central application configuration.

CRITICAL RULE (per project spec, Section 6 / 52 / 62):
Nothing in this file may hard-code model IDs, prompts, secrets, or
business logic. Every value here is sourced from environment variables
and consumed elsewhere via dependency injection on the `Settings` object.

Unresolved / pending items are explicitly marked TODO-CONFIRM below.
Do not fill these with guesses — they must come from the backend team
or product owner before the relevant phase is implemented.
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class AuthMode(str, Enum):
    """
    Supported authentication modes for incoming requests to this AI service.

    jwt: The main backend issues a signed JWT (confirmed by product owner).
         Verification details (algorithm, issuer, audience, public key /
         shared secret, and the exact claim names for user_id and
         adult_eligible) are TODO-CONFIRM — see AUTH section below.
    local_api_key: Simple static API key, local development only.
    internal_service_token: Static shared-secret bearer token, for
         service-to-service calls that are not per-user JWTs
         (e.g. internal maintenance/deletion endpoints).
    """

    jwt = "jwt"
    local_api_key = "local_api_key"
    internal_service_token = "internal_service_token"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    APP_ENV: AppEnv = AppEnv.development
    APP_NAME: str = "meet-elysia-ai"
    LOG_LEVEL: str = "INFO"
    API_V1_PREFIX: str = "/api/v1"

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/meet_elysia",
        description="Async SQLAlchemy connection string (asyncpg driver).",
    )
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 5
    DATABASE_SSL_MODE: str = Field(
        default="prefer",
        description=(
            "asyncpg ssl mode: 'disable', 'prefer', or 'require'. "
            "Hosted providers (e.g. Supabase) require 'require'."
        ),
    )
    DATABASE_DISABLE_STATEMENT_CACHE: bool = Field(
        default=False,
        description=(
            "Set true when connecting through a PgBouncer transaction-mode "
            "pooler (e.g. Supabase's port-6543 pooler), which does not "
            "support asyncpg's prepared-statement caching. Not needed for "
            "a direct connection (Supabase port 5432) or self-hosted Postgres."
        ),
    )

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------------
    # OpenAI — provider credentials
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = ""

    # ------------------------------------------------------------------
    # OpenAI — model configuration
    # Confirmed by product owner (2026-08-24):
    #   Chat                -> GPT-5.6 Terra
    #   Memory extraction    -> GPT-5.6 Luna
    #   Conversation summary -> GPT-5.6 Luna
    #   Emotion/context      -> GPT-5.6 Luna
    #   Evaluation           -> GPT-5.6 Sol
    # These map onto the three model "roles" defined in the original spec:
    #   OPENAI_CHAT_MODEL        -> primary conversational generation
    #   OPENAI_BACKGROUND_MODEL  -> memory extraction, summarization,
    #                                emotion/context classification
    #                                (all lower-cost background tasks)
    #   OPENAI_REASONING_MODEL   -> evaluation / regression suite
    # No model ID is ever referenced directly in business logic — all
    # call sites read these settings fields.
    # ------------------------------------------------------------------
    OPENAI_CHAT_MODEL: str = "gpt-5.6-terra"
    OPENAI_BACKGROUND_MODEL: str = "gpt-5.6-luna"
    OPENAI_REASONING_MODEL: str = "gpt-5.6-sol"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MODERATION_MODEL: str = "omni-moderation-latest"
    OPENAI_IMAGE_MODEL: str = "gpt-image-2"

    # ------------------------------------------------------------------
    # Auth
    # Confirmed by product owner (2026-08-24): AUTH_MODE = jwt.
    # user_id / conversation_id / companion_id are UUIDs.
    # adult_eligible is delivered via JWT claim.
    #
    # TODO-CONFIRM before Phase 5/9 implementation (still needed from
    # backend team — not guessed):
    #   - JWT signing algorithm (HS256 shared secret vs RS256/ES256
    #     public key)
    #   - JWT_PUBLIC_KEY or JWT_SECRET value/source (e.g. JWKS URL vs
    #     static key)
    #   - Expected `iss` (issuer) and `aud` (audience) claim values
    #   - Exact claim name carrying the internal user UUID
    #     (e.g. "sub" vs "user_id")
    #   - Exact claim name carrying the adult-eligibility flag
    #     (e.g. "adult_eligible" vs "entitlements.adult")
    #   - Exact claim name carrying the entitlement/subscription flag
    #   - Token transport: `Authorization: Bearer <jwt>` assumed but
    #     not yet confirmed
    #
    # Until confirmed, `security.py` implements NO verification logic —
    # only the interface — per Section 66 ("do not fabricate
    # functionality").
    # ------------------------------------------------------------------
    AUTH_MODE: AuthMode = AuthMode.jwt
    JWT_ALGORITHM: str = Field(
        default="RS256",
        description="TODO-CONFIRM with backend team before Phase 5.",
    )
    JWT_PUBLIC_KEY: str = Field(
        default="",
        description="TODO-CONFIRM: PEM public key or JWKS URL from backend team.",
    )
    JWT_SECRET: str = Field(
        default="",
        description="Only used if JWT_ALGORITHM is HS*. TODO-CONFIRM.",
    )
    JWT_ISSUER: str = Field(default="", description="TODO-CONFIRM.")
    JWT_AUDIENCE: str = Field(default="", description="TODO-CONFIRM.")
    JWT_USER_ID_CLAIM: str = Field(
        default="sub",
        description="TODO-CONFIRM exact claim name with backend team.",
    )
    JWT_ADULT_ELIGIBLE_CLAIM: str = Field(
        default="adult_eligible",
        description="TODO-CONFIRM exact claim name with backend team.",
    )
    JWT_ENTITLED_CLAIM: str = Field(
        default="entitled",
        description="TODO-CONFIRM exact claim name with backend team.",
    )

    # Local dev / internal service fallback modes (still supported per
    # spec Section 30, independent of JWT mode)
    LOCAL_API_KEY: str = ""
    INTERNAL_SERVICE_TOKEN: str = ""

    # ------------------------------------------------------------------
    # Memory / context tuning
    # ------------------------------------------------------------------
    MEMORY_TOP_K: int = 5
    MEMORY_MIN_SCORE: float = 0.75
    RECENT_MESSAGE_LIMIT: int = 20
    SUMMARY_TRIGGER_MESSAGE_COUNT: int = 30
    AI_DATA_RETENTION_DAYS: int = 365

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    ENABLE_IMAGE_GENERATION: bool = False

    # ------------------------------------------------------------------
    # Rate limiting (Redis-backed, configured not hard-coded)
    # ------------------------------------------------------------------
    RATE_LIMIT_PER_USER_PER_MINUTE: int = 20
    RATE_LIMIT_PER_IP_PER_MINUTE: int = 60
    RATE_LIMIT_CONVERSATION_BURST: int = 5

    # ------------------------------------------------------------------
    # Prompt versioning (spec Section 34)
    # ------------------------------------------------------------------
    PROMPT_VERSION: str = "elysia-v1"

    @field_validator("DATABASE_URL")
    @classmethod
    def _require_asyncpg_driver(cls, v: str) -> str:
        """
        Fail fast with a clear message instead of a confusing
        ModuleNotFoundError deep inside SQLAlchemy. Supabase and most
        other providers give you a plain `postgresql://` string by
        default — it must be `postgresql+asyncpg://` for this async
        service.
        """
        if v.startswith("postgresql://") or v.startswith("postgres://"):
            raise ValueError(
                "DATABASE_URL must use the asyncpg driver: "
                "'postgresql+asyncpg://...', not 'postgresql://...' or "
                "'postgres://...'. This commonly happens when pasting a "
                "connection string directly from a provider's dashboard "
                "(e.g. Supabase) — just add '+asyncpg' right after "
                "'postgresql'."
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor for use as a FastAPI dependency."""
    return Settings()
