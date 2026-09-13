"""
Central application configuration.

CRITICAL RULE (per project spec, Section 6 / 52 / 62):
Nothing in this file may hard-code model IDs, prompts, secrets, or
business logic. Every value here is sourced from environment variables
and consumed elsewhere via dependency injection on the `Settings` object.

Deployment-owned identifiers and secrets remain empty until supplied through
the environment; the application never invents them.
"""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class AuthMode(str, Enum):
    """
    Supported authentication modes for incoming requests to this AI service.

    jwt: Verify end-user JWTs directly when the backend's complete claim
         contract is available.
    local_api_key: Simple static API key, local development only.
    internal_service_token: Authenticate the main backend, which forwards a
         verified user ID and its authorization decisions. This is the
         recommended production boundary when the backend owns auth/billing.
    """

    jwt = "jwt"
    local_api_key = "local_api_key"
    internal_service_token = "internal_service_token"


class MediaStorageBackend(str, Enum):
    """Storage adapters supported by this service."""

    local = "local"
    cloudinary = "cloudinary"


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
    # Companion catalogue
    # The main companion backend is the source of truth for public profiles.
    # ------------------------------------------------------------------
    COMPANION_CATALOGUE_BASE_URL: str = "https://natalieapi.duckdns.org/api/v1"
    COMPANION_CATALOGUE_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=300)

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
    OPENAI_CHAT_MODEL: str = "gpt-5.6-terra"

    # ------------------------------------------------------------------
    # xAI — conversation and preferred image-generation provider.
    # ------------------------------------------------------------------
    XAI_API_KEY: str = ""
    XAI_BASE_URL: str = "https://api.x.ai/v1"
    XAI_MODEL: str = "grok-4.6"
    XAI_REASONING_EFFORT: Literal["low", "medium", "high", "xhigh"] = "low"
    XAI_IMAGE_MODEL: str = "grok-imagine-image-2.0"
    CHAT_MAX_OUTPUT_TOKENS: int = Field(default=800, gt=0, le=8192)
    PROVIDER_TIMEOUT_SECONDS: float = Field(default=45.0, gt=0, le=300)
    PROVIDER_MAX_RETRIES: int = Field(default=2, ge=0, le=10)

    # ------------------------------------------------------------------
    # OpenAI — model configuration
    # Confirmed by product owner (2026-08-24):
    #   Memory extraction    -> GPT-5.6 Luna
    #   Conversation summary -> GPT-5.6 Luna
    #   Emotion/context      -> GPT-5.6 Luna
    #   Evaluation           -> GPT-5.6 Sol
    # OpenAI is no longer used for user-facing conversation. It remains for:
    #   OPENAI_BACKGROUND_MODEL  -> memory extraction, summarization,
    #                                emotion/context classification
    #                                (all lower-cost background tasks)
    #   OPENAI_REASONING_MODEL   -> evaluation / regression suite
    # No model ID is ever referenced directly in business logic — all
    # call sites read these settings fields.
    # ------------------------------------------------------------------
    OPENAI_BACKGROUND_MODEL: str = "gpt-5.6-luna"
    OPENAI_REASONING_MODEL: str = "gpt-5.6-sol"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_IMAGE_MODEL: str = "gpt-image-2"

    # ------------------------------------------------------------------
    # Auth
    # The main backend owns login, age eligibility, plans/credits, profile,
    # notifications, and companion access. For that deployment topology use
    # internal_service_token and the X-Backend-* contract documented in the
    # README. Direct JWT verification remains supported as an alternative and
    # fails closed when eligibility/entitlement claims are absent.
    # ------------------------------------------------------------------
    AUTH_MODE: AuthMode = AuthMode.jwt
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="Confirmed via backend team's NestJS source (2026-08-29).",
    )
    JWT_PUBLIC_KEY: str = Field(
        default="",
        description="Not used — backend uses a symmetric HS256 secret, not a key pair.",
    )
    JWT_SECRET: str = Field(
        default="",
        description="Set this to the backend team's actual ACCESS_TOKEN_SECRET value.",
    )
    JWT_ISSUER: str = Field(
        default="",
        description="Confirmed unused — backend does not set an iss claim. Leave blank.",
    )
    JWT_AUDIENCE: str = Field(
        default="",
        description="Confirmed unused — backend does not set an aud claim. Leave blank.",
    )
    JWT_USER_ID_CLAIM: str = Field(
        default="id",
        description="Confirmed via backend team's NestJS source (2026-08-29).",
    )
    JWT_ADULT_ELIGIBLE_CLAIM: str = Field(
        default="adult_eligible",
        description="Claim carrying the backend's adult-eligibility decision; "
        "missing values fail closed.",
    )
    JWT_ENTITLED_CLAIM: str = Field(
        default="entitled",
        description="Legacy claim carrying the backend's entitlement decision; "
        "missing values fail closed.",
    )
    JWT_AI_FEATURES_CLAIM: str = Field(
        default="ai_features",
        description=(
            "Optional backend-authorized feature list (for example voice_input, "
            "voice_output, image). The AI service never computes plan access."
        ),
    )
    JWT_REQUIRE_EXP: bool = True

    # Local dev / internal service fallback modes (still supported per
    # spec Section 30, independent of JWT mode)
    LOCAL_API_KEY: str = ""
    INTERNAL_SERVICE_TOKEN: str = ""
    BACKEND_COMPANION_ID_MAP: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "JSON object mapping companion IDs owned by the main backend to this "
            "service's stable slugs, for example {\"backend-id\": \"lina\"}."
        ),
    )
    COMPANION_VOICE_ID_MAP: dict[str, str] = Field(
        default_factory=dict,
        description="JSON object mapping stable companion slugs to ElevenLabs voice IDs.",
    )
    COMPANION_REFERENCE_IMAGE_MAP: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "JSON object mapping stable companion slugs to paths relative to "
            "COMPANION_ASSET_ROOT."
        ),
    )

    # ------------------------------------------------------------------
    # Voice — ElevenLabs STT/TTS
    # ------------------------------------------------------------------
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_BASE_URL: str = "https://api.elevenlabs.io"
    ELEVENLABS_STT_MODEL: str = "scribe_v2"
    ELEVENLABS_TTS_MODEL: str = "eleven_flash_v2_5"
    ELEVENLABS_OUTPUT_FORMAT: str = "mp3_44100_128"
    ENABLE_VOICE_INPUT: bool = False
    MAX_AUDIO_UPLOAD_BYTES: int = Field(default=10 * 1024 * 1024, gt=0)
    MAX_AUDIO_DURATION_SECONDS: float = Field(default=300.0, gt=0, le=3600)
    MAX_TRANSCRIPT_CHARACTERS: int = Field(default=4000, gt=0, le=4000)
    MAX_TTS_CHARACTERS: int = Field(default=4000, gt=0, le=10000)

    # ------------------------------------------------------------------
    # Memory / context tuning
    # ------------------------------------------------------------------
    MEMORY_TOP_K: int = 5
    MEMORY_MIN_SCORE: float = 0.75
    MEMORY_MIN_CONFIDENCE: float = Field(default=0.6, ge=0.0, le=1.0)
    RECENT_MESSAGE_LIMIT: int = 20
    SUMMARY_TRIGGER_MESSAGE_COUNT: int = 30
    SUMMARY_MAX_OUTPUT_TOKENS: int = Field(default=600, gt=0, le=4096)
    RELATIONSHIP_FAMILIAR_AFTER_MESSAGES: int = Field(default=20, gt=0)
    RELATIONSHIP_ESTABLISHED_AFTER_MESSAGES: int = Field(default=100, gt=0)
    RELATIONSHIP_DEEP_AFTER_MESSAGES: int = Field(default=200, gt=0)
    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    ENABLE_IMAGE_GENERATION: bool = False
    ENABLE_VOICE_GENERATION: bool = False

    # ------------------------------------------------------------------
    # Image generation + private media storage
    # ------------------------------------------------------------------
    IMAGE_OUTPUT_SIZE: str = "1024x1536"
    IMAGE_OUTPUT_QUALITY: str = "medium"
    MAX_REFERENCE_IMAGE_BYTES: int = Field(default=20 * 1024 * 1024, gt=0)
    COMPANION_ASSET_ROOT: str = "config/companion_assets"
    COMPANION_REFERENCE_IMAGE_ALLOWED_HOSTS: list[str] = ["res.cloudinary.com"]

    MEDIA_STORAGE_BACKEND: MediaStorageBackend = MediaStorageBackend.local
    MEDIA_STORAGE_ROOT: str = "storage/media"
    MEDIA_URL_PREFIX: str = "/api/v1/media"
    MAX_GENERATED_MEDIA_BYTES: int = Field(default=25 * 1024 * 1024, gt=0)
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""
    CLOUDINARY_FOLDER: str = "meet-elysia"

    # ------------------------------------------------------------------
    # Rate limiting (Redis-backed, configured not hard-coded)
    # ------------------------------------------------------------------
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_FAIL_CLOSED: bool = True
    RATE_LIMIT_PER_USER_PER_MINUTE: int = 20
    RATE_LIMIT_PER_IP_PER_MINUTE: int = 60
    RATE_LIMIT_CONVERSATION_BURST: int = 5

    # ------------------------------------------------------------------
    # Prompt versioning (spec Section 34)
    # ------------------------------------------------------------------
    PROMPT_VERSION: str = "elysia-v2"

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
