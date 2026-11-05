# Meet Elysia AI Companion Engine

This repository is the AI/media service for Meet Elysia. It owns companion
personas, Grok conversation generation, memory and relationship continuity,
ElevenLabs speech, Grok Imagine image generation with GPT fallback, private media storage,
and companion-world story context.

The main backend owns end-user authentication, profiles, companion access,
adult eligibility, subscriptions/payments/credits, scheduling, notifications,
and usage accounting. This repository contains no frontend. The backend should
therefore authorize each operation before forwarding a trusted user and
companion reference to this service.

## Implemented flows

- Text: trusted identity → persona/history/memory/story context → Grok 4.6 → persistence.
- Voice input: validated audio → ElevenLabs Scribe STT → the same text-chat
  pipeline → optional, explicitly requested ElevenLabs TTS.
- Voice output: a per-companion cloned voice ID, resolved from deployment
  configuration; no unnecessary automatic TTS.
- Images: authorized request/contextual trigger → approved companion reference image +
  visual profile/history/story → Grok Imagine image edit (GPT image fallback) → private storage.
- Proactive interactions: the main backend schedules an eligible notification;
  this service generates the contextual text or optional voice response.
- Living-world continuity: the backend/social pipeline upserts global companion
  story events, which are reused by chat, proactive responses, and images.
- Companion routines: each backend profile's optional weekly routine supplies
  the current companion-local activity to text, voice, proactive replies, and images.

Text, audio, and image assistant messages share one conversation/message model.
Long-term memories use pgvector, rolling summaries and deterministic
relationship stages are maintained after chat turns, and generated media can
only be retrieved by its owning user.

## Local setup

Copy `.env.example` to `.env`, populate development credentials, then run:

```powershell
docker compose up --build
```

The Compose stack starts PostgreSQL with pgvector, Redis, and the API. It
automatically applies migrations and seeds the companion catalogue on startup.
Authentication configuration is read from `.env`; for JWT authentication set
`AUTH_MODE=jwt` and populate the matching `JWT_*` values there. Stop it with
`docker compose down`. Its database and generated local media are stored in
named Docker volumes, so they survive restarts. To reset this local environment
completely, run `docker compose down --volumes`.

Swagger UI is available at `http://localhost:8000/docs`. Liveness and readiness
are `GET /health` and `GET /health/ready`; readiness checks the database, Redis,
provider configuration, auth configuration, and enabled media features.

For direct local calls, set `AUTH_MODE=local_api_key` and send
`Authorization: Bearer <LOCAL_API_KEY>`. This mode is rejected in production.

## Main-backend contract

Recommended production mode:

```env
AUTH_MODE=internal_service_token
INTERNAL_SERVICE_TOKEN=<long randomly generated secret>
BACKEND_COMPANION_ID_MAP={"<backend Elena ID>":"elena","<backend Chloé ID>":"chloe","<backend Thalia ID>":"thalia","<backend Lina ID>":"lina","<backend Luna ID>":"luna"}
```

Every private request must be made over a protected network/TLS connection with:

```text
Authorization: Bearer <INTERNAL_SERVICE_TOKEN>
X-Backend-User-Id: <UUID authenticated by the main backend>
X-Backend-Adult-Eligible: true|false
X-Backend-Entitled: true|false
X-Backend-AI-Features: chat,voice_input,voice_output,image,proactive,story_context
```

When a feature list is present it is authoritative. The main backend must check
companion access and reserve/debit plan usage or credits before making an
expensive call. Every generation request includes a UUID idempotency key so a
backend retry does not purchase duplicate output.

The adapter endpoints accept the backend's companion ID (mapped to a stable AI
slug) and an optional string/UUID/cuid `external_conversation_id`:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/internal/companions/{ref}/messages` | Text chat or explicit image request |
| `POST` | `/api/v1/internal/companions/{ref}/voice` | Text chat plus requested TTS |
| `POST` | `/api/v1/internal/companions/{ref}/voice-input` | Multipart audio STT, chat, and optional TTS |
| `POST` | `/api/v1/internal/companions/{ref}/images` | Requested or contextual image |
| `POST` | `/api/v1/internal/companions/{ref}/proactive` | Scheduled text/voice check-in |
| `PUT` | `/api/v1/internal/companions/{ref}/story-events` | Idempotent public-story event upsert |

If the main backend does not supply a conversation ID, the adapter reuses one
durable conversation per user/companion. The original UUID-based endpoints
remain available under `/api/v1/chat`, `/api/v1/chat/voice`,
`/api/v1/images/generate`, `/api/v1/media/{media_id}`, and
`/api/v1/conversations/*`.

Both message endpoints conservatively recognize explicit natural-language photo
requests (for example, “Show me what you're wearing tonight”) and return an
image-typed `ChatResponse` with a private media descriptor. They require one
caller-generated UUID `idempotency_key`; the same key is reused through routing
so retries cannot purchase both text and image output.

Direct JWT verification is also implemented, but should only be selected when
the JWT contract supplies a UUID user ID plus authoritative adult, entitlement,
feature, and expiry claims.

## Companion media activation

Provider keys, cloned voice IDs, and licensed reference images are deployment
assets and are intentionally absent from source control.

```env
XAI_API_KEY= # preferred for Grok Imagine image generation
OPENAI_API_KEY= # automatic GPT Image fallback
ELEVENLABS_API_KEY=

COMPANION_VOICE_ID_MAP={"elena":"...","chloe":"...","thalia":"...","lina":"...","luna":"..."}
COMPANION_REFERENCE_IMAGE_MAP={"elena":["elena/reference.webp"],"chloe":["chloe/reference.webp"],"thalia":["thalia/reference.webp"],"lina":["lina/reference.webp"],"luna":["luna/reference.webp"]}

ENABLE_VOICE_INPUT=true
ENABLE_VOICE_GENERATION=true
ENABLE_IMAGE_GENERATION=true
MAX_AUDIO_UPLOAD_BYTES=10485760
MAX_AUDIO_DURATION_SECONDS=300
MAX_TRANSCRIPT_CHARACTERS=4000
```

Reference paths are relative to `COMPANION_ASSET_ROOT`; see
`config/companion_assets/README.md`. Run the companion seed after changing either
mapping. Features fail closed if a required provider key, voice ID, reference
asset, or backend grant is missing.
`/api/v1/chat` is the unified public chat endpoint. It continues to accept its
original JSON text body and also accepts `multipart/form-data` with an optional
`audio` file. Explicit photo requests automatically return an image; audio
input automatically returns a spoken reply. Voice IDs remain server-only
deployment configuration.

`MEDIA_STORAGE_BACKEND=local` writes opaque files under `MEDIA_STORAGE_ROOT`.
For production, mount that directory as durable private storage. A multi-instance
deployment should add an object-storage adapter rather than sharing public paths.

## Database and tests

Apply migrations in order and seed the five companions (Elena, Chloé, Thalia,
Lina, and Luna):

```powershell
.\venv\Scripts\alembic.exe upgrade head
.\venv\Scripts\python.exe -m app.scripts.seed_companions
```

Run unit tests without external services:

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\ruff.exe check app tests alembic
```

Database integration tests are skipped unless an explicitly isolated, migrated
database is provided. They never inherit `DATABASE_URL` from `.env`:

```powershell
$env:TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/meet_elysia_test"
$env:DATABASE_URL = $env:TEST_DATABASE_URL
if ($env:DATABASE_URL -ne $env:TEST_DATABASE_URL) { throw "Refusing non-test database" }
.\venv\Scripts\alembic.exe upgrade head
.\venv\Scripts\python.exe -m app.scripts.seed_companions
.\venv\Scripts\python.exe -m pytest -q -m integration
```

Do not point `TEST_DATABASE_URL` at staging or production. Provider calls are
mocked in integration tests.

## Operational safeguards

- Redis enforces per-user, per-IP, and short-burst limits and can fail closed.
- PostgreSQL advisory locks plus unique idempotency keys protect paid operations
  from concurrent retries.
- Audio/image size and MIME signatures are validated; reference paths are
  confined to the configured asset root.
- Provider errors are normalized; API responses never expose provider secrets,
  raw stack traces, or local storage paths.
- User/conversation deletion removes both database state and owned media bytes.
- Raw conversation content is excluded from structured application logs.

The API contract supports a separate frontend, but recording/playback/rendering
UI must be implemented in the frontend repository owned by that team.

## Backend-owned companion routines

The catalogue detail response may include `data.routine` for any companion:

```json
{
  "timezone": "Asia/Dubai",
  "weekly": [
    {"day": 1, "start": "12:00", "end": "18:00", "activity": "Work at a café"},
    {"day": 1, "start": "23:00", "end": "06:00", "activity": "Sleep"}
  ]
}
```

Days follow ISO numbering (Monday = 1, Sunday = 7). Times are local `HH:MM`,
with inclusive starts and exclusive ends. An end earlier than its start crosses
midnight: Monday 23:00–06:00 ends Tuesday morning. Sunday overnight slots also
carry into Monday. Each companion uses its own IANA timezone, including daylight
saving transitions; user and server timezones do not choose the activity.

The AI service resolves one activity per new generation. It does not calculate
sub-schedules for combined labels such as "Coffee at home, gym". Current routine
context takes precedence over older whereabouts in conversation/story context.
Current selfies use that activity as their default scene; explicit imagined or
past scene requests may differ. A sleep slot still allows a conversational reply.

Missing/null routines, empty schedules, gaps, or invalid schedules contribute no
routine prompt section, preserving the existing response flow. Invalid timezones,
blank activities, equal start/end times, and overlapping slots (including across
midnight/week boundaries) cause the optional routine to be ignored and emit a
content-free warning. Admin input should be validated by the main backend too.

Profiles are fetched from the backend without a local profile cache; slug lookups
fetch the full detail after finding the companion ID. Admin edits apply to the
next new generation once the catalogue returns the update. Completed idempotent
retries replay their original output. No routine database migration, per-companion
code, new endpoint, or timer is required; automatic messages/photos remain owned
by the main backend's existing scheduling and authorization flows.

Current routine replies also pass a semantic consistency check using the configured
background model. The latest resolved scene is repeated after conversation history
so older locations and times do not set the current scene. Ordinary activity
questions receive an in-character answer; direct questions about AI identity still
receive a truthful disclosure. The companion uses its own local daypart and the
user's language.

An inconsistent draft is regenerated once in the original persona and checked
again. If the checker is unavailable, returns malformed data, or the second draft
still conflicts, the API returns a provider error rather than persisting or speaking
an unverified draft. Active routine turns therefore incur one additional background
model call normally, or two checks plus a second generation when repaired. Missing
or unmatched routines use the previous generation path without these extra calls.

The `chat_completion` and `proactive_message` AI event metadata records
`routine_context` (the exact local time, daypart, activity and slot) and
`routine_validation` (verdicts, checker usage, policy version and whether a repair
was needed). Main generation token counts include both drafts when repaired;
checker tokens are recorded separately in the metadata. Standard logs include
slot/time identifiers and rejection reason codes, without raw conversation text.

Image prompts use the same resolved snapshot after user scene/style context so
present-moment selfies follow the activity and local time rather than stale
evening aesthetics. Image asset/event metadata also records `routine_context`.
This does not add an automatic visual verifier for generated pixels.
