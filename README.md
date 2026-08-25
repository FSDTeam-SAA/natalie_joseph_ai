# Meet Elysia — AI Companion Engine

AI backend service for the Meet Elysia companion platform. This repo
owns the AI layer only: companion personas, conversation orchestration,
memory, context assembly, and safety moderation. It does **not** own
the frontend, payments UI, or authentication UI — those belong to the
main backend/frontend team.

This README will grow with each phase. Right now it documents **Phase 1**.

---

## Phase 1 scope (delivered)

- Project scaffold matching the full target directory structure
- `app/core/config.py` — typed settings, all confirmed env vars, all
  pending items explicitly marked `TODO-CONFIRM`
- `app/core/logging.py` — structured JSON logging, never logs raw
  conversation content
- `app/core/exceptions.py` — full application error hierarchy
- `app/core/security.py` — `AuthProvider` **interface only**; JWT
  verification is intentionally not implemented yet (see below)
- `app/db/base.py`, `app/db/session.py` — async SQLAlchemy setup, no
  models yet (Phase 2)
- `app/api/health.py`, `app/main.py` — `/health` and `/health/ready`,
  wired into a real FastAPI app with global error handlers
- `Dockerfile`, `docker-compose.yml` (app, postgres+pgvector, redis,
  worker placeholder)
- `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako` — ready for
  Phase 2 migrations
- `requirements.txt`, `pyproject.toml`, `Makefile`
- `.env.example`, `.gitignore`
- `tests/unit/test_health.py` — passing smoke test

**Verified working** (not just written): dependencies installed
cleanly in a fresh venv, `app.main` imports without error, the app
boots under uvicorn, `/health` returns `200 {"status": "ok"}`, and
`pytest` passes.

## What is intentionally NOT in Phase 1

- No database models or migrations yet (Phase 2)
- No companion configs or companion API yet (Phase 3)
- No chat endpoint, no LLM calls yet (Phases 4–5)
- **No working authentication.** `AuthProvider` is an interface with a
  `NotConfiguredAuthProvider` that always fails closed. Real JWT
  verification needs the items listed below from the backend team.

---

## Running locally

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY once Phase 4 needs it.
# JWT_* fields remain blank until backend team confirms details (see below).

docker compose up --build
```

Swagger UI: http://localhost:8000/docs
Health: http://localhost:8000/health and http://localhost:8000/health/ready

## Running tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

---

## Backend integration contract — what's confirmed vs. still needed

### Confirmed (2026-08-24)
| Item | Value |
|---|---|
| Auth mechanism | JWT |
| `user_id`, `conversation_id`, `companion_id` format | UUID |
| Adult eligibility delivery | JWT claim |
| Chat model | GPT-5.6 Terra |
| Memory extraction model | GPT-5.6 Luna |
| Conversation summary model | GPT-5.6 Luna |
| Emotion/context classification model | GPT-5.6 Luna |
| Evaluation model | GPT-5.6 Sol |

### Still needed from the backend team before Phase 5 (auth) can be implemented
- JWT signing algorithm (HS256 shared secret vs RS256/ES256 public key / JWKS)
- The actual public key, JWKS URL, or shared secret
- Expected `iss` (issuer) and `aud` (audience) claim values
- Exact claim name carrying the user UUID (e.g. `sub` vs `user_id`)
- Exact claim name carrying the adult-eligibility flag
- Exact claim name carrying the entitlement/subscription flag
- Confirmation that the token is sent as `Authorization: Bearer <jwt>`

### Still needed before later phases
- Whether the main backend needs webhooks/callbacks for background job
  results (memory extraction, summarization) or is purely fire-and-forget
- Any existing per-companion visual/image asset pipeline details, if
  Phase 12 image generation is ever turned on

## Division of responsibility

**AI developer (this repo) owns:**
Everything under `app/`, `config/`, `alembic/`, provider abstractions,
memory/context/safety logic, this service's own database, Docker setup
for this service, and this service's tests.

**Backend team owns:**
Issuing and signing JWTs, deciding entitlement/adult-eligibility logic
upstream, the frontend, payments, and calling this service's `/api/v1/*`
endpoints with correctly-authenticated requests.

---

## Full phase plan

1. **Foundation** (this phase) — scaffold, config, Docker, health checks
2. **Data layer** — SQLAlchemy models, Alembic migration, repositories
3. **Companion configuration & API** — 5 companion JSON configs, companion endpoints
4. **Provider layer** — LLMProvider/EmbeddingProvider/ModerationProvider/ImageProvider (OpenAI)
5. **Conversation core + basic chat** — conversation endpoints, non-streaming chat, real auth
6. **Context assembly & prompt builder**
7. **Memory engine** — extraction, dedup, embeddings, retrieval
8. **Summarization + relationship context + background jobs**
9. **Safety layer** — moderation pipeline, injection defense, dependency safeguards, age gating
10. **Streaming + rate limiting + cost tracking**
11. **User data deletion + matching engine**
12. **Testing, evaluation suites, docs, final polish**

## Known limitations (Phase 1)

- Service has no real authentication yet — every request would be
  rejected by design until Phase 5.
- No database tables exist yet — `/health/ready` will report a DB
  connection error until Phase 2's migration is applied (it can still
  connect if Postgres is up; it just has no schema).
- Celery worker command in `docker-compose.yml` references
  `app.workers.celery_app`, which doesn't exist until Phase 8 — the
  `worker` service will fail to start until then. This is expected and
  documented, not a bug to chase in Phase 1.
