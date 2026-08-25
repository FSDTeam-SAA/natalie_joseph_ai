.PHONY: up down build logs migrate seed run test lint fmt shell

up:
	docker compose up --build

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

migrate:
	docker compose exec app alembic upgrade head

seed:
	docker compose exec app python -m app.scripts.seed_companions

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -v

lint:
	ruff check .

fmt:
	black .
	ruff check --fix .

shell:
	docker compose exec app /bin/bash
