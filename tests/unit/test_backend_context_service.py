from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import AuthorizationError
from app.core.security import AuthContext
from app.services.backend_context_service import BackendContextService


def _auth(*, trusted: bool) -> AuthContext:
    return AuthContext(
        user_id=uuid.uuid4(),
        adult_eligible=False,
        entitled=True,
        raw_claims={"source": "trusted_backend"} if trusted else {},
    )


async def test_resolves_backend_companion_and_creates_durable_conversation() -> None:
    user = SimpleNamespace(id=uuid.uuid4())
    companion = SimpleNamespace(id=uuid.uuid4(), active=True)
    session = SimpleNamespace(commit=AsyncMock())
    conversation_repo = SimpleNamespace(
        session=session,
        get_by_external_id_for_user=AsyncMock(return_value=None),
        external_id_exists=AsyncMock(return_value=False),
        get_latest_for_user_and_companion=AsyncMock(return_value=None),
        acquire_creation_lock=AsyncMock(),
        add=AsyncMock(),
    )

    async def add_conversation(conversation):
        conversation.id = uuid.uuid4()
        return conversation

    conversation_repo.add.side_effect = add_conversation
    companion_repo = SimpleNamespace(
        resolve_backend_reference=AsyncMock(return_value=companion)
    )
    service = BackendContextService(
        user_repo=SimpleNamespace(
            get_or_create_by_external_user_id=AsyncMock(return_value=user)
        ),
        companion_repo=companion_repo,
        conversation_repo=conversation_repo,
        settings=Settings(
            _env_file=None,
            BACKEND_COMPANION_ID_MAP={"backend-lina": "lina"},
        ),
    )

    result = await service.resolve(
        _auth(trusted=True),
        companion_reference="backend-lina",
        external_conversation_id="nestjs-thread-cuid",
    )

    assert result.companion is companion
    assert result.conversation.external_conversation_id is not None
    companion_repo.resolve_backend_reference.assert_awaited_once_with(
        "backend-lina", id_map={"backend-lina": "lina"}
    )
    session.commit.assert_awaited_once()


async def test_backend_context_rejects_direct_client_identity() -> None:
    service = BackendContextService(
        user_repo=SimpleNamespace(),
        companion_repo=SimpleNamespace(),
        conversation_repo=SimpleNamespace(),
        settings=Settings(_env_file=None),
    )
    with pytest.raises(AuthorizationError, match="trusted backend"):
        await service.resolve(
            _auth(trusted=False),
            companion_reference="lina",
            external_conversation_id=None,
        )
