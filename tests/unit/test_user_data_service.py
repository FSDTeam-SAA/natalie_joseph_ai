from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.core.security import AuthContext
from app.services.user_data_service import UserDataService


def _auth(user_id: uuid.UUID) -> AuthContext:
    return AuthContext(
        user_id=user_id,
        adult_eligible=False,
        entitled=False,
        raw_claims={"source": "trusted_backend"},
    )


async def test_user_ai_data_deletion_removes_media_then_cascading_database_row() -> None:
    external_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4())
    assets = [
        SimpleNamespace(storage_key="0" * 32 + ".png"),
        SimpleNamespace(storage_key="1" * 32 + ".mp3"),
    ]
    session = SimpleNamespace(commit=AsyncMock())
    user_repo = SimpleNamespace(
        session=session,
        get_by_external_user_id=AsyncMock(return_value=user),
        delete=AsyncMock(),
    )
    media_repo = SimpleNamespace(list_for_user=AsyncMock(return_value=assets))
    storage = SimpleNamespace(delete=AsyncMock())
    service = UserDataService(
        user_repo=user_repo,
        media_repo=media_repo,
        storage=storage,
    )

    await service.delete_current_user_data(_auth(external_id))

    assert storage.delete.await_count == 2
    user_repo.delete.assert_awaited_once_with(user)
    session.commit.assert_awaited_once()


async def test_deleting_absent_user_is_idempotent() -> None:
    user_repo = SimpleNamespace(
        session=SimpleNamespace(commit=AsyncMock()),
        get_by_external_user_id=AsyncMock(return_value=None),
        delete=AsyncMock(),
    )
    storage = SimpleNamespace(delete=AsyncMock())
    service = UserDataService(
        user_repo=user_repo,
        media_repo=SimpleNamespace(list_for_user=AsyncMock()),
        storage=storage,
    )

    await service.delete_current_user_data(_auth(uuid.uuid4()))

    user_repo.delete.assert_not_awaited()
    storage.delete.assert_not_awaited()
    user_repo.session.commit.assert_not_awaited()
