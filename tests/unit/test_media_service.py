from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import NotFoundError
from app.core.security import AuthContext
from app.services.media_service import MediaService


def _auth(user_id: uuid.UUID) -> AuthContext:
    return AuthContext(
        user_id=user_id,
        adult_eligible=False,
        entitled=True,
        raw_claims={},
    )


async def test_private_media_is_loaded_only_after_owned_database_lookup() -> None:
    external_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4())
    media_id = uuid.uuid4()
    asset = SimpleNamespace(
        id=media_id,
        storage_key="0" * 32 + ".png",
        mime_type="image/png",
    )
    storage = SimpleNamespace(get=AsyncMock(return_value=b"private image"))
    media_repo = SimpleNamespace(get_by_id_for_user=AsyncMock(return_value=asset))
    service = MediaService(
        user_repo=SimpleNamespace(get_by_external_user_id=AsyncMock(return_value=user)),
        media_repo=media_repo,
        storage=storage,
    )

    result = await service.get_for_user(_auth(external_id), media_id)

    media_repo.get_by_id_for_user.assert_awaited_once_with(media_id, user.id)
    storage.get.assert_awaited_once_with(asset.storage_key)
    assert result.data == b"private image"
    assert result.mime_type == "image/png"


async def test_unknown_or_foreign_media_never_reads_storage() -> None:
    user = SimpleNamespace(id=uuid.uuid4())
    storage = SimpleNamespace(get=AsyncMock())
    service = MediaService(
        user_repo=SimpleNamespace(get_by_external_user_id=AsyncMock(return_value=user)),
        media_repo=SimpleNamespace(get_by_id_for_user=AsyncMock(return_value=None)),
        storage=storage,
    )

    with pytest.raises(NotFoundError, match="Media file not found"):
        await service.get_for_user(_auth(uuid.uuid4()), uuid.uuid4())

    storage.get.assert_not_awaited()
