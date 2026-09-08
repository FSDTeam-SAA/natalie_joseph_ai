from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.deps import get_user_data_service
from app.core.security import AuthContext, get_current_auth_context
from app.services.user_data_service import UserDataService

router = APIRouter(tags=["AI Data"])


@router.delete(
    "/users/me/ai-data",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete all AI-service data for the authenticated backend user",
)
async def delete_my_ai_data(
    auth: AuthContext = Depends(get_current_auth_context),
    service: UserDataService = Depends(get_user_data_service),
) -> None:
    await service.delete_current_user_data(auth)
