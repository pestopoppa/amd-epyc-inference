"""Statistics endpoints."""

from fastapi import APIRouter, Depends

from src.api.dependencies import dep_app_state
from src.api.models import StatsResponse
from src.api.state import AppState

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats(state: AppState = Depends(dep_app_state)) -> StatsResponse:
    """Get API usage statistics."""
    stats = state.get_stats()
    return StatsResponse(**stats)


@router.post("/stats/reset")
async def reset_stats(state: AppState = Depends(dep_app_state)) -> dict[str, str]:
    """Reset API usage statistics."""
    state.reset_stats()
    return {"status": "reset"}
