"""Statistics endpoints."""

from fastapi import APIRouter

from src.api.models import StatsResponse
from src.api.state import get_state

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get API usage statistics."""
    state = get_state()
    stats = state.get_stats()
    return StatsResponse(**stats)


@router.post("/stats/reset")
async def reset_stats() -> dict[str, str]:
    """Reset API usage statistics."""
    state = get_state()
    state.reset_stats()
    return {"status": "reset"}
