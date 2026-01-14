"""Health check endpoint."""

from fastapi import APIRouter

from src.api.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded=0,  # In mock mode, no models loaded
        mock_mode_available=True,
        version="0.1.0",
    )
