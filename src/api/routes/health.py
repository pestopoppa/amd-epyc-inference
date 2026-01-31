"""Health check endpoint with backend health aggregation."""

from fastapi import APIRouter

from src.api.models import HealthResponse
from src.api.state import get_state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Reports overall status and per-backend circuit breaker state.
    Status is "ok" when all tracked backends are healthy, "degraded"
    when any circuit is open or half-open.
    """
    state = get_state()
    tracker = state.health_tracker

    backend_health = tracker.get_status() if tracker else {}
    backends_healthy = sum(
        1 for s in backend_health.values() if s["state"] == "closed"
    )
    backends_total = len(backend_health)

    if backends_total == 0:
        status = "ok"
    elif backends_healthy == backends_total:
        status = "ok"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        models_loaded=backends_healthy,
        mock_mode_available=True,
        version="0.1.0",
        backend_health=backend_health,
    )
