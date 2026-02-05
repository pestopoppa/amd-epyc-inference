"""Health check endpoint with backend health aggregation."""

import logging
from typing import Any

from fastapi import APIRouter, Depends

from src.api.dependencies import dep_health_tracker
from src.api.health_tracker import BackendHealthTracker
from src.api.models import HealthResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Cache knowledge tool status at module load to avoid repeated import checks
_knowledge_tools_status: dict[str, Any] | None = None


def _check_knowledge_tools() -> dict[str, Any]:
    """Check availability of knowledge tools dependencies.

    Returns status dict with each tool's availability and any error messages.
    This is called once at startup and cached.
    """
    global _knowledge_tools_status
    if _knowledge_tools_status is not None:
        return _knowledge_tools_status

    status: dict[str, Any] = {
        "available": True,
        "tools": {},
    }

    # Check arxiv
    try:
        import arxiv  # noqa: F401

        status["tools"]["search_arxiv"] = {"available": True, "error": None}
    except ImportError as e:
        status["tools"]["search_arxiv"] = {"available": False, "error": str(e)}
        status["available"] = False

    # Check semanticscholar
    try:
        import semanticscholar  # noqa: F401

        status["tools"]["search_papers"] = {"available": True, "error": None}
    except ImportError as e:
        status["tools"]["search_papers"] = {"available": False, "error": str(e)}
        status["available"] = False

    # Check mwclient (Wikipedia)
    try:
        import mwclient  # noqa: F401

        status["tools"]["search_wikipedia"] = {"available": True, "error": None}
        status["tools"]["get_wikipedia_article"] = {"available": True, "error": None}
    except ImportError as e:
        status["tools"]["search_wikipedia"] = {"available": False, "error": str(e)}
        status["tools"]["get_wikipedia_article"] = {"available": False, "error": str(e)}
        status["available"] = False

    # Check google-api-python-client (Books)
    try:
        from googleapiclient.discovery import build  # noqa: F401

        status["tools"]["search_books"] = {"available": True, "error": None}
    except ImportError as e:
        status["tools"]["search_books"] = {"available": False, "error": str(e)}
        status["available"] = False

    # Log warnings for unavailable tools (warn only, don't fail startup)
    unavailable = [name for name, info in status["tools"].items() if not info["available"]]
    if unavailable:
        logger.warning(
            f"Knowledge tools unavailable: {', '.join(unavailable)}. "
            "Install with: pip install -e '.[knowledge]'"
        )

    _knowledge_tools_status = status
    return status


@router.get("/health", response_model=HealthResponse)
async def health(
    tracker: BackendHealthTracker = Depends(dep_health_tracker),
) -> HealthResponse:
    """Health check endpoint.

    Reports overall status and per-backend circuit breaker state.
    Status is "ok" when all tracked backends are healthy, "degraded"
    when any circuit is open or half-open.
    """
    backend_health = tracker.get_status() if tracker else {}
    backends_healthy = sum(1 for s in backend_health.values() if s["state"] == "closed")
    backends_total = len(backend_health)

    if backends_total == 0:
        status = "ok"
    elif backends_healthy == backends_total:
        status = "ok"
    else:
        status = "degraded"

    # Check knowledge tools availability (cached)
    knowledge_status = _check_knowledge_tools()

    return HealthResponse(
        status=status,
        models_loaded=backends_healthy,
        mock_mode_available=True,
        version="0.1.0",
        backend_health=backend_health,
        knowledge_tools=knowledge_status,
    )
