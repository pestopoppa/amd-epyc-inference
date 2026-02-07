"""Health check endpoint with backend health aggregation."""

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends

from src.api.dependencies import dep_health_tracker
from src.api.health_tracker import BackendHealthTracker
from src.api.models import HealthResponse
from src.config import get_config

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


async def _probe_backend(url: str, timeout: float = 2.0) -> dict[str, Any]:
    """Probe a backend for liveness."""
    import httpx

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/health")
        ok = resp.status_code == 200
    except Exception:
        ok = False
    latency_ms = (time.perf_counter() - start) * 1000
    return {"ok": ok, "latency_ms": round(latency_ms, 1), "url": url}


async def _probe_core_backends() -> dict[str, Any]:
    """Probe core backend roles for liveness."""
    server_urls = get_config().server_urls.as_dict()
    core_roles = ["frontdoor", "coder_escalation", "architect_general", "architect_coding"]
    probes: dict[str, Any] = {}
    tasks = []
    role_list = []
    for role in core_roles:
        url = server_urls.get(role)
        if url:
            role_list.append(role)
            tasks.append(_probe_backend(url))
    if not tasks:
        return probes
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for role, result in zip(role_list, results):
        if isinstance(result, Exception):
            probes[role] = {"ok": False, "latency_ms": None, "url": server_urls.get(role)}
        else:
            probes[role] = result
    return probes


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

    # Probe core backends for liveness
    backend_probes = await _probe_core_backends()
    if backend_probes:
        any_down = any(not p.get("ok", False) for p in backend_probes.values())
        if any_down and status == "ok":
            status = "degraded"

    return HealthResponse(
        status=status,
        models_loaded=backends_healthy,
        mock_mode_available=True,
        version="0.1.0",
        backend_health=backend_health,
        backend_probes=backend_probes or None,
        knowledge_tools=knowledge_status,
    )
