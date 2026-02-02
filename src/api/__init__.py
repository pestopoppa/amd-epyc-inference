"""FastAPI application factory for the orchestrator API.

This module provides the main FastAPI application with all routes,
middleware, and lifespan management.

Usage:
    # Development server
    uvicorn src.api:app --reload --port 8000

    # Production
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.features import features
from src.api.state import get_state, AppState
from src.api.routes import create_api_router
from src.api.services.memrl import (
    load_optional_imports,
    background_cleanup,
    ensure_memrl_initialized,
    shutdown_scoring,
    get_progress_logger_class,
    get_tool_registry_class,
    get_script_registry_class,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan.

    Initialization order:
    1. Load feature flags from environment
    2. Load optional module imports based on features
    3. Load registry (YAML parsing only)
    4. Initialize core components (LLM primitives, gate runner, failure router)
    5. Initialize optional components based on features
    6. Eagerly initialize MemRL (Q-scorer, TaskEmbedder, HybridRouter)
    7. Start background tasks if MemRL enabled
    """
    state = get_state()
    f = features()

    # Validate feature dependencies at startup
    validation_errors = f.validate()
    if validation_errors:
        for err in validation_errors:
            logger.error(f"Feature validation error: {err}")

    # Load feature flags and optional imports
    load_optional_imports()

    logger.info(f"Starting orchestrator with features: {f.enabled_features()}")

    # Load registry for role-based generation defaults (YAML parsing only, no models)
    try:
        from src.registry_loader import RegistryLoader
        state.registry = RegistryLoader(validate_paths=False)
    except Exception as e:
        logger.info(f"Registry file not found or invalid, using defaults: {e}")
        state.registry = None

    # Core components (always initialized)
    from src.llm_primitives import LLMPrimitives
    from src.gate_runner import GateRunner
    from src.failure_router import FailureRouter

    ProgressLogger = get_progress_logger_class()

    state.llm_primitives = LLMPrimitives(
        mock_mode=f.mock_mode,
        registry=state.registry,
        health_tracker=state.health_tracker,
    )
    state.progress_logger = ProgressLogger() if ProgressLogger else None
    state.gate_runner = GateRunner(progress_logger=state.progress_logger)
    state.failure_router = FailureRouter()

    # Tool registry (feature-gated)
    ToolRegistry = get_tool_registry_class()
    if f.tools and ToolRegistry:
        try:
            from src.tool_registry import load_from_yaml as load_tools_from_yaml

            state.tool_registry = ToolRegistry()

            # Load tools from YAML registry (22 tools in orchestration/tool_registry.yaml)
            from src.config import get_config as _get_config
            _paths = _get_config().paths
            loaded = load_tools_from_yaml(
                state.tool_registry,
                str(_paths.tool_registry_path),
            )
            logger.info(f"Loaded {loaded} tools from YAML registry")

            # Load role permissions from model registry
            state.tool_registry.load_permissions_from_registry(
                str(_paths.registry_path),
            )

            # Register built-in tools (programmatic tools)
            # These may overlap with YAML tools - duplicates are skipped
            try:
                from src.builtin_tools import register_builtin_tools
                register_builtin_tools(state.tool_registry)
            except ImportError:
                logger.debug("No builtin_tools module found")
            except ValueError as e:
                # Tool already registered from YAML - this is fine
                logger.debug(f"Some builtin tools skipped (already in YAML): {e}")
        except Exception as e:
            logger.info(f"Tool registry not available: {e}")
            state.tool_registry = None
    else:
        state.tool_registry = None

    # Script registry (feature-gated, requires tools)
    ScriptRegistry = get_script_registry_class()
    if f.scripts and ScriptRegistry and state.tool_registry:
        try:
            state.script_registry = ScriptRegistry()
            state.script_registry.load_from_directory(
                str(_paths.script_registry_dir),
            )
        except Exception as e:
            logger.info(f"Script registry not available: {e}")
            state.script_registry = None
    else:
        state.script_registry = None

    # Eagerly initialize MemRL components at startup (loads TaskEmbedder model).
    # This avoids a ~10-30s block on the first real_mode request.
    if f.memrl:
        ensure_memrl_initialized(state)

    # Background Q-scoring task (only if MemRL feature enabled)
    if f.memrl:
        state._q_scorer_task = asyncio.create_task(background_cleanup(state))
    else:
        state._q_scorer_task = None

    yield

    # Shutdown - cancel background task first
    if state._q_scorer_task:
        state._q_scorer_task.cancel()
        try:
            await state._q_scorer_task
        except asyncio.CancelledError:
            pass

    # Shutdown Q-scorer thread pool
    shutdown_scoring()

    # Flush progress logger before cleanup
    if state.progress_logger:
        state.progress_logger.flush()

    # Shutdown vision batch processor if initialized
    if state.vision_batch_processor is not None:
        state.vision_batch_processor.shutdown()

    # Clear state
    state.llm_primitives = None
    state.gate_runner = None
    state.failure_router = None
    state.progress_logger = None
    state.q_scorer = None
    state.episodic_store = None
    state.hybrid_router = None
    state.tool_registry = None
    state.script_registry = None
    state.registry = None
    state.vision_batch_processor = None
    state.health_tracker.reset()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Orchestrator API",
        description="HTTP interface for the hierarchical orchestration system",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware (order matters: last added = first executed)
    from src.config import get_config as _get_config
    _api_cfg = _get_config().api

    # CORS — explicit origins required when credentials are enabled
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_api_cfg.cors_origins,
        allow_credentials=_api_cfg.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting — per-IP token bucket
    from src.api.rate_limit import RateLimitMiddleware
    app.add_middleware(
        RateLimitMiddleware,
        rpm=_api_cfg.rate_limit_rpm,
        burst=_api_cfg.rate_limit_burst,
    )

    # Include all routes
    router = create_api_router()
    app.include_router(router)

    return app


# Create the default application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
