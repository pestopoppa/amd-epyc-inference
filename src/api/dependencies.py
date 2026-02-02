"""FastAPI dependency injection layer for the orchestrator API.

Provides typed dependency functions using FastAPI's Depends() pattern.
Replaces direct get_state() calls in route handlers, enabling:
- Testability: Override dependencies in tests without monkeypatching
- Type safety: Each handler declares exactly what it needs
- Validation: Required components raise HTTP 503 if unavailable

Usage in route handlers:
    from fastapi import Depends
    from src.api.dependencies import get_app_state, dep_gate_runner

    @router.post("/gates")
    async def run_gates(gate_runner: GateRunner = Depends(dep_gate_runner)):
        ...

Usage in tests:
    from src.api.dependencies import dep_gate_runner
    app.dependency_overrides[dep_gate_runner] = lambda: mock_gate_runner
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException

from src.api.state import get_state

if TYPE_CHECKING:
    from src.api.health_tracker import BackendHealthTracker
    from src.api.state import AppState
    from src.gate_runner import GateRunner
    from src.llm_primitives import LLMPrimitives
    from src.features import Features
    from src.services.document_preprocessor import DocumentPreprocessor
    from src.vision.pipeline import VisionPipeline
    from src.vision.batch import BatchProcessor
    from src.vision.search import VisionSearch
    from src.vision.video import VideoProcessor
    from src.session import SQLiteSessionStore
    from src.api.protocols import (
        HybridRouterProtocol,
        ProgressLoggerProtocol,
        RegistryLoaderProtocol,
        ToolRegistryProtocol,
        ScriptRegistryProtocol,
    )


# ── Core state dependency ─────────────────────────────────────────────────


def dep_app_state() -> "AppState":
    """Get the global application state."""
    return get_state()


# ── Required components (raise 503 if unavailable) ────────────────────────


def dep_llm_primitives() -> "LLMPrimitives":
    """Get LLMPrimitives. Raises 503 if not initialized."""
    state = get_state()
    if state.llm_primitives is None:
        raise HTTPException(
            status_code=503,
            detail="LLM primitives not initialized (server not ready)",
        )
    return state.llm_primitives


def dep_gate_runner() -> "GateRunner":
    """Get GateRunner. Raises 503 if not initialized."""
    state = get_state()
    if state.gate_runner is None:
        raise HTTPException(
            status_code=503,
            detail="Gate runner not initialized",
        )
    return state.gate_runner


def dep_health_tracker() -> "BackendHealthTracker":
    """Get BackendHealthTracker (always available on AppState)."""
    return get_state().health_tracker


# ── Optional components (return None if unavailable) ──────────────────────


def dep_progress_logger() -> "ProgressLoggerProtocol | None":
    """Get ProgressLogger if available."""
    return get_state().progress_logger


def dep_hybrid_router() -> "HybridRouterProtocol | None":
    """Get HybridRouter if available."""
    return get_state().hybrid_router


def dep_tool_registry() -> "ToolRegistryProtocol | None":
    """Get ToolRegistry if available."""
    return get_state().tool_registry


def dep_script_registry() -> "ScriptRegistryProtocol | None":
    """Get ScriptRegistry if available."""
    return get_state().script_registry


def dep_registry_loader() -> "RegistryLoaderProtocol | None":
    """Get RegistryLoader if available."""
    return get_state().registry


# ── Feature flags ─────────────────────────────────────────────────────────


def dep_features() -> "Features":
    """Get the current feature flags configuration."""
    from src.features import features
    return features()


# ── Document processing ───────────────────────────────────────────────────


def dep_document_preprocessor() -> "DocumentPreprocessor":
    """Get or lazily initialize the document preprocessor."""
    state = get_state()
    if state.document_preprocessor is None:
        from src.services.document_preprocessor import DocumentPreprocessor
        state.document_preprocessor = DocumentPreprocessor()
    return state.document_preprocessor


# ── Vision processing ─────────────────────────────────────────────────────


def dep_vision_pipeline() -> "VisionPipeline":
    """Get or lazily initialize the vision pipeline."""
    state = get_state()
    if state.vision_pipeline is None:
        from src.vision.pipeline import VisionPipeline
        state.vision_pipeline = VisionPipeline()
    return state.vision_pipeline


def dep_vision_batch_processor() -> "BatchProcessor":
    """Get or lazily initialize the vision batch processor."""
    state = get_state()
    if state.vision_batch_processor is None:
        from src.vision.batch import BatchProcessor
        state.vision_batch_processor = BatchProcessor()
    return state.vision_batch_processor


def dep_vision_search() -> "VisionSearch":
    """Get or lazily initialize the vision search engine."""
    state = get_state()
    if state.vision_search is None:
        from src.vision.search import VisionSearch
        state.vision_search = VisionSearch()
    return state.vision_search


def dep_vision_video_processor() -> "VideoProcessor":
    """Get or lazily initialize the vision video processor."""
    state = get_state()
    if state.vision_video_processor is None:
        from src.vision.video import VideoProcessor
        state.vision_video_processor = VideoProcessor()
    return state.vision_video_processor


# ── Session management ────────────────────────────────────────────────────


def dep_session_store() -> "SQLiteSessionStore":
    """Get or lazily initialize the session store."""
    state = get_state()
    if state.session_store is None:
        from src.session import SQLiteSessionStore
        import logging
        state.session_store = SQLiteSessionStore()
        logging.getLogger(__name__).info("Initialized SQLiteSessionStore")
    return state.session_store
