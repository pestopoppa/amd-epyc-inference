"""Unit tests for FastAPI dependency injection layer.

Tests that dependency functions return correct types, raise 503 for
uninitialized required components, and return None for optional components.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestDepAppState:
    """Tests for dep_app_state — returns the global AppState."""

    def test_returns_app_state(self):
        from src.api.dependencies import dep_app_state
        from src.api.state import AppState

        state = dep_app_state()
        assert isinstance(state, AppState)

    def test_returns_same_instance(self):
        from src.api.dependencies import dep_app_state

        s1 = dep_app_state()
        s2 = dep_app_state()
        assert s1 is s2


class TestDepLLMPrimitives:
    """Tests for dep_llm_primitives — required, raises 503 if None."""

    def test_raises_503_when_none(self):
        from fastapi import HTTPException
        from src.api.dependencies import dep_llm_primitives
        from src.api.state import get_state

        state = get_state()
        state.llm_primitives = None
        with pytest.raises(HTTPException) as exc_info:
            dep_llm_primitives()
        assert exc_info.value.status_code == 503
        assert "LLM primitives" in exc_info.value.detail

    def test_returns_primitives_when_set(self):
        from src.api.dependencies import dep_llm_primitives
        from src.api.state import get_state

        state = get_state()
        mock_primitives = MagicMock()
        state.llm_primitives = mock_primitives
        try:
            result = dep_llm_primitives()
            assert result is mock_primitives
        finally:
            state.llm_primitives = None


class TestDepGateRunner:
    """Tests for dep_gate_runner — required, raises 503 if None."""

    def test_raises_503_when_none(self):
        from fastapi import HTTPException
        from src.api.dependencies import dep_gate_runner
        from src.api.state import get_state

        state = get_state()
        state.gate_runner = None
        with pytest.raises(HTTPException) as exc_info:
            dep_gate_runner()
        assert exc_info.value.status_code == 503
        assert "Gate runner" in exc_info.value.detail

    def test_returns_gate_runner_when_set(self):
        from src.api.dependencies import dep_gate_runner
        from src.api.state import get_state

        state = get_state()
        mock_runner = MagicMock()
        state.gate_runner = mock_runner
        try:
            result = dep_gate_runner()
            assert result is mock_runner
        finally:
            state.gate_runner = None


class TestDepHealthTracker:
    """Tests for dep_health_tracker — always available."""

    def test_returns_health_tracker(self):
        from src.api.dependencies import dep_health_tracker
        from src.api.health_tracker import BackendHealthTracker

        tracker = dep_health_tracker()
        assert isinstance(tracker, BackendHealthTracker)


class TestDepOptionalComponents:
    """Tests for optional dependency functions — return None when not set."""

    def test_progress_logger_none(self):
        from src.api.dependencies import dep_progress_logger
        from src.api.state import get_state

        state = get_state()
        state.progress_logger = None
        assert dep_progress_logger() is None

    def test_progress_logger_returns_value(self):
        from src.api.dependencies import dep_progress_logger
        from src.api.state import get_state

        state = get_state()
        mock = MagicMock()
        state.progress_logger = mock
        try:
            assert dep_progress_logger() is mock
        finally:
            state.progress_logger = None

    def test_hybrid_router_none(self):
        from src.api.dependencies import dep_hybrid_router
        from src.api.state import get_state

        state = get_state()
        state.hybrid_router = None
        assert dep_hybrid_router() is None

    def test_tool_registry_none(self):
        from src.api.dependencies import dep_tool_registry
        from src.api.state import get_state

        state = get_state()
        state.tool_registry = None
        assert dep_tool_registry() is None

    def test_script_registry_none(self):
        from src.api.dependencies import dep_script_registry
        from src.api.state import get_state

        state = get_state()
        state.script_registry = None
        assert dep_script_registry() is None

    def test_registry_loader_none(self):
        from src.api.dependencies import dep_registry_loader
        from src.api.state import get_state

        state = get_state()
        state.registry = None
        assert dep_registry_loader() is None


class TestDepFeatures:
    """Tests for dep_features — returns current feature flags."""

    def test_returns_features_instance(self):
        from src.api.dependencies import dep_features
        from src.features import Features

        result = dep_features()
        assert isinstance(result, Features)

    def test_returns_current_features(self):
        from src.api.dependencies import dep_features

        result = dep_features()
        # Should return a Features object with some default values
        assert hasattr(result, "summary")
        assert isinstance(result.summary(), dict)


class TestDepDocumentPreprocessor:
    """Tests for dep_document_preprocessor — lazy initialization."""

    def test_lazy_initialization(self):
        from src.api.dependencies import dep_document_preprocessor
        from src.api.state import get_state
        from src.services.document_preprocessor import DocumentPreprocessor

        state = get_state()
        state.document_preprocessor = None

        result = dep_document_preprocessor()
        assert isinstance(result, DocumentPreprocessor)
        assert state.document_preprocessor is result

    def test_returns_existing_instance(self):
        from src.api.dependencies import dep_document_preprocessor
        from src.api.state import get_state

        state = get_state()
        mock = MagicMock()
        state.document_preprocessor = mock

        try:
            result = dep_document_preprocessor()
            assert result is mock
        finally:
            state.document_preprocessor = None


class TestDepVisionComponents:
    """Tests for vision pipeline dependency functions."""

    def test_vision_pipeline_lazy_init(self):
        from src.api.dependencies import dep_vision_pipeline
        from src.api.state import get_state
        from src.vision.pipeline import VisionPipeline

        state = get_state()
        state.vision_pipeline = None

        result = dep_vision_pipeline()
        assert isinstance(result, VisionPipeline)
        assert state.vision_pipeline is result

    def test_vision_batch_processor_lazy_init(self):
        from src.api.dependencies import dep_vision_batch_processor
        from src.api.state import get_state
        from src.vision.batch import BatchProcessor

        state = get_state()
        state.vision_batch_processor = None

        result = dep_vision_batch_processor()
        assert isinstance(result, BatchProcessor)
        assert state.vision_batch_processor is result

    def test_vision_search_lazy_init(self):
        from src.api.dependencies import dep_vision_search
        from src.api.state import get_state
        from src.vision.search import VisionSearch

        state = get_state()
        state.vision_search = None

        result = dep_vision_search()
        assert isinstance(result, VisionSearch)
        assert state.vision_search is result

    def test_vision_video_processor_lazy_init(self):
        from src.api.dependencies import dep_vision_video_processor
        from src.api.state import get_state
        from src.vision.video import VideoProcessor

        state = get_state()
        state.vision_video_processor = None

        result = dep_vision_video_processor()
        assert isinstance(result, VideoProcessor)
        assert state.vision_video_processor is result


class TestDepSessionStore:
    """Tests for dep_session_store — lazy initialization."""

    def test_lazy_initialization(self):
        from src.api.dependencies import dep_session_store
        from src.api.state import get_state
        from src.session import SQLiteSessionStore

        state = get_state()
        state.session_store = None

        result = dep_session_store()
        assert isinstance(result, SQLiteSessionStore)
        assert state.session_store is result

    def test_returns_existing_instance(self):
        from src.api.dependencies import dep_session_store
        from src.api.state import get_state

        state = get_state()
        mock = MagicMock()
        state.session_store = mock

        try:
            result = dep_session_store()
            assert result is mock
        finally:
            state.session_store = None
