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
