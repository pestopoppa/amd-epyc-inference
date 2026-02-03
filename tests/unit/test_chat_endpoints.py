"""Tests for chat endpoints (src/api/routes/chat.py).

Tests the main /chat and /chat/stream endpoints including:
- Request handling and state management
- Mock mode vs real mode
- Reward injection endpoint
- SSE streaming
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat import router, _handle_chat
from src.api.dependencies import dep_app_state
from src.api.state import AppState


@pytest.fixture
def mock_app_state():
    """Create a mock AppState for testing."""
    state = MagicMock(spec=AppState)
    state.progress_logger = MagicMock()
    state.tool_registry = MagicMock()
    state.script_registry = MagicMock()
    state.registry = MagicMock()
    state.hybrid_router = None
    state.increment_active = MagicMock()
    state.decrement_active = MagicMock()
    state.increment_request = MagicMock()
    return state


@pytest.fixture
def test_app(mock_app_state):
    """Create a FastAPI test app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[dep_app_state] = lambda: mock_app_state
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


# ─────────────────────────────────────────────────────────────────────────────
# /chat endpoint tests
# ─────────────────────────────────────────────────────────────────────────────


class TestChatEndpoint:
    """Tests for POST /chat endpoint."""

    def test_mock_mode_returns_response(self, client, mock_app_state):
        """Mock mode request returns ChatResponse."""
        with patch("src.api.routes.chat._handle_chat") as mock_handle:
            mock_response = ChatResponse(
                answer="[MOCK] Test response",
                turns=1,
                tokens_used=0,
                elapsed_seconds=0.1,
                mock_mode=True,
                real_mode=False,
            )
            mock_handle.return_value = mock_response

            response = client.post(
                "/chat",
                json={"prompt": "Hello", "mock_mode": True},
            )

            assert response.status_code == 200
            data = response.json()
            assert "[MOCK]" in data["answer"]
            mock_handle.assert_called_once()

    def test_increments_and_decrements_active(self, client, mock_app_state):
        """Chat endpoint tracks active requests."""
        with patch("src.api.routes.chat._handle_chat") as mock_handle:
            mock_response = ChatResponse(
                answer="Test",
                turns=1,
                tokens_used=0,
                elapsed_seconds=0.1,
                mock_mode=True,
                real_mode=False,
            )
            mock_handle.return_value = mock_response

            client.post("/chat", json={"prompt": "Hello", "mock_mode": True})

            mock_app_state.increment_active.assert_called_once()
            mock_app_state.decrement_active.assert_called_once()

    def test_decrements_even_on_exception(self, client, mock_app_state):
        """Active counter is decremented even if handler raises."""
        with patch("src.api.routes.chat._handle_chat") as mock_handle:
            mock_handle.side_effect = RuntimeError("Handler error")

            with pytest.raises(Exception):
                client.post("/chat", json={"prompt": "Hello", "mock_mode": True})

            mock_app_state.increment_active.assert_called_once()
            mock_app_state.decrement_active.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# /chat/reward endpoint tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardEndpoint:
    """Tests for POST /chat/reward endpoint."""

    def test_reward_injection_success(self, client):
        """Successful reward injection returns success=True."""
        with patch("src.api.routes.chat.store_external_reward") as mock_store:
            mock_store.return_value = True

            response = client.post(
                "/chat/reward",
                json={
                    "task_description": "Test task",
                    "action": "route_to_coder",
                    "reward": 1.0,
                    "context": {"suite": "thinking", "tier": 1},
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_store.assert_called_once()

    def test_reward_injection_failure(self, client):
        """Failed reward injection returns success=False."""
        with patch("src.api.routes.chat.store_external_reward") as mock_store:
            mock_store.return_value = False

            response = client.post(
                "/chat/reward",
                json={
                    "task_description": "Test task",
                    "action": "route_to_coder",
                    "reward": -1.0,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False


# ─────────────────────────────────────────────────────────────────────────────
# _handle_chat tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleChat:
    """Tests for _handle_chat internal function."""

    @pytest.mark.asyncio
    async def test_mock_mode_path(self, mock_app_state):
        """Mock mode returns early from _execute_mock."""
        request = ChatRequest(prompt="Test", mock_mode=True, real_mode=False)

        with patch("src.api.routes.chat._route_request") as mock_route:
            mock_routing = MagicMock()
            mock_routing.use_mock = True
            mock_route.return_value = mock_routing

            with patch("src.api.routes.chat._preprocess"):
                with patch("src.api.routes.chat._execute_mock") as mock_exec:
                    mock_response = ChatResponse(
                        answer="[MOCK]",
                        turns=1,
                        tokens_used=0,
                        elapsed_seconds=0.1,
                        mock_mode=True,
                        real_mode=False,
                    )
                    mock_exec.return_value = mock_response

                    result = await _handle_chat(request, mock_app_state)

                    assert result.mock_mode is True
                    mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_vision_early_return(self, mock_app_state):
        """Vision pipeline returns early when it produces a response."""
        request = ChatRequest(prompt="Test", real_mode=True, image_path="/test.png")

        with patch("src.api.routes.chat._route_request") as mock_route:
            mock_routing = MagicMock()
            mock_routing.use_mock = False
            mock_routing.routing_decision = ["worker_vision"]
            mock_routing.document_result = None
            mock_route.return_value = mock_routing

            with patch("src.api.routes.chat._preprocess"):
                with patch("src.api.routes.chat._init_primitives"):
                    with patch("src.api.routes.chat._plan_review_gate"):
                        with patch("src.api.routes.chat._execute_vision") as mock_vision:
                            mock_response = ChatResponse(
                                answer="Vision result",
                                turns=1,
                                tokens_used=100,
                                elapsed_seconds=1.0,
                                mock_mode=False,
                                real_mode=True,
                            )
                            mock_vision.return_value = mock_response

                            with patch("src.api.routes.chat._annotate_error") as mock_annotate:
                                mock_annotate.return_value = mock_response
                                result = await _handle_chat(request, mock_app_state)

                                assert result.answer == "Vision result"

    @pytest.mark.asyncio
    async def test_proactive_delegation_path(self, mock_app_state):
        """Proactive delegation returns early when successful."""
        request = ChatRequest(prompt="Complex task", real_mode=True)

        with patch("src.api.routes.chat._route_request") as mock_route:
            mock_routing = MagicMock()
            mock_routing.use_mock = False
            mock_routing.routing_decision = ["frontdoor"]
            mock_routing.document_result = None
            mock_route.return_value = mock_routing

            with patch("src.api.routes.chat._preprocess"):
                with patch("src.api.routes.chat._init_primitives"):
                    with patch("src.api.routes.chat._plan_review_gate"):
                        with patch("src.api.routes.chat._execute_vision") as mock_vision:
                            mock_vision.return_value = None  # No vision result

                            with patch("src.api.routes.chat._execute_proactive") as mock_proactive:
                                mock_response = ChatResponse(
                                    answer="Proactive result",
                                    turns=3,
                                    tokens_used=500,
                                    elapsed_seconds=5.0,
                                    mock_mode=False,
                                    real_mode=True,
                                    mode="proactive",
                                )
                                mock_proactive.return_value = mock_response

                                with patch("src.api.routes.chat._annotate_error") as mock_annotate:
                                    mock_annotate.return_value = mock_response
                                    result = await _handle_chat(request, mock_app_state)

                                    assert result.mode == "proactive"

    @pytest.mark.asyncio
    async def test_direct_mode_path(self, mock_app_state):
        """Direct mode executes _execute_direct."""
        request = ChatRequest(prompt="Simple question", real_mode=True, force_mode="direct")

        with patch("src.api.routes.chat._route_request") as mock_route:
            mock_routing = MagicMock()
            mock_routing.use_mock = False
            mock_routing.routing_decision = ["frontdoor"]
            mock_routing.document_result = None
            mock_route.return_value = mock_routing

            with patch("src.api.routes.chat._preprocess"):
                with patch("src.api.routes.chat._init_primitives"):
                    with patch("src.api.routes.chat._plan_review_gate"):
                        with patch("src.api.routes.chat._execute_vision") as mock_vision:
                            mock_vision.return_value = None

                            with patch("src.api.routes.chat._execute_proactive") as mock_proactive:
                                mock_proactive.return_value = None

                                with patch("src.api.routes.chat.features") as mock_features:
                                    mock_features.return_value.architect_delegation = False

                                    with patch(
                                        "src.api.routes.chat._execute_direct"
                                    ) as mock_direct:
                                        mock_response = ChatResponse(
                                            answer="Direct answer",
                                            turns=1,
                                            tokens_used=100,
                                            elapsed_seconds=0.5,
                                            mock_mode=False,
                                            real_mode=True,
                                            mode="direct",
                                        )
                                        mock_direct.return_value = mock_response

                                        with patch(
                                            "src.api.routes.chat._annotate_error"
                                        ) as mock_annotate:
                                            mock_annotate.return_value = mock_response
                                            result = await _handle_chat(request, mock_app_state)

                                            assert result.mode == "direct"

    @pytest.mark.asyncio
    async def test_repl_fallback_path(self, mock_app_state):
        """REPL mode is the default fallback."""
        request = ChatRequest(prompt="Complex task", real_mode=True)

        with patch("src.api.routes.chat._route_request") as mock_route:
            mock_routing = MagicMock()
            mock_routing.use_mock = False
            mock_routing.routing_decision = ["frontdoor"]
            mock_routing.document_result = None
            mock_route.return_value = mock_routing

            with patch("src.api.routes.chat._preprocess"):
                with patch("src.api.routes.chat._init_primitives"):
                    with patch("src.api.routes.chat._plan_review_gate"):
                        with patch("src.api.routes.chat._execute_vision") as mock_vision:
                            mock_vision.return_value = None

                            with patch("src.api.routes.chat._execute_proactive") as mock_proactive:
                                mock_proactive.return_value = None

                                with patch("src.api.routes.chat._select_mode") as mock_mode:
                                    mock_mode.return_value = "repl"

                                    with patch("src.api.routes.chat.features") as mock_features:
                                        mock_features.return_value.architect_delegation = False

                                        with patch(
                                            "src.api.routes.chat._execute_repl"
                                        ) as mock_repl:
                                            mock_response = ChatResponse(
                                                answer="REPL answer",
                                                turns=2,
                                                tokens_used=200,
                                                elapsed_seconds=2.0,
                                                mock_mode=False,
                                                real_mode=True,
                                            )
                                            mock_repl.return_value = mock_response

                                            with patch(
                                                "src.api.routes.chat._annotate_error"
                                            ) as mock_annotate:
                                                mock_annotate.return_value = mock_response
                                                result = await _handle_chat(request, mock_app_state)

                                                assert result.answer == "REPL answer"

    @pytest.mark.asyncio
    async def test_document_result_forces_repl(self, mock_app_state):
        """Document preprocessing result forces REPL mode."""
        request = ChatRequest(prompt="Describe document", real_mode=True)

        with patch("src.api.routes.chat._route_request") as mock_route:
            mock_routing = MagicMock()
            mock_routing.use_mock = False
            mock_routing.routing_decision = ["worker_vision"]
            mock_routing.document_result = MagicMock()  # Non-None document result
            mock_route.return_value = mock_routing

            with patch("src.api.routes.chat._preprocess"):
                with patch("src.api.routes.chat._init_primitives"):
                    with patch("src.api.routes.chat._plan_review_gate"):
                        with patch("src.api.routes.chat._execute_vision") as mock_vision:
                            mock_vision.return_value = None

                            with patch("src.api.routes.chat._execute_proactive") as mock_proactive:
                                mock_proactive.return_value = None

                                with patch("src.api.routes.chat._execute_repl") as mock_repl:
                                    mock_response = ChatResponse(
                                        answer="Document analysis",
                                        turns=1,
                                        tokens_used=200,
                                        elapsed_seconds=3.0,
                                        mock_mode=False,
                                        real_mode=True,
                                    )
                                    mock_repl.return_value = mock_response

                                    with patch(
                                        "src.api.routes.chat._annotate_error"
                                    ) as mock_annotate:
                                        mock_annotate.return_value = mock_response
                                        await _handle_chat(request, mock_app_state)

                                        # REPL should be called with frontdoor role
                                        assert mock_repl.called


# ─────────────────────────────────────────────────────────────────────────────
# /chat/stream endpoint tests (SSE streaming)
# ─────────────────────────────────────────────────────────────────────────────


class TestChatStreamEndpoint:
    """Tests for POST /chat/stream SSE endpoint."""

    def test_stream_mock_mode(self, client, mock_app_state):
        """Mock mode streams simulated tokens."""
        with patch("src.api.routes.chat.score_completed_task"):
            response = client.post(
                "/chat/stream",
                json={"prompt": "Hello", "mock_mode": True},
            )

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            # Parse SSE events
            content = response.text
            assert "[MOCK]" in content or "turn_start" in content

    def test_stream_mock_mode_with_thinking(self, client, mock_app_state):
        """Mock mode with thinking_budget emits thinking events."""
        with patch("src.api.routes.chat.score_completed_task"):
            response = client.post(
                "/chat/stream",
                json={
                    "prompt": "Hello",
                    "mock_mode": True,
                    "thinking_budget": 100,
                },
            )

            assert response.status_code == 200
            content = response.text
            # Should contain thinking events
            assert "thinking" in content or "Analyzing" in content

    def test_stream_plan_mode(self, client, mock_app_state):
        """Plan mode emits analysis without full execution."""
        with patch("src.api.routes.chat.score_completed_task"):
            response = client.post(
                "/chat/stream",
                json={
                    "prompt": "Hello",
                    "mock_mode": True,
                    "permission_mode": "plan",
                },
            )

            assert response.status_code == 200
            content = response.text
            assert "PLAN MODE" in content or "Would process" in content

    def test_stream_real_mode_initialization_error(self, client, mock_app_state):
        """Real mode handles LLMPrimitives initialization errors."""
        with patch("src.api.routes.chat.ensure_memrl_initialized"):
            with patch("src.api.routes.chat.get_config") as mock_config:
                mock_config.return_value.server_urls.as_dict.return_value = {}
                with patch("src.api.routes.chat.LLMPrimitives") as mock_prims:
                    mock_prims.side_effect = RuntimeError("Backend unavailable")
                    with patch("src.api.routes.chat.score_completed_task"):
                        response = client.post(
                            "/chat/stream",
                            json={"prompt": "Hello", "real_mode": True},
                        )

                        assert response.status_code == 200
                        content = response.text
                        # Should contain error event
                        assert "error" in content.lower() or "unavailable" in content.lower()

    def test_stream_logs_task_start(self, client, mock_app_state):
        """Stream endpoint logs task start via progress logger."""
        with patch("src.api.routes.chat.score_completed_task"):
            client.post(
                "/chat/stream",
                json={"prompt": "Hello", "mock_mode": True},
            )

            mock_app_state.progress_logger.log_task_started.assert_called_once()

    def test_stream_logs_task_completed(self, client, mock_app_state):
        """Stream endpoint logs task completion."""
        with patch("src.api.routes.chat.score_completed_task"):
            client.post(
                "/chat/stream",
                json={"prompt": "Hello", "mock_mode": True},
            )

            mock_app_state.progress_logger.log_task_completed.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestChatEdgeCases:
    """Edge case tests for chat endpoints."""

    @pytest.mark.asyncio
    async def test_delegated_mode_returns_none_fallthrough(self, mock_app_state):
        """Delegated mode returning None falls through to next mode."""
        request = ChatRequest(prompt="Task", real_mode=True)

        with patch("src.api.routes.chat._route_request") as mock_route:
            mock_routing = MagicMock()
            mock_routing.use_mock = False
            mock_routing.routing_decision = ["architect_general"]
            mock_routing.document_result = None
            mock_route.return_value = mock_routing

            with patch("src.api.routes.chat._preprocess"):
                with patch("src.api.routes.chat._init_primitives"):
                    with patch("src.api.routes.chat._plan_review_gate"):
                        with patch("src.api.routes.chat._execute_vision") as mock_vision:
                            mock_vision.return_value = None

                            with patch("src.api.routes.chat._execute_proactive") as mock_proactive:
                                mock_proactive.return_value = None

                                with patch("src.api.routes.chat._select_mode") as mock_mode:
                                    mock_mode.return_value = "delegated"

                                    with patch("src.api.routes.chat.features") as mock_features:
                                        mock_features.return_value.architect_delegation = True

                                        with patch(
                                            "src.api.routes.chat._execute_delegated"
                                        ) as mock_deleg:
                                            mock_deleg.return_value = None  # Falls through

                                            with patch(
                                                "src.api.routes.chat._execute_react"
                                            ) as mock_react:
                                                mock_react.return_value = None

                                                with patch(
                                                    "src.api.routes.chat._execute_repl"
                                                ) as mock_repl:
                                                    mock_response = ChatResponse(
                                                        answer="REPL fallback",
                                                        turns=1,
                                                        tokens_used=100,
                                                        elapsed_seconds=1.0,
                                                        mock_mode=False,
                                                        real_mode=True,
                                                    )
                                                    mock_repl.return_value = mock_response

                                                    with patch(
                                                        "src.api.routes.chat._annotate_error"
                                                    ) as mock_annotate:
                                                        mock_annotate.return_value = mock_response
                                                        result = await _handle_chat(
                                                            request, mock_app_state
                                                        )

                                                        assert result.answer == "REPL fallback"

    @pytest.mark.asyncio
    async def test_react_mode_returns_none_fallthrough(self, mock_app_state):
        """React mode returning None falls through to REPL."""
        request = ChatRequest(prompt="Task", real_mode=True, force_mode="react")

        with patch("src.api.routes.chat._route_request") as mock_route:
            mock_routing = MagicMock()
            mock_routing.use_mock = False
            mock_routing.routing_decision = ["frontdoor"]
            mock_routing.document_result = None
            mock_route.return_value = mock_routing

            with patch("src.api.routes.chat._preprocess"):
                with patch("src.api.routes.chat._init_primitives"):
                    with patch("src.api.routes.chat._plan_review_gate"):
                        with patch("src.api.routes.chat._execute_vision") as mock_vision:
                            mock_vision.return_value = None

                            with patch("src.api.routes.chat._execute_proactive") as mock_proactive:
                                mock_proactive.return_value = None

                                with patch("src.api.routes.chat.features") as mock_features:
                                    mock_features.return_value.architect_delegation = False

                                    with patch("src.api.routes.chat._execute_react") as mock_react:
                                        mock_react.return_value = None  # Falls through

                                        with patch(
                                            "src.api.routes.chat._execute_repl"
                                        ) as mock_repl:
                                            mock_response = ChatResponse(
                                                answer="REPL fallback after react",
                                                turns=1,
                                                tokens_used=100,
                                                elapsed_seconds=1.0,
                                                mock_mode=False,
                                                real_mode=True,
                                            )
                                            mock_repl.return_value = mock_response

                                            with patch(
                                                "src.api.routes.chat._annotate_error"
                                            ) as mock_annotate:
                                                mock_annotate.return_value = mock_response
                                                result = await _handle_chat(request, mock_app_state)

                                                assert result.answer == "REPL fallback after react"
