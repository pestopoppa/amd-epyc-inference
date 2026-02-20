"""Characterization tests for src/api/routes/chat.py.

Tests the chat route handlers and helper functions:
- _try_cheap_first() edge cases (disabled, forced, vision, delegated, empty, short)
- /chat endpoint routing via TestClient with mock mode
- /chat/reward endpoint
- _handle_chat dispatcher behavior
- /chat/stream endpoint returns StreamingResponse
"""

from __future__ import annotations

import time
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from src.api.dependencies import dep_app_state
from src.api.models import ChatRequest, ChatResponse, RewardRequest
from src.api.routes.chat import _handle_chat, _try_cheap_first, chat, chat_stream, inject_reward, router
from src.api.routes.chat_utils import RoutingResult
from src.api.state import AppState


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_state():
    """Create a mock AppState with required attributes."""
    state = MagicMock(spec=AppState)
    state.progress_logger = None
    state.hybrid_router = None
    state.tool_registry = None
    state.script_registry = None
    state.registry = None
    state.health_tracker = MagicMock()
    state.admission = MagicMock()
    state.increment_active = MagicMock()
    state.decrement_active = MagicMock()
    state.increment_request = MagicMock()
    return state


@pytest.fixture
def mock_primitives():
    """Create a mock LLMPrimitives."""
    primitives = MagicMock()
    primitives.llm_call = MagicMock(return_value="This is a sufficiently long mock answer for testing purposes.")
    return primitives


@pytest.fixture
def base_routing():
    """Create a baseline RoutingResult for cheap-first tests."""
    return RoutingResult(
        task_id="test-abc123",
        task_ir={"task_type": "chat", "objective": "test"},
        use_mock=False,
        routing_decision=["frontdoor"],
        routing_strategy="rules",
        skill_ids=[],
    )


@pytest.fixture
def base_request():
    """Create a baseline ChatRequest."""
    return ChatRequest(
        prompt="What is the meaning of life?",
        mock_mode=False,
        real_mode=True,
    )


@pytest.fixture
def test_app(mock_state):
    """Create a FastAPI test app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[dep_app_state] = lambda: mock_state
    return app


@pytest.fixture
def client(test_app):
    """Create a synchronous test client."""
    with TestClient(test_app) as test_client:
        yield test_client


# ── _try_cheap_first edge cases ─────────────────────────────────────────────


class TestTryCheapFirstEdgeCases:
    """Tests for the _try_cheap_first speculative pre-filter."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When try_cheap_first_enabled=False, returns None immediately."""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = False
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_forced_mode(
        self, base_routing, mock_primitives, mock_state
    ):
        """When request has force_mode set, returns None."""
        request = ChatRequest(
            prompt="test prompt",
            mock_mode=False,
            real_mode=True,
            force_mode="direct",
        )
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_forced_role(
        self, base_routing, mock_primitives, mock_state
    ):
        """When request has force_role set, returns None."""
        request = ChatRequest(
            prompt="test prompt",
            mock_mode=False,
            real_mode=True,
            force_role="coder",
        )
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_vision_role(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When initial_role is a vision role, returns None (already cheap)."""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "worker_vision",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_worker_explore_role(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When initial_role is worker_explore, returns None (already cheap)."""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "worker_explore",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_delegated_mode(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When execution_mode is 'delegated', returns None."""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "delegated",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_answer(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM returns empty string, returns None."""
        mock_primitives.llm_call.return_value = ""
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_explore"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_short_answer(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM returns answer shorter than 20 chars, returns None."""
        mock_primitives.llm_call.return_value = "Too short"
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_explore"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error_answer(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM returns an error-like answer, returns None."""
        mock_primitives.llm_call.return_value = "[ERROR] Something went wrong with inference"
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_explore"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_response_on_good_answer(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM returns a good answer, returns ChatResponse."""
        mock_primitives.llm_call.return_value = (
            "The meaning of life is a philosophical question that has been debated for centuries."
        )
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_explore"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            with patch(
                "src.api.routes.chat_review._detect_output_quality_issue",
                return_value=None,
            ):
                result = await _try_cheap_first(
                    base_request,
                    base_routing,
                    mock_primitives,
                    mock_state,
                    time.perf_counter(),
                    "frontdoor",
                    "direct",
                )
                assert result is not None
                assert isinstance(result, ChatResponse)
                assert result.cheap_first_attempted is True
                assert result.cheap_first_passed is True
                assert result.routed_to == "worker_explore"
                assert "cheap_first" in result.routing_strategy

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_exception(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When LLM call raises exception, returns None gracefully."""
        mock_primitives.llm_call.side_effect = ConnectionError("Backend down")
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_explore"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            result = await _try_cheap_first(
                base_request,
                base_routing,
                mock_primitives,
                mock_state,
                time.perf_counter(),
                "frontdoor",
                "direct",
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_quality_issue(
        self, base_request, base_routing, mock_primitives, mock_state
    ):
        """When quality detector flags an issue, returns None."""
        mock_primitives.llm_call.return_value = (
            "Here is a long enough answer that should pass length checks easily."
        )
        with patch("src.api.routes.chat.get_config") as mock_cfg:
            mock_cfg.return_value.chat.try_cheap_first_enabled = True
            mock_cfg.return_value.chat.try_cheap_first_phase = "A"
            mock_cfg.return_value.chat.try_cheap_first_role = "worker_explore"
            mock_cfg.return_value.chat.try_cheap_first_max_tokens = 1024
            with patch(
                "src.api.routes.chat_review._detect_output_quality_issue",
                return_value="repetitive output",
            ):
                result = await _try_cheap_first(
                    base_request,
                    base_routing,
                    mock_primitives,
                    mock_state,
                    time.perf_counter(),
                    "frontdoor",
                    "direct",
                )
                assert result is None


# ── /chat endpoint tests ────────────────────────────────────────────────────


class TestChatEndpoint:
    """Tests for POST /chat via TestClient."""

    @pytest.mark.asyncio
    async def test_mock_mode_returns_200(self, mock_state):
        """Mock mode chat request returns 200 with mock answer."""
        mock_response = ChatResponse(
            answer="[MOCK] Processed prompt: Hello...",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
            real_mode=False,
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            response = await chat(ChatRequest(prompt="Hello", mock_mode=True), _FakeRequest(), mock_state)
            assert response is mock_response
            data = response.model_dump()
            assert data["mock_mode"] is True
            assert "[MOCK]" in data["answer"]

    @pytest.mark.asyncio
    async def test_chat_returns_200_for_successful_request(self, mock_state):
        """Chat endpoint returns 200 for a successful non-error response."""
        mock_response = ChatResponse(
            answer="A real answer to the user question.",
            turns=2,
            elapsed_seconds=1.5,
            mock_mode=False,
            real_mode=True,
            routed_to="frontdoor",
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            response = await chat(
                ChatRequest(prompt="Explain recursion", mock_mode=False, real_mode=True),
                _FakeRequest(),
                mock_state,
            )
            data = response.model_dump()
            assert data["answer"] == "A real answer to the user question."
            assert data["turns"] == 2

    @pytest.mark.asyncio
    async def test_chat_returns_error_status_on_error_code(self, mock_state):
        """When _handle_chat returns error_code, HTTP status matches."""
        mock_response = ChatResponse(
            answer="Backend timeout",
            turns=1,
            elapsed_seconds=60.0,
            mock_mode=False,
            real_mode=True,
            error_code=504,
            error_detail="Request timed out",
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            response = await chat(
                ChatRequest(prompt="Slow query", mock_mode=False, real_mode=True),
                _FakeRequest(),
                mock_state,
            )
            assert response.status_code == 504
            data = json.loads(response.body.decode("utf-8"))
            assert data["error_code"] == 504

    @pytest.mark.asyncio
    async def test_chat_503_includes_retry_after(self, mock_state):
        """When error_code=503, response includes Retry-After header."""
        mock_response = ChatResponse(
            answer="Service unavailable",
            turns=0,
            elapsed_seconds=0.0,
            mock_mode=False,
            real_mode=True,
            error_code=503,
            error_detail="Backend down",
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            response = await chat(
                ChatRequest(prompt="test", mock_mode=False, real_mode=True),
                _FakeRequest(),
                mock_state,
            )
            assert response.status_code == 503
            assert response.headers.get("retry-after") == "30"

    @pytest.mark.asyncio
    async def test_chat_increments_and_decrements_active(self, mock_state):
        """Chat endpoint calls increment_active/decrement_active around handling."""
        mock_response = ChatResponse(
            answer="ok",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
        )

        class _FakeRequest:
            async def is_disconnected(self) -> bool:
                return False

        async def fake_handle_chat(*_args, **_kwargs):
            return mock_response

        with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
            await chat(ChatRequest(prompt="test"), _FakeRequest(), mock_state)
            mock_state.increment_active.assert_called_once()
            mock_state.decrement_active.assert_called_once()


# ── /chat/reward endpoint tests ─────────────────────────────────────────────


class TestRewardEndpoint:
    """Tests for POST /chat/reward."""

    @pytest.mark.asyncio
    async def test_inject_reward_returns_success(self, mock_state):
        """inject_reward endpoint returns success dict."""
        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("src.api.routes.chat.store_external_reward", return_value=True) as mock_store, \
             patch("src.api.routes.chat.asyncio.to_thread", new=fake_to_thread):
            response = await inject_reward(
                RewardRequest(
                    task_description="Solve fizzbuzz",
                    action="coder:direct",
                    reward=0.85,
                ),
                mock_state,
            )
            assert response["success"] is True
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_reward_returns_failure(self, mock_state):
        """inject_reward returns success=False when store fails."""
        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("src.api.routes.chat.store_external_reward", return_value=False), \
             patch("src.api.routes.chat.asyncio.to_thread", new=fake_to_thread):
            response = await inject_reward(
                RewardRequest(
                    task_description="Hard problem",
                    action="architect:delegated",
                    reward=-0.5,
                ),
                mock_state,
            )
            assert response["success"] is False

    @pytest.mark.asyncio
    async def test_inject_reward_with_embedding(self, mock_state):
        """inject_reward passes precomputed embedding to store."""
        embedding = [0.1, 0.2, 0.3]
        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("src.api.routes.chat.store_external_reward", return_value=True) as mock_store, \
             patch("src.api.routes.chat.asyncio.to_thread", new=fake_to_thread):
            response = await inject_reward(
                RewardRequest(
                    task_description="Test task",
                    action="frontdoor:direct",
                    reward=1.0,
                    embedding=embedding,
                ),
                mock_state,
            )
            assert response["success"] is True
            # Verify embedding was passed through
            call_args = mock_store.call_args
            assert call_args[0][5] == embedding or call_args[1].get("embedding") == embedding


# ── _handle_chat dispatcher tests ───────────────────────────────────────────


class TestHandleChat:
    """Tests for _handle_chat dispatcher logic."""

    @pytest.mark.asyncio
    async def test_dispatches_to_execute_mock(self, mock_state):
        """_handle_chat returns mock response when use_mock is True."""
        request = ChatRequest(prompt="Hello", mock_mode=True, real_mode=False)
        mock_routing = RoutingResult(
            task_id="test-123",
            task_ir={"task_type": "chat", "objective": "Hello"},
            use_mock=True,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        mock_response = ChatResponse(
            answer="[MOCK] Hello",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
        )
        with patch("src.api.routes.chat._route_request", return_value=mock_routing) as mock_route, \
             patch("src.api.routes.chat._preprocess"), \
             patch("src.api.routes.chat._execute_mock", return_value=mock_response) as mock_exec:
            result = await _handle_chat(request, mock_state)
            mock_route.assert_called_once_with(request, mock_state)
            mock_exec.assert_called_once()
            assert result.mock_mode is True
            assert result.answer == "[MOCK] Hello"

    @pytest.mark.asyncio
    async def test_calls_route_request_first(self, mock_state):
        """_handle_chat calls _route_request as the first stage."""
        request = ChatRequest(prompt="Test", mock_mode=True)
        mock_routing = RoutingResult(
            task_id="test-456",
            task_ir={"task_type": "chat", "objective": "Test"},
            use_mock=True,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        mock_response = ChatResponse(
            answer="[MOCK] Test",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
        )
        with patch("src.api.routes.chat._route_request", return_value=mock_routing) as mock_route, \
             patch("src.api.routes.chat._preprocess"), \
             patch("src.api.routes.chat._execute_mock", return_value=mock_response):
            await _handle_chat(request, mock_state)
            mock_route.assert_called_once_with(request, mock_state)

    @pytest.mark.asyncio
    async def test_calls_preprocess_before_execution(self, mock_state):
        """_handle_chat calls _preprocess after routing."""
        request = ChatRequest(prompt="Preprocess test", mock_mode=True)
        mock_routing = RoutingResult(
            task_id="test-789",
            task_ir={"task_type": "chat", "objective": "Preprocess test"},
            use_mock=True,
            routing_decision=["frontdoor"],
            routing_strategy="mock",
        )
        mock_response = ChatResponse(
            answer="[MOCK] Preprocessed",
            turns=1,
            elapsed_seconds=0.01,
            mock_mode=True,
        )
        with patch("src.api.routes.chat._route_request", return_value=mock_routing), \
             patch("src.api.routes.chat._preprocess") as mock_pre, \
             patch("src.api.routes.chat._execute_mock", return_value=mock_response):
            await _handle_chat(request, mock_state)
            mock_pre.assert_called_once_with(request, mock_state, mock_routing)


# ── /chat/stream endpoint tests ─────────────────────────────────────────────


class TestChatStreamEndpoint:
    """Tests for POST /chat/stream."""

    @pytest.mark.asyncio
    async def test_stream_returns_streaming_response(self, mock_state):
        """chat_stream endpoint returns a streaming response (200)."""
        # Patch features to use legacy streaming (not unified)
        with patch("src.api.routes.chat.features") as mock_features:
            mock_features.return_value.unified_streaming = False
            mock_state.progress_logger = None

            response = await chat_stream(
                ChatRequest(prompt="Stream test", mock_mode=True, real_mode=False),
                mock_state,
            )
            assert isinstance(response, StreamingResponse)
            assert "text/event-stream" in (response.media_type or "")

    @pytest.mark.asyncio
    async def test_stream_mock_mode_contains_done(self, mock_state):
        """Mock mode stream contains [DONE] sentinel."""
        with patch("src.api.routes.chat.features") as mock_features:
            mock_features.return_value.unified_streaming = False
            mock_state.progress_logger = None

            response = await chat_stream(
                ChatRequest(prompt="Hello stream", mock_mode=True, real_mode=False),
                mock_state,
            )
            chunks = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    chunks.append(chunk.decode("utf-8", errors="replace"))
                else:
                    chunks.append(str(chunk))
            body = "".join(chunks)
            assert "[DONE]" in body
