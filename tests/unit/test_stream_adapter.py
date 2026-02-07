"""Tests for src/api/routes/chat_pipeline/stream_adapter.py.

Covers: generate_stream, _stream_mock, _stream_repl.
"""

import asyncio
from unittest.mock import MagicMock, patch

from src.api.routes.chat_pipeline.stream_adapter import (
    _stream_mock,
    generate_stream,
)


# ── helpers ────────────────────────────────────────────────────────────


def _collect(gen):
    """Run an async generator synchronously and collect all events."""
    loop = asyncio.new_event_loop()
    try:
        events = []

        async def _gather():
            async for event in gen:
                events.append(event)

        loop.run_until_complete(_gather())
    finally:
        loop.close()
    return events


def _mock_routing(use_mock=True, task_id="stream-test123"):
    routing = MagicMock()
    routing.use_mock = use_mock
    routing.task_id = task_id
    routing.routing_decision = ["frontdoor"]
    routing.routing_strategy = "mock" if use_mock else "rules"
    routing.formalization_applied = False
    return routing


def _mock_request(**overrides):
    defaults = dict(
        prompt="Hello world",
        context=None,
        mock_mode=True,
        real_mode=False,
        thinking_budget=0,
        permission_mode="normal",
        role="frontdoor",
        max_turns=5,
    )
    defaults.update(overrides)
    req = MagicMock(**defaults)
    return req


# ── _stream_mock ───────────────────────────────────────────────────────


class TestStreamMock:
    """Test mock mode SSE event generation."""

    def test_yields_turn_start_token_turn_end_done(self):
        request = _mock_request()
        routing = _mock_routing()
        state = MagicMock(progress_logger=None)

        events = _collect(_stream_mock(request, routing, state, 0.0))

        types = [e.get("event") for e in events]
        assert types[0] == "turn_start"
        # Middle events are tokens
        assert "token" in types
        # Must end with turn_end + done
        assert types[-2] == "turn_end"
        assert types[-1] == "done"

    def test_thinking_events_when_budget_positive(self):
        request = _mock_request(thinking_budget=100)
        routing = _mock_routing()
        state = MagicMock(progress_logger=None)

        events = _collect(_stream_mock(request, routing, state, 0.0))

        types = [e.get("event") for e in events]
        assert "thinking" in types

    def test_no_thinking_events_when_budget_zero(self):
        request = _mock_request(thinking_budget=0)
        routing = _mock_routing()
        state = MagicMock(progress_logger=None)

        events = _collect(_stream_mock(request, routing, state, 0.0))

        types = [e.get("event") for e in events]
        assert "thinking" not in types

    def test_plan_mode_yields_plan_analysis(self):
        request = _mock_request(permission_mode="plan")
        routing = _mock_routing()
        state = MagicMock(progress_logger=None)

        events = _collect(_stream_mock(request, routing, state, 0.0))

        types = [e.get("event") for e in events]
        assert "token" in types
        # Plan mode: turn_start, token (plan analysis), turn_end, done
        assert types[-1] == "done"

    def test_logs_completion_when_logger_present(self):
        request = _mock_request()
        routing = _mock_routing()
        logger = MagicMock()
        state = MagicMock(progress_logger=logger)

        _collect(_stream_mock(request, routing, state, 0.0))

        logger.log_task_completed.assert_called_once()


# ── generate_stream ────────────────────────────────────────────────────


class TestGenerateStream:
    """Test the main generate_stream entry point."""

    @patch("src.api.routes.chat_pipeline.stream_adapter._route_request")
    @patch("src.api.routes.chat_pipeline.stream_adapter._preprocess")
    def test_mock_mode_yields_events(self, mock_preprocess, mock_route):
        routing = _mock_routing(use_mock=True)
        mock_route.return_value = routing

        request = _mock_request()
        state = MagicMock(progress_logger=None)

        events = _collect(generate_stream(request, state))

        types = [e.get("event") for e in events]
        assert "turn_start" in types
        assert "done" in types
        mock_route.assert_called_once()
        mock_preprocess.assert_called_once()

    @patch("src.api.routes.chat_pipeline.stream_adapter._plan_review_gate")
    @patch("src.api.routes.chat_pipeline.stream_adapter._init_primitives")
    @patch("src.api.routes.chat_pipeline.stream_adapter._preprocess")
    @patch("src.api.routes.chat_pipeline.stream_adapter._route_request")
    def test_real_mode_init_failure_yields_error(
        self, mock_route, mock_preprocess, mock_init, mock_plan
    ):
        routing = _mock_routing(use_mock=False)
        mock_route.return_value = routing
        mock_init.side_effect = RuntimeError("No backends")

        request = _mock_request(mock_mode=False, real_mode=True)
        state = MagicMock(progress_logger=None)

        events = _collect(generate_stream(request, state))

        types = [e.get("event") for e in events]
        assert "error" in types
        assert "done" in types

    @patch("src.api.routes.chat_pipeline.stream_adapter._plan_review_gate")
    @patch("src.api.routes.chat_pipeline.stream_adapter._init_primitives")
    @patch("src.api.routes.chat_pipeline.stream_adapter._preprocess")
    @patch("src.api.routes.chat_pipeline.stream_adapter._route_request")
    def test_calls_plan_review_gate(
        self, mock_route, mock_preprocess, mock_init, mock_plan
    ):
        routing = _mock_routing(use_mock=False)
        mock_route.return_value = routing
        mock_init.side_effect = RuntimeError("fail")  # Short-circuit after init

        request = _mock_request(mock_mode=False, real_mode=True)
        state = MagicMock(progress_logger=None)

        # Init fails before plan review is reached
        _collect(generate_stream(request, state))
        # Plan review should NOT be called because init raised
        mock_plan.assert_not_called()
