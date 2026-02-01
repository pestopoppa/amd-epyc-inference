#!/usr/bin/env python3
"""Unit tests for src/sse_utils.py."""

import json

import pytest

from src.sse_utils import (
    done_event,
    error_event,
    final_event,
    format_done_event,
    format_sse_manual,
    sse_event,
    thinking_event,
    token_event,
    turn_end_event,
    turn_start_event,
)


class TestSSEEvent:
    """Test sse_event() function."""

    def test_sse_event_with_dict(self):
        """Test sse_event with dict data."""
        event = sse_event("token", {"content": "Hello"})

        assert event["event"] == "token"
        assert "data" in event

        # Data should be JSON string with type included
        data = json.loads(event["data"])
        assert data["type"] == "token"
        assert data["content"] == "Hello"

    def test_sse_event_with_string(self):
        """Test sse_event with string data."""
        event = sse_event("error", "Something went wrong")

        assert event["event"] == "error"
        assert event["data"] == "Something went wrong"

    def test_sse_event_type_in_data(self):
        """Test sse_event includes type in data for backward compatibility."""
        event = sse_event("turn_start", {"turn": 1})
        data = json.loads(event["data"])

        assert data["type"] == "turn_start"
        assert data["turn"] == 1


class TestFormatSSEManual:
    """Test format_sse_manual() function."""

    def test_format_sse_manual(self):
        """Test manual SSE formatting."""
        event = sse_event("token", {"content": "test"})
        formatted = format_sse_manual(event)

        assert formatted.startswith("data: ")
        assert formatted.endswith("\n\n")
        assert "token" in formatted

    def test_format_done_event(self):
        """Test [DONE] event formatting."""
        formatted = format_done_event()

        assert formatted == "data: [DONE]\n\n"


class TestConvenienceFunctions:
    """Test convenience event functions."""

    def test_token_event(self):
        """Test token_event() creates correct event."""
        event = token_event("Hello")

        assert event["event"] == "token"
        data = json.loads(event["data"])
        assert data["content"] == "Hello"

    def test_thinking_event(self):
        """Test thinking_event() creates correct event."""
        event = thinking_event("Let me think...")

        assert event["event"] == "thinking"
        data = json.loads(event["data"])
        assert data["content"] == "Let me think..."

    def test_turn_start_event(self):
        """Test turn_start_event() creates correct event."""
        event = turn_start_event(turn=3, role="coder")

        assert event["event"] == "turn_start"
        data = json.loads(event["data"])
        assert data["turn"] == 3
        assert data["role"] == "coder"

    def test_turn_end_event(self):
        """Test turn_end_event() creates correct event."""
        event = turn_end_event(tokens=100, elapsed_ms=5000)

        assert event["event"] == "turn_end"
        data = json.loads(event["data"])
        assert data["tokens"] == 100
        assert data["elapsed_ms"] == 5000

    def test_error_event(self):
        """Test error_event() creates correct event."""
        event = error_event("Something failed")

        assert event["event"] == "error"
        data = json.loads(event["data"])
        assert data["message"] == "Something failed"

    def test_final_event(self):
        """Test final_event() creates correct event."""
        event = final_event("The answer is 42")

        assert event["event"] == "final"
        data = json.loads(event["data"])
        assert data["answer"] == "The answer is 42"

    def test_done_event(self):
        """Test done_event() creates special DONE marker."""
        event = done_event()

        assert event["event"] == "done"
        assert event["data"] == "[DONE]"
