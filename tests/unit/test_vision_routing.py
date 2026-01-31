"""Unit tests for vision routing: tool whitelists, executable tools, and stop constants.

All tests mock httpx and OCR service — no live servers required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Constant relationship tests ─────────────────────────────────────────────


class TestVisionConstants:
    """Tests for vision-related constants in prompt_builders and chat."""

    def test_executable_tools_match_descriptions_keys(self):
        """VISION_REACT_EXECUTABLE_TOOLS must match VISION_TOOL_DESCRIPTIONS keys."""
        from src.prompt_builders import (
            VISION_REACT_EXECUTABLE_TOOLS,
            VISION_TOOL_DESCRIPTIONS,
        )

        assert VISION_REACT_EXECUTABLE_TOOLS == frozenset(VISION_TOOL_DESCRIPTIONS.keys())

    def test_executable_tools_subset_of_whitelist(self):
        """VISION_REACT_EXECUTABLE_TOOLS must be a subset of VISION_REACT_TOOL_WHITELIST."""
        from src.prompt_builders import (
            VISION_REACT_EXECUTABLE_TOOLS,
            VISION_REACT_TOOL_WHITELIST,
        )

        assert VISION_REACT_EXECUTABLE_TOOLS <= VISION_REACT_TOOL_WHITELIST

    def test_qwen_stop_constant_exists(self):
        """QWEN_STOP constant exists and equals '<|im_end|>'."""
        from src.api.routes.chat import QWEN_STOP

        assert QWEN_STOP == "<|im_end|>"

    def test_vision_tool_descriptions_non_empty(self):
        """Each tool description is a non-empty string."""
        from src.prompt_builders import VISION_TOOL_DESCRIPTIONS

        for name, desc in VISION_TOOL_DESCRIPTIONS.items():
            assert isinstance(desc, str), f"Description for '{name}' is not a string"
            assert len(desc) > 10, f"Description for '{name}' is too short"


# ── _execute_vision_tool tests ──────────────────────────────────────────────


def _run_async(coro):
    """Run an async coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestExecuteVisionTool:
    """Tests for _execute_vision_tool() dispatch."""

    def test_calculate_tool(self):
        """calculate(expression="2+3") returns '5'."""
        from src.api.routes.chat import _execute_vision_tool

        result = _run_async(_execute_vision_tool('calculate(expression="2+3")', "dummy_b64"))
        assert result == "5"

    def test_get_current_date(self):
        """get_current_date() returns a date string."""
        from src.api.routes.chat import _execute_vision_tool

        result = _run_async(_execute_vision_tool("get_current_date()", "dummy_b64"))
        # Should contain year-month-day pattern
        assert len(result) >= 10
        assert "-" in result

    def test_get_current_time(self):
        """get_current_time() returns an ISO timestamp."""
        from src.api.routes.chat import _execute_vision_tool

        result = _run_async(_execute_vision_tool("get_current_time()", "dummy_b64"))
        assert "T" in result  # ISO format has T separator

    @patch("httpx.AsyncClient")
    def test_ocr_extract_success(self, mock_client_cls):
        """ocr_extract sends POST to port 9001 and returns text."""
        from src.api.routes.chat import _execute_vision_tool

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "Extracted OCR text here"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = _run_async(
            _execute_vision_tool('ocr_extract(image_base64="current")', "test_b64_data")
        )
        assert result == "Extracted OCR text here"

    @patch("httpx.AsyncClient")
    def test_ocr_extract_http_error(self, mock_client_cls):
        """OCR HTTP error returns error message."""
        from src.api.routes.chat import _execute_vision_tool

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = _run_async(
            _execute_vision_tool('ocr_extract(image_base64="current")', "test_b64")
        )
        assert "[OCR error: HTTP 500]" in result

    def test_unknown_tool(self):
        """Unknown tool returns error listing available tools."""
        from src.api.routes.chat import _execute_vision_tool

        result = _run_async(
            _execute_vision_tool('unknown_tool(arg="val")', "dummy_b64")
        )
        assert "not available" in result
        assert "unknown_tool" in result
        # Should list available tools
        assert "calculate" in result
        assert "ocr_extract" in result

    def test_unparseable_action(self):
        """Unparseable action string returns parse error."""
        from src.api.routes.chat import _execute_vision_tool

        result = _run_async(
            _execute_vision_tool("this is not a valid action", "dummy_b64")
        )
        assert "[ERROR: Could not parse action" in result
