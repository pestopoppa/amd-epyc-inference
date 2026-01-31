#!/usr/bin/env python3
"""Tests for ReAct-style tool loop in chat.py.

Tests the ReAct mode detection, argument parsing, and answer extraction
without requiring a live LLM backend.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestParseReactArgs:
    """Tests for _parse_react_args()."""

    def test_empty_string(self):
        from src.api.routes.chat_react import _parse_react_args
        assert _parse_react_args("") == {}

    def test_single_string_arg(self):
        from src.api.routes.chat_react import _parse_react_args
        result = _parse_react_args('query="quantum computing"')
        assert result == {"query": "quantum computing"}

    def test_multiple_args(self):
        from src.api.routes.chat_react import _parse_react_args
        result = _parse_react_args('query="machine learning", max_results=5')
        assert result == {"query": "machine learning", "max_results": 5}

    def test_numeric_arg(self):
        from src.api.routes.chat_react import _parse_react_args
        result = _parse_react_args("expression=\"2+2\"")
        assert result == {"expression": "2+2"}

    def test_single_quotes(self):
        from src.api.routes.chat_react import _parse_react_args
        result = _parse_react_args("query='test value'")
        assert result == {"query": "test value"}

    def test_boolean_arg(self):
        from src.api.routes.chat_react import _parse_react_args
        result = _parse_react_args("verbose=True")
        assert result == {"verbose": True}

    def test_comma_in_quoted_string(self):
        from src.api.routes.chat_react import _parse_react_args
        result = _parse_react_args('query="hello, world"')
        assert result == {"query": "hello, world"}

    def test_no_equals(self):
        from src.api.routes.chat_react import _parse_react_args
        # Should skip parts without =
        result = _parse_react_args("orphan_value")
        assert result == {}


class TestShouldUseReactMode:
    """Tests for _should_use_react_mode()."""

    @patch("src.api.routes.chat_react.features")
    def test_disabled_by_feature_flag(self, mock_features):
        from src.api.routes.chat_react import _should_use_react_mode
        mock_features.return_value = MagicMock(react_mode=False)
        assert _should_use_react_mode("search for quantum computing") is False

    @patch("src.api.routes.chat_react.features")
    def test_enabled_with_search_keyword(self, mock_features):
        from src.api.routes.chat_react import _should_use_react_mode
        mock_features.return_value = MagicMock(react_mode=True)
        assert _should_use_react_mode("search for quantum computing papers") is True

    @patch("src.api.routes.chat_react.features")
    def test_enabled_with_calculate(self, mock_features):
        from src.api.routes.chat_react import _should_use_react_mode
        mock_features.return_value = MagicMock(react_mode=True)
        assert _should_use_react_mode("calculate the area of a circle with radius 5") is True

    @patch("src.api.routes.chat_react.features")
    def test_enabled_with_date_query(self, mock_features):
        from src.api.routes.chat_react import _should_use_react_mode
        mock_features.return_value = MagicMock(react_mode=True)
        assert _should_use_react_mode("what is the current date?") is True

    @patch("src.api.routes.chat_react.features")
    def test_no_match_on_plain_question(self, mock_features):
        from src.api.routes.chat_react import _should_use_react_mode
        mock_features.return_value = MagicMock(react_mode=True)
        assert _should_use_react_mode("explain the theory of relativity") is False

    @patch("src.api.routes.chat_react.features")
    def test_large_context_prevents_react(self, mock_features):
        from src.api.routes.chat_react import _should_use_react_mode
        mock_features.return_value = MagicMock(react_mode=True)
        assert _should_use_react_mode("search for info", "x" * 6000) is False


class TestReactModeAnswer:
    """Tests for _react_mode_answer() with mocked LLM."""

    def test_direct_final_answer(self):
        """LLM immediately produces a Final Answer."""
        from src.api.routes.chat_react import _react_mode_answer

        mock_primitives = MagicMock()
        mock_primitives.llm_call.return_value = (
            "Thought: I know the answer without tools.\n"
            "Final Answer: 42"
        )

        result, tools, _ = _react_mode_answer(
            prompt="What is 6 times 7?",
            context="",
            primitives=mock_primitives,
            role="frontdoor",
            tool_registry=None,
        )
        assert result == "42"
        assert tools == 0

    def test_tool_call_then_answer(self):
        """LLM calls a tool then produces Final Answer."""
        from src.api.routes.chat_react import _react_mode_answer

        mock_primitives = MagicMock()
        # First call: Action
        # Second call: Final Answer after observation
        mock_primitives.llm_call.side_effect = [
            'Thought: I need to calculate this.\nAction: calculate(expression="6*7")',
            "Thought: The calculation returned 42.\nFinal Answer: 42",
        ]

        mock_registry = MagicMock()
        mock_registry.invoke.return_value = "42"
        mock_registry.list_tools.return_value = []

        result, tools, _ = _react_mode_answer(
            prompt="What is 6 times 7?",
            context="",
            primitives=mock_primitives,
            role="frontdoor",
            tool_registry=mock_registry,
        )
        assert result == "42"
        assert tools == 1
        mock_registry.invoke.assert_called_once_with("calculate", "frontdoor", expression="6*7")

    def test_disallowed_tool_rejected(self):
        """Tool not in whitelist is rejected."""
        from src.api.routes.chat_react import _react_mode_answer

        mock_primitives = MagicMock()
        mock_primitives.llm_call.side_effect = [
            'Thought: Let me run a shell command.\nAction: run_shell(cmd="ls -la")',
            "Thought: That tool is not available.\nFinal Answer: Cannot access shell.",
        ]

        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = []

        result, tools, _ = _react_mode_answer(
            prompt="list my files",
            context="",
            primitives=mock_primitives,
            role="frontdoor",
            tool_registry=mock_registry,
        )
        assert "Cannot access shell" in result
        assert tools == 0  # Disallowed tool not counted

    def test_no_action_treated_as_answer(self):
        """Response with no Action and no Final Answer is treated as answer."""
        from src.api.routes.chat_react import _react_mode_answer

        mock_primitives = MagicMock()
        mock_primitives.llm_call.return_value = (
            "Thought: The answer is simply 42."
        )

        result, tools, _ = _react_mode_answer(
            prompt="What is the meaning of life?",
            context="",
            primitives=mock_primitives,
            role="frontdoor",
        )
        assert "42" in result
        assert tools == 0

    def test_max_turns_reached(self):
        """Test behavior when max turns are reached."""
        from src.api.routes.chat_react import _react_mode_answer

        mock_primitives = MagicMock()
        # Always produce an action without final answer
        mock_primitives.llm_call.return_value = (
            'Thought: Still searching.\nAction: calculate(expression="1+1")'
        )

        mock_registry = MagicMock()
        mock_registry.invoke.return_value = "2"
        mock_registry.list_tools.return_value = []

        result, tools, _ = _react_mode_answer(
            prompt="infinite loop test",
            context="",
            primitives=mock_primitives,
            role="frontdoor",
            tool_registry=mock_registry,
            max_turns=2,
        )
        assert "max turns" in result.lower() or "Still searching" in result
        assert tools == 2  # Tool invoked each of 2 turns


class TestBuildReactPrompt:
    """Tests for build_react_prompt()."""

    def test_basic_prompt_structure(self):
        from src.prompt_builders import build_react_prompt
        result = build_react_prompt("What is 2+2?")
        assert "Question: What is 2+2?" in result
        assert "Final Answer:" in result
        assert "Action:" in result
        assert "Observation:" in result

    def test_with_context(self):
        from src.prompt_builders import build_react_prompt
        result = build_react_prompt("Summarize this", context="Some text here")
        assert "Context:" in result
        assert "Some text here" in result

    def test_static_tool_descriptions(self):
        from src.prompt_builders import build_react_prompt
        result = build_react_prompt("Calculate something")
        assert "calculate" in result


class TestStripToolOutputs:
    """Tests for delimiter-based _strip_tool_outputs()."""

    def test_strip_delimited_output(self):
        from src.api.routes.chat_utils import _strip_tool_outputs
        text = "Hello <<<TOOL_OUTPUT>>>{\"role\": \"frontdoor\"}<<<END_TOOL_OUTPUT>>> World"
        result = _strip_tool_outputs(text, [])
        assert result == "Hello  World"

    def test_strip_multiline_delimited_output(self):
        from src.api.routes.chat_utils import _strip_tool_outputs
        text = "Before\n<<<TOOL_OUTPUT>>>line1\nline2\nline3<<<END_TOOL_OUTPUT>>>\nAfter"
        result = _strip_tool_outputs(text, [])
        assert "Before" in result
        assert "After" in result
        assert "line1" not in result

    def test_legacy_fallback(self):
        from src.api.routes.chat_utils import _strip_tool_outputs
        text = 'Hello {"role": "frontdoor"} World'
        result = _strip_tool_outputs(text, ['{"role": "frontdoor"}'])
        assert result == "Hello  World"

    def test_empty_text(self):
        from src.api.routes.chat_utils import _strip_tool_outputs
        assert _strip_tool_outputs("", []) == ""


class TestDetectFormatConstraints:
    """Tests for detect_format_constraints()."""

    def test_word_count(self):
        from src.prompt_builders import detect_format_constraints
        result = detect_format_constraints("Answer in exactly 5 words")
        assert any("5 words" in c for c in result)

    def test_json_format(self):
        from src.prompt_builders import detect_format_constraints
        result = detect_format_constraints("Respond in JSON format")
        assert any("JSON" in c for c in result)

    def test_numbered_list(self):
        from src.prompt_builders import detect_format_constraints
        result = detect_format_constraints("Give me a numbered list of items")
        assert any("numbered list" in c for c in result)

    def test_no_constraints(self):
        from src.prompt_builders import detect_format_constraints
        result = detect_format_constraints("Tell me about quantum physics")
        assert result == []

    def test_uppercase(self):
        from src.prompt_builders import detect_format_constraints
        result = detect_format_constraints("Write the answer in UPPER case")
        assert any("uppercase" in c for c in result)

    def test_comma_separated(self):
        from src.prompt_builders import detect_format_constraints
        result = detect_format_constraints("List them comma-separated")
        assert any("comma-separated" in c for c in result)
