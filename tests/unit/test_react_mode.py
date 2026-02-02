#!/usr/bin/env python3
"""Tests for ReAct-style tool loop in chat.py.

Tests the ReAct mode detection, argument parsing, and answer extraction
without requiring a live LLM backend.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features import Features


# Lightweight test stub for LLMPrimitives (replaces MagicMock)
class StubLLMPrimitives:
    """Test stub for LLMPrimitives that replaces MagicMock.

    Only mocks the external LLM call - all internal state is real.
    """

    def __init__(self, responses=None):
        """Initialize stub with predefined responses.

        Args:
            responses: Either a single string response or a list of responses
                      (for side_effect-like behavior)
        """
        if responses is None:
            self.responses = []
        elif isinstance(responses, str):
            self.responses = [responses]
        else:
            self.responses = list(responses)
        self.call_count = 0
        self.call_log = []

    def llm_call(self, prompt, role=None, n_tokens=None, skip_suffix=None, stop_sequences=None, **kwargs):
        """Mock llm_call that returns predefined responses."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        elif self.responses:
            # If we run out of responses, repeat the last one
            response = self.responses[-1]
        else:
            response = "Mock response"

        self.call_count += 1
        self.call_log.append({
            'prompt': prompt,
            'role': role,
            'n_tokens': n_tokens,
            'skip_suffix': skip_suffix,
            'stop_sequences': stop_sequences,
        })
        return response


# Lightweight test stub for ToolRegistry (replaces MagicMock)
class StubToolRegistry:
    """Test stub for ToolRegistry that replaces MagicMock."""

    def __init__(self, tool_results=None):
        """Initialize stub with predefined tool results.

        Args:
            tool_results: Dict mapping tool names to their return values
        """
        self.tool_results = tool_results or {}
        self.invocations = []

    def invoke(self, tool_name, role, **kwargs):
        """Mock tool invocation."""
        self.invocations.append({
            'tool_name': tool_name,
            'role': role,
            'kwargs': kwargs,
        })
        return self.tool_results.get(tool_name, f"Result from {tool_name}")

    def list_tools(self):
        """Mock tool listing - returns list of tool info dicts."""
        return [
            {
                "name": name,
                "description": f"Mock tool {name}",
                "parameters": {}
            }
            for name in self.tool_results.keys()
        ]


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

    def test_disabled_by_feature_flag(self):
        from src.api.routes.chat_react import _should_use_react_mode
        # Use real Features object instead of MagicMock
        with patch("src.api.routes.chat_react.features") as mock_features:
            mock_features.return_value = Features(react_mode=False)
            assert _should_use_react_mode("search for quantum computing") is False

    def test_enabled_with_search_keyword(self):
        from src.api.routes.chat_react import _should_use_react_mode
        # Use real Features object instead of MagicMock
        with patch("src.api.routes.chat_react.features") as mock_features:
            mock_features.return_value = Features(react_mode=True)
            assert _should_use_react_mode("search for quantum computing papers") is True

    def test_enabled_with_calculate(self):
        from src.api.routes.chat_react import _should_use_react_mode
        # Use real Features object instead of MagicMock
        with patch("src.api.routes.chat_react.features") as mock_features:
            mock_features.return_value = Features(react_mode=True)
            assert _should_use_react_mode("calculate the area of a circle with radius 5") is True

    def test_enabled_with_date_query(self):
        from src.api.routes.chat_react import _should_use_react_mode
        # Use real Features object instead of MagicMock
        with patch("src.api.routes.chat_react.features") as mock_features:
            mock_features.return_value = Features(react_mode=True)
            assert _should_use_react_mode("what is the current date?") is True

    def test_no_match_on_plain_question(self):
        from src.api.routes.chat_react import _should_use_react_mode
        # Use real Features object instead of MagicMock
        with patch("src.api.routes.chat_react.features") as mock_features:
            mock_features.return_value = Features(react_mode=True)
            assert _should_use_react_mode("explain the theory of relativity") is False

    def test_large_context_prevents_react(self):
        from src.api.routes.chat_react import _should_use_react_mode
        # Use real Features object instead of MagicMock
        with patch("src.api.routes.chat_react.features") as mock_features:
            mock_features.return_value = Features(react_mode=True)
            assert _should_use_react_mode("search for info", "x" * 6000) is False


class TestReactModeAnswer:
    """Tests for _react_mode_answer() with mocked LLM."""

    def test_direct_final_answer(self):
        """LLM immediately produces a Final Answer."""
        from src.api.routes.chat_react import _react_mode_answer

        # Use real stub instead of MagicMock
        primitives = StubLLMPrimitives(
            "Thought: I know the answer without tools.\n"
            "Final Answer: 42"
        )

        result, tools, _ = _react_mode_answer(
            prompt="What is 6 times 7?",
            context="",
            primitives=primitives,
            role="frontdoor",
            tool_registry=None,
        )
        assert result == "42"
        assert tools == 0

    def test_tool_call_then_answer(self):
        """LLM calls a tool then produces Final Answer."""
        from src.api.routes.chat_react import _react_mode_answer

        # Use real stub with multiple responses instead of MagicMock side_effect
        primitives = StubLLMPrimitives([
            'Thought: I need to calculate this.\nAction: calculate(expression="6*7")',
            "Thought: The calculation returned 42.\nFinal Answer: 42",
        ])

        # Use real stub instead of MagicMock
        registry = StubToolRegistry({"calculate": "42"})

        result, tools, _ = _react_mode_answer(
            prompt="What is 6 times 7?",
            context="",
            primitives=primitives,
            role="frontdoor",
            tool_registry=registry,
        )
        assert result == "42"
        assert tools == 1
        # Verify tool was called with correct args
        assert len(registry.invocations) == 1
        assert registry.invocations[0]['tool_name'] == "calculate"
        assert registry.invocations[0]['kwargs']['expression'] == "6*7"

    def test_disallowed_tool_rejected(self):
        """Tool not in whitelist is rejected."""
        from src.api.routes.chat_react import _react_mode_answer

        # Use real stub with multiple responses
        primitives = StubLLMPrimitives([
            'Thought: Let me run a shell command.\nAction: run_shell(cmd="ls -la")',
            "Thought: That tool is not available.\nFinal Answer: Cannot access shell.",
        ])

        # Use real stub
        registry = StubToolRegistry()

        result, tools, _ = _react_mode_answer(
            prompt="list my files",
            context="",
            primitives=primitives,
            role="frontdoor",
            tool_registry=registry,
        )
        assert "Cannot access shell" in result
        assert tools == 0  # Disallowed tool not counted

    def test_no_action_treated_as_answer(self):
        """Response with no Action and no Final Answer is treated as answer."""
        from src.api.routes.chat_react import _react_mode_answer

        # Use real stub
        primitives = StubLLMPrimitives(
            "Thought: The answer is simply 42."
        )

        result, tools, _ = _react_mode_answer(
            prompt="What is the meaning of life?",
            context="",
            primitives=primitives,
            role="frontdoor",
        )
        assert "42" in result
        assert tools == 0

    def test_max_turns_reached(self):
        """Test behavior when max turns are reached."""
        from src.api.routes.chat_react import _react_mode_answer

        # Use real stub that always produces an action
        primitives = StubLLMPrimitives(
            'Thought: Still searching.\nAction: calculate(expression="1+1")'
        )

        # Use real stub
        registry = StubToolRegistry({"calculate": "2"})

        result, tools, _ = _react_mode_answer(
            prompt="infinite loop test",
            context="",
            primitives=primitives,
            role="frontdoor",
            tool_registry=registry,
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
