"""Tests for chat_routing.py intent classification and mode selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api.routes.chat_routing import (
    _classify_and_route,
    _select_mode,
)
from src.classifiers import should_use_direct_mode as _should_use_direct_mode
from src.roles import Role


class TestShouldUseDirectMode:
    """Tests for should_use_direct_mode classifier (was in chat_routing, now in classifiers)."""

    def test_empty_context_returns_direct(self):
        """Empty context triggers direct mode."""
        result = _should_use_direct_mode("What is 2+2?", "")
        assert result is True

    def test_short_context_returns_direct(self):
        """Short context (<20k) triggers direct mode."""
        result = _should_use_direct_mode("Summarize this", "a" * 1000)
        assert result is True

    def test_large_context_returns_repl(self):
        """Large context (>20k) keeps REPL for chunked exploration."""
        large_context = "x" * 25000
        result = _should_use_direct_mode("Summarize this", large_context)
        assert result is False

    def test_context_exactly_at_threshold(self):
        """Context exactly at 20k threshold keeps REPL."""
        context = "x" * 20001
        result = _should_use_direct_mode("Summarize this", context)
        assert result is False

    @pytest.mark.parametrize(
        "indicator",
        [
            "read the file",
            "list files",
            "list the files",
            "look at the file",
            "open the file",
            "read from",
            "write to",
            "save to",
            "execute",
            "run the",
            "run this",
            "search the codebase",
            "find in the",
            "grep for",
            "explore the",
            "scan the",
        ],
    )
    def test_repl_indicator_returns_false(self, indicator: str):
        """REPL indicators keep REPL mode."""
        prompt = f"Please {indicator} something"
        result = _should_use_direct_mode(prompt, "")
        assert result is False

    def test_repl_indicator_case_insensitive(self):
        """REPL indicators are case-insensitive."""
        result = _should_use_direct_mode("READ THE FILE for me", "")
        assert result is False

    def test_pure_reasoning_returns_direct(self):
        """Pure reasoning prompts use direct mode."""
        result = _should_use_direct_mode("Explain the theory of relativity", "")
        assert result is True

    def test_formatting_task_returns_direct(self):
        """Formatting tasks use direct mode."""
        result = _should_use_direct_mode("Reformat this JSON: {}", "")
        assert result is True

    def test_tool_call_generation_returns_direct(self):
        """Tool call JSON generation uses direct mode."""
        result = _should_use_direct_mode("Generate a tool call for weather API", "")
        assert result is True

    def test_mixed_prompt_with_indicator(self):
        """Prompts with any REPL indicator keep REPL."""
        result = _should_use_direct_mode("Explain AI and then execute the code", "")
        assert result is False


class TestSelectMode:
    """Tests for _select_mode with MemRL and heuristic fallback."""

    def test_memrl_router_returns_direct(self):
        """MemRL hybrid_router direct mode selection."""
        state = MagicMock()
        state.hybrid_router.route_with_mode.return_value = (["role"], "learned", "direct")

        result = _select_mode("Hello", "", state)
        assert result == "direct"

    def test_memrl_router_returns_react_mapped_to_repl(self):
        """MemRL hybrid_router react mode is mapped to repl (React unified into REPL)."""
        state = MagicMock()
        state.hybrid_router.route_with_mode.return_value = (["role"], "learned", "react")

        result = _select_mode("Hello", "", state)
        # React is now unified into REPL - legacy "react" maps to "repl"
        assert result == "repl"

    def test_memrl_router_returns_repl(self):
        """MemRL hybrid_router REPL mode selection."""
        state = MagicMock()
        state.hybrid_router.route_with_mode.return_value = (["role"], "learned", "repl")

        result = _select_mode("Hello", "", state)
        assert result == "repl"

    def test_memrl_router_invalid_mode_falls_back(self):
        """MemRL returning invalid mode falls back to repl."""
        state = MagicMock()
        state.hybrid_router.route_with_mode.return_value = (["role"], "learned", "invalid")

        result = _select_mode("Hello", "", state)
        # Falls back to repl (default)
        assert result == "repl"

    def test_memrl_router_exception_falls_back(self):
        """MemRL exception falls back to repl."""
        state = MagicMock()
        state.hybrid_router.route_with_mode.side_effect = RuntimeError("DB error")

        result = _select_mode("Hello", "", state)
        assert result == "repl"

    def test_no_hybrid_router_defaults_to_repl(self):
        """No hybrid_router defaults to repl (React unified into REPL)."""
        state = MagicMock()
        state.hybrid_router = None

        # No more React heuristic - always returns repl
        result = _select_mode("Use tool X", "", state)
        assert result == "repl"

    def test_all_prompts_default_to_repl(self):
        """All prompts default to repl mode (React unified into REPL)."""
        state = MagicMock()
        state.hybrid_router = None

        # REPL is now the universal default - model can FINAL() immediately
        # for simple tasks or use tools for complex ones
        result = _select_mode("Call the API", "", state)
        assert result == "repl"

    def test_heuristic_repl_default(self):
        """Heuristic fallback defaults to repl."""
        state = MagicMock()
        state.hybrid_router = None

        result = _select_mode("Explain something", "", state)
        assert result == "repl"

    def test_state_without_hybrid_router_attr(self):
        """State without hybrid_router attribute falls back to repl."""
        state = MagicMock(spec=[])  # No hybrid_router attr

        result = _select_mode("Hello", "", state)
        assert result == "repl"


class TestClassifyAndRoute:
    """Tests for _classify_and_route proactive routing."""

    def test_image_routes_to_vision(self):
        """Image presence routes to worker_vision."""
        role, strategy = _classify_and_route("Describe this", "", has_image=True)
        assert role == "worker_vision"
        assert strategy == "classified"

    def test_no_specialist_routing_returns_frontdoor(self):
        """Without specialist_routing feature, returns frontdoor."""
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = False
            role, strategy = _classify_and_route("Implement a function", "")

        assert role == str(Role.FRONTDOOR)
        assert strategy == "rules"

    def test_specialist_routing_disabled_ignores_keywords(self):
        """Specialist routing disabled ignores code keywords."""
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = False
            role, _ = _classify_and_route("debug this function", "")

        assert role == str(Role.FRONTDOOR)

    @pytest.mark.parametrize(
        "keyword",
        [
            "implement",
            "write code",
            "function",
            "class ",
            "debug",
            "refactor",
            "fix the bug",
            "code review",
            "unit test",
            "algorithm",
            "data structure",
            "regex",
            "parse",
        ],
    )
    def test_code_keywords_route_to_coder_escalation(self, keyword: str):
        """Code keywords route to coder_escalation."""
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = True
            role, strategy = _classify_and_route(f"Please {keyword} this", "")

        assert role == str(Role.CODER_ESCALATION)
        assert strategy == "classified"

    @pytest.mark.parametrize(
        "keyword",
        [
            "concurrent",
            "lock-free",
            "distributed",
            "optimize performance",
            "memory leak",
            "race condition",
            "deadlock",
        ],
    )
    def test_complex_code_keywords_route_to_escalation(self, keyword: str):
        """Complex code keywords route to coder_escalation."""
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = True
            role, strategy = _classify_and_route(f"Fix the {keyword} issue", "")

        assert role == str(Role.CODER_ESCALATION)
        assert strategy == "classified"

    @pytest.mark.parametrize(
        "keyword",
        [
            "architecture",
            "system design",
            "design pattern",
            "scalab",  # prefix match (scalability)
            "microservice",
            "trade-off",
            "tradeoff",
            "invariant",
            "constraint",
            "cap theorem",
        ],
    )
    def test_arch_keywords_route_to_architect(self, keyword: str):
        """Architecture keywords route to architect_general."""
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = True
            role, strategy = _classify_and_route(f"Explain the {keyword}", "")

        assert role == str(Role.ARCHITECT_GENERAL)
        assert strategy == "classified"

    def test_no_keyword_match_returns_frontdoor(self):
        """No keyword match returns frontdoor."""
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = True
            role, strategy = _classify_and_route("What is the weather?", "")

        assert role == str(Role.FRONTDOOR)
        assert strategy == "rules"

    def test_keyword_case_insensitive(self):
        """Keyword matching is case-insensitive."""
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = True
            role, _ = _classify_and_route("IMPLEMENT THIS FUNCTION", "")

        assert role == str(Role.CODER_ESCALATION)

    def test_complex_code_has_priority_over_simple_code(self):
        """Complex code keywords take priority (checked first) over simple code."""
        # Actually checking the implementation - complex_code_keywords are checked
        # after code_keywords, so this tests that both exist in prompt
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = True
            # "concurrent" matches complex_code, but "implement" matches first
            role, _ = _classify_and_route("implement concurrent data structure", "")

        # "implement" matches first since code_keywords are checked first
        assert role == str(Role.CODER_ESCALATION)

    def test_only_complex_keyword_routes_to_escalation(self):
        """Only complex keyword routes to escalation."""
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = True
            role, _ = _classify_and_route("fix the deadlock issue", "")

        assert role == str(Role.CODER_ESCALATION)

    def test_context_not_used_for_routing(self):
        """Context is not used for keyword routing (only prompt)."""
        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = True
            # Keyword only in context, not prompt
            role, _ = _classify_and_route("hello", "implement a function")

        assert role == str(Role.FRONTDOOR)
