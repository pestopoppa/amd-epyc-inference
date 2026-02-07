"""Integration tests for the pydantic-graph orchestration flow.

Tests end-to-end graph execution with mock LLM and REPL.
"""

from __future__ import annotations

import pytest

from src.graph import (
    run_task,
    TaskDeps,
    TaskResult,
    GraphConfig,
)
from src.roles import Role


# Reuse mock infrastructure from test_graph_nodes
from tests.unit.test_graph_nodes import (
    MockREPLResult,
    make_deps,
    make_state,
)


class TestRunTask:
    """Test the run_task() convenience function."""

    @pytest.mark.asyncio
    async def test_basic_success(self):
        state = make_state(prompt="What is 2+2?")
        deps = make_deps(
            repl_results=[MockREPLResult(output="4", is_final=True)],
        )
        result = await run_task(state, deps, start_role=Role.FRONTDOOR)
        assert isinstance(result, TaskResult)
        assert result.success is True
        assert result.turns == 1

    @pytest.mark.asyncio
    async def test_worker_to_coder_escalation(self):
        state = make_state(current_role=Role.WORKER_GENERAL)
        deps = make_deps(
            repl_results=[
                MockREPLResult(error="SyntaxError in test"),
                MockREPLResult(error="SyntaxError in test"),
                MockREPLResult(output="coder fixed", is_final=True),
            ],
            config=GraphConfig(max_retries=2, max_escalations=2, max_turns=10),
        )
        result = await run_task(state, deps, start_role=Role.WORKER_GENERAL)
        assert isinstance(result, TaskResult)
        # Should have escalated
        assert len(state.role_history) > 1

    @pytest.mark.asyncio
    async def test_full_chain_frontdoor_to_architect(self):
        """Test escalation from frontdoor -> coder -> architect."""
        state = make_state(current_role=Role.FRONTDOOR)
        deps = make_deps(
            repl_results=[
                # Frontdoor fails twice
                MockREPLResult(error="SyntaxError"),
                MockREPLResult(error="SyntaxError"),
                # Coder fails twice
                MockREPLResult(error="Logic error"),
                MockREPLResult(error="Logic error"),
                # Architect succeeds
                MockREPLResult(output="architect solution", is_final=True),
            ],
            config=GraphConfig(max_retries=2, max_escalations=3, max_turns=15),
        )
        result = await run_task(state, deps, start_role=Role.FRONTDOOR)
        assert isinstance(result, TaskResult)
        assert state.escalation_count >= 2

    @pytest.mark.asyncio
    async def test_no_primitives(self):
        """Should fail gracefully when no LLM primitives."""
        state = make_state()
        deps = TaskDeps(config=GraphConfig(max_turns=2))
        result = await run_task(state, deps, start_role=Role.FRONTDOOR)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_role_history_tracking(self):
        state = make_state(current_role=Role.CODER_PRIMARY, role_history=[])
        deps = make_deps(
            repl_results=[MockREPLResult(output="done", is_final=True)],
        )
        await run_task(state, deps, start_role=Role.CODER_PRIMARY)
        assert "coder_primary" in state.role_history

    @pytest.mark.asyncio
    async def test_format_error_never_escalates(self):
        """FORMAT errors should retry only, never escalate."""
        state = make_state(current_role=Role.CODER_PRIMARY)
        deps = make_deps(
            repl_results=[
                MockREPLResult(error="ruff format check failed"),
                MockREPLResult(error="ruff format check failed"),
                MockREPLResult(error="ruff format check failed"),
            ],
            config=GraphConfig(max_retries=2, max_escalations=2, max_turns=10),
        )
        result = await run_task(state, deps, start_role=Role.CODER_PRIMARY)
        # Should fail without escalating (FORMAT is no_escalate)
        assert result.success is False
        # escalation_count should be 0 for format errors
        assert state.escalation_count == 0
