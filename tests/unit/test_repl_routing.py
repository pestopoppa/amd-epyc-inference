#!/usr/bin/env python3
"""Unit tests for the REPL routing tools (_RoutingMixin)."""

import json
from unittest.mock import Mock, MagicMock, patch
import pytest

from src.repl_environment import REPLEnvironment, REPLConfig


class TestMyRole:
    """Test _my_role() / my_role() function."""

    def test_my_role_returns_json(self):
        """Test my_role() returns role information as JSON."""
        repl = REPLEnvironment(context="test", role="worker_explore")
        result = repl.execute("output = my_role(); print('role' in output)")

        assert result.error is None
        assert "True" in result.output

    def test_my_role_contains_tier_info(self):
        """Test my_role() includes tier information."""
        repl = REPLEnvironment(context="test", role="coder_primary")
        # Execute and capture the output
        result = repl.execute("""
output = my_role()
# Strip tool delimiters if present
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(data['role'])
print(data['tier'])
""")

        assert result.error is None
        assert "coder_primary" in result.output
        assert "tier" in result.output.lower() or result.output.strip()

    def test_my_role_default_role(self):
        """Test my_role() returns default role when not set."""
        repl = REPLEnvironment(context="test")
        # Default role should be worker_general
        result = repl.execute("""
output = my_role()
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(data['role'])
""")

        assert result.error is None
        assert "worker_general" in result.output


class TestResolveRoleAlias:
    """Test _resolve_role_alias() method."""

    def test_resolve_known_alias(self):
        """Test resolving known role aliases."""
        repl = REPLEnvironment(context="test")

        # Test directly via the method
        assert repl._resolve_role_alias("researcher_agent") == "worker_explore"
        assert repl._resolve_role_alias("coder_agent") == "coder_primary"
        assert repl._resolve_role_alias("reviewer_agent") == "architect_general"

    def test_resolve_unknown_returns_original(self):
        """Test unknown aliases return original value."""
        repl = REPLEnvironment(context="test")

        assert repl._resolve_role_alias("unknown_role") == "unknown_role"
        assert repl._resolve_role_alias("custom_agent") == "custom_agent"

    def test_resolve_passthrough_for_actual_roles(self):
        """Test actual role names pass through unchanged."""
        repl = REPLEnvironment(context="test")

        assert repl._resolve_role_alias("coder_primary") == "coder_primary"
        assert repl._resolve_role_alias("worker_explore") == "worker_explore"


class TestEscalate:
    """Test _escalate() / escalate() function."""

    def test_escalate_sets_artifact(self):
        """Test escalate() sets escalation request in artifacts."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("escalate('Task too complex')")

        assert result.error is None
        assert repl.artifacts.get("_escalation_requested") is True
        assert repl.artifacts.get("_escalation_reason") == "Task too complex"

    def test_escalate_with_target_role(self):
        """Test escalate() with specific target role."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("escalate('Need architecture review', 'architect_general')")

        assert result.error is None
        assert repl.artifacts.get("_escalation_target") == "architect_general"
        assert repl.artifacts.get("_escalation_requested") is True

    def test_escalate_resolves_role_alias(self):
        """Test escalate() resolves target role aliases."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("escalate('Need review', 'reviewer_agent')")

        assert result.error is None
        # reviewer_agent should resolve to architect_general
        assert repl.artifacts.get("_escalation_target") == "architect_general"

    def test_escalate_returns_message(self):
        """Test escalate() returns acknowledgment message."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(escalate('Too difficult'))")

        assert result.error is None
        assert "ESCALATION REQUESTED" in result.output
        assert "Too difficult" in result.output

    def test_escalate_logs_event(self):
        """Test escalate() is logged as exploration call."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("escalate('complex task')")

        # Escalate does NOT increment exploration_calls (it's a signal, not exploration)
        # But it should still set artifacts
        assert repl.artifacts.get("_escalation_requested") is True


class TestRecall:
    """Test _recall() / recall() function with episodic memory."""

    def test_recall_without_retriever(self):
        """Test recall() returns empty results when no retriever."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = recall('similar tasks')
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(len(data.get('results', [])))
print('error' in data)
""")

        assert result.error is None
        # Should return empty results or error when retriever not available
        assert "0" in result.output or "True" in result.output

    def test_recall_with_mock_retriever(self):
        """Test recall() with mocked retriever."""
        mock_retriever = Mock()
        mock_memory = Mock()
        mock_memory.context = {"objective": "test task", "role": "worker_general"}
        mock_memory.action = "test action"
        mock_memory.outcome = "success"

        mock_result = Mock()
        mock_result.memory = mock_memory
        mock_result.q_value = 0.85
        mock_result.similarity = 0.92
        mock_result.combined_score = 0.88

        mock_retriever.retrieve_for_exploration = Mock(return_value=[mock_result])
        mock_retriever.get_best_action = Mock(return_value=("test action", 0.88))

        repl = REPLEnvironment(context="test", retriever=mock_retriever)
        result = repl.execute("""
output = recall('similar task', limit=5)
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(len(data.get('results', [])))
print(data.get('best_action'))
""")

        assert result.error is None
        assert "1" in result.output  # 1 result
        assert "test action" in result.output

    def test_recall_increments_exploration(self):
        """Test recall() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("recall('task')")

        assert repl._exploration_calls > initial_calls


class TestRouteAdvice:
    """Test _route_advice() / route_advice() function."""

    def test_route_advice_without_router(self):
        """Test route_advice() returns error when no router."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = route_advice('implement feature X')
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(data.get('strategy'))
print('unavailable' in str(data.get('warnings', [])))
""")

        assert result.error is None
        assert "unavailable" in result.output.lower()

    def test_route_advice_with_mock_router(self):
        """Test route_advice() with mocked hybrid router."""
        mock_router = Mock()
        mock_retriever = Mock()

        # Mock retrieval results
        mock_memory = Mock()
        mock_memory.context = {"role": "coder_primary"}
        mock_memory.action = "coder_primary"
        mock_memory.outcome = "success"

        mock_result = Mock()
        mock_result.memory = mock_memory
        mock_result.q_value = 0.90
        mock_result.combined_score = 0.85
        mock_result.warnings = []

        mock_retriever.retrieve_for_routing = Mock(return_value=[mock_result])
        mock_router.retriever = mock_retriever
        mock_router.route = Mock(return_value=(("coder_primary", 0.85), "memrl"))

        repl = REPLEnvironment(context="test", hybrid_router=mock_router)
        result = repl.execute("""
output = route_advice('write a function')
if '<<<TOOL_OUTPUT>>>' in output:
    start = output.find('<<<TOOL_OUTPUT>>>') + len('<<<TOOL_OUTPUT>>>')
    end = output.find('<<<END_TOOL_OUTPUT>>>')
    output = output[start:end]
data = json.loads(output)
print(data.get('recommended_role'))
print(data.get('strategy'))
""")

        assert result.error is None
        assert "coder_primary" in result.output
        assert "memrl" in result.output

    def test_route_advice_increments_exploration(self):
        """Test route_advice() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("route_advice('task')")

        assert repl._exploration_calls > initial_calls


class TestDelegate:
    """Test _delegate() / delegate() function."""

    def test_delegate_without_llm_primitives(self):
        """Test delegate() returns error when no LLM primitives."""
        # Use a non-worker role to bypass tier guard
        repl = REPLEnvironment(context="test", role="frontdoor")
        result = repl.execute("print(delegate('subtask', 'worker_general'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "primitives" in result.output.lower()

    def test_delegate_with_mock_llm(self):
        """Test delegate() with mocked LLM primitives."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="Task completed successfully")

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="coder_primary")
        result = repl.execute("output = delegate('summarize this', 'worker_summarize', 'need summary')")

        assert result.error is None
        mock_llm.llm_call.assert_called_once()

    def test_delegate_resolves_role_alias(self):
        """Test delegate() resolves target role aliases."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="response")

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")
        result = repl.execute("delegate('task', 'researcher_agent')")

        assert result.error is None
        # Should call with resolved role (worker_explore)
        call_args = mock_llm.llm_call.call_args
        assert call_args[1]["role"] == "worker_explore"

    def test_delegate_tier_guard_workers_cannot_delegate(self):
        """Test workers (Tier C) cannot delegate."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="response")

        # Set role to a worker (Tier C)
        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="worker_general")
        result = repl.execute("print(delegate('task', 'worker_math'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "Tier C" in result.output or "cannot delegate" in result.output

    def test_delegate_tracks_delegation(self):
        """Test delegate() tracks delegation in artifacts."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="result")

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="coder_primary")
        result = repl.execute("delegate('task', 'worker_math', 'need help')")

        assert result.error is None
        assert "_delegations" in repl.artifacts
        delegations = repl.artifacts["_delegations"]
        assert len(delegations) == 1
        # worker_math doesn't have an alias, should stay as is
        assert delegations[0]["to_role"] == "worker_math"
        assert delegations[0]["reason"] == "need help"

    def test_delegate_increments_exploration(self):
        """Test delegate() increments exploration calls."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(return_value="result")

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")
        initial_calls = repl._exploration_calls
        repl.execute("delegate('task', 'worker_general')")

        assert repl._exploration_calls > initial_calls

    def test_delegate_handles_exception(self):
        """Test delegate() handles LLM call exceptions."""
        mock_llm = Mock()
        mock_llm.llm_call = Mock(side_effect=Exception("LLM error"))

        repl = REPLEnvironment(context="test", llm_primitives=mock_llm, role="frontdoor")
        result = repl.execute("print(delegate('task', 'worker_general'))")

        assert result.error is None
        assert "DELEGATION FAILED" in result.output
        assert "LLM error" in result.output
