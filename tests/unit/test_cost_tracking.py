#!/usr/bin/env python3
"""Unit tests for src/llm_primitives/cost_tracking.py.

The CostTrackingMixin is tested through LLMPrimitives in mock mode.
"""

import pytest

from src.llm_primitives import LLMPrimitives


class TestCostTracking:
    """Test cost tracking through LLMPrimitives."""

    @pytest.fixture
    def llm(self):
        """Create a mock-mode LLMPrimitives instance."""
        return LLMPrimitives(mock_mode=True)

    def test_query_lifecycle(self, llm):
        """Test start_query, track usage, end_query lifecycle."""
        # Start a query
        llm.start_query("test-query-1")

        # Make a call (mock mode returns fixed response)
        llm.llm_call("test prompt", "test context")

        # End the query
        query_cost = llm.end_query()

        # Verify query cost is returned
        assert query_cost is not None
        assert query_cost.query_id == "test-query-1"
        assert query_cost.prompt_tokens >= 0
        assert query_cost.completion_tokens >= 0

    def test_get_completed_queries(self, llm):
        """Test get_completed_queries returns all completed queries."""
        # Complete two queries
        llm.start_query("query-1")
        llm.llm_call("prompt 1", "")
        llm.end_query()

        llm.start_query("query-2")
        llm.llm_call("prompt 2", "")
        llm.end_query()

        # Get all completed queries
        completed = llm.get_completed_queries()
        assert len(completed) == 2
        assert completed[0].query_id == "query-1"
        assert completed[1].query_id == "query-2"

    def test_get_completed_queries_last_n(self, llm):
        """Test get_completed_queries with last_n parameter."""
        # Complete three queries
        for i in range(3):
            llm.start_query(f"query-{i}")
            llm.llm_call(f"prompt {i}", "")
            llm.end_query()

        # Get last 2 queries
        last_two = llm.get_completed_queries(last_n=2)
        assert len(last_two) == 2
        assert last_two[0].query_id == "query-1"
        assert last_two[1].query_id == "query-2"

    def test_get_total_cost(self, llm):
        """Test get_total_cost sums all completed queries."""
        # Complete two queries
        llm.start_query("query-1")
        llm.llm_call("prompt 1", "")
        cost1 = llm.end_query()

        llm.start_query("query-2")
        llm.llm_call("prompt 2", "")
        cost2 = llm.end_query()

        # Total cost should be sum of individual costs
        total = llm.get_total_cost()
        expected = cost1.estimated_cost + cost2.estimated_cost
        assert total == pytest.approx(expected)

    def test_start_query_ends_previous(self, llm):
        """Test start_query automatically ends previous query."""
        llm.start_query("query-1")
        llm.llm_call("prompt 1", "")

        # Start a new query without explicitly ending the first
        llm.start_query("query-2")

        # First query should be in completed queries
        completed = llm.get_completed_queries()
        assert len(completed) == 1
        assert completed[0].query_id == "query-1"

    def test_end_query_without_start(self, llm):
        """Test end_query returns None if no query active."""
        result = llm.end_query()
        assert result is None
