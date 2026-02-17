"""Tests for the graph SQLiteStatePersistence adapter."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from pydantic_graph import End

from src.graph.persistence import SQLiteStatePersistence, _state_to_dict
from src.graph.nodes import FrontdoorNode, CoderNode
from src.graph.state import TaskState, TaskResult
from src.roles import Role


class TestStateToDict:
    """Test TaskState serialization."""

    def test_basic_serialization(self):
        state = TaskState(
            task_id="test-001",
            prompt="hello world",
            current_role=Role.CODER_ESCALATION,
            consecutive_failures=1,
            escalation_count=2,
            role_history=["frontdoor", "coder_escalation"],
            turns=3,
            last_error="some error",
        )
        d = _state_to_dict(state)
        assert d["task_id"] == "test-001"
        assert d["current_role"] == "coder_escalation"
        assert d["consecutive_failures"] == 1
        assert d["escalation_count"] == 2
        assert d["role_history"] == ["frontdoor", "coder_escalation"]
        assert d["turns"] == 3
        assert d["last_error"] == "some error"

    def test_prompt_truncation(self):
        state = TaskState(prompt="x" * 1000)
        d = _state_to_dict(state)
        assert len(d["prompt"]) == 500

    def test_error_truncation(self):
        state = TaskState(last_error="e" * 500)
        d = _state_to_dict(state)
        assert len(d["last_error"]) == 200


class TestSQLiteStatePersistence:
    """Test persistence adapter with mock session store."""

    def test_init(self):
        store = MagicMock()
        persistence = SQLiteStatePersistence(store, "session-001")
        assert persistence._session_id == "session-001"

    @pytest.mark.asyncio
    async def test_snapshot_node(self):
        store = MagicMock()
        persistence = SQLiteStatePersistence(store, "session-001")

        state = TaskState(task_id="test", current_role=Role.FRONTDOOR)
        node = FrontdoorNode()

        await persistence.snapshot_node(state, node)
        store.save_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_snapshot_end(self):
        store = MagicMock()
        persistence = SQLiteStatePersistence(store, "session-001")

        state = TaskState(task_id="test")
        end = End(TaskResult(answer="done", success=True))

        await persistence.snapshot_end(state, end)
        store.save_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_next_returns_none(self):
        """load_next returns None (resume not yet implemented)."""
        store = MagicMock()
        persistence = SQLiteStatePersistence(store, "session-001")
        result = await persistence.load_next()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_all_returns_snapshots(self):
        store = MagicMock()
        persistence = SQLiteStatePersistence(store, "session-001")

        state = TaskState(task_id="test")
        node = CoderNode()

        await persistence.snapshot_node(state, node)
        all_snaps = await persistence.load_all()
        assert len(all_snaps) == 1
        assert all_snaps[0]["type"] == "node"
        assert all_snaps[0]["node_class"] == "CoderNode"

    def test_write_with_no_store(self):
        """_write does nothing when store is None."""
        persistence = SQLiteStatePersistence(None, "session-001")
        persistence._write({"test": True})  # Should not raise

    @pytest.mark.asyncio
    async def test_record_run_context_manager(self):
        """record_run should work as an async context manager."""
        store = MagicMock()
        persistence = SQLiteStatePersistence(store, "session-001")
        async with persistence.record_run("run-001"):
            pass  # Should not raise
