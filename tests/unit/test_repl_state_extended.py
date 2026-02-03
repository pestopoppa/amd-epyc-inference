"""Extended tests for REPL state management.

Tests coverage for src/repl_environment/state.py (58% coverage).
Focus on uncovered checkpoint/restore logic and edge cases.
"""

import time
from unittest.mock import Mock

import pytest

from src.repl_environment.state import _StateMixin
from src.repl_environment.types import ExplorationEvent, ExplorationLog


class MockREPLEnvironment(_StateMixin):
    """Mock REPL environment for testing state mixin."""

    def __init__(self):
        self.config = Mock()
        self.context = "Test context with some data"
        self.artifacts = {}
        self._exploration_calls = 0
        self._exploration_log = ExplorationLog()
        self._execution_count = 0
        self._final_answer = None
        self._grep_hits_buffer = []
        self._findings_buffer = []
        self._globals = {}
        self.progress_logger = None
        self.task_id = "test_task_123"

    def _build_globals(self):
        """Build globals dict."""
        return {"artifacts": self.artifacts}


class TestGetState:
    """Test get_state() method."""

    def test_get_state_empty_artifacts(self):
        """Test state summary with no artifacts."""
        env = MockREPLEnvironment()
        state = env.get_state()

        assert "context: str" in state
        assert "artifacts: {}" in state

    def test_get_state_with_artifacts(self):
        """Test state summary with artifacts."""
        env = MockREPLEnvironment()
        env.artifacts["result"] = "Test result value"
        env.artifacts["count"] = 42

        state = env.get_state()

        assert "artifacts: ['result', 'count']" in state
        assert "Test result value" in state


class TestExplorationLog:
    """Test exploration logging methods."""

    def test_get_exploration_log(self):
        """Test retrieving exploration log."""
        env = MockREPLEnvironment()
        log = env.get_exploration_log()

        assert isinstance(log, ExplorationLog)
        assert log.events == []

    def test_get_grep_history(self):
        """Test retrieving grep history."""
        env = MockREPLEnvironment()
        env._grep_hits_buffer = [{"match": "test1"}, {"match": "test2"}]

        history = env.get_grep_history()

        assert len(history) == 2
        assert history[0]["match"] == "test1"

    def test_clear_grep_history(self):
        """Test clearing grep history."""
        env = MockREPLEnvironment()
        env._grep_hits_buffer = [{"match": "test"}]

        env.clear_grep_history()

        assert env._grep_hits_buffer == []

    def test_get_exploration_strategy(self):
        """Test getting exploration strategy summary."""
        env = MockREPLEnvironment()
        env._exploration_log.events.append(
            ExplorationEvent(
                function="grep",
                args={"pattern": "test"},
                result_size=100,
                timestamp=time.time(),
                token_estimate=25,
            )
        )

        strategy = env.get_exploration_strategy()

        assert isinstance(strategy, dict)
        assert "function_counts" in strategy or "total_tokens" in strategy


class TestLogExplorationCompleted:
    """Test log_exploration_completed() method."""

    def test_log_exploration_completed_without_logger(self):
        """Test completion logging without progress logger."""
        env = MockREPLEnvironment()
        env.progress_logger = None

        data = env.log_exploration_completed(success=True, result="Test result")

        assert data["success"] is True
        assert "strategy" in data
        assert "efficiency" in data

    def test_log_exploration_completed_with_logger(self):
        """Test completion logging with progress logger."""
        env = MockREPLEnvironment()
        env.progress_logger = Mock()

        data = env.log_exploration_completed(success=True, result="Test result")

        env.progress_logger.log_exploration.assert_called_once()
        assert data["success"] is True


class TestSuggestExploration:
    """Test suggest_exploration() method."""

    def test_suggest_exploration_without_retriever(self):
        """Test suggestions without retriever."""
        env = MockREPLEnvironment()
        env.context = "Short context"

        suggestions = env.suggest_exploration("Find data", retriever=None)

        assert len(suggestions) > 0
        assert any("peek" in s for s in suggestions)

    def test_suggest_exploration_long_context(self):
        """Test suggestions for long context."""
        env = MockREPLEnvironment()
        env.context = "x" * 5000  # Long context

        suggestions = env.suggest_exploration("Task", retriever=None)

        # Should suggest grep for long contexts
        assert any("grep" in s for s in suggestions)

    def test_suggest_exploration_with_retriever(self):
        """Test suggestions with retriever providing episodic memories."""
        env = MockREPLEnvironment()
        retriever = Mock()

        # Mock retrieval result
        mock_memory = Mock()
        mock_memory.outcome = "success"
        mock_memory.context = {
            "exploration_strategy": {
                "strategy_type": "scan",
                "function_counts": {"peek": 3, "grep": 2},
            }
        }

        mock_result = Mock()
        mock_result.q_value = 0.8
        mock_result.memory = mock_memory

        retriever.retrieve_for_exploration.return_value = [mock_result]

        suggestions = env.suggest_exploration("Task description", retriever=retriever)

        # Should include episodic suggestions from similar tasks
        assert len(suggestions) > 0


class TestCheckpoint:
    """Test checkpoint() method."""

    def test_checkpoint_basic(self):
        """Test basic checkpoint creation."""
        env = MockREPLEnvironment()
        env.artifacts["key"] = "value"
        env._execution_count = 5

        checkpoint = env.checkpoint()

        assert checkpoint["version"] == 1
        assert checkpoint["artifacts"]["key"] == "value"
        assert checkpoint["execution_count"] == 5
        assert checkpoint["task_id"] == "test_task_123"

    def test_checkpoint_with_exploration_events(self):
        """Test checkpoint includes exploration events."""
        env = MockREPLEnvironment()
        env._exploration_log.events.append(
            ExplorationEvent(
                function="grep",
                args={"pattern": "test"},
                result_size=100,
                timestamp=time.time(),
                token_estimate=25,
            )
        )

        checkpoint = env.checkpoint()

        assert len(checkpoint["exploration_events"]) == 1
        assert checkpoint["exploration_events"][0]["function"] == "grep"

    def test_checkpoint_sanitizes_unserializable(self):
        """Test checkpoint sanitizes non-JSON-serializable values."""
        env = MockREPLEnvironment()

        # Add non-serializable object
        class CustomObject:
            pass

        env.artifacts["obj"] = CustomObject()
        env.artifacts["good_val"] = "serializable"

        checkpoint = env.checkpoint()

        # Good value should be preserved
        assert checkpoint["artifacts"]["good_val"] == "serializable"

        # Bad value should be marked
        assert checkpoint["artifacts"]["obj"]["__unserializable__"] is True
        assert "CustomObject" in checkpoint["artifacts"]["obj"]["type"]

    def test_checkpoint_with_nested_artifacts(self):
        """Test checkpoint handles nested artifact structures."""
        env = MockREPLEnvironment()
        env.artifacts["nested"] = {"level1": {"level2": "value", "list": [1, 2, 3]}}

        checkpoint = env.checkpoint()

        # Should preserve nested structure
        assert checkpoint["artifacts"]["nested"]["level1"]["level2"] == "value"
        assert checkpoint["artifacts"]["nested"]["level1"]["list"] == [1, 2, 3]


class TestRestore:
    """Test restore() method."""

    def test_restore_basic(self):
        """Test basic restore from checkpoint."""
        env = MockREPLEnvironment()

        checkpoint = {
            "version": 1,
            "artifacts": {"key": "value"},
            "execution_count": 5,
            "exploration_calls": 3,
            "exploration_tokens": 100,
            "exploration_events": [],
            "grep_hits_buffer": [],
            "findings_buffer": [],
            "context_length": 100,
            "task_id": "restored_task",
        }

        env.restore(checkpoint)

        assert env.artifacts["key"] == "value"
        assert env._execution_count == 5
        assert env._exploration_calls == 3

    def test_restore_invalid_version(self):
        """Test restore with unsupported version."""
        env = MockREPLEnvironment()

        checkpoint = {
            "version": 99,  # Unsupported version
        }

        with pytest.raises(ValueError, match="Unsupported checkpoint version"):
            env.restore(checkpoint)

    def test_restore_with_exploration_events(self):
        """Test restore rebuilds exploration log."""
        env = MockREPLEnvironment()

        checkpoint = {
            "version": 1,
            "artifacts": {},
            "execution_count": 0,
            "exploration_calls": 0,
            "exploration_tokens": 50,
            "exploration_events": [
                {
                    "function": "grep",
                    "args": {"pattern": "test"},
                    "result_size": 100,
                    "timestamp": time.time(),
                    "token_estimate": 25,
                }
            ],
            "grep_hits_buffer": [],
            "findings_buffer": [],
        }

        env.restore(checkpoint)

        assert len(env._exploration_log.events) == 1
        assert env._exploration_log.events[0].function == "grep"
        assert env._exploration_log.total_exploration_tokens == 50

    def test_restore_rebuilds_globals(self):
        """Test restore rebuilds globals dict."""
        env = MockREPLEnvironment()

        checkpoint = {
            "version": 1,
            "artifacts": {"data": "test_data"},
            "execution_count": 0,
            "exploration_calls": 0,
            "exploration_tokens": 0,
            "exploration_events": [],
            "grep_hits_buffer": [],
            "findings_buffer": [],
        }

        env.restore(checkpoint)

        # Globals should be rebuilt with restored artifacts
        assert env._globals["artifacts"]["data"] == "test_data"


class TestCheckpointMetadata:
    """Test get_checkpoint_metadata() method."""

    def test_get_checkpoint_metadata(self):
        """Test retrieving checkpoint metadata."""
        env = MockREPLEnvironment()
        env._execution_count = 10
        env._exploration_calls = 5
        env.artifacts = {"a": 1, "b": 2}
        env._grep_hits_buffer = [{"match": "x"}]
        env._findings_buffer = [{"finding": "y"}]

        metadata = env.get_checkpoint_metadata()

        assert metadata["execution_count"] == 10
        assert metadata["exploration_calls"] == 5
        assert metadata["artifact_count"] == 2
        assert metadata["context_length"] == len(env.context)
        assert metadata["grep_hits_count"] == 1
        assert metadata["findings_count"] == 1


class TestReset:
    """Test reset() method."""

    def test_reset_clears_state(self):
        """Test reset clears all state except context."""
        env = MockREPLEnvironment()
        env.artifacts = {"key": "value"}
        env._final_answer = "Some answer"
        env._execution_count = 10
        env._exploration_calls = 5
        env._grep_hits_buffer = [{"match": "x"}]
        env._findings_buffer = [{"finding": "y"}]

        original_context = env.context

        env.reset()

        assert env.artifacts == {}
        assert env._final_answer is None
        assert env._execution_count == 0
        assert env._exploration_calls == 0
        assert env._grep_hits_buffer == []
        assert env._findings_buffer == []
        # Context should be preserved
        assert env.context == original_context

    def test_reset_creates_new_exploration_log(self):
        """Test reset creates fresh exploration log."""
        env = MockREPLEnvironment()
        env._exploration_log.events.append(ExplorationEvent("test", {}, 0, time.time(), 0))

        env.reset()

        assert len(env._exploration_log.events) == 0
        assert env._exploration_log.total_exploration_tokens == 0
