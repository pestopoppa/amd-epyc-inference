#!/usr/bin/env python3
"""Unit tests for src/repl_environment/state.py."""

import pytest

from src.repl_environment import REPLEnvironment
from src.repl_environment.types import REPLConfig


class TestREPLStateCheckpoint:
    """Test checkpoint/restore functionality."""

    @pytest.fixture
    def repl(self):
        """Create a REPL environment for testing."""
        config = REPLConfig()
        return REPLEnvironment("test context", config=config, task_id="test-task")

    def test_checkpoint_basic(self, repl):
        """Test checkpoint captures basic state."""
        # Add some artifacts
        repl.artifacts["key1"] = "value1"
        repl.artifacts["key2"] = [1, 2, 3]

        checkpoint = repl.checkpoint()

        assert checkpoint["version"] == 1
        assert checkpoint["artifacts"]["key1"] == "value1"
        assert checkpoint["artifacts"]["key2"] == [1, 2, 3]
        assert checkpoint["task_id"] == "test-task"

    def test_checkpoint_execution_counts(self, repl):
        """Test checkpoint captures execution state."""
        # Execute some code
        repl.execute("x = 1 + 1")

        checkpoint = repl.checkpoint()

        assert checkpoint["execution_count"] == 1
        assert checkpoint["exploration_calls"] >= 0

    def test_checkpoint_json_serializable(self, repl):
        """Test checkpoint produces JSON-serializable dict."""
        import json

        repl.artifacts["data"] = {"nested": "value"}
        checkpoint = repl.checkpoint()

        # Should not raise
        json.dumps(checkpoint)

    def test_checkpoint_non_serializable_artifacts(self, repl):
        """Test checkpoint handles non-serializable artifacts."""

        # Add a non-serializable object
        class CustomClass:
            pass

        repl.artifacts["obj"] = CustomClass()

        checkpoint = repl.checkpoint()

        # Non-serializable artifact should be marked
        assert "__unserializable__" in checkpoint["artifacts"]["obj"]
        assert checkpoint["artifacts"]["obj"]["type"] == "CustomClass"

    def test_restore_from_checkpoint(self, repl):
        """Test restore() rebuilds state from checkpoint."""
        # Set up initial state
        repl.artifacts["key1"] = "value1"
        repl.execute("x = 42")

        checkpoint = repl.checkpoint()

        # Create new REPL and restore
        config = REPLConfig()
        new_repl = REPLEnvironment("test context", config=config, task_id="test-task")
        new_repl.restore(checkpoint)

        # Artifacts should be restored
        assert new_repl.artifacts["key1"] == "value1"
        assert new_repl._execution_count == 1

    def test_restore_invalid_version(self, repl):
        """Test restore() raises error for invalid version."""
        bad_checkpoint = {"version": 99}

        with pytest.raises(ValueError, match="Unsupported checkpoint version"):
            repl.restore(bad_checkpoint)

    def test_get_checkpoint_metadata(self, repl):
        """Test get_checkpoint_metadata returns state summary."""
        repl.artifacts["a"] = 1
        repl.artifacts["b"] = 2
        repl.execute("x = 1")

        metadata = repl.get_checkpoint_metadata()

        assert metadata["execution_count"] == 1
        assert metadata["artifact_count"] == 2
        assert metadata["context_length"] == len("test context")
        assert "exploration_calls" in metadata

    def test_reset_clears_state(self, repl):
        """Test reset() clears artifacts and state."""
        repl.artifacts["key"] = "value"
        repl.execute("x = 1")

        repl.reset()

        assert len(repl.artifacts) == 0
        assert repl._execution_count == 0
        assert repl._exploration_calls == 0
