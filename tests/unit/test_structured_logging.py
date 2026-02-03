"""Tests for structured logging module."""

from __future__ import annotations

import json
import logging

from src.api.structured_logging import JSONFormatter, configure_json_logging, task_extra


class TestTaskExtra:
    """Tests for task_extra helper."""

    def test_empty_returns_empty_dict(self):
        assert task_extra() == {}

    def test_includes_only_non_none(self):
        result = task_extra(task_id="abc", role="frontdoor")
        assert result == {"task_id": "abc", "role": "frontdoor"}
        assert "latency_ms" not in result

    def test_all_fields(self):
        result = task_extra(
            task_id="t1",
            role="coder",
            latency_ms=42.567,
            stage="execute",
            strategy="rules",
            mode="direct",
            turn=3,
            error_type="TimeoutError",
            tokens=150,
            prompt_len=500,
        )
        assert result["task_id"] == "t1"
        assert result["latency_ms"] == 42.6  # rounded to 1 decimal
        assert result["turn"] == 3
        assert len(result) == 10

    def test_latency_rounding(self):
        result = task_extra(latency_ms=42.3456789)
        assert result["latency_ms"] == 42.3


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_basic_format(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["msg"] == "hello world"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert "ts" in parsed

    def test_structured_fields_included(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="routed",
            args=(),
            exc_info=None,
        )
        record.task_id = "chat-abc"
        record.role = "frontdoor"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["task_id"] == "chat-abc"
        assert parsed["role"] == "frontdoor"

    def test_unrecognized_fields_excluded(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.random_field = "should not appear"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "random_field" not in parsed

    def test_format_with_args(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="user %s did %d things",
            args=("alice", 3),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["msg"] == "user alice did 3 things"


class TestConfigureJSONLogging:
    """Tests for configure_json_logging."""

    def test_installs_json_handler(self):
        # Save original state
        root = logging.getLogger()
        orig_handlers = root.handlers[:]
        orig_level = root.level

        try:
            configure_json_logging(level=logging.DEBUG)
            assert len(root.handlers) == 1
            assert isinstance(root.handlers[0].formatter, JSONFormatter)
            assert root.level == logging.DEBUG
        finally:
            # Restore
            root.handlers = orig_handlers
            root.level = orig_level
