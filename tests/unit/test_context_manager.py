"""Unit tests for context_manager module."""

from pathlib import Path

from src.context_manager import (
    ContextConfig,
    ContextEntry,
    ContextManager,
    ContextType,
)


class TestContextType:
    """Tests for ContextType enum."""

    def test_type_values(self):
        """Test context type enum values."""
        assert ContextType.TEXT.value == "text"
        assert ContextType.ARTIFACT.value == "artifact"
        assert ContextType.STRUCTURED.value == "structured"
        assert ContextType.SUMMARY.value == "summary"


class TestContextEntry:
    """Tests for ContextEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a context entry."""
        entry = ContextEntry(
            key="test_key",
            value="test value",
            context_type=ContextType.TEXT,
        )

        assert entry.key == "test_key"
        assert entry.value == "test value"
        assert entry.context_type == ContextType.TEXT
        assert entry.step_id is None
        assert entry.truncated is False

    def test_entry_with_step_id(self):
        """Test entry with step ID."""
        entry = ContextEntry(
            key="output",
            value="result",
            context_type=ContextType.TEXT,
            step_id="S1",
        )

        assert entry.step_id == "S1"

    def test_entry_to_dict(self):
        """Test converting entry to dictionary."""
        entry = ContextEntry(
            key="test",
            value="short value",
            context_type=ContextType.TEXT,
            step_id="S1",
            size_bytes=11,
        )

        d = entry.to_dict()
        assert d["key"] == "test"
        assert d["value"] == "short value"
        assert d["context_type"] == "text"
        assert d["step_id"] == "S1"

    def test_entry_to_dict_truncates_long_text(self):
        """Test that long text is truncated in serialization."""
        long_text = "x" * 2000
        entry = ContextEntry(
            key="test",
            value=long_text,
            context_type=ContextType.TEXT,
        )

        d = entry.to_dict()
        assert len(d["value"]) < len(long_text)
        assert "[truncated]" in d["value"]

    def test_entry_to_dict_artifact(self):
        """Test artifact serialization."""
        entry = ContextEntry(
            key="file",
            value=Path("/tmp/test.py"),
            context_type=ContextType.ARTIFACT,
        )

        d = entry.to_dict()
        assert d["value"] == "/tmp/test.py"


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContextConfig()

        assert config.max_entry_size == 10000
        assert config.max_total_size == 100000
        assert config.auto_summarize is True
        assert config.summary_threshold == 5000

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContextConfig(
            max_entry_size=5000,
            max_total_size=50000,
            auto_summarize=False,
        )

        assert config.max_entry_size == 5000
        assert config.max_total_size == 50000
        assert config.auto_summarize is False


class TestContextManager:
    """Tests for ContextManager class."""

    def test_manager_creation(self):
        """Test creating a context manager."""
        ctx = ContextManager()

        assert ctx.count() == 0
        assert ctx.size() == 0

    def test_manager_with_config(self):
        """Test creating manager with custom config."""
        config = ContextConfig(max_entry_size=100)
        ctx = ContextManager(config=config)

        assert ctx.config.max_entry_size == 100

    def test_set_and_get(self):
        """Test setting and getting values."""
        ctx = ContextManager()

        ctx.set("key1", "value1")
        result = ctx.get("key1")

        assert result == "value1"

    def test_get_missing_key(self):
        """Test getting a missing key."""
        ctx = ContextManager()

        result = ctx.get("missing")
        assert result is None

        result = ctx.get("missing", "default")
        assert result == "default"

    def test_set_with_step_id(self):
        """Test setting with step ID."""
        ctx = ContextManager()

        entry = ctx.set("output", "result", step_id="S1")

        assert entry.step_id == "S1"

    def test_set_auto_detect_text(self):
        """Test auto-detection of text type."""
        ctx = ContextManager()

        entry = ctx.set("text", "hello world")

        assert entry.context_type == ContextType.TEXT

    def test_set_auto_detect_structured(self):
        """Test auto-detection of structured type."""
        ctx = ContextManager()

        entry = ctx.set("data", {"key": "value"})

        assert entry.context_type == ContextType.STRUCTURED

    def test_set_auto_detect_artifact(self):
        """Test auto-detection of artifact type."""
        ctx = ContextManager()

        entry = ctx.set("file", Path("/tmp/test.py"))

        assert entry.context_type == ContextType.ARTIFACT

    def test_set_truncates_large_text(self):
        """Test that large text is truncated."""
        config = ContextConfig(max_entry_size=100)
        ctx = ContextManager(config=config)

        large_text = "x" * 500
        entry = ctx.set("big", large_text)

        assert entry.truncated is True
        assert len(entry.value) <= 100 + len("\n... [truncated]")

    def test_has(self):
        """Test has method."""
        ctx = ContextManager()

        assert ctx.has("missing") is False

        ctx.set("key", "value")
        assert ctx.has("key") is True

    def test_delete(self):
        """Test deleting entries."""
        ctx = ContextManager()

        ctx.set("key", "value")
        assert ctx.has("key") is True

        result = ctx.delete("key")
        assert result is True
        assert ctx.has("key") is False

    def test_delete_missing(self):
        """Test deleting missing key."""
        ctx = ContextManager()

        result = ctx.delete("missing")
        assert result is False

    def test_add_artifact(self):
        """Test adding file artifacts."""
        ctx = ContextManager()

        entry = ctx.add_artifact("code", "/tmp/output.py", step_id="S1")

        assert entry.context_type == ContextType.ARTIFACT
        assert entry.step_id == "S1"
        assert "path" in entry.metadata

    def test_get_entry(self):
        """Test getting full entry."""
        ctx = ContextManager()

        ctx.set("key", "value", step_id="S1")
        entry = ctx.get_entry("key")

        assert entry is not None
        assert entry.key == "key"
        assert entry.step_id == "S1"

    def test_get_entry_missing(self):
        """Test getting missing entry."""
        ctx = ContextManager()

        entry = ctx.get_entry("missing")
        assert entry is None

    def test_get_for_step(self):
        """Test getting entries for a specific step."""
        ctx = ContextManager()

        ctx.set("out1", "value1", step_id="S1")
        ctx.set("out2", "value2", step_id="S1")
        ctx.set("out3", "value3", step_id="S2")

        s1_entries = ctx.get_for_step("S1")

        assert len(s1_entries) == 2
        assert "out1" in s1_entries
        assert "out2" in s1_entries
        assert "out3" not in s1_entries

    def test_get_inputs(self):
        """Test getting multiple inputs."""
        ctx = ContextManager()

        ctx.set("a", "value_a")
        ctx.set("b", "value_b")
        ctx.set("c", "value_c")

        inputs = ctx.get_inputs(["a", "c", "missing"])

        assert len(inputs) == 2
        assert inputs["a"] == "value_a"
        assert inputs["c"] == "value_c"
        assert "missing" not in inputs

    def test_build_prompt_context(self):
        """Test building prompt context."""
        ctx = ContextManager()

        ctx.set("analysis", "The code looks good")
        ctx.set("data", {"score": 95})

        prompt = ctx.build_prompt_context(["analysis", "data"])

        assert "### analysis" in prompt
        assert "The code looks good" in prompt
        assert "### data" in prompt
        assert "score" in prompt

    def test_build_prompt_context_missing_key(self):
        """Test prompt context with missing keys."""
        ctx = ContextManager()

        ctx.set("exists", "value")

        prompt = ctx.build_prompt_context(["exists", "missing"])

        assert "### exists" in prompt
        assert "missing" not in prompt

    def test_build_prompt_context_truncation(self):
        """Test prompt context respects max_chars."""
        ctx = ContextManager()

        ctx.set("large", "x" * 10000)

        prompt = ctx.build_prompt_context(["large"], max_chars=500)

        assert len(prompt) <= 500 + 50  # Some buffer for headers

    def test_clear(self):
        """Test clearing all entries."""
        ctx = ContextManager()

        ctx.set("a", "1")
        ctx.set("b", "2")
        assert ctx.count() == 2

        ctx.clear()

        assert ctx.count() == 0
        assert ctx.size() == 0

    def test_keys(self):
        """Test getting keys in order."""
        ctx = ContextManager()

        ctx.set("first", "1")
        ctx.set("second", "2")
        ctx.set("third", "3")

        keys = ctx.keys()

        assert keys == ["first", "second", "third"]

    def test_values(self):
        """Test getting values in order."""
        ctx = ContextManager()

        ctx.set("a", "1")
        ctx.set("b", "2")

        values = ctx.values()

        assert values == ["1", "2"]

    def test_items(self):
        """Test getting items in order."""
        ctx = ContextManager()

        ctx.set("a", "1")
        ctx.set("b", "2")

        items = ctx.items()

        assert items == [("a", "1"), ("b", "2")]

    def test_entries(self):
        """Test getting entries in order."""
        ctx = ContextManager()

        ctx.set("a", "1")
        ctx.set("b", "2")

        entries = ctx.entries()

        assert len(entries) == 2
        assert entries[0].key == "a"
        assert entries[1].key == "b"

    def test_size_tracking(self):
        """Test size is tracked correctly."""
        ctx = ContextManager()

        ctx.set("key", "12345")  # 5 bytes
        assert ctx.size() == 5

        ctx.set("key2", "67890")  # Another 5 bytes
        assert ctx.size() == 10

    def test_count(self):
        """Test entry count."""
        ctx = ContextManager()

        assert ctx.count() == 0

        ctx.set("a", "1")
        assert ctx.count() == 1

        ctx.set("b", "2")
        assert ctx.count() == 2

    def test_to_dict(self):
        """Test exporting to dictionary."""
        ctx = ContextManager()

        ctx.set("key", "value", step_id="S1")

        d = ctx.to_dict()

        assert "entries" in d
        assert "total_size" in d
        assert "count" in d
        assert d["count"] == 1

    def test_from_dict(self):
        """Test importing from dictionary."""
        ctx = ContextManager()

        data = {
            "entries": [
                {
                    "key": "imported",
                    "value": "test value",
                    "context_type": "text",
                    "step_id": "S1",
                }
            ]
        }

        ctx.from_dict(data)

        assert ctx.has("imported")
        assert ctx.get("imported") == "test value"

    def test_overwrite_updates_size(self):
        """Test that overwriting an entry updates size correctly."""
        ctx = ContextManager()

        ctx.set("key", "short")
        initial_size = ctx.size()

        ctx.set("key", "much longer value")
        assert ctx.size() != initial_size
        assert ctx.count() == 1

    def test_total_size_limit_enforcement(self):
        """Test that total size limit is enforced."""
        config = ContextConfig(max_total_size=100)
        ctx = ContextManager(config=config)

        # Add entries that exceed the limit
        for i in range(20):
            ctx.set(f"key{i}", "x" * 10)

        # Oldest entries should be removed
        assert ctx.size() <= 100
        assert ctx.count() < 20


class TestContextManagerEdgeCases:
    """Edge case tests for ContextManager."""

    def test_empty_value(self):
        """Test setting empty value."""
        ctx = ContextManager()

        ctx.set("empty", "")

        assert ctx.get("empty") == ""
        assert ctx.size() == 0

    def test_none_value(self):
        """Test setting None value."""
        ctx = ContextManager()

        ctx.set("none", None)

        assert ctx.get("none") is None

    def test_list_value(self):
        """Test setting list value."""
        ctx = ContextManager()

        entry = ctx.set("list", [1, 2, 3])

        assert entry.context_type == ContextType.STRUCTURED
        assert ctx.get("list") == [1, 2, 3]

    def test_nested_dict(self):
        """Test setting nested dictionary."""
        ctx = ContextManager()

        data = {"level1": {"level2": {"level3": "value"}}}
        ctx.set("nested", data)

        assert ctx.get("nested") == data

    def test_metadata_preserved(self):
        """Test that metadata is preserved."""
        ctx = ContextManager()

        ctx.set("key", "value", metadata={"custom": "data"})
        entry = ctx.get_entry("key")

        assert entry.metadata["custom"] == "data"
