#!/usr/bin/env python3
"""Unit tests for REPL context management tools (_ContextMixin)."""

import pytest
from unittest.mock import MagicMock

from src.repl_environment import REPLConfig, REPLEnvironment, ExecutionResult, FinalSignal


class TestBasicContextTools:
    """Test basic context tools."""

    def test_context_len_returns_correct_length(self):
        """Test CONTEXT_LEN() returns correct length."""
        context = "A" * 12345
        repl = REPLEnvironment(context=context)
        result = repl.execute('artifacts["length"] = context_len()')

        assert result.error is None
        assert repl.artifacts["length"] == 12345

    def test_final_sets_is_final_true(self):
        """Test FINAL() sets is_final=True with answer."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('FINAL("This is my answer")')

        assert result.is_final is True
        assert result.final_answer == "This is my answer"

    def test_final_var_returns_artifact_value(self):
        """Test FINAL_VAR() returns artifact value."""
        repl = REPLEnvironment(context="test")
        repl.execute('artifacts["my_result"] = "Result value"')
        result = repl.execute('FINAL_VAR("my_result")')

        assert result.is_final is True
        assert result.final_answer == "Result value"

    def test_mark_finding_adds_to_findings_buffer(self):
        """Test MARK_FINDING() adds to findings buffer."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('artifacts["f"] = mark_finding("Important discovery", tags=["critical"])')

        assert result.error is None
        findings = repl.get_findings()
        assert len(findings) == 1
        assert findings[0]["content"] == "Important discovery"
        assert "critical" in findings[0]["tags"]

    def test_list_findings_returns_marked_findings(self):
        """Test LIST_FINDINGS() returns marked findings."""
        repl = REPLEnvironment(context="test")
        repl.execute('mark_finding("Finding 1")')
        repl.execute('mark_finding("Finding 2")')
        result = repl.execute('artifacts["findings"] = list_findings()')

        assert result.error is None
        findings_list = repl.artifacts["findings"]
        assert len(findings_list) == 2
        assert "Finding 1" in findings_list[0]["content"]
        assert "Finding 2" in findings_list[1]["content"]

    def test_chunk_context_splits_correctly(self):
        """Test CHUNK_CONTEXT() splits context correctly."""
        context = "A" * 10000
        repl = REPLEnvironment(context=context)
        result = repl.execute('artifacts["chunks"] = chunk_context(n_chunks=4)')

        assert result.error is None
        chunks = repl.artifacts["chunks"]
        assert len(chunks) == 4
        assert all("index" in c for c in chunks)
        assert all("text" in c for c in chunks)
        assert all("char_count" in c for c in chunks)

    def test_list_tools_returns_available_tools(self):
        """Test LIST_TOOLS() returns available tools for role."""
        # Create mock tool registry
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"},
        ]

        repl = REPLEnvironment(context="test", tool_registry=mock_registry, role="worker_general")
        result = repl.execute('artifacts["tools"] = list_tools()')

        assert result.error is None
        tools = repl.artifacts["tools"]
        assert len(tools) == 2
        assert tools[0]["name"] == "tool1"

    def test_context_len_empty_context(self):
        """Test context_len with empty context."""
        repl = REPLEnvironment(context="")
        result = repl.execute('artifacts["length"] = context_len()')

        assert result.error is None
        assert repl.artifacts["length"] == 0

    def test_chunk_context_caps_at_20_chunks(self):
        """Test chunk_context caps n_chunks at 20."""
        context = "X" * 100000
        repl = REPLEnvironment(context=context)
        result = repl.execute('artifacts["chunks"] = chunk_context(n_chunks=50)')

        assert result.error is None
        chunks = repl.artifacts["chunks"]
        # Should be capped at 20
        assert len(chunks) <= 20

    def test_mark_finding_with_source_metadata(self):
        """Test mark_finding with source metadata."""
        repl = REPLEnvironment(context="test")
        result = repl.execute('''
artifacts["f"] = mark_finding(
    "Finding text",
    tags=["tag1", "tag2"],
    source_file="/path/to/file.txt",
    source_page=42,
    source_section="Section 3"
)
''')

        assert result.error is None
        findings = repl.get_findings()
        assert len(findings) == 1
        assert findings[0]["source"]["file"] == "/path/to/file.txt"
        assert findings[0]["source"]["page"] == 42
        assert findings[0]["source"]["section"] == "Section 3"

    def test_clear_findings_returns_count(self):
        """Test clear_findings() returns count of cleared findings."""
        repl = REPLEnvironment(context="test")
        repl.execute('mark_finding("Finding 1")')
        repl.execute('mark_finding("Finding 2")')
        repl.execute('mark_finding("Finding 3")')

        assert len(repl.get_findings()) == 3
        count = repl.clear_findings()
        assert count == 3
        assert len(repl.get_findings()) == 0

    def test_final_validates_exploration_requirement(self):
        """Test FINAL() validates exploration requirement when enabled."""
        config = REPLConfig(require_exploration_before_final=True, min_exploration_calls=2)
        repl = REPLEnvironment(context="test context", config=config)

        # Should fail without exploration
        result = repl.execute('FINAL("answer")')
        assert result.error is not None
        assert "Premature FINAL" in result.error

        # After exploration, should succeed
        repl.execute('peek(10)')
        repl.execute('grep("test")')
        result = repl.execute('FINAL("answer")')
        assert result.is_final is True

    def test_chunk_context_with_overlap(self):
        """Test chunk_context creates overlap between chunks."""
        context = "0123456789" * 100  # 1000 chars
        repl = REPLEnvironment(context=context)
        result = repl.execute('artifacts["chunks"] = chunk_context(n_chunks=4, overlap=50)')

        assert result.error is None
        chunks = repl.artifacts["chunks"]
        assert len(chunks) == 4

        # Verify overlap exists (each chunk except first should start before previous ends)
        for i in range(1, len(chunks)):
            assert chunks[i]["start"] < chunks[i-1]["end"]

    def test_list_tools_without_registry(self):
        """Test list_tools is not defined when no registry."""
        repl = REPLEnvironment(context="test", tool_registry=None)
        result = repl.execute('artifacts["tools"] = list_tools()')

        # Should get NameError because list_tools is not in globals when tool_registry is None
        assert result.error is not None
        assert "NameError" in result.error
