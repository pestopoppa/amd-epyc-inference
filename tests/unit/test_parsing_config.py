#!/usr/bin/env python3
"""Unit tests for src/parsing_config.py."""

import re

import pytest
from pydantic import ValidationError

from src.parsing_config import (
    ArchitectureIR,
    FormalizationIR,
    ParsingMode,
    TaskIR,
    ToolCall,
    TOOLRUNNER_PATTERNS,
    get_parsing_mode,
    parse_toolrunner_output,
)


class TestParsingMode:
    """Test ParsingMode enum."""

    def test_enum_values(self):
        """Test that ParsingMode has expected values."""
        assert ParsingMode.INSTRUCTOR == "instructor"
        assert ParsingMode.GBNF == "gbnf"
        assert ParsingMode.REGEX == "regex"
        assert ParsingMode.NONE == "none"

    def test_enum_membership(self):
        """Test ParsingMode enum membership."""
        assert "instructor" in [m.value for m in ParsingMode]
        assert "gbnf" in [m.value for m in ParsingMode]
        assert "regex" in [m.value for m in ParsingMode]
        assert "none" in [m.value for m in ParsingMode]


class TestGetParsingMode:
    """Test get_parsing_mode() function."""

    def test_frontdoor_role(self):
        """Test parsing mode for frontdoor role."""
        assert get_parsing_mode("frontdoor") == ParsingMode.INSTRUCTOR

    def test_coder_roles(self):
        """Test parsing mode for coder roles."""
        assert get_parsing_mode("coder_escalation") == ParsingMode.INSTRUCTOR

    def test_architect_roles(self):
        """Test parsing mode for architect roles."""
        assert get_parsing_mode("architect_general") == ParsingMode.INSTRUCTOR
        assert get_parsing_mode("architect_coding") == ParsingMode.INSTRUCTOR

    def test_formalizer_role(self):
        """Test parsing mode for formalizer role."""
        assert get_parsing_mode("formalizer") == ParsingMode.GBNF

    def test_ingest_role(self):
        """Test parsing mode for ingest role."""
        assert get_parsing_mode("ingest_long_context") == ParsingMode.NONE

    def test_worker_roles(self):
        """Test parsing mode for worker roles."""
        assert get_parsing_mode("worker_general") == ParsingMode.NONE
        assert get_parsing_mode("worker_math") == ParsingMode.NONE
        assert get_parsing_mode("worker_vision") == ParsingMode.NONE
        assert get_parsing_mode("worker_summarize") == ParsingMode.NONE

    def test_toolrunner_role(self):
        """Test parsing mode for toolrunner role."""
        assert get_parsing_mode("toolrunner") == ParsingMode.REGEX

    def test_draft_roles(self):
        """Test parsing mode for draft roles."""
        assert get_parsing_mode("draft_coder") == ParsingMode.NONE
        assert get_parsing_mode("draft_general") == ParsingMode.NONE

    def test_wildcard_draft_pattern(self):
        """Test wildcard pattern for draft_ roles."""
        assert get_parsing_mode("draft_anything") == ParsingMode.NONE
        assert get_parsing_mode("draft_custom_model") == ParsingMode.NONE

    def test_wildcard_worker_pattern(self):
        """Test wildcard pattern for worker_ roles."""
        assert get_parsing_mode("worker_custom") == ParsingMode.NONE
        assert get_parsing_mode("worker_exploration") == ParsingMode.NONE

    def test_unknown_role(self):
        """Test parsing mode for unknown role defaults to NONE."""
        assert get_parsing_mode("unknown_role") == ParsingMode.NONE
        assert get_parsing_mode("") == ParsingMode.NONE


class TestToolRunnerPatterns:
    """Test TOOLRUNNER_PATTERNS regex patterns."""

    def test_tool_status_pattern_success(self):
        """Test tool_status pattern matches SUCCESS."""
        pattern = TOOLRUNNER_PATTERNS["tool_status"]
        match = re.search(pattern, "SUCCESS: bash_linter")
        assert match is not None
        assert match.group(1) == "SUCCESS"
        assert match.group(2) == "bash_linter"

    def test_tool_status_pattern_failure(self):
        """Test tool_status pattern matches FAILURE."""
        pattern = TOOLRUNNER_PATTERNS["tool_status"]
        match = re.search(pattern, "FAILURE: pytest")
        assert match is not None
        assert match.group(1) == "FAILURE"
        assert match.group(2) == "pytest"

    def test_tool_status_pattern_timeout(self):
        """Test tool_status pattern matches TIMEOUT."""
        pattern = TOOLRUNNER_PATTERNS["tool_status"]
        match = re.search(pattern, "TIMEOUT: slow_tool")
        assert match is not None
        assert match.group(1) == "TIMEOUT"
        assert match.group(2) == "slow_tool"

    def test_output_summary_pattern(self):
        """Test output_summary pattern."""
        pattern = TOOLRUNNER_PATTERNS["output_summary"]
        match = re.search(pattern, "Output: All tests passed")
        assert match is not None
        assert match.group(1) == "All tests passed"

    def test_next_action_pattern_retry(self):
        """Test next_action pattern matches RETRY."""
        pattern = TOOLRUNNER_PATTERNS["next_action"]
        match = re.search(pattern, "Next: RETRY")
        assert match is not None
        assert match.group(1) == "RETRY"

    def test_next_action_pattern_escalate(self):
        """Test next_action pattern matches ESCALATE."""
        pattern = TOOLRUNNER_PATTERNS["next_action"]
        match = re.search(pattern, "Next: ESCALATE")
        assert match is not None
        assert match.group(1) == "ESCALATE"

    def test_next_action_pattern_complete(self):
        """Test next_action pattern matches COMPLETE."""
        pattern = TOOLRUNNER_PATTERNS["next_action"]
        match = re.search(pattern, "Next: COMPLETE")
        assert match is not None
        assert match.group(1) == "COMPLETE"

    def test_error_message_pattern(self):
        """Test error_message pattern."""
        pattern = TOOLRUNNER_PATTERNS["error_message"]
        match = re.search(pattern, "Error: File not found")
        assert match is not None
        assert match.group(1) == "File not found"


class TestParseToolrunnerOutput:
    """Test parse_toolrunner_output() function."""

    def test_full_output(self):
        """Test parsing complete toolrunner output."""
        output = """SUCCESS: bash_linter
Output: All checks passed
Next: COMPLETE"""
        result = parse_toolrunner_output(output)
        assert result["status"] == "SUCCESS"
        assert result["tool_name"] == "bash_linter"
        assert result["summary"] == "All checks passed"
        assert result["next_action"] == "COMPLETE"
        assert result["error"] is None

    def test_failure_with_error(self):
        """Test parsing failure output with error."""
        output = """FAILURE: pytest
Error: 3 tests failed
Next: RETRY"""
        result = parse_toolrunner_output(output)
        assert result["status"] == "FAILURE"
        assert result["tool_name"] == "pytest"
        assert result["error"] == "3 tests failed"
        assert result["next_action"] == "RETRY"

    def test_timeout_output(self):
        """Test parsing timeout output."""
        output = """TIMEOUT: slow_command
Next: ABORT"""
        result = parse_toolrunner_output(output)
        assert result["status"] == "TIMEOUT"
        assert result["tool_name"] == "slow_command"
        assert result["next_action"] == "ABORT"

    def test_empty_output(self):
        """Test parsing empty output."""
        result = parse_toolrunner_output("")
        assert result["status"] is None
        assert result["tool_name"] is None
        assert result["summary"] is None
        assert result["next_action"] is None
        assert result["error"] is None


class TestToolCall:
    """Test ToolCall Pydantic model."""

    def test_valid_tool_call(self):
        """Test valid ToolCall creation."""
        tool = ToolCall(tool="bash_linter", args={"file": "script.sh"})
        assert tool.tool == "bash_linter"
        assert tool.args == {"file": "script.sh"}

    def test_tool_name_pattern_valid(self):
        """Test tool name pattern validation accepts valid names."""
        # Valid: lowercase start, can have numbers and underscores
        ToolCall(tool="my_tool_123", args={})
        ToolCall(tool="tool", args={})
        ToolCall(tool="tool_name", args={})

    def test_tool_name_pattern_invalid(self):
        """Test tool name pattern validation rejects invalid names."""
        # Invalid: starts with uppercase
        with pytest.raises(ValidationError):
            ToolCall(tool="MyTool", args={})

        # Invalid: starts with number
        with pytest.raises(ValidationError):
            ToolCall(tool="123tool", args={})


class TestTaskIR:
    """Test TaskIR Pydantic model."""

    def test_valid_task_ir(self):
        """Test valid TaskIR creation."""
        task = TaskIR(
            task_id="task-123",
            task_type="code",
            priority="interactive",
            objective="Fix bug",
            agents=[{"role": "coder"}],
            gates=["lint", "test"],
        )
        assert task.task_id == "task-123"
        assert task.task_type == "code"
        assert task.priority == "interactive"

    def test_task_type_validation(self):
        """Test task_type literal validation."""
        # Valid types
        for task_type in ["code", "doc", "ingest", "manage", "chat"]:
            TaskIR(
                task_id="t1",
                task_type=task_type,
                priority="batch",
                objective="Test",
                agents=[],
                gates=[],
            )

    def test_priority_validation(self):
        """Test priority literal validation."""
        # Valid priorities
        for priority in ["interactive", "batch"]:
            TaskIR(
                task_id="t1",
                task_type="code",
                priority=priority,
                objective="Test",
                agents=[],
                gates=[],
            )


class TestFormalizationIR:
    """Test FormalizationIR Pydantic model."""

    def test_valid_formalization_ir(self):
        """Test valid FormalizationIR creation."""
        fir = FormalizationIR(
            problem_type="algorithm",
            variables=[{"name": "x", "type": "int", "constraints": ["x > 0"]}],
            constraints=["x < 100"],
            edge_cases=[{"input": "0", "expected": "error"}],
            acceptance_criteria=["All tests pass"],
            confidence=0.95,
        )
        assert fir.problem_type == "algorithm"
        assert fir.confidence == 0.95

    def test_confidence_bounds(self):
        """Test confidence field bounds validation."""
        # Valid: 0.0 to 1.0
        FormalizationIR(
            problem_type="proof",
            variables=[],
            constraints=[],
            edge_cases=[],
            acceptance_criteria=[],
            confidence=0.0,
        )
        FormalizationIR(
            problem_type="proof",
            variables=[],
            constraints=[],
            edge_cases=[],
            acceptance_criteria=[],
            confidence=1.0,
        )

        # Invalid: outside bounds
        with pytest.raises(ValidationError):
            FormalizationIR(
                problem_type="proof",
                variables=[],
                constraints=[],
                edge_cases=[],
                acceptance_criteria=[],
                confidence=-0.1,
            )
        with pytest.raises(ValidationError):
            FormalizationIR(
                problem_type="proof",
                variables=[],
                constraints=[],
                edge_cases=[],
                acceptance_criteria=[],
                confidence=1.1,
            )


class TestArchitectureIR:
    """Test ArchitectureIR Pydantic model."""

    def test_valid_architecture_ir(self):
        """Test valid ArchitectureIR creation."""
        air = ArchitectureIR(
            components=[{"name": "API", "responsibility": "HTTP", "interfaces": []}],
            relationships=[{"from": "API", "to": "DB", "type": "calls"}],
            invariants=["API never calls DB directly"],
            constraints=["Use async"],
            rationale="Performance",
        )
        assert len(air.components) == 1
        assert air.rationale == "Performance"

    def test_optional_alternatives(self):
        """Test alternatives_considered is optional."""
        air = ArchitectureIR(
            components=[],
            relationships=[],
            invariants=[],
            constraints=[],
            rationale="Test",
        )
        assert air.alternatives_considered is None

        # Can also be provided
        air2 = ArchitectureIR(
            components=[],
            relationships=[],
            invariants=[],
            constraints=[],
            rationale="Test",
            alternatives_considered=[{"option": "REST", "rejected": "complexity"}],
        )
        assert air2.alternatives_considered is not None
