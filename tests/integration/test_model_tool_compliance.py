#!/usr/bin/env python3
"""Integration tests for model REPL tool compliance.

These tests validate that models in the orchestration hierarchy correctly use
REPL tools instead of Python imports. This is critical because:

1. The REPL sandbox blocks imports for security (os, subprocess, pathlib, etc.)
2. Models must use provided tools: list_dir(), peek(), grep(), etc.
3. Import failures waste turns and degrade user experience

Test coverage:
- Each role is tested against their required tools
- Checks for forbidden import patterns
- Validates correct tool function patterns in responses

To run all tests:
    pytest tests/integration/test_model_tool_compliance.py -v

To run with live models (requires orchestrator running):
    pytest tests/integration/test_model_tool_compliance.py -v --run-live-models

See handoffs/active/model_repl_tool_compliance.md for full requirements.
"""

import re
import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Tool Definitions
# =============================================================================

# Tools available in the REPL environment
REPL_TOOLS = {
    # Context & Files
    "peek": {
        "signature": "peek(n, file_path=None)",
        "description": "Return first n characters of context or file",
        "expected_pattern": r"peek\s*\(",
    },
    "grep": {
        "signature": "grep(pattern, file_path=None)",
        "description": "Search context or file with regex",
        "expected_pattern": r"grep\s*\(",
    },
    "list_dir": {
        "signature": "list_dir(path)",
        "description": "List directory contents as JSON",
        "expected_pattern": r"list_dir\s*\(",
    },
    "file_info": {
        "signature": "file_info(path)",
        "description": "Get file metadata",
        "expected_pattern": r"file_info\s*\(",
    },
    # Document Processing
    "ocr_document": {
        "signature": "ocr_document(path)",
        "description": "Extract text and figure bboxes from PDF",
        "expected_pattern": r"ocr_document\s*\(",
    },
    "analyze_figure": {
        "signature": "analyze_figure(image_path, prompt)",
        "description": "Analyze image with vision model",
        "expected_pattern": r"analyze_figure\s*\(",
    },
    "extract_figure": {
        "signature": "extract_figure(pdf_path, page, bbox)",
        "description": "Crop figure from PDF",
        "expected_pattern": r"extract_figure\s*\(",
    },
    # LLM Delegation
    "llm_call": {
        "signature": "llm_call(prompt, role='worker')",
        "description": "Call a sub-LM for a task",
        "expected_pattern": r"llm_call\s*\(",
    },
    "llm_batch": {
        "signature": "llm_batch(prompts, role='worker')",
        "description": "Parallel sub-LM calls",
        "expected_pattern": r"llm_batch\s*\(",
    },
    "escalate": {
        "signature": "escalate(reason)",
        "description": "Request escalation to higher-tier model",
        "expected_pattern": r"escalate\s*\(",
    },
    "recall": {
        "signature": "recall(query)",
        "description": "Search episodic memory",
        "expected_pattern": r"recall\s*\(",
    },
    # Completion
    "FINAL": {
        "signature": "FINAL(answer)",
        "description": "Signal completion with final answer",
        "expected_pattern": r"FINAL\s*\(",
    },
    # Web & Shell
    "web_fetch": {
        "signature": "web_fetch(url)",
        "description": "Fetch web content",
        "expected_pattern": r"web_fetch\s*\(",
    },
    "run_shell": {
        "signature": "run_shell(cmd)",
        "description": "Run sandboxed shell command",
        "expected_pattern": r"run_shell\s*\(",
    },
}

# Forbidden patterns - imports that are blocked by the sandbox
FORBIDDEN_PATTERNS = [
    (r"^import\s+", "import statement"),
    (r"^from\s+\w+\s+import", "from import statement"),
    (r"os\.listdir\s*\(", "os.listdir (use list_dir)"),
    (r"os\.path\.", "os.path (use file_info or list_dir)"),
    (r"pathlib\.", "pathlib (use list_dir or file_info)"),
    (r"Path\s*\(", "Path() (use list_dir or file_info)"),
    (r"open\s*\(", "open() (use peek)"),
    (r"subprocess\.", "subprocess (use run_shell)"),
    (r"glob\.glob\s*\(", "glob.glob (use list_dir)"),
]

# Role → Required tools mapping
ROLE_TOOL_REQUIREMENTS = {
    "frontdoor": {
        "required": ["peek", "grep", "list_dir", "file_info", "FINAL"],
        "optional": ["llm_call", "escalate", "recall"],
    },
    "coder_escalation": {
        "required": ["peek", "grep", "llm_call", "FINAL"],
        "optional": ["list_dir", "run_shell", "escalate"],
    },
    "worker_general": {
        "required": ["peek", "grep", "FINAL"],
        "optional": ["llm_call"],
    },
    "ingest_long_context": {
        "required": ["ocr_document", "peek", "FINAL"],
        "optional": ["extract_figure", "analyze_figure"],
    },
    "architect_general": {
        "required": ["peek", "grep", "llm_call", "FINAL"],
        "optional": ["escalate", "recall"],
    },
}


# =============================================================================
# Test Prompts
# =============================================================================

# Prompts designed to test specific tool usage
TOOL_TEST_PROMPTS = {
    "list_dir": {
        "prompt": "List all files in /mnt/raid0/llm/claude/tmp",
        "expected_tool": "list_dir",
        "forbidden_alternative": "os.listdir",
    },
    "peek": {
        "prompt": "Show the first 100 characters of /mnt/raid0/llm/claude/README.md",
        "expected_tool": "peek",
        "forbidden_alternative": "open(",
    },
    "grep": {
        "prompt": "Search for all lines containing 'def test_' in the context",
        "expected_tool": "grep",
        "forbidden_alternative": "re.findall",
    },
    "file_info": {
        "prompt": "Get the file size and modification date of /mnt/raid0/llm/claude/CLAUDE.md",
        "expected_tool": "file_info",
        "forbidden_alternative": "os.stat",
    },
    "ocr_document": {
        "prompt": "Extract text from /mnt/raid0/llm/tmp/document.pdf",
        "expected_tool": "ocr_document",
        "forbidden_alternative": "PyPDF",
    },
}


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_primitives():
    """Create mock LLM primitives for testing."""
    from src.llm_primitives import LLMPrimitives

    primitives = LLMPrimitives(mock_mode=True)
    return primitives


@pytest.fixture
def mock_repl_environment():
    """Create a mock REPL environment for testing."""
    from src.repl_environment import REPLEnvironment, REPLConfig

    config = REPLConfig(timeout_seconds=10, output_cap=4096)
    repl = REPLEnvironment(
        context="Sample context for testing tool compliance.",
        config=config,
    )
    return repl


@pytest.fixture
def compliant_response_list_dir():
    """Example of a compliant response for list_dir task."""
    return """result = list_dir('/mnt/raid0/llm/claude/tmp')
FINAL(result)"""


@pytest.fixture
def non_compliant_response_list_dir():
    """Example of a non-compliant response that uses os.listdir."""
    return """import os
files = os.listdir('/mnt/raid0/llm/claude/tmp')
FINAL(files)"""


@pytest.fixture
def compliant_response_peek():
    """Example of a compliant response for peek task."""
    return """content = peek(100, file_path='/mnt/raid0/llm/claude/README.md')
FINAL(content)"""


@pytest.fixture
def non_compliant_response_peek():
    """Example of a non-compliant response that uses open()."""
    return """with open('/mnt/raid0/llm/claude/README.md', 'r') as f:
    content = f.read(100)
FINAL(content)"""


# =============================================================================
# Compliance Checking Functions
# =============================================================================


def check_forbidden_patterns(response: str) -> list[tuple[str, str]]:
    """Check if response contains forbidden patterns.

    Args:
        response: Model response to check.

    Returns:
        List of (pattern_description, matched_text) tuples for violations.
    """
    violations = []
    lines = response.split("\n")

    for line in lines:
        stripped = line.strip()
        for pattern, description in FORBIDDEN_PATTERNS:
            if re.search(pattern, stripped, re.MULTILINE):
                violations.append((description, stripped[:50]))

    return violations


def check_tool_usage(response: str, expected_tool: str) -> bool:
    """Check if response uses the expected REPL tool.

    Args:
        response: Model response to check.
        expected_tool: Name of expected tool (e.g., "list_dir").

    Returns:
        True if the expected tool pattern is found.
    """
    tool_info = REPL_TOOLS.get(expected_tool)
    if not tool_info:
        return False

    pattern = tool_info["expected_pattern"]
    return bool(re.search(pattern, response))


def get_compliance_score(response: str, expected_tools: list[str]) -> dict:
    """Calculate compliance score for a response.

    Args:
        response: Model response to check.
        expected_tools: List of expected tool names.

    Returns:
        Dict with score details:
        - tools_used: List of tools found in response
        - tools_missing: List of expected tools not found
        - forbidden_violations: List of forbidden pattern violations
        - is_compliant: True if no violations and at least one expected tool used
    """
    tools_used = []
    tools_missing = []

    for tool_name in expected_tools:
        if check_tool_usage(response, tool_name):
            tools_used.append(tool_name)
        else:
            tools_missing.append(tool_name)

    violations = check_forbidden_patterns(response)

    # Compliance requires:
    # 1. No forbidden pattern violations
    # 2. At least one expected tool used (if expected_tools non-empty)
    is_compliant = len(violations) == 0 and (len(expected_tools) == 0 or len(tools_used) > 0)

    return {
        "tools_used": tools_used,
        "tools_missing": tools_missing,
        "forbidden_violations": violations,
        "is_compliant": is_compliant,
    }


# =============================================================================
# Unit Tests for Compliance Checking
# =============================================================================


class TestComplianceChecker:
    """Unit tests for the compliance checking functions."""

    def test_check_forbidden_patterns_clean(self):
        """Test that clean code passes."""
        code = """result = list_dir('/tmp')
FINAL(result)"""
        violations = check_forbidden_patterns(code)
        assert len(violations) == 0

    def test_check_forbidden_patterns_import(self):
        """Test that import is detected."""
        code = """import os
files = os.listdir('/tmp')"""
        violations = check_forbidden_patterns(code)
        assert len(violations) >= 1
        assert any("import" in v[0] for v in violations)

    def test_check_forbidden_patterns_os_listdir(self):
        """Test that os.listdir is detected."""
        code = """files = os.listdir('/tmp')"""
        violations = check_forbidden_patterns(code)
        assert len(violations) >= 1
        assert any("os.listdir" in v[0] for v in violations)

    def test_check_forbidden_patterns_pathlib(self):
        """Test that pathlib is detected."""
        code = """from pathlib import Path
p = Path('/tmp')"""
        violations = check_forbidden_patterns(code)
        assert len(violations) >= 1

    def test_check_tool_usage_list_dir(self):
        """Test list_dir detection."""
        code = "result = list_dir('/tmp')"
        assert check_tool_usage(code, "list_dir") is True

    def test_check_tool_usage_peek(self):
        """Test peek detection."""
        code = "content = peek(100, file_path='/path')"
        assert check_tool_usage(code, "peek") is True

    def test_check_tool_usage_grep(self):
        """Test grep detection."""
        code = "matches = grep(r'pattern')"
        assert check_tool_usage(code, "grep") is True

    def test_check_tool_usage_final(self):
        """Test FINAL detection."""
        code = "FINAL(result)"
        assert check_tool_usage(code, "FINAL") is True

    def test_check_tool_usage_missing(self):
        """Test when tool is not used."""
        code = "print('hello')"
        assert check_tool_usage(code, "list_dir") is False

    def test_compliance_score_compliant(self, compliant_response_list_dir):
        """Test compliance score for compliant response."""
        score = get_compliance_score(
            compliant_response_list_dir,
            ["list_dir", "FINAL"],
        )
        assert score["is_compliant"] is True
        assert "list_dir" in score["tools_used"]
        assert "FINAL" in score["tools_used"]
        assert len(score["forbidden_violations"]) == 0

    def test_compliance_score_non_compliant(self, non_compliant_response_list_dir):
        """Test compliance score for non-compliant response."""
        score = get_compliance_score(
            non_compliant_response_list_dir,
            ["list_dir", "FINAL"],
        )
        assert score["is_compliant"] is False
        assert len(score["forbidden_violations"]) >= 1


# =============================================================================
# REPL Execution Tests
# =============================================================================


class TestREPLToolExecution:
    """Tests for REPL tool execution."""

    def test_list_dir_executes_successfully(self, mock_repl_environment):
        """Test that list_dir can be executed in REPL."""
        code = "result = list_dir('/mnt/raid0/llm/claude')"
        result = mock_repl_environment.execute(code)

        # Should execute without security error
        assert result.error is None or "SecurityError" not in str(result.error)

    def test_peek_executes_successfully(self, mock_repl_environment):
        """Test that peek can be executed in REPL."""
        code = "content = peek(50)"
        result = mock_repl_environment.execute(code)

        assert result.error is None
        # Peek returns first 50 chars of context
        assert result.output is not None or result.is_final is False

    def test_grep_executes_successfully(self, mock_repl_environment):
        """Test that grep can be executed in REPL."""
        code = "matches = grep(r'test')"
        result = mock_repl_environment.execute(code)

        assert result.error is None

    def test_import_blocked(self, mock_repl_environment):
        """Test that import statements are blocked."""
        code = "import os"
        result = mock_repl_environment.execute(code)

        # Should get a security error
        assert result.error is not None
        assert "import" in result.error.lower() or "security" in result.error.lower()

    def test_os_module_blocked(self, mock_repl_environment):
        """Test that os module operations are blocked."""
        code = "os.listdir('/tmp')"
        result = mock_repl_environment.execute(code)

        # Should fail (os not available)
        assert result.error is not None

    def test_compliant_code_succeeds(self, mock_repl_environment, compliant_response_list_dir):
        """Test that compliant code executes successfully."""
        result = mock_repl_environment.execute(compliant_response_list_dir)

        # Should complete with FINAL
        assert result.is_final is True or result.error is None


# =============================================================================
# Model Response Tests (Mock Mode)
# =============================================================================


class TestModelResponsePatterns:
    """Tests for validating expected model response patterns."""

    @pytest.mark.parametrize("role", ["frontdoor", "coder_escalation", "worker_general"])
    def test_role_has_required_tools(self, role):
        """Test that each role has defined required tools."""
        assert role in ROLE_TOOL_REQUIREMENTS
        requirements = ROLE_TOOL_REQUIREMENTS[role]
        assert "required" in requirements
        assert len(requirements["required"]) > 0

    @pytest.mark.parametrize("tool_name", ["list_dir", "peek", "grep", "file_info"])
    def test_tool_has_definition(self, tool_name):
        """Test that each tool has a proper definition."""
        assert tool_name in REPL_TOOLS
        tool_info = REPL_TOOLS[tool_name]
        assert "signature" in tool_info
        assert "expected_pattern" in tool_info
        assert "description" in tool_info

    @pytest.mark.parametrize("tool_name,test_config", list(TOOL_TEST_PROMPTS.items()))
    def test_tool_prompt_has_config(self, tool_name, test_config):
        """Test that each tool test prompt has required config."""
        assert "prompt" in test_config
        assert "expected_tool" in test_config
        assert test_config["expected_tool"] in REPL_TOOLS


# =============================================================================
# Live Model Tests (requires --run-live-models)
# =============================================================================


@pytest.mark.skipif(
    "not config.getoption('--run-live-models', default=False)",
    reason="Requires --run-live-models flag",
)
class TestLiveModelCompliance:
    """Live tests against running models.

    These tests require the orchestrator to be running with live models.
    Run with: pytest --run-live-models tests/integration/test_model_tool_compliance.py
    """

    @pytest.fixture
    def live_llm_primitives(self):
        """Get LLM primitives connected to live servers."""
        from src.llm_primitives import LLMPrimitives

        # Use default server URLs from orchestrator config
        primitives = LLMPrimitives(mock_mode=False)
        return primitives

    @pytest.mark.parametrize("role", ["frontdoor", "coder_escalation"])
    @pytest.mark.parametrize("tool_name", ["list_dir", "peek"])
    def test_model_uses_correct_tool(self, live_llm_primitives, role, tool_name):
        """Test that model uses REPL tool instead of imports."""
        test_config = TOOL_TEST_PROMPTS[tool_name]
        prompt = test_config["prompt"]

        # Call the model
        response = live_llm_primitives.llm_call(
            prompt=prompt,
            role=role,
        )

        # Check compliance
        score = get_compliance_score(response, [tool_name])

        # Assert tool was used
        assert tool_name in score["tools_used"], (
            f"{role} did not use {tool_name} for prompt: {prompt}\nResponse: {response[:200]}"
        )

        # Assert no forbidden patterns
        assert len(score["forbidden_violations"]) == 0, (
            f"{role} used forbidden patterns: {score['forbidden_violations']}\n"
            f"Response: {response[:200]}"
        )


# =============================================================================
# Compliance Dashboard Data (for tracking over time)
# =============================================================================


class TestComplianceDashboard:
    """Tests for compliance dashboard data collection."""

    def test_collect_compliance_metrics(self):
        """Test that we can collect compliance metrics."""
        # Example responses to score
        responses = {
            "frontdoor_list_dir": "result = list_dir('/tmp')\nFINAL(result)",
            "frontdoor_list_dir_bad": "import os\nfiles = os.listdir('/tmp')",
            "coder_peek": "content = peek(100)\nFINAL(content)",
        }

        metrics = {}
        for name, response in responses.items():
            name.split("_")[0]
            tool = (
                "_".join(name.split("_")[1:-1]) if "bad" in name else "_".join(name.split("_")[1:])
            )

            score = get_compliance_score(response, [tool, "FINAL"] if "bad" not in name else [tool])
            metrics[name] = {
                "is_compliant": score["is_compliant"],
                "tools_used": score["tools_used"],
                "violations": len(score["forbidden_violations"]),
            }

        # Verify we can collect metrics
        assert len(metrics) == 3
        assert metrics["frontdoor_list_dir"]["is_compliant"] is True
        assert metrics["frontdoor_list_dir_bad"]["is_compliant"] is False


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-live-models",
        action="store_true",
        default=False,
        help="Run tests against live models (requires orchestrator)",
    )
