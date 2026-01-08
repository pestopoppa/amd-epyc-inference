#!/usr/bin/env python3
"""Test runner tool for executing pytest.

Provides test execution with:
- Specific test file/function targeting
- Output capture and formatting
- Exit code handling
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from src.tool_registry import Tool, ToolCategory, ToolRegistry
from src.tools.base import truncate_output

logger = logging.getLogger(__name__)

# Allowed working directories
ALLOWED_PATHS = [
    "/mnt/raid0/llm/claude",
]


def _validate_path(path: str) -> tuple[bool, str | None]:
    """Validate that a path is allowed."""
    path_obj = Path(path).resolve()
    path_str = str(path_obj)

    for allowed in ALLOWED_PATHS:
        if path_str.startswith(allowed):
            return True, None

    return False, f"Path not in allowed locations: {ALLOWED_PATHS}"


def run_tests(
    test_path: str = "tests/",
    test_pattern: str | None = None,
    verbose: bool = False,
    timeout: int = 300,
    working_dir: str = "/mnt/raid0/llm/claude",
) -> dict[str, Any]:
    """Run pytest tests.

    Args:
        test_path: Path to test file or directory.
        test_pattern: Specific test pattern (-k flag).
        verbose: Enable verbose output.
        timeout: Maximum execution time in seconds.
        working_dir: Working directory for test execution.

    Returns:
        Dict with success, output, and test results.
    """
    # Validate paths
    is_valid, error = _validate_path(working_dir)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "test_path": test_path,
        }

    # Build pytest command
    cmd = ["python", "-m", "pytest", test_path]

    if test_pattern:
        cmd.extend(["-k", test_pattern])

    if verbose:
        cmd.append("-v")

    # Always show summary
    cmd.append("-q")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )

        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr

        # Truncate if needed
        output, truncated = truncate_output(output, max_length=16000)

        # Parse summary line for counts
        passed = 0
        failed = 0
        errors = 0

        for line in output.split("\n"):
            line_lower = line.lower()
            if "passed" in line_lower:
                import re
                match = re.search(r"(\d+)\s+passed", line_lower)
                if match:
                    passed = int(match.group(1))
            if "failed" in line_lower:
                import re
                match = re.search(r"(\d+)\s+failed", line_lower)
                if match:
                    failed = int(match.group(1))
            if "error" in line_lower:
                import re
                match = re.search(r"(\d+)\s+error", line_lower)
                if match:
                    errors = int(match.group(1))

        return {
            "success": result.returncode == 0,
            "output": output,
            "test_path": test_path,
            "exit_code": result.returncode,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "truncated": truncated,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Test execution timed out after {timeout}s",
            "test_path": test_path,
        }
    except Exception as e:
        logger.exception(f"Error running tests: {e}")
        return {
            "success": False,
            "error": str(e),
            "test_path": test_path,
        }


def register_run_tests_tool(registry: ToolRegistry) -> int:
    """Register the run_tests tool.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered (1).
    """
    tool = Tool(
        name="run_tests",
        description="Run pytest tests",
        category=ToolCategory.CODE,
        parameters={
            "test_path": {
                "type": "string",
                "description": "Path to test file or directory",
                "required": False,
            },
            "test_pattern": {
                "type": "string",
                "description": "Test name pattern (-k flag)",
                "required": False,
            },
            "verbose": {
                "type": "boolean",
                "description": "Enable verbose output",
                "required": False,
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time (default 300s)",
                "required": False,
            },
        },
        handler=run_tests,
    )

    registry.register_tool(tool)
    return 1
