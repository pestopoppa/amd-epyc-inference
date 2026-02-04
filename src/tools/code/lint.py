#!/usr/bin/env python3
"""Linting tool for code quality checks.

Provides linting with:
- Ruff for Python (fast)
- Multiple file support
- JSON output parsing
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.config import _registry_timeout
from src.tool_registry import Tool, ToolCategory, ToolRegistry
from src.tools.base import truncate_output

logger = logging.getLogger(__name__)

# Default lint timeout from registry
_LINT_TIMEOUT = int(_registry_timeout("tools", "lint", 60))


def _get_project_root() -> str:
    """Get project root from config with fallback."""
    import os
    from pathlib import Path

    try:
        from src.config import get_config

        root = str(get_config().paths.project_root)
        if Path(root).exists():
            return root
    except Exception:
        pass

    # Fallback: try hardcoded path, then cwd
    if Path("/mnt/raid0/llm/claude").exists():
        return "/mnt/raid0/llm/claude"
    return os.getcwd()


# Allowed paths (computed at module load)
ALLOWED_PATHS = [
    _get_project_root(),
]


def _validate_path(path: str) -> tuple[bool, str | None]:
    """Validate that a path is allowed."""
    path_obj = Path(path).resolve()
    path_str = str(path_obj)

    for allowed in ALLOWED_PATHS:
        if path_str.startswith(allowed):
            return True, None

    return False, f"Path not in allowed locations: {ALLOWED_PATHS}"


def lint_python(
    file_path: str,
    fix: bool = False,
    select: str | None = None,
    ignore: str | None = None,
    timeout: int = _LINT_TIMEOUT,
) -> dict[str, Any]:
    """Run ruff linter on Python files.

    Args:
        file_path: Path to file or directory to lint.
        fix: Automatically fix fixable issues.
        select: Rule codes to check (comma-separated).
        ignore: Rule codes to ignore (comma-separated).
        timeout: Maximum execution time in seconds (from registry).

    Returns:
        Dict with success, issues, and output.
    """
    # Validate path
    is_valid, error = _validate_path(file_path)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "file_path": file_path,
        }

    path = Path(file_path)
    if not path.exists():
        return {
            "success": False,
            "error": f"Path not found: {file_path}",
            "file_path": file_path,
        }

    # Build ruff command
    cmd = ["python", "-m", "ruff", "check"]

    if fix:
        cmd.append("--fix")

    if select:
        cmd.extend(["--select", select])

    if ignore:
        cmd.extend(["--ignore", ignore])

    # Output format
    cmd.extend(["--output-format", "json"])

    cmd.append(str(path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_get_project_root(),
        )

        # Parse JSON output
        issues = []
        if result.stdout.strip():
            try:
                issues = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Fall back to raw output
                pass

        # Format issues for readability
        formatted_issues = []
        for issue in issues:
            if isinstance(issue, dict):
                formatted_issues.append(
                    {
                        "file": issue.get("filename", ""),
                        "line": issue.get("location", {}).get("row", 0),
                        "code": issue.get("code", ""),
                        "message": issue.get("message", ""),
                        "fix_available": issue.get("fix") is not None,
                    }
                )

        output = ""
        if formatted_issues:
            output = "\n".join(
                f"{i['file']}:{i['line']} [{i['code']}] {i['message']}" for i in formatted_issues
            )
        elif result.stderr:
            output = result.stderr

        output, truncated = truncate_output(output, max_length=8000)

        return {
            "success": result.returncode == 0,
            "output": output,
            "file_path": file_path,
            "issue_count": len(formatted_issues),
            "issues": formatted_issues[:50],  # Limit to 50 issues
            "exit_code": result.returncode,
            "truncated": truncated,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Linting timed out after {timeout}s",
            "file_path": file_path,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "Ruff not installed. Install with: pip install ruff",
            "file_path": file_path,
        }
    except Exception as e:
        logger.exception(f"Error running lint: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


def register_lint_tool(registry: ToolRegistry) -> int:
    """Register the lint_python tool.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered (1).
    """
    tool = Tool(
        name="lint_python",
        description="Run ruff linter on Python files",
        category=ToolCategory.CODE,
        parameters={
            "file_path": {
                "type": "string",
                "description": "Path to file or directory to lint",
                "required": True,
            },
            "fix": {
                "type": "boolean",
                "description": "Automatically fix fixable issues",
                "required": False,
            },
            "select": {
                "type": "string",
                "description": "Rule codes to check (comma-separated)",
                "required": False,
            },
            "ignore": {
                "type": "string",
                "description": "Rule codes to ignore (comma-separated)",
                "required": False,
            },
        },
        handler=lint_python,
    )

    registry.register_tool(tool)
    return 1
