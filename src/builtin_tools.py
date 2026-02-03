#!/usr/bin/env python3
"""Built-in tools for the orchestrator REPL.

These tools are registered with the ToolRegistry at startup and can be
invoked via TOOL(name, **kwargs) in the REPL environment.

Tools are organized by category:
- CODE: lint_python, run_tests, format_python
- WEB: fetch_docs, web_search (stub)
- DATA: read_json, write_json
- FILE: read_file, list_files
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.tool_registry import ToolCategory, ToolRegistry

logger = logging.getLogger(__name__)


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools with the registry.

    Args:
        registry: The ToolRegistry to register tools with.
    """
    _register_code_tools(registry)
    _register_data_tools(registry)
    _register_file_tools(registry)
    logger.info(f"Registered {len(registry._tools)} built-in tools")


def _register_code_tools(registry: ToolRegistry) -> None:
    """Register code-related tools."""

    @registry.register_handler(
        name="lint_python",
        description="Run Python linting on code or file",
        category=ToolCategory.CODE,
        parameters={
            "code": {"type": "string", "description": "Python code to lint", "required": False},
            "file_path": {
                "type": "string",
                "description": "Path to Python file",
                "required": False,
            },
        },
    )
    def lint_python(code: str | None = None, file_path: str | None = None) -> dict[str, Any]:
        """Lint Python code using ruff if available, otherwise basic ast check."""
        import ast
        import tempfile

        if code is None and file_path is None:
            return {"error": "Either 'code' or 'file_path' must be provided"}

        # If file path provided, read it
        if file_path:
            try:
                code = Path(file_path).read_text()
            except Exception as e:
                return {"error": f"Could not read file: {e}"}

        # Basic syntax check using ast
        try:
            ast.parse(code)
            syntax_ok = True
            syntax_error = None
        except SyntaxError as e:
            syntax_ok = False
            syntax_error = f"Line {e.lineno}: {e.msg}"

        result = {
            "syntax_valid": syntax_ok,
            "syntax_error": syntax_error,
            "issues": [],
        }

        # Try ruff if available
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            proc = subprocess.run(
                [sys.executable, "-m", "ruff", "check", temp_path, "--output-format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.stdout:
                issues = json.loads(proc.stdout)
                result["issues"] = [
                    {"line": i.get("location", {}).get("row"), "message": i.get("message")}
                    for i in issues
                ]
            Path(temp_path).unlink()
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass  # ruff not available or failed

        return result

    @registry.register_handler(
        name="run_tests",
        description="Run pytest tests on a file or directory",
        category=ToolCategory.CODE,
        parameters={
            "path": {
                "type": "string",
                "description": "Path to test file or directory",
                "required": True,
            },
            "verbose": {"type": "boolean", "description": "Verbose output", "required": False},
        },
    )
    def run_tests(path: str, verbose: bool = False) -> dict[str, Any]:
        """Run pytest on the specified path."""
        cmd = [sys.executable, "-m", "pytest", path]
        if verbose:
            cmd.append("-v")
        cmd.append("--tb=short")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {
                "passed": proc.returncode == 0,
                "return_code": proc.returncode,
                "stdout": proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout,
                "stderr": proc.stderr[-1000:] if len(proc.stderr) > 1000 else proc.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Test execution timed out after 120s", "passed": False}
        except Exception as e:
            return {"error": str(e), "passed": False}

    @registry.register_handler(
        name="format_python",
        description="Format Python code using black or ruff",
        category=ToolCategory.CODE,
        parameters={
            "code": {"type": "string", "description": "Python code to format", "required": True},
        },
    )
    def format_python(code: str) -> dict[str, Any]:
        """Format Python code."""
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            # Try ruff format first
            subprocess.run(
                [sys.executable, "-m", "ruff", "format", temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            formatted = Path(temp_path).read_text()
            Path(temp_path).unlink()

            return {
                "formatted": formatted,
                "changed": formatted != code,
            }
        except Exception as e:
            return {"error": str(e), "formatted": code, "changed": False}


def _register_data_tools(registry: ToolRegistry) -> None:
    """Register data-related tools."""

    @registry.register_handler(
        name="read_json",
        description="Read and parse a JSON file",
        category=ToolCategory.DATA,
        parameters={
            "path": {"type": "string", "description": "Path to JSON file", "required": True},
        },
    )
    def read_json(path: str) -> dict[str, Any]:
        """Read and parse a JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            return {"data": data, "success": True}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

    @registry.register_handler(
        name="write_json",
        description="Write data to a JSON file",
        category=ToolCategory.DATA,
        parameters={
            "path": {"type": "string", "description": "Path to JSON file", "required": True},
            "data": {"type": "object", "description": "Data to write", "required": True},
            "indent": {"type": "integer", "description": "Indentation level", "required": False},
        },
    )
    def write_json(path: str, data: Any, indent: int = 2) -> dict[str, Any]:
        """Write data to a JSON file."""
        # Security: only allow writing to RAID array
        from src.config import get_config

        _raid_prefix = get_config().paths.raid_prefix
        if not path.startswith(_raid_prefix):
            return {"error": f"Can only write to {_raid_prefix}", "success": False}

        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=indent)
            return {"path": path, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}


def _register_file_tools(registry: ToolRegistry) -> None:
    """Register file-related tools."""

    @registry.register_handler(
        name="read_file",
        description="Read contents of a file",
        category=ToolCategory.FILE,
        parameters={
            "path": {"type": "string", "description": "Path to file", "required": True},
            "max_lines": {
                "type": "integer",
                "description": "Maximum lines to read",
                "required": False,
            },
        },
    )
    def read_file(path: str, max_lines: int | None = None) -> dict[str, Any]:
        """Read contents of a file."""
        try:
            with open(path) as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = "".join(lines)
                    truncated = True
                else:
                    content = f.read()
                    truncated = False

            return {
                "content": content,
                "truncated": truncated,
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    @registry.register_handler(
        name="list_files",
        description="List files in a directory",
        category=ToolCategory.FILE,
        parameters={
            "path": {"type": "string", "description": "Directory path", "required": True},
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter",
                "required": False,
            },
        },
    )
    def list_files(path: str, pattern: str = "*") -> dict[str, Any]:
        """List files in a directory."""
        try:
            dir_path = Path(path)
            if not dir_path.is_dir():
                return {"error": f"Not a directory: {path}", "success": False}

            files = list(dir_path.glob(pattern))
            return {
                "files": [str(f) for f in files[:100]],  # Limit to 100
                "count": len(files),
                "truncated": len(files) > 100,
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}
