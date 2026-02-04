#!/usr/bin/env python3
"""File read tool for reading file contents.

Provides safe file reading with:
- Line number prefixing
- Offset and limit support
- Path validation (RAID only)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.tool_registry import Tool, ToolCategory, ToolRegistry
from src.tools.base import truncate_output

logger = logging.getLogger(__name__)


def _get_allowed_paths() -> list[str]:
    """Get allowed path prefixes from config with fallback."""
    try:
        from src.config import get_config

        cfg = get_config()
        llm_root = str(cfg.paths.llm_root)
        # Ensure trailing slash for prefix matching
        return [
            llm_root if llm_root.endswith("/") else f"{llm_root}/",
            "/tmp/",
        ]
    except Exception:
        return ["/mnt/raid0/llm/", "/tmp/"]


# Allowed path prefixes (computed at module load)
ALLOWED_PATHS = _get_allowed_paths()


def _validate_path(path: str) -> tuple[bool, str | None]:
    """Validate that a path is allowed.

    Args:
        path: Path to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    path_obj = Path(path).resolve()
    path_str = str(path_obj)

    for allowed in ALLOWED_PATHS:
        if path_str.startswith(allowed):
            return True, None

    return False, f"Path not in allowed locations: {ALLOWED_PATHS}"


def read_file(
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
    show_line_numbers: bool = True,
) -> dict[str, Any]:
    """Read contents of a file.

    Args:
        file_path: Path to file to read.
        offset: Line number to start from (0-indexed).
        limit: Maximum number of lines to read.
        show_line_numbers: Whether to prefix lines with numbers.

    Returns:
        Dict with content, line_count, and metadata.
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
            "error": f"File not found: {file_path}",
            "file_path": file_path,
        }

    if not path.is_file():
        return {
            "success": False,
            "error": f"Not a file: {file_path}",
            "file_path": file_path,
        }

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        selected_lines = all_lines[offset : offset + limit]

        # Format with line numbers if requested
        if show_line_numbers:
            formatted_lines = []
            for i, line in enumerate(selected_lines, start=offset + 1):
                # Truncate long lines
                if len(line) > 2000:
                    line = line[:2000] + "...[truncated]\n"
                formatted_lines.append(f"{i:>6}\t{line.rstrip()}")
            content = "\n".join(formatted_lines)
        else:
            content = "".join(selected_lines)

        # Truncate if needed
        content, truncated = truncate_output(content, max_length=32000)

        return {
            "success": True,
            "content": content,
            "file_path": file_path,
            "total_lines": total_lines,
            "lines_returned": len(selected_lines),
            "offset": offset,
            "truncated": truncated,
        }

    except Exception as e:
        logger.exception(f"Error reading file {file_path}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


def register_read_tool(registry: ToolRegistry) -> int:
    """Register the read_file tool.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered (1).
    """
    tool = Tool(
        name="read_file",
        description="Read contents of a file",
        category=ToolCategory.FILE,
        parameters={
            "file_path": {
                "type": "string",
                "description": "Path to file to read",
                "required": True,
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start from (0-indexed)",
                "required": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read (default 2000)",
                "required": False,
            },
            "show_line_numbers": {
                "type": "boolean",
                "description": "Prefix lines with numbers",
                "required": False,
            },
        },
        handler=read_file,
    )

    registry.register_tool(tool)
    return 1
