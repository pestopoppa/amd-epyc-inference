#!/usr/bin/env python3
"""Directory listing tool.

Provides directory listing with:
- File type indicators
- Size information
- Glob pattern filtering
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.tool_registry import Tool, ToolCategory, ToolRegistry

logger = logging.getLogger(__name__)

# Allowed path prefixes (safety)
ALLOWED_PATHS = [
    "/mnt/raid0/llm/",
    "/tmp/",
]


def _validate_path(path: str) -> tuple[bool, str | None]:
    """Validate that a path is allowed."""
    path_obj = Path(path).resolve()
    path_str = str(path_obj)

    for allowed in ALLOWED_PATHS:
        if path_str.startswith(allowed):
            return True, None

    return False, f"Path not in allowed locations: {ALLOWED_PATHS}"


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
        size /= 1024
    return f"{size:.1f}TB"


def list_dir(
    directory: str,
    pattern: str = "*",
    show_hidden: bool = False,
    show_size: bool = True,
    recursive: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """List contents of a directory.

    Args:
        directory: Directory path to list.
        pattern: Glob pattern to filter (default "*").
        show_hidden: Include hidden files (starting with .).
        show_size: Show file sizes.
        recursive: Recursively list subdirectories.
        limit: Maximum entries to return.

    Returns:
        Dict with entries list and metadata.
    """
    # Validate path
    is_valid, error = _validate_path(directory)
    if not is_valid:
        return {
            "success": False,
            "error": error,
            "directory": directory,
        }

    path = Path(directory)

    if not path.exists():
        return {
            "success": False,
            "error": f"Directory not found: {directory}",
            "directory": directory,
        }

    if not path.is_dir():
        return {
            "success": False,
            "error": f"Not a directory: {directory}",
            "directory": directory,
        }

    try:
        # Get entries
        if recursive:
            entries = list(path.rglob(pattern))
        else:
            entries = list(path.glob(pattern))

        # Filter hidden if needed
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        # Sort: directories first, then by name
        entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))

        # Limit entries
        total_count = len(entries)
        entries = entries[:limit]

        # Format entries
        formatted = []
        for entry in entries:
            info = {
                "name": entry.name,
                "path": str(entry),
                "type": "dir" if entry.is_dir() else "file",
            }

            if show_size and entry.is_file():
                try:
                    info["size"] = _format_size(entry.stat().st_size)
                except OSError:
                    info["size"] = "?"

            formatted.append(info)

        return {
            "success": True,
            "entries": formatted,
            "directory": directory,
            "pattern": pattern,
            "total_count": total_count,
            "returned_count": len(formatted),
            "truncated": total_count > limit,
        }

    except Exception as e:
        logger.exception(f"Error listing directory {directory}")
        return {
            "success": False,
            "error": str(e),
            "directory": directory,
        }


def register_list_tool(registry: ToolRegistry) -> int:
    """Register the list_dir tool.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered (1).
    """
    tool = Tool(
        name="list_dir",
        description="List contents of a directory",
        category=ToolCategory.FILE,
        parameters={
            "directory": {
                "type": "string",
                "description": "Directory path to list",
                "required": True,
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter (default '*')",
                "required": False,
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Include hidden files",
                "required": False,
            },
            "show_size": {
                "type": "boolean",
                "description": "Show file sizes",
                "required": False,
            },
            "recursive": {
                "type": "boolean",
                "description": "Recursively list subdirectories",
                "required": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum entries (default 100)",
                "required": False,
            },
        },
        handler=list_dir,
    )

    registry.register_tool(tool)
    return 1
