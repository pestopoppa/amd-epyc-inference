"""File tools for file system operations.

Tools for reading files and listing directories:
- read_file: Read file content with line limits
- list_dir: List directory contents
"""

from __future__ import annotations

import logging

from src.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_file_tools(registry: ToolRegistry) -> int:
    """Register all file tools with the registry.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered.
    """
    from src.tools.file.read import register_read_tool
    from src.tools.file.list import register_list_tool

    count = 0
    count += register_read_tool(registry)
    count += register_list_tool(registry)
    return count
