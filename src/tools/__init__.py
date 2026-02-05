"""Tools package for orchestrator models.

This package provides tools that orchestrator models can invoke:
- web/     Web/knowledge retrieval (fetch, search, crawl)
- file/    File operations (read, write, list)
- code/    Code execution (run_tests, lint, format)
- data/    Data processing (json_ops, yaml_ops, text)

Usage:
    from src.tools import register_all_tools
    from src.tool_registry import get_registry

    registry = get_registry()
    register_all_tools(registry)
"""

from __future__ import annotations

import logging

from src.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_all_tools(registry: ToolRegistry) -> int:
    """Register all built-in tools with a registry.

    Args:
        registry: ToolRegistry to register tools with.

    Returns:
        Number of tools registered.
    """
    count = 0

    # Import and register web tools
    try:
        from src.tools.web import register_web_tools

        count += register_web_tools(registry)
    except ImportError as e:
        logger.debug("Web tools not available: %s", e)

    # Import and register file tools
    try:
        from src.tools.file import register_file_tools

        count += register_file_tools(registry)
    except ImportError as e:
        logger.debug("File tools not available: %s", e)

    # Import and register code tools
    try:
        from src.tools.code import register_code_tools

        count += register_code_tools(registry)
    except ImportError as e:
        logger.debug("Code tools not available: %s", e)

    # Import and register data tools
    try:
        from src.tools.data import register_data_tools

        count += register_data_tools(registry)
    except ImportError as e:
        logger.debug("Data tools not available: %s", e)

    # Import and register knowledge retrieval tools
    try:
        from src.tools.knowledge import register_knowledge_tools

        count += register_knowledge_tools(registry)
    except ImportError as e:
        logger.debug("Knowledge tools not available: %s", e)

    return count
