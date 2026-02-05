"""Data tools for data processing.

Tools for JSON and text operations:
- json_parse: Parse and validate JSON
- text_transform: Text manipulation utilities
"""

from __future__ import annotations

import json as json_module
import logging

from src.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def register_data_tools(registry: ToolRegistry) -> int:
    """Register all data tools with the registry.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered.
    """
    # Data tools are simpler - register inline
    from src.tool_registry import Tool, ToolCategory

    def json_parse(
        content: str,
        extract_path: str | None = None,
    ) -> dict:
        """Parse JSON content.

        Args:
            content: JSON string to parse.
            extract_path: Optional JSONPath-like path to extract.

        Returns:
            Dict with parsed data.
        """
        try:
            data = json_module.loads(content)

            if extract_path:
                # Simple path extraction (supports dot notation)
                parts = extract_path.split(".")
                result = data
                for part in parts:
                    if isinstance(result, dict):
                        result = result.get(part)
                    elif isinstance(result, list) and part.isdigit():
                        result = result[int(part)]
                    else:
                        return {
                            "success": False,
                            "error": f"Cannot extract '{part}' from {type(result).__name__}",
                        }
                data = result

            return {
                "success": True,
                "data": data,
                "type": type(data).__name__,
            }

        except json_module.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON parse error: {e}",
            }

    tool = Tool(
        name="json_parse",
        description="Parse and validate JSON content",
        category=ToolCategory.DATA,
        parameters={
            "content": {
                "type": "string",
                "description": "JSON string to parse",
                "required": True,
            },
            "extract_path": {
                "type": "string",
                "description": "Path to extract (dot notation)",
                "required": False,
            },
        },
        handler=json_parse,
    )

    registry.register_tool(tool)
    return 1
