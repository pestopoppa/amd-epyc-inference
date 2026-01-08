"""Web tools for knowledge retrieval.

Tools for fetching and searching web content:
- fetch: Fetch content from a URL
- search: Web search wrapper
- doc_fetch: Documentation-specific fetching with source registry
"""

from src.tool_registry import ToolRegistry


def register_web_tools(registry: ToolRegistry) -> int:
    """Register all web tools with the registry.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered.
    """
    from src.tools.web.fetch import register_fetch_tool
    from src.tools.web.search import register_search_tool

    count = 0
    count += register_fetch_tool(registry)
    count += register_search_tool(registry)
    return count
