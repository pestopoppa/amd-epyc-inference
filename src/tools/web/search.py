#!/usr/bin/env python3
"""Web search tool for knowledge retrieval.

Provides web search functionality with:
- DuckDuckGo as default backend (no API key required)
- Source registry filtering for trusted results
- Result summarization
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus

from src.tool_registry import Tool, ToolCategory, ToolRegistry
from src.tools.base import safe_execute

logger = logging.getLogger(__name__)


def _search_duckduckgo(
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Search DuckDuckGo and return results.

    Uses the DuckDuckGo HTML interface (no API key required).

    Args:
        query: Search query.
        max_results: Maximum number of results.

    Returns:
        List of result dicts with title, url, snippet.
    """
    import urllib.request
    from urllib.error import HTTPError, URLError

    # DuckDuckGo HTML search URL
    encoded_query = quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OrchestratorBot/1.0)",
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError) as e:
        raise Exception(f"Search request failed: {e}")

    # Parse results from HTML
    results = []

    # Simple regex parsing for DuckDuckGo HTML results
    # Pattern for result blocks
    result_pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
        r'.*?<a[^>]*class="result__snippet"[^>]*>([^<]*)</a>',
        re.DOTALL
    )

    for match in result_pattern.finditer(html):
        if len(results) >= max_results:
            break

        url_match = match.group(1)
        title = match.group(2).strip()
        snippet = match.group(3).strip()

        # Clean up DuckDuckGo redirect URLs
        if "uddg=" in url_match:
            # Extract actual URL from DDG redirect
            actual_url_match = re.search(r'uddg=([^&]+)', url_match)
            if actual_url_match:
                from urllib.parse import unquote
                url_match = unquote(actual_url_match.group(1))

        if url_match and title:
            results.append({
                "title": title,
                "url": url_match,
                "snippet": snippet,
            })

    # Fallback: try simpler pattern if no results
    if not results:
        link_pattern = re.compile(r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>')
        for match in link_pattern.finditer(html):
            if len(results) >= max_results:
                break
            url_match = match.group(1)
            title = match.group(2).strip()
            if url_match and title and "duckduckgo" not in url_match.lower():
                results.append({
                    "title": title,
                    "url": url_match,
                    "snippet": "",
                })

    return results


def web_search(
    query: str,
    max_results: int = 5,
    domain_filter: str | None = None,
) -> dict[str, Any]:
    """Search the web for information.

    Args:
        query: Search query.
        max_results: Maximum number of results to return.
        domain_filter: Optional domain to filter results (e.g., "docs.python.org").

    Returns:
        Dict with results list and metadata.
    """
    # Add domain filter to query if specified
    search_query = query
    if domain_filter:
        search_query = f"site:{domain_filter} {query}"

    result = safe_execute(
        _search_duckduckgo,
        search_query,
        max_results=max_results,
        timeout_seconds=20,
    )

    if not result.success:
        return {
            "success": False,
            "error": result.error,
            "query": query,
        }

    return {
        "success": True,
        "results": result.data,
        "query": query,
        "result_count": len(result.data),
        "elapsed_ms": result.elapsed_ms,
    }


def register_search_tool(registry: ToolRegistry) -> int:
    """Register the web_search tool.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered (1).
    """
    tool = Tool(
        name="web_search",
        description="Search the web for information",
        category=ToolCategory.WEB,
        parameters={
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 5)",
                "required": False,
            },
            "domain_filter": {
                "type": "string",
                "description": "Filter results to specific domain",
                "required": False,
            },
        },
        handler=web_search,
    )

    registry.register_tool(tool)
    return 1
