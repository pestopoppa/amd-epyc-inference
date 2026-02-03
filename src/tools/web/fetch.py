#!/usr/bin/env python3
"""Web fetch tool for retrieving content from URLs.

Fetches web pages and extracts readable content, with support for:
- HTML to markdown conversion
- Content selectors for documentation sites
- Trust level checking via source registry
- Caching for repeated fetches
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from src.tool_registry import Tool, ToolCategory, ToolRegistry
from src.tools.base import safe_execute

logger = logging.getLogger(__name__)

# Simple cache for fetched content (URL -> content)
_fetch_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 900  # 15 minutes


def _get_project_root() -> Path:
    """Get project root from config with fallback."""
    import os

    try:
        from src.config import get_config

        root = get_config().paths.project_root
        if root.exists():
            return root
    except Exception:
        pass

    # Fallback: try hardcoded path, then cwd
    if Path("/mnt/raid0/llm/claude").exists():
        return Path("/mnt/raid0/llm/claude")
    return Path(os.getcwd())


def _load_source_registry() -> dict[str, Any]:
    """Load the source registry configuration."""
    registry_path = _get_project_root() / "orchestration" / "source_registry.yaml"
    if not registry_path.exists():
        return {}

    with open(registry_path) as f:
        return yaml.safe_load(f) or {}


def _get_domain_trust(url: str, source_registry: dict[str, Any]) -> int | None:
    """Get trust level for a domain from source registry.

    Args:
        url: URL to check.
        source_registry: Loaded source registry.

    Returns:
        Trust level (1-3) or None if not in registry.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # Remove www. prefix for matching
    if domain.startswith("www."):
        domain = domain[4:]

    # Search through all categories
    for category in ["coding", "research", "llm_specific", "general"]:
        cat_data = source_registry.get(category, {})
        for section in cat_data.values():
            if isinstance(section, list):
                for entry in section:
                    if isinstance(entry, dict) and entry.get("domain", "").lower() == domain:
                        return entry.get("trust")

    return None


def _extract_content(html: str, url: str) -> str:
    """Extract readable content from HTML.

    Args:
        html: Raw HTML content.
        url: Source URL (for site-specific extraction).

    Returns:
        Extracted text content.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: strip HTML tags with regex
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Try to find main content
    main_content = None
    for selector in ["article", "main", ".content", ".documentation", "#content", ".markdown-body"]:
        found = soup.select_one(selector)
        if found:
            main_content = found
            break

    if main_content is None:
        main_content = soup.body if soup.body else soup

    # Extract text
    text = main_content.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def _fetch_url(url: str, max_length: int = 8000, timeout: int = 30) -> str:
    """Fetch content from a URL.

    Args:
        url: URL to fetch.
        max_length: Maximum content length to return.
        timeout: Request timeout in seconds.

    Returns:
        Extracted content from the URL.

    Raises:
        Exception: If fetch fails.
    """
    import time
    import urllib.request
    from urllib.error import HTTPError, URLError

    # Check cache
    cache_key = url
    if cache_key in _fetch_cache:
        content, cached_at = _fetch_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL_SECONDS:
            logger.debug(f"Cache hit for {url}")
            return content

    # Prepare request with headers
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OrchestratorBot/1.0)",
        "Accept": "text/html,application/xhtml+xml,text/plain",
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                html = response.read().decode("utf-8", errors="replace")
                content = _extract_content(html, url)
            else:
                # Plain text or other
                content = response.read().decode("utf-8", errors="replace")

            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + f"\n\n[... truncated at {max_length} chars]"

            # Cache result
            _fetch_cache[cache_key] = (content, time.time())

            return content

    except HTTPError as e:
        raise Exception(f"HTTP error {e.code}: {e.reason}")
    except URLError as e:
        raise Exception(f"URL error: {e.reason}")


def fetch_docs(
    url: str,
    max_length: int = 8000,
    check_trust: bool = True,
) -> dict[str, Any]:
    """Fetch documentation from a URL.

    Args:
        url: URL to fetch.
        max_length: Maximum content length.
        check_trust: Whether to check source registry trust level.

    Returns:
        Dict with content, trust_level, and metadata.
    """
    # Load source registry for trust checking
    source_registry = _load_source_registry() if check_trust else {}
    trust_level = _get_domain_trust(url, source_registry)

    # Fetch content
    result = safe_execute(
        _fetch_url,
        url,
        max_length=max_length,
        timeout_seconds=30,
    )

    if not result.success:
        return {
            "success": False,
            "error": result.error,
            "url": url,
            "trust_level": trust_level,
        }

    return {
        "success": True,
        "content": result.data,
        "url": url,
        "trust_level": trust_level,
        "elapsed_ms": result.elapsed_ms,
        "truncated": result.truncated,
    }


def register_fetch_tool(registry: ToolRegistry) -> int:
    """Register the fetch_docs tool.

    Args:
        registry: ToolRegistry to register with.

    Returns:
        Number of tools registered (1).
    """
    tool = Tool(
        name="fetch_docs",
        description="Fetch documentation or content from a URL",
        category=ToolCategory.WEB,
        parameters={
            "url": {
                "type": "string",
                "description": "URL to fetch content from",
                "required": True,
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum content length (default 8000)",
                "required": False,
            },
            "check_trust": {
                "type": "boolean",
                "description": "Check source registry for trust level",
                "required": False,
            },
        },
        handler=fetch_docs,
    )

    registry.register_tool(tool)
    return 1
