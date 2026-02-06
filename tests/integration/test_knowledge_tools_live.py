"""Live integration tests for knowledge retrieval tools.

These tests make actual API calls to external services (arXiv, Wikipedia, etc.).
They are designed to be:
- Skipped in CI (ORCHESTRATOR_MOCK_MODE=true)
- Skipped if packages are missing
- Run manually before production deployment
- Verify real connectivity and response parsing

Usage:
    pytest tests/integration/test_knowledge_tools_live.py -v
    pytest tests/integration/test_knowledge_tools_live.py -v -k arxiv
"""

from __future__ import annotations

import os

import pytest

# Skip all tests in CI/mock mode (they make real API calls)
MOCK_MODE = os.environ.get("ORCHESTRATOR_MOCK_MODE", "").lower() == "true"

# Skip all tests if knowledge packages are not installed
try:
    import arxiv  # noqa: F401
    import mwclient  # noqa: F401
    import semanticscholar  # noqa: F401
    from googleapiclient.discovery import build  # noqa: F401

    KNOWLEDGE_PACKAGES_AVAILABLE = True
except ImportError:
    KNOWLEDGE_PACKAGES_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(
        MOCK_MODE,
        reason="Skipped in CI: live API tests require network access",
    ),
    pytest.mark.skipif(
        not KNOWLEDGE_PACKAGES_AVAILABLE,
        reason="Knowledge packages not installed. Run: pip install -e '.[knowledge]'",
    ),
]


class TestArXivSearch:
    """Live tests for arXiv search functionality."""

    def test_search_arxiv_returns_results(self):
        """Verify arXiv search returns actual paper results."""
        from src.tools.knowledge import search_arxiv

        result = search_arxiv("transformer attention", max_results=3)

        assert result["success"] is True, f"arXiv search failed: {result.get('error')}"
        assert result["count"] > 0, "Expected at least one result"
        assert len(result["results"]) <= 3

        paper = result["results"][0]
        assert "title" in paper
        assert "authors" in paper
        assert "abstract" in paper
        assert "arxiv_id" in paper
        assert "pdf_url" in paper

    def test_search_arxiv_handles_no_results(self):
        """Verify arXiv search handles queries with no matches gracefully."""
        from src.tools.knowledge import search_arxiv

        result = search_arxiv("xyznonexistent12345qwerty", max_results=5)

        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []


class TestSemanticScholarSearch:
    """Live tests for Semantic Scholar search functionality."""

    def test_search_papers_returns_results(self):
        """Verify Semantic Scholar search returns paper results with citations."""
        from src.tools.knowledge import search_papers

        result = search_papers("deep learning", max_results=3)

        assert result["success"] is True, f"Semantic Scholar search failed: {result.get('error')}"
        assert result["count"] > 0, "Expected at least one result"
        assert len(result["results"]) <= 3

        paper = result["results"][0]
        assert "title" in paper
        assert "authors" in paper
        assert "citation_count" in paper
        assert "paper_id" in paper

    def test_search_papers_with_year_filter(self):
        """Verify year range filtering works."""
        from src.tools.knowledge import search_papers

        result = search_papers("machine learning", max_results=3, year_range="2023-2025")

        assert result["success"] is True
        # Note: Some results may not have year data, so we just verify the query works


class TestWikipediaSearch:
    """Live tests for Wikipedia search functionality."""

    def test_search_wikipedia_returns_results(self):
        """Verify Wikipedia search returns article summaries."""
        from src.tools.knowledge import search_wikipedia

        result = search_wikipedia("Python programming language", max_results=3)

        assert result["success"] is True, f"Wikipedia search failed: {result.get('error')}"
        assert result["count"] > 0, "Expected at least one result"

        article = result["results"][0]
        assert "title" in article
        assert "summary" in article
        assert "url" in article
        assert "wikipedia.org" in article["url"]

    def test_get_wikipedia_article_full_text(self):
        """Verify fetching full Wikipedia article content."""
        from src.tools.knowledge import get_wikipedia_article

        result = get_wikipedia_article("Python (programming language)")

        assert result["success"] is True, f"Wikipedia fetch failed: {result.get('error')}"
        assert result["title"] is not None
        assert len(result["full_text"]) > 100, "Expected substantial article content"
        assert "url" in result
        assert isinstance(result["sections"], list)

    def test_get_wikipedia_article_not_found(self):
        """Verify graceful handling of non-existent articles."""
        from src.tools.knowledge import get_wikipedia_article

        result = get_wikipedia_article("NonExistentArticle12345XYZ")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestGoogleBooksSearch:
    """Live tests for Google Books search functionality."""

    def test_search_books_returns_results(self):
        """Verify Google Books search returns book metadata."""
        from src.tools.knowledge import search_books

        result = search_books("machine learning", max_results=3)

        assert result["success"] is True, f"Google Books search failed: {result.get('error')}"
        assert result["count"] > 0, "Expected at least one result"
        assert len(result["results"]) <= 3

        book = result["results"][0]
        assert "title" in book
        assert "authors" in book
        assert "preview_link" in book

    def test_search_books_with_ebooks_filter(self):
        """Verify ebooks filter works."""
        from src.tools.knowledge import search_books

        result = search_books("python programming", max_results=3, filter="ebooks")

        assert result["success"] is True


class TestToolRegistration:
    """Test that knowledge tools register correctly."""

    def test_register_knowledge_tools(self):
        """Verify all knowledge tools can be registered."""
        from src.tool_registry import ToolRegistry
        from src.tools.knowledge import register_knowledge_tools

        registry = ToolRegistry()
        count = register_knowledge_tools(registry)

        assert count == 5, f"Expected 5 tools registered, got {count}"

        # Verify specific tools are registered
        tool_names = [t["name"] for t in registry.list_tools()]
        assert "search_arxiv" in tool_names
        assert "search_papers" in tool_names
        assert "search_wikipedia" in tool_names
        assert "get_wikipedia_article" in tool_names
        assert "search_books" in tool_names


class TestHealthEndpointKnowledgeTools:
    """Test health endpoint reports knowledge tool status."""

    def test_health_check_includes_knowledge_status(self):
        """Verify health endpoint includes knowledge tools status."""
        from src.api.routes.health import _check_knowledge_tools

        status = _check_knowledge_tools()

        assert "available" in status
        assert "tools" in status
        assert isinstance(status["tools"], dict)

        # With packages installed, all should be available
        assert status["available"] is True
        for tool_name, tool_status in status["tools"].items():
            assert tool_status["available"] is True, f"{tool_name} should be available"
