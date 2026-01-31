#!/usr/bin/env python3
"""Tests for knowledge retrieval tools.

All external API calls are mocked — no network access required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.tools.knowledge import (
    _strip_wikitext,
    get_wikipedia_article,
    search_arxiv,
    search_books,
    search_papers,
    search_wikipedia,
)


# ============ search_arxiv ============


class TestSearchArxiv:
    def test_success(self):
        """Returns formatted results on success."""
        mock_paper = MagicMock()
        mock_paper.title = "Attention Is All You Need"
        mock_paper.authors = [MagicMock(name="Vaswani")]
        mock_paper.summary = "We propose a new architecture..."
        mock_paper.entry_id = "http://arxiv.org/abs/1706.03762"
        mock_paper.published = MagicMock()
        mock_paper.published.isoformat.return_value = "2017-06-12T00:00:00"
        mock_paper.updated = MagicMock()
        mock_paper.updated.isoformat.return_value = "2017-06-12T00:00:00"
        mock_paper.pdf_url = "http://arxiv.org/pdf/1706.03762"
        mock_paper.categories = ["cs.CL"]

        mock_client = MagicMock()
        mock_client.results.return_value = [mock_paper]

        mock_arxiv = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        mock_arxiv.SortCriterion.Relevance = "relevance"

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            result = search_arxiv("transformer attention")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["error"] is None
        assert result["results"][0]["title"] == "Attention Is All You Need"
        assert result["results"][0]["arxiv_id"] == "1706.03762"
        assert result["results"][0]["pdf_url"] == "http://arxiv.org/pdf/1706.03762"

    def test_empty_results(self):
        """Returns empty list when no papers match."""
        mock_client = MagicMock()
        mock_client.results.return_value = []

        mock_arxiv = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        mock_arxiv.SortCriterion.Relevance = "relevance"

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            result = search_arxiv("zzzznonexistentquery")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []

    def test_api_error(self):
        """Gracefully handles API errors."""
        mock_arxiv = MagicMock()
        mock_arxiv.Client.side_effect = ConnectionError("Network unreachable")
        mock_arxiv.SortCriterion.Relevance = "relevance"

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            result = search_arxiv("test")

        assert result["success"] is False
        assert "ConnectionError" in result["error"]


# ============ search_papers ============


class TestSearchPapers:
    def test_success(self):
        """Returns formatted results on success."""
        mock_paper = MagicMock()
        mock_paper.title = "Speculative Decoding"
        mock_paper.authors = [MagicMock(name="Leviathan")]
        mock_paper.abstract = "We introduce speculative decoding..."
        mock_paper.citationCount = 150
        mock_paper.year = 2023
        mock_paper.externalIds = {"DOI": "10.1234/test"}
        mock_paper.url = "https://semanticscholar.org/paper/abc"
        mock_paper.venue = "ICML"
        mock_paper.paperId = "abc123"

        mock_ss = MagicMock()
        mock_ss.search_paper.return_value = [mock_paper]

        mock_ss_module = MagicMock()
        mock_ss_module.SemanticScholar.return_value = mock_ss

        with patch.dict(sys.modules, {"semanticscholar": mock_ss_module}):
            result = search_papers("speculative decoding")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Speculative Decoding"
        assert result["results"][0]["citation_count"] == 150
        assert result["results"][0]["doi"] == "10.1234/test"

    def test_with_year_range(self):
        """Passes year_range to API."""
        mock_ss = MagicMock()
        mock_ss.search_paper.return_value = []

        mock_ss_module = MagicMock()
        mock_ss_module.SemanticScholar.return_value = mock_ss

        with patch.dict(sys.modules, {"semanticscholar": mock_ss_module}):
            search_papers("test", year_range="2023-2025")

        mock_ss.search_paper.assert_called_once()
        _, kwargs = mock_ss.search_paper.call_args
        assert kwargs.get("year") == "2023-2025"

    def test_no_doi(self):
        """Handles papers without DOI."""
        mock_paper = MagicMock()
        mock_paper.title = "No DOI Paper"
        mock_paper.authors = []
        mock_paper.abstract = None
        mock_paper.citationCount = 0
        mock_paper.year = 2024
        mock_paper.externalIds = None
        mock_paper.url = ""
        mock_paper.venue = ""
        mock_paper.paperId = "xyz"

        mock_ss = MagicMock()
        mock_ss.search_paper.return_value = [mock_paper]

        mock_ss_module = MagicMock()
        mock_ss_module.SemanticScholar.return_value = mock_ss

        with patch.dict(sys.modules, {"semanticscholar": mock_ss_module}):
            result = search_papers("test")

        assert result["success"] is True
        assert result["results"][0]["doi"] is None

    def test_api_error(self):
        """Gracefully handles API errors."""
        mock_ss_module = MagicMock()
        mock_ss_module.SemanticScholar.side_effect = ConnectionError("timeout")

        with patch.dict(sys.modules, {"semanticscholar": mock_ss_module}):
            result = search_papers("test")

        assert result["success"] is False
        assert "ConnectionError" in result["error"]


# ============ search_wikipedia ============


class TestSearchWikipedia:
    def test_success(self):
        """Returns formatted search results."""
        mock_page = MagicMock()
        mock_page.exists = True
        mock_page.text.return_value = "'''Machine learning''' is a subset of [[artificial intelligence]]."

        mock_site = MagicMock()
        mock_site.search.return_value = [
            {"title": "Machine learning", "pageid": 12345},
        ]
        mock_site.pages.__getitem__ = MagicMock(return_value=mock_page)

        mock_mw = MagicMock()
        mock_mw.Site.return_value = mock_site

        with patch.dict(sys.modules, {"mwclient": mock_mw}):
            result = search_wikipedia("machine learning")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Machine learning"
        assert "Machine learning" in result["results"][0]["summary"]
        assert "wikipedia.org" in result["results"][0]["url"]

    def test_language_param(self):
        """Uses correct language site."""
        mock_site = MagicMock()
        mock_site.search.return_value = []

        mock_mw = MagicMock()
        mock_mw.Site.return_value = mock_site

        with patch.dict(sys.modules, {"mwclient": mock_mw}):
            search_wikipedia("test", language="de")

        mock_mw.Site.assert_called_once_with("de.wikipedia.org")

    def test_api_error(self):
        """Gracefully handles API errors."""
        mock_mw = MagicMock()
        mock_mw.Site.side_effect = ConnectionError("DNS resolution failed")

        with patch.dict(sys.modules, {"mwclient": mock_mw}):
            result = search_wikipedia("test")

        assert result["success"] is False
        assert "ConnectionError" in result["error"]


# ============ get_wikipedia_article ============


class TestGetWikipediaArticle:
    def test_success(self):
        """Returns full article text."""
        mock_category = MagicMock()
        mock_category.name = "Category:Programming languages"

        mock_page = MagicMock()
        mock_page.exists = True
        mock_page.name = "Python (programming language)"
        mock_page.text.return_value = (
            "'''Python''' is a programming language.\n\n"
            "== History ==\nCreated by Guido.\n\n"
            "== Features ==\nDynamic typing."
        )
        mock_page.categories.return_value = [mock_category]

        mock_site = MagicMock()
        mock_site.pages.__getitem__ = MagicMock(return_value=mock_page)

        mock_mw = MagicMock()
        mock_mw.Site.return_value = mock_site

        with patch.dict(sys.modules, {"mwclient": mock_mw}):
            result = get_wikipedia_article("Python (programming language)")

        assert result["success"] is True
        assert result["title"] == "Python (programming language)"
        assert "programming language" in result["full_text"]
        assert "History" in result["sections"]
        assert "Features" in result["sections"]
        assert "Programming languages" in result["categories"]

    def test_article_not_found(self):
        """Returns error for non-existent article."""
        mock_page = MagicMock()
        mock_page.exists = False

        mock_site = MagicMock()
        mock_site.pages.__getitem__ = MagicMock(return_value=mock_page)

        mock_mw = MagicMock()
        mock_mw.Site.return_value = mock_site

        with patch.dict(sys.modules, {"mwclient": mock_mw}):
            result = get_wikipedia_article("Nonexistent Article XYZ")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_api_error(self):
        """Gracefully handles API errors."""
        mock_mw = MagicMock()
        mock_mw.Site.side_effect = TimeoutError("Connection timed out")

        with patch.dict(sys.modules, {"mwclient": mock_mw}):
            result = get_wikipedia_article("Test")

        assert result["success"] is False
        assert "TimeoutError" in result["error"]


# ============ search_books ============


class TestSearchBooks:
    def test_success(self):
        """Returns formatted book results."""
        mock_service = MagicMock()
        mock_service.volumes.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "volumeInfo": {
                        "title": "Deep Learning",
                        "authors": ["Ian Goodfellow", "Yoshua Bengio"],
                        "publisher": "MIT Press",
                        "publishedDate": "2016-11-18",
                        "description": "An introduction to deep learning.",
                        "industryIdentifiers": [
                            {"type": "ISBN_13", "identifier": "9780262035613"},
                        ],
                        "pageCount": 800,
                        "previewLink": "https://books.google.com/preview",
                        "language": "en",
                    }
                }
            ]
        }

        mock_api_module = MagicMock()
        mock_api_module.build.return_value = mock_service

        with patch.dict(sys.modules, {"googleapiclient": mock_api_module, "googleapiclient.discovery": mock_api_module}):
            result = search_books("deep learning")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Deep Learning"
        assert result["results"][0]["isbn"] == "9780262035613"
        assert result["results"][0]["authors"] == ["Ian Goodfellow", "Yoshua Bengio"]

    def test_no_isbn(self):
        """Handles books without ISBN."""
        mock_service = MagicMock()
        mock_service.volumes.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "volumeInfo": {
                        "title": "Open Access Book",
                        "industryIdentifiers": [],
                    }
                }
            ]
        }

        mock_api_module = MagicMock()
        mock_api_module.build.return_value = mock_service

        with patch.dict(sys.modules, {"googleapiclient": mock_api_module, "googleapiclient.discovery": mock_api_module}):
            result = search_books("test")

        assert result["success"] is True
        assert result["results"][0]["isbn"] is None

    def test_with_filter(self):
        """Passes filter parameter to API."""
        mock_service = MagicMock()
        mock_list = mock_service.volumes.return_value.list
        mock_list.return_value.execute.return_value = {"items": []}

        mock_api_module = MagicMock()
        mock_api_module.build.return_value = mock_service

        with patch.dict(sys.modules, {"googleapiclient": mock_api_module, "googleapiclient.discovery": mock_api_module}):
            search_books("test", filter="free-ebooks")

        _, kwargs = mock_list.call_args
        assert kwargs.get("filter") == "free-ebooks"

    def test_api_error(self):
        """Gracefully handles API errors."""
        mock_api_module = MagicMock()
        mock_api_module.build.side_effect = Exception("API quota exceeded")

        with patch.dict(sys.modules, {"googleapiclient": mock_api_module, "googleapiclient.discovery": mock_api_module}):
            result = search_books("test")

        assert result["success"] is False
        assert "API quota exceeded" in result["error"]


# ============ _strip_wikitext helper ============


class TestStripWikitext:
    def test_links(self):
        """Strips wiki links, keeping display text."""
        assert "text" in _strip_wikitext("[[link|text]]")
        assert "link" not in _strip_wikitext("[[link|text]]")

    def test_plain_links(self):
        """Strips plain wiki links, keeping target."""
        assert "Python" in _strip_wikitext("[[Python]]")

    def test_bold_italic(self):
        """Strips bold and italic markers."""
        assert _strip_wikitext("'''bold''' and ''italic''") == "bold and italic"

    def test_empty(self):
        """Handles empty input."""
        assert _strip_wikitext("") == ""
        assert _strip_wikitext(None) == ""

    def test_references(self):
        """Strips reference tags."""
        text = "Some text<ref>citation</ref> continues."
        result = _strip_wikitext(text)
        assert "<ref>" not in result
        assert "continues" in result

    def test_templates(self):
        """Strips template markup."""
        text = "Text {{template|arg}} continues."
        result = _strip_wikitext(text)
        assert "{{" not in result
        assert "continues" in result
