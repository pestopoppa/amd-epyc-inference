#!/usr/bin/env python3
"""Tests for knowledge retrieval tools.

All external API calls are mocked — no network access required.
"""

from __future__ import annotations

import sys
from unittest.mock import patch
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from src.tools.knowledge import (
    _strip_wikitext,
    get_wikipedia_article,
    search_arxiv,
    search_books,
    search_papers,
    search_wikipedia,
)


# Stub classes for external API responses
@dataclass
class ArxivAuthor:
    """Stub arXiv author."""
    name: str


@dataclass
class ArxivPaper:
    """Stub arXiv paper result."""
    title: str
    authors: list[ArxivAuthor]
    summary: str
    entry_id: str
    published: datetime
    updated: datetime
    pdf_url: str
    categories: list[str]


@dataclass
class SemanticScholarAuthor:
    """Stub Semantic Scholar author."""
    name: str


@dataclass
class SemanticScholarPaper:
    """Stub Semantic Scholar paper result."""
    title: str
    authors: list[SemanticScholarAuthor]
    abstract: str | None
    citationCount: int
    year: int
    externalIds: dict | None
    url: str
    venue: str
    paperId: str


class StubArxivSearch:
    """Stub arXiv search object."""
    def __init__(self, query: str, max_results: int, sort_by: Any):
        self.query = query
        self.max_results = max_results
        self.sort_by = sort_by


class StubArxivClient:
    """Stub arXiv client."""
    def __init__(self, results: list = None):
        self._results = results or []

    def results(self, search: StubArxivSearch):
        """Return stub results."""
        return self._results


class StubSemanticScholar:
    """Stub Semantic Scholar client."""
    def __init__(self, results: list = None):
        self._results = results or []

    def search_paper(self, query: str, limit: int = 10, year: str | None = None,
                     fields_of_study: list | None = None, fields: list | None = None):
        """Return stub search results."""
        return self._results


class StubWikipediaPage:
    """Stub Wikipedia page."""
    def __init__(self, exists: bool = True, name: str = "", text_content: str = "",
                 categories_list: list = None):
        self.exists = exists
        self.name = name
        self._text_content = text_content
        self._categories = categories_list or []

    def text(self, section: int | None = None):
        """Return page text."""
        return self._text_content

    def categories(self):
        """Return page categories."""
        return self._categories


class StubWikipediaCategory:
    """Stub Wikipedia category."""
    def __init__(self, name: str):
        self.name = name


class StubWikipediaSite:
    """Stub Wikipedia site."""
    def __init__(self, search_results: list = None, pages_dict: dict = None):
        self._search_results = search_results or []
        self._pages_dict = pages_dict or {}

    def search(self, query: str, namespace: int = 0, limit: int = 5):
        """Return search results."""
        return self._search_results

    @property
    def pages(self):
        """Return pages dict."""
        return self._pages_dict


# ============ search_arxiv ============


class TestSearchArxiv:
    def test_success(self):
        """Returns formatted results on success."""
        paper = ArxivPaper(
            title="Attention Is All You Need",
            authors=[ArxivAuthor(name="Vaswani")],
            summary="We propose a new architecture...",
            entry_id="http://arxiv.org/abs/1706.03762",
            published=datetime(2017, 6, 12),
            updated=datetime(2017, 6, 12),
            pdf_url="http://arxiv.org/pdf/1706.03762",
            categories=["cs.CL"],
        )

        client = StubArxivClient(results=[paper])

        # Create stub arxiv module
        stub_arxiv = type('module', (), {
            'Client': lambda: client,
            'Search': StubArxivSearch,
            'SortCriterion': type('obj', (), {'Relevance': 'relevance'}),
        })

        with patch.dict(sys.modules, {"arxiv": stub_arxiv}):
            result = search_arxiv("transformer attention")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["error"] is None
        assert result["results"][0]["title"] == "Attention Is All You Need"
        assert result["results"][0]["arxiv_id"] == "1706.03762"
        assert result["results"][0]["pdf_url"] == "http://arxiv.org/pdf/1706.03762"

    def test_empty_results(self):
        """Returns empty list when no papers match."""
        client = StubArxivClient(results=[])

        stub_arxiv = type('module', (), {
            'Client': lambda: client,
            'Search': StubArxivSearch,
            'SortCriterion': type('obj', (), {'Relevance': 'relevance'}),
        })

        with patch.dict(sys.modules, {"arxiv": stub_arxiv}):
            result = search_arxiv("zzzznonexistentquery")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []

    def test_api_error(self):
        """Gracefully handles API errors."""
        def failing_client():
            raise ConnectionError("Network unreachable")

        stub_arxiv = type('module', (), {
            'Client': failing_client,
            'SortCriterion': type('obj', (), {'Relevance': 'relevance'}),
        })

        with patch.dict(sys.modules, {"arxiv": stub_arxiv}):
            result = search_arxiv("test")

        assert result["success"] is False
        assert "ConnectionError" in result["error"]


# ============ search_papers ============


class TestSearchPapers:
    def test_success(self):
        """Returns formatted results on success."""
        paper = SemanticScholarPaper(
            title="Speculative Decoding",
            authors=[SemanticScholarAuthor(name="Leviathan")],
            abstract="We introduce speculative decoding...",
            citationCount=150,
            year=2023,
            externalIds={"DOI": "10.1234/test"},
            url="https://semanticscholar.org/paper/abc",
            venue="ICML",
            paperId="abc123",
        )

        ss = StubSemanticScholar(results=[paper])

        stub_ss_module = type('module', (), {
            'SemanticScholar': lambda: ss,
        })

        with patch.dict(sys.modules, {"semanticscholar": stub_ss_module}):
            result = search_papers("speculative decoding")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Speculative Decoding"
        assert result["results"][0]["citation_count"] == 150
        assert result["results"][0]["doi"] == "10.1234/test"

    def test_with_year_range(self):
        """Passes year_range to API."""
        call_args = {}

        class TrackingSemanticScholar:
            def search_paper(self, query, limit=10, year=None, fields_of_study=None, fields=None):
                call_args["year"] = year
                return []

        stub_ss_module = type('module', (), {
            'SemanticScholar': TrackingSemanticScholar,
        })

        with patch.dict(sys.modules, {"semanticscholar": stub_ss_module}):
            search_papers("test", year_range="2023-2025")

        assert call_args.get("year") == "2023-2025"

    def test_no_doi(self):
        """Handles papers without DOI."""
        paper = SemanticScholarPaper(
            title="No DOI Paper",
            authors=[],
            abstract=None,
            citationCount=0,
            year=2024,
            externalIds=None,
            url="",
            venue="",
            paperId="xyz",
        )

        ss = StubSemanticScholar(results=[paper])

        stub_ss_module = type('module', (), {
            'SemanticScholar': lambda: ss,
        })

        with patch.dict(sys.modules, {"semanticscholar": stub_ss_module}):
            result = search_papers("test")

        assert result["success"] is True
        assert result["results"][0]["doi"] is None

    def test_api_error(self):
        """Gracefully handles API errors."""
        def failing_init():
            raise ConnectionError("timeout")

        stub_ss_module = type('module', (), {
            'SemanticScholar': failing_init,
        })

        with patch.dict(sys.modules, {"semanticscholar": stub_ss_module}):
            result = search_papers("test")

        assert result["success"] is False
        assert "ConnectionError" in result["error"]


# ============ search_wikipedia ============


class TestSearchWikipedia:
    def test_success(self):
        """Returns formatted search results."""
        page = StubWikipediaPage(
            exists=True,
            name="Machine learning",
            text_content="'''Machine learning''' is a subset of [[artificial intelligence]].",
        )

        pages_dict = {"Machine learning": page}
        site = StubWikipediaSite(
            search_results=[{"title": "Machine learning", "pageid": 12345}],
            pages_dict=pages_dict,
        )

        stub_mw = type('module', (), {
            'Site': lambda host: site,
        })

        with patch.dict(sys.modules, {"mwclient": stub_mw}):
            result = search_wikipedia("machine learning")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Machine learning"
        assert "Machine learning" in result["results"][0]["summary"]
        assert "wikipedia.org" in result["results"][0]["url"]

    def test_language_param(self):
        """Uses correct language site."""
        call_args = {}

        def tracking_site(host):
            call_args["host"] = host
            return StubWikipediaSite(search_results=[])

        stub_mw = type('module', (), {
            'Site': tracking_site,
        })

        with patch.dict(sys.modules, {"mwclient": stub_mw}):
            search_wikipedia("test", language="de")

        assert call_args["host"] == "de.wikipedia.org"

    def test_api_error(self):
        """Gracefully handles API errors."""
        def failing_site(host):
            raise ConnectionError("DNS resolution failed")

        stub_mw = type('module', (), {
            'Site': failing_site,
        })

        with patch.dict(sys.modules, {"mwclient": stub_mw}):
            result = search_wikipedia("test")

        assert result["success"] is False
        assert "ConnectionError" in result["error"]


# ============ get_wikipedia_article ============


class TestGetWikipediaArticle:
    def test_success(self):
        """Returns full article text."""
        category = StubWikipediaCategory(name="Category:Programming languages")
        page = StubWikipediaPage(
            exists=True,
            name="Python (programming language)",
            text_content=(
                "'''Python''' is a programming language.\n\n"
                "== History ==\nCreated by Guido.\n\n"
                "== Features ==\nDynamic typing."
            ),
            categories_list=[category],
        )

        pages_dict = {"Python (programming language)": page}
        site = StubWikipediaSite(pages_dict=pages_dict)

        stub_mw = type('module', (), {
            'Site': lambda host: site,
        })

        with patch.dict(sys.modules, {"mwclient": stub_mw}):
            result = get_wikipedia_article("Python (programming language)")

        assert result["success"] is True
        assert result["title"] == "Python (programming language)"
        assert "programming language" in result["full_text"]
        assert "History" in result["sections"]
        assert "Features" in result["sections"]
        assert "Programming languages" in result["categories"]

    def test_article_not_found(self):
        """Returns error for non-existent article."""
        page = StubWikipediaPage(exists=False)

        pages_dict = {"Nonexistent Article XYZ": page}
        site = StubWikipediaSite(pages_dict=pages_dict)

        stub_mw = type('module', (), {
            'Site': lambda host: site,
        })

        with patch.dict(sys.modules, {"mwclient": stub_mw}):
            result = get_wikipedia_article("Nonexistent Article XYZ")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_api_error(self):
        """Gracefully handles API errors."""
        def failing_site(host):
            raise TimeoutError("Connection timed out")

        stub_mw = type('module', (), {
            'Site': failing_site,
        })

        with patch.dict(sys.modules, {"mwclient": stub_mw}):
            result = get_wikipedia_article("Test")

        assert result["success"] is False
        assert "TimeoutError" in result["error"]


# ============ search_books ============


class TestSearchBooks:
    def test_success(self):
        """Returns formatted book results."""
        response_data = {
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

        class StubVolumes:
            def list(self, **kwargs):
                self.list_kwargs = kwargs
                return self

            def execute(self):
                return response_data

        class StubService:
            def volumes(self):
                return StubVolumes()

        stub_api_module = type('module', (), {
            'build': lambda service, version, developerKey: StubService(),
            'discovery': type('obj', (), {}),
        })

        with patch.dict(sys.modules, {
            "googleapiclient": stub_api_module,
            "googleapiclient.discovery": stub_api_module,
        }):
            result = search_books("deep learning")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Deep Learning"
        assert result["results"][0]["isbn"] == "9780262035613"
        assert result["results"][0]["authors"] == ["Ian Goodfellow", "Yoshua Bengio"]

    def test_no_isbn(self):
        """Handles books without ISBN."""
        response_data = {
            "items": [
                {
                    "volumeInfo": {
                        "title": "Open Access Book",
                        "industryIdentifiers": [],
                    }
                }
            ]
        }

        class StubVolumes:
            def list(self, **kwargs):
                return self

            def execute(self):
                return response_data

        class StubService:
            def volumes(self):
                return StubVolumes()

        stub_api_module = type('module', (), {
            'build': lambda service, version, developerKey: StubService(),
            'discovery': type('obj', (), {}),
        })

        with patch.dict(sys.modules, {
            "googleapiclient": stub_api_module,
            "googleapiclient.discovery": stub_api_module,
        }):
            result = search_books("test")

        assert result["success"] is True
        assert result["results"][0]["isbn"] is None

    def test_with_filter(self):
        """Passes filter parameter to API."""
        call_kwargs = {}

        class StubVolumes:
            def list(self, **kwargs):
                call_kwargs.update(kwargs)
                return self

            def execute(self):
                return {"items": []}

        class StubService:
            def volumes(self):
                return StubVolumes()

        stub_api_module = type('module', (), {
            'build': lambda service, version, developerKey: StubService(),
            'discovery': type('obj', (), {}),
        })

        with patch.dict(sys.modules, {
            "googleapiclient": stub_api_module,
            "googleapiclient.discovery": stub_api_module,
        }):
            search_books("test", filter="free-ebooks")

        assert call_kwargs.get("filter") == "free-ebooks"

    def test_api_error(self):
        """Gracefully handles API errors."""
        def failing_build(service, version, developerKey):
            raise Exception("API quota exceeded")

        stub_api_module = type('module', (), {
            'build': failing_build,
            'discovery': type('obj', (), {}),
        })

        with patch.dict(sys.modules, {
            "googleapiclient": stub_api_module,
            "googleapiclient.discovery": stub_api_module,
        }):
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
