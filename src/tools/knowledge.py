#!/usr/bin/env python3
"""Knowledge retrieval tools for academic papers, Wikipedia, and books.

Provides search access to free public APIs:
- arXiv (academic preprints)
- Semantic Scholar (academic papers with citation data)
- Wikipedia (encyclopedia articles)
- Google Books (book metadata)

All APIs are free and require no API keys.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def search_arxiv(query: str, max_results: int = 10) -> dict[str, Any]:
    """Search arXiv for academic papers.

    Args:
        query: Search query (supports arXiv query syntax).
        max_results: Maximum number of results to return.

    Returns:
        Dict with success status and list of paper results.
    """
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary,
                "arxiv_id": paper.entry_id.split("/abs/")[-1],
                "published": paper.published.isoformat() if paper.published else None,
                "updated": paper.updated.isoformat() if paper.updated else None,
                "pdf_url": paper.pdf_url,
                "categories": paper.categories,
            })

        return {"success": True, "results": results, "count": len(results), "error": None}

    except Exception as e:
        logger.exception(f"arXiv search failed: {e}")
        return {"success": False, "results": [], "count": 0, "error": f"{type(e).__name__}: {e}"}


def search_papers(
    query: str,
    max_results: int = 10,
    year_range: str | None = None,
    fields_of_study: list[str] | None = None,
) -> dict[str, Any]:
    """Search Semantic Scholar for academic papers with citation data.

    Args:
        query: Search query.
        max_results: Maximum number of results.
        year_range: Optional year range filter (e.g., "2020-2025").
        fields_of_study: Optional list of fields (e.g., ["Computer Science"]).

    Returns:
        Dict with success status and list of paper results.
    """
    try:
        from semanticscholar import SemanticScholar

        sch = SemanticScholar()
        results_raw = sch.search_paper(
            query,
            limit=max_results,
            year=year_range,
            fields_of_study=fields_of_study,
            fields=[
                "title", "authors", "abstract", "citationCount",
                "year", "externalIds", "url", "venue",
            ],
        )

        results = []
        for paper in results_raw:
            doi = None
            if paper.externalIds:
                doi = paper.externalIds.get("DOI")

            results.append({
                "title": paper.title,
                "authors": [a.name for a in (paper.authors or [])],
                "abstract": paper.abstract,
                "citation_count": paper.citationCount,
                "year": paper.year,
                "doi": doi,
                "url": paper.url,
                "venue": paper.venue,
                "paper_id": paper.paperId,
            })

        return {"success": True, "results": results, "count": len(results), "error": None}

    except Exception as e:
        logger.exception(f"Semantic Scholar search failed: {e}")
        return {"success": False, "results": [], "count": 0, "error": f"{type(e).__name__}: {e}"}


def search_wikipedia(
    query: str,
    max_results: int = 5,
    language: str = "en",
) -> dict[str, Any]:
    """Search Wikipedia for articles matching a query.

    Args:
        query: Search query.
        max_results: Maximum number of results.
        language: Wikipedia language code (default "en").

    Returns:
        Dict with success status and list of article summaries.
    """
    try:
        import mwclient

        site = mwclient.Site(f"{language}.wikipedia.org")
        search_results = site.search(query, namespace=0, limit=max_results)

        results = []
        for result in search_results:
            title = result.get("title", "")
            page = site.pages[title]

            # Get summary (first section)
            text = page.text(section=0) if page.exists else ""
            # Strip wikitext markup (basic cleanup)
            summary = _strip_wikitext(text)

            results.append({
                "title": title,
                "summary": summary[:500] if summary else "",
                "url": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "page_id": result.get("pageid"),
            })

        return {"success": True, "results": results, "count": len(results), "error": None}

    except Exception as e:
        logger.exception(f"Wikipedia search failed: {e}")
        return {"success": False, "results": [], "count": 0, "error": f"{type(e).__name__}: {e}"}


def get_wikipedia_article(
    title: str,
    language: str = "en",
) -> dict[str, Any]:
    """Fetch the full text of a Wikipedia article by title.

    Args:
        title: Exact article title.
        language: Wikipedia language code (default "en").

    Returns:
        Dict with success status and article content.
    """
    try:
        import mwclient

        site = mwclient.Site(f"{language}.wikipedia.org")
        page = site.pages[title]

        if not page.exists:
            return {
                "success": False,
                "error": f"Article not found: {title}",
                "title": title,
                "full_text": "",
                "url": "",
                "categories": [],
                "sections": [],
            }

        full_text = page.text()
        clean_text = _strip_wikitext(full_text)

        # Extract section headings
        sections = []
        for line in full_text.split("\n"):
            line = line.strip()
            if line.startswith("==") and line.endswith("=="):
                heading = line.strip("= ")
                if heading:
                    sections.append(heading)

        # Get categories
        categories = [cat.name.replace("Category:", "") for cat in page.categories()]

        url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"

        return {
            "success": True,
            "title": page.name,
            "full_text": clean_text,
            "url": url,
            "categories": categories[:20],
            "sections": sections,
            "error": None,
        }

    except Exception as e:
        logger.exception(f"Wikipedia article fetch failed: {e}")
        return {
            "success": False,
            "title": title,
            "full_text": "",
            "url": "",
            "categories": [],
            "sections": [],
            "error": f"{type(e).__name__}: {e}",
        }


def search_books(
    query: str,
    max_results: int = 10,
    filter: str | None = None,
) -> dict[str, Any]:
    """Search Google Books for publications.

    Args:
        query: Search query.
        max_results: Maximum number of results.
        filter: Optional filter: "ebooks", "free-ebooks", "paid-ebooks",
                "partial", "full".

    Returns:
        Dict with success status and list of book results.
    """
    try:
        from googleapiclient.discovery import build

        service = build("books", "v1", developerKey=None)

        kwargs: dict[str, Any] = {
            "q": query,
            "maxResults": min(max_results, 40),
        }
        if filter:
            kwargs["filter"] = filter

        response = service.volumes().list(**kwargs).execute()

        results = []
        for item in response.get("items", []):
            info = item.get("volumeInfo", {})
            identifiers = info.get("industryIdentifiers", [])
            isbn = None
            for ident in identifiers:
                if ident.get("type") in ("ISBN_13", "ISBN_10"):
                    isbn = ident.get("identifier")
                    break

            results.append({
                "title": info.get("title", ""),
                "authors": info.get("authors", []),
                "publisher": info.get("publisher", ""),
                "published_date": info.get("publishedDate", ""),
                "description": (info.get("description", "") or "")[:300],
                "isbn": isbn,
                "page_count": info.get("pageCount"),
                "preview_link": info.get("previewLink", ""),
                "language": info.get("language", ""),
            })

        return {"success": True, "results": results, "count": len(results), "error": None}

    except Exception as e:
        logger.exception(f"Google Books search failed: {e}")
        return {"success": False, "results": [], "count": 0, "error": f"{type(e).__name__}: {e}"}


def _strip_wikitext(text: str) -> str:
    """Strip common wikitext markup to produce plain text.

    This is a lightweight cleanup — not a full parser.
    Handles: links, bold/italic, templates, HTML tags, references.
    """
    import re

    if not text:
        return ""

    # Remove templates {{...}} (non-greedy, handles simple nesting)
    result = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove references <ref>...</ref> and <ref ... />
    result = re.sub(r"<ref[^>]*>.*?</ref>", "", result, flags=re.DOTALL)
    result = re.sub(r"<ref[^/]*/\s*>", "", result)
    # Remove HTML tags
    result = re.sub(r"<[^>]+>", "", result)
    # Convert [[link|text]] to text, [[link]] to link
    result = re.sub(r"\[\[[^|\]]*\|([^\]]+)\]\]", r"\1", result)
    result = re.sub(r"\[\[([^\]]+)\]\]", r"\1", result)
    # Remove external links [url text] -> text
    result = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", result)
    result = re.sub(r"\[https?://\S+\]", "", result)
    # Remove bold/italic markers
    result = result.replace("'''", "").replace("''", "")
    # Collapse whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r" {2,}", " ", result)

    return result.strip()
