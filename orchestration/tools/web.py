"""Web tools - HTTP requests, search, scraping."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def http_get(url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    """Fetch content from URL via HTTP GET."""
    start = time.time()
    req_headers = {"User-Agent": "OrchestrationBot/1.0"}
    if headers:
        req_headers.update(headers)

    request = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "status_code": response.status,
                "headers": dict(response.headers),
                "body": body[:100000],  # Limit response size
                "elapsed_ms": int((time.time() - start) * 1000),
            }
    except urllib.error.HTTPError as e:
        return {
            "status_code": e.code,
            "headers": dict(e.headers) if e.headers else {},
            "body": e.read().decode("utf-8", errors="replace")[:10000],
            "elapsed_ms": int((time.time() - start) * 1000),
            "error": str(e),
        }
    except Exception as e:
        return {
            "status_code": 0,
            "headers": {},
            "body": "",
            "elapsed_ms": int((time.time() - start) * 1000),
            "error": str(e),
        }


def http_post(url: str, data: dict | None = None, headers: dict | None = None,
              timeout: int = 30) -> dict:
    """Send data to URL via HTTP POST."""
    start = time.time()
    req_headers = {
        "User-Agent": "OrchestrationBot/1.0",
        "Content-Type": "application/json",
    }
    if headers:
        req_headers.update(headers)

    body_bytes = json.dumps(data).encode("utf-8") if data else None
    request = urllib.request.Request(url, data=body_bytes, headers=req_headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "status_code": response.status,
                "headers": dict(response.headers),
                "body": body[:100000],
                "elapsed_ms": int((time.time() - start) * 1000),
            }
    except Exception as e:
        return {
            "status_code": 0,
            "headers": {},
            "body": "",
            "elapsed_ms": int((time.time() - start) * 1000),
            "error": str(e),
        }


def _classify_error(error: str) -> str:
    """Classify an HTTP error into a typed category."""
    err_lower = error.lower()
    if "timed out" in err_lower or "timeout" in err_lower:
        return "timeout"
    if "429" in err_lower or "rate" in err_lower:
        return "rate_limit"
    if any(k in err_lower for k in ("connection", "refused", "unreachable", "dns", "name resolution")):
        return "network"
    return "parse_error"


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#x27;", "'").replace("&nbsp;", " ")
    return text.strip()


def _fallback_wikipedia_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Wikipedia via opensearch API as fallback when DDG fails."""
    params = urllib.parse.urlencode({
        "action": "opensearch",
        "search": query,
        "limit": max_results,
        "namespace": "0",
        "format": "json",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"

    result = http_get(url, timeout=10)
    if result.get("error") or result.get("status_code") != 200:
        return [{"error": f"wikipedia_fallback_failed: {result.get('error', 'unknown')}",
                 "error_type": "network"}]

    try:
        data = json.loads(result["body"])
        # opensearch returns [query, [titles], [snippets], [urls]]
        if len(data) < 4:
            return [{"error": "wikipedia_empty_response", "error_type": "parse_error"}]
        titles, snippets, urls = data[1], data[2], data[3]
        results = []
        for title, snippet, page_url in zip(titles, snippets, urls):
            results.append({
                "title": title,
                "url": page_url,
                "snippet": snippet,
                "source": "wikipedia",
            })
        return results if results else [{"error": "wikipedia_no_results", "error_type": "parse_error"}]
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        return [{"error": f"wikipedia_parse_failed: {e}", "error_type": "parse_error"}]


def web_search(query: str, max_results: int = 10, region: str = "wt-wt") -> list[dict]:
    """Search web using DuckDuckGo HTML with retry + Wikipedia fallback."""
    params = urllib.parse.urlencode({"q": query, "kl": region})
    url = f"https://html.duckduckgo.com/html/?{params}"

    last_error = ""
    for attempt in range(2):
        if attempt > 0:
            time.sleep(1)

        result = http_get(url, timeout=15)
        if result.get("error") or result.get("status_code") != 200:
            last_error = result.get("error", "unknown")
            logger.warning("web_search attempt %d failed: %s", attempt + 1, last_error)
            continue

        body = result["body"]

        # Extract title+URL pairs
        title_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>'
        title_matches = re.findall(title_pattern, body, re.DOTALL)

        # Extract snippets
        snippet_pattern = r'<a class="result__snippet"[^>]*>(.*?)</a>'
        snippet_matches = re.findall(snippet_pattern, body, re.DOTALL)

        results = []
        for idx, (raw_url, raw_title) in enumerate(title_matches[:max_results]):
            # Clean URL (DuckDuckGo wraps URLs)
            clean_url = raw_url
            if "uddg=" in raw_url:
                clean_url = urllib.parse.unquote(raw_url.split("uddg=")[1].split("&")[0])

            snippet = ""
            if idx < len(snippet_matches):
                snippet = _strip_html_tags(snippet_matches[idx])

            results.append({
                "title": _strip_html_tags(raw_title),
                "url": clean_url,
                "snippet": snippet,
            })

        if results:
            return results

        last_error = "no_results_parsed"
        logger.warning("web_search attempt %d: parsed 0 results from DDG", attempt + 1)

    # Both DDG attempts failed — fall back to Wikipedia
    logger.info("web_search: DDG failed (%s), falling back to Wikipedia for: %s",
                last_error, query)
    wiki_results = _fallback_wikipedia_search(query, max_results=max_results)

    # If Wikipedia also failed, return typed error
    if wiki_results and "error" in wiki_results[0]:
        error_type = _classify_error(last_error)
        return [{"error": f"search_failed: ddg={last_error}, wiki={wiki_results[0]['error']}",
                 "error_type": error_type}]

    return wiki_results


def fetch_wikipedia(title: str, sentences: int = 5) -> dict:
    """Fetch Wikipedia article summary via API."""
    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
    url = (
        f"https://en.wikipedia.org/w/api.php?action=query&titles={encoded_title}"
        f"&prop=extracts&exintro=1&exsentences={sentences}"
        f"&explaintext=1&format=json&redirects=1"
    )

    result = http_get(url, timeout=10)
    if result.get("error") or result.get("status_code") != 200:
        return {"error": result.get("error", "Wikipedia fetch failed")}

    try:
        data = json.loads(result["body"])
        pages = data.get("query", {}).get("pages", {})
        # Get first page (API returns pages keyed by page ID)
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return {"error": f"Wikipedia page not found: {title}"}
            return {
                "title": page_data.get("title", title),
                "summary": page_data.get("extract", ""),
                "url": f"https://en.wikipedia.org/wiki/{encoded_title}",
                "categories": [],
            }
        return {"error": "Wikipedia returned empty pages"}
    except json.JSONDecodeError:
        return {"error": "Failed to parse Wikipedia response"}
