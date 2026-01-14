"""Web tools - HTTP requests, search, scraping."""

import json
import time
import urllib.parse
import urllib.request
from typing import Any


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


def web_search(query: str, max_results: int = 10, region: str = "wt-wt") -> list[dict]:
    """Search web using DuckDuckGo HTML (no API key needed)."""
    # Use DuckDuckGo HTML interface
    params = urllib.parse.urlencode({"q": query, "kl": region})
    url = f"https://html.duckduckgo.com/html/?{params}"

    result = http_get(url, timeout=15)
    if result.get("error") or result.get("status_code") != 200:
        return [{"error": result.get("error", "Search failed")}]

    # Parse results (simple extraction)
    import re
    results = []
    body = result["body"]

    # Extract result blocks
    pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
    matches = re.findall(pattern, body)

    for url, title in matches[:max_results]:
        # Clean URL (DuckDuckGo wraps URLs)
        if "uddg=" in url:
            url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])
        results.append({
            "title": title.strip(),
            "url": url,
            "snippet": "",  # Would need more parsing
        })

    return results


def fetch_wikipedia(title: str, sentences: int = 5) -> dict:
    """Fetch Wikipedia article summary via API."""
    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"

    result = http_get(url, timeout=10)
    if result.get("error") or result.get("status_code") != 200:
        return {"error": result.get("error", "Wikipedia fetch failed")}

    try:
        data = json.loads(result["body"])
        return {
            "title": data.get("title", title),
            "summary": data.get("extract", "")[:sentences * 200],  # Approximate
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "categories": [],  # Would need separate API call
        }
    except json.JSONDecodeError:
        return {"error": "Failed to parse Wikipedia response"}
