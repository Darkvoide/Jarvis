"""Web search tool — DuckDuckGo via ddgs, no API key required."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 5) -> dict:
    """Search the web for a query and return the top results.

    Uses DuckDuckGo's search API — no API key or account required.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return. Default is 5.

    Returns:
        A dict with:
          - results: list of dicts, each with keys: title, href, body.
          - query: the original query.
          - count: number of results returned.
    """
    logger.info("search_web: %s", query)
    try:
        try:
            from ddgs import DDGS  # new package name
        except ImportError:
            from duckduckgo_search import DDGS  # fallback to old name

        with DDGS() as ddgs_client:
            raw_results = list(ddgs_client.text(query, max_results=max(1, min(max_results, 10))))

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw_results
        ]
        return {"query": query, "results": results, "count": len(results)}

    except ImportError:
        return {"error": "ddgs is not installed. Run: pip install ddgs"}
    except Exception as exc:
        logger.error("search_web error: %s", exc)
        return {"error": f"Web search failed: {exc}"}
