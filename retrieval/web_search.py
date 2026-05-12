"""
retrieval/web_search.py — DuckDuckGo web search fallback.

Used when local retrieval returns insufficient results.
No API key required.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from retrieval.vector_store import ChunkMetadata, RetrievedChunk
from observability.logger import get_logger

logger = get_logger(__name__)

# Rate limiting
_last_search_time: float = 0.0
_MIN_INTERVAL_SECONDS: float = 1.0


class WebSearcher:
    """
    DuckDuckGo-based web search for supplementing local retrieval.

    Features:
    - Rate limiting to respect API limits
    - Result deduplication
    - Async-compatible
    """

    def __init__(self, max_results: int = 5):
        self._max_results = max_results

    def search(self, query: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        """
        Search the web and return results as RetrievedChunk objects.

        Args:
            query: Search query.
            k: Max results (defaults to self._max_results).

        Returns:
            List of RetrievedChunk from web sources.
        """
        global _last_search_time
        k = k or self._max_results

        # Rate limiting
        now = time.time()
        elapsed = now - _last_search_time
        if elapsed < _MIN_INTERVAL_SECONDS:
            time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
        _last_search_time = time.time()

        try:
            from duckduckgo_search import DDGS

            results = []
            seen_urls: set[str] = set()

            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=k):
                    url = r.get("href", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = r.get("title", "")
                    body = r.get("body", "")
                    text = f"{title}\n{body}" if title else body

                    results.append(RetrievedChunk(
                        text=text.strip(),
                        score=0.4,  # Web results get lower base score
                        source=url,
                        retrieval_method="web",
                        metadata=ChunkMetadata(
                            source=url,
                            extra={"title": title},
                        ),
                    ))

            logger.info("Web search completed", query=query, results=len(results))
            return results

        except Exception as exc:
            logger.warning("Web search failed", error=str(exc))
            return []

    async def async_search(
        self, query: str, k: Optional[int] = None
    ) -> list[RetrievedChunk]:
        """Async wrapper for web search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, k)
