"""
retrieval/hybrid.py — Multi-source retrieval with Reciprocal Rank Fusion (RRF).

Merges results from vector search, graph traversal, and web search
into a unified ranked list using RRF.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from retrieval.vector_store import RetrievedChunk, VectorStore
from retrieval.graph_store import GraphStore
from retrieval.web_search import WebSearcher
from retrieval.reranker import Reranker
from observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for the hybrid retrieval step."""
    top_k: int = 5
    vector_weight: float = 1.0
    graph_weight: float = 0.7
    web_weight: float = 0.5
    use_reranker: bool = True
    use_web_search: bool = False  
    rrf_k: int = 60  


class HybridRetriever:
    """
    Multi-source retriever combining vector, graph, and web search
    with Reciprocal Rank Fusion (RRF) for merging.

    RRF formula: score(d) = Σ (weight_i / (rrf_k + rank_i(d)))

    This avoids the pitfalls of raw score combination where different
    retrieval methods have incompatible score distributions.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: Optional[GraphStore] = None,
        web_searcher: Optional[WebSearcher] = None,
        reranker: Optional[Reranker] = None,
    ):
        self._vector = vector_store
        self._graph = graph_store
        self._web = web_searcher
        self._reranker = reranker

    async def retrieve(
        self, query: str, config: Optional[RetrievalConfig] = None
    ) -> list[RetrievedChunk]:
        """
        Retrieve and merge results from all available sources.

        Args:
            query: User query.
            config: Retrieval configuration.

        Returns:
            Merged, deduplicated, and ranked list of chunks.
        """
        config = config or RetrievalConfig()
        fetch_k = config.top_k * 3 

     
        tasks = []

      
        if self._vector.is_built:
            tasks.append(("vector", self._vector.async_search(query, k=fetch_k)))
        else:
            tasks.append(("vector", asyncio.coroutine(lambda: [])()))

      
        if self._graph and self._graph.is_available:
            loop = asyncio.get_event_loop()
            tasks.append(("graph", loop.run_in_executor(
                None, self._graph.search, query, fetch_k
            )))

   
        if config.use_web_search and self._web:
            tasks.append(("web", self._web.async_search(query, k=config.top_k)))

    
        source_results: dict[str, list[RetrievedChunk]] = {}
        task_names = [t[0] for t in tasks]
        task_coros = [t[1] for t in tasks]

        results = await asyncio.gather(*task_coros, return_exceptions=True)

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.warning(f"{name} search failed", error=str(result))
                source_results[name] = []
            else:
                source_results[name] = result

     
        weight_map = {
            "vector": config.vector_weight,
            "graph": config.graph_weight,
            "web": config.web_weight,
        }

        merged = self._rrf_merge(source_results, weight_map, config.rrf_k)

    
        if config.use_reranker and self._reranker and len(merged) > 0:
            merged = await self._reranker.async_rerank(
                query, merged, top_k=config.top_k
            )
        else:
            merged = merged[:config.top_k]

        logger.info(
            "Hybrid retrieval complete",
            sources={k: len(v) for k, v in source_results.items()},
            final_count=len(merged),
        )
        return merged

    def _rrf_merge(
        self,
        source_results: dict[str, list[RetrievedChunk]],
        weights: dict[str, float],
        rrf_k: int,
    ) -> list[RetrievedChunk]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.

        RRF avoids the problem of incompatible score distributions
        across different retrieval methods by using only rank positions.
        """
      
        scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}
        sources_map: dict[str, set[str]] = defaultdict(set)

        for source_name, chunks in source_results.items():
            weight = weights.get(source_name, 1.0)
            for rank, chunk in enumerate(chunks):
                key = chunk.text.strip()[:200]  
                scores[key] += weight / (rrf_k + rank + 1)
                sources_map[key].add(source_name)

                if key not in chunk_map:
                    chunk_map[key] = chunk

      
        merged = []
        for key in sorted(scores.keys(), key=lambda k: scores[k], reverse=True):
            chunk = chunk_map[key]
            chunk.score = scores[key]

         
            sources = sources_map[key]
            if len(sources) > 1:
                chunk.retrieval_method = "hybrid"

            merged.append(chunk)

        return merged
