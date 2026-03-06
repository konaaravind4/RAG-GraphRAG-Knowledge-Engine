"""
Hybrid retriever: fuses FAISS vector search + Neo4j graph traversal with re-ranking.
Score fusion: λ·cos(q,d) + (1-λ)·graph_score
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain.schema import Document

from rag.vector_store import VectorStore
from rag.graph_retriever import GraphRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    document: Document
    vector_score: float = 0.0
    graph_score: float = 0.0
    combined_score: float = 0.0
    source: str = "vector"  # "vector" | "graph" | "both"


class HybridRetriever:
    """
    Fuses dense vector retrieval and graph-based retrieval with configurable λ.

    score(q, d) = λ · cos(q, d) + (1 - λ) · graph_score(d)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_retriever: GraphRetriever,
        lambda_weight: float = 0.6,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.lambda_weight = lambda_weight
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """
        Perform hybrid retrieval and return top-k fused, re-ranked chunks.
        """
        k = k or self.top_k
        multiplier = 2  # fetch 2x to allow fusion dedup

        # ── Vector retrieval ──────────────────────────────────────────
        vec_results = self.vector_store.search(query, k=k * multiplier)
        vec_map: dict[str, RetrievedChunk] = {}
        for doc, score in vec_results:
            key = doc.page_content[:120]
            vec_map[key] = RetrievedChunk(
                document=doc,
                vector_score=score,
                source="vector",
            )

        # ── Graph retrieval ───────────────────────────────────────────
        graph_docs = self.graph_retriever.retrieve(query, limit=k * multiplier)
        for i, doc in enumerate(graph_docs):
            # Assign a decaying graph score (1.0 for best match, linear decay)
            graph_score = 1.0 - (i / max(len(graph_docs), 1)) * 0.5
            key = doc.page_content[:120]
            if key in vec_map:
                vec_map[key].graph_score = graph_score
                vec_map[key].source = "both"
            else:
                vec_map[key] = RetrievedChunk(
                    document=doc,
                    graph_score=graph_score,
                    source="graph",
                )

        # ── Fusion scoring ────────────────────────────────────────────
        chunks = list(vec_map.values())
        λ = self.lambda_weight
        for chunk in chunks:
            chunk.combined_score = (
                λ * chunk.vector_score + (1 - λ) * chunk.graph_score
            )

        # ── Re-rank and return top-k ──────────────────────────────────
        chunks.sort(key=lambda c: c.combined_score, reverse=True)
        return chunks[:k]

    def retrieve_documents(self, query: str, k: int | None = None) -> list[Document]:
        """Convenience method — returns just the Document objects."""
        return [chunk.document for chunk in self.retrieve(query, k=k)]
