"""
rag/retriever.py — Hybrid vector + graph retriever for the RAG Knowledge Engine.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Parameters for the hybrid retrieval step."""
    top_k: int = 5
    lambda_: float = 0.6          # weight: vector vs graph (0=graph only, 1=vector only)
    score_threshold: float = 0.0  # minimum combined score to include


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    retrieval_method: str  # 'vector' | 'graph' | 'hybrid'


class VectorRetriever:
    """
    FAISS-backed dense retriever using HuggingFace sentence-transformers.
    Lazily initialised — call .build(docs) before .search().
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self._model_name = model_name
        self._index = None
        self._docs: list[str] = []
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._embedder = SentenceTransformer(self._model_name)
        return self._embedder

    def build(self, docs: list[str]) -> None:
        import faiss  # type: ignore
        import numpy as np

        self._docs = docs
        embedder = self._get_embedder()
        vectors = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        vectors = np.array(vectors, dtype="float32")

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        logger.info("FAISS index built with %d documents.", len(docs))

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        if self._index is None:
            raise RuntimeError("Call .build(docs) before .search().")
        import numpy as np

        embedder = self._get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")

        scores, indices = self._index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(RetrievedChunk(
                text=self._docs[idx],
                source=f"doc_{idx}",
                score=float(score),
                retrieval_method="vector",
            ))
        return results


class HybridRetriever:
    """
    Combines VectorRetriever with a LangChain / Neo4j graph traversal.
    Falls back to pure-vector when graph is not configured.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_url: Optional[str] = None,
        graph_user: str = "neo4j",
        graph_password: str = "password",
    ):
        self._vector = vector_retriever
        self._graph_url = graph_url
        self._graph = None

        if graph_url:
            try:
                from langchain_community.graphs import Neo4jGraph  # type: ignore
                self._graph = Neo4jGraph(
                    url=graph_url,
                    username=graph_user,
                    password=graph_password,
                )
                logger.info("Neo4j graph connected at %s.", graph_url)
            except Exception as exc:
                logger.warning("Graph connection failed (%s); using vector-only mode.", exc)

    def retrieve(self, query: str, cfg: RetrievalConfig) -> list[RetrievedChunk]:
        vector_results = self._vector.search(query, k=cfg.top_k * 2)

        graph_results: list[RetrievedChunk] = []
        if self._graph:
            try:
                cypher = (
                    "MATCH (n)-[r]->(m) "
                    "WHERE n.name CONTAINS $q OR m.name CONTAINS $q "
                    "RETURN n.text AS text, n.source AS source LIMIT $k"
                )
                rows = self._graph.query(cypher, params={"q": query, "k": cfg.top_k})
                for row in rows:
                    graph_results.append(RetrievedChunk(
                        text=row.get("text", ""),
                        source=row.get("source", "graph"),
                        score=0.5,  # placeholder; re-ranked below
                        retrieval_method="graph",
                    ))
            except Exception as exc:
                logger.warning("Graph query failed: %s", exc)

        # ── Merge & re-rank ───────────────────────────────────────────────────
        combined: dict[str, RetrievedChunk] = {}
        for chunk in vector_results:
            combined[chunk.text] = RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                score=cfg.lambda_ * chunk.score,
                retrieval_method="hybrid" if graph_results else "vector",
            )
        for chunk in graph_results:
            if chunk.text in combined:
                combined[chunk.text].score += (1 - cfg.lambda_) * chunk.score
                combined[chunk.text].retrieval_method = "hybrid"
            else:
                combined[chunk.text] = RetrievedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    score=(1 - cfg.lambda_) * chunk.score,
                    retrieval_method="graph",
                )

        ranked = sorted(combined.values(), key=lambda c: c.score, reverse=True)
        return [c for c in ranked[:cfg.top_k] if c.score >= cfg.score_threshold]
