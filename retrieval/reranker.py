"""
retrieval/reranker.py — Cross-encoder reranker for precision improvement.

Re-scores top-N candidates from initial retrieval using a more accurate
(but slower) cross-encoder model.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from retrieval.vector_store import RetrievedChunk
from observability.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Lazily loads the model on first use. Falls back gracefully
    if the model isn't available (returns input unchanged).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None
        self._available = True

    def _get_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                logger.info("Loaded reranker model", model=self._model_name)
            except Exception as exc:
                logger.warning("Reranker model unavailable", error=str(exc))
                self._available = False
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Re-score and re-sort chunks using the cross-encoder.

        Args:
            query: The user's query.
            chunks: Pre-retrieved chunks to rerank.
            top_k: Return only top-k after reranking.

        Returns:
            Reranked chunks sorted by cross-encoder score.
        """
        if not chunks:
            return []

        model = self._get_model()
        if model is None or not self._available:
            logger.debug("Reranker unavailable, returning original order")
            return chunks[:top_k] if top_k else chunks

        try:
            # Cross-encoder takes (query, document) pairs
            pairs = [(query, chunk.text) for chunk in chunks]
            scores = model.predict(pairs)

            # Update scores
            for chunk, score in zip(chunks, scores):
                chunk.score = float(score)

            # Sort descending
            reranked = sorted(chunks, key=lambda c: c.score, reverse=True)

            if top_k:
                reranked = reranked[:top_k]

            logger.info(
                "Reranked chunks",
                input_count=len(chunks),
                output_count=len(reranked),
                top_score=round(reranked[0].score, 4) if reranked else 0,
            )
            return reranked

        except Exception as exc:
            logger.warning("Reranking failed, returning original order", error=str(exc))
            return chunks[:top_k] if top_k else chunks

    async def async_rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Async wrapper for reranking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, chunks, top_k)
