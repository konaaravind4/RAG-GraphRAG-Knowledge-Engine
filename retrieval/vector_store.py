"""
retrieval/vector_store.py — FAISS-backed dense vector retriever.

Features:
- Lazy embedding model initialization
- Async-compatible (blocking ops in thread pool)
- Index persistence (save/load to disk)
- Metadata support per chunk
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a stored chunk."""
    source: str = ""
    page: int = 0
    chunk_index: int = 0
    content_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with score and provenance."""
    text: str
    score: float
    source: str
    retrieval_method: str  # "vector" | "graph" | "web" | "hybrid"
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "retrieval_method": self.retrieval_method,
            "metadata": {
                "page": self.metadata.page,
                "chunk_index": self.metadata.chunk_index,
            },
        }


class VectorStore:
    """
    FAISS-backed dense retriever using sentence-transformers.

    Usage:
        store = VectorStore(model_name="BAAI/bge-base-en-v1.5")
        store.add_documents(texts, metadatas)
        results = store.search("query", k=5)
        store.save("./data/index")
        store.load("./data/index")
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", namespace: str = "default"):
        self._model_name = model_name
        self._namespace = namespace
        self._embedder = None
        self._index = None
        self._documents: list[str] = []
        self._metadatas: list[ChunkMetadata] = []
        self._dimension: int = 0

    @property
    def namespace(self) -> str:
        """Return the namespace this store is associated with.

        The namespace is used to logically partition knowledge (e.g. 'financial',
        'code_review') without requiring separate FAISS index files.
        """
        return self._namespace

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def is_built(self) -> bool:
        return self._index is not None and len(self._documents) > 0

    def count(self) -> int:
        """Return the number of chunks currently stored in the index.

        Convenience alias for :attr:`document_count` that reads naturally in
        procedural contexts (e.g. ``store.count() == 0``).
        """
        return len(self._documents)

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._model_name)
            logger.info("Loaded embedding model", model=self._model_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning normalized vectors."""
        embedder = self._get_embedder()
        vectors = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return np.array(vectors, dtype="float32")

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[ChunkMetadata]] = None,
    ) -> int:
        """
        Add documents to the index. Can be called multiple times (incremental).

        Returns:
            Number of new documents added (after dedup).
        """
        import faiss

        if not texts:
            return 0

        # Generate metadatas if not provided
        if metadatas is None:
            metadatas = [
                ChunkMetadata(
                    source=f"doc_{len(self._documents) + i}",
                    chunk_index=i,
                    content_hash=hashlib.md5(t.encode()).hexdigest()[:8],
                )
                for i, t in enumerate(texts)
            ]

        # Dedup by content hash
        existing_hashes = {m.content_hash for m in self._metadatas}
        new_texts, new_metas = [], []
        for text, meta in zip(texts, metadatas):
            if not meta.content_hash:
                meta.content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            if meta.content_hash not in existing_hashes:
                new_texts.append(text)
                new_metas.append(meta)
                existing_hashes.add(meta.content_hash)

        if not new_texts:
            logger.info("No new documents to add (all duplicates)")
            return 0

        vectors = self._embed(new_texts)
        self._dimension = vectors.shape[1]

        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dimension)

        self._index.add(vectors)
        self._documents.extend(new_texts)
        self._metadatas.extend(new_metas)

        logger.info(
            "Added documents to index",
            new=len(new_texts),
            total=len(self._documents),
        )
        return len(new_texts)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """
        Search the index for the k most similar chunks.

        Raises:
            RuntimeError: If no documents have been indexed.
        """
        if not self.is_built:
            raise RuntimeError("No documents indexed. Call add_documents() first.")

        q_vec = self._embed([query])
        k = min(k, len(self._documents))
        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(RetrievedChunk(
                text=self._documents[idx],
                score=float(score),
                source=self._metadatas[idx].source,
                retrieval_method="vector",
                metadata=self._metadatas[idx],
            ))
        return results

    async def async_search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Async wrapper — runs blocking search in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, k)

    async def async_add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[ChunkMetadata]] = None,
    ) -> int:
        """Async wrapper — runs blocking add in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.add_documents, texts, metadatas)

    def save(self, directory: str) -> None:
        """Persist the index and documents to disk."""
        import faiss

        if not self.is_built:
            logger.warning("Nothing to save — index is empty")
            return

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "store.pkl", "wb") as f:
            pickle.dump({
                "documents": self._documents,
                "metadatas": self._metadatas,
                "dimension": self._dimension,
                "model_name": self._model_name,
            }, f)

        logger.info("Saved index", path=str(path), documents=len(self._documents))

    def load(self, directory: str) -> bool:
        """Load a persisted index from disk. Returns True if loaded."""
        import faiss

        path = Path(directory)
        index_path = path / "index.faiss"
        store_path = path / "store.pkl"

        if not index_path.exists() or not store_path.exists():
            return False

        self._index = faiss.read_index(str(index_path))
        with open(store_path, "rb") as f:
            data = pickle.load(f)

        self._documents = data["documents"]
        self._metadatas = data["metadatas"]
        self._dimension = data["dimension"]

        logger.info("Loaded index", path=str(path), documents=len(self._documents))
        return True

    def clear(self) -> None:
        """Remove all documents and reset the index."""
        self._index = None
        self._documents.clear()
        self._metadatas.clear()
        self._dimension = 0
        logger.info("Cleared vector store")

    def clear_namespace(self, namespace: str) -> int:
        """Remove all chunks whose source begins with *namespace*.

        This enables soft multi-tenancy: each ingested document is tagged with
        a source like ``'financial/doc_0'``.  Calling
        ``clear_namespace('financial')`` removes only those chunks while leaving
        other namespaces intact.

        The FAISS index is rebuilt from the remaining documents after removal.

        Args:
            namespace: Source-prefix to match (case-sensitive, e.g. 'financial').

        Returns:
            Number of chunks removed.

        Raises:
            RuntimeError: If FAISS is not installed.
        """
        import faiss

        prefix = namespace.rstrip("/") + "/"
        keep_texts: list[str] = []
        keep_metas: list[ChunkMetadata] = []

        for text, meta in zip(self._documents, self._metadatas):
            if not meta.source.startswith(prefix) and meta.source != namespace:
                keep_texts.append(text)
                keep_metas.append(meta)

        removed = len(self._documents) - len(keep_texts)

        if removed == 0:
            logger.info("clear_namespace: no chunks matched", namespace=namespace)
            return 0

        # Rebuild index from remaining chunks
        self._documents = keep_texts
        self._metadatas = keep_metas

        if keep_texts:
            vectors = self._embed(keep_texts)
            self._dimension = vectors.shape[1]
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(vectors)
        else:
            self._index = None
            self._dimension = 0

        logger.info(
            "clear_namespace complete",
            namespace=namespace,
            removed=removed,
            remaining=len(self._documents),
        )
        return removed
