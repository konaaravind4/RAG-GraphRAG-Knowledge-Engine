"""
FAISS vector store for semantic chunk retrieval.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
INDEX_PATH = Path("./data/faiss_index")


class VectorStore:
    """
    FAISS-backed vector store for dense semantic retrieval.
    Uses BAAI/bge-base-en-v1.5 (768d) embeddings.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info("Loading embedding model: %s", model_name)
        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: list[Document] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[Document]) -> None:
        """Embed and add documents to the FAISS index."""
        if not docs:
            return
        texts = [d.page_content for d in docs]
        embeddings = self.embedder.embed_documents(texts)
        vectors = np.array(embeddings, dtype="float32")

        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # inner product = cosine (normalized)

        self.index.add(vectors)
        self.documents.extend(docs)
        logger.info("Indexed %d documents. Total: %d", len(docs), len(self.documents))

    def save(self, path: Path = INDEX_PATH) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        logger.info("Index saved to %s", path)

    def load(self, path: Path = INDEX_PATH) -> None:
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "docs.pkl", "rb") as f:
            self.documents = pickle.load(f)
        logger.info("Loaded %d documents from %s", len(self.documents), path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """Return top-k (document, score) pairs for the query."""
        if self.index is None or len(self.documents) == 0:
            return []

        query_vec = np.array(
            [self.embedder.embed_query(query)], dtype="float32"
        )
        scores, indices = self.index.search(query_vec, min(k, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.documents[idx], float(score)))
        return results
