"""Tests for retrieval components."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

from retrieval.vector_store import VectorStore, RetrievedChunk, ChunkMetadata


class TestVectorStore:
    """Tests for the VectorStore class."""

    def test_initial_state(self):
        store = VectorStore()
        assert store.document_count == 0
        assert not store.is_built

    def test_search_raises_when_empty(self):
        store = VectorStore()
        with pytest.raises(RuntimeError, match="No documents indexed"):
            store.search("test query")

    @patch("retrieval.vector_store.VectorStore._get_embedder")
    def test_add_and_search(self, mock_get_embedder):
        """Test add + search with mocked embedder and FAISS."""
        import faiss

        # Mock embedder
        mock_embedder = MagicMock()
        dim = 3
        mock_embedder.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32"
        )
        mock_get_embedder.return_value = mock_embedder

        store = VectorStore()
        docs = ["Doc about Paris", "Doc about Berlin"]
        added = store.add_documents(docs)

        assert added == 2
        assert store.document_count == 2
        assert store.is_built

        # Mock query embedding
        mock_embedder.encode.return_value = np.array(
            [[0.1, 0.2, 0.3]], dtype="float32"
        )
        results = store.search("Paris", k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert results[0].retrieval_method == "vector"

    @patch("retrieval.vector_store.VectorStore._get_embedder")
    def test_deduplication(self, mock_get_embedder):
        """Adding duplicate documents should be skipped."""
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array(
            [[0.1, 0.2, 0.3]], dtype="float32"
        )
        mock_get_embedder.return_value = mock_embedder

        store = VectorStore()
        added1 = store.add_documents(["Same document"])
        assert added1 == 1

        added2 = store.add_documents(["Same document"])
        assert added2 == 0  # Duplicate
        assert store.document_count == 1

    def test_clear(self):
        store = VectorStore()
        store._documents = ["doc1"]
        store._index = MagicMock()
        store.clear()
        assert store.document_count == 0
        assert not store.is_built


class TestRetrievedChunk:
    """Tests for the RetrievedChunk dataclass."""

    def test_to_dict(self):
        chunk = RetrievedChunk(
            text="Test text",
            score=0.9123,
            source="doc_0",
            retrieval_method="vector",
            metadata=ChunkMetadata(page=1, chunk_index=0),
        )
        d = chunk.to_dict()
        assert d["text"] == "Test text"
        assert d["score"] == 0.9123
        assert d["source"] == "doc_0"
        assert d["metadata"]["page"] == 1


class TestChunkMetadata:
    """Tests for ChunkMetadata."""

    def test_defaults(self):
        meta = ChunkMetadata()
        assert meta.source == ""
        assert meta.page == 0
        assert meta.chunk_index == 0
        assert meta.extra == {}
