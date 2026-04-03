"""
tests/test_retriever.py — Unit tests for the RAG retriever (no live Neo4j or OpenAI needed).
"""

import pytest
from unittest.mock import MagicMock, patch


# ─── RetrievalConfig ──────────────────────────────────────────────────────────

class TestRetrievalConfig:
    def test_defaults(self):
        from rag.retriever import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.top_k == 5
        assert cfg.lambda_ == 0.6

    def test_custom(self):
        from rag.retriever import RetrievalConfig
        cfg = RetrievalConfig(top_k=10, lambda_=0.3)
        assert cfg.top_k == 10
        assert cfg.lambda_ == 0.3


# ─── VectorRetriever ─────────────────────────────────────────────────────────

class TestVectorRetriever:
    @patch("rag.retriever.faiss", create=True)
    @patch("rag.retriever.VectorRetriever._get_embedder")
    def test_build_and_search(self, mock_get_embedder, mock_faiss):
        """Test build + search interaction with mocked FAISS and embedder."""
        import numpy as np
        from rag.retriever import VectorRetriever, RetrievalConfig

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
        mock_get_embedder.return_value = mock_embedder

        # Mock FAISS index
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.7]]),
            np.array([[0, 1]])
        )

        import faiss
        faiss.IndexFlatIP = MagicMock(return_value=mock_index)

        docs = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
        retriever = VectorRetriever()
        retriever.build(docs)
        results = retriever.search("capital of France", k=2)

        assert len(results) == 2
        assert results[0].retrieval_method == "vector"
        assert results[0].score > 0

    def test_search_raises_when_not_built(self):
        from rag.retriever import VectorRetriever
        r = VectorRetriever()
        with pytest.raises(RuntimeError, match="build"):
            r.search("test")


# ─── HybridRetriever (vector-only mode) ──────────────────────────────────────

class TestHybridRetriever:
    def test_retrieve_vector_only(self):
        """HybridRetriever should work with no graph configured."""
        from rag.retriever import VectorRetriever, HybridRetriever, RetrievalConfig, RetrievedChunk

        mock_vector = MagicMock(spec=VectorRetriever)
        mock_vector.search.return_value = [
            RetrievedChunk(text="Paris is in France.", source="doc_0", score=0.9, retrieval_method="vector"),
            RetrievedChunk(text="London is in England.", source="doc_1", score=0.7, retrieval_method="vector"),
        ]

        retriever = HybridRetriever(vector_retriever=mock_vector, graph_url=None)
        cfg = RetrievalConfig(top_k=2, lambda_=1.0)
        results = retriever.retrieve("Paris", cfg)

        assert len(results) == 2
        assert results[0].score >= results[1].score


# ─── FastAPI endpoint ─────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self):
        from fastapi.testclient import TestClient
        import api.main as main_module

        # Inject mocks to bypass lifespan
        mock_vector = MagicMock()
        mock_vector._docs = ["doc1"]
        mock_retriever = MagicMock()

        main_module._vector_store = mock_vector
        main_module._retriever = mock_retriever

        client = TestClient(main_module.app, raise_server_exceptions=True)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
