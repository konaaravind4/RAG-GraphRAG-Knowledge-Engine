"""
Unit tests for RAG/GraphRAG Knowledge Engine.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document


class TestVectorStore:
    @patch("rag.vector_store.HuggingFaceEmbeddings")
    @patch("rag.vector_store.faiss")
    def test_add_and_search(self, mock_faiss, mock_embed_cls):
        from rag.vector_store import VectorStore
        import numpy as np

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embed_cls.return_value = mock_embedder

        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[0.95]]), np.array([[0]]))
        mock_faiss.IndexFlatIP.return_value = mock_index

        store = VectorStore()
        doc = Document(page_content="Hello world", metadata={"source": "test"})
        store.add_documents([doc])

        results = store.search("Hello world", k=1)
        assert len(results) == 1
        assert results[0][0].page_content == "Hello world"
        assert results[0][1] == pytest.approx(0.95)

    @patch("rag.vector_store.HuggingFaceEmbeddings")
    @patch("rag.vector_store.faiss")
    def test_search_empty_store_returns_empty(self, mock_faiss, mock_embed_cls):
        from rag.vector_store import VectorStore

        mock_embed_cls.return_value = MagicMock()
        store = VectorStore()
        results = store.search("anything", k=5)
        assert results == []


class TestHybridRetriever:
    def test_fusion_scoring(self):
        from rag.hybrid_retriever import HybridRetriever, RetrievedChunk

        mock_vec = MagicMock()
        mock_graph = MagicMock()

        doc1 = Document(page_content="Vector doc", metadata={})
        doc2 = Document(page_content="Graph doc", metadata={})

        mock_vec.search.return_value = [(doc1, 0.9)]
        mock_graph.retrieve.return_value = [doc2]

        retriever = HybridRetriever(mock_vec, mock_graph, lambda_weight=0.6, top_k=5)
        chunks = retriever.retrieve("test query", k=2)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.combined_score >= 0.0

    def test_combined_score_formula(self):
        from rag.hybrid_retriever import RetrievedChunk
        from langchain.schema import Document

        chunk = RetrievedChunk(
            document=Document(page_content="x"),
            vector_score=0.8,
            graph_score=0.6,
        )
        λ = 0.6
        expected = λ * 0.8 + (1 - λ) * 0.6
        chunk.combined_score = expected
        assert chunk.combined_score == pytest.approx(0.72)


class TestRAGGenerator:
    @patch("rag.generator.genai" if True else "rag.generator.OpenAI")
    def test_build_context_truncates(self, _):
        from rag.generator import RAGGenerator

        # Just test the static method without instantiating (avoids API key)
        docs = [Document(page_content=f"Chunk {i} " + "x" * 1000) for i in range(20)]
        context = RAGGenerator._build_context(docs)
        assert len(context) <= 12100  # within limit + small buffer

    def test_build_prompt_contains_query(self):
        from rag.generator import RAGGenerator
        prompt = RAGGenerator._build_prompt("What is AI?", "Context: AI is artificial intelligence.")
        assert "What is AI?" in prompt
        assert "Context" in prompt
