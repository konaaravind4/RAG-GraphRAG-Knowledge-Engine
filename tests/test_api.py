"""Tests for API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    import api.main as main_module

    # Mock global state
    mock_vector = MagicMock()
    mock_vector.document_count = 5
    mock_vector.is_built = True

    mock_orch = MagicMock()
    mock_orch.run = AsyncMock(return_value=MagicMock(
        answer="Test answer",
        query="Test query",
        conversation_id="test",
        sources=[],
        route_decision="retrieval",
        iterations=1,
        trace=None,
    ))

    mock_pipeline = MagicMock()
    mock_pipeline.get_ingested_sources.return_value = ["doc1.pdf"]

    main_module._vector_store = mock_vector
    main_module._orchestrator = mock_orch
    main_module._pipeline = mock_pipeline

    return TestClient(main_module.app, raise_server_exceptions=True)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"

class TestChatEndpoint:
    def test_chat_success(self, client):
        resp = client.post("/chat", json={"query": "What is RAG?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "conversation_id" in data

    def test_chat_empty_query(self, client):
        resp = client.post("/chat", json={"query": ""})
        assert resp.status_code == 422  # Validation error

class TestDocumentsEndpoint:
    def test_list_documents(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_chunks" in data
        assert "sources" in data
