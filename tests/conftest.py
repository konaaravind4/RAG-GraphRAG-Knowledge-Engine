"""
tests/conftest.py — Shared fixtures for all tests.

Provides mocked LLM clients, sample documents, and test configurations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config.settings import Settings


@pytest.fixture
def test_settings():
    """Settings with test defaults (no real API keys)."""
    return Settings(
        openai_api_key="sk-test-fake-key",
        llm_model="gpt-4o-mini",
        llm_fast_model="gpt-4o-mini",
        embed_model="BAAI/bge-base-en-v1.5",
        neo4j_url=None,
        chunk_size=128,
        chunk_overlap=16,
        max_agent_iterations=2,
        enable_tracing=True,
        log_level="WARNING",
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Berlin is the capital and largest city of Germany.",
        "Machine learning is a subset of artificial intelligence.",
        "Retrieval-augmented generation combines retrieval with language model generation.",
        "FAISS is a library for efficient similarity search developed by Meta.",
        "Neo4j is a graph database management system.",
        "Python is a high-level, general-purpose programming language.",
    ]


@pytest.fixture
def mock_llm_client():
    """Mocked LLM client that returns predictable responses."""
    client = MagicMock()
    client.usage = MagicMock()
    client.usage.to_dict.return_value = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    # Mock generate
    client.generate = AsyncMock(return_value="This is a test answer based on the context.")

    # Mock generate_structured
    client.generate_structured = AsyncMock()

    # Mock generate_stream
    async def mock_stream(*args, **kwargs):
        for word in ["This ", "is ", "a ", "test ", "answer."]:
            yield word
    client.generate_stream = mock_stream

    # Mock generate_with_messages
    client.generate_with_messages = AsyncMock(return_value="Multi-turn answer.")

    return client


@pytest.fixture
def sample_chunks():
    """Sample RetrievedChunk objects for testing."""
    from retrieval.vector_store import RetrievedChunk, ChunkMetadata

    return [
        RetrievedChunk(
            text="Paris is the capital of France.",
            score=0.95,
            source="doc_0",
            retrieval_method="vector",
            metadata=ChunkMetadata(source="doc_0", chunk_index=0),
        ),
        RetrievedChunk(
            text="The Eiffel Tower is in Paris.",
            score=0.85,
            source="doc_1",
            retrieval_method="vector",
            metadata=ChunkMetadata(source="doc_1", chunk_index=1),
        ),
        RetrievedChunk(
            text="Berlin is the capital of Germany.",
            score=0.45,
            source="doc_2",
            retrieval_method="vector",
            metadata=ChunkMetadata(source="doc_2", chunk_index=2),
        ),
    ]
