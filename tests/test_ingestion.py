"""Tests for ingestion pipeline."""

import pytest
from unittest.mock import MagicMock, patch

from ingestion.parser import DocumentParser, ParsedDocument


class TestDocumentParser:
    """Tests for the DocumentParser class."""

    def test_parse_text(self):
        parser = DocumentParser()
        doc = parser.parse_text("Hello world", source="test")
        assert doc.content == "Hello world"
        assert doc.source == "test"
        assert doc.metadata["format"] == "text"

    def test_parse_text_strips_whitespace(self):
        parser = DocumentParser()
        doc = parser.parse_text("  Hello world  ", source="test")
        assert doc.content == "Hello world"

    def test_parse_bytes_text(self):
        parser = DocumentParser()
        content = b"Hello from bytes"
        doc = parser.parse_bytes(content, "test.txt")
        assert doc.content == "Hello from bytes"
        assert doc.source == "test.txt"

    def test_parse_file_not_found(self):
        parser = DocumentParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/file.txt")

    def test_parsed_document_dataclass(self):
        doc = ParsedDocument(
            content="Test content",
            source="test.txt",
            num_pages=3,
            metadata={"format": "pdf"},
        )
        assert doc.content == "Test content"
        assert doc.num_pages == 3
        assert doc.metadata["format"] == "pdf"


class TestIngestionPipeline:
    """Tests for the IngestionPipeline class."""

    @pytest.mark.asyncio
    async def test_ingest_texts(self, test_settings):
        from ingestion.pipeline import IngestionPipeline
        from retrieval.vector_store import VectorStore

        # Mock the vector store to avoid loading real models
        mock_store = MagicMock(spec=VectorStore)
        mock_store.document_count = 0
        mock_store._metadatas = []
        mock_store.async_add_documents = MagicMock(return_value=3)

        # Make async_add_documents a coroutine
        async def mock_add(*args, **kwargs):
            return 3
        mock_store.async_add_documents = mock_add

        pipeline = IngestionPipeline(
            vector_store=mock_store,
            settings=test_settings,
        )

        result = await pipeline.ingest_texts(
            ["Doc one.", "Doc two.", "Doc three."],
            source="test",
        )

        assert result.success_count == 3
        assert result.error_count == 0

    @pytest.mark.asyncio
    async def test_ingest_texts_dedup(self, test_settings):
        from ingestion.pipeline import IngestionPipeline
        from retrieval.vector_store import VectorStore

        mock_store = MagicMock(spec=VectorStore)
        mock_store.document_count = 0
        mock_store._metadatas = []

        async def mock_add(*args, **kwargs):
            return 1
        mock_store.async_add_documents = mock_add

        pipeline = IngestionPipeline(
            vector_store=mock_store,
            settings=test_settings,
        )

        # Ingest same text twice
        await pipeline.ingest_texts(["Same content"], source="test1")
        result = await pipeline.ingest_texts(["Same content"], source="test2")

        # Second ingestion should be deduped
        assert result.success_count == 0
