"""
ingestion/pipeline.py — End-to-end document ingestion pipeline.

Orchestrates: Parse → Chunk → Embed → Index
Supports: file upload, URL ingestion, raw text
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import get_settings
from ingestion.chunker import TextChunker
from ingestion.parser import DocumentParser, ParsedDocument
from retrieval.vector_store import ChunkMetadata, VectorStore
from observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentRecord:
    """Record of an ingested document."""
    doc_id: str
    source: str
    content_hash: str
    chunk_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestResult:
    """Result of a batch ingestion operation."""
    documents: list[DocumentRecord] = field(default_factory=list)
    total_chunks: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.documents)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_ingested": self.success_count,
            "total_chunks": self.total_chunks,
            "errors": self.error_count,
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "source": d.source,
                    "chunks": d.chunk_count,
                }
                for d in self.documents
            ],
            "error_details": self.errors,
        }


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline(vector_store)
        result = await pipeline.ingest_file(file_content, filename)
        result = await pipeline.ingest_url("https://example.com/article")
        result = await pipeline.ingest_texts(["doc1", "doc2"])
    """

    def __init__(
        self,
        vector_store: VectorStore,
        settings=None,
    ):
        self._settings = settings or get_settings()
        self._vector_store = vector_store
        self._parser = DocumentParser()
        self._chunker = TextChunker(
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        self._ingested_hashes: set[str] = set()

    @property
    def document_count(self) -> int:
        return self._vector_store.document_count

    async def ingest_file(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None,
    ) -> IngestResult:
        """Ingest an uploaded file."""
        result = IngestResult()
        try:
            doc = self._parser.parse_bytes(content, filename, content_type)
            await self._process_document(doc, result)
        except Exception as exc:
            logger.error("File ingestion failed", filename=filename, error=str(exc))
            result.errors.append({"source": filename, "error": str(exc)})
        return result

    async def ingest_url(self, url: str) -> IngestResult:
        """Ingest content from a URL."""
        result = IngestResult()
        try:
            doc = self._parser.parse_url(url)
            await self._process_document(doc, result)
        except Exception as exc:
            logger.error("URL ingestion failed", url=url, error=str(exc))
            result.errors.append({"source": url, "error": str(exc)})
        return result

    async def ingest_texts(
        self,
        texts: list[str],
        source: str = "manual",
    ) -> IngestResult:
        """Ingest a list of raw text strings."""
        result = IngestResult()
        for i, text in enumerate(texts):
            try:
                doc = self._parser.parse_text(text, source=f"{source}_{i}")
                await self._process_document(doc, result)
            except Exception as exc:
                logger.error("Text ingestion failed", index=i, error=str(exc))
                result.errors.append({"source": f"{source}_{i}", "error": str(exc)})
        return result

    async def _process_document(
        self, doc: ParsedDocument, result: IngestResult
    ) -> None:
        """Process a single parsed document: chunk → embed → index."""
        # Check for duplicate content
        content_hash = hashlib.sha256(doc.content.encode()).hexdigest()[:16]
        if content_hash in self._ingested_hashes:
            logger.info("Skipping duplicate document", source=doc.source)
            return

        # Chunk
        chunks = self._chunker.chunk(doc.content, source=doc.source)
        if not chunks:
            logger.warning("Document produced no chunks", source=doc.source)
            return

        # Build metadata for each chunk
        texts = [c.text for c in chunks]
        metadatas = [
            ChunkMetadata(
                source=doc.source,
                page=doc.metadata.get("page", 0),
                chunk_index=c.chunk_index,
                content_hash=hashlib.md5(c.text.encode()).hexdigest()[:8],
                extra={
                    "token_count": c.token_count,
                    "doc_hash": content_hash,
                },
            )
            for c in chunks
        ]

        # Index
        added = await self._vector_store.async_add_documents(texts, metadatas)
        self._ingested_hashes.add(content_hash)

        doc_id = uuid.uuid4().hex[:8]
        record = DocumentRecord(
            doc_id=doc_id,
            source=doc.source,
            content_hash=content_hash,
            chunk_count=added,
            metadata=doc.metadata,
        )
        result.documents.append(record)
        result.total_chunks += added

        logger.info(
            "Document ingested",
            doc_id=doc_id,
            source=doc.source,
            chunks=added,
        )

    def get_ingested_sources(self) -> list[str]:
        """Return list of ingested document sources."""
        sources = set()
        for meta in self._vector_store._metadatas:
            sources.add(meta.source)
        return sorted(sources)
