"""Ingestion module — document parsing, chunking, and indexing pipeline."""

from ingestion.parser import DocumentParser, ParsedDocument
from ingestion.chunker import TextChunker, TextChunk
from ingestion.pipeline import IngestionPipeline, IngestResult, DocumentRecord

__all__ = [
    "DocumentParser",
    "ParsedDocument",
    "TextChunker",
    "TextChunk",
    "IngestionPipeline",
    "IngestResult",
    "DocumentRecord",
]
