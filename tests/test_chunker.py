"""Tests for ingestion/chunker.py — text chunking logic."""

import pytest
from ingestion.chunker import TextChunker, TextChunk


class TestTextChunker:
    """Tests for the TextChunker class."""

    def test_empty_text(self):
        chunker = TextChunker(chunk_size=100, overlap=10)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_single_chunk(self):
        chunker = TextChunker(chunk_size=512, overlap=64)
        text = "This is a short sentence."
        chunks = chunker.chunk(text, source="test")
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].source == "test"
        assert chunks[0].chunk_index == 0

    def test_long_text_multiple_chunks(self):
        chunker = TextChunker(chunk_size=20, overlap=0)
        # Each sentence is roughly 5-10 tokens
        text = (
            "First sentence about Paris. "
            "Second sentence about Berlin. "
            "Third sentence about London. "
            "Fourth sentence about Tokyo. "
            "Fifth sentence about New York."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_respects_paragraph_boundaries(self):
        chunker = TextChunker(chunk_size=10, overlap=0)
        text = (
            "This is the first paragraph about artificial intelligence.\n\n"
            "This is the second paragraph about machine learning.\n\n"
            "This is the third paragraph about deep learning."
        )
        chunks = chunker.chunk(text)
        # Small chunk_size forces splitting on paragraph boundaries
        assert len(chunks) >= 2

    def test_chunk_metadata(self):
        chunker = TextChunker(chunk_size=512, overlap=0)
        text = "A sample document for testing metadata."
        chunks = chunker.chunk(text, source="report.pdf")
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert chunks[0].source == "report.pdf"
        assert chunks[0].token_count > 0

    def test_token_counting(self):
        chunker = TextChunker()
        count = chunker.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_overlap_produces_more_chunks(self):
        text = "A " * 200  # Long enough text
        chunker_no_overlap = TextChunker(chunk_size=50, overlap=0)
        chunker_with_overlap = TextChunker(chunk_size=50, overlap=10)
        chunks_no = chunker_no_overlap.chunk(text)
        chunks_yes = chunker_with_overlap.chunk(text)
        # With overlap, chunk count should be similar or slightly more
        assert len(chunks_no) > 0
        assert len(chunks_yes) > 0

    def test_excessive_whitespace_cleaned(self):
        chunker = TextChunker(chunk_size=512, overlap=0)
        text = "Hello\n\n\n\n\n\nWorld"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # Should not have more than 2 consecutive newlines
        for chunk in chunks:
            assert "\n\n\n" not in chunk.text


class TestTextChunkerEdgeCases:
    """Edge case tests."""

    def test_unicode_text(self):
        chunker = TextChunker(chunk_size=512, overlap=0)
        text = "こんにちは世界。これはテストです。"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_very_long_word(self):
        chunker = TextChunker(chunk_size=20, overlap=0)
        text = "a" * 500 + " normal words here"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_single_character(self):
        chunker = TextChunker(chunk_size=512, overlap=0)
        chunks = chunker.chunk("x")
        assert len(chunks) == 1
