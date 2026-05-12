"""
ingestion/chunker.py — Intelligent text chunking for RAG.

Features:
- Recursive character splitting respecting sentence/paragraph boundaries
- Configurable chunk size and overlap
- Token counting via tiktoken
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with position metadata."""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    source: str = ""


class TextChunker:
    """
    Intelligent text chunker that respects natural boundaries.

    Uses a hierarchy of separators:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (period + space)
    4. Spaces (word boundaries)
    5. Characters (last resort)

    Usage:
        chunker = TextChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk("Long document text...", source="report.pdf")
    """

    # Separator hierarchy — try splitting on these in order
    SEPARATORS = [
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentence ends
        "? ",      # Question ends
        "! ",      # Exclamation ends
        "; ",      # Semicolons
        ", ",      # Commas
        " ",       # Word boundaries
    ]

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        """
        Args:
            chunk_size: Target chunk size in tokens.
            overlap: Number of overlapping tokens between consecutive chunks.
        """
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazy-load the tiktoken tokenizer."""
        if self._tokenizer is None:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text, disallowed_special=()))

    def chunk(self, text: str, source: str = "") -> list[TextChunk]:
        """
        Split text into overlapping chunks respecting natural boundaries.

        Args:
            text: Full document text.
            source: Source identifier for metadata.

        Returns:
            List of TextChunk objects.
        """
        if not text.strip():
            return []

        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        raw_chunks = self._recursive_split(text, self.SEPARATORS)
        final_chunks = self._merge_with_overlap(raw_chunks, source)

        logger.info(
            "Chunked document",
            source=source,
            input_chars=len(text),
            chunks=len(final_chunks),
            avg_tokens=round(
                sum(c.token_count for c in final_chunks) / max(len(final_chunks), 1)
            ),
        )
        return final_chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the separator hierarchy.

        If a split produces pieces that are too large, recurse with
        the next separator in the hierarchy.
        """
        if self.count_tokens(text) <= self._chunk_size:
            return [text] if text.strip() else []

        # Find the best separator
        separator = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if separator:
            parts = text.split(separator)
        else:
            # Last resort: character split
            mid = len(text) // 2
            parts = [text[:mid], text[mid:]]

        # Re-merge parts that are too small, split those too large
        result = []
        current = ""

        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if self.count_tokens(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if self.count_tokens(part) > self._chunk_size and remaining_seps:
                    # Recurse with finer separators
                    result.extend(self._recursive_split(part, remaining_seps))
                else:
                    current = part

        if current:
            result.append(current)

        return [r for r in result if r.strip()]

    def _merge_with_overlap(
        self, raw_chunks: list[str], source: str
    ) -> list[TextChunk]:
        """
        Build final chunks with overlap from raw splits.

        Overlap ensures context continuity across chunk boundaries.
        """
        if not raw_chunks:
            return []

        tokenizer = self._get_tokenizer()
        final: list[TextChunk] = []
        char_pos = 0

        for i, chunk_text in enumerate(raw_chunks):
            # Add overlap from previous chunk
            if i > 0 and self._overlap > 0 and final:
                prev_text = final[-1].text
                prev_tokens = tokenizer.encode(prev_text, disallowed_special=())
                overlap_tokens = prev_tokens[-self._overlap:]
                overlap_text = tokenizer.decode(overlap_tokens)
                chunk_text = overlap_text + " " + chunk_text

            tokens = self.count_tokens(chunk_text)
            final.append(TextChunk(
                text=chunk_text.strip(),
                chunk_index=i,
                start_char=char_pos,
                end_char=char_pos + len(chunk_text),
                token_count=tokens,
                source=source,
            ))
            char_pos += len(raw_chunks[i])

        return final
