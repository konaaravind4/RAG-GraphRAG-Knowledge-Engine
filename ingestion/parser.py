"""
ingestion/parser.py — Multi-format document parser.

Supports: PDF, plain text, Markdown, and URLs (HTML → text).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedDocument:
    """A parsed document with content and metadata."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    num_pages: int = 0


class DocumentParser:
    """
    Multi-format document parser.

    Usage:
        parser = DocumentParser()
        doc = parser.parse_file("report.pdf")
        doc = parser.parse_url("https://example.com/article")
        doc = parser.parse_text("Raw text content", source="manual")
    """

    def parse_file(self, filepath: Union[str, Path]) -> ParsedDocument:
        """
        Parse a file based on its extension.

        Supported: .pdf, .txt, .md, .text
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        source = str(path.name)

        if ext == ".pdf":
            return self._parse_pdf(path, source)
        elif ext in (".txt", ".md", ".text", ".rst"):
            return self._parse_text_file(path, source)
        else:
            # Try as plain text
            logger.warning("Unknown file type, treating as plain text", ext=ext)
            return self._parse_text_file(path, source)

    def parse_bytes(
        self, content: bytes, filename: str, content_type: Optional[str] = None
    ) -> ParsedDocument:
        """Parse uploaded file content from bytes."""
        if filename.lower().endswith(".pdf") or content_type == "application/pdf":
            return self._parse_pdf_bytes(content, filename)
        else:
            text = content.decode("utf-8", errors="replace")
            return ParsedDocument(
                content=text.strip(),
                source=filename,
                metadata={"format": "text", "filename": filename},
            )

    def parse_url(self, url: str) -> ParsedDocument:
        """
        Fetch a URL and parse the HTML content to clean text.

        Uses BeautifulSoup to extract meaningful content.
        """
        import requests
        from bs4 import BeautifulSoup

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; AgenticRAG/1.0; "
                    "+https://github.com/konaaravind4/RAG-GraphRAG-Knowledge-Engine)"
                ),
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script, style, and nav elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Extract title
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Extract main content
            main = soup.find("main") or soup.find("article") or soup.body or soup
            text = main.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = "\n".join(lines)

            logger.info("Parsed URL", url=url, chars=len(content))
            return ParsedDocument(
                content=content,
                source=url,
                metadata={
                    "format": "url",
                    "title": title,
                    "url": url,
                    "content_length": len(content),
                },
            )

        except Exception as exc:
            logger.error("URL parsing failed", url=url, error=str(exc))
            raise ValueError(f"Failed to parse URL: {url} — {exc}") from exc

    def parse_text(self, text: str, source: str = "manual") -> ParsedDocument:
        """Parse raw text input."""
        return ParsedDocument(
            content=text.strip(),
            source=source,
            metadata={"format": "text"},
        )

    # ── Private methods ──────────────────────────────────────────────────────

    def _parse_pdf(self, path: Path, source: str) -> ParsedDocument:
        """Parse a PDF file using pypdf."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(text.strip())

        content = "\n\n".join(pages)
        logger.info("Parsed PDF", source=source, pages=len(pages), chars=len(content))

        return ParsedDocument(
            content=content,
            source=source,
            num_pages=len(reader.pages),
            metadata={
                "format": "pdf",
                "pages": len(reader.pages),
                "filename": source,
            },
        )

    def _parse_pdf_bytes(self, content: bytes, source: str) -> ParsedDocument:
        """Parse PDF from bytes."""
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

        text_content = "\n\n".join(pages)
        return ParsedDocument(
            content=text_content,
            source=source,
            num_pages=len(reader.pages),
            metadata={
                "format": "pdf",
                "pages": len(reader.pages),
                "filename": source,
            },
        )

    def _parse_text_file(self, path: Path, source: str) -> ParsedDocument:
        """Parse a plain text or markdown file."""
        content = path.read_text(encoding="utf-8", errors="replace")

        return ParsedDocument(
            content=content.strip(),
            source=source,
            metadata={
                "format": path.suffix.lstrip(".") or "text",
                "filename": source,
            },
        )
