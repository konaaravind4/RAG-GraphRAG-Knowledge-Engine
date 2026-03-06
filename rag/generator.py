"""
LLM generator — uses retrieved context to answer queries via Gemini/OpenAI.
"""
from __future__ import annotations

import logging
import os
from typing import Iterator

from langchain.schema import Document

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


SYSTEM_PROMPT = """You are a precise knowledge assistant. Answer questions based ONLY on the provided context.
If the context does not contain enough information, say "I don't have enough information to answer this with confidence."
Always cite your sources when referencing specific facts."""


class RAGGenerator:
    """
    Generates grounded answers from retrieved context using Gemini or OpenAI.
    """

    def __init__(self, provider: str = "gemini", model: str | None = None):
        self.provider = provider
        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model or "gemini-1.5-flash",
                system_instruction=SYSTEM_PROMPT,
            )
        elif provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self._model_name = model or "gpt-4o-mini"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, query: str, context_docs: list[Document]) -> str:
        """Generate a grounded answer given retrieved context chunks."""
        context = self._build_context(context_docs)
        prompt = self._build_prompt(query, context)

        if self.provider == "gemini":
            response = self.model.generate_content(prompt)
            return response.text.strip()
        else:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()

    def generate_with_sources(
        self, query: str, context_docs: list[Document]
    ) -> dict[str, object]:
        """Returns answer + source citations."""
        answer = self.generate(query, context_docs)
        sources = list({
            doc.metadata.get("source", doc.page_content[:60])
            for doc in context_docs
        })
        return {"answer": answer, "sources": sources, "context_chunks": len(context_docs)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(docs: list[Document]) -> str:
        parts = []
        total = 0
        for i, doc in enumerate(docs):
            chunk = f"[{i+1}] {doc.page_content}"
            if total + len(chunk) > MAX_CONTEXT_CHARS:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return f"""Context:
{context}

Question: {query}

Answer based only on the context above:"""
