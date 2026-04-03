"""
rag/generator.py — LLM answer generator using retrieved context.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def generate_answer(
    query: str,
    context_chunks: list[str],
    model: str = "gpt-4o-mini",
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Generate an answer from retrieved context using OpenAI or a compatible LLM.

    Args:
        query: The user's original question.
        context_chunks: List of retrieved text passages for grounding.
        model: LLM model identifier.
        max_tokens: Maximum generation length.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        Generated answer string.
    """
    context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))

    system_prompt = (
        "You are a precise, knowledgeable AI assistant. "
        "Answer the question using ONLY the provided context. "
        "If the context does not contain enough information, say so. "
        "Be concise and cite context numbers [1], [2], etc."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        raise
