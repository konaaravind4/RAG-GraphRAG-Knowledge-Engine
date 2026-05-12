"""
agent/grader.py — LLM-as-judge for relevance grading and hallucination detection.

Three evaluation functions:
1. grade_relevance — scores retrieval quality per chunk
2. check_hallucination — verifies answer is grounded in context
3. check_completeness — verifies answer addresses the full question
"""

from __future__ import annotations

import asyncio
from typing import Optional

from agent.schemas import HallucinationCheck, RelevanceGrade
from llm.client import LLMClient
from llm.prompts import (
    GRADER_SYSTEM,
    GRADER_USER_TEMPLATE,
    HALLUCINATION_SYSTEM,
    HALLUCINATION_USER_TEMPLATE,
)
from retrieval.vector_store import RetrievedChunk
from observability.logger import get_logger

logger = get_logger(__name__)


class RetrievalGrader:
    """
    LLM-based evaluator for retrieval quality and answer grounding.

    Uses a cheap/fast model for all grading operations to minimize cost.
    """

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def grade_relevance(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        threshold: float = 0.5,
    ) -> list[tuple[RetrievedChunk, RelevanceGrade]]:
        """
        Grade the relevance of each retrieved chunk to the query.

        Runs all grading calls concurrently for speed.

        Args:
            query: The user's question.
            chunks: Retrieved chunks to grade.
            threshold: Minimum relevance score to keep.

        Returns:
            List of (chunk, grade) tuples, filtered by threshold.
        """
        if not chunks:
            return []

        # Grade all chunks concurrently
        tasks = [self._grade_single(query, chunk) for chunk in chunks]
        grades = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for chunk, grade in zip(chunks, grades):
            if isinstance(grade, Exception):
                logger.warning("Grading failed for chunk", error=str(grade))
                # Default to keeping the chunk with moderate score
                grade = RelevanceGrade(score=0.5, reasoning="Grading failed", is_relevant=True)

            if grade.score >= threshold:
                results.append((chunk, grade))

        # Sort by relevance score
        results.sort(key=lambda x: x[1].score, reverse=True)

        logger.info(
            "Grading complete",
            total=len(chunks),
            relevant=len(results),
            filtered=len(chunks) - len(results),
        )
        return results

    async def _grade_single(
        self, query: str, chunk: RetrievedChunk
    ) -> RelevanceGrade:
        """Grade a single chunk's relevance."""
        prompt = GRADER_USER_TEMPLATE.format(query=query, chunk=chunk.text[:1000])

        return await self._llm.generate_structured(
            prompt=prompt,
            system=GRADER_SYSTEM,
            response_model=RelevanceGrade,
            fast=True,
            temperature=0.0,
        )

    async def check_hallucination(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> HallucinationCheck:
        """
        Check if the generated answer is grounded in the provided context.

        Args:
            query: The user's question.
            context: The context used for generation.
            answer: The generated answer to check.

        Returns:
            HallucinationCheck with grounding assessment.
        """
        prompt = HALLUCINATION_USER_TEMPLATE.format(
            query=query,
            context=context[:3000],  # Truncate to fit context window
            answer=answer,
        )

        try:
            result = await self._llm.generate_structured(
                prompt=prompt,
                system=HALLUCINATION_SYSTEM,
                response_model=HallucinationCheck,
                fast=True,
                temperature=0.0,
            )

            logger.info(
                "Hallucination check",
                is_grounded=result.is_grounded,
                confidence=result.confidence,
                issues=len(result.issues),
            )
            return result

        except Exception as exc:
            logger.warning("Hallucination check failed", error=str(exc))
            return HallucinationCheck(
                is_grounded=True,
                confidence=0.5,
                issues=[f"Check failed: {exc}"],
            )
