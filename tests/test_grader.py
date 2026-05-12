"""Tests for agent/grader.py — relevance grading and hallucination detection."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent.grader import RetrievalGrader
from agent.schemas import RelevanceGrade, HallucinationCheck
from retrieval.vector_store import RetrievedChunk, ChunkMetadata


class TestRetrievalGrader:
    """Tests for the RetrievalGrader class."""

    @pytest.mark.asyncio
    async def test_grade_relevance_filters_low_scores(self, mock_llm_client, sample_chunks):
        """Chunks below threshold should be filtered out."""
        # Make structured output return alternating relevant/irrelevant
        grades = [
            RelevanceGrade(score=0.9, reasoning="Very relevant", is_relevant=True),
            RelevanceGrade(score=0.8, reasoning="Relevant", is_relevant=True),
            RelevanceGrade(score=0.2, reasoning="Not relevant", is_relevant=False),
        ]
        mock_llm_client.generate_structured = AsyncMock(side_effect=grades)

        grader = RetrievalGrader(mock_llm_client)
        results = await grader.grade_relevance("What is the capital of France?", sample_chunks)

        # Only 2 chunks should pass (score >= 0.5)
        assert len(results) == 2
        assert results[0][1].score >= results[1][1].score  # Sorted by score

    @pytest.mark.asyncio
    async def test_grade_relevance_empty_chunks(self, mock_llm_client):
        """Empty chunk list should return empty."""
        grader = RetrievalGrader(mock_llm_client)
        results = await grader.grade_relevance("query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_grade_relevance_handles_errors(self, mock_llm_client, sample_chunks):
        """Failed grading should default to keeping the chunk."""
        mock_llm_client.generate_structured = AsyncMock(
            side_effect=Exception("API error")
        )

        grader = RetrievalGrader(mock_llm_client)
        results = await grader.grade_relevance("query", sample_chunks)

        # All chunks should be kept with default score
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_check_hallucination_grounded(self, mock_llm_client):
        """Test hallucination check when answer is grounded."""
        mock_llm_client.generate_structured = AsyncMock(
            return_value=HallucinationCheck(
                is_grounded=True,
                confidence=0.95,
                issues=[],
            )
        )

        grader = RetrievalGrader(mock_llm_client)
        result = await grader.check_hallucination(
            query="What is Paris?",
            context="Paris is the capital of France.",
            answer="Paris is the capital of France.",
        )

        assert result.is_grounded is True
        assert result.confidence > 0.9
        assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_check_hallucination_detected(self, mock_llm_client):
        """Test hallucination detection."""
        mock_llm_client.generate_structured = AsyncMock(
            return_value=HallucinationCheck(
                is_grounded=False,
                confidence=0.85,
                issues=["Claims Paris has 10 million people, not in context"],
            )
        )

        grader = RetrievalGrader(mock_llm_client)
        result = await grader.check_hallucination(
            query="Population of Paris?",
            context="Paris is the capital of France.",
            answer="Paris has 10 million people.",
        )

        assert result.is_grounded is False
        assert len(result.issues) > 0

    @pytest.mark.asyncio
    async def test_check_hallucination_error_fallback(self, mock_llm_client):
        """Hallucination check error should default to grounded."""
        mock_llm_client.generate_structured = AsyncMock(
            side_effect=Exception("API error")
        )

        grader = RetrievalGrader(mock_llm_client)
        result = await grader.check_hallucination(
            query="q", context="c", answer="a"
        )

        assert result.is_grounded is True  # Safe default
        assert result.confidence == 0.5
