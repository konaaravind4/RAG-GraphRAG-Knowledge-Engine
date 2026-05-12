"""Tests for the agent orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.orchestrator import AgentOrchestrator
from agent.memory import ConversationMemory
from agent.schemas import RouteDecision, RouteType, DecomposedQuery, RelevanceGrade, HallucinationCheck


@pytest.fixture
def orchestrator(mock_llm_client, sample_chunks, test_settings):
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=sample_chunks)
    memory = ConversationMemory(max_turns=10)
    return AgentOrchestrator(llm=mock_llm_client, retriever=mock_retriever, memory=memory, settings=test_settings)


class TestOrchestrator:

    @pytest.mark.asyncio
    async def test_direct_route(self, orchestrator, mock_llm_client):
        mock_llm_client.generate_structured = AsyncMock(
            return_value=RouteDecision(decision=RouteType.DIRECT, reasoning="Simple", confidence=0.95)
        )
        mock_llm_client.generate = AsyncMock(return_value="Hello!")
        resp = await orchestrator.run("Hello!", conversation_id="t1")
        assert resp.route_decision == "direct"
        assert len(resp.sources) == 0

    @pytest.mark.asyncio
    async def test_retrieval_route(self, orchestrator, mock_llm_client):
        route = RouteDecision(decision=RouteType.RETRIEVAL, reasoning="Needs KB", confidence=0.9)
        grade = RelevanceGrade(score=0.9, reasoning="Relevant", is_relevant=True)
        hall = HallucinationCheck(is_grounded=True, confidence=0.95, issues=[])
        mock_llm_client.generate_structured = AsyncMock(side_effect=[route, grade, grade, grade, hall])
        mock_llm_client.generate = AsyncMock(return_value="Paris is the capital [1].")
        resp = await orchestrator.run("Capital of France?", conversation_id="t2")
        assert resp.route_decision == "retrieval"
        assert len(resp.sources) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator, mock_llm_client):
        # Make ALL LLM calls fail so the top-level catch-all triggers
        mock_llm_client.generate_structured = AsyncMock(side_effect=Exception("fail"))
        mock_llm_client.generate = AsyncMock(side_effect=Exception("fail"))
        resp = await orchestrator.run("Error", conversation_id="t3")
        assert "error" in resp.answer.lower()

    @pytest.mark.asyncio
    async def test_trace_included(self, orchestrator, mock_llm_client):
        mock_llm_client.generate_structured = AsyncMock(
            return_value=RouteDecision(decision=RouteType.DIRECT, reasoning="S", confidence=0.9)
        )
        mock_llm_client.generate = AsyncMock(return_value="A")
        resp = await orchestrator.run("Test", conversation_id="t4")
        assert resp.trace is not None
        assert "steps" in resp.trace

    @pytest.mark.asyncio
    async def test_stream(self, orchestrator, mock_llm_client):
        mock_llm_client.generate_structured = AsyncMock(
            return_value=RouteDecision(decision=RouteType.DIRECT, reasoning="S", confidence=0.9)
        )
        tokens = []
        async for t in orchestrator.run_stream("Stream", conversation_id="t5"):
            tokens.append(t)
        assert len(tokens) > 0
