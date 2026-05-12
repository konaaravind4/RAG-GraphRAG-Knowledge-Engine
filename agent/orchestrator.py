"""
agent/orchestrator.py — ReAct agent loop for Agentic RAG.

This is the "brain" of the system. It implements an autonomous control loop:

1. ROUTE    → Classify intent
2. PLAN     → Decompose complex queries
3. RETRIEVE → Execute retrieval
4. GRADE    → Score relevance (LLM-as-judge)
5. DECIDE   → Re-query if insufficient (max N iterations)
6. GENERATE → Produce grounded answer
7. REFLECT  → Hallucination check → loop back if failed
8. RESPOND  → Return answer + trace
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from agent.decomposer import QueryDecomposer
from agent.grader import RetrievalGrader
from agent.memory import ConversationMemory
from agent.router import QueryRouter
from agent.schemas import (
    AgentResponse,
    AgentState,
    DecomposedQuery,
    RefinedQuery,
    RouteType,
)
from config.settings import get_settings
from llm.client import LLMClient
from llm.prompts import (
    CLARIFICATION_SYSTEM,
    DIRECT_ANSWER_SYSTEM,
    GENERATOR_SYSTEM,
    GENERATOR_STREAM_SYSTEM,
    GENERATOR_USER_TEMPLATE,
    QUERY_REFINEMENT_SYSTEM,
    QUERY_REFINEMENT_USER_TEMPLATE,
)
from retrieval.hybrid import HybridRetriever, RetrievalConfig
from retrieval.vector_store import RetrievedChunk
from observability.logger import get_logger
from observability.tracer import TraceContext, TraceStepType

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    ReAct agent orchestrator for Agentic RAG.

    Manages the full lifecycle of a query:
    routing → planning → retrieval → grading → generation → reflection.

    Supports:
    - Multi-turn conversations via ConversationMemory
    - Self-correcting retrieval (re-query on low relevance)
    - Hallucination detection and retry
    - Full execution tracing for observability
    - Streaming generation

    Usage:
        orchestrator = AgentOrchestrator(
            llm=llm_client,
            retriever=hybrid_retriever,
            memory=memory,
        )
        response = await orchestrator.run("What is RAG?", conversation_id="abc")
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever,
        memory: ConversationMemory,
        settings=None,
    ):
        self._settings = settings or get_settings()
        self._llm = llm
        self._retriever = retriever
        self._memory = memory

        # Sub-components
        self._router = QueryRouter(llm)
        self._decomposer = QueryDecomposer(llm)
        self._grader = RetrievalGrader(llm)

        self._max_iterations = self._settings.max_agent_iterations

    async def run(
        self,
        query: str,
        conversation_id: str = "default",
    ) -> AgentResponse:
        """
        Execute the full agentic RAG pipeline.

        Args:
            query: User's question.
            conversation_id: ID for multi-turn context.

        Returns:
            AgentResponse with answer, sources, and trace.
        """
        trace = TraceContext(query=query, conversation_id=conversation_id)

        try:
            # Add user turn to memory
            await self._memory.add_turn(conversation_id, "user", query)
            context = await self._memory.get_context(conversation_id, last_n=5)

            # ── Step 1: ROUTE ────────────────────────────────────────────
            with trace.step(TraceStepType.ROUTE) as step:
                route = await self._router.route(query, context)
                step.output_data = {
                    "decision": route.decision.value,
                    "confidence": route.confidence,
                }
                step.token_usage = self._llm.usage.to_dict()

            # ── Handle non-retrieval routes ──────────────────────────────
            if route.decision == RouteType.DIRECT:
                return await self._handle_direct(query, conversation_id, trace)

            if route.decision == RouteType.CLARIFICATION:
                return await self._handle_clarification(query, conversation_id, trace)

            # ── Step 2: PLAN (if multi-hop) ─────────────────────────────
            queries_to_retrieve = [query]
            if route.decision == RouteType.MULTI_HOP:
                with trace.step(TraceStepType.DECOMPOSE) as step:
                    decomposed = await self._decomposer.decompose(query)
                    queries_to_retrieve = decomposed.sub_queries
                    step.output_data = {
                        "sub_queries": queries_to_retrieve,
                        "reasoning": decomposed.reasoning,
                    }

            # ── Agentic Loop: RETRIEVE → GRADE → DECIDE ────────────────
            all_relevant_chunks: list[RetrievedChunk] = []
            iteration = 0

            for iteration in range(self._max_iterations):
                # Step 3: RETRIEVE
                with trace.step(TraceStepType.RETRIEVE, iteration=iteration) as step:
                    config = RetrievalConfig(
                        top_k=5,
                        use_web_search=(iteration > 0),  # Enable web on retries
                    )
                    chunks: list[RetrievedChunk] = []
                    for sub_query in queries_to_retrieve:
                        sub_chunks = await self._retriever.retrieve(sub_query, config)
                        chunks.extend(sub_chunks)

                    step.output_data = {
                        "chunks_retrieved": len(chunks),
                        "sources": list({c.source for c in chunks}),
                    }

                if not chunks:
                    logger.warning("No chunks retrieved", iteration=iteration)
                    if iteration < self._max_iterations - 1:
                        # Refine query and retry
                        queries_to_retrieve = [await self._refine_query(query, context)]
                        continue
                    break

                # Step 4: GRADE
                with trace.step(TraceStepType.GRADE, iteration=iteration) as step:
                    graded = await self._grader.grade_relevance(query, chunks)
                    relevant_chunks = [chunk for chunk, grade in graded]
                    step.output_data = {
                        "total_chunks": len(chunks),
                        "relevant_chunks": len(relevant_chunks),
                        "scores": [
                            round(grade.score, 3) for _, grade in graded
                        ],
                    }

                # Step 5: DECIDE
                if relevant_chunks:
                    all_relevant_chunks = relevant_chunks
                    break
                else:
                    logger.info(
                        "Insufficient relevant context, refining query",
                        iteration=iteration,
                    )
                    queries_to_retrieve = [await self._refine_query(query, context)]

            # ── Step 6: GENERATE ─────────────────────────────────────────
            if not all_relevant_chunks:
                answer = (
                    "I couldn't find sufficient relevant information in the knowledge base "
                    "to answer your question. Please try rephrasing your question or "
                    "upload relevant documents first."
                )
                await self._memory.add_turn(conversation_id, "assistant", answer)
                return AgentResponse(
                    answer=answer,
                    query=query,
                    conversation_id=conversation_id,
                    route_decision=route.decision.value,
                    iterations=iteration + 1,
                    trace=trace.to_dict() if self._settings.enable_tracing else None,
                )

            with trace.step(TraceStepType.GENERATE) as step:
                context_text = self._format_context(all_relevant_chunks)
                prompt = GENERATOR_USER_TEMPLATE.format(
                    context=context_text, query=query
                )
                answer = await self._llm.generate(
                    prompt=prompt,
                    system=GENERATOR_SYSTEM,
                    temperature=0.2,
                    max_tokens=1024,
                )
                step.output_data = {"answer_length": len(answer)}
                step.token_usage = self._llm.usage.to_dict()

            # ── Step 7: REFLECT (hallucination check) ────────────────────
            with trace.step(TraceStepType.REFLECT) as step:
                hallucination = await self._grader.check_hallucination(
                    query=query,
                    context=context_text,
                    answer=answer,
                )
                step.output_data = {
                    "is_grounded": hallucination.is_grounded,
                    "confidence": hallucination.confidence,
                    "issues": hallucination.issues,
                }

                if not hallucination.is_grounded and hallucination.confidence > 0.7:
                    logger.warning(
                        "Hallucination detected, regenerating",
                        issues=hallucination.issues,
                    )
                    # Regenerate with stricter prompt
                    stricter_prompt = (
                        f"{prompt}\n\n"
                        "IMPORTANT: Only state facts directly supported by the context above. "
                        "Do not add information from your own knowledge."
                    )
                    answer = await self._llm.generate(
                        prompt=stricter_prompt,
                        system=GENERATOR_SYSTEM,
                        temperature=0.0,
                        max_tokens=1024,
                    )

            # ── Step 8: RESPOND ──────────────────────────────────────────
            await self._memory.add_turn(conversation_id, "assistant", answer)

            return AgentResponse(
                answer=answer,
                query=query,
                conversation_id=conversation_id,
                sources=[c.to_dict() for c in all_relevant_chunks],
                route_decision=route.decision.value,
                iterations=iteration + 1,
                trace=trace.to_dict() if self._settings.enable_tracing else None,
            )

        except Exception as exc:
            logger.error("Agent orchestration failed", error=str(exc))
            return AgentResponse(
                answer=f"An error occurred while processing your query: {exc}",
                query=query,
                conversation_id=conversation_id,
                trace=trace.to_dict() if self._settings.enable_tracing else None,
            )

    async def run_stream(
        self,
        query: str,
        conversation_id: str = "default",
    ) -> AsyncIterator[str]:
        """
        Stream the agent's response token-by-token.

        Yields:
            Tokens of the generated answer.
        """
        # For streaming, we do a simplified pipeline:
        # Route → Retrieve → Grade → Stream Generate
        await self._memory.add_turn(conversation_id, "user", query)
        context = await self._memory.get_context(conversation_id, last_n=5)

        # Route
        route = await self._router.route(query, context)

        # Direct answer streaming
        if route.decision == RouteType.DIRECT:
            full_answer = ""
            async for token in self._llm.generate_stream(
                prompt=query,
                system=DIRECT_ANSWER_SYSTEM,
            ):
                full_answer += token
                yield token
            await self._memory.add_turn(conversation_id, "assistant", full_answer)
            return

        # Retrieval + generation streaming
        config = RetrievalConfig(top_k=5)
        chunks = await self._retriever.retrieve(query, config)

        if not chunks:
            msg = "I couldn't find relevant information. Please try rephrasing."
            yield msg
            await self._memory.add_turn(conversation_id, "assistant", msg)
            return

        context_text = self._format_context(chunks)
        prompt = GENERATOR_USER_TEMPLATE.format(context=context_text, query=query)

        full_answer = ""
        async for token in self._llm.generate_stream(
            prompt=prompt,
            system=GENERATOR_STREAM_SYSTEM,
        ):
            full_answer += token
            yield token

        await self._memory.add_turn(conversation_id, "assistant", full_answer)

    # ── Private Helpers ──────────────────────────────────────────────────────

    async def _handle_direct(
        self, query: str, conversation_id: str, trace: TraceContext
    ) -> AgentResponse:
        """Handle queries that don't need retrieval."""
        with trace.step(TraceStepType.GENERATE) as step:
            answer = await self._llm.generate(
                prompt=query,
                system=DIRECT_ANSWER_SYSTEM,
                temperature=0.3,
            )
            step.output_data = {"answer_length": len(answer)}

        await self._memory.add_turn(conversation_id, "assistant", answer)
        return AgentResponse(
            answer=answer,
            query=query,
            conversation_id=conversation_id,
            route_decision="direct",
            trace=trace.to_dict() if self._settings.enable_tracing else None,
        )

    async def _handle_clarification(
        self, query: str, conversation_id: str, trace: TraceContext
    ) -> AgentResponse:
        """Handle ambiguous queries by asking for clarification."""
        with trace.step(TraceStepType.GENERATE) as step:
            answer = await self._llm.generate(
                prompt=query,
                system=CLARIFICATION_SYSTEM,
                temperature=0.5,
            )
            step.output_data = {"answer_length": len(answer)}

        await self._memory.add_turn(conversation_id, "assistant", answer)
        return AgentResponse(
            answer=answer,
            query=query,
            conversation_id=conversation_id,
            route_decision="clarification",
            trace=trace.to_dict() if self._settings.enable_tracing else None,
        )

    async def _refine_query(self, query: str, context: str) -> str:
        """Refine a query for better retrieval on retry."""
        try:
            prompt = QUERY_REFINEMENT_USER_TEMPLATE.format(
                query=query, context=context
            )
            refined = await self._llm.generate_structured(
                prompt=prompt,
                system=QUERY_REFINEMENT_SYSTEM,
                response_model=RefinedQuery,
                fast=True,
            )
            logger.info("Query refined", original=query[:60], refined=refined.refined_query[:60])
            return refined.refined_query
        except Exception:
            return query

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks as numbered context for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.source or "unknown"
            parts.append(f"[{i}] (Source: {source})\n{chunk.text}")
        return "\n\n".join(parts)
