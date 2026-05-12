"""
agent/router.py — Intent classifier for query routing.

Classifies queries into: direct, retrieval, multi_hop, or clarification.
Uses LLM structured output for reliable classification.
"""

from __future__ import annotations

from agent.schemas import RouteDecision, RouteType
from llm.client import LLMClient
from llm.prompts import ROUTER_SYSTEM, ROUTER_USER_TEMPLATE
from observability.logger import get_logger

logger = get_logger(__name__)


class QueryRouter:
    """
    Routes queries to the appropriate processing path.

    Uses a cheap/fast LLM call with structured output to classify intent.
    """

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def route(
        self,
        query: str,
        conversation_context: str = "",
        n_turns: int = 0,
    ) -> RouteDecision:
        """
        Classify the intent of a query.

        Args:
            query: The user's question.
            conversation_context: Recent conversation history for context.
            n_turns: Number of conversation turns included.

        Returns:
            RouteDecision with the classification.
        """
        prompt = ROUTER_USER_TEMPLATE.format(
            query=query,
            context=conversation_context or "No previous context.",
            n_turns=n_turns,
        )

        try:
            decision = await self._llm.generate_structured(
                prompt=prompt,
                system=ROUTER_SYSTEM,
                response_model=RouteDecision,
                fast=True,
            )

            logger.info(
                "Query routed",
                decision=decision.decision.value,
                confidence=decision.confidence,
                reasoning=decision.reasoning[:100],
            )
            return decision

        except Exception as exc:
            logger.warning("Router failed, defaulting to retrieval", error=str(exc))
            return RouteDecision(
                decision=RouteType.RETRIEVAL,
                reasoning=f"Router error ({exc}), defaulting to retrieval",
                confidence=0.5,
            )
