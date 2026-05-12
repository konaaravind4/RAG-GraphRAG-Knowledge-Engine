"""
agent/decomposer.py — Query decomposition for multi-hop questions.

Breaks complex queries into atomic sub-queries for independent retrieval.
"""

from __future__ import annotations

from agent.schemas import DecomposedQuery
from llm.client import LLMClient
from llm.prompts import DECOMPOSER_SYSTEM, DECOMPOSER_USER_TEMPLATE
from observability.logger import get_logger

logger = get_logger(__name__)


class QueryDecomposer:
    """
    Decomposes complex queries into simpler atomic sub-queries.

    Used when the router classifies a query as "multi_hop".
    Each sub-query can be independently retrieved and answered.
    """

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    async def decompose(self, query: str) -> DecomposedQuery:
        """
        Decompose a complex query into 2-5 atomic sub-queries.

        Args:
            query: The complex user question.

        Returns:
            DecomposedQuery with sub_queries and reasoning.
        """
        prompt = DECOMPOSER_USER_TEMPLATE.format(query=query)

        try:
            result = await self._llm.generate_structured(
                prompt=prompt,
                system=DECOMPOSER_SYSTEM,
                response_model=DecomposedQuery,
                fast=True,
            )

            # Ensure we have at least the original query
            if not result.sub_queries:
                result.sub_queries = [query]

            logger.info(
                "Query decomposed",
                original=query[:80],
                sub_queries=len(result.sub_queries),
            )
            return result

        except Exception as exc:
            logger.warning("Decomposition failed, using original query", error=str(exc))
            return DecomposedQuery(
                sub_queries=[query],
                reasoning=f"Decomposition failed ({exc}), using original query",
            )
