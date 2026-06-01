"""
agent/schemas.py — Internal data models for the agentic RAG pipeline.

All inter-component communication uses these Pydantic models.

Enhancements:
- AgentResponse gains `tokens_used`, `cost_estimate_usd`, and `namespace` fields
  for cost tracking and multi-tenant namespace routing.
- EcosystemSearchRequest / EcosystemSearchResponse support the /ecosystem/search API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Route Decision ──────────────────────────────────────────────────────────

class RouteType(str, Enum):
    DIRECT = "direct"
    RETRIEVAL = "retrieval"
    MULTI_HOP = "multi_hop"
    CLARIFICATION = "clarification"


class RouteDecision(BaseModel):
    """Output of the intent router."""
    decision: RouteType
    reasoning: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


# ─── Query Decomposition ────────────────────────────────────────────────────

class DecomposedQuery(BaseModel):
    """Output of the query decomposer."""
    sub_queries: list[str] = Field(default_factory=list)
    reasoning: str = ""


# ─── Grading ─────────────────────────────────────────────────────────────────

class RelevanceGrade(BaseModel):
    """Relevance score for a single chunk."""
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    is_relevant: bool = True


class HallucinationCheck(BaseModel):
    """Result of hallucination detection."""
    is_grounded: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)


class CompletenessCheck(BaseModel):
    """Result of answer completeness check."""
    is_complete: bool = True
    missing_aspects: list[str] = Field(default_factory=list)


# ─── Query Refinement ───────────────────────────────────────────────────────

class RefinedQuery(BaseModel):
    """Output of query refinement."""
    refined_query: str
    reasoning: str = ""


# ─── Agent State ─────────────────────────────────────────────────────────────

class AgentState(str, Enum):
    """States in the agent's ReAct loop."""
    ROUTING = "routing"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    GRADING = "grading"
    GENERATING = "generating"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    ERROR = "error"


# ─── Conversation ────────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    """A single turn in the conversation."""
    role: str  # "user" | "assistant"
    content: str


# ─── Agent Response ──────────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    """Final response from the agent orchestrator.

    Fields:
        answer: The generated answer text.
        query: The original user query.
        conversation_id: Session / conversation identifier.
        sources: Retrieved source chunks used to ground the answer.
        route_decision: Which route the agent chose (direct/retrieval/multi_hop/clarification).
        iterations: Number of retrieve-grade loops executed.
        trace: Full execution trace for observability (None when tracing is disabled).
        tokens_used: Total tokens consumed across all LLM calls for this request.
        cost_estimate_usd: Approximate USD cost derived from tokens_used.
        namespace: Knowledge namespace this response was generated against.
    """
    answer: str
    query: str
    conversation_id: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    route_decision: Optional[str] = None
    iterations: int = 1
    trace: Optional[dict[str, Any]] = None
    tokens_used: int = 0
    cost_estimate_usd: float = 0.0
    namespace: str = "default"

    model_config = {"arbitrary_types_allowed": True}


# ─── Ecosystem Search ─────────────────────────────────────────────────────────

class EcosystemSearchRequest(BaseModel):
    """Request payload for the /ecosystem/search endpoint.

    Fields:
        query: Natural language question to search the knowledge base with.
        namespace: Knowledge namespace to search within.  Defaults to 'financial'.
        top_k: Maximum number of results to return (1–20).
    """
    query: str = Field(..., min_length=1, description="Search query")
    namespace: str = Field(default="financial", description="Knowledge namespace")
    top_k: int = Field(default=3, ge=1, le=20, description="Max results to return")


class EcosystemSearchResponse(BaseModel):
    """Response payload for the /ecosystem/search endpoint.

    Fields:
        results: List of matching knowledge chunks with text, source, score.
        namespace: The namespace that was searched.
        count: Number of results returned.
    """
    results: list[dict[str, Any]] = Field(default_factory=list)
    namespace: str
    count: int = 0
