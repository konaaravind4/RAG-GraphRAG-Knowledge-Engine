"""
agent/schemas.py — Internal data models for the agentic RAG pipeline.

All inter-component communication uses these Pydantic models.
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
    """Final response from the agent orchestrator."""
    answer: str
    query: str
    conversation_id: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    route_decision: Optional[str] = None
    iterations: int = 1
    trace: Optional[dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}
