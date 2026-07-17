"""
api/schemas.py — API request/response models.

Separated from internal agent schemas to maintain API contract stability.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field




class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    query: str = Field(..., min_length=1, max_length=4000, description="User question")
    conversation_id: Optional[str] = Field(
        default=None, description="ID for multi-turn conversation"
    )

    model_config = {"json_schema_extra": {
        "examples": [{"query": "What is retrieval-augmented generation?"}]
    }}


class SourceChunk(BaseModel):
    """A source chunk used to generate the answer."""
    text: str
    score: float
    source: str
    retrieval_method: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""
    answer: str
    query: str
    conversation_id: str
    sources: list[SourceChunk] = Field(default_factory=list)
    route_decision: Optional[str] = None
    iterations: int = 1
    trace: Optional[dict[str, Any]] = None




class IngestURLRequest(BaseModel):
    """Request body for /ingest/url."""
    url: str = Field(..., description="URL to ingest")


class IngestTextRequest(BaseModel):
    """Request body for /ingest/text."""
    documents: list[str] = Field(
        ..., min_length=1, max_length=1000, description="List of text documents"
    )
    source: str = Field(default="manual", description="Source identifier")


class IngestResponse(BaseModel):
    """Response body for ingestion endpoints."""
    documents_ingested: int
    total_chunks: int
    errors: int = 0
    documents: list[dict[str, Any]] = Field(default_factory=list)
    error_details: list[dict[str, str]] = Field(default_factory=list)




class DocumentListResponse(BaseModel):
    """Response body for /documents."""
    total_documents: int
    total_chunks: int
    sources: list[str]




class HealthResponse(BaseModel):
    """Response body for /health."""
    status: str
    version: str
    components: dict[str, Any] = Field(default_factory=dict)




class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    detail: str = ""
    request_id: Optional[str] = None
