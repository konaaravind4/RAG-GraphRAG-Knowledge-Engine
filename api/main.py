"""
api/main.py — FastAPI application for the Agentic RAG Knowledge Engine.

Endpoints:
    POST /chat             — conversational Q&A with full agent pipeline
    POST /chat/stream      — SSE streaming version
    POST /ingest/file      — upload documents (multipart form)
    POST /ingest/url       — ingest from URL
    POST /ingest/text      — ingest raw text
    GET  /documents        — list indexed documents
    GET  /health           — health check with component status
    POST /ecosystem/search — search ecosystem knowledge namespaces
    GET  /ecosystem/namespaces — list available namespaces
    GET  /stats            — orchestrator runtime statistics
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent.memory import ConversationMemory
from agent.orchestrator import AgentOrchestrator
from agent.schemas import EcosystemSearchRequest, EcosystemSearchResponse
from api.middleware import RequestTrackingMiddleware
from api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    IngestURLRequest,
    SourceChunk,
)
from config.settings import get_settings
from ecosystem.gateway import EcosystemRAGClient, KNOWLEDGE_BASES
from ingestion.pipeline import IngestionPipeline
from llm.client import LLMClient
from retrieval.graph_store import GraphStore
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore
from retrieval.web_search import WebSearcher
from observability.logger import get_logger, setup_logging

logger = get_logger(__name__)



_orchestrator: AgentOrchestrator | None = None
_pipeline: IngestionPipeline | None = None
_vector_store: VectorStore | None = None
_settings = None
# Shared ecosystem RAG client (uses local fallback when gateway is down)
_ecosystem_client: EcosystemRAGClient = EcosystemRAGClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    global _orchestrator, _pipeline, _vector_store, _settings

    _settings = get_settings()
    setup_logging(_settings.log_level)

    logger.info("Initializing Agentic RAG Engine …")


    _vector_store = VectorStore(model_name=_settings.embed_model)

    # Try to load persisted index
    if _vector_store.load(_settings.index_persist_dir):
        logger.info("Loaded persisted index", documents=_vector_store.document_count)


    graph_store = GraphStore(
        url=_settings.neo4j_url,
        username=_settings.neo4j_user,
        password=_settings.neo4j_password,
    )

   
    web_searcher = WebSearcher(max_results=5)


    reranker = Reranker(model_name=_settings.reranker_model)

   
    retriever = HybridRetriever(
        vector_store=_vector_store,
        graph_store=graph_store if graph_store.is_available else None,
        web_searcher=web_searcher,
        reranker=reranker,
    )


    llm = LLMClient(settings=_settings)

    
    memory = ConversationMemory(max_turns=_settings.max_conversation_turns)

    
    _orchestrator = AgentOrchestrator(
        llm=llm,
        retriever=retriever,
        memory=memory,
        settings=_settings,
    )

    
    _pipeline = IngestionPipeline(
        vector_store=_vector_store,
        settings=_settings,
    )

    logger.info(
        "Agentic RAG Engine ready",
        graph_available=graph_store.is_available,
        indexed_documents=_vector_store.document_count,
    )

    yield

    # Cleanup
    if _vector_store and _vector_store.is_built:
        _vector_store.save(_settings.index_persist_dir)
        logger.info("Index persisted on shutdown")

    if graph_store:
        graph_store.close()




app = FastAPI(
    title="Agentic RAG Knowledge Engine",
    description=(
        "Production-level Agentic RAG with ReAct orchestration, hybrid retrieval "
        "(FAISS + Neo4j + Web), self-reflective grading, and streaming generation."
    ),
    version="2.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse},
    },
)

# Middleware
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check with component status."""
    return HealthResponse(
        status="ok",
        version="2.0.0",
        components={
            "vector_store": {
                "status": "ok" if _vector_store else "not_initialized",
                "document_count": _vector_store.document_count if _vector_store else 0,
            },
            "orchestrator": "ok" if _orchestrator else "not_initialized",
            "ingestion": "ok" if _pipeline else "not_initialized",
        },
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Ask a question with full agentic RAG pipeline.

    The agent will:
    1. Route the query (direct / retrieval / multi-hop / clarification)
    2. Decompose complex queries into sub-queries
    3. Retrieve from vector store, graph, and optionally web
    4. Grade relevance of retrieved chunks
    5. Generate a grounded answer with citations
    6. Self-check for hallucinations
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    conversation_id = req.conversation_id or uuid.uuid4().hex[:8]

    response = await _orchestrator.run(
        query=req.query,
        conversation_id=conversation_id,
    )

    return ChatResponse(
        answer=response.answer,
        query=response.query,
        conversation_id=conversation_id,
        sources=[
            SourceChunk(**s) for s in response.sources
        ] if response.sources else [],
        route_decision=response.route_decision,
        iterations=response.iterations,
        trace=response.trace,
    )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """
    Stream the agent's response token-by-token via Server-Sent Events.
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    conversation_id = req.conversation_id or uuid.uuid4().hex[:8]

    async def event_generator():
        async for token in _orchestrator.run_stream(
            query=req.query,
            conversation_id=conversation_id,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a document (PDF, TXT, MD)."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    content = await file.read()
    result = await _pipeline.ingest_file(
        content=content,
        filename=file.filename or "unnamed",
        content_type=file.content_type,
    )

    return IngestResponse(**result.to_dict())


@app.post("/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_url(req: IngestURLRequest):
    """Ingest content from a URL."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result = await _pipeline.ingest_url(req.url)
    return IngestResponse(**result.to_dict())


@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(req: IngestTextRequest):
    """Ingest raw text documents."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result = await _pipeline.ingest_texts(
        texts=req.documents,
        source=req.source,
    )
    return IngestResponse(**result.to_dict())


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    if not _vector_store:
        raise HTTPException(status_code=503, detail="Store not initialized")

    sources = _pipeline.get_ingested_sources() if _pipeline else []

    return DocumentListResponse(
        total_documents=len(sources),
        total_chunks=_vector_store.document_count,
        sources=sources,
    )




@app.post(
    "/ecosystem/search",
    response_model=EcosystemSearchResponse,
    tags=["Ecosystem"],
    summary="Search a knowledge namespace",
)
async def ecosystem_search(req: EcosystemSearchRequest) -> EcosystemSearchResponse:
    """Search the ecosystem knowledge base for a given namespace.

    Uses the :class:`~ecosystem.gateway.EcosystemRAGClient` with a local
    keyword-match fallback so the endpoint works even when the external
    gateway is unavailable.

    Args:
        req: Request body with *query*, *namespace*, and *top_k* fields.

    Returns:
        :class:`~agent.schemas.EcosystemSearchResponse` with matched chunks.
    """
    if req.namespace not in KNOWLEDGE_BASES:
        raise HTTPException(
            status_code=404,
            detail=f"Namespace '{req.namespace}' not found. "
                   f"Available: {list(KNOWLEDGE_BASES.keys())}",
        )

    logger.info(
        "Ecosystem search",
        namespace=req.namespace,
        query=req.query[:80],
        top_k=req.top_k,
    )

    results = _ecosystem_client._local_search(
        query=req.query,
        namespace=req.namespace,
        top_k=req.top_k,
    )

    serialised = [
        {"text": r.text, "source": r.source, "score": r.score, "namespace": r.namespace}
        for r in results
    ]
    return EcosystemSearchResponse(
        results=serialised,
        namespace=req.namespace,
        count=len(serialised),
    )


@app.get(
    "/ecosystem/namespaces",
    tags=["Ecosystem"],
    summary="List available knowledge namespaces",
)
async def ecosystem_namespaces() -> dict:
    """Return all available knowledge namespaces in the ecosystem.

    Returns:
        JSON object ``{"namespaces": [...]}``.  The list is derived directly
        from the in-process ``KNOWLEDGE_BASES`` dict so it is always accurate.
    """
    namespaces = _ecosystem_client.available_namespaces()
    logger.debug("Listing ecosystem namespaces", count=len(namespaces))
    return {"namespaces": namespaces}





@app.get(
    "/stats",
    tags=["System"],
    summary="Orchestrator runtime statistics",
)
async def stats() -> dict:
    """Return live orchestrator telemetry.

    Exposes the counters maintained by :attr:`AgentOrchestrator.stats`:

    - **request_count** — total ``/chat`` requests handled since startup.
    - **total_tokens**  — cumulative LLM tokens consumed.
    - **avg_tokens_per_request** — mean tokens per request.

    Returns:
        503 if the orchestrator has not been initialized yet.
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _orchestrator.stats
