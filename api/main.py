"""
api/main.py — FastAPI REST service for the RAG + GraphRAG Knowledge Engine.

Endpoints:
    POST /ask          — ask a question and get a grounded answer
    POST /index        — index a list of documents
    GET  /health       — health check
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from rag.retriever import VectorRetriever, HybridRetriever, RetrievalConfig
from rag.generator import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Global State ─────────────────────────────────────────────────────────────

_retriever: Optional[HybridRetriever] = None
_vector_store: Optional[VectorRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _vector_store
    logger.info("Initialising RAG retriever …")
    _vector_store = VectorRetriever(
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    )
    _retriever = HybridRetriever(
        vector_retriever=_vector_store,
        graph_url=os.getenv("NEO4J_URL"),
        graph_user=os.getenv("NEO4J_USER", "neo4j"),
        graph_password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    logger.info("RAG engine ready.")
    yield


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG & GraphRAG Knowledge Engine",
    description="Hybrid retrieval-augmented generation: FAISS vector search + Neo4j graph traversal.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    documents: list[str] = Field(..., min_length=1, max_length=10000)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    lambda_: float = Field(0.6, ge=0.0, le=1.0)
    model: str = Field("gpt-4o-mini")


class ChunkResponse(BaseModel):
    text: str
    source: str
    score: float
    retrieval_method: str


class AskResponse(BaseModel):
    query: str
    answer: str
    chunks: list[ChunkResponse]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "indexed_docs": len(_vector_store._docs) if _vector_store else 0}


@app.post("/index")
async def index_documents(req: IndexRequest):
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised.")
    try:
        _vector_store.build(req.documents)
        return {"indexed": len(req.documents)}
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised.")
    if not _vector_store or not _vector_store._docs:
        raise HTTPException(status_code=400, detail="No documents indexed. POST /index first.")
    try:
        cfg = RetrievalConfig(top_k=req.top_k, lambda_=req.lambda_)
        chunks = _retriever.retrieve(req.query, cfg)
        context_texts = [c.text for c in chunks]
        answer = generate_answer(req.query, context_texts, model=req.model)
        return AskResponse(
            query=req.query,
            answer=answer,
            chunks=[ChunkResponse(**c.__dict__) for c in chunks],
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))
