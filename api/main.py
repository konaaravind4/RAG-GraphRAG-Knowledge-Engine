"""
FastAPI endpoints for the RAG GraphRAG Knowledge Engine.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag.vector_store import VectorStore
from rag.graph_retriever import GraphRetriever
from rag.hybrid_retriever import HybridRetriever
from rag.generator import RAGGenerator

logger = logging.getLogger(__name__)

# ── Global components ──────────────────────────────────────────────────────

vector_store: Optional[VectorStore] = None
graph_retriever: Optional[GraphRetriever] = None
hybrid_retriever: Optional[HybridRetriever] = None
generator: Optional[RAGGenerator] = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, graph_retriever, hybrid_retriever, generator
    vector_store = VectorStore()
    graph_retriever = GraphRetriever()
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        graph_retriever=graph_retriever,
        lambda_weight=float(os.getenv("RETRIEVAL_LAMBDA", "0.6")),
    )
    generator = RAGGenerator(provider=os.getenv("LLM_PROVIDER", "gemini"))
    yield
    graph_retriever.close()


app = FastAPI(
    title="RAG & GraphRAG Knowledge Engine",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=20)
    lambda_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    return_sources: bool = True


class IngestRequest(BaseModel):
    content: str = Field(..., min_length=10)
    source: str = ""
    chunk: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "vector_docs": len(vector_store.documents) if vector_store else 0,
    }


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    """Ingest text content into the vector + graph stores."""
    doc = Document(page_content=req.content, metadata={"source": req.source})
    docs = text_splitter.split_documents([doc]) if req.chunk else [doc]
    vector_store.add_documents(docs)
    for d in docs:
        graph_retriever.ingest_document(d)
    return {"ingested_chunks": len(docs), "source": req.source}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """Upload a text file for ingestion."""
    content = (await file.read()).decode("utf-8", errors="replace")
    doc = Document(page_content=content, metadata={"source": file.filename})
    chunks = text_splitter.split_documents([doc])
    vector_store.add_documents(chunks)
    return {"filename": file.filename, "chunks": len(chunks)}


@app.post("/query")
async def query(req: QueryRequest) -> dict:
    """Query the knowledge engine with hybrid retrieval + LLM generation."""
    if not vector_store or not generator:
        raise HTTPException(status_code=503, detail="Engine not ready")

    if req.lambda_weight is not None:
        hybrid_retriever.lambda_weight = req.lambda_weight

    chunks = hybrid_retriever.retrieve(req.query, k=req.k)
    docs = [c.document for c in chunks]
    result = generator.generate_with_sources(req.query, docs)

    return {
        **result,
        "retrieval_chunks": [
            {
                "content": c.document.page_content[:200],
                "combined_score": round(c.combined_score, 4),
                "source_type": c.source,
            }
            for c in chunks
        ],
    }


@app.post("/save-index")
async def save_index() -> dict:
    vector_store.save()
    return {"status": "saved"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
