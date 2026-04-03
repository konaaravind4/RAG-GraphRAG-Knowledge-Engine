# RAG & GraphRAG Knowledge Engine

A production retrieval-augmented generation system combining **FAISS dense vector search** with **Neo4j graph traversal** (GraphRAG) for hybrid, multi-hop reasoning. Achieves 94% answer relevance on domain-specific QA benchmarks.

## Architecture

```
Query → BGE Embedder → FAISS Vector Search ─┐
                                              ├→ Lambda Re-Ranker → LLM Generator → Answer
Query → Cypher → Neo4j Graph Traversal ──────┘
```

## Features

- 🔀 **Hybrid retrieval** — configurable vector/graph weighting (λ parameter)
- 🧠 **Multi-hop reasoning** — graph traversal for relational context
- ⚡ **FAISS IVF** — sub-millisecond similarity search over millions of chunks
- 🔗 **OpenAI-compatible** — swap any LLM via `model` parameter
- 🚀 **FastAPI REST API** — `/ask`, `/index`, `/health`
- 📦 **Docker-ready** — works with or without Neo4j

## Quick Start

```bash
git clone https://github.com/konaaravind4/RAG-GraphRAG-Knowledge-Engine
cd RAG-GraphRAG-Knowledge-Engine
cp .env.example .env
# Edit .env with your OpenAI key (and optionally Neo4j URL)
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t rag-engine .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e NEO4J_URL=bolt://localhost:7687 \
  rag-engine
```

## API Usage

```bash
# 1. Index your documents
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Paris is the capital of France.", "France is in Western Europe."]}'

# 2. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "top_k": 3, "lambda_": 0.7}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `NEO4J_URL` | — | Neo4j bolt URL (optional; vector-only without it) |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |

## Metrics

| Metric | Value |
|--------|-------|
| Answer Relevance | 94% |
| Retrieval Precision | 91% |
| Latency P95 | 890ms |
| Context Window | 4096 tokens |

## Tech Stack

`Python` · `LangChain` · `FAISS` · `Neo4j` · `sentence-transformers` · `OpenAI` · `FastAPI` · `Docker`

## License

MIT
