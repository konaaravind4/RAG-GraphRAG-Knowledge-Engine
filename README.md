# Agentic RAG Knowledge Engine 🧠

> **Production-level Agentic RAG with ReAct orchestration, hybrid retrieval (FAISS + Neo4j + Web), self-reflective grading, hallucination detection, streaming generation, and shared Ecosystem Gateway.**

[![CI](https://github.com/konaaravind4/RAG-GraphRAG-Knowledge-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/konaaravind4/RAG-GraphRAG-Knowledge-Engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal)](https://fastapi.tiangolo.com)
[![Stars](https://img.shields.io/github/stars/konaaravind4/RAG-GraphRAG-Knowledge-Engine?style=social)](https://github.com/konaaravind4/RAG-GraphRAG-Knowledge-Engine)

A **production-level Agentic Retrieval-Augmented Generation** system built from scratch. Unlike traditional RAG pipelines that blindly retrieve-then-generate, this system uses an autonomous **ReAct agent loop** that plans, retrieves, evaluates, and self-corrects — producing grounded, hallucination-checked answers.

Now also serves as the **shared knowledge microservice** for the entire Kona AI Ecosystem.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI Gateway                            │
│  POST /chat · POST /chat/stream · POST /ingest · GET /health    │
│  POST /ecosystem/search · GET /ecosystem/namespaces (NEW)        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Agent Orchestrator  │  ◄── ReAct control loop
                 │  (The "Brain")       │
                 └──┬───┬───┬───┬──────┘
                    │   │   │   │
        ┌───────────┘   │   │   └────────────┐
        ▼               ▼   ▼                ▼
  ┌──────────┐   ┌──────────┐  ┌──────────┐  ┌────────────┐
  │  Router  │   │  Query   │  │ Retriever│  │  Evaluator │
  │          │   │  Decomp  │  │ (Hybrid) │  │  (Grader)  │
  └──────────┘   └──────────┘  └──────────┘  └────────────┘
                                    │
                       ┌────────────┼────────────┐
                       ▼            ▼            ▼
                 ┌──────────┐ ┌────────┐ ┌──────────┐
                 │  FAISS   │ │ Neo4j  │ │  Web     │
                 │  Vector  │ │ Graph  │ │  Search  │
                 └──────────┘ └────────┘ └──────────┘
                       │
                       ▼
               ┌──────────────┐
               │ KonaDB Vector│  ◄── Alternative FAISS backend (NEW)
               │ (Persistent) │
               └──────────────┘
```

### Agent Loop (ReAct Pattern)

```
1. ROUTE    → Classify intent: "retrieval" | "direct" | "multi_hop" | "clarification"
2. PLAN     → If multi_hop → decompose into atomic sub-queries
3. RETRIEVE → Execute sub-queries against vector / graph / web tools
4. GRADE    → LLM-as-judge scores each chunk for relevance (concurrent)
5. DECIDE   → If context insufficient → refine query and retry (max 3 loops)
6. GENERATE → Produce grounded answer with [1], [2] citations
7. REFLECT  → Self-check for hallucination → regenerate if detected
8. RESPOND  → Return answer + sources + full execution trace
```

---

## ✨ What Makes It "Agentic"

| Feature | Traditional RAG | This System |
|---------|----------------|-------------|
| Query handling | Single retrieval pass | Intent routing → adaptive strategy |
| Complex queries | Fails silently | Decomposes into sub-queries |
| Bad retrieval | Returns low-quality answer | Self-corrects: refines, retries, adds web search |
| Hallucination | Undetected | LLM-as-judge verification + regeneration |
| Observability | Black box | Full trace of every decision |
| **Ecosystem** | Isolated | **Shared gateway for all projects (NEW)** |
| **Storage** | FAISS only | **FAISS + Neo4j + KonaDB (NEW)** |

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/konaaravind4/RAG-GraphRAG-Knowledge-Engine
cd RAG-GraphRAG-Knowledge-Engine
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, NEO4J_URI, NEO4J_PASSWORD
```

### 3. Ingest Documents

```bash
# Ingest a PDF or text file
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "docs/kronos_paper.pdf", "namespace": "financial"}'
```

### 4. Chat

```bash
# Single query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Binary Spherical Quantization in Kronos?"}'

# Streaming response
curl -X POST http://localhost:8000/chat/stream \
  -d '{"query": "Explain the ReAct loop", "stream": true}'
```

### 5. Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Agentic RAG query (blocking) |
| `POST` | `/chat/stream` | Streaming RAG response |
| `POST` | `/ingest` | Add documents to knowledge base |
| `GET`  | `/health` | Health check + component status |
| `POST` | `/ecosystem/search` | Search by namespace (for ecosystem projects) |
| `GET`  | `/ecosystem/namespaces` | List available namespaces |

```bash
# Ecosystem search (used by Code Review Bot, Kronos, etc.)
curl -X POST http://localhost:8000/ecosystem/search \
  -d '{"query": "SQL injection prevention", "namespace": "code_review", "top_k": 3}'

# Response
{
  "results": [
    {
      "text": "SQL injection occurs when user input is interpolated directly...",
      "source": "owasp_top10",
      "score": 0.94,
      "namespace": "code_review"
    }
  ]
}
```

---

## 🗃️ KonaDB Vector Backend (New!)

Use [KonaDB](https://github.com/konaaravind4/kona-db) as a fully persistent vector store instead of FAISS:

```python
# In config/.env
VECTOR_BACKEND=kona          # Options: faiss (default), kona
KONA_DB_PATH=knowledge.kona
```

```python
# Programmatically
from kona.vector import VectorStore
import kona

conn = kona.connect("knowledge.kona")
vs = VectorStore(conn, namespace="financial")

# Embeddings persist across restarts — no need to re-index!
vs.add(embedding=my_embedding, text="Kronos uses BSQ tokenization", metadata={"source": "paper"})
```

---

## 🌍 Ecosystem Gateway (New!)

This system is the **knowledge backbone** of the Kona AI Ecosystem:

```
Agentic-Code-Review-Bot ──► /ecosystem/search?namespace=code_review
Kronos Reproduction      ──► /ecosystem/search?namespace=financial
AI-SQL-Data-Analyst      ──► /ecosystem/search?namespace=sql
Sentiment Dashboard      ──► /ecosystem/search?namespace=sentiment
```

Run as a standalone knowledge microservice:

```bash
python -m ecosystem.gateway --port 9000
```

Pre-loaded namespaces:
- **financial** — Kronos, OHLCVA, quant finance, backtesting
- **code_review** — OWASP, PEP8, ESLint, clean code patterns
- **sql** — Query optimization, indexing, schema design
- **sentiment** — RoBERTa, emotion analysis, Kafka streaming

---

## 🐳 Docker

```bash
docker-compose up
```

```yaml
# docker-compose.yml includes:
# - API server (port 8000)
# - Neo4j graph database (port 7687)
# - Ecosystem Gateway (port 9000)
```

---

## 🤝 Related Projects

| Project | How It Uses This |
|---------|-----------------|
| [kona-db](https://github.com/konaaravind4/kona-db) | Alternative FAISS backend for vector storage |
| [Agentic-Code-Review-Bot](https://github.com/konaaravind4/Agentic-Code-Review-Bot) | Fetches code review best practices |
| [kronos-reproduction](https://github.com/konaaravind4/kronos-reproduction) | Financial knowledge retrieval |
| [AI-SQL-Data-Analyst](https://github.com/konaaravind4/AI-SQL-Data-Analyst) | SQL optimization knowledge |
| [Real-time-Sentiment-Intelligence-Dashboard](https://github.com/konaaravind4/Real-time-Sentiment-Intelligence-Dashboard) | Contextualizes emotions with events |

---

## 📄 License

MIT © [konaaravind4](https://github.com/konaaravind4)
