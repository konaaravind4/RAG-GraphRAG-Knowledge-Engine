#  Agentic RAG Knowledge Engine

A **production-level Agentic Retrieval-Augmented Generation** system built from scratch. Unlike traditional RAG pipelines that blindly retrieve-then-generate, this system uses an autonomous **ReAct agent loop** that plans, retrieves, evaluates, and self-corrects — producing grounded, hallucination-checked answers.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI Gateway                            │
│  POST /chat · POST /chat/stream · POST /ingest · GET /health    │
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

## What Makes It "Agentic"

| Feature | Traditional RAG | This System |
|---------|----------------|-------------|
| Query handling | Single retrieval pass | Intent routing → adaptive strategy |
| Complex queries | Fails silently | Decomposes into sub-queries |
| Bad retrieval | Returns low-quality answer | Self-corrects: refines query, retries, adds web search |
| Hallucination | Undetected | LLM-as-judge verification + regeneration |
| Observability | Black box | Full trace of every decision step |
| Conversation | Stateless | Multi-turn memory with sliding window |

## Features

-  **ReAct Agent Loop** — autonomous planning, retrieval, evaluation, and self-correction
-  **Hybrid Retrieval** — FAISS vector + Neo4j graph + DuckDuckGo web, merged via Reciprocal Rank Fusion
-  **Cross-Encoder Reranking** — precision boost on top-N candidates
-  **LLM-as-Judge Grading** — concurrent relevance scoring with filtering
-  **Hallucination Detection** — self-reflective grounding verification
-  **Query Decomposition** — multi-hop question breakdown
-  **Document Ingestion** — PDF, text, markdown, URL parsing with intelligent chunking
-  **Multi-Turn Conversations** — sliding window memory per conversation
-  **Streaming Responses** — Server-Sent Events for real-time generation
-  **Observability** — structured JSON tracing of every agent step
-  **Docker-Ready** — multi-stage build, non-root user, healthchecks
-  **CI/CD** — GitHub Actions: lint, type check, tests, Docker build

## Quick Start

### Local Development

```bash
git clone https://github.com/konaaravind4/RAG-GraphRAG-Knowledge-Engine
cd RAG-GraphRAG-Knowledge-Engine

# Setup
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Install
pip install -r requirements.txt

# Run
make dev
# Server starts at http://localhost:8000
```

### Docker

```bash
# App only
docker build -t agentic-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... agentic-rag

# Full stack (app + Neo4j)
docker compose up -d
```

## API Usage

### 1. Ingest Documents

```bash
# Upload a PDF
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@report.pdf"

# Ingest from URL
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}'

# Ingest raw text
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."]}'
```

### 2. Ask Questions

```bash
# Standard chat (full agent pipeline)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "conversation_id": "session1"}'
```

**Response:**
```json
{
  "answer": "Paris is the capital of France [1].",
  "query": "What is the capital of France?",
  "conversation_id": "session1",
  "sources": [
    {
      "text": "Paris is the capital of France.",
      "score": 0.9512,
      "source": "manual_0",
      "retrieval_method": "vector"
    }
  ],
  "route_decision": "retrieval",
  "iterations": 1,
  "trace": {
    "trace_id": "a1b2c3d4e5f6",
    "steps": [
      {"step_type": "route", "duration_ms": 234.5},
      {"step_type": "retrieve", "duration_ms": 45.2},
      {"step_type": "grade", "duration_ms": 312.1},
      {"step_type": "generate", "duration_ms": 567.3},
      {"step_type": "reflect", "duration_ms": 289.8}
    ]
  }
}
```

### 3. Streaming

```bash
curl -N http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain RAG in detail"}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Primary generation model |
| `LLM_FAST_MODEL` | `gpt-4o-mini` | Cheap model for routing/grading |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `NEO4J_URL` | — | Neo4j bolt URL (optional) |
| `CHUNK_SIZE` | `512` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between chunks |
| `MAX_AGENT_ITERATIONS` | `3` | Max retrieval retry loops |
| `ENABLE_TRACING` | `true` | Include execution trace in responses |

## Project Structure

```
├── agent/                    #    Agent brain
│   ├── orchestrator.py       #    ReAct control loop
│   ├── router.py             #    Intent classification
│   ├── decomposer.py         #    Multi-hop query decomposition
│   ├── grader.py             #    Relevance grading + hallucination check
│   ├── memory.py             #    Conversation memory
│   └── schemas.py            #    Internal data models
├── retrieval/                #    Multi-source retrieval
│   ├── vector_store.py       #    FAISS dense retriever
│   ├── graph_store.py        #    Neo4j graph traversal
│   ├── web_search.py         #    DuckDuckGo web search
│   ├── hybrid.py             #    RRF multi-source fusion
│   └── reranker.py           #    Cross-encoder reranker
├── ingestion/                #    Document processing
│   ├── parser.py             #    PDF / text / URL parsing
│   ├── chunker.py            #    Intelligent text chunking
│   └── pipeline.py           #    End-to-end ingestion
├── llm/                      #    LLM abstraction
│   ├── client.py             #    OpenAI client with retry + streaming
│   └── prompts.py            #    All system prompts (versioned)
├── api/                      #    FastAPI application
│   ├── main.py               #    Endpoints + lifespan
│   ├── schemas.py            #    API models
│   └── middleware.py         #    Request tracking
├── config/                   #    Configuration
│   └── settings.py           #    Pydantic Settings
├── observability/            #    Monitoring
│   ├── tracer.py             #    Agent trace capture
│   └── logger.py             #    Structured JSON logging
├── tests/                    #    Test suite
├── Dockerfile                # Multi-stage, non-root
├── docker-compose.yml        # Full stack
├── Makefile                  # Dev commands
└── .github/workflows/ci.yml  # CI pipeline
```

## Development

```bash
make install        # Install dependencies
make dev            # Run dev server with hot-reload
make test           # Run all tests
make test-fast      # Skip slow/integration tests
make lint           # Lint with ruff
make format         # Auto-format code
make docker         # Build Docker image
make docker-up      # Start full stack
```

## Tech Stack

`Python 3.11` · `FastAPI` · `OpenAI` · `FAISS` · `sentence-transformers` · `Neo4j` · `Pydantic` · `structlog` · `Docker` · `GitHub Actions`

## License

MIT
