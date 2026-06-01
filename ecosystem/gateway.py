"""
RAG Ecosystem Gateway — Shared Microservice for the Kona AI Ecosystem
======================================================================
Exposes the RAG-GraphRAG-Knowledge-Engine as a shared HTTP microservice
that any project in the ecosystem can call for knowledge retrieval.

Each project gets its own namespace with pre-loaded knowledge:
  - financial   → Kronos financial docs + OHLCVA guides
  - code_review → OWASP, PEP8, ESLint, clean code best practices
  - sql         → SQL optimization, schema design patterns
  - sentiment   → Emotion analysis papers, NLP guides

Usage (as a standalone server):
    python -m ecosystem.gateway --port 9000

Usage (as a Python client):
    from ecosystem.gateway import EcosystemRAGClient

    client = EcosystemRAGClient(base_url="http://localhost:9000")
    results = client.search("What is binary spherical quantization?", namespace="financial")
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


# ── Pre-loaded knowledge bases ────────────────────────────────────────────────

KNOWLEDGE_BASES: dict[str, list[dict[str, str]]] = {
    "financial": [
        {"text": "Kronos uses Binary Spherical Quantization (BSQ) to tokenize OHLCVA candlestick data into discrete tokens.", "source": "kronos_paper"},
        {"text": "OHLCVA stands for Open, High, Low, Close, Volume, Amount — the 6 dimensions of a candlestick.", "source": "kronos_paper"},
        {"text": "RankIC (Rank Information Coefficient) measures the correlation between predicted and actual asset returns.", "source": "quant_finance"},
        {"text": "Volatility forecasting uses GARCH models as a baseline. Lower MAE = better performance.", "source": "quant_finance"},
        {"text": "Backtesting with Sharpe Ratio > 1.0 indicates risk-adjusted returns better than the risk-free rate.", "source": "portfolio_theory"},
        {"text": "Market sentiment (bullish/bearish) from social media can be used as a leading indicator for short-term price movements.", "source": "behavioral_finance"},
        {"text": "Financial time-series models should be evaluated on out-of-sample data to avoid look-ahead bias.", "source": "ml_finance"},
        {"text": "The Efficient Market Hypothesis suggests that all public information is already priced in.", "source": "academic"},
    ],
    "code_review": [
        {"text": "SQL injection occurs when user input is interpolated directly into SQL queries. Always use parameterized queries.", "source": "owasp_top10"},
        {"text": "N+1 query problem: calling the database inside a loop causes O(n) queries. Use batch fetching instead.", "source": "performance_patterns"},
        {"text": "OWASP Top 10 (2021): Injection, Broken Auth, Sensitive Data Exposure, XXE, Broken Access Control, Security Misconfig, XSS, Insecure Deserialization, Known Vulnerabilities, Insufficient Logging.", "source": "owasp"},
        {"text": "PEP 8: Use 4 spaces for indentation, max line length 79 chars, two blank lines between top-level definitions.", "source": "pep8"},
        {"text": "Type annotations (PEP 484) improve code readability and enable static analysis with mypy.", "source": "python_docs"},
        {"text": "Hardcoded credentials (passwords, API keys) should never appear in source code. Use environment variables.", "source": "security_best_practices"},
        {"text": "O(n²) complexity nested loops are a common performance issue. Consider using hash maps or sorting for O(n log n).", "source": "algorithms"},
        {"text": "Functions should do one thing and have a maximum cyclomatic complexity of 10.", "source": "clean_code"},
    ],
    "sql": [
        {"text": "Use EXPLAIN to analyze query execution plans and identify missing indexes.", "source": "sql_optimization"},
        {"text": "Index columns used in WHERE, JOIN, and ORDER BY clauses for faster query performance.", "source": "indexing_guide"},
        {"text": "Avoid SELECT * in production queries — only select columns you need to reduce I/O.", "source": "sql_best_practices"},
        {"text": "Use covering indexes (include all columns needed by the query) to avoid table lookups.", "source": "indexing_guide"},
        {"text": "Normalized schemas (3NF) reduce data redundancy but may require more JOINs.", "source": "database_design"},
        {"text": "Partitioning large tables by date or range improves query performance and data management.", "source": "advanced_sql"},
    ],
    "sentiment": [
        {"text": "RoBERTa (Robustly Optimized BERT) outperforms BERT on many NLP benchmarks due to better training.", "source": "roberta_paper"},
        {"text": "The go_emotions dataset has 28 emotion classes. SamLowe/roberta-base-go_emotions reduces to 8 practical classes.", "source": "go_emotions"},
        {"text": "Emotion classification should consider negation (not happy ≠ happy).", "source": "nlp_guide"},
        {"text": "Financial text requires domain-specific sentiment analysis distinct from general sentiment.", "source": "finbert_paper"},
        {"text": "Kafka enables real-time stream processing at 12K+ messages per second for social media monitoring.", "source": "kafka_docs"},
    ],
}


# ── Client ────────────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    text: str
    source: str
    score: float
    namespace: str


class EcosystemRAGClient:
    """
    HTTP client for the ecosystem RAG gateway.
    
    Falls back to local knowledge base if the gateway is unavailable.
    """

    def __init__(self, base_url: str = "http://localhost:9000"):
        self.base_url = base_url.rstrip("/")

    def search(
        self,
        query: str,
        namespace: str = "financial",
        top_k: int = 3,
    ) -> list[RAGResult]:
        """
        Search the ecosystem knowledge base.

        Args:
            query: Natural language question.
            namespace: Knowledge namespace (financial, code_review, sql, sentiment).
            top_k: Number of results to return.

        Returns:
            List of RAGResult with relevant knowledge chunks.
        """
        try:
            import urllib.request
            url = f"{self.base_url}/search"
            payload = json.dumps({
                "query": query,
                "namespace": namespace,
                "top_k": top_k,
            }).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return [RAGResult(**r) for r in data.get("results", [])]
        except Exception:
            # Fallback: keyword match against local knowledge base
            return self._local_search(query, namespace, top_k)

    def _local_search(self, query: str, namespace: str, top_k: int) -> list[RAGResult]:
        """Fast keyword fallback when gateway is unreachable."""
        docs = KNOWLEDGE_BASES.get(namespace, [])
        query_words = set(query.lower().split())
        scored = []
        for doc in docs:
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RAGResult(text=doc["text"], source=doc["source"],
                      score=round(score / max(len(query_words), 1), 3),
                      namespace=namespace)
            for score, doc in scored[:top_k]
        ]

    def available_namespaces(self) -> list[str]:
        """Return list of available knowledge namespaces."""
        return list(KNOWLEDGE_BASES.keys())


# ── Standalone gateway server ─────────────────────────────────────────────────

def _run_gateway_server(port: int = 9000) -> None:
    """Run a simple HTTP gateway server exposing RAG search."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class GatewayHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[RAG Gateway] {fmt % args}")

        def do_POST(self):
            if self.path == "/search":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                client = EcosystemRAGClient.__new__(EcosystemRAGClient)
                results = client._local_search(
                    query=body.get("query", ""),
                    namespace=body.get("namespace", "financial"),
                    top_k=body.get("top_k", 3),
                )
                resp = json.dumps({"results": [
                    {"text": r.text, "source": r.source, "score": r.score, "namespace": r.namespace}
                    for r in results
                ]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                resp = json.dumps({"status": "ok", "namespaces": list(KNOWLEDGE_BASES.keys())}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp)
            elif self.path == "/namespaces":
                resp = json.dumps({"namespaces": list(KNOWLEDGE_BASES.keys())}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp)

    server = HTTPServer(("0.0.0.0", port), GatewayHandler)
    print(f"[RAG Ecosystem Gateway] Running on http://0.0.0.0:{port}")
    print(f"[RAG Ecosystem Gateway] Namespaces: {', '.join(KNOWLEDGE_BASES.keys())}")
    server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Ecosystem Gateway")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    _run_gateway_server(args.port)
