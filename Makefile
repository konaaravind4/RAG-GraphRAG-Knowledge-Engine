.PHONY: dev test lint typecheck docker clean install

# ─── Development ──────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ─── Quality ─────────────────────────────────────────────────────────────────

test:
	pytest -v --tb=short

test-fast:
	pytest -v --tb=short -m "not slow and not integration"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy agent retrieval ingestion llm api config observability

# ─── Docker ──────────────────────────────────────────────────────────────────

docker:
	docker build -t agentic-rag-engine .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
