"""Centralized configuration via Pydantic Settings."""

from __future__ import annotations

import json
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    All values have sensible defaults so the system boots without config
    (except OPENAI_API_KEY which is required for LLM calls).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini", description="Primary generation model")
    llm_fast_model: str = Field(
        default="gpt-4o-mini", description="Cheap model for routing/grading"
    )


    embed_model: str = Field(
        default="BAAI/bge-base-en-v1.5", description="HuggingFace sentence-transformer model"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder reranker"
    )


    neo4j_url: Optional[str] = Field(default=None, description="Neo4j bolt URL")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")


    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=512)


    max_agent_iterations: int = Field(default=3, ge=1, le=10)
    max_conversation_turns: int = Field(default=20, ge=1, le=100)


    log_level: str = Field(default="INFO")
    enable_tracing: bool = Field(default=True)
    cors_origins: str = Field(default='["*"]')

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from JSON string."""
        try:
            return json.loads(self.cors_origins)
        except (json.JSONDecodeError, TypeError):
            return ["*"]

    index_persist_dir: str = Field(
        default="./data/indices", description="Directory for persisted FAISS indices"
    )


def get_settings() -> Settings:
    """Factory — cached at module level for performance."""
    return Settings()
