"""
llm/client.py — Unified LLM client wrapping OpenAI API.

Features:
- Structured output (JSON mode / response_format)
- Streaming via async generators
- Retry with exponential backoff
- Token usage tracking
- Model routing (cheap vs. expensive)
"""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import Any, AsyncIterator, Optional, Type, TypeVar

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from config.settings import get_settings
from observability.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class TokenUsage:
    """Tracks token usage across calls."""

    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    def add(self, usage) -> None:
        if usage:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class LLMClient:
    """
    Unified LLM client with sync and async support.

    Usage:
        client = LLMClient()
        answer = await client.generate("What is RAG?", system="You are helpful.")
        parsed = await client.generate_structured(
            "Classify this query", system="...", response_model=RouteDecision
        )
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
        self.usage = TokenUsage()

    @property
    def sync_client(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(api_key=self._settings.openai_api_key)
        return self._sync_client

    @property
    def async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._async_client

    async def generate(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        fast: bool = False,
    ) -> str:
        """
        Generate a text completion.

        Args:
            prompt: User message.
            system: System prompt.
            model: Override model (defaults to settings).
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            fast: If True, use the cheap/fast model.

        Returns:
            Generated text.
        """
        model = model or (self._settings.llm_fast_model if fast else self._settings.llm_model)

        response = await self._call_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.usage.add(response.usage)
        return response.choices[0].message.content.strip()

    async def generate_structured(
        self,
        prompt: str,
        system: str,
        response_model: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.1,
        fast: bool = True,
    ) -> T:
        """
        Generate a structured response parsed into a Pydantic model.

        Uses OpenAI's JSON mode + function calling for reliable parsing.
        """
        model = model or (self._settings.llm_fast_model if fast else self._settings.llm_model)

        schema = response_model.model_json_schema()
        schema_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"{schema}"
        )

        response = await self._call_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": schema_prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        self.usage.add(response.usage)
        raw = response.choices[0].message.content.strip()
        return response_model.model_validate_json(raw)

    async def generate_stream(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream tokens as an async generator."""
        model = model or self._settings.llm_model

        stream = await self.async_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def generate_with_messages(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        fast: bool = False,
    ) -> str:
        """Generate from a full message list (for multi-turn)."""
        model = model or (self._settings.llm_fast_model if fast else self._settings.llm_model)

        response = await self._call_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.usage.add(response.usage)
        return response.choices[0].message.content.strip()

    async def _call_with_retry(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs,
    ):
        """Call the API with exponential backoff retry."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await self.async_client.chat.completions.create(**kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM call failed, retrying",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
        raise last_error


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Singleton LLM client."""
    return LLMClient()
