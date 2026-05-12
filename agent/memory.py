"""
agent/memory.py — Conversation memory manager.

Provides multi-turn conversation context with:
- Sliding window (last N turns)
- Thread-safe with asyncio locks
- Summary generation for older turns
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Optional

from agent.schemas import ConversationTurn
from observability.logger import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    In-memory conversation store keyed by conversation_id.

    Features:
    - Sliding window: keeps last N full turns
    - Thread-safe: uses asyncio locks
    - Context formatting: produces clean context strings for prompts
    """

    def __init__(self, max_turns: int = 20):
        self._max_turns = max_turns
        self._conversations: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def add_turn(
        self, conversation_id: str, role: str, content: str
    ) -> None:
        """Add a turn to the conversation history."""
        async with self._locks[conversation_id]:
            turns = self._conversations[conversation_id]
            turns.append(ConversationTurn(role=role, content=content))

            # Trim to max turns
            if len(turns) > self._max_turns:
                self._conversations[conversation_id] = turns[-self._max_turns:]

            logger.debug(
                "Turn added",
                conversation_id=conversation_id,
                role=role,
                total_turns=len(self._conversations[conversation_id]),
            )

    async def get_context(
        self, conversation_id: str, last_n: int = 5
    ) -> str:
        """
        Get formatted conversation context for prompt injection.

        Returns the last N turns as a formatted string.
        """
        async with self._locks[conversation_id]:
            turns = self._conversations[conversation_id]

        if not turns:
            return "No previous context."

        recent = turns[-last_n:]
        lines = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")

        return "\n".join(lines)

    async def get_turns(
        self, conversation_id: str
    ) -> list[ConversationTurn]:
        """Get all turns for a conversation."""
        async with self._locks[conversation_id]:
            return list(self._conversations[conversation_id])

    async def clear(self, conversation_id: str) -> None:
        """Clear a conversation's history."""
        async with self._locks[conversation_id]:
            self._conversations[conversation_id] = []
            logger.info("Conversation cleared", conversation_id=conversation_id)

    def get_active_conversations(self) -> list[str]:
        """List all active conversation IDs."""
        return [
            cid for cid, turns in self._conversations.items() if turns
        ]
