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
from datetime import datetime, timezone
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

    async def get_recent_topics(self, conversation_id: str, n: int = 5) -> list[str]:
        """Return the last *n* unique non-empty user messages from the conversation.

        This acts as a lightweight "topic tracker" — each distinct user turn is
        treated as a topic.  Duplicates are deduplicated while preserving order so
        that the most-recent topic appears last.

        Args:
            conversation_id: The conversation to inspect.
            n: Maximum number of unique topics to return.

        Returns:
            Ordered list of up to *n* unique user message strings.
        """
        async with self._locks[conversation_id]:
            turns = self._conversations[conversation_id]

        seen: dict[str, None] = {}  # ordered-set via insertion-order dict
        for turn in turns:
            if turn.role == "user" and turn.content.strip():
                seen[turn.content.strip()] = None

        unique = list(seen.keys())
        result = unique[-n:]
        logger.debug(
            "get_recent_topics",
            conversation_id=conversation_id,
            n=n,
            found=len(result),
        )
        return result

    def summary(self, conversation_id: str) -> dict:
        """Return a lightweight summary dict for a given conversation.

        The summary is intentionally synchronous and lock-free so it can be
        called safely from non-async contexts (e.g. logging middleware).

        Returns a dict with:
            turn_count      — total number of turns stored.
            unique_topics   — number of distinct user messages.
            session_id      — the conversation_id (alias for tracing).
        """
        turns = self._conversations.get(conversation_id, [])
        unique_topics = len({t.content.strip() for t in turns if t.role == "user" and t.content.strip()})
        return {
            "turn_count": len(turns),
            "unique_topics": unique_topics,
            "session_id": conversation_id,
        }

    async def export(self, conversation_id: str) -> list[dict]:
        """Export the full conversation as a list of serialisable dicts.

        Each entry has:
            role      — 'user' or 'assistant'.
            content   — the message text.
            timestamp — ISO-8601 UTC string stamped at export time (same for all
                         rows in a single export call, since ConversationTurn does
                         not store per-message timestamps).

        Args:
            conversation_id: The conversation to export.

        Returns:
            List of dicts ordered from oldest to newest turn.
        """
        async with self._locks[conversation_id]:
            turns = list(self._conversations[conversation_id])

        now_iso = datetime.now(timezone.utc).isoformat()
        exported = [
            {"role": turn.role, "content": turn.content, "timestamp": now_iso}
            for turn in turns
        ]
        logger.info(
            "Conversation exported",
            conversation_id=conversation_id,
            turns_exported=len(exported),
        )
        return exported
