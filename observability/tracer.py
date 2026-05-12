"""Lightweight agent trace capture for observability."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TraceStepType(str, Enum):
    """Types of steps in the agent trace."""
    ROUTE = "route"
    DECOMPOSE = "decompose"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GRADE = "grade"
    GENERATE = "generate"
    REFLECT = "reflect"
    WEB_SEARCH = "web_search"
    ERROR = "error"


@dataclass
class TraceStep:
    """A single step in the agent execution trace."""
    step_type: TraceStepType
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_type": self.step_type.value,
            "input": self.input_data,
            "output": self.output_data,
            "duration_ms": round(self.duration_ms, 2),
            "token_usage": self.token_usage,
            "metadata": self.metadata,
        }


class TraceContext:
    """
    Captures the full execution trace of an agent run.

    Usage:
        trace = TraceContext(query="What is RAG?")
        with trace.step(TraceStepType.ROUTE) as step:
            step.output_data = {"decision": "retrieval"}
        # trace.steps now contains the completed step
    """

    def __init__(self, query: str, conversation_id: Optional[str] = None):
        self.trace_id: str = uuid.uuid4().hex[:12]
        self.query: str = query
        self.conversation_id: str = conversation_id or "anonymous"
        self.steps: list[TraceStep] = []
        self.start_time: float = time.perf_counter()
        self.total_tokens: int = 0

    class _StepContextManager:
        """Context manager for timing a trace step."""

        def __init__(self, trace: TraceContext, step: TraceStep):
            self._trace = trace
            self.step = step
            self._start: float = 0.0

        def __enter__(self) -> TraceStep:
            self._start = time.perf_counter()
            return self.step

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            self.step.duration_ms = (time.perf_counter() - self._start) * 1000
            if exc_type:
                self.step.metadata["error"] = str(exc_val)
                self.step.step_type = TraceStepType.ERROR
            tokens = sum(self.step.token_usage.values())
            self._trace.total_tokens += tokens
            self._trace.steps.append(self.step)
            return None  # don't suppress exceptions

    def step(self, step_type: TraceStepType, **metadata) -> _StepContextManager:
        """Create a traced step context manager."""
        trace_step = TraceStep(step_type=step_type, metadata=metadata)
        return self._StepContextManager(self, trace_step)

    @property
    def total_duration_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "query": self.query,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_tokens": self.total_tokens,
            "num_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }
