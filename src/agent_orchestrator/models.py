"""Core domain models shared by providers, agents and orchestration strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    ROUTE_SELECTED = "route_selected"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class AgentResult:
    agent_name: str
    content: str
    model: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class OrchestrationEvent:
    event_type: EventType
    run_id: str
    timestamp: datetime = field(default_factory=utc_now)
    agent_name: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    run_id: str
    strategy: str
    task: str
    final_answer: str
    agent_results: list[AgentResult]
    events: list[OrchestrationEvent]
    started_at: datetime
    finished_at: datetime
    status: str = "completed"

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @classmethod
    def new_run_id(cls) -> str:
        return str(uuid4())
