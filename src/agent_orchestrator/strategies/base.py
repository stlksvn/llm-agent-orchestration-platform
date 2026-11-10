from __future__ import annotations

from typing import Protocol

from agent_orchestrator.models import RunResult


class OrchestrationStrategy(Protocol):
    name: str

    async def run(self, task: str) -> RunResult: ...
