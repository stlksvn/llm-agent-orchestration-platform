"""Deterministic provider for tests and offline demonstrations."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence

from agent_orchestrator.errors import ProviderError
from agent_orchestrator.models import ChatMessage, LLMResponse


class MockProvider:
    def __init__(
        self, responses: list[str | Exception] | None = None, delay_seconds: float = 0.0
    ) -> None:
        self._responses = deque(responses or [])
        self.delay_seconds = delay_seconds
        self.calls: list[list[ChatMessage]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        timeout_seconds: float = 90.0,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        response = self._responses.popleft() if self._responses else "Mock response"
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, str):
            raise ProviderError("Mock response must be text")
        return LLMResponse(content=response, model="mock-model", latency_seconds=self.delay_seconds)
