"""LLM provider abstraction. Strategies never depend on a vendor SDK."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent_orchestrator.models import ChatMessage, LLMResponse


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        timeout_seconds: float = 90.0,
    ) -> LLMResponse: ...
