"""Shared retry and timeout policy for transient provider failures."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from agent_orchestrator.errors import ProviderError, ProviderTimeoutError
from agent_orchestrator.models import ChatMessage, LLMResponse
from agent_orchestrator.providers.base import LLMProvider


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 2
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("Retry delays cannot be negative")


async def chat_with_retry(
    provider: LLMProvider,
    messages: Sequence[ChatMessage],
    *,
    temperature: float,
    timeout_seconds: float,
    retry_policy: RetryPolicy,
) -> tuple[LLMResponse, int]:
    """Call a provider with bounded retries; cancellation is never swallowed."""
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            response = await asyncio.wait_for(
                provider.chat(
                    messages,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                ),
                timeout=timeout_seconds + 1.0,
            )
            return response, attempt
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            error: ProviderError = ProviderTimeoutError(
                f"Provider request exceeded {timeout_seconds:.1f}s"
            )
            error.__cause__ = exc
        except ProviderError as exc:
            error = exc

        if attempt >= retry_policy.max_attempts:
            raise error
        delay = min(
            retry_policy.base_delay_seconds * (2 ** (attempt - 1)),
            retry_policy.max_delay_seconds,
        )
        if delay:
            await asyncio.sleep(delay)

    raise AssertionError("Retry loop exited unexpectedly")
