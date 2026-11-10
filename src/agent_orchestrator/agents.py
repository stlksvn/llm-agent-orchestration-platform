"""Role-specific agents implemented on top of a provider-neutral LLM interface."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agent_orchestrator.models import AgentResult, ChatMessage
from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.reliability import RetryPolicy, chat_with_retry


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    system_prompt: str


AGENT_SPECS: dict[str, AgentSpec] = {
    "planner": AgentSpec(
        "planner",
        (
            "You are a planning agent. Break the user's task into a concise plan, "
            "identify constraints, and define what a good result must contain. "
            "Do not solve the task fully."
        ),
    ),
    "researcher": AgentSpec(
        "researcher",
        (
            "You are a research agent. Work only from the supplied task and context. "
            "Separate known facts from assumptions and identify missing information. "
            "Do not invent sources or claim to have browsed the web."
        ),
    ),
    "critic": AgentSpec(
        "critic",
        (
            "You are a critical reviewer. Find logical gaps, unsupported claims, "
            "ambiguities, and practical risks in the supplied draft. "
            "Give actionable corrections, not generic praise."
        ),
    ),
    "executor": AgentSpec(
        "executor",
        (
            "You are an execution agent. Produce the requested deliverable using "
            "the supplied task and context. Follow constraints exactly and state "
            "uncertainty where needed."
        ),
    ),
    "synthesizer": AgentSpec(
        "synthesizer",
        (
            "You are a synthesis agent. Produce a coherent final answer from the "
            "task and agent contributions. Resolve repetition, preserve useful "
            "disagreement, and do not invent facts."
        ),
    ),
}


class Agent:
    def __init__(
        self,
        spec: AgentSpec,
        provider: LLMProvider,
        *,
        temperature: float = 0.2,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.temperature = temperature
        self.retry_policy = retry_policy or RetryPolicy()

    @property
    def name(self) -> str:
        return self.spec.name

    async def run(
        self,
        task: str,
        context: str = "",
        timeout_seconds: float = 90.0,
    ) -> AgentResult:
        user_content = f"TASK:\n{task}"

        if context.strip():
            user_content += f"\n\nCONTEXT FROM PREVIOUS STEPS:\n{context}"

        started = time.perf_counter()

        response, attempts = await chat_with_retry(
            self.provider,
            [
                ChatMessage("system", self.spec.system_prompt),
                ChatMessage("user", user_content),
            ],
            temperature=self.temperature,
            timeout_seconds=timeout_seconds,
            retry_policy=self.retry_policy,
        )

        elapsed = time.perf_counter() - started

        return AgentResult(
            agent_name=self.name,
            content=response.content,
            model=response.model,
            latency_seconds=response.latency_seconds or elapsed,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            attempts=attempts,
        )