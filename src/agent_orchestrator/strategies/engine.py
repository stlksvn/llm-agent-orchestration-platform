"""Four deliberately distinct orchestration strategies with bounded execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from agent_orchestrator.agents import AGENT_SPECS, Agent
from agent_orchestrator.errors import StrategyError
from agent_orchestrator.models import (
    AgentResult,
    ChatMessage,
    EventType,
    OrchestrationEvent,
    RunResult,
    utc_now,
)
from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.reliability import RetryPolicy, chat_with_retry
from agent_orchestrator.routing import parse_routing_decision


class StrategyEngine:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        timeout_seconds: float = 90.0,
        max_steps: int = 5,
        max_concurrency: int = 2,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.max_steps = max_steps

        if timeout_seconds <= 0 or max_steps < 1 or max_concurrency < 1:
            raise ValueError(
                "timeout_seconds, max_steps and max_concurrency must be positive"
            )

        self.max_concurrency = max_concurrency
        self.retry_policy = retry_policy or RetryPolicy()
        self.agents = {
            name: Agent(spec, provider, retry_policy=self.retry_policy)
            for name, spec in AGENT_SPECS.items()
        }

    async def run(self, strategy: str, task: str) -> RunResult:
        if not task.strip():
            raise ValueError("Task must not be empty")

        runners: dict[str, Callable[[str], object]] = {
            "sequential": self._sequential,
            "parallel": self._parallel,
            "centralized": self._centralized,
            "distributed": self._distributed,
        }

        if strategy not in runners:
            raise ValueError(f"Unknown strategy: {strategy}")

        run_id = RunResult.new_run_id()
        started = utc_now()

        events = [
            OrchestrationEvent(
                EventType.RUN_STARTED,
                run_id,
                message=f"Strategy: {strategy}",
            )
        ]

        try:
            final, results = await runners[strategy](task)  # type: ignore[misc]

            for result in results:
                events.append(
                    OrchestrationEvent(
                        EventType.AGENT_COMPLETED,
                        run_id,
                        agent_name=result.agent_name,
                        message=f"Completed using {result.model}",
                        metadata={
                            "latency_seconds": result.latency_seconds,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "attempts": result.attempts,
                        },
                    )
                )

            events.append(
                OrchestrationEvent(
                    EventType.RUN_COMPLETED,
                    run_id,
                    message="Run completed",
                )
            )

            return RunResult(
                run_id,
                strategy,
                task,
                final,
                results,
                events,
                started,
                utc_now(),
            )

        except Exception as exc:
            events.append(
                OrchestrationEvent(
                    EventType.RUN_FAILED,
                    run_id,
                    message=str(exc),
                )
            )
            raise StrategyError(f"{strategy} strategy failed: {exc}") from exc

    async def _call(self, name: str, task: str, context: str = ""):
        return await self.agents[name].run(
            task,
            context,
            self.timeout_seconds,
        )

    async def _sequential(self, task: str):
        results = []
        context = ""

        for name in ("planner", "researcher", "executor", "critic"):
            result = await self._call(name, task, context)
            results.append(result)
            context += f"\n\n[{name.upper()}]\n{result.content}"

        final = await self._call("synthesizer", task, context)
        results.append(final)

        return final.content, results

    async def _parallel(self, task: str):
        names = ("planner", "researcher", "executor", "critic")
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def bounded(name: str):
            async with semaphore:
                return await self._call(name, task)

        gathered = await asyncio.gather(
            *(bounded(name) for name in names),
            return_exceptions=True,
        )

        results = [
            item for item in gathered
            if not isinstance(item, BaseException)
        ]
        failures = [
            item for item in gathered
            if isinstance(item, BaseException)
        ]

        if failures:
            failed_names = [
                names[index]
                for index, item in enumerate(gathered)
                if isinstance(item, BaseException)
            ]
            details = "; ".join(str(item) for item in failures)
            raise RuntimeError(
                "Parallel execution is incomplete. Failed agents: "
                f"{', '.join(failed_names)}. Details: {details}"
            )

        if len(results) != len(names):
            raise RuntimeError(
                f"Expected {len(names)} parallel results, received {len(results)}"
            )

        context = "\n\n".join(
            f"[{result.agent_name.upper()}]\n{result.content}"
            for result in results
        )

        final = await self._call("synthesizer", task, context)
        results.append(final)

        return final.content, results

    async def _centralized(self, task: str):
        """A supervisor LLM chooses the next worker at each bounded step."""

        results: list[AgentResult] = []
        context = ""
        available = {"planner", "researcher", "executor", "critic"}

        for _ in range(self.max_steps):
            decision, _ = await chat_with_retry(
                self.provider,
                [
                    ChatMessage(
                        "system",
                        (
                            "You are a central supervisor. Choose exactly one next "
                            "worker from planner, researcher, executor, critic, "
                            "or FINISH. Return one JSON object with keys "
                            '"next" and "reason". Do not choose an already used worker.'
                        ),
                    ),
                    ChatMessage(
                        "user",
                        (
                            f"Task: {task}\n"
                            f"Used workers: {[result.agent_name for result in results]}\n"
                            f"Context: {context}"
                        ),
                    ),
                ],
                temperature=0.0,
                timeout_seconds=self.timeout_seconds,
                retry_policy=self.retry_policy,
            )

            parsed = parse_routing_decision(decision.content, available)
            next_agent = parsed.next_agent if parsed else ""
            used = {result.agent_name for result in results}

            if next_agent == "finish":
                break

            if next_agent not in available or next_agent in used:
                next_agent = next(
                    (
                        name
                        for name in ("planner", "researcher", "executor", "critic")
                        if name not in used
                    ),
                    "",
                )

            if not next_agent:
                break

            result = await self._call(next_agent, task, context)
            results.append(result)
            context += f"\n\n[{next_agent.upper()}]\n{result.content}"

        if not results:
            results.append(await self._call("planner", task))
            context = results[0].content

        final = await self._call("synthesizer", task, context)
        results.append(final)

        return final.content, results

    async def _distributed(self, task: str):
        """Agents propose peer handoffs; the runtime validates and bounds them."""

        results = []
        context = ""
        current = "planner"
        visited: set[str] = set()
        allowed = {"planner", "researcher", "executor", "critic"}

        for _ in range(self.max_steps):
            if current not in allowed or current in visited:
                break

            routing_instruction = (
                "\n\nAfter completing your work, append exactly one JSON object with "
                'keys "next" and "reason". The next value must be one of '
                '"planner", "researcher", "executor", "critic", or "finish". '
                "Do not include Markdown fences around this routing object."
            )

            result = await self._call(
                current,
                task,
                context + routing_instruction,
            )

            results.append(result)
            visited.add(current)
            context += f"\n\n[{current.upper()}]\n{result.content}"

            decision = parse_routing_decision(result.content, allowed)
            proposed = decision.next_agent if decision else ""

            if not proposed:
                # Invalid model output must not create an unbounded or arbitrary route.
                proposed = next(
                    (
                        name
                        for name in ("researcher", "executor", "critic")
                        if name not in visited
                    ),
                    "finish",
                )

            if proposed == "finish" or proposed not in allowed or proposed in visited:
                break

            current = proposed

        if not results:
            raise RuntimeError("Distributed workflow did not execute an agent")

        final = await self._call("synthesizer", task, context)
        results.append(final)

        return final.content, results