"""Command-line entry point for local and optional OpenAI-compatible models."""

from __future__ import annotations

import argparse
import asyncio
import os

from agent_orchestrator.providers import OllamaProvider, OpenAICompatibleProvider
from agent_orchestrator.providers.base import LLMProvider
from agent_orchestrator.strategies import StrategyEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a multi-agent orchestration strategy"
    )

    parser.add_argument(
        "task",
        help="Task to give the agent team",
    )
    parser.add_argument(
        "--strategy",
        choices=["sequential", "parallel", "centralized", "distributed"],
        default="sequential",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai-compatible"],
        default=os.getenv("LLM_PROVIDER", "ollama"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY"),
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable Qwen3 thinking mode (Ollama only)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=512,
        help="Maximum number of generated tokens (Ollama only)",
    )

    return parser


async def _run(args: argparse.Namespace) -> None:
    provider: LLMProvider

    if args.provider == "ollama":
        provider = OllamaProvider(
            model=args.model,
            base_url=args.base_url,
            think=args.think,
            num_predict=args.num_predict,
        )
    else:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )

    engine = StrategyEngine(
        provider,
        timeout_seconds=180.0,
    )
    result = await engine.run(args.strategy, args.task)

    print(f"\n=== FINAL ANSWER ({result.strategy}) ===\n")
    print(result.final_answer)

    print("\n=== RUN METRICS ===")
    print(f"Run ID: {result.run_id}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Agents: {len(result.agent_results)}")

    for item in result.agent_results:
        print(
            f"- {item.agent_name}: "
            f"{item.latency_seconds:.2f}s, model={item.model}"
        )


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()