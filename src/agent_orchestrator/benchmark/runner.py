"""Repeatable benchmark runner for comparing orchestration strategies.

The runner records operational metrics. It deliberately does not claim to measure
semantic answer quality: that requires a task-specific reference or human rubric.
"""
from __future__ import annotations

import asyncio
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_orchestrator.strategies import StrategyEngine

STRATEGIES = ("sequential", "parallel", "centralized", "distributed")


@dataclass(slots=True)
class BenchmarkRecord:
    task_id: str
    strategy: str
    repetition: int
    status: str
    duration_seconds: float
    agent_count: int
    provider_calls: int
    prompt_tokens: int | None
    completion_tokens: int | None
    answer_characters: int
    nonempty_answer: bool
    ends_with_terminal_punctuation: bool
    error: str | None = None


def _sum_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if isinstance(value, int)]
    return sum(present) if present else None


def _record(task_id: str, strategy: str, repetition: int, result: Any,
            elapsed: float, calls_before: int, calls_after: int) -> BenchmarkRecord:
    answer = result.final_answer.strip()
    agent_results = result.agent_results
    return BenchmarkRecord(
        task_id=task_id,
        strategy=strategy,
        repetition=repetition,
        status="completed",
        duration_seconds=elapsed,
        agent_count=len(agent_results),
        provider_calls=calls_after - calls_before,
        prompt_tokens=_sum_optional([item.prompt_tokens for item in agent_results]),
        completion_tokens=_sum_optional([item.completion_tokens for item in agent_results]),
        answer_characters=len(answer),
        nonempty_answer=bool(answer),
        ends_with_terminal_punctuation=answer.endswith((".", "!", "?", "…", ")", "]", "}")),
    )


async def run_benchmark(provider: Any, tasks: list[dict[str, str]], *, repetitions: int = 3,
                        timeout_seconds: float = 180.0) -> list[BenchmarkRecord]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if not tasks:
        raise ValueError("at least one benchmark task is required")
    engine = StrategyEngine(provider, timeout_seconds=timeout_seconds)
    records: list[BenchmarkRecord] = []
    for task in tasks:
        task_id, prompt = task.get("id", "task"), task.get("prompt", "")
        if not prompt.strip():
            raise ValueError(f"Task {task_id!r} has an empty prompt")
        for repetition in range(1, repetitions + 1):
            for strategy in STRATEGIES:
                calls_before = len(getattr(provider, "calls", []))
                started = time.perf_counter()
                try:
                    result = await engine.run(strategy, prompt)
                    elapsed = time.perf_counter() - started
                    calls_after = len(getattr(provider, "calls", []))
                    records.append(_record(task_id, strategy, repetition, result,
                                           elapsed, calls_before, calls_after))
                except Exception as exc:  # retain failures as data, don't abort the matrix
                    elapsed = time.perf_counter() - started
                    calls_after = len(getattr(provider, "calls", []))
                    records.append(BenchmarkRecord(
                        task_id=task_id, strategy=strategy, repetition=repetition,
                        status="failed", duration_seconds=elapsed, agent_count=0,
                        provider_calls=calls_after - calls_before, prompt_tokens=None,
                        completion_tokens=None, answer_characters=0, nonempty_answer=False,
                        ends_with_terminal_punctuation=False,
                        error=f"{type(exc).__name__}: {exc}"[:1000],
                    ))
    return records


def summarize(records: list[BenchmarkRecord]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        group = [r for r in records if r.strategy == strategy]
        completed = [r for r in group if r.status == "completed"]
        durations = [r.duration_seconds for r in completed]
        prompts = [r.prompt_tokens for r in completed if r.prompt_tokens is not None]
        completions = [r.completion_tokens for r in completed if r.completion_tokens is not None]
        summary.append({
            "strategy": strategy,
            "runs": len(group),
            "successful_runs": len(completed),
            "success_rate": len(completed) / len(group) if group else 0.0,
            "mean_duration_seconds": statistics.mean(durations) if durations else None,
            "median_duration_seconds": statistics.median(durations) if durations else None,
            "mean_provider_calls": statistics.mean([r.provider_calls for r in group]) if group else None,
            "mean_prompt_tokens": statistics.mean(prompts) if prompts else None,
            "mean_completion_tokens": statistics.mean(completions) if completions else None,
            "nonempty_answer_rate": (sum(r.nonempty_answer for r in completed) / len(completed)
                                     if completed else None),
        })
    return summary


def write_results(records: list[BenchmarkRecord], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"benchmark_{stamp}.csv"
    json_path = output_dir / f"benchmark_{stamp}.json"
    summary_path = output_dir / f"benchmark_summary_{stamp}.csv"
    rows = [asdict(record) for record in records]
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {"created_at": datetime.now(UTC).isoformat(),
               "metric_notes": {
                   "duration": "Wall-clock time for one complete orchestration run.",
                   "tokens": "Provider-reported counts; null when unavailable.",
                   "answer_quality": "Not automatically evaluated; use a task-specific rubric.",
                   "completion_proxy": "Non-empty output and terminal punctuation are weak diagnostics only.",
               }, "records": rows, "summary": summarize(records)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = payload["summary"]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)
    return csv_path, json_path, summary_path


def load_tasks(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError("Dataset must be a JSON array of {id, prompt} objects")
    return [{"id": str(item.get("id", f"task-{i+1}")),
             "prompt": str(item.get("prompt", ""))} for i, item in enumerate(data)]


def main() -> None:
    import argparse
    from agent_orchestrator.providers import OllamaProvider, OpenAICompatibleProvider

    parser = argparse.ArgumentParser(description="Benchmark all orchestration strategies")
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/tasks.json"))
    parser.add_argument("--provider", choices=("ollama", "openai-compatible"), default="ollama")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()
    provider = (OllamaProvider(model=args.model, base_url=args.base_url)
               if args.provider == "ollama" else
               OpenAICompatibleProvider(base_url=args.base_url, model=args.model,
                                        api_key=args.api_key))
    records = asyncio.run(run_benchmark(provider, load_tasks(args.dataset),
                                        repetitions=args.repetitions,
                                        timeout_seconds=args.timeout))
    paths = write_results(records, args.output_dir)
    print("\nStrategy comparison")
    for row in summarize(records):
        duration = row["mean_duration_seconds"]
        duration_text = f"{duration:.2f}s" if duration is not None else "n/a"
        print(f"{row['strategy']:12} success={row['successful_runs']}/{row['runs']} "
              f"mean={duration_text} calls={row['mean_provider_calls']}")
    print("\nSaved:", *(str(path) for path in paths), sep="\n  ")


if __name__ == "__main__":
    main()
