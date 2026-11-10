from __future__ import annotations

import asyncio
from pathlib import Path

from agent_orchestrator.benchmark.runner import (
    STRATEGIES, load_tasks, run_benchmark, summarize, write_results,
)
from agent_orchestrator.providers.mock import MockProvider


def test_benchmark_runs_full_strategy_matrix_and_summarizes(tmp_path: Path) -> None:
    provider = MockProvider()
    records = asyncio.run(run_benchmark(
        provider, [{"id": "case-1", "prompt": "Explain a queue."}], repetitions=2
    ))

    assert len(records) == len(STRATEGIES) * 2
    assert {record.strategy for record in records} == set(STRATEGIES)
    assert all(record.status == "completed" for record in records)
    assert all(record.nonempty_answer for record in records)
    summary = summarize(records)
    assert len(summary) == 4
    assert all(row["success_rate"] == 1.0 for row in summary)
    csv_path, json_path, summary_path = write_results(records, tmp_path)
    assert csv_path.exists() and json_path.exists() and summary_path.exists()


def test_load_tasks_rejects_empty_prompt(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text('[{"id":"empty","prompt":" "}]', encoding="utf-8")
    tasks = load_tasks(path)
    try:
        asyncio.run(run_benchmark(MockProvider(), tasks))
    except ValueError as exc:
        assert "empty prompt" in str(exc)
    else:
        raise AssertionError("Expected empty prompt to be rejected")
