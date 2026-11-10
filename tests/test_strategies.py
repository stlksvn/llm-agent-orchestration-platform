import pytest

from agent_orchestrator.providers.mock import MockProvider
from agent_orchestrator.strategies import StrategyEngine


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["sequential", "parallel", "centralized", "distributed"])
async def test_each_strategy_produces_final_answer(strategy):
    # The response sequence intentionally includes supervisor decisions and agent content.
    responses = [
        '{"next":"planner","reason":"plan"}',
        "Plan and constraints. NEXT: researcher",
        "Research findings. NEXT: executor",
        "Draft result. NEXT: critic",
        "Review complete. NEXT: FINISH",
        "Final synthesized answer.",
    ]
    provider = MockProvider(responses)
    engine = StrategyEngine(provider, max_steps=4, max_concurrency=2)
    result = await engine.run(strategy, "Explain a multi-agent system")
    assert result.final_answer
    assert result.status == "completed"
    assert result.duration_seconds >= 0
    assert result.events[-1].event_type.value == "run_completed"


@pytest.mark.asyncio
async def test_empty_task_is_rejected():
    engine = StrategyEngine(MockProvider())
    with pytest.raises(ValueError, match="Task must not be empty"):
        await engine.run("sequential", "   ")


@pytest.mark.asyncio
async def test_unknown_strategy_is_rejected():
    engine = StrategyEngine(MockProvider())
    with pytest.raises(ValueError, match="Unknown strategy"):
        await engine.run("magic", "task")


@pytest.mark.asyncio
async def test_centralized_strategy_uses_supervisor_routing_decisions():
    provider = MockProvider(
        [
            '{"next":"planner","reason":"Create a plan"}',
            "Plan ready.",
            '{"next":"critic","reason":"Review the plan"}',
            "Review complete.",
            '{"next":"finish","reason":"Sufficient review"}',
            "Final answer.",
        ]
    )
    result = await StrategyEngine(provider, max_steps=4).run("centralized", "Review a design")
    assert [item.agent_name for item in result.agent_results] == [
        "planner", "critic", "synthesizer"
    ]
    assert result.final_answer == "Final answer."


@pytest.mark.asyncio
async def test_distributed_strategy_follows_peer_handoff():
    provider = MockProvider(
        [
            'Plan ready. {"next":"critic","reason":"Needs review"}',
            'Review complete. {"next":"finish","reason":"Done"}',
            "Final answer.",
        ]
    )
    result = await StrategyEngine(provider, max_steps=4).run("distributed", "Review a design")
    assert [item.agent_name for item in result.agent_results] == [
        "planner", "critic", "synthesizer"
    ]
