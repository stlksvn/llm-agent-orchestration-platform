import pytest

from agent_orchestrator.models import ChatMessage
from agent_orchestrator.providers.mock import MockProvider


@pytest.mark.asyncio
async def test_mock_provider_returns_deterministic_response():
    provider = MockProvider(["first", "second"])
    first = await provider.chat([ChatMessage("user", "hello")])
    second = await provider.chat([ChatMessage("user", "again")])
    assert first.content == "first"
    assert second.content == "second"
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_agent_retries_transient_provider_error():
    from agent_orchestrator.agents import AGENT_SPECS, Agent
    from agent_orchestrator.errors import ProviderError

    provider = MockProvider([ProviderError("temporary failure"), "recovered response"])
    agent = Agent(AGENT_SPECS["planner"], provider)
    result = await agent.run("make a plan")

    assert result.content == "recovered response"
    assert result.attempts == 2
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_agent_does_not_retry_after_attempt_limit():
    from agent_orchestrator.agents import AGENT_SPECS, Agent
    from agent_orchestrator.errors import ProviderError
    from agent_orchestrator.reliability import RetryPolicy

    provider = MockProvider([ProviderError("persistent failure")])
    agent = Agent(
        AGENT_SPECS["planner"],
        provider,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(ProviderError, match="persistent failure"):
        await agent.run("make a plan")
    assert len(provider.calls) == 1

@pytest.mark.asyncio
async def test_ollama_provider_parses_successful_response(monkeypatch):
    import json
    from agent_orchestrator.providers import ollama
    from agent_orchestrator.providers.ollama import OllamaProvider

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({
                "model": "qwen3:1.7b",
                "message": {"content": "  completed  "},
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 7,
            }).encode()

    monkeypatch.setattr(ollama, "urlopen", lambda *args, **kwargs: Response())
    result = await OllamaProvider().chat([ChatMessage("user", "hello")])
    assert result.content == "completed"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 7


@pytest.mark.asyncio
async def test_ollama_provider_expands_generation_budget_after_length(monkeypatch):
    import json
    from agent_orchestrator.providers import ollama
    from agent_orchestrator.providers.ollama import OllamaProvider

    limits = []
    class Response:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return self.body

    def fake_urlopen(request, **kwargs):
        payload = json.loads(request.data)
        limits.append(payload["options"]["num_predict"])
        body = {"model": "test", "message": {"content": "answer"},
                "done_reason": "length" if len(limits) == 1 else "stop"}
        return Response(json.dumps(body).encode())

    monkeypatch.setattr(ollama, "urlopen", fake_urlopen)
    result = await OllamaProvider(num_predict=16, max_num_predict=32).chat(
        [ChatMessage("user", "hello")]
    )
    assert result.content == "answer"
    assert limits == [16, 32]


@pytest.mark.asyncio
async def test_ollama_provider_rejects_invalid_json(monkeypatch):
    from agent_orchestrator.errors import ProviderError
    from agent_orchestrator.providers import ollama
    from agent_orchestrator.providers.ollama import OllamaProvider

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b"not-json"

    monkeypatch.setattr(ollama, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ProviderError, match="invalid JSON"):
        await OllamaProvider().chat([ChatMessage("user", "hello")])


def test_ollama_provider_rejects_invalid_token_limits():
    from agent_orchestrator.providers.ollama import OllamaProvider
    with pytest.raises(ValueError):
        OllamaProvider(num_predict=0)
    with pytest.raises(ValueError):
        OllamaProvider(num_predict=20, max_num_predict=10)
