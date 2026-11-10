"""Ollama chat provider using its local HTTP API and only the Python stdlib."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_orchestrator.errors import ProviderError, ProviderTimeoutError
from agent_orchestrator.models import ChatMessage, LLMResponse


class OllamaProvider:
    def __init__(
        self,
        model: str = "qwen3:1.7b",
        base_url: str = "http://localhost:11434",
        keep_alive: str = "5m",
        think: bool = False,
        num_predict: int = 1024,
        max_num_predict: int = 4096,
    ) -> None:
        if num_predict < 1:
            raise ValueError("num_predict must be greater than 0")
        if max_num_predict < num_predict:
            raise ValueError("max_num_predict must be greater than or equal to num_predict")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.think = think
        self.num_predict = num_predict
        self.max_num_predict = max_num_predict

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        timeout_seconds: float = 90.0,
    ) -> LLMResponse:
        started = time.perf_counter()

        current_limit = self.num_predict
        data: dict = {}

        # Ollama reports done_reason="length" when generation hits num_predict.
        # Retry a bounded number of times with a larger output budget instead of
        # silently returning a truncated agent response.
        while True:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                "stream": False,
                "keep_alive": self.keep_alive,
                "think": self.think,
                "options": {
                    "temperature": temperature,
                    "num_predict": current_limit,
                },
            }

            request = Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )

            def send() -> dict:
                try:
                    with urlopen(request, timeout=timeout_seconds) as response:
                        raw_body = response.read().decode("utf-8")
                    try:
                        parsed = json.loads(raw_body)
                    except json.JSONDecodeError as exc:
                        raise ProviderError("Ollama returned invalid JSON") from exc
                    if not isinstance(parsed, dict):
                        raise ProviderError("Ollama returned an unexpected response structure")
                    return parsed
                except HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise ProviderError(f"Ollama HTTP {exc.code}: {detail}") from exc
                except URLError as exc:
                    raise ProviderError(
                        f"Cannot reach Ollama at {self.base_url}. "
                        f"Is Ollama running? ({exc})"
                    ) from exc
                except TimeoutError as exc:
                    raise ProviderTimeoutError(
                        f"Ollama request exceeded {timeout_seconds:.1f}s"
                    ) from exc
                except OSError as exc:
                    raise ProviderError(f"Ollama connection failed: {exc}") from exc

            try:
                data = await asyncio.wait_for(
                    asyncio.to_thread(send),
                    timeout=timeout_seconds + 2,
                )
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    f"Ollama request exceeded {timeout_seconds:.1f}s"
                ) from exc

            if data.get("done_reason") != "length" or current_limit >= self.max_num_predict:
                break
            current_limit = min(current_limit * 2, self.max_num_predict)

        content = data.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Ollama returned an empty or malformed message")

        if data.get("done_reason") == "length":
            raise ProviderError(
                f"Ollama response was truncated at the maximum of "
                f"{self.max_num_predict} generated tokens"
            )

        return LLMResponse(
            content=content.strip(),
            model=str(data.get("model", self.model)),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_seconds=time.perf_counter() - started,
            raw=data,
        )