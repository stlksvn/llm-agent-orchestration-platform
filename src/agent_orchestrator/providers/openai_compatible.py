"""OpenAI-compatible chat completions provider (optional cloud/local endpoints)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_orchestrator.errors import ProviderError, ProviderTimeoutError
from agent_orchestrator.models import ChatMessage, LLMResponse


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("LLM_API_KEY")

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        timeout_seconds: float = 90.0,
    ) -> LLMResponse:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        def send() -> dict:
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise ProviderError(f"Provider HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                raise ProviderError(f"Cannot reach provider at {self.base_url}: {exc}") from exc
            except TimeoutError as exc:
                raise ProviderTimeoutError("Provider request timed out") from exc

        try:
            data = await asyncio.wait_for(asyncio.to_thread(send), timeout=timeout_seconds + 2)
        except TimeoutError as exc:
            raise ProviderTimeoutError("Provider request timed out") from exc
        try:
            choice = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Provider returned an invalid chat-completions response") from exc
        if not isinstance(choice, str) or not choice.strip():
            raise ProviderError("Provider returned empty content")
        usage = data.get("usage") or {}
        return LLMResponse(
            content=choice.strip(),
            model=str(data.get("model", self.model)),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_seconds=time.perf_counter() - started,
            raw=data,
        )
