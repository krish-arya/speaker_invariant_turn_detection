"""Token-streaming chat completions: Groq (OpenAI-compatible SSE) and Ollama (NDJSON)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from turnsense.adapters.base import ChatMessage, LLMAdapter, ProviderError
from turnsense.events import ErrorSource


def parse_sse_line(line: str) -> str | None:
    """Text delta from one ``data: {...}`` SSE line; '' for keep-alives; None at [DONE]."""
    if not line.startswith("data:"):
        return ""
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except ValueError:
        return ""
    choices = obj.get("choices") or [{}]
    return (choices[0].get("delta") or {}).get("content") or ""


def _http_error(exc: httpx.HTTPStatusError) -> ProviderError:
    code = exc.response.status_code
    if code == 429:
        return ProviderError(ErrorSource.LLM, "rate_limited", "LLM provider rate limit reached")
    return ProviderError(
        ErrorSource.LLM,
        f"http_{code}",
        f"LLM provider returned HTTP {code}",
        recoverable=code >= 500,
    )


class OpenAICompatibleLLM(LLMAdapter):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 160,
        temperature: float = 0.6,
    ) -> None:
        self.name = name
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=30.0))

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        body = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        try:
            async with self._client.stream(
                "POST", self._url, json=body, headers=self._headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    delta = parse_sse_line(line)
                    if delta is None:
                        return
                    if delta:
                        yield delta
        except httpx.HTTPStatusError as exc:
            raise _http_error(exc) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                ErrorSource.LLM, "network_error", f"LLM request failed: {exc}"
            ) from exc


def groq_llm(api_key: str, model: str, base_url: str, max_tokens: int) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        f"groq_{model.replace('-', '_').replace('.', '_')}", base_url, api_key, model, max_tokens
    )


class OllamaLLM(LLMAdapter):
    def __init__(self, base_url: str, model: str, max_tokens: int = 160) -> None:
        self.name = f"ollama_{model}"
        self._url = base_url.rstrip("/") + "/api/chat"
        self._model = model
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=60.0))

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        body = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"num_predict": self._max_tokens},
        }
        try:
            async with self._client.stream("POST", self._url, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    text = (obj.get("message") or {}).get("content") or ""
                    if text:
                        yield text
                    if obj.get("done"):
                        return
        except httpx.HTTPStatusError as exc:
            raise _http_error(exc) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                ErrorSource.LLM, "network_error", f"Ollama request failed: {exc}"
            ) from exc
