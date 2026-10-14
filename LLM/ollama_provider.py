"""Ollama provider - local models such as ``qwen2.5-coder`` or ``llama3.2``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from config import Settings
from LLM.base import LLMProvider, ProviderError, StreamChunk, ToolCall

__all__ = ["OllamaProvider"]


class OllamaProvider(LLMProvider):
    """Talks to a local (or LAN) Ollama server over its HTTP API.

    The provider is considered configured when ``OLLAMA_HOST`` points at an
    Ollama instance and is reachable (``GET /api/tags``).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.host: str = (self.settings.ollama_host or "http://localhost:11434").rstrip(
            "/"
        )
        self.model: str = self.settings.ollama_model

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "ollama"

    @property
    def model_label(self) -> str:
        """Model label shown in the UI."""
        return self.model or "ollama"

    def is_configured(self) -> bool:
        """True when a host and a model name are configured."""
        return bool(self.host and self.model)

    async def probe(self) -> bool:
        """Check that the Ollama daemon answers ``GET /api/tags``."""
        if not self.is_configured():
            return False
        try:
            async with httpx.AsyncClient(timeout=self.settings.probe_timeout) as client:
                response = await client.get(f"{self.host}/api/tags")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        *,
        market_data: Mapping[str, Any] | None = None,
        phase: str | None = None,
    ) -> str:
        """Run a single-turn chat completion against ``/api/chat``.

        Args:
            prompt: Request text.
            system_prompt: Agent persona.
            market_data: Ignored (remote providers receive it inside the prompt).
            phase: Ignored.

        Returns:
            The assistant message content.

        Raises:
            ProviderError: When the daemon is unreachable or returns an error.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": self.settings.effective_temperature,
                "num_predict": self.settings.max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            data = await self._post_json(client, f"{self.host}/api/chat", payload)

        if data.get("error"):
            raise ProviderError(f"ollama: {data['error']}")

        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            raise ProviderError("ollama: empty completion returned")
        return str(content)

    async def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        market_data: Mapping[str, Any] | None = None,
        phase: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream one turn from ``/api/chat`` (server sent events).

        Args:
            messages: Chat messages of the current turn.
            tools: OpenAI style function schemas (optional).
            market_data: Ignored.
            phase: Ignored.
            temperature: Optional sampling override.
            max_tokens: Optional output length override.

        Yields:
            ``reasoning`` (``message.thinking``), ``content`` and finally a
            ``tool_calls`` chunk when the model requested tools.

        Raises:
            ProviderError: When the daemon is unreachable or errors.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "messages": list(messages),
            "options": {
                "temperature": (
                    self.settings.effective_temperature if temperature is None else temperature
                ),
                "num_predict": max_tokens or self.settings.max_tokens,
            },
        }
        if tools:
            payload["tools"] = list(tools)

        pending: dict[str, str] = {"id": "", "name": "", "arguments": ""}
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.request_timeout
            ) as client, client.stream(
                "POST", f"{self.host}/api/chat", json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        f"ollama: HTTP {response.status_code} - "
                        f"{body.strip()[:200]}"
                    )
                async for raw_line in response.aiter_lines():
                    line = (raw_line or "").strip()
                    if not line:
                        continue
                    # Native Ollama streams newline-delimited JSON (no prefix);
                    # some proxies wrap it in Server-Sent Events ("data: ...").
                    if line.startswith("data:"):
                        line = line[5:].strip()
                        if line == "[DONE]":
                            break
                        if not line:
                            continue
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if event.get("error"):
                        raise ProviderError(f"ollama: {event['error']}")
                    message = event.get("message")
                    if not isinstance(message, dict):
                        continue
                    thinking = message.get("thinking")
                    if thinking:
                        yield StreamChunk(kind="reasoning", text=str(thinking))
                    content = message.get("content")
                    if content:
                        yield StreamChunk(kind="content", text=str(content))
                    for call in message.get("tool_calls") or []:
                        function = (
                            call.get("function")
                            if isinstance(call, dict)
                            else None
                        )
                        if not isinstance(function, dict):
                            continue
                        if function.get("name"):
                            pending["name"] = str(function["name"])
                        if function.get("arguments"):
                            arguments = function["arguments"]
                            pending["arguments"] += (
                                arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments)
                            )
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama: request failed ({exc})") from exc

        if pending["name"]:
            yield StreamChunk(
                kind="tool_calls",
                tool_calls=(
                    ToolCall(
                        id=pending["id"] or "call_0",
                        name=pending["name"],
                        arguments=pending["arguments"],
                    ),
                ),
            )
