"""Ollama streams newline-delimited JSON, not SSE - guard that parsing."""

from __future__ import annotations

import json
from typing import Any

import LLM.ollama_provider as ollama_mod
from config import Settings
from LLM.ollama_provider import OllamaProvider

_EVENTS: list[dict[str, Any]] = [
    {"message": {"role": "assistant", "content": "Hel"}, "done": False},
    {"message": {"role": "assistant", "content": "lo"}, "done": False},
    {"message": {"role": "assistant", "thinking": "hmm"}, "done": False},
    {"message": {"role": "assistant", "content": ""}, "done": True},
]


class _FakeResponse:
    status_code = 200

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def aiter_lines(self):
        for event in _EVENTS:
            yield json.dumps(event)


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, *args: object, **kwargs: object) -> _FakeStream:
        return _FakeStream(_FakeResponse())


async def test_ollama_parses_ndjson_stream(monkeypatch) -> None:
    monkeypatch.setattr(ollama_mod.httpx, "AsyncClient", _FakeClient)
    provider = OllamaProvider(Settings())

    kinds: list[tuple[str, str]] = []
    async for chunk in provider.stream([{"role": "user", "content": "hi"}]):
        if chunk.kind in ("content", "reasoning"):
            kinds.append((chunk.kind, chunk.text))

    assert ("content", "Hel") in kinds
    assert ("content", "lo") in kinds
    assert ("reasoning", "hmm") in kinds
