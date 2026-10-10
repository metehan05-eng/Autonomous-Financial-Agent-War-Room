"""Deterministic, network-free LLM provider used by the test suite.

The provider implements the real :class:`LLM.base.LLMProvider` streaming
contract (reasoning / content / tool-call chunks) so tests can drive the
orchestrator without ever touching a remote gateway or yfinance.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from config import Settings, load_settings
from LLM.base import LLMProvider, StreamChunk, ToolCall

__all__ = ["MockEvrenProvider", "build_mock_router"]


class MockEvrenProvider(LLMProvider):
    """Configurable fake Evren provider.

    Args:
        settings: Optional configuration override.
        latency: Artificial delay (seconds) before each streamed turn.
        emit_tools: When ``True`` the first streamed call of every turn
            requests a tool, exercising the tool-call loop.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        latency: float = 0.0,
        emit_tools: bool = True,
    ) -> None:
        super().__init__(settings)
        self.latency = latency
        self.emit_tools = emit_tools
        #: One record per ``stream()`` call: the phase and whether tools
        #: were offered (``None`` means the tool schemas were withheld).
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "evren"

    @property
    def model_label(self) -> str:
        """Model label (fake)."""
        return "mock-evren"

    def is_configured(self) -> bool:
        """The mock is always configured."""
        return True

    async def probe(self) -> bool:
        """The mock is always reachable."""
        return True

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        *,
        market_data: Mapping[str, Any] | None = None,
        phase: str | None = None,
    ) -> str:
        """Return a canned completion."""
        return f"mock completion for phase {phase or 'unknown'}"

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
        """Stream a reasoning delta, a tool call (once per turn) or an answer."""
        self.calls.append({"phase": phase, "tools": tools})
        if self.latency > 0:
            await asyncio.sleep(self.latency)

        already_used_tool = any(m.get("role") == "tool" for m in messages)
        if tools and self.emit_tools and not already_used_tool:
            yield StreamChunk(kind="reasoning", text="Let me verify the data.")
            yield StreamChunk(
                kind="tool_calls",
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="get_stock_quote",
                        arguments='{"symbol": "AAPL"}',
                    ),
                ),
            )
            return

        yield StreamChunk(kind="reasoning", text=f"Reasoning for {phase or 'turn'}.")
        yield StreamChunk(
            kind="content", text=f"{phase or 'turn'} answer from the mock provider."
        )


def build_mock_router(
    settings: Settings | None = None,
    *,
    latency: float = 0.0,
    emit_tools: bool = True,
):
    """Build an :class:`LLM.base.LLMRouter` wired to a single mock provider.

    Args:
        settings: Optional configuration override.
        latency: Artificial streaming delay (seconds).
        emit_tools: Whether the mock should request tools.

    Returns:
        ``(router, provider)`` so tests can inspect the recorded calls.
    """
    from LLM.base import LLMRouter

    resolved = settings or load_settings()
    provider = MockEvrenProvider(resolved, latency=latency, emit_tools=emit_tools)
    router = LLMRouter(resolved, candidates=[provider])
    return router, provider
