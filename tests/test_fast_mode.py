"""Fast mode must withhold the market-data tools outside the pitch round."""

from __future__ import annotations

from dataclasses import replace

from agents.orchestrator import DebateOrchestrator
from config import load_settings
from tests.mock_evren import build_mock_router


async def test_fast_mode_disables_tools_in_cross() -> None:
    settings = replace(load_settings(), fast=True, provider="evren", debate_rounds=3)
    router, provider = build_mock_router(settings)

    orchestrator = DebateOrchestrator(router, rounds=3)
    events = [event async for event in orchestrator.run("AAPL")]

    assert events[-1].kind == "done"

    pitch_calls = [call for call in provider.calls if call["phase"] == "pitch"]
    cross_calls = [call for call in provider.calls if call["phase"] == "cross"]

    assert pitch_calls and all(call["tools"] is not None for call in pitch_calls)
    assert cross_calls and all(call["tools"] is None for call in cross_calls)
