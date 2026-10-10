"""End-to-end orchestrator smoke test against an in-process mock provider."""

from __future__ import annotations

from agents.orchestrator import DebateOrchestrator
from config import load_settings
from tests.mock_evren import build_mock_router


async def test_orchestrator_with_mock_provider_runs() -> None:
    settings = load_settings()
    router, provider = build_mock_router(settings, latency=0.001)

    orchestrator = DebateOrchestrator(router, rounds=2)
    events = [event async for event in orchestrator.run("AAPL")]

    kinds = [event.kind for event in events]
    assert kinds[-1] == "done"
    assert "verdict" in kinds
    assert "data" in kinds
    assert provider.calls, "the mock provider should have been called"
