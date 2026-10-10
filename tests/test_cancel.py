"""Cancellation behaviour: an interrupted debate must close cleanly."""

from __future__ import annotations

from agents.orchestrator import DebateOrchestrator
from config import load_settings
from tests.mock_evren import build_mock_router


async def test_cancel_mid_run() -> None:
    settings = load_settings()
    router, _ = build_mock_router(settings, latency=0.01)

    orchestrator = DebateOrchestrator(router, rounds=3)
    stream = orchestrator.run("AAPL")

    first = await stream.__anext__()
    assert first.kind == "status"

    await stream.aclose()
    assert True
