from __future__ import annotations

from agents.orchestrator import DebateOrchestrator
from config import Settings
from main import parse_args
from tests.mock_evren import build_mock_router


def test_default_cli_runs_two_rounds(monkeypatch) -> None:
    monkeypatch.setattr("main.load_settings", lambda: Settings())

    assert parse_args([]).rounds == 2
    assert parse_args(["--rounds", "3"]).rounds == 3


async def test_default_debate_makes_four_base_llm_calls() -> None:
    settings = Settings()
    router, provider = build_mock_router(settings, emit_tools=False)
    orchestrator = DebateOrchestrator(router, rounds=settings.debate_rounds)

    events = [event async for event in orchestrator.run("AAPL")]

    assert events[-1].kind == "done"
    assert [call["phase"] for call in provider.calls].count("pitch") == 3
    assert [call["phase"] for call in provider.calls].count("cross") == 0
    assert [call["phase"] for call in provider.calls].count("synthesis") == 1
