from __future__ import annotations

from textual.app import App, ComposeResult

from agents.base_agent import AgentSpec
from ui.widgets import DebateLog


class _CountingDebateLog(DebateLog):
    def __init__(self, **kwargs) -> None:
        self.rebuild_count = 0
        super().__init__(**kwargs)

    def _rebuild(self) -> None:
        self.rebuild_count += 1
        super()._rebuild()


class _StreamTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield _CountingDebateLog(id="debate-log")


async def test_stream_deltas_are_batched_for_rendering() -> None:
    app = _StreamTestApp()
    spec = AgentSpec(
        key="test",
        name="Test Agent",
        emoji="T",
        color="green",
        short="TEST",
    )

    async with app.run_test(size=(80, 24)) as pilot:
        log = app.query_one("#debate-log", _CountingDebateLog)
        log.begin_stream(spec, 1)
        rebuilds_before_deltas = log.rebuild_count

        for _ in range(100):
            log.push_stream(spec, 1, "content", "token ")

        assert log.rebuild_count == rebuilds_before_deltas

        await pilot.pause(0.05)

        assert log.rebuild_count == rebuilds_before_deltas + 1
        assert log._streams["1:test"].content == "token " * 100
