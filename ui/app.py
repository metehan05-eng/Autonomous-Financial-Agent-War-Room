"""Main Textual application: layout, events and keyboard shortcuts."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Static
from textual.worker import Worker

from agents.moderator import Verdict
from agents.orchestrator import DebateEvent, DebateOrchestrator
from config import APP_NAME, APP_SUBTITLE, BASE_DIR, Settings, load_settings
from LLM import build_router
from LLM.base import LLMRouter
from ui.transcript import DebateTranscript
from ui.widgets import DebateLog, MetricCard, VerdictBanner

__all__ = ["WarRoomApp"]


class WarRoomApp(App):
    """Autonomous Financial Agent War Room - terminal user interface.

    Layout:

    * Header
    * Body - left sidebar (35%) with the financial metrics card, right
      panel (65%) with the streaming debate log
    * Bottom dock - CRO verdict banner (decision, confidence meter, rationale)
    * Input bar - ticker input plus the *Start War Room* button
    * Footer with keyboard shortcuts
    """

    TITLE = APP_NAME
    SUB_TITLE = APP_SUBTITLE

    CSS = """
    Screen {
        background: $background;
    }

    #body {
        height: 1fr;
    }

    #sidebar {
        width: 35%;
        min-width: 30;
        height: 100%;
        border-right: round $accent;
        padding: 1 1 0 1;
        overflow-y: auto;
    }

    #metrics {
        height: auto;
        padding: 0 1;
    }

    #backend {
        height: auto;
        padding: 1 1;
        color: $text-muted;
        border-top: solid $accent;
    }

    #main_panel {
        width: 65%;
        height: 100%;
        padding: 0 1;
    }

    #debate_log {
        height: 1fr;
        border: round $panel;
    }

    #verdict {
        height: auto;
        min-height: 7;
        max-height: 45%;
        margin: 0 1;
        padding: 0 1;
        border: round $warning;
        background: $surface;
    }

    #dock {
        height: 3;
        dock: bottom;
        padding: 0 1;
    }

    #ticker {
        width: 1fr;
    }

    #start {
        width: 26;
        min-width: 20;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "Quit", key_display="Q"),
        Binding("escape", "cancel_debate", "Cancel", key_display="Esc"),
        Binding("ctrl+s", "save_transcript", "Export", key_display="Ctrl+S"),
        Binding("ctrl+r", "clear_log", "Clear", key_display="Ctrl+R"),
        Binding("ctrl+enter", "start_war_room", "Start", key_display="Ctrl+Enter"),
    ]

    def __init__(
        self,
        initial_ticker: str | None = None,
        settings: Settings | None = None,
        router: LLMRouter | None = None,
    ) -> None:
        """Create the application.

        Args:
            initial_ticker: Ticker pre-filled in the input bar.
            settings: Optional configuration override.
            router: Optional pre-built LLM router (tests).
        """
        super().__init__()
        self.settings: Settings = settings or load_settings()
        self.router: LLMRouter = router or build_router(self.settings)
        self.initial_ticker: str = (
            initial_ticker or self.settings.default_ticker
        ).upper()
        self._run_sequence: int = 0
        self._debate_worker: Worker | None = None
        self.transcript: DebateTranscript = DebateTranscript()

    # ------------------------------------------------------------------ #
    # Composition
    # ------------------------------------------------------------------ #
    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield MetricCard(id="metrics")
                yield Static("Backend: pending probe", id="backend")
            with Vertical(id="main_panel"):
                yield DebateLog(id="debate_log")
        yield VerdictBanner(id="verdict")
        with Horizontal(id="dock"):
            yield Input(
                value=self.initial_ticker,
                placeholder="Ticker — AAPL, BTC-USD, GC=F, EURUSD=X",
                id="ticker",
            )
            yield Button("Start War Room ⚔️", variant="primary", id="start")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the ticker input and greet the operator."""
        self.query_one("#ticker", Input).focus()
        log = self.query_one("#debate_log", DebateLog)
        log.banner(f"{APP_NAME} ready", style="bold yellow")
        log.system(
            "Enter a ticker, press Start (or Enter). Shortcuts: Q quit · Ctrl+R clear log."
        )
        log.system(
            "Provider order: Evren (deepseek-v4.1-flash + glm-5.3 tools) → Ollama → "
            "OpenAI → Gemini → Anthropic. Rule engine only with --no-llm."
        )
        log.system(
            "Reasoning streams live while each agent thinks. Esc=cancel · "
            "Ctrl+S=export transcript. Default: pitch + synthesis; "
            "--rounds 3 adds cross-examination."
        )

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #
    @on(Input.Submitted, "#ticker")
    def _on_ticker_submitted(self, event: Input.Submitted) -> None:
        """Enter inside the ticker input starts the debate."""
        self.action_start_war_room()

    @on(Button.Pressed, "#start")
    def _on_start_pressed(self, event: Button.Pressed) -> None:
        """Start button starts the debate."""
        self.action_start_war_room()

    def action_clear_log(self) -> None:
        """Ctrl+R - clear the debate log and reset the verdict banner."""
        self.query_one("#debate_log", DebateLog).clear()
        self.query_one("#verdict", VerdictBanner).set_idle()
        self.notify("Log cleared", timeout=1.5)

    def action_start_war_room(self) -> None:
        """Start (or restart) the multi-agent debate for the typed ticker."""
        ticker = self.current_ticker()
        if not ticker:
            self.notify("Enter a ticker symbol first", severity="warning", timeout=2)
            return
        self._debate_worker = self.run_debate(ticker)

    def action_cancel_debate(self) -> None:
        """Esc - cancel the running debate cleanly."""
        worker = self._debate_worker
        if worker is None or worker.is_finished:
            self.notify("No debate in progress", timeout=1.5)
            return
        self.notify("Cancelling debate ...", timeout=1.5)
        worker.cancel()

    def action_save_transcript(self) -> None:
        """Ctrl+S - export the current debate transcript to a Markdown file."""
        path = self._save_transcript()
        if path is None:
            self.notify("Nothing to export yet", severity="warning", timeout=2)

    def _save_transcript(self) -> Path | None:
        """Write the accumulated transcript under ``exports/``.

        Returns:
            The written file path, or ``None`` when there is nothing to save.
        """
        if not (self.transcript.ticker or self.transcript.rounds):
            return None
        directory = BASE_DIR / "exports"
        try:
            path = self.transcript.write(directory)
        except Exception as exc:
            self.notify(f"Cannot export transcript: {exc}", severity="error", timeout=3)
            return None
        self.notify(f"Transcript saved: {path.name}", timeout=2)
        return path

    # ------------------------------------------------------------------ #
    # Debate worker
    # ------------------------------------------------------------------ #
    @work(exclusive=True, group="debate")
    async def run_debate(self, ticker: str) -> None:
        """Stream the orchestrated debate into the UI.

        Args:
            ticker: Symbol to analyse. A new invocation cancels any run that
                is still in progress (same worker group).
        """
        log = self.query_one("#debate_log", DebateLog)
        metrics = self.query_one("#metrics", MetricCard)
        banner = self.query_one("#verdict", VerdictBanner)
        backend = self.query_one("#backend", Static)
        start_button = self.query_one("#start", Button)

        self._run_sequence += 1
        token = self._run_sequence

        start_button.disabled = True
        banner.set_working(ticker)
        log.end_all_streams()  # close blocks left over from a cancelled run
        log.banner(f"WAR ROOM CONVENE — {ticker}", style="bold yellow")
        self.transcript.reset()
        self.transcript.set_ticker(ticker)

        orchestrator = DebateOrchestrator(
            self.router,
            rounds=self.settings.debate_rounds,
            period=self.settings.history_period,
        )
        try:
            async for event in orchestrator.run(ticker):
                self._dispatch(event, log, metrics, banner, backend)
                self.transcript.accept(event)
                if event.kind == "done":
                    saved = self._save_transcript()
                    if saved is not None:
                        log.system(f"Transcript saved: {saved}")
        except asyncio.CancelledError:
            # The user pressed Esc: close the live region and stay consistent.
            try:
                log.end_all_streams()
                log.system("— debate cancelled —")
            except Exception:
                pass
            raise
        except Exception as exc:
            with contextlib.suppress(Exception):
                log.error(f"Unexpected failure: {exc}")
        finally:
            # Only the most recent run may restore the idle state; the widget
            # tree may also be gone when the app is shutting down.
            self._debate_worker = None
            if token == self._run_sequence:
                try:
                    start_button.disabled = False
                    ticker_input = self.query_one_optional("#ticker", Input)
                    if ticker_input is not None:
                        ticker_input.focus()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Event dispatch
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dispatch(
        event: DebateEvent,
        log: DebateLog,
        metrics: MetricCard,
        banner: VerdictBanner,
        backend: Static,
    ) -> None:
        """Render a single orchestrator event."""
        if event.kind == "status":
            log.system(event.text)
            if event.text.startswith("LLM backend:"):
                backend.update(f"Backend: {event.text.split(':', 1)[1].strip()}")
        elif event.kind == "data":
            if event.data:
                metrics.update_metrics(event.data)
            log.info(event.text)
        elif event.kind == "round":
            log.separator(event.round_index, event.round_title)
        elif event.kind == "stream_start" and event.agent is not None:
            log.begin_stream(event.agent, event.round_index)
        elif event.kind == "stream_delta" and event.agent is not None:
            log.push_stream(
                event.agent, event.round_index, event.mode or "content", event.text
            )
        elif event.kind == "stream_end" and event.agent is not None:
            log.end_stream(event.agent, event.round_index, event.text)
        elif event.kind == "tool" and event.agent is not None:
            log.stream_tool(event.agent, event.round_index, event.text)
        elif event.kind == "tool_result" and event.agent is not None:
            log.stream_tool_result(event.agent, event.round_index, event.text)
        elif event.kind == "message" and event.agent is not None:
            log.agent_entry(event.agent, event.text)
            log.write("")  # visual gap between statements
        elif event.kind == "verdict":
            verdict: Verdict | None = event.verdict
            if verdict is not None:
                banner.show_verdict(verdict)
                log.banner(
                    f"CONSENSUS: {verdict.normalized} · confidence {verdict.confidence}/100",
                    style=verdict.color,
                )
        elif event.kind == "error":
            log.error(event.text)
        elif event.kind == "done":
            log.end_all_streams()
            log.system("— session closed —")

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def current_ticker(self) -> str:
        """Ticker currently typed in the input bar."""
        return self.query_one("#ticker", Input).value.strip().upper()


def run(ticker: str | None = None) -> None:
    """Launch the TUI (used by ``main.py``).

    Args:
        ticker: Optional ticker to pre-fill.
    """
    WarRoomApp(initial_ticker=ticker).run()
