"""Custom Rich/Textual widgets: metric card, debate log and verdict banner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rich.cells import cell_len
from rich.measure import measure_renderables
from rich.segment import Segment
from rich.text import Text
from textual.events import Resize
from textual.geometry import Size
from textual.strip import Strip
from textual.widgets import RichLog, Static

from agents.base_agent import AgentSpec
from agents.moderator import Verdict
from data.fetcher import format_large, format_price, sparkline

__all__ = ["DebateLog", "MetricCard", "VerdictBanner", "confidence_meter"]

#: Characters of reasoning kept in memory for one live entry.
_MAX_REASONING_CHARS = 4000
#: Characters of reasoning shown in the rendered log line.
_VISIBLE_REASONING_CHARS = 600


def _row(label: str, value: str, value_style: str = "") -> Text:
    """Render one aligned ``label  value`` line."""
    line = Text()
    line.append(f"{label:<15}", style="dim")
    line.append(value, style=value_style or "bold not italic")
    return line


def _pct_style(value: float | None) -> str:
    """Rich style for a signed percentage."""
    if value is None:
        return "dim"
    return "bold green" if value >= 0 else "bold red"


def confidence_meter(confidence: float, width: int = 24) -> Text:
    """Render a block based confidence meter.

    Args:
        confidence: Score between 0 and 100.
        width: Number of characters of the meter.

    Returns:
        A :class:`rich.text.Text` with filled/empty blocks.
    """
    score = max(0.0, min(100.0, float(confidence)))
    filled = round((score / 100.0) * width)
    if score >= 70:
        style = "bold green"
    elif score >= 45:
        style = "bold yellow"
    else:
        style = "bold red"
    meter = Text()
    meter.append("█" * filled, style=style)
    meter.append("░" * (width - filled), style="dim")
    return meter


class MetricCard(Static):
    """Left sidebar card showing the live financial metrics of a ticker."""

    def on_mount(self) -> None:
        """Render the empty state before the first fetch."""
        self.show_placeholder()

    def show_placeholder(self) -> None:
        """Display the pre-run placeholder content."""
        body = Text()
        body.append("No snapshot loaded\n\n", style="bold yellow")
        body.append("Enter a ticker and press\n", style="dim")
        body.append("  Start War Room ⚔️\n\n", style="bold")
        body.append("Examples\n", style="underline")
        body.append("  AAPL      equities\n", style="dim")
        body.append("  BTC-USD   crypto\n", style="dim")
        body.append("  GC=F      commodities\n", style="dim")
        body.append("  EURUSD=X  forex\n", style="dim")
        self.update(body)

    def update_metrics(self, data: Mapping[str, Any]) -> None:
        """Render a market snapshot produced by the data fetcher.

        Args:
            data: Snapshot dictionary (``ok`` must be ``True``).
        """
        currency = str(data.get("currency") or "USD")
        price = _fmt_price(data.get("price"), currency)
        change_pct = _num(data.get("change_pct"))
        change_style = _pct_style(data.get("change_pct"))
        arrow = "▲" if (data.get("change_pct") or 0) >= 0 else "▼"

        body = Text()
        header = Text()
        header.append(f"{data.get('ticker', '?')}", style="bold cyan")
        header.append("  ", style="")
        header.append(str(data.get("name", ""))[:26], style="bold")
        body.append_text(header)
        body.append("\n")

        sector = str(data.get("sector") or "n/a")
        body.append_text(_row("Sector", sector[:22]))
        body.append("\n")

        price_line = Text()
        price_line.append(f"{'Price':<15}", style="dim")
        price_line.append(price, style="bold white")
        price_line.append(f"  {arrow} {change_pct}%", style=change_style)
        body.append_text(price_line)
        body.append("\n")

        body.append_text(_row("Prev close", _fmt_price(data.get("previous_close"), currency)))
        body.append("\n")
        body.append_text(_row("52w range", _range(data, currency)))
        body.append("\n")
        body.append_text(
            _row(
                "From 52w high",
                "n/a"
                if data.get("pct_from_52w_high") is None
                else f"{data['pct_from_52w_high']:+.2f}%",
                _pct_style(data.get("pct_from_52w_high")),
            )
        )
        body.append("\n\n")

        body.append_text(_row("RSI (14)", _num(data.get("rsi_14")), _rsi_style(data.get("rsi_14"))))
        body.append("  ")
        body.append(str(data.get("rsi_zone") or ""), style=_rsi_style(data.get("rsi_14")))
        body.append("\n")
        body.append_text(
            _row(
                "SMA 50",
                _fmt_price(data.get("sma_50"), currency),
                _above_below(data.get("price_vs_sma_50")),
            )
        )
        body.append("\n")
        body.append_text(
            _row(
                "SMA 200",
                _fmt_price(data.get("sma_200"), currency),
                _above_below(data.get("price_vs_sma_200")),
            )
        )
        body.append("\n")
        body.append_text(_row("Trend", str(data.get("trend_label") or "n/a"), _trend_style(data)))
        body.append("\n")
        body.append_text(
            _row("MACD", _num(data.get("macd"), 3), _pct_style(data.get("macd_histogram")))
        )
        body.append("\n\n")

        body.append_text(_row("Fwd P/E", _num(data.get("forward_pe"), 2)))
        body.append("\n")
        body.append_text(_row("Market cap", format_large(_as_float(data.get("market_cap")))))
        body.append("\n")
        body.append_text(_row("Beta", _num(data.get("beta"), 2)))
        body.append("\n")
        body.append_text(
            _row("Volatility", f"{_num(data.get('volatility_ann'), 1)}% (ann.)")
        )
        body.append("\n")
        body.append_text(
            _row("Volume", f"{_num(data.get('volume_ratio'), 2)}x 20d avg")
        )
        body.append("\n\n")

        series: Sequence[float] = data.get("sparkline") or []
        if series:
            body.append_text(_row("Trend (40d)", ""))
            body.append("\n")
            body.append("  ")
            body.append(sparkline([float(v) for v in series], width=30), style="cyan")
            body.append("\n")

        body.append("\n")
        body.append(str(data.get("fetched_at") or ""), style="dim italic")
        self.update(body)


@dataclass
class _StreamEntry:
    """Live rendering state of one agent turn inside the debate log.

    One entry exists per (round, agent) pair while that agent streams. The
    rendered strips are cached in ``frozen`` once the turn finished, so
    re-renders triggered by *other* agents never repaint completed turns.
    """

    spec: AgentSpec
    round_index: int
    reasoning: str = ""
    content: str = ""
    tools: list[str] = field(default_factory=list)
    finished: bool = False
    frozen: list[Strip] | None = None


class DebateLog(RichLog):
    """Color-coded streaming dialogue of the four agents.

    A ``RichLog`` is used instead of the plain ``Log`` widget because agent
    entries must render Rich styles (per-agent colors, separators, banners).

    On top of the append-only log, the widget keeps a *live region* at the
    bottom of the strip list: while an agent streams, its reasoning and
    answer deltas are re-rendered in place (reasoning line first, then the
    statement), so three concurrent speakers in Round 1 each update their
    own block. When a turn ends the block is frozen and later committed to
    the permanent part of the log.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("wrap", True)
        kwargs.setdefault("markup", False)
        kwargs.setdefault("highlight", False)
        kwargs.setdefault("auto_scroll", True)
        kwargs.setdefault("max_lines", 4000)
        kwargs.setdefault("min_width", 32)
        super().__init__(**kwargs)
        self._streams: dict[str, _StreamEntry] = {}
        self._stream_order: list[str] = []
        self._anchor: int = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def clear(self) -> DebateLog:
        """Clear the log and forget every live streaming entry."""
        self._streams.clear()
        self._stream_order.clear()
        self._anchor = 0
        super().clear()
        return self

    def on_resize(self, event: Resize) -> None:
        """Re-anchor and re-render the live region after a size change."""
        known_before = self._size_known
        super().on_resize(event)
        if not known_before:
            # Deferred writes just flushed: the live region starts below them.
            self._anchor = len(self.lines)
        for entry in self._streams.values():
            entry.frozen = None
        if self._streams:
            self._rebuild()

    # ------------------------------------------------------------------ #
    # Write splicing (permanent lines always stay above the live region)
    # ------------------------------------------------------------------ #
    def write(  # type: ignore[override]
        self,
        content: Any,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
        animate: bool = False,
    ) -> DebateLog:
        """Append a permanent line *above* the live streaming region."""
        if not (self._streams and self._size_known):
            super().write(
                content,
                width=width,
                expand=expand,
                shrink=shrink,
                scroll_end=scroll_end,
                animate=animate,
            )
            if not self._streams:
                # No live region: permanent lines end where the log ends, so a
                # later _rebuild must not treat them as live and delete them.
                self._anchor = len(self.lines)
            return self

        live = self.lines[self._anchor :]
        del self.lines[self._anchor :]
        try:
            super().write(
                content,
                width=width,
                expand=expand,
                shrink=shrink,
                scroll_end=False,
                animate=False,
            )
        finally:
            self._anchor = len(self.lines)
            self.lines.extend(live)
        self._line_cache.clear()
        self._update_metrics()
        if scroll_end if scroll_end is not None else self.auto_scroll:
            self.scroll_end(animate=bool(animate), immediate=True, x_axis=False)
        return self

    # ------------------------------------------------------------------ #
    # Streaming API used by the app event dispatch
    # ------------------------------------------------------------------ #
    def begin_stream(self, agent_spec: AgentSpec, round_index: int) -> None:
        """Open a live block for ``agent_spec`` inside ``round_index``."""
        key = self._key(agent_spec, round_index)
        if key not in self._streams:
            self._streams[key] = _StreamEntry(spec=agent_spec, round_index=round_index)
            self._stream_order.append(key)
        self._rebuild()

    def push_stream(
        self, agent_spec: AgentSpec, round_index: int, mode: str, text: str
    ) -> None:
        """Append a streamed delta to a live block.

        Args:
            agent_spec: Speaking agent.
            round_index: Current round.
            mode: ``reasoning`` or ``content``.
            text: Delta fragment.
        """
        entry = self._streams.get(self._key(agent_spec, round_index))
        if entry is None:
            self.begin_stream(agent_spec, round_index)
            entry = self._streams.get(self._key(agent_spec, round_index))
        if entry is None or not text:
            return
        if mode == "reasoning":
            if entry.tools and entry.reasoning and not entry.reasoning.endswith("⋯"):
                # a new thinking segment started after a tool round trip
                entry.reasoning += " ⋯ "
            entry.reasoning = (entry.reasoning + text)[-_MAX_REASONING_CHARS:]
        else:
            entry.content += text
        entry.frozen = None
        self._rebuild()

    def stream_tool(
        self, agent_spec: AgentSpec, round_index: int, text: str
    ) -> None:
        """Show a tool call line inside the agent's live block."""
        entry = self._ensure(agent_spec, round_index)
        if entry is None:
            return
        entry.tools.append(text)
        entry.frozen = None
        self._rebuild()

    def stream_tool_result(
        self, agent_spec: AgentSpec, round_index: int, text: str
    ) -> None:
        """Show the observation line of a tool call."""
        self.stream_tool(agent_spec, round_index, f"    {text}")

    def end_stream(
        self, agent_spec: AgentSpec, round_index: int, final_text: str = ""
    ) -> None:
        """Freeze a live block and commit finished history to the log.

        Args:
            agent_spec: Speaking agent.
            round_index: Current round.
            final_text: Complete statement (replaces streamed content).
        """
        entry = self._streams.get(self._key(agent_spec, round_index))
        if entry is None:
            entry = self._ensure(agent_spec, round_index)
        if entry is None:
            return
        if final_text:
            entry.content = final_text
        entry.finished = True
        entry.frozen = None
        self._rebuild()
        self._commit_finished()

    def end_all_streams(self) -> None:
        """Freeze every live block (used when a run ends or is cancelled)."""
        changed = False
        for key in list(self._stream_order):
            entry = self._streams.get(key)
            if entry is None or entry.finished:
                continue
            entry.finished = True
            entry.frozen = None
            changed = True
        if changed:
            self._rebuild()
        self._commit_finished()

    # ------------------------------------------------------------------ #
    # Static log helpers
    # ------------------------------------------------------------------ #
    def system(self, message: str) -> None:
        """Write a dimmed status line."""
        self.write(Text(message, style="dim"))

    def info(self, message: str) -> None:
        """Write an informational line."""
        self.write(Text(message, style="cyan"))

    def error(self, message: str) -> None:
        """Write an error line."""
        block = Text()
        block.append("⚠ ", style="bold red")
        block.append(message, style="red")
        self.write(block)

    def separator(self, round_index: int, title: str) -> None:
        """Write a round header bar."""
        bar = Text()
        bar.append("─" * 3, style="magenta")
        bar.append(f" ROUND {round_index} · {title.upper()} ", style="bold magenta")
        bar.append("─" * 60, style="magenta")
        self.write(bar)

    def entry(self, badge: str, color: str, body: str) -> None:
        """Write one agent statement with an aligned colored badge.

        Args:
            badge: Label rendered before the statement (``🐂 Bullish ...``).
            color: Rich color/style for the badge.
            body: Statement text; continuation lines are indented.
        """
        prefix = f"{badge}  "
        width = cell_len(prefix)
        lines = (body or "").strip().splitlines() or [""]
        block = Text()
        block.append(prefix, style=f"bold {color}")
        block.append(lines[0])
        for line in lines[1:]:
            block.append("\n")
            block.append(" " * width, style="")
            block.append(line)
        self.write(block)

    def agent_entry(self, agent_spec: Any, body: str) -> None:
        """Convenience wrapper around :meth:`entry` for an ``AgentSpec``."""
        self.entry(f"{agent_spec.emoji} {agent_spec.name}", agent_spec.color, body)

    def banner(self, text: str, style: str = "bold yellow") -> None:
        """Write a full-width highlighted line."""
        line = Text()
        line.append("▌ ", style=style)
        line.append(text, style=style)
        self.write(line)

    # ------------------------------------------------------------------ #
    # Live region internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(agent_spec: AgentSpec, round_index: int) -> str:
        """Dictionary key of one live block."""
        return f"{round_index}:{agent_spec.key}"

    def _ensure(self, agent_spec: AgentSpec, round_index: int) -> _StreamEntry | None:
        """Return (creating when needed) the live block of a turn."""
        key = self._key(agent_spec, round_index)
        entry = self._streams.get(key)
        if entry is None:
            self.begin_stream(agent_spec, round_index)
            entry = self._streams.get(key)
        return entry

    def _rebuild(self) -> None:
        """Re-render the live region from the tracked entries."""
        if not self._size_known or not self._streams:
            return
        if self._anchor > len(self.lines):
            self._anchor = len(self.lines)
        del self.lines[self._anchor :]
        for key in self._stream_order:
            entry = self._streams.get(key)
            if entry is None:
                continue
            if entry.frozen is None:
                entry.frozen = self._render_strips(self._render_entry(entry))
            self.lines.extend(entry.frozen)
        self._line_cache.clear()
        self._update_metrics()
        self.refresh()
        if self.auto_scroll:
            self.scroll_end(immediate=True, x_axis=False)

    def _commit_finished(self) -> None:
        """Promote the finished prefix of the live region to permanent lines."""
        if not self._size_known:
            return
        while self._stream_order:
            key = self._stream_order[0]
            entry = self._streams.get(key)
            if entry is None:
                self._stream_order.pop(0)
                continue
            if not entry.finished or entry.frozen is None:
                break
            count = len(entry.frozen)
            if self._anchor + count > len(self.lines):
                break
            self._anchor += count
            del self._streams[key]
            self._stream_order.pop(0)

    def _render_strips(self, renderable: Text) -> list[Strip]:
        """Render a ``Text`` into wrap-adjusted strips (same as ``write``)."""
        console = self.app.console
        options = console.options
        if not self.wrap:
            options = options.update(overflow="ignore", no_wrap=True)
        width = measure_renderables(console, options, [renderable]).maximum
        region = self.scrollable_content_region.width
        if region and width > region:
            width = region
        width = max(width, self.min_width)
        options = options.update_width(width)
        segments = console.render(renderable, options)
        lines = list(Segment.split_lines(segments))
        if not lines:
            return [Strip.blank(width)]
        strips = Strip.from_lines(lines)
        for strip in strips:
            strip.adjust_cell_length(width)
        return strips

    @staticmethod
    def _render_entry(entry: _StreamEntry) -> Text:
        """Render one live block: reasoning line, tool lines, statement."""
        spec = entry.spec
        body = Text()

        reasoning = entry.reasoning.strip()
        if reasoning:
            if len(reasoning) > _VISIBLE_REASONING_CHARS:
                reasoning = "…" + reasoning[-_VISIBLE_REASONING_CHARS:]
            header = f"🧠 [{spec.name} Thinking]: "
            body.append(header, style="cyan")
            body.append(f"{reasoning}\n", style="dim cyan")

        for line in entry.tools:
            body.append(f"{line}\n", style="dim yellow")

        prefix = f"{spec.emoji} {spec.name}: "
        width = cell_len(prefix)
        content = entry.content.strip()
        if content:
            lines = content.splitlines()
            body.append(prefix, style=f"bold {spec.color}")
            body.append('"', style=f"bold {spec.color}")
            body.append(lines[0])
            for line in lines[1:]:
                body.append("\n")
                body.append(" " * width, style="")
                body.append(line)
            body.append('"', style=f"bold {spec.color}")
            if not entry.finished:
                body.append("▌", style=f"dim {spec.color}")
        else:
            body.append(prefix, style=f"bold {spec.color}")
            if entry.finished:
                body.append("(no answer)", style="dim")
            else:
                body.append("▌", style=f"dim {spec.color}")

        body.append("\n")
        if entry.finished:
            body.append("\n")
        return body

    def _update_metrics(self) -> None:
        """Recompute the widest line and the virtual size of the log."""
        widest = 0
        for strip in self.lines:
            cell = strip.cell_length
            if cell > widest:
                widest = cell
        self._widest_line_width = widest
        self.virtual_size = Size(widest, len(self.lines))


class VerdictBanner(Static):
    """Bottom dock widget showing the CRO consensus and confidence meter."""

    def on_mount(self) -> None:
        """Render the idle state."""
        self.set_idle()

    def set_idle(self) -> None:
        """Reset to the waiting-for-verdict state."""
        body = Text()
        body.append("CRO VERDICT  ", style="bold yellow")
        body.append("awaiting debate ...\n", style="dim")
        body.append("  ", style="")
        body.append("░" * 24, style="dim")
        body.append("  0/100\n", style="dim")
        body.append("  The Chief Risk Officer will publish the consensus here.", style="dim")
        self.update(body)

    def set_working(self, ticker: str) -> None:
        """Show the in-progress state while a debate runs."""
        body = Text()
        body.append("CRO VERDICT  ", style="bold yellow")
        body.append(f"debating {ticker.upper()} ...\n", style="bold cyan")
        body.append("  ", style="")
        body.append("░" * 24, style="dim")
        body.append("  pending\n", style="dim")
        body.append("  Rounds: pitch → cross-examination → synthesis", style="dim")
        self.update(body)

    def show_verdict(self, verdict: Verdict) -> None:
        """Render the final consensus.

        Args:
            verdict: Parsed consensus produced by the CRO.
        """
        body = Text()
        title = Text()
        title.append("CRO VERDICT  ", style="bold yellow")
        title.append(f"{verdict.emoji} {verdict.normalized}", style=f"bold {verdict.color}")
        title.append(f"   confidence {verdict.confidence}/100", style="bold white")
        body.append_text(title)
        body.append("\n  ")
        body.append(confidence_meter(verdict.confidence))
        body.append(f"  {verdict.confidence}%\n", style="bold white")

        for item in verdict.rationale[:3]:
            body.append("  • ", style="bold yellow")
            body.append(str(item))
            body.append("\n")

        source = verdict.source or "rules"
        body.append(f"  source: {source}   ", style="dim")
        body.append("Q quit · Esc cancel · Ctrl+S save", style="dim italic")
        self.update(body)


# --------------------------------------------------------------------------- #
# Formatting helpers used by the metric card
# --------------------------------------------------------------------------- #
def _as_float(value: Any) -> float | None:
    """Coerce a value to ``float`` or ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _num(value: Any, digits: int = 2) -> str:
    """Format a number with fixed precision."""
    number = _as_float(value)
    return "n/a" if number is None else f"{number:,.{digits}f}"


def _fmt_price(value: Any, currency: str) -> str:
    """Format a price with its currency symbol."""
    return format_price(_as_float(value), currency)


def _range(data: Mapping[str, Any], currency: str) -> str:
    """Format the 52-week low/high range."""
    low = _fmt_price(data.get("fifty_two_week_low"), currency)
    high = _fmt_price(data.get("fifty_two_week_high"), currency)
    return f"{low} – {high}"


def _rsi_style(value: Any) -> str:
    """Rich style for an RSI reading."""
    number = _as_float(value)
    if number is None:
        return "dim"
    if number >= 70:
        return "bold red"
    if number >= 55:
        return "bold green"
    if number > 45:
        return "white"
    if number > 30:
        return "bold yellow"
    return "bold cyan"


def _above_below(value: Any) -> str:
    """Rich style describing price vs moving average."""
    number = _as_float(value)
    if number is None:
        return "dim"
    return "bold green" if number >= 0 else "bold red"


def _trend_style(data: Mapping[str, Any]) -> str:
    """Rich style for the trend classification."""
    trend = str(data.get("trend") or "neutral")
    if "up" in trend:
        return "bold green"
    if "down" in trend:
        return "bold red"
    return "bold yellow"
