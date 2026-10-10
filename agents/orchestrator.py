"""Asynchronous multi-round debate orchestrator.

Drives the four agents through the debate and exposes the whole session as
an asynchronous stream of :class:`DebateEvent` objects that the Textual UI
consumes as they happen:

* **Round 1** - Initial Pitch: Bull, Bear and Quant reason and pitch
  **concurrently** (``asyncio.gather`` over a shared event queue), with
  their chain-of-thought streamed live into the log.
* **Round 2 .. N-1** - Cross-Examination: sequential, every agent receives
  the full transcript of all previous outputs as context.
* **Round N** - CRO Synthesis & Final Verdict: the Chief Risk Officer
  returns the structured consensus (``decision``/``confidence``/
  ``rationale`` JSON) with no tools attached.

Every turn is a real LLM call in the default (strict) mode; the rule
engine only runs with ``provider=fallback`` (``--no-llm``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import AgentSpec, BaseAgent
from agents.bearish import BearishAgent
from agents.bullish import BullishAgent
from agents.moderator import ModeratorAgent, Verdict
from agents.quant import QuantAgent
from data.fetcher import fetch_market_data
from LLM.base import LLMRouter

__all__ = ["DebateEvent", "DebateOrchestrator"]

#: Human readable titles for the three canonical phases.
ROUND_TITLES: dict[int, str] = {
    1: "Initial Pitch",
    2: "Cross-Examination",
    3: "CRO Synthesis & Final Verdict",
}


@dataclass
class DebateEvent:
    """Single item emitted by :meth:`DebateOrchestrator.run`.

    Attributes:
        kind: One of ``status``, ``data``, ``round``, ``stream_start``,
            ``stream_delta``, ``stream_end``, ``tool``, ``tool_result``,
            ``verdict``, ``error`` or ``done``.
        text: Event payload for status/error/tool/verdict events; the full
            statement for ``stream_end``.
        agent: Agent owning a streaming/tool event.
        round_index: 1-based round number.
        round_title: Title of the round (for ``round`` events).
        mode: ``reasoning`` or ``content`` (for ``stream_delta`` events).
        data: Market snapshot carried by the ``data`` event.
        verdict: Consensus carried by the ``verdict`` event.
    """

    kind: str
    text: str = ""
    agent: AgentSpec | None = None
    round_index: int = 0
    round_title: str = ""
    mode: str = ""
    data: dict | None = field(default_factory=dict)
    verdict: Verdict | None = None

    # ------------------------------------------------------------------ #
    @classmethod
    def status(cls, text: str) -> DebateEvent:
        """Progress information shown in dim style."""
        return cls(kind="status", text=text)

    @classmethod
    def round_start(cls, index: int, title: str) -> DebateEvent:
        """Round separator."""
        return cls(kind="round", round_index=index, round_title=title, text=title)

    @classmethod
    def stream_start(cls, agent: AgentSpec, round_index: int) -> DebateEvent:
        """An agent begins streaming a turn."""
        return cls(kind="stream_start", agent=agent, round_index=round_index)

    @classmethod
    def stream_delta(
        cls, agent: AgentSpec, round_index: int, mode: str, text: str
    ) -> DebateEvent:
        """Incremental reasoning or content text of an agent turn.

        Args:
            agent: Speaking agent.
            round_index: Current round.
            mode: ``reasoning`` or ``content``.
            text: Delta fragment to append.
        """
        return cls(
            kind="stream_delta",
            agent=agent,
            round_index=round_index,
            mode=mode,
            text=text,
        )

    @classmethod
    def stream_end(
        cls, agent: AgentSpec, round_index: int, text: str = ""
    ) -> DebateEvent:
        """An agent finished its turn; ``text`` is the full statement."""
        return cls(
            kind="stream_end", agent=agent, round_index=round_index, text=text
        )

    @classmethod
    def tool(cls, agent: AgentSpec, round_index: int, text: str) -> DebateEvent:
        """A tool call was issued by ``agent``."""
        return cls(kind="tool", agent=agent, round_index=round_index, text=text)

    @classmethod
    def tool_result(
        cls, agent: AgentSpec, round_index: int, text: str
    ) -> DebateEvent:
        """The observation of a tool call arrived."""
        return cls(
            kind="tool_result", agent=agent, round_index=round_index, text=text
        )

    @classmethod
    def error(cls, text: str) -> DebateEvent:
        """Recoverable error surfaced in the log."""
        return cls(kind="error", text=text)

    @classmethod
    def done(cls) -> DebateEvent:
        """Terminal event - always the last one."""
        return cls(kind="done")


def _clip(text: str, limit: int = 700) -> str:
    """Trim a statement so prompts stay within sane token bounds."""
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _market_block(data: dict[str, Any]) -> str:
    """Compact, human/LLM friendly rendering of the market snapshot."""
    lines = [
        f"Ticker: {data.get('ticker')} ({data.get('name')})",
        f"Sector: {data.get('sector', 'n/a')} | Currency: {data.get('currency', 'n/a')}",
        f"Price: {data.get('price')} | Change: {data.get('change_pct')}% | "
        f"52w: {data.get('fifty_two_week_low')} - {data.get('fifty_two_week_high')}",
        f"RSI(14): {data.get('rsi_14')} ({data.get('rsi_zone')}) | "
        f"SMA50: {data.get('sma_50')} | SMA200: {data.get('sma_200')} | "
        f"Trend: {data.get('trend_label')}",
        f"MACD: {data.get('macd')} / signal {data.get('macd_signal')} | "
        f"Volatility(ann): {data.get('volatility_ann')}% | "
        f"Volume vs 20d avg: {data.get('volume_ratio')}x",
        f"Forward P/E: {data.get('forward_pe')} | Market cap: {data.get('market_cap')} | "
        f"Beta: {data.get('beta')}",
    ]
    summary = str(data.get("summary") or "")
    if summary:
        lines.append(f"Business: {_clip(summary, 400)}")
    return "\n".join(lines)


def _transcript_block(
    entries: list[tuple[AgentSpec, str]], clip: int = 700
) -> str:
    """Render previous statements for inclusion in a prompt."""
    if not entries:
        return "(no statements yet)"
    blocks = []
    for spec, text in entries:
        blocks.append(f"{spec.emoji} {spec.name}:\n{_clip(text, clip)}")
    return "\n\n".join(blocks)


class DebateOrchestrator:
    """Runs the multi-round debate and streams it as events.

    Args:
        router: Shared LLM router used by every agent.
        rounds: Total number of debate rounds (minimum 2: pitch + synthesis).
        period: History window requested from the market data fetcher.
    """

    def __init__(
        self,
        router: LLMRouter,
        rounds: int = 3,
        period: str = "1y",
    ) -> None:
        self.router = router
        self.rounds: int = max(2, int(rounds or 3))
        self.period = period
        self.bull = BullishAgent(router)
        self.bear = BearishAgent(router)
        self.quant = QuantAgent(router)
        self.cro = ModeratorAgent(router)
        self.speakers: list[BaseAgent] = [self.bull, self.bear, self.quant]

    # ------------------------------------------------------------------ #
    # Prompt builders
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pitch_prompt(data: dict[str, Any]) -> str:
        return (
            "ROUND 1 - INITIAL PITCH\n"
            "Deliver your opening thesis on this instrument based on the snapshot "
            "below. You may first call the market data tools to double check the "
            "figures you want to cite.\n\n"
            f"{_market_block(data)}"
        )

    @staticmethod
    def _cross_prompt(
        data: dict[str, Any],
        transcript: list[tuple[AgentSpec, str]],
        iteration: int,
    ) -> str:
        return (
            f"ROUND 2.{iteration} - CROSS-EXAMINATION\n"
            "Challenge the weakest claims made by the other desks. Quote or paraphrase "
            "the specific claim you reject, then refute it with figures from the "
            "snapshot (or with a fresh tool call). Do not repeat your opening pitch.\n\n"
            f"Everything said so far:\n{_transcript_block(transcript)}\n\n"
            f"{_market_block(data)}"
        )

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    async def run(self, ticker: str) -> AsyncGenerator[DebateEvent, None]:
        """Run the full debate for ``ticker``.

        Args:
            ticker: Yahoo Finance symbol (``AAPL``, ``BTC-USD``, ``GC=F``...).

        Yields:
            :class:`DebateEvent` objects in chronological order; the last one
            is always a ``done`` event (unless the consumer stops early).
        """
        symbol = (ticker or "").strip().upper()
        if not symbol:
            yield DebateEvent.error("Enter a ticker symbol before starting the war room.")
            yield DebateEvent.done()
            return

        yield DebateEvent.status(f"Fetching market data for {symbol} ...")
        try:
            data = await asyncio.to_thread(fetch_market_data, symbol, self.period)
        except Exception as exc:
            yield DebateEvent.error(f"Market data request failed: {exc}")
            yield DebateEvent.done()
            return

        if not data.get("ok"):
            yield DebateEvent.error(str(data.get("error") or "Unknown market data error."))
            yield DebateEvent.done()
            return

        yield DebateEvent(kind="data", data=data, text=f"Snapshot loaded for {symbol}")

        # Strict mode: fail fast (with a clear message) when no LLM answers.
        try:
            provider = await self.router.resolve()
        except Exception as exc:
            yield DebateEvent.error(str(exc))
            yield DebateEvent.done()
            return
        label = f"{provider.name} / {provider.model_label}"
        if provider.name == "fallback":
            label = f"{label} (rule engine - deterministic mode)"
        yield DebateEvent.status(f"LLM backend: {label}")

        transcript: list[tuple[AgentSpec, str]] = []
        cross_rounds = self.rounds - 2

        try:
            # ---------------- Round 1: concurrent pitches ---------------- #
            yield DebateEvent.round_start(1, ROUND_TITLES[1])
            async for event in self._round_one(data, transcript):
                yield event

            # ---------------- Round 2..N-1: cross-examination ------------ #
            for iteration in range(1, cross_rounds + 1):
                round_index = 1 + iteration
                title = (
                    ROUND_TITLES[2]
                    if round_index == 2
                    else f"Cross-Examination ({iteration})"
                )
                yield DebateEvent.round_start(round_index, title)
                for speaker in self.speakers:
                    yield DebateEvent.status(
                        f"{speaker.spec.emoji} {speaker.spec.name} is challenging "
                        "the floor ..."
                    )
                    prompt = self._cross_prompt(data, transcript, iteration)
                    sink: dict[str, str] = {}
                    async for event in self._stream_turn(
                        speaker, prompt, data, "cross", round_index, sink
                    ):
                        yield event
                    if sink.get("text"):
                        transcript.append((speaker.spec, sink["text"]))

            # ---------------- Final round: CRO synthesis ----------------- #
            final_round = 1 + cross_rounds + 1
            yield DebateEvent.round_start(final_round, ROUND_TITLES[3])
            yield DebateEvent.status(
                "⚖️ Chief Risk Officer is weighing every argument ..."
            )
            prompt = self.cro.build_synthesis_prompt(
                symbol, _transcript_block(transcript, clip=900)
            )
            sink = {}
            async for event in self._stream_turn(
                self.cro, prompt, data, "synthesis", final_round, sink
            ):
                yield event
            cro_text = sink.get("text", "")
            verdict = self.cro.to_verdict(
                cro_text, data, source=self._source_label() if cro_text else ""
            )
            yield DebateEvent(kind="verdict", verdict=verdict, text=verdict.decision)
            yield DebateEvent.status(
                f"Debate closed: {verdict.decision} @ {verdict.confidence}/100 "
                f"({verdict.source})"
            )
        except Exception as exc:
            yield DebateEvent.error(f"Debate aborted: {exc}")

        yield DebateEvent.done()

    # ------------------------------------------------------------------ #
    # Round runners
    # ------------------------------------------------------------------ #
    async def _round_one(
        self,
        data: dict[str, Any],
        statements: list[tuple[AgentSpec, str]],
    ) -> AsyncIterator[DebateEvent]:
        """Run the three initial pitches concurrently.

        Every speaker streams into its own sink while a single producer task
        fans their events onto one queue, so the UI receives interleaved
        reasoning/content deltas from all three agents.

        Args:
            data: Market snapshot.
            statements: Transcript list filled in speaker order once every
                pitch has finished.
        """
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        sinks: dict[str, dict[str, str]] = {
            speaker.spec.key: {} for speaker in self.speakers
        }

        async def pump(agent: BaseAgent) -> None:
            sink = sinks[agent.spec.key]
            try:
                async for event in self._stream_turn(
                    agent, self._pitch_prompt(data), data, "pitch", 1, sink
                ):
                    await queue.put(("event", event))
            except Exception as exc:
                await queue.put(
                    ("event", DebateEvent.error(f"{agent.spec.name} failed: {exc}"))
                )

        async def produce() -> None:
            await asyncio.gather(*(pump(agent) for agent in self.speakers))
            await queue.put(("all_done", None))

        producer = asyncio.create_task(produce())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "all_done":
                    break
                yield payload  # type: ignore[misc]
        finally:
            producer.cancel()
            producer.add_done_callback(_consume_task_exception)

        for speaker in self.speakers:
            text = sinks.get(speaker.spec.key, {}).get("text", "")
            if text:
                statements.append((speaker.spec, text))

    async def _stream_turn(
        self,
        agent: BaseAgent,
        prompt: str,
        data: dict[str, Any],
        phase: str,
        round_index: int,
        sink: dict[str, str],
    ) -> AsyncIterator[DebateEvent]:
        """Stream one agent turn as events, never raising.

        Args:
            agent: Speaking agent.
            prompt: Phase specific user prompt.
            data: Market snapshot.
            phase: ``pitch``, ``cross`` or ``synthesis``.
            round_index: Current round number.
            sink: Mutable dict; receives the full statement under ``text``.
        """
        messages = agent.build_messages(prompt)
        yield DebateEvent.stream_start(agent.spec, round_index)

        parts: list[str] = []
        try:
            async for chunk in agent.respond(
                messages, market_data=data, phase=phase
            ):
                if chunk.kind == "reasoning":
                    yield DebateEvent.stream_delta(
                        agent.spec, round_index, "reasoning", chunk.text
                    )
                elif chunk.kind == "content":
                    parts.append(chunk.text)
                    yield DebateEvent.stream_delta(
                        agent.spec, round_index, "content", chunk.text
                    )
                elif chunk.kind == "tool":
                    label = f"🔧 {chunk.tool_name}({chunk.tool_args})"
                    yield DebateEvent.tool(agent.spec, round_index, label)
                elif chunk.kind == "tool_result":
                    yield DebateEvent.tool_result(
                        agent.spec, round_index, f"↳ {chunk.text}"
                    )
                elif chunk.kind == "error":
                    yield DebateEvent.error(chunk.text)
        except Exception as exc:
            yield DebateEvent.error(f"{agent.spec.name} failed: {exc}")

        text = "".join(parts).strip()
        sink["text"] = text
        yield DebateEvent.stream_end(agent.spec, round_index, text)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _source_label(self) -> str:
        """Provider/model label stored on the final verdict."""
        name = self.router.active_name
        model = self.router.active_model
        if not name or name == "pending":
            return "llm"
        if name == "fallback":
            return "rules"
        return f"{name}/{model}" if model else name


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    """Retrieve a task result so cancellations never warn on shutdown."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
