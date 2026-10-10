"""Markdown transcript of a debate run, assembled from orchestrator events."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agents.base_agent import AgentSpec
from agents.moderator import Verdict
from agents.orchestrator import DebateEvent

__all__ = ["DebateTranscript"]


class RoundBlock:
    """One round: a header plus the agent statements in speaking order."""

    def __init__(self, index: int, title: str) -> None:
        self.index: int = index
        self.title: str = title
        self.agents: list[AgentBlock] = []


class AgentBlock:
    """Everything one agent produced during a single turn."""

    def __init__(self, spec: AgentSpec, round_index: int) -> None:
        self.spec: AgentSpec = spec
        self.round_index: int = round_index
        self.reasoning: list[str] = []
        self.tools: list[str] = []
        self.content: str = ""

    @property
    def thinking(self) -> str:
        """Joined reasoning chain (may span tool round trips)."""
        return "\n".join(part for part in self.reasoning if part)


class DebateTranscript:
    """Accumulates debate events and renders a Markdown report.

    Feed every :class:`DebateEvent` from the orchestrator through
    :meth:`accept`; call :meth:`markdown` (or :meth:`write`) once the run has
    ended or whenever a snapshot is wanted. Reasoning text is preserved in
    full (it can be large with reasoning models), tool observations are
    summarised as ``tool -> N chars received``.
    """

    def __init__(self) -> None:
        self.ticker: str = ""
        self.backend: str = ""
        self.rounds: list[RoundBlock] = []
        self.verdict: Verdict | None = None
        self.ran_to_completion: bool = False
        self._current_round: RoundBlock | None = None
        self._by_key: dict[str, AgentBlock] = {}

    # ------------------------------------------------------------------ #
    def accept(self, event: DebateEvent) -> None:
        """Fold one orchestrator event into the transcript."""
        if event.kind == "status":
            if event.text.startswith("LLM backend:"):
                self.backend = event.text.split(":", 1)[1].strip()
        elif event.kind == "round":
            round_block = RoundBlock(event.round_index, event.round_title)
            self.rounds.append(round_block)
            self._current_round = round_block
        elif event.kind == "stream_start" and event.agent is not None:
            agent_block = self._ensure_agent(event.agent, event.round_index)
            agent_block.reasoning = []
            agent_block.content = ""
        elif event.kind == "stream_delta" and event.agent is not None:
            agent_block = self._ensure_agent(event.agent, event.round_index)
            if event.mode == "reasoning":
                agent_block.reasoning.append(event.text)
            elif event.mode == "content":
                agent_block.content += event.text
        elif event.kind == "stream_end" and event.agent is not None:
            agent_block = self._ensure_agent(event.agent, event.round_index)
            if event.text:
                agent_block.content = event.text
        elif event.kind == "tool" and event.agent is not None:
            agent_block = self._ensure_agent(event.agent, event.round_index)
            agent_block.tools.append(event.text)
        elif event.kind == "tool_result" and event.agent is not None:
            agent_block = self._ensure_agent(event.agent, event.round_index)
            if agent_block.tools:
                agent_block.tools[-1] = f"{agent_block.tools[-1]}  →  {event.text}"
        elif event.kind == "verdict":
            self.verdict = event.verdict
        elif event.kind == "done":
            self.ran_to_completion = True

    def set_ticker(self, ticker: str) -> None:
        """Record the analysed symbol (called when a run convenes)."""
        self.ticker = (ticker or "").upper()

    def reset(self) -> None:
        """Clear the transcript for a new run."""
        self.ticker = ""
        self.backend = ""
        self.rounds = []
        self.verdict = None
        self.ran_to_completion = False
        self._current_round = None
        self._by_key = {}

    # ------------------------------------------------------------------ #
    def _ensure_agent(self, spec: AgentSpec, round_index: int) -> AgentBlock:
        key = f"{round_index}:{spec.key}"
        block = self._by_key.get(key)
        if block is None:
            block = AgentBlock(spec, round_index)
            self._by_key[key] = block
            if self._current_round is None:
                self._current_round = RoundBlock(round_index, f"Round {round_index}")
                self.rounds.append(self._current_round)
            self._current_round.agents.append(block)
        return block

    # ------------------------------------------------------------------ #
    def markdown(self) -> str:
        """Render the collected debate as a Markdown document."""
        lines: list[str] = []
        lines.append(f"# Autonomous Financial Agent War Room — {self.ticker}")
        lines.append("")
        lines.append(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if self.backend:
            lines.append(f"- **Backend:** {self.backend}")
        lines.append(f"- **Status:** {'completed' if self.ran_to_completion else 'partial'}")
        lines.append("")

        if not self.rounds:
            lines.append("_No debate output recorded yet._")
            lines.append("")

        for round_block in self.rounds:
            lines.append(f"## Round {round_block.index} · {round_block.title}")
            lines.append("")
            for block in round_block.agents:
                lines.append(f"### {block.spec.emoji} {block.spec.name}")
                lines.append("")
                if block.thinking:
                    lines.append("**Thinking:**")
                    lines.append("")
                    lines.append("```")
                    lines.extend(block.thinking.splitlines())
                    lines.append("```")
                    lines.append("")
                if block.tools:
                    lines.append("**Tools:**")
                    lines.append("")
                    for tool in block.tools:
                        lines.append(f"- {tool}")
                    lines.append("")
                lines.append("**Statement:**")
                lines.append("")
                depth = "  "
                statement = (block.content or "(no answer)").strip()
                lines.extend(f"{depth}{line}" for line in statement.splitlines())
                lines.append("")

        verdict = self.verdict
        lines.append("## CRO Verdict")
        lines.append("")
        if verdict is None:
            lines.append("_No verdict yet._")
        else:
            lines.append(f"**Decision:** {verdict.emoji} {verdict.normalized}")
            lines.append(f"**Confidence:** {verdict.confidence}/100")
            lines.append("")
            lines.append("**Rationale:**")
            lines.append("")
            for item in verdict.rationale[:3]:
                lines.append(f"- {item}")
            lines.append("")
            if verdict.source:
                lines.append(f"**Source:** {verdict.source}")
                lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def write(self, directory: Path, ticker: str | None = None) -> Path:
        """Write the transcript to ``directory`` and return the file path.

        Args:
            directory: Export folder (created when missing).
            ticker: Override for the file name (defaults to the transcript's
                ticker, sanitised for the filesystem).
        """
        directory.mkdir(parents=True, exist_ok=True)
        symbol = (ticker or self.ticker or "DEBATE").upper()
        safe = "".join(ch for ch in symbol if ch.isalnum()) or "DEBATE"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = directory / f"{safe}-{stamp}.md"
        path.write_text(self.markdown(), encoding="utf-8")
        return path
