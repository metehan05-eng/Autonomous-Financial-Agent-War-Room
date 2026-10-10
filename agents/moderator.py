"""⚖️ Chief Risk Officer (moderator) agent and the consensus verdict."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from agents.base_agent import AgentSpec, BaseAgent
from LLM.fallback import derive_verdict

__all__ = ["DECISIONS", "ModeratorAgent", "Verdict", "parse_verdict"]

#: Consensus decisions ordered from most to least bullish.
DECISIONS: tuple[str, ...] = ("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL")

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class Verdict:
    """Final CRO consensus handed to the UI."""

    decision: str
    confidence: int
    rationale: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""
    source: str = "fallback"

    @property
    def normalized(self) -> str:
        """Decision restricted to :data:`DECISIONS` (defaults to ``HOLD``)."""
        decision = (self.decision or "HOLD").upper().strip()
        return decision if decision in DECISIONS else "HOLD"

    @property
    def is_bullish(self) -> bool:
        """True for BUY decisions."""
        return self.normalized in ("BUY", "STRONG BUY")

    @property
    def is_bearish(self) -> bool:
        """True for SELL decisions."""
        return self.normalized in ("SELL", "STRONG SELL")

    @property
    def emoji(self) -> str:
        """Icon matching the consensus direction."""
        if self.normalized == "STRONG BUY":
            return "🚀"
        if self.normalized == "BUY":
            return "📈"
        if self.normalized == "SELL":
            return "📉"
        if self.normalized == "STRONG SELL":
            return "💥"
        return "⚖️"

    @property
    def color(self) -> str:
        """Rich color name matching the consensus direction."""
        if self.is_bullish:
            return "green" if self.normalized == "BUY" else "bold green"
        if self.is_bearish:
            return "red" if self.normalized == "SELL" else "bold red"
        return "yellow"

    @property
    def level(self) -> float:
        """Confidence expressed as 0.0 - 1.0."""
        return max(0.0, min(100.0, float(self.confidence))) / 100.0


def _coerce_verdict(raw: Mapping[str, Any]) -> Verdict | None:
    """Validate a decoded JSON object and turn it into a :class:`Verdict`."""
    decision = str(raw.get("decision", "")).upper().replace("-", " ").strip()
    aliases = {
        "STRONGBUY": "STRONG BUY",
        "STRONG SELL": "STRONG SELL",
        "STRONG BUY": "STRONG BUY",
        "OVERWEIGHT": "BUY",
        "UNDERWEIGHT": "SELL",
        "NEUTRAL": "HOLD",
        "ACCUMULATE": "BUY",
        "REDUCE": "SELL",
    }
    decision = aliases.get(decision, decision)
    if decision not in DECISIONS:
        return None

    try:
        confidence = round(float(raw.get("confidence", 50)))
    except (TypeError, ValueError):
        confidence = 50
    confidence = max(0, min(100, confidence))

    rationale_raw = raw.get("rationale")
    rationale: list[str] = []
    if isinstance(rationale_raw, Sequence) and not isinstance(rationale_raw, str):
        rationale = [str(item).strip() for item in rationale_raw if str(item).strip()]
    elif isinstance(rationale_raw, str) and rationale_raw.strip():
        rationale = [rationale_raw.strip()]
    if not rationale:
        return None

    return Verdict(
        decision=decision,
        confidence=confidence,
        rationale=tuple(rationale[:3]),
        summary=str(raw.get("summary", "") or ""),
        source=str(raw.get("source", "") or ""),
    )


def parse_verdict(text: str) -> Verdict | None:
    """Extract the JSON verdict block written by the CRO.

    Several shapes are accepted: a fenced ```json block, a trailing bare
    object, or the last JSON object anywhere in the answer.

    Args:
        text: Raw moderator output.

    Returns:
        A :class:`Verdict` or ``None`` when the payload is missing/invalid.
    """
    if not text:
        return None

    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))

    brace_start = text.rfind("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1])

    candidates.extend(match.group(0) for match in _OBJECT_RE.finditer(text))

    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(decoded, dict) or "decision" not in decoded:
            continue
        verdict = _coerce_verdict(decoded)
        if verdict is not None:
            return verdict
    return None


class ModeratorAgent(BaseAgent):
    """⚖️ Chief Risk Officer: moderates, filters and delivers the verdict."""

    spec: ClassVar[AgentSpec] = AgentSpec(
        key="cro",
        name="Chief Risk Officer",
        emoji="⚖️",
        color="yellow",
        short="CRO",
    )

    #: The CRO reasons over the transcript only - no market data tools.
    uses_tools: ClassVar[bool] = False

    system_prompt: ClassVar[str] = (
        "You are the CHIEF RISK OFFICER (⚖️) and moderator of the Autonomous Financial "
        "Agent War Room, a live terminal debate between a 🐂 Bullish Growth Strategist, "
        "a 🐻 Bearish Risk Manager and a 📐 Quantitative Analyst.\n"
        "Your persona: sober, decisive and evidence-driven. You filter out weak or "
        "unsupported arguments, weigh the strongest points from each desk against the "
        "market snapshot, and synthesise a single consensus.\n"
        "Output format - follow it exactly:\n"
        "1. First line: 'CRO SYNTHESIS - <TICKER>'\n"
        "2. Two or three sentences weighing the three desks and naming the decisive evidence.\n"
        "3. A blank line, then exactly: 'Consensus: <STRONG BUY|BUY|HOLD|SELL|STRONG SELL>'\n"
        "4. Then exactly: 'Confidence: <0-100>/100'\n"
        "5. Then exactly three rationale bullets, each on its own line starting with a "
        "bullet character, each a concrete, data-backed reason (max 25 words each).\n"
        "6. End with a fenced ```json code block containing exactly this object:\n"
        '{"decision": "<STRONG BUY|BUY|HOLD|SELL|STRONG SELL>", "confidence": <int>, '
        '"rationale": ["<reason 1>", "<reason 2>", "<reason 3>"]}\n'
        "Never output any other JSON, never exceed 220 words in total."
    )

    def build_synthesis_prompt(self, ticker: str, transcript: str) -> str:
        """Compose the synthesis request for the final round.

        Args:
            ticker: Symbol under debate.
            transcript: Joined statements of rounds 1..N-1.

        Returns:
            The user prompt carrying the full transcript.
        """
        return (
            f"ROUND 3 - CRO SYNTHESIS & FINAL VERDICT\n"
            f"Ticker: {ticker}\n\n"
            f"Full debate transcript:\n{transcript}\n\n"
            f"Produce your synthesis now, ending with the JSON verdict block exactly as "
            f"specified in your instructions."
        )

    def to_verdict(
        self,
        text: str,
        market_data: Mapping[str, Any],
        source: str = "llm",
    ) -> Verdict:
        """Turn the streamed CRO answer into a :class:`Verdict`.

        Args:
            text: Raw moderator output of the synthesis turn.
            market_data: Current market snapshot.
            source: Provider label stored on the verdict.

        Returns:
            The parsed consensus; when the answer carries no valid JSON the
            decision is derived deterministically from the indicators (with
            ``source`` suffixed ``+rules``) so a consensus always exists.
        """
        parsed = parse_verdict(text)

        if parsed is None:
            rules = derive_verdict(market_data)
            decision = str(rules["decision"])
            confidence = int(rules["confidence"])
            rationale: list[str] = [str(item) for item in rules["rationale"]]
            source = f"{source}+rules"
        else:
            decision = parsed.decision
            confidence = parsed.confidence
            rationale = list(parsed.rationale)

        # Guarantee exactly three bullets even when the LLM is terse.
        if len(rationale) < 3:
            for item in derive_verdict(market_data)["rationale"]:
                if len(rationale) >= 3:
                    break
                if item not in rationale:
                    rationale.append(str(item))

        return Verdict(
            decision=decision,
            confidence=confidence,
            rationale=tuple(rationale[:3]),
            summary=text,
            source=source,
        )
