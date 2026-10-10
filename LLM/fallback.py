"""Deterministic rule-based fallback engine.

Used whenever no LLM provider is configured, when every remote endpoint is
unreachable, or when a provider fails mid-debate. It inspects the same
market snapshot the AI agents receive and produces structured, quant-like
prose so the War Room always reaches a verdict.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from data.fetcher import format_large, format_price
from LLM.base import LLMProvider

__all__ = ["FallbackProvider", "derive_verdict", "detect_role"]

_ROLE_MARKERS: tuple[tuple[str, str], ...] = (
    ("chief risk officer", "cro"),
    ("moderator", "cro"),
    ("bullish growth strategist", "bull"),
    ("bearish risk manager", "bear"),
    ("quantitative analyst", "quant"),
)

BULLET = "•"


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def _get(md: Mapping[str, Any], key: str) -> float | None:
    """Return a float metric from the snapshot or ``None``."""
    value = md.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _num(md: Mapping[str, Any], key: str, digits: int = 2) -> str:
    """Format a metric as a plain number (``61.42`` / ``n/a``)."""
    value = _get(md, key)
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _pct(value: float | None, digits: int = 2, signed: bool = True) -> str:
    """Format a percentage (``+1.48%`` / ``n/a``)."""
    if value is None:
        return "n/a"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value:,.{digits}f}%"


def _direction(value: float | None, positive: str, negative: str, neutral: str = "in line with") -> str:
    """Pick a phrase based on the sign of ``value``."""
    if value is None:
        return neutral
    if value > 0:
        return positive
    if value < 0:
        return negative
    return neutral


def _currency(md: Mapping[str, Any]) -> str:
    """Currency code of the snapshot."""
    return str(md.get("currency") or "USD")


def _price(md: Mapping[str, Any], key: str) -> str:
    """Formatted price for a metric."""
    return format_price(_get(md, key), _currency(md))


def detect_role(system_prompt: str) -> str:
    """Map a system prompt onto an agent role key.

    Args:
        system_prompt: Persona definition written by the agent classes.

    Returns:
        One of ``cro``, ``bull``, ``bear``, ``quant`` or ``quant``.
    """
    lowered = (system_prompt or "").lower()
    for marker, role in _ROLE_MARKERS:
        if marker in lowered:
            return role
    return "quant"


def _summary_snippet(md: Mapping[str, Any], limit: int = 190) -> str:
    """First sentence of the company summary, trimmed to ``limit`` chars."""
    summary = str(md.get("summary") or "").strip()
    if not summary:
        return ""
    match = re.match(r"(.*?[.!?])(\s|$)", summary)
    sentence = match.group(1) if match else summary
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence.rstrip(". ")


def _ticker(md: Mapping[str, Any]) -> str:
    """Symbol of the snapshot."""
    return str(md.get("ticker") or "the ticker")


def _name(md: Mapping[str, Any]) -> str:
    """Company/asset name of the snapshot."""
    return str(md.get("name") or _ticker(md))


# --------------------------------------------------------------------------- #
# Verdict derivation (shared with the moderator's parsing fallback)
# --------------------------------------------------------------------------- #
def derive_verdict(md: Mapping[str, Any]) -> dict[str, Any]:
    """Compute a consensus decision from raw indicators.

    A transparent point system scores trend structure, momentum, MACD,
    valuation, extension from the 52-week high, volatility and the daily
    move. The composite score is mapped onto one of
    ``STRONG BUY / BUY / HOLD / SELL / STRONG SELL``.

    Args:
        md: Market snapshot produced by :class:`data.fetcher.MarketDataFetcher`.

    Returns:
        ``{"decision": str, "confidence": int, "rationale": list[str]}``
    """
    sma_50 = _get(md, "sma_50")
    sma_200 = _get(md, "sma_200")
    rsi = _get(md, "rsi_14")
    macd = _get(md, "macd")
    macd_signal = _get(md, "macd_signal")
    forward_pe = _get(md, "forward_pe")
    pct_high = _get(md, "pct_from_52w_high")
    volatility = _get(md, "volatility_ann")
    change_pct = _get(md, "change_pct")
    trend = str(md.get("trend") or "neutral")

    contributions: list[tuple[str, float]] = []

    # Trend structure ------------------------------------------------------
    if sma_50 is not None and sma_200 is not None:
        spread = ((sma_50 - sma_200) / sma_200) * 100.0
        if trend == "strong_uptrend":
            contributions.append(
                (f"Trend: price above a rising 50/200 DMA stack ({_pct(spread)} spread) - strong uptrend", 2.0)
            )
        elif trend == "uptrend":
            contributions.append(
                (f"Trend: price above the 200 DMA, 50/200 spread {_pct(spread)} - uptrend intact", 1.0)
            )
        elif trend == "strong_downtrend":
            contributions.append(
                (f"Trend: price below a falling 50/200 DMA stack ({_pct(spread)} spread) - strong downtrend", -2.0)
            )
        elif trend == "downtrend":
            contributions.append(
                (f"Trend: price below the 200 DMA, 50/200 spread {_pct(spread)} - downtrend", -1.2)
            )
        else:
            contributions.append(("Trend: moving averages are mixed - no clean directional edge", 0.0))

        cross_bias = 0.5 if sma_50 > sma_200 else -0.5
        cross_name = "golden" if sma_50 > sma_200 else "death"
        contributions.append(
            (f"50/200 DMA posture: {cross_name} cross {'supports' if cross_bias > 0 else 'pressures'} price", cross_bias)
        )

    # Momentum -------------------------------------------------------------
    if rsi is not None:
        if rsi >= 70:
            contributions.append((f"RSI(14) at {rsi:.1f} is overbought - elevated pullback risk", -1.5))
        elif rsi >= 60:
            contributions.append((f"RSI(14) at {rsi:.1f} shows strong momentum but is running hot", -0.3))
        elif rsi >= 45:
            contributions.append((f"RSI(14) at {rsi:.1f} is neutral-to-constructive", 0.5))
        elif rsi >= 30:
            contributions.append((f"RSI(14) at {rsi:.1f} is soft - sellers still in control", -0.5))
        else:
            contributions.append((f"RSI(14) at {rsi:.1f} is oversold - bounce potential", 1.2))

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            contributions.append((f"MACD ({macd:.2f}) is above its signal line - bullish crossover bias", 0.7))
        else:
            contributions.append((f"MACD ({macd:.2f}) is below its signal line - bearish crossover bias", -0.7))

    # Valuation ------------------------------------------------------------
    if forward_pe is not None:
        if forward_pe < 15:
            contributions.append((f"Forward P/E of {forward_pe:.1f} is below the 15-20 market norm - valuation support", 1.0))
        elif forward_pe < 25:
            contributions.append((f"Forward P/E of {forward_pe:.1f} is close to fair value", 0.3))
        elif forward_pe < 40:
            contributions.append((f"Forward P/E of {forward_pe:.1f} is a premium multiple - compression risk", -0.6))
        else:
            contributions.append((f"Forward P/E of {forward_pe:.1f} is stretched - priced for perfection", -1.3))

    # Extension from the high ---------------------------------------------
    if pct_high is not None:
        if pct_high > -3:
            contributions.append((f"Trading {abs(pct_high):.1f}% from the 52-week high - little margin of safety", -0.5))
        elif pct_high < -25:
            contributions.append((f"Trading {abs(pct_high):.1f}% below the 52-week high - deep discount / broken chart", 0.3))
        else:
            contributions.append((f"Trading {abs(pct_high):.1f}% below the 52-week high - ordinary pullback range", 0.0))

    # Volatility -----------------------------------------------------------
    if volatility is not None:
        if volatility > 60:
            contributions.append((f"Realised volatility of {volatility:.0f}% annualised implies wide drawdown bands", -1.0))
        elif volatility < 25:
            contributions.append((f"Realised volatility of {volatility:.0f}% annualised is contained", 0.3))

    # Daily impulse --------------------------------------------------------
    if change_pct is not None:
        if change_pct > 2:
            contributions.append((f"Daily impulse of {_pct(change_pct)} - buyers pressing", 0.5))
        elif change_pct < -2:
            contributions.append((f"Daily impulse of {_pct(change_pct)} - sellers pressing", -0.5))

    if not contributions:
        return {
            "decision": "HOLD",
            "confidence": 50,
            "rationale": [
                f"Insufficient structured data for { _ticker(md) }",
                "No technical or fundamental edge detected",
                "Await a fuller dataset before committing capital",
            ],
        }

    score = sum(weight for _, weight in contributions)

    if score >= 2.5:
        decision = "STRONG BUY"
    elif score >= 1.0:
        decision = "BUY"
    elif score > -1.0:
        decision = "HOLD"
    elif score > -2.5:
        decision = "SELL"
    else:
        decision = "STRONG SELL"

    if decision == "HOLD":
        confidence = round(50 + (1.0 - abs(score)) * 14)
        confidence = max(45, min(80, confidence))
    else:
        confidence = round(55 + abs(score) * 7)
        confidence = max(55, min(95, confidence))

    ranked = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
    rationale = [text for text, _ in ranked[:3]]
    defaults = [
        f"Composite indicator score of {score:+.1f} points on a -6..+6 scale",
        f"{md.get('trend_label', 'Neutral')} structure on the daily chart",
        "Position sizing should stay disciplined while signals remain mixed",
    ]
    for filler in defaults:
        if len(rationale) >= 3:
            break
        rationale.append(filler)

    return {
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale[:3],
        "score": round(score, 2),
    }


# --------------------------------------------------------------------------- #
# Argument builders
# --------------------------------------------------------------------------- #
def _bull_pitch(md: Mapping[str, Any]) -> str:
    """Opening statement of the Bullish Growth Strategist."""
    lines = [
        f"Opening thesis on {_ticker(md)} ({_name(md)}) - the path of least resistance is UP.",
    ]

    trend_label = md.get("trend_label", "Neutral")
    sma_50, sma_200 = _get(md, "sma_50"), _get(md, "sma_200")
    if sma_50 is not None and sma_200 is not None:
        vs50 = _get(md, "price_vs_sma_50")
        cross = "golden cross" if sma_50 > sma_200 else "death cross"
        lines.append(
            f"{BULLET} Trend structure: price is {_pct(vs50)} vs the 50-day SMA "
            f"({format_price(sma_50, _currency(md))}) and {_pct(_get(md, 'price_vs_sma_200'))} "
            f"vs the 200-day SMA ({format_price(sma_200, _currency(md))}) - {trend_label}, "
            f"with a {cross} regime in force."
        )
    else:
        lines.append(
            f"{BULLET} Trend structure: {trend_label} - the primary trend still favours the bulls."
        )

    rsi = _get(md, "rsi_14")
    if rsi is not None:
        zone = str(md.get("rsi_zone", "neutral")).lower()
        lines.append(
            f"{BULLET} Momentum: RSI(14) = {rsi:.1f} ({zone}) and MACD histogram "
            f"{_num(md, 'macd_histogram')} - buying pressure is intact, not exhausted."
        )

    pct_high = _get(md, "pct_from_52w_high")
    if pct_high is not None and pct_high < -1:
        upside = abs(pct_high) / (100.0 + pct_high) * 100.0
        lines.append(
            f"{BULLET} Upside optionality: still {abs(pct_high):.1f}% below the 52-week high "
            f"({_price(md, 'fifty_two_week_high')}) - a simple retest implies roughly "
            f"{upside:.1f}% upside from here."
        )
    else:
        lines.append(
            f"{BULLET} Upside optionality: price is pressing fresh highs - momentum buyers "
            f"are getting validated, not punished."
        )

    volatility = _get(md, "volatility_ann")
    volume_ratio = _get(md, "volume_ratio")
    extras = []
    if volatility is not None:
        extras.append(f"realised volatility of {volatility:.0f}% p.a. is the fair cost of that trend")
    if volume_ratio is not None and volume_ratio >= 1.0:
        extras.append(f"volume running {volume_ratio:.2f}x the 20-day average confirms participation")
    if extras:
        lines.append(f"{BULLET} Participation: {'; '.join(extras)}.")

    pe = _get(md, "forward_pe")
    snippet = _summary_snippet(md)
    catalyst_bits = []
    if str(md.get("sector") or "n/a") not in ("", "n/a"):
        catalyst_bits.append(f"{md['sector']} exposure")
    if pe is not None:
        catalyst_bits.append(f"forward P/E of {pe:.1f} on a {format_large(_get(md, 'market_cap'))} cap base")
    if snippet:
        catalyst_bits.append(snippet)
    if catalyst_bits:
        lines.append(f"{BULLET} Catalyst: {'; '.join(catalyst_bits)}.")

    return "\n".join(lines)


def _bear_pitch(md: Mapping[str, Any]) -> str:
    """Opening statement of the Bearish Risk Manager."""
    lines = [
        f"Opening thesis on {_ticker(md)} ({_name(md)}) - the risk/reward is skewed DOWN.",
    ]

    pe = _get(md, "forward_pe")
    if pe is not None:
        if pe >= 25:
            verdict = "well above the 15-20 long-run market average - multiple compression is the path of least resistance"
        else:
            verdict = "no longer cheap once growth slows even slightly"
        lines.append(
            f"{BULLET} Valuation: forward P/E of {pe:.1f} is {verdict}."
        )
    else:
        lines.append(
            f"{BULLET} Valuation: no usable multiple on this instrument - price is pure "
            f"sentiment, which means no fundamental floor either."
        )

    pct_high = _get(md, "pct_from_52w_high")
    if pct_high is not None:
        if pct_high > -5:
            lines.append(
                f"{BULLET} Positioning: only {abs(pct_high):.1f}% from the 52-week high "
                f"({_price(md, 'fifty_two_week_high')}) - late buyers are paying peak prices "
                f"with the easy money already made."
            )
        else:
            lines.append(
                f"{BULLET} Positioning: {abs(pct_high):.1f}% off the 52-week high and "
                f"{str(md.get('trend_label', 'neutral')).lower()} - this is a distribution "
                f"pattern, not a base."
            )

    zone = str(md.get("rsi_zone", "n/a")).lower()
    lines.append(
        f"{BULLET} Technicals: RSI(14) = {_num(md, 'rsi_14')} ({zone}), price "
        f"{_pct(_get(md, 'price_vs_sma_50'))} vs the 50-day SMA and MACD histogram "
        f"{_num(md, 'macd_histogram')} - downside momentum can build faster than it looks."
    )

    vol = _get(md, "volatility_ann")
    low = _get(md, "fifty_two_week_low")
    bits = []
    if vol is not None:
        bits.append(f"{vol:.0f}% annualised volatility (≈{vol / 15.87:.1f}% "
                    f"1-sigma daily moves)")
    if low is not None:
        bits.append(f"the 52-week low at {format_price(low, _currency(md))} is the "
                    f"next structural shelf")
    if bits:
        lines.append(f"{BULLET} Downside map: {'; '.join(bits)}.")

    beta = _get(md, "beta")
    macro = []
    if beta is not None:
        macro.append(f"beta of {beta:.2f} amplifies any market-wide de-rating")
    macro.append("rates, inflation and liquidity surprises remain the black-swan transmission belt")
    macro.append("crowded positioning leaves no cushion if guidance slips")
    lines.append(f"{BULLET} Black-swan surface: {'; '.join(macro)}.")

    return "\n".join(lines)


def _quant_pitch(md: Mapping[str, Any]) -> str:
    """Opening statement of the Quantitative Analyst."""
    lines = [
        f"Data read on {_ticker(md)} ({_name(md)}) - numbers first, narrative second.",
    ]

    rsi = _get(md, "rsi_14")
    if rsi is not None:
        zone = str(md.get("rsi_zone", "n/a")).lower()
        lines.append(
            f"{BULLET} RSI(14): {rsi:.2f} -> {zone}; the 70/30 thresholds are "
            f"{'not yet' if 30 < rsi < 70 else 'currently'} tripped."
        )

    sma_50, sma_200 = _get(md, "sma_50"), _get(md, "sma_200")
    if sma_50 is not None and sma_200 is not None:
        lines.append(
            f"{BULLET} Moving averages: SMA50 {format_price(sma_50, _currency(md))}, "
            f"SMA200 {format_price(sma_200, _currency(md))}; price {_pct(_get(md, 'price_vs_sma_50'))} "
            f"/ {_pct(_get(md, 'price_vs_sma_200'))} respectively, 50-200 spread "
            f"{_pct(_get(md, 'sma_spread'))} ({'bullish' if sma_50 > sma_200 else 'bearish'} regime)."
        )

    macd, signal = _get(md, "macd"), _get(md, "macd_signal")
    if macd is not None and signal is not None:
        posture = "above" if macd > signal else "below"
        lines.append(
            f"{BULLET} MACD(12,26,9): line {macd:.3f} vs signal {signal:.3f} "
            f"(histogram {_num(md, 'macd_histogram', 3)}) - momentum {posture} zero-line bias."
        )

    vol = _get(md, "volatility_ann")
    vol_ratio = _get(md, "volume_ratio")
    position = _get(md, "range_position")
    parts = []
    if vol is not None:
        parts.append(f"21d annualised vol {vol:.1f}%")
    if vol_ratio is not None:
        parts.append(f"volume {vol_ratio:.2f}x its 20-day mean")
    if position is not None:
        parts.append(f"52-week range position {position:.1f}% (0=low, 100=high)")
    if parts:
        lines.append(f"{BULLET} Volatility & volume: {'; '.join(parts)}.")

    score_hint = derive_verdict(md).get("score", 0.0)
    bias = "bullish" if score_hint > 1 else "bearish" if score_hint < -1 else "neutral"
    lines.append(
        f"{BULLET} Composite: trend + momentum + valuation score {score_hint:+.1f} "
        f"(range -6..+6) -> quant bias {bias}. Invalidation sits at the "
        f"{('200' if bias != 'bullish' else '50')}-day moving average."
    )
    return "\n".join(lines)


def _cross_bull(md: Mapping[str, Any]) -> str:
    """Cross-examination remarks of the Bull."""
    sma_50 = _get(md, "sma_50")
    rsi = _get(md, "rsi_14")
    lines = [f"Rebuttal to the Bear's downside case on {_ticker(md)}:"]
    if sma_50 is not None:
        lines.append(
            f"{BULLET} The 'crack-up' thesis has no technical trigger yet: price is still "
            f"{_pct(_get(md, 'price_vs_sma_50'))} versus the 50-day SMA "
            f"({format_price(sma_50, _currency(md))}). Break that and I will listen."
        )
    pe = _get(md, "forward_pe")
    if pe is not None:
        lines.append(
            f"{BULLET} Valuation alone never timed a single trade. At forward P/E {pe:.1f} "
            f"the market is funding growth, and growth compounds - the Bear is pricing "
            f"average outcomes onto an above-average story."
        )
    if rsi is not None:
        lines.append(
            f"{BULLET} RSI {rsi:.1f} ({str(md.get('rsi_zone', 'neutral')).lower()}) is momentum, "
            f"not blow-off exhaustion; overbought readings persist precisely in strong trends."
        )
    pct_high = _get(md, "pct_from_52w_high")
    if pct_high is not None:
        lines.append(
            f"{BULLET} And the 'priced for perfection' claim is odd when we trade "
            f"{abs(pct_high):.1f}% {'below' if pct_high < 0 else 'from'} the 52-week high."
        )
    return "\n".join(lines)


def _cross_bear(md: Mapping[str, Any]) -> str:
    """Cross-examination remarks of the Bear."""
    pe = _get(md, "forward_pe")
    lines = [f"Challenging the Bull's upside pitch on {_ticker(md)}:"]
    if pe is not None:
        lines.append(
            f"{BULLET} The momentum narrative ignores the price of admission: forward P/E "
            f"{pe:.1f} against a 15-20 norm means expectations, not earnings, must keep "
            f"surprising. That is a fragile engine."
        )
    else:
        lines.append(
            f"{BULLET} With no fundamental anchor, the Bull's case is pure reflexivity - "
            f"it works until the day it stops, with no valuation cushion underneath."
        )
    rsi = _get(md, "rsi_14")
    if rsi is not None:
        lines.append(
            f"{BULLET} The Bear's math: RSI {rsi:.1f} ({str(md.get('rsi_zone', 'n/a')).lower()}) "
            f"and {_pct(_get(md, 'price_vs_sma_50'))} extension above the 50-day SMA is the "
            f"definition of late-cycle risk."
        )
    vol = _get(md, "volatility_ann")
    if vol is not None:
        daily_2sigma = 2 * vol / 15.87
        lines.append(
            f"{BULLET} At {vol:.0f}% realised volatility a routine 2-sigma daily shakeout "
            f"is ≈{daily_2sigma:.1f}% - an easy {format_price(_get(md, 'fifty_two_week_low'), _currency(md))} "
            f"tour is on the table."
        )
    return "\n".join(lines)


def _cross_quant(md: Mapping[str, Any]) -> str:
    """Cross-examination remarks of the Quant."""
    score = derive_verdict(md).get("score", 0.0)
    bias = "bullish" if score > 1 else "bearish" if score < -1 else "neutral"
    lines = [f"Correcting both desks with the tape on {_ticker(md)}:"]
    rsi = _get(md, "rsi_14")
    if rsi is not None:
        lines.append(
            f"{BULLET} Bull: RSI {rsi:.1f} ({str(md.get('rsi_zone', 'n/a')).lower()}) is the "
            f"ceiling on your optimism; the read is {bias}, not a blank cheque."
        )
    vol = _get(md, "volatility_ann")
    if vol is not None:
        lines.append(
            f"{BULLET} Bear: your drawdown scenario needs {vol:.0f}% realised volatility to "
            f"persist - until the 50-day SMA breaks on a closing basis, that is a hypothesis, "
            f"not a signal."
        )
    macd = _get(md, "macd")
    signal = _get(md, "macd_signal")
    if macd is not None and signal is not None:
        lines.append(
            f"{BULLET} Objective state: MACD {macd:.3f} vs signal {signal:.3f}, 50/200 spread "
            f"{_pct(_get(md, 'sma_spread'))}, composite score {score:+.1f}. Trade the level, "
            f"not the opinion."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
def _verdict_block(verdict: Mapping[str, Any]) -> str:
    """Render the machine readable verdict line."""
    payload = {
        "decision": str(verdict.get("decision", "HOLD")).upper(),
        "confidence": int(verdict.get("confidence", 50)),
        "rationale": [str(item) for item in list(verdict.get("rationale", []))[:3]],
    }
    return json.dumps(payload, ensure_ascii=False)


def render_synthesis(md: Mapping[str, Any], verdict: Mapping[str, Any]) -> str:
    """Compose the CRO's final synthesis text (prose + verdict JSON).

    Args:
        md: Market snapshot.
        verdict: Output of :func:`derive_verdict`.

    Returns:
        Multi-line markdown-ish text ending with a JSON verdict block.
    """
    decision = str(verdict.get("decision", "HOLD")).upper()
    confidence = int(verdict.get("confidence", 50))
    rationale = [str(item) for item in list(verdict.get("rationale", []))[:3]]
    score = verdict.get("score", 0.0)

    lines = [
        f"CRO SYNTHESIS - {_ticker(md)} ({_name(md)})",
        "",
        "After weighing the Bull's momentum case, the Bear's valuation and macro "
        "warnings, and the Quant's objective read of the tape, the desk is "
        f"composite-scored at {float(score):+.1f} (range -6..+6).",
        "",
        f"Consensus: {decision}",
        f"Confidence: {confidence}/100",
        "",
    ]
    lines.extend(f"{BULLET} {item}" for item in rationale)

    invalidation = _get(md, "sma_50") or _get(md, "sma_200")
    if invalidation is not None:
        lines.append(
            f"{BULLET} Risk control: a daily close through "
            f"{format_price(invalidation, _currency(md))} voids this call - size the "
            f"position so that level is survivable."
        )

    lines.extend(
        [
            "",
            "```json",
            _verdict_block(verdict),
            "```",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class FallbackProvider(LLMProvider):
    """Deterministic, offline generator used when no LLM is available."""

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "fallback"

    @property
    def model_label(self) -> str:
        """Engine label shown in the UI."""
        return "rule-engine v1"

    def is_configured(self) -> bool:
        """The rule engine is always available."""
        return True

    async def probe(self) -> bool:
        """The rule engine never fails its probe."""
        return True

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        *,
        market_data: Mapping[str, Any] | None = None,
        phase: str | None = None,
    ) -> str:
        """Produce a structured response for the requested role and phase.

        Args:
            prompt: Request text (unused - the snapshot drives the answer).
            system_prompt: Agent persona used for role detection.
            market_data: Market snapshot.
            phase: ``pitch``, ``cross`` or ``synthesis``.

        Returns:
            Deterministic debate text.
        """
        md: Mapping[str, Any] = market_data or {}
        role = detect_role(system_prompt)
        phase_name = (phase or "pitch").lower()

        if role == "cro" or phase_name == "synthesis":
            verdict = derive_verdict(md)
            return render_synthesis(md, verdict)

        if phase_name.startswith("cross"):
            if role == "bull":
                return _cross_bull(md)
            if role == "bear":
                return _cross_bear(md)
            return _cross_quant(md)

        if role == "bull":
            return _bull_pitch(md)
        if role == "bear":
            return _bear_pitch(md)
        return _quant_pitch(md)
