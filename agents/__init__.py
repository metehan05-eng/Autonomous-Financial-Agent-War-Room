"""Multi-agent debate package."""

from __future__ import annotations

from agents.base_agent import AgentSpec, BaseAgent
from agents.bearish import BearishAgent
from agents.bullish import BullishAgent
from agents.moderator import ModeratorAgent, Verdict, parse_verdict
from agents.orchestrator import DebateEvent, DebateOrchestrator
from agents.quant import QuantAgent

__all__ = [
    "AgentSpec",
    "BaseAgent",
    "BearishAgent",
    "BullishAgent",
    "DebateEvent",
    "DebateOrchestrator",
    "ModeratorAgent",
    "QuantAgent",
    "Verdict",
    "parse_verdict",
]
