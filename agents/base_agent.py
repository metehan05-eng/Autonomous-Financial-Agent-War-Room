"""Base class shared by every debate agent.

Agents no longer return a finished string: they stream their turn as
:class:`AgentChunk` objects (reasoning deltas, answer deltas, tool calls)
so the TUI can render the model's chain of thought while it is generated.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from data.fetcher import TOOL_SCHEMAS, execute_tool
from LLM.base import GenerationResult, LLMRouter, ToolCall

__all__ = ["MAX_TOOL_ROUNDS", "AgentChunk", "AgentSpec", "BaseAgent"]

#: Maximum tool-call round trips allowed inside a single agent turn.
MAX_TOOL_ROUNDS = 3


@dataclass(frozen=True)
class AgentSpec:
    """Static description of an agent used by the UI for rendering."""

    key: str
    name: str
    emoji: str
    color: str
    short: str

    @property
    def badge(self) -> str:
        """Colored, Rich-markup badge for log entries."""
        return f"[bold {self.color}]{self.emoji} {self.name}[/bold {self.color}]"

    @property
    def label(self) -> str:
        """Plain ``emoji name`` label."""
        return f"{self.emoji} {self.name}"


@dataclass(frozen=True)
class AgentChunk:
    """Single piece of an agent turn forwarded to the debate log.

    Attributes:
        kind: ``reasoning`` (chain-of-thought delta), ``content`` (visible
            statement delta), ``tool`` (tool call started), ``tool_result``
            (tool observation returned) or ``error`` (recoverable problem).
        text: Delta / message text.
        tool_name: Name of the tool being executed (``tool`` chunks).
        tool_args: Raw argument string of the call (``tool`` chunks).
    """

    kind: str
    text: str = ""
    tool_name: str = ""
    tool_args: str = ""


def _clip(text: str, limit: int = 200) -> str:
    """Trim a string so log lines stay readable."""
    cleaned = (text or "").strip().replace("\n", " ")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


class BaseAgent(ABC):
    """Common plumbing for the four debate archetypes.

    Subclasses only declare :attr:`spec` and :attr:`system_prompt`; the
    orchestration layer decides *when* an agent speaks.

    Args:
        router: Shared :class:`LLM.base.LLMRouter` that resolves the active
            backend (Evren / Ollama / OpenAI / Gemini / Anthropic).
    """

    spec: ClassVar[AgentSpec]
    system_prompt: ClassVar[str]
    #: Whether this agent may call the market data tools.
    uses_tools: ClassVar[bool] = True

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    # ------------------------------------------------------------------ #
    # Message building
    # ------------------------------------------------------------------ #
    def build_messages(self, prompt: str) -> list[dict[str, Any]]:
        """Build the chat message list for one turn.

        Args:
            prompt: Phase specific prompt built by the orchestrator.

        Returns:
            ``[system, user]`` messages ready for a chat completion API.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    # ------------------------------------------------------------------ #
    # Streaming turn with tool loop
    # ------------------------------------------------------------------ #
    async def respond(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        market_data: Mapping[str, Any],
        phase: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[AgentChunk]:
        """Stream one assistant turn, executing tool calls when requested.

        The loop: stream a completion -> if the model asked for tools,
        execute them (yfinance runs in a worker thread), append the
        observations to the conversation and stream again, up to
        :data:`MAX_TOOL_ROUNDS` round trips.

        Args:
            messages: Chat messages of this turn (copied before mutation).
            market_data: Current market snapshot (used by the rule engine).
            phase: ``pitch``, ``cross`` or ``synthesis``. Tools are only
                offered when the phase is in
                :attr:`config.Settings.tool_phases` (``--fast`` restricts
                them to the pitch round).
            tools: Optional tool schemas; defaults to
                :data:`data.fetcher.TOOL_SCHEMAS`.

        Yields:
            :class:`AgentChunk` objects in streaming order.
        """
        conversation: list[dict[str, Any]] = [dict(m) for m in messages]
        active_tools: Sequence[Mapping[str, Any]] | None = (
            TOOL_SCHEMAS if tools is None else tools
        )
        allowed = self.uses_tools and phase in self.router.settings.tool_phases
        if not allowed:
            active_tools = None

        empty_retried = False
        for _ in range(MAX_TOOL_ROUNDS + 1):
            content_parts: list[str] = []
            pending: list[ToolCall] = []
            saw_reasoning = False

            async for chunk in self.router.stream(
                conversation,
                tools=active_tools,
                market_data=market_data,
                phase=phase,
            ):
                if chunk.kind == "reasoning" and chunk.text:
                    saw_reasoning = True
                    yield AgentChunk(kind="reasoning", text=chunk.text)
                elif chunk.kind == "content" and chunk.text:
                    content_parts.append(chunk.text)
                    yield AgentChunk(kind="content", text=chunk.text)
                elif chunk.kind == "tool_calls":
                    pending.extend(chunk.tool_calls)

            if not pending:
                if not content_parts:
                    if saw_reasoning and not empty_retried:
                        # Reasoning models can burn the whole token budget on
                        # chain-of-thought; give them one chance to answer
                        # directly instead of failing the turn.
                        empty_retried = True
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "You used your whole output budget on "
                                    "reasoning and produced no answer. Now reply "
                                    "directly with the final answer, max 160 "
                                    "words, no extra thinking."
                                ),
                            }
                        )
                        continue
                    yield AgentChunk(
                        kind="error",
                        text=f"{self.spec.name}: the model returned an empty answer.",
                    )
                break

            spoken = "".join(content_parts).strip()
            conversation.append(
                {
                    "role": "assistant",
                    "content": spoken or None,
                    "tool_calls": [
                        {
                            "id": call.id or f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments or "{}",
                            },
                        }
                        for index, call in enumerate(pending)
                    ],
                }
            )

            for index, call in enumerate(pending):
                yield AgentChunk(
                    kind="tool",
                    tool_name=call.name,
                    tool_args=_clip(call.arguments),
                )
                observation = await execute_tool(call.name, call.arguments)
                yield AgentChunk(
                    kind="tool_result",
                    text=f"{call.name} → {len(observation)} chars received",
                )
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or f"call_{index}",
                        "name": call.name,
                        "content": observation,
                    }
                )
        else:
            yield AgentChunk(
                kind="error",
                text=f"{self.spec.name}: tool-call limit reached "
                f"({MAX_TOOL_ROUNDS} round trips).",
            )

    # ------------------------------------------------------------------ #
    # Classic (non-streaming) helper
    # ------------------------------------------------------------------ #
    async def speak(
        self,
        prompt: str,
        market_data: Mapping[str, Any],
        phase: str,
    ) -> GenerationResult:
        """Generate this agent's contribution without streaming.

        Args:
            prompt: Phase specific prompt built by the orchestrator.
            market_data: Current market snapshot.
            phase: ``pitch``, ``cross`` or ``synthesis``.

        Returns:
            The provider result (strict mode: providers only, never rules
            unless ``provider=fallback``).
        """
        return await self.router.generate(
            prompt,
            self.system_prompt,
            market_data=market_data,
            phase=phase,
        )

    def system_prompt_for(self, extra: str | None = None) -> str:
        """Return the system prompt, optionally with extra instructions."""
        if not extra:
            return self.system_prompt
        return f"{self.system_prompt}\n\n{extra}"
