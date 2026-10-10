"""Abstract LLM provider interface and the resilient provider router."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
)

import httpx

from config import Settings, load_settings

__all__ = [
    "FALLBACK_NAME",
    "GenerationResult",
    "LLMProvider",
    "LLMRouter",
    "ProviderError",
    "StreamChunk",
    "ToolCall",
]

FALLBACK_NAME = "fallback"


class ProviderError(RuntimeError):
    """Raised when a provider cannot honour a generation request."""


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of a single completion request.

    Attributes:
        text: Generated completion (never empty).
        provider: Name of the backend that produced ``text``.
        model: Model/engine label of that backend.
        fallback_used: ``True`` when the deterministic rule engine produced
            ``text`` instead of a language model.
        error: Description of the failure that forced the fallback, if any.
    """

    text: str
    provider: str
    model: str = ""
    fallback_used: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ToolCall:
    """Function call requested by the model inside one assistant turn.

    Attributes:
        id: Provider side correlation id (may be empty for some gateways).
        name: Function name advertised in the tool schemas.
        arguments: Raw (possibly partial) JSON arguments as a string.
        index: Choice index used to aggregate streamed argument fragments.
    """

    id: str = ""
    name: str = ""
    arguments: str = ""
    index: int = 0


@dataclass
class StreamChunk:
    """Single piece of a streamed completion.

    Attributes:
        kind: ``reasoning`` (chain-of-thought delta), ``content`` (visible
            answer delta) or ``tool_calls`` (finished tool calls of the turn).
        text: Delta text for ``reasoning`` / ``content`` chunks.
        tool_calls: Populated for ``tool_calls`` chunks.
    """

    kind: str
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


def flatten_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Reduce a chat message list to a ``(prompt, system_prompt)`` pair.

    Providers that only understand a classic ``generate(prompt, system)``
    API (the rule engine) use this helper.

    Args:
        messages: Chat messages with ``role`` / ``content`` keys.

    Returns:
        ``(prompt, system_prompt)`` - both may be empty strings.
    """
    system_parts: list[str] = []
    prompt_parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        else:
            prompt_parts.append(f"{role}: {content}" if role != "user" else content)
    return "\n\n".join(prompt_parts), "\n\n".join(system_parts)


class LLMProvider(ABC):
    """Minimal interface every backend must implement.

    Implementations may raise :class:`ProviderError`; the router then tries
    the next candidate and eventually the rule-based fallback, so a broken
    endpoint never breaks the UI.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or load_settings()

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier of the provider (``ollama``, ``openai``...)."""

    @property
    @abstractmethod
    def model_label(self) -> str:
        """Model / engine label surfaced in the UI."""

    @abstractmethod
    def is_configured(self) -> bool:
        """True when the provider has enough configuration to be attempted."""

    async def probe(self) -> bool:
        """Cheap reachability check. Defaults to :meth:`is_configured`."""
        return self.is_configured()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        *,
        market_data: Mapping[str, Any] | None = None,
        phase: str | None = None,
    ) -> str:
        """Return a completion for ``prompt``.

        Args:
            prompt: User/developer content of the request.
            system_prompt: Persona definition of the requesting agent.
            market_data: Structured market snapshot (used by the rule-based
                fallback, ignored by remote providers).
            phase: Debate phase (``pitch``, ``cross``, ``synthesis``).

        Returns:
            The generated text, never ``None``.
        """

    async def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        market_data: Mapping[str, Any] | None = None,
        phase: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion turn.

        The default implementation flattens ``messages`` and delegates to
        :meth:`generate`, yielding the result as a single ``content`` chunk.
        Providers that speak a native chat/stream API override this method.

        Args:
            messages: Chat messages (``system`` / ``user`` / ``assistant`` /
                ``tool`` roles) for the current turn.
            tools: OpenAI style function schemas the model may call.
            market_data: Structured market snapshot (rule engine only).
            phase: Debate phase (``pitch``, ``cross``, ``synthesis``).
            temperature: Optional sampling override.
            max_tokens: Optional output length override.

        Yields:
            :class:`StreamChunk` objects describing the turn.
        """
        prompt, system_prompt = flatten_messages(messages)
        text = await self.generate(
            prompt,
            system_prompt,
            market_data=market_data,
            phase=phase,
        )
        if text:
            yield StreamChunk(kind="content", text=text)

    async def _post_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        """POST ``payload`` to ``url`` and decode a JSON object."""
        try:
            response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: request failed ({exc})") from exc
        if response.status_code >= 400:
            detail = response.text.strip().replace("\n", " ")[:200]
            raise ProviderError(f"{self.name}: HTTP {response.status_code} - {detail}")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: invalid JSON response") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{self.name}: unexpected response payload")
        return data


class LLMRouter:
    """Selects the best available provider and streams every completion.

    Resolution strategy:

    1. Probe candidates in order (Evren first, then local Ollama and the
       configured cloud APIs) with a short timeout.
    2. Cache the first healthy provider for the rest of the session.
    3. If a provider fails mid-session, move on to the next candidate.
    4. In strict mode (the default) a session without any reachable LLM
       raises :class:`ProviderError` - the War Room never fabricates
       debate text from rules. Only ``provider=fallback`` (``--no-llm``)
       engages the deterministic rule engine.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        candidates: Sequence[LLMProvider] | None = None,
    ) -> None:
        self.settings: Settings = settings or load_settings()
        self._candidates: Sequence[LLMProvider] | None = candidates
        self._active: LLMProvider | None = None
        self._unhealthy: set[str] = set()
        self._lock = asyncio.Lock()
        self.last_error: str | None = None
        # Imported lazily: LLM.fallback imports this module as well.
        from LLM.fallback import FallbackProvider

        self._fallback: LLMProvider = FallbackProvider(self.settings)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def require_llm(self) -> bool:
        """True unless the deterministic rule engine was requested."""
        return self.settings.provider != "fallback"

    @property
    def active_name(self) -> str:
        """Name of the provider that will answer (``pending`` before run)."""
        if self._active is not None:
            return self._active.name
        if self.settings.provider == "fallback":
            return FALLBACK_NAME
        return "pending"

    @property
    def active_model(self) -> str:
        """Model/engine label of the active provider."""
        return self._active.model_label if self._active is not None else ""

    @property
    def candidate_names(self) -> list[str]:
        """Names of the providers that will be probed."""
        if self._candidates is None:
            from LLM import build_candidates

            self._candidates = build_candidates(self.settings)
        return [candidate.name for candidate in self._candidates]

    async def resolve(self, strict: bool | None = None) -> LLMProvider:
        """Probe candidates and cache the first reachable provider.

        Args:
            strict: When ``True`` and no candidate is reachable, raise
                :class:`ProviderError` instead of degrading to the rule
                engine. Defaults to :attr:`require_llm`.

        Returns:
            The first healthy provider (or the rule engine in non-strict
            mode).

        Raises:
            ProviderError: In strict mode when every candidate failed.
        """
        if strict is None:
            strict = self.require_llm
        async with self._lock:
            if self._active is not None:
                return self._active
            if self._candidates is None:
                # Late import avoids a circular dependency with LLM/__init__.
                from LLM import build_candidates

                self._candidates = build_candidates(self.settings)

            if self.settings.provider == "fallback":
                self._active = self._fallback
                return self._active

            errors: list[str] = []
            for candidate in self._candidates:
                if candidate.name in self._unhealthy or not candidate.is_configured():
                    continue
                try:
                    healthy = await asyncio.wait_for(
                        candidate.probe(), timeout=self.settings.probe_timeout + 1.0
                    )
                except Exception as exc:
                    healthy = False
                    message = f"{candidate.name}: {exc}"
                    errors.append(message)
                    self.last_error = message
                if healthy:
                    self._active = candidate
                    return candidate
                self._unhealthy.add(candidate.name)
                if not errors:
                    errors.append(f"{candidate.name}: probe failed")
                    self.last_error = errors[-1]

            if strict:
                detail = "; ".join(errors) or self.last_error or "no provider configured"
                raise ProviderError(
                    f"no reachable LLM provider ({detail}). Set EVREN_API_KEY "
                    "or another provider key, or run with --no-llm for the "
                    "deterministic rule engine."
                )

            self._active = self._fallback
            return self._active

    def reset(self) -> None:
        """Forget cached health state (used by tests / reconnects)."""
        self._active = None
        self._unhealthy.clear()
        self.last_error = None

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    async def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        market_data: Mapping[str, Any] | None = None,
        phase: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from the best healthy provider.

        Falls through to the next candidate when a provider fails before
        emitting any output. In strict mode the stream raises
        :class:`ProviderError` when nothing could be produced; the rule
        engine is only consulted when ``provider=fallback``.

        Args:
            messages: Chat messages of the current turn.
            tools: OpenAI style function schemas (optional).
            market_data: Market snapshot (rule engine only).
            phase: Debate phase.
            temperature: Optional sampling override.
            max_tokens: Optional output length override.

        Yields:
            :class:`StreamChunk` objects of the active provider's turn.

        Raises:
            ProviderError: When no provider produced any output.
        """
        primary = await self.resolve()
        chain: list[LLMProvider] = [primary]
        if self._candidates:
            chain.extend(
                candidate
                for candidate in self._candidates
                if candidate is not primary and candidate.name not in self._unhealthy
            )

        errors: list[str] = []
        emitted = False
        for candidate in chain:
            if candidate is self._fallback:
                continue
            try:
                async for chunk in candidate.stream(
                    messages,
                    tools=tools,
                    market_data=market_data,
                    phase=phase,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                message = f"{candidate.name}: {exc}"
                errors.append(message)
                self._unhealthy.add(candidate.name)
                self.last_error = message
                if emitted:
                    raise ProviderError(message) from exc
                continue

        if not self.require_llm:
            async for chunk in self._fallback.stream(
                messages,
                market_data=market_data,
                phase=phase,
            ):
                yield chunk
            return

        detail = "; ".join(errors) or self.last_error or "no provider configured"
        raise ProviderError(f"LLM stream failed ({detail})")

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        *,
        market_data: Mapping[str, Any] | None = None,
        phase: str | None = None,
    ) -> GenerationResult:
        """Generate a completion from the best healthy provider.

        In strict mode (default) every configured provider is tried before
        :class:`ProviderError` is raised; the rule engine is only used when
        ``provider=fallback``.

        Args:
            prompt: Request text.
            system_prompt: Agent persona.
            market_data: Structured market snapshot.
            phase: Current debate phase.

        Returns:
            A :class:`GenerationResult` that always contains usable text.

        Raises:
            ProviderError: When strict mode is active and every provider
                failed to produce text.
        """
        primary = await self.resolve()
        chain: list[LLMProvider] = [primary]
        if self._candidates:
            chain.extend(
                candidate
                for candidate in self._candidates
                if candidate is not primary and candidate.name not in self._unhealthy
            )

        errors: list[str] = []
        for candidate in chain:
            if candidate is self._fallback:
                break
            try:
                text = await candidate.generate(
                    prompt,
                    system_prompt,
                    market_data=market_data,
                    phase=phase,
                )
            except Exception as exc:
                message = f"{candidate.name}: {exc}"
                errors.append(message)
                self._unhealthy.add(candidate.name)
                self.last_error = message
                continue
            cleaned = (text or "").strip()
            if cleaned:
                return GenerationResult(
                    text=cleaned,
                    provider=candidate.name,
                    model=candidate.model_label,
                )
            errors.append(f"{candidate.name}: empty response")

        if self.require_llm:
            detail = "; ".join(errors) or self.last_error or "no provider configured"
            raise ProviderError(f"LLM generation failed ({detail})")

        text = await self._fallback.generate(
            prompt, system_prompt, market_data=market_data, phase=phase
        )
        return GenerationResult(
            text=(text or "").strip(),
            provider=FALLBACK_NAME,
            model=self._fallback.model_label,
            fallback_used=True,
            error="; ".join(errors) or None,
        )
