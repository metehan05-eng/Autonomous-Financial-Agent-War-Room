"""OpenAI-compatible chat completion providers built on the official SDK.

A single implementation covers the Evren gateway, OpenAI itself and the two
other clouds that expose OpenAI-compatible ``/chat/completions`` endpoints:

* Google Gemini - ``https://generativelanguage.googleapis.com/v1beta/openai/``
* Anthropic Claude - ``https://api.anthropic.com/v1/``

Streaming is implemented with ``openai.AsyncOpenAI``: every chunk is
translated into :class:`LLM.base.StreamChunk` objects, so reasoning deltas
(``delta.reasoning_content`` / ``delta.reasoning``) and finished tool calls
reach the debate log as they arrive.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from config import Settings, load_settings
from LLM.base import LLMProvider, ProviderError, StreamChunk, ToolCall

__all__ = ["OpenAIProvider"]

OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"

#: Non-standard reasoning field names seen across OpenAI-compatible gateways.
_REASONING_FIELDS: tuple[str, ...] = (
    "reasoning_content",
    "reasoning",
    "thinking",
)


def _delta_field(delta: Any, name: str) -> str:
    """Read a possibly non-standard field from a streamed choice delta.

    The official SDK allows unknown keys (``extra="allow"``), but they are
    only reachable through ``__pydantic_extra__``.

    Args:
        delta: ``ChoiceDelta`` of the current chunk.
        name: Field name to read (``reasoning_content``, ``reasoning``...).

    Returns:
        The field value as a string, or ``""`` when it is absent/empty.
    """
    value = getattr(delta, name, None)
    if value is None:
        extra = getattr(delta, "__pydantic_extra__", None) or {}
        value = extra.get(name)
    if not value:
        return ""
    return str(value)


class OpenAIProvider(LLMProvider):
    """Generic client for OpenAI-style ``/chat/completions`` endpoints.

    Args:
        settings: Application configuration.
        provider_name: Identifier surfaced in the UI (``openai``...).
        base_url: Root of the API (``.../v1``).
        api_key: Bearer token.
        model: Model identifier.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider_name: str,
        base_url: str,
        api_key: str | None,
        model: str,
    ) -> None:
        super().__init__(settings)
        self._name = provider_name
        self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client: Any = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def openai(cls, settings: Settings | None = None) -> OpenAIProvider:
        """Provider configured from ``OPENAI_API_KEY`` / ``OPENAI_MODEL``."""
        settings = settings or load_settings()
        return cls(
            settings,
            provider_name="openai",
            base_url=OPENAI_BASE_URL,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    @classmethod
    def gemini(cls, settings: Settings | None = None) -> OpenAIProvider:
        """Provider configured from ``GEMINI_API_KEY`` / ``GEMINI_MODEL``."""
        settings = settings or load_settings()
        return cls(
            settings,
            provider_name="gemini",
            base_url=GEMINI_BASE_URL,
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )

    @classmethod
    def anthropic(cls, settings: Settings | None = None) -> OpenAIProvider:
        """Provider configured from ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_MODEL``."""
        settings = settings or load_settings()
        return cls(
            settings,
            provider_name="anthropic",
            base_url=ANTHROPIC_BASE_URL,
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )

    # ------------------------------------------------------------------ #
    # LLMProvider interface
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Provider identifier."""
        return self._name

    @property
    def model_label(self) -> str:
        """Model label shown in the UI."""
        return self.model or self._name

    def is_configured(self) -> bool:
        """True when an API key and a model are configured."""
        return bool(self.api_key and self.model)

    def _headers(self) -> dict[str, str]:
        """Authorization headers used by the HTTP probe."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def probe(self) -> bool:
        """Check the endpoint with a ``GET /models`` call."""
        if not self.is_configured():
            return False
        try:
            async with httpx.AsyncClient(timeout=self.settings.probe_timeout) as client:
                response = await client.get(
                    f"{self.base_url}/models", headers=self._headers()
                )
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    # ------------------------------------------------------------------ #
    # SDK plumbing
    # ------------------------------------------------------------------ #
    def _async_client(self) -> Any:
        """Create (or reuse for the current event loop) an ``AsyncOpenAI``.

        Returns:
            The SDK client bound to the running event loop.

        Raises:
            ProviderError: When the ``openai`` package is missing.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - openai is a dependency
            raise ProviderError(
                f"{self._name}: the 'openai' package is not installed "
                "(pip install openai)"
            ) from exc

        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = AsyncOpenAI(
                api_key=self.api_key or "",
                base_url=self.base_url,
                timeout=self.settings.request_timeout,
                max_retries=1,
            )
            self._client_loop = loop
        return self._client

    @staticmethod
    def _describe_api_error(exc: Exception) -> str:
        """Render an SDK exception as a short, log friendly message."""
        status = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                detail = str(error["message"])
                return f"HTTP {status}: {detail}" if status else detail
            if body.get("message"):
                return str(body["message"])
        message = str(exc).strip() or type(exc).__name__
        if status:
            return f"HTTP {status}: {message}"
        return message

    def _pick_model(self, tools: Sequence[Mapping[str, Any]] | None) -> str:
        """Model used for a request (subclasses may switch on ``tools``)."""
        return self.model

    def _completion_kwargs(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Assemble the keyword arguments of a chat completion request."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": (
                self.settings.effective_temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream"] = True
        return kwargs

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
        """Stream reasoning, content and tool calls of one completion turn.

        Args:
            messages: Chat messages of the current turn.
            tools: OpenAI style function schemas (optional).
            market_data: Ignored (remote providers receive data in prompts).
            phase: Ignored.
            temperature: Optional sampling override.
            max_tokens: Optional output length override.

        Yields:
            ``reasoning`` / ``content`` deltas while they stream, followed by
            a single ``tool_calls`` chunk when the model requested tools.

        Raises:
            ProviderError: On transport, HTTP or payload problems.
        """
        if not self.is_configured():
            raise ProviderError(f"{self._name}: missing API key or model")

        model = self._pick_model(tools)
        kwargs = self._completion_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        client = self._async_client()
        try:
            stream_obj = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(
                f"{self._name}: {self._describe_api_error(exc)}"
            ) from exc

        pending: dict[int, dict[str, str]] = {}
        try:
            async for event in stream_obj:
                for choice in getattr(event, "choices", None) or []:
                    delta = choice.delta
                    if delta is None:
                        continue
                    reasoning = ""
                    for field_name in _REASONING_FIELDS:
                        reasoning = _delta_field(delta, field_name)
                        if reasoning:
                            break
                    if reasoning:
                        yield StreamChunk(kind="reasoning", text=reasoning)
                    content = delta.content
                    if content:
                        yield StreamChunk(kind="content", text=str(content))
                    for tool_delta in delta.tool_calls or []:
                        index = int(tool_delta.index or 0)
                        bucket = pending.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tool_delta.id:
                            bucket["id"] = tool_delta.id
                        function = tool_delta.function
                        if function is not None:
                            if function.name:
                                bucket["name"] = str(function.name)
                            if function.arguments:
                                bucket["arguments"] += str(function.arguments)
        except Exception as exc:
            raise ProviderError(
                f"{self._name}: stream interrupted - {self._describe_api_error(exc)}"
            ) from exc
        finally:
            close = getattr(stream_obj, "close", None)
            if close is not None:
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass

        if pending:
            calls = tuple(
                ToolCall(
                    id=pending[index]["id"] or f"call_{index}",
                    name=pending[index]["name"],
                    arguments=pending[index]["arguments"],
                    index=index,
                )
                for index in sorted(pending)
            )
            yield StreamChunk(kind="tool_calls", tool_calls=calls)

    # ------------------------------------------------------------------ #
    # Non-streaming generation
    # ------------------------------------------------------------------ #
    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        *,
        market_data: Mapping[str, Any] | None = None,
        phase: str | None = None,
    ) -> str:
        """Run a single (non-streaming) chat completion request.

        Args:
            prompt: Request text.
            system_prompt: Agent persona.
            market_data: Ignored (already embedded in the prompt).
            phase: Ignored.

        Returns:
            The assistant message content.

        Raises:
            ProviderError: On transport, HTTP or payload problems.
        """
        if not self.is_configured():
            raise ProviderError(f"{self._name}: missing API key or model")

        kwargs = self._completion_kwargs(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            temperature=None,
            max_tokens=None,
            stream=False,
        )
        kwargs.pop("stream", None)
        client = self._async_client()
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(
                f"{self._name}: {self._describe_api_error(exc)}"
            ) from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ProviderError(f"{self._name}: no choices in response")
        message = choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, list):  # some SDKs return content parts
            content = "".join(
                str(part.get("text", ""))
                if isinstance(part, dict)
                else str(getattr(part, "text", ""))
                for part in content
            )
        if not content:
            content = _delta_field(message, "reasoning_content") or _delta_field(
                message, "reasoning"
            )
        if not content:
            raise ProviderError(f"{self._name}: empty completion returned")
        return str(content)
