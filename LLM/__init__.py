"""Hybrid LLM provider package.

Providers implement an async :meth:`LLMProvider.generate` method plus a
native :meth:`LLMProvider.stream` generator (reasoning / content / tool
call deltas). They are selected by :class:`LLM.base.LLMRouter`, which
probes the Evren gateway first, then local Ollama and the configured
cloud APIs. In the default strict mode every agent answer comes from a
real LLM; the deterministic rule engine is only used with ``--no-llm``.
"""

from __future__ import annotations

from config import Settings, load_settings
from LLM.base import (
    GenerationResult,
    LLMProvider,
    LLMRouter,
    ProviderError,
    StreamChunk,
    ToolCall,
)
from LLM.evren_provider import EvrenProvider
from LLM.fallback import FallbackProvider, derive_verdict
from LLM.ollama_provider import OllamaProvider
from LLM.openai_provider import OpenAIProvider

__all__ = [
    "EvrenProvider",
    "FallbackProvider",
    "GenerationResult",
    "LLMProvider",
    "LLMRouter",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "StreamChunk",
    "ToolCall",
    "build_provider",
    "build_router",
    "derive_verdict",
    "load_settings",
]


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the provider explicitly requested by ``settings.provider``.

    Args:
        settings: Application configuration.

    Returns:
        The configured :class:`LLMProvider` (never ``None`` - unknown or
        unconfigured selections resolve to :class:`FallbackProvider`).
    """
    choice = settings.provider
    if choice == "evren":
        return EvrenProvider(settings)
    if choice == "ollama":
        return OllamaProvider(settings)
    if choice == "openai":
        return OpenAIProvider.openai(settings)
    if choice == "gemini":
        return OpenAIProvider.gemini(settings)
    if choice == "anthropic":
        return OpenAIProvider.anthropic(settings)
    return FallbackProvider(settings)


def build_candidates(settings: Settings) -> list[LLMProvider]:
    """Ordered list of providers to try (Evren first, then local/clouds).

    Args:
        settings: Application configuration.

    Returns:
        Candidate providers excluding the always-present fallback engine.
    """
    if settings.provider == "fallback":
        return []
    if settings.provider != "auto":
        provider = build_provider(settings)
        return [] if isinstance(provider, FallbackProvider) else [provider]

    candidates: list[LLMProvider] = []
    if settings.evren_api_key:
        candidates.append(EvrenProvider(settings))
    candidates.append(OllamaProvider(settings))
    if settings.openai_api_key:
        candidates.append(OpenAIProvider.openai(settings))
    if settings.gemini_api_key:
        candidates.append(OpenAIProvider.gemini(settings))
    if settings.anthropic_api_key:
        candidates.append(OpenAIProvider.anthropic(settings))
    return candidates


def build_router(settings: Settings | None = None) -> LLMRouter:
    """Create the router used by every debate agent.

    Args:
        settings: Optional configuration, defaults to the process settings.

    Returns:
        A ready to use :class:`LLMRouter`.
    """
    return LLMRouter(settings or load_settings())
