"""Evren gateway provider - streaming reasoning + tool calling.

Talks to the Evren LLM gateway (``https://evren-llmapi.ssyz.org.tr/v1``)
through the official :mod:`openai` SDK. Two models are used side by side:

* ``EVREN_MODEL`` (default ``deepseek-v4.1-flash``) for pure debate turns,
* ``EVREN_TOOL_MODEL`` (default ``glm-5.3``) whenever the request carries
  tool definitions - i.e. every analyst round that may call yfinance tools.

Reasoning deltas (``delta.reasoning_content`` / ``delta.reasoning``) are
streamed into the TUI as ``reasoning`` chunks before the visible answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from config import Settings, load_settings
from LLM.openai_provider import OpenAIProvider

__all__ = ["EvrenProvider"]


class EvrenProvider(OpenAIProvider):
    """Official-SDK client for the Evren chat completion gateway.

    Args:
        settings: Application configuration (``EVREN_*`` variables).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or load_settings()
        super().__init__(
            settings,
            provider_name="evren",
            base_url=settings.evren_base_url,
            api_key=settings.evren_api_key,
            model=settings.evren_model,
        )
        self.tool_model: str = settings.evren_tool_model or settings.evren_model

    @property
    def model_label(self) -> str:
        """Model label shown in the UI (both models when tools are wired)."""
        if self.tool_model and self.tool_model != self.model:
            return f"{self.model} + {self.tool_model} (tools)"
        return self.model or self._name

    def _pick_model(self, tools: Sequence[Mapping[str, Any]] | None) -> str:
        """Route tool-calling turns to the dedicated tool model."""
        if tools and self.tool_model:
            return self.tool_model
        return self.model
