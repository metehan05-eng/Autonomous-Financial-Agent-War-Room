"""Configuration and environment loading for the Financial War Room.

Every tunable of the application (LLM provider selection, model names,
network timeouts, debate parameters) lives here so that no other module
has to touch ``os.environ`` directly.

Environment variables are read from a project level ``.env`` file (if
present) as well as from the process environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

APP_NAME = "Autonomous Financial Agent War Room"
APP_SUBTITLE = "Financial-WarRoom"

#: Provider identifiers understood by :func:`LLM.build_provider`.
KNOWN_PROVIDERS: tuple[str, ...] = (
    "auto",
    "evren",
    "ollama",
    "openai",
    "gemini",
    "anthropic",
    "fallback",
)


def _load_env() -> None:
    """Load the project ``.env`` file into the process environment."""
    if load_dotenv is not None:
        load_dotenv(ENV_FILE, override=False)


def _env_str(key: str, default: str = "") -> str:
    """Read a string environment variable."""
    value = os.environ.get(key)
    return value.strip() if value is not None else default


def _env_optional(key: str) -> str | None:
    """Read an optional (possibly unset) string environment variable."""
    value = os.environ.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_float(key: str, default: float) -> float:
    """Read a float environment variable, falling back to ``default``."""
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``."""
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the application configuration."""

    #: Provider selection: auto | evren | ollama | openai | gemini | anthropic | fallback
    provider: str = "auto"

    #: Evren gateway (official ``openai`` SDK, streaming reasoning + tools).
    evren_base_url: str = "https://evren-llmapi.ssyz.org.tr/v1"
    evren_api_key: str | None = None
    evren_model: str = "deepseek-v4.1-flash"
    #: Model used when a request carries tool definitions (analyst rounds).
    evren_tool_model: str = "glm-5.3"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-haiku-latest"

    #: Timeout for a single completion request (seconds).
    request_timeout: float = 45.0
    #: Timeout used while probing whether a provider is reachable (seconds).
    probe_timeout: float = 2.5
    #: Number of debate rounds (Round 3 is always the CRO synthesis).
    debate_rounds: int = 3
    max_tokens: int = 3000
    temperature: float = 0.7
    #: Fast mode: agents only call tools during the initial pitch, the debate
    #: runs fewer LLM round trips and a lower temperature speeds generation.
    fast: bool = False

    #: Ticker pre-filled in the input bar.
    default_ticker: str = "AAPL"
    #: History window used for technical analysis.
    history_period: str = "1y"

    @property
    def provider_label(self) -> str:
        """Human readable label of the requested provider."""
        return self.provider or "auto"

    @property
    def has_cloud_keys(self) -> bool:
        """True when at least one cloud API key is configured."""
        return bool(
            self.evren_api_key
            or self.openai_api_key
            or self.gemini_api_key
            or self.anthropic_api_key
        )

    @property
    def tool_phases(self) -> tuple[str, ...]:
        """Debate phases in which agents may call market data tools."""
        return ("pitch",) if self.fast else ("pitch", "cross")

    @property
    def effective_temperature(self) -> float:
        """Sampling temperature actually used in fast mode."""
        return 0.3 if self.fast else self.temperature


def load_settings() -> Settings:
    """Build a :class:`Settings` object from the process environment.

    Returns:
        A fully populated, immutable :class:`Settings` instance.
    """
    _load_env()
    provider = _env_str("LLM_PROVIDER", _env_str("AI_PROVIDER", "auto")).lower()
    if provider not in KNOWN_PROVIDERS:
        provider = "auto"
    return Settings(
        provider=provider,
        evren_base_url=_env_str(
            "EVREN_BASE_URL", "https://evren-llmapi.ssyz.org.tr/v1"
        ).rstrip("/"),
        evren_api_key=_env_optional("EVREN_API_KEY"),
        evren_model=_env_str("EVREN_MODEL", "deepseek-v4.1-flash"),
        evren_tool_model=_env_str("EVREN_TOOL_MODEL", "glm-5.3"),
        ollama_host=_env_str("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_model=_env_str("OLLAMA_MODEL", "qwen2.5-coder"),
        openai_api_key=_env_optional("OPENAI_API_KEY"),
        openai_model=_env_str("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=_env_optional("GEMINI_API_KEY") or _env_optional("GOOGLE_API_KEY"),
        gemini_model=_env_str("GEMINI_MODEL", "gemini-2.0-flash"),
        anthropic_api_key=_env_optional("ANTHROPIC_API_KEY"),
        anthropic_model=_env_str("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        request_timeout=_env_float("LLM_TIMEOUT", 45.0),
        probe_timeout=_env_float("LLM_PROBE_TIMEOUT", 2.5),
        debate_rounds=max(1, _env_int("LLM_ROUNDS", 3)),
        max_tokens=_env_int("LLM_MAX_TOKENS", 3000),
        temperature=_env_float("LLM_TEMPERATURE", 0.7),
        fast=_env_str("LLM_FAST", "0").lower() in ("1", "true", "yes", "on"),
        default_ticker=_env_str("DEFAULT_TICKER", "AAPL").upper(),
        history_period=_env_str("HISTORY_PERIOD", "1y"),
    )


#: Process wide configuration singleton.
SETTINGS: Settings = load_settings()
