"""Market data retrieval and technical indicator calculation.

The :class:`MarketDataFetcher` pulls price history and fundamental metadata
for any ticker supported by Yahoo Finance (equities, ETFs, crypto, futures,
FX) and derives the technical indicators the debate agents reason about:
RSI(14), SMA(50), SMA(200), MACD, realised volatility and volume profile.

Every public entry point is exception safe: failures are reported inside
the returned dictionary (``ok``/``error`` keys) so the TUI never crashes
because of a typo'd ticker or a network hiccup.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import ta
import yfinance as yf

__all__ = [
    "TOOL_SCHEMAS",
    "FetchError",
    "MarketDataFetcher",
    "execute_tool",
    "fetch_market_data",
    "get_latest_news",
    "get_rsi_and_indicators",
    "get_stock_quote",
]


class FetchError(RuntimeError):
    """Raised when market data cannot be retrieved for a ticker."""


#: Currency symbols used when formatting prices.
_CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "TRY": "₺",
    "CHF": "CHF ",
    "CAD": "C$",
    "AUD": "A$",
    "CNY": "¥",
    "HKD": "HK$",
    "INR": "₹",
    "KRW": "₩",
    "BRL": "R$",
    "ZAR": "R ",
}


def _safe_float(value: Any) -> float | None:
    """Convert ``value`` to a finite float, or ``None`` when impossible."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def format_price(value: float | None, currency: str = "USD") -> str:
    """Format a monetary value with its currency symbol.

    Args:
        value: Amount to format, may be ``None``.
        currency: ISO 4217 currency code.

    Returns:
        Human readable price string (e.g. ``$340.12``).
    """
    if value is None:
        return "n/a"
    symbol = _CURRENCY_SYMBOLS.get(currency.upper(), f"{currency.upper()} ")
    decimals = 0 if abs(value) >= 10_000 else 2
    return f"{symbol}{value:,.{decimals}f}"


def format_large(value: float | None) -> str:
    """Format a large number with B/M/K suffixes (e.g. ``$2.94T``)."""
    if value is None:
        return "n/a"
    absolute = abs(value)
    for divisor, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if absolute >= divisor:
            return f"{value / divisor:,.2f}{suffix}"
    return f"{value:,.0f}"


def sparkline(values: list[float], width: int = 32) -> str:
    """Render a unicode block sparkline for a price series.

    Args:
        values: Sequence of numeric points.
        width: Maximum number of characters to emit.

    Returns:
        A single line string built from ``▁▂▃▄▅▆▇█``.
    """
    if not values:
        return ""
    points = values[-width:]
    blocks = "▁▂▃▄▅▆▇█"
    low, high = min(points), max(points)
    span = high - low
    if span == 0:
        return blocks[3] * len(points)
    return "".join(blocks[int((point - low) / span * (len(blocks) - 1))] for point in points)


class MarketDataFetcher:
    """Fetches market data and computes technical indicators for a ticker.

    Args:
        ticker: Symbol understood by Yahoo Finance (``AAPL``, ``BTC-USD``,
            ``GC=F``, ``EURUSD=X`` ...).
        period: History window passed to ``yfinance`` (default ``1y`` so a
            200 day moving average can be computed).
    """

    def __init__(self, ticker: str, period: str = "1y") -> None:
        self.ticker: str = (ticker or "").strip().upper()
        self.period: str = period or "1y"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch(self) -> dict[str, Any]:
        """Retrieve market data and indicators.

        Returns:
            Dictionary with an ``ok`` flag. On success it contains price,
            fundamentals, indicators and a sparkline series; on failure it
            contains ``ok=False`` plus a human readable ``error``.
        """
        if not self.ticker:
            return self._failure("No ticker supplied.")

        try:
            history = self._history()
        except Exception as exc:
            return self._failure(f"Unable to download history for {self.ticker}: {exc}")

        if history is None or history.empty or "Close" not in history:
            return self._failure(
                f"No data returned for '{self.ticker}'. Check the symbol "
                "(stocks: AAPL, crypto: BTC-USD, commodities: GC=F)."
            )

        try:
            info = self._info()
        except Exception:
            info = {}

        try:
            return self._build_result(history, info)
        except Exception as exc:
            return self._failure(f"Failed to process data for {self.ticker}: {exc}")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _failure(self, message: str) -> dict[str, Any]:
        """Build a well-formed error payload."""
        return {
            "ok": False,
            "error": message,
            "ticker": self.ticker,
            "name": self.ticker,
            "price": None,
            "change_pct": None,
            "rsi_14": None,
            "sma_50": None,
            "sma_200": None,
            "forward_pe": None,
            "sparkline": [],
        }

    def _history(self) -> pd.DataFrame | None:
        """Download adjusted price history for the ticker."""
        ticker_obj = yf.Ticker(self.ticker)
        history = ticker_obj.history(period=self.period, auto_adjust=True)
        if history is None or history.empty:
            # Second attempt without adjustment - some symbols refuse it.
            history = ticker_obj.history(period=self.period)
        return history

    def _info(self) -> dict[str, Any]:
        """Best effort fundamentals lookup (short summary, P/E, caps ...)."""
        try:
            info = yf.Ticker(self.ticker).info
        except Exception:
            return {}
        return info if isinstance(info, dict) else {}

    def _build_result(self, history: pd.DataFrame, info: dict[str, Any]) -> dict[str, Any]:
        """Assemble the final metrics dictionary from raw Yahoo data."""
        close = history["Close"].dropna().astype(float)
        volume = history["Volume"].astype(float) if "Volume" in history else pd.Series(dtype=float)
        highs = history["High"].astype(float) if "High" in history else close
        lows = history["Low"].astype(float) if "Low" in history else close

        price = _safe_float(close.iloc[-1])
        if price is None:
            raise FetchError("Latest close price is not available.")

        previous_close = _safe_float(close.iloc[-2]) if len(close) > 1 else None
        if previous_close is None:
            previous_close = _safe_float(info.get("previousClose"))
        change = None if previous_close is None else price - previous_close
        change_pct = (
            None
            if (change is None or not previous_close)
            else (change / previous_close) * 100.0
        )

        currency = str(info.get("currency") or "USD").upper()
        year_high = _safe_float(info.get("fiftyTwoWeekHigh")) or _safe_float(highs.max())
        year_low = _safe_float(info.get("fiftyTwoWeekLow")) or _safe_float(lows.min())

        rsi = self._rsi(close)
        sma_50 = _safe_float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma_200 = _safe_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        macd_line, macd_signal = self._macd(close)
        volatility = self._volatility(close)
        avg_volume = _safe_float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else None
        last_volume = _safe_float(volume.iloc[-1]) if len(volume) else None

        pct_from_high = (
            None if not year_high else ((price - year_high) / year_high) * 100.0
        )
        range_position = None
        if year_high is not None and year_low is not None and year_high != year_low:
            range_position = (price - year_low) / (year_high - year_low) * 100.0

        trend = self._classify_trend(price, sma_50, sma_200)
        macd_hist = (
            None
            if (macd_line is None or macd_signal is None)
            else macd_line - macd_signal
        )

        return {
            "ok": True,
            "error": None,
            "ticker": self.ticker,
            "name": str(info.get("longName") or info.get("shortName") or self.ticker),
            "sector": str(info.get("sector") or "n/a"),
            "industry": str(info.get("industry") or "n/a"),
            "summary": str(info.get("longBusinessSummary") or "").strip(),
            "currency": currency,
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_pct": change_pct,
            "market_cap": _safe_float(info.get("marketCap")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "trailing_pe": _safe_float(info.get("trailingPE")),
            "dividend_yield": _safe_float(info.get("dividendYield")),
            "beta": _safe_float(info.get("beta")),
            "fifty_two_week_high": year_high,
            "fifty_two_week_low": year_low,
            "pct_from_52w_high": pct_from_high,
            "range_position": range_position,
            "rsi_14": rsi,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "price_vs_sma_50": self._pct_vs(price, sma_50),
            "price_vs_sma_200": self._pct_vs(price, sma_200),
            "sma_spread": self._pct_vs(sma_50, sma_200),
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "volume": last_volume,
            "avg_volume_20": avg_volume,
            "volume_ratio": (
                None
                if not (last_volume and avg_volume)
                else last_volume / avg_volume
            ),
            "volatility_ann": volatility,
            "trend": trend,
            "trend_label": self._trend_label(trend),
            "rsi_zone": self._rsi_zone(rsi),
            "sparkline": [_safe_float(v) for v in close.iloc[-40:].tolist() if v is not None],
            "history_points": len(close),
            "fetched_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        }

    # ------------------------------------------------------------------ #
    # Indicator helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> float | None:
        """14 period Relative Strength Index."""
        if len(close) <= window:
            return None
        try:
            return _safe_float(
                ta.momentum.RSIIndicator(close=close, window=window).rsi().iloc[-1]
            )
        except Exception:
            return None

    @staticmethod
    def _macd(close: pd.Series) -> tuple[float | None, float | None]:
        """MACD line and signal line (12/26/9)."""
        if len(close) < 35:
            return None, None
        try:
            indicator = ta.trend.MACD(close=close)
            return _safe_float(indicator.macd().iloc[-1]), _safe_float(
                indicator.macd_signal().iloc[-1]
            )
        except Exception:
            return None, None

    @staticmethod
    def _volatility(close: pd.Series, window: int = 21) -> float | None:
        """Annualised realised volatility expressed in percent."""
        if len(close) <= window:
            return None
        try:
            returns = close.pct_change().dropna()
            daily = returns.rolling(window).std().iloc[-1]
            number = _safe_float(daily)
            return None if number is None else number * math.sqrt(252) * 100.0
        except Exception:
            return None

    @staticmethod
    def _pct_vs(value: float | None, reference: float | None) -> float | None:
        """Percentage distance of ``value`` from ``reference``."""
        if value is None or reference in (None, 0):
            return None
        return (value - reference) / reference * 100.0

    @staticmethod
    def _classify_trend(price: float, sma_50: float | None, sma_200: float | None) -> str:
        """Classify the medium/long term trend from moving averages."""
        if sma_50 is None or sma_200 is None:
            return "neutral"
        if price > sma_50 > sma_200:
            return "strong_uptrend"
        if price > sma_200:
            return "uptrend"
        if price < sma_50 < sma_200:
            return "strong_downtrend"
        return "downtrend"

    @staticmethod
    def _trend_label(trend: str) -> str:
        """Human readable trend label."""
        return {
            "strong_uptrend": "Strong Uptrend",
            "uptrend": "Uptrend",
            "strong_downtrend": "Strong Downtrend",
            "downtrend": "Downtrend",
            "neutral": "Neutral / Unclear",
        }.get(trend, "Neutral")

    @staticmethod
    def _rsi_zone(rsi: float | None) -> str:
        """Classify the RSI reading."""
        if rsi is None:
            return "n/a"
        if rsi >= 70:
            return "Overbought"
        if rsi >= 55:
            return "Bullish momentum"
        if rsi > 45:
            return "Neutral"
        if rsi > 30:
            return "Bearish momentum"
        return "Oversold"


def fetch_market_data(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Convenience wrapper around :class:`MarketDataFetcher.fetch`.

    Args:
        ticker: Yahoo Finance symbol.
        period: History window.

    Returns:
        The metrics dictionary produced by :meth:`MarketDataFetcher.fetch`.
    """
    return MarketDataFetcher(ticker, period=period).fetch()


# --------------------------------------------------------------------------- #
# Tool functions exposed to the LLM agents (OpenAI function-calling schema)
# --------------------------------------------------------------------------- #
def get_stock_quote(symbol: str) -> dict[str, Any]:
    """Fetch a light quote snapshot for ``symbol``.

    Args:
        symbol: Yahoo Finance symbol (``AAPL``, ``BTC-USD``, ``GC=F``...).

    Returns:
        Quote dictionary (price, change, 52-week range, valuation) with an
        ``ok`` flag; failures carry ``ok=False`` and an ``error`` message.
    """
    data = MarketDataFetcher(symbol).fetch()
    if not data.get("ok"):
        return {"ok": False, "symbol": symbol.upper(), "error": data.get("error")}
    keys = (
        "ticker",
        "name",
        "sector",
        "industry",
        "currency",
        "price",
        "previous_close",
        "change",
        "change_pct",
        "market_cap",
        "forward_pe",
        "trailing_pe",
        "dividend_yield",
        "beta",
        "fifty_two_week_high",
        "fifty_two_week_low",
        "pct_from_52w_high",
        "range_position",
        "volume",
        "avg_volume_20",
        "summary",
        "fetched_at",
    )
    result: dict[str, Any] = {"ok": True, "symbol": data.get("ticker")}
    result.update({key: data.get(key) for key in keys})
    return result


def get_rsi_and_indicators(symbol: str) -> dict[str, Any]:
    """Fetch the technical indicator block for ``symbol``.

    Args:
        symbol: Yahoo Finance symbol.

    Returns:
        Indicator dictionary (RSI, SMAs, MACD, volatility, trend) with an
        ``ok`` flag; failures carry ``ok=False`` and an ``error`` message.
    """
    data = MarketDataFetcher(symbol).fetch()
    if not data.get("ok"):
        return {"ok": False, "symbol": symbol.upper(), "error": data.get("error")}
    keys = (
        "ticker",
        "price",
        "change_pct",
        "rsi_14",
        "rsi_zone",
        "sma_50",
        "sma_200",
        "price_vs_sma_50",
        "price_vs_sma_200",
        "sma_spread",
        "macd",
        "macd_signal",
        "macd_histogram",
        "volatility_ann",
        "volume_ratio",
        "range_position",
        "pct_from_52w_high",
        "trend",
        "trend_label",
        "history_points",
        "fetched_at",
    )
    result: dict[str, Any] = {"ok": True, "symbol": data.get("ticker")}
    result.update({key: data.get(key) for key in keys})
    return result


def get_latest_news(symbol: str, limit: int = 5) -> dict[str, Any]:
    """Fetch the most recent Yahoo Finance news items for ``symbol``.

    Args:
        symbol: Yahoo Finance symbol.
        limit: Maximum number of headlines to return (1-10).

    Returns:
        Dictionary with ``ok``, ``symbol``, ``count`` and ``items``; each
        item carries ``title``, ``publisher``, ``published`` and ``url``.
    """
    symbol = (symbol or "").strip().upper()
    limit = max(1, min(10, int(limit or 5)))
    if not symbol:
        return {"ok": False, "symbol": "", "error": "No symbol supplied.", "items": []}
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception as exc:
        return {
            "ok": False,
            "symbol": symbol,
            "error": f"News request failed for {symbol}: {exc}",
            "items": [],
        }

    items: list[dict[str, Any]] = []
    for entry in raw:
        if len(items) >= limit:
            break
        if not isinstance(entry, dict):
            continue
        item = _news_item(entry)
        if item is not None:
            items.append(item)
    return {"ok": True, "symbol": symbol, "count": len(items), "items": items}


def _news_item(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalise one yfinance news entry (old and new payload shapes)."""
    content = entry.get("content")
    if isinstance(content, dict):
        canonical = content.get("canonicalUrl")
        click = content.get("clickThroughUrl")
        provider = content.get("provider")
        url = ""
        if isinstance(canonical, dict):
            url = str(canonical.get("url") or "")
        if not url and isinstance(click, dict):
            url = str(click.get("url") or "")
        title = str(content.get("title") or "").strip()
        publisher = ""
        if isinstance(provider, dict):
            publisher = str(provider.get("displayName") or "").strip()
        published = str(
            content.get("pubDate") or content.get("displayTime") or ""
        ).strip()
        if not title:
            return None
        return {
            "title": title,
            "publisher": publisher,
            "published": published,
            "url": url,
        }

    title = str(entry.get("title") or "").strip()
    if not title:
        return None
    published = ""
    stamp = entry.get("providerPublishTime")
    if stamp is not None:
        try:
            published = datetime.fromtimestamp(
                float(stamp), tz=timezone.utc
            ).isoformat()
        except (TypeError, ValueError, OSError):
            published = str(stamp)
    return {
        "title": title,
        "publisher": str(entry.get("publisher") or "").strip(),
        "published": published,
        "url": str(entry.get("link") or "").strip(),
    }


#: JSON schemas advertised to the model through OpenAI function calling.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": (
                "Fetch a live Yahoo Finance quote snapshot for a symbol: last "
                "price, daily change, 52-week range, market cap, P/E, beta, "
                "volume and company profile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": (
                            "Yahoo Finance symbol, e.g. 'AAPL', 'BTC-USD', "
                            "'GC=F', 'EURUSD=X'."
                        ),
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rsi_and_indicators",
            "description": (
                "Fetch computed technical indicators for a symbol: RSI(14), "
                "SMA50/SMA200 and their spread, MACD(12,26,9), annualised "
                "volatility, volume ratio, range position and trend regime."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Yahoo Finance symbol, e.g. 'AAPL'.",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_news",
            "description": (
                "Fetch the most recent news headlines published for a symbol "
                "with publisher, publication date and link."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Yahoo Finance symbol, e.g. 'AAPL'.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of headlines (1-10, default 5).",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
]

#: Dispatch table mapping a tool name onto its callable.
_TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_stock_quote": get_stock_quote,
    "get_rsi_and_indicators": get_rsi_and_indicators,
    "get_latest_news": get_latest_news,
}


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Decode the ``arguments`` payload of a tool call.

    Args:
        arguments: JSON string, mapping or empty value emitted by the model.

    Returns:
        A decoded mapping; malformed payloads become ``{}``.
    """
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


async def execute_tool(name: str, arguments: Any) -> str:
    """Execute one tool call on behalf of an agent.

    The blocking yfinance requests run in a worker thread so the debate
    (and the UI) keep streaming while market data is retrieved.

    Args:
        name: Tool name advertised in :data:`TOOL_SCHEMAS`.
        arguments: Raw arguments produced by the model.

    Returns:
        A JSON string with ``ok=True`` and the payload, or ``ok=False``
        plus an ``error`` description - never raises.
    """
    payload = parse_tool_arguments(arguments)
    function = _TOOL_FUNCTIONS.get(name)
    if function is None:
        return json.dumps(
            {"ok": False, "error": f"Unknown tool '{name}'."},
            ensure_ascii=False,
        )

    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol:
        return json.dumps(
            {"ok": False, "error": f"{name} requires a 'symbol' argument."},
            ensure_ascii=False,
        )

    try:
        if name == "get_latest_news":
            result = await asyncio.to_thread(
                function, symbol, int(payload.get("limit") or 5)
            )
        else:
            result = await asyncio.to_thread(function, symbol)
    except Exception as exc:
        return json.dumps(
            {"ok": False, "symbol": symbol, "error": str(exc)},
            ensure_ascii=False,
        )
    return json.dumps(result, ensure_ascii=False, default=str)
