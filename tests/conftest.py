"""Shared pytest fixtures: keep the test suite fully offline and fast."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import base_agent, orchestrator  # noqa: E402


@pytest.fixture
def sample_market_data() -> dict[str, Any]:
    """A deterministic market snapshot shared by the orchestrator tests."""
    return {
        "ok": True,
        "error": None,
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "summary": "Apple designs and sells consumer electronics worldwide.",
        "currency": "USD",
        "price": 190.0,
        "previous_close": 188.0,
        "change": 2.0,
        "change_pct": 1.06,
        "market_cap": 2_950_000_000_000.0,
        "forward_pe": 28.5,
        "trailing_pe": 30.1,
        "dividend_yield": 0.005,
        "beta": 1.25,
        "fifty_two_week_high": 200.0,
        "fifty_two_week_low": 150.0,
        "pct_from_52w_high": -5.0,
        "range_position": 80.0,
        "rsi_14": 58.0,
        "rsi_zone": "Bullish momentum",
        "sma_50": 185.0,
        "sma_200": 175.0,
        "price_vs_sma_50": 2.70,
        "price_vs_sma_200": 8.57,
        "sma_spread": 5.71,
        "macd": 1.2,
        "macd_signal": 0.9,
        "macd_histogram": 0.3,
        "volume": 60_000_000.0,
        "avg_volume_20": 55_000_000.0,
        "volume_ratio": 1.09,
        "volatility_ann": 24.0,
        "trend": "strong_uptrend",
        "trend_label": "Strong Uptrend",
        "sparkline": [180.0, 184.0, 188.0, 190.0],
        "history_points": 250,
        "fetched_at": "2026-01-01 00:00 UTC",
    }


@pytest.fixture(autouse=True)
def offline_market_data(monkeypatch, sample_market_data):
    """Replace network calls with deterministic, offline stand-ins."""

    def fake_fetch(ticker: str, period: str = "1y") -> dict[str, Any]:
        data = dict(sample_market_data)
        data["ticker"] = (ticker or "AAPL").upper()
        return data

    async def fake_execute_tool(name: str, arguments: Any) -> str:
        return json.dumps(
            {"ok": True, "tool": name, "symbol": "AAPL", "note": "mock observation"}
        )

    monkeypatch.setattr(orchestrator, "fetch_market_data", fake_fetch)
    monkeypatch.setattr(base_agent, "execute_tool", fake_execute_tool)
    return sample_market_data
