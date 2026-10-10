"""Market data package: yfinance retrieval and technical analysis."""

from __future__ import annotations

from data.fetcher import FetchError, MarketDataFetcher, fetch_market_data

__all__ = ["FetchError", "MarketDataFetcher", "fetch_market_data"]
