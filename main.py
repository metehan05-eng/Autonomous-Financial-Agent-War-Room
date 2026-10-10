#!/usr/bin/env python3
"""Autonomous Financial Agent War Room - entry point.

Usage:
    python main.py                 # pre-filled ticker from config (AAPL)
    python main.py --ticker BTC-USD
    python main.py --no-llm        # force the deterministic rule engine
    python main.py --rounds 4      # pitch + 2 cross-examinations + synthesis
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace

from config import APP_NAME, load_settings
from LLM import build_router
from ui.app import WarRoomApp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list, defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    settings = load_settings()
    parser = argparse.ArgumentParser(
        prog="financial-warroom",
        description=f"{APP_NAME} - multi-agent market debate in your terminal.",
    )
    parser.add_argument(
        "-t",
        "--ticker",
        default=settings.default_ticker,
        help="Ticker to pre-fill (AAPL, BTC-USD, GC=F, EURUSD=X).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip remote LLM probing and use the rule-based engine only.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=settings.debate_rounds,
        help="Total debate rounds (>=2). Default: 3 = pitch, cross, synthesis.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick debate: tools only in the pitch round, lower temperature. "
        "Fewer LLM round trips - ideal for impatient gateways.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.max_tokens,
        help="Maximum output tokens per call. Keep large enough for the answer "
        "to survive chain-of-thought (default 3000).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the TUI.

    Args:
        argv: Optional argument list override.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    settings = replace(
        load_settings(),
        debate_rounds=max(2, args.rounds),
        max_tokens=max(256, args.max_tokens),
        fast=args.fast,
    )
    if args.no_llm:
        settings = replace(settings, provider="fallback")

    app = WarRoomApp(
        initial_ticker=args.ticker,
        settings=settings,
        router=build_router(settings),
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
