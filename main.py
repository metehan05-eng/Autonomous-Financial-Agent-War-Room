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
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from config import APP_NAME, Settings, load_settings
from LLM import build_router
from ui.app import WarRoomApp

#: Fallback used when the package is run from a source checkout (not installed).
_FALLBACK_VERSION = "0.1.0"


def _app_version() -> str:
    """Installed package version, or a sensible default for source runs."""
    try:
        return _pkg_version("financial-agent-war-room")
    except PackageNotFoundError:
        return _FALLBACK_VERSION


def _check_backend(settings: Settings) -> int:
    """Probe the configured LLM backend without starting the TUI.

    Returns:
        ``0`` when a backend is reachable, ``1`` otherwise.
    """
    router = build_router(settings)
    try:
        provider = asyncio.run(router.resolve())
    except Exception as exc:
        print(f"✗ No usable LLM backend: {exc}", file=sys.stderr)
        return 1
    kind = "rule engine" if provider.name == "fallback" else "LLM provider"
    print(f"✓ Backend ready — {kind}: {provider.name} / {provider.model_label}")
    return 0


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
        help="Total debate rounds (>=2). Default: 2 = pitch and synthesis; "
        "3 adds a cross-examination.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Only offer market-data tools in the pitch round and lower "
        "the temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.max_tokens,
        help="Maximum output tokens per call. Keep large enough for the answer "
        "to survive chain-of-thought (default 3000).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Probe the configured LLM backend, print the result and exit "
        "without launching the UI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_app_version()}",
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

    if args.check:
        return _check_backend(settings)

    app = WarRoomApp(
        initial_ticker=args.ticker,
        settings=settings,
        router=build_router(settings),
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
