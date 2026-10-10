# Autonomous Financial Agent War Room

A terminal **multi-agent "war room"** where four AI desks debate a financial
instrument and a Chief Risk Officer delivers a single, evidence-backed
verdict. Everything streams live into a [Textual](https://textual.textualize.io/)
TUI: each agent's chain-of-thought, tool calls and final statement appear as
they are generated.

> The desk debates, the CRO decides. No silent fallbacks — if no LLM answers,
> the debate stops (unless you explicitly run the deterministic rule engine).

---

## How it works

The debate is orchestrated by `agents/orchestrator.py` and follows a fixed,
multi-round protocol:

| Round | Phase | Who | Notes |
|-------|-------|-----|-------|
| 1 | **Initial Pitch** | 🐂 Bull, 🐻 Bear, 📐 Quant (concurrently) | Each desk opens with a thesis; market-data tools are available. |
| 2 … N-1 | **Cross-Examination** | All three desks (sequential) | Each agent receives the full transcript and must challenge the weakest claim. Tools available unless `--fast`. |
| N | **CRO Synthesis** | ⚖️ Chief Risk Officer | The moderator weighs every argument and emits a structured verdict. No tools. |

The final verdict is parsed into `decision` / `confidence` / `rationale`:

```
decision   STRONG BUY | BUY | HOLD | SELL | STRONG SELL
confidence 0 … 100
rationale  three concrete, data-backed reasons
```

If the CRO answer does not contain valid JSON, the decision is derived
deterministically from the same indicators, so a consensus always exists.

### The four desks

| Agent | Emoji | Role |
|-------|-------|------|
| Bullish Growth Strategist | 🐂 | Momentum, catalysts, upside optionality. |
| Bearish Risk Manager | 🐻 | Valuation, macro risk, downside technicals. |
| Quantitative Analyst | 📐 | Pure technical evidence (RSI, SMA, MACD, volatility, volume). |
| Chief Risk Officer | ⚖️ | Moderates the debate and issues the final consensus. |

### Market data & tools

`data/fetcher.py` pulls price history and fundamentals from Yahoo Finance
(equities, ETFs, crypto, futures, FX) and derives **RSI(14)**, **SMA(50/200)**,
**MACD(12,26,9)**, realised **volatility**, **volume** profile, 52-week range
position and a trend regime. Agents may call three tools during a turn:

- `get_stock_quote`
- `get_rsi_and_indicators`
- `get_latest_news`

### LLM providers

Providers are probed in order and the first healthy one is cached for the
session:

```
auto → Evren → Ollama → OpenAI → Gemini → Anthropic
```

- **Evren gateway** is the default (official `openai` SDK), using
  `EVREN_MODEL` for debate turns and `EVREN_TOOL_MODEL` for tool-calling turns.
- **Reasoning deltas** (`delta.reasoning_content` / `reasoning`) stream live.
- **Strict mode by default**: a session without any reachable LLM raises a
  clear error instead of fabricating text.
- The **deterministic rule engine** (`LLM/fallback.py`) only runs with
  `--no-llm` / `LLM_PROVIDER=fallback`.

---

## Requirements

- **Python 3.10+**
- An LLM backend: the Evren gateway, a local [Ollama](https://ollama.com/)
  server, or an OpenAI / Gemini / Anthropic API key. Not needed with `--no-llm`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env
```

Then edit `.env`. Every variable is optional; the most common ones:

```dotenv
# auto | evren | ollama | openai | gemini | anthropic | fallback
LLM_PROVIDER=auto

# Evren gateway (default)
EVREN_BASE_URL=https://evren-llmapi.ssyz.org.tr/v1
EVREN_API_KEY=your-key-here
EVREN_MODEL=deepseek-v4.1-flash
EVREN_TOOL_MODEL=glm-5.3

# Local models
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder

# Cloud fallbacks
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=

# Behaviour
LLM_TIMEOUT=45
LLM_ROUNDS=3
LLM_MAX_TOKENS=3000
LLM_TEMPERATURE=0.7
DEFAULT_TICKER=AAPL
HISTORY_PERIOD=1y
```

> **Never commit `.env`.** It is already listed in `.gitignore`; `.env.example`
> is the safe template.

## Usage

```bash
python main.py                      # pre-filled ticker (AAPL)
python main.py --ticker BTC-USD     # pre-fill another symbol
python main.py --no-llm             # deterministic rule engine, offline
python main.py --rounds 4           # pitch + 2 cross-examinations + synthesis
python main.py --fast               # quicker run: tools only in the pitch round
python main.py --max-tokens 4000    # larger output budget per call
```

Run `python main.py --help` for the full flag list.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Start the war room |
| `Enter` (in ticker box) | Start the war room |
| `Esc` | Cancel the running debate |
| `Ctrl+S` | Export the transcript (Markdown, under `exports/`) |
| `Ctrl+R` | Clear the log |
| `Q` | Quit |

Each finished debate is also exported automatically to
`exports/<TICKER>-<timestamp>.md`, including the reasoning, tool calls,
statements and the CRO verdict.

---

## Project layout

```
main.py               CLI entry point (argparse → Textual app)
config.py             Immutable Settings loaded from the environment
agents/
  base_agent.py       Shared agent plumbing (streaming + tool loop)
  bullish.py          🐂 Bullish Growth Strategist
  bearish.py          🐻 Bearish Risk Manager
  quant.py            📐 Quantitative Analyst
  moderator.py        ⚖️ CRO, verdict parsing + fallback derivation
  orchestrator.py     Multi-round debate engine (async event stream)
LLM/
  base.py             Provider interface + resilient router
  openai_provider.py  OpenAI-compatible client (OpenAI/Gemini/Anthropic/Evren)
  evren_provider.py   Evren gateway specialization
  ollama_provider.py  Local Ollama client
  fallback.py         Deterministic rule engine
data/fetcher.py       Yahoo Finance data, indicators and LLM tools
ui/
  app.py              Textual application, layout and event dispatch
  widgets.py          Metric card, streaming debate log, verdict banner
  transcript.py       Markdown transcript builder / exporter
tests/                Offline test suite (mock provider + fixtures)
```

## Development

```bash
ruff check .            # lint
mypy .                  # type-check
pytest                  # tests (fully offline, no network)
```

The test suite never touches the network: `tests/mock_evren.py` implements the
real provider streaming contract and `tests/conftest.py` stubs market data and
tool execution.

## Notes & limitations

- Technical indicators are derived from daily closes; short histories
  (< 200 sessions) yield `n/a` for SMA(200) / MACD.
- Yahoo Finance may throttle or block `Ticker.info`; fundamentals are treated
  as best effort and never crash the UI.
- Output depends on the chosen model; reasoning-heavy models may take several
  minutes per debate. Use `--rounds 2` or `--fast` for a quicker run.
- This tool is for research and debate simulation — **not financial advice**.
# Autonomous-Financial-Agent-War-Room
