from agents.base_agent import AgentSpec
from agents.orchestrator import DebateEvent
from ui.transcript import DebateTranscript


def _spec() -> AgentSpec:
    return AgentSpec(
        key="bull",
        name="Bullish Growth Strategist",
        emoji="🐂",
        color="green",
        short="Bull",
    )


def test_transcript_basic_export() -> None:
    tr = DebateTranscript()
    tr.set_ticker("AAPL")
    events = [
        DebateEvent(kind="round", round_index=1, round_title="Round 1"),
        DebateEvent(kind="done"),
    ]
    for e in events:
        tr.accept(e)
    md = tr.markdown()
    assert "AAPL" in md
    assert "CRO Verdict" in md


def test_reasoning_tokens_are_concatenated_not_split() -> None:
    tr = DebateTranscript()
    spec = _spec()
    tr.accept(
        DebateEvent(
            kind="stream_start", agent=spec, round_index=1, round_title="Round 1"
        )
    )
    for token in ["The ", "bull ", "case ", "is ", "oversold."]:
        tr.accept(
            DebateEvent(kind="stream_delta", agent=spec, round_index=1, mode="reasoning", text=token)
        )
    block = tr._by_key["1:bull"]
    assert block.thinking == "The bull case is oversold."
    assert "\n" not in block.thinking


def test_reasoning_segments_split_on_tool_round_trip() -> None:
    tr = DebateTranscript()
    spec = _spec()
    tr.accept(
        DebateEvent(
            kind="stream_start", agent=spec, round_index=1, round_title="Round 1"
        )
    )
    for token in ["Let me ", "check the data."]:
        tr.accept(
            DebateEvent(kind="stream_delta", agent=spec, round_index=1, mode="reasoning", text=token)
        )
    tr.accept(DebateEvent(kind="tool", agent=spec, round_index=1, text="get_stock_quote({})"))

    for token in ["Data ", "confirms it."]:
        tr.accept(
            DebateEvent(kind="stream_delta", agent=spec, round_index=1, mode="reasoning", text=token)
        )
    block = tr._by_key["1:bull"]
    assert block.thinking == "Let me check the data.\n\nData confirms it."
