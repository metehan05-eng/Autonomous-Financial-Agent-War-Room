from agents.orchestrator import DebateEvent
from ui.transcript import DebateTranscript


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
