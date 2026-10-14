"""CLI surface: version flag and the non-TUI ``--check`` diagnostic."""

from __future__ import annotations

import pytest

import main


def test_version_flag_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main.parse_args(["--version"])
    assert excinfo.value.code == 0
    assert "financial-warroom" in capsys.readouterr().out


def test_check_backend_reports_reachable_provider(monkeypatch, capsys) -> None:
    class _Provider:
        name = "ollama"
        model_label = "qwen2.5-coder"

    class _Router:
        async def resolve(self):
            return _Provider()

    monkeypatch.setattr(main, "build_router", lambda settings: _Router())

    assert main._check_backend(main.load_settings()) == 0
    assert "ollama" in capsys.readouterr().out


def test_check_backend_reports_failure(monkeypatch, capsys) -> None:
    class _Router:
        async def resolve(self):
            raise RuntimeError("no reachable LLM provider")

    monkeypatch.setattr(main, "build_router", lambda settings: _Router())

    assert main._check_backend(main.load_settings()) == 1
    assert "No usable LLM backend" in capsys.readouterr().err
