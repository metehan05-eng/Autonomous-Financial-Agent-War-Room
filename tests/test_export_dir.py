"""Transcript exports must land in the working directory, not in the package."""

from __future__ import annotations

from pathlib import Path

from ui import app as app_module


def test_export_directory_prefers_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    directory = app_module._export_directory()
    assert directory == tmp_path / "exports"
    assert directory.is_dir()
