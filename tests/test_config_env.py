"""The ``.env`` discovery order matters for pipx/global installs."""

from __future__ import annotations

import os
from pathlib import Path

import config


def test_env_candidates_prefer_current_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    candidates = config._env_candidates()
    assert candidates[0] == tmp_path / ".env"
    assert config.USER_ENV_FILE in candidates


def test_load_env_reads_dotenv_from_cwd(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("WARROOM_TEST_KEY=hello\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WARROOM_TEST_KEY", raising=False)

    config._load_env()

    assert os.environ["WARROOM_TEST_KEY"] == "hello"
    monkeypatch.delenv("WARROOM_TEST_KEY", raising=False)
