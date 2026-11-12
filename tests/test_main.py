from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from mujoco_sim.main import cli


def test_cli_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["mujoco_sim", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        cli()

    assert exc_info.value.code == 0
    version = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]["version"]
    assert capsys.readouterr().out == f"mujoco-sim {version}\n"
