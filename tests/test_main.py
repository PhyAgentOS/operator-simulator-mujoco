from __future__ import annotations

import pytest
from mujoco_sim.main import cli


def test_cli_version(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.argv", ["mujoco_sim", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        cli()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == "mujoco-sim 1.1.0\n"
