"""Display recording is optional, portable and never overwrites a prior run."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_sim.state_recording import StateRecording


def test_recording_without_development_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("NAV_SIM_DEV_ROOT", raising=False)
    path = tmp_path / "session" / "state.jsonl"
    writer = StateRecording(path)
    data = SimpleNamespace(time=0.02, qpos=np.array([1.0, 2.0]),
                           qvel=np.zeros(2), mocap_pos=np.empty((0, 3)),
                           mocap_quat=np.empty((0, 4)))
    writer.capture(data, 3)
    writer.close()
    row = json.loads(path.read_text())
    assert row["qpos"] == [1.0, 2.0]
    assert row["simulation_time"] == 0.02
    assert row["episode"] == 3
    assert row["display_only"] is True
    with pytest.raises(FileExistsError):
        StateRecording(path)
    assert json.loads(path.read_text()) == row


def test_disabled_recording_needs_no_file():
    writer = StateRecording(None)
    writer.capture(None, 0)
    writer.close()
    assert writer.stream is None
