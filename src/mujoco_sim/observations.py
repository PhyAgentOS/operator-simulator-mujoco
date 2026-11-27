"""Opt-in read-only state and contact observations; no recording or evaluation."""

from __future__ import annotations

import time


def state_snapshot(model, data, episode):
    return {
        "schema_version": 1,
        "episode": episode,
        "monotonic_ns": time.monotonic_ns(),
        "simulation_time": float(data.time),
        "nq": model.nq,
        "nv": model.nv,
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "mocap_pos": data.mocap_pos.tolist(),
        "mocap_quat": data.mocap_quat.tolist(),
    }


def contact_sample(data):
    """Include contact candidates unchanged; consumers decide what counts."""
    return {
        "capture_timestamp_ns": round(float(data.time) * 1e9),
        "contacts": [
            {
                "geom1": int(c.geom1),
                "geom2": int(c.geom2),
                "distance": float(c.dist),
                "normal": c.frame[:3].tolist(),
            }
            for c in data.contact[: data.ncon]
        ],
    }
