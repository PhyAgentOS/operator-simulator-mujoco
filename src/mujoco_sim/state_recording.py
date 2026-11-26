"""Optional display-only native-state capture. Never supplies control inputs."""
import json
from pathlib import Path
import queue
import sys
import threading
import time


class StateRecording:
    def __init__(self, filename):
        self.stream = None
        self.last = -1.0
        self.pending = queue.Queue(maxsize=256)
        self.thread = None
        self.error = None
        self.dropped = 0
        if filename:
            path = Path(filename).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = path.open("x", buffering=1)
            self.thread = threading.Thread(target=self._write, name="display-state-writer", daemon=True)
            self.thread.start()

    def _write(self):
        try:
            while True:
                record = self.pending.get()
                if record is None:
                    break
                self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        except Exception as error:
            self.error = type(error).__name__
            print("display-only recording stopped: " + self.error, file=sys.stderr)
        finally:
            self.stream.close()

    def capture(self, data, episode):
        if self.stream is None or self.error is not None:
            return
        now = time.monotonic()
        if now - self.last < .039:
            return
        self.last = now
        record = {"display_only": True,
            "source": "actual_mujoco_data_after_physics_step", "episode": episode,
            "monotonic_ns": time.monotonic_ns(), "simulation_time": float(data.time),
            "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
            "mocap_pos": data.mocap_pos.tolist(), "mocap_quat": data.mocap_quat.tolist(),
            "dropped_frames": self.dropped}
        try:
            self.pending.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            try:
                self.pending.put(None, timeout=2)
            except queue.Full:
                self.error = "writer_shutdown_timeout"
            self.thread.join(timeout=2)
        self.stream = None
