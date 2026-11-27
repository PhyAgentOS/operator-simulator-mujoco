"""Mid360-only subset of the MIT-licensed MuJoCo-LiDAR scan generator."""
from pathlib import Path

import numpy as np


class LivoxGenerator:
    def __init__(self, name: str):
        if name != "mid360":
            raise ValueError("Only the bundled Mid360 pattern is supported")
        self.samples = 24000
        self.ray_angles = np.load(Path(__file__).parent / "scan_mode" / "mid360.npy", allow_pickle=False)
        self.n_rays = len(self.ray_angles)
        self.currStartIndex = 0

    def sample_ray_angles(self, downsample: int = 1) -> tuple[np.ndarray, np.ndarray]:
        if self.currStartIndex + self.samples > self.n_rays:
            self.ray_part1 = self.ray_angles[self.currStartIndex :]
            self.ray_part2 = self.ray_angles[: self.samples - len(self.ray_part1)]
            self.currStartIndex = self.samples - len(self.ray_part1)
            self.ray_out = np.concatenate([self.ray_part1, self.ray_part2], axis=0)
        else:
            self.ray_part1 = self.ray_angles[
                self.currStartIndex : self.currStartIndex + self.samples
            ]
            self.currStartIndex += self.samples
            self.ray_out = self.ray_part1
        if downsample > 1:
            self.ray_out = self.ray_out[::downsample]
        return self.ray_out[:, 0], self.ray_out[:, 1]
