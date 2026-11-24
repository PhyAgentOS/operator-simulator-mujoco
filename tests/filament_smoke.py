"""Run with a working offscreen graphics driver, including in headless CI."""

import mujoco
import numpy as np

from mujoco_sim.rendering import _create_renderer


def main():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><statistic extent="1"/><visual><map znear=".1" zfar="10"/></visual>
      <worldbody>
        <camera name="eye" pos="0 0 2"/>
        <camera name="ortho" pos="0 0 2" projection="orthographic" fovy="2"/>
        <body pos="0 .35 0"><joint name="lift" type="slide" axis="0 0 1"/>
          <geom type="box" size=".22 .18 .1" rgba="1 .05 .03 1"/>
        </body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    renderer = _create_renderer(model, height=80, width=100)
    try:
        for z in (0, .45):
            data.qpos[0] = z
            mujoco.mj_forward(model, data)
            for camera in ("eye", "ortho"):
                depth = renderer.render(data, camera, depth=True)
                rgb = renderer.render(data, camera)
                assert rgb.shape == (80, 100, 3) and rgb.dtype == np.uint8
                assert depth.shape == (80, 100) and depth.dtype == np.float32
                assert np.all(np.isfinite(depth))
                foreground = depth < 5
                y, x = np.where(foreground)
                assert 0 < y.mean() < 40
                row, col = round(y.mean()), round(x.mean())
                assert rgb[row, col, 0] > rgb[row, col, 1] + 10
                np.testing.assert_allclose(depth[row, col], 1.9-z, atol=.002)
                np.testing.assert_allclose(depth[-1, 0], 10, atol=.002)
    finally:
        renderer.close()
        renderer.close()


if __name__ == "__main__":
    main()
