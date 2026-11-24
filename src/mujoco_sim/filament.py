"""Offscreen RGB and metric optical-axis depth using MuJoCo's Filament backend."""

import mujoco
from mujoco import mjrf
from mujoco.rendering.filament.model_scene import ModelScene
from mujoco.rendering.filament.renderer import Renderer, View
import numpy as np


class FilamentRenderer:
    def __init__(self, model: mujoco.MjModel, height: int, width: int):
        self.model = model
        # mjv_updateCamera needs only the stereo camera pair, no drawable geoms.
        self.camera_scene = mujoco.MjvScene(model, maxgeom=1)
        self.fixed = mujoco.MjvCamera()
        self.fixed.type = mujoco.mjtCamera.mjCAMERA_FIXED

        ctx = mjrf.Context(mjrf.ContextConfig())
        self.renderer = Renderer(ctx)
        scene = self.renderer.scene("model")
        self.model_scene = ModelScene(ctx, model, scene)
        self.rgb = np.empty((height, width, 3), np.uint8)
        self.depth = np.empty((height, width), np.float32)
        self.renderer.target("rgb", self.rgb)
        self.renderer.target("depth", self.depth, mjrf.PixelFormat.PIXEL_FORMAT_R32F)
        # MuJoCo 4.0.1's Python wrapper does not expose GetHeadlightIntensity.
        intensity = 40000.0
        custom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "filament.fallback.head_light_intensity")
        if custom >= 0:
            intensity = float(model.numeric_data[model.numeric_adr[custom]])
        self.color_view = View(
            "model", "rgb",
            headlight=bool(model.vis.headlight.active) and not np.any(model.light_intensity > 0),
            headlight_color=tuple(model.vis.headlight.diffuse),
            headlight_intensity=intensity,
        )
        self.depth_view = View(
            "model", "depth", draw_mode=mjrf.DrawMode.DRAW_MODE_DEPTH,
            post_processing=False, reflections=False, shadows=False, skybox=False, haze=False,
        )

    def render(self, data: mujoco.MjData, camera: str, *, depth: bool = False):
        self.fixed.fixedcamid = self.model.camera(camera).id
        mujoco.mjv_updateCamera(self.model, data, self.fixed, self.camera_scene)
        gl = mujoco.mjv_averageCamera(*self.camera_scene.camera)
        # Passing MjvGLCamera straight to the 4.0.1 Python bridge drops three
        # projection fields used by calibrated and orthographic cameras.
        view = self.depth_view if depth else self.color_view
        view.camera = mjrf.Camera(
            pos=gl.pos, forward=gl.forward, up=gl.up, orthographic=gl.orthographic,
            frustum_width=gl.frustum_width, frustum_center=gl.frustum_center,
            frustum_top=gl.frustum_top, frustum_bottom=gl.frustum_bottom,
            frustum_near=gl.frustum_near, frustum_far=gl.frustum_far,
        )
        self.model_scene.update(data)
        self.renderer.render([view])
        if not depth:
            return self.rgb

        # Filament's unlit depth pass writes reversed clip depth (near=1, far=0).
        near, far = gl.frustum_near, gl.frustum_far
        if gl.orthographic:
            return far - self.depth * (far - near)
        return (near * far) / (near + self.depth * (far - near))

    def close(self):
        # Release retained objects before the renderer and its native context.
        self.model_scene = None
        self.renderer = None
