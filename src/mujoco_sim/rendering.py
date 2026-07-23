"""MuJoCo renderer setup, image encoding, and camera timing helpers."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from forge_msgs import CompressedImage, Image

from .config import CameraConfig, MuJoCoNodeConfig


def _patch_mujoco_renderer_close() -> None:
    """Avoid close errors when Renderer initialization fails before context setup."""

    def _safe_close(self) -> None:
        try:
            if getattr(self, "_gl_context", None):
                self._gl_context.free()
            if getattr(self, "_mjr_context", None):
                self._mjr_context.free()
        except Exception:
            pass
        try:
            object.__setattr__(self, "_gl_context", None)
            object.__setattr__(self, "_mjr_context", None)
        except Exception:
            pass

    mujoco.Renderer.close = _safe_close


_patch_mujoco_renderer_close()


def _create_renderer(model: mujoco.MjModel, height: int = 480, width: int = 640):
    """Create a MuJoCo offscreen renderer."""
    return mujoco.Renderer(model, height, width)


def create_renderer_for_config(
    model: mujoco.MjModel,
    config: MuJoCoNodeConfig,
):
    """Create the shared renderer at the first explicitly configured resolution."""
    render_height, render_width = 480, 640
    for _, cam_config in config.camera_outputs:
        if cam_config.width is not None and cam_config.height is not None:
            render_width = int(cam_config.width)
            render_height = int(cam_config.height)
            break
    return _create_renderer(model, height=render_height, width=render_width)


def load_cv2_module(config: MuJoCoNodeConfig):
    """Load OpenCV only when compressed image output is configured."""
    if config.image_format not in {"jpeg", "png"}:
        return None

    import cv2  # type: ignore

    return cv2


def _build_image_payload(
    pixels: np.ndarray,
    config: MuJoCoNodeConfig,
    cv2_module: Any | None,
):
    """Encode rendered pixels as raw Image or jpeg/png CompressedImage."""
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    if config.image_format == "raw":
        return Image(
            height=height,
            width=width,
            encoding="rgb8",
            step=width * 3,
            data=pixels.tobytes(),
        ).to_arrow()

    if cv2_module is None:
        raise RuntimeError("image_format 为 jpeg/png 时需要安装 opencv-python")

    cv2 = cv2_module
    # OpenCV 编码器默认按 BGR 解释 3 通道输入，这里将 RGB 转为 BGR 再压缩。
    bgr = pixels[..., ::-1]
    if config.image_format == "jpeg":
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(config.image_jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return CompressedImage(format="jpeg", data=encoded.tobytes()).to_arrow()

    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return CompressedImage(format="png", data=encoded.tobytes()).to_arrow()


def _build_depth_payload(depth_m: np.ndarray):
    """Encode MuJoCo depth in float32 meters as an Image with 32FC1 encoding."""
    depth32 = np.ascontiguousarray(depth_m, dtype=np.float32)
    return Image.from_numpy(depth32, encoding="32FC1").to_arrow()


def render_color_payload(
    renderer,
    data: mujoco.MjData,
    camera: str,
    config: MuJoCoNodeConfig,
    cv2_module: Any | None,
):
    """Render RGB, explicitly disabling any previous depth rendering mode."""
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    return _build_image_payload(pixels, config, cv2_module)


def render_depth_payload(renderer, data: mujoco.MjData, camera: str):
    """Render depth in meters."""
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth_m = renderer.render()
    return _build_depth_payload(depth_m)


def should_emit_camera(
    sim_time: float,
    next_camera_time: dict[str, float],
    key: str,
    fps: float | None,
) -> bool:
    if fps is None:
        return True
    if sim_time < next_camera_time[key]:
        return False
    next_camera_time[key] += 1.0 / fps
    return True


def camera_output_key(cam_name: str, cam_config: CameraConfig) -> str:
    """Include the output topic because one camera may emit multiple image streams."""
    return f"{cam_config.image_type}:{cam_name}:{cam_config.output_id}"


def initial_camera_times(config: MuJoCoNodeConfig) -> dict[str, float]:
    """Initialize the next simulation time for every camera output stream."""
    return {
        camera_output_key(cam_name, cam_config): 0.0
        for cam_name, cam_config in config.camera_outputs
    }
