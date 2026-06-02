# -*- mode: python ; coding: utf-8 -*-
# 使用 SPEC 所在目录，避免 PyInstaller 按 spec 目录解析时路径重复。

import os

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

_spec_dir = os.path.dirname(os.path.abspath(SPEC))
_node_dir = os.path.dirname(_spec_dir)

mujoco_binaries = collect_dynamic_libs("mujoco")

hiddenimports = (
    collect_submodules("forge_msgs")
    + collect_submodules("forge_common")
    + collect_submodules("cv2")
    + [
        "config",
        "handlers",
        "simulator",
        "dora",
        "mujoco",
        "numpy",
        "pyarrow",
        "yaml",
        "OpenGL.platform.egl",
        "OpenGL.EGL",
        "OpenGL.platform.glfw",
    ]
)

a = Analysis(
    [os.path.join(_node_dir, "main.py")],
    pathex=[_node_dir],
    binaries=mujoco_binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="mujoco_sim",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
