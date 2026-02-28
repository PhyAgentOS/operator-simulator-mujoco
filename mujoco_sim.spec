# -*- mode: python ; coding: utf-8 -*-
# 使用 SPEC 所在目录，避免 PyInstaller 按 spec 目录解析时路径重复。

import os

_spec_dir = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(_spec_dir, 'main.py')],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['OpenGL.platform.egl', 'OpenGL.EGL'],
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
    name='mujoco_sim',
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

