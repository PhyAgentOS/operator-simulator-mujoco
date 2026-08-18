# Contributing to MuJoCo Simulator

Thank you for contributing to the X-ERA MuJoCo simulator adapter. Keep changes focused, preserve the documented Dora message and configuration contracts, and add tests for behavior changes.

## Development setup

The project requires Python 3.12 or later and uses `uv.lock` as the dependency source of truth:

```bash
uv lock --check
uv sync --frozen --all-groups --python 3.12
```

Camera rendering requires one of EGL, OSMesa, or GLFW and the corresponding system OpenGL libraries. On Ubuntu 24.04:

```bash
sudo apt install libegl1 libgl1 libosmesa6 libglfw3
```

## Required checks

Before opening a pull request, run:

```bash
uv lock --check
uv run --frozen pytest tests -q
uv run --frozen python -m compileall -q src tests examples
uv build
```

If binary packaging is affected, also run:

```bash
bash scripts/build_pyinstaller.sh
dist/mujoco_sim --version
dist/mujoco_sim --help
```

The public binary archive must contain only `mujoco_sim`.

## Pull request guidelines

- Keep each pull request focused on one behavior or maintenance concern.
- Preserve the `forge_msgs` payload semantics and the `mujoco_sim` CLI and executable names.
- Update `README.md`, `CHANGELOG.md`, tests, and examples when their contracts change.
- Prefer small inline MJCF fixtures that do not depend on external assets.
- Do not commit generated executables, build directories, virtual environments, model captures, or simulation recordings.
- Do not include secrets, credentials, private URLs, personal filesystem paths, private model assets, or private identifiers.
- Do not add binary blobs or new dependencies without a clear need and a license review.

## License

By contributing, you agree that your contributions are licensed under the Apache License, Version 2.0.
