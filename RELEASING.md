# Releasing MuJoCo Simulator

## Release identity

- Maintainer identity: X-ERA.
- Public repository: <https://github.com/Forgelab-Robotics/operator-simulator-mujoco>.
- Release branch: `master`.
- Python distribution: `mujoco-sim`.
- Command and executable: `mujoco_sim`.
- Current release: `1.1.1`, dated 2026-08-18.

The project follows Semantic Versioning. Release tags are immutable annotated tags named `v<version>`.

## Version preparation

1. Update the version in `pyproject.toml` and `src/mujoco_sim/__init__.py`.
2. Regenerate `uv.lock` when package metadata or dependencies change.
3. Update version-sensitive tests and `CHANGELOG.md`.
4. Prepare the release from a clean, reviewed commit on `master`.

## Release checks

Use Python 3.12:

```bash
uv lock --check
uv sync --frozen --all-groups --python 3.12
uv run --frozen pytest tests -q
uv run --frozen python -m compileall -q src tests examples
uv build
```

The wheel and source distribution are packaging checks only; this project does not publish them as release assets. Confirm they contain `LICENSE` and contain no private dependency source, private URL, local path, generated binary, or virtual environment.

## Linux x86_64 binary

Build on the Ubuntu 24.04 release host. The published binary uses glibc 2.39 as its deployment baseline:

```bash
sudo apt install libegl1 libgl1 libosmesa6 libglfw3
rm -rf dist build/pyinstaller
bash scripts/build_pyinstaller.sh
```

The script creates the PyInstaller one-file executable at `dist/mujoco_sim` and validates its version. Remove stale `dist/` and PyInstaller work output before building.

Verify:

```bash
dist/mujoco_sim --version
dist/mujoco_sim --help
file dist/mujoco_sim
ldd dist/mujoco_sim
```

The binary requires compatible glibc and one usable EGL, OSMesa, or GLFW/OpenGL backend. PyInstaller one-file extraction also requires a temporary directory that permits file creation, writing, and execution.

## Normalized archive and SHA-256

For release `1.1.1`, create `mujoco_sim-v1.1.1-linux-x86_64.tar.gz` containing exactly one member named `mujoco_sim`:

```bash
rm -rf release-stage
install -D -m 0755 dist/mujoco_sim release-stage/mujoco_sim
tar --sort=name --owner=0 --group=0 --numeric-owner \
  --mode='u=rwx,go=rx' --mtime='@0' \
  -C release-stage -cf - mujoco_sim \
  | gzip -n > mujoco_sim-v1.1.1-linux-x86_64.tar.gz
sha256sum mujoco_sim-v1.1.1-linux-x86_64.tar.gz \
  > mujoco_sim-v1.1.1-linux-x86_64.tar.gz.sha256
```

Inspect the archive with `tar -tvf`, verify it with `sha256sum --check`, extract it into a clean directory, and repeat the version and help checks against the extracted executable. Project documentation and licensing remain available in the repository and GitHub-generated source archives.

## Publication

Before creating the tag:

1. Confirm CI passes for the reviewed release commit on `master`.
2. Confirm the working tree is clean and the binary was built from that commit.
3. Confirm the archive contains only `mujoco_sim` with normalized metadata.
4. Confirm the binary was tested on the glibc 2.39 deployment baseline with the intended rendering backend.
5. Create and push the annotated `v1.1.1` tag to GitLab and GitHub.
6. Create the GitHub Release and upload only the `.tar.gz` archive and `.sha256` file. Include the checksum and compatibility baseline in the release notes.

Do not store repository tokens in the repository. Published archives, checksums, tags, and release entries must not be replaced.
