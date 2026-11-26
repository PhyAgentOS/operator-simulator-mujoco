[English](README.md) | [中文](README_zh.md)

[English](README.md) | [简体中文](README_zh.md)

# MuJoCo simulator

## BREAKING CHANGE: Forge 2.x / Dora 1.x migration

This migration from Forge 1.x / Dora 0.4.x requires coordinated deployment. Align the Forge packages, Dora CLI/coordinator/daemon, this node and its peer nodes before connecting a dataflow; replacing only one node in a Forge 1.x deployment is unsupported. Rebuild and redeploy executable artifacts with the matching environment. Keep `stable/1.x` for deployments that still use the previous stack.

The validated Python baseline is `forge-msgs==2.1.0`, `Pillow==13.0.0`, `forge-common==2.1.0`, the applicable `forge-robot` / `forge-policy` / `forge-kinematics` / `forge-tool` packages at `2.0.0`, and `dora-rs==1.0.1`. Respect this repository's Python and SDK platform requirements. Forge package families have independent versions. Rust nodes retain the published `forge_msgs` crate `2.0.0` with Dora `1.0.1`; Python Msgs `2.1.0` does not require a nonexistent Rust `2.1.0` release. The `forge_msgs.v1` message namespace remains unchanged.

This notice describes the deployment break from the previous stack; it does not declare hardware, real-model or sustained-run acceptance complete. See [this round's verification report](migration/msgs-2.1.0-report.json) for actual tests and outstanding gates.

A standalone Forge simulator with joint/locomotion commands, state/pose outputs and configurable RGB/depth rendering.

## Migration status

Target node version: **2.1.0**. These changes are candidates pending the platform,
hardware and sustained-stream acceptance gates below. A version declaration does not
mean a stable release has been accepted.

Forge Common/Msgs and Dora move to the 2.x/1.x lines. Renderers close when later initialization fails; Dora errors return failure and cleanup remains idempotent.

## Runtime and platform

The Forge 2.x node defaults to MuJoCo Filament (`mujoco-core` and `mujoco-studio` 4.0.1). Headless Linux uses Vulkan and requires a Vulkan loader and a working driver. On Ubuntu, install `libvulkan1` and the GPU vendor driver, or `mesa-vulkan-drivers` for Mesa software rendering. macOS uses OpenGL. Existing `image_forward` configurations emit RGB or optical-axis depth in float32 metres (`32FC1`).

Select `renderer_backend: filament` (default) or `renderer_backend: classic` for all camera streams in the node YAML. For classic rendering on headless Linux, set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` before startup and install the matching OpenGL runtime.

For Filament, set `filament_graphics_api: auto` in the node YAML to use the platform default, or set `vulkan` or `opengl` explicitly. Use `auto` or `vulkan` on headless Linux. The Linux 4.0.1 OpenGL wheel requires an X display. The setting applies to every configured RGB and depth stream.

- First validation combination: Dora CLI/coordinator/daemon/node APIs 1.0.1,
  Forge Msgs 2.1.0 and Common 2.1.0; use only the families required by this node.
- Forge wire schemas remain `forge_msgs.v1`. Dora 0.x and 1.x cannot share a runtime.
- Use Python 3.12 for Python builds. Camera-only Python packages may also declare
  Python 3.11 support; test that interpreter separately before claiming it.
- Rust packages enable `arrow-v59`; the committed lock uses Arrow 59.3.0.
- Hardware camera/robot artifacts target Linux. macOS development tests do not
  qualify Linux binaries. GPU/rendering and vendor SDK prerequisites still apply.
- Keep a recoverable `stable/1.x` runtime and artifact combination.

## Setup and commands

Run from the repository root:

```sh
uv sync --frozen --python 3.12
uv run --frozen mujoco_sim --config config.example.yaml
```

Use `mujoco_sim --help` (or `uv run --frozen mujoco_sim --help`) for supported CLI options.
The example configuration is `config.example.yaml`; choose the actual device and supported
stream profiles before starting. This command requires an appropriate Dora dataflow.

## Message contract

Inputs include tick, action or action/<source>, locomotion_command and sim_command. Outputs include proprio_state, optional pose_set, sim_status and configured camera outputs.

Keep bounded queues for high-rate image consumers and document whether old images
are dropped or producers experience backpressure. Preserve image dimensions, dtype,
encoding, units and capture metadata across the node boundary.

## Tests and build

```sh
uv run --frozen pytest tests -q
bash scripts/build_pyinstaller.sh
```

Use the committed `uv.lock` and frozen/locked installs. Python vendor SDK tests require
the supported SDK platform. Run `uv run --frozen python tests/camera_smoke.py` to verify RGB alignment and metric depth using the installed graphics driver.

Validate every delivered example or rendered template with:

```sh
dora validate path/to/dataflow.yaml
```

Schema validation is followed by a real start, message exchange and stop test.
Keep the existing detailed configuration and device setup reference in
[the Chinese guide](README_zh.md) and the repository's `config/`, `examples/` and
`scripts/` directories where present.

## Acceptance and release

1. Build candidate archives from the selected commit/tag and record actual embedded
   dependency versions, platform, archive SHA256 and unpacked executable SHA256.
2. Install the final archive into a clean supported environment. Verify the CLI,
   message contract, resource cleanup and error exits.
3. For image producers, run real supported RGB/depth configurations for at least
   four continuous hours after warmup. Exercise slow/multiple consumers, stalled
   consumption, failures and repeated start/stop. Define throughput, latency and
   memory limits before testing, and retain time series rather than frame payloads.
4. Exercise the relevant device; synthetic streams do not qualify hardware capture.
   Demonstrate rollback to the complete old node/runtime combination.
5. Publish the exact accepted archive. Rebuilding invalidates that artifact's
   acceptance; repeat it before publication. Never overwrite published tags/assets.

## License

Apache-2.0. See `LICENSE` and any bundled third-party notices.

## Versioned releases

**Mandatory tag-first release rule:** merge the version/lock change, create an annotated version tag on that reviewed commit, build that exact tag, validate the artifacts, then publish. Moving branch names, untagged commit IDs and replacement of published tags are not release inputs.

| Release line | Source | Node/package version | Runtime |
| --- | --- | --- | --- |
| Forge 1.x | `stable/1.x` | `1.x`, tag `v1.X.Y` | Forge 1.x / Dora 0.4.x |
| Forge 2.x | `master` | `2.x`, tag `v2.X.Y` | Forge 2.x / Dora 1.x |

**Legacy normalization:** nodes previously at 0.x on `stable/1.x` release as **1.0.0**, retaining Forge 1.x / Dora 0.4.x dependencies and runtime behavior. Existing 1.x package versions are retained. Node/package majors now follow the Forge compatibility line; a 0.x tag is rejected.

Run **Build versioned release** from `master` for either line. First merge the version/lock changes into the selected source branch, then create an unused **annotated** tag on that reviewed commit. The tag must match package metadata. Never move existing tags or reuse a published version. Dispatch uses the workflow's reviewed tooling with the exact tagged source, so legacy tags need not contain the new release helper.

```bash
# Replace with an existing, reviewed and unpublished tag matching source metadata.
TAG=v1.X.Y
gh workflow run release.yml --ref master -f release_line=1.x -f tag="$TAG"
TAG=v2.X.Y
gh workflow run release.yml --ref master -f release_line=2.x -f tag="$TAG"
```

The workflow verifies tag type, source commit, branch ancestry, source version and locked Forge/Dora compatibility. It builds source archives and wheel/sdist where supported, checks Python distribution metadata and SHA-256, and creates or attaches to a **draft GitHub Release**. Published releases/assets are not overwritten; review all build results and acceptance evidence before publishing. This workflow does not publish node packages to PyPI.

Set `build_binaries=true` to call the existing Ubuntu 20.04 binary workflow; its existing architecture, CLI and archive checks remain in force. You can also run `build-ubuntu20.yml` directly with `ref`, `version` and `release_line`; `ref` must be an annotated tag, not a moving branch. Binary artifacts are attached only after all selected jobs succeed.

The source archive preserves the tagged tree; external SDKs/submodules, model downloads and site configuration remain separate prerequisites. A dependency or SDK build failure is not successful release acceptance. Build success does not establish single-node or Gateway-chain `dora run`, hardware, real-model, sustained-run or rollback acceptance. Every acceptance chain must include `adapter-forge-gateway` from the same compatibility line. Forge 2.x is a **breaking change**: align Forge libraries, node versions and Dora across the entire deployment; keep 1.x on its maintenance branch.

## Optional navigation simulation

Version 2.1.0 adds opt-in navigation support; profiles without these settings keep their existing behavior.

- `joint_planar_base` names the body and x/y slide plus yaw hinge joints. Velocity commands apply generalized forces through MuJoCo dynamics, so obstacles produce physical contacts. It is mutually exclusive with the existing mocap `kinematic_planar_base`. Commands expire on simulation time and a wall-clock watchdog. `base_feedback` reports measured velocities and inferred differential wheel speeds.
- `navigation_sensors` names a LiDAR site, IMU site and native gyro/accelerometer sensors. The packaged Mid360 scan pattern is raycast incrementally at physics steps. `point_cloud` is a Forge `PointCloudBuffer` with x/y/z in metres, per-point `time_offset_ns`, synthetic reflectivity and sensor-index line/tag fields. It does not model hardware reflectivity or firing electronics. `imu` is Forge `Imu` (rad/s and m/s²).
- Sensor metadata includes `capture_timestamp_ns`, `frame_id`, `clock_domain=simulation`, `time_mapping_id` and `ros_time_offset_ns=0`. `clock` is an Arrow int64 nanosecond simulation timestamp, emitted before the period's sensor payloads. A scene reset changes `time_mapping_id`; restart dependent localization/navigation sessions rather than mixing time epochs.
- Geometry groups 4 (visual decals) and 5 (collision proxies) are excluded from LiDAR rays. Physical scene geometry must remain in groups 0–3. An optional auxiliary LiDAR must be rigidly mounted; its returns are transformed into the primary sensor frame using asset extrinsics.
- `state_recording_file` enables an exclusive-create JSONL file containing actual qpos/qvel, mocap state, simulation time and monotonic capture time. A bounded writer drops display frames rather than blocking physics. This stream and optional contact evaluation are for display/acceptance only, never navigation truth feedback.

The MIT-licensed CPU ray support and scan pattern are vendored privately from MuJoCo-LiDAR. See `src/mujoco_sim/_vendor/lidar/SOURCE.json` and `LICENSE` for the pinned source, local compatibility change and checksums. No legacy MuJoCo 3 package is installed. Run `pytest tests/test_navigation_simulation.py tests/test_contact_evaluation.py` for contact, watchdog, timestamp and sensor checks. These tests do not certify an external navigation stack or physical robot.
