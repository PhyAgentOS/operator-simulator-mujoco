[English](README.md) | [中文](README_zh.md)

# MuJoCo simulator

## BREAKING CHANGE: Forge 2.x / Dora 1.x migration

This migration from Forge 1.x / Dora 0.4.x requires coordinated deployment. Align the Forge packages, Dora CLI/coordinator/daemon, this node and its peer nodes before connecting a dataflow; replacing only one node in a Forge 1.x deployment is unsupported. Rebuild and redeploy executable artifacts with the matching environment. Keep `stable/1.x` for deployments that still use the previous stack.

The validated Python baseline is `forge-msgs==2.1.0`, `Pillow==13.0.0`, `forge-common==2.1.0`, the applicable `forge-robot` / `forge-policy` / `forge-kinematics` / `forge-tool` packages at `2.0.0`, and `dora-rs==1.0.1`. Respect this repository's Python and SDK platform requirements. Forge package families have independent versions. Rust nodes retain the published `forge_msgs` crate `2.0.0` with Dora `1.0.1`; Python Msgs `2.1.0` does not require a nonexistent Rust `2.1.0` release. The `forge_msgs.v1` message namespace remains unchanged.

This notice describes the deployment break from the previous stack; it does not declare hardware, real-model or sustained-run acceptance complete. See [this round's verification report](migration/msgs-2.1.0-report.json) for actual tests and outstanding gates.

A standalone Forge simulator with joint/locomotion commands, state/pose outputs and configurable RGB/depth rendering.

## Migration status

Target node version: **2.0.0**. These changes are candidates pending the platform,
hardware and sustained-stream acceptance gates below. A version declaration does not
mean a stable release has been accepted.

Forge Common/Msgs and Dora move to the 2.x/1.x lines. Renderers close when later initialization fails; Dora errors return failure and cleanup remains idempotent.

## Runtime and platform

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
the supported SDK platform. For Linux headless MuJoCo rendering, install the required
EGL or OSMesa libraries and select the appropriate `MUJOCO_GL` backend.

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
