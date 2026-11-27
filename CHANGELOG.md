# Changelog

All notable changes to this project are documented in this file.

The project follows Semantic Versioning. Dates use the `YYYY-MM-DD` format.

## Unreleased

- Add independently configured native IMU and Mid360 ray scans with physics-step timestamps and reset epochs.
- Add opt-in direct joint velocity feedback and per-actuator velocity-command expiry; preserve legacy defaults.
- Expose optional state and per-step contact observations without recording or task-specific collision rules.

## 2.0.2 - 2026-11-24

- Choose Filament (default) or classic rendering for RGB and optical-axis depth with `renderer_backend`. Camera topics and float32 depth units in metres are preserved.
- Configure the graphics API with `filament_graphics_api` (`auto`, `vulkan`, or `opengl`). Headless Linux uses Vulkan by default.
- Bundle the Filament native extension, shaders and environment lighting in the Ubuntu 20.04 x86_64 executable.

## 2.0.1 - 2026-11-24

- Upgrade the Forge 2.x simulator to MuJoCo 4.0.1 and use the classic renderer’s native cleanup.
- Include the OSMesa platform module in the Ubuntu 20.04 executable for headless RGB and depth rendering.

## 1.1.1 - 2026-08-18

### Added

- Added the Apache License 2.0 license, X-ERA authorship metadata, public repository links, and contributor and release documentation.
- Added GitHub CI for the locked Python 3.12 test and binary-build workflows.

### Changed

- Replaced private dependency source overrides with the publicly published `forge-common` and `forge-msgs` packages while preserving their existing compatible version ranges.
- Documented the normalized Linux x86_64 binary archive and SHA-256 publication procedure.

There are no intentional changes to the simulator configuration, Dora messages, control behavior, physics stepping, rendering, or image encoding contracts in this release.

## 1.1.0 - 2026-08-15

### Changed

- Moved initial joint positions from `initial_positions` to the extensible `initial_state.joint_positions` configuration path.
- Applied configured initial joint positions at startup and scene reset while preserving model defaults for unspecified joints and synchronizing actuator targets.
- Rejected removed or unsupported initial-state fields instead of silently accepting them.

## 1.0.0 - 2026-08-02

### Added

- Initial standalone `mujoco_sim` release with Dora tick-driven MuJoCo stepping, sparse joint commands, joint-state output, optional kinematic planar locomotion, scene reset, pose output, and configurable color/depth camera outputs.
- Added Python 3.12 packaging, the `mujoco_sim` console entry point, a minimal example, tests, and a PyInstaller build script.
