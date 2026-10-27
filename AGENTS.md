# Repository Guide

Go2 RL Lab trains and evaluates Unitree Go2 velocity locomotion policies with Isaac Lab 3.0 and a project-specific RSL-RL implementation. The active branch is `lab3.0`. Run commands from the repository root.

## Local validation files

- Treat the entire `tests/` directory as local-only, including Python scripts, Markdown reports, logs, images, videos, converted assets, and checkpoints.
- Never stage, commit, force-add, push, upload, or attach any file under `tests/` to a remote service, pull request, or release.
- Place new local validation scripts and reports under `tests/`; do not move them elsewhere to bypass this policy.
- Keep `/tests/` ignored without exceptions. `.gitignore` does not exclude files already tracked: remove such files from the Git index while preserving their local copies.
- Before a commit or push, verify that `git ls-files tests` is empty and that no added or modified `tests/` file appears in the index. Staged deletions of previously tracked test files are allowed to enforce this policy.
- Keep validation results locally under `tests/upgrade_isaaclab_3_ea/`. Reference their outcomes in a concise change summary without publishing the files themselves.
- Keep `logs/`, exported policies, generated USD files, videos, and training checkpoints out of commits. Preserve the bundled Go2 USD assets and their Git LFS configuration.

## Framework and repository layout

| Path | Responsibility |
| --- | --- |
| `go2_rl_lab.sh` | Project CLI for editable installation, task discovery, training, distributed training, playback, and MuJoCo deployment |
| `scripts/rsl_rl/` | Training/playback orchestration, CLI overrides, runtime bootstrap, visualization ports, and video encoding |
| `scripts/utils/` | Task listing and editor configuration utilities |
| `scripts/tools/` | Asset conversion and GPU utilization tools |
| `source/go2_rl_lab/go2_rl_lab/tasks/` | Gym registrations, scene/environment configs, task MDP terms, terrain generation, and agent configs |
| `source/go2_rl_lab/go2_rl_lab/assets/` | Go2 USD and URDF assets, articulation configs, and custom motor model |
| `source/go2_rl_lab/go2_rl_lab/cmd_utils/` | Body-frame velocity sampling, limits, curricula, state, and visualization |
| `source/go2_rl_lab/go2_rl_lab/physics/` | Newton MuJoCo-Warp extensions for initialization-time center-of-mass buckets and sparse hard resets |
| `source/go2_rl_lab/go2_rl_lab/utils/` | Terrain helpers, privileged height scans, symmetry, curricula, and configuration composition |
| `source/rsl_rl/rsl_rl/` | Vendored RSL-RL algorithms, policies, storage, runners, logging, export, and RoboGauge integration |
| `deploy/deploy_mujoco/` | Standalone MuJoCo inference and simulation-to-simulation deployment |
| `docs/` | Project documentation and upstream license notices |

Isaac Lab owns simulation, assets, sensors, scene replication, and manager-based environments. Gym registrations connect each task ID to its environment implementation and configuration entry points. The project configures observation, action, command, event, reward, termination, and curriculum managers rather than implementing a separate simulator loop.

The project-owned vector environment wrapper adapts Isaac Lab observations to TensorDict groups for the vendored RSL-RL runners. Runner and policy classes are resolved from configuration; preserve that dispatch for PPO and MoE-CTS.

### Task families and training pipeline

- `go2-flat`: flat-ground PPO baseline with fixed velocity ranges and mixed-task rewards.
- `go2-mixed` and `go2-mixed-moe-cts`: mixed-terrain PPO and mixture-of-experts concurrent teacher/student training.

All tasks use body-frame commands `[v_x, v_y, yaw_rate]` in `m/s`, `m/s`, and `rad/s`. Policy observations contain a 45-dimensional frame and ten-frame history. Critic observations and the MoE teacher include privileged simulation state and terrain heights. MuJoCo deployment receives raw velocity commands; TorchScript exports maintain observation history internally.

Discover the registered task set with `./go2_rl_lab.sh --list`; update this guide when task families change. Preserve observation layouts, symmetry mappings, checkpoint formats, task-specific rewards, and the MoE teacher/student split when updating compatibility code. Historical Go2 RL Gym checkpoints require their own simulator, observation, and controller settings; do not assume cross-runtime checkpoint compatibility or numerical equivalence.

## Runtime and installation

Inspect the active branch, working tree, and tracking remote before pulling or incorporating upstream commits. Keep Isaac Lab 3.0 changes on `lab3.0`.

The configured runtime is Docker image `wtyyy/isaaclab:3.0.0`, with Python 3.12, Isaac Lab `3.0.0`, Isaac Sim 6.2, PyTorch 2.11, Newton 1.6.1, and Warp 1.17. Inspect installed versions when the image changes.

In the current workspace, the host checkout `/mnt/data/Coding/go2_rl_lab` is mounted at `/home/user/Coding/go2_rl_lab` in container `go2-rl-lab-3.0`. Use the container's Python for simulation and GPU validation. Do not modify unrelated checkouts or running workloads.

```bash
# Install both project packages in the configured container environment
./go2_rl_lab.sh --install

# Reinstall only the task package without replacing the existing runtime stack
python -m pip install --no-deps -e source/go2_rl_lab

# Optional recording encoder
python -m pip install -e 'source/go2_rl_lab[video]'

# List current Gym tasks
./go2_rl_lab.sh --list
```

Keep the editable `source/rsl_rl` package. Its distribution version is `3.3.0`, with project PPO, MoE-CTS, symmetry, export, and RoboGauge extensions. Upstream Isaac Lab declares a newer RSL-RL dependency; replacing the vendored package can remove required project behavior. Inspect and report runtime dependency conflicts instead of blindly upgrading packages.

## Isaac Lab 3.0 contracts

- Supported physics selectors are `newton_mjwarp` and `isaacsim_physx`. Newton is the project default. Use the explicit Isaac Sim PhysX selector so backend-specific sensor and reward presets resolve consistently.
- Bootstrap the required runtime before importing implementation modules that load USD bindings. Kit and standalone USD bindings must not be mixed within one process. The provisional bootstrap selects the runtime; the composed task config owns the actual solver, scene, randomization, and sensors.
- Resolve registered configuration and presets through `isaaclab_tasks.utils.resolve_task_config`. Restore temporary Gym registry changes in `finally`. Keep the project's separate `play_env_cfg_entry_point` support.
- Asset and sensor data use `ProxyArray`; access `.torch` or `.warp` explicitly. Isaac Lab quaternions are XYZW.
- Use keyword-only `write_*_to_sim_index` for selected environment IDs and `write_*_to_sim_mask` for masks. Preserve the distinction between partial indexed data and full masked data.
- Keep `sim.use_newton_actuators=False` for the custom Go2 actuator. Its torque-speed curve, delayed commands, and friction require the Python execution path.
- Keep `Go2ActuatorCfg.joint_effort_limit=1e9` so the Go2 motor model limits torque without additional solver clipping. Read applied effort through `asset.actuators.applied_effort.torch`.
- URDF conversion enables articulation self-collision by default. Author matching `newton:selfCollisionEnabled` and `physxArticulation:enabledSelfCollisions` values on the USD articulation root, support the explicit `--no-self-collision` opt-out, and preserve collision filtering between adjacent links.
- Preserve the principal-axis rotations authored by the URDF importer and mirror `newton:velocityLimit` into `physxJoint:maxJointVelocity` when converting. Both velocity attributes use degrees/s for angular joints and distance/s for prismatic joints.
- Use `--viz none` to disable Isaac Lab visualizers. Use visualizer names such as `newton_gl`, `rerun`, `viser`, `kit`, and `newton_rtx`; do not revive the obsolete Isaac Lab `--headless` flag. Standalone MuJoCo deployment has its own `--headless` option.
- Recording is configured through `env_cfg.video_recorders`, not Gym `RecordVideo` or `render_mode="rgb_array"`. Automatic recording uses headless Newton GL with either supported physics backend and the project FFmpeg encoder adapter.
- Kit/RTX recording currently fails in the configured container with CUDA 719; the Newton/Kit path also reports a missing cubric bridge. Preserve the tested GL default and document the limit rather than claiming RTX validation.

## Development and validation

Inspect the working tree before pulling, editing, committing, or pushing. Preserve local adaptation changes when incorporating upstream commits. Back up or stash affected files before integration, resolve conflicts by combining the upstream behavior with Isaac Lab 3.0 contracts, and retain local validation artifacts.

Use four-space Python indentation and concise docstrings. Add a purpose and minimal command to runnable script headers. Keep `README.md` and the existing Chinese `README.zh-CN.md` synchronized when changing documented behavior or commands.

```bash
# Static validation
git diff --check
bash -n go2_rl_lab.sh
python -m compileall -q scripts source/go2_rl_lab source/rsl_rl

# Focused training smoke in the Isaac Lab 3.0 container
./go2_rl_lab.sh --train --task go2-flat --physics newton_mjwarp \
  --viz none --num_envs 32 --max_iterations 2 --logger tensorboard
```

For changes affecting shared task/runtime APIs, validate every registered task on both supported physics backends. For MoE-CTS, verify optimizer updates in both teacher and student paths. Validate the privileged terrain-height scan for mixed and plane configurations and retain production training defaults.

A passing train smoke requires actual rollout and optimizer updates, exit code 0, finite model tensors and scalar losses, changed trainable weights, and saved checkpoints. Exit code 0 alone is insufficient: Kit initialization failures can exit without training. Keep simulation and runner devices aligned after Isaac Lab resolves the runtime. Distributed device selection must use the process-visible CUDA index while retaining the actual rank for seeding and process-group membership. Keep GPU tests sequential on the current single-GPU host and distinguish smoke execution from training convergence or production-scale performance.

Validate reset and policy export behavior when those paths change. Decode recorded videos to check their frame count and visible content; load TorchScript exports for finite inference and check ONNX models. Verify that deployment metadata identifies `command/body/velocity` and that MuJoCo forward, backward, lateral, and yaw commands use the same observation interface. State any untested interactive visualization, large-scale, or multi-GPU paths explicitly in the local report.

Do not create commits or publish changes unless the user requests it. Before any authorized publication, enforce the local-only `tests/` rule above.
