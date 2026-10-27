<div align="center">
  <h1>Go2 RL Lab</h1>
  <p><strong>High-speed, agile velocity locomotion control for Unitree Go2 with Isaac Lab<br><a href="https://github.com/wty-yy/go2_rl_lab/tree/gym">(original go2_rl_gym branch)</a></strong></p>
  <p>
    <a href="https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0"><img height="20" src="https://img.shields.io/badge/Isaac%20Lab-3.0-silver.svg" alt="Isaac Lab 3.0"></a>&nbsp;
    <a href="https://swanlab.cn/@wty-yy/go2-rl-lab/v1/lxqfax/overview"><img height="20" src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg" alt="SwanLab"></a>&nbsp;
    <a href="LICENSE"><img height="20" src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  </p>
  <p><strong>🌎 English</strong>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="README.zh-CN.md">🇨🇳 中文</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://www.roboticsproceedings.org/rss22/p156.pdf">📄 Paper [RSS 2026]</a></p>
</div>

📢 `go2_rl_gym` has been renamed to `go2_rl_lab`, built on **Isaac Lab 3.0 / Isaac Sim 6.2** with **Newton MJWarp** as the physics backend and numerous improvements. See [from_gym_to_lab.md](docs/from_gym_to_lab.md) for the update notes

<table width="100%">
  <thead>
    <tr>
      <th width="33%" align="center"><div align="center">IsaacSim</div></th>
      <th width="33%" align="center"><div align="center">MuJoCo</div></th>
      <th width="33%" align="center"><div align="center">Real Robot</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="Kit demo" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_isaaclab_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="MuJoCo demo" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/python_mujoco_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="Real robot stair demo" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/forward_stairs_DJI_20261017144009_0111_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%" align="center"><div align="center">3.5 m/s Forward and 4.4 m/s Backward</div></th>
      <th width="50%" align="center"><div align="center">Running (Top View)</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="Hardware forward and backward walking" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/forward_backward_1x_then_0.5x.gif" /></td>
      <td align="center"><img width="100%" alt="Hardware fast running" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/overlook_fastrun_DJI_20261017143527_0110_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%" align="center"><div align="center">Slope</div></th>
      <th width="50%" align="center"><div align="center">Stairwell</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="Hardware slope walking" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/slope_DJI_20261017142100_0108_D_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="Hardware stairwell walking" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/stairwell_DJI_20261017145051_0117_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

## 📦 Installation

Requirements: Linux, an NVIDIA GPU, Python 3.12, Isaac Sim 6.2, and Isaac Lab 3.0

Two environment installation methods are available: Docker and uv

<details>
<summary><strong>(Method 1) Docker environment</strong></summary>

The configured Docker image is [wtyyy/isaaclab:3.0.0](https://hub.docker.com/r/wtyyy/isaaclab). Use the project RSL-RL package from `source/rsl_rl`, version `3.3.0`

> If the current host user is root, follow the [container startup instructions on Docker Hub](https://hub.docker.com/repository/docker/wtyyy/isaaclab/general).

```bash
# Host: start the Isaac Lab 3.0 runtime; run subsequent project commands inside the container
mkdir -p "${HOME}/isaaclab_docker/.cache/ov" "${HOME}/isaaclab_docker/.nvidia-omniverse"
docker run -it --name go2-rl-lab-3.0 \
  -e DEFAULT_UID="$(id -u)" \
  -e DEFAULT_GID="$(id -g)" \
  -e DISPLAY \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e "__NV_PRIME_RENDER_OFFLOAD=1" \
  -e "__GLX_VENDOR_LIBRARY_NAME=nvidia" \
  -v /etc/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json:ro \
  --device /dev/input \
  -e EXTRA_GIDS="$(getent group input | cut -d: -f3)" \
  --net=host \
  -v /path/to/Coding:/home/user/Coding \
  -v ${HOME}/isaaclab_docker/.cache/ov:/home/user/.cache/ov \
  -v ${HOME}/isaaclab_docker/.nvidia-omniverse:/home/user/.nvidia-omniverse \
  wtyyy/isaaclab:3.0.0 zsh
```

Replace `/path/to/Coding` with the actual directory containing your code on the host

</details>

<details>
<summary><strong>(Method 2) uv environment</strong></summary>

```bash
uv venv ~/go2-rl-lab-env --python 3.12 --seed --prompt go2-rl-lab  # Create the virtual environment under ~/go2-rl-lab-env
source ~/go2-rl-lab-env/bin/activate  # Activate the virtual environment
uv pip install "torch==2.11.0" "torchvision==0.26.0" --index-url "https://download.pytorch.org/whl/cu128"  # PyTorch
uv pip install --prerelease="allow" --index-strategy unsafe-best-match "isaaclab[isaacsim,all]==3.0.0" --extra-index-url https://pypi.nvidia.com  # Isaac Lab
```

</details>

After setting up the environment, install this repository:

```bash
# Clone the branch and fetch bundled Go2 USD assets
sudo apt update && sudo apt install -y git-lfs

cd /home/user/Coding  # Container path; replace with your actual path for host installation
git clone --branch lab3.0 https://github.com/wty-yy/go2_rl_lab.git
cd go2_rl_lab
git lfs pull

# Install go2_rl_lab and local rsl_rl in editable mode, add video dependencies, and generate VS Code settings
./go2_rl_lab.sh --install

# List the registered tasks
./go2_rl_lab.sh --list
```

## 🛠️ Usage Guide

### 🚀 1. Train

```bash
# Mixed-terrain MoE-CTS; Newton MuJoCo-Warp and no visualization by default
./go2_rl_lab.sh --train --task go2-mixed-moe-cts

# Resume a checkpoint in the same task
./go2_rl_lab.sh --train --task go2-mixed-moe-cts --resume --checkpoint /path/to/model_500.pt

# Distributed training: one worker per visible GPU; num_envs is the environment count per worker
CUDA_VISIBLE_DEVICES=0,1 ./go2_rl_lab.sh --train-distributed --task go2-mixed-moe-cts --num_envs 4096
```

Common CLI arguments:

- `--task TASK`: required task ID; no default; choices are `{go2-mixed-moe-cts, go2-mixed, go2-flat}`
- `--physics {newton_mjwarp,isaacsim_physx}`: physics backend; defaults to Newton MuJoCo-Warp
- `--viz NAME`: select a visualizer, such as `newton_gl`, `rerun`, or `kit`; visualization is disabled by default
- `--num_envs N`, `--max_iterations N`, `--seed N`: environment count, learning iterations, and seed; all training tasks default to 16384 environments
- `--experiment_name NAME`, `--run_name NAME`: experiment directory and run suffix
- `--resume`, `--load_run RUN`, `--checkpoint PATH`: checkpoint loading controls
- `--logger {tensorboard,wandb,swanlab}`, `--log_project_name NAME`: training logger and remote project; defaults to tensorboard. Install the selected optional client
- `--robogauge`, `--robogauge_port PORT`: asynchronous evaluation and server port

Use `./go2_rl_lab.sh --train -h` for all options. Checkpoints are saved to `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/model_<iteration>.pt`

#### 📋 Tasks

Use `./go2_rl_lab.sh --list` to list all registered tasks:

| Task ID | Algorithm | Terrain and commands |
| --- | --- | --- |
| `go2-mixed` | PPO | Mixed flat, wave, slope, stair, obstacle, tyre, and small-column terrains; terrain-dependent velocity limits and command curricula |
| `go2-mixed-moe-cts` | MoE-CTS | Same mixed environment; privileged teacher and eight-expert student with a 3:1 teacher/student environment split |
| `go2-flat` | PPO | Plane with the mixed reward set; fixed `v_x/v_y ∈ [-1, 1] m/s`, `yaw_rate ∈ [-2, 2] rad/s`, and no command-range curriculum |

#### 📊 RoboGauge Evaluation

[RoboGauge](https://github.com/wty-yy/RoboGauge) evaluates exported policies asynchronously during training; evaluation results are shown below

| Model | Score | Tracking | Safety | Quality | Level | Download |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| go2_mixed_moe_cts (Lab) | **0.6930** | **0.7004** | 0.7818 | **0.7818** | **8.40** | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_rl_lab/2026-10-13_15-35-09_mixed1.3.2resume_82.5k_0.6929) |
| go2_moe_cts (Gym) | 0.6713 | 0.6669 | **0.7857** | 0.7392 | 7.85 | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_moe_cts_137000_0.6713) |
| [CTS](https://arxiv.org/pdf/2405.10830) vanilla | 0.5786 | 0.5755 | 0.7066 | 0.6624 | 6.83 | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_cts_vanilla2_103.5k_0.5786) |
| [HIM](https://arxiv.org/abs/2312.11460) | 0.5379 | 0.5453 | 0.6476 | 0.6050 | 6.19 | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/blob/main/go2_him_21k_0.5379.pt) |
| [DreamWaQ](https://arxiv.org/abs/2301.10602) | 0.5054 | 0.5105 | 0.6149 | 0.5730 | 5.74 | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/blob/main/go2_dwaq_119.5k_0.5054.pt) |

<details>
<summary><strong>RoboGauge installation and startup</strong></summary>

Install the client in the training environment and start the server in the RoboGauge environment:

```bash
# RoboGauge repository: install client and full evaluation runtime
git clone https://github.com/wty-yy/RoboGauge.git
cd RoboGauge
pip install -e .
./robogauge.sh --install

# Terminal 1: start the server with the specified port and process count
./robogauge.sh --server --port 9973 --num-processes 30

# Terminal 2: Go2 RL Lab repository; use the matching server port
cd go2_rl_lab
./go2_rl_lab.sh --train --task go2-mixed-moe-cts \
  --robogauge --robogauge_port 9973
```

The client connects to `127.0.0.1`, so the server and training process must share a host or network namespace. Reduce `--num-processes` when resources are limited. Evaluation files are stored in the run directory as `jit_models/policy_jit_<iteration>.pt` and `robogauge_results/results_<iteration>.yaml`; scalar scores appear under `RoboGauge/` in the training logger

</details>

### 🎬 2. Play

```bash
# Load a checkpoint, export its policy, and visualize playback
./go2_rl_lab.sh --play --task go2-mixed-moe-cts --checkpoint /path/to/model_500.pt --viz rerun

# Override ports for concurrent Rerun sessions
./go2_rl_lab.sh --play --task go2-mixed-moe-cts --checkpoint /path/to/model_500.pt --viz rerun --rerun-web-port 9091 --rerun-grpc-port 9092
```

- Without `--checkpoint`, playback loads the latest matching checkpoint in the experiment directory
- Available visualizers are `newton_gl`, `rerun`, `viser`, `kit`, `newton_rtx`, and `none`; separate multiple visualizers with commas

#### 💾 Policy Export

Playback writes these files to the checkpoint directory's `exported/` subdirectory:

- `policy.pt`: stateful TorchScript actor accepting a 45-dimensional single frame and maintaining ten-frame history internally
- `policy.onnx`: actor accepting the flattened observation history
- `config.yaml`: observation, action, command, controller, and checkpoint metadata

#### 🎥 Visualization

<table width="100%">
  <thead>
    <tr>
      <th width="33%" align="center"><div align="center">Kit</div></th>
      <th width="33%" align="center"><div align="center">Rerun</div></th>
      <th width="33%" align="center"><div align="center">Newton</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="Kit visualization" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_isaaclab_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="Rerun visualization" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_rerun_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="Newton visualization" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_newton_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

### 🕹️ 3. Sim2Sim (MuJoCo)

`--deploy-mujoco` checks the current Python environment and automatically installs MuJoCo, gamepad, and video dependencies. When an available gamepad is detected, it provides command input

```bash
# Run an exported policy with a fixed velocity command
./go2_rl_lab.sh --deploy-mujoco --policy /path/to/exported/policy.pt --terrain stairs
```

Configure deployment in [`deploy/deploy_mujoco/config.yaml`](deploy/deploy_mujoco/config.yaml); available scenes are `flat`, `stairs`, `cross_slope`, `cross_stairs`, and `race_track`

<p align="center">
  <img width="50%" alt="MuJoCo deployment demo" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/python_mujoco_encoded.gif" />
</p>

### 🤖 4. Sim2Real

Hardware deployment uses C++. See [unitree_cpp_deploy - go2 branch](https://github.com/wty-yy/unitree_cpp_deploy/tree/go2) for code and instructions

## 🧩 Framework

```text
.
├── go2_rl_lab.sh                # Unified CLI
├── scripts/rsl_rl/              # Training, playback, and visualization
├── source/
│   ├── go2_rl_lab/go2_rl_lab/
│   │   ├── tasks/               # Task, environment, and algorithm configs
│   │   ├── cmd_utils/           # Velocity commands and curricula
│   │   ├── assets/              # Go2 assets and motor model
│   │   └── physics/             # Newton physics extensions
│   └── rsl_rl/rsl_rl/           # PPO, MoE-CTS, and policy export
└── deploy/deploy_mujoco/        # MuJoCo inference and scenes
```

## 🎉 Acknowledgements

- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab): project foundation and robot training structure
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): simulation and environment framework
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl): reinforcement learning infrastructure
- [Go2 RL Gym](https://github.com/wty-yy/go2_rl_lab/tree/gym): velocity baseline and MuJoCo deployment reference
- [Go2 RL RobotLab](https://github.com/wertyuilife2/go2_rl_robotlab): locomotion configuration reference
- [RoboGauge](https://github.com/wty-yy/RoboGauge): policy evaluation
- [MuJoCo](https://github.com/google-deepmind/mujoco): standalone simulation

## 📄 License

Project-owned code uses the [MIT License](LICENSE), copyright 2026 Tianyang Wu. Third-party code retains its original terms:

- Unitree RL Lab-derived code uses Apache 2.0
- Isaac Lab and original RSL-RL code use BSD 3-Clause
- MuJoCo deployment retains the [Go2 RL Gym notice](docs/licenses/go2-rl-gym-license.txt)
- Go2 USD assets come from the [Unitree model dataset](https://huggingface.co/datasets/unitreerobotics/unitree_model)

See the consolidated [LICENSE](LICENSE), [vendored RSL-RL license](source/rsl_rl/LICENSE), and [license notices](docs/licenses/)

## 📚 Citation

If you find our work helpful, please cite:

```bibtex
@INPROCEEDINGS{WuT-RSS-26,
    AUTHOR    = {Tianyang Wu AND Hanwei Guo AND Yuhang Wang AND Junshu Yang AND Xinyang Sui AND Jiayi Xie AND Xingyu Chen AND Zeyang Liu AND Xuguang Lan},
    TITLE     = {{Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion}},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2026},
    ADDRESS   = {Sydney, Australia},
    MONTH     = {July},
    DOI       = {10.15607/RSS.2026.XXII.156}
}
```
