<div align="center">
  <h1>Go2 RL Gym → Go2 RL Lab</h1>
  <p><strong>Migration notes and performance comparison · 2026-10-17</strong></p>
  <p>
    <a href="https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0"><img height="20" src="https://img.shields.io/badge/Isaac%20Lab-3.0-silver.svg" alt="Isaac Lab 3.0"></a>&nbsp;
    <a href="../LICENSE"><img height="20" src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  </p>
  <p><strong>🌎 English</strong>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="from_gym_to_lab.zh-CN.md">🇨🇳 中文</a></p>
</div>

`go2_rl_gym` is now `go2_rl_lab`, built on **Isaac Lab 3.0 / Isaac Sim 6.2**, with **Newton MJWarp** as the physics backend.

The original Isaac Gym implementation is preserved on the [gym branch](https://github.com/wty-yy/go2_rl_lab/tree/gym); the current Isaac Lab implementation uses the [lab3.0 branch](https://github.com/wty-yy/go2_rl_lab/tree/lab3.0).

## Policy performance

Single-stage training and single-policy deployment, using only proprioceptive sensing at deployment.

- **Terrain traversal**: 20 cm stairs, 30-degree slopes, and 30 cm obstacles
- **Flat-ground locomotion**: up to **4 m/s** in both forward and backward directions
- **Best RoboGauge score**: 0.6715 → **0.6929**
- **Maximum angular velocity command**: 2 → **3 rad/s**

## Training performance

RTX 4090 · MoE-CTS · Visualization disabled · 50 measured iterations after 10 warm-up iterations. Measurements were recorded on 2026-10-17, before the 3.0 release.

| Parallel environments | Framework | Steady-state GPU memory (GiB) | Rollout time (s/iter) | Total iteration time (s/iter) |
| ---: | --- | ---: | ---: | ---: |
| 8192 | Gym | 9.76 | 1.78 | 2.30 |
| 8192 | **Lab** | **9.13** | **1.72** | **2.65** |
| 16384 | Gym | 15.63 | 3.79 | 4.70 |
| 16384 | **Lab** | **17.80** | **2.68** | **4.37** |

## Algorithm and environment

- **Parallelism and history**: increase the default environment count from 8192 to 16384 and observation history from 5 to 10 frames
- **Heights and augmentation**: use RayCaster for terrain-height queries and introduce left-right symmetry augmentation, absent from the Gym version
- **Motors and terrains**: use Unitree's official Go2HV motor model as a reference; add tyre and small-column terrains
- **Terrain-dependent limits**: dynamically adjust velocity-command limits according to the terrain at the robot's current position

## Engineering support

The unified `go2_rl_lab.sh` entry point supports distributed training, recovery of invalid environments, reward clipping, and SwanLab / W&B logging. All Sim2Sim / Sim2Real code has been adapted.
