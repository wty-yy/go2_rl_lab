<div align="center">
  <h1>Go2 RL Gym → Go2 RL Lab</h1>
  <p><strong>迁移说明与性能对比 · 2026-10-17</strong></p>
  <p>
    <a href="https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0"><img height="20" src="https://img.shields.io/badge/Isaac%20Lab-3.0-silver.svg" alt="Isaac Lab 3.0"></a>&nbsp;
    <a href="../LICENSE"><img height="20" src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  </p>
  <p><a href="from_gym_to_lab.md">🌎 English</a>&nbsp;&nbsp;·&nbsp;&nbsp;<strong>🇨🇳 中文</strong></p>
</div>

`go2_rl_gym` 正式更名为 `go2_rl_lab`，基于 **Isaac Lab 3.0 / Isaac Sim 6.2** 构建，物理后端切换为 **Newton MJWarp**。

旧版 Isaac Gym 实现保留在 [gym 分支](https://github.com/wty-yy/go2_rl_lab/tree/gym)，当前 Isaac Lab 实现使用 [lab3.0 分支](https://github.com/wty-yy/go2_rl_lab/tree/lab3.0)。

## 策略表现

单阶段训练、单策略部署，部署仅使用本体感知。

- **多地形越障**：20 cm 楼梯、30 度斜坡、30 cm 障碍
- **平地运动**：前进、后退速度最高均为 **4 m/s**
- **RoboGauge 最优得分**：0.6715 → **0.6929**
- **最大角速度指令**：2 → **3 rad/s**

## 训练性能

RTX 4090 · MoE-CTS · 关闭可视化 · 预热 10 个迭代后统计 50 个迭代。数据采集于 2026-10-17，早于 3.0 正式版发布。

| 并行环境数 | 框架 | 稳态显存（GiB） | 采样时间（s/iter） | 总迭代时间（s/iter） |
| ---: | --- | ---: | ---: | ---: |
| 8192 | Gym | 9.76 | 1.78 | 2.30 |
| 8192 | **Lab** | **9.13** | **1.72** | **2.65** |
| 16384 | Gym | 15.63 | 3.79 | 4.70 |
| 16384 | **Lab** | **17.80** | **2.68** | **4.37** |

## 算法与环境

- **并行与历史**：默认并行环境数 8192 → 16384，观测历史 5 → 10 帧
- **高程与增强**：使用 RayCaster 计算地形高程，引入左右对称数据增强（Gym 中没有该增强）
- **电机与地形**：参考宇树官方 Go2HV 电机模型，新增轮胎和小方柱地形
- **地形限速**：根据机器人当前位置所在的地形，动态调整速度指令上限

## 工程支持

统一入口 `go2_rl_lab.sh`，支持分布式训练、异常环境恢复、奖励异常裁剪以及 SwanLab / W&B 日志，已完成所有 Sim2Sim / Sim2Real 代码适配。
