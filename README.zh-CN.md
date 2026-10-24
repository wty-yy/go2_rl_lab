<div align="center">
  <h1>Go2 RL Lab</h1>
  <p><strong>基于 Isaac Lab 的 Unitree Go2 高速灵巧速度运动控制<br><a href="https://github.com/wty-yy/go2_rl_lab/tree/gym">（原 go2_rl_gym 分支）</a></strong></p>
  <p>
    <a href="https://isaac-sim.github.io/IsaacLab/"><img height="20" src="https://img.shields.io/badge/Isaac%20Lab-3.0%20EA-silver.svg" alt="Isaac Lab 3.0 EA"></a>&nbsp;
    <a href="https://swanlab.cn/@wty-yy/go2-rl-lab/v1/lxqfax/overview"><img height="20" src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg" alt="SwanLab"></a>&nbsp;
    <a href="LICENSE"><img height="20" src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  </p>
  <p><a href="README.md">🌎 English</a>&nbsp;&nbsp;·&nbsp;&nbsp;<strong>🇨🇳 中文</strong>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://www.roboticsproceedings.org/rss22/p156.pdf">📄 Paper [RSS 2026]</a></p>
</div>

📢 `go2_rl_gym` 正式更名为 `go2_rl_lab`，基于 **Isaac Lab 3.0 EA / Isaac Sim 6.1** 构建，物理后端切换为 **Newton MJWarp**，并做大量改进，更新说明见 [from_gym_to_lab.zh-CN.md](docs/from_gym_to_lab.zh-CN.md)

<table width="100%">
  <thead>
    <tr>
      <th width="33%" align="center"><div align="center">IsaacSim</div></th>
      <th width="33%" align="center"><div align="center">MuJoCo</div></th>
      <th width="33%" align="center"><div align="center">真机</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="Kit 演示" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_isaaclab_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="MuJoCo 演示" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/python_mujoco_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="真机楼梯演示" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/forward_stairs_DJI_20261017144009_0111_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%" align="center"><div align="center">3.5m/s前进与4.4m/s后退</div></th>
      <th width="50%" align="center"><div align="center">奔跑俯视</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="真机前进与后退" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/forward_backward_1x_then_0.5x.gif" /></td>
      <td align="center"><img width="100%" alt="真机快速奔跑" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/overlook_fastrun_DJI_20261017143527_0110_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%" align="center"><div align="center">斜坡</div></th>
      <th width="50%" align="center"><div align="center">楼梯间</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="真机斜坡行走" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/slope_DJI_20261017142100_0108_D_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="真机楼梯间行走" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/stairwell_DJI_20261017145051_0117_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

## 📦 安装

环境要求：Linux、NVIDIA GPU、Python 3.12、Isaac Sim 6.1 和 Isaac Lab 3.0 EA

环境安装包含两种方式：Docker容器和uv环境安装

<details>
<summary><strong>（方法一）Docker 容器环境</strong></summary>

已配置的 Docker 镜像为 [wtyyy/isaaclab:3.0.0](https://hub.docker.com/r/wtyyy/isaaclab)。使用 `source/rsl_rl` 中的项目版 RSL-RL `3.3.0`

> 如果当前宿主机用户为 root，请参考 [Docker Hub 中的容器启动方法](https://hub.docker.com/repository/docker/wtyyy/isaaclab/general)。

```bash
# 宿主机：启动 Isaac Lab 3.0 运行环境，后续项目命令均在容器内执行
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

将 `/path/to/Coding` 替换为宿主机上你的实际代码存放目录

</details>

<details>
<summary><strong>（方法二）uv 环境安装</strong></summary>

```bash
uv venv ~/go2-rl-lab-env --python 3.12 --seed --prompt go2-rl-lab  # ~/go2-rl-lab-env 下创建虚拟环境
source ~/go2-rl-lab-env/bin/activate  # 激活虚拟环境
uv pip install "torch==2.11.0" "torchvision==0.26.0" --index-url "https://download.pytorch.org/whl/cu128"  # pytorch
uv pip install --prerelease="allow" --index-strategy unsafe-best-match "isaaclab[isaacsim,all]==3.0.0rc1" --extra-index-url https://pypi.nvidia.com  # isaaclab
```

</details>

完成环境安装后安装本仓库

```bash
# 克隆该分支并获取内置 Go2 USD 资源
sudo apt update && sudo apt install -y git-lfs

cd /home/user/Coding  # 容器路径；宿主机安装时替换为实际路径
git clone --branch lab https://github.com/wty-yy/go2_rl_lab.git
cd go2_rl_lab
git lfs pull

# 安装可编辑的 go2_rl_lab、本地 rsl_rl 及录像依赖，并自动生成 VS Code 配置
./go2_rl_lab.sh --install

# 列出已注册任务
./go2_rl_lab.sh --list
```

## 🛠️ 使用指南

### 🚀 1. 训练

```bash
# 混合地形 MoE-CTS；默认使用 Newton MuJoCo-Warp，关闭可视化
./go2_rl_lab.sh --train --task go2-mixed-moe-cts

# 从同一任务的检查点恢复训练
./go2_rl_lab.sh --train --task go2-mixed-moe-cts --resume --checkpoint /path/to/model_500.pt

# 分布式训练：每块可见 GPU 启动一个 worker，num_envs在分布式中为每个 worker 的环境数量
CUDA_VISIBLE_DEVICES=0,1 ./go2_rl_lab.sh --train-distributed --task go2-mixed-moe-cts --num_envs 4096
```

常用cli参数：

- `--task TASK`：必填任务 ID，无默认值，可选 `{go2-mixed-moe-cts, go2-mixed, go2-flat}`
- `--physics {newton_mjwarp,isaacsim_physx}`：物理后端，默认 Newton MuJoCo-Warp
- `--viz NAME`：选择可视化器，例如 `newton_gl`、`rerun` 或 `kit`；默认关闭可视化
- `--num_envs N`、`--max_iterations N`、`--seed N`：环境数量、学习迭代次数和随机种子；全部训练任务默认使用 16384 个环境
- `--experiment_name NAME`、`--run_name NAME`：实验目录和运行名称后缀
- `--resume`、`--load_run RUN`、`--checkpoint PATH`：检查点加载选项
- `--logger {tensorboard,wandb,swanlab}`、`--log_project_name NAME`：训练日志器和远端项目，需安装对应的可选客户端（默认为tensorboard）
- `--robogauge`、`--robogauge_port PORT`：异步评估和服务端端口

使用 `./go2_rl_lab.sh --train -h` 查看全部选项。检查点路径为 `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/model_<iteration>.pt`

#### 📋 任务

通过 `./go2_rl_lab.sh --list` 可列出所有注册任务：

| 任务 ID | 算法 | 地形与指令 |
| --- | --- | --- |
| `go2-mixed` | PPO | 平地、波浪、斜坡、楼梯、障碍物、轮胎和小柱体混合地形；按地形限制速度，使用指令课程 |
| `go2-mixed-moe-cts` | MoE-CTS | 同一混合地形环境；特权教师和八专家学生，教师与学生环境按 3:1 划分 |
| `go2-flat` | PPO | 平地及完整 mixed 奖励；固定 `v_x/v_y ∈ [-1, 1] m/s`、`yaw_rate ∈ [-2, 2] rad/s`，无指令范围课程 |

#### 📊 RoboGauge 评估

[RoboGauge](https://github.com/wty-yy/RoboGauge) 可在训练期间异步评估导出策略，评估结果如下表所示

| Model | Score | Tracking | Safety | Quality | Level | Download |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| go2_mixed_moe_cts (Lab) | **0.6930** | **0.7004** | 0.7818 | **0.7818** | **8.40** | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_rl_lab/2026-10-13_15-35-09_mixed1.3.2resume_82.5k_0.6929) |
| go2_moe_cts (Gym) | 0.6713 | 0.6669 | **0.7857** | 0.7392 | 7.85 | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_moe_cts_137000_0.6713) |
| [CTS](https://arxiv.org/pdf/2405.10830) vanilla | 0.5786 | 0.5755 | 0.7066 | 0.6624 | 6.83 | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/tree/main/go2_cts_vanilla2_103.5k_0.5786) |
| [HIM](https://arxiv.org/abs/2312.11460) | 0.5379 | 0.5453 | 0.6476 | 0.6050 | 6.19 | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/blob/main/go2_him_21k_0.5379.pt) |
| [DreamWaQ](https://arxiv.org/abs/2301.10602) | 0.5054 | 0.5105 | 0.6149 | 0.5730 | 5.74 | [ckpt](https://huggingface.co/wty-yy/go2_rl_gym_data/blob/main/go2_dwaq_119.5k_0.5054.pt) |

<details>
<summary><strong>RoboGauge 安装与启动</strong></summary>

在训练环境安装客户端，并在 RoboGauge 环境启动服务端：

```bash
# RoboGauge 仓库：安装客户端和完整评估运行环境
git clone https://github.com/wty-yy/RoboGauge.git
cd RoboGauge
pip install -e .
./robogauge.sh --install

# 终端1，启动服务端，指定端口和进程数
./robogauge.sh --server --port 9973 --num-processes 30

# 终端2，Go2 RL Lab 仓库，使用与服务端一致的端口
cd go2_rl_lab
./go2_rl_lab.sh --train --task go2-mixed-moe-cts \
  --robogauge --robogauge_port 9973
```

客户端连接 `127.0.0.1`，因此服务端和训练进程需共享主机或网络命名空间。资源有限时可减小 `--num-processes`。评估文件保存在运行目录的 `jit_models/policy_jit_<iteration>.pt` 和 `robogauge_results/results_<iteration>.yaml`，标量分数记录在训练日志的 `RoboGauge/` 下

</details>

### 🎬 2. 回放

```bash
# 加载检查点、导出策略并可视化回放
./go2_rl_lab.sh --play --task go2-mixed-moe-cts --checkpoint /path/to/model_500.pt --viz rerun

# 为并发 Rerun 会话指定独立端口
./go2_rl_lab.sh --play --task go2-mixed-moe-cts --checkpoint /path/to/model_500.pt --viz rerun --rerun-web-port 9091 --rerun-grpc-port 9092
```

- 未指定 `--checkpoint` 时，加载实验目录中最新的匹配检查点
- 可视化选项包括 `newton_gl`、`rerun`、`viser`、`kit`、`newton_rtx` 和 `none`（多个可视化器使用逗号分隔）

#### 💾 策略导出

回放在所选检查点目录的 `exported/` 子目录中生成：

- `policy.pt`：接收 45 维单帧观测、在内部维护十帧历史的有状态 TorchScript actor
- `policy.onnx`：接收展平观测历史的 actor
- `config.yaml`：观测、动作、指令、控制器和检查点元数据

#### 🎥 可视化效果

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
      <td align="center"><img width="100%" alt="Kit 可视化效果" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_isaaclab_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="Rerun 可视化效果" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_rerun_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="Newton 可视化效果" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_newton_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

### 🕹️ 3. Sim2Sim（MuJoCo）

`--deploy-mujoco` 会检查当前 Python 环境，自动安装 MuJoCo、手柄和录像依赖。若检测到可用手柄，将使用手柄输入指令

```bash
# 使用固定速度指令运行导出策略
./go2_rl_lab.sh --deploy-mujoco --policy /path/to/exported/policy.pt --terrain stairs
```

[`deploy/deploy_mujoco/config.yaml`](deploy/deploy_mujoco/config.yaml) 中进行配置，可选场景：`flat`、`stairs`、`cross_slope`、`cross_stairs` 和 `race_track`

<p align="center">
  <img width="50%" alt="MuJoCo 部署演示" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/python_mujoco_encoded.gif" />
</p>

### 🤖 4. Sim2Real

真机部署使用 C++，代码与使用说明见 [unitree_cpp_deploy - go2 branch](https://github.com/wty-yy/unitree_cpp_deploy/tree/go2)

## 🧩 框架

```text
.
├── go2_rl_lab.sh                # 统一命令入口
├── scripts/rsl_rl/              # 训练、回放与可视化
├── source/
│   ├── go2_rl_lab/go2_rl_lab/
│   │   ├── tasks/               # 任务、环境与算法配置
│   │   ├── cmd_utils/           # 速度指令与课程
│   │   ├── assets/              # Go2 资源与电机模型
│   │   └── physics/             # Newton 物理扩展
│   └── rsl_rl/rsl_rl/           # PPO、MoE-CTS 与策略导出
└── deploy/deploy_mujoco/        # MuJoCo 推理与场景
```

## 🎉 致谢

- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)：项目基础和机器人训练结构
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)：仿真与环境框架
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)：强化学习基础设施
- [Go2 RL Gym](https://github.com/wty-yy/go2_rl_lab/tree/gym)：速度基线和 MuJoCo 部署参考
- [Go2 RL RobotLab](https://github.com/wertyuilife2/go2_rl_robotlab)：运动控制配置参考
- [RoboGauge](https://github.com/wty-yy/RoboGauge)：策略评估
- [MuJoCo](https://github.com/google-deepmind/mujoco)：独立仿真

## 📄 许可证

项目自有代码采用 [MIT License](LICENSE)，版权归 2026 Tianyang Wu。第三方代码保留原有条款：

- Unitree RL Lab 衍生代码采用 Apache 2.0
- Isaac Lab 及原始 RSL-RL 代码采用 BSD 3-Clause
- MuJoCo 部署保留 [Go2 RL Gym 声明](docs/licenses/go2-rl-gym-license.txt)
- Go2 USD 资源来自 [Unitree 模型数据集](https://huggingface.co/datasets/unitreerobotics/unitree_model)

完整条款见合并后的 [LICENSE](LICENSE)、[内置 RSL-RL 许可证](source/rsl_rl/LICENSE) 和 [许可证声明](docs/licenses/)

## 📚 引用

如果我们的工作对你有所帮助，请引用：

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
