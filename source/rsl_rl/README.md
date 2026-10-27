# RSL-RL for Go2 Velocity Locomotion

Vendored RSL-RL 3.3.0 with feed-forward PPO, MoE-CTS, symmetry augmentation, policy export, and asynchronous RoboGauge evaluation. Install this package from the checkout to retain the project extensions.

## Usage

Run commands from the repository root inside the configured Isaac Lab 3.0 environment.

```bash
# Install the vendored learning package
python -m pip install --no-deps -e source/rsl_rl

# Train PPO or MoE-CTS
./go2_rl_lab.sh --train --task go2-mixed --viz none
./go2_rl_lab.sh --train --task go2-mixed-moe-cts --viz none

# Evaluate checkpoints with an existing RoboGauge server
./go2_rl_lab.sh --train --task go2-mixed --viz none --robogauge --robogauge_port 9973
```

RoboGauge is disabled by default. Enabled PPO and MoE-CTS runners submit a TorchScript policy every 500 iterations, collect completed scores at later checkpoint saves, write metrics under `RoboGauge/`, and retain full results in the training run's `robogauge_results/` directory. Configure evaluator settings through `RoboGaugeEvaluatorCfg` in the task's runner configuration.

The default task prefix is `go2_lab_1.3.1`, matching command observation scales `[0.5, 1.0, 1.0]`; the RoboGauge server must register the corresponding versioned task family.

## Implementation

| Path | Responsibility |
| --- | --- |
| `rsl_rl/base/` | Algorithm interface, runner lifecycle, vector environment contract, shared rollout fields, MLPs, and observation normalization |
| `rsl_rl/ppo/` | Feed-forward actor-critic PPO, symmetry augmentation, rollout batching, and policy exporters |
| `rsl_rl/moe_cts/` | Concurrent privileged teacher/student PPO, student mixture-of-experts encoder, latent matching, segmented rollout batches, and student exporters |
| `rsl_rl/utils/` | Logging, callable/observation resolution, deployment metadata, and RoboGauge integration |

Each algorithm owns its configuration, runner, policy, storage, and exporter. Gym task configurations select runtime classes with fully qualified names such as `rsl_rl.ppo:PPORunner` and `rsl_rl.moe_cts:MoeCtsRunner`. Shared infrastructure contains the rollout loop, distributed gradient synchronization, checkpointing, and logger lifecycle.

PPO and MoE-CTS mini-batches contain eight fields: observations, actions, target values, advantages, returns, previous action log probabilities, previous action means, and previous action standard deviations. Symmetry augmentation preserves this interface. MoE-CTS keeps teacher and student samples in separate contiguous segments.

TorchScript policies accept one observation frame and maintain the feature-major history internally. ONNX policies accept the stacked policy observations. Playback also writes deployment metadata and the original training configurations through `export_model_config`. Metadata identifies the body velocity command as `command/body/velocity`, lists its `lin_vel_x`, `lin_vel_y`, and `ang_vel_z` elements, and records the command observation scale when available in the saved configuration.

## Attribution

Based on [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl). See [LICENSE](LICENSE), [CONTRIBUTORS.md](CONTRIBUTORS.md), [CITATION.cff](CITATION.cff), and [licenses/](licenses/) for attribution, citation, and dependency notices.
