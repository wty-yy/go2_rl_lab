# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified for the Go2-only Go2 RL Lab project.

"""Train Go2 policies with the vendored RSL-RL runners.

Quick Start:
    ./go2_rl_lab.sh --train --task go2-flat

Full Command:
    ./go2_rl_lab.sh --train --task go2-mixed --num_envs 4096 \
        --max_iterations 50000 --physics newton_mjwarp

Notes:
    Requires the configured Isaac Lab Python environment.
    Hydra accepts env.* and agent.* overrides after the CLI arguments.
"""

import argparse
import signal
import sys

from isaaclab.app import add_launcher_args, launch_simulation

# local imports
import cli_args  # isort: skip
import port_args  # isort: skip
import runtime_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Task ID to train.")
parser.add_argument(
    "--physics",
    type=str,
    choices=("isaacsim_physx", "newton_mjwarp"),
    default=None,
    help="Physics preset. Defaults to Newton MuJoCo-Warp.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append Rerun port arguments
port_args.add_rerun_port_args(parser)
# append runtime launcher cli args
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
port_args.validate_rerun_port_args(parser, args_cli)

# Translate the conventional argparse form into Isaac Lab's Hydra preset selector.
if args_cli.physics is not None:
    if any(arg.lstrip("-").startswith("physics=") for arg in hydra_args):
        parser.error("Use either '--physics NAME' or 'physics=NAME', not both.")
    hydra_args.append(f"physics={args_cli.physics}")

# Kit capture needs RTX cameras; automatic GL capture uses its own renderer.
if args_cli.video and "kit" in (args_cli.visualizer or []):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0], *hydra_args]

runtime_args.bootstrap_runtime(args_cli, hydra_args)

import inspect
import os
import shutil
import torch
from datetime import datetime

import video_utils

if args_cli.video:
    video_utils.configure_video_writer()

import go2_rl_lab.tasks  # ruff: ignore[unused-import]
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.entrypoints.common import apply_video_recording

from rsl_rl.base.base_runner import BaseRunner
from rsl_rl.base.base_runner_config import RslRlBaseRunnerCfg
from rsl_rl.base.env import RslRlVecEnvWrapper
from rsl_rl.utils import resolve_callable

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _exit_on_sigint(signum: int, _frame: object) -> None:
    """Exit a distributed worker immediately on Ctrl-C so NCCL teardown cannot hang."""
    print(f"[INFO] Received signal {signum}; terminating worker immediately.", file=sys.stderr, flush=True)
    os._exit(128 + signum)


def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Robot meshes are unused by pure headless training. Disable them before scene cloning,
    # while preserving them for video recording or an active visualizer.
    visualizer_enabled = any(name != "none" for name in (args_cli.visualizer or []))
    env_cfg.scene.robot.spawn.visuals_enabled = bool(args_cli.video or visualizer_enabled)
    if isinstance(env_cfg.sim.physics, NewtonCfg):
        env_cfg.sim.physics.load_visual_shapes = env_cfg.scene.robot.spawn.visuals_enabled
    port_args.apply_rerun_port_args(env_cfg, args_cli)

    # multi-gpu training configuration
    if args_cli.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible_gpus = torch.cuda.device_count()
        env_cfg.sim.device = f"cuda:{local_rank}" if local_rank < visible_gpus else "cuda:0"
        agent_cfg.device = env_cfg.sim.device

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

        # Preserve the upstream immediate Ctrl-C exit for distributed workers.
        signal.signal(signal.SIGINT, _exit_on_sigint)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Recording is owned by the environment and configured before runtime selection.
    video_utils.configure_video_capture(env_cfg, args_cli)
    apply_video_recording(env_cfg, log_dir, args_cli)
    launcher_args = vars(args_cli).copy()
    launcher_args["physics"] = None  # Already composed through the task preset.

    with launch_simulation(env_cfg, launcher_args):
        agent_cfg.device = env_cfg.sim.device
        train(env_cfg, agent_cfg, log_dir)


def train(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg, log_dir: str) -> None:
    """Create the selected environment and run training inside its simulation runtime."""
    log_root_path = os.path.dirname(log_dir)
    env = gym.make(args_cli.task, cfg=env_cfg)

    completed = False
    try:
        # save resume path before creating a new log_dir
        resume_path = None
        if agent_cfg.resume:
            if args_cli.checkpoint is not None:
                resume_path = retrieve_file_path(args_cli.checkpoint)
            else:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
            clip_rewards=agent_cfg.clip_rewards,
        )

        agent_cfg_dict = agent_cfg.to_dict()

        # create runner from rsl-rl
        runner_class = resolve_callable(agent_cfg.class_name)
        runner: BaseRunner = runner_class(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
        # write git state to logs
        runner.add_git_repo_to_log(__file__)
        # load the checkpoint
        if resume_path is not None:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # load previously trained model
            runner.load(resume_path)
            # iteration-based curriculum continue instead of restarting.
            completed_iterations = runner.current_learning_iteration + 1
            env.unwrapped.common_step_counter = completed_iterations * agent_cfg.num_steps_per_env
            print(
                "[INFO]: Restored environment curriculum progress: "
                f"iteration={completed_iterations}, common_step_counter={env.unwrapped.common_step_counter}"
            )

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg_dict)
        # copy the environment configuration file to the log directory
        shutil.copy(
            inspect.getfile(env_cfg.__class__),
            os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
        )

        # run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        completed = True

    finally:
        env.close()
        # Tear down NCCL only after every rank completed normally.
        if completed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main(*resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point", overrides=hydra_args))
    finally:
        runtime_args.shutdown_runtime()
