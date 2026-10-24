# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified for the Go2-only Go2 RL Lab project.

"""Play and export a trained Go2 RSL-RL policy.

Quick Start:
    ./go2_rl_lab.sh --play --task go2-flat --checkpoint path/to/model.pt --viz newton_gl

Full Command:
    ./go2_rl_lab.sh --play --task go2-flat --checkpoint path/to/model.pt \
        --num_envs 4 --physics newton_mjwarp --video --video_length 200

Notes:
    Requires the configured Isaac Lab Python environment.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

# local imports
import cli_args  # isort: skip
import port_args  # isort: skip
import runtime_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained Go2 RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Task ID to play.")
parser.add_argument(
    "--physics",
    type=str,
    choices=("isaacsim_physx", "newton_mjwarp"),
    default=None,
    help="Physics preset. Defaults to Newton MuJoCo-Warp.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append Rerun port arguments
port_args.add_rerun_port_args(parser)
# append runtime launcher cli args
add_launcher_args(parser)
args_cli = parser.parse_args()
port_args.validate_rerun_port_args(parser, args_cli)
# Kit capture needs RTX cameras; automatic GL capture uses its own renderer.
if args_cli.video and "kit" in (args_cli.visualizer or []):
    args_cli.enable_cameras = True

runtime_args.bootstrap_runtime(args_cli)

import os
import time
import torch

import video_utils

if args_cli.video:
    video_utils.configure_video_writer()

import go2_rl_lab.tasks  # ruff: ignore[unused-import]
import gymnasium as gym
from go2_rl_lab.utils.parser_cfg import parse_env_cfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_newton.physics import NewtonCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.entrypoints.common import apply_video_recording

from rsl_rl.base.base_runner import BaseRunner
from rsl_rl.base.base_runner_config import RslRlBaseRunnerCfg
from rsl_rl.base.env import RslRlVecEnvWrapper
from rsl_rl.utils import export_model_config, resolve_callable


def _configure_play_visualizers(env_cfg) -> None:
    """Disable playback charts and select the project's Rerun scene layout."""
    from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

    if env_cfg.sim.default_visualizer_cfg is None:
        env_cfg.sim.default_visualizer_cfg = VisualizerCfg(enable_live_plots=False)
    else:
        env_cfg.sim.default_visualizer_cfg.enable_live_plots = False
    cfgs = env_cfg.sim.visualizer_cfgs
    cfgs = list(cfgs) if isinstance(cfgs, list) else ([] if cfgs is None else [cfgs])
    for cfg in cfgs:
        cfg.enable_live_plots = False
    rerun_cfg = next((cfg for cfg in cfgs if cfg.visualizer_type == "rerun"), None)
    if "rerun" in (args_cli.visualizer or []) or rerun_cfg is not None:
        import rerun as rr
        import rerun.blueprint as rrb
        from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg

        class PlaybackRerunVisualizer(RerunVisualizer):
            """Use the standard Rerun backend with the project's playback layout."""

            def _apply_camera_pose(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
                """Update the playback camera and display a single 3D scene view."""
                super()._apply_camera_pose(pose)
                if self._viewer is None or self._streaming_view_active:
                    return
                rr.send_blueprint(
                    rrb.Blueprint(
                        rrb.Spatial3DView(
                            name="3D View",
                            origin="/",
                            line_grid=True,
                            eye_controls=rrb.EyeControls3D(position=pose[0], look_target=pose[1]),
                        ),
                        rrb.TimePanel(state="hidden"),
                        collapse_panels=True,
                    )
                )

        if rerun_cfg is None:
            rerun_cfg = RerunVisualizerCfg(class_type=PlaybackRerunVisualizer, enable_live_plots=False)
            cfgs.append(rerun_cfg)
        else:
            rerun_cfg.class_type = PlaybackRerunVisualizer
        rerun_cfg.keep_scalar_history = False
    env_cfg.sim.visualizer_cfgs = cfgs


def main() -> None:
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
        physics=args_cli.physics,
    )
    env_cfg.scene.robot.spawn.visuals_enabled = True
    if isinstance(env_cfg.sim.physics, NewtonCfg):
        env_cfg.sim.physics.load_visual_shapes = True
    port_args.apply_rerun_port_args(env_cfg, args_cli)
    _configure_play_visualizers(env_cfg)
    agent_cfg: RslRlBaseRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    args_cli.video_interval = 0
    video_utils.configure_video_capture(env_cfg, args_cli)
    apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=resume_path)
    launcher_args = vars(args_cli).copy()
    launcher_args["physics"] = None
    with launch_simulation(env_cfg, launcher_args):
        agent_cfg.device = env_cfg.sim.device
        play(env_cfg, agent_cfg, resume_path)


def play(env_cfg, agent_cfg, resume_path: str) -> None:
    """Load, export, and step the policy inside the selected simulation runtime."""
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
            clip_rewards=agent_cfg.clip_rewards,
        )

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner_class = resolve_callable(agent_cfg.class_name)
        runner: BaseRunner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # extract the neural network module
        policy_nn = runner.alg.policy

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        else:
            normalizer = None

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        current_obs = env.get_observations()
        dt = env.unwrapped.step_dt
        runner.export_policy_as_jit(
            path=export_model_dir,
            normalizer=normalizer,
            num_single_obs=int(current_obs["single_obs"].shape[-1]),
            num_actions=env.num_actions,
            filename="policy.pt",
        )
        runner.export_policy_as_onnx(
            path=export_model_dir,
            normalizer=normalizer,
            filename="policy.onnx",
        )

        config_path = export_model_config(
            path=export_model_dir,
            checkpoint_path=resume_path,
            checkpoint_iteration=runner.current_learning_iteration,
            task=args_cli.task,
            experiment_name=agent_cfg.experiment_name,
            num_single_obs=int(current_obs["single_obs"].shape[-1]),
            num_actor_obs=int(current_obs["policy"].shape[-1]),
            num_actions=env.num_actions,
            control_dt=dt,
        )
        print(f"[INFO]: Exported model configuration to: {config_path}")

        # reset environment
        obs = env.get_observations()
        policy_nn.reset()
        timestep = 0
        # simulate environment
        while env.unwrapped.sim.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
            timestep += 1

            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        runtime_args.shutdown_runtime()
