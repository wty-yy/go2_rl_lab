"""Custom domain-randomization events for the Go2 mixed-terrain task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the selected root poses and velocities with uniform noise."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=env.device)[env_ids or slice(None)]

    root_states = asset.data.default_root_state.torch[env_ids].clone()
    pose_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
        device=asset.device,
    )
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0],
        pose_ranges[:, 1],
        (len(env_ids), 6),
        device=asset.device,
    )
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_samples[:, 0:3]
    orientation_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3],
        pose_samples[:, 4],
        pose_samples[:, 5],
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
        device=asset.device,
    )
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0],
        velocity_ranges[:, 1],
        (len(env_ids), 6),
        device=asset.device,
    )
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim_index(
        root_pose=torch.cat([positions, orientations], dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def randomize_action_joint_pos_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    action_term_name: str,
    offset_range: tuple[float, float],
):
    """Randomize the motor zero offset of a joint-position action term."""
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=env.device)[env_ids or slice(None)]
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    action_term = env.action_manager.get_term(action_term_name)
    if not hasattr(action_term, "_offset") or not isinstance(action_term._offset, torch.Tensor):
        raise TypeError(
            f"Action term '{action_term_name}' does not expose a tensor '_offset', "
            "so it cannot be used for motor zero-offset randomization."
        )

    cache_name = f"_default_action_offset_{action_term_name}"
    default_offset = getattr(env, cache_name, None)
    if default_offset is None or default_offset.shape != action_term._offset.shape:
        default_offset = action_term._offset.clone()
        setattr(env, cache_name, default_offset)

    offset_noise = math_utils.sample_uniform(
        offset_range[0],
        offset_range[1],
        (len(env_ids), action_term._offset.shape[1]),
        device=env.device,
    )
    action_term._offset[env_ids] = default_offset[env_ids] + offset_noise
