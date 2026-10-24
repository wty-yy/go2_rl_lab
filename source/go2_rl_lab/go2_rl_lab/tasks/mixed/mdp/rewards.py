"""Reward terms used by the Go2 mixed-terrain task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from go2_rl_lab.utils.height_scan import get_cropped_ray_hit_heights
from go2_rl_lab.utils.terrains import get_terrain_masks

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import ContactSensor
    from isaaclab.envs import ManagerBasedRLEnv


def _get_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> torch.Tensor:
    """Return the base height above a crop of the shared RayCaster samples."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w.torch[:, 2]
    if sensor_cfg is None:
        return base_z

    base_height_samples = get_cropped_ray_hit_heights(env, sensor_cfg, x_range, y_range)
    estimated_ground_z = torch.mean(base_height_samples, dim=1)
    return base_z - estimated_ground_z


def _is_robot_on_flat_terrain(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether each robot's current base position lies on a flat sub-terrain."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    terrain_generator = terrain.cfg.terrain_generator
    if terrain.cfg.terrain_type != "generator" or terrain_generator is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain_generator.sub_terrains is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_masks = get_terrain_masks(env)
    flat_mask = terrain_masks.get("flat")
    if flat_mask is not None:
        return flat_mask

    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _velocity_command_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Resolve the raw Go2 Gym body-frame velocity target."""
    return env.command_manager.get_command(command_name)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands in the base xy-plane."""
    asset: RigidObject = env.scene[asset_cfg.name]
    error = torch.sum(
        torch.square(
            _velocity_command_b(env, command_name)[:, :2]
            - asset.data.root_lin_vel_b.torch[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of yaw-angular-velocity commands."""
    asset: RigidObject = env.scene[asset_cfg.name]
    error = torch.square(
        _velocity_command_b(env, command_name)[:, 2]
        - asset.data.root_ang_vel_b.torch[:, 2]
    )
    return torch.exp(-error / std**2)


def joint_power(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute mechanical power at the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(
            asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
            * asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def joint_pos_penalty_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    stand_cmd_idxs: list[int] = [0, 1, 2],
) -> torch.Tensor:
    """Penalize joint offsets, with a stronger penalty while standing on flat terrain."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = torch.linalg.norm(
        _velocity_command_b(env, command_name)[:, stand_cmd_idxs], dim=1
    )
    position_error = torch.linalg.norm(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids],
        dim=1,
        ord=1,
    )
    stand_still = torch.logical_and(command < 1.0e-3, _is_robot_on_flat_terrain(env))
    return torch.where(stand_still, stand_still_scale * position_error, position_error)


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the second-order difference between consecutive actions."""
    action = env.action_manager.action
    previous_action = env.action_manager.prev_action
    previous_previous_action = getattr(env, "_previous_previous_action", None)
    if previous_previous_action is None or previous_previous_action.shape != action.shape:
        previous_previous_action = torch.zeros_like(action)
    previous_previous_action[env.episode_length_buf <= 1] = 0.0
    env._previous_previous_action = previous_action.clone()

    difference = torch.square(action - 2.0 * previous_action + previous_previous_action)
    difference *= previous_action != 0
    difference *= previous_previous_action != 0
    return torch.sum(difference, dim=1)


def clipped_action_rate_l2(env: ManagerBasedRLEnv, max_penalty: float) -> torch.Tensor:
    """Penalize changes between consecutive actions, capped before reward weighting."""
    difference = env.action_manager.action - env.action_manager.prev_action
    penalty = torch.sum(torch.square(difference), dim=1)
    return _clamp_finite_max(penalty, max_penalty)


def clipped_action_smoothness_l2(env: ManagerBasedRLEnv, max_penalty: float) -> torch.Tensor:
    """Penalize second-order action differences, capped before reward weighting."""
    penalty = action_smoothness_l2(env)
    return _clamp_finite_max(penalty, max_penalty)


def _clamp_finite_max(values: torch.Tensor, maximum: float) -> torch.Tensor:
    """Cap finite values while preserving NaN/Inf for the environment safety check."""
    return torch.where(torch.isfinite(values), torch.clamp_max(values, maximum), values)


def clipped_joint_acc_l2(
    env: ManagerBasedRLEnv,
    max_penalty: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared joint acceleration, capped before reward weighting."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(
        torch.square(asset.data.joint_acc.torch[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return _clamp_finite_max(penalty, max_penalty)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    x_range: tuple[float, float] = (-0.2, 0.2),
    y_range: tuple[float, float] = (-0.15, 0.15),
) -> torch.Tensor:
    """Penalize deviation from a fixed target height above the ground."""
    base_height = _get_base_height(env, asset_cfg, sensor_cfg, x_range, y_range)
    return torch.square(base_height - target_height)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base linear velocity along the z-axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base angular velocity around the x- and y-axes."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :2]), dim=1)


def undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize contacts whose force exceeds the threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history.torch
    is_contact = (
        torch.max(torch.norm(contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    return torch.sum(is_contact, dim=1).float()


def feet_regulation(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    x_range: tuple[float, float] = (-0.2, 0.2),
    y_range: tuple[float, float] = (-0.15, 0.15),
) -> torch.Tensor:
    """Penalize fast horizontal foot motion close to the ground."""
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_position_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :]
    base_position_w = asset.data.root_pos_w.torch.unsqueeze(1)
    feet_velocity_xy_w = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    base_height = _get_base_height(env, asset_cfg, sensor_cfg, x_range, y_range)

    gravity_w = torch.tensor(env.sim.cfg.gravity, device=env.device, dtype=feet_position_w.dtype)
    down_w = gravity_w / torch.norm(gravity_w)
    feet_from_base_w = feet_position_w - base_position_w
    feet_to_base_height = torch.sum(feet_from_base_w * down_w.view(1, 1, 3), dim=-1)
    feet_height = torch.clamp(base_height.unsqueeze(1) - feet_to_base_height, min=0.0)

    return (
        feet_velocity_xy_w.pow(2).sum(dim=-1)
        * torch.exp(-feet_height / (0.025 * base_height_target))
    ).sum(dim=-1)
