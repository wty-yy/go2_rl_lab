"""Velocity commands following the Go2 RL Gym sampling scheme.

The ``rl_gym`` name records the source of the command-generation convention:
the implementation follows ``_resample_commands`` from
https://github.com/wty-yy/go2_rl_gym/blob/master/legged_gym/envs/base/legged_robot.py.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.utils.configclass import configclass

from go2_rl_lab.utils.terrains import get_terrain_masks, get_terrain_position_lookup

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


_COMMAND_NAMES = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")


def _apply_command_clip_rules(
    commands: torch.Tensor,
    env_ids: Sequence[int] | slice,
    clip_rules: Sequence[dict],
) -> None:
    """Clamp coupled command dimensions when a priority rule overflows.

    Each rule triggers when the magnitude of its ``command`` exceeds ``limit``,
    then clamps every command listed in ``clip`` into the given range. Rules are
    applied in list order, so a later rule sees the values already clipped by an
    earlier rule and an earlier rule wins on the dimensions it clips.
    """
    command_indices = {name: index for index, name in enumerate(_COMMAND_NAMES)}
    selected = commands[env_ids]
    for rule in clip_rules:
        source_index = command_indices[rule["command"]]
        overflow = selected[:, source_index].abs() > rule["limit"]
        for command_name, command_range in rule["clip"].items():
            command_index = command_indices[command_name]
            selected[:, command_index] = torch.where(
                overflow,
                selected[:, command_index].clamp(*command_range),
                selected[:, command_index],
            )
    commands[env_ids] = selected


def _sample_disjoint_intervals(
    env_ids: Sequence[int],
    limit_bound: torch.Tensor,
    cfg_min: torch.Tensor,
    cfg_max: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Sample from ``[cfg_min, -limit] U [limit, cfg_max]``."""
    width_negative = torch.nn.functional.relu(-limit_bound - cfg_min)
    width_positive = torch.nn.functional.relu(cfg_max - limit_bound)
    total_width = width_negative + width_positive + 1.0e-6
    uniform_sample = torch.rand(len(env_ids), device=device) * total_width
    return torch.where(
        uniform_sample < width_negative,
        cfg_min + uniform_sample,
        cfg_max - width_positive + (uniform_sample - width_negative),
    )


def _sample_single_interval(
    env_ids: Sequence[int],
    cfg_min: torch.Tensor,
    cfg_max: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Sample uniformly from ``[cfg_min, cfg_max]``."""
    uniform_sample = torch.rand(len(env_ids), device=device)
    return cfg_min + uniform_sample * (cfg_max - cfg_min)


class RLGymVelCommand(CommandTerm):
    """Generate velocity commands using the Go2 RL Gym sampling scheme."""

    cfg: RLGymVelCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: RLGymVelCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.visualizer_max_speed <= 0.0:
            raise ValueError("visualizer_max_speed must be positive.")
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.max_move_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.velocity_command_b = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        self._requested_velocity_command_b = torch.zeros_like(self.velocity_command_b)

        self.command_ranges = self.cfg.ranges.to_dict()
        self._global_range_values = None
        self._env_command_ranges = torch.empty(self.num_envs, 3, 2, device=self.device)
        self.env_command_ranges = {
            name: self._env_command_ranges[:, index] for index, name in enumerate(_COMMAND_NAMES)
        }
        self.max_lin_vel = self._get_max_lin_vel()
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._initialize_terrain_command_limits()
        self.zero_command_prob = 0.0
        self.max_command_x = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_command_y = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_command_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cfg.command_range_curriculum = sorted(
            self.cfg.command_range_curriculum,
            key=lambda stage: stage["iter"],
            reverse=True,
        )
        self.cfg.limit_vel_curriculum = sorted(
            self.cfg.limit_vel_curriculum,
            key=lambda stage: stage["iter"],
            reverse=True,
        )
        self.limit_vel: dict[str, list[int]] = {}
        self.limit_vel_comb = torch.zeros(0, len(_COMMAND_NAMES), dtype=torch.long, device=self.device)
        self._update_limit_vel_curriculum()
        if not self.limit_vel and (self.cfg.limit_vel_prob > 0.0 or self.cfg.limit_vel_hold_prob > 0.0):
            raise ValueError(
                "limit_vel_curriculum must define a stage at iteration 0 when limit-velocity "
                "sampling is enabled."
            )
        self._validate_command_clip_rules()

    def __str__(self) -> str:
        return f"RLGymVelCommand:\nCommand shape: {self.velocity_command_b.shape}"

    def _validate_command_clip_rules(self):
        for rule in self.cfg.command_clip_rules:
            if rule["command"] not in _COMMAND_NAMES:
                raise ValueError(
                    f"Unknown command_clip_rules command: {rule['command']!r}."
                )
            for command_name, command_range in rule["clip"].items():
                if command_name not in _COMMAND_NAMES:
                    raise ValueError(
                        f"Unknown command_clip_rules clip target: {command_name!r}."
                    )
                if command_range[0] > command_range[1]:
                    raise ValueError(
                        "command_clip_rules range for "
                        f"{command_name!r} must be ordered (min, max), got {command_range}."
                    )

    @property
    def command(self) -> torch.Tensor:
        return self.velocity_command_b

    @property
    def terrain_masks(self) -> dict[str, torch.Tensor]:
        """Resolve current terrain masks for command visualization."""
        return get_terrain_masks(self._env)

    def _get_max_lin_vel(self) -> float:
        return max(
            abs(self.command_ranges["lin_vel_x"][0]),
            abs(self.command_ranges["lin_vel_x"][1]),
            abs(self.command_ranges["lin_vel_y"][0]),
            abs(self.command_ranges["lin_vel_y"][1]),
        )

    def _update_metrics(self):
        self.max_command_x[:] = self.command_ranges["lin_vel_x"][1]
        self.max_command_y[:] = self.command_ranges["lin_vel_y"][1]
        self.max_command_yaw[:] = self.command_ranges["ang_vel_yaw"][1]
        self.metrics["max_velocity_command_x"] = self.max_command_x
        self.metrics["max_velocity_command_y"] = self.max_command_y
        self.metrics["max_velocity_command_yaw"] = self.max_command_yaw

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        """Reset command state for all or selected environments."""
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids or slice(None)]
        self.time_left[env_ids] = self.cfg.resampling_time
        self.commands_xy_accumulation[env_ids] = 0.0
        self.max_move_distance[env_ids] = 0.0
        self.last_is_limit_vel[env_ids] = False
        return super().reset(env_ids)

    def _resample(self, env_ids: Sequence[int] | slice):
        """Sample velocity commands for the selected environments."""
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return

        self._update_command_range_curriculum()
        self._update_limit_vel_curriculum()
        self._update_env_command_ranges()
        remaining_distance = torch.clip(
            0.625 * self.cfg.command_area_length
            - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.resampling_time,
            min=0.0,
        )
        self.time_left[env_ids] = self.cfg.resampling_time

        if self.cfg.dynamic_resample_commands:
            remaining_steps = self._env.max_episode_length - self._env.episode_length_buf[env_ids] + 1
            velocity_lower_bound = torch.clip(
                remaining_distance / (remaining_steps * self._env.step_dt + 1.0e-9),
                min=0.0,
            )
            self.velocity_command_b[env_ids, 0] = _sample_disjoint_intervals(
                env_ids,
                velocity_lower_bound,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device,
            )
            self.velocity_command_b[env_ids, 1] = _sample_disjoint_intervals(
                env_ids,
                velocity_lower_bound,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device,
            )
            self.velocity_command_b[env_ids, 2] = _sample_single_interval(
                env_ids,
                self.env_command_ranges["ang_vel_yaw"][env_ids, 0],
                self.env_command_ranges["ang_vel_yaw"][env_ids, 1],
                self.device,
            )
        else:
            self.velocity_command_b[env_ids, 0] = _sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device,
            )
            self.velocity_command_b[env_ids, 1] = _sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device,
            )
            self.velocity_command_b[env_ids, 2] = _sample_single_interval(
                env_ids,
                self.env_command_ranges["ang_vel_yaw"][env_ids, 0],
                self.env_command_ranges["ang_vel_yaw"][env_ids, 1],
                self.device,
            )
            linear_command_is_large = torch.norm(self.velocity_command_b[env_ids, :2], dim=1) > 0.2
            self.velocity_command_b[env_ids, :2] *= linear_command_is_large.unsqueeze(1)

        random_probability = torch.rand(len(env_ids), device=self.device)
        min_probability = 0.0
        max_probability = 0.0

        if self.cfg.limit_vel_prob > 0.0:
            max_probability += self.cfg.limit_vel_prob
            limit_mask = (random_probability >= min_probability) & (random_probability < max_probability)
            limit_env_ids = env_ids[limit_mask]
            if len(limit_env_ids) > 0:
                change_limit_env_ids = limit_env_ids
                if self.cfg.limit_vel_invert_when_continuous:
                    was_limited = self.last_is_limit_vel[limit_env_ids]
                    invert_env_ids = limit_env_ids[was_limited]
                    self.velocity_command_b[invert_env_ids] *= -1.0
                    change_limit_env_ids = limit_env_ids[~was_limited]

                velocity_index = torch.randint(
                    0,
                    self.limit_vel_comb.shape[0],
                    (len(change_limit_env_ids),),
                    device=self.device,
                )
                self._apply_limit_commands(change_limit_env_ids, velocity_index)
                self.last_is_limit_vel[env_ids] = False
                self.last_is_limit_vel[limit_env_ids] = True
            else:
                self.last_is_limit_vel[env_ids] = False
            min_probability += self.cfg.limit_vel_prob

        if self.cfg.limit_vel_hold_prob > 0.0:
            max_probability += self.cfg.limit_vel_hold_prob
            episode_start = self.command_counter[env_ids] == 0
            hold_mask = (
                (random_probability >= min_probability)
                & (random_probability < max_probability)
                & episode_start
            )
            hold_env_ids = env_ids[hold_mask]
            if len(hold_env_ids) > 0:
                combo_index = self._limit_vel_combo_index(self.cfg.limit_vel_hold_combo)
                self._apply_limit_commands(hold_env_ids, combo_index.expand(len(hold_env_ids)))
                self.time_left[hold_env_ids] = self._env.max_episode_length_s
                self.last_is_limit_vel[hold_env_ids] = True
            min_probability += self.cfg.limit_vel_hold_prob

        if self.cfg.zero_command_curriculum is not None:
            self.zero_command_prob = self._get_current_scale(self.cfg.zero_command_curriculum)
        if self.zero_command_prob > 0.0:
            max_probability += self.zero_command_prob
            next_time_left = torch.clip(
                self._env.max_episode_length_s
                - self._env.episode_length_buf[env_ids] * self._env.step_dt
                - remaining_distance / (0.8 * self.max_lin_vel + 1.0e-9),
                min=0.0,
                max=self.cfg.resampling_time,
            )
            zero_mask = (
                (random_probability >= min_probability)
                & (random_probability < max_probability)
                & (next_time_left > 0.0)
            )
            zero_env_ids = env_ids[zero_mask]
            if len(zero_env_ids) > 0:
                self.velocity_command_b[zero_env_ids] = 0.0
                self.time_left[zero_env_ids] = next_time_left[zero_mask]
                self._add_limit_yaw_to_zero_commands(zero_env_ids)

        self._requested_velocity_command_b[env_ids] = self.velocity_command_b[env_ids]
        self._apply_terrain_command_limits(env_ids)

        self.commands_xy_accumulation[env_ids] += self.velocity_command_b[env_ids, :2]
        self.command_counter[env_ids] += 1

    def _update_command_range_curriculum(self):
        current_iteration = self._env.common_step_counter // self.cfg.num_steps_per_iter
        for index in range(len(self.cfg.command_range_curriculum) - 1, -1, -1):
            stage = self.cfg.command_range_curriculum[index]
            if current_iteration < stage["iter"]:
                continue
            self.command_ranges["lin_vel_x"] = stage["lin_vel_x"]
            self.command_ranges["lin_vel_y"] = stage["lin_vel_y"]
            self.command_ranges["ang_vel_yaw"] = stage["ang_vel_yaw"]
            self.max_lin_vel = self._get_max_lin_vel()
            self.cfg.command_range_curriculum.pop(index)
            self._update_env_command_ranges()
            print(f"Command range updated at iteration {current_iteration}: {self.command_ranges}")

    def _resolve_limit_vel_stage(self, stage: dict) -> dict[str, list[int]]:
        """Return the complete ``limit_vel`` configuration defined by one curriculum stage.

        Every stage fully specifies the three direction-code lists, so activating a stage
        replaces the whole active configuration instead of merging into it.

        Args:
            stage: Curriculum entry holding ``iter`` and the three direction-code lists.

        Returns:
            The validated direction-code configuration of the stage.
        """
        stage_keys = set(stage) - {"iter"}
        if stage_keys != set(_COMMAND_NAMES):
            raise ValueError(
                f"limit_vel_curriculum stage {stage!r} must define exactly {_COMMAND_NAMES}."
            )
        limit_vel: dict[str, list[int]] = {}
        for command_name in _COMMAND_NAMES:
            directions = stage[command_name]
            if not directions:
                raise ValueError(
                    f"limit_vel_curriculum directions for {command_name!r} must not be empty."
                )
            limit_vel[command_name] = list(directions)
        return limit_vel

    def _build_limit_vel_comb(self) -> torch.Tensor:
        """Build the direction-code combinations allowed by the active ``limit_vel``."""
        return torch.tensor(
            list(
                product(
                    self.limit_vel["lin_vel_x"],
                    self.limit_vel["lin_vel_y"],
                    self.limit_vel["ang_vel_yaw"],
                )
            ),
            device=self.device,
        )

    def _update_limit_vel_curriculum(self):
        """Activate every limit-velocity curriculum stage reached so far.

        Stages are applied from the earliest to the latest reached iteration; each one
        replaces ``limit_vel`` and rebuilds ``limit_vel_comb`` before being removed from
        the curriculum.
        """
        current_iteration = self._env.common_step_counter // self.cfg.num_steps_per_iter
        for index in range(len(self.cfg.limit_vel_curriculum) - 1, -1, -1):
            stage = self.cfg.limit_vel_curriculum[index]
            if current_iteration < stage["iter"]:
                continue
            self.limit_vel = self._resolve_limit_vel_stage(stage)
            self.limit_vel_comb = self._build_limit_vel_comb()
            self.cfg.limit_vel_curriculum.pop(index)
            print(f"Limit velocity updated at iteration {current_iteration}: {self.limit_vel}")

    def _update_env_command_ranges(self):
        """Intersect current curriculum ranges with limits under each robot's base."""
        values = tuple(tuple(self.command_ranges[name]) for name in _COMMAND_NAMES)
        if values != self._global_range_values:
            self._global_range_values = values
            self._global_command_ranges = torch.tensor(
                values, dtype=self._env_command_ranges.dtype, device=self.device
            )
        if self._terrain_lookup is None:
            self._env_command_ranges.copy_(self._global_command_ranges)
            return
        self.current_terrain_ids = self._terrain_lookup.query(self.robot.data.root_pos_w.torch)
        limits = self._terrain_max_ranges[self.current_terrain_ids + 1]
        # Saturate disjoint ranges too: a held request [3, 3] on a [-1, 1]
        # terrain becomes [1, 1], preserving ordered sampling bounds.
        self._env_command_ranges.copy_(
            torch.clamp(self._global_command_ranges, min=limits[..., :1], max=limits[..., 1:])
        )

    def _initialize_terrain_command_limits(self):
        """Precompute category limits, applying flat-terrain limits outside the generated grid."""
        self._terrain_lookup = get_terrain_position_lookup(self._env)
        if self._terrain_lookup is None:
            self._update_env_command_ranges()
            return

        terrain_names = self._terrain_lookup.names
        missing_terrain_names = set(terrain_names) - set(self.cfg.terrain_max_command_ranges)
        if missing_terrain_names:
            raise ValueError(
                "Missing maximum command ranges for terrain types: "
                f"{sorted(missing_terrain_names)}."
            )

        ranges = torch.empty(len(terrain_names) + 1, 3, 2, device=self.device)
        ranges[..., 0] = -torch.inf
        ranges[..., 1] = torch.inf
        for terrain_index, terrain_name in enumerate(terrain_names, start=1):
            for axis, command_name in enumerate(_COMMAND_NAMES):
                limits = self.cfg.terrain_max_command_ranges[terrain_name].get(command_name)
                if limits is not None:
                    if limits[0] > limits[1]:
                        raise ValueError(f"Unordered command limits for {terrain_name}/{command_name}: {limits}.")
                    ranges[terrain_index, axis] = torch.tensor(limits, device=self.device)
        for axis, command_name in enumerate(_COMMAND_NAMES):
            limits = self.cfg.terrain_max_command_ranges.get("flat", {}).get(command_name)
            if limits is not None:
                ranges[0, axis] = torch.tensor(limits, device=self.device)
        if not torch.all(ranges[..., 0] <= ranges[..., 1]):
            raise ValueError("Terrain command limits must contain ordered ranges.")
        self._terrain_max_ranges = ranges
        self._update_env_command_ranges()

    def _apply_terrain_command_limits(self, env_ids: Sequence[int] | None = None) -> None:
        """Clip saved requests using current limits without discarding their original values."""
        if env_ids is None:
            env_ids = slice(None)
        limits = self._env_command_ranges[env_ids]
        self.velocity_command_b[env_ids] = torch.clamp(
            self._requested_velocity_command_b[env_ids], min=limits[..., 0], max=limits[..., 1]
        )
        _apply_command_clip_rules(self.velocity_command_b, env_ids, self.cfg.command_clip_rules)

    def _apply_limit_commands(self, env_ids: torch.Tensor, velocity_index: torch.Tensor):
        if len(env_ids) == 0:
            return
        for command_index in range(len(_COMMAND_NAMES)):
            limit_direction = self.limit_vel_comb[velocity_index, command_index]
            command_value = torch.where(
                limit_direction == -1,
                self._global_command_ranges[command_index, 0],
                self._global_command_ranges[command_index, 1],
            )
            command_value[limit_direction == 0] = 0.0
            self.velocity_command_b[env_ids, command_index] = command_value

    def _limit_vel_combo_index(self, combo: Sequence[int]) -> torch.Tensor:
        """Return the matching row index of a direction-code combination in ``limit_vel_comb``."""
        target = torch.tensor(combo, dtype=self.limit_vel_comb.dtype, device=self.device)
        matches = torch.all(self.limit_vel_comb == target, dim=1)
        if not bool(matches.any()):
            raise ValueError(
                "limit_vel_hold_combo "
                f"{tuple(int(value) for value in combo)} is not a combination of the active "
                f"limit_vel {self.limit_vel}."
            )
        return matches.nonzero(as_tuple=False).squeeze(-1)[0]

    def _add_limit_yaw_to_zero_commands(self, env_ids: torch.Tensor):
        if self.cfg.limit_ang_vel_at_zero_command_prob <= 0.0 or len(env_ids) == 0:
            return
        add_yaw_mask = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.limit_ang_vel_at_zero_command_prob
        )
        add_yaw_env_ids = env_ids[add_yaw_mask]
        if len(add_yaw_env_ids) == 0:
            return
        use_lower_bound = torch.rand(len(add_yaw_env_ids), device=self.device) < 0.5
        self.velocity_command_b[add_yaw_env_ids, 2] = torch.where(
            use_lower_bound,
            self.env_command_ranges["ang_vel_yaw"][add_yaw_env_ids, 0],
            self.env_command_ranges["ang_vel_yaw"][add_yaw_env_ids, 1],
        )

    def _get_current_scale(self, config: dict) -> float:
        current_iteration = self._env.common_step_counter // self.cfg.num_steps_per_iter
        percentage = (current_iteration - config["start_iter"]) / (
            config["end_iter"] - config["start_iter"]
        )
        percentage = max(min(percentage, 1.0), 0.0)
        return (1.0 - percentage) * config["start_value"] + percentage * config["end_value"]

    def _update_command(self):
        self._update_env_command_ranges()
        self._apply_terrain_command_limits()
        current_distance = torch.linalg.norm(
            self.robot.data.root_pos_w.torch[:, :2] - self._env.scene.env_origins[:, :2],
            dim=1,
        )
        self.max_move_distance = torch.maximum(self.max_move_distance, current_distance)

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        elif hasattr(self, "goal_vel_visualizer"):
            self.goal_vel_visualizer.set_visibility(False)
            self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        del event
        env_ids = self._debug_vis_env_ids()
        if env_ids.numel() == 0:
            return
        base_position_w = self.robot.data.root_pos_w.torch[env_ids].clone()
        base_position_w[:, 2] += 0.5
        desired_scale, desired_quaternion = self._resolve_xy_velocity_to_arrow(
            self.velocity_command_b[env_ids, :2], env_ids
        )
        current_scale, current_quaternion = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b.torch[env_ids, :2], env_ids
        )
        self.goal_vel_visualizer.visualize(base_position_w, desired_quaternion, desired_scale)
        self.current_vel_visualizer.visualize(base_position_w, current_quaternion, current_scale)

    def _debug_vis_env_ids(self) -> torch.Tensor:
        """Return the environments selected for command visualization."""
        terrain_names = self.cfg.debug_vis_terrain_names
        if terrain_names is None:
            return torch.arange(self.num_envs, device=self.device)
        terrain_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for terrain_name in terrain_names:
            terrain_mask |= self.terrain_masks.get(terrain_name, torch.zeros_like(terrain_mask))
        return terrain_mask.nonzero(as_tuple=False).squeeze(-1)

    def _resolve_xy_velocity_to_arrow(
        self,
        xy_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visualizer = getattr(self, "current_vel_visualizer", None)
        if visualizer is None:
            visualizer = self.goal_vel_visualizer
        default_scale = visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        safe_velocity = torch.nan_to_num(xy_velocity, nan=0.0, posinf=0.0, neginf=0.0)
        safe_velocity = safe_velocity.clamp(
            min=-self.cfg.visualizer_max_speed,
            max=self.cfg.visualizer_max_speed,
        )
        speed = torch.linalg.norm(safe_velocity, dim=1)
        speed = torch.nan_to_num(
            speed,
            nan=0.0,
            posinf=self.cfg.visualizer_max_speed,
            neginf=0.0,
        ).clamp_max(self.cfg.visualizer_max_speed)
        arrow_scale[:, 0] *= speed * 3.0
        heading_angle = torch.atan2(safe_velocity[:, 1], safe_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        root_heading = self.robot.data.heading_w.torch
        if env_ids is not None:
            root_heading = root_heading[env_ids]
        arrow_quaternion = math_utils.quat_from_euler_xyz(
            zeros, zeros, root_heading + heading_angle
        )
        return arrow_scale, arrow_quaternion


@configclass
class RLGymVelCommandCfg(CommandTermCfg):
    """Configuration for :class:`RLGymVelCommand`."""

    class_type: type = RLGymVelCommand
    cmd_kind: str = "command/body/velocity"
    element_names: list[str] = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
    asset_name: str = "robot"
    dynamic_resample_commands: bool = True
    limit_vel_invert_when_continuous: bool = True
    zero_command_curriculum: dict | None = {
        "start_iter": 0,
        "end_iter": 1500,
        "start_value": 0.0,
        "end_value": 0.1,
    }
    limit_vel_curriculum: list[dict] = [
        {
            "iter": 0,
            "lin_vel_x": [-1, 1],
            "lin_vel_y": [-1, 1],
            "ang_vel_yaw": [-1, 0, 1],
        },
    ]
    """Iteration-indexed complete ``limit_vel`` configurations, one dict per stage."""
    command_range_curriculum: list[dict] = [
        {
            "iter": 5000,
            "lin_vel_x": [-1.0, 1.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-2.0, 2.0],
        },
        {
            "iter": 15000,
            "lin_vel_x": [-2.0, 2.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-3.0, 3.0],
        },
        {
            "iter": 25000,
            "lin_vel_x": [-3.0, 3.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-3.0, 3.0],
        },
        {
            "iter": 35000,
            "lin_vel_x": [-4.0, 4.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-3.0, 3.0],
        },
        {
            "iter": 45000,
            "lin_vel_x": [-4.5, 4.2],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_yaw": [-3.0, 3.0],
        },
    ]
    terrain_max_command_ranges: dict[str, dict[str, tuple[float, float]]] = {
        "flat": {
            "lin_vel_x": (-4.5, 4.2),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-3.0, 3.0),
        },
        "wave": {
            "lin_vel_x": (-1.5, 1.5),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-1.5, 1.5),
        },
        "slope_up": {
            "lin_vel_x": (-1.5, 1.5),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-1.5, 1.5),
        },
        "slope_down": {
            "lin_vel_x": (-1.5, 1.5),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-1.5, 1.5),
        },
        "rough_slope": {
            "lin_vel_x": (-1.5, 1.5),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-1.5, 1.5),
        },
        "stairs_up": {
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-1.5, 1.5),
        },
        "stairs_down": {
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-1.5, 1.5),
        },
        "obstacles": {
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-1.5, 1.5),
        },
        "wheels": {
            "lin_vel_x": (-2.0, 1.5),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-1.5, 1.5),
        },
        "small_squares": {
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_yaw": (-1.5, 1.5),
        },
    }
    command_clip_rules: list[dict] = [  # checking is performed according to the order of the list
        {
            "command": "lin_vel_x",
            "limit": 2.0,  # if |lin_vel_x| > 2.0, clip lin_vel_y and ang_vel_yaw
            "clip": {
                "lin_vel_y": (-0.5, 0.5),
                "ang_vel_yaw": (-1.0, 1.0),
            },
        },
        {
            "command": "ang_vel_yaw",
            "limit": 2.0,  # if |ang_vel_yaw| > 2.0, clip lin_vel_x and lin_vel_y
            "clip": {
                "lin_vel_x": (-2.0, 2.0),
                "lin_vel_y": (-1.0, 1.0),
            },
        },
    ]
    """Priority-ordered coupling rules applied after sampling and current-terrain clipping.

    Each rule triggers when the magnitude of its ``command`` exceeds ``limit``,
    then clamps every command listed in ``clip`` into the given range. Rules are
    evaluated in list order, so a later rule sees the values already clipped by
    an earlier rule and an earlier rule wins on the dimensions it clips.
    """
    resampling_time: float = 5.0
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    limit_ang_vel_at_zero_command_prob: float = 0.2
    limit_vel_prob: float = 0.2
    limit_vel_hold_prob: float = 0.0
    """Probability that a fresh episode holds ``limit_vel_hold_combo`` until the episode ends."""
    limit_vel_hold_combo: tuple[int, int, int] = (1, 0, 0)
    """Direction-code combination held for a whole episode (same encoding as ``limit_vel``)."""
    num_steps_per_iter: int = 24
    command_area_length: float = 8.0
    visualizer_max_speed: float = 1.0
    """Cap debug-arrow length when a simulated velocity becomes unstable."""
    debug_vis_terrain_names: tuple[str, ...] | None = None
    """Terrain names to include in debug visualization, or all when unset."""

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-0.5, 0.5)
        lin_vel_y: tuple[float, float] = (-0.5, 0.5)
        ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

    ranges: Ranges = Ranges()
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)



__all__ = ["RLGymVelCommand", "RLGymVelCommandCfg"]
