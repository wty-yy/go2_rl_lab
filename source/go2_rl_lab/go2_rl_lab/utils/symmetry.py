"""Symmetry observation and action augmentation for Go2."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict


class Go2SymmetryMapper:
    """Mirror Go2 observations and actions across the robot sagittal plane."""

    JOINT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    JOINT_SIGN = [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]
    FOOT_PERM = [1, 0, 3, 2]
    VECTOR_SIGNS = {
        "base_lin_vel": [1, -1, 1],
        "base_ang_vel": [-1, 1, -1],
        "projected_gravity": [1, -1, 1],
        "velocity_commands": [1, -1, -1],
    }
    JOINT_TERMS = {"joint_pos", "joint_vel", "joint_acc", "joint_torque", "actions"}

    def __init__(self, env) -> None:
        """Store the environment used for observation term metadata."""
        self.env = env

    @property
    def base_env(self):
        """Return the unwrapped environment that owns managers and configs."""
        return self.env.unwrapped

    def reverse_vector(self, value: torch.Tensor, signs: list[int]) -> torch.Tensor:
        """Flip coordinate axes for a vector observation term."""
        return value * value.new_tensor(signs)

    def reverse_joints(self, value: torch.Tensor) -> torch.Tensor:
        """Swap left/right legs and flip the hip abduction/adduction axes."""
        perm = torch.tensor(self.JOINT_PERM, device=value.device)
        signs = value.new_tensor(self.JOINT_SIGN)
        return value.index_select(-1, perm) * signs

    def permute_joints(self, value: torch.Tensor) -> torch.Tensor:
        """Swap left/right legs without sign changes for standard deviations."""
        perm = torch.tensor(self.JOINT_PERM, device=value.device)
        return value.index_select(-1, perm)

    def restore_history(self, value: torch.Tensor, cfg) -> torch.Tensor:
        """Restore a flattened history term before applying symmetry."""
        history_length = cfg.history_length
        if history_length > 0 and cfg.flatten_history_dim:
            term_dim = value.shape[-1] // history_length
            return value.reshape(*value.shape[:-1], history_length, term_dim)
        return value

    def flatten_history(self, value: torch.Tensor, cfg) -> torch.Tensor:
        """Flatten a restored history term back to the observation-manager layout."""
        history_length = cfg.history_length
        if history_length > 0 and cfg.flatten_history_dim:
            return value.reshape(*value.shape[:-2], -1)
        return value

    def reverse_feet(self, value: torch.Tensor) -> torch.Tensor:
        """Swap FL/FR and RL/RR for packed foot-contact features."""
        if value.shape[-1] % len(self.FOOT_PERM) != 0:
            return value
        perm = torch.tensor(self.FOOT_PERM, device=value.device)
        return value.reshape(*value.shape[:-1], -1, len(self.FOOT_PERM)).index_select(-1, perm).reshape_as(value)

    def height_scan_shape(self, cfg) -> tuple[int, int, str] | None:
        """Infer the configured two-dimensional height-scan grid shape."""
        sensor_cfg = cfg.params.get("sensor_cfg")
        scene_cfg = getattr(self.base_env.cfg, "scene", None)
        sensor_scene_cfg = getattr(scene_cfg, sensor_cfg.name, None) if sensor_cfg is not None else None
        pattern_cfg = getattr(sensor_scene_cfg, "pattern_cfg", None)
        if pattern_cfg is None:
            return None

        nx = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        ny = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        return nx, ny, pattern_cfg.ordering

    def reverse_height_scan(self, cfg, value: torch.Tensor) -> torch.Tensor:
        """Flip the height-scan grid along its lateral axis."""
        shape = self.height_scan_shape(cfg)
        if shape is None:
            return value

        nx, ny, ordering = shape
        if value.shape[-1] != nx * ny:
            return value

        if ordering == "xy":
            return value.reshape(*value.shape[:-1], ny, nx).flip(-2).reshape_as(value)
        return value.reshape(*value.shape[:-1], nx, ny).flip(-1).reshape_as(value)

    def reverse_term(self, name: str, cfg, value: torch.Tensor) -> torch.Tensor:
        """Mirror one observation term according to its semantic name."""
        if name in self.VECTOR_SIGNS:
            return self.reverse_vector(value, self.VECTOR_SIGNS[name])
        if name in self.JOINT_TERMS:
            return self.reverse_joints(value)
        if name == "contact_force":
            return self.reverse_feet(value)
        if name == "height_scan":
            return self.reverse_height_scan(cfg, value)
        return value

    def reverse_obs_group(self, group_name: str, value: torch.Tensor) -> torch.Tensor:
        """Mirror a flattened observation group term by term."""
        manager = self.base_env.observation_manager
        names = manager._group_obs_term_names[group_name]
        dims = manager._group_obs_term_dim[group_name]
        cfgs = manager._group_obs_term_cfgs[group_name]
        widths = [int(math.prod(dim)) for dim in dims]
        if sum(widths) != value.shape[-1]:
            return value

        chunks = torch.split(value, widths, dim=-1)
        reversed_chunks = []
        for name, cfg, chunk in zip(names, cfgs, chunks):
            term = self.restore_history(chunk, cfg)
            term = self.reverse_term(name, cfg, term)
            reversed_chunks.append(self.flatten_history(term, cfg))
        return torch.cat(reversed_chunks, dim=-1)

    def reverse_obs(self, obs: TensorDict) -> TensorDict:
        """Mirror every observation group known to the observation manager."""
        manager = self.base_env.observation_manager
        reversed_obs = obs.clone()
        for key, value in obs.items():
            if key in manager._group_obs_term_names:
                reversed_obs[key] = self.reverse_obs_group(key, value)
        return reversed_obs


    def data_augmentation(self, obs: TensorDict, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        """Return the original batch followed by its mirrored counterpart."""
        return torch.cat([obs, self.reverse_obs(obs)], dim=0), torch.cat(
            [actions, self.reverse_joints(actions)], dim=0
        )


class Go2Symmetry:
    """Go2 symmetry augmentation for PPO mini-batches."""

    def __init__(self, env) -> None:
        """Create the shared Go2 symmetry mapper."""
        self.mapper = Go2SymmetryMapper(env)
        self.num_aug = 2

    def repeat_batch(self, value: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Repeat a PPO tensor for its original and mirrored samples."""
        return torch.cat([value] * self.num_aug, dim=dim)

    def get_original_mask(
        self,
        teacher_samples: int,
        student_samples: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Mark original samples in a segmented, symmetry-augmented MoE-CTS batch."""
        mask = torch.zeros(
            (teacher_samples + student_samples) * self.num_aug,
            dtype=torch.bool,
            device=device,
        )
        mask[:teacher_samples] = True
        student_start = teacher_samples * self.num_aug
        mask[student_start : student_start + student_samples] = True
        return mask

    def repeat_segmented_batch(self, value: torch.Tensor, teacher_samples: int) -> torch.Tensor:
        """Repeat teacher and student segments without changing their CTS ordering."""
        repeat_shape = (self.num_aug, *([1] * (value.ndim - 1)))
        teacher_batch = value[:teacher_samples].repeat(repeat_shape)
        student_batch = value[teacher_samples:].repeat(repeat_shape)
        return torch.cat([teacher_batch, student_batch], dim=0)

    def augment_segment(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append mirrored samples to one contiguous teacher or student segment."""
        augmented_obs, augmented_actions = self.mapper.data_augmentation(obs_batch, actions_batch)
        return (
            augmented_obs,
            augmented_actions,
            torch.cat([old_mu_batch, self.mapper.reverse_joints(old_mu_batch)], dim=0),
            torch.cat([old_sigma_batch, self.mapper.permute_joints(old_sigma_batch)], dim=0),
        )

    def augment_batch(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ) -> tuple[
        TensorDict,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Append mirrored feed-forward observations, actions, and PPO targets."""
        obs_batch, actions_batch = self.mapper.data_augmentation(obs_batch, actions_batch)
        old_mu_batch = torch.cat([old_mu_batch, self.mapper.reverse_joints(old_mu_batch)], dim=0)
        old_sigma_batch = torch.cat([old_sigma_batch, self.mapper.permute_joints(old_sigma_batch)], dim=0)
        return (
            obs_batch,
            actions_batch,
            self.repeat_batch(target_values_batch),
            self.repeat_batch(advantages_batch),
            self.repeat_batch(returns_batch),
            self.repeat_batch(old_actions_log_prob_batch),
            old_mu_batch,
            old_sigma_batch,
        )


    def augment_moe_cts_batch(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
        teacher_samples: int,
    ) -> tuple[
        TensorDict,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Append mirrored teacher and student samples while preserving segmented order."""
        teacher_obs, teacher_actions, teacher_mu, teacher_sigma = self.augment_segment(
            obs_batch[:teacher_samples],
            actions_batch[:teacher_samples],
            old_mu_batch[:teacher_samples],
            old_sigma_batch[:teacher_samples],
        )
        student_obs, student_actions, student_mu, student_sigma = self.augment_segment(
            obs_batch[teacher_samples:],
            actions_batch[teacher_samples:],
            old_mu_batch[teacher_samples:],
            old_sigma_batch[teacher_samples:],
        )
        return (
            torch.cat([teacher_obs, student_obs], dim=0),
            torch.cat([teacher_actions, student_actions], dim=0),
            self.repeat_segmented_batch(target_values_batch, teacher_samples),
            self.repeat_segmented_batch(advantages_batch, teacher_samples),
            self.repeat_segmented_batch(returns_batch, teacher_samples),
            self.repeat_segmented_batch(old_actions_log_prob_batch, teacher_samples),
            torch.cat([teacher_mu, student_mu], dim=0),
            torch.cat([teacher_sigma, student_sigma], dim=0),
        )

    def augment_moe_cts_batch_generator(self, generator, teacher_samples: int):
        """Apply segmented MoE-CTS augmentation to every storage mini-batch."""
        for batch in generator:
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
            ) = batch
            augmented = self.augment_moe_cts_batch(
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                teacher_samples,
            )
            yield augmented
