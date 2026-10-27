"""Terrain-related helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SolidColorTerrainImporter(TerrainImporter):
    """Replace the ground's checker texture with the configured solid material."""

    def import_ground_plane(self, name: str, size: tuple[float, float] | None = None) -> None:
        """Bind a constant-color surface to the ground's visual mesh.

        The standard importer retains the collision plane. Rebinding the visual
        geometry removes the default checker texture instead of only tinting it.
        """
        material_cfg = self.cfg.visual_material
        if not isinstance(material_cfg, sim_utils.PreviewSurfaceCfg):
            raise ValueError("Solid-color ground requires a PreviewSurfaceCfg visual material.")
        super().import_ground_plane(name, size)
        prim_path = f"{self.cfg.prim_path}/{name}"
        material_path = f"{prim_path}/Looks/solid_color"
        material_cfg.func(material_path, material_cfg)
        sim_utils.bind_visual_material(f"{prim_path}/Environment/Geometry", material_path)


class HighToLowTerrainImporter(TerrainImporter):
    """Assign environments deterministically across terrains from hardest to easiest."""

    envs_per_non_flat_terrain: int = 1

    @staticmethod
    def _column_is_flat(generator_cfg: TerrainGeneratorCfg) -> list[bool]:
        """Return whether each generator column uses the flat sub-terrain."""
        names = list(generator_cfg.sub_terrains.keys())
        proportions = np.array(
            [sub_cfg.proportion for sub_cfg in generator_cfg.sub_terrains.values()],
            dtype=np.float64,
        )
        proportions /= proportions.sum()
        cumulative = np.cumsum(proportions)
        return [
            names[int(np.argmax(col / generator_cfg.num_cols + 0.001 < cumulative))] == "flat"
            for col in range(generator_cfg.num_cols)
        ]

    @classmethod
    def column_replicas(cls, generator_cfg: TerrainGeneratorCfg) -> list[int]:
        """Number of environments assigned to each generator column."""
        return [
            1 if flat else cls.envs_per_non_flat_terrain
            for flat in cls._column_is_flat(generator_cfg)
        ]


    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        """Cycle through terrain cells in descending difficulty and ascending terrain type."""
        num_rows = origins.shape[0]
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)

        self.max_terrain_level = num_rows
        replicas = self.column_replicas(self.cfg.terrain_generator)
        cells = torch.tensor(
            [
                (level, col)
                for level in range(max_init_level, -1, -1)
                for col, repeats in enumerate(replicas)
                for _ in range(repeats)
            ],
            device=self.device,
        )
        env_cells = cells[torch.arange(num_envs, device=self.device) % cells.shape[0]]
        self.terrain_levels = env_cells[:, 0]
        self.terrain_types = env_cells[:, 1]
        return origins[self.terrain_levels, self.terrain_types]


class TerrainPositionLookup:
    """Query current terrain categories on a centered curriculum-generated grid."""

    def __init__(self, cfg: TerrainGeneratorCfg, device: str) -> None:
        """Cache grid dimensions and the generator's exact column-to-category assignment."""
        if not cfg.curriculum:
            raise ValueError("Position-based terrain categories require curriculum-generated columns.")
        self.names = tuple(cfg.sub_terrains)
        proportions = np.array(
            [terrain.proportion for terrain in cfg.sub_terrains.values()], dtype=np.float64
        )
        if not np.isfinite(proportions).all() or (proportions < 0).any() or proportions.sum() <= 0:
            raise ValueError("Terrain proportions must be finite, nonnegative, and have a positive sum.")
        cumulative = np.cumsum(proportions / proportions.sum())
        columns = np.arange(cfg.num_cols) / cfg.num_cols + 0.001
        column_types = np.searchsorted(cumulative, columns, side="right")
        if (column_types >= len(self.names)).any():
            raise ValueError("Terrain column allocation exceeds the configured category count.")
        self.column_types = torch.as_tensor(column_types, dtype=torch.long, device=device)
        self.size = torch.tensor(cfg.size, dtype=torch.float32, device=device)
        self.shape = torch.tensor([cfg.num_rows, cfg.num_cols], device=device)
        self.minimum_xy = -0.5 * self.size * self.shape

    def query(self, positions_w: torch.Tensor) -> torch.Tensor:
        """Return one category ID per world position, with -1 outside the generated grid.

        Half-open cells assign exact seams to the next row or column. Border
        ground and nonfinite positions return -1 instead of an edge category.
        """
        cells = torch.floor((positions_w[:, :2] - self.minimum_xy) / self.size).long()
        inside = ((cells >= 0) & (cells < self.shape)).all(dim=-1)
        inside &= torch.isfinite(positions_w[:, :2]).all(dim=-1)
        columns = cells[:, 1].clamp(0, len(self.column_types) - 1)
        return torch.where(inside, self.column_types[columns], -1)


def get_terrain_position_lookup(env: ManagerBasedEnv) -> TerrainPositionLookup | None:
    """Cache geometric lookup metadata for generated terrain, leaving plane tasks unrestricted."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.cfg.terrain_type != "generator":
        return None
    generator = terrain.cfg.terrain_generator
    if generator is None or not generator.sub_terrains:
        return None
    lookup = getattr(terrain, "_position_lookup", None)
    if lookup is None:
        lookup = TerrainPositionLookup(generator, env.device)
        terrain._position_lookup = lookup
    return lookup


def get_terrain_masks(env: ManagerBasedEnv) -> dict[str, torch.Tensor]:
    """Return terrain masks from each robot's current base world position."""
    lookup = get_terrain_position_lookup(env)
    if lookup is None:
        return {}
    terrain_ids = lookup.query(env.scene["robot"].data.root_pos_w.torch)
    return {name: terrain_ids == index for index, name in enumerate(lookup.names)}


def get_assigned_terrain_masks(env: ManagerBasedEnv) -> dict[str, torch.Tensor]:
    """Return cached terrain assignments for reset placement and curriculum statistics."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return {}

    cached_masks = getattr(terrain, "_assigned_terrain_masks", None)
    if cached_masks is not None:
        return cached_masks

    terrain_masks: dict[str, torch.Tensor] = {}
    terrain_generator = terrain.cfg.terrain_generator
    if (
        terrain.cfg.terrain_type == "generator"
        and terrain_generator is not None
        and terrain_generator.sub_terrains is not None
        and hasattr(terrain, "terrain_types")
    ):
        lookup = get_terrain_position_lookup(env)
        terrain_ids = lookup.column_types[terrain.terrain_types]
        terrain_masks = {name: terrain_ids == index for index, name in enumerate(lookup.names)}

    terrain._assigned_terrain_masks = terrain_masks
    return terrain_masks


def configure_play_sub_terrains(
    terrain_generator: TerrainGeneratorCfg,
    proportions: Mapping[str, float] | None = None,
) -> None:
    """Restrict playback terrain generation to the requested sub-terrain proportions.

    Curriculum generation assigns sub-terrains to grid columns through normalized
    cumulative proportions over every entry of ``terrain_generator.sub_terrains``,
    so unselected terrains are zeroed out instead of removed and columns keep the
    sub-terrain definition order. Passing ``None`` keeps every sub-terrain with one
    column each, while a mapping keeps only its entries and quantizes the relative
    weights into whole columns, for example ``{"flat": 0.75, "wheels": 0.25}``
    yields three flat columns followed by one wheels column.

    Args:
        terrain_generator: Terrain generator configuration modified in place.
        proportions: Optional sub-terrain name to relative weight mapping.

    Raises:
        ValueError: If a name is unknown or no weight is positive.
    """
    sub_terrains = terrain_generator.sub_terrains
    if sub_terrains is None:
        return
    if proportions is None:
        proportions = {name: 1.0 for name in sub_terrains}

    unknown_names = set(proportions).difference(sub_terrains)
    if unknown_names:
        raise ValueError(
            f"Unknown sub-terrain names {sorted(unknown_names)}; "
            f"available names are {sorted(sub_terrains)}."
        )
    weights: dict[str, float] = {}
    for name, weight in proportions.items():
        value = float(weight)
        if value > 0.0:
            weights[name] = value
    if not weights:
        raise ValueError("At least one sub-terrain proportion must be positive.")

    smallest_weight = min(weights.values())
    column_counts = {
        name: max(1, round(weight / smallest_weight)) for name, weight in weights.items()
    }
    total_columns = sum(column_counts.values())
    terrain_generator.num_cols = total_columns
    for name, sub_terrain_cfg in sub_terrains.items():
        column_count = column_counts.get(name, 0)
        sub_terrain_cfg.proportion = column_count / total_columns if column_count else 0.0


__all__ = [
    "HighToLowTerrainImporter",
    "SolidColorTerrainImporter",
    "TerrainPositionLookup",
    "configure_play_sub_terrains",
    "get_assigned_terrain_masks",
    "get_terrain_masks",
    "get_terrain_position_lookup",
]
