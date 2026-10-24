"""Direct-mesh generation backend for the Go2 mixed terrain.

Each sub-terrain is converted into a closed watertight triangular mesh at
generation time, giving sharp stair and obstacle edges without the regular
height-field tessellation. The height-field counterpart lives in ``terrain.py``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.sim.utils import get_current_stage
from isaaclab.terrains import SubTerrainBaseCfg, TerrainImporter
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

from go2_rl_lab.tasks.mixed.terrain import (
    MESH_BASE_THICKNESS,
    MIXED_TERRAIN_CFG,
    DirectMeshTerrainCfg,
    PerSubTerrainSlopeThresholdGenerator,
    RoughSlopeTerrainCfg,
    WaveTerrainCfg,
)
from go2_rl_lab.utils.terrains import HighToLowTerrainImporter


def _resolve_height_function(function: Callable) -> Callable:
    """Resolve config-style callables and remove the height-field decorator."""
    if hasattr(function, "_resolve"):
        function = function._resolve()
    return getattr(function, "__wrapped__", function)


def _height_field_with_border(
    difficulty: float,
    cfg: HfTerrainBaseCfg,
    horizontal_scale_override: float | None = None,
) -> tuple[np.ndarray, tuple[float, float], float, float]:
    """Build the same bordered height field used by Isaac Lab's decorator."""
    work_cfg = cfg.copy()
    horizontal_scale = float(
        horizontal_scale_override
        if horizontal_scale_override is not None
        else work_cfg.horizontal_scale
    )
    work_cfg.horizontal_scale = horizontal_scale
    vertical_scale = float(work_cfg.vertical_scale)
    width_pixels = int(work_cfg.size[0] / horizontal_scale) + 1
    length_pixels = int(work_cfg.size[1] / horizontal_scale) + 1
    border_pixels = int(work_cfg.border_width / horizontal_scale) + 1
    heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    terrain_size = tuple(float(value) for value in work_cfg.size)
    sub_terrain_size = (
        (width_pixels - 2 * border_pixels) * horizontal_scale,
        (length_pixels - 2 * border_pixels) * horizontal_scale,
    )
    work_cfg.size = sub_terrain_size
    slope_threshold_override = getattr(work_cfg, "slope_threshold_override", None)
    if slope_threshold_override is not None:
        work_cfg.slope_threshold = slope_threshold_override
    height_function = _resolve_height_function(work_cfg.function)
    generated_heights, _ = height_function(difficulty, work_cfg)
    heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = generated_heights
    return heights, terrain_size, horizontal_scale, vertical_scale


def _boundary_edges(num_rows: int, num_cols: int) -> np.ndarray:
    """Return counter-clockwise boundary edges for a regular grid."""
    edges: list[tuple[int, int]] = []
    for col in range(num_cols - 1):
        edges.append((col + 1, col))
        edges.append(((num_rows - 1) * num_cols + col, (num_rows - 1) * num_cols + col + 1))
    for row in range(num_rows - 1):
        edges.append((row * num_cols, (row + 1) * num_cols))
        edges.append(((row + 1) * num_cols + num_cols - 1, row * num_cols + num_cols - 1))
    return np.asarray(edges, dtype=np.int64)


def _height_field_origin(
    heights: np.ndarray,
    terrain_size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    """Return the centered spawn origin used by Isaac Lab's height-field wrapper."""
    center_x_1 = int((terrain_size[0] * 0.5 - 1.0) / horizontal_scale)
    center_x_2 = int((terrain_size[0] * 0.5 + 1.0) / horizontal_scale)
    center_y_1 = int((terrain_size[1] * 0.5 - 1.0) / horizontal_scale)
    center_y_2 = int((terrain_size[1] * 0.5 + 1.0) / horizontal_scale)
    return np.array(
        (
            terrain_size[0] * 0.5,
            terrain_size[1] * 0.5,
            float(np.max(heights[center_x_1:center_x_2, center_y_1:center_y_2]))
            * vertical_scale,
        ),
        dtype=np.float64,
    )


def _effective_slope_threshold(cfg: HfTerrainBaseCfg) -> float | None:
    """Return the per-sub-terrain slope threshold override when present."""
    return (
        cfg.slope_threshold_override
        if getattr(cfg, "slope_threshold_override", None) is not None
        else cfg.slope_threshold
    )


def _close_height_field_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid_shape: tuple[int, int],
    terrain_size: tuple[float, float],
    base_z: float,
) -> trimesh.Trimesh:
    """Close one regular height-field surface with side walls and a bottom cap."""
    num_rows, num_cols = grid_shape
    boundary = _boundary_edges(num_rows, num_cols)
    bottom_indices: dict[int, int] = {}
    bottom_vertices: list[np.ndarray] = []
    for top_index in boundary.reshape(-1):
        if int(top_index) in bottom_indices:
            continue
        bottom_indices[int(top_index)] = len(vertices) + len(bottom_vertices)
        bottom_vertex = vertices[top_index].copy()
        bottom_vertex[2] = base_z
        bottom_vertices.append(bottom_vertex)

    center_bottom = len(vertices) + len(bottom_vertices)
    bottom_vertices.append(
        np.array((terrain_size[0] * 0.5, terrain_size[1] * 0.5, base_z), dtype=np.float32)
    )
    side_faces = np.empty((len(boundary) * 2, 3), dtype=np.int64)
    bottom_faces = np.empty((len(boundary), 3), dtype=np.int64)
    for index, (top_a, top_b) in enumerate(boundary):
        bottom_a = bottom_indices[int(top_a)]
        bottom_b = bottom_indices[int(top_b)]
        side_faces[2 * index] = (top_a, bottom_a, bottom_b)
        side_faces[2 * index + 1] = (top_a, bottom_b, top_b)
        bottom_faces[index] = (center_bottom, bottom_b, bottom_a)

    mesh = trimesh.Trimesh(
        vertices=np.vstack((vertices, np.asarray(bottom_vertices, dtype=np.float32))),
        faces=np.vstack((faces, side_faces, bottom_faces)),
        process=False,
    )
    mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_volume:
        raise RuntimeError("Generated mixed mesh is not a watertight volume.")
    return mesh


def _compression_indices(heights: np.ndarray, axis: int) -> np.ndarray:
    """Keep only grid lines that participate in a height discontinuity."""
    differences = np.diff(heights, axis=axis)
    changed_rows = np.any(differences != 0, axis=1 - axis)
    changed_indices = np.flatnonzero(changed_rows)
    selected = {0, heights.shape[axis] - 1}
    selected.update(changed_indices.tolist())
    selected.update((changed_indices + 1).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _closed_height_field_mesh(
    difficulty: float,
    cfg: HfTerrainBaseCfg,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Generate a closed mesh from one mixed-terrain height field."""
    smooth_terrain = isinstance(
        cfg,
        (WaveTerrainCfg, RoughSlopeTerrainCfg, terrain_gen.HfPyramidSlopedTerrainCfg),
    )
    heights, terrain_size, horizontal_scale, vertical_scale = _height_field_with_border(
        difficulty,
        cfg,
        horizontal_scale_override=0.2 if smooth_terrain else None,
    )
    vertices, faces = convert_height_field_to_mesh(
        heights,
        horizontal_scale,
        vertical_scale,
        _effective_slope_threshold(cfg),
    )
    base_z = float(np.min(heights)) * vertical_scale - MESH_BASE_THICKNESS
    mesh = _close_height_field_surface(
        vertices,
        faces,
        heights.shape,
        terrain_size,
        base_z,
    )
    return mesh, _height_field_origin(heights, terrain_size, horizontal_scale, vertical_scale)


def _box_obstacle_mesh(
    difficulty: float,
    cfg: terrain_gen.HfDiscreteObstaclesTerrainCfg,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Generate the obstacle height field as a union of axis-aligned boxes."""
    heights, terrain_size, horizontal_scale, vertical_scale = _height_field_with_border(
        difficulty, cfg
    )
    row_indices = _compression_indices(heights, axis=0)
    column_indices = _compression_indices(heights, axis=1)
    compressed = heights[np.ix_(row_indices, column_indices)]

    cell_heights = compressed[:-1, :-1]
    x_edges = row_indices * horizontal_scale
    y_edges = column_indices * horizontal_scale
    base_z = float(np.min(heights)) * vertical_scale - MESH_BASE_THICKNESS
    height_levels = np.unique(cell_heights)
    z_edges = np.concatenate(([base_z], height_levels.astype(np.float64) * vertical_scale))
    occupied = cell_heights[:, :, None] >= height_levels[None, None, :]
    num_rows, num_cols, num_levels = occupied.shape

    vertices: list[tuple[float, float, float]] = []
    vertex_indices: dict[tuple[int, int, int], int] = {}
    faces: list[tuple[int, int, int]] = []

    def vertex(row: int, col: int, level: int) -> int:
        key = (row, col, level)
        if key not in vertex_indices:
            vertex_indices[key] = len(vertices)
            vertices.append(
                (
                    float(x_edges[row]),
                    float(y_edges[col]),
                    float(z_edges[level]),
                )
            )
        return vertex_indices[key]

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        faces.extend(((a, b, c), (a, c, d)))

    for row, col, level in np.argwhere(occupied):
        if row == 0 or not occupied[row - 1, col, level]:
            add_quad(
                vertex(row, col, level),
                vertex(row, col, level + 1),
                vertex(row, col + 1, level + 1),
                vertex(row, col + 1, level),
            )
        if row == num_rows - 1 or not occupied[row + 1, col, level]:
            add_quad(
                vertex(row + 1, col, level),
                vertex(row + 1, col + 1, level),
                vertex(row + 1, col + 1, level + 1),
                vertex(row + 1, col, level + 1),
            )
        if col == 0 or not occupied[row, col - 1, level]:
            add_quad(
                vertex(row, col, level),
                vertex(row + 1, col, level),
                vertex(row + 1, col, level + 1),
                vertex(row, col, level + 1),
            )
        if col == num_cols - 1 or not occupied[row, col + 1, level]:
            add_quad(
                vertex(row, col + 1, level),
                vertex(row, col + 1, level + 1),
                vertex(row + 1, col + 1, level + 1),
                vertex(row + 1, col + 1, level),
            )
        if level == 0 or not occupied[row, col, level - 1]:
            add_quad(
                vertex(row, col, level),
                vertex(row, col + 1, level),
                vertex(row + 1, col + 1, level),
                vertex(row + 1, col, level),
            )
        if level == num_levels - 1 or not occupied[row, col, level + 1]:
            add_quad(
                vertex(row, col, level + 1),
                vertex(row + 1, col, level + 1),
                vertex(row + 1, col + 1, level + 1),
                vertex(row, col + 1, level + 1),
            )

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    return mesh, _height_field_origin(heights, terrain_size, horizontal_scale, vertical_scale)


def _stairs_mesh(
    difficulty: float,
    cfg: terrain_gen.HfPyramidStairsTerrainCfg,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Convert the exact discrete stair height field into closed box rings."""
    heights, terrain_size, horizontal_scale, vertical_scale = _height_field_with_border(
        difficulty, cfg
    )
    vertices, faces = convert_height_field_to_mesh(
        heights,
        horizontal_scale,
        vertical_scale,
        _effective_slope_threshold(cfg),
    )
    surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh = PerSubTerrainSlopeThresholdGenerator._heightfield_stairs_to_boxes(surface, cfg)
    return mesh, _height_field_origin(heights, terrain_size, horizontal_scale, vertical_scale)


def _closed_flat_mesh(
    difficulty: float,
    cfg: SubTerrainBaseCfg,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Generate a closed flat slab with its walkable surface at ``z=0``."""
    del difficulty
    size_x, size_y = (float(value) for value in cfg.size)
    mesh = trimesh.creation.box(
        extents=(size_x, size_y, MESH_BASE_THICKNESS),
        transform=trimesh.transformations.translation_matrix(
            (size_x * 0.5, size_y * 0.5, -MESH_BASE_THICKNESS * 0.5)
        ),
    )
    return mesh, np.array((size_x * 0.5, size_y * 0.5, 0.0), dtype=np.float64)


def _stairs_crease_segments(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Collect step creases of one stair tile, dropping slab and tile perimeter edges."""
    selected = np.degrees(mesh.face_adjacency_angles) > 30.0
    if not np.any(selected):
        return None
    segments = mesh.vertices[mesh.face_adjacency_edges[selected]]
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    segments = segments[lengths > 1.0e-6]
    bounds = mesh.bounds
    segments = segments[segments[:, :, 2].max(axis=1) > bounds[0, 2] + 1.0e-3]
    on_x = (np.abs(segments[:, :, 0] - bounds[0, 0]) < 1.0e-3) | (
        np.abs(segments[:, :, 0] - bounds[1, 0]) < 1.0e-3
    )
    on_y = (np.abs(segments[:, :, 1] - bounds[0, 1]) < 1.0e-3) | (
        np.abs(segments[:, :, 1] - bounds[1, 1]) < 1.0e-3
    )
    segments = segments[~((on_x[:, 0] & on_x[:, 1]) | (on_y[:, 0] & on_y[:, 1]))]
    return segments if len(segments) > 0 else None


class MixedMeshTerrainGenerator(PerSubTerrainSlopeThresholdGenerator):
    """Generate a closed triangular mesh for each configured mixed sub-terrain."""

    def __init__(self, cfg, device: str = "cpu"):
        self._stair_outline_chunks: list[np.ndarray] = []
        super().__init__(cfg=cfg, device=device)
        if self._stair_outline_chunks:
            segments = np.concatenate(self._stair_outline_chunks)
            transform = np.eye(4)
            transform[:2, -1] = -cfg.size[0] * cfg.num_rows * 0.5, -cfg.size[1] * cfg.num_cols * 0.5
            segments += transform[:3, -1]
            self.terrain_mesh.stair_outline_segments = segments

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        work_cfg = copy.deepcopy(cfg)
        if isinstance(work_cfg, terrain_gen.HfPyramidStairsTerrainCfg):
            mesh, origin = _stairs_mesh(difficulty, work_cfg)
        elif isinstance(work_cfg, terrain_gen.HfDiscreteObstaclesTerrainCfg):
            mesh, origin = _box_obstacle_mesh(difficulty, work_cfg)
        elif isinstance(work_cfg, DirectMeshTerrainCfg):
            meshes, origin = work_cfg.function(difficulty, work_cfg)
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(work_cfg, HfTerrainBaseCfg):
            mesh, origin = _closed_height_field_mesh(difficulty, work_cfg)
        else:
            mesh, origin = _closed_flat_mesh(difficulty, work_cfg)

        transform = np.eye(4)
        transform[:2, 3] = -work_cfg.size[0] * 0.5, -work_cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        return mesh, np.asarray(origin, dtype=np.float64) + transform[:3, 3]

    def _add_sub_terrain(
        self,
        mesh: trimesh.Trimesh,
        origin: np.ndarray,
        row: int,
        col: int,
        sub_terrain_cfg: SubTerrainBaseCfg,
    ) -> None:
        """Store step creases of stair tiles once the tile is placed in the world frame."""
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        if isinstance(sub_terrain_cfg, terrain_gen.HfPyramidStairsTerrainCfg):
            segments = _stairs_crease_segments(mesh)
            if segments is not None:
                self._stair_outline_chunks.append(segments)


MIXED_MESH_TERRAIN_CFG = copy.deepcopy(MIXED_TERRAIN_CFG)
MIXED_MESH_TERRAIN_CFG.class_type = MixedMeshTerrainGenerator
MIXED_MESH_TERRAIN_CFG.use_cache = False
MIXED_MESH_TERRAIN_CFG.color_scheme = "none"


def _rod_mesh(segments: np.ndarray, radius: float, sections: int) -> trimesh.Trimesh:
    """Build one capped tube per segment with a vectorized rod construction."""
    starts = segments[:, 0]
    directions = segments[:, 1] - starts
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / lengths
    helper = np.zeros_like(directions)
    helper[np.arange(len(directions)), np.argmin(np.abs(directions), axis=1)] = 1.0
    side = np.cross(directions, helper)
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    up = np.cross(directions, side)

    angles = np.arange(sections) * (2.0 * np.pi / sections)
    ring = (
        np.cos(angles)[None, :, None] * side[:, None, :]
        + np.sin(angles)[None, :, None] * up[:, None, :]
    )
    ring *= radius

    vertices = np.empty((len(segments), 2 * sections + 2, 3), dtype=np.float64)
    vertices[:, :sections] = starts[:, None, :] + ring
    vertices[:, sections : 2 * sections] = segments[:, 1, None, :] + ring
    vertices[:, 2 * sections] = starts
    vertices[:, 2 * sections + 1] = segments[:, 1]

    ring_index = np.arange(sections)
    next_index = (ring_index + 1) % sections
    rod_faces = np.concatenate(
        (
            np.stack((ring_index, next_index, sections + next_index), axis=1),
            np.stack((ring_index, sections + next_index, sections + ring_index), axis=1),
            np.stack((np.full(sections, 2 * sections), next_index, ring_index), axis=1),
            np.stack((np.full(sections, 2 * sections + 1), sections + ring_index, sections + next_index), axis=1),
        ),
        axis=0,
    )
    faces = rod_faces[None] + (np.arange(len(segments)) * (2 * sections + 2))[:, None, None]
    return trimesh.Trimesh(
        vertices=vertices.reshape(-1, 3).astype(np.float32),
        faces=faces.reshape(-1, 3).astype(np.int64),
        process=False,
    )


def _author_visual_mesh(path: str, mesh: trimesh.Trimesh, color: tuple[float, float, float]) -> None:
    """Author a uniformly colored, collision-free mesh prim."""
    from pxr import UsdGeom, UsdPhysics

    prim = UsdGeom.Mesh.Define(get_current_stage(), path)
    prim.CreatePointsAttr().Set(np.asarray(mesh.vertices, dtype=np.float32))
    prim.CreateFaceVertexIndicesAttr().Set(np.asarray(mesh.faces, dtype=np.int32).reshape(-1))
    prim.CreateFaceVertexCountsAttr().Set(np.full(len(mesh.faces), 3, dtype=np.int32))
    prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    prim.CreateDisplayColorAttr().Set([color])
    prim.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.constant)
    prim.CreateDisplayOpacityAttr().Set([1.0])
    prim.GetDisplayOpacityPrimvar().SetInterpolation(UsdGeom.Tokens.constant)
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim()).CreateCollisionEnabledAttr().Set(False)


class MixedTerrainImporter(TerrainImporter):
    """Import the closed mixed-terrain collider with neutral material and stair outlines."""

    outline_color: tuple[float, float, float] = (0.004, 0.007, 0.012)
    outline_radius: float = 0.01
    outline_sections: int = 6

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        """Import the collider and outline the stair creases."""
        super().import_mesh(name, mesh)
        segments = getattr(mesh, "stair_outline_segments", None)
        if segments is None or len(segments) == 0:
            return
        _author_visual_mesh(
            f"{self.cfg.prim_path}/{name}/outlines",
            _rod_mesh(segments, self.outline_radius, self.outline_sections),
            self.outline_color,
        )



class MixedPlayTerrainImporter(MixedTerrainImporter, HighToLowTerrainImporter):
    """Use the outlined neutral mixed terrain with deterministic play tile ordering."""


__all__ = [
    "MIXED_MESH_TERRAIN_CFG",
    "MixedMeshTerrainGenerator",
    "MixedPlayTerrainImporter",
    "MixedTerrainImporter",
]
