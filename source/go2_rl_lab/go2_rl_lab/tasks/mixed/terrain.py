"""Height-field generation backend and shared terrain configuration for the Go2 mixed task."""

from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh, height_field_to_mesh
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.utils.configclass import configclass


MESH_BASE_THICKNESS = 0.10
SPAWN_CENTER_HALF_EXTENT = 1.0
"""Half extent of the centered spawn region used to derive the terrain origin height."""


class PerSubTerrainSlopeThresholdGenerator(TerrainGenerator):
    """Terrain generator with RobotLab's per-sub-terrain slope handling."""

    def __init__(self, cfg, device: str = "cpu"):
        self._apply_sub_terrain_border_width(cfg)
        super().__init__(cfg, device)

    @staticmethod
    def _apply_sub_terrain_border_width(cfg) -> None:
        border_width = getattr(cfg, "sub_terrain_border_width", None)
        if border_width is None:
            return
        for sub_cfg in cfg.sub_terrains.values():
            if hasattr(sub_cfg, "border_width"):
                sub_cfg.border_width = border_width

    def _get_terrain_mesh(self, difficulty, cfg):
        override = getattr(cfg, "slope_threshold_override", None)
        stride = int(getattr(self.cfg, "mesh_simplification_stride", 1))
        stairs_representation = getattr(self.cfg, "stairs_representation", "heightfield")
        if stairs_representation not in ("heightfield", "boxes"):
            raise ValueError(
                "stairs_representation must be either 'heightfield' or 'boxes', "
                f"got {stairs_representation!r}."
            )
        original_slope_threshold = getattr(cfg, "slope_threshold", None)
        if override is not None:
            cfg.slope_threshold = override
        try:
            use_box_stairs = stairs_representation == "boxes" and "stair" in type(cfg).__name__.lower()
            if stride > 1 or use_box_stairs:
                sparse_mesh = self._generate_heightfield_mesh(difficulty, cfg)
            else:
                sparse_mesh = None
            if sparse_mesh is None:
                mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
            else:
                mesh, origin = sparse_mesh
        finally:
            if override is not None:
                cfg.slope_threshold = original_slope_threshold

        if use_box_stairs:
            mesh = self._heightfield_stairs_to_boxes(mesh, cfg)
        elif stride > 1:
            mesh = self._simplify_heightfield_mesh(mesh, cfg, stride)
        return mesh, origin

    @staticmethod
    def _generate_heightfield_mesh(difficulty, cfg):
        """Rebuild a height-field mesh without trimesh vertex deduplication.

        Isaac Lab's ``height_field_to_mesh`` decorator constructs the regular
        grid and then lets ``trimesh`` process it, which merges vertices around
        vertical stair/obstacle faces.  The merged topology is inconvenient for
        structured decimation.  This is the same conversion with
        ``process=False``; all heights, slope correction, borders and origins
        are unchanged.
        """
        horizontal_scale = getattr(cfg, "horizontal_scale", None)
        vertical_scale = getattr(cfg, "vertical_scale", None)
        if horizontal_scale is None or vertical_scale is None or not hasattr(cfg, "border_width"):
            return None

        function = cfg.function
        if hasattr(function, "_resolve"):
            function = function._resolve()
        raw_function = getattr(function, "__wrapped__", function)
        if not callable(raw_function):
            return None

        work_cfg = cfg.copy()
        width_pixels = int(work_cfg.size[0] / horizontal_scale) + 1
        length_pixels = int(work_cfg.size[1] / horizontal_scale) + 1
        border_pixels = int(work_cfg.border_width / horizontal_scale) + 1
        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        sub_terrain_size = [
            (width_pixels - 2 * border_pixels) * horizontal_scale,
            (length_pixels - 2 * border_pixels) * horizontal_scale,
        ]
        terrain_size = work_cfg.size
        work_cfg.size = tuple(sub_terrain_size)
        z_gen, _ = raw_function(difficulty, work_cfg)
        heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = z_gen

        vertices, triangles = convert_height_field_to_mesh(
            heights, horizontal_scale, vertical_scale, work_cfg.slope_threshold
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        x1 = int((terrain_size[0] * 0.5 - 1) / horizontal_scale)
        x2 = int((terrain_size[0] * 0.5 + 1) / horizontal_scale)
        y1 = int((terrain_size[1] * 0.5 - 1) / horizontal_scale)
        y2 = int((terrain_size[1] * 0.5 + 1) / horizontal_scale)
        origin = np.array(
            [0.5 * terrain_size[0], 0.5 * terrain_size[1], np.max(heights[x1:x2, y1:y2]) * vertical_scale]
        )
        transform = np.eye(4)
        transform[0:2, -1] = -terrain_size[0] * 0.5, -terrain_size[1] * 0.5
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]
        return mesh, origin

    @staticmethod
    def _simplify_heightfield_mesh(mesh: trimesh.Trimesh, cfg, stride: int) -> trimesh.Trimesh:
        """Coarsen regular height-field meshes while retaining sharp terrain edges."""
        horizontal_scale = float(getattr(cfg, "horizontal_scale", 0.0))
        size = getattr(cfg, "size", None)
        if horizontal_scale <= 0.0 or size is None:
            return mesh

        num_rows = int(size[0] / horizontal_scale) + 1
        num_cols = int(size[1] / horizontal_scale) + 1
        if len(mesh.vertices) != num_rows * num_cols:
            # Mesh terrains (for example the flat plane) do not use the
            # height-field layout and are already minimal.
            return mesh

        grid = np.asarray(mesh.vertices).reshape(num_rows, num_cols, 3)
        row_indices = np.arange(0, num_rows, stride, dtype=np.int64)
        col_indices = np.arange(0, num_cols, stride, dtype=np.int64)
        if row_indices[-1] != num_rows - 1:
            row_indices = np.append(row_indices, num_rows - 1)
        if col_indices[-1] != num_cols - 1:
            col_indices = np.append(col_indices, num_cols - 1)

        # Preserve every grid line next to a real height discontinuity for
        # stairs and obstacles.  Continuous waves/slopes remain regular.
        cfg_name = type(cfg).__name__.lower()
        if "stair" not in cfg_name and "obstacle" not in cfg_name:
            # Keep wave, rough-slope and pyramid-slope meshes at their native
            # 10 cm sampling.  Their curvature/noise is part of the policy's
            # observation distribution and is not safely represented by a
            # coarse linear patch without a model-performance revalidation.
            return mesh

        heights = grid[:, :, 2]
        edge_threshold = max(0.04, 2.0 * float(getattr(cfg, "vertical_scale", 0.005)))
        row_edges = np.flatnonzero(np.any(np.abs(np.diff(heights, axis=0)) >= edge_threshold, axis=1))
        col_edges = np.flatnonzero(np.any(np.abs(np.diff(heights, axis=1)) >= edge_threshold, axis=0))
        row_indices = np.unique(np.concatenate((row_indices, row_edges, row_edges + 1)))
        col_indices = np.unique(np.concatenate((col_indices, col_edges, col_edges + 1)))

        coarse = grid[np.ix_(row_indices, col_indices)]
        coarse_rows, coarse_cols = coarse.shape[:2]
        vertices = coarse.reshape(-1, 3)
        cell_rows = np.arange(coarse_rows - 1, dtype=np.int64)[:, None]
        cell_cols = np.arange(coarse_cols - 1, dtype=np.int64)[None, :]
        base = cell_rows * coarse_cols + cell_cols
        next_row = base + coarse_cols
        faces = np.stack(
            (
                np.stack((base, next_row + 1, base + 1), axis=-1),
                np.stack((base, next_row, next_row + 1), axis=-1),
            ),
            axis=2,
        ).reshape(-1, 3)
        # Slope correction can place two neighboring samples at the same
        # 3-D point on a sharp edge.  Such zero-area triangles carry no
        # collision or raycast surface and should not be uploaded.
        face_vertices = vertices[faces]
        cross = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
        faces = faces[np.linalg.norm(cross, axis=1) > 1.0e-9]
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    @staticmethod
    def _heightfield_stairs_to_boxes(mesh: trimesh.Trimesh, cfg) -> trimesh.Trimesh:
        """Represent the quantized pyramid-stair surface with concentric box rings."""
        horizontal_scale = float(getattr(cfg, "horizontal_scale", 0.0))
        vertical_scale = float(getattr(cfg, "vertical_scale", 0.0))
        size = getattr(cfg, "size", None)
        if horizontal_scale <= 0.0 or vertical_scale <= 0.0 or size is None:
            raise ValueError("Box stairs require positive terrain scales and a configured size.")

        num_rows = int(size[0] / horizontal_scale) + 1
        num_cols = int(size[1] / horizontal_scale) + 1
        if len(mesh.vertices) != num_rows * num_cols:
            raise ValueError("Box stairs require a regular height-field mesh.")

        grid = np.asarray(mesh.vertices).reshape(num_rows, num_cols, 3)
        heights = np.unique(grid[:, :, 2])
        levels = []
        for height in heights:
            level_vertices = grid[np.isclose(grid[:, :, 2], height, atol=1.0e-7)]
            bounds = np.array(
                [
                    np.min(level_vertices[:, 0]),
                    np.max(level_vertices[:, 0]),
                    np.min(level_vertices[:, 1]),
                    np.max(level_vertices[:, 1]),
                ]
            )
            levels.append((float(height), bounds))

        # The generated pyramid consists of nested squares.  Sorting by area
        # reconstructs the visible outer-to-inner stair rings independently of
        # whether the terrain climbs upward or downward toward its center.
        levels.sort(
            key=lambda level: (level[1][1] - level[1][0]) * (level[1][3] - level[1][2]),
            reverse=True,
        )
        tolerance = max(1.0e-6, 0.01 * horizontal_scale)
        for (_, outer), (_, inner) in zip(levels, levels[1:]):
            if not (
                outer[0] < inner[0] - tolerance
                and outer[1] > inner[1] + tolerance
                and outer[2] < inner[2] - tolerance
                and outer[3] > inner[3] + tolerance
            ):
                raise ValueError("Box stairs require strictly nested rectangular height levels.")

        bottom_z = min(height for height, _ in levels) - vertical_scale
        boxes: list[trimesh.Trimesh] = []

        def add_box(x_min: float, x_max: float, y_min: float, y_max: float, top_z: float) -> None:
            if x_max - x_min <= tolerance or y_max - y_min <= tolerance:
                return
            extents = (x_max - x_min, y_max - y_min, top_z - bottom_z)
            center = (0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (top_z + bottom_z))
            boxes.append(
                trimesh.creation.box(
                    extents=extents,
                    transform=trimesh.transformations.translation_matrix(center),
                )
            )

        def add_triangular_prism(top_vertices: np.ndarray) -> None:
            """Extrude one slope-correction corner triangle down to the common base."""
            top_vertices = top_vertices.copy()
            if np.cross(top_vertices[1] - top_vertices[0], top_vertices[2] - top_vertices[0])[2] < 0.0:
                top_vertices[[1, 2]] = top_vertices[[2, 1]]
            bottom_vertices = top_vertices.copy()
            bottom_vertices[:, 2] = bottom_z
            vertices = np.concatenate((top_vertices, bottom_vertices))
            faces = np.array(
                [
                    [0, 1, 2],
                    [3, 5, 4],
                    [0, 3, 4],
                    [0, 4, 1],
                    [1, 4, 5],
                    [1, 5, 2],
                    [2, 5, 3],
                    [2, 3, 0],
                ],
                dtype=np.int64,
            )
            boxes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))

        face_vertices = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        face_centers = np.mean(face_vertices[:, :, :2], axis=1)
        horizontal_faces = np.ptp(face_vertices[:, :, 2], axis=1) < 1.0e-7
        for (top_z, outer), (_, inner) in zip(levels, levels[1:]):
            x_min, x_max, y_min, y_max = outer
            inner_x_min, inner_x_max, inner_y_min, inner_y_max = inner
            add_box(x_min, x_max, y_min, inner_y_min, top_z)
            add_box(x_min, x_max, inner_y_max, y_max, top_z)
            add_box(x_min, inner_x_min, inner_y_min, inner_y_max, top_z)
            add_box(inner_x_max, x_max, inner_y_min, inner_y_max, top_z)

            # Inverted height-field stairs retain two diagonal half-cells per
            # level because of Isaac Lab's corner slope correction.  Preserve
            # those exact triangles instead of expanding the lower box top
            # through them.
            corner_faces = (
                horizontal_faces
                & np.all(np.isclose(face_vertices[:, :, 2], top_z, atol=1.0e-7), axis=1)
                & (face_centers[:, 0] > inner_x_min + tolerance)
                & (face_centers[:, 0] < inner_x_max - tolerance)
                & (face_centers[:, 1] > inner_y_min + tolerance)
                & (face_centers[:, 1] < inner_y_max - tolerance)
            )
            for triangle in face_vertices[corner_faces]:
                add_triangular_prism(triangle)

        center_height, center = levels[-1]
        add_box(center[0], center[1], center[2], center[3], center_height)
        return trimesh.util.concatenate(boxes)


def flat_slab_terrain(
    difficulty: float,
    cfg: MeshFlatSlabTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create a closed flat slab whose top surface is at z=0."""
    del difficulty
    transform = trimesh.transformations.translation_matrix(
        (cfg.size[0] / 2.0, cfg.size[1] / 2.0, -cfg.thickness / 2.0)
    )
    slab = trimesh.creation.box(
        extents=(cfg.size[0], cfg.size[1], cfg.thickness),
        transform=transform,
    )
    origin = np.array((cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0))
    return [slab], origin


@configclass
class MeshFlatSlabTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a volume-bearing flat mesh compatible with Newton."""

    function = flat_slab_terrain

    thickness: float = 0.1
    """Thickness below z=0 in meters."""


_COLUMN_BOX_FACES = np.array(
    (
        (0, 3, 2),
        (0, 2, 1),  # bottom (-z)
        (4, 5, 6),
        (4, 6, 7),  # top (+z)
        (0, 4, 7),
        (0, 7, 3),  # -x
        (1, 2, 6),
        (1, 6, 5),  # +x
        (0, 1, 5),
        (0, 5, 4),  # -y
        (3, 7, 6),
        (3, 6, 2),  # +y
    ),
    dtype=np.int64,
)


def _column_boxes_mesh(
    heights: np.ndarray,
    cell_size: float,
    base_z: float,
) -> trimesh.Trimesh:
    """Build one closed box per flat-topped column, vectorized over the grid.

    Column walls are not merged with their neighbours: adjacent boxes simply
    share coincident wall planes. This keeps every box watertight with exact
    flat tops and avoids the T-junctions of a merged column surface.
    """
    num_rows, num_cols = heights.shape
    rows = np.arange(num_rows, dtype=np.float64)[:, None]
    cols = np.arange(num_cols, dtype=np.float64)[None, :]
    x0 = np.broadcast_to(rows * cell_size, heights.shape)
    x1 = x0 + cell_size
    y0 = np.broadcast_to(cols * cell_size, heights.shape)
    y1 = y0 + cell_size
    z0 = np.full(heights.shape, float(base_z))
    z1 = heights.astype(np.float64)

    corners = np.stack(
        (
            np.stack((x0, y0, z0), axis=-1),
            np.stack((x1, y0, z0), axis=-1),
            np.stack((x1, y1, z0), axis=-1),
            np.stack((x0, y1, z0), axis=-1),
            np.stack((x0, y0, z1), axis=-1),
            np.stack((x1, y0, z1), axis=-1),
            np.stack((x1, y1, z1), axis=-1),
            np.stack((x0, y1, z1), axis=-1),
        ),
        axis=-2,
    )
    vertices = corners.reshape(-1, 3).astype(np.float32)
    offsets = (np.arange(num_rows * num_cols, dtype=np.int64) * 8)[:, None, None]
    faces = (_COLUMN_BOX_FACES[None] + offsets).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def wheels_terrain(
    difficulty: float,
    cfg: WheelsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a field of flat tyres around a tyre-free spawn disc.

    Tyres are placed on a square grid whose pitch keeps neighbouring tyres in
    contact. Every tyre overlapping the configured clear radius around the
    terrain center is skipped, so the robot spawns on flat ground and the tyres
    fill the rest of the tile. Inner radius grows and heights interpolate with
    the difficulty level.
    """
    level = float(np.clip(difficulty, 0.0, 1.0))
    size_x, size_y = (float(value) for value in cfg.size)
    center = np.array((size_x * 0.5, size_y * 0.5))
    ground = trimesh.creation.box(
        extents=(size_x, size_y, MESH_BASE_THICKNESS),
        transform=trimesh.transformations.translation_matrix(
            (center[0], center[1], -MESH_BASE_THICKNESS * 0.5)
        ),
    )
    meshes = [ground]

    inner_radius = float(np.interp(level, (0.0, 1.0), cfg.inner_radius_range))
    height_range = (1.0 - level) * np.asarray(cfg.easy_height_range) + level * np.asarray(
        cfg.height_range
    )
    pitch = 2.0 * cfg.outer_radius + level * cfg.max_gap
    skip_radius = cfg.clear_radius + cfg.outer_radius
    origin_height = 0.0
    num_cells = int(np.ceil(max(size_x, size_y) / pitch)) + 1
    for row in range(-num_cells, num_cells + 1):
        for col in range(-num_cells, num_cells + 1):
            offset = np.array(((row + 0.5) * pitch, (col + 0.5) * pitch))
            if np.linalg.norm(offset) < skip_radius:
                continue
            position = center + offset
            if not (
                cfg.outer_radius <= position[0] <= size_x - cfg.outer_radius
                and cfg.outer_radius <= position[1] <= size_y - cfg.outer_radius
            ):
                continue
            height = float(np.random.uniform(*height_range))
            tyre = trimesh.creation.annulus(
                r_min=inner_radius,
                r_max=cfg.outer_radius,
                height=height,
                sections=cfg.mesh_segments,
            )
            tyre.apply_translation((position[0], position[1], height * 0.5))
            meshes.append(tyre)
            if (
                abs(position[0] - center[0])
                <= SPAWN_CENTER_HALF_EXTENT + cfg.outer_radius
                and abs(position[1] - center[1])
                <= SPAWN_CENTER_HALF_EXTENT + cfg.outer_radius
            ):
                origin_height = max(origin_height, height)
    return meshes, np.array((center[0], center[1], origin_height))


def small_squares_terrain(
    difficulty: float,
    cfg: SmallSquaresTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a closed block terrain of randomly raised square columns.

    Every square of ``cfg.square_size`` fills the whole tile and its height is
    sampled uniformly from ``[0, x]``, where ``x`` grows linearly from
    ``height_range[0]`` to ``height_range[1]`` with the difficulty level.
    """
    level = float(np.clip(difficulty, 0.0, 1.0))
    size_x, size_y = (float(value) for value in cfg.size)
    max_height = float(np.interp(level, (0.0, 1.0), cfg.height_range))
    num_rows = int(round(size_x / cfg.square_size))
    num_cols = int(round(size_y / cfg.square_size))
    heights = np.random.uniform(0.0, max_height, size=(num_rows, num_cols))
    mesh = _column_boxes_mesh(heights, cfg.square_size, -MESH_BASE_THICKNESS)

    half = max(1, int(round(SPAWN_CENTER_HALF_EXTENT / cfg.square_size)))
    center_row, center_col = num_rows // 2, num_cols // 2
    center_heights = heights[
        max(0, center_row - half) : min(num_rows, center_row + half),
        max(0, center_col - half) : min(num_cols, center_col + half),
    ]
    origin_height = float(np.max(center_heights))
    return [mesh], np.array((size_x * 0.5, size_y * 0.5, origin_height))


@configclass
class DirectMeshTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for sub-terrains generated directly as mesh solids."""


@configclass
class WheelsTerrainCfg(DirectMeshTerrainCfg):
    """Configuration for a field of flat tyres around a tyre-free spawn disc."""

    function = wheels_terrain

    outer_radius: float = 0.30
    """Tyre outer radius in meters (0.6 m diameter)."""

    inner_radius_range: tuple[float, float] = (0.05, 0.25)
    """Tyre hole radius at difficulty 0 and 1; grows with the difficulty level."""

    easy_height_range: tuple[float, float] = (0.01, 0.03)
    """Per-tyre height range at difficulty 0."""

    height_range: tuple[float, float] = (0.15, 0.17)
    """Per-tyre height range at difficulty 1; interpolated from ``easy_height_range``."""

    max_gap: float = 0.0
    """Extra center-to-center gap at difficulty 1; 0 keeps neighbouring tyres in contact."""

    clear_radius: float = 1.0
    """Radius of the tyre-free disc around the terrain center."""

    mesh_segments: int = 32
    """Number of circumferential tyre segments."""


@configclass
class SmallSquaresTerrainCfg(DirectMeshTerrainCfg):
    """Configuration for a block terrain of randomly raised 10 cm squares."""

    function = small_squares_terrain

    square_size: float = 0.1
    """Edge length of every square column in meters."""

    height_range: tuple[float, float] = (0.01, 0.17)
    """Maximum square height at difficulty 0 and 1; heights are uniform in ``[0, x]``."""


@configclass
class Go2TerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    """RobotLab terrain-generator options used by the mixed task."""

    class_type: type = PerSubTerrainSlopeThresholdGenerator
    sub_terrain_border_width: float | None = None
    # Keep the native 10 cm height-field mesh. Coarsening stair and obstacle
    # meshes changes Newton contact-force statistics at sharp boundaries.
    mesh_simplification_stride: int = 1
    # Preserve the original height-field stair surface instead of rebuilding
    # it from box rings, which amplifies calf contacts under Newton.
    stairs_representation: str = "heightfield"


def with_slope_threshold(sub_terrain_cfg, slope_threshold: float | None):
    """Attach a slope-threshold override to one height-field terrain."""
    sub_terrain_cfg.slope_threshold_override = slope_threshold
    return sub_terrain_cfg


def _height_field_with_origin(heights: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Attach a centered spawn origin to a raw height field for Isaac Lab 3.0."""
    center_x, center_y = (size // 2 for size in heights.shape)
    radius = max(1, int(SPAWN_CENTER_HALF_EXTENT / cfg.horizontal_scale))
    spawn_height = np.max(
        heights[max(0, center_x - radius) : center_x + radius, max(0, center_y - radius) : center_y + radius]
    )
    origin = np.array(
        [
            (heights.shape[0] - 1) * cfg.horizontal_scale / 2,
            (heights.shape[1] - 1) * cfg.horizontal_scale / 2,
            spawn_height * cfg.vertical_scale,
        ]
    )
    return heights, origin


@height_field_to_mesh
def robotlab_wave_terrain(difficulty: float, cfg) -> tuple[np.ndarray, np.ndarray]:
    """RobotLab wave terrain: sinusoidal waves plus fine roughness."""
    wave, _ = hf_terrains.wave_terrain.__wrapped__(difficulty, cfg)
    rough, _ = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return _height_field_with_origin(np.rint(wave + rough).astype(np.int16), cfg)


@height_field_to_mesh
def robotlab_rough_slope_terrain(difficulty: float, cfg) -> tuple[np.ndarray, np.ndarray]:
    """RobotLab rough slope terrain: pyramid slope plus fine roughness."""
    slope, _ = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    rough, _ = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return _height_field_with_origin(np.rint(slope + rough).astype(np.int16), cfg)


@configclass
class WaveTerrainCfg(terrain_gen.HfWaveTerrainCfg):
    function = robotlab_wave_terrain
    amplitude_range: tuple[float, float] = (0.1, 0.28)
    num_waves: int = 5
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


@configclass
class RoughSlopeTerrainCfg(terrain_gen.HfPyramidSlopedTerrainCfg):
    function = robotlab_rough_slope_terrain
    slope_range: tuple[float, float] = (0.1, 0.568)
    platform_width: float = 3.0
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


# Shared RobotLab terrain configuration for the mixed-terrain task.
MIXED_TERRAIN_CFG = Go2TerrainGeneratorCfg(
    size=(9.0, 9.0),
    border_width=25.0,
    sub_terrain_border_width=0.5,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    # Global default slope correction: 0.75 is approximately 36.9 degrees.
    # Each height-field terrain below overrides this with its recommended value.
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "wave": with_slope_threshold(
            WaveTerrainCfg(proportion=0.05),
            10.0,  # effectively disable slope correction for wave terrain
        ),
        "slope_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.1, 0.568),
                platform_width=3.0,
            ),
            10.0,  # effectively disable slope correction for slope-up terrain
        ),
        "slope_down": with_slope_threshold(
            terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.1, 0.568),
                platform_width=3.0,
            ),
            10.0,  # effectively disable slope correction for slope-down terrain
        ),
        "rough_slope": with_slope_threshold(
            RoughSlopeTerrainCfg(proportion=0.05),
            10.0,  # effectively disable slope correction for rough-slope terrain
        ),
        "stairs_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                proportion=0.25,
                step_height_range=(0.05, 0.257),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25,  # preserve sharp stair edges with approximately 14.0-degree correction
        ),
        "stairs_down": with_slope_threshold(
            terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.05, 0.257),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25,  # preserve sharp stair edges with approximately 14.0-degree correction
        ),
        "obstacles": with_slope_threshold(
            terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.10,
                obstacle_width_range=(1.0, 2.0),
                obstacle_height_range=(0.05, 0.275),
                num_obstacles=20,
                platform_width=3.0,
            ),
            0.25,  # preserve obstacle edges with approximately 14.0-degree correction
        ),
        # Mesh terrains: height-field slope correction does not apply.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "wheels": WheelsTerrainCfg(proportion=0.05),
        "small_squares": SmallSquaresTerrainCfg(proportion=0.05),
    },
)
