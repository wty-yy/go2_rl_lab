"""MuJoCo Warp manager with explicit hard per-world resets."""

from __future__ import annotations

import torch
import warp as wp
from newton import Model, ModelBuilder, ShapeFlags

from isaaclab.physics import PhysicsManager
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonManager

from .randomize import apply_body_com_buckets


@wp.kernel(enable_backward=False)
def _mark_hard_reset_worlds(
    env_ids: wp.array(dtype=wp.int64),
    reset_mask: wp.array(dtype=wp.bool),
):
    """Accumulate sparse environment IDs into a persistent boolean mask."""
    reset_mask[env_ids[wp.tid()]] = True


class NewtonMJWarpResettableManager(NewtonMJWarpManager):
    """Add an explicit hard reset path without changing normal environment resets.

    The class name intentionally starts with ``Newton``. Isaac Lab uses the
    physics-manager class name as its backend identifier and dispatches Newton
    scene, view, and factory implementations with ``startswith("newton")``.

    Normal episode resets remain lightweight and use the upstream Newton behavior:
    reset root/joint and manager state, recompute FK, and synchronize coordinates on
    the next solver step. Call :meth:`request_hard_reset` only after detecting a
    corrupted world. The hard path additionally clears MuJoCo Warp acceleration,
    warm-start, force, contact, and other per-world solver data.

    Requested hard resets are queued until the next physics step so they run at a
    safe solver boundary. ``SolverMuJoCo.step`` then copies the freshly reset Newton
    coordinates and velocities into MuJoCo Warp before advancing physics.
    """

    _hard_reset_mask: wp.array | None = None
    _has_pending_hard_reset = False

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Apply sensor-BVH filtering and CoM buckets before model finalization."""
        super()._prepare_builder_for_finalize(builder)
        cls._restore_static_mesh_visibility(builder)
        solver_cfg = PhysicsManager._cfg.solver_cfg
        if solver_cfg.sensor_bvh_shape_flags is not None:
            builder.default_bvh_cfg.shape_flags = ShapeFlags(solver_cfg.sensor_bvh_shape_flags)
        if solver_cfg.body_com_buckets is not None:
            apply_body_com_buckets(builder, solver_cfg.body_com_buckets)

    @staticmethod
    def _restore_static_mesh_visibility(builder: ModelBuilder) -> None:
        """Keep USD-visible terrain meshes visible alongside their decorative outlines.

        Newton groups static shapes under body -1, so importing a visual outline
        can hide the terrain collider as though the outline replaced its surface.
        Restore only mesh colliders that USD explicitly allows in the viewport.
        """
        from pxr import UsdGeom

        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        for index, (body, label) in enumerate(zip(builder.shape_body, builder.shape_label)):
            flags = builder.shape_flags[index]
            if not label or body != -1 or flags & ShapeFlags.VISIBLE or not flags & ShapeFlags.COLLIDE_SHAPES:
                continue
            prim = stage.GetPrimAtPath(label)
            if not prim or not prim.IsA(UsdGeom.Mesh):
                continue
            imageable = UsdGeom.Imageable(prim)
            if (
                imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
                and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy)
            ):
                builder.shape_flags[index] |= ShapeFlags.VISIBLE

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        if solver_cfg.use_mujoco_cpu:
            raise ValueError(f"{cls.__name__} only supports the MuJoCo Warp backend, not use_mujoco_cpu=True.")
        if solver_cfg.update_data_interval != 1:
            raise ValueError(
                f"{cls.__name__} requires update_data_interval=1 so reset Newton state is synchronized "
                "to MuJoCo Warp before the next physics step."
            )
        super()._build_solver(model, solver_cfg)

    @classmethod
    def step(cls) -> None:
        sim = PhysicsManager._sim
        if sim is None or not sim.is_playing():
            return super().step()
        if cls._has_pending_hard_reset:
            cls._apply_pending_hard_resets()
        super().step()

    @classmethod
    def request_hard_reset(cls, env_ids: torch.Tensor) -> None:
        """Queue MuJoCo Warp solver-data resets for the selected environments."""
        if env_ids.numel() == 0:
            return

        world_reset_mask = NewtonManager._world_reset_mask
        solver = NewtonManager._solver
        if world_reset_mask is None or solver is None:
            raise RuntimeError("Cannot request a hard reset before Newton world masks are initialized.")

        # Newton includes a global-world sentinel; MuJoCo accepts only simulated worlds.
        reset_shape = (solver.mjw_data.nworld,)
        hard_reset_mask = cls._hard_reset_mask
        if (
            hard_reset_mask is None
            or hard_reset_mask.shape != reset_shape
            or hard_reset_mask.device != world_reset_mask.device
        ):
            hard_reset_mask = wp.zeros(reset_shape, dtype=wp.bool, device=world_reset_mask.device)
            cls._hard_reset_mask = hard_reset_mask

        env_ids = env_ids.to(device=str(world_reset_mask.device), dtype=torch.int64)
        with wp.ScopedDevice(world_reset_mask.device):
            wp.launch(
                _mark_hard_reset_worlds,
                dim=env_ids.shape[0],
                inputs=[env_ids],
                outputs=[hard_reset_mask],
                device=world_reset_mask.device,
            )
        cls._has_pending_hard_reset = True

    @classmethod
    def _apply_pending_hard_resets(cls) -> None:
        hard_reset_mask = cls._hard_reset_mask
        solver = NewtonManager._solver
        if hard_reset_mask is None or solver is None:
            cls._has_pending_hard_reset = False
            return

        with wp.ScopedDevice(hard_reset_mask.device):
            _, mujoco_warp = solver.import_mujoco()
            mujoco_warp.reset_data(solver.mjw_model, solver.mjw_data, hard_reset_mask)
            hard_reset_mask.zero_()
        cls._has_pending_hard_reset = False

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._hard_reset_mask = None
        cls._has_pending_hard_reset = False
        super()._solver_specific_clear()


__all__ = ["NewtonMJWarpResettableManager"]
