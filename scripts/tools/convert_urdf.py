"""Convert a URDF to a self-contained USD with articulation self-collision enabled.

Quick Start:
    python scripts/tools/convert_urdf.py robot.urdf robot.usd

Full Command:
    python scripts/tools/convert_urdf.py \
        source/go2_rl_lab/go2_rl_lab/assets/urdf/go2.urdf \
        source/go2_rl_lab/go2_rl_lab/assets/usd/go2.usd \
        --ros-package go2=source/go2_rl_lab/go2_rl_lab/assets/urdf --viz none

Notes:
    Requires the configured Isaac Lab 3.0 environment.
    --no-self-collision explicitly disables articulation self-collision.
    Parent-child joint collision filtering remains authored by the importer.
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path, help="Input URDF path.")
parser.add_argument("output", type=Path, help="Output USD path.")
parser.add_argument(
    "--merge-fixed-joints",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Merge links connected by fixed joints (default: enabled).",
)
parser.add_argument(
    "--self-collision",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable articulation self-collision in the exported USD (default: enabled).",
)
parser.add_argument(
    "--ros-package",
    action="append",
    default=[],
    metavar="NAME=PATH",
    help="Resolve a package:// URL using a ROS package name and directory. May be repeated.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _resolve_ros_packages(values: list[str]) -> list[dict[str, str]]:
    """Parse and validate ``NAME=PATH`` ROS package mappings."""
    mappings = []
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Invalid ROS package mapping {value!r}; expected NAME=PATH.")
        package_path = Path(raw_path).expanduser().resolve()
        if not package_path.is_dir():
            raise FileNotFoundError(f"ROS package directory does not exist: {package_path}")
        mappings.append({"name": name, "path": str(package_path)})
    return mappings


def _merge_unprotected_fixed_joints(input_path: Path, temporary_dir: Path) -> tuple[Path, int]:
    """Merge fixed joints while preserving URDF joints marked ``dont_collapse``."""
    import omni.kit.app

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    if not extension_manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
        extension_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    from isaacsim.asset.importer.urdf.impl.urdf_utils import merge_fixed_joints

    protected_joint_type = "fixed_preserved_by_exporter"
    marked_path = temporary_dir / "marked.urdf"
    merged_path = temporary_dir / "merged.urdf"

    tree = ET.parse(input_path)
    protected_count = 0
    for joint in tree.getroot().findall("joint"):
        if joint.get("type") == "fixed" and joint.get("dont_collapse", "false").lower() == "true":
            joint.set("type", protected_joint_type)
            protected_count += 1
    tree.write(marked_path, xml_declaration=True, encoding="UTF-8")

    merge_fixed_joints(str(marked_path), str(merged_path))

    tree = ET.parse(merged_path)
    for joint in tree.getroot().findall("joint"):
        if joint.get("type") == protected_joint_type:
            joint.set("type", "fixed")
            joint.attrib.pop("dont_collapse", None)
    tree.write(merged_path, xml_declaration=True, encoding="UTF-8")
    return merged_path, protected_count


def _normalize_rigid_body_hierarchy(stage) -> int:
    """Move rigid-body links below the default prim while preserving their authored poses."""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError("Generated USD has no default prim.")

    body_paths = [prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_relative_transforms = {
        path: xform_cache.ComputeRelativeTransform(stage.GetPrimAtPath(path), root)[0] for path in body_paths
    }
    for source_path in sorted(body_paths, key=lambda path: path.pathElementCount, reverse=True):
        prim = stage.GetPrimAtPath(source_path)
        if not prim or prim.GetParent() == root:
            continue
        destination_path = root.GetPath().AppendChild(prim.GetName())
        if stage.GetPrimAtPath(destination_path):
            raise RuntimeError(f"Cannot normalize duplicate rigid-body name: {destination_path}")
        editor = Usd.NamespaceEditor(stage)
        editor.MovePrimAtPath(source_path, destination_path)
        can_apply = editor.CanApplyEdits()
        if not can_apply:
            raise RuntimeError(f"Cannot move {source_path} to {destination_path}: {can_apply}")
        if not editor.ApplyEdits():
            raise RuntimeError(f"Failed to move {source_path} to {destination_path}")

        moved_prim = stage.GetPrimAtPath(destination_path)
        xformable = UsdGeom.Xformable(moved_prim)
        xformable.ClearXformOpOrder()
        xformable.SetResetXformStack(False)
        xformable.AddTransformOp(
            precision=UsdGeom.XformOp.PrecisionDouble,
            opSuffix="reparent",
        ).Set(root_relative_transforms[source_path])

    geometry_path = root.GetPath().AppendChild("Geometry")
    geometry_prim = stage.GetPrimAtPath(geometry_path)
    if geometry_prim and not geometry_prim.GetChildren():
        stage.RemovePrim(geometry_path)
    return len(body_paths)


def _normalize_visual_hierarchy(stage) -> int:
    """Rename importer-created per-body visual nodes to ``body/visuals``.

    Apart from matching the previous Go2 hierarchy, this prevents broad rigid-body sensor patterns
    from attempting to treat a visual Xform such as ``Robot/base/base`` as another rigid body.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    body_paths = [prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    normalized_count = 0
    for body_path in body_paths:
        body_prim = stage.GetPrimAtPath(body_path)
        visual_nodes = []
        for child in body_prim.GetChildren():
            if not child.IsA(UsdGeom.Xform):
                continue
            descendants = Usd.PrimRange(
                child,
                Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
            )
            has_visual_mesh = False
            has_collision = False
            for descendant in descendants:
                has_visual_mesh |= descendant.IsA(UsdGeom.Mesh)
                has_collision |= descendant.HasAPI(UsdPhysics.CollisionAPI)
            if has_visual_mesh and not has_collision:
                visual_nodes.append(child)
        if not visual_nodes:
            continue
        if len(visual_nodes) != 1:
            raise RuntimeError(f"Expected one visual node below {body_path}, found {len(visual_nodes)}")
        source_path = visual_nodes[0].GetPath()
        destination_path = body_path.AppendChild("visuals")
        if stage.GetPrimAtPath(destination_path):
            raise RuntimeError(f"Cannot normalize duplicate visual node: {destination_path}")
        editor = Usd.NamespaceEditor(stage)
        editor.MovePrimAtPath(source_path, destination_path)
        can_apply = editor.CanApplyEdits()
        if not can_apply:
            raise RuntimeError(f"Cannot move {source_path} to {destination_path}: {can_apply}")
        if not editor.ApplyEdits():
            raise RuntimeError(f"Failed to move {source_path} to {destination_path}")
        normalized_count += 1
    return normalized_count


def _hide_collision_geometry(stage) -> int:
    """Hide collision branches from renderers without disabling their physics APIs.

    Newton deliberately renders colliders as a fallback for bodies without visual-only geometry.
    The legacy Go2 asset instead inherited ``visibility=invisible`` from its collider hierarchy, so
    author the same rendering intent on each importer-created collision branch.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    body_paths = [prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    hidden_count = 0
    for body_path in body_paths:
        body_prim = stage.GetPrimAtPath(body_path)
        for child in body_prim.GetChildren():
            descendants = Usd.PrimRange(
                child,
                Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
            )
            if not any(descendant.HasAPI(UsdPhysics.CollisionAPI) for descendant in descendants):
                continue
            imageable = UsdGeom.Imageable(child)
            if not imageable:
                raise RuntimeError(f"Collision branch is not imageable: {child.GetPath()}")
            imageable.MakeInvisible()
            hidden_count += 1
    return hidden_count


def _preserve_physx_joint_velocity_limits(stage) -> int:
    """Mirror importer-authored Newton velocity limits into the equivalent PhysX attribute.

    Both USD attributes use degrees per second for revolute joints and distance per second for
    prismatic joints. The URDF importer authors only the Newton attribute.
    """
    from pxr import Sdf, UsdPhysics

    joint_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        velocity_limit = prim.GetAttribute("newton:velocityLimit")
        if not velocity_limit or not velocity_limit.HasAuthoredValueOpinion():
            continue
        if not prim.AddAppliedSchema("PhysxJointAPI"):
            raise RuntimeError(f"Cannot apply PhysxJointAPI to {prim.GetPath()}")
        attribute = prim.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float, custom=False)
        if not attribute.Set(velocity_limit.Get()):
            raise RuntimeError(f"Cannot preserve the PhysX joint velocity limit on {prim.GetPath()}")
        joint_count += 1
    return joint_count


def _configure_self_collision(stage, enabled: bool) -> int:
    """Author matching Newton and PhysX self-collision settings on every articulation root."""
    from pxr import Sdf, UsdPhysics

    roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not roots:
        raise RuntimeError("Generated USD contains no articulation root for self-collision configuration.")
    for prim in roots:
        for schema in ("NewtonArticulationRootAPI", "PhysxArticulationAPI"):
            if not prim.AddAppliedSchema(schema):
                raise RuntimeError(f"Cannot apply {schema} to {prim.GetPath()}")
        for name in ("newton:selfCollisionEnabled", "physxArticulation:enabledSelfCollisions"):
            attribute = prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool, custom=False)
            if not attribute.Set(enabled):
                raise RuntimeError(f"Cannot set {name} on {prim.GetPath()}")
    return len(roots)


def main() -> None:
    """Run the Isaac Lab URDF converter and flatten its layered output."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {input_path}")
    if output_path.suffix != ".usd":
        raise ValueError(f"Output must use the .usd suffix: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ros_package_paths = _resolve_ros_packages(args.ros_package)

    # The importer emits a layered asset. Generate it in a temporary directory,
    # then flatten the composed stage so the requested output is the only USD
    # file that callers need to retain.
    with tempfile.TemporaryDirectory(prefix=f"{output_path.stem}_urdf_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        converter_input_path = input_path
        protected_fixed_joint_count = 0
        if args.merge_fixed_joints:
            converter_input_path, protected_fixed_joint_count = _merge_unprotected_fixed_joints(
                input_path, temporary_path
            )
        converter = UrdfConverter(
            UrdfConverterCfg(
                asset_path=str(converter_input_path),
                usd_dir=temporary_dir,
                usd_file_name=output_path.name,
                fix_base=False,
                # Fixed joints were selectively merged above so the converter
                # cannot discard links marked dont_collapse by the URDF.
                merge_fixed_joints=False,
                self_collision=args.self_collision,
                force_usd_conversion=True,
                ros_package_paths=ros_package_paths,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                    target_type="none",
                ),
            )
        )
        generated_stage = Usd.Stage.Open(str(converter.usd_path), load=Usd.Stage.LoadAll)
        if generated_stage is None:
            raise RuntimeError(f"Failed to open generated USD: {converter.usd_path}")

        flattened_layer = generated_stage.Flatten()
        flattened_stage = Usd.Stage.Open(flattened_layer, load=Usd.Stage.LoadAll)
        rigid_body_count = _normalize_rigid_body_hierarchy(flattened_stage)
        normalized_visual_count = _normalize_visual_hierarchy(flattened_stage)
        hidden_collision_branch_count = _hide_collision_geometry(flattened_stage)
        physx_velocity_limit_count = _preserve_physx_joint_velocity_limits(flattened_stage)
        articulation_count = _configure_self_collision(flattened_stage, args.self_collision)
        mesh_count = sum(
            prim.IsA(UsdGeom.Mesh)
            for prim in Usd.PrimRange(flattened_stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        )
        if mesh_count == 0:
            raise RuntimeError(
                "The generated USD contains no Mesh prims. Check package:// paths and --ros-package mappings."
            )
        if not flattened_stage.GetRootLayer().Export(str(output_path), args={"format": "usdc"}):
            raise RuntimeError(f"Failed to export flattened USD: {output_path}")

    output_layer = Sdf.Layer.FindOrOpen(str(output_path))
    if output_layer is None:
        raise RuntimeError(f"Failed to reopen exported USD: {output_path}")
    external_dependencies = {
        dependency
        for dependency in (
            *output_layer.GetCompositionAssetDependencies(),
            *output_layer.GetExternalAssetDependencies(),
        )
        if dependency
    }
    if external_dependencies:
        raise RuntimeError(
            f"Generated USD is not self-contained; external dependencies: {sorted(external_dependencies)}"
        )
    output_stage = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadAll)
    if output_stage is None:
        raise RuntimeError(f"Failed to reopen exported USD stage: {output_path}")
    saved_roots = [prim for prim in output_stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(saved_roots) != articulation_count:
        raise RuntimeError("Exported USD articulation root count differs from the generated stage.")
    for prim in saved_roots:
        applied_schemas = prim.GetMetadata("apiSchemas").GetAppliedItems()
        for schema in ("NewtonArticulationRootAPI", "PhysxArticulationAPI"):
            if schema not in applied_schemas:
                raise RuntimeError(f"Exported {prim.GetPath()} is missing {schema}.")
        for name in ("newton:selfCollisionEnabled", "physxArticulation:enabledSelfCollisions"):
            attribute = prim.GetAttribute(name)
            if not attribute.HasAuthoredValueOpinion() or attribute.Get() != args.self_collision:
                raise RuntimeError(f"Exported {prim.GetPath()}.{name} does not match the requested setting.")
    print(f"Generated self-contained USD: {output_path}")
    print(f"Mesh prims: {mesh_count}")
    print(f"Rigid-body links: {rigid_body_count}")
    print(f"Normalized visual nodes: {normalized_visual_count}")
    print(f"Hidden collision branches: {hidden_collision_branch_count}")
    print(f"Preserved PhysX joint velocity limits: {physx_velocity_limit_count}")
    print(f"Articulation self-collision: {args.self_collision} ({articulation_count} roots)")
    if args.merge_fixed_joints:
        print(f"Preserved fixed joints: {protected_fixed_joint_count}")


try:
    main()
finally:
    simulation_app.close()
