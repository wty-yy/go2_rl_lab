"""Environment configuration loading and preset resolution utilities."""

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import resolve_task_config


def parse_env_cfg(
    task_name: str,
    device: str | None = None,
    num_envs: int | None = None,
    use_fabric: bool | None = None,
    entry_point_key: str = "env_cfg_entry_point",
    physics: str | None = None,
) -> ManagerBasedRLEnvCfg:
    """Parse configuration for an environment and override based on inputs.

    Args:
        task_name: The name of the environment.
        device: The simulation device, or None to preserve the task default.
        num_envs: Number of environments to create. Defaults to None, in which case it is left unchanged.
        use_fabric: Whether to enable/disable fabric interface. If false, all read/write operations go through USD.
            This slows down the simulation but allows seeing the changes in the USD through the USD stage.
            Defaults to None, in which case it is left unchanged.
        physics: Optional physics preset name, such as ``"isaacsim_physx"`` or ``"newton_mjwarp"``.

    Returns:
        The parsed configuration object.

    Raises:
        RuntimeError: If the configuration for the task is not a class. We assume users always use a class for the
            environment configuration.
    """
    # Compose the selected entry point through the shared task boundary.
    # Playback configs are registered separately from training configs in this project.
    spec = gym.spec(task_name.split(":")[-1])
    original_entry = spec.kwargs["env_cfg_entry_point"]
    spec.kwargs["env_cfg_entry_point"] = spec.kwargs[entry_point_key]
    try:
        overrides = [f"physics={physics}"] if physics is not None else []
        cfg, _ = resolve_task_config(task_name, None, overrides=overrides)
    finally:
        spec.kwargs["env_cfg_entry_point"] = original_entry

    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for task {task_name!r} must be a configuration class.")

    # simulation device
    if device is not None:
        cfg.sim.device = device
    # disable fabric to read/write through USD
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    # number of environments
    if num_envs is not None:
        cfg.scene.num_envs = num_envs

    return cfg
