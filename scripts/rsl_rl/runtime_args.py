"""Select the initial runtime before importing custom task implementations."""

import atexit
import sys
from contextlib import ExitStack

from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsCfg
from isaaclab.sim import SimulationCfg


_runtime_stack: ExitStack | None = None


def bootstrap_runtime(args_cli, overrides=()) -> None:
    """Initialize the selected runtime before importing task implementations.

    The preliminary config selects the runtime; the composed task config owns the
    solver and scene. Keep the runtime open until shutdown so task imports use its
    USD bindings. Interpreter-exit cleanup also covers failures during imports.
    """
    global _runtime_stack
    if _runtime_stack is not None:
        raise RuntimeError("The simulation runtime has already been bootstrapped.")

    physics = args_cli.physics or "newton_mjwarp"
    for override in overrides:
        key, separator, value = override.partition("=")
        if separator and key.lstrip("-") in {"physics", "env.sim.physics"}:
            physics = value
    cfg = SimulationCfg(physics=PhysicsCfg())
    if args_cli.video and not args_cli.visualizer and not getattr(args_cli, "visualizer_explicit", False):
        # Select EGL before task/terrain imports can cache pyglet's X11 display backend.
        import pyglet

        pyglet.options["headless"] = True
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        cfg.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True)]
    launcher_args = vars(args_cli).copy()
    launcher_args["physics"] = physics
    # Capture is configured on the final env; the launcher does not own encoding.
    launcher_args["video"] = False
    stack = ExitStack()
    stack.enter_context(launch_simulation(cfg, launcher_args))
    _runtime_stack = stack
    atexit.register(shutdown_runtime)


def shutdown_runtime() -> None:
    """Close the bootstrapped runtime once, preserving an active failure."""
    global _runtime_stack
    stack = _runtime_stack
    if stack is None:
        return
    _runtime_stack = None

    exception_type, exception, traceback = sys.exc_info()
    if exception is None:
        exception = getattr(sys, "last_exc", None)
        if exception is not None:
            exception_type, traceback = type(exception), exception.__traceback__
    stack.__exit__(exception_type, exception, traceback)
