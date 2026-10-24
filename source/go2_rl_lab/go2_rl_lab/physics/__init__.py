"""Project-local physics backend extensions."""

from .randomize import BodyComBucketRandomizationCfg, RandomizedMJWarpSolverCfg

__all__ = [
    "BodyComBucketRandomizationCfg",
    "NewtonMJWarpResettableManager",
    "RandomizedMJWarpSolverCfg",
]


def __getattr__(name: str):
    """Load solver implementations only when the active runtime requests them."""
    if name == "NewtonMJWarpResettableManager":
        from .resettable_mjwarp_manager import NewtonMJWarpResettableManager

        return NewtonMJWarpResettableManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
