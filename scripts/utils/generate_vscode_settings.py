"""Generate Isaac Lab VS Code settings and this project's Pyright import paths.

Quick Start:
    ./go2_rl_lab.sh --generate-vscode-settings

Full Command:
    ./go2_rl_lab.sh --generate-vscode-settings --verbose

Notes:
    Run in the configured Isaac Lab 3.0 Python environment.
    Use --verbose to print the discovered import paths.
    Isaac Sim extension paths are discovered even without its VS Code settings.
    Pylance diagnostics are disabled while code navigation and completion remain available.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from importlib.util import find_spec
from pathlib import Path

PROJECT_EXTRA_PATHS = ("source/rsl_rl", "source/go2_rl_lab")
ISAACSIM_MISSING_WARNING = "[WARN] Isaac Sim was not found; simulator extension paths were not added."


def discover_isaacsim_extra_paths(project_root: Path) -> list[str]:
    """Find installed Isaac Sim extension and Kit Python paths without starting Kit."""
    candidates = []
    if isaac_path := os.environ.get("ISAAC_PATH"):
        candidates.append(Path(isaac_path).expanduser())
    candidates.append(project_root / "_isaac_sim")
    sim_spec = find_spec("isaacsim")
    if sim_spec is not None and sim_spec.origin is not None:
        candidates.append(Path(sim_spec.origin).parent)

    for candidate in candidates:
        if (candidate / "exts").is_dir() or (candidate / "extscache").is_dir():
            sim_root = candidate.resolve()
            break
    else:
        return []

    kit_roots = [sim_root / "kit"]
    omni_spec = find_spec("omni")
    if omni_spec is not None:
        kit_roots.extend(
            Path(location)
            for location in (omni_spec.submodule_search_locations or [])
            if Path(location).parent.resolve() == sim_root.parent
        )

    paths = [root / "kernel/py" for root in kit_roots]
    extension_dirs = [
        sim_root / name for name in ("exts", "extscache", "extsDeprecated", "extsPhysics")
    ]
    extension_dirs.extend(root / name for root in kit_roots for name in ("exts", "extscore"))
    for directory in extension_dirs:
        for config_path in sorted(directory.glob("*/config/extension.toml")):
            extension_root = config_path.parent.parent
            paths.append(extension_root)
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            paths.extend(
                extension_root / module.get("path", ".")
                for module in config.get("python", {}).get("module", [])
            )

    return list(dict.fromkeys(path.resolve().as_posix() for path in paths if path.is_dir()))


def add_project_extra_paths(config_path: Path, isaacsim_paths: list[str]) -> None:
    """Add project and simulator paths while preserving the installed package priority."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["extraPaths"] = list(
        dict.fromkeys((*PROJECT_EXTRA_PATHS, *config.get("extraPaths", []), *isaacsim_paths))
    )
    config["typeCheckingMode"] = "off"
    config["ignore"] = ["**"]
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    """Generate the base settings with Isaac Lab, then add project paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", action="store_true", help="Show Isaac Lab path discovery output."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    isaacsim_paths = discover_isaacsim_extra_paths(project_root)
    command = [sys.executable, "-m", "isaaclab", "--editor"]
    if args.verbose:
        command.append("--verbose")

    result = subprocess.run(command, cwd=project_root, stdout=subprocess.PIPE, text=True, check=False)
    if result.returncode:
        print(result.stdout, end="")
        result.check_returncode()

    settings_path = project_root / ".vscode" / "settings.json"
    if not settings_path.is_file():
        raise FileNotFoundError(f"Isaac Lab did not generate {settings_path}")

    config_path = project_root / "pyrightconfig.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Isaac Lab did not generate {config_path}")

    add_project_extra_paths(config_path, isaacsim_paths)
    output = result.stdout
    if isaacsim_paths:
        output = output.replace(
            ISAACSIM_MISSING_WARNING,
            f"[INFO] Discovered {len(isaacsim_paths)} Isaac Sim import paths from installed extensions.",
        )
    print(output, end="")
    if args.verbose:
        for path in isaacsim_paths:
            print(f"Registered Isaac Sim search path: {path}")
    print(f"Added project source paths to {config_path}")


if __name__ == "__main__":
    main()
