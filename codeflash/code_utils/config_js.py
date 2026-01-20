"""JavaScript/TypeScript configuration parsing from package.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACKAGE_JSON_CACHE: dict[Path, Path] = {}


def find_package_json(config_file: Path | None = None) -> Path | None:
    """Find package.json file for JavaScript/TypeScript projects.

    Args:
        config_file: Optional explicit config file path.

    Returns:
        Path to package.json if found, None otherwise.
    """
    if config_file is not None:
        config_file = Path(config_file)
        if config_file.name == "package.json" and config_file.exists():
            return config_file
        return None

    dir_path = Path.cwd()
    cur_path = dir_path

    if cur_path in PACKAGE_JSON_CACHE:
        return PACKAGE_JSON_CACHE[cur_path]

    while dir_path != dir_path.parent:
        config_file = dir_path / "package.json"
        if config_file.exists():
            PACKAGE_JSON_CACHE[cur_path] = config_file
            return config_file
        dir_path = dir_path.parent

    return None


def parse_package_json_config(package_json_path: Path) -> tuple[dict[str, Any], Path] | None:
    """Parse codeflash config from package.json.

    Config is stored under the "codeflash" key using camelCase convention.
    Keys are converted to snake_case for internal use.

    Args:
        package_json_path: Path to package.json file.

    Returns:
        Tuple of (config dict, path) if codeflash config exists, None otherwise.
    """
    try:
        with package_json_path.open(encoding="utf8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    codeflash_config = data.get("codeflash")
    if not codeflash_config or not isinstance(codeflash_config, dict):
        return None

    # Convert camelCase to snake_case and resolve paths
    config: dict[str, Any] = {}
    parent_dir = package_json_path.parent

    # Map camelCase keys to snake_case
    key_mapping = {
        "moduleRoot": "module_root",
        "testsRoot": "tests_root",
        "benchmarksRoot": "benchmarks_root",
        "ignorePaths": "ignore_paths",
        "formatterCmds": "formatter_cmds",
        "pytestCmd": "pytest_cmd",
        "gitRemote": "git_remote",
        "disableTelemetry": "disable_telemetry",
        "disableImportsSorting": "disable_imports_sorting",
        "overrideFixtures": "override_fixtures",
        "language": "language",
    }

    for camel_key, snake_key in key_mapping.items():
        if camel_key in codeflash_config:
            value = codeflash_config[camel_key]

            # Resolve path keys
            if snake_key in ("module_root", "tests_root", "benchmarks_root"):
                if value:
                    config[snake_key] = str((parent_dir / Path(value)).resolve())
            elif snake_key == "ignore_paths":
                if value:
                    config[snake_key] = [str((parent_dir / path).resolve()) for path in value]
            else:
                config[snake_key] = value

    # Set defaults appropriate for JS/TS projects
    config.setdefault("pytest_cmd", "jest")  # Default to jest for JS/TS
    config.setdefault("git_remote", "origin")
    config.setdefault("disable_telemetry", False)
    config.setdefault("disable_imports_sorting", False)
    config.setdefault("override_fixtures", False)
    config.setdefault("formatter_cmds", ["npx prettier --write $file"])
    config.setdefault("ignore_paths", [])

    return config, package_json_path


def clear_cache() -> None:
    """Clear the package.json path cache."""
    PACKAGE_JSON_CACHE.clear()
