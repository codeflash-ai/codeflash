"""JavaScript/TypeScript configuration parsing from package.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACKAGE_JSON_CACHE: dict[Path, Path] = {}
PACKAGE_JSON_DATA_CACHE: dict[Path, dict[str, Any]] = {}


def get_package_json_data(package_json_path: Path) -> dict[str, Any] | None:
    """Load and cache package.json data.

    Args:
        package_json_path: Path to package.json file.

    Returns:
        Parsed package.json data or None if invalid.

    """
    if package_json_path in PACKAGE_JSON_DATA_CACHE:
        return PACKAGE_JSON_DATA_CACHE[package_json_path]

    try:
        with package_json_path.open(encoding="utf8") as f:
            data: dict[str, Any] = json.load(f)
            PACKAGE_JSON_DATA_CACHE[package_json_path] = data
            return data
    except (json.JSONDecodeError, OSError):
        return None


def detect_language(project_root: Path) -> str:
    """Detect project language from tsconfig.json presence.

    Args:
        project_root: Root directory of the project.

    Returns:
        "typescript" if tsconfig.json exists, "javascript" otherwise.

    """
    tsconfig_path = project_root / "tsconfig.json"
    return "typescript" if tsconfig_path.exists() else "javascript"


def detect_module_root(project_root: Path, package_data: dict[str, Any]) -> str:
    """Detect module root from package.json fields or directory conventions.

    Detection order:
    1. package.json "exports" field (extract directory from main export)
    2. package.json "module" field (ESM entry point)
    3. package.json "main" field (CJS entry point)
    4. "src/" directory if it exists
    5. Fall back to "." (project root)

    Args:
        project_root: Root directory of the project.
        package_data: Parsed package.json data.

    Returns:
        Detected module root path (relative to project root).

    """
    # Check exports field (modern Node.js)
    exports = package_data.get("exports")
    if exports:
        entry_path = None
        if isinstance(exports, str):
            entry_path = exports
        elif isinstance(exports, dict):
            # Handle {"." : "./src/index.js"} or {".": {"import": "./src/index.js"}}
            main_export = exports.get(".") or exports.get("import") or exports.get("default")
            if isinstance(main_export, str):
                entry_path = main_export
            elif isinstance(main_export, dict):
                entry_path = main_export.get("import") or main_export.get("default") or main_export.get("require")

        if entry_path and isinstance(entry_path, str):
            parent = Path(entry_path).parent
            if parent != Path() and (project_root / parent).is_dir():
                return parent.as_posix()

    # Check module field (ESM)
    module_field = package_data.get("module")
    if module_field and isinstance(module_field, str):
        parent = Path(module_field).parent
        if parent != Path() and (project_root / parent).is_dir():
            return parent.as_posix()

    # Check main field (CJS)
    main_field = package_data.get("main")
    if main_field and isinstance(main_field, str):
        parent = Path(main_field).parent
        if parent != Path() and (project_root / parent).is_dir():
            return parent.as_posix()

    # Check for src/ directory convention
    if (project_root / "src").is_dir():
        return "src"

    # Default to project root
    return "."


def detect_test_runner(project_root: Path, package_data: dict[str, Any]) -> str:  # noqa: ARG001
    """Detect test runner from devDependencies or scripts.test.

    Detection order:
    1. Check devDependencies for vitest, jest, mocha
    2. Parse scripts.test for runner hints
    3. Fall back to "jest" as default

    Args:
        project_root: Root directory of the project.
        package_data: Parsed package.json data.

    Returns:
        Detected test runner command (e.g., "jest", "vitest", "mocha").

    """
    runners = ["vitest", "jest", "mocha"]
    dev_deps = package_data.get("devDependencies", {})
    deps = package_data.get("dependencies", {})
    all_deps = {**deps, **dev_deps}

    # Check devDependencies (order matters - prefer more modern runners)
    for runner in runners:
        if runner in all_deps:
            return runner

    # Parse scripts.test for hints
    scripts = package_data.get("scripts", {})
    test_script = scripts.get("test", "")
    if isinstance(test_script, str):
        test_lower = test_script.lower()
        for runner in runners:
            if runner in test_lower:
                return runner

    # Default to jest
    return "jest"


def detect_formatter(project_root: Path, package_data: dict[str, Any]) -> list[str] | None:  # noqa: ARG001
    """Detect formatter from devDependencies.

    Detection order:
    1. Check devDependencies for prettier
    2. Check devDependencies for eslint (with --fix)
    3. Return None if no formatter detected

    Args:
        project_root: Root directory of the project.
        package_data: Parsed package.json data.

    Returns:
        List of formatter commands or None if not detected.

    """
    dev_deps = package_data.get("devDependencies", {})
    deps = package_data.get("dependencies", {})
    all_deps = {**deps, **dev_deps}

    # Check for prettier (preferred)
    if "prettier" in all_deps:
        return ["npx prettier --write $file"]

    # Check for eslint (can format with --fix)
    if "eslint" in all_deps:
        return ["npx eslint --fix $file"]

    return None


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
    """Parse codeflash config from package.json with auto-detection.

    Most configuration is auto-detected from package.json and project structure.
    Only minimal config is stored in the "codeflash" key:
    - benchmarksRoot: Where to store benchmark files (optional, defaults to __benchmarks__)
    - ignorePaths: Paths to exclude from optimization (optional)
    - disableTelemetry: Privacy preference (optional, defaults to false)
    - formatterCmds: Override auto-detected formatter (optional)

    Auto-detected values (not stored in config):
    - language: Detected from tsconfig.json presence
    - moduleRoot: Detected from package.json exports/module/main or src/ convention
    - testRunner: Detected from devDependencies (vitest/jest/mocha)
    - formatter: Detected from devDependencies (prettier/eslint)

    Args:
        package_json_path: Path to package.json file.

    Returns:
        Tuple of (config dict, path) if package.json exists, None otherwise.

    """
    package_data = get_package_json_data(package_json_path)
    if package_data is None:
        return None

    project_root = package_json_path.parent
    codeflash_config = package_data.get("codeflash", {})
    if not isinstance(codeflash_config, dict):
        codeflash_config = {}

    config: dict[str, Any] = {}

    # Auto-detect language
    config["language"] = detect_language(project_root)

    # Auto-detect module root (can be overridden)
    if codeflash_config.get("moduleRoot"):
        config["module_root"] = str((project_root / Path(codeflash_config["moduleRoot"])).resolve())
    else:
        detected_module_root = detect_module_root(project_root, package_data)
        config["module_root"] = str((project_root / Path(detected_module_root)).resolve())

    # Auto-detect test runner
    config["test_runner"] = detect_test_runner(project_root, package_data)
    # Keep pytest_cmd for backwards compatibility with existing code
    config["pytest_cmd"] = config["test_runner"]

    # Auto-detect formatter (with optional override from config)
    if "formatterCmds" in codeflash_config:
        config["formatter_cmds"] = codeflash_config["formatterCmds"]
    else:
        detected_formatter = detect_formatter(project_root, package_data)
        config["formatter_cmds"] = detected_formatter if detected_formatter else []

    # Parse optional config values from codeflash section
    if codeflash_config.get("benchmarksRoot"):
        config["benchmarks_root"] = str((project_root / Path(codeflash_config["benchmarksRoot"])).resolve())

    if codeflash_config.get("ignorePaths"):
        config["ignore_paths"] = [str((project_root / path).resolve()) for path in codeflash_config["ignorePaths"]]
    else:
        config["ignore_paths"] = []

    config["disable_telemetry"] = codeflash_config.get("disableTelemetry", False)

    # Git remote (from config or default to "origin")
    config["git_remote"] = codeflash_config.get("gitRemote", "origin")

    # Set remaining defaults for backwards compatibility
    config.setdefault("disable_imports_sorting", False)
    config.setdefault("override_fixtures", False)

    return config, package_json_path


def clear_cache() -> None:
    """Clear all package.json caches."""
    PACKAGE_JSON_CACHE.clear()
    PACKAGE_JSON_DATA_CACHE.clear()
