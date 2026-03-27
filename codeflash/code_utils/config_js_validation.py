"""JavaScript/TypeScript module resolution validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.code_utils.config_js import detect_module_root, get_package_json_data
from codeflash.languages.javascript.test_runner import find_node_project_root

if TYPE_CHECKING:
    from pathlib import Path


def validate_js_module_resolution(source_file: Path, project_root: Path, module_root: Path) -> tuple[bool, str]:
    """Validate that a JS/TS source file can be resolved within the configured module root.

    Checks:
    1. Source file exists
    2. Source file is within project_root
    3. package.json exists in project_root
    4. Source file is within module_root

    Returns:
        (True, "") on success, (False, error_message) on failure.

    """
    source_file = source_file.resolve()
    project_root = project_root.resolve()
    module_root = module_root.resolve()

    if not source_file.exists():
        return False, f"Source file does not exist: {source_file}"

    try:
        source_file.relative_to(project_root)
    except ValueError:
        return False, f"Source file {source_file} is not within project root {project_root}"

    package_json = project_root / "package.json"
    if not package_json.exists():
        return False, f"No package.json found at {project_root}"

    try:
        source_file.relative_to(module_root)
    except ValueError:
        return False, (
            f"Source file {source_file} is not within module root {module_root}. "
            f"Check the 'codeflash.moduleRoot' setting in package.json."
        )

    return True, ""


def infer_js_module_root(source_file: Path, project_root: Path | None = None) -> Path:
    """Infer the JavaScript/TypeScript module root for a source file.

    Uses find_node_project_root to locate package.json, then detect_module_root
    to determine the module root from package.json fields and directory conventions.

    Falls back to the source file's parent directory if no package.json is found.

    Returns:
        Absolute path to the inferred module root.

    """
    source_file = source_file.resolve()

    if project_root is None:
        project_root = find_node_project_root(source_file)

    if project_root is None:
        return source_file.parent

    project_root = project_root.resolve()
    package_json_path = project_root / "package.json"
    package_data = get_package_json_data(package_json_path)

    if package_data is None:
        return project_root

    detected = detect_module_root(project_root, package_data)
    return (project_root / detected).resolve()
