"""Module system detection for JavaScript/TypeScript projects.

Determines whether a project uses CommonJS (require/module.exports) or
ES Modules (import/export).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModuleSystem:
    """Enum-like class for module systems."""

    COMMONJS = "commonjs"
    ES_MODULE = "esm"
    UNKNOWN = "unknown"


def detect_module_system(project_root: Path, file_path: Path | None = None) -> str:
    """Detect the module system used by a JavaScript/TypeScript project.

    Detection strategy:
    1. Check package.json for "type" field
    2. If file_path provided, check file extension (.mjs = ESM, .cjs = CommonJS)
    3. Analyze import statements in the file
    4. Default to CommonJS if uncertain

    Args:
        project_root: Root directory of the project containing package.json.
        file_path: Optional specific file to analyze.

    Returns:
        ModuleSystem constant (COMMONJS, ES_MODULE, or UNKNOWN).

    """
    # Strategy 1: Check package.json
    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            with package_json.open("r") as f:
                pkg = json.load(f)
                pkg_type = pkg.get("type", "commonjs")

                if pkg_type == "module":
                    logger.debug("Detected ES Module from package.json type field")
                    return ModuleSystem.ES_MODULE
                if pkg_type == "commonjs":
                    logger.debug("Detected CommonJS from package.json type field")
                    return ModuleSystem.COMMONJS

        except Exception as e:
            logger.warning(f"Failed to parse package.json: {e}")

    # Strategy 2: Check file extension
    if file_path:
        suffix = file_path.suffix
        if suffix == ".mjs":
            logger.debug("Detected ES Module from .mjs extension")
            return ModuleSystem.ES_MODULE
        if suffix == ".cjs":
            logger.debug("Detected CommonJS from .cjs extension")
            return ModuleSystem.COMMONJS

        # Strategy 3: Analyze file content
        if file_path.exists():
            try:
                content = file_path.read_text()

                # Look for ES module syntax
                has_import = "import " in content and "from " in content
                has_export = "export " in content or "export default" in content or "export {" in content

                # Look for CommonJS syntax
                has_require = "require(" in content
                has_module_exports = "module.exports" in content or "exports." in content

                # Determine based on what we found
                if has_import or has_export:
                    if not (has_require or has_module_exports):
                        logger.debug("Detected ES Module from import/export statements")
                        return ModuleSystem.ES_MODULE

                if has_require or has_module_exports:
                    if not (has_import or has_export):
                        logger.debug("Detected CommonJS from require/module.exports")
                        return ModuleSystem.COMMONJS

            except Exception as e:
                logger.warning(f"Failed to analyze file {file_path}: {e}")

    # Default to CommonJS (more common and backward compatible)
    logger.debug("Defaulting to CommonJS")
    return ModuleSystem.COMMONJS


def get_import_statement(
    module_system: str, target_path: Path, source_path: Path, imported_names: list[str] | None = None
) -> str:
    """Generate the appropriate import statement for the module system.

    Args:
        module_system: ModuleSystem constant (COMMONJS or ES_MODULE).
        target_path: Path to the module being imported.
        source_path: Path to the file doing the importing.
        imported_names: List of names to import (for named imports).

    Returns:
        Import statement string.

    """
    # Calculate relative import path
    rel_path = _get_relative_import_path(target_path, source_path)

    if module_system == ModuleSystem.ES_MODULE:
        if imported_names:
            names = ", ".join(imported_names)
            return f"import {{ {names} }} from '{rel_path}';"
        # Default import
        module_name = target_path.stem
        return f"import {module_name} from '{rel_path}';"
    if imported_names:
        names = ", ".join(imported_names)
        return f"const {{ {names} }} = require('{rel_path}');"
    # Require entire module
    module_name = target_path.stem
    return f"const {module_name} = require('{rel_path}');"


def _get_relative_import_path(target_path: Path, source_path: Path) -> str:
    """Calculate relative import path from source to target.

    For JavaScript imports, we calculate the path from the source file's directory
    to the target file.

    Args:
        target_path: Absolute path to the file being imported.
        source_path: Absolute path to the file doing the importing.

    Returns:
        Relative import path (without file extension for .js files).

    """
    # Both paths should be absolute - get the directory containing source
    source_dir = source_path.parent

    # Try to use os.path.relpath for accuracy
    import os

    rel_path_str = os.path.relpath(str(target_path), str(source_dir))

    # Normalize to forward slashes
    rel_path_str = rel_path_str.replace("\\", "/")

    # Remove .js extension (Node.js convention)
    rel_path_str = rel_path_str.removesuffix(".js")

    # Ensure it starts with ./ or ../ for relative imports
    if not rel_path_str.startswith("./") and not rel_path_str.startswith("../"):
        rel_path_str = "./" + rel_path_str

    return rel_path_str
