"""Module system detection for JavaScript/TypeScript projects.

Determines whether a project uses CommonJS (require/module.exports) or
ES Modules (import/export).
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ModuleSystem:
    """Enum-like class for module systems."""

    COMMONJS = "commonjs"
    ES_MODULE = "esm"
    UNKNOWN = "unknown"


# Pattern for destructured require: const { a, b } = require('...')
destructured_require = re.compile(
    r"(const|let|var)\s+\{\s*([^}]+)\s*\}\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*;?"
)

# Pattern for require with property access: const foo = require('...').propertyName
# This must come before simple_require to match first
property_access_require = re.compile(
    r"(const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\.(\w+)\s*;?"
)

# Pattern for simple require: const foo = require('...')
simple_require = re.compile(r"(const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*;?")


def detect_module_system(project_root: Path, file_path: Path | None = None) -> str:
    """Detect the module system used by a JavaScript/TypeScript project.

    Detection strategy:
    1. Check file extension for explicit module type (.mjs, .cjs, .ts, .tsx, .mts)
       - TypeScript files always use ESM syntax regardless of package.json
    2. Check package.json for explicit "type" field (only if explicitly set)
    3. Analyze import/export statements in the file content
    4. Default to CommonJS if uncertain

    Args:
        project_root: Root directory of the project containing package.json.
        file_path: Optional specific file to analyze.

    Returns:
        ModuleSystem constant (COMMONJS, ES_MODULE, or UNKNOWN).

    """
    # Strategy 1: Check file extension first for explicit module type indicators
    # TypeScript files always use ESM syntax (import/export)
    if file_path:
        suffix = file_path.suffix.lower()
        if suffix == ".mjs":
            logger.debug("Detected ES Module from .mjs extension")
            return ModuleSystem.ES_MODULE
        if suffix == ".cjs":
            logger.debug("Detected CommonJS from .cjs extension")
            return ModuleSystem.COMMONJS
        if suffix in (".ts", ".tsx", ".mts"):
            # TypeScript always uses ESM syntax (import/export)
            # even if package.json doesn't have "type": "module"
            logger.debug("Detected ES Module from TypeScript file extension")
            return ModuleSystem.ES_MODULE

    # Strategy 2: Check package.json for explicit type field
    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            with package_json.open("r") as f:
                pkg = json.load(f)
                pkg_type = pkg.get("type")  # Don't default - only use if explicitly set

                if pkg_type == "module":
                    logger.debug("Detected ES Module from package.json type field")
                    return ModuleSystem.ES_MODULE
                if pkg_type == "commonjs":
                    logger.debug("Detected CommonJS from package.json type field")
                    return ModuleSystem.COMMONJS
                # If type is not explicitly set, continue to file content analysis

        except Exception as e:
            logger.warning("Failed to parse package.json: %s", e)

    # Strategy 3: Analyze file content for import/export patterns
    if file_path and file_path.exists():
        try:
            content = file_path.read_text()

            # Look for ES module syntax
            has_import = "import " in content and "from " in content
            has_export = "export " in content or "export default" in content or "export {" in content

            # Look for CommonJS syntax
            has_require = "require(" in content
            has_module_exports = "module.exports" in content or "exports." in content

            # Determine based on what we found
            if (has_import or has_export) and not (has_require or has_module_exports):
                logger.debug("Detected ES Module from import/export statements")
                return ModuleSystem.ES_MODULE

            if (has_require or has_module_exports) and not (has_import or has_export):
                logger.debug("Detected CommonJS from require/module.exports")
                return ModuleSystem.COMMONJS

        except Exception as e:
            logger.warning("Failed to analyze file %s: %s", file_path, e)

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


def add_js_extension(module_path: str) -> str:
    """Process module path for ESM compatibility.

    NOTE: This function intentionally does NOT add extensions because:
    1. TypeScript projects resolve modules without explicit extensions
    2. Adding .js to .ts imports causes "Cannot find module" errors
    3. Modern bundlers (webpack, vite, etc.) handle extension resolution automatically

    The function name is preserved for backward compatibility but the behavior
    has been changed to NOT add extensions.
    """
    # Previously this function added .js extensions, but this caused module resolution
    # errors in TypeScript projects. We now preserve paths without adding extensions.
    return module_path


# Replace destructured requires with named imports
def replace_destructured(match: re.Match) -> str:
    names = match.group(2).strip()
    module_path = add_js_extension(match.group(3))
    return f"import {{ {names} }} from '{module_path}';"


# Replace property access requires with named imports with alias
# e.g., const foo = require('./module').bar -> import { bar as foo } from './module';
def replace_property_access(match: re.Match) -> str:
    alias_name = match.group(2)  # The variable name (e.g., missingAuthHeader)
    module_path = add_js_extension(match.group(3))
    property_name = match.group(4)  # The property being accessed (e.g., missingAuthorizationHeader)

    # Special case: .default means default export
    if property_name == "default":
        return f"import {alias_name} from '{module_path}';"

    # Named export with alias
    if alias_name == property_name:
        return f"import {{ {property_name} }} from '{module_path}';"
    return f"import {{ {property_name} as {alias_name} }} from '{module_path}';"


# Replace simple requires with default imports
def replace_simple(match: re.Match) -> str:
    name = match.group(2)
    module_path = add_js_extension(match.group(3))
    return f"import {name} from '{module_path}';"


def convert_commonjs_to_esm(code: str) -> str:
    """Convert CommonJS require statements to ES Module imports.

    Converts:
        const { foo, bar } = require('./module');  ->  import { foo, bar } from './module';
        const foo = require('./module');           ->  import foo from './module';
        const foo = require('./module').default;   ->  import foo from './module';
        const foo = require('./module').bar;       ->  import { bar as foo } from './module';

    Special handling:
        - Local codeflash helper (./codeflash-jest-helper) is converted to npm package codeflash
          because the local helper uses CommonJS exports which don't work in ESM projects

    Args:
        code: JavaScript code with CommonJS require statements.

    Returns:
        Code with ES Module import statements.

    """
    # Apply conversions (most specific patterns first)
    code = destructured_require.sub(replace_destructured, code)
    code = property_access_require.sub(replace_property_access, code)
    return simple_require.sub(replace_simple, code)


def convert_esm_to_commonjs(code: str) -> str:
    """Convert ES Module imports to CommonJS require statements.

    Converts:
        import { foo, bar } from './module';  ->  const { foo, bar } = require('./module');
        import foo from './module';           ->  const foo = require('./module');

    Args:
        code: JavaScript code with ES Module import statements.

    Returns:
        Code with CommonJS require statements.

    """
    import re

    # Pattern for named import: import { a, b } from '...'; (semicolon optional)
    named_import = re.compile(r"import\s+\{\s*([^}]+)\s*\}\s+from\s+['\"]([^'\"]+)['\"];?")

    # Pattern for default import: import foo from '...'; (semicolon optional)
    default_import = re.compile(r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"];?")

    # Replace named imports with destructured requires
    def replace_named(match) -> str:
        names = match.group(1).strip()
        module_path = match.group(2)
        # Remove .js extension for CommonJS (optional but cleaner)
        module_path = module_path.removesuffix(".js")
        return f"const {{ {names} }} = require('{module_path}');"

    # Replace default imports with simple requires
    def replace_default(match) -> str:
        name = match.group(1)
        module_path = match.group(2)
        # Remove .js extension for CommonJS
        module_path = module_path.removesuffix(".js")
        return f"const {name} = require('{module_path}');"

    # Apply conversions (named first as it's more specific)
    code = named_import.sub(replace_named, code)
    return default_import.sub(replace_default, code)


def ensure_module_system_compatibility(code: str, target_module_system: str) -> str:
    """Ensure code uses the correct module system syntax.

    Detects the current module system in the code and converts if needed.
    Handles mixed-style code (e.g., ESM imports with CommonJS require for npm packages).

    Args:
        code: JavaScript code to check and potentially convert.
        target_module_system: Target ModuleSystem (COMMONJS or ES_MODULE).

    Returns:
        Code with correct module system syntax.

    """
    # Detect current module system in code
    has_require = "require(" in code
    has_import = "import " in code and "from " in code

    if target_module_system == ModuleSystem.ES_MODULE:
        # Convert any require() statements to imports for ESM projects
        # This handles mixed code (ESM imports + CommonJS requires for npm packages)
        if has_require:
            logger.debug("Converting CommonJS requires to ESM imports")
            return convert_commonjs_to_esm(code)
    elif target_module_system == ModuleSystem.COMMONJS:
        # Convert any import statements to requires for CommonJS projects
        if has_import:
            logger.debug("Converting ESM imports to CommonJS requires")
            return convert_esm_to_commonjs(code)

    return code


def ensure_vitest_imports(code: str, test_framework: str) -> str:
    """Ensure vitest test globals are imported when using vitest framework.

    Vitest by default does not enable globals (describe, test, expect, etc.),
    so they must be explicitly imported. This function adds the import if missing.

    Args:
        code: JavaScript/TypeScript test code.
        test_framework: The test framework being used (vitest, jest, mocha).

    Returns:
        Code with vitest imports added if needed.

    """
    if test_framework != "vitest":
        return code

    # Check if vitest imports already exist
    if "from 'vitest'" in code or 'from "vitest"' in code:
        return code

    # Check if the code uses test functions that need to be imported
    test_globals = ["describe", "test", "it", "expect", "vi", "beforeEach", "afterEach", "beforeAll", "afterAll"]

    # Combine detection and collection into a single pass
    used_globals = [g for g in test_globals if f"{g}(" in code or f"{g} (" in code]
    if not used_globals:
        return code

    # Build the import statement
    import_statement = f"import {{ {', '.join(used_globals)} }} from 'vitest';\n"

    # Find the first line that isn't a comment or empty
    lines = code.split("\n")
    insert_index = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("//")
            and not stripped.startswith("/*")
            and not stripped.startswith("*")
        ):
            # Check if this line is an import/require - insert after imports
            if stripped.startswith(("import ", "const ", "let ")):
                continue
            insert_index = i
            break
        insert_index = i + 1

    # Find the last import line to insert after it
    last_import_index = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") and "from " in stripped:
            last_import_index = i

    if last_import_index >= 0:
        # Insert after the last import
        lines.insert(last_import_index + 1, import_statement.rstrip())
    else:
        # Insert at the beginning (after any leading comments)
        lines.insert(insert_index, import_statement.rstrip())

    logger.debug("Added vitest imports: %s", used_globals)
    return "\n".join(lines)
