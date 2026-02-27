"""JavaScript test runner using Jest.

This module provides functions for running Jest tests for behavioral
verification and performance benchmarking.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.cli_cmds.init_javascript import get_package_install_command
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.config_consts import STABILITY_CENTER_TOLERANCE, STABILITY_SPREAD_TOLERANCE
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles

# Track created config files (jest configs and tsconfigs) for cleanup
_created_config_files: set[Path] = set()


def get_created_config_files() -> list[Path]:
    """Get list of config files created by codeflash for cleanup.

    Returns:
        List of paths to created config files (jest.codeflash.config.js, tsconfig.codeflash.json)
        that should be cleaned up after optimization.

    """
    return list(_created_config_files)


def clear_created_config_files() -> None:
    """Clear the set of tracked config files after cleanup."""
    _created_config_files.clear()


def _detect_bundler_module_resolution(project_root: Path) -> bool:
    """Detect if the project uses moduleResolution: 'bundler' in tsconfig.

    TypeScript 5+ supports 'bundler' moduleResolution which requires
    module: 'preserve' or ES2015+. This can cause issues with ts-jest
    in some configurations.

    This function also resolves extended tsconfigs to find bundler setting
    in parent configs.

    Args:
        project_root: Root of the project to check.

    Returns:
        True if the project uses bundler moduleResolution.

    """
    tsconfig_path = project_root / "tsconfig.json"
    if not tsconfig_path.exists():
        return False

    visited_configs: set[Path] = set()

    def check_tsconfig(config_path: Path) -> bool:
        """Recursively check tsconfig and its extends for bundler moduleResolution."""
        if config_path in visited_configs:
            return False
        visited_configs.add(config_path)

        if not config_path.exists():
            return False

        try:
            content = config_path.read_text()
            tsconfig = json.loads(content)

            # Check direct moduleResolution setting
            compiler_options = tsconfig.get("compilerOptions", {})
            module_resolution = compiler_options.get("moduleResolution", "").lower()
            if module_resolution == "bundler":
                return True

            # Check extended config if present
            extends = tsconfig.get("extends")
            if extends:
                # Resolve the extended config path
                if extends.startswith("."):
                    # Relative path
                    extended_path = (config_path.parent / extends).resolve()
                    if not extended_path.suffix:
                        extended_path = extended_path.with_suffix(".json")
                else:
                    # Package reference (e.g., "@n8n/typescript-config/modern/tsconfig.json")
                    # Try to find it in node_modules
                    node_modules_path = project_root / "node_modules" / extends
                    if not node_modules_path.suffix:
                        node_modules_path = node_modules_path.with_suffix(".json")
                    if node_modules_path.exists():
                        extended_path = node_modules_path
                    else:
                        # Try parent directories for monorepo support
                        current = project_root.parent
                        extended_path = None
                        while current != current.parent:
                            candidate = current / "node_modules" / extends
                            if not candidate.suffix:
                                candidate = candidate.with_suffix(".json")
                            if candidate.exists():
                                extended_path = candidate
                                break
                            # Also check packages directory for workspace packages
                            packages_candidate = current / "packages" / extends
                            if not packages_candidate.suffix:
                                packages_candidate = packages_candidate.with_suffix(".json")
                            if packages_candidate.exists():
                                extended_path = packages_candidate
                                break
                            current = current.parent

                if extended_path and extended_path.exists():
                    return check_tsconfig(extended_path)

            return False
        except Exception as e:
            logger.debug(f"Failed to read {config_path}: {e}")
            return False

    return check_tsconfig(tsconfig_path)


def _create_codeflash_tsconfig(project_root: Path) -> Path:
    """Create a codeflash-compatible tsconfig for projects using bundler moduleResolution.

    This creates a tsconfig that inherits from the project's tsconfig but overrides
    moduleResolution to 'Node' for compatibility with ts-jest.

    Args:
        project_root: Root of the project.

    Returns:
        Path to the created tsconfig.codeflash.json file.

    """
    codeflash_tsconfig_path = project_root / "tsconfig.codeflash.json"

    # If it already exists, use it
    if codeflash_tsconfig_path.exists():
        logger.debug(f"Using existing {codeflash_tsconfig_path}")
        return codeflash_tsconfig_path

    # Read the original tsconfig to preserve most settings
    original_tsconfig_path = project_root / "tsconfig.json"
    try:
        original_content = original_tsconfig_path.read_text()
        original_tsconfig = json.loads(original_content)
    except Exception:
        original_tsconfig = {}

    # Create a new tsconfig that extends the original but fixes moduleResolution
    codeflash_tsconfig = {
        "extends": "./tsconfig.json",
        "compilerOptions": {
            # Override bundler to Node for ts-jest compatibility
            "moduleResolution": "Node",
            # Ensure module is set to a compatible value
            "module": "ESNext",
            # These are generally safe defaults for testing
            "esModuleInterop": True,
            "skipLibCheck": True,
            "isolatedModules": True,
        },
    }

    # Preserve include/exclude from original if not in extends
    if "include" in original_tsconfig:
        codeflash_tsconfig["include"] = original_tsconfig["include"]
    if "exclude" in original_tsconfig:
        codeflash_tsconfig["exclude"] = original_tsconfig["exclude"]

    try:
        codeflash_tsconfig_path.write_text(json.dumps(codeflash_tsconfig, indent=2))
        _created_config_files.add(codeflash_tsconfig_path)
        logger.debug(f"Created {codeflash_tsconfig_path} with Node moduleResolution")
    except Exception as e:
        logger.warning(f"Failed to create codeflash tsconfig: {e}")

    return codeflash_tsconfig_path


def _has_ts_jest_dependency(project_root: Path) -> bool:
    """Check if the project has ts-jest as a dependency.

    Args:
        project_root: Root of the project.

    Returns:
        True if ts-jest is found in dependencies or devDependencies.

    """
    package_json = project_root / "package.json"
    if not package_json.exists():
        return False

    try:
        content = json.loads(package_json.read_text())
        deps = {**content.get("dependencies", {}), **content.get("devDependencies", {})}
        return "ts-jest" in deps
    except (json.JSONDecodeError, OSError):
        return False


def _create_codeflash_jest_config(
    project_root: Path, original_jest_config: Path | None, *, for_esm: bool = False
) -> Path | None:
    """Create a Jest config that handles ESM packages and TypeScript properly.

    Args:
        project_root: Root of the project.
        original_jest_config: Path to the original Jest config, or None.
        for_esm: If True, configure for ESM package transformation.

    Returns:
        Path to the codeflash Jest config, or None if creation failed.

    """
    # For ESM projects (type: module), use .cjs extension since config uses CommonJS require/module.exports
    # This prevents "ReferenceError: module is not defined" errors
    is_esm = _is_esm_project(project_root)
    config_ext = ".cjs" if is_esm else ".js"

    # Create codeflash config in the same directory as the original config
    # This ensures relative paths work correctly
    if original_jest_config:
        codeflash_jest_config_path = original_jest_config.parent / f"jest.codeflash.config{config_ext}"
    else:
        codeflash_jest_config_path = project_root / f"jest.codeflash.config{config_ext}"

    # If it already exists, use it (check both extensions)
    if codeflash_jest_config_path.exists():
        logger.debug(f"Using existing {codeflash_jest_config_path}")
        return codeflash_jest_config_path

    # Also check if the alternate extension exists
    alt_ext = ".js" if is_esm else ".cjs"
    alt_path = codeflash_jest_config_path.with_suffix(alt_ext)
    if alt_path.exists():
        logger.debug(f"Using existing {alt_path}")
        return alt_path

    # Common ESM-only packages that need to be transformed
    # These packages ship only ESM and will cause "Cannot use import statement" errors
    esm_packages = [
        "p-queue",
        "p-limit",
        "p-timeout",
        "yocto-queue",
        "eventemitter3",
        "chalk",
        "ora",
        "strip-ansi",
        "ansi-regex",
        "string-width",
        "wrap-ansi",
        "is-unicode-supported",
        "is-interactive",
        "log-symbols",
        "figures",
    ]
    esm_pattern = "|".join(esm_packages)

    # Check if ts-jest is available in the project
    has_ts_jest = _has_ts_jest_dependency(project_root)

    # Build transform config only if ts-jest is available
    if has_ts_jest:
        transform_config = """
  // Ensure TypeScript files are transformed using ts-jest
  transform: {
    '^.+\\\\.(ts|tsx)$': ['ts-jest', { isolatedModules: true }],
    // Use ts-jest for JS files in ESM packages too
    '^.+\\\\.js$': ['ts-jest', { isolatedModules: true }],
  },"""
    else:
        transform_config = ""
        logger.debug("ts-jest not found in project dependencies, skipping transform config")

    # Create a wrapper Jest config
    if original_jest_config:
        # Since codeflash config is in the same directory as original, use simple relative path
        config_require_path = f"./{original_jest_config.name}"

        # Extend the original config
        jest_config_content = f"""// Auto-generated by codeflash for ESM compatibility
const originalConfig = require('{config_require_path}');

module.exports = {{
  ...originalConfig,
  // Transform ESM packages that don't work with Jest's default config
  // Pattern handles both npm/yarn (node_modules/pkg) and pnpm (node_modules/.pnpm/pkg@version/node_modules/pkg)
  transformIgnorePatterns: [
    'node_modules/(?!(\\\\.pnpm/)?({esm_pattern}))',
  ],{transform_config}
}};
"""
    else:
        # Create a minimal Jest config for TypeScript with ESM support
        jest_config_content = f"""// Auto-generated by codeflash for ESM compatibility
module.exports = {{
  verbose: true,
  testEnvironment: 'node',
  testRegex: '\\\\.(test|spec)\\\\.(js|ts|tsx)$',
  testPathIgnorePatterns: ['/dist/'],
  // Transform ESM packages that don't work with Jest's default config
  // Pattern handles both npm/yarn and pnpm directory structures
  transformIgnorePatterns: [
    'node_modules/(?!(\\\\.pnpm/)?({esm_pattern}))',
  ],{transform_config}
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
}};
"""

    try:
        codeflash_jest_config_path.write_text(jest_config_content)
        _created_config_files.add(codeflash_jest_config_path)
        logger.debug(f"Created {codeflash_jest_config_path} with ESM package support")
        return codeflash_jest_config_path
    except Exception as e:
        logger.warning(f"Failed to create codeflash Jest config: {e}")
        return None


def _is_multi_project_jest_config(config_path: Path) -> bool:
    """Check if a Jest config file uses a multi-project setup (projects: [...]).

    Multi-project Jest configs restrict test matching to each project's testMatch patterns,
    which prevents codeflash-generated tests from being discovered since they use a
    different naming convention (test_*.test.ts vs *.spec.ts).
    """
    if config_path is None or not config_path.exists():
        return False
    try:
        content = config_path.read_text(encoding="utf-8")
        # Look for "projects:" or "projects =" in the config — both TS and JS forms
        import re

        return bool(re.search(r"\bprojects\s*[=:]", content))
    except Exception:
        return False


def _create_flat_jest_config_for_generated_tests(project_root: Path, original_config: Path) -> Path | None:
    """Create a simple single-project Jest config for running codeflash-generated tests.

    When a project uses a multi-project Jest config (projects: [...]), each project has its own
    testMatch/testPathPattern. Generated tests (test_*.test.ts) don't match these patterns.
    This function creates a flat config that inherits transform/preset settings but uses
    a broad testMatch so generated tests can be discovered.
    """
    is_esm = _is_esm_project(project_root)
    config_ext = ".cjs" if is_esm else ".js"
    codeflash_config_path = original_config.parent / f"jest.codeflash.config{config_ext}"

    # Check if it already exists
    if codeflash_config_path.exists():
        return codeflash_config_path
    alt_ext = ".js" if is_esm else ".cjs"
    alt_path = codeflash_config_path.with_suffix(alt_ext)
    if alt_path.exists():
        return alt_path

    has_ts_jest = _has_ts_jest_dependency(project_root)

    if has_ts_jest:
        transform_block = """
  transform: {
    '^.+\\\\.(ts|tsx)$': ['ts-jest', { isolatedModules: true }],
    '^.+\\\\.js$': ['ts-jest', { isolatedModules: true }],
  },"""
    else:
        transform_block = ""

    # Create a flat single-project config (no 'projects' array) with broad testMatch
    jest_config_content = f"""// Auto-generated by codeflash — flat config for generated test files
// This replaces the multi-project config so codeflash-generated tests can be discovered
module.exports = {{
  rootDir: '{project_root.as_posix()}',
  testEnvironment: 'node',
  testMatch: ['**/*.test.ts', '**/*.test.js', '**/*.test.tsx', '**/*.test.jsx', '**/*.spec.ts', '**/*.spec.js'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],{transform_block}
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
}};
"""
    try:
        codeflash_config_path.write_text(jest_config_content)
        _created_config_files.add(codeflash_config_path)
        logger.info(f"Created flat Jest config for generated tests: {codeflash_config_path}")
        return codeflash_config_path
    except Exception as e:
        logger.warning(f"Failed to create flat Jest config: {e}")
        return None


def _get_jest_config_for_project(project_root: Path) -> Path | None:
    """Get the appropriate Jest config for the project.

    If the project uses bundler moduleResolution, creates and returns a
    codeflash-compatible Jest config. Otherwise, returns the project's
    existing Jest config. For multi-project Jest configs, creates a flat
    single-project config so codeflash-generated tests can be discovered.

    Args:
        project_root: Root of the project.

    Returns:
        Path to the Jest config to use, or None if not found.

    """
    # First check for existing Jest config
    original_jest_config = _find_jest_config(project_root)

    # Check if project uses bundler moduleResolution
    if _detect_bundler_module_resolution(project_root):
        logger.info("Detected bundler moduleResolution - creating compatible config")
        # Create codeflash-compatible tsconfig
        _create_codeflash_tsconfig(project_root)
        # Create codeflash Jest config that uses it
        codeflash_jest_config = _create_codeflash_jest_config(project_root, original_jest_config)
        if codeflash_jest_config:
            return codeflash_jest_config

    # Handle multi-project Jest configs (projects: [...]) — these restrict testMatch
    # per-project and prevent generated tests from being discovered
    if _is_multi_project_jest_config(original_jest_config):
        logger.info("Detected multi-project Jest config — creating flat config for generated tests")
        flat_config = _create_flat_jest_config_for_generated_tests(project_root, original_jest_config)
        if flat_config:
            return flat_config

    return original_jest_config


def _find_node_project_root(file_path: Path) -> Path | None:
    """Find the Node.js project root by looking for package.json.

    Traverses up from the given file path to find the nearest directory
    containing package.json or jest.config.js.

    Args:
        file_path: A file path within the Node.js project.

    Returns:
        The project root directory, or None if not found.

    """
    current = file_path.parent if file_path.is_file() else file_path
    while current != current.parent:  # Stop at filesystem root
        if (
            (current / "package.json").exists()
            or (current / "jest.config.js").exists()
            or (current / "jest.config.ts").exists()
            or (current / "tsconfig.json").exists()
        ):
            return current
        current = current.parent
    return None


def _find_monorepo_root(start_path: Path) -> Path | None:
    """Find the monorepo workspace root by looking for workspace markers.

    Traverses up from the given path to find a directory containing
    monorepo workspace markers like yarn.lock, pnpm-workspace.yaml, etc.

    Args:
        start_path: A path within the monorepo.

    Returns:
        The monorepo root directory, or None if not found.

    """
    monorepo_markers = ["yarn.lock", "pnpm-workspace.yaml", "lerna.json", "package-lock.json"]
    current = start_path if start_path.is_dir() else start_path.parent

    while current != current.parent:
        # Check for monorepo markers
        if any((current / marker).exists() for marker in monorepo_markers):
            # Verify it has node_modules (it's the workspace root)
            if (current / "node_modules").exists():
                return current
        current = current.parent

    return None


def _get_jest_major_version(project_root: Path) -> int | None:
    """Detect the major version of Jest installed in the project.

    Args:
        project_root: Root of the project to check.

    Returns:
        Major version number (e.g., 29, 30), or None if not detected.

    """
    # First try to check package.json for explicit version
    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            content = json.loads(package_json.read_text())
            deps = {**content.get("devDependencies", {}), **content.get("dependencies", {})}
            jest_version = deps.get("jest", "")
            # Parse version like "30.0.5", "^30.0.5", "~30.0.5"
            if jest_version:
                # Strip leading version prefixes (^, ~, =, v)
                version_str = jest_version.lstrip("^~=v")
                if version_str and version_str[0].isdigit():
                    major = version_str.split(".")[0]
                    if major.isdigit():
                        return int(major)
        except (json.JSONDecodeError, OSError):
            pass

    # Also check monorepo root
    monorepo_root = _find_monorepo_root(project_root)
    if monorepo_root and monorepo_root != project_root:
        monorepo_package = monorepo_root / "package.json"
        if monorepo_package.exists():
            try:
                content = json.loads(monorepo_package.read_text())
                deps = {**content.get("devDependencies", {}), **content.get("dependencies", {})}
                jest_version = deps.get("jest", "")
                if jest_version:
                    version_str = jest_version.lstrip("^~=v")
                    if version_str and version_str[0].isdigit():
                        major = version_str.split(".")[0]
                        if major.isdigit():
                            return int(major)
            except (json.JSONDecodeError, OSError):
                pass

    return None


def _find_jest_config(project_root: Path) -> Path | None:
    """Find Jest configuration file in the project.

    Searches for common Jest config file names in the project root and parent
    directories (for monorepo support). This is important for TypeScript projects
    that require specific transformation configurations (e.g., next/jest, ts-jest, babel-jest).

    Args:
        project_root: Root of the project to search.

    Returns:
        Path to Jest config file, or None if not found.

    """
    # Common Jest config file names, in order of preference
    config_names = ["jest.config.ts", "jest.config.js", "jest.config.mjs", "jest.config.cjs", "jest.config.json"]

    # First check the project root itself
    for config_name in config_names:
        config_path = project_root / config_name
        if config_path.exists():
            logger.debug(f"Found Jest config: {config_path}")
            return config_path

    # For monorepos, search parent directories up to the filesystem root
    # Stop at common monorepo root indicators (git root, package.json with workspaces)
    current = project_root.parent
    max_depth = 5  # Don't search too far up
    depth = 0

    while current != current.parent and depth < max_depth:
        for config_name in config_names:
            config_path = current / config_name
            if config_path.exists():
                logger.debug(f"Found Jest config in parent directory: {config_path}")
                return config_path

        # Check if this looks like a monorepo root
        package_json = current / "package.json"
        if package_json.exists():
            try:
                import json

                with package_json.open("r") as f:
                    pkg = json.load(f)
                    if "workspaces" in pkg:
                        # This is likely the monorepo root, stop here
                        break
            except Exception:
                pass

        # Check for git root as another stopping point
        if (current / ".git").exists():
            break

        current = current.parent
        depth += 1

    return None


def _is_esm_project(project_root: Path) -> bool:
    """Check if the project uses ES Modules.

    Detects ESM by checking package.json for "type": "module".

    Args:
        project_root: The project root directory.

    Returns:
        True if the project uses ES Modules, False otherwise.

    """
    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            with package_json.open("r") as f:
                pkg = json.load(f)
                return pkg.get("type") == "module"
        except Exception as e:
            logger.debug(f"Failed to read package.json: {e}")
    return False


def _uses_ts_jest(project_root: Path) -> bool:
    """Check if the project uses ts-jest for TypeScript transformation.

    ts-jest handles ESM transformation internally, so we don't need the
    --experimental-vm-modules flag when it's being used. Adding that flag
    can actually break Jest's module resolution for jest.mock() with relative paths.

    Args:
        project_root: The project root directory.

    Returns:
        True if ts-jest is being used, False otherwise.

    """
    # Check for ts-jest in devDependencies
    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            with package_json.open("r") as f:
                pkg = json.load(f)
                dev_deps = pkg.get("devDependencies", {})
                deps = pkg.get("dependencies", {})
                if "ts-jest" in dev_deps or "ts-jest" in deps:
                    return True
        except Exception as e:
            logger.debug(f"Failed to read package.json for ts-jest detection: {e}")

    # Also check for jest.config with ts-jest preset
    for config_file in ["jest.config.js", "jest.config.cjs", "jest.config.ts", "jest.config.mjs"]:
        config_path = project_root / config_file
        if config_path.exists():
            try:
                content = config_path.read_text()
                if "ts-jest" in content:
                    return True
            except Exception as e:
                logger.debug(f"Failed to read {config_file}: {e}")

    return False


_ENV_VAR_RE = __import__("re").compile(r"""([A-Z_][A-Z0-9_]*)=(?:'([^']*)'|"([^"]*)"|(\S+))""")


def _extract_env_vars_from_test_script(project_root: Path) -> dict[str, str]:
    """Extract environment variables from the project's jest/test scripts in package.json.

    Parses scripts like: TZ=UTC TS_NODE_COMPILER_OPTIONS='{"allowJs": false}' jest
    """
    pkg_json = project_root / "package.json"
    if not pkg_json.exists():
        return {}

    try:
        pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
        scripts = pkg.get("scripts", {})
    except Exception:
        return {}

    # Look for jest-related scripts: test, testunit, .testunit:jest, etc.
    script_keys = [k for k in scripts if "jest" in scripts[k].lower() or k in ("test", "testunit")]
    env_vars: dict[str, str] = {}
    for key in script_keys:
        script = scripts[key]
        for match in _ENV_VAR_RE.finditer(script):
            name = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4) or ""
            env_vars[name] = value

    if env_vars:
        logger.debug(f"Extracted env vars from package.json scripts: {list(env_vars.keys())}")
    return env_vars


def _configure_esm_environment(jest_env: dict[str, str], project_root: Path) -> None:
    """Configure environment variables for ES Module support in Jest.

    Jest requires --experimental-vm-modules flag for ESM support.
    This is passed via NODE_OPTIONS environment variable.

    IMPORTANT: When ts-jest is being used, we skip adding --experimental-vm-modules
    because ts-jest handles ESM transformation internally. Adding this flag can
    break Jest's module resolution for jest.mock() calls with relative paths.

    Args:
        jest_env: Environment variables dictionary to modify.
        project_root: The project root directory.

    """
    if _is_esm_project(project_root):
        # Skip if ts-jest is being used - it handles ESM internally and
        # --experimental-vm-modules breaks module resolution for relative mocks
        if _uses_ts_jest(project_root):
            logger.debug("Skipping --experimental-vm-modules: ts-jest handles ESM transformation")
            return

        logger.debug("Configuring Jest for ES Module support")
        existing_node_options = jest_env.get("NODE_OPTIONS", "")
        esm_flag = "--experimental-vm-modules"
        if esm_flag not in existing_node_options:
            jest_env["NODE_OPTIONS"] = f"{existing_node_options} {esm_flag}".strip()


def _ensure_runtime_files(project_root: Path) -> None:
    """Ensure JavaScript runtime package is installed in the project.

    Installs codeflash package if not already present.
    The package provides all runtime files needed for test instrumentation.
    Uses the project's detected package manager (npm, pnpm, yarn, or bun).

    Args:
        project_root: The project root directory.

    """
    node_modules_pkg = project_root / "node_modules" / "codeflash"
    if node_modules_pkg.exists():
        logger.debug("codeflash already installed")
        return

    install_cmd = get_package_install_command(project_root, "codeflash", dev=True)
    try:
        result = subprocess.run(install_cmd, check=False, cwd=project_root, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.debug(f"Installed codeflash using {install_cmd[0]}")
            return
        logger.warning(f"Failed to install codeflash: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error installing codeflash: {e}")

    logger.error(f"Could not install codeflash. Please install it manually: {' '.join(install_cmd)}")


def run_jest_behavioral_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    enable_coverage: bool = False,
    candidate_index: int = 0,
) -> tuple[Path, subprocess.CompletedProcess[str], Path | None, Path | None]:
    """Run Jest tests and return results in a format compatible with pytest output.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: JavaScript project root (directory containing package.json).
        enable_coverage: Whether to collect coverage information.
        candidate_index: Index of the candidate being tested.

    Returns:
        Tuple of (result_file_path, subprocess_result, coverage_json_path, None).

    """
    result_file_path = get_run_tmp_file(Path("jest_results.xml"))

    # Get test files to run
    test_files = [str(file.instrumented_behavior_file_path) for file in test_paths.test_files]
    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        first_test_file = Path(test_files[0])
        project_root = _find_node_project_root(first_test_file)

    # Use the project root, or fall back to provided cwd
    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Jest working directory: {effective_cwd}")

    # Ensure the codeflash npm package is installed
    _ensure_runtime_files(effective_cwd)

    # Coverage output directory
    coverage_dir = get_run_tmp_file(Path("jest_coverage"))
    coverage_json_path = coverage_dir / "coverage-final.json" if enable_coverage else None

    # Build Jest command
    jest_cmd = [
        "npx",
        "jest",
        "--reporters=default",
        "--reporters=jest-junit",
        "--runInBand",  # Run tests serially for consistent timing
        "--forceExit",
    ]

    # Add Jest config if found - needed for TypeScript transformation
    # Uses codeflash-compatible config if project has bundler moduleResolution
    jest_config = _get_jest_config_for_project(effective_cwd)
    if jest_config:
        jest_cmd.append(f"--config={jest_config}")

    # Add coverage flags if enabled
    if enable_coverage:
        jest_cmd.extend(["--coverage", "--coverageReporters=json", f"--coverageDirectory={coverage_dir}"])

    if test_files:
        jest_cmd.append("--runTestsByPath")
        resolved_test_files = [str(Path(f).resolve()) for f in test_files]
        jest_cmd.extend(resolved_test_files)
        # Add --roots to include directories containing test files
        # This is needed because some projects configure Jest with restricted roots
        # (e.g., roots: ["<rootDir>/src"]) which excludes the test directory
        test_dirs = {str(Path(f).resolve().parent) for f in test_files}
        for test_dir in sorted(test_dirs):
            jest_cmd.extend(["--roots", test_dir])

    if timeout:
        jest_cmd.append(f"--testTimeout={timeout * 1000}")  # Jest uses milliseconds

    # Set up environment
    jest_env = test_env.copy()
    jest_env["JEST_JUNIT_OUTPUT_FILE"] = str(result_file_path)
    jest_env["JEST_JUNIT_OUTPUT_DIR"] = str(result_file_path.parent)
    jest_env["JEST_JUNIT_OUTPUT_NAME"] = result_file_path.name
    # Configure jest-junit to use filepath-based classnames for proper parsing
    jest_env["JEST_JUNIT_CLASSNAME"] = "{filepath}"
    jest_env["JEST_JUNIT_SUITE_NAME"] = "{filepath}"
    jest_env["JEST_JUNIT_ADD_FILE_ATTRIBUTE"] = "true"
    # Include console.log output in JUnit XML for timing marker parsing
    jest_env["JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT"] = "true"
    # Set codeflash output file for the jest helper to write timing/behavior data (SQLite format)
    # Use candidate_index to differentiate between baseline (0) and optimization candidates
    codeflash_sqlite_file = get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite"))
    jest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    jest_env["CODEFLASH_TEST_ITERATION"] = str(candidate_index)
    jest_env["CODEFLASH_LOOP_INDEX"] = "1"
    jest_env["CODEFLASH_MODE"] = "behavior"
    # Seed random number generator for reproducible test runs across original and optimized code
    jest_env["CODEFLASH_RANDOM_SEED"] = "42"

    # Configure ESM support if project uses ES Modules
    _configure_esm_environment(jest_env, effective_cwd)

    # Extract env vars from project's test scripts (e.g. TZ, TS_NODE_COMPILER_OPTIONS)
    script_env_vars = _extract_env_vars_from_test_script(effective_cwd)
    for key, value in script_env_vars.items():
        if key not in jest_env:
            jest_env[key] = value

    # Increase Node.js heap size for large TypeScript projects
    # Default heap is often not enough for monorepos with many dependencies
    existing_node_options = jest_env.get("NODE_OPTIONS", "")
    if "--max-old-space-size" not in existing_node_options:
        jest_env["NODE_OPTIONS"] = f"{existing_node_options} --max-old-space-size=4096".strip()

    logger.debug(f"Running Jest tests with command: {' '.join(jest_cmd)}")

    # Calculate subprocess timeout: needs to be much larger than per-test timeout
    # to account for Jest startup, TypeScript compilation, module loading, etc.
    # Use at least 120 seconds, or 10x the per-test timeout, whichever is larger
    subprocess_timeout = max(120, (timeout or 15) * 10, 600) if timeout else 600

    start_time_ns = time.perf_counter_ns()
    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=jest_env, timeout=subprocess_timeout, check=False, text=True, capture_output=True
        )
        logger.debug(f"Jest subprocess timeout: {subprocess_timeout}s (per-test timeout: {timeout}s)")
        result = subprocess.run(jest_cmd, **run_args)  # noqa: PLW1510
        # Jest sends console.log output to stderr by default - move it to stdout
        # so our timing markers (printed via console.log) are in the expected place
        if result.stderr and not result.stdout:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stderr, stderr=""
            )
        elif result.stderr:
            # Combine stderr into stdout if both have content
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stdout + "\n" + result.stderr, stderr=""
            )
        logger.debug(f"Jest result: returncode={result.returncode}")
        # Log Jest output at WARNING level if tests fail and no XML output will be created
        # This helps debug issues like import errors that cause Jest to fail early
        if result.returncode != 0 and not result_file_path.exists():
            logger.warning(
                f"Jest failed with returncode={result.returncode}.\n"
                f"Jest stdout: {result.stdout[:2000] if result.stdout else '(empty)'}\n"
                f"Jest stderr: {result.stderr[:500] if result.stderr else '(empty)'}"
            )
    except subprocess.TimeoutExpired:
        logger.warning(f"Jest tests timed out after {subprocess_timeout}s")
        result = subprocess.CompletedProcess(args=jest_cmd, returncode=-1, stdout="", stderr="Test execution timed out")
    except FileNotFoundError:
        logger.error("Jest not found. Make sure Jest is installed (npm install jest)")
        result = subprocess.CompletedProcess(
            args=jest_cmd, returncode=-1, stdout="", stderr="Jest not found. Run: npm install jest jest-junit"
        )
    finally:
        wall_clock_ns = time.perf_counter_ns() - start_time_ns
        logger.debug(f"Jest behavioral tests completed in {wall_clock_ns / 1e9:.2f}s")

    print(result.stdout)
    return result_file_path, result, coverage_json_path, None


def _parse_timing_from_jest_output(stdout: str) -> dict[str, int]:
    """Parse timing data from Jest stdout markers.

    Extracts timing information from markers like:
    !######testModule:testFunc:funcName:loopIndex:invocationId:durationNs######!

    Args:
        stdout: Jest stdout containing timing markers.

    Returns:
        Dictionary mapping test case IDs to duration in nanoseconds.

    """
    import re

    # Pattern: !######module:testFunc:funcName:loopIndex:invocationId:durationNs######!
    pattern = re.compile(r"!######([^:]+):([^:]*):([^:]+):([^:]+):([^:]+):(\d+)######!")

    timings: dict[str, int] = {}
    for match in pattern.finditer(stdout):
        module, test_class, func_name, _loop_index, invocation_id, duration_ns = match.groups()
        # Create test case ID (same format as Python)
        test_id = f"{module}:{test_class}:{func_name}:{invocation_id}"
        timings[test_id] = int(duration_ns)

    return timings


def _should_stop_stability(
    runtimes: list[int],
    window: int,
    min_window_size: int,
    center_rel_tol: float = STABILITY_CENTER_TOLERANCE,
    spread_rel_tol: float = STABILITY_SPREAD_TOLERANCE,
) -> bool:
    """Check if performance has stabilized (matches Python's pytest_plugin.should_stop exactly).

    This function implements the same stability criteria as the Python pytest_plugin.py
    to ensure consistent behavior between Python and JavaScript performance testing.

    Args:
        runtimes: List of aggregate runtimes (sum of min per test case).
        window: Size of the window to check for stability.
        min_window_size: Minimum number of data points required.
        center_rel_tol: Center tolerance - all recent points must be within this fraction of median.
        spread_rel_tol: Spread tolerance - (max-min)/min must be within this fraction.

    Returns:
        True if performance has stabilized, False otherwise.

    """
    if len(runtimes) < window:
        return False

    if len(runtimes) < min_window_size:
        return False

    recent = runtimes[-window:]

    # Use sorted array for faster median and min/max operations
    recent_sorted = sorted(recent)
    mid = window // 2
    m = recent_sorted[mid] if window % 2 else (recent_sorted[mid - 1] + recent_sorted[mid]) / 2

    # 1) All recent points close to the median
    centered = True
    for r in recent:
        if abs(r - m) / m > center_rel_tol:
            centered = False
            break

    # 2) Window spread is small
    r_min, r_max = recent_sorted[0], recent_sorted[-1]
    if r_min == 0:
        return False
    spread_ok = (r_max - r_min) / r_min <= spread_rel_tol

    return centered and spread_ok


def run_jest_benchmarking_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    min_loops: int = 5,
    max_loops: int = 100,
    target_duration_ms: int = 10_000,  # 10 seconds for benchmarking tests
    stability_check: bool = True,
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    """Run Jest benchmarking tests with in-process session-level looping.

    Uses a custom Jest runner (codeflash/loop-runner) to loop all tests
    within a single Jest process, eliminating process startup overhead.

    This matches Python's pytest_plugin behavior:
    - All tests are run multiple times within a single Jest process
    - Timing data is collected per iteration
    - Stability is checked within the runner

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds for the entire benchmark run.
        project_root: JavaScript project root (directory containing package.json).
        min_loops: Minimum number of loop iterations.
        max_loops: Maximum number of loop iterations.
        target_duration_ms: Target TOTAL duration in milliseconds for all loops.
        stability_check: Whether to enable stability-based early stopping.

    Returns:
        Tuple of (result_file_path, subprocess_result with stdout from all iterations).

    """
    result_file_path = get_run_tmp_file(Path("jest_perf_results.xml"))

    # Get performance test files
    test_files = [str(file.benchmarking_file_path) for file in test_paths.test_files if file.benchmarking_file_path]
    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        first_test_file = Path(test_files[0])
        project_root = _find_node_project_root(first_test_file)

    effective_cwd = project_root if project_root else cwd

    # Ensure the codeflash npm package is installed
    _ensure_runtime_files(effective_cwd)

    # Detect Jest version for logging
    jest_major_version = _get_jest_major_version(effective_cwd)
    if jest_major_version:
        logger.debug(f"Jest {jest_major_version} detected - using loop-runner for batched looping")

    # Build Jest command for performance tests
    jest_cmd = [
        "npx",
        "jest",
        "--reporters=default",
        "--reporters=jest-junit",
        "--runInBand",  # Ensure serial execution
        "--forceExit",
        "--runner=codeflash/loop-runner",  # Use custom loop runner for in-process looping
    ]

    # Add Jest config if found - needed for TypeScript transformation
    # Uses codeflash-compatible config if project has bundler moduleResolution
    jest_config = _get_jest_config_for_project(effective_cwd)
    if jest_config:
        jest_cmd.append(f"--config={jest_config}")

    if test_files:
        jest_cmd.append("--runTestsByPath")
        resolved_test_files = [str(Path(f).resolve()) for f in test_files]
        jest_cmd.extend(resolved_test_files)
        # Add --roots to include directories containing test files
        test_dirs = {str(Path(f).resolve().parent) for f in test_files}
        for test_dir in sorted(test_dirs):
            jest_cmd.extend(["--roots", test_dir])

    if timeout:
        jest_cmd.append(f"--testTimeout={timeout * 1000}")

    # Base environment setup
    jest_env = test_env.copy()
    jest_env["JEST_JUNIT_OUTPUT_FILE"] = str(result_file_path)
    jest_env["JEST_JUNIT_OUTPUT_DIR"] = str(result_file_path.parent)
    jest_env["JEST_JUNIT_OUTPUT_NAME"] = result_file_path.name
    jest_env["JEST_JUNIT_CLASSNAME"] = "{filepath}"
    jest_env["JEST_JUNIT_SUITE_NAME"] = "{filepath}"
    jest_env["JEST_JUNIT_ADD_FILE_ATTRIBUTE"] = "true"
    jest_env["JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT"] = "true"

    # Pass monorepo root to loop-runner for jest-runner resolution
    monorepo_root = _find_monorepo_root(effective_cwd)
    if monorepo_root:
        jest_env["CODEFLASH_MONOREPO_ROOT"] = str(monorepo_root)
        logger.debug(f"Detected monorepo root: {monorepo_root}")
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
    jest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    jest_env["CODEFLASH_TEST_ITERATION"] = "0"
    jest_env["CODEFLASH_MODE"] = "performance"
    jest_env["CODEFLASH_RANDOM_SEED"] = "42"

    # Internal loop configuration for capturePerf (eliminates Jest environment overhead)
    # Looping happens inside capturePerf() for maximum efficiency
    jest_env["CODEFLASH_PERF_LOOP_COUNT"] = str(max_loops)
    jest_env["CODEFLASH_PERF_MIN_LOOPS"] = str(min_loops)
    jest_env["CODEFLASH_PERF_TARGET_DURATION_MS"] = str(target_duration_ms)
    jest_env["CODEFLASH_PERF_STABILITY_CHECK"] = "true" if stability_check else "false"
    jest_env["CODEFLASH_LOOP_INDEX"] = "1"  # Initial value for compatibility

    # Enable console output for timing markers
    # Some projects mock console.log in test setup (e.g., based on LOG_LEVEL or DEBUG)
    # We need console.log to work for capturePerf timing markers
    jest_env["LOG_LEVEL"] = "info"  # Disable console.log mocking in projects that check LOG_LEVEL
    jest_env["DEBUG"] = "1"  # Disable console.log mocking in projects that check DEBUG

    # Debug logging for loop behavior verification (set CODEFLASH_DEBUG_LOOPS=true to enable)
    if os.environ.get("CODEFLASH_DEBUG_LOOPS") == "true":
        jest_env["CODEFLASH_DEBUG_LOOPS"] = "true"
        logger.info("Loop debug logging enabled - will show capturePerf loop details")

    # Configure ESM support if project uses ES Modules
    _configure_esm_environment(jest_env, effective_cwd)

    # Extract env vars from project's test scripts (e.g. TZ, TS_NODE_COMPILER_OPTIONS)
    script_env_vars = _extract_env_vars_from_test_script(effective_cwd)
    for key, value in script_env_vars.items():
        if key not in jest_env:
            jest_env[key] = value

    # Increase Node.js heap size for large TypeScript projects
    existing_node_options = jest_env.get("NODE_OPTIONS", "")
    if "--max-old-space-size" not in existing_node_options:
        jest_env["NODE_OPTIONS"] = f"{existing_node_options} --max-old-space-size=4096".strip()

    # Total timeout for the entire benchmark run (longer than single-loop timeout)
    # Account for startup overhead + target duration + buffer
    total_timeout = max(120, (target_duration_ms // 1000) + 60, timeout or 120)

    logger.debug(f"Running Jest benchmarking tests with in-process loop runner: {' '.join(jest_cmd)}")
    logger.debug(
        f"Jest benchmarking config: min_loops={min_loops}, max_loops={max_loops}, "
        f"target_duration={target_duration_ms}ms, stability_check={stability_check}"
    )

    total_start_time = time.time()

    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=jest_env, timeout=total_timeout, check=False, text=True, capture_output=True
        )
        result = subprocess.run(jest_cmd, **run_args)  # noqa: PLW1510

        # Combine stderr into stdout for timing markers
        stdout = result.stdout or ""
        if result.stderr:
            stdout = stdout + "\n" + result.stderr if stdout else result.stderr

        # Create result with combined stdout
        result = subprocess.CompletedProcess(args=result.args, returncode=result.returncode, stdout=stdout, stderr="")
        if result.returncode != 0:
            logger.info(f"Jest benchmarking failed with return code {result.returncode}")
            logger.info(f"Jest benchmarking stdout: {result.stdout}")
            logger.info(f"Jest benchmarking stderr: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.warning(f"Jest benchmarking timed out after {total_timeout}s")
        result = subprocess.CompletedProcess(args=jest_cmd, returncode=-1, stdout="", stderr="Benchmarking timed out")
    except FileNotFoundError:
        logger.error("Jest not found for benchmarking")
        result = subprocess.CompletedProcess(args=jest_cmd, returncode=-1, stdout="", stderr="Jest not found")

    wall_clock_seconds = time.time() - total_start_time
    logger.debug(f"Jest benchmarking completed in {wall_clock_seconds:.2f}s")
    return result_file_path, result


def run_jest_line_profile_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    line_profile_output_file: Path | None = None,
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    """Run Jest tests for line profiling.

    This runs tests against source code that has been instrumented with line profiler.
    The instrumentation collects execution counts and timing per line.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds for the subprocess.
        project_root: JavaScript project root (directory containing package.json).
        line_profile_output_file: Path where line profile results will be written.

    Returns:
        Tuple of (result_file_path, subprocess_result).

    """
    result_file_path = get_run_tmp_file(Path("jest_line_profile_results.xml"))

    # Get test files to run - use instrumented behavior files if available, otherwise benchmarking files
    test_files = []
    for file in test_paths.test_files:
        if file.instrumented_behavior_file_path:
            test_files.append(str(file.instrumented_behavior_file_path))
        elif file.benchmarking_file_path:
            test_files.append(str(file.benchmarking_file_path))

    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        first_test_file = Path(test_files[0])
        project_root = _find_node_project_root(first_test_file)

    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Jest line profiling working directory: {effective_cwd}")

    # Ensure the codeflash npm package is installed
    _ensure_runtime_files(effective_cwd)

    # Build Jest command for line profiling - simple run without benchmarking loops
    jest_cmd = [
        "npx",
        "jest",
        "--reporters=default",
        "--reporters=jest-junit",
        "--runInBand",  # Run tests serially for consistent line profiling
        "--forceExit",
    ]

    # Add Jest config if found - needed for TypeScript transformation
    # Uses codeflash-compatible config if project has bundler moduleResolution
    jest_config = _get_jest_config_for_project(effective_cwd)
    if jest_config:
        jest_cmd.append(f"--config={jest_config}")

    if test_files:
        jest_cmd.append("--runTestsByPath")
        resolved_test_files = [str(Path(f).resolve()) for f in test_files]
        jest_cmd.extend(resolved_test_files)
        # Add --roots to include directories containing test files
        test_dirs = {str(Path(f).resolve().parent) for f in test_files}
        for test_dir in sorted(test_dirs):
            jest_cmd.extend(["--roots", test_dir])

    if timeout:
        jest_cmd.append(f"--testTimeout={timeout * 1000}")

    # Set up environment
    jest_env = test_env.copy()
    jest_env["JEST_JUNIT_OUTPUT_FILE"] = str(result_file_path)
    jest_env["JEST_JUNIT_OUTPUT_DIR"] = str(result_file_path.parent)
    jest_env["JEST_JUNIT_OUTPUT_NAME"] = result_file_path.name
    jest_env["JEST_JUNIT_CLASSNAME"] = "{filepath}"
    jest_env["JEST_JUNIT_SUITE_NAME"] = "{filepath}"
    jest_env["JEST_JUNIT_ADD_FILE_ATTRIBUTE"] = "true"
    jest_env["JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT"] = "true"
    # Set codeflash output file for the jest helper
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_line_profile.sqlite"))
    jest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    jest_env["CODEFLASH_TEST_ITERATION"] = "0"
    jest_env["CODEFLASH_LOOP_INDEX"] = "1"
    jest_env["CODEFLASH_MODE"] = "line_profile"
    # Seed random number generator for reproducibility
    jest_env["CODEFLASH_RANDOM_SEED"] = "42"
    # Pass the line profile output file path to the instrumented code
    if line_profile_output_file:
        jest_env["CODEFLASH_LINE_PROFILE_OUTPUT"] = str(line_profile_output_file)

    # Configure ESM support if project uses ES Modules
    _configure_esm_environment(jest_env, effective_cwd)

    # Extract env vars from project's test scripts (e.g. TZ, TS_NODE_COMPILER_OPTIONS)
    script_env_vars = _extract_env_vars_from_test_script(effective_cwd)
    for key, value in script_env_vars.items():
        if key not in jest_env:
            jest_env[key] = value

    # Increase Node.js heap size for large TypeScript projects
    existing_node_options = jest_env.get("NODE_OPTIONS", "")
    if "--max-old-space-size" not in existing_node_options:
        jest_env["NODE_OPTIONS"] = f"{existing_node_options} --max-old-space-size=4096".strip()

    subprocess_timeout = timeout or 600

    logger.debug(f"Running Jest line profile tests: {' '.join(jest_cmd)}")

    start_time_ns = time.perf_counter_ns()
    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=jest_env, timeout=subprocess_timeout, check=False, text=True, capture_output=True
        )
        result = subprocess.run(jest_cmd, **run_args)  # noqa: PLW1510
        # Jest sends console.log output to stderr by default - move it to stdout
        if result.stderr and not result.stdout:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stderr, stderr=""
            )
        elif result.stderr:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stdout + "\n" + result.stderr, stderr=""
            )
        logger.debug(f"Jest line profile result: returncode={result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Jest line profile tests timed out after {subprocess_timeout}s")
        result = subprocess.CompletedProcess(
            args=jest_cmd, returncode=-1, stdout="", stderr="Line profile tests timed out"
        )
    except FileNotFoundError:
        logger.error("Jest not found for line profiling")
        result = subprocess.CompletedProcess(args=jest_cmd, returncode=-1, stdout="", stderr="Jest not found")
    finally:
        wall_clock_ns = time.perf_counter_ns() - start_time_ns
        logger.debug(f"Jest line profile tests completed in {wall_clock_ns / 1e9:.2f}s")

    return result_file_path, result
