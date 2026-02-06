"""JavaScript tracer runner.

This module provides functionality to run JavaScript code with function tracing
enabled. It spawns a Node.js subprocess with the trace-runner.js script and
generates replay tests after tracing completes.

The tracer supports two modes:
1. Script mode: Trace a specific JavaScript file
2. Test mode: Trace tests running under Jest or Vitest

Usage from CLI:
    codeflash trace --language javascript script.js
    codeflash trace --language javascript --jest --testPathPattern=mytest
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)


def find_node_executable() -> Optional[Path]:
    """Find the Node.js executable.

    Returns:
        Path to node executable, or None if not found.

    """
    # Try common locations
    node_path = shutil.which("node")
    if node_path:
        return Path(node_path)

    # Try npx as fallback
    npx_path = shutil.which("npx")
    if npx_path:
        return Path(npx_path)

    return None


def find_trace_runner() -> Optional[Path]:
    """Find the trace-runner.js script.

    Returns:
        Path to trace-runner.js, or None if not found.

    """
    # First, try to find it in the installed codeflash npm package
    # Check common node_modules locations
    cwd = Path.cwd()

    # Check project-local node_modules
    local_path = cwd / "node_modules" / "codeflash" / "runtime" / "trace-runner.js"
    if local_path.exists():
        return local_path

    # Check global npm packages
    try:
        result = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, check=True)
        global_modules = Path(result.stdout.strip())
        global_path = global_modules / "codeflash" / "runtime" / "trace-runner.js"
        if global_path.exists():
            return global_path
    except Exception:
        pass

    # Fall back to the bundled version in the Python package
    bundled_path = Path(__file__).parent.parent.parent.parent / "packages" / "codeflash" / "runtime" / "trace-runner.js"
    if bundled_path.exists():
        return bundled_path

    return None


def run_javascript_tracer(args: Namespace, config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Run JavaScript code with function tracing enabled.

    Args:
        args: Command line arguments.
        config: Project configuration.
        project_root: Project root directory.

    Returns:
        Dictionary with tracing results:
        - success: Whether tracing succeeded
        - trace_file: Path to trace database
        - replay_test_file: Path to generated replay test (if any)
        - error: Error message (if failed)

    """
    result: dict[str, Any] = {"success": False, "trace_file": None, "replay_test_file": None, "error": None}

    # Find Node.js
    node_path = find_node_executable()
    if not node_path:
        result["error"] = "Node.js not found. Please install Node.js to use JavaScript tracing."
        logger.error(result["error"])
        return result

    # Find trace runner
    trace_runner_path = find_trace_runner()
    if not trace_runner_path:
        result["error"] = "trace-runner.js not found. Please install the codeflash npm package."
        logger.error(result["error"])
        return result

    # Determine output paths
    outfile = getattr(args, "outfile", None) or "codeflash.trace.sqlite"
    trace_file = Path(outfile).resolve()

    # Build environment
    env = os.environ.copy()
    env["CODEFLASH_TRACE_DB"] = str(trace_file)
    env["CODEFLASH_PROJECT_ROOT"] = str(project_root)

    # Set max function count
    max_count = getattr(args, "max_function_count", 256)
    env["CODEFLASH_MAX_FUNCTION_COUNT"] = str(max_count)

    # Set timeout if specified
    timeout = getattr(args, "tracer_timeout", None)
    if timeout:
        env["CODEFLASH_TRACER_TIMEOUT"] = str(timeout)

    # Set functions to trace if specified
    only_functions = getattr(args, "only_functions", None)
    if only_functions:
        env["CODEFLASH_FUNCTIONS"] = json.dumps(only_functions)

    # Build command
    cmd = [str(node_path), str(trace_runner_path)]

    # Add trace runner options
    cmd.extend(["--trace-db", str(trace_file)])
    cmd.extend(["--project-root", str(project_root)])

    if max_count:
        cmd.extend(["--max-function-count", str(max_count)])

    if timeout:
        cmd.extend(["--timeout", str(timeout)])

    if only_functions:
        cmd.extend(["--functions", json.dumps(only_functions)])

    # Determine mode and add appropriate flags
    is_module = getattr(args, "module", False)
    script_args = []

    # Get the remaining arguments after parsing
    if hasattr(args, "script_args"):
        script_args = args.script_args
    elif hasattr(args, "unknown_args"):
        script_args = args.unknown_args

    if is_module and script_args and script_args[0] == "jest":
        cmd.append("--jest")
        cmd.append("--")
        cmd.extend(script_args[1:])
    elif is_module and script_args and script_args[0] == "vitest":
        cmd.append("--vitest")
        cmd.append("--")
        cmd.extend(script_args[1:])
    elif script_args:
        # Regular script mode
        cmd.extend(script_args)

    # Run the tracer
    logger.info("Running JavaScript tracer: %s", " ".join(cmd))

    try:
        process = subprocess.run(cmd, cwd=project_root, env=env, capture_output=False, check=False)

        if process.returncode != 0:
            result["error"] = f"Tracing failed with exit code {process.returncode}"
            logger.error(result["error"])
            return result

    except Exception as e:
        result["error"] = f"Failed to run tracer: {e}"
        logger.exception(result["error"])
        return result

    # Check if trace file was created
    if not trace_file.exists():
        result["error"] = f"Trace file not created: {trace_file}"
        logger.error(result["error"])
        return result

    result["success"] = True
    result["trace_file"] = str(trace_file)

    # Generate replay test if not in trace-only mode
    trace_only = getattr(args, "trace_only", False)
    if not trace_only:
        replay_test_path = generate_replay_test(trace_file=trace_file, project_root=project_root, config=config)
        if replay_test_path:
            result["replay_test_file"] = str(replay_test_path)
            logger.info("Generated replay test: %s", replay_test_path)

    return result


def generate_replay_test(
    trace_file: Path, project_root: Path, config: dict[str, Any], output_path: Optional[Path] = None
) -> Optional[Path]:
    """Generate a replay test file from trace data.

    Args:
        trace_file: Path to trace SQLite database.
        project_root: Project root directory.
        config: Project configuration.
        output_path: Optional custom output path.

    Returns:
        Path to generated test file, or None if generation failed.

    """
    from codeflash.languages.javascript.replay_test import create_replay_test_file

    # Determine test framework from config or detect from project
    framework = detect_test_framework(project_root, config)

    # Determine output path
    if output_path is None:
        tests_root = config.get("tests_root", "tests")
        tests_dir = project_root / tests_root
        output_path = tests_dir / "codeflash_replay.test.js"

    return create_replay_test_file(
        trace_file=trace_file,
        output_path=output_path,
        framework=framework,
        max_run_count=100,
        project_root=project_root,
    )


def detect_test_framework(project_root: Path, config: dict[str, Any]) -> str:
    """Detect the test framework used by the project.

    Args:
        project_root: Project root directory.
        config: Project configuration.

    Returns:
        Test framework name ('jest' or 'vitest').

    """
    # Check config first
    if "test_framework" in config:
        framework: str = config["test_framework"]
        return framework

    # Check for vitest config files
    vitest_configs = ["vitest.config.js", "vitest.config.ts", "vitest.config.mjs"]
    for conf in vitest_configs:
        if (project_root / conf).exists():
            return "vitest"

    # Check for jest config files
    jest_configs = ["jest.config.js", "jest.config.ts", "jest.config.mjs", "jest.config.json"]
    for conf in jest_configs:
        if (project_root / conf).exists():
            return "jest"

    # Check package.json for test script
    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            with package_json.open() as f:
                pkg = json.load(f)
                test_script = pkg.get("scripts", {}).get("test", "")
                if "vitest" in test_script:
                    return "vitest"
                if "jest" in test_script:
                    return "jest"

                # Check dependencies
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "vitest" in deps:
                    return "vitest"
                if "jest" in deps:
                    return "jest"
        except Exception:
            pass

    # Default to Jest
    return "jest"


def check_javascript_tracer_available() -> bool:
    """Check if JavaScript tracing is available.

    Returns:
        True if all requirements are met for JavaScript tracing.

    """
    # Check for Node.js
    if not find_node_executable():
        return False

    # Check for trace runner
    if not find_trace_runner():
        return False

    return True


def get_tracer_requirements_message() -> str:
    """Get a message about tracer requirements.

    Returns:
        Human-readable message about what's needed for JavaScript tracing.

    """
    missing = []

    if not find_node_executable():
        missing.append("Node.js (v18+)")

    if not find_trace_runner():
        missing.append("codeflash npm package (npm install codeflash)")

    if not missing:
        return "All requirements met for JavaScript tracing."

    return "Missing requirements for JavaScript tracing:\n- " + "\n- ".join(missing)
