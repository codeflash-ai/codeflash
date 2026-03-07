from __future__ import annotations

import re
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import custom_addopts
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args
from codeflash.languages.registry import get_language_support

# Pattern to extract timing from stdout markers: !######...:<duration_ns>######!
# Jest markers have multiple colons: !######module:test:func:loop:id:duration######!
# Python markers: !######module:class.test:func:loop:id:duration######!
_TIMING_MARKER_PATTERN = re.compile(r"!######.+:(\d+)######!")


def _calculate_utilization_fraction(stdout: str, wall_clock_ns: int, test_type: str = "unknown") -> None:
    """Calculate and log the function utilization fraction.

    Utilization = sum(function_runtimes_from_markers) / total_wall_clock_time

    This metric shows how much of the test execution time was spent in actual
    function calls vs overhead (Jest startup, test framework, I/O, etc.).

    Args:
        stdout: The stdout from the test subprocess containing timing markers.
        wall_clock_ns: Total wall clock time for the subprocess in nanoseconds.
        test_type: Type of test for logging context (e.g., "behavioral", "performance").

    """
    if not stdout or wall_clock_ns <= 0:
        return

    # Extract all timing values from stdout markers
    matches = _TIMING_MARKER_PATTERN.findall(stdout)
    if not matches:
        logger.debug(f"[{test_type}] No timing markers found in stdout, cannot calculate utilization")
        return

    # Sum all function runtimes
    total_function_runtime_ns = sum(int(m) for m in matches)

    # Calculate utilization fraction
    utilization = total_function_runtime_ns / wall_clock_ns if wall_clock_ns > 0 else 0
    utilization_pct = utilization * 100

    # Log metrics
    logger.debug(
        f"[{test_type}] Function Utilization Fraction: {utilization_pct:.2f}% "
        f"(function_time={total_function_runtime_ns / 1e6:.1f}ms, "
        f"wall_time={wall_clock_ns / 1e6:.1f}ms, "
        f"overhead={100 - utilization_pct:.1f}%, "
        f"num_markers={len(matches)})"
    )


def _ensure_runtime_files(project_root: Path, language: str = "javascript") -> None:
    """Ensure runtime environment is set up for the project.

    For JavaScript/TypeScript: Installs codeflash npm package.
    Falls back to copying runtime files if package installation fails.

    Args:
        project_root: The project root directory.
        language: The programming language (e.g., "javascript", "typescript").

    """
    try:
        language_support = get_language_support(language)
    except (KeyError, ValueError):
        logger.debug(f"No language support found for {language}, skipping runtime file setup")
        return

    # Try to install npm package (for JS/TS) or other language-specific setup
    if language_support.ensure_runtime_environment(project_root):
        return  # Package installed successfully

    # Fall back to copying runtime files directly
    runtime_files = language_support.get_runtime_files()
    for runtime_file in runtime_files:
        dest_path = project_root / runtime_file.name
        # Always copy to ensure we have the latest version
        if not dest_path.exists() or dest_path.stat().st_mtime < runtime_file.stat().st_mtime:
            shutil.copy2(runtime_file, dest_path)
            logger.debug(f"Copied {runtime_file.name} to {project_root}")


def execute_test_subprocess(
    cmd_list: list[str], cwd: Path, env: dict[str, str] | None, timeout: int = 600
) -> subprocess.CompletedProcess:
    """Execute a subprocess with the given command list, working directory, environment variables, and timeout."""
    logger.debug(f"executing test run with command: {' '.join(cmd_list)}")
    with custom_addopts():
        run_args = get_cross_platform_subprocess_run_args(
            cwd=cwd, env=env, timeout=timeout, check=False, text=True, capture_output=True
        )
        return subprocess.run(cmd_list, **run_args)  # noqa: PLW1510
