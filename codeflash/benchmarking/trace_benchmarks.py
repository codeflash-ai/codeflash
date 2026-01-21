from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE, is_compiled_or_bundled_binary
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args


# Embedded benchmark tracing script for use when running as compiled binary
# Read the script content at module load time to embed it
_BENCHMARK_TRACING_SCRIPT_PATH = Path(__file__).parent / "pytest_new_process_trace_benchmarks.py"

# Read and store the script content as a constant at module import time
try:
    with open(_BENCHMARK_TRACING_SCRIPT_PATH, encoding="utf-8") as _f:
        _BENCHMARK_TRACING_SCRIPT_CONTENT = _f.read()
except Exception:
    # If we can't read it, set to None and we'll try the file path directly
    _BENCHMARK_TRACING_SCRIPT_CONTENT = None


def _get_benchmark_tracing_script_path() -> Path:
    """Get path to pytest_new_process_trace_benchmarks.py, creating it from embedded source if needed."""
    script_name = "pytest_new_process_trace_benchmarks.py"

    if is_compiled_or_bundled_binary():
        if _BENCHMARK_TRACING_SCRIPT_CONTENT is None:
            logger.error("Error: Benchmark tracing script content not embedded in compiled binary")
            raise RuntimeError("Benchmark tracing script content not available in compiled mode")

        # Write embedded script to a temporary file
        temp_dir = Path(tempfile.gettempdir()) / "codeflash_scripts"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_script_path = temp_dir / script_name

        # Write the embedded script content
        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(_BENCHMARK_TRACING_SCRIPT_CONTENT)

        return temp_script_path

    # When not compiled, use the original file
    return _BENCHMARK_TRACING_SCRIPT_PATH


def trace_benchmarks_pytest(
    benchmarks_root: Path, tests_root: Path, project_root: Path, trace_file: Path, timeout: int = 300
) -> None:
    benchmark_env = os.environ.copy()
    if "PYTHONPATH" not in benchmark_env:
        benchmark_env["PYTHONPATH"] = str(project_root)
    else:
        benchmark_env["PYTHONPATH"] += os.pathsep + str(project_root)
    run_args = get_cross_platform_subprocess_run_args(
        cwd=project_root, env=benchmark_env, timeout=timeout, check=False, text=True, capture_output=True
    )
    result = subprocess.run(  # noqa: PLW1510
        [
            SAFE_SYS_EXECUTABLE,
            _get_benchmark_tracing_script_path(),
            benchmarks_root,
            tests_root,
            trace_file,
        ],
        **run_args,
    )
    if result.returncode != 0:
        # Combine stdout and stderr for error reporting (errors often go to stderr)
        combined_output = result.stdout
        if result.stderr:
            combined_output = combined_output + "\n" + result.stderr if combined_output else result.stderr

        if "ERROR collecting" in combined_output:
            # Pattern matches "===== ERRORS =====" (any number of =) and captures everything after
            error_pattern = r"={3,}\s*ERRORS\s*={3,}\n([\s\S]*?)(?:={3,}|$)"
            match = re.search(error_pattern, combined_output)
            error_section = match.group(1) if match else combined_output
        elif "FAILURES" in combined_output:
            # Pattern matches "===== FAILURES =====" (any number of =) and captures everything after
            error_pattern = r"={3,}\s*FAILURES\s*={3,}\n([\s\S]*?)(?:={3,}|$)"
            match = re.search(error_pattern, combined_output)
            error_section = match.group(1) if match else combined_output
        else:
            error_section = combined_output
        logger.warning(f"Error collecting benchmarks - Pytest Exit code: {result.returncode}, {error_section}")
        logger.debug(f"Full pytest output:\n{combined_output}")
