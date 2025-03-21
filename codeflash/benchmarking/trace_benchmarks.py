from __future__ import annotations

import re

from pytest import ExitCode

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
from pathlib import Path
import subprocess

def trace_benchmarks_pytest(benchmarks_root: Path, tests_root:Path, project_root: Path, trace_file: Path) -> None:
    result = subprocess.run(
        [
            SAFE_SYS_EXECUTABLE,
            Path(__file__).parent / "pytest_new_process_trace_benchmarks.py",
            benchmarks_root,
            tests_root,
            trace_file,
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(project_root)},
    )
    if result.returncode != 0:
        if "ERROR collecting" in result.stdout:
            # Pattern matches "===== ERRORS =====" (any number of =) and captures everything after
            error_pattern = r"={3,}\s*ERRORS\s*={3,}\n([\s\S]*?)(?:={3,}|$)"
            match = re.search(error_pattern, result.stdout)
            error_section = match.group(1) if match else result.stdout
        elif "FAILURES" in result.stdout:
            # Pattern matches "===== FAILURES =====" (any number of =) and captures everything after
            error_pattern = r"={3,}\s*FAILURES\s*={3,}\n([\s\S]*?)(?:={3,}|$)"
            match = re.search(error_pattern, result.stdout)
            error_section = match.group(1) if match else result.stdout
        else:
            error_section = result.stdout
        logger.warning(
            f"Error collecting benchmarks - Pytest Exit code: {result.returncode}={ExitCode(result.returncode).name}\n {error_section}"
        )