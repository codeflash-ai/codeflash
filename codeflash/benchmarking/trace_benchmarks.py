from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE


def trace_benchmarks_pytest(
    benchmarks_root: Path, tests_root: Path, project_root: Path, trace_file: Path, timeout: int = 300
) -> None:
    benchmark_env = os.environ.copy()
    if "PYTHONPATH" not in benchmark_env:
        benchmark_env["PYTHONPATH"] = str(project_root)
    else:
        benchmark_env["PYTHONPATH"] += os.pathsep + str(project_root)
    
    is_windows = sys.platform == "win32"
    cmd_list = [
        SAFE_SYS_EXECUTABLE,
        Path(__file__).parent / "pytest_new_process_trace_benchmarks.py",
        benchmarks_root,
        tests_root,
        trace_file,
    ]
    
    if is_windows:
        # Use Windows-safe subprocess handling to avoid file locking issues
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=project_root,
            env=benchmark_env,
            text=True,
            creationflags=creationflags,
        )
        try:
            stdout_content, stderr_content = process.communicate(timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                process.kill()
            stdout_content, stderr_content = process.communicate(timeout=5)
            raise subprocess.TimeoutExpired(
                cmd_list, timeout, output=stdout_content, stderr=stderr_content
            ) from None
        result = subprocess.CompletedProcess(cmd_list, returncode, stdout_content, stderr_content)
    else:
        result = subprocess.run(
            cmd_list,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            env=benchmark_env,
            timeout=timeout,
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
        logger.warning(f"Error collecting benchmarks - Pytest Exit code: {result.returncode}, {error_section}")
