from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args, make_env_with_project_root


def trace_unit_tests_pytest(tests_root: Path, project_root: Path, trace_file: Path, timeout: int = 600) -> None:
    env = make_env_with_project_root(project_root)
    run_args = get_cross_platform_subprocess_run_args(
        cwd=project_root, env=env, timeout=timeout, check=False, text=True, capture_output=True
    )
    result = subprocess.run(  # noqa: PLW1510
        [
            SAFE_SYS_EXECUTABLE,
            Path(__file__).parent / "pytest_new_process_trace_unit_tests.py",
            str(tests_root),
            str(trace_file),
        ],
        **run_args,
    )
    if result.returncode != 0:
        combined_output = result.stdout
        if result.stderr:
            combined_output = combined_output + "\n" + result.stderr if combined_output else result.stderr

        if "ERROR collecting" in combined_output:
            error_pattern = r"={3,}\s*ERRORS\s*={3,}\n([\s\S]*?)(?:={3,}|$)"
            match = re.search(error_pattern, combined_output)
            error_section = match.group(1) if match else combined_output
        elif "FAILURES" in combined_output:
            error_pattern = r"={3,}\s*FAILURES\s*={3,}\n([\s\S]*?)(?:={3,}|$)"
            match = re.search(error_pattern, combined_output)
            error_section = match.group(1) if match else combined_output
        else:
            error_section = combined_output
        logger.warning(f"Error tracing unit tests - Pytest Exit code: {result.returncode}, {error_section}")
        logger.debug(f"Full pytest output:\n{combined_output}")


def time_unit_tests_pytest(
    test_files: list[Path], project_root: Path, output_json: Path, timeout: int = 300
) -> dict[str, int]:
    env = make_env_with_project_root(project_root)
    run_args = get_cross_platform_subprocess_run_args(
        cwd=project_root, env=env, timeout=timeout, check=False, text=True, capture_output=True
    )
    result = subprocess.run(  # noqa: PLW1510
        [
            SAFE_SYS_EXECUTABLE,
            Path(__file__).parent / "pytest_new_process_time_unit_tests.py",
            str(output_json),
            *[str(f) for f in test_files],
        ],
        **run_args,
    )
    if result.returncode != 0:
        combined_output = result.stdout
        if result.stderr:
            combined_output = combined_output + "\n" + result.stderr if combined_output else result.stderr
        logger.warning(f"Error timing unit tests - Pytest Exit code: {result.returncode}, {combined_output}")
        logger.debug(f"Full pytest output:\n{combined_output}")

    if output_json.exists():
        return json.loads(output_json.read_text(encoding="utf-8"))
    return {}
