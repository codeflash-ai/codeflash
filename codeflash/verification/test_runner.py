"""Standalone test runner for the PythonPlugin adapter.

Extracted from the codeflash-next-gen test runner, adapted to use codeflash imports.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.code_utils.code_utils import custom_addopts, get_run_tmp_file
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args
from codeflash.languages.base import TestResult

if TYPE_CHECKING:
    import threading
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_TIMING_MARKER_PATTERN = re.compile(r"!######.+:(\d+)######!")

PYTEST_CMD: str = "pytest"


def setup_pytest_cmd(pytest_cmd: str | None) -> None:
    global PYTEST_CMD
    PYTEST_CMD = pytest_cmd or "pytest"


def pytest_cmd_tokens(is_posix: bool) -> list[str]:
    import shlex

    return shlex.split(PYTEST_CMD, posix=is_posix)


def build_pytest_cmd(safe_sys_executable: str, is_posix: bool) -> list[str]:
    return [safe_sys_executable, "-m", *pytest_cmd_tokens(is_posix)]


def run_tests(
    test_files: Sequence[Path],
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    *,
    min_loops: int = 1,
    max_loops: int = 1,
    target_seconds: float | None = None,
    stability_check: bool = False,
    enable_coverage: bool = False,
) -> tuple[list[TestResult], Path, Path | None, Path | None]:
    import contextlib
    import shlex
    import sys

    from codeflash.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
    from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE

    if target_seconds is None:
        target_seconds = TOTAL_LOOPING_TIME_EFFECTIVE

    junit_xml = get_run_tmp_file(Path("pytest_results.xml"))

    pytest_args = [
        "--capture=tee-sys",
        "-q",
        "--codeflash_loops_scope=session",
        f"--codeflash_min_loops={min_loops}",
        f"--codeflash_max_loops={max_loops}",
        f"--codeflash_seconds={target_seconds}",
    ]
    if stability_check:
        pytest_args.append("--codeflash_stability_check=true")
    if timeout:
        pytest_args.append(f"--timeout={timeout}")

    result_args = [f"--junitxml={junit_xml.as_posix()}", "-o", "junit_logging=all"]

    pytest_env = env.copy()
    pytest_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"

    blocklisted_plugins = ["benchmark", "codspeed", "xdist", "sugar"]
    if min_loops > 1:
        blocklisted_plugins.extend(["cov", "profiling"])

    test_file_args = [str(f) for f in test_files]

    coverage_database_file: Path | None = None
    coverage_config_file: Path | None = None

    try:
        if enable_coverage:
            from codeflash.languages.python.static_analysis.coverage_utils import prepare_coverage_files

            coverage_database_file, coverage_config_file = prepare_coverage_files()
            pytest_env["NUMBA_DISABLE_JIT"] = str(1)
            pytest_env["TORCHDYNAMO_DISABLE"] = str(1)
            pytest_env["PYTORCH_JIT"] = str(0)
            pytest_env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
            pytest_env["TF_ENABLE_ONEDNN_OPTS"] = str(0)
            pytest_env["JAX_DISABLE_JIT"] = str(0)

            is_windows = sys.platform == "win32"
            if is_windows:
                if coverage_database_file.exists():
                    with contextlib.suppress(PermissionError, OSError):
                        coverage_database_file.unlink()
            else:
                cov_erase = execute_test_subprocess(
                    shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage erase"), cwd=cwd, env=pytest_env, timeout=30
                )
                logger.debug(cov_erase)

            coverage_cmd = [
                SAFE_SYS_EXECUTABLE,
                "-m",
                "coverage",
                "run",
                f"--rcfile={coverage_config_file.as_posix()}",
                "-m",
            ]
            coverage_cmd.extend(pytest_cmd_tokens(IS_POSIX))

            blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins if plugin != "cov"]
            result = execute_test_subprocess(
                coverage_cmd + pytest_args + blocklist_args + result_args + test_file_args,
                cwd=cwd,
                env=pytest_env,
                timeout=600,
            )
        else:
            pytest_cmd_list = build_pytest_cmd(SAFE_SYS_EXECUTABLE, IS_POSIX)
            blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins]

            result = execute_test_subprocess(
                pytest_cmd_list + pytest_args + blocklist_args + result_args + test_file_args,
                cwd=cwd,
                env=pytest_env,
                timeout=600,
            )

        logger.debug("Result return code: %s, %s", result.returncode, result.stderr or "")
        results = parse_test_results(junit_xml, result.stdout or "")
        return results, junit_xml, coverage_database_file, coverage_config_file

    except Exception as e:
        logger.exception("Test execution failed: %s", e)
        return [], junit_xml, coverage_database_file, coverage_config_file


def parse_test_results(junit_xml_path: Path, stdout: str) -> list[TestResult]:
    import xml.etree.ElementTree as ET

    results: list[TestResult] = []

    if not junit_xml_path.exists():
        return results

    try:
        tree = ET.parse(junit_xml_path)
        root = tree.getroot()

        for testcase in root.iter("testcase"):
            name = testcase.get("name", "unknown")
            classname = testcase.get("classname", "")
            time_str = testcase.get("time", "0")

            try:
                runtime_ns = int(float(time_str) * 1_000_000_000)
            except ValueError:
                runtime_ns = None

            failure = testcase.find("failure")
            error = testcase.find("error")
            passed = failure is None and error is None

            error_message = None
            if failure is not None:
                error_message = failure.get("message", failure.text)
            elif error is not None:
                error_message = error.get("message", error.text)

            test_file = Path(classname.replace(".", "/") + ".py") if classname else Path("unknown")

            results.append(
                TestResult(
                    test_name=name,
                    test_file=test_file,
                    passed=passed,
                    runtime_ns=runtime_ns,
                    error_message=error_message,
                    stdout=stdout,
                )
            )
    except Exception as e:
        logger.warning("Failed to parse JUnit XML: %s", e)

    return results


def process_generated_test_strings(
    generated_test_source: str,
    instrumented_behavior_test_source: str,
    instrumented_perf_test_source: str,
    function_to_optimize: object,
    test_path: Path,
    test_cfg: object,
    project_module_system: str | None,
) -> tuple[str, str, str]:
    temp_run_dir = get_run_tmp_file(Path()).as_posix()
    instrumented_behavior_test_source = instrumented_behavior_test_source.replace(
        "{codeflash_run_tmp_dir_client_side}", temp_run_dir
    )
    instrumented_perf_test_source = instrumented_perf_test_source.replace(
        "{codeflash_run_tmp_dir_client_side}", temp_run_dir
    )
    return generated_test_source, instrumented_behavior_test_source, instrumented_perf_test_source


def execute_test_subprocess(
    cmd_list: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    timeout: int = 600,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess:
    """Execute a subprocess with the given command list, working directory, environment variables, and timeout.

    If *cancel_event* is provided and becomes set while the process is running,
    the subprocess is terminated immediately and a CompletedProcess with
    returncode -15 is returned.
    """
    import time

    logger.debug("executing test run with command: %s", " ".join(cmd_list))
    with custom_addopts():
        if cancel_event is None:
            run_args = get_cross_platform_subprocess_run_args(
                cwd=cwd, env=env, timeout=timeout, check=False, text=True, capture_output=True
            )
            return subprocess.run(cmd_list, **run_args)  # type: ignore[no-matching-overload]  # noqa: PLW1510

        # Use Popen so we can poll for cancellation
        run_args = get_cross_platform_subprocess_run_args(
            cwd=cwd, env=env, timeout=None, check=False, text=True, capture_output=False
        )
        # Remove keys that don't apply to Popen
        run_args.pop("check", None)
        run_args.pop("timeout", None)
        run_args.pop("capture_output", None)
        proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **run_args)  # type: ignore[no-matching-overload]
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if cancel_event.is_set():
                    proc.terminate()
                    proc.wait(timeout=5)
                    return subprocess.CompletedProcess(cmd_list, -15, stdout="", stderr="cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.terminate()
                    proc.wait(timeout=5)
                    msg = f"Timed out after {timeout}s"
                    raise subprocess.TimeoutExpired(cmd_list, timeout, output="", stderr=msg)  # noqa: TRY301
                # Poll every 200ms
                cancel_event.wait(min(0.2, remaining))
            stdout, stderr = proc.communicate(timeout=5)
            return subprocess.CompletedProcess(cmd_list, proc.returncode, stdout=stdout or "", stderr=stderr or "")
        except BaseException:
            proc.kill()
            proc.wait()
            raise
