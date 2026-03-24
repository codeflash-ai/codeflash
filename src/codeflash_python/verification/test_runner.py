from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash_python.code_utils.shell_utils import get_cross_platform_subprocess_run_args
from codeflash_python.models.test_result import TestResult
from codeflash_python.verification.addopts import custom_addopts

if TYPE_CHECKING:
    import threading
    from collections.abc import Sequence

# Pattern to extract timing from stdout markers: !######module:class.test:func:loop:id:duration######!

logger = logging.getLogger("codeflash_python")

_TIMING_MARKER_PATTERN = re.compile(r"!######.+:(\d+)######!")


def calculate_utilization_fraction(stdout: str, wall_clock_ns: int, test_type: str = "unknown") -> None:
    """Calculate and log the function utilization fraction.

    Utilization = sum(function_runtimes_from_markers) / total_wall_clock_time

    This metric shows how much of the test execution time was spent in actual
    function calls vs overhead (test framework, I/O, etc.).

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
        logger.debug("[%s] No timing markers found in stdout, cannot calculate utilization", test_type)
        return

    # Sum all function runtimes
    total_function_runtime_ns = sum(int(m) for m in matches)

    # Calculate utilization fraction
    utilization = total_function_runtime_ns / wall_clock_ns if wall_clock_ns > 0 else 0
    utilization_pct = utilization * 100

    # Log metrics
    logger.debug(
        "[%s] Function Utilization Fraction: %.2f%% "
        "(function_time=%.1fms, wall_time=%.1fms, overhead=%.1f%%, num_markers=%s)",
        test_type,
        utilization_pct,
        total_function_runtime_ns / 1e6,
        wall_clock_ns / 1e6,
        100 - utilization_pct,
        len(matches),
    )


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

    from codeflash_python.code_utils.code_utils import get_run_tmp_file
    from codeflash_python.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
    from codeflash_python.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE

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
    pytest_env["PYTEST_PLUGINS"] = "codeflash_python.verification.pytest_plugin"

    blocklisted_plugins = ["benchmark", "codspeed", "xdist", "sugar"]
    if min_loops > 1:
        blocklisted_plugins.extend(["cov", "profiling"])

    test_file_args = [str(f) for f in test_files]

    coverage_database_file: Path | None = None
    coverage_config_file: Path | None = None

    try:
        if enable_coverage:
            from codeflash_python.static_analysis.coverage_utils import prepare_coverage_files

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


def run_behavioral_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    enable_coverage: bool = False,
    candidate_index: int = 0,
) -> tuple[Path, Any, Path | None, Path | None]:
    import contextlib
    import shlex
    import sys

    from codeflash.models.models import TestType
    from codeflash_python.code_utils.code_utils import get_run_tmp_file
    from codeflash_python.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
    from codeflash_python.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
    from codeflash_python.static_analysis.coverage_utils import prepare_coverage_files

    blocklisted_plugins = ["benchmark", "codspeed", "xdist", "sugar"]

    test_files: list[str] = []
    for file in test_paths.test_files:
        if file.test_type == TestType.REPLAY_TEST:
            if file.tests_in_file:
                test_files.extend(
                    [
                        str(file.instrumented_behavior_file_path) + "::" + test.test_function
                        for test in file.tests_in_file
                    ]
                )
        else:
            test_files.append(str(file.instrumented_behavior_file_path))

    pytest_cmd_list = build_pytest_cmd(SAFE_SYS_EXECUTABLE, IS_POSIX)
    test_files = list(set(test_files))

    common_pytest_args = [
        "--capture=tee-sys",
        "-q",
        "--codeflash_loops_scope=session",
        "--codeflash_min_loops=1",
        "--codeflash_max_loops=1",
        f"--codeflash_seconds={TOTAL_LOOPING_TIME_EFFECTIVE}",
    ]
    if timeout is not None:
        common_pytest_args.append(f"--timeout={timeout}")

    result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
    result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]

    pytest_test_env = test_env.copy()
    pytest_test_env["PYTEST_PLUGINS"] = "codeflash_python.verification.pytest_plugin"

    coverage_database_file: Path | None = None
    coverage_config_file: Path | None = None

    if enable_coverage:
        coverage_database_file, coverage_config_file = prepare_coverage_files()
        pytest_test_env["NUMBA_DISABLE_JIT"] = str(1)
        pytest_test_env["TORCHDYNAMO_DISABLE"] = str(1)
        pytest_test_env["PYTORCH_JIT"] = str(0)
        pytest_test_env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
        pytest_test_env["TF_ENABLE_ONEDNN_OPTS"] = str(0)
        pytest_test_env["JAX_DISABLE_JIT"] = str(0)

        is_windows = sys.platform == "win32"
        if is_windows:
            if coverage_database_file.exists():
                with contextlib.suppress(PermissionError, OSError):
                    coverage_database_file.unlink()
        else:
            cov_erase = execute_test_subprocess(
                shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage erase"), cwd=cwd, env=pytest_test_env, timeout=30
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
        results = execute_test_subprocess(
            coverage_cmd + common_pytest_args + blocklist_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,
        )
        logger.debug("Result return code: %s, %s", results.returncode, results.stderr or "")
    else:
        blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins]

        results = execute_test_subprocess(
            pytest_cmd_list + common_pytest_args + blocklist_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,
        )
        logger.debug("Result return code: %s, %s", results.returncode, results.stderr or "")

    return result_file_path, results, coverage_database_file, coverage_config_file


def run_benchmarking_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    min_loops: int = 5,
    max_loops: int = 100_000,
    target_duration_seconds: float = 10.0,
) -> tuple[Path, Any]:

    from codeflash_python.code_utils.code_utils import get_run_tmp_file
    from codeflash_python.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE

    blocklisted_plugins = ["codspeed", "cov", "benchmark", "profiling", "xdist", "sugar"]

    pytest_cmd_list = build_pytest_cmd(SAFE_SYS_EXECUTABLE, IS_POSIX)
    test_files: list[str] = list({str(file.benchmarking_file_path) for file in test_paths.test_files})
    pytest_args = [
        "--capture=tee-sys",
        "-q",
        "--codeflash_loops_scope=session",
        f"--codeflash_min_loops={min_loops}",
        f"--codeflash_max_loops={max_loops}",
        f"--codeflash_seconds={target_duration_seconds}",
        "--codeflash_stability_check=true",
    ]
    if timeout is not None:
        pytest_args.append(f"--timeout={timeout}")

    result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
    result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]
    pytest_test_env = test_env.copy()
    pytest_test_env["PYTEST_PLUGINS"] = "codeflash_python.verification.pytest_plugin"
    blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins]
    results = execute_test_subprocess(
        pytest_cmd_list + pytest_args + blocklist_args + result_args + test_files,
        cwd=cwd,
        env=pytest_test_env,
        timeout=600,
    )
    return result_file_path, results


def run_line_profile_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    line_profile_output_file: Path | None = None,
) -> tuple[Path, Any]:

    from codeflash_python.code_utils.code_utils import get_run_tmp_file
    from codeflash_python.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
    from codeflash_python.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE

    blocklisted_plugins = ["codspeed", "cov", "benchmark", "profiling", "xdist", "sugar"]

    pytest_cmd_list = build_pytest_cmd(SAFE_SYS_EXECUTABLE, IS_POSIX)
    test_files: list[str] = list({str(file.benchmarking_file_path) for file in test_paths.test_files})
    pytest_args = [
        "--capture=tee-sys",
        "-q",
        "--codeflash_loops_scope=session",
        "--codeflash_min_loops=1",
        "--codeflash_max_loops=1",
        f"--codeflash_seconds={TOTAL_LOOPING_TIME_EFFECTIVE}",
    ]
    if timeout is not None:
        pytest_args.append(f"--timeout={timeout}")
    result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
    result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]
    pytest_test_env = test_env.copy()
    pytest_test_env["PYTEST_PLUGINS"] = "codeflash_python.verification.pytest_plugin"
    blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins]
    pytest_test_env["LINE_PROFILE"] = "1"
    results = execute_test_subprocess(
        pytest_cmd_list + pytest_args + blocklist_args + result_args + test_files,
        cwd=cwd,
        env=pytest_test_env,
        timeout=600,
    )
    return result_file_path, results


def process_generated_test_strings(
    generated_test_source: str,
    instrumented_behavior_test_source: str,
    instrumented_perf_test_source: str,
    function_to_optimize: Any,
    test_path: Path,
    test_cfg: Any,
    project_module_system: str | None,
) -> tuple[str, str, str]:
    from codeflash_python.code_utils.code_utils import get_run_tmp_file

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
