from __future__ import annotations

import contextlib
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import custom_addopts, get_run_tmp_file
from codeflash.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
from codeflash.languages.python.static_analysis.coverage_utils import prepare_coverage_files
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args
from codeflash.languages import is_python
from codeflash.languages.registry import get_language_support, get_language_support_by_framework
from codeflash.models.models import TestFiles, TestType

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles

BEHAVIORAL_BLOCKLISTED_PLUGINS = ["benchmark", "codspeed", "xdist", "sugar"]
BENCHMARKING_BLOCKLISTED_PLUGINS = ["codspeed", "cov", "benchmark", "profiling", "xdist", "sugar"]

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


def run_behavioral_tests(
    test_paths: TestFiles,
    test_framework: str,
    test_env: dict[str, str],
    cwd: Path,
    *,
    pytest_timeout: int | None = None,
    pytest_cmd: str = "pytest",
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME_EFFECTIVE,
    enable_coverage: bool = False,
    js_project_root: Path | None = None,
    candidate_index: int = 0,
) -> tuple[Path, subprocess.CompletedProcess, Path | None, Path | None]:
    """Run behavioral tests with optional coverage."""
    # Check if there's a language support for this test framework that implements run_behavioral_tests
    language_support = get_language_support_by_framework(test_framework)
    if language_support is not None and hasattr(language_support, "run_behavioral_tests"):
        # Java tests need longer timeout due to Maven startup overhead
        # Use Java-specific timeout if no explicit timeout provided
        from codeflash.code_utils.config_consts import JAVA_TESTCASE_TIMEOUT

        effective_timeout = pytest_timeout
        if test_framework in ("junit4", "junit5", "testng") and pytest_timeout is not None:
            # For Java, use a minimum timeout to account for Maven overhead
            effective_timeout = max(pytest_timeout, JAVA_TESTCASE_TIMEOUT)
            if effective_timeout != pytest_timeout:
                logger.debug(
                    f"Increased Java test timeout from {pytest_timeout}s to {effective_timeout}s "
                    "to account for Maven startup overhead"
                )

        return language_support.run_behavioral_tests(
            test_paths=test_paths,
            test_env=test_env,
            cwd=cwd,
            timeout=effective_timeout,
            project_root=js_project_root,
            enable_coverage=enable_coverage,
            candidate_index=candidate_index,
        )
    if is_python():
        test_files: list[str] = []
        for file in test_paths.test_files:
            if file.test_type == TestType.REPLAY_TEST:
                # Replay tests need specific test targeting because one file contains tests for multiple functions
                if file.tests_in_file:
                    test_files.extend(
                        [
                            str(file.instrumented_behavior_file_path) + "::" + test.test_function
                            for test in file.tests_in_file
                        ]
                    )
            else:
                test_files.append(str(file.instrumented_behavior_file_path))

        pytest_cmd_list = (
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m pytest", posix=IS_POSIX)
            if pytest_cmd == "pytest"
            else [SAFE_SYS_EXECUTABLE, "-m", *shlex.split(pytest_cmd, posix=IS_POSIX)]
        )
        test_files = list(set(test_files))  # remove multiple calls in the same test function

        common_pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=1",
            "--codeflash_max_loops=1",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
        ]
        if pytest_timeout is not None:
            common_pytest_args.append(f"--timeout={pytest_timeout}")

        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]

        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"

        if enable_coverage:
            coverage_database_file, coverage_config_file = prepare_coverage_files()
            # disable jit for coverage
            pytest_test_env["NUMBA_DISABLE_JIT"] = str(1)
            pytest_test_env["TORCHDYNAMO_DISABLE"] = str(1)
            pytest_test_env["PYTORCH_JIT"] = str(0)
            pytest_test_env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
            pytest_test_env["TF_ENABLE_ONEDNN_OPTS"] = str(0)
            pytest_test_env["JAX_DISABLE_JIT"] = str(0)

            is_windows = sys.platform == "win32"
            if is_windows:
                # On Windows, delete coverage database file directly instead of using 'coverage erase', to avoid locking issues
                if coverage_database_file.exists():
                    with contextlib.suppress(PermissionError, OSError):
                        coverage_database_file.unlink()
            else:
                cov_erase = execute_test_subprocess(
                    shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage erase"), cwd=cwd, env=pytest_test_env, timeout=30
                )  # this cleanup is necessary to avoid coverage data from previous runs, if there are any, then the current run will be appended to the previous data, which skews the results
                logger.debug(cov_erase)
            coverage_cmd = [
                SAFE_SYS_EXECUTABLE,
                "-m",
                "coverage",
                "run",
                f"--rcfile={coverage_config_file.as_posix()}",
                "-m",
            ]

            if pytest_cmd == "pytest":
                coverage_cmd.extend(["pytest"])
            else:
                coverage_cmd.extend(shlex.split(pytest_cmd, posix=IS_POSIX)[1:])

            blocklist_args = [f"-p no:{plugin}" for plugin in BEHAVIORAL_BLOCKLISTED_PLUGINS if plugin != "cov"]
            results = execute_test_subprocess(
                coverage_cmd + common_pytest_args + blocklist_args + result_args + test_files,
                cwd=cwd,
                env=pytest_test_env,
                # Timeout for test subprocess execution (seconds).
                # Override via CODEFLASH_TEST_TIMEOUT env var. Default: 600s.
                timeout=600,
            )
            logger.debug(
                f"Result return code: {results.returncode}, "
                f"{'Result stderr:' + str(results.stderr) if results.stderr else ''}"
            )
        else:
            blocklist_args = [f"-p no:{plugin}" for plugin in BEHAVIORAL_BLOCKLISTED_PLUGINS]

            results = execute_test_subprocess(
                pytest_cmd_list + common_pytest_args + blocklist_args + result_args + test_files,
                cwd=cwd,
                env=pytest_test_env,
                # Timeout for test subprocess execution (seconds).
                # Override via CODEFLASH_TEST_TIMEOUT env var. Default: 600s.
                timeout=600,
            )
            logger.debug(
                f"""Result return code: {results.returncode}, {"Result stderr:" + str(results.stderr) if results.stderr else ""}"""
            )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)

    return (
        result_file_path,
        results,
        coverage_database_file if enable_coverage else None,
        coverage_config_file if enable_coverage else None,
    )


def run_line_profile_tests(
    test_paths: TestFiles,
    pytest_cmd: str,
    test_env: dict[str, str],
    cwd: Path,
    test_framework: str,
    *,
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME_EFFECTIVE,
    pytest_timeout: int | None = None,
    pytest_min_loops: int = 5,
    pytest_max_loops: int = 100_000,
    js_project_root: Path | None = None,
    line_profiler_output_file: Path | None = None,
) -> tuple[Path, subprocess.CompletedProcess]:
    # Check if there's a language support for this test framework that implements run_line_profile_tests
    language_support = get_language_support_by_framework(test_framework)
    if language_support is not None and hasattr(language_support, "run_line_profile_tests"):
        from codeflash.code_utils.config_consts import JAVA_TESTCASE_TIMEOUT

        effective_timeout = pytest_timeout
        if test_framework in ("junit4", "junit5", "testng") and pytest_timeout is not None:
            # For Java, use a minimum timeout to account for Maven overhead
            effective_timeout = max(pytest_timeout, JAVA_TESTCASE_TIMEOUT)
        return language_support.run_line_profile_tests(
            test_paths=test_paths,
            test_env=test_env,
            cwd=cwd,
            timeout=effective_timeout,
            project_root=js_project_root,
            line_profile_output_file=line_profiler_output_file,
        )
    if is_python():  # pytest runs both pytest and unittest tests
        pytest_cmd_list = (
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m pytest", posix=IS_POSIX)
            if pytest_cmd == "pytest"
            else shlex.split(pytest_cmd)
        )
        # Always use file path - pytest discovers all tests including parametrized ones
        test_files: list[str] = list(
            {str(file.benchmarking_file_path) for file in test_paths.test_files}
        )  # remove multiple calls in the same test function
        pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=1",
            "--codeflash_max_loops=1",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
        ]
        if pytest_timeout is not None:
            pytest_args.append(f"--timeout={pytest_timeout}")
        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]
        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
        blocklist_args = [f"-p no:{plugin}" for plugin in BENCHMARKING_BLOCKLISTED_PLUGINS]
        pytest_test_env["LINE_PROFILE"] = "1"
        results = execute_test_subprocess(
            pytest_cmd_list + pytest_args + blocklist_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            # Timeout for line-profiling subprocess execution (seconds).
            # Override via CODEFLASH_TEST_TIMEOUT env var. Default: 600s.
            timeout=600,
        )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)
    return result_file_path, results


def run_benchmarking_tests(
    test_paths: TestFiles,
    pytest_cmd: str,
    test_env: dict[str, str],
    cwd: Path,
    test_framework: str,
    *,
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME_EFFECTIVE,
    pytest_timeout: int | None = None,
    pytest_min_loops: int = 5,
    pytest_max_loops: int = 100_000,
    js_project_root: Path | None = None,
) -> tuple[Path, subprocess.CompletedProcess]:
    # Check if there's a language support for this test framework that implements run_benchmarking_tests
    language_support = get_language_support_by_framework(test_framework)
    if language_support is not None and hasattr(language_support, "run_benchmarking_tests"):
        # Java tests need longer timeout due to Maven startup overhead
        # Use Java-specific timeout if no explicit timeout provided
        from codeflash.code_utils.config_consts import JAVA_TESTCASE_TIMEOUT

        effective_timeout = pytest_timeout
        if test_framework in ("junit4", "junit5", "testng") and pytest_timeout is not None:
            # For Java, use a minimum timeout to account for Maven overhead
            effective_timeout = max(pytest_timeout, JAVA_TESTCASE_TIMEOUT)
            if effective_timeout != pytest_timeout:
                logger.debug(
                    f"Increased Java test timeout from {pytest_timeout}s to {effective_timeout}s "
                    "to account for Maven startup overhead"
                )

        return language_support.run_benchmarking_tests(
            test_paths=test_paths,
            test_env=test_env,
            cwd=cwd,
            timeout=effective_timeout,
            project_root=js_project_root,
            min_loops=pytest_min_loops,
            max_loops=pytest_max_loops,
            target_duration_seconds=pytest_target_runtime_seconds,
        )
    if is_python():  # pytest runs both pytest and unittest tests
        pytest_cmd_list = (
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m pytest", posix=IS_POSIX)
            if pytest_cmd == "pytest"
            else shlex.split(pytest_cmd)
        )
        # Always use file path - pytest discovers all tests including parametrized ones
        test_files: list[str] = list(
            {str(file.benchmarking_file_path) for file in test_paths.test_files}
        )  # remove multiple calls in the same test function
        pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            f"--codeflash_min_loops={pytest_min_loops}",
            f"--codeflash_max_loops={pytest_max_loops}",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
            "--codeflash_stability_check=true",
        ]
        if pytest_timeout is not None:
            pytest_args.append(f"--timeout={pytest_timeout}")

        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]
        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
        blocklist_args = [f"-p no:{plugin}" for plugin in BENCHMARKING_BLOCKLISTED_PLUGINS]
        results = execute_test_subprocess(
            pytest_cmd_list + pytest_args + blocklist_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            # Timeout for benchmarking subprocess execution (seconds).
            # Override via CODEFLASH_TEST_TIMEOUT env var. Default: 600s.
            timeout=600,
        )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)
    return result_file_path, results
