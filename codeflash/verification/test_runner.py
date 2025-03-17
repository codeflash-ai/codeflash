from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from auditwall.core import SideEffectDetected

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME
from codeflash.code_utils.coverage_utils import prepare_coverage_files
from codeflash.models.models import TestFiles
from codeflash.verification.codeflash_auditwall import transform_code
from codeflash.verification.test_results import TestType

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles


def execute_test_subprocess(
    cmd_list: list[str], cwd: Path, env: dict[str, str] | None, timeout: int = 600
) -> subprocess.CompletedProcess:
    """Execute a subprocess with the given command list, working directory, environment variables, and timeout."""
    logger.debug(f"executing test run with command: {' '.join(cmd_list)}")
    return subprocess.run(cmd_list, capture_output=True, cwd=cwd, env=env, text=True, timeout=timeout, check=False)


def run_behavioral_tests(
    test_paths: TestFiles,
    test_framework: str,
    test_env: dict[str, str],
    cwd: Path,
    pytest_timeout: int | None = None,
    pytest_cmd: str = "pytest",
    verbose: bool = False,
    pytest_target_runtime_seconds: int = TOTAL_LOOPING_TIME,
    enable_coverage: bool = False,
) -> tuple[Path, subprocess.CompletedProcess, Path | None]:
    if test_framework not in ["pytest", "unittest"]:
        raise ValueError(f"Unsupported test framework: {test_framework}")

    test_files: list[str] = []
    for file in test_paths.test_files:
        if file.test_type == TestType.REPLAY_TEST:
            # TODO: Does this work for unittest framework?
            test_files.extend(
                [str(file.instrumented_behavior_file_path) + "::" + test.test_function for test in file.tests_in_file]
            )
        else:
            test_files.append(str(file.instrumented_behavior_file_path))

    source_code = next((file.original_source for file in test_paths.test_files if file.original_source), None)
    if not source_code:
        raise ValueError("No source code found for auditing")

    audit_code = transform_code(source_code)
    pytest_cmd_list = shlex.split(pytest_cmd, posix=IS_POSIX)
    common_pytest_args = [
        "--capture=tee-sys",
        f"--timeout={pytest_timeout}",
        "-q",
        "--codeflash_loops_scope=session",
        "--codeflash_min_loops=1",
        "--codeflash_max_loops=1",
        f"--codeflash_seconds={pytest_target_runtime_seconds}",
        "-p",
        "no:cacheprovider",
    ]

    result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
    result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]

    pytest_test_env = test_env.copy()
    pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"

    with tempfile.TemporaryDirectory(
        dir=Path(test_paths.test_files[0].instrumented_behavior_file_path).parent
    ) as temp_dir:
        audited_file_path = Path(temp_dir) / "audited_code.py"
        audited_file_path.write_text(audit_code, encoding="utf8")

        auditing_res = execute_test_subprocess(
            pytest_cmd_list + common_pytest_args + [audited_file_path.as_posix()],
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,
        )
        if auditing_res.returncode != 0:
            line_co = next(
                (
                    line
                    for line in auditing_res.stderr.splitlines() + auditing_res.stdout.splitlines()
                    if "codeflash.verification._auditwall.SideEffectDetectedError" in line
                ),
                None,
            )
            if line_co:
                match = re.search(r"codeflash has detected: (.+).", line_co)
                if match:
                    msg = match.group(1)
                    raise SideEffectDetected(msg)
                logger.debug(auditing_res.stderr)
            logger.debug(auditing_res.stdout)

    if test_framework == "pytest":
        coverage_database_file, coveragercfile = prepare_coverage_files()
        cov_erase = execute_test_subprocess(
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage erase"), cwd=cwd, env=pytest_test_env
        )
        logger.debug(cov_erase)

        results = execute_test_subprocess(
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage run --rcfile={coveragercfile.as_posix()} -m")
            + pytest_cmd_list
            + common_pytest_args
            + result_args
            + list(set(test_files)),  # remove duplicates
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,
        )
    else:  # unittest
        if enable_coverage:
            raise ValueError("Coverage is not supported yet for unittest framework")
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_files = [file.instrumented_behavior_file_path for file in test_paths.test_files]
        result_file_path, results = run_unittest_tests(verbose, test_files, test_env, cwd)

    return result_file_path, results, coverage_database_file if enable_coverage else None


def run_benchmarking_tests(
    test_paths: TestFiles,
    pytest_cmd: str,
    test_env: dict[str, str],
    cwd: Path,
    test_framework: str,
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME,
    verbose: bool = False,
    pytest_timeout: int | None = None,
    pytest_min_loops: int = 5,
    pytest_max_loops: int = 100_000,
):
    if test_framework == "pytest":
        pytest_cmd_list = shlex.split(pytest_cmd, posix=IS_POSIX)
        test_files: list[str] = []
        for file in test_paths.test_files:
            if file.test_type in [TestType.REPLAY_TEST, TestType.EXISTING_UNIT_TEST] and file.tests_in_file:
                test_files.extend(
                    [
                        str(file.benchmarking_file_path)
                        + "::"
                        + (test.test_class + "::" if test.test_class else "")
                        + (test.test_function.split("[", 1)[0] if "[" in test.test_function else test.test_function)
                        for test in file.tests_in_file
                    ]
                )
            else:
                test_files.append(str(file.benchmarking_file_path))
        test_files = list(set(test_files))  # remove multiple calls in the same test function
        pytest_args = [
            "--capture=tee-sys",
            f"--timeout={pytest_timeout}",
            "-q",
            "--codeflash_loops_scope=session",
            f"--codeflash_min_loops={pytest_min_loops}",
            f"--codeflash_max_loops={pytest_max_loops}",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
        ]
        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]
        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
        results = execute_test_subprocess(
            pytest_cmd_list + pytest_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,  # TODO: Make this dynamic
        )
    elif test_framework == "unittest":
        test_files = [file.benchmarking_file_path for file in test_paths.test_files]
        result_file_path, results = run_unittest_tests(verbose, test_files, test_env, cwd)
    else:
        raise ValueError(f"Unsupported test framework: {test_framework}")
    return result_file_path, results


def run_unittest_tests(verbose: bool, test_file_paths: list[Path], test_env: dict[str, str], cwd: Path):
    result_file_path = get_run_tmp_file(Path("unittest_results.xml"))
    unittest_cmd_list = [SAFE_SYS_EXECUTABLE, "-m", "xmlrunner"]
    log_level = ["-v"] if verbose else []
    files = [str(file) for file in test_file_paths]
    output_file = ["--output-file", str(result_file_path)]

    results = execute_test_subprocess(
        unittest_cmd_list + log_level + files + output_file, cwd=cwd, env=test_env, timeout=600
    )
    return result_file_path, results
