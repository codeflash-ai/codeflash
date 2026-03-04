"""Mocha test runner for JavaScript/TypeScript.

This module provides functions for running Mocha tests for behavioral
verification and performance benchmarking. Uses Mocha's built-in JSON reporter
and converts the output to JUnit XML in Python, avoiding extra npm dependencies.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, SubElement, tostring

from codeflash.cli_cmds.console import logger
from codeflash.cli_cmds.init_javascript import get_package_install_command
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles


def _find_mocha_project_root(file_path: Path) -> Path | None:
    """Find the Mocha project root by looking for .mocharc.* or package.json.

    Traverses up from the given file path to find the directory containing
    a Mocha config file. Falls back to package.json if no Mocha config is found.

    Args:
        file_path: A file path within the Mocha project.

    Returns:
        The project root directory, or None if not found.

    """
    current = file_path.parent if file_path.is_file() else file_path
    package_json_dir = None

    mocha_config_names = (
        ".mocharc.yml",
        ".mocharc.yaml",
        ".mocharc.json",
        ".mocharc.js",
        ".mocharc.cjs",
        ".mocharc.mjs",
    )

    while current != current.parent:
        if any((current / cfg).exists() for cfg in mocha_config_names):
            return current
        if package_json_dir is None and (current / "package.json").exists():
            package_json_dir = current
        current = current.parent

    return package_json_dir


def _ensure_runtime_files(project_root: Path) -> None:
    """Ensure JavaScript runtime package is installed in the project.

    Installs codeflash package if not already present.
    The package provides all runtime files needed for test instrumentation.

    Args:
        project_root: The project root directory.

    """
    node_modules_pkg = project_root / "node_modules" / "codeflash"
    if node_modules_pkg.exists():
        logger.debug("codeflash already installed")
        return

    install_cmd = get_package_install_command(project_root, "codeflash", dev=True)
    try:
        result = subprocess.run(install_cmd, check=False, cwd=project_root, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.debug(f"Installed codeflash using {install_cmd[0]}")
            return
        logger.warning(f"Failed to install codeflash: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error installing codeflash: {e}")

    logger.error(f"Could not install codeflash. Please install it manually: {' '.join(install_cmd)}")


def mocha_json_to_junit_xml(json_str: str, output_file: Path) -> None:
    """Convert Mocha's JSON reporter output to JUnit XML.

    Mocha JSON format:
        { stats: {...}, tests: [...], failures: [...], passes: [...], pending: [...] }

    Each test object has: fullTitle, title, duration, err, ...

    Args:
        json_str: JSON string from Mocha's --reporter json output.
        output_file: Path to write the JUnit XML file.

    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Mocha JSON output")
        # Write a minimal empty JUnit XML so downstream parsing doesn't break
        output_file.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<testsuites />\n')
        return

    tests = data.get("tests", [])
    stats = data.get("stats", {})

    testsuites = Element("testsuites")
    testsuites.set("tests", str(stats.get("tests", len(tests))))
    testsuites.set("failures", str(stats.get("failures", 0)))
    testsuites.set("time", str((stats.get("duration", 0) or 0) / 1000.0))

    # Group tests by suite (parent describe block)
    suites: dict[str, list[dict]] = {}
    for test in tests:
        full_title = test.get("fullTitle", "")
        title = test.get("title", "")
        # Suite name = fullTitle minus the test's own title
        suite_name = full_title[: -len(title)].strip() if title and full_title.endswith(title) else "root"
        suite_name = suite_name or "root"
        suites.setdefault(suite_name, []).append(test)

    for suite_name, suite_tests in suites.items():
        testsuite = SubElement(testsuites, "testsuite")
        testsuite.set("name", suite_name)
        testsuite.set("tests", str(len(suite_tests)))

        suite_failures = 0
        suite_time = 0.0

        for test in suite_tests:
            testcase = SubElement(testsuite, "testcase")
            testcase.set("classname", suite_name)
            testcase.set("name", test.get("title", "unknown"))
            duration_ms = test.get("duration", 0) or 0
            duration_s = duration_ms / 1000.0
            testcase.set("time", str(duration_s))
            suite_time += duration_s

            err = test.get("err", {})
            if err and err.get("message"):
                suite_failures += 1
                failure = SubElement(testcase, "failure")
                failure.set("message", err.get("message", ""))
                failure.text = err.get("stack", err.get("message", ""))

            if test.get("pending"):
                skipped = SubElement(testcase, "skipped")
                skipped.set("message", "pending")

        testsuite.set("failures", str(suite_failures))
        testsuite.set("time", str(suite_time))

    xml_bytes = tostring(testsuites, encoding="unicode")
    output_file.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_bytes}\n')


def _extract_mocha_json(stdout: str) -> str | None:
    """Extract Mocha JSON output from stdout that may contain mixed content.

    Mocha's JSON reporter writes the JSON blob to stdout, but other output
    (console.log from tests, codeflash markers) may be interleaved.
    We look for the JSON object by finding the outermost { ... } that
    contains the expected "stats" key.

    Args:
        stdout: Full stdout from the Mocha subprocess.

    Returns:
        The extracted JSON string, or None if not found.

    """
    # Try the whole stdout first
    stripped = stdout.strip()
    if stripped.startswith("{") and '"stats"' in stripped:
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    # Find the outermost JSON object containing "stats"
    depth = 0
    start = None
    for i, ch in enumerate(stdout):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = stdout[start : i + 1]
                if '"stats"' in candidate:
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        pass
                start = None

    return None


def _build_mocha_behavioral_command(
    test_files: list[Path], timeout: int | None = None, project_root: Path | None = None
) -> list[str]:
    """Build Mocha command for behavioral tests.

    Args:
        test_files: List of test files to run.
        timeout: Optional timeout in seconds (converted to ms for Mocha).
        project_root: Project root directory.

    Returns:
        Command list for subprocess execution.

    """
    cmd = ["npx", "mocha", "--reporter", "json", "--jobs", "1", "--exit"]

    if timeout:
        cmd.extend(["--timeout", str(timeout * 1000)])
    else:
        cmd.extend(["--timeout", "60000"])

    cmd.extend(str(f.resolve()) for f in test_files)

    return cmd


def _build_mocha_benchmarking_command(
    test_files: list[Path], timeout: int | None = None, project_root: Path | None = None
) -> list[str]:
    """Build Mocha command for benchmarking tests.

    Args:
        test_files: List of test files to run.
        timeout: Optional timeout in seconds (converted to ms for Mocha).
        project_root: Project root directory.

    Returns:
        Command list for subprocess execution.

    """
    cmd = ["npx", "mocha", "--reporter", "json", "--jobs", "1", "--exit"]

    if timeout:
        cmd.extend(["--timeout", str(timeout * 1000)])
    else:
        cmd.extend(["--timeout", "120000"])

    cmd.extend(str(f.resolve()) for f in test_files)

    return cmd


def _build_mocha_line_profile_command(
    test_files: list[Path], timeout: int | None = None, project_root: Path | None = None
) -> list[str]:
    """Build Mocha command for line profiling tests.

    Args:
        test_files: List of test files to run.
        timeout: Optional timeout in seconds (converted to ms for Mocha).
        project_root: Project root directory.

    Returns:
        Command list for subprocess execution.

    """
    cmd = ["npx", "mocha", "--reporter", "json", "--jobs", "1", "--exit"]

    if timeout:
        cmd.extend(["--timeout", str(timeout * 1000)])
    else:
        cmd.extend(["--timeout", "60000"])

    cmd.extend(str(f.resolve()) for f in test_files)

    return cmd


def _run_mocha_and_convert(
    mocha_cmd: list[str],
    mocha_env: dict[str, str],
    effective_cwd: Path,
    result_file_path: Path,
    subprocess_timeout: int,
    label: str,
) -> subprocess.CompletedProcess:
    """Run Mocha subprocess, extract JSON output, and convert to JUnit XML.

    Args:
        mocha_cmd: Mocha command list.
        mocha_env: Environment variables.
        effective_cwd: Working directory.
        result_file_path: Path to write JUnit XML.
        subprocess_timeout: Timeout in seconds.
        label: Label for log messages (e.g. "behavioral", "benchmarking").

    Returns:
        CompletedProcess with combined stdout/stderr.

    """
    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=mocha_env, timeout=subprocess_timeout, check=False, text=True, capture_output=True
        )
        result = subprocess.run(mocha_cmd, **run_args)  # noqa: PLW1510

        # Combine stderr into stdout
        stdout = result.stdout or ""
        if result.stderr:
            stdout = stdout + "\n" + result.stderr if stdout else result.stderr

        result = subprocess.CompletedProcess(args=result.args, returncode=result.returncode, stdout=stdout, stderr="")

        logger.debug(f"Mocha {label} result: returncode={result.returncode}")
        if result.returncode != 0:
            logger.warning(
                f"Mocha {label} failed with returncode={result.returncode}.\n"
                f"Command: {' '.join(mocha_cmd)}\n"
                f"Stdout: {stdout[:2000] if stdout else '(empty)'}"
            )

    except subprocess.TimeoutExpired:
        logger.warning(f"Mocha {label} tests timed out after {subprocess_timeout}s")
        result = subprocess.CompletedProcess(
            args=mocha_cmd, returncode=-1, stdout="", stderr=f"{label} tests timed out"
        )
    except FileNotFoundError:
        logger.error("Mocha not found. Make sure Mocha is installed (npm install mocha)")
        result = subprocess.CompletedProcess(
            args=mocha_cmd, returncode=-1, stdout="", stderr="Mocha not found. Run: npm install mocha"
        )

    # Extract Mocha JSON from stdout and convert to JUnit XML
    if result.stdout:
        mocha_json = _extract_mocha_json(result.stdout)
        if mocha_json:
            mocha_json_to_junit_xml(mocha_json, result_file_path)
            logger.debug(f"Converted Mocha JSON to JUnit XML: {result_file_path}")
        else:
            logger.warning(f"Could not extract Mocha JSON from stdout (len={len(result.stdout)})")
            result_file_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<testsuites />\n')
    else:
        result_file_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<testsuites />\n')

    return result


def run_mocha_behavioral_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    enable_coverage: bool = False,
    candidate_index: int = 0,
) -> tuple[Path, subprocess.CompletedProcess, Path | None, Path | None]:
    """Run Mocha tests and return results in a format compatible with pytest output.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: Mocha project root (directory containing .mocharc.* or package.json).
        enable_coverage: Whether to collect coverage information (not yet supported for Mocha).
        candidate_index: Index of the candidate being tested.

    Returns:
        Tuple of (result_file_path, subprocess_result, coverage_json_path, None).

    """
    result_file_path = get_run_tmp_file(Path("mocha_results.xml"))

    test_files = [Path(file.instrumented_behavior_file_path) for file in test_paths.test_files]

    if project_root is None and test_files:
        project_root = _find_mocha_project_root(test_files[0])

    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Mocha working directory: {effective_cwd}")

    _ensure_runtime_files(effective_cwd)

    mocha_cmd = _build_mocha_behavioral_command(test_files=test_files, timeout=timeout, project_root=effective_cwd)

    mocha_env = test_env.copy()
    codeflash_sqlite_file = get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite"))
    mocha_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    mocha_env["CODEFLASH_TEST_ITERATION"] = str(candidate_index)
    mocha_env["CODEFLASH_LOOP_INDEX"] = "1"
    mocha_env["CODEFLASH_MODE"] = "behavior"
    mocha_env["CODEFLASH_RANDOM_SEED"] = "42"

    logger.debug(f"Running Mocha behavioral tests: {' '.join(mocha_cmd)}")

    subprocess_timeout = max(120, (timeout or 60) * 10)

    start_time_ns = time.perf_counter_ns()
    try:
        result = _run_mocha_and_convert(
            mocha_cmd=mocha_cmd,
            mocha_env=mocha_env,
            effective_cwd=effective_cwd,
            result_file_path=result_file_path,
            subprocess_timeout=subprocess_timeout,
            label="behavioral",
        )
    finally:
        wall_clock_ns = time.perf_counter_ns() - start_time_ns
        logger.debug(f"Mocha behavioral tests completed in {wall_clock_ns / 1e9:.2f}s")

    if result_file_path.exists():
        file_size = result_file_path.stat().st_size
        logger.debug(f"Mocha JUnit XML created: {result_file_path} ({file_size} bytes)")
    else:
        logger.warning(f"Mocha JUnit XML not created at {result_file_path}")

    return result_file_path, result, None, None


def run_mocha_benchmarking_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    min_loops: int = 5,
    max_loops: int = 100,
    target_duration_ms: int = 10_000,
    stability_check: bool = True,
) -> tuple[Path, subprocess.CompletedProcess]:
    """Run Mocha benchmarking tests with internal looping via capturePerf.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds for the entire benchmark run.
        project_root: Mocha project root.
        min_loops: Minimum number of loop iterations.
        max_loops: Maximum number of loop iterations.
        target_duration_ms: Target total duration in milliseconds for all loops.
        stability_check: Whether to enable stability-based early stopping.

    Returns:
        Tuple of (result_file_path, subprocess_result with stdout from all iterations).

    """
    result_file_path = get_run_tmp_file(Path("mocha_perf_results.xml"))

    test_files = [Path(file.benchmarking_file_path) for file in test_paths.test_files if file.benchmarking_file_path]

    logger.debug(
        f"Mocha benchmark test file selection: {len(test_files)}/{len(test_paths.test_files)} have benchmarking_file_path"
    )
    if not test_files:
        logger.warning("No perf test files found! Cannot run benchmarking tests.")

    if project_root is None and test_files:
        project_root = _find_mocha_project_root(test_files[0])

    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Mocha benchmarking working directory: {effective_cwd}")

    _ensure_runtime_files(effective_cwd)

    mocha_cmd = _build_mocha_benchmarking_command(test_files=test_files, timeout=timeout, project_root=effective_cwd)

    mocha_env = test_env.copy()
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
    mocha_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    mocha_env["CODEFLASH_TEST_ITERATION"] = "0"
    mocha_env["CODEFLASH_MODE"] = "performance"
    mocha_env["CODEFLASH_RANDOM_SEED"] = "42"

    mocha_env["CODEFLASH_PERF_LOOP_COUNT"] = str(max_loops)
    mocha_env["CODEFLASH_PERF_MIN_LOOPS"] = str(min_loops)
    mocha_env["CODEFLASH_PERF_TARGET_DURATION_MS"] = str(target_duration_ms)
    mocha_env["CODEFLASH_PERF_STABILITY_CHECK"] = "true" if stability_check else "false"
    mocha_env["CODEFLASH_LOOP_INDEX"] = "1"

    if test_files:
        test_module_path = str(
            test_files[0].relative_to(effective_cwd)
            if test_files[0].is_relative_to(effective_cwd)
            else test_files[0].name
        )
        mocha_env["CODEFLASH_TEST_MODULE"] = test_module_path

    total_timeout = max(120, (target_duration_ms // 1000) + 60, timeout or 120)

    logger.debug(f"Running Mocha benchmarking tests: {' '.join(mocha_cmd)}")
    logger.debug(
        f"Config: min_loops={min_loops}, max_loops={max_loops}, "
        f"target_duration={target_duration_ms}ms, stability_check={stability_check}"
    )

    total_start_time = time.time()
    try:
        result = _run_mocha_and_convert(
            mocha_cmd=mocha_cmd,
            mocha_env=mocha_env,
            effective_cwd=effective_cwd,
            result_file_path=result_file_path,
            subprocess_timeout=total_timeout,
            label="benchmarking",
        )
    finally:
        wall_clock_seconds = time.time() - total_start_time
        logger.debug(f"Mocha benchmarking completed in {wall_clock_seconds:.2f}s, returncode={result.returncode}")

    return result_file_path, result


def run_mocha_line_profile_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    line_profile_output_file: Path | None = None,
) -> tuple[Path, subprocess.CompletedProcess]:
    """Run Mocha tests for line profiling.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds for the subprocess.
        project_root: Mocha project root.
        line_profile_output_file: Path where line profile results will be written.

    Returns:
        Tuple of (result_file_path, subprocess_result).

    """
    result_file_path = get_run_tmp_file(Path("mocha_line_profile_results.xml"))

    test_files = []
    for file in test_paths.test_files:
        if file.instrumented_behavior_file_path:
            test_files.append(Path(file.instrumented_behavior_file_path))
        elif file.benchmarking_file_path:
            test_files.append(Path(file.benchmarking_file_path))

    if project_root is None and test_files:
        project_root = _find_mocha_project_root(test_files[0])

    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Mocha line profiling working directory: {effective_cwd}")

    _ensure_runtime_files(effective_cwd)

    mocha_cmd = _build_mocha_line_profile_command(test_files=test_files, timeout=timeout, project_root=effective_cwd)

    mocha_env = test_env.copy()
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_line_profile.sqlite"))
    mocha_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    mocha_env["CODEFLASH_TEST_ITERATION"] = "0"
    mocha_env["CODEFLASH_LOOP_INDEX"] = "1"
    mocha_env["CODEFLASH_MODE"] = "line_profile"
    mocha_env["CODEFLASH_RANDOM_SEED"] = "42"

    if line_profile_output_file:
        mocha_env["CODEFLASH_LINE_PROFILE_OUTPUT"] = str(line_profile_output_file)

    subprocess_timeout = max(120, (timeout or 60) * 10)

    logger.debug(f"Running Mocha line profile tests: {' '.join(mocha_cmd)}")

    start_time_ns = time.perf_counter_ns()
    try:
        result = _run_mocha_and_convert(
            mocha_cmd=mocha_cmd,
            mocha_env=mocha_env,
            effective_cwd=effective_cwd,
            result_file_path=result_file_path,
            subprocess_timeout=subprocess_timeout,
            label="line_profile",
        )
    finally:
        wall_clock_ns = time.perf_counter_ns() - start_time_ns
        logger.debug(f"Mocha line profile tests completed in {wall_clock_ns / 1e9:.2f}s")

    return result_file_path, result
