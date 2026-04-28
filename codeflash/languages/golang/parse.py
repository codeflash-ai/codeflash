from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults

if TYPE_CHECKING:
    import subprocess
    from pathlib import Path

    from codeflash.models.models import TestFiles
    from codeflash.verification.verification_utils import TestConfig

logger = logging.getLogger(__name__)

BENCHMARK_RE = re.compile(
    r"^(Benchmark\w+)(?:-\d+)?\s+"
    r"(\d+)\s+"
    r"([\d.]+)\s+ns/op"
    r"(?:\s+(\d+)\s+B/op)?"
    r"(?:\s+(\d+)\s+allocs/op)?"
)


def parse_go_test_output(
    test_json_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
) -> TestResults:
    test_results = TestResults()

    content = _read_json_output(test_json_path, run_result)
    if not content:
        logger.warning("No Go test output to parse from %s", test_json_path)
        return test_results

    events = _parse_json_lines(content)
    if not events:
        logger.warning("No valid JSON events found in %s", test_json_path)
        return test_results

    iterations: list[_TestIteration] = []
    active: dict[str, _TestIteration] = {}

    for event in events:
        action = event.get("Action")
        test_name = event.get("Test")
        package = event.get("Package", "")

        if test_name is None:
            if action == "output":
                output_text = event.get("Output", "")
                bench_match = BENCHMARK_RE.search(output_text)
                if bench_match:
                    bench_name = bench_match.group(1)
                    it = _TestIteration(test_name=bench_name, package=package)
                    it.passed = True
                    it.bench_ns_per_op = float(bench_match.group(3))
                    it.bench_iterations = int(bench_match.group(2))
                    it.stdout = output_text
                    iterations.append(it)
            continue

        if action == "run":
            if test_name in active:
                iterations.append(active[test_name])
            active[test_name] = _TestIteration(test_name=test_name, package=package)
            continue

        it = active.get(test_name)
        if it is None:
            it = _TestIteration(test_name=test_name, package=package)
            active[test_name] = it

        if action == "output":
            output_text = event.get("Output", "")
            it.stdout += output_text
            bench_match = BENCHMARK_RE.search(output_text)
            if bench_match:
                it.bench_ns_per_op = float(bench_match.group(3))
                it.bench_iterations = int(bench_match.group(2))
        elif action in ("pass", "fail"):
            it.passed = action == "pass"
            elapsed = event.get("Elapsed", 0)
            it.elapsed_ns = int(elapsed * 1_000_000_000) if elapsed else None
            iterations.append(active.pop(test_name))

    for it in active.values():
        if it.passed is not None:
            iterations.append(it)

    loop_counters: dict[str, int] = {}
    base_dir = test_config.tests_project_rootdir

    for it in iterations:
        if it.passed is None:
            continue

        loop_index = loop_counters.get(it.test_name, 0) + 1
        loop_counters[it.test_name] = loop_index

        runtime_ns = it.bench_ns_per_op if it.bench_ns_per_op is not None else it.elapsed_ns
        if runtime_ns is not None:
            runtime_ns = int(runtime_ns)

        test_file_path = _resolve_test_file(it.test_name, it.package, test_files, base_dir)
        test_type = _resolve_test_type(test_file_path, test_files)
        if test_type is None:
            logger.debug("Skipping test %s: could not resolve test type", it.test_name)
            continue

        test_results.add(
            FunctionTestInvocation(
                loop_index=loop_index,
                id=InvocationId(
                    test_module_path=it.package,
                    test_class_name=None,
                    test_function_name=it.test_name,
                    function_getting_tested="",
                    iteration_id="",
                ),
                file_name=test_file_path,
                runtime=runtime_ns,
                test_framework="go-test",
                did_pass=it.passed,
                test_type=test_type,
                return_value=None,
                timed_out=False,
                stdout=it.stdout,
            )
        )

    if not test_results:
        logger.info("No Go test results parsed from %s", test_json_path)
        if run_result is not None:
            logger.debug("stdout: %s\nstderr: %s", run_result.stdout, run_result.stderr)

    logger.debug("[BENCHMARK-DONE] Got %d benchmark results", len(test_results))

    return test_results


class _TestIteration:
    __slots__ = ("bench_iterations", "bench_ns_per_op", "elapsed_ns", "package", "passed", "stdout", "test_name")

    def __init__(self, test_name: str, package: str) -> None:
        self.test_name = test_name
        self.package = package
        self.passed: bool | None = None
        self.elapsed_ns: int | None = None
        self.bench_ns_per_op: float | None = None
        self.bench_iterations: int | None = None
        self.stdout: str = ""


def _read_json_output(path: Path, run_result: subprocess.CompletedProcess | None) -> str:
    try:
        content = path.read_text(encoding="utf-8")
        if content.strip():
            return content
    except Exception:
        pass
    if run_result is not None:
        stdout = run_result.stdout
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        return stdout or ""
    return ""


def _parse_json_lines(content: str) -> list[dict]:
    events: list[dict] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _resolve_test_file(test_name: str, package: str, test_files: TestFiles, base_dir: Path) -> Path:

    for tf in test_files.test_files:
        behavior_path = tf.instrumented_behavior_file_path
        if behavior_path.exists():
            return behavior_path
        if tf.original_file_path and tf.original_file_path.exists():
            return tf.original_file_path

    if package:
        return base_dir / package.replace("/", "_")
    return base_dir / f"{test_name}.go"


def _resolve_test_type(test_file_path: Path, test_files: TestFiles):
    from codeflash.models.test_type import TestType

    test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
    if test_type is not None:
        return test_type
    test_type = test_files.get_test_type_by_original_file_path(test_file_path)
    if test_type is not None:
        return test_type
    if test_files.test_files:
        return test_files.test_files[0].test_type
    return TestType.GENERATED_REGRESSION
