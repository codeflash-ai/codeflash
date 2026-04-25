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

    test_states: dict[str, _TestState] = {}
    benchmark_results: dict[str, _BenchmarkResult] = {}

    for event in events:
        action = event.get("Action")
        test_name = event.get("Test")
        package = event.get("Package", "")

        if test_name is None:
            continue

        if test_name not in test_states:
            test_states[test_name] = _TestState(package=package)

        state = test_states[test_name]

        if action == "output":
            output_text = event.get("Output", "")
            state.stdout += output_text
            bench_match = BENCHMARK_RE.search(output_text)
            if bench_match:
                bench_name = bench_match.group(1)
                iterations = int(bench_match.group(2))
                ns_per_op = float(bench_match.group(3))
                b_per_op = int(bench_match.group(4)) if bench_match.group(4) else None
                allocs_per_op = int(bench_match.group(5)) if bench_match.group(5) else None
                benchmark_results[bench_name] = _BenchmarkResult(
                    ns_per_op=ns_per_op, iterations=iterations, b_per_op=b_per_op, allocs_per_op=allocs_per_op
                )
        elif action == "pass":
            state.passed = True
            elapsed = event.get("Elapsed", 0)
            state.runtime_ns = int(elapsed * 1_000_000_000) if elapsed else None
        elif action == "fail":
            state.passed = False
            elapsed = event.get("Elapsed", 0)
            state.runtime_ns = int(elapsed * 1_000_000_000) if elapsed else None

    base_dir = test_config.tests_project_rootdir

    for test_name, state in test_states.items():
        if state.passed is None:
            continue

        test_file_path = _resolve_test_file(test_name, state.package, test_files, base_dir)
        test_type = _resolve_test_type(test_file_path, test_files)
        if test_type is None:
            logger.debug("Skipping test %s: could not resolve test type", test_name)
            continue

        runtime_ns = state.runtime_ns
        bench = benchmark_results.get(test_name)
        if bench is not None:
            runtime_ns = int(bench.ns_per_op)

        test_results.add(
            FunctionTestInvocation(
                loop_index=1,
                id=InvocationId(
                    test_module_path=state.package,
                    test_class_name=None,
                    test_function_name=test_name,
                    function_getting_tested="",
                    iteration_id="",
                ),
                file_name=test_file_path,
                runtime=runtime_ns,
                test_framework="go-test",
                did_pass=state.passed,
                test_type=test_type,
                return_value=None,
                timed_out=False,
                stdout=state.stdout,
            )
        )

    if not test_results:
        logger.info("No Go test results parsed from %s", test_json_path)
        if run_result is not None:
            logger.debug("stdout: %s\nstderr: %s", run_result.stdout, run_result.stderr)

    return test_results


class _TestState:
    __slots__ = ("package", "passed", "runtime_ns", "stdout")

    def __init__(self, package: str) -> None:
        self.package = package
        self.passed: bool | None = None
        self.runtime_ns: int | None = None
        self.stdout: str = ""


class _BenchmarkResult:
    __slots__ = ("allocs_per_op", "b_per_op", "iterations", "ns_per_op")

    def __init__(
        self, ns_per_op: float, iterations: int, b_per_op: int | None = None, allocs_per_op: int | None = None
    ) -> None:
        self.ns_per_op = ns_per_op
        self.iterations = iterations
        self.b_per_op = b_per_op
        self.allocs_per_op = allocs_per_op


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
