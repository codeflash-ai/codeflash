from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TestInvocationResult:
    test_module_path: str
    test_class_name: str | None
    test_function_name: str
    function_getting_tested: str | None
    loop_index: int
    iteration_id: str | None
    runtime_ns: int | None
    did_pass: bool
    timed_out: bool = False
    error_message: str | None = None


@dataclass
class BehavioralRunResult:
    run_id: str
    total_tests: int
    passed: int
    failed: int
    best_summed_runtime_ns: int
    test_results: list[TestInvocationResult]
    errors: list[str] = field(default_factory=list)


@dataclass
class CompareResult:
    equivalent: bool
    total_compared: int
    diffs: list[DiffEntry] = field(default_factory=list)


@dataclass
class DiffEntry:
    scope: str
    test_name: str
    original_value: str
    candidate_value: str
    original_passed: bool
    candidate_passed: bool


@dataclass
class SpeedupInfo:
    baseline_run_id: str
    baseline_runtime_ns: int
    candidate_runtime_ns: int
    performance_gain: float
    speedup_x: str
    speedup_pct: str


@dataclass
class BenchmarkRunResult:
    run_id: str
    best_summed_runtime_ns: int
    loops_executed: int
    test_results: list[TestInvocationResult]
    speedup: SpeedupInfo | None = None
