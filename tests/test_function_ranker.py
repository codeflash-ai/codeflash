from pathlib import Path

import pytest

from codeflash.benchmarking.function_ranker import FunctionRanker
from codeflash.discovery.functions_to_optimize import find_all_functions_in_file


@pytest.fixture
def trace_file():
    return Path(__file__).parent.parent / "code_to_optimize/code_directories/simple_tracer_e2e/codeflash.trace"


@pytest.fixture
def workload_functions():
    workloads_file = Path(__file__).parent.parent / "code_to_optimize/code_directories/simple_tracer_e2e/workload.py"
    functions_dict = find_all_functions_in_file(workloads_file)
    all_functions = []
    for functions_list in functions_dict.values():
        all_functions.extend(functions_list)
    return all_functions


@pytest.fixture
def function_ranker(trace_file):
    return FunctionRanker(trace_file)


def test_function_ranker_initialization(trace_file):
    ranker = FunctionRanker(trace_file)
    assert ranker.trace_file_path == trace_file
    assert ranker._profile_stats is not None
    assert isinstance(ranker._function_stats, dict)


def test_load_function_stats(function_ranker):
    assert len(function_ranker._function_stats) > 0

    # Check that funcA is loaded with expected structure
    func_a_key = None
    for key, stats in function_ranker._function_stats.items():
        if stats["function_name"] == "funcA":
            func_a_key = key
            break

    assert func_a_key is not None
    func_a_stats = function_ranker._function_stats[func_a_key]

    # Verify funcA stats structure
    expected_keys = {
        "filename",
        "function_name",
        "qualified_name",
        "class_name",
        "line_number",
        "call_count",
        "own_time_ns",
        "cumulative_time_ns",
        "time_in_callees_ns",
        "addressable_time_ns",
    }
    assert set(func_a_stats.keys()) == expected_keys

    # Verify funcA specific values
    assert func_a_stats["function_name"] == "funcA"
    assert func_a_stats["call_count"] == 1
    assert func_a_stats["own_time_ns"] == 153000
    assert func_a_stats["cumulative_time_ns"] == 1324000


def test_get_function_addressable_time(function_ranker, workload_functions):
    func_a = None
    for func in workload_functions:
        if func.function_name == "funcA":
            func_a = func
            break

    assert func_a is not None
    addressable_time = function_ranker.get_function_addressable_time(func_a)

    # Expected addressable time: own_time + (time_in_callees / call_count)
    # = 153000 + ((1324000 - 153000) / 1) = 1324000
    assert addressable_time == 1324000


def test_rank_functions(function_ranker, workload_functions):
    ranked_functions = function_ranker.rank_functions(workload_functions)

    # Should filter out functions below importance threshold and sort by addressable time
    assert len(ranked_functions) <= len(workload_functions)
    assert len(ranked_functions) > 0  # At least some functions should pass the threshold

    # funcA should pass the importance threshold
    func_a_in_results = any(f.function_name == "funcA" for f in ranked_functions)
    assert func_a_in_results

    # Verify functions are sorted by addressable time in descending order
    for i in range(len(ranked_functions) - 1):
        current_time = function_ranker.get_function_addressable_time(ranked_functions[i])
        next_time = function_ranker.get_function_addressable_time(ranked_functions[i + 1])
        assert current_time >= next_time


def test_get_function_stats_summary(function_ranker, workload_functions):
    func_a = None
    for func in workload_functions:
        if func.function_name == "funcA":
            func_a = func
            break

    assert func_a is not None
    stats = function_ranker.get_function_stats_summary(func_a)

    assert stats is not None
    assert stats["function_name"] == "funcA"
    assert stats["own_time_ns"] == 153000
    assert stats["cumulative_time_ns"] == 1324000
    assert stats["addressable_time_ns"] == 1324000


def test_importance_calculation(function_ranker):
    total_program_time = sum(
        s["own_time_ns"] for s in function_ranker._function_stats.values() if s.get("own_time_ns", 0) > 0
    )

    func_a_stats = None
    for stats in function_ranker._function_stats.values():
        if stats["function_name"] == "funcA":
            func_a_stats = stats
            break

    assert func_a_stats is not None
    importance = func_a_stats["own_time_ns"] / total_program_time

    # funcA importance should be approximately 1.9% (153000/7958000)
    assert abs(importance - 0.019) < 0.01
