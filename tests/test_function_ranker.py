import pytest
from pathlib import Path
from unittest.mock import patch

from codeflash.benchmarking.function_ranker import FunctionRanker
from codeflash.discovery.functions_to_optimize import FunctionToOptimize, find_all_functions_in_file
from codeflash.models.models import FunctionParent


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
        "filename", "function_name", "qualified_name", "class_name", 
        "line_number", "call_count", "own_time_ns", "cumulative_time_ns", 
        "time_in_callees_ns", "ttx_score"
    }
    assert set(func_a_stats.keys()) == expected_keys
    
    # Verify funcA specific values
    assert func_a_stats["function_name"] == "funcA"
    assert func_a_stats["call_count"] == 1
    assert func_a_stats["own_time_ns"] == 63000
    assert func_a_stats["cumulative_time_ns"] == 5443000


def test_get_function_ttx_score(function_ranker, workload_functions):
    func_a = None
    for func in workload_functions:
        if func.function_name == "funcA":
            func_a = func
            break
    
    assert func_a is not None
    ttx_score = function_ranker.get_function_ttx_score(func_a)
    
    # Expected ttX score: own_time + (time_in_callees / call_count)
    # = 63000 + ((5443000 - 63000) / 1) = 5443000
    assert ttx_score == 5443000


def test_rank_functions(function_ranker, workload_functions):
    ranked_functions = function_ranker.rank_functions(workload_functions)
    
    assert len(ranked_functions) == len(workload_functions)
    
    # Verify functions are sorted by ttX score in descending order
    for i in range(len(ranked_functions) - 1):
        current_score = function_ranker.get_function_ttx_score(ranked_functions[i])
        next_score = function_ranker.get_function_ttx_score(ranked_functions[i + 1])
        assert current_score >= next_score


def test_rerank_and_filter_functions(function_ranker, workload_functions):
    filtered_ranked = function_ranker.rerank_and_filter_functions(workload_functions)
    
    # Should filter out functions below importance threshold
    assert len(filtered_ranked) <= len(workload_functions)
    
    # funcA should pass the importance threshold (0.33% > 0.1%)
    func_a_in_results = any(f.function_name == "funcA" for f in filtered_ranked)
    assert func_a_in_results


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
    assert stats["own_time_ns"] == 63000
    assert stats["cumulative_time_ns"] == 5443000
    assert stats["ttx_score"] == 5443000




def test_importance_calculation(function_ranker):
    total_program_time = sum(
        s["own_time_ns"] for s in function_ranker._function_stats.values() 
        if s.get("own_time_ns", 0) > 0
    )
    
    func_a_stats = None
    for stats in function_ranker._function_stats.values():
        if stats["function_name"] == "funcA":
            func_a_stats = stats
            break
    
    assert func_a_stats is not None
    importance = func_a_stats["own_time_ns"] / total_program_time
    
    # funcA importance should be approximately 0.57% (63000/10968000)
    assert abs(importance - 0.0057) < 0.001


def test_simple_model_predict_stats(function_ranker, workload_functions):
    # Find SimpleModel::predict function
    predict_func = None
    for func in workload_functions:
        if func.function_name == "predict":
            predict_func = func
            break
    
    assert predict_func is not None
    
    stats = function_ranker.get_function_stats_summary(predict_func)
    assert stats is not None
    assert stats["function_name"] == "predict"
    assert stats["call_count"] == 1
    assert stats["own_time_ns"] == 2289000
    assert stats["cumulative_time_ns"] == 4017000
    assert stats["ttx_score"] == 4017000
    
    # Test ttX score calculation
    ttx_score = function_ranker.get_function_ttx_score(predict_func)
    # Expected ttX score: own_time + (time_in_callees / call_count)
    # = 2289000 + ((4017000 - 2289000) / 1) = 4017000
    assert ttx_score == 4017000
    
    # Test importance calculation for predict function
    total_program_time = sum(
        s["own_time_ns"] for s in function_ranker._function_stats.values() 
        if s.get("own_time_ns", 0) > 0
    )
    importance = stats["own_time_ns"] / total_program_time
    # predict importance should be approximately 20.9% (2289000/10968000)
    assert abs(importance - 0.209) < 0.01
