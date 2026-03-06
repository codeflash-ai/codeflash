from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from codeflash.code_utils.config_consts import BEHAVIORAL_SLOWDOWN_SKIP_THRESHOLD
from codeflash.either import Failure, is_successful
from codeflash.models.models import FunctionTestInvocation, InvocationId, OriginalCodeBaseline, TestResults
from codeflash.models.test_type import TestType

MOCK_FILE_PATH = Path("tests/test_example.py")


def _make_test_results(runtime_ns: int) -> TestResults:
    results = TestResults()
    invocation = FunctionTestInvocation(
        loop_index=1,
        id=InvocationId(
            test_module_path="tests.test_example",
            test_class_name=None,
            test_function_name="test_func",
            function_getting_tested="func",
            iteration_id="0",
        ),
        file_name=MOCK_FILE_PATH,
        did_pass=True,
        runtime=runtime_ns,
        test_framework="pytest",
        test_type=TestType.EXISTING_UNIT_TEST,
        return_value=None,
        timed_out=False,
    )
    results.add(invocation)
    return results


def _make_baseline(behavior_runtime_ns: int) -> OriginalCodeBaseline:
    return OriginalCodeBaseline(
        behavior_test_results=_make_test_results(behavior_runtime_ns),
        benchmarking_test_results=TestResults(),
        line_profile_results={},
        runtime=behavior_runtime_ns,
        coverage_results=None,
    )


def _make_optimizer_mock(mock_run_and_parse: MagicMock, mock_compare_results: MagicMock, **kwargs: object) -> MagicMock:
    optimizer = MagicMock()
    optimizer.function_to_optimize.file_path = MOCK_FILE_PATH
    optimizer.function_to_optimize.is_async = False
    optimizer.test_files = []
    optimizer.run_and_parse_tests = mock_run_and_parse
    optimizer.compare_candidate_results = mock_compare_results
    for key, value in kwargs.items():
        setattr(optimizer, key, value) if "." not in key else None
    return optimizer


def _run_candidate(optimizer: MagicMock, baseline: OriginalCodeBaseline) -> Failure | object:
    from codeflash.optimization.function_optimizer import FunctionOptimizer

    with patch.object(Path, "read_text", return_value="def func(): pass"):
        return FunctionOptimizer.run_optimized_candidate(
            optimizer,
            optimization_candidate_index=1,
            baseline_results=baseline,
            original_helper_code={},
            file_path_to_helper_classes={},
            eval_ctx=MagicMock(),
            code_context=MagicMock(),
            candidate=MagicMock(),
            exp_type="test",
        )


class TestBehavioralTimingGate:
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.run_and_parse_tests")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.compare_candidate_results")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.write_code_and_helpers")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.instrument_capture")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.get_test_env")
    @patch("codeflash.optimization.function_optimizer.get_run_tmp_file")
    def test_slow_candidate_skips_benchmarking(
        self,
        mock_get_run_tmp_file: MagicMock,
        mock_get_test_env: MagicMock,
        mock_instrument_capture: MagicMock,
        mock_write_code_and_helpers: MagicMock,
        mock_compare_results: MagicMock,
        mock_run_and_parse: MagicMock,
    ) -> None:
        """A candidate 15x slower than baseline should be rejected without benchmarking."""
        baseline = _make_baseline(1000)
        candidate_results = _make_test_results(15000)  # 15x slower

        mock_get_run_tmp_file.return_value = MagicMock()
        mock_get_test_env.return_value = {}
        mock_run_and_parse.return_value = (candidate_results, None)
        mock_compare_results.return_value = (True, [])

        optimizer = _make_optimizer_mock(mock_run_and_parse, mock_compare_results)
        optimizer.write_code_and_helpers = mock_write_code_and_helpers
        optimizer.instrument_capture = mock_instrument_capture
        optimizer.get_test_env = mock_get_test_env

        result = _run_candidate(optimizer, baseline)

        assert not is_successful(result)
        assert isinstance(result, Failure)
        assert f"{BEHAVIORAL_SLOWDOWN_SKIP_THRESHOLD:.0f}x" in str(result.value)
        assert mock_run_and_parse.call_count == 1

    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.run_and_parse_tests")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.compare_candidate_results")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.write_code_and_helpers")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.instrument_capture")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.get_test_env")
    @patch("codeflash.optimization.function_optimizer.get_run_tmp_file")
    def test_acceptable_candidate_proceeds_to_benchmarking(
        self,
        mock_get_run_tmp_file: MagicMock,
        mock_get_test_env: MagicMock,
        mock_instrument_capture: MagicMock,
        mock_write_code_and_helpers: MagicMock,
        mock_compare_results: MagicMock,
        mock_run_and_parse: MagicMock,
    ) -> None:
        """A candidate 5x slower than baseline (under threshold) should proceed to benchmarking."""
        baseline = _make_baseline(1000)
        candidate_behavior_results = _make_test_results(5000)  # 5x slower, under 10x threshold
        candidate_benchmark_results = _make_test_results(5000)

        mock_get_run_tmp_file.return_value = MagicMock()
        mock_get_test_env.return_value = {}
        mock_run_and_parse.side_effect = [(candidate_behavior_results, None), (candidate_benchmark_results, None)]
        mock_compare_results.return_value = (True, [])

        optimizer = _make_optimizer_mock(mock_run_and_parse, mock_compare_results)
        optimizer.write_code_and_helpers = mock_write_code_and_helpers
        optimizer.instrument_capture = mock_instrument_capture
        optimizer.get_test_env = mock_get_test_env
        optimizer.args.benchmark = False
        optimizer.collect_async_metrics.return_value = (None, None)

        result = _run_candidate(optimizer, baseline)

        assert mock_run_and_parse.call_count == 2

    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.run_and_parse_tests")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.compare_candidate_results")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.write_code_and_helpers")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.instrument_capture")
    @patch("codeflash.optimization.function_optimizer.FunctionOptimizer.get_test_env")
    @patch("codeflash.optimization.function_optimizer.get_run_tmp_file")
    def test_zero_baseline_runtime_skips_check(
        self,
        mock_get_run_tmp_file: MagicMock,
        mock_get_test_env: MagicMock,
        mock_instrument_capture: MagicMock,
        mock_write_code_and_helpers: MagicMock,
        mock_compare_results: MagicMock,
        mock_run_and_parse: MagicMock,
    ) -> None:
        """When baseline behavioral runtime is 0, the timing gate should be skipped."""
        baseline = OriginalCodeBaseline(
            behavior_test_results=TestResults(),  # Empty = 0 runtime
            benchmarking_test_results=TestResults(),
            line_profile_results={},
            runtime=0,
            coverage_results=None,
        )
        candidate_behavior_results = _make_test_results(50000)
        candidate_benchmark_results = _make_test_results(50000)

        mock_get_run_tmp_file.return_value = MagicMock()
        mock_get_test_env.return_value = {}
        mock_run_and_parse.side_effect = [(candidate_behavior_results, None), (candidate_benchmark_results, None)]
        mock_compare_results.return_value = (True, [])

        optimizer = _make_optimizer_mock(mock_run_and_parse, mock_compare_results)
        optimizer.write_code_and_helpers = mock_write_code_and_helpers
        optimizer.instrument_capture = mock_instrument_capture
        optimizer.get_test_env = mock_get_test_env
        optimizer.args.benchmark = False
        optimizer.collect_async_metrics.return_value = (None, None)

        result = _run_candidate(optimizer, baseline)

        assert mock_run_and_parse.call_count == 2
