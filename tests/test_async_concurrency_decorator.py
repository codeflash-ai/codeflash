from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

from codeflash.code_utils.codeflash_wrap_decorator import codeflash_concurrency_async
from codeflash.models.models import ConcurrencyMetrics, TestResults
from codeflash.verification.parse_test_output import parse_concurrency_metrics


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
class TestConcurrencyAsyncDecorator:
    """Integration tests for codeflash_concurrency_async decorator."""

    @pytest.fixture
    def concurrency_env_setup(self, request):
        """Set up environment variables for concurrency testing."""
        original_env = {}
        test_env = {
            "CODEFLASH_LOOP_INDEX": "1",
            "CODEFLASH_TEST_MODULE": __name__,
            "CODEFLASH_TEST_CLASS": "TestConcurrencyAsyncDecorator",
            "CODEFLASH_TEST_FUNCTION": request.node.name,
            "CODEFLASH_CONCURRENCY_FACTOR": "5",  # Use smaller factor for faster tests
        }

        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        yield test_env

        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

    @pytest.mark.asyncio
    async def test_concurrency_decorator_nonblocking_function(self, concurrency_env_setup, capsys):
        """Test that non-blocking async functions show high concurrency ratio."""

        @codeflash_concurrency_async
        async def nonblocking_sleep(duration: float) -> str:
            await asyncio.sleep(duration)
            return "done"

        result = await nonblocking_sleep(0.01)

        assert result == "done"

        captured = capsys.readouterr()
        output = captured.out

        # Verify the output format
        assert "!@######CONC:" in output
        assert "######@!" in output

        # Parse the output manually to verify format
        lines = [line for line in output.strip().split("\n") if "!@######CONC:" in line]
        assert len(lines) == 1

        line = lines[0]
        # Format: !@######CONC:{test_module}:{test_class}:{test_function}:{function_name}:{loop_index}:{seq_time}:{conc_time}:{factor}######@!
        assert "nonblocking_sleep" in line
        assert ":5######@!" in line  # concurrency factor

        # Extract timing values
        parts = line.replace("!@######CONC:", "").replace("######@!", "").split(":")
        # parts should be: [test_module, test_class, test_function, function_name, loop_index, seq_time, conc_time, factor]
        assert len(parts) == 8

        seq_time = int(parts[5])
        conc_time = int(parts[6])
        factor = int(parts[7])

        assert seq_time > 0
        assert conc_time > 0
        assert factor == 5

        # For non-blocking async, concurrent time should be much less than sequential
        # Sequential runs 5 iterations of 10ms = ~50ms
        # Concurrent runs 5 iterations in parallel = ~10ms
        # So ratio should be around 5 (with some overhead tolerance)
        ratio = seq_time / conc_time if conc_time > 0 else 1.0
        assert ratio > 2.0, f"Non-blocking function should have ratio > 2.0, got {ratio}"

    @pytest.mark.asyncio
    async def test_concurrency_decorator_blocking_function(self, concurrency_env_setup, capsys):
        """Test that blocking functions show low concurrency ratio (~1.0)."""

        @codeflash_concurrency_async
        async def blocking_sleep(duration: float) -> str:
            time.sleep(duration)  # Blocking sleep
            return "done"

        result = await blocking_sleep(0.005)  # 5ms blocking

        assert result == "done"

        captured = capsys.readouterr()
        output = captured.out

        assert "!@######CONC:" in output

        lines = [line for line in output.strip().split("\n") if "!@######CONC:" in line]
        assert len(lines) == 1

        line = lines[0]
        parts = line.replace("!@######CONC:", "").replace("######@!", "").split(":")
        assert len(parts) == 8

        seq_time = int(parts[5])
        conc_time = int(parts[6])

        # For blocking code, sequential and concurrent times should be similar
        # Because time.sleep blocks the entire event loop
        ratio = seq_time / conc_time if conc_time > 0 else 1.0
        # Blocking code should have ratio close to 1.0 (within reasonable tolerance)
        assert ratio < 2.0, f"Blocking function should have ratio < 2.0, got {ratio}"

    @pytest.mark.asyncio
    async def test_concurrency_decorator_with_computation(self, concurrency_env_setup, capsys):
        """Test concurrency with CPU-bound computation."""

        @codeflash_concurrency_async
        async def compute_intensive(n: int) -> int:
            # CPU-bound work (blocked by GIL in concurrent execution)
            total = 0
            for i in range(n):
                total += i * i
            return total

        result = await compute_intensive(10000)

        assert result == sum(i * i for i in range(10000))

        captured = capsys.readouterr()
        output = captured.out

        assert "!@######CONC:" in output
        assert "compute_intensive" in output


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
class TestParseConcurrencyMetrics:
    """Integration tests for parse_concurrency_metrics function."""

    def test_parse_concurrency_metrics_from_real_output(self):
        """Test parsing concurrency metrics from simulated stdout."""
        # Simulate stdout from codeflash_concurrency_async decorator
        perf_stdout = """Some other output
!@######CONC:test_module:TestClass:test_func:my_async_func:1:50000000:10000000:5######@!
More output here
"""
        test_results = TestResults(test_results=[], perf_stdout=perf_stdout)

        metrics = parse_concurrency_metrics(test_results, "my_async_func")

        assert metrics is not None
        assert isinstance(metrics, ConcurrencyMetrics)
        assert metrics.sequential_time_ns == 50000000
        assert metrics.concurrent_time_ns == 10000000
        assert metrics.concurrency_factor == 5
        assert metrics.concurrency_ratio == 5.0  # 50M / 10M = 5.0

    def test_parse_concurrency_metrics_multiple_entries(self):
        """Test parsing when multiple concurrency entries exist."""
        perf_stdout = """!@######CONC:test_module:TestClass:test_func:target_func:1:40000000:10000000:5######@!
!@######CONC:test_module:TestClass:test_func:target_func:2:60000000:10000000:5######@!
!@######CONC:test_module:TestClass:test_func:other_func:1:30000000:15000000:5######@!
"""
        test_results = TestResults(test_results=[], perf_stdout=perf_stdout)

        metrics = parse_concurrency_metrics(test_results, "target_func")

        assert metrics is not None
        # Should average the two entries for target_func
        # (40M + 60M) / 2 = 50M seq, (10M + 10M) / 2 = 10M conc
        assert metrics.sequential_time_ns == 50000000
        assert metrics.concurrent_time_ns == 10000000
        assert metrics.concurrency_ratio == 5.0

    def test_parse_concurrency_metrics_no_match(self):
        """Test parsing when function name doesn't match."""
        perf_stdout = """!@######CONC:test_module:TestClass:test_func:other_func:1:50000000:10000000:5######@!
"""
        test_results = TestResults(test_results=[], perf_stdout=perf_stdout)

        metrics = parse_concurrency_metrics(test_results, "nonexistent_func")

        assert metrics is None

    def test_parse_concurrency_metrics_empty_stdout(self):
        """Test parsing with empty stdout."""
        test_results = TestResults(test_results=[], perf_stdout="")

        metrics = parse_concurrency_metrics(test_results, "any_func")

        assert metrics is None

    def test_parse_concurrency_metrics_none_stdout(self):
        """Test parsing with None stdout."""
        test_results = TestResults(test_results=[], perf_stdout=None)

        metrics = parse_concurrency_metrics(test_results, "any_func")

        assert metrics is None


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
class TestConcurrencyRatioComparison:
    """Test comparing blocking vs non-blocking concurrency ratios."""

    @pytest.fixture
    def comparison_env_setup(self, request):
        """Set up environment variables for comparison testing."""
        original_env = {}
        test_env = {
            "CODEFLASH_LOOP_INDEX": "1",
            "CODEFLASH_TEST_MODULE": __name__,
            "CODEFLASH_TEST_CLASS": "TestConcurrencyRatioComparison",
            "CODEFLASH_TEST_FUNCTION": request.node.name,
            "CODEFLASH_CONCURRENCY_FACTOR": "10",
        }

        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        yield test_env

        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

    @pytest.mark.asyncio
    async def test_blocking_vs_nonblocking_comparison(self, comparison_env_setup, capsys):
        """Compare concurrency ratios between blocking and non-blocking implementations."""

        @codeflash_concurrency_async
        async def blocking_impl() -> str:
            time.sleep(0.002)  # 2ms blocking
            return "blocking"

        @codeflash_concurrency_async
        async def nonblocking_impl() -> str:
            await asyncio.sleep(0.002)  # 2ms non-blocking
            return "nonblocking"

        # Run blocking version
        await blocking_impl()
        blocking_output = capsys.readouterr().out

        # Run non-blocking version
        await nonblocking_impl()
        nonblocking_output = capsys.readouterr().out

        # Parse blocking metrics
        blocking_line = [l for l in blocking_output.split("\n") if "!@######CONC:" in l][0]
        blocking_parts = blocking_line.replace("!@######CONC:", "").replace("######@!", "").split(":")
        blocking_seq = int(blocking_parts[5])
        blocking_conc = int(blocking_parts[6])
        blocking_ratio = blocking_seq / blocking_conc if blocking_conc > 0 else 1.0

        # Parse non-blocking metrics
        nonblocking_line = [l for l in nonblocking_output.split("\n") if "!@######CONC:" in l][0]
        nonblocking_parts = nonblocking_line.replace("!@######CONC:", "").replace("######@!", "").split(":")
        nonblocking_seq = int(nonblocking_parts[5])
        nonblocking_conc = int(nonblocking_parts[6])
        nonblocking_ratio = nonblocking_seq / nonblocking_conc if nonblocking_conc > 0 else 1.0

        # Non-blocking should have significantly higher concurrency ratio
        assert nonblocking_ratio > blocking_ratio, (
            f"Non-blocking ratio ({nonblocking_ratio:.2f}) should be greater than blocking ratio ({blocking_ratio:.2f})"
        )

        # The difference should be substantial (non-blocking should be at least 2x better)
        ratio_improvement = nonblocking_ratio / blocking_ratio if blocking_ratio > 0 else 0
        assert ratio_improvement > 2.0, (
            f"Non-blocking should show >2x improvement in concurrency ratio, got {ratio_improvement:.2f}x"
        )
