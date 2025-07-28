from __future__ import annotations

import importlib.util

import pytest

from codeflash.benchmarking.plugin.plugin import codeflash_benchmark_plugin

PYTEST_BENCHMARK_INSTALLED = importlib.util.find_spec("pytest_benchmark") is not None


def pytest_configure(config: pytest.Config) -> None:
    """Register the benchmark marker and disable conflicting plugins."""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark that should be run with codeflash tracing")

    if config.getoption("--codeflash-trace") and PYTEST_BENCHMARK_INSTALLED:
        config.option.benchmark_disable = True
        config.pluginmanager.set_blocked("pytest_benchmark")
        config.pluginmanager.set_blocked("pytest-benchmark")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--codeflash-trace", action="store_true", default=False, help="Enable CodeFlash tracing for benchmarks"
    )



@pytest.fixture
def benchmark(request: pytest.FixtureRequest) -> object:
    """Benchmark fixture that works with or without pytest-benchmark installed."""
    config = request.config

    # If --codeflash-trace is enabled, use our implementation
    if config.getoption("--codeflash-trace"):
        return codeflash_benchmark_plugin.Benchmark(request)

    # If pytest-benchmark is installed and --codeflash-trace is not enabled,
    # return the normal pytest-benchmark fixture
    if PYTEST_BENCHMARK_INSTALLED:
        from pytest_benchmark.fixture import BenchmarkFixture as BSF  # noqa: N814

        bs = getattr(config, "_benchmarksession", None)
        if bs and bs.skip:
            pytest.skip("Benchmarks are skipped (--benchmark-skip was used).")

        node = request.node
        marker = node.get_closest_marker("benchmark")
        options = dict(marker.kwargs) if marker else {}

        if bs:
            return BSF(
                node,
                add_stats=bs.benchmarks.append,
                logger=bs.logger,
                warner=request.node.warn,
                disabled=bs.disabled,
                **dict(bs.options, **options),
            )
        return lambda func, *args, **kwargs: func(*args, **kwargs)

    return lambda func, *args, **kwargs: func(*args, **kwargs)
