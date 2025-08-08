from __future__ import annotations

import importlib.util

import pytest

from codeflash.benchmarking.plugin.plugin import codeflash_benchmark_plugin

PYTEST_BENCHMARK_INSTALLED = importlib.util.find_spec("pytest_benchmark") is not None

benchmark_options = [
    ("--benchmark-columns", "store", None, "Benchmark columns"),
    ("--benchmark-group-by", "store", None, "Benchmark group by"),
    ("--benchmark-name", "store", None, "Benchmark name pattern"),
    ("--benchmark-sort", "store", None, "Benchmark sort column"),
    ("--benchmark-json", "store", None, "Benchmark JSON output file"),
    ("--benchmark-save", "store", None, "Benchmark save name"),
    ("--benchmark-warmup", "store", None, "Benchmark warmup"),
    ("--benchmark-warmup-iterations", "store", None, "Benchmark warmup iterations"),
    ("--benchmark-min-time", "store", None, "Benchmark minimum time"),
    ("--benchmark-max-time", "store", None, "Benchmark maximum time"),
    ("--benchmark-min-rounds", "store", None, "Benchmark minimum rounds"),
    ("--benchmark-timer", "store", None, "Benchmark timer"),
    ("--benchmark-calibration-precision", "store", None, "Benchmark calibration precision"),
    ("--benchmark-disable", "store_true", False, "Disable benchmarks"),
    ("--benchmark-skip", "store_true", False, "Skip benchmarks"),
    ("--benchmark-only", "store_true", False, "Only run benchmarks"),
    ("--benchmark-verbose", "store_true", False, "Verbose benchmark output"),
    ("--benchmark-histogram", "store", None, "Benchmark histogram"),
    ("--benchmark-compare", "store", None, "Benchmark compare"),
    ("--benchmark-compare-fail", "store", None, "Benchmark compare fail threshold"),
]


def pytest_configure(config: pytest.Config) -> None:
    """Register the benchmark marker and disable conflicting plugins."""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark that should be run with codeflash tracing")

    if config.getoption("--codeflash-trace"):
        # When --codeflash-trace is used, ignore all benchmark options by resetting them to defaults
        for option, _, default, _ in benchmark_options:
            option_name = option.replace("--", "").replace("-", "_")
            if hasattr(config.option, option_name):
                setattr(config.option, option_name, default)

        if PYTEST_BENCHMARK_INSTALLED:
            config.pluginmanager.set_blocked("pytest_benchmark")
            config.pluginmanager.set_blocked("pytest-benchmark")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--codeflash-trace", action="store_true", default=False, help="Enable CodeFlash tracing for benchmarks"
    )
    # These options are ignored when --codeflash-trace is used
    for option, action, default, help_text in benchmark_options:
        help_suffix = " (ignored when --codeflash-trace is used)"
        parser.addoption(option, action=action, default=default, help=help_text + help_suffix)


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
        from pytest_benchmark.fixture import BenchmarkFixture as BSF  # pyright: ignore[reportMissingImports]  # noqa: I001, N814

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
