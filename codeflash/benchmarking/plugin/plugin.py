import sys

import pytest
import time
import os
class CodeFlashBenchmarkPlugin:
    benchmark_timings = []

    class Benchmark:
        def __init__(self, request):
            self.request = request

        def __call__(self, func, *args, **kwargs):
            benchmark_file_name = self.request.node.fspath.basename
            benchmark_function_name = self.request.node.name
            line_number = str(sys._getframe(1).f_lineno)  # 1 frame up in the call stack

            os.environ["CODEFLASH_BENCHMARK_FUNCTION_NAME"] = benchmark_function_name
            os.environ["CODEFLASH_BENCHMARK_FILE_NAME"] = benchmark_file_name
            os.environ["CODEFLASH_BENCHMARK_LINE_NUMBER"] = line_number
            os.environ["CODEFLASH_BENCHMARKING"] = "True"

            start = time.perf_counter_ns()
            result = func(*args, **kwargs)
            end = time.perf_counter_ns()

            os.environ["CODEFLASH_BENCHMARKING"] = "False"
            CodeFlashBenchmarkPlugin.benchmark_timings.append(
                (benchmark_file_name, benchmark_function_name, line_number, end - start))
            return result
    @staticmethod
    def pytest_addoption(parser):
        parser.addoption(
            "--codeflash-trace",
            action="store_true",
            default=False,
            help="Enable CodeFlash tracing"
        )

    @staticmethod
    def pytest_plugin_registered(plugin, manager):
        if hasattr(plugin, "name") and plugin.name == "pytest-benchmark":
            manager.unregister(plugin)

    @staticmethod
    def pytest_collection_modifyitems(config, items):
        if not config.getoption("--codeflash-trace"):
            return

        skip_no_benchmark = pytest.mark.skip(reason="Test requires benchmark fixture")
        for item in items:
            if hasattr(item, "fixturenames") and "benchmark" in item.fixturenames:
                continue
            item.add_marker(skip_no_benchmark)

    @staticmethod
    @pytest.fixture
    def benchmark(request):
        if not request.config.getoption("--codeflash-trace"):
            return None

        return CodeFlashBenchmarkPlugin.Benchmark(request)