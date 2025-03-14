import sys

import pytest
import time
import os
class CodeFlashPlugin:
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

        class Benchmark:

            def __call__(self, func, *args, **kwargs):
                os.environ["CODEFLASH_BENCHMARK_FUNCTION_NAME"] = request.node.name
                os.environ["CODEFLASH_BENCHMARK_FILE_NAME"] = request.node.fspath.basename
                os.environ["CODEFLASH_BENCHMARK_LINE_NUMBER"] = str(sys._getframe(1).f_lineno) # 1 frame up in the call stack
                os.environ["CODEFLASH_BENCHMARKING"] = "True"
                start = time.process_time_ns()
                result = func(*args, **kwargs)
                end = time.process_time_ns()
                os.environ["CODEFLASH_BENCHMARKING"] = "False"
                print(f"Benchmark: {func.__name__} took {end - start} ns")
                return result

        return Benchmark()
