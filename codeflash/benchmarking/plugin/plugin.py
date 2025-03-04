import pytest

from codeflash.tracer import Tracer
from pathlib import Path

class CodeFlashPlugin:
    @staticmethod
    def pytest_addoption(parser):
        parser.addoption(
            "--codeflash-trace",
            action="store_true",
            default=False,
            help="Enable CodeFlash tracing"
        )
        parser.addoption(
            "--functions",
            action="store",
            default="",
            help="Comma-separated list of additional functions to trace"
        )
        parser.addoption(
            "--benchmarks-root",
            action="store",
            default=".",
            help="Root directory for benchmarks"
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
                func_name = func.__name__
                test_name = request.node.name
                additional_functions = request.config.getoption("--functions").split(",")
                trace_functions = [f for f in additional_functions if f]
                print("Tracing functions: ", trace_functions)

                # Get benchmarks root directory from command line option
                benchmarks_root = Path(request.config.getoption("--benchmarks-root"))

                # Create .trace directory if it doesn't exist
                trace_dir = benchmarks_root / '.codeflash_trace'
                trace_dir.mkdir(exist_ok=True)

                # Set output path to the .trace directory
                output_path = trace_dir / f"{test_name}.trace"

                tracer = Tracer(
                    output=str(output_path),  # Convert Path to string for Tracer
                    functions=trace_functions,
                    max_function_count=256,
                    benchmark=True
                )

                with tracer:
                    result = func(*args, **kwargs)

                return result

        return Benchmark()
