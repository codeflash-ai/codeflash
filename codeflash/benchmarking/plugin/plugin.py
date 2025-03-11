import pytest
import time

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
                start = time.time_ns()
                result = func(*args, **kwargs)
                end = time.time_ns()
                print(f"Benchmark: {func.__name__} took {end - start} ns")
                return result

        return Benchmark()
