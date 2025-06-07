from __future__ import annotations

from typing import Any, Callable

import pytest


class CodeFlashBenchmarkDummyPlugin:
    @staticmethod
    def pytest_plugin_registered(plugin, manager) -> None:  # noqa: ANN001
        # Not necessary since run with -p no:benchmark, but just in case
        if hasattr(plugin, "name") and plugin.name == "pytest-benchmark":
            manager.unregister(plugin)

    @staticmethod
    def pytest_configure(config: pytest.Config) -> None:
        """Register the benchmark marker."""
        config.addinivalue_line(
            "markers",
            "benchmark: mark test as a benchmark that should be run without modification if the benchmark fixture is disabled",
        )

    # Benchmark fixture
    class DummyBenchmark:
        """A dummy benchmark object that mimics pytest-benchmark's interface."""

        def __init__(self) -> None:
            self.stats = {}

        def __call__(self, func: Callable, *args: tuple[Any, ...], **kwargs: dict) -> Any:
            """Call the function and return its result without benchmarking."""
            return func(*args, **kwargs)

        def pedantic(
            self,
            target: Callable,
            args: tuple = (),
            kwargs: dict = None,
            iterations: int = 1,
            rounds: int = 1,
            warmup_rounds: int = 0,
            setup: Callable = None,
        ) -> Any:
            """Mimics the pedantic method of pytest-benchmark."""
            if setup:
                setup()
            if kwargs is None:
                kwargs = {}
            return target(*args, **kwargs)

        @property
        def group(self):
            """Return a dummy group object."""
            return type("Group", (), {"name": "dummy"})()

        @property
        def name(self):
            """Return a dummy name."""
            return "dummy_benchmark"

        @property
        def fullname(self):
            """Return a dummy fullname."""
            return "dummy::benchmark"

        @property
        def params(self):
            """Return empty params."""
            return {}

        @property
        def extra_info(self):
            """Return empty extra info."""
            return {}

    @staticmethod
    @pytest.fixture
    def benchmark(request: pytest.FixtureRequest) -> object:
        # Check if benchmark fixture is already available (pytest-benchmark is active)
        if "benchmark" in request.fixturenames and hasattr(request, "_fixturemanager"):
            try:
                return request.getfixturevalue("benchmark")
            except (pytest.FixtureLookupError, AttributeError):
                pass
        return CodeFlashBenchmarkDummyPlugin.DummyBenchmark(request)


codeflash_benchmark_plugin = CodeFlashBenchmarkDummyPlugin()
