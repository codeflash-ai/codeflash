from __future__ import annotations

from typing import Callable, Protocol

import pytest


class GroupProtocol(Protocol):
    """A protocol for objects with a 'name' attribute."""

    name: str


class CodeFlashBenchmarkCustomPlugin:
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
    class CustomBenchmark:
        """A custom benchmark object that mimics pytest-benchmark's interface."""

        def __init__(self) -> None:
            self.stats = {}

        def __call__(self, func: Callable, *args, **kwargs):  # type: ignore  # noqa: ANN002, ANN003, ANN204, PGH003
            """Call the function and return its result without benchmarking."""
            return func(*args, **kwargs)

        def pedantic(  # noqa: ANN201
            self,
            target,  # noqa: ANN001
            args,  # noqa: ANN001
            kwargs,  # noqa: ANN001
            iterations: int = 1,  # noqa: ARG002
            rounds: int = 1,  # noqa: ARG002
            warmup_rounds: int = 0,  # noqa: ARG002
            setup=None,  # noqa: ANN001
        ):
            """Mimics the pedantic method of pytest-benchmark."""
            if kwargs is None:
                kwargs = {}
            if setup:
                setup()
            if kwargs is None:
                kwargs = {}
            return target(*args, **kwargs)

        @property
        def group(self) -> GroupProtocol:
            """Return a custom group object."""
            return type("Group", (), {"name": "custom"})()

        @property
        def name(self) -> str:
            """Return a custom name."""
            return "custom_benchmark"

        @property
        def fullname(self) -> str:
            """Return a custom fullname."""
            return "custom::benchmark"

        @property
        def params(self) -> dict:
            """Return empty params."""
            return {}

        @property
        def extra_info(self) -> dict:
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
        return CodeFlashBenchmarkCustomPlugin.CustomBenchmark(request)


codeflash_benchmark_plugin = CodeFlashBenchmarkCustomPlugin()
