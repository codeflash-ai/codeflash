from typing import Callable, Any

import pytest
from code_to_optimize.bubble_sort import sorter


class DummyBenchmark:
    """A dummy benchmark object that mimics pytest-benchmark's interface."""

    def __init__(self):
        self.stats = {}

    def __call__(self, func: Callable, *args, **kwargs) -> Any:
        """Call the function and return its result without benchmarking."""
        return func(*args, **kwargs)

    def pedantic(self, target: Callable, args: tuple = (), kwargs: dict = None,
                 iterations: int = 1, rounds: int = 1, warmup_rounds: int = 0,
                 setup: Callable = None) -> Any:
        """Mimics the pedantic method of pytest-benchmark."""
        if setup:
            setup()
        if kwargs is None:
            kwargs = {}
        return target(*args, **kwargs)

    @property
    def group(self):
        """Return a dummy group object."""
        return type('Group', (), {'name': 'dummy'})()

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


@pytest.fixture
def benchmark(request):
    """
    Provide a benchmark fixture that works whether pytest-benchmark is installed or not.

    When pytest-benchmark is disabled with '-p no:benchmark', this provides a dummy
    implementation that allows tests to run without modification.
    """
    # Check if benchmark fixture is already available (pytest-benchmark is active)
    if 'benchmark' in request.fixturenames and hasattr(request, '_fixturemanager'):
        try:
            # Try to get the real benchmark fixture
            return request.getfixturevalue('benchmark')
        except (pytest.FixtureLookupError, AttributeError):
            pass

    # Return dummy benchmark if real one is not available
    return DummyBenchmark()

def test_sort(benchmark):
    result = benchmark(sorter, list(reversed(range(500))))
    assert result == list(range(500))

# This should not be picked up as a benchmark test
def test_sort2():
    result = sorter(list(reversed(range(500))))
    assert result == list(range(500))
