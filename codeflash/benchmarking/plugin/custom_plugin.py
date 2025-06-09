from typing import Any, Callable, Optional

import pytest


@pytest.fixture
def benchmark(request):  # noqa: ANN201, ANN001
    class CustomBenchmark:
        def __init__(self) -> None:
            self.stats = []

        def __call__(self, func, *args, **kwargs):  # noqa: ANN204, ANN001, ANN002, ANN003
            # Just call the function without measuring anything
            return func(*args, **kwargs)

        def __getattr__(self, name):  # noqa: ANN204, ANN001
            # Return a no-op callable for any attribute
            return lambda *args, **kwargs: None  # noqa: ARG005

        def pedantic(
            self,
            target: Callable,
            args: tuple = (),
            kwargs: Optional[dict] = None,  # noqa: FA100
            iterations: int = 1,  # noqa: ARG002
            rounds: int = 1,  # noqa: ARG002
            warmup_rounds: int = 0,  # noqa: ARG002
            setup: Optional[Callable] = None,  # noqa: FA100
        ) -> Any:  # noqa: ANN401
            """Mimics the pedantic method of pytest-benchmark."""
            if setup:
                setup()
            if kwargs is None:
                kwargs = {}
            return target(*args, **kwargs)

        @property
        def group(self):  # noqa: ANN202
            """Return a dummy group object."""
            return type("Group", (), {"name": "dummy"})()

        @property
        def name(self) -> str:
            """Return a dummy name."""
            return "dummy_benchmark"

        @property
        def fullname(self) -> str:
            """Return a dummy fullname."""
            return "dummy::benchmark"

        @property
        def params(self):  # noqa: ANN202
            """Return empty params."""
            return {}

        @property
        def extra_info(self):  # noqa: ANN202
            """Return empty extra info."""
            return {}

    # Check if benchmark fixture is already available (pytest-benchmark is active)
    if "benchmark" in request.fixturenames and hasattr(request, "_fixturemanager"):
        try:
            # Try to get the real benchmark fixture
            return request.getfixturevalue("benchmark")
        except (pytest.FixtureLookupError, AttributeError):
            pass
    return CustomBenchmark()
