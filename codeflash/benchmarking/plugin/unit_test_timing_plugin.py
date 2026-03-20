from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class UnitTestTimingPlugin:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.timings: dict[str, int] = {}
        self._start_time: int = 0

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: pytest.Item) -> None:
        self._start_time = time.perf_counter_ns()
        yield
        duration_ns = time.perf_counter_ns() - self._start_time
        self.timings[item.nodeid] = duration_ns

    @pytest.hookimpl
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self.output_path.write_text(json.dumps(self.timings), encoding="utf-8")
