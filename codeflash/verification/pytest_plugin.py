from __future__ import annotations

import contextlib
import inspect

# System Imports
import logging
import os
import platform
import re
import sys
import time as _time_module
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from unittest import TestCase

# PyTest Imports
import pytest
from pluggy import HookspecMarker

if TYPE_CHECKING:
    from _pytest.config import Config, Parser
    from _pytest.main import Session
    from _pytest.python import Metafunc

SECONDS_IN_HOUR: float = 3600
SECONDS_IN_MINUTE: float = 60
SHORTEST_AMOUNT_OF_TIME: float = 0
hookspec = HookspecMarker("pytest")


class InvalidTimeParameterError(Exception):
    pass


class UnexpectedError(Exception):
    pass


if platform.system() == "Linux":
    import resource

    # We set the memory limit to 85% of total system memory + swap when swap exists
    swap_file_path = Path("/proc/swaps")
    swap_exists = swap_file_path.is_file()
    swap_size = 0

    if swap_exists:
        with swap_file_path.open("r") as f:
            swap_lines = f.readlines()
            swap_exists = len(swap_lines) > 1  # First line is header

            if swap_exists:
                # Parse swap size from lines after header
                for line in swap_lines[1:]:
                    parts = line.split()
                    if len(parts) >= 3:
                        # Swap size is in KB in the 3rd column
                        with contextlib.suppress(ValueError, IndexError):
                            swap_size += int(parts[2]) * 1024  # Convert KB to bytes

    # Get total system memory
    total_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    # Add swap to total available memory if swap exists
    if swap_exists:
        total_memory += swap_size

    # Set the memory limit to 85% of total memory (RAM plus swap)
    memory_limit = int(total_memory * 0.85)

    # Set both soft and hard limits
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


# Store references to original functions before any patching
_ORIGINAL_TIME_TIME = _time_module.time
_ORIGINAL_PERF_COUNTER = _time_module.perf_counter
_ORIGINAL_TIME_SLEEP = _time_module.sleep


# Apply deterministic patches for reproducible test execution
def _apply_deterministic_patches() -> None:
    """Apply patches to make all sources of randomness deterministic."""
    import datetime
    import random
    import time
    import uuid

    # Store original functions (these are already saved globally above)
    _original_time = time.time
    _original_perf_counter = time.perf_counter
    _original_datetime_now = datetime.datetime.now
    _original_datetime_utcnow = datetime.datetime.utcnow
    _original_uuid4 = uuid.uuid4
    _original_uuid1 = uuid.uuid1
    _original_random = random.random

    # Fixed deterministic values
    fixed_timestamp = 1609459200.0  # 2021-01-01 00:00:00 UTC
    fixed_datetime = datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-123456789012")

    # Counter for perf_counter to maintain relative timing
    _perf_counter_start = fixed_timestamp
    _perf_counter_calls = 0

    def mock_time_time() -> float:
        """Return fixed timestamp while preserving performance characteristics."""
        _original_time()  # Maintain performance characteristics
        return fixed_timestamp

    def mock_perf_counter() -> float:
        """Return incrementing counter for relative timing."""
        nonlocal _perf_counter_calls
        _original_perf_counter()  # Maintain performance characteristics
        _perf_counter_calls += 1
        return _perf_counter_start + (_perf_counter_calls * 0.001)  # Increment by 1ms each call

    def mock_datetime_now(tz: datetime.timezone | None = None) -> datetime.datetime:
        """Return fixed datetime while preserving performance characteristics."""
        _original_datetime_now(tz)  # Maintain performance characteristics
        if tz is None:
            return fixed_datetime
        return fixed_datetime.replace(tzinfo=tz)

    def mock_datetime_utcnow() -> datetime.datetime:
        """Return fixed UTC datetime while preserving performance characteristics."""
        _original_datetime_utcnow()  # Maintain performance characteristics
        return fixed_datetime

    def mock_uuid4() -> uuid.UUID:
        """Return fixed UUID4 while preserving performance characteristics."""
        _original_uuid4()  # Maintain performance characteristics
        return fixed_uuid

    def mock_uuid1(node: int | None = None, clock_seq: int | None = None) -> uuid.UUID:
        """Return fixed UUID1 while preserving performance characteristics."""
        _original_uuid1(node, clock_seq)  # Maintain performance characteristics
        return fixed_uuid

    def mock_random() -> float:
        """Return deterministic random value while preserving performance characteristics."""
        _original_random()  # Maintain performance characteristics
        return 0.123456789  # Fixed random value

    # Apply patches
    time.time = mock_time_time
    time.perf_counter = mock_perf_counter
    uuid.uuid4 = mock_uuid4
    uuid.uuid1 = mock_uuid1

    # Seed random module for other random functions
    random.seed(42)
    random.random = mock_random

    # For datetime, we need to use a different approach since we can't patch class methods
    # Store original methods for potential later use
    import builtins

    builtins._original_datetime_now = _original_datetime_now  # noqa: SLF001
    builtins._original_datetime_utcnow = _original_datetime_utcnow  # noqa: SLF001
    builtins._mock_datetime_now = mock_datetime_now  # noqa: SLF001
    builtins._mock_datetime_utcnow = mock_datetime_utcnow  # noqa: SLF001

    # Patch numpy.random if available
    try:
        import numpy as np

        # Use modern numpy random generator approach
        np.random.default_rng(42)
        np.random.seed(42)  # Keep legacy seed for compatibility  # noqa: NPY002
    except ImportError:
        pass

    # Patch os.urandom if needed
    try:
        import os

        _original_urandom = os.urandom

        def mock_urandom(n: int) -> bytes:
            _original_urandom(n)  # Maintain performance characteristics
            return b"\x42" * n  # Fixed bytes

        os.urandom = mock_urandom
    except (ImportError, AttributeError):
        pass


# Note: Deterministic patches are applied conditionally, not globally
# They should only be applied when running CodeFlash optimization tests


def pytest_addoption(parser: Parser) -> None:
    """Add command line options."""
    pytest_loops = parser.getgroup("loops")
    pytest_loops.addoption(
        "--codeflash_delay",
        action="store",
        default=0,
        type=float,
        help="The amount of time to wait between each test loop.",
    )
    pytest_loops.addoption(
        "--codeflash_hours", action="store", default=0, type=float, help="The number of hours to loop the tests for."
    )
    pytest_loops.addoption(
        "--codeflash_minutes",
        action="store",
        default=0,
        type=float,
        help="The number of minutes to loop the tests for.",
    )
    pytest_loops.addoption(
        "--codeflash_seconds",
        action="store",
        default=0,
        type=float,
        help="The number of seconds to loop the tests for.",
    )

    pytest_loops.addoption(
        "--codeflash_loops", action="store", default=1, type=int, help="The number of times to loop each test"
    )

    pytest_loops.addoption(
        "--codeflash_min_loops",
        action="store",
        default=1,
        type=int,
        help="The minimum number of times to loop each test",
    )

    pytest_loops.addoption(
        "--codeflash_max_loops",
        action="store",
        default=100_000,
        type=int,
        help="The maximum number of times to loop each test",
    )

    pytest_loops.addoption(
        "--codeflash_loops_scope",
        action="store",
        default="function",
        type=str,
        choices=("function", "class", "module", "session"),
        help="Scope for looping tests",
    )


@pytest.hookimpl(trylast=True)
def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "loops(n): run the given test function `n` times.")
    config.pluginmanager.register(PytestLoops(config), PytestLoops.name)

    # Apply deterministic patches when the plugin is configured
    _apply_deterministic_patches()


class PytestLoops:
    name: str = "pytest-loops"

    def __init__(self, config: Config) -> None:
        # Turn debug prints on only if "-vv" or more passed
        level = logging.DEBUG if config.option.verbose > 1 else logging.INFO
        logging.basicConfig(level=level)
        self.logger = logging.getLogger(self.name)

    @hookspec(firstresult=True)
    def pytest_runtestloop(self, session: Session) -> bool:
        """Reimplement the test loop but loop for the user defined amount of time."""
        if session.testsfailed and not session.config.option.continue_on_collection_errors:
            msg = "{} error{} during collection".format(session.testsfailed, "s" if session.testsfailed != 1 else "")
            raise session.Interrupted(msg)

        if session.config.option.collectonly:
            return True

        start_time: float = _ORIGINAL_TIME_TIME()
        total_time: float = self._get_total_time(session)

        count: int = 0

        while total_time >= SHORTEST_AMOUNT_OF_TIME:  # need to run at least one for normal tests
            count += 1
            total_time = self._get_total_time(session)

            for index, item in enumerate(session.items):
                item: pytest.Item = item  # noqa: PLW0127, PLW2901
                item._report_sections.clear()  # clear reports for new test  # noqa: SLF001

                if total_time > SHORTEST_AMOUNT_OF_TIME:
                    item._nodeid = self._set_nodeid(item._nodeid, count)  # noqa: SLF001

                next_item: pytest.Item = session.items[index + 1] if index + 1 < len(session.items) else None

                self._clear_lru_caches(item)

                item.config.hook.pytest_runtest_protocol(item=item, nextitem=next_item)
                if session.shouldfail:
                    raise session.Failed(session.shouldfail)
                if session.shouldstop:
                    raise session.Interrupted(session.shouldstop)
            if self._timed_out(session, start_time, count):
                break  # exit loop
            _ORIGINAL_TIME_SLEEP(self._get_delay_time(session))
        return True

    def _clear_lru_caches(self, item: pytest.Item) -> None:
        processed_functions: set[Callable] = set()
        protected_modules = {
            "gc",
            "inspect",
            "os",
            "sys",
            "time",
            "functools",
            "pathlib",
            "typing",
            "dill",
            "pytest",
            "importlib",
        }

        def _clear_cache_for_object(obj: Any) -> None:  # noqa: ANN401
            if obj in processed_functions:
                return
            processed_functions.add(obj)

            if hasattr(obj, "__wrapped__"):
                module_name = obj.__wrapped__.__module__
            else:
                try:
                    obj_module = inspect.getmodule(obj)
                    module_name = obj_module.__name__.split(".")[0] if obj_module is not None else None
                except Exception:
                    module_name = None

            if module_name in protected_modules:
                return

            if hasattr(obj, "cache_clear") and callable(obj.cache_clear):
                with contextlib.suppress(Exception):
                    obj.cache_clear()

        _clear_cache_for_object(item.function)  # type: ignore[attr-defined]

        try:
            if hasattr(item.function, "__module__"):  # type: ignore[attr-defined]
                module_name = item.function.__module__  # type: ignore[attr-defined]
                try:
                    module = sys.modules.get(module_name)
                    if module:
                        for _, obj in inspect.getmembers(module):
                            if callable(obj):
                                _clear_cache_for_object(obj)
                except Exception:  # noqa: S110
                    pass
        except Exception:  # noqa: S110
            pass

    def _set_nodeid(self, nodeid: str, count: int) -> str:
        """Set loop count when using duration.

        :param nodeid: Name of test function.
        :param count: Current loop count.
        :return: Formatted string for test name.
        """
        pattern = r"\[ \d+ \]"
        run_str = f"[ {count} ]"
        os.environ["CODEFLASH_LOOP_INDEX"] = str(count)
        return re.sub(pattern, run_str, nodeid) if re.search(pattern, nodeid) else nodeid + run_str

    def _get_delay_time(self, session: Session) -> float:
        """Extract delay time from session.

        :param session: Pytest session object.
        :return: Returns the delay time for each test loop.
        """
        return session.config.option.codeflash_delay

    def _get_total_time(self, session: Session) -> float:
        """Take all the user available time options, add them and return it in seconds.

        :param session: Pytest session object.
        :return: Returns total amount of time in seconds.
        """
        hours_in_seconds = session.config.option.codeflash_hours * SECONDS_IN_HOUR
        minutes_in_seconds = session.config.option.codeflash_minutes * SECONDS_IN_MINUTE
        seconds = session.config.option.codeflash_seconds
        total_time = hours_in_seconds + minutes_in_seconds + seconds
        if total_time < SHORTEST_AMOUNT_OF_TIME:
            msg = f"Total time cannot be less than: {SHORTEST_AMOUNT_OF_TIME}!"
            raise InvalidTimeParameterError(msg)
        return total_time

    def _timed_out(self, session: Session, start_time: float, count: int) -> bool:
        """Check if the user specified amount of time has lapsed.

        :param session: Pytest session object.
        :return: Returns True if the timeout has expired, False otherwise.
        """
        return count >= session.config.option.codeflash_max_loops or (
            count >= session.config.option.codeflash_min_loops
            and _ORIGINAL_TIME_TIME() - start_time > self._get_total_time(session)
        )

    @pytest.fixture
    def __pytest_loop_step_number(self, request: pytest.FixtureRequest) -> int:
        """Set step number for loop.

        :param request: The number to print.
        :return: request.param.
        """
        marker = request.node.get_closest_marker("loops")
        count = (marker and marker.args[0]) or request.config.option.codeflash_loops
        if count > 1:
            try:
                return request.param
            except AttributeError:
                if issubclass(request.cls, TestCase):
                    warnings.warn("Repeating unittest class tests not supported", stacklevel=2)
                else:
                    msg = "This call couldn't work with pytest-loops. Please consider raising an issue with your usage."
                    raise UnexpectedError(msg) from None
        return count

    @pytest.hookimpl(trylast=True)
    def pytest_generate_tests(self, metafunc: Metafunc) -> None:
        """Create tests based on loop value.

        :param metafunc: pytest metafunction
        :return: None.
        """
        count = metafunc.config.option.codeflash_loops
        m = metafunc.definition.get_closest_marker("loops")

        if m is not None:
            count = int(m.args[0])
        if count > 1:
            metafunc.fixturenames.append("__pytest_loop_step_number")

            def make_progress_id(i: int, n: int = count) -> str:
                return f"{n}/{i + 1}"

            scope = metafunc.config.option.codeflash_loops_scope
            metafunc.parametrize(
                "__pytest_loop_step_number", range(count), indirect=True, ids=make_progress_id, scope=scope
            )
