import os
import sys
import types
from typing import NoReturn
from unittest.mock import patch

import pytest
from _pytest.config import Config

from codeflash.verification.pytest_plugin import (
    InvalidTimeParameterError,
    PytestLoops,
    get_runtime_from_stdout,
    should_stop,
)


@pytest.fixture
def pytest_loops_instance(pytestconfig: Config) -> PytestLoops:
    return PytestLoops(pytestconfig)


@pytest.fixture
def mock_item() -> type:
    class MockItem:
        def __init__(self, function: types.FunctionType, name: str = "test_func", cls: type = None, module: types.ModuleType = None) -> None:
            self.function = function
            self.name = name
            self.cls = cls
            self.module = module

    return MockItem


def create_mock_module(module_name: str, source_code: str, register: bool = False) -> types.ModuleType:
    module = types.ModuleType(module_name)
    exec(source_code, module.__dict__)  # noqa: S102
    if register:
        sys.modules[module_name] = module
    return module


def mock_session(**kwargs):
    """Create a mock session with config options."""
    defaults = {
        "codeflash_hours": 0,
        "codeflash_minutes": 0,
        "codeflash_seconds": 10,
        "codeflash_delay": 0.0,
        "codeflash_loops": 1,
        "codeflash_min_loops": 1,
        "codeflash_max_loops": 100_000,
    }
    defaults.update(kwargs)

    class Option:
        pass

    option = Option()
    for k, v in defaults.items():
        setattr(option, k, v)

    class MockConfig:
        pass

    config = MockConfig()
    config.option = option

    class MockSession:
        pass

    session = MockSession()
    session.config = config
    return session


# --- get_runtime_from_stdout ---


class TestGetRuntimeFromStdout:
    def test_valid_payload(self) -> None:
        assert get_runtime_from_stdout("!######test_func:12345######!") == 12345

    def test_valid_payload_with_surrounding_text(self) -> None:
        assert get_runtime_from_stdout("some output\n!######mod.func:99999######!\nmore output") == 99999

    def test_empty_string(self) -> None:
        assert get_runtime_from_stdout("") is None

    def test_no_markers(self) -> None:
        assert get_runtime_from_stdout("just some output") is None

    def test_missing_end_marker(self) -> None:
        assert get_runtime_from_stdout("!######test:123") is None

    def test_missing_start_marker(self) -> None:
        assert get_runtime_from_stdout("test:123######!") is None

    def test_no_colon_in_payload(self) -> None:
        assert get_runtime_from_stdout("!######nocolon######!") is None

    def test_non_integer_value(self) -> None:
        assert get_runtime_from_stdout("!######test:notanumber######!") is None

    def test_multiple_markers_uses_last(self) -> None:
        stdout = "!######first:111######! middle !######second:222######!"
        assert get_runtime_from_stdout(stdout) == 222


# --- should_stop ---


class TestShouldStop:
    def test_not_enough_data_for_window(self) -> None:
        assert should_stop([100, 100], window=5, min_window_size=3) is False

    def test_below_min_window_size(self) -> None:
        assert should_stop([100, 100], window=2, min_window_size=5) is False

    def test_stable_runtimes_stops(self) -> None:
        runtimes = [1000000] * 10
        assert should_stop(runtimes, window=5, min_window_size=3, center_rel_tol=0.01, spread_rel_tol=0.01) is True

    def test_unstable_runtimes_continues(self) -> None:
        runtimes = [100, 200, 100, 200, 100]
        assert should_stop(runtimes, window=5, min_window_size=3, center_rel_tol=0.01, spread_rel_tol=0.01) is False

    def test_zero_runtimes_raises(self) -> None:
        # All-zero runtimes cause ZeroDivisionError in median check.
        # In practice the caller guards with best_runtime_until_now > 0.
        runtimes = [0, 0, 0, 0, 0]
        with pytest.raises(ZeroDivisionError):
            should_stop(runtimes, window=5, min_window_size=3)

    def test_even_window_median(self) -> None:
        # Even window: median is average of two middle values
        runtimes = [1000, 1000, 1001, 1001]
        assert should_stop(runtimes, window=4, min_window_size=2, center_rel_tol=0.01, spread_rel_tol=0.01) is True

    def test_centered_but_spread_too_large(self) -> None:
        # All close to median but spread exceeds tolerance
        runtimes = [1000, 1050, 1000, 1050, 1000]
        assert should_stop(runtimes, window=5, min_window_size=3, center_rel_tol=0.1, spread_rel_tol=0.001) is False


# --- _set_nodeid ---


class TestSetNodeid:
    def test_appends_count_to_plain_nodeid(self, pytest_loops_instance: PytestLoops) -> None:
        result = pytest_loops_instance._set_nodeid("test_module.py::test_func", 3)  # noqa: SLF001
        assert result == "test_module.py::test_func[ 3 ]"
        assert os.environ["CODEFLASH_LOOP_INDEX"] == "3"

    def test_replaces_existing_count(self, pytest_loops_instance: PytestLoops) -> None:
        result = pytest_loops_instance._set_nodeid("test_module.py::test_func[ 1 ]", 5)  # noqa: SLF001
        assert result == "test_module.py::test_func[ 5 ]"

    def test_replaces_only_loop_pattern(self, pytest_loops_instance: PytestLoops) -> None:
        # Parametrize brackets like [param0] should not be replaced
        result = pytest_loops_instance._set_nodeid("test_mod.py::test_func[param0]", 2)  # noqa: SLF001
        assert result == "test_mod.py::test_func[param0][ 2 ]"


# --- _get_total_time ---


class TestGetTotalTime:
    def test_seconds_only(self, pytest_loops_instance: PytestLoops) -> None:
        session = mock_session(codeflash_seconds=30)
        assert pytest_loops_instance._get_total_time(session) == 30  # noqa: SLF001

    def test_mixed_units(self, pytest_loops_instance: PytestLoops) -> None:
        session = mock_session(codeflash_hours=1, codeflash_minutes=30, codeflash_seconds=45)
        assert pytest_loops_instance._get_total_time(session) == 3600 + 1800 + 45  # noqa: SLF001

    def test_zero_time_is_valid(self, pytest_loops_instance: PytestLoops) -> None:
        session = mock_session(codeflash_hours=0, codeflash_minutes=0, codeflash_seconds=0)
        assert pytest_loops_instance._get_total_time(session) == 0  # noqa: SLF001

    def test_negative_time_raises(self, pytest_loops_instance: PytestLoops) -> None:
        session = mock_session(codeflash_hours=0, codeflash_minutes=0, codeflash_seconds=-1)
        with pytest.raises(InvalidTimeParameterError):
            pytest_loops_instance._get_total_time(session)  # noqa: SLF001


# --- _timed_out ---


class TestTimedOut:
    def test_exceeds_max_loops(self, pytest_loops_instance: PytestLoops) -> None:
        session = mock_session(codeflash_max_loops=10, codeflash_min_loops=1, codeflash_seconds=9999)
        assert pytest_loops_instance._timed_out(session, start_time=0, count=10) is True  # noqa: SLF001

    def test_below_min_loops_never_times_out(self, pytest_loops_instance: PytestLoops) -> None:
        session = mock_session(codeflash_max_loops=100_000, codeflash_min_loops=50, codeflash_seconds=0)
        # Even with 0 seconds budget, count < min_loops means not timed out
        assert pytest_loops_instance._timed_out(session, start_time=0, count=5) is False  # noqa: SLF001

    def test_above_min_loops_and_time_exceeded(self, pytest_loops_instance: PytestLoops) -> None:
        session = mock_session(codeflash_max_loops=100_000, codeflash_min_loops=1, codeflash_seconds=1)
        # start_time far in the past → time exceeded
        assert pytest_loops_instance._timed_out(session, start_time=0, count=2) is True  # noqa: SLF001


# --- _get_delay_time ---


class TestGetDelayTime:
    def test_returns_configured_delay(self, pytest_loops_instance: PytestLoops) -> None:
        session = mock_session(codeflash_delay=0.5)
        assert pytest_loops_instance._get_delay_time(session) == 0.5  # noqa: SLF001


# --- pytest_runtest_logreport ---


class TestRunTestLogReport:
    def test_skipped_when_stability_check_disabled(self, pytestconfig: Config) -> None:
        instance = PytestLoops(pytestconfig)
        instance.enable_stability_check = False

        class MockReport:
            when = "call"
            passed = True
            capstdout = "!######func:12345######!"
            nodeid = "test::func"

        instance.pytest_runtest_logreport(MockReport())
        assert instance.runtime_data_by_test_case == {}

    def test_records_runtime_on_passed_call(self, pytestconfig: Config) -> None:
        instance = PytestLoops(pytestconfig)
        instance.enable_stability_check = True

        class MockReport:
            when = "call"
            passed = True
            capstdout = "!######func:12345######!"
            nodeid = "test::func [ 1 ]"

        instance.pytest_runtest_logreport(MockReport())
        assert "test::func" in instance.runtime_data_by_test_case
        assert instance.runtime_data_by_test_case["test::func"] == [12345]

    def test_ignores_non_call_phase(self, pytestconfig: Config) -> None:
        instance = PytestLoops(pytestconfig)
        instance.enable_stability_check = True

        class MockReport:
            when = "setup"
            passed = True
            capstdout = "!######func:12345######!"
            nodeid = "test::func"

        instance.pytest_runtest_logreport(MockReport())
        assert instance.runtime_data_by_test_case == {}


# --- pytest_runtest_setup / teardown ---


class TestRunTestSetupTeardown:
    def test_setup_sets_env_vars(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        module = types.ModuleType("my_test_module")

        class MyTestClass:
            pass

        item = mock_item(lambda: None, name="test_something[param1]", cls=MyTestClass, module=module)
        pytest_loops_instance.pytest_runtest_setup(item)

        assert os.environ["CODEFLASH_TEST_MODULE"] == "my_test_module"
        assert os.environ["CODEFLASH_TEST_CLASS"] == "MyTestClass"
        assert os.environ["CODEFLASH_TEST_FUNCTION"] == "test_something"

    def test_setup_no_class(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        module = types.ModuleType("my_test_module")
        item = mock_item(lambda: None, name="test_plain", cls=None, module=module)
        pytest_loops_instance.pytest_runtest_setup(item)

        assert os.environ["CODEFLASH_TEST_CLASS"] == ""

    def test_teardown_clears_env_vars(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        os.environ["CODEFLASH_TEST_MODULE"] = "leftover"
        os.environ["CODEFLASH_TEST_CLASS"] = "leftover"
        os.environ["CODEFLASH_TEST_FUNCTION"] = "leftover"

        item = mock_item(lambda: None)
        pytest_loops_instance.pytest_runtest_teardown(item)

        assert "CODEFLASH_TEST_MODULE" not in os.environ
        assert "CODEFLASH_TEST_CLASS" not in os.environ
        assert "CODEFLASH_TEST_FUNCTION" not in os.environ


# --- _clear_lru_caches ---


class TestClearLruCaches:
    def test_clears_lru_cached_function(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        source_code = """
import functools

@functools.lru_cache(maxsize=None)
def my_func(x):
    return x * 2

my_func(10)
my_func(10)
"""
        mock_module = create_mock_module("test_module_func", source_code)
        item = mock_item(mock_module.my_func)
        pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
        assert mock_module.my_func.cache_info().hits == 0
        assert mock_module.my_func.cache_info().misses == 0
        assert mock_module.my_func.cache_info().currsize == 0

    def test_clears_class_method_cache(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        source_code = """
import functools

class MyClass:
    @functools.lru_cache(maxsize=None)
    def my_method(self, x):
        return x * 3

obj = MyClass()
obj.my_method(5)
obj.my_method(5)
# """
        mock_module = create_mock_module("test_module_class", source_code)
        item = mock_item(mock_module.MyClass.my_method)
        pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
        assert mock_module.MyClass.my_method.cache_info().hits == 0
        assert mock_module.MyClass.my_method.cache_info().misses == 0
        assert mock_module.MyClass.my_method.cache_info().currsize == 0

    def test_handles_exception_in_cache_clear(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        class BrokenCache:
            def cache_clear(self) -> NoReturn:
                msg = "Cache clearing failed!"
                raise ValueError(msg)

        item = mock_item(BrokenCache())
        pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001

    def test_handles_no_cache(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        def no_cache_func(x: int) -> int:
            return x

        item = mock_item(no_cache_func)
        pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001

    def test_clears_module_level_caches_via_sys_modules(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        module_name = "_cf_test_module_scan"
        source_code = """
import functools

@functools.lru_cache(maxsize=None)
def cached_a(x):
    return x + 1

@functools.lru_cache(maxsize=None)
def cached_b(x):
    return x + 2

def plain_func(x):
    return x

cached_a(1)
cached_a(1)
cached_b(2)
cached_b(2)
"""
        mock_module = create_mock_module(module_name, source_code, register=True)
        try:
            item = mock_item(mock_module.plain_func)
            pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001

            assert mock_module.cached_a.cache_info().currsize == 0
            assert mock_module.cached_b.cache_info().currsize == 0
        finally:
            sys.modules.pop(module_name, None)

    def test_skips_protected_modules(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        module_name = "_cf_test_protected"
        source_code = """
import functools

@functools.lru_cache(maxsize=None)
def user_func(x):
    return x
"""
        mock_module = create_mock_module(module_name, source_code, register=True)
        try:
            mock_module.os_exists = os.path.exists
            item = mock_item(mock_module.user_func)
            pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
        finally:
            sys.modules.pop(module_name, None)

    def test_caches_scan_result(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        module_name = "_cf_test_cache_reuse"
        source_code = """
import functools

@functools.lru_cache(maxsize=None)
def cached_fn(x):
    return x
"""
        mock_module = create_mock_module(module_name, source_code, register=True)
        try:
            item = mock_item(mock_module.cached_fn)

            pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
            assert module_name in pytest_loops_instance._module_clearables  # noqa: SLF001

            mock_module.cached_fn(42)
            assert mock_module.cached_fn.cache_info().currsize == 1

            with patch("codeflash.verification.pytest_plugin.inspect.getmembers") as mock_getmembers:
                pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
                mock_getmembers.assert_not_called()

            assert mock_module.cached_fn.cache_info().currsize == 0
        finally:
            sys.modules.pop(module_name, None)

    def test_handles_wrapped_function(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        module_name = "_cf_test_wrapped"
        source_code = """
import functools

@functools.lru_cache(maxsize=None)
def inner(x):
    return x

def wrapper(x):
    return inner(x)

wrapper.__wrapped__ = inner
wrapper.__module__ = __name__

inner(1)
inner(1)
"""
        mock_module = create_mock_module(module_name, source_code, register=True)
        try:
            item = mock_item(mock_module.wrapper)
            pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
            assert mock_module.inner.cache_info().currsize == 0
        finally:
            sys.modules.pop(module_name, None)

    def test_handles_function_without_module(self, pytest_loops_instance: PytestLoops, mock_item: type) -> None:
        def func() -> None:
            pass

        func.__module__ = None  # type: ignore[assignment]
        item = mock_item(func)
        pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
