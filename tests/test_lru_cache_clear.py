import types
from typing import NoReturn

import pytest
from _pytest.config import Config

from codeflash.verification.pytest_plugin import PyTest_Loops


@pytest.fixture
def pytest_loops_instance(pytestconfig: Config) -> PyTest_Loops:
    return PyTest_Loops(pytestconfig)


@pytest.fixture
def mock_item() -> type:
    class MockItem:
        def __init__(self, function: types.FunctionType) -> None:
            self.function = function

    return MockItem


def create_mock_module(module_name: str, source_code: str) -> types.ModuleType:
    module = types.ModuleType(module_name)
    exec(source_code, module.__dict__)  # noqa: S102
    return module


def test_clear_lru_caches_function(pytest_loops_instance: PyTest_Loops, mock_item: type) -> None:
    source_code = """
import functools

@functools.lru_cache(maxsize=None)
def my_func(x):
    return x * 2

my_func(10)  # miss the cache
my_func(10)  # hit the cache
"""
    mock_module = create_mock_module("test_module_func", source_code)
    item = mock_item(mock_module.my_func)
    pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
    assert mock_module.my_func.cache_info().hits == 0
    assert mock_module.my_func.cache_info().misses == 0
    assert mock_module.my_func.cache_info().currsize == 0


def test_clear_lru_caches_class_method(pytest_loops_instance: PyTest_Loops, mock_item: type) -> None:
    source_code = """
import functools

class MyClass:
    @functools.lru_cache(maxsize=None)
    def my_method(self, x):
        return x * 3

obj = MyClass()
obj.my_method(5)  # Pre-populate the cache
obj.my_method(5)  # Hit the cache
# """
    mock_module = create_mock_module("test_module_class", source_code)
    item = mock_item(mock_module.MyClass.my_method)
    pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
    assert mock_module.MyClass.my_method.cache_info().hits == 0
    assert mock_module.MyClass.my_method.cache_info().misses == 0
    assert mock_module.MyClass.my_method.cache_info().currsize == 0


def test_clear_lru_caches_exception_handling(pytest_loops_instance: PyTest_Loops, mock_item: type) -> None:
    """Test that exceptions during clearing are handled."""

    class BrokenCache:
        def cache_clear(self) -> NoReturn:
            msg = "Cache clearing failed!"
            raise ValueError(msg)

    item = mock_item(BrokenCache())
    pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001


def test_clear_lru_caches_no_cache(pytest_loops_instance: PyTest_Loops, mock_item: type) -> None:
    def no_cache_func(x: int) -> int:
        return x

    item = mock_item(no_cache_func)
    pytest_loops_instance._clear_lru_caches(item)  # noqa: SLF001
