import tempfile
from pathlib import Path
import pytest

from codeflash.discovery.functions_to_optimize import (
    find_all_functions_in_file,
    get_functions_to_optimize,
    inspect_top_level_functions_or_methods,
)
from codeflash.verification.verification_utils import TestConfig


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp:
        yield Path(temp)


def test_async_function_detection(temp_dir):
    async_function = """
async def async_function_with_return():
    await some_async_operation()
    return 42

async def async_function_without_return():
    await some_async_operation()
    print("No return")

def regular_function():
    return 10
"""
    
    file_path = temp_dir / "test_file.py"
    file_path.write_text(async_function)
    functions_found = find_all_functions_in_file(file_path)
    
    function_names = [fn.function_name for fn in functions_found[file_path]]
    
    assert "async_function_with_return" in function_names
    assert "regular_function" in function_names
    assert "async_function_without_return" not in function_names


def test_async_method_in_class(temp_dir):
    code_with_async_method = """
class AsyncClass:
    async def async_method(self):
        await self.do_something()
        return "result"
    
    async def async_method_no_return(self):
        await self.do_something()
        pass
    
    def sync_method(self):
        return "sync result"
"""
    
    file_path = temp_dir / "test_file.py"
    file_path.write_text(code_with_async_method)
    functions_found = find_all_functions_in_file(file_path)
    
    found_functions = functions_found[file_path]
    function_names = [fn.function_name for fn in found_functions]
    qualified_names = [fn.qualified_name for fn in found_functions]
    
    assert "async_method" in function_names
    assert "AsyncClass.async_method" in qualified_names
    
    assert "sync_method" in function_names
    assert "AsyncClass.sync_method" in qualified_names
    
    assert "async_method_no_return" not in function_names


def test_nested_async_functions(temp_dir):
    nested_async = """
async def outer_async():
    async def inner_async():
        return "inner"
    
    result = await inner_async()
    return result

def outer_sync():
    async def inner_async():
        return "inner from sync"
    
    return inner_async
"""
    
    file_path = temp_dir / "test_file.py"
    file_path.write_text(nested_async)
    functions_found = find_all_functions_in_file(file_path)
    
    function_names = [fn.function_name for fn in functions_found[file_path]]
    
    assert "outer_async" in function_names
    assert "outer_sync" in function_names
    assert "inner_async" not in function_names


def test_async_staticmethod_and_classmethod(temp_dir):
    async_decorators = """
class MyClass:
    @staticmethod
    async def async_static_method():
        await some_operation()
        return "static result"
    
    @classmethod
    async def async_class_method(cls):
        await cls.some_operation()
        return "class result"
    
    @property
    async def async_property(self):
        return await self.get_value()
"""
    
    file_path = temp_dir / "test_file.py"
    file_path.write_text(async_decorators)
    functions_found = find_all_functions_in_file(file_path)
    
    function_names = [fn.function_name for fn in functions_found[file_path]]
    
    assert "async_static_method" in function_names
    assert "async_class_method" in function_names
    
    assert "async_property" not in function_names


def test_async_generator_functions(temp_dir):
    async_generators = """
async def async_generator_with_return():
    for i in range(10):
        yield i
    return "done"

async def async_generator_no_return():
    for i in range(10):
        yield i

async def regular_async_with_return():
    result = await compute()
    return result
"""
    
    file_path = temp_dir / "test_file.py"
    file_path.write_text(async_generators)
    functions_found = find_all_functions_in_file(file_path)
    
    function_names = [fn.function_name for fn in functions_found[file_path]]
    
    assert "async_generator_with_return" in function_names
    assert "regular_async_with_return" in function_names
    assert "async_generator_no_return" not in function_names


def test_inspect_async_top_level_functions(temp_dir):
    code = """
async def top_level_async():
    return 42

class AsyncContainer:
    async def async_method(self):
        async def nested_async():
            return 1
        return await nested_async()
    
    @staticmethod
    async def async_static():
        return "static"
    
    @classmethod
    async def async_classmethod(cls):
        return "classmethod"
"""
    
    file_path = temp_dir / "test_file.py"
    file_path.write_text(code)
    
    result = inspect_top_level_functions_or_methods(file_path, "top_level_async")
    assert result.is_top_level
    
    result = inspect_top_level_functions_or_methods(file_path, "async_method", class_name="AsyncContainer")
    assert result.is_top_level
    
    result = inspect_top_level_functions_or_methods(file_path, "nested_async", class_name="AsyncContainer")
    assert not result.is_top_level
    
    result = inspect_top_level_functions_or_methods(file_path, "async_static", class_name="AsyncContainer")
    assert result.is_top_level
    assert result.is_staticmethod
    
    result = inspect_top_level_functions_or_methods(file_path, "async_classmethod", class_name="AsyncContainer")
    assert result.is_top_level
    assert result.is_classmethod


def test_get_functions_to_optimize_with_async(temp_dir):
    mixed_code = """
async def async_func_one():
    return await operation_one()

def sync_func_one():
    return operation_one()

async def async_func_two():
    print("no return")

class MixedClass:
    async def async_method(self):
        return await self.operation()
    
    def sync_method(self):
        return self.operation()
"""
    
    file_path = temp_dir / "test_file.py"
    file_path.write_text(mixed_code)
    
    test_config = TestConfig(
        tests_root="tests",
        project_root_path=".",
        test_framework="pytest",
        tests_project_rootdir=Path()
    )
    
    functions, functions_count, _ = get_functions_to_optimize(
        optimize_all=None,
        replay_test=None,
        file=file_path,
        only_get_this_function=None,
        test_cfg=test_config,
        ignore_paths=[],
        project_root=file_path.parent,
        module_root=file_path.parent,
    )
    
    assert functions_count == 4
    
    function_names = [fn.function_name for fn in functions[file_path]]
    assert "async_func_one" in function_names
    assert "sync_func_one" in function_names
    assert "async_method" in function_names
    assert "sync_method" in function_names
    
    assert "async_func_two" not in function_names


def test_async_function_parents(temp_dir):
    complex_structure = """
class OuterClass:
    async def outer_method(self):
        return 1
    
    class InnerClass:
        async def inner_method(self):
            return 2

async def module_level_async():
    class LocalClass:
        async def local_method(self):
            return 3
    return LocalClass()
"""
    
    file_path = temp_dir / "test_file.py"
    file_path.write_text(complex_structure)
    functions_found = find_all_functions_in_file(file_path)
    
    found_functions = functions_found[file_path]
    
    for fn in found_functions:
        if fn.function_name == "outer_method":
            assert len(fn.parents) == 1
            assert fn.parents[0].name == "OuterClass"
            assert fn.qualified_name == "OuterClass.outer_method"
        elif fn.function_name == "inner_method":
            assert len(fn.parents) == 2
            assert fn.parents[0].name == "OuterClass"
            assert fn.parents[1].name == "InnerClass"
        elif fn.function_name == "module_level_async":
            assert len(fn.parents) == 0
            assert fn.qualified_name == "module_level_async"