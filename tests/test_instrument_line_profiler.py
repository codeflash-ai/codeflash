import os
from pathlib import Path

from codeflash.code_utils.line_profile_utils import add_decorator_imports
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


def test_add_decorator_imports_helper_in_class():
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_classmethod.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    original_cwd = Path.cwd()
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sort_classmethod", parents=[], file_path=code_path)
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    os.chdir(run_cwd)
    #func_optimizer = pass
    try:
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        expected_code_main = f"""from line_profiler import profile as codeflash_line_profile
codeflash_line_profile.enable(output_prefix='{line_profiler_output_file}')

from code_to_optimize.bubble_sort_in_class import BubbleSortClass


@codeflash_line_profile
def sort_classmethod(x):
    y = BubbleSortClass()
    return y.sorter(x)
"""
        expected_code_helper = """from line_profiler import profile as codeflash_line_profile


def hi():
    pass


class BubbleSortClass:
    def __init__(self):
        pass

    @codeflash_line_profile
    def sorter(self, arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def helper(self, arr, j):
        return arr[j] > arr[j + 1]
"""
        assert code_path.read_text("utf-8") == expected_code_main
        assert code_context.helper_functions[0].file_path.read_text("utf-8") == expected_code_helper
    finally:
        func_optimizer.write_code_and_helpers(
            func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
        )

def test_add_decorator_imports_helper_in_nested_class():
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_nested_classmethod.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    original_cwd = Path.cwd()
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sort_classmethod", parents=[], file_path=code_path)
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    os.chdir(run_cwd)
    #func_optimizer = pass
    try:
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        expected_code_main = f"""from line_profiler import profile as codeflash_line_profile
codeflash_line_profile.enable(output_prefix='{line_profiler_output_file}')

from code_to_optimize.bubble_sort_in_nested_class import WrapperClass


@codeflash_line_profile
def sort_classmethod(x):
    y = WrapperClass.BubbleSortClass()
    return y.sorter(x)
"""
        expected_code_helper = """from line_profiler import profile as codeflash_line_profile


def hi():
    pass


class WrapperClass:
    def __init__(self):
        pass

    class BubbleSortClass:
        def __init__(self):
            pass

        @codeflash_line_profile
        def sorter(self, arr):
            def inner_helper(arr, j):
                return arr[j] > arr[j + 1]

            for i in range(len(arr)):
                for j in range(len(arr) - 1):
                    if arr[j] > arr[j + 1]:
                        temp = arr[j]
                        arr[j] = arr[j + 1]
                        arr[j + 1] = temp
            return arr

        def helper(self, arr, j):
            return arr[j] > arr[j + 1]
"""
        assert code_path.read_text("utf-8") == expected_code_main
        assert code_context.helper_functions[0].file_path.read_text("utf-8") == expected_code_helper
    finally:
        func_optimizer.write_code_and_helpers(
            func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
        )
