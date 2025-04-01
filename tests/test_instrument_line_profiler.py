import os
from pathlib import Path
from tempfile import TemporaryDirectory

from codeflash.code_utils.line_profile_utils import add_decorator_imports
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


def test_add_decorator_imports_helper_in_class():
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_classmethod.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
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
    #Need to invert the assert once the helper detection is fixed
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_nested_classmethod.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
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
        assert code_path.read_text("utf-8") == expected_code_main
        assert code_context.helper_functions.__len__() == 0
    finally:
        func_optimizer.write_code_and_helpers(
            func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
        )

def test_add_decorator_imports_nodeps():
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
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


@codeflash_line_profile
def sorter(arr):
    print("codeflash stdout: Sorting list")
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    print(f"result: {{arr}}")
    return arr
"""
        assert code_path.read_text("utf-8") == expected_code_main
    finally:
        func_optimizer.write_code_and_helpers(
            func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
        )

def test_add_decorator_imports_helper_outside():
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_deps.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sorter_deps", parents=[], file_path=code_path)
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

from code_to_optimize.bubble_sort_dep1_helper import dep1_comparer
from code_to_optimize.bubble_sort_dep2_swap import dep2_swap


@codeflash_line_profile
def sorter_deps(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if dep1_comparer(arr, j):
                dep2_swap(arr, j)
    return arr

"""
        expected_code_helper1 = """from line_profiler import profile as codeflash_line_profile


@codeflash_line_profile
def dep1_comparer(arr, j: int) -> bool:
    return arr[j] > arr[j + 1]
"""
        expected_code_helper2="""from line_profiler import profile as codeflash_line_profile


@codeflash_line_profile
def dep2_swap(arr, j):
    temp = arr[j]
    arr[j] = arr[j + 1]
    arr[j + 1] = temp
"""
        assert code_path.read_text("utf-8") == expected_code_main
        assert code_context.helper_functions[0].file_path.read_text("utf-8") == expected_code_helper1
        assert code_context.helper_functions[1].file_path.read_text("utf-8") == expected_code_helper2
    finally:
        func_optimizer.write_code_and_helpers(
            func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
        )

def test_add_decorator_imports_helper_in_dunder_class():
    code_str = """def sorter(arr):
    ans = helper(arr)
    return ans
class helper:
    def __init__(self, arr):
        return arr.sort()"""
    code_path = TemporaryDirectory()
    code_write_path = Path(code_path.name) / "dunder_class.py"
    code_write_path.write_text(code_str,"utf-8")
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = Path(code_path.name)
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_write_path)
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


@codeflash_line_profile
def sorter(arr):
    ans = helper(arr)
    return ans
class helper:
    def __init__(self, arr):
        return arr.sort()
"""
        assert code_write_path.read_text("utf-8") == expected_code_main
    finally:
        pass
