import tempfile
from pathlib import Path

from codeflash.discovery.functions_to_optimize import (
    filter_files_optimized,
    find_all_functions_in_file,
    get_functions_to_optimize,
    inspect_top_level_functions_or_methods,
)
from codeflash.verification.verification_utils import TestConfig


def test_function_eligible_for_optimization() -> None:
    function = """def test_function_eligible_for_optimization():
    a = 5
    return a**2
    """
    functions_found = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "temp_file.py"
        temp_file.write_text(function)
        functions_found = find_all_functions_in_file(temp_file)
    assert functions_found[temp_file][0].function_name == "test_function_eligible_for_optimization"

    # Has no return statement
    function = """def test_function_not_eligible_for_optimization():
    a = 5
    print(a)
    """
    functions_found = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "temp_file.py"
        temp_file.write_text(function)
        functions_found = find_all_functions_in_file(temp_file)
    assert len(functions_found[temp_file]) == 0


def test_find_top_level_function_or_method():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "temp_file.py"
        temp_file.write_text(
            """def functionA():
    def functionB():
        return 5
    class E:
        def functionF():
            pass
    return functionA()
class A:
    def functionC():
        def functionD():
            pass
        return 6
class AirbyteEntrypoint(object):
    @staticmethod
    def handle_record_counts(message: AirbyteMessage, stream_message_count: DefaultDict[HashableStreamDescriptor, float]) -> AirbyteMessage:
        return "idontcare"
    @classmethod
    def functionE(cls, num):
        return AirbyteEntrypoint.handle_record_counts(num)
def non_classmethod_function(cls, name):
    return cls.name
    """
        )
        path_obj_name = temp_file
        assert inspect_top_level_functions_or_methods(path_obj_name, "functionA").is_top_level
        assert not inspect_top_level_functions_or_methods(path_obj_name, "functionB").is_top_level
        assert inspect_top_level_functions_or_methods(path_obj_name, "functionC", class_name="A").is_top_level
        assert not inspect_top_level_functions_or_methods(path_obj_name, "functionD", class_name="A").is_top_level
        assert not inspect_top_level_functions_or_methods(path_obj_name, "functionF", class_name="E").is_top_level
        assert not inspect_top_level_functions_or_methods(path_obj_name, "functionA").has_args
        staticmethod_func = inspect_top_level_functions_or_methods(
            path_obj_name, "handle_record_counts", class_name=None, line_no=15
        )
        assert staticmethod_func.is_staticmethod
        assert staticmethod_func.staticmethod_class_name == "AirbyteEntrypoint"
        assert inspect_top_level_functions_or_methods(
            path_obj_name, "functionE", class_name="AirbyteEntrypoint"
        ).is_classmethod
        assert not inspect_top_level_functions_or_methods(
            path_obj_name, "non_classmethod_function", class_name="AirbyteEntrypoint"
        ).is_top_level
        # needed because this will be traced with a class_name being passed


def test_class_method_discovery():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "temp_file.py"
        temp_file.write_text(
            """class A:
    def functionA():
        return True
    def functionB():
        return False
class X:
    def functionA():
        return True
    def functionB():
        return False
def functionA():
    return True"""
        )
        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )
        path_obj_name = temp_file
        functions, functions_count = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=path_obj_name,
            only_get_this_function="A.functionA",
            test_cfg=test_config,
            ignore_paths=[Path("/bruh/")],
            project_root=path_obj_name.parent,
            module_root=path_obj_name.parent,
        )
        assert len(functions) == 1
        for file in functions:
            assert functions[file][0].qualified_name == "A.functionA"
            assert functions[file][0].function_name == "functionA"
            assert functions[file][0].top_level_parent_name == "A"

        functions, functions_count = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=path_obj_name,
            only_get_this_function="X.functionA",
            test_cfg=test_config,
            ignore_paths=[Path("/bruh/")],
            project_root=path_obj_name.parent,
            module_root=path_obj_name.parent,
        )
        assert len(functions) == 1
        for file in functions:
            assert functions[file][0].qualified_name == "X.functionA"
            assert functions[file][0].function_name == "functionA"
            assert functions[file][0].top_level_parent_name == "X"

        functions, functions_count = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=path_obj_name,
            only_get_this_function="functionA",
            test_cfg=test_config,
            ignore_paths=[Path("/bruh/")],
            project_root=path_obj_name.parent,
            module_root=path_obj_name.parent,
        )
        assert len(functions) == 1
        for file in functions:
            assert functions[file][0].qualified_name == "functionA"
            assert functions[file][0].function_name == "functionA"


def test_filter_files_optimized():
    tests_root = Path("tests").resolve()
    module_root = Path().resolve()
    ignore_paths = []

    file_path_test = Path("tests/test_function_discovery.py").resolve()
    file_path_same_level = Path("file.py").resolve()
    file_path_different_level = Path("src/file.py").resolve()
    file_path_above_level = Path("../file.py").resolve()

    assert not filter_files_optimized(file_path_test, tests_root, ignore_paths, module_root)
    assert filter_files_optimized(file_path_same_level, tests_root, ignore_paths, module_root)
    assert filter_files_optimized(file_path_different_level, tests_root, ignore_paths, module_root)
    assert not filter_files_optimized(file_path_above_level, tests_root, ignore_paths, module_root)
