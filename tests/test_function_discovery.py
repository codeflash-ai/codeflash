import tempfile
from pathlib import Path
import os
import unittest.mock

from codeflash.discovery.functions_to_optimize import (
    filter_files_optimized,
    find_all_functions_in_file,
    get_functions_to_optimize,
    inspect_top_level_functions_or_methods,
    filter_functions
)
from codeflash.verification.verification_utils import TestConfig
from codeflash.code_utils.compat import codeflash_temp_dir


def test_function_eligible_for_optimization() -> None:
    function = """def test_function_eligible_for_optimization():
    a = 5
    return a**2
    """
    functions_found = {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(function)
        f.flush()
        functions_found = find_all_functions_in_file(Path(f.name))
    assert functions_found[Path(f.name)][0].function_name == "test_function_eligible_for_optimization"

    # Has no return statement
    function = """def test_function_not_eligible_for_optimization():
    a = 5
    print(a)
    """
    functions_found = {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(function)
        f.flush()
        functions_found = find_all_functions_in_file(Path(f.name))
    assert len(functions_found[Path(f.name)]) == 0


    # we want to trigger an error in the function discovery
    function = """def test_invalid_code():"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(function)
        f.flush()
        functions_found = find_all_functions_in_file(Path(f.name))
    assert functions_found == {}




def test_find_top_level_function_or_method():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(
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
        f.flush()
        path_obj_name = Path(f.name)
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

    # we want to write invalid code to ensure that the function discovery does not crash
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(
            """def functionA():
"""
        )
        f.flush()
        path_obj_name = Path(f.name)
        assert not inspect_top_level_functions_or_methods(path_obj_name, "functionA")

def test_class_method_discovery():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(
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
        f.flush()
        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )
        path_obj_name = Path(f.name)
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


def test_nested_function():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(
"""
import copy 

def propagate_attributes(
    nodes: dict[str, dict], edges: list[dict], source_node_id: str, attribute: str
) -> dict[str, dict]:
    modified_nodes = copy.deepcopy(nodes)

    # Build an adjacency list for faster traversal
    adjacency = {}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append(tgt)

    # Track visited nodes to avoid cycles
    visited = set()

    def traverse(node_id):
        if node_id in visited:
            return
        visited.add(node_id)

        # Propagate attribute from source node
        if (
            node_id != source_node_id
            and source_node_id in modified_nodes
            and attribute in modified_nodes[source_node_id]
        ):
            if node_id in modified_nodes:
                modified_nodes[node_id][attribute] = modified_nodes[source_node_id][
                    attribute
                ]

        # Continue propagation to neighbors
        for neighbor in adjacency.get(node_id, []):
            traverse(neighbor)

    traverse(source_node_id)
    return modified_nodes
"""
        )
        f.flush()
        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )
        path_obj_name = Path(f.name)
        functions, functions_count = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=path_obj_name,
            test_cfg=test_config,
            only_get_this_function=None,
            ignore_paths=[Path("/bruh/")],
            project_root=path_obj_name.parent,
            module_root=path_obj_name.parent,
        )

        assert len(functions) == 1
        assert functions_count == 1

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(
"""
def outer_function():
    def inner_function():
        pass

    return inner_function
"""
        )
        f.flush()
        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )
        path_obj_name = Path(f.name)
        functions, functions_count = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=path_obj_name,
            test_cfg=test_config,
            only_get_this_function=None,
            ignore_paths=[Path("/bruh/")],
            project_root=path_obj_name.parent,
            module_root=path_obj_name.parent,
        )

        assert len(functions) == 1
        assert functions_count == 1

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(
"""
def outer_function():
    def inner_function():
        pass
    
    def another_inner_function():
        pass
    return inner_function, another_inner_function
"""
        )
        f.flush()
        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )
        path_obj_name = Path(f.name)
        functions, functions_count = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=path_obj_name,
            test_cfg=test_config,
            only_get_this_function=None,
            ignore_paths=[Path("/bruh/")],
            project_root=path_obj_name.parent,
            module_root=path_obj_name.parent,
        )

        assert len(functions) == 1
        assert functions_count == 1


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

def test_filter_functions():
    with codeflash_temp_dir.joinpath("test_get_functions_to_optimize.py").open("w") as f:
        f.write(
"""
import copy 

def propagate_attributes(
    nodes: dict[str, dict], edges: list[dict], source_node_id: str, attribute: str
) -> dict[str, dict]:
    modified_nodes = copy.deepcopy(nodes)

    # Build an adjacency list for faster traversal
    adjacency = {}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append(tgt)

    # Track visited nodes to avoid cycles
    visited = set()

    def traverse(node_id):
        if node_id in visited:
            return
        visited.add(node_id)

        # Propagate attribute from source node
        if (
            node_id != source_node_id
            and source_node_id in modified_nodes
            and attribute in modified_nodes[source_node_id]
        ):
            if node_id in modified_nodes:
                modified_nodes[node_id][attribute] = modified_nodes[source_node_id][
                    attribute
                ]

        # Continue propagation to neighbors
        for neighbor in adjacency.get(node_id, []):
            traverse(neighbor)

    traverse(source_node_id)
    return modified_nodes

def vanilla_function():
    return "This is a vanilla function."
    
def not_in_checkpoint_function():
    return "This function is not in the checkpoint."
"""
        )
        f.flush()
        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )

        file_path = codeflash_temp_dir.joinpath("test_get_functions_to_optimize.py")
        discovered = find_all_functions_in_file(file_path)
        modified_functions = {file_path: discovered[file_path]}
        filtered, count = filter_functions(
            modified_functions,
            tests_root=Path("tests"),
            ignore_paths=[],
            project_root=file_path.parent,
            module_root=file_path.parent,
        )
        function_names = [fn.function_name for fn in filtered.get(file_path, [])]
        assert "propagate_attributes" in function_names
        assert count == 3

    tests_root_dir = codeflash_temp_dir.joinpath("tests")
    tests_root_dir.mkdir(exist_ok=True)
    
    test_file_path = tests_root_dir.joinpath("test_functions.py")
    with test_file_path.open("w") as f:
        f.write(
"""
def test_function_in_tests_dir():
    return "This function is in a test directory and should be filtered out."
"""
        )
    
    discovered_test_file = find_all_functions_in_file(test_file_path)
    modified_functions_test = {test_file_path: discovered_test_file.get(test_file_path, [])}
    
    filtered_test_file, count_test_file = filter_functions(
        modified_functions_test,
        tests_root=tests_root_dir,
        ignore_paths=[],
        project_root=codeflash_temp_dir,
        module_root=codeflash_temp_dir,
    )
    
    assert not filtered_test_file
    assert count_test_file == 0

    with codeflash_temp_dir.joinpath("ignored_dir").open("w") as f:
        f.write("def ignored_func(): return 1")
    
    ignored_file_path = codeflash_temp_dir.joinpath("ignored_dir")
    discovered_ignored = find_all_functions_in_file(ignored_file_path)
    modified_functions_ignored = {ignored_file_path: discovered_ignored.get(ignored_file_path, [])}

    filtered_ignored, count_ignored = filter_functions(
        modified_functions_ignored,
        tests_root=Path("tests"),
        ignore_paths=[ignored_file_path.parent],
        project_root=file_path.parent,
        module_root=file_path.parent,
    )
    assert not filtered_ignored
    assert count_ignored == 0

    with unittest.mock.patch("codeflash.discovery.functions_to_optimize.ignored_submodule_paths", return_value=[str(codeflash_temp_dir.joinpath("submodule_dir"))]):
        with codeflash_temp_dir.joinpath("submodule_dir").open("w") as f:
            f.write("def submodule_func(): return 1")
        
        submodule_file_path = codeflash_temp_dir.joinpath("submodule_dir")
        discovered_submodule = find_all_functions_in_file(submodule_file_path)
        modified_functions_submodule = {submodule_file_path: discovered_submodule.get(submodule_file_path, [])}
        
        filtered_submodule, count_submodule = filter_functions(
            modified_functions_submodule,
            tests_root=Path("tests"),
            ignore_paths=[],
            project_root=file_path.parent,
            module_root=file_path.parent,
        )
        assert not filtered_submodule
        assert count_submodule == 0

    with unittest.mock.patch("codeflash.discovery.functions_to_optimize.path_belongs_to_site_packages", return_value=True):
        with codeflash_temp_dir.joinpath("site_package_file.py").open("w") as f:
            f.write("def site_package_func(): return 1")

        site_package_file_path = codeflash_temp_dir.joinpath("site_package_file.py")
        discovered_site_package = find_all_functions_in_file(site_package_file_path)
        modified_functions_site_package = {site_package_file_path: discovered_site_package.get(site_package_file_path, [])}

        filtered_site_package, count_site_package = filter_functions(
            modified_functions_site_package,
            tests_root=Path("tests"),
            ignore_paths=[],
            project_root=file_path.parent,
            module_root=file_path.parent,
        )
        assert not filtered_site_package
        assert count_site_package == 0
    
    outside_module_root_path = codeflash_temp_dir.parent.joinpath("outside_module_root_file.py")
    with outside_module_root_path.open("w") as f:
        f.write("def func_outside_module_root(): return 1")
    
    discovered_outside_module = find_all_functions_in_file(outside_module_root_path)
    modified_functions_outside_module = {outside_module_root_path: discovered_outside_module.get(outside_module_root_path, [])}

    filtered_outside_module, count_outside_module = filter_functions(
        modified_functions_outside_module,
        tests_root=Path("tests"),
        ignore_paths=[],
        project_root=file_path.parent,
        module_root=file_path.parent,
    )
    assert not filtered_outside_module
    assert count_outside_module == 0
    os.remove(outside_module_root_path)

    invalid_module_file_path = codeflash_temp_dir.joinpath("invalid-module-name.py")
    with invalid_module_file_path.open("w") as f:
        f.write("def func_in_invalid_module(): return 1")

    discovered_invalid_module = find_all_functions_in_file(invalid_module_file_path)
    modified_functions_invalid_module = {invalid_module_file_path: discovered_invalid_module.get(invalid_module_file_path, [])}
    
    filtered_invalid_module, count_invalid_module = filter_functions(
        modified_functions_invalid_module,
        tests_root=Path("tests"),
        ignore_paths=[],
        project_root=file_path.parent,
        module_root=file_path.parent,
    )
    assert not filtered_invalid_module
    assert count_invalid_module == 0

    with unittest.mock.patch("codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={file_path.name: {"propagate_attributes", "other_blocklisted_function"}}):
        filtered_funcs, count= filter_functions(
            modified_functions,
            tests_root=Path("tests"),
            ignore_paths=[],
            project_root=file_path.parent,
            module_root=file_path.parent,
        )
        assert "propagate_attributes" not in [fn.function_name for fn in filtered_funcs.get(file_path, [])]
        assert count == 2


    module_name = "test_get_functions_to_optimize"
    qualified_name_for_checkpoint = f"{module_name}.propagate_attributes"
    other_qualified_name_for_checkpoint = f"{module_name}.vanilla_function"
    
    with unittest.mock.patch("codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}):
        filtered_checkpoint, count_checkpoint = filter_functions(
            modified_functions,
            tests_root=Path("tests"),
            ignore_paths=[],
            project_root=file_path.parent,
            module_root=file_path.parent,
            previous_checkpoint_functions={qualified_name_for_checkpoint: {"status": "optimized"}, other_qualified_name_for_checkpoint: {}}
        )
        assert filtered_checkpoint.get(file_path)
        assert count_checkpoint == 1

        remaining_functions = [fn.function_name for fn in filtered_checkpoint.get(file_path, [])]
        assert "not_in_checkpoint_function" in remaining_functions
        assert "propagate_attributes" not in remaining_functions
        assert "vanilla_function" not in remaining_functions