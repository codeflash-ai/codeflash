import tempfile
import unittest.mock
from pathlib import Path

from codeflash.discovery.functions_to_optimize import (
    filter_files_optimized,
    filter_functions,
    find_all_functions_in_file,
    get_all_files_and_functions,
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
        temp_dir_path = Path(temp_dir)
        file_path = temp_dir_path / "test_function.py"

        with file_path.open("w") as f:
            f.write(function)

        functions_found = find_all_functions_in_file(file_path)
    assert functions_found[file_path][0].function_name == "test_function_eligible_for_optimization"

    # Has no return statement
    function = """def test_function_not_eligible_for_optimization():
    a = 5
    print(a)
    """
    functions_found = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        file_path = temp_dir_path / "test_function.py"

        with file_path.open("w") as f:
            f.write(function)

        functions_found = find_all_functions_in_file(file_path)
    assert len(functions_found[file_path]) == 0

    # we want to trigger an error in the function discovery
    function = """def test_invalid_code():"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        file_path = temp_dir_path / "test_function.py"

        with file_path.open("w") as f:
            f.write(function)

        functions_found = find_all_functions_in_file(file_path)
    assert functions_found == {}


def test_find_top_level_function_or_method():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        file_path = temp_dir_path / "test_function.py"

        with file_path.open("w") as f:
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

        assert inspect_top_level_functions_or_methods(file_path, "functionA").is_top_level
        assert not inspect_top_level_functions_or_methods(file_path, "functionB").is_top_level
        assert inspect_top_level_functions_or_methods(file_path, "functionC", class_name="A").is_top_level
        assert not inspect_top_level_functions_or_methods(file_path, "functionD", class_name="A").is_top_level
        assert not inspect_top_level_functions_or_methods(file_path, "functionF", class_name="E").is_top_level
        assert not inspect_top_level_functions_or_methods(file_path, "functionA").has_args
        staticmethod_func = inspect_top_level_functions_or_methods(
            file_path, "handle_record_counts", class_name=None, line_no=15
        )
        assert staticmethod_func.is_staticmethod
        assert staticmethod_func.staticmethod_class_name == "AirbyteEntrypoint"
        assert inspect_top_level_functions_or_methods(
            file_path, "functionE", class_name="AirbyteEntrypoint"
        ).is_classmethod
        assert not inspect_top_level_functions_or_methods(
            file_path, "non_classmethod_function", class_name="AirbyteEntrypoint"
        ).is_top_level
        # needed because this will be traced with a class_name being passed

    # we want to write invalid code to ensure that the function discovery does not crash
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        file_path = temp_dir_path / "test_function.py"

        with file_path.open("w") as f:
            f.write(
                """def functionA():
"""
            )

        assert not inspect_top_level_functions_or_methods(file_path, "functionA")


def test_class_method_discovery():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        file_path = temp_dir_path / "test_function.py"

        with file_path.open("w") as f:
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

        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )
        functions, functions_count, _ = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=file_path,
            only_get_this_function="A.functionA",
            test_cfg=test_config,
            ignore_paths=[Path("/bruh/")],
            project_root=file_path.parent,
            module_root=file_path.parent,
        )
        assert len(functions) == 1
        for file in functions:
            assert functions[file][0].qualified_name == "A.functionA"
            assert functions[file][0].function_name == "functionA"
            assert functions[file][0].top_level_parent_name == "A"

        functions, functions_count, _ = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=file_path,
            only_get_this_function="X.functionA",
            test_cfg=test_config,
            ignore_paths=[Path("/bruh/")],
            project_root=file_path.parent,
            module_root=file_path.parent,
        )
        assert len(functions) == 1
        for file in functions:
            assert functions[file][0].qualified_name == "X.functionA"
            assert functions[file][0].function_name == "functionA"
            assert functions[file][0].top_level_parent_name == "X"

        functions, functions_count, _ = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=file_path,
            only_get_this_function="functionA",
            test_cfg=test_config,
            ignore_paths=[Path("/bruh/")],
            project_root=file_path.parent,
            module_root=file_path.parent,
        )
        assert len(functions) == 1
        for file in functions:
            assert functions[file][0].qualified_name == "functionA"
            assert functions[file][0].function_name == "functionA"


def test_nested_function():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        file_path = temp_dir_path / "test_function.py"

        with file_path.open("w") as f:
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

        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )
        functions, functions_count, _ = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=file_path,
            test_cfg=test_config,
            only_get_this_function=None,
            ignore_paths=[Path("/bruh/")],
            project_root=file_path.parent,
            module_root=file_path.parent,
        )

        assert len(functions) == 1
        assert functions_count == 1

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        file_path = temp_dir_path / "test_function.py"

        with file_path.open("w") as f:
            f.write(
                """
def outer_function():
    def inner_function():
        pass

    return inner_function
"""
            )

        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )
        functions, functions_count, _ = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=file_path,
            test_cfg=test_config,
            only_get_this_function=None,
            ignore_paths=[Path("/bruh/")],
            project_root=file_path.parent,
            module_root=file_path.parent,
        )

        assert len(functions) == 1
        assert functions_count == 1

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        file_path = temp_dir_path / "test_function.py"

        with file_path.open("w") as f:
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

        test_config = TestConfig(
            tests_root="tests", project_root_path=".", test_framework="pytest", tests_project_rootdir=Path()
        )
        functions, functions_count, _ = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=file_path,
            test_cfg=test_config,
            only_get_this_function=None,
            ignore_paths=[Path("/bruh/")],
            project_root=file_path.parent,
            module_root=file_path.parent,
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
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Create a test file in the temporary directory
        test_file_path = temp_dir.joinpath("test_get_functions_to_optimize.py")
        with test_file_path.open("w") as f:
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

        discovered = find_all_functions_in_file(test_file_path)
        modified_functions = {test_file_path: discovered[test_file_path]}
        # Use an absolute path for tests_root that won't match the temp directory
        # This avoids path resolution issues in CI where the working directory might differ
        tests_root_absolute = (temp_dir.parent / "nonexistent_tests_dir").resolve()
        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                modified_functions,
                tests_root=tests_root_absolute,
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )
        function_names = [fn.function_name for fn in filtered.get(test_file_path, [])]
        assert "propagate_attributes" in function_names
        assert count == 3

        # Create a tests directory inside our temp directory
        tests_root_dir = temp_dir.joinpath("tests")
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
            project_root=temp_dir,
            module_root=temp_dir,
        )

        assert not filtered_test_file
        assert count_test_file == 0

        # Test ignored directory
        ignored_dir = temp_dir.joinpath("ignored_dir")
        ignored_dir.mkdir(exist_ok=True)
        ignored_file_path = ignored_dir.joinpath("ignored_file.py")
        with ignored_file_path.open("w") as f:
            f.write("def ignored_func(): return 1")

        discovered_ignored = find_all_functions_in_file(ignored_file_path)
        modified_functions_ignored = {ignored_file_path: discovered_ignored.get(ignored_file_path, [])}

        filtered_ignored, count_ignored = filter_functions(
            modified_functions_ignored,
            tests_root=Path("tests"),
            ignore_paths=[ignored_dir],
            project_root=temp_dir,
            module_root=temp_dir,
        )
        assert not filtered_ignored
        assert count_ignored == 0

        # Test submodule paths
        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.ignored_submodule_paths",
            return_value=[str(temp_dir.joinpath("submodule_dir"))],
        ):
            submodule_dir = temp_dir.joinpath("submodule_dir")
            submodule_dir.mkdir(exist_ok=True)
            submodule_file_path = submodule_dir.joinpath("submodule_file.py")
            with submodule_file_path.open("w") as f:
                f.write("def submodule_func(): return 1")

            discovered_submodule = find_all_functions_in_file(submodule_file_path)
            modified_functions_submodule = {submodule_file_path: discovered_submodule.get(submodule_file_path, [])}

            filtered_submodule, count_submodule = filter_functions(
                modified_functions_submodule,
                tests_root=Path("tests"),
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )
            assert not filtered_submodule
            assert count_submodule == 0

        # Test site packages
        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.path_belongs_to_site_packages", return_value=True
        ):
            site_package_file_path = temp_dir.joinpath("site_package_file.py")
            with site_package_file_path.open("w") as f:
                f.write("def site_package_func(): return 1")

            discovered_site_package = find_all_functions_in_file(site_package_file_path)
            modified_functions_site_package = {
                site_package_file_path: discovered_site_package.get(site_package_file_path, [])
            }

            filtered_site_package, count_site_package = filter_functions(
                modified_functions_site_package,
                tests_root=Path("tests"),
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )
            assert not filtered_site_package
            assert count_site_package == 0

        # Test outside module root
        parent_dir = temp_dir.parent
        outside_module_root_path = parent_dir.joinpath("outside_module_root_file.py")
        try:
            with outside_module_root_path.open("w") as f:
                f.write("def func_outside_module_root(): return 1")

            discovered_outside_module = find_all_functions_in_file(outside_module_root_path)
            modified_functions_outside_module = {
                outside_module_root_path: discovered_outside_module.get(outside_module_root_path, [])
            }

            filtered_outside_module, count_outside_module = filter_functions(
                modified_functions_outside_module,
                tests_root=Path("tests"),
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )
            assert not filtered_outside_module
            assert count_outside_module == 0
        finally:
            outside_module_root_path.unlink(missing_ok=True)

        # Test invalid module name
        invalid_module_file_path = temp_dir.joinpath("invalid-module-name.py")
        with invalid_module_file_path.open("w") as f:
            f.write("def func_in_invalid_module(): return 1")

        discovered_invalid_module = find_all_functions_in_file(invalid_module_file_path)
        modified_functions_invalid_module = {
            invalid_module_file_path: discovered_invalid_module.get(invalid_module_file_path, [])
        }

        filtered_invalid_module, count_invalid_module = filter_functions(
            modified_functions_invalid_module,
            tests_root=Path("tests"),
            ignore_paths=[],
            project_root=temp_dir,
            module_root=temp_dir,
        )
        assert not filtered_invalid_module
        assert count_invalid_module == 0

        original_file_path = temp_dir.joinpath("test_get_functions_to_optimize.py")
        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions",
            return_value={original_file_path.name: {"propagate_attributes", "other_blocklisted_function"}},
        ):
            filtered_funcs, count = filter_functions(
                modified_functions,
                tests_root=Path("tests"),
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )
            assert "propagate_attributes" not in [fn.function_name for fn in filtered_funcs.get(original_file_path, [])]
            assert count == 2

        module_name = "test_get_functions_to_optimize"
        qualified_name_for_checkpoint = f"{module_name}.propagate_attributes"
        other_qualified_name_for_checkpoint = f"{module_name}.vanilla_function"

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered_checkpoint, count_checkpoint = filter_functions(
                modified_functions,
                tests_root=Path("tests"),
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
                previous_checkpoint_functions={
                    qualified_name_for_checkpoint: {"status": "optimized"},
                    other_qualified_name_for_checkpoint: {},
                },
            )
            assert filtered_checkpoint.get(original_file_path)
            assert count_checkpoint == 1

            remaining_functions = [fn.function_name for fn in filtered_checkpoint.get(original_file_path, [])]
            assert "not_in_checkpoint_function" in remaining_functions
            assert "propagate_attributes" not in remaining_functions
            assert "vanilla_function" not in remaining_functions
        files_and_funcs = get_all_files_and_functions(module_root_path=temp_dir, ignore_paths=[])
        assert len(files_and_funcs) == 6


def test_filter_functions_tests_root_overlaps_source():
    """Test that source files are not filtered when tests_root equals module_root or project_root.

    This is a critical test for monorepo structures where tests live alongside source code
    (e.g., TypeScript projects with .test.ts files in the same directories as source).
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Create a source file (NOT a test file)
        source_file = temp_dir / "utils.py"
        with source_file.open("w") as f:
            f.write("""
def process_data(items):
    return [item * 2 for item in items]

def calculate_sum(numbers):
    return sum(numbers)
""")

        # Create a test file with standard naming pattern
        test_file = temp_dir / "utils.test.py"
        with test_file.open("w") as f:
            f.write("""
def test_process_data():
    return "test"
""")

        # Create a test file with _test suffix pattern
        test_file_underscore = temp_dir / "utils_test.py"
        with test_file_underscore.open("w") as f:
            f.write("""
def test_calculate_sum():
    return "test"
""")

        # Create a spec file
        spec_file = temp_dir / "utils.spec.py"
        with spec_file.open("w") as f:
            f.write("""
def spec_function():
    return "spec"
""")

        # Create a file in a tests subdirectory
        tests_subdir = temp_dir / "tests"
        tests_subdir.mkdir()
        tests_subdir_file = tests_subdir / "test_main.py"
        with tests_subdir_file.open("w") as f:
            f.write("""
def test_in_tests_dir():
    return "test"
""")

        # Create a file in __tests__ subdirectory (common in JS/TS projects)
        dunder_tests_subdir = temp_dir / "__tests__"
        dunder_tests_subdir.mkdir()
        dunder_tests_file = dunder_tests_subdir / "main.py"
        with dunder_tests_file.open("w") as f:
            f.write("""
def test_in_dunder_tests():
    return "test"
""")

        # Discover all functions
        discovered_source = find_all_functions_in_file(source_file)
        discovered_test = find_all_functions_in_file(test_file)
        discovered_test_underscore = find_all_functions_in_file(test_file_underscore)
        discovered_spec = find_all_functions_in_file(spec_file)
        discovered_tests_dir = find_all_functions_in_file(tests_subdir_file)
        discovered_dunder_tests = find_all_functions_in_file(dunder_tests_file)

        # Combine all discovered functions
        all_functions = {}
        for discovered in [discovered_source, discovered_test, discovered_test_underscore,
                          discovered_spec, discovered_tests_dir, discovered_dunder_tests]:
            all_functions.update(discovered)

        # Test Case 1: tests_root == module_root (overlapping case)
        # This is the bug scenario where all functions were being filtered
        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                all_functions,
                tests_root=temp_dir,  # Same as module_root
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,  # Same as tests_root
            )

        # Strict check: only source_file should remain in filtered results
        assert set(filtered.keys()) == {source_file}, (
            f"Expected only source file in filtered results, got: {set(filtered.keys())}"
        )

        # Strict check: exactly these two functions should be present
        source_functions = sorted([fn.function_name for fn in filtered.get(source_file, [])])
        assert source_functions == ["calculate_sum", "process_data"], (
            f"Expected ['calculate_sum', 'process_data'], got {source_functions}"
        )

        # Strict check: exactly 2 functions remaining
        assert count == 2, f"Expected exactly 2 functions, got {count}"

        # Test Case 2: tests_root == project_root (another overlapping case)
        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered2, count2 = filter_functions(
                {source_file: discovered_source[source_file]},
                tests_root=temp_dir,  # Same as project_root
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )

        # Strict check: only source_file should remain
        assert set(filtered2.keys()) == {source_file}, (
            f"Expected only source file when tests_root == project_root, got: {set(filtered2.keys())}"
        )
        assert count2 == 2, f"Expected exactly 2 functions, got {count2}"


def test_filter_functions_strict_string_matching():
    """Test that test file pattern matching uses strict string matching.

    Ensures patterns like '.test.' only match actual test files and don't
    accidentally match files with similar names like 'contest.py' or 'latest.py'.
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Files that should NOT be filtered (contain 'test' as substring but not as pattern)
        contest_file = temp_dir / "contest.py"
        with contest_file.open("w") as f:
            f.write("def run_contest(): return 1")

        latest_file = temp_dir / "latest.py"
        with latest_file.open("w") as f:
            f.write("def get_latest(): return 1")

        attestation_file = temp_dir / "attestation.py"
        with attestation_file.open("w") as f:
            f.write("def verify_attestation(): return 1")

        # File that SHOULD be filtered (matches .test. pattern)
        actual_test_file = temp_dir / "utils.test.py"
        with actual_test_file.open("w") as f:
            f.write("def test_utils(): return 1")

        # File that SHOULD be filtered (matches _test. pattern)
        underscore_test_file = temp_dir / "utils_test.py"
        with underscore_test_file.open("w") as f:
            f.write("def test_stuff(): return 1")

        # Discover all functions
        all_functions = {}
        for file_path in [contest_file, latest_file, attestation_file, actual_test_file, underscore_test_file]:
            discovered = find_all_functions_in_file(file_path)
            all_functions.update(discovered)

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                all_functions,
                tests_root=temp_dir,  # Overlapping case to trigger pattern matching
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )

        # Strict check: exactly these 3 files should remain (those with 'test' as substring only)
        expected_files = {contest_file, latest_file, attestation_file}
        assert set(filtered.keys()) == expected_files, (
            f"Expected files {expected_files}, got {set(filtered.keys())}"
        )

        # Strict check: each file should have exactly 1 function with the expected name
        assert [fn.function_name for fn in filtered[contest_file]] == ["run_contest"], (
            f"Expected ['run_contest'], got {[fn.function_name for fn in filtered[contest_file]]}"
        )
        assert [fn.function_name for fn in filtered[latest_file]] == ["get_latest"], (
            f"Expected ['get_latest'], got {[fn.function_name for fn in filtered[latest_file]]}"
        )
        assert [fn.function_name for fn in filtered[attestation_file]] == ["verify_attestation"], (
            f"Expected ['verify_attestation'], got {[fn.function_name for fn in filtered[attestation_file]]}"
        )

        # Strict check: exactly 3 functions remaining
        assert count == 3, f"Expected exactly 3 functions, got {count}"


def test_filter_functions_test_directory_patterns():
    """Test that test directory patterns work correctly with strict matching.

    Ensures that /test/, /tests/, and /__tests__/ patterns only match actual
    test directories and not directories that happen to contain 'test' in name.
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Directory that should NOT be filtered (contains 'test' but not as /test/ pattern)
        contest_dir = temp_dir / "contest_results"
        contest_dir.mkdir()
        contest_file = contest_dir / "scores.py"
        with contest_file.open("w") as f:
            f.write("def get_scores(): return [1, 2, 3]")

        latest_dir = temp_dir / "latest_data"
        latest_dir.mkdir()
        latest_file = latest_dir / "data.py"
        with latest_file.open("w") as f:
            f.write("def load_data(): return {}")

        # Directory that SHOULD be filtered (matches /tests/ pattern)
        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        tests_file = tests_dir / "test_main.py"
        with tests_file.open("w") as f:
            f.write("def test_main(): return True")

        # Directory that SHOULD be filtered (matches /test/ pattern - singular)
        test_dir = temp_dir / "test"
        test_dir.mkdir()
        test_file = test_dir / "test_utils.py"
        with test_file.open("w") as f:
            f.write("def test_utils(): return True")

        # Directory that SHOULD be filtered (matches /__tests__/ pattern)
        dunder_tests_dir = temp_dir / "__tests__"
        dunder_tests_dir.mkdir()
        dunder_file = dunder_tests_dir / "component.py"
        with dunder_file.open("w") as f:
            f.write("def test_component(): return True")

        # Nested test directory
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        nested_tests_dir = src_dir / "tests"
        nested_tests_dir.mkdir()
        nested_test_file = nested_tests_dir / "test_nested.py"
        with nested_test_file.open("w") as f:
            f.write("def test_nested(): return True")

        # Discover all functions
        all_functions = {}
        for file_path in [contest_file, latest_file, tests_file, test_file, dunder_file, nested_test_file]:
            discovered = find_all_functions_in_file(file_path)
            all_functions.update(discovered)

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                all_functions,
                tests_root=temp_dir,  # Overlapping case
                ignore_paths=[],
                project_root=temp_dir,
                module_root=temp_dir,
            )

        # Strict check: exactly these 2 files should remain (those in non-test directories)
        expected_files = {contest_file, latest_file}
        assert set(filtered.keys()) == expected_files, (
            f"Expected files {expected_files}, got {set(filtered.keys())}"
        )

        # Strict check: each file should have exactly 1 function with the expected name
        assert [fn.function_name for fn in filtered[contest_file]] == ["get_scores"], (
            f"Expected ['get_scores'], got {[fn.function_name for fn in filtered[contest_file]]}"
        )
        assert [fn.function_name for fn in filtered[latest_file]] == ["load_data"], (
            f"Expected ['load_data'], got {[fn.function_name for fn in filtered[latest_file]]}"
        )

        # Strict check: exactly 2 functions remaining
        assert count == 2, f"Expected exactly 2 functions, got {count}"


def test_filter_functions_non_overlapping_tests_root():
    """Test that the original directory-based filtering still works when tests_root is separate.

    When tests_root is a distinct directory (e.g., 'tests/'), the original behavior
    of filtering files that start with tests_root should still work.
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Create source directory structure
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        source_file = src_dir / "utils.py"
        with source_file.open("w") as f:
            f.write("def process(): return 1")

        # Create a file with .test. pattern in source (should NOT be filtered in non-overlapping mode)
        # because directory-based filtering takes precedence
        test_in_src = src_dir / "helper.test.py"
        with test_in_src.open("w") as f:
            f.write("def helper_test(): return 1")

        # Create separate tests directory
        tests_dir = temp_dir / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_utils.py"
        with test_file.open("w") as f:
            f.write("def test_process(): return 1")

        # Discover functions
        all_functions = {}
        for file_path in [source_file, test_in_src, test_file]:
            discovered = find_all_functions_in_file(file_path)
            all_functions.update(discovered)

        # Non-overlapping case: tests_root is a separate directory
        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                all_functions,
                tests_root=tests_dir,  # Separate from module_root
                ignore_paths=[],
                project_root=temp_dir,
                module_root=src_dir,  # Different from tests_root
            )

        # Strict check: exactly these 2 files should remain (both in src/, not in tests/)
        expected_files = {source_file, test_in_src}
        assert set(filtered.keys()) == expected_files, (
            f"Expected files {expected_files}, got {set(filtered.keys())}"
        )

        # Strict check: each file should have exactly 1 function with the expected name
        assert [fn.function_name for fn in filtered[source_file]] == ["process"], (
            f"Expected ['process'], got {[fn.function_name for fn in filtered[source_file]]}"
        )
        assert [fn.function_name for fn in filtered[test_in_src]] == ["helper_test"], (
            f"Expected ['helper_test'], got {[fn.function_name for fn in filtered[test_in_src]]}"
        )

        # Strict check: exactly 2 functions remaining
        assert count == 2, f"Expected exactly 2 functions, got {count}"
