import ast
import site
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.code_utils.code_utils import (
    cleanup_paths,
    file_name_from_test_module_name,
    file_path_from_module_name,
    get_all_function_names,
    get_imports_from_file,
    get_qualified_name,
    get_run_tmp_file,
    is_class_defined_in_file,
    module_name_from_file_path,
    path_belongs_to_site_packages,
    validate_python_code,
)
from codeflash.code_utils.concolic_utils import clean_concolic_tests
from codeflash.code_utils.coverage_utils import extract_dependent_function, generate_candidates, prepare_coverage_files
from codeflash.models.models import CodeStringsMarkdown
from codeflash.verification.parse_test_output import resolve_test_file_from_class_path


@pytest.fixture
def multiple_existing_and_non_existing_files(tmp_path: Path) -> list[Path]:
    existing_files = [tmp_path / f"existing_file{i}.txt" for i in range(3)]
    non_existing_files = [tmp_path / f"non_existing_file{i}.txt" for i in range(2)]
    for file in existing_files:
        file.touch()
    return existing_files + non_existing_files


@pytest.fixture
def mock_get_run_tmp_file() -> Generator[MagicMock, None, None]:
    with patch("codeflash.code_utils.coverage_utils.get_run_tmp_file") as mock:
        yield mock


def test_get_qualified_name_valid() -> None:
    module_name = "codeflash"
    full_qualified_name = "codeflash.utils.module"

    result = get_qualified_name(module_name, full_qualified_name)
    assert result == "utils.module"


def test_get_qualified_name_invalid_prefix() -> None:
    module_name = "codeflash"
    full_qualified_name = "otherflash.utils.module"
    with pytest.raises(ValueError, match="does not start with codeflash"):
        get_qualified_name(module_name, full_qualified_name)


def test_get_qualified_name_same_name() -> None:
    module_name = "codeflash"
    full_qualified_name = "codeflash"
    with pytest.raises(ValueError, match="is the same as codeflash"):
        get_qualified_name(module_name, full_qualified_name)


# tests for module_name_from_file_path
def test_module_name_from_file_path() -> None:
    project_root_path = Path("/Users/codeflashuser/PycharmProjects/codeflash")
    file_path = project_root_path / "cli/codeflash/code_utils/code_utils.py"

    module_name = module_name_from_file_path(file_path, project_root_path)
    assert module_name == "cli.codeflash.code_utils.code_utils"


def test_module_name_from_file_path_with_subdirectory() -> None:
    project_root_path = Path("/Users/codeflashuser/PycharmProjects/codeflash")
    file_path = project_root_path / "cli/codeflash/code_utils/subdir/code_utils.py"

    module_name = module_name_from_file_path(file_path, project_root_path)
    assert module_name == "cli.codeflash.code_utils.subdir.code_utils"


def test_module_name_from_file_path_with_different_root() -> None:
    project_root_path = Path("/Users/codeflashuser/PycharmProjects")
    file_path = project_root_path / "codeflash/cli/codeflash/code_utils/code_utils.py"

    module_name = module_name_from_file_path(file_path, project_root_path)
    assert module_name == "codeflash.cli.codeflash.code_utils.code_utils"


def test_module_name_from_file_path_with_root_as_file() -> None:
    project_root_path = Path("/Users/codeflashuser/PycharmProjects/codeflash/cli/codeflash/code_utils")
    file_path = project_root_path / "code_utils.py"

    module_name = module_name_from_file_path(file_path, project_root_path)
    assert module_name == "code_utils"


def test_get_imports_from_file_with_file_path(tmp_path: Path) -> None:
    test_file = tmp_path / "test_file.py"
    test_file.write_text("import os\nfrom sys import path\n")

    imports = get_imports_from_file(file_path=test_file)
    assert len(imports) == 2
    assert isinstance(imports[0], ast.Import)
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[0].names[0].name == "os"
    assert imports[1].module == "sys"
    assert imports[1].names[0].name == "path"


def test_get_imports_from_file_with_file_string() -> None:
    file_string = "import os\nfrom sys import path\n"

    imports = get_imports_from_file(file_string=file_string)
    assert len(imports) == 2
    assert isinstance(imports[0], ast.Import)
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[0].names[0].name == "os"
    assert imports[1].module == "sys"
    assert imports[1].names[0].name == "path"


def test_get_imports_from_file_with_file_ast() -> None:
    file_string = "import os\nfrom sys import path\n"
    file_ast = ast.parse(file_string)

    imports = get_imports_from_file(file_ast=file_ast)
    assert len(imports) == 2
    assert isinstance(imports[0], ast.Import)
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[0].names[0].name == "os"
    assert imports[1].module == "sys"
    assert imports[1].names[0].name == "path"


def test_get_imports_from_file_with_syntax_error(caplog: pytest.LogCaptureFixture) -> None:
    file_string = "import os\nfrom sys import path\ninvalid syntax"

    imports = get_imports_from_file(file_string=file_string)
    assert len(imports) == 0
    assert "Syntax error in code" in caplog.text


def test_get_imports_from_file_with_no_input() -> None:
    with pytest.raises(AssertionError, match="Must provide exactly one of file_path, file_string, or file_ast"):
        get_imports_from_file()


# tests for file_path_from_module_name
def test_file_path_from_module_name() -> None:
    project_root_path = Path("/Users/codeflashuser/PycharmProjects/codeflash")
    module_name = "cli.codeflash.code_utils.code_utils"

    file_path = file_path_from_module_name(module_name, project_root_path)
    assert file_path == project_root_path / "cli/codeflash/code_utils/code_utils.py"


def test_file_path_from_module_name_with_subdirectory() -> None:
    project_root_path = Path("/Users/codeflashuser/PycharmProjects/codeflash")
    module_name = "cli.codeflash.code_utils.subdir.code_utils"

    file_path = file_path_from_module_name(module_name, project_root_path)
    assert file_path == project_root_path / "cli/codeflash/code_utils/subdir/code_utils.py"


def test_file_path_from_module_name_with_different_root() -> None:
    project_root_path = Path("/Users/codeflashuser/PycharmProjects")
    module_name = "codeflash.cli.codeflash.code_utils.code_utils"

    file_path = file_path_from_module_name(module_name, project_root_path)
    assert file_path == project_root_path / "codeflash/cli/codeflash/code_utils/code_utils.py"


def test_file_path_from_module_name_with_root_as_file() -> None:
    project_root_path = Path("/Users/codeflashuser/PycharmProjects/codeflash/cli/codeflash/code_utils")
    module_name = "code_utils"

    file_path = file_path_from_module_name(module_name, project_root_path)
    assert file_path == project_root_path / "code_utils.py"


# tests for get_all_function_names
def test_get_all_function_names_with_valid_code() -> None:
    code = """
def foo():
    pass

async def bar():
    pass
"""
    success, function_names = get_all_function_names(code)
    assert success is True
    assert function_names == ["foo", "bar"]


def test_get_all_function_names_with_syntax_error(caplog: pytest.LogCaptureFixture) -> None:
    code = """
def foo():
    pass

async def bar():
    pass

invalid syntax
"""
    success, function_names = get_all_function_names(code)
    assert success is False
    assert function_names == []
    assert "Syntax error in code" in caplog.text


def test_get_all_function_names_with_no_functions() -> None:
    code = """
x = 1
y = 2
"""
    success, function_names = get_all_function_names(code)
    assert success is True
    assert function_names == []


def test_get_all_function_names_with_nested_functions() -> None:
    code = """
def outer():
    def inner():
        pass
    return inner
"""
    success, function_names = get_all_function_names(code)
    assert success is True
    assert function_names == ["outer", "inner"]


# tests for get_run_tmp_file
def test_get_run_tmp_file_creates_temp_directory() -> None:
    file_path = Path("test_file.py")
    tmp_file_path = get_run_tmp_file(file_path)

    assert tmp_file_path.name == "test_file.py"
    assert tmp_file_path.parent.name.startswith("codeflash_")
    assert tmp_file_path.parent.exists()


def test_get_run_tmp_file_reuses_temp_directory() -> None:
    file_path1 = Path("test_file1.py")
    file_path2 = Path("test_file2.py")

    tmp_file_path1 = get_run_tmp_file(file_path1)
    tmp_file_path2 = get_run_tmp_file(file_path2)

    assert tmp_file_path1.parent == tmp_file_path2.parent
    assert tmp_file_path1.name == "test_file1.py"
    assert tmp_file_path2.name == "test_file2.py"
    assert tmp_file_path1.parent.name.startswith("codeflash_")
    assert tmp_file_path1.parent.exists()


def test_path_belongs_to_site_packages_with_site_package_path(monkeypatch: pytest.MonkeyPatch) -> None:
    site_packages = [Path("/usr/local/lib/python3.9/site-packages").resolve()]
    monkeypatch.setattr(site, "getsitepackages", lambda: site_packages)

    file_path = Path("/usr/local/lib/python3.9/site-packages/some_package")
    assert path_belongs_to_site_packages(file_path) is True


def test_path_belongs_to_site_packages_with_non_site_package_path(monkeypatch: pytest.MonkeyPatch) -> None:
    site_packages = [Path("/usr/local/lib/python3.9/site-packages")]
    monkeypatch.setattr(site, "getsitepackages", lambda: site_packages)

    file_path = Path("/usr/local/lib/python3.9/other_directory/some_package")
    assert path_belongs_to_site_packages(file_path) is False


def test_path_belongs_to_site_packages_with_relative_path(monkeypatch: pytest.MonkeyPatch) -> None:
    site_packages = [Path("/usr/local/lib/python3.9/site-packages")]
    monkeypatch.setattr(site, "getsitepackages", lambda: site_packages)

    file_path = Path("some_package")
    assert path_belongs_to_site_packages(file_path) is False


def test_path_belongs_to_site_packages_with_symlinked_site_packages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_site_packages = tmp_path / "real_site_packages"
    real_site_packages.mkdir()
    
    symlinked_site_packages = tmp_path / "symlinked_site_packages"
    symlinked_site_packages.symlink_to(real_site_packages)
    
    package_file = real_site_packages / "some_package" / "__init__.py"
    package_file.parent.mkdir()
    package_file.write_text("# package file")
    
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(symlinked_site_packages)])
    
    assert path_belongs_to_site_packages(package_file) is True
    
    symlinked_package_file = symlinked_site_packages / "some_package" / "__init__.py"
    assert path_belongs_to_site_packages(symlinked_package_file) is True


def test_path_belongs_to_site_packages_with_complex_symlinks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_site_packages = tmp_path / "real" / "lib" / "python3.9" / "site-packages"
    real_site_packages.mkdir(parents=True)
    
    link1 = tmp_path / "link1"
    link1.symlink_to(real_site_packages.parent.parent.parent)
    
    link2 = tmp_path / "link2" 
    link2.symlink_to(link1)
    
    package_file = real_site_packages / "test_package" / "module.py"
    package_file.parent.mkdir()
    package_file.write_text("# test module")
    
    site_packages_via_links = link2 / "lib" / "python3.9" / "site-packages"
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages_via_links)])
    
    assert path_belongs_to_site_packages(package_file) is True
    
    file_via_links = site_packages_via_links / "test_package" / "module.py"
    assert path_belongs_to_site_packages(file_via_links) is True


def test_path_belongs_to_site_packages_resolved_paths_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    site_packages_dir = tmp_path / "lib" / "python3.9" / "site-packages"
    site_packages_dir.mkdir(parents=True)
    
    package_dir = site_packages_dir / "mypackage"
    package_dir.mkdir()
    package_file = package_dir / "module.py"
    package_file.write_text("# module")
    
    complex_site_packages_path = tmp_path / "lib" / "python3.9" / "other" / ".." / "site-packages" / "."
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(complex_site_packages_path)])
    
    assert path_belongs_to_site_packages(package_file) is True
    
    complex_file_path = tmp_path / "lib" / "python3.9" / "site-packages" / "other" / ".." / "mypackage" / "module.py"
    assert path_belongs_to_site_packages(complex_file_path) is True


# tests for is_class_defined_in_file
def test_is_class_defined_in_file_with_existing_class(tmp_path: Path) -> None:
    test_file = tmp_path / "test_file.py"
    test_file.write_text("""
class MyClass:
    pass
""")

    assert is_class_defined_in_file("MyClass", test_file) is True


def test_is_class_defined_in_file_with_non_existing_class(tmp_path: Path) -> None:
    test_file = tmp_path / "test_file.py"
    test_file.write_text("""
class MyClass:
    pass
""")

    assert is_class_defined_in_file("OtherClass", test_file) is False


def test_is_class_defined_in_file_with_no_classes(tmp_path: Path) -> None:
    test_file = tmp_path / "test_file.py"
    test_file.write_text("""
def my_function():
    pass
""")

    assert is_class_defined_in_file("MyClass", test_file) is False


@pytest.fixture
def mock_code_context():
    """Mock CodeOptimizationContext for testing extract_dependent_function."""
    from unittest.mock import MagicMock
    from codeflash.models.models import CodeOptimizationContext
    
    context = MagicMock(spec=CodeOptimizationContext)
    context.preexisting_objects = []
    return context


def test_extract_dependent_function_sync_and_async(mock_code_context):
    """Test extract_dependent_function with both sync and async functions."""
    # Test sync function extraction
    mock_code_context.testgen_context = CodeStringsMarkdown.parse_markdown_code("""```python:file.py
def main_function():
    pass

def helper_function():
    pass
```
""")
    assert extract_dependent_function("main_function", mock_code_context) == "helper_function"
    
    # Test async function extraction
    mock_code_context.testgen_context = CodeStringsMarkdown.parse_markdown_code("""```python:file.py
def main_function():
    pass

async def async_helper_function():
    pass
```
""")

    assert extract_dependent_function("main_function", mock_code_context) == "async_helper_function"


def test_extract_dependent_function_edge_cases(mock_code_context):
    """Test extract_dependent_function edge cases."""
    # No dependent functions
    mock_code_context.testgen_context = CodeStringsMarkdown.parse_markdown_code("""```python:file.py
def main_function():
    pass
```
""")
    assert extract_dependent_function("main_function", mock_code_context) is False
    
    # Multiple dependent functions
    mock_code_context.testgen_context = CodeStringsMarkdown.parse_markdown_code("""```python:file.py
def main_function():
    pass
def helper1():
    pass

async def helper2():
    pass
```
""")
    assert extract_dependent_function("main_function", mock_code_context) is False


def test_extract_dependent_function_mixed_scenarios(mock_code_context):
    """Test extract_dependent_function with mixed sync/async scenarios."""
    # Async main with sync helper
    mock_code_context.testgen_context = CodeStringsMarkdown.parse_markdown_code("""```python:file.py
async def async_main():
    pass

def sync_helper():
    pass
```
""")
    assert extract_dependent_function("async_main", mock_code_context) == "sync_helper"
    
    # Only async functions
    mock_code_context.testgen_context = CodeStringsMarkdown.parse_markdown_code("""```python:file.py
async def async_main():
    pass

async def async_helper():
    pass
```
""")

    assert extract_dependent_function("async_main", mock_code_context) == "async_helper"


def test_is_class_defined_in_file_with_non_existing_file() -> None:
    non_existing_file = Path("/non/existing/file.py")

    assert is_class_defined_in_file("MyClass", non_existing_file) is False


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    base_dir = tmp_path / "project"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "test_module.py").touch()
    (base_dir / "subdir").mkdir(exist_ok=True)
    (base_dir / "subdir" / "test_submodule.py").touch()
    return base_dir


def test_existing_module(base_dir: Path) -> None:
    result = file_name_from_test_module_name("test_module", base_dir)
    assert result == base_dir / "test_module.py"


def test_existing_submodule(base_dir: Path) -> None:
    result = file_name_from_test_module_name("subdir.test_submodule", base_dir)
    assert result == base_dir / "subdir" / "test_submodule.py"


def test_non_existing_module(base_dir: Path) -> None:
    result = file_name_from_test_module_name("non_existing_module", base_dir)
    assert result is None


def test_partial_module_name(base_dir: Path) -> None:
    result = file_name_from_test_module_name("subdir.test_submodule.TestClass", base_dir)
    assert result == base_dir / "subdir" / "test_submodule.py"


def test_partial_module_name2(base_dir: Path) -> None:
    result = file_name_from_test_module_name("subdir.test_submodule.TestClass.TestClass2", base_dir)
    assert result == base_dir / "subdir" / "test_submodule.py"


def test_pytest_unittest_path_resolution_with_prefix(tmp_path: Path) -> None:
    """Test path resolution when pytest includes parent directory in classname.
    
    This handles the case where pytest's base_dir is /path/to/tests but the
    classname includes the parent directory like "project.tests.unittest.test_file.TestClass".
    """
    # Setup directory structure: /tmp/code_to_optimize/tests/unittest/
    project_root = tmp_path / "code_to_optimize"
    tests_root = project_root / "tests"
    unittest_dir = tests_root / "unittest"
    unittest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test files
    test_file = unittest_dir / "test_bubble_sort.py"
    test_file.touch()
    
    generated_test = unittest_dir / "test_sorter__unit_test_0.py"
    generated_test.touch()
    
    # Case 1: pytest reports classname with full path including "code_to_optimize.tests"
    # but base_dir is .../tests (not the project root)
    result = resolve_test_file_from_class_path(
        "code_to_optimize.tests.unittest.test_bubble_sort.TestPigLatin",
        tests_root
    )
    assert result == test_file
    
    # Case 2: Generated test file with class name
    result = resolve_test_file_from_class_path(
        "code_to_optimize.tests.unittest.test_sorter__unit_test_0.TestSorter",
        tests_root
    )
    assert result == generated_test
    
    # Case 3: Without the class name (just the module path)
    result = resolve_test_file_from_class_path(
        "code_to_optimize.tests.unittest.test_bubble_sort",
        tests_root
    )
    assert result == test_file


def test_pytest_unittest_multiple_prefix_levels(tmp_path: Path) -> None:
    """Test path resolution with multiple levels of prefix stripping."""
    # Setup: /tmp/org/project/src/tests/unit/
    base = tmp_path / "org" / "project" / "src" / "tests"
    unit_dir = base / "unit"
    unit_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = unit_dir / "test_example.py"
    test_file.touch()
    
    # pytest might report: org.project.src.tests.unit.test_example.TestClass
    # with base_dir being .../src/tests or .../tests
    result = resolve_test_file_from_class_path(
        "org.project.src.tests.unit.test_example.TestClass",
        base
    )
    assert result == test_file
    
    # Also test with base_dir at different level
    result = resolve_test_file_from_class_path(
        "project.src.tests.unit.test_example.TestClass",
        base
    )
    assert result == test_file


def test_pytest_unittest_instrumented_files(tmp_path: Path) -> None:
    """Test path resolution for instrumented test files."""
    tests_root = tmp_path / "tests" / "unittest"
    tests_root.mkdir(parents=True, exist_ok=True)
    
    # Create instrumented test file
    instrumented_file = tests_root / "test_bubble_sort__perfinstrumented.py"
    instrumented_file.touch()
    
    # pytest classname includes parent directories
    result = resolve_test_file_from_class_path(
        "code_to_optimize.tests.unittest.test_bubble_sort__perfinstrumented.TestPigLatin",
        tmp_path / "tests"
    )
    assert result == instrumented_file


def test_pytest_unittest_nested_classes(tmp_path: Path) -> None:
    """Test path resolution with nested class names."""
    tests_root = tmp_path / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    
    test_file = tests_root / "test_nested.py"
    test_file.touch()
    
    # Some unittest frameworks use nested classes
    result = resolve_test_file_from_class_path(
        "project.tests.test_nested.OuterClass.InnerClass",
        tests_root
    )
    assert result == test_file


def test_pytest_unittest_no_match_returns_none(tmp_path: Path) -> None:
    """Test that non-existent files return None even with prefix stripping."""
    tests_root = tmp_path / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    
    # File doesn't exist
    result = resolve_test_file_from_class_path(
        "code_to_optimize.tests.unittest.nonexistent_test.TestClass",
        tests_root
    )
    assert result is None


def test_pytest_unittest_single_component(tmp_path: Path) -> None:
    """Test that single-component paths still work."""
    base_dir = tmp_path
    test_file = base_dir / "test_simple.py"
    test_file.touch()
    
    result = file_name_from_test_module_name("test_simple", base_dir)
    assert result == test_file
    
    # With class name
    result = file_name_from_test_module_name("test_simple.TestClass", base_dir)
    assert result == test_file


def test_cleanup_paths(multiple_existing_and_non_existing_files: list[Path]) -> None:
    cleanup_paths(multiple_existing_and_non_existing_files)
    for file in multiple_existing_and_non_existing_files:
        assert not file.exists()


def test_generate_candidates() -> None:
    source_code_path = Path("/Users/krrt7/Desktop/work/codeflash/cli/codeflash/code_utils/coverage_utils.py")
    expected_candidates = {
        "coverage_utils.py",
        "code_utils/coverage_utils.py",
        "codeflash/code_utils/coverage_utils.py",
        "cli/codeflash/code_utils/coverage_utils.py",
        "codeflash/cli/codeflash/code_utils/coverage_utils.py",
        "work/codeflash/cli/codeflash/code_utils/coverage_utils.py",
        "Desktop/work/codeflash/cli/codeflash/code_utils/coverage_utils.py",
        "krrt7/Desktop/work/codeflash/cli/codeflash/code_utils/coverage_utils.py",
        "Users/krrt7/Desktop/work/codeflash/cli/codeflash/code_utils/coverage_utils.py",
        "/Users/krrt7/Desktop/work/codeflash/cli/codeflash/code_utils/coverage_utils.py"
    }
    assert generate_candidates(source_code_path) == expected_candidates


def test_prepare_coverage_files(mock_get_run_tmp_file: MagicMock) -> None:
    mock_coverage_file = MagicMock(spec=Path)
    mock_coveragerc_file = MagicMock(spec=Path)
    mock_get_run_tmp_file.side_effect = [mock_coverage_file, mock_coveragerc_file]

    coverage_database_file, coveragercfile = prepare_coverage_files()
    assert coverage_database_file == mock_coverage_file
    assert coveragercfile == mock_coveragerc_file
    mock_coveragerc_file.write_text.assert_called_once_with(f"[run]\n branch = True\ndata_file={mock_coverage_file}\n")


def test_clean_concolic_tests() -> None:
    original_code = """
def test_add_numbers(x: int, y: int) -> None:
    assert add_numbers(1, 2) == 3


def test_concatenate_strings(s1: str, s2: str) -> None:
    assert concatenate_strings("hello", "world") == "helloworld"


def test_append_to_list(my_list: list[int], element: int) -> None:
    assert append_to_list([1, 2, 3], 4) == [1, 2, 3, 4]


def test_get_dict_value(my_dict: dict[str, int], key: str) -> None:
    assert get_dict_value({"a": 1, "b": 2}, "a") == 1


def test_union_sets(set1: set[int], set2: set[int]) -> None:
    assert union_sets({1, 2, 3}, {3, 4, 5}) == {1, 2, 3, 4, 5}

def test_calculate_tuple_sum(my_tuple: tuple[int, int, int]) -> None:
    assert calculate_tuple_sum((1, 2, 3)) == 6
"""

    cleaned_code = clean_concolic_tests(original_code)
    expected_cleaned_code = """
def test_add_numbers(x: int, y: int) -> None:
    add_numbers(1, 2)

def test_concatenate_strings(s1: str, s2: str) -> None:
    concatenate_strings('hello', 'world')

def test_append_to_list(my_list: list[int], element: int) -> None:
    append_to_list([1, 2, 3], 4)

def test_get_dict_value(my_dict: dict[str, int], key: str) -> None:
    get_dict_value({'a': 1, 'b': 2}, 'a')

def test_union_sets(set1: set[int], set2: set[int]) -> None:
    union_sets({1, 2, 3}, {3, 4, 5})

def test_calculate_tuple_sum(my_tuple: tuple[int, int, int]) -> None:
    calculate_tuple_sum((1, 2, 3))
"""
    assert cleaned_code == expected_cleaned_code.strip()

    concolic_generated_repr_code = """from src.blib2to3.pgen2.grammar import Grammar

def test_Grammar_copy():
    assert Grammar.copy(Grammar()) == <src.blib2to3.pgen2.grammar.Grammar object at 0x104c30f50>
"""
    cleaned_code = clean_concolic_tests(concolic_generated_repr_code)
    expected_cleaned_code = """
from src.blib2to3.pgen2.grammar import Grammar

def test_Grammar_copy():
    Grammar.copy(Grammar())
"""
    assert cleaned_code == expected_cleaned_code.strip()


def test_validate_python_code_valid() -> None:
    code = "def hello():\n    return 'world'"
    result = validate_python_code(code)
    assert result == code


def test_validate_python_code_invalid() -> None:
    code = "def hello(:\n    return 'world'"
    with pytest.raises(ValueError, match="Invalid Python code"):
        validate_python_code(code)


def test_validate_python_code_empty() -> None:
    code = ""
    result = validate_python_code(code)
    assert result == code


def test_validate_python_code_complex_invalid() -> None:
    code = "if True\n    print('missing colon')"
    with pytest.raises(ValueError, match="Invalid Python code.*line 1.*column 8"):
        validate_python_code(code)


def test_validate_python_code_valid_complex() -> None:
    code = """
def calculate(a, b):
    if a > b:
        return a + b
    else:
        return a * b
        
class MyClass:
    def __init__(self):
        self.value = 42
"""
    result = validate_python_code(code)
    assert result == code
