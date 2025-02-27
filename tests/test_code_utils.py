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
)
from codeflash.code_utils.concolic_utils import clean_concolic_tests
from codeflash.code_utils.coverage_utils import generate_candidates, prepare_coverage_files


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
    site_packages = [Path("/usr/local/lib/python3.9/site-packages")]
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


def test_cleanup_paths(multiple_existing_and_non_existing_files: list[Path]) -> None:
    cleanup_paths(multiple_existing_and_non_existing_files)
    for file in multiple_existing_and_non_existing_files:
        assert not file.exists()


def test_generate_candidates() -> None:
    source_code_path = Path("/Users/krrt7/Desktop/work/codeflash/cli/codeflash/code_utils/coverage_utils.py")
    expected_candidates = [
        "coverage_utils.py",
        "code_utils/coverage_utils.py",
        "codeflash/code_utils/coverage_utils.py",
        "cli/codeflash/code_utils/coverage_utils.py",
        "codeflash/cli/codeflash/code_utils/coverage_utils.py",
        "work/codeflash/cli/codeflash/code_utils/coverage_utils.py",
        "Desktop/work/codeflash/cli/codeflash/code_utils/coverage_utils.py",
        "krrt7/Desktop/work/codeflash/cli/codeflash/code_utils/coverage_utils.py",
        "Users/krrt7/Desktop/work/codeflash/cli/codeflash/code_utils/coverage_utils.py",
    ]
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
