import ast
from pathlib import Path

import pytest

from codeflash.code_utils.code_utils import get_imports_from_file, module_name_from_file_path


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


# def test_get_imports_from_file_with_file_path(tmp_path: Path):
#     test_file = tmp_path / "test_file.py"
#     test_file.write_text("import os\nfrom sys import path\n")

#     imports = get_imports_from_file(file_path=test_file)
#     assert len(imports) == 2
#     assert isinstance(imports[0], ast.Import)
#     assert isinstance(imports[1], ast.ImportFrom)
#     assert imports[0].names[0].name == "os"
#     assert imports[1].module == "sys"
#     assert imports[1].names[0].name == "path"


# def test_get_imports_from_file_with_file_string():
#     file_string = "import os\nfrom sys import path\n"

#     imports = get_imports_from_file(file_string=file_string)
#     assert len(imports) == 2
#     assert isinstance(imports[0], ast.Import)
#     assert isinstance(imports[1], ast.ImportFrom)
#     assert imports[0].names[0].name == "os"
#     assert imports[1].module == "sys"
#     assert imports[1].names[0].name == "path"


# def test_get_imports_from_file_with_file_ast():
#     file_string = "import os\nfrom sys import path\n"
#     file_ast = ast.parse(file_string)

#     imports = get_imports_from_file(file_ast=file_ast)
#     assert len(imports) == 2
#     assert isinstance(imports[0], ast.Import)
#     assert isinstance(imports[1], ast.ImportFrom)
#     assert imports[0].names[0].name == "os"
#     assert imports[1].module == "sys"
#     assert imports[1].names[0].name == "path"


# def test_get_imports_from_file_with_syntax_error(caplog):
#     file_string = "import os\nfrom sys import path\ninvalid syntax"

#     imports = get_imports_from_file(file_string=file_string)
#     assert len(imports) == 0
#     assert "Syntax error in code" in caplog.text


# def test_get_imports_from_file_with_no_input():
#     with pytest.raises(
#         AssertionError, match="Must provide exactly one of file_path, file_string, or file_ast"
#     ):
#         get_imports_from_file()
