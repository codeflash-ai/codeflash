import tempfile
from unittest import mock
from unittest.mock import patch

from codeflash.discovery.functions_to_optimize import (
    find_all_functions_in_file,
    FunctionToOptimize,
    filter_functions,
)


def test_function_eligible_for_optimization():
    function = """def test_function_eligible_for_optimization():
    a = 5
    return a**2
    """
    functions_found = {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(function)
        f.flush()
        functions_found = find_all_functions_in_file(f.name)
    assert "test_function_eligible_for_optimization" == functions_found[f.name][0].function_name

    # Has no return statement
    function = """def test_function_not_eligible_for_optimization():
    a = 5
    print(a)
    """
    functions_found = {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(function)
        f.flush()
        functions_found = find_all_functions_in_file(f.name)
    assert len(functions_found[f.name]) == 0


@patch("cli.codeflash.discovery.functions_to_optimize.git.Repo")
def test_filter_functions(mock_git_repo):
    # Mock the git.Repo call to raise an exception, simulating a non-git repo
    # Mock the git.Repo call to simulate a valid git repo for the path "/user/projects/nuitka"
    mock_git_repo.side_effect = lambda *args, **kwargs: (
        mock.MagicMock(
            head=mock.MagicMock(
                commit=mock.MagicMock(tree=mock.MagicMock(join=mock.MagicMock(return_value=None)))
            ),
            working_dir=args[0],
        )
        if args[0] == "/user/projects/nuitka"
        else mock.DEFAULT
    )

    functions = {
        "/user/projects/nuitka/build/inline_copy/lib/scons/hi/SCons/Utilities/sconsign.py": [
            FunctionToOptimize(
                "function_name",
                "/user/projects/nuitka/build/inline_copy/lib/scons/hi/SCons/Utilities/sconsign.py",
                [],
                None,
                None,
            )
        ]
    }

    with patch(
        "cli.codeflash.discovery.functions_to_optimize.is_not_git_module_file"
    ) as mock_is_not_git_module_file:
        mock_is_not_git_module_file.return_value = False
        _, functions_count = filter_functions(
            functions, "/user/projects/nuitka/tests", [], "/user/projects", "/user/projects/nuitka"
        )
    assert functions_count == 1

    functions = {
        "/user/projects/nuitka/build/inline_copy/lib/scons/4.3.0/SCons/Utilities/sconsign.py": [
            FunctionToOptimize(
                "function_name",
                "/user/projects/nuitka/build/inline_copy/lib/scons/4.3.0/SCons/Utilities/sconsign.py",
                [],
                None,
                None,
            )
        ]
    }

    with patch(
        "cli.codeflash.discovery.functions_to_optimize.is_not_git_module_file"
    ) as mock_is_not_git_module_file:
        mock_is_not_git_module_file.return_value = True
        _, functions_count = filter_functions(
            functions, "/user/projects/nuitka/tests", [], "/user/projects", "/user/projects/nuitka"
        )
    assert functions_count == 0
