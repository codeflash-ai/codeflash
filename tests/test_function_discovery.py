import tempfile

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


def test_filter_functions():
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

    _, functions_count = filter_functions(
        functions,
        "/user/projects/nuitka/tests",
        [],
        "/user/projects",
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

    _, functions_count = filter_functions(
        functions,
        "/user/projects/nuitka/tests",
        [],
        "/user/projects",
    )
    assert functions_count == 0
