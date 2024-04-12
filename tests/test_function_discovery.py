import tempfile

from codeflash.discovery.functions_to_optimize import (
    find_all_functions_in_file,
    is_function_or_method_top_level,
)


def test_function_eligible_for_optimization() -> None:
    function = """def test_function_eligible_for_optimization():
    a = 5
    return a**2
    """
    functions_found = {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(function)
        f.flush()
        functions_found = find_all_functions_in_file(f.name)
    assert functions_found[f.name][0].function_name == "test_function_eligible_for_optimization"

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

    """,
        )
        f.flush()
        assert is_function_or_method_top_level(f.name, "functionA")
        assert not is_function_or_method_top_level(f.name, "functionB")
        assert is_function_or_method_top_level(f.name, "functionC", class_name="A")
        assert not is_function_or_method_top_level(f.name, "functionD", class_name="A")
        assert not is_function_or_method_top_level(f.name, "functionF", class_name="E")
