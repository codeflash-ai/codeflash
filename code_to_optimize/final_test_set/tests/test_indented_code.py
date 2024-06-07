from code_to_optimize.final_test_set.indented_code import indentedCode


def test_basic_indentation():
    codes = ["def foo():", "    return 42"]
    indented = indentedCode(codes, 4)
    expected = "    def foo():\n        return 42"
    assert indented == expected, "Basic indentation failed"


def test_zero_indentation():
    codes = ["print('hello')", "print('world')"]
    indented = indentedCode(codes, 0)
    expected = "print('hello')\nprint('world')"
    assert indented == expected, "Zero indentation should leave text unchanged"


def test_empty_string_lines():
    codes = ["", "print('hello')", ""]
    indented = indentedCode(codes, 2)
    expected = "\n  print('hello')\n"
    assert indented == expected, "Empty lines should be handled correctly"


def test_no_lines():
    codes = []
    indented = indentedCode(codes, 4)
    assert indented == "", "Empty code list should return an empty string"


def test_large_indentation():
    codes = ["if True:", "    pass"]
    indented = indentedCode(codes, 8)
    expected = "        if True:\n            pass"
    assert indented == expected, "Large indentation failed"


def test_mixed_content():
    codes = ["", "def test():", "    assert True", ""]
    indented = indentedCode(codes, 1)
    expected = "\n def test():\n     assert True\n"
    assert indented == expected, "Mixed content with empty lines failed"
