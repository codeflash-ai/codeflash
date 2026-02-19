from __future__ import annotations

import pytest

from codeflash.languages.python.static_analysis.concolic_utils import AssertCleanup, is_valid_concolic_test


class TestFirstTopLevelArg:
    @pytest.fixture
    def cleanup(self) -> AssertCleanup:
        return AssertCleanup()

    def test_single_argument_no_comma(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("foo") == "foo"

    def test_single_argument_with_whitespace(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("  foo  ") == "foo"

    def test_two_simple_arguments(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("a, b") == "a"

    def test_multiple_arguments(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("x, y, z") == "x"

    def test_nested_parentheses_comma_ignored(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("func(a, b), c") == "func(a, b)"

    def test_nested_brackets_comma_ignored(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("[1, 2], x") == "[1, 2]"

    def test_nested_braces_comma_ignored(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("{a: b, c: d}, e") == "{a: b, c: d}"

    def test_deeply_nested_parentheses(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("f(g(h(i))), j") == "f(g(h(i)))"

    def test_mixed_bracket_types(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("func([{a, b}], c), d") == "func([{a, b}], c)"

    def test_empty_string(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("") == ""

    def test_only_whitespace(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("   ") == ""

    def test_comma_at_start(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg(", a") == ""

    def test_no_top_level_comma(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("func(a, b)") == "func(a, b)"

    def test_empty_parentheses(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("(), b") == "()"

    def test_empty_brackets(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("[], b") == "[]"

    def test_empty_braces(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("{}, b") == "{}"

    def test_whitespace_around_comma(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("a  ,  b") == "a"

    def test_complex_nested_structure(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("{'key': [1, (2, 3)]}, other") == "{'key': [1, (2, 3)]}"

    def test_string_literal_with_comma(self, cleanup: AssertCleanup) -> None:
        # Note: this function doesn't handle string literals specially
        # commas inside strings are treated as top-level
        assert cleanup._first_top_level_arg('"a,b", c') == '"a'

    def test_unbalanced_opening_bracket(self, cleanup: AssertCleanup) -> None:
        # With unbalanced opening, no top-level comma found
        assert cleanup._first_top_level_arg("(a, b") == "(a, b"

    def test_multiple_consecutive_commas(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg(",,") == ""

    def test_attribute_access(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("obj.attr, b") == "obj.attr"

    def test_numeric_arguments(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("123, 456") == "123"

    def test_negative_number(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("-42, x") == "-42"

    def test_float_argument(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("3.14, x") == "3.14"

    def test_newline_in_argument(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("a\nb, c") == "a\nb"

    def test_tab_whitespace(self, cleanup: AssertCleanup) -> None:
        assert cleanup._first_top_level_arg("\ta\t, b") == "a"


class TestIsValidConcolicTest:
    def test_valid_test_passes_collection(self) -> None:
        valid_test = """
import pytest

def test_simple():
    assert 1 + 1 == 2
"""
        assert is_valid_concolic_test(valid_test) is True

    def test_invalid_test_with_side_effect_detected(self) -> None:
        invalid_test = """
from typing import TextIO
from some_module import some_function
import pytest

def test_some_function():
    with pytest.raises(SideEffectDetected, match="We've blocked a file writing operation"):
        some_function(filename="", file=TextIO())
"""
        assert is_valid_concolic_test(invalid_test) is False

    def test_syntax_error_is_invalid(self) -> None:
        invalid_syntax = "def test_broken( assert True"
        assert is_valid_concolic_test(invalid_syntax) is False

    def test_empty_string_fails_collection(self) -> None:
        assert is_valid_concolic_test("") is False

    def test_import_only_fails_collection(self) -> None:
        import_only = "from some_module import some_function"
        assert is_valid_concolic_test(import_only) is False

    def test_valid_import_but_no_tests_fails_collection(self) -> None:
        valid_import_no_tests = "import pytest\nimport os"
        assert is_valid_concolic_test(valid_import_no_tests) is False
