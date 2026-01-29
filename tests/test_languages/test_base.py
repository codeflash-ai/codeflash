"""Extensive tests for the language abstraction base types.

These tests verify that the core data structures work correctly
and maintain their contracts.
"""

from pathlib import Path

import pytest

from codeflash.languages.base import (
    CodeContext,
    FunctionFilterCriteria,
    FunctionInfo,
    HelperFunction,
    Language,
    ParentInfo,
    TestInfo,
    TestResult,
    convert_parents_to_tuple,
)


class TestLanguageEnum:
    """Tests for the Language enum."""

    def test_language_values(self):
        """Test that language enum has expected values."""
        assert Language.PYTHON.value == "python"
        assert Language.JAVASCRIPT.value == "javascript"
        assert Language.TYPESCRIPT.value == "typescript"

    def test_language_str(self):
        """Test string conversion of Language enum."""
        assert str(Language.PYTHON) == "python"
        assert str(Language.JAVASCRIPT) == "javascript"

    def test_language_from_string(self):
        """Test creating Language from string."""
        assert Language("python") == Language.PYTHON
        assert Language("javascript") == Language.JAVASCRIPT
        assert Language("typescript") == Language.TYPESCRIPT

    def test_invalid_language_raises(self):
        """Test that invalid language string raises ValueError."""
        with pytest.raises(ValueError):
            Language("invalid_language")


class TestParentInfo:
    """Tests for the ParentInfo dataclass."""

    def test_parent_info_creation(self):
        """Test creating ParentInfo."""
        parent = ParentInfo(name="Calculator", type="ClassDef")
        assert parent.name == "Calculator"
        assert parent.type == "ClassDef"

    def test_parent_info_frozen(self):
        """Test that ParentInfo is immutable."""
        parent = ParentInfo(name="Calculator", type="ClassDef")
        with pytest.raises(AttributeError):
            parent.name = "NewName"

    def test_parent_info_str(self):
        """Test string representation of ParentInfo."""
        parent = ParentInfo(name="Calculator", type="ClassDef")
        assert str(parent) == "ClassDef:Calculator"

    def test_parent_info_equality(self):
        """Test ParentInfo equality."""
        p1 = ParentInfo(name="Calculator", type="ClassDef")
        p2 = ParentInfo(name="Calculator", type="ClassDef")
        p3 = ParentInfo(name="Other", type="ClassDef")

        assert p1 == p2
        assert p1 != p3

    def test_parent_info_hash(self):
        """Test that ParentInfo is hashable."""
        p1 = ParentInfo(name="Calculator", type="ClassDef")
        p2 = ParentInfo(name="Calculator", type="ClassDef")

        # Should be able to use in sets/dicts
        s = {p1, p2}
        assert len(s) == 1


class TestFunctionInfo:
    """Tests for the FunctionInfo dataclass."""

    def test_function_info_creation_minimal(self):
        """Test creating FunctionInfo with minimal args."""
        func = FunctionInfo(name="add", file_path=Path("/test/example.py"), start_line=1, end_line=3)
        assert func.name == "add"
        assert func.file_path == Path("/test/example.py")
        assert func.start_line == 1
        assert func.end_line == 3
        assert func.parents == ()
        assert func.is_async is False
        assert func.is_method is False
        assert func.language == Language.PYTHON

    def test_function_info_creation_full(self):
        """Test creating FunctionInfo with all args."""
        parents = (ParentInfo(name="Calculator", type="ClassDef"),)
        func = FunctionInfo(
            name="add",
            file_path=Path("/test/example.py"),
            start_line=10,
            end_line=15,
            parents=parents,
            is_async=True,
            is_method=True,
            language=Language.PYTHON,
            start_col=4,
            end_col=20,
        )
        assert func.name == "add"
        assert func.parents == parents
        assert func.is_async is True
        assert func.is_method is True
        assert func.start_col == 4
        assert func.end_col == 20

    def test_function_info_frozen(self):
        """Test that FunctionInfo is immutable."""
        func = FunctionInfo(name="add", file_path=Path("/test/example.py"), start_line=1, end_line=3)
        with pytest.raises(AttributeError):
            func.name = "new_name"

    def test_qualified_name_no_parents(self):
        """Test qualified_name without parents."""
        func = FunctionInfo(name="add", file_path=Path("/test/example.py"), start_line=1, end_line=3)
        assert func.qualified_name == "add"

    def test_qualified_name_with_class(self):
        """Test qualified_name with class parent."""
        func = FunctionInfo(
            name="add",
            file_path=Path("/test/example.py"),
            start_line=1,
            end_line=3,
            parents=(ParentInfo(name="Calculator", type="ClassDef"),),
        )
        assert func.qualified_name == "Calculator.add"

    def test_qualified_name_nested(self):
        """Test qualified_name with nested parents."""
        func = FunctionInfo(
            name="inner",
            file_path=Path("/test/example.py"),
            start_line=1,
            end_line=3,
            parents=(ParentInfo(name="Outer", type="ClassDef"), ParentInfo(name="Inner", type="ClassDef")),
        )
        assert func.qualified_name == "Outer.Inner.inner"

    def test_class_name_with_class(self):
        """Test class_name property with class parent."""
        func = FunctionInfo(
            name="add",
            file_path=Path("/test/example.py"),
            start_line=1,
            end_line=3,
            parents=(ParentInfo(name="Calculator", type="ClassDef"),),
        )
        assert func.class_name == "Calculator"

    def test_class_name_without_class(self):
        """Test class_name property without class parent."""
        func = FunctionInfo(name="add", file_path=Path("/test/example.py"), start_line=1, end_line=3)
        assert func.class_name is None

    def test_class_name_nested_function(self):
        """Test class_name for function nested in another function."""
        func = FunctionInfo(
            name="inner",
            file_path=Path("/test/example.py"),
            start_line=1,
            end_line=3,
            parents=(ParentInfo(name="outer", type="FunctionDef"),),
        )
        assert func.class_name is None

    def test_class_name_method_in_nested_class(self):
        """Test class_name for method in nested class."""
        func = FunctionInfo(
            name="method",
            file_path=Path("/test/example.py"),
            start_line=1,
            end_line=3,
            parents=(ParentInfo(name="Outer", type="ClassDef"), ParentInfo(name="Inner", type="ClassDef")),
        )
        # Should return the immediate parent class
        assert func.class_name == "Inner"

    def test_top_level_parent_name_no_parents(self):
        """Test top_level_parent_name without parents."""
        func = FunctionInfo(name="add", file_path=Path("/test/example.py"), start_line=1, end_line=3)
        assert func.top_level_parent_name == "add"

    def test_top_level_parent_name_with_parents(self):
        """Test top_level_parent_name with parents."""
        func = FunctionInfo(
            name="method",
            file_path=Path("/test/example.py"),
            start_line=1,
            end_line=3,
            parents=(ParentInfo(name="Outer", type="ClassDef"), ParentInfo(name="Inner", type="ClassDef")),
        )
        assert func.top_level_parent_name == "Outer"

    def test_function_info_str(self):
        """Test string representation."""
        func = FunctionInfo(
            name="add",
            file_path=Path("/test/example.py"),
            start_line=1,
            end_line=3,
            parents=(ParentInfo(name="Calculator", type="ClassDef"),),
        )
        s = str(func)
        assert "Calculator.add" in s
        assert "example.py" in s
        assert "1-3" in s


class TestHelperFunction:
    """Tests for the HelperFunction dataclass."""

    def test_helper_function_creation(self):
        """Test creating HelperFunction."""
        helper = HelperFunction(
            name="multiply",
            qualified_name="Calculator.multiply",
            file_path=Path("/test/helpers.py"),
            source_code="def multiply(a, b): return a * b",
            start_line=10,
            end_line=12,
        )
        assert helper.name == "multiply"
        assert helper.qualified_name == "Calculator.multiply"
        assert helper.file_path == Path("/test/helpers.py")
        assert "return a * b" in helper.source_code


class TestCodeContext:
    """Tests for the CodeContext dataclass."""

    def test_code_context_creation_minimal(self):
        """Test creating CodeContext with minimal args."""
        ctx = CodeContext(target_code="def add(a, b): return a + b", target_file=Path("/test/example.py"))
        assert ctx.target_code == "def add(a, b): return a + b"
        assert ctx.target_file == Path("/test/example.py")
        assert ctx.helper_functions == []
        assert ctx.read_only_context == ""
        assert ctx.imports == []
        assert ctx.language == Language.PYTHON

    def test_code_context_creation_full(self):
        """Test creating CodeContext with all args."""
        helper = HelperFunction(
            name="multiply",
            qualified_name="multiply",
            file_path=Path("/test/helpers.py"),
            source_code="def multiply(a, b): return a * b",
            start_line=1,
            end_line=2,
        )
        ctx = CodeContext(
            target_code="def add(a, b): return a + b",
            target_file=Path("/test/example.py"),
            helper_functions=[helper],
            read_only_context="# Constants\nMAX_VALUE = 100",
            imports=["import math", "from typing import List"],
            language=Language.JAVASCRIPT,
        )
        assert len(ctx.helper_functions) == 1
        assert ctx.read_only_context == "# Constants\nMAX_VALUE = 100"
        assert len(ctx.imports) == 2
        assert ctx.language == Language.JAVASCRIPT


class TestTestInfo:
    """Tests for the TestInfo dataclass."""

    def test_test_info_creation(self):
        """Test creating TestInfo."""
        info = TestInfo(test_name="test_add", test_file=Path("/tests/test_calc.py"), test_class="TestCalculator")
        assert info.test_name == "test_add"
        assert info.test_file == Path("/tests/test_calc.py")
        assert info.test_class == "TestCalculator"

    def test_test_info_without_class(self):
        """Test TestInfo without test class."""
        info = TestInfo(test_name="test_add", test_file=Path("/tests/test_calc.py"))
        assert info.test_class is None

    def test_full_test_path_with_class(self):
        """Test full_test_path with class."""
        info = TestInfo(test_name="test_add", test_file=Path("/tests/test_calc.py"), test_class="TestCalculator")
        assert info.full_test_path == "/tests/test_calc.py::TestCalculator::test_add"

    def test_full_test_path_without_class(self):
        """Test full_test_path without class."""
        info = TestInfo(test_name="test_add", test_file=Path("/tests/test_calc.py"))
        assert info.full_test_path == "/tests/test_calc.py::test_add"


class TestTestResult:
    """Tests for the TestResult dataclass."""

    def test_test_result_passed(self):
        """Test TestResult for passing test."""
        result = TestResult(
            test_name="test_add",
            test_file=Path("/tests/test_calc.py"),
            passed=True,
            runtime_ns=1000000,  # 1ms
        )
        assert result.passed is True
        assert result.runtime_ns == 1000000
        assert result.error_message is None

    def test_test_result_failed(self):
        """Test TestResult for failing test."""
        result = TestResult(
            test_name="test_add",
            test_file=Path("/tests/test_calc.py"),
            passed=False,
            error_message="AssertionError: 1 != 2",
        )
        assert result.passed is False
        assert result.error_message == "AssertionError: 1 != 2"

    def test_test_result_with_output(self):
        """Test TestResult with stdout/stderr."""
        result = TestResult(
            test_name="test_add",
            test_file=Path("/tests/test_calc.py"),
            passed=True,
            stdout="Debug: calculating...",
            stderr="Warning: deprecated",
        )
        assert result.stdout == "Debug: calculating..."
        assert result.stderr == "Warning: deprecated"


class TestFunctionFilterCriteria:
    """Tests for the FunctionFilterCriteria dataclass."""

    def test_default_criteria(self):
        """Test default filter criteria."""
        criteria = FunctionFilterCriteria()
        assert criteria.require_return is True
        assert criteria.include_async is True
        assert criteria.include_methods is True
        assert criteria.include_patterns == []
        assert criteria.exclude_patterns == []
        assert criteria.min_lines is None
        assert criteria.max_lines is None

    def test_custom_criteria(self):
        """Test custom filter criteria."""
        criteria = FunctionFilterCriteria(
            include_patterns=["process_*", "handle_*"],
            exclude_patterns=["_private_*"],
            require_return=False,
            include_async=False,
            include_methods=False,
            min_lines=3,
            max_lines=50,
        )
        assert criteria.include_patterns == ["process_*", "handle_*"]
        assert criteria.exclude_patterns == ["_private_*"]
        assert criteria.require_return is False
        assert criteria.include_async is False
        assert criteria.min_lines == 3
        assert criteria.max_lines == 50


class TestConvertParentsToTuple:
    """Tests for the convert_parents_to_tuple helper function."""

    def test_empty_parents(self):
        """Test conversion of empty list."""
        result = convert_parents_to_tuple([])
        assert result == ()

    def test_convert_from_list(self):
        """Test conversion from list of parent-like objects."""

        class MockParent:
            def __init__(self, name: str, type_: str):
                self.name = name
                self.type = type_

        parents = [MockParent("Outer", "ClassDef"), MockParent("inner", "FunctionDef")]
        result = convert_parents_to_tuple(parents)

        assert len(result) == 2
        assert result[0].name == "Outer"
        assert result[0].type == "ClassDef"
        assert result[1].name == "inner"
        assert result[1].type == "FunctionDef"

    def test_convert_from_tuple(self):
        """Test conversion from tuple (should work the same)."""

        class MockParent:
            def __init__(self, name: str, type_: str):
                self.name = name
                self.type = type_

        parents = (MockParent("Calculator", "ClassDef"),)
        result = convert_parents_to_tuple(parents)

        assert len(result) == 1
        assert result[0].name == "Calculator"
