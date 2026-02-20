from __future__ import annotations

from typing import Any

from codeflash.code_utils.coverage_utils import build_fully_qualified_name, extract_dependent_function
from codeflash.models.function_types import FunctionParent
from codeflash.models.models import CodeOptimizationContext, CodeString, CodeStringsMarkdown
from codeflash.verification.coverage_utils import CoverageUtils


def _make_code_context(
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]],
    testgen_code_strings: list[CodeString] | None = None,
) -> CodeOptimizationContext:
    """Helper to create a minimal CodeOptimizationContext for testing."""
    return CodeOptimizationContext(
        testgen_context=CodeStringsMarkdown(code_strings=testgen_code_strings or []),
        read_writable_code=CodeStringsMarkdown(),
        helper_functions=[],
        preexisting_objects=preexisting_objects,
    )


class TestBuildFullyQualifiedName:
    def test_bare_name_with_class_parent(self) -> None:
        ctx = _make_code_context({("__init__", (FunctionParent(name="HttpInterface", type="ClassDef"),))})
        assert build_fully_qualified_name("__init__", ctx) == "HttpInterface.__init__"

    def test_bare_name_no_parent(self) -> None:
        ctx = _make_code_context({("helper_func", ())})
        assert build_fully_qualified_name("helper_func", ctx) == "helper_func"

    def test_already_qualified_name_returned_as_is(self) -> None:
        """If name already contains a dot, skip preexisting_objects lookup."""
        ctx = _make_code_context({("__init__", (FunctionParent(name="WrongClass", type="ClassDef"),))})
        result = build_fully_qualified_name("HttpInterface.__init__", ctx)
        assert result == "HttpInterface.__init__"

    def test_bare_name_picks_first_match_from_set(self) -> None:
        """With multiple __init__ entries, bare name picks an arbitrary one."""
        ctx = _make_code_context(
            {
                ("__init__", (FunctionParent(name="ClassA", type="ClassDef"),)),
                ("__init__", (FunctionParent(name="ClassB", type="ClassDef"),)),
            }
        )
        result = build_fully_qualified_name("__init__", ctx)
        assert result in {"ClassA.__init__", "ClassB.__init__"}

    def test_qualified_name_avoids_ambiguity(self) -> None:
        """Qualified name bypasses preexisting_objects entirely, avoiding ambiguity."""
        ctx = _make_code_context(
            {
                ("__init__", (FunctionParent(name="ClassA", type="ClassDef"),)),
                ("__init__", (FunctionParent(name="ClassB", type="ClassDef"),)),
            }
        )
        assert build_fully_qualified_name("ClassB.__init__", ctx) == "ClassB.__init__"

    def test_bare_name_not_in_preexisting_objects(self) -> None:
        ctx = _make_code_context(set())
        assert build_fully_qualified_name("some_func", ctx) == "some_func"

    def test_nested_class_parent(self) -> None:
        """Bare name under nested class parents gets fully qualified."""
        ctx = _make_code_context(
            {("method", (FunctionParent(name="Outer", type="ClassDef"), FunctionParent(name="Inner", type="ClassDef")))}
        )
        assert build_fully_qualified_name("method", ctx) == "Inner.Outer.method"

    def test_non_classdef_parent_ignored(self) -> None:
        """Only ClassDef parents are prepended to the name."""
        ctx = _make_code_context({("helper", (FunctionParent(name="wrapper", type="FunctionDef"),))})
        assert build_fully_qualified_name("helper", ctx) == "helper"


class TestExtractDependentFunction:
    def test_single_dependent_function(self) -> None:
        ctx = _make_code_context(
            preexisting_objects={("helper", ())},
            testgen_code_strings=[CodeString(code="def main_func(): pass\ndef helper(): pass")],
        )
        result = extract_dependent_function("main_func", ctx)
        assert result == "helper"

    def test_qualified_main_function_discards_bare_match(self) -> None:
        """Qualified main_function should still discard the matching bare name."""
        ctx = _make_code_context(
            preexisting_objects={("helper", ())},
            testgen_code_strings=[CodeString(code="def __init__(): pass\ndef helper(): pass")],
        )
        result = extract_dependent_function("HttpInterface.__init__", ctx)
        assert result == "helper"

    def test_bare_main_function_discards_match(self) -> None:
        """Bare main_function should still work for discarding."""
        ctx = _make_code_context(
            preexisting_objects={("helper", ())},
            testgen_code_strings=[CodeString(code="def main_func(): pass\ndef helper(): pass")],
        )
        result = extract_dependent_function("main_func", ctx)
        assert result == "helper"

    def test_no_dependent_functions(self) -> None:
        ctx = _make_code_context(preexisting_objects=set(), testgen_code_strings=[CodeString(code="x = 1\n")])
        result = extract_dependent_function("main_func", ctx)
        assert result is False

    def test_multiple_dependent_functions_returns_false(self) -> None:
        ctx = _make_code_context(
            preexisting_objects=set(),
            testgen_code_strings=[CodeString(code="def helper_a(): pass\ndef helper_b(): pass")],
        )
        result = extract_dependent_function("main_func", ctx)
        assert result is False

    def test_dependent_function_gets_qualified(self) -> None:
        """The dependent function returned should be qualified via build_fully_qualified_name."""
        ctx = _make_code_context(
            preexisting_objects={("helper", (FunctionParent(name="MyClass", type="ClassDef"),))},
            testgen_code_strings=[CodeString(code="def main_func(): pass\ndef helper(): pass")],
        )
        result = extract_dependent_function("main_func", ctx)
        assert result == "MyClass.helper"

    def test_only_main_in_code_returns_false(self) -> None:
        """When code only contains the main function, no dependent function exists."""
        ctx = _make_code_context(
            preexisting_objects=set(), testgen_code_strings=[CodeString(code="def __init__(): pass")]
        )
        result = extract_dependent_function("HttpInterface.__init__", ctx)
        assert result is False

    def test_async_functions_extracted(self) -> None:
        """Async function definitions are also extracted as dependent functions."""
        ctx = _make_code_context(
            preexisting_objects={("async_helper", ())},
            testgen_code_strings=[CodeString(code="def main(): pass\nasync def async_helper(): pass")],
        )
        result = extract_dependent_function("main", ctx)
        assert result == "async_helper"


class TestGrabDependentFunctionFromCoverageData:
    def _make_func_data(self, coverage_pct: float = 80.0) -> dict[str, Any]:
        return {
            "summary": {"percent_covered": coverage_pct},
            "executed_lines": [1, 2, 3],
            "missing_lines": [4],
            "executed_branches": [[1, 0]],
            "missing_branches": [[2, 1]],
        }

    def test_exact_match_in_coverage_data(self) -> None:
        coverage_data = {"HttpInterface.__init__": self._make_func_data(90.0)}
        result = CoverageUtils.grab_dependent_function_from_coverage_data("HttpInterface.__init__", coverage_data, {})
        assert result.name == "HttpInterface.__init__"
        assert result.coverage == 90.0

    def test_fallback_exact_match_in_original_data(self) -> None:
        original_cov_data = {
            "files": {"http_api.py": {"functions": {"HttpInterface.__init__": self._make_func_data(75.0)}}}
        }
        result = CoverageUtils.grab_dependent_function_from_coverage_data(
            "HttpInterface.__init__", {}, original_cov_data
        )
        assert result.name == "HttpInterface.__init__"
        assert result.coverage == 75.0

    def test_fallback_suffix_match_in_original_data(self) -> None:
        """Qualified dependent name matches via suffix in original coverage data."""
        original_cov_data = {
            "files": {"http_api.py": {"functions": {"module.HttpInterface.__init__": self._make_func_data(60.0)}}}
        }
        result = CoverageUtils.grab_dependent_function_from_coverage_data(
            "HttpInterface.__init__", {}, original_cov_data
        )
        assert result.name == "HttpInterface.__init__"
        assert result.coverage == 60.0

    def test_no_false_substring_match_bare_init(self) -> None:
        """Bare __init__ should NOT match PathAwareCORSMiddleware.__init__ via substring."""
        original_cov_data = {
            "files": {"cors.py": {"functions": {"PathAwareCORSMiddleware.__init__": self._make_func_data(50.0)}}}
        }
        result = CoverageUtils.grab_dependent_function_from_coverage_data("__init__", {}, original_cov_data)
        assert result.coverage == 0

    def test_no_false_substring_match_different_class(self) -> None:
        """Qualified name for one class should not match another class's method."""
        original_cov_data = {
            "files": {
                "api.py": {
                    "functions": {
                        "PathAwareCORSMiddleware.__init__": self._make_func_data(50.0),
                        "HttpInterface.__init__": self._make_func_data(85.0),
                    }
                }
            }
        }
        result = CoverageUtils.grab_dependent_function_from_coverage_data(
            "HttpInterface.__init__", {}, original_cov_data
        )
        assert result.name == "HttpInterface.__init__"
        assert result.coverage == 85.0

    def test_no_match_returns_zero_coverage(self) -> None:
        result = CoverageUtils.grab_dependent_function_from_coverage_data("nonexistent_func", {}, {"files": {}})
        assert result.coverage == 0
        assert result.executed_lines == []

    def test_qualified_suffix_no_match_for_partial_name(self) -> None:
        """Ensure suffix match requires a dot boundary, not just string suffix."""
        original_cov_data = {
            "files": {"api.py": {"functions": {"XHttpInterface.__init__": self._make_func_data(40.0)}}}
        }
        # "HttpInterface.__init__" should NOT match "XHttpInterface.__init__" via suffix
        result = CoverageUtils.grab_dependent_function_from_coverage_data(
            "HttpInterface.__init__", {}, original_cov_data
        )
        assert result.coverage == 0

    def test_bare_name_exact_match_in_fallback(self) -> None:
        """Bare function name should still work with exact match in fallback."""
        original_cov_data = {"files": {"utils.py": {"functions": {"helper_func": self._make_func_data(95.0)}}}}
        result = CoverageUtils.grab_dependent_function_from_coverage_data("helper_func", {}, original_cov_data)
        assert result.name == "helper_func"
        assert result.coverage == 95.0
