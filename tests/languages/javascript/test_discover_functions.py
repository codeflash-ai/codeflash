"""Tests for JavaScript/TypeScript function discovery logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.languages.base import FunctionFilterCriteria
from codeflash.languages.javascript.support import JavaScriptSupport


class TestFunctionDiscovery:
    """Tests for discover_functions method."""

    @pytest.fixture
    def js_support(self) -> JavaScriptSupport:
        """Create a JavaScriptSupport instance."""
        return JavaScriptSupport()

    def test_discovers_top_level_function(self, js_support: JavaScriptSupport) -> None:
        """Should discover top-level exported functions."""
        code = """
export function topLevelFunc() {
    return 42;
}
"""
        functions = js_support.discover_functions(
            code,
            Path("/tmp/test.js"),
            FunctionFilterCriteria(require_export=True, require_return=True),
        )

        assert len(functions) == 1
        assert functions[0].function_name == "topLevelFunc"
        assert functions[0].parents == []

    def test_skips_nested_functions_in_closures(self, js_support: JavaScriptSupport) -> None:
        """Should skip nested functions that are defined inside other functions.

        Nested functions depend on closure variables from their parent scope and cannot
        be optimized in isolation without extracting the entire parent context.

        Bug: Previously, nested functions were discovered and attempted to be optimized,
        but the extraction logic only captured the nested function body, causing
        validation errors like "Undefined variable(s): base, streamFn, record".
        """
        code = """
export function wrapStreamFn(streamFn) {
    const base = { id: 1 };
    const record = (event) => { };

    const wrapped = (model, context, options) => {
        if (!model) {
            return streamFn(model, context, options);
        }
        record({ data: base });
        return base;
    };

    return wrapped;
}
"""
        functions = js_support.discover_functions(
            code,
            Path("/tmp/test.js"),
            FunctionFilterCriteria(require_export=True, require_return=True),
        )

        # Should only discover the top-level function, not the nested ones
        assert len(functions) == 1, f"Expected 1 function but found {len(functions)}: {[f.function_name for f in functions]}"
        assert functions[0].function_name == "wrapStreamFn"
        assert functions[0].parents == []

    def test_discovers_class_methods(self, js_support: JavaScriptSupport) -> None:
        """Should discover class methods (these are handled specially with class wrapping)."""
        code = """
export class MyClass {
    myMethod() {
        return 42;
    }
}
"""
        functions = js_support.discover_functions(
            code,
            Path("/tmp/test.js"),
            FunctionFilterCriteria(require_export=True, require_return=True, include_methods=True),
        )

        assert len(functions) == 1
        assert functions[0].function_name == "myMethod"
        assert len(functions[0].parents) == 1
        assert functions[0].parents[0].name == "MyClass"
        assert functions[0].parents[0].type == "ClassDef"

    def test_skips_nested_functions_with_multiple_levels(self, js_support: JavaScriptSupport) -> None:
        """Should skip deeply nested functions."""
        code = """
export function outer() {
    const middle = () => {
        const inner = () => {
            return 42;
        };
        return inner();
    };
    return middle();
}
"""
        functions = js_support.discover_functions(
            code,
            Path("/tmp/test.js"),
            FunctionFilterCriteria(require_export=True, require_return=True),
        )

        # Should only discover the top-level function
        assert len(functions) == 1
        assert functions[0].function_name == "outer"
