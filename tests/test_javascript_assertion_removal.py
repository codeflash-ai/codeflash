"""Comprehensive tests for JavaScript assertion removal in test instrumentation.

This module tests the removal of expect() assertions from LLM-generated tests,
covering all patterns that might be seen in the wild.
"""

from __future__ import annotations

import pytest

from codeflash.languages.javascript.instrument import (
    ExpectCallTransformer,
    TestingMode,
    instrument_generated_js_test,
    transform_expect_calls,
)


class TestExpectCallTransformer:
    """Tests for the ExpectCallTransformer class."""

    def test_basic_toBe_assertion(self) -> None:
        """Test basic .toBe() assertion removal."""
        code = "expect(fibonacci(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "fibonacci", "fibonacci", "capture", remove_assertions=True)
        assert result == "codeflash.capture('fibonacci', '1', fibonacci, 5);"

    def test_basic_toEqual_assertion(self) -> None:
        """Test .toEqual() assertion removal."""
        code = "expect(func([1, 2, 3])).toEqual([1, 2, 3]);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, [1, 2, 3]);"

    def test_toStrictEqual_assertion(self) -> None:
        """Test .toStrictEqual() assertion removal."""
        code = "expect(func({a: 1})).toStrictEqual({a: 1});"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, {a: 1});"

    def test_toBeCloseTo_with_precision(self) -> None:
        """Test .toBeCloseTo() with precision argument."""
        code = "expect(func(3.14159)).toBeCloseTo(3.14, 2);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 3.14159);"

    def test_toBeTruthy_no_args(self) -> None:
        """Test .toBeTruthy() assertion without arguments."""
        code = "expect(func(true)).toBeTruthy();"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, true);"

    def test_toBeFalsy_no_args(self) -> None:
        """Test .toBeFalsy() assertion without arguments."""
        code = "expect(func(0)).toBeFalsy();"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 0);"

    def test_toBeNull(self) -> None:
        """Test .toBeNull() assertion."""
        code = "expect(func(null)).toBeNull();"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, null);"

    def test_toBeUndefined(self) -> None:
        """Test .toBeUndefined() assertion."""
        code = "expect(func()).toBeUndefined();"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func);"

    def test_toBeDefined(self) -> None:
        """Test .toBeDefined() assertion."""
        code = "expect(func(1)).toBeDefined();"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 1);"

    def test_toBeNaN(self) -> None:
        """Test .toBeNaN() assertion."""
        code = "expect(func(NaN)).toBeNaN();"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, NaN);"

    def test_toBeGreaterThan(self) -> None:
        """Test .toBeGreaterThan() assertion."""
        code = "expect(func(10)).toBeGreaterThan(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 10);"

    def test_toBeLessThan(self) -> None:
        """Test .toBeLessThan() assertion."""
        code = "expect(func(3)).toBeLessThan(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 3);"

    def test_toBeGreaterThanOrEqual(self) -> None:
        """Test .toBeGreaterThanOrEqual() assertion."""
        code = "expect(func(5)).toBeGreaterThanOrEqual(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 5);"

    def test_toBeLessThanOrEqual(self) -> None:
        """Test .toBeLessThanOrEqual() assertion."""
        code = "expect(func(5)).toBeLessThanOrEqual(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 5);"

    def test_toContain(self) -> None:
        """Test .toContain() assertion."""
        code = "expect(func([1, 2, 3])).toContain(2);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, [1, 2, 3]);"

    def test_toContainEqual(self) -> None:
        """Test .toContainEqual() assertion."""
        code = "expect(func([{a: 1}])).toContainEqual({a: 1});"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, [{a: 1}]);"

    def test_toHaveLength(self) -> None:
        """Test .toHaveLength() assertion."""
        code = "expect(func([1, 2, 3])).toHaveLength(3);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, [1, 2, 3]);"

    def test_toMatch_string(self) -> None:
        """Test .toMatch() with string pattern."""
        code = "expect(func('hello')).toMatch('ell');"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 'hello');"

    def test_toMatch_regex(self) -> None:
        """Test .toMatch() with regex pattern."""
        code = "expect(func('hello')).toMatch(/ell/);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 'hello');"

    def test_toMatchObject(self) -> None:
        """Test .toMatchObject() assertion."""
        code = "expect(func({a: 1, b: 2})).toMatchObject({a: 1});"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, {a: 1, b: 2});"

    def test_toHaveProperty(self) -> None:
        """Test .toHaveProperty() assertion."""
        code = "expect(func({a: 1})).toHaveProperty('a');"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, {a: 1});"

    def test_toHaveProperty_with_value(self) -> None:
        """Test .toHaveProperty() with value."""
        code = "expect(func({a: 1})).toHaveProperty('a', 1);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, {a: 1});"

    def test_toBeInstanceOf(self) -> None:
        """Test .toBeInstanceOf() assertion."""
        code = "expect(func()).toBeInstanceOf(Array);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func);"


class TestNegatedAssertions:
    """Tests for negated assertions with .not modifier."""

    def test_not_toBe(self) -> None:
        """Test .not.toBe() assertion removal."""
        code = "expect(func(5)).not.toBe(10);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 5);"

    def test_not_toEqual(self) -> None:
        """Test .not.toEqual() assertion removal."""
        code = "expect(func([1, 2])).not.toEqual([3, 4]);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, [1, 2]);"

    def test_not_toBeTruthy(self) -> None:
        """Test .not.toBeTruthy() assertion removal."""
        code = "expect(func(0)).not.toBeTruthy();"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 0);"

    def test_not_toContain(self) -> None:
        """Test .not.toContain() assertion removal."""
        code = "expect(func([1, 2, 3])).not.toContain(4);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, [1, 2, 3]);"

    def test_not_toBeNull(self) -> None:
        """Test .not.toBeNull() assertion removal."""
        code = "expect(func(1)).not.toBeNull();"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 1);"


class TestAsyncAssertions:
    """Tests for async assertions with .resolves and .rejects modifiers."""

    def test_resolves_toBe(self) -> None:
        """Test .resolves.toBe() assertion removal."""
        code = "expect(asyncFunc(5)).resolves.toBe(10);"
        result, _ = transform_expect_calls(code, "asyncFunc", "asyncFunc", "capture", remove_assertions=True)
        assert result == "codeflash.capture('asyncFunc', '1', asyncFunc, 5);"

    def test_resolves_toEqual(self) -> None:
        """Test .resolves.toEqual() assertion removal."""
        code = "expect(asyncFunc()).resolves.toEqual({data: 'test'});"
        result, _ = transform_expect_calls(code, "asyncFunc", "asyncFunc", "capture", remove_assertions=True)
        assert result == "codeflash.capture('asyncFunc', '1', asyncFunc);"

    def test_rejects_toThrow(self) -> None:
        """Test .rejects.toThrow() assertion removal."""
        code = "expect(asyncFunc()).rejects.toThrow();"
        result, _ = transform_expect_calls(code, "asyncFunc", "asyncFunc", "capture", remove_assertions=True)
        assert result == "codeflash.capture('asyncFunc', '1', asyncFunc);"

    def test_rejects_toThrow_with_message(self) -> None:
        """Test .rejects.toThrow() with error message."""
        code = "expect(asyncFunc()).rejects.toThrow('Error message');"
        result, _ = transform_expect_calls(code, "asyncFunc", "asyncFunc", "capture", remove_assertions=True)
        assert result == "codeflash.capture('asyncFunc', '1', asyncFunc);"

    def test_not_resolves_toBe(self) -> None:
        """Test .not.resolves.toBe() (rare but valid)."""
        code = "expect(asyncFunc()).not.resolves.toBe(5);"
        result, _ = transform_expect_calls(code, "asyncFunc", "asyncFunc", "capture", remove_assertions=True)
        assert result == "codeflash.capture('asyncFunc', '1', asyncFunc);"


class TestNestedParentheses:
    """Tests for handling nested parentheses in function arguments."""

    def test_nested_function_call(self) -> None:
        """Test nested function call in arguments."""
        code = "expect(func(getN(5))).toBe(10);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, getN(5));"

    def test_deeply_nested_calls(self) -> None:
        """Test deeply nested function calls."""
        code = "expect(func(outer(inner(deep(1))))).toBe(100);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, outer(inner(deep(1))));"

    def test_multiple_nested_args(self) -> None:
        """Test multiple arguments with nested calls."""
        code = "expect(func(getA(), getB(getC()))).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, getA(), getB(getC()));"

    def test_object_with_nested_calls(self) -> None:
        """Test object argument with nested function calls."""
        code = "expect(func({key: getValue()})).toEqual({key: 1});"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, {key: getValue()});"

    def test_array_with_nested_calls(self) -> None:
        """Test array argument with nested function calls."""
        code = "expect(func([getA(), getB()])).toEqual([1, 2]);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, [getA(), getB()]);"


class TestStringLiterals:
    """Tests for handling string literals with special characters."""

    def test_string_with_parentheses(self) -> None:
        """Test string argument containing parentheses."""
        code = "expect(func('hello (world)')).toBe('result');"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 'hello (world)');"

    def test_double_quoted_string_with_parens(self) -> None:
        """Test double-quoted string with parentheses."""
        code = 'expect(func("hello (world)")).toBe("result");'
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, \"hello (world)\");"

    def test_template_literal(self) -> None:
        """Test template literal argument."""
        code = "expect(func(`template ${value}`)).toBe('result');"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, `template ${value}`);"

    def test_template_literal_with_parens(self) -> None:
        """Test template literal with parentheses inside."""
        code = "expect(func(`hello (${name})`)).toBe('greeting');"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, `hello (${name})`);"

    def test_escaped_quotes(self) -> None:
        """Test string with escaped quotes."""
        code = "expect(func('it\\'s working')).toBe('yes');"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 'it\\'s working');"


class TestWhitespaceHandling:
    """Tests for various whitespace patterns."""

    def test_leading_whitespace_preserved(self) -> None:
        """Test that leading whitespace is preserved."""
        code = "        expect(func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "        codeflash.capture('func', '1', func, 5);"

    def test_tab_indentation(self) -> None:
        """Test tab indentation is preserved."""
        code = "\t\texpect(func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "\t\tcodeflash.capture('func', '1', func, 5);"

    def test_no_space_after_expect(self) -> None:
        """Test expect without space before parenthesis."""
        code = "expect(func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 5);"

    def test_space_after_expect(self) -> None:
        """Test expect with space before parenthesis."""
        code = "expect (func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 5);"

    def test_newline_in_assertion(self) -> None:
        """Test assertion split across lines."""
        code = """expect(func(5))
            .toBe(5);"""
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 5);"

    def test_newline_after_expect_close(self) -> None:
        """Test newline after expect closing paren."""
        code = """expect(func(5))
.toBe(5);"""
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 5);"


class TestMultipleAssertions:
    """Tests for multiple assertions in the same code."""

    def test_multiple_assertions_same_function(self) -> None:
        """Test multiple assertions for the same function."""
        code = """expect(func(1)).toBe(1);
expect(func(2)).toBe(2);
expect(func(3)).toBe(3);"""
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        expected = """codeflash.capture('func', '1', func, 1);
codeflash.capture('func', '2', func, 2);
codeflash.capture('func', '3', func, 3);"""
        assert result == expected

    def test_multiple_different_assertions(self) -> None:
        """Test multiple different assertion types."""
        code = """expect(func(1)).toBe(1);
expect(func(2)).toEqual(2);
expect(func(3)).not.toBe(0);"""
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        expected = """codeflash.capture('func', '1', func, 1);
codeflash.capture('func', '2', func, 2);
codeflash.capture('func', '3', func, 3);"""
        assert result == expected

    def test_mixed_with_other_code(self) -> None:
        """Test assertions mixed with other code."""
        code = """const x = 5;
expect(func(x)).toBe(10);
console.log('done');
expect(func(x + 1)).toBe(12);"""
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        expected = """const x = 5;
codeflash.capture('func', '1', func, x);
console.log('done');
codeflash.capture('func', '2', func, x + 1);"""
        assert result == expected


class TestSemicolonHandling:
    """Tests for semicolon handling."""

    def test_with_semicolon(self) -> None:
        """Test assertion with trailing semicolon."""
        code = "expect(func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 5);"

    def test_without_semicolon(self) -> None:
        """Test assertion without trailing semicolon."""
        code = "expect(func(5)).toBe(5)"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func, 5);"

    def test_multiple_without_semicolons(self) -> None:
        """Test multiple assertions without semicolons (common in some styles)."""
        code = """expect(func(1)).toBe(1)
expect(func(2)).toBe(2)"""
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        expected = """codeflash.capture('func', '1', func, 1);
codeflash.capture('func', '2', func, 2);"""
        assert result == expected


class TestPreservingAssertions:
    """Tests for keeping assertions intact (for existing user tests)."""

    def test_preserve_toBe(self) -> None:
        """Test preserving .toBe() assertion."""
        code = "expect(func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=False)
        assert result == "expect(codeflash.capture('func', '1', func, 5)).toBe(5);"

    def test_preserve_not_toBe(self) -> None:
        """Test preserving .not.toBe() assertion."""
        code = "expect(func(5)).not.toBe(10);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=False)
        assert result == "expect(codeflash.capture('func', '1', func, 5)).not.toBe(10);"

    def test_preserve_resolves(self) -> None:
        """Test preserving .resolves assertion."""
        code = "expect(asyncFunc(5)).resolves.toBe(10);"
        result, _ = transform_expect_calls(code, "asyncFunc", "asyncFunc", "capture", remove_assertions=False)
        assert result == "expect(codeflash.capture('asyncFunc', '1', asyncFunc, 5)).resolves.toBe(10);"

    def test_preserve_toBeCloseTo(self) -> None:
        """Test preserving .toBeCloseTo() with args."""
        code = "expect(func(3.14159)).toBeCloseTo(3.14, 2);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=False)
        assert result == "expect(codeflash.capture('func', '1', func, 3.14159)).toBeCloseTo(3.14, 2);"


class TestCaptureFunction:
    """Tests for different capture function modes."""

    def test_behavior_mode_uses_capture(self) -> None:
        """Test behavior mode uses capture function."""
        code = "expect(func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert "codeflash.capture(" in result

    def test_performance_mode_uses_capturePerf(self) -> None:
        """Test performance mode uses capturePerf function."""
        code = "expect(func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capturePerf", remove_assertions=True)
        assert "codeflash.capturePerf(" in result


class TestQualifiedNames:
    """Tests for qualified function names."""

    def test_simple_qualified_name(self) -> None:
        """Test simple qualified name."""
        code = "expect(func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "module.func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('module.func', '1', func, 5);"

    def test_nested_qualified_name(self) -> None:
        """Test nested qualified name."""
        code = "expect(func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "pkg.module.func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('pkg.module.func', '1', func, 5);"


class TestEdgeCases:
    """Tests for edge cases and potential issues."""

    def test_function_name_as_substring(self) -> None:
        """Test that function name matching is exact."""
        code = "expect(myFunc(5)).toBe(5); expect(func(10)).toBe(10);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        # Should only transform func, not myFunc
        assert "expect(myFunc(5)).toBe(5)" in result
        assert "codeflash.capture('func', '1', func, 10)" in result

    def test_empty_args(self) -> None:
        """Test function call with no arguments."""
        code = "expect(func()).toBe(undefined);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == "codeflash.capture('func', '1', func);"

    def test_object_method_style(self) -> None:
        """Test that method calls on objects are not matched."""
        code = "expect(obj.func(5)).toBe(5);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        # Should not transform method calls
        assert result == "expect(obj.func(5)).toBe(5);"

    def test_non_matching_code_unchanged(self) -> None:
        """Test that non-matching code remains unchanged."""
        code = "const x = func(5); console.log(x);"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        assert result == code

    def test_expect_without_assertion(self) -> None:
        """Test expect without assertion is not transformed."""
        code = "const result = expect(func(5));"
        result, _ = transform_expect_calls(code, "func", "func", "capture", remove_assertions=True)
        # Should not transform as there's no assertion
        assert result == code


class TestInstrumentGeneratedJsTest:
    """Integration tests for instrument_generated_js_test function."""

    def test_full_test_file_behavior_mode(self) -> None:
        """Test instrumenting a full test file in behavior mode."""
        code = """import { fibonacci } from '../fibonacci.js';

describe('fibonacci', () => {
    test('basic', () => {
        expect(fibonacci(5)).toBe(5);
        expect(fibonacci(10)).toBe(55);
    });
});"""
        result = instrument_generated_js_test(code, "fibonacci", "fibonacci", TestingMode.BEHAVIOR)
        assert "import codeflash from 'codeflash'" in result
        assert "codeflash.capture('fibonacci'" in result
        assert ".toBe(" not in result

    def test_full_test_file_performance_mode(self) -> None:
        """Test instrumenting a full test file in performance mode."""
        code = """import { fibonacci } from '../fibonacci.js';

describe('fibonacci', () => {
    test('basic', () => {
        expect(fibonacci(5)).toBe(5);
    });
});"""
        result = instrument_generated_js_test(code, "fibonacci", "fibonacci", TestingMode.PERFORMANCE)
        assert "import codeflash from 'codeflash'" in result
        assert "codeflash.capturePerf('fibonacci'" in result
        assert ".toBe(" not in result

    def test_commonjs_import_style(self) -> None:
        """Test CommonJS require style."""
        code = """const { fibonacci } = require('../fibonacci');

describe('fibonacci', () => {
    test('basic', () => {
        expect(fibonacci(5)).toBe(5);
    });
});"""
        result = instrument_generated_js_test(code, "fibonacci", "fibonacci", TestingMode.BEHAVIOR)
        assert "const codeflash = require('codeflash')" in result
        assert "codeflash.capture('fibonacci'" in result

    def test_various_assertion_types(self) -> None:
        """Test file with various assertion types."""
        code = """import { func } from './func.js';

describe('func', () => {
    test('various assertions', () => {
        expect(func(5)).toBe(5);
        expect(func(-5)).not.toBe(5);
        expect(func(0.5)).toBeCloseTo(0.5, 2);
        expect(func(true)).toBeTruthy();
        expect(func(null)).toBeNull();
    });
});"""
        result = instrument_generated_js_test(code, "func", "func", TestingMode.BEHAVIOR)
        # All assertions should be removed
        assert ".toBe(" not in result
        assert ".not." not in result
        assert ".toBeCloseTo(" not in result
        assert ".toBeTruthy(" not in result
        assert ".toBeNull(" not in result
        # All should have capture calls
        assert result.count("codeflash.capture(") == 5

    def test_empty_code(self) -> None:
        """Test with empty code."""
        result = instrument_generated_js_test("", "func", "func", TestingMode.BEHAVIOR)
        assert result == ""

    def test_whitespace_only_code(self) -> None:
        """Test with whitespace-only code."""
        result = instrument_generated_js_test("   \n\t  ", "func", "func", TestingMode.BEHAVIOR)
        assert result == "   \n\t  "


class TestRealWorldPatterns:
    """Tests based on real-world LLM-generated test patterns."""

    def test_jest_describe_test_structure(self) -> None:
        """Test standard Jest describe/test structure."""
        code = """import { processData } from '../processData';

describe('processData', () => {
    describe('with valid input', () => {
        test('returns processed result', () => {
            expect(processData({input: 'test'})).toEqual({output: 'TEST'});
        });

        test('handles arrays', () => {
            expect(processData([1, 2, 3])).toEqual([2, 4, 6]);
        });
    });

    describe('with invalid input', () => {
        test('returns null for undefined', () => {
            expect(processData(undefined)).toBeNull();
        });
    });
});"""
        result = instrument_generated_js_test(code, "processData", "processData", TestingMode.BEHAVIOR)
        assert result.count("codeflash.capture(") == 3
        assert "toEqual(" not in result
        assert "toBeNull(" not in result

    def test_vitest_it_structure(self) -> None:
        """Test Vitest it() style tests."""
        code = """import { calculate } from './calculate';

describe('calculate', () => {
    it('should add numbers', () => {
        expect(calculate(1, 2, 'add')).toBe(3);
    });

    it('should multiply numbers', () => {
        expect(calculate(2, 3, 'mul')).toBe(6);
    });
});"""
        result = instrument_generated_js_test(code, "calculate", "calculate", TestingMode.BEHAVIOR)
        assert result.count("codeflash.capture(") == 2
        assert ".toBe(" not in result

    def test_async_await_pattern(self) -> None:
        """Test async/await test pattern."""
        code = """import { fetchData } from './api';

describe('fetchData', () => {
    test('fetches data successfully', async () => {
        expect(fetchData('/api/users')).resolves.toEqual([{id: 1}]);
    });

    test('handles errors', async () => {
        expect(fetchData('/invalid')).rejects.toThrow('Not found');
    });
});"""
        result = instrument_generated_js_test(code, "fetchData", "fetchData", TestingMode.BEHAVIOR)
        assert result.count("codeflash.capture(") == 2
        assert ".resolves." not in result
        assert ".rejects." not in result

    def test_numeric_precision_tests(self) -> None:
        """Test numeric precision test patterns."""
        code = """import { calculatePi } from './math';

describe('calculatePi', () => {
    test('calculates pi to 2 decimal places', () => {
        expect(calculatePi(2)).toBeCloseTo(3.14, 2);
    });

    test('calculates pi to 5 decimal places', () => {
        expect(calculatePi(5)).toBeCloseTo(3.14159, 5);
    });
});"""
        result = instrument_generated_js_test(code, "calculatePi", "calculatePi", TestingMode.BEHAVIOR)
        assert result.count("codeflash.capture(") == 2
        assert ".toBeCloseTo(" not in result