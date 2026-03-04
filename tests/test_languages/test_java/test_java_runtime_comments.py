"""Tests for inline runtime comments in Java generated tests."""

from __future__ import annotations

import pytest

from codeflash.languages.java.replacement import add_runtime_comments


class TestAddRuntimeComments:
    def test_single_call_inline_comment(self) -> None:
        source = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        Fibonacci.fibonacci(10);
    }
}
"""
        original = {"FibonacciTest.testFibonacci#L8": 2_890_000}
        optimized = {"FibonacciTest.testFibonacci#L8": 26_200}
        result = add_runtime_comments(source, original, optimized)
        lines = result.splitlines()
        assert "// 2.89ms ->" in lines[7]
        assert "faster" in lines[7]

    def test_multiple_calls_different_lines(self) -> None:
        source = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void testMultiple() {
        Fibonacci.fibonacci(5);
        Fibonacci.fibonacci(10);
    }
}
"""
        original = {"FibTest.testMultiple#L8": 1_000_000, "FibTest.testMultiple#L9": 5_000_000}
        optimized = {"FibTest.testMultiple#L8": 100_000, "FibTest.testMultiple#L9": 500_000}
        result = add_runtime_comments(source, original, optimized)
        lines = result.splitlines()
        assert "//" in lines[7]
        assert "//" in lines[8]

    def test_multiple_test_methods(self) -> None:
        source = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void testSmall() {
        Fibonacci.fibonacci(5);
    }

    @Test
    void testLarge() {
        Fibonacci.fibonacci(100);
    }
}
"""
        original = {"FibTest.testSmall#L8": 500_000, "FibTest.testLarge#L13": 10_000_000}
        optimized = {"FibTest.testSmall#L8": 50_000, "FibTest.testLarge#L13": 1_000_000}
        result = add_runtime_comments(source, original, optimized)
        lines = result.splitlines()
        assert "//" in lines[7]
        assert "//" in lines[12]

    def test_no_runtime_data_unchanged(self) -> None:
        source = "public class Test {}\n"
        assert add_runtime_comments(source, {}, {}) == source
        assert add_runtime_comments(source, {"k": 1}, {}) == source
        assert add_runtime_comments(source, {}, {"k": 1}) == source

    def test_only_original_no_optimized_unchanged(self) -> None:
        source = "public class Test {}\n"
        assert add_runtime_comments(source, {"FibTest.test#L1": 100}, {}) == source

    def test_key_without_line_prefix_ignored(self) -> None:
        source = """\
package com.example;

public class FibTest {
    void test() {
        Fibonacci.fibonacci(10);
    }
}
"""
        original = {"FibTest.test#1": 1_000_000}
        optimized = {"FibTest.test#1": 500_000}
        result = add_runtime_comments(source, original, optimized)
        assert result == source

    def test_same_line_sums_runtimes(self) -> None:
        source = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void test() {
        Fibonacci.fibonacci(10);
    }
}
"""
        # Two invocation IDs on the same line (e.g. "L8_1" and "L8_2" both map to "L8" in _build_runtime_map)
        # After _build_runtime_map, these are already summed into a single key "FibTest.test#L8"
        original = {"FibTest.test#L8": 3_000_000}  # sum of both calls
        optimized = {"FibTest.test#L8": 300_000}
        result = add_runtime_comments(source, original, optimized)
        lines = result.splitlines()
        assert "//" in lines[7]
        assert "faster" in lines[7]


class TestBuildRuntimeMap:
    def test_new_line_format_groups_by_line(self) -> None:
        from unittest.mock import MagicMock

        from codeflash.languages.java.support import JavaSupport

        support = MagicMock(spec=JavaSupport)
        support._build_runtime_map = JavaSupport._build_runtime_map.__get__(support, JavaSupport)

        inv_id_1 = MagicMock()
        inv_id_1.test_class_name = "FibTest"
        inv_id_1.test_function_name = "testFib"
        inv_id_1.iteration_id = "L15_1"

        inv_id_2 = MagicMock()
        inv_id_2.test_class_name = "FibTest"
        inv_id_2.test_function_name = "testFib"
        inv_id_2.iteration_id = "L15_2"

        inv_id_runtimes = {inv_id_1: [100, 200, 150], inv_id_2: [300, 400, 350]}

        result = support._build_runtime_map(inv_id_runtimes)
        # Both L15_1 and L15_2 map to "L15", so their min runtimes (100 + 300 = 400) are summed
        assert result == {"FibTest.testFib#L15": 400}

    def test_different_lines_separate_keys(self) -> None:
        from unittest.mock import MagicMock

        from codeflash.languages.java.support import JavaSupport

        support = MagicMock(spec=JavaSupport)
        support._build_runtime_map = JavaSupport._build_runtime_map.__get__(support, JavaSupport)

        inv_id_1 = MagicMock()
        inv_id_1.test_class_name = "FibTest"
        inv_id_1.test_function_name = "testFib"
        inv_id_1.iteration_id = "L10_1"

        inv_id_2 = MagicMock()
        inv_id_2.test_class_name = "FibTest"
        inv_id_2.test_function_name = "testFib"
        inv_id_2.iteration_id = "L20_1"

        inv_id_runtimes = {inv_id_1: [100, 200], inv_id_2: [500, 600]}

        result = support._build_runtime_map(inv_id_runtimes)
        assert result == {"FibTest.testFib#L10": 100, "FibTest.testFib#L20": 500}
