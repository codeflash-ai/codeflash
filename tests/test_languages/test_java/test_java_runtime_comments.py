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
        expected = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibonacciTest {
    @Test
    void testFibonacci() {
        Fibonacci.fibonacci(10); // 2.89ms -> 26.2\u03bcs (10931% faster)
    }
}
"""
        assert result == expected

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
        expected = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void testMultiple() {
        Fibonacci.fibonacci(5); // 1.00ms -> 100\u03bcs (900% faster)
        Fibonacci.fibonacci(10); // 5.00ms -> 500\u03bcs (900% faster)
    }
}
"""
        assert result == expected

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
        expected = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void testSmall() {
        Fibonacci.fibonacci(5); // 500\u03bcs -> 50.0\u03bcs (900% faster)
    }

    @Test
    void testLarge() {
        Fibonacci.fibonacci(100); // 10.0ms -> 1.00ms (900% faster)
    }
}
"""
        assert result == expected

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

    def test_annotations_only_for_matching_classes(self) -> None:
        """Bug: add_runtime_comments ignores class/method prefixes and only uses line numbers.

        When runtime keys from different test classes map to the same line number,
        annotations from unrelated classes leak into the source. Only annotations
        whose class.method prefix matches a class in the test source should be applied.
        """
        source = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void testSmall() {
        Fibonacci.fibonacci(5);
    }
}
"""
        # Keys from FibTest should match, keys from UnrelatedTest should NOT
        original = {
            "FibTest.testSmall#L8": 500_000,
            "UnrelatedTest.testOther#L8": 2_000_000,  # same line number, different class
        }
        optimized = {"FibTest.testSmall#L8": 50_000, "UnrelatedTest.testOther#L8": 200_000}
        result = add_runtime_comments(source, original, optimized)
        # Only FibTest annotation should appear, NOT the summed value from both classes
        expected = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void testSmall() {
        Fibonacci.fibonacci(5); // 500\u03bcs -> 50.0\u03bcs (900% faster)
    }
}
"""
        assert result == expected

    def test_annotations_different_classes_different_lines_no_leak(self) -> None:
        """Annotations from classes not present in the source should not appear at all."""
        source = """\
package com.example;

import org.junit.jupiter.api.Test;

public class AlphaTest {
    @Test
    void testAlpha() {
        Alpha.run();
    }
}
"""
        # BetaTest.testBeta#L10 should not annotate line 10 in AlphaTest source
        original = {"AlphaTest.testAlpha#L8": 1_000_000, "BetaTest.testBeta#L10": 3_000_000}
        optimized = {"AlphaTest.testAlpha#L8": 100_000, "BetaTest.testBeta#L10": 300_000}
        result = add_runtime_comments(source, original, optimized)
        lines = result.splitlines()
        # Line 10 (1-indexed) = line 9 (0-indexed) = "}" - should NOT have annotation
        assert "//" not in lines[9]
        # Line 8 should have the AlphaTest annotation
        assert "// 1.00ms -> 100\u03bcs" in lines[7]

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
        expected = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void test() {
        Fibonacci.fibonacci(10); // 3.00ms -> 300\u03bcs (900% faster)
    }
}
"""
        assert result == expected


class TestAddRuntimeCommentsToGeneratedTests:
    """Tests for add_runtime_comments_to_generated_tests on JavaSupport."""

    def test_multi_file_annotations_dont_leak(self) -> None:
        """Bug: annotations from one test file should not appear in another test file.

        When multiple test files have different classes, the runtime map from file 1
        should not affect file 2. Currently, all runtime keys are merged into one map
        and applied to every file, causing cross-file annotation leaking.
        """
        from pathlib import Path
        from unittest.mock import MagicMock

        from codeflash.languages.java.support import JavaSupport
        from codeflash.models.models import GeneratedTests, GeneratedTestsList, InvocationId

        support = MagicMock(spec=JavaSupport)
        support._build_runtime_map = JavaSupport._build_runtime_map.__get__(support, JavaSupport)
        support.add_runtime_comments = JavaSupport.add_runtime_comments.__get__(support, JavaSupport)
        support._analyzer = None
        support.add_runtime_comments_to_generated_tests = JavaSupport.add_runtime_comments_to_generated_tests.__get__(
            support, JavaSupport
        )

        test1_source = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FooTest {
    @Test
    void testFoo() {
        Foo.run();
    }
}
"""
        test2_source = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FooTest_2 {
    @Test
    void testFoo2() {
        Foo.run2();
    }
}
"""
        generated_tests = GeneratedTestsList(
            generated_tests=[
                GeneratedTests(
                    generated_original_test_source=test1_source,
                    instrumented_behavior_test_source="",
                    instrumented_perf_test_source="",
                    behavior_file_path=Path("FooTest.java"),
                    perf_file_path=Path("FooTest_perf.java"),
                ),
                GeneratedTests(
                    generated_original_test_source=test2_source,
                    instrumented_behavior_test_source="",
                    instrumented_perf_test_source="",
                    behavior_file_path=Path("FooTest_2.java"),
                    perf_file_path=Path("FooTest_2_perf.java"),
                ),
            ]
        )

        # Runtime data: FooTest has call at L8, FooTest_2 has call at L8 too
        inv_id_1 = InvocationId(
            test_module_path="com.example",
            test_class_name="FooTest",
            test_function_name="testFoo",
            function_getting_tested="run",
            iteration_id="L8_1",
        )
        inv_id_2 = InvocationId(
            test_module_path="com.example",
            test_class_name="FooTest_2",
            test_function_name="testFoo2",
            function_getting_tested="run2",
            iteration_id="L8_1",
        )

        original = {inv_id_1: [1_000_000], inv_id_2: [2_000_000]}
        optimized = {inv_id_1: [100_000], inv_id_2: [200_000]}

        result = support.add_runtime_comments_to_generated_tests(generated_tests, original, optimized)

        result_test1 = result.generated_tests[0].generated_original_test_source
        result_test2 = result.generated_tests[1].generated_original_test_source

        # Test file 1 should have FooTest annotation ONLY (1ms -> 100us)
        assert "// 1.00ms -> 100\u03bcs" in result_test1
        assert "// 2.00ms -> 200\u03bcs" not in result_test1  # FooTest_2 annotation should NOT appear

        # Test file 2 should have FooTest_2 annotation ONLY (2ms -> 200us)
        assert "// 2.00ms -> 200\u03bcs" in result_test2
        assert "// 1.00ms -> 100\u03bcs" not in result_test2  # FooTest annotation should NOT appear


class TestRuntimeCommentsAfterFunctionRemoval:
    """Tests that runtime comments remain correct when test functions are removed."""

    def test_removal_after_annotation_preserves_line_alignment(self) -> None:
        """Bug: when test functions are removed BEFORE adding runtime comments,
        line numbers shift and annotations end up on wrong lines.

        The fix is to add runtime comments first, then remove test functions.
        This test verifies the correct ordering by simulating both operations.
        """
        from codeflash.languages.java.replacement import add_runtime_comments, remove_test_functions

        source = """\
package com.example;

import org.junit.jupiter.api.Test;

public class FibTest {
    @Test
    void testFailing() {
        Fibonacci.fibonacci(0);
    }

    @Test
    void testWorking() {
        Fibonacci.fibonacci(100);
    }
}
"""
        original = {"FibTest.testWorking#L13": 10_000_000}
        optimized = {"FibTest.testWorking#L13": 1_000_000}

        # Correct order: annotate THEN remove
        annotated = add_runtime_comments(source, original, optimized)
        result_correct = remove_test_functions(annotated, ["testFailing"])
        # The annotation should be on the Fibonacci.fibonacci(100) line
        for line in result_correct.splitlines():
            if "Fibonacci.fibonacci(100)" in line:
                assert "// 10.0ms -> 1.00ms" in line, f"Annotation missing from call line: {line}"
                break
        else:
            raise AssertionError("Fibonacci.fibonacci(100) line not found in result")

        # Wrong order (old behavior): remove THEN annotate - annotation lands on wrong line
        removed_first = remove_test_functions(source, ["testFailing"])
        result_wrong = add_runtime_comments(removed_first, original, optimized)
        # After removal, testFailing (lines 7-9) is gone, shifting testWorking up.
        # Line 13 in the post-removal source is beyond the file end or on a wrong line.
        # Verify the annotation does NOT correctly land on the fibonacci(100) call
        for line in result_wrong.splitlines():
            if "Fibonacci.fibonacci(100)" in line:
                has_annotation = "// 10.0ms -> 1.00ms" in line
                assert not has_annotation, "Wrong ordering accidentally worked - test needs different line numbers"
                break


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
