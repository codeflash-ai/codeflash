"""Tests for Java code instrumentation.

Tests the instrumentation functions with exact string equality assertions
to ensure the generated code matches expected output exactly.
"""

import re
from pathlib import Path

import pytest

from codeflash.languages.base import FunctionInfo, Language
from codeflash.languages.java.discovery import discover_functions_from_source
from codeflash.languages.java.instrumentation import (
    _add_timing_instrumentation,
    create_benchmark_test,
    instrument_existing_test,
    instrument_for_behavior,
    instrument_for_benchmarking,
    instrument_generated_java_test,
    remove_instrumentation,
)


class TestInstrumentForBehavior:
    """Tests for instrument_for_behavior."""

    def test_returns_source_unchanged(self):
        """Test that source is returned unchanged (Java uses JUnit pass/fail)."""
        source = """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        functions = discover_functions_from_source(source)
        result = instrument_for_behavior(source, functions)

        assert result == source

    def test_no_functions_unchanged(self):
        """Test that source is unchanged when no functions provided."""
        source = """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        result = instrument_for_behavior(source, [])
        assert result == source


class TestInstrumentForBenchmarking:
    """Tests for instrument_for_benchmarking."""

    def test_returns_source_unchanged(self):
        """Test that source is returned unchanged (Java uses Maven Surefire timing)."""
        source = """import org.junit.jupiter.api.Test;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
"""
        func = FunctionInfo(
            name="add",
            file_path=Path("Calculator.java"),
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        result = instrument_for_benchmarking(source, func)
        assert result == source


class TestInstrumentExistingTest:
    """Tests for instrument_existing_test with exact string equality."""

    def test_instrument_behavior_mode_simple(self, tmp_path: Path):
        """Test instrumenting a simple test in behavior mode."""
        test_file = tmp_path / "CalculatorTest.java"
        source = """import org.junit.jupiter.api.Test;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
"""
        test_file.write_text(source)

        func = FunctionInfo(
            name="add",
            file_path=tmp_path / "Calculator.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="behavior",
        )

        expected = """import org.junit.jupiter.api.Test;

public class CalculatorTest__perfinstrumented {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
"""
        assert success is True
        assert result == expected

    def test_instrument_performance_mode_simple(self, tmp_path: Path):
        """Test instrumenting a simple test in performance mode."""
        test_file = tmp_path / "CalculatorTest.java"
        source = """import org.junit.jupiter.api.Test;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
"""
        test_file.write_text(source)

        func = FunctionInfo(
            name="add",
            file_path=tmp_path / "Calculator.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="performance",
        )

        expected = """import org.junit.jupiter.api.Test;

public class CalculatorTest__perfonlyinstrumented {
    @Test
    public void testAdd() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "CalculatorTest";
        String _cf_cls1 = "CalculatorTest";
        String _cf_fn1 = "add";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            Calculator calc = new Calculator();
            assertEquals(4, calc.add(2, 2));
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_instrument_performance_mode_multiple_tests(self, tmp_path: Path):
        """Test instrumenting multiple test methods in performance mode."""
        test_file = tmp_path / "MathTest.java"
        source = """import org.junit.jupiter.api.Test;

public class MathTest {
    @Test
    public void testAdd() {
        assertEquals(4, add(2, 2));
    }

    @Test
    public void testSubtract() {
        assertEquals(0, subtract(2, 2));
    }
}
"""
        test_file.write_text(source)

        func = FunctionInfo(
            name="calculate",
            file_path=tmp_path / "Math.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="performance",
        )

        expected = """import org.junit.jupiter.api.Test;

public class MathTest__perfonlyinstrumented {
    @Test
    public void testAdd() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "MathTest";
        String _cf_cls1 = "MathTest";
        String _cf_fn1 = "calculate";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            assertEquals(4, add(2, 2));
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }

    @Test
    public void testSubtract() {
        // Codeflash timing instrumentation
        int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter2 = 2;
        String _cf_mod2 = "MathTest";
        String _cf_cls2 = "MathTest";
        String _cf_fn2 = "calculate";
        System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + "######$!");
        long _cf_start2 = System.nanoTime();
        try {
            assertEquals(0, subtract(2, 2));
        } finally {
            long _cf_end2 = System.nanoTime();
            long _cf_dur2 = _cf_end2 - _cf_start2;
            System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + ":" + _cf_dur2 + "######!");
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_instrument_preserves_annotations(self, tmp_path: Path):
        """Test that annotations other than @Test are preserved."""
        test_file = tmp_path / "ServiceTest.java"
        source = """import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Disabled;

public class ServiceTest {
    @Test
    @DisplayName("Test service call")
    public void testService() {
        service.call();
    }

    @Disabled
    @Test
    public void testDisabled() {
        service.other();
    }
}
"""
        test_file.write_text(source)

        func = FunctionInfo(
            name="call",
            file_path=tmp_path / "Service.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="performance",
        )

        expected = """import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Disabled;

public class ServiceTest__perfonlyinstrumented {
    @Test
    @DisplayName("Test service call")
    public void testService() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "ServiceTest";
        String _cf_cls1 = "ServiceTest";
        String _cf_fn1 = "call";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            service.call();
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }

    @Disabled
    @Test
    public void testDisabled() {
        // Codeflash timing instrumentation
        int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter2 = 2;
        String _cf_mod2 = "ServiceTest";
        String _cf_cls2 = "ServiceTest";
        String _cf_fn2 = "call";
        System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + "######$!");
        long _cf_start2 = System.nanoTime();
        try {
            service.other();
        } finally {
            long _cf_end2 = System.nanoTime();
            long _cf_dur2 = _cf_end2 - _cf_start2;
            System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + ":" + _cf_dur2 + "######!");
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_missing_file(self, tmp_path: Path):
        """Test handling missing test file."""
        test_file = tmp_path / "NonExistent.java"

        func = FunctionInfo(
            name="add",
            file_path=tmp_path / "Calculator.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="behavior",
        )

        assert success is False


class TestAddTimingInstrumentation:
    """Tests for _add_timing_instrumentation helper function."""

    def test_single_test_method(self):
        """Test timing instrumentation for a single test method."""
        source = """public class SimpleTest {
    @Test
    public void testSomething() {
        doSomething();
    }
}
"""
        result = _add_timing_instrumentation(source, "SimpleTest", "targetFunc")

        expected = """public class SimpleTest {
    @Test
    public void testSomething() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "SimpleTest";
        String _cf_cls1 = "SimpleTest";
        String _cf_fn1 = "targetFunc";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            doSomething();
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }
}
"""
        assert result == expected

    def test_multiple_test_methods(self):
        """Test timing instrumentation for multiple test methods."""
        source = """public class MultiTest {
    @Test
    public void testFirst() {
        first();
    }

    @Test
    public void testSecond() {
        second();
    }
}
"""
        result = _add_timing_instrumentation(source, "MultiTest", "func")

        expected = """public class MultiTest {
    @Test
    public void testFirst() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "MultiTest";
        String _cf_cls1 = "MultiTest";
        String _cf_fn1 = "func";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            first();
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }

    @Test
    public void testSecond() {
        // Codeflash timing instrumentation
        int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter2 = 2;
        String _cf_mod2 = "MultiTest";
        String _cf_cls2 = "MultiTest";
        String _cf_fn2 = "func";
        System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + "######$!");
        long _cf_start2 = System.nanoTime();
        try {
            second();
        } finally {
            long _cf_end2 = System.nanoTime();
            long _cf_dur2 = _cf_end2 - _cf_start2;
            System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + ":" + _cf_dur2 + "######!");
        }
    }
}
"""
        assert result == expected

    def test_timing_markers_format(self):
        """Test that timing markers have the correct format."""
        source = """public class MarkerTest {
    @Test
    public void testMarkers() {
        action();
    }
}
"""
        result = _add_timing_instrumentation(source, "TestClass", "targetMethod")

        expected = """public class MarkerTest {
    @Test
    public void testMarkers() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "TestClass";
        String _cf_cls1 = "TestClass";
        String _cf_fn1 = "targetMethod";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            action();
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }
}
"""
        assert result == expected


class TestCreateBenchmarkTest:
    """Tests for create_benchmark_test."""

    def test_create_benchmark(self):
        """Test creating a benchmark test."""
        func = FunctionInfo(
            name="add",
            file_path=Path("Calculator.java"),
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        result = create_benchmark_test(
            func,
            test_setup_code="Calculator calc = new Calculator();",
            invocation_code="calc.add(2, 2)",
            iterations=1000,
        )

        expected = """
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

/**
 * Benchmark test for add.
 * Generated by CodeFlash.
 */
public class TargetBenchmark {

    @Test
    @DisplayName("Benchmark add")
    public void benchmarkAdd() {
        Calculator calc = new Calculator();

        // Warmup phase
        for (int i = 0; i < 100; i++) {
            calc.add(2, 2);
        }

        // Measurement phase
        long startTime = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            calc.add(2, 2);
        }
        long endTime = System.nanoTime();

        long totalNanos = endTime - startTime;
        long avgNanos = totalNanos / 1000;

        System.out.println("CODEFLASH_BENCHMARK:add:total_ns=" + totalNanos + ",avg_ns=" + avgNanos + ",iterations=1000");
    }
}
"""
        assert result == expected

    def test_create_benchmark_different_iterations(self):
        """Test benchmark with different iteration count."""
        func = FunctionInfo(
            name="multiply",
            file_path=Path("Math.java"),
            start_line=1,
            end_line=3,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        result = create_benchmark_test(
            func,
            test_setup_code="",
            invocation_code="multiply(5, 3)",
            iterations=5000,
        )

        # Note: Empty test_setup_code still has 8-space indentation on its line
        expected = (
            "\n"
            "import org.junit.jupiter.api.Test;\n"
            "import org.junit.jupiter.api.DisplayName;\n"
            "\n"
            "/**\n"
            " * Benchmark test for multiply.\n"
            " * Generated by CodeFlash.\n"
            " */\n"
            "public class TargetBenchmark {\n"
            "\n"
            "    @Test\n"
            "    @DisplayName(\"Benchmark multiply\")\n"
            "    public void benchmarkMultiply() {\n"
            "        \n"  # Empty test_setup_code with 8-space indent
            "\n"
            "        // Warmup phase\n"
            "        for (int i = 0; i < 500; i++) {\n"
            "            multiply(5, 3);\n"
            "        }\n"
            "\n"
            "        // Measurement phase\n"
            "        long startTime = System.nanoTime();\n"
            "        for (int i = 0; i < 5000; i++) {\n"
            "            multiply(5, 3);\n"
            "        }\n"
            "        long endTime = System.nanoTime();\n"
            "\n"
            "        long totalNanos = endTime - startTime;\n"
            "        long avgNanos = totalNanos / 5000;\n"
            "\n"
            "        System.out.println(\"CODEFLASH_BENCHMARK:multiply:total_ns=\" + totalNanos + \",avg_ns=\" + avgNanos + \",iterations=5000\");\n"
            "    }\n"
            "}\n"
        )
        assert result == expected


class TestRemoveInstrumentation:
    """Tests for remove_instrumentation."""

    def test_returns_source_unchanged(self):
        """Test that source is returned unchanged (no-op for Java)."""
        source = """import com.codeflash.CodeFlash;
import org.junit.jupiter.api.Test;

public class Test {}
"""
        result = remove_instrumentation(source)
        assert result == source

    def test_preserves_regular_code(self):
        """Test that regular code is preserved."""
        source = """public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        result = remove_instrumentation(source)
        assert result == source


class TestInstrumentGeneratedJavaTest:
    """Tests for instrument_generated_java_test."""

    def test_instrument_generated_test_behavior_mode(self):
        """Test instrumenting generated test in behavior mode."""
        test_code = """import org.junit.jupiter.api.Test;

public class CalculatorTest {
    @Test
    public void testAdd() {
        assertEquals(4, new Calculator().add(2, 2));
    }
}
"""
        result = instrument_generated_java_test(
            test_code,
            function_name="add",
            qualified_name="Calculator.add",
            mode="behavior",
        )

        expected = """import org.junit.jupiter.api.Test;

public class CalculatorTest__perfinstrumented {
    @Test
    public void testAdd() {
        assertEquals(4, new Calculator().add(2, 2));
    }
}
"""
        assert result == expected

    def test_instrument_generated_test_performance_mode(self):
        """Test instrumenting generated test in performance mode."""
        test_code = """import org.junit.jupiter.api.Test;

public class GeneratedTest {
    @Test
    public void testMethod() {
        target.method();
    }
}
"""
        result = instrument_generated_java_test(
            test_code,
            function_name="method",
            qualified_name="Target.method",
            mode="performance",
        )

        expected = """import org.junit.jupiter.api.Test;

public class GeneratedTest__perfonlyinstrumented {
    @Test
    public void testMethod() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "GeneratedTest";
        String _cf_cls1 = "GeneratedTest";
        String _cf_fn1 = "method";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            target.method();
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }
}
"""
        assert result == expected


class TestTimingMarkerParsing:
    """Tests for parsing timing markers from stdout."""

    def test_timing_markers_can_be_parsed(self):
        """Test that generated timing markers can be parsed with the standard regex."""
        # Simulate stdout from instrumented test
        stdout = """
!$######TestModule:TestClass:targetFunc:1:1######$!
Running test...
!######TestModule:TestClass:targetFunc:1:1:12345678######!
"""
        # Use the same regex patterns from parse_test_output.py
        start_pattern = re.compile(r"!\$######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+)######\$!")
        end_pattern = re.compile(r"!######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+):([^:]+)######!")

        start_matches = start_pattern.findall(stdout)
        end_matches = end_pattern.findall(stdout)

        assert len(start_matches) == 1
        assert len(end_matches) == 1

        # Verify parsed values
        start = start_matches[0]
        assert start[0] == "TestModule"
        assert start[1] == "TestClass"
        assert start[2] == "targetFunc"
        assert start[3] == "1"
        assert start[4] == "1"

        end = end_matches[0]
        assert end[0] == "TestModule"
        assert end[1] == "TestClass"
        assert end[2] == "targetFunc"
        assert end[3] == "1"
        assert end[4] == "1"
        assert end[5] == "12345678"  # Duration in nanoseconds

    def test_multiple_timing_markers(self):
        """Test parsing multiple timing markers."""
        stdout = """
!$######Module:Class:func:1:1######$!
test 1
!######Module:Class:func:1:1:100000######!
!$######Module:Class:func:2:1######$!
test 2
!######Module:Class:func:2:1:200000######!
!$######Module:Class:func:3:1######$!
test 3
!######Module:Class:func:3:1:150000######!
"""
        end_pattern = re.compile(r"!######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+):([^:]+)######!")
        end_matches = end_pattern.findall(stdout)

        assert len(end_matches) == 3
        # Verify durations
        durations = [int(m[5]) for m in end_matches]
        assert durations == [100000, 200000, 150000]


class TestInstrumentedCodeValidity:
    """Tests to verify that instrumented code is syntactically valid Java."""

    def test_instrumented_code_has_balanced_braces(self, tmp_path: Path):
        """Test that instrumented code has balanced braces."""
        test_file = tmp_path / "BraceTest.java"
        source = """import org.junit.jupiter.api.Test;

public class BraceTest {
    @Test
    public void testOne() {
        if (true) {
            doSomething();
        }
    }

    @Test
    public void testTwo() {
        for (int i = 0; i < 10; i++) {
            process(i);
        }
    }
}
"""
        test_file.write_text(source)

        func = FunctionInfo(
            name="process",
            file_path=tmp_path / "Processor.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="performance",
        )

        expected = """import org.junit.jupiter.api.Test;

public class BraceTest__perfonlyinstrumented {
    @Test
    public void testOne() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "BraceTest";
        String _cf_cls1 = "BraceTest";
        String _cf_fn1 = "process";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            if (true) {
                doSomething();
            }
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }

    @Test
    public void testTwo() {
        // Codeflash timing instrumentation
        int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter2 = 2;
        String _cf_mod2 = "BraceTest";
        String _cf_cls2 = "BraceTest";
        String _cf_fn2 = "process";
        System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + "######$!");
        long _cf_start2 = System.nanoTime();
        try {
            for (int i = 0; i < 10; i++) {
                process(i);
            }
        } finally {
            long _cf_end2 = System.nanoTime();
            long _cf_dur2 = _cf_end2 - _cf_start2;
            System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + ":" + _cf_dur2 + "######!");
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_instrumented_code_preserves_imports(self, tmp_path: Path):
        """Test that imports are preserved in instrumented code."""
        test_file = tmp_path / "ImportTest.java"
        source = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import java.util.ArrayList;

public class ImportTest {
    @Test
    public void testCollections() {
        List<String> list = new ArrayList<>();
        assertEquals(0, list.size());
    }
}
"""
        test_file.write_text(source)

        func = FunctionInfo(
            name="size",
            file_path=tmp_path / "Collection.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="performance",
        )

        expected = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import java.util.ArrayList;

public class ImportTest__perfonlyinstrumented {
    @Test
    public void testCollections() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "ImportTest";
        String _cf_cls1 = "ImportTest";
        String _cf_fn1 = "size";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            List<String> list = new ArrayList<>();
            assertEquals(0, list.size());
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }
}
"""
        assert success is True
        assert result == expected


class TestEdgeCases:
    """Edge cases for Java instrumentation."""

    def test_empty_test_method(self, tmp_path: Path):
        """Test instrumenting an empty test method."""
        test_file = tmp_path / "EmptyTest.java"
        source = """import org.junit.jupiter.api.Test;

public class EmptyTest {
    @Test
    public void testEmpty() {
    }
}
"""
        test_file.write_text(source)

        func = FunctionInfo(
            name="empty",
            file_path=tmp_path / "Empty.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="performance",
        )

        expected = """import org.junit.jupiter.api.Test;

public class EmptyTest__perfonlyinstrumented {
    @Test
    public void testEmpty() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "EmptyTest";
        String _cf_cls1 = "EmptyTest";
        String _cf_fn1 = "empty";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_test_with_nested_braces(self, tmp_path: Path):
        """Test instrumenting code with nested braces."""
        test_file = tmp_path / "NestedTest.java"
        source = """import org.junit.jupiter.api.Test;

public class NestedTest {
    @Test
    public void testNested() {
        if (condition) {
            for (int i = 0; i < 10; i++) {
                if (i > 5) {
                    process(i);
                }
            }
        }
    }
}
"""
        test_file.write_text(source)

        func = FunctionInfo(
            name="process",
            file_path=tmp_path / "Processor.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="performance",
        )

        expected = """import org.junit.jupiter.api.Test;

public class NestedTest__perfonlyinstrumented {
    @Test
    public void testNested() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "NestedTest";
        String _cf_cls1 = "NestedTest";
        String _cf_fn1 = "process";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            if (condition) {
                for (int i = 0; i < 10; i++) {
                    if (i > 5) {
                        process(i);
                    }
                }
            }
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_class_with_inner_class(self, tmp_path: Path):
        """Test instrumenting test class with inner class."""
        test_file = tmp_path / "InnerClassTest.java"
        source = """import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

public class InnerClassTest {
    @Test
    public void testOuter() {
        outerMethod();
    }

    @Nested
    class InnerTests {
        @Test
        public void testInner() {
            innerMethod();
        }
    }
}
"""
        test_file.write_text(source)

        func = FunctionInfo(
            name="testMethod",
            file_path=tmp_path / "Target.java",
            start_line=1,
            end_line=5,
            parents=(),
            is_method=True,
            language=Language.JAVA,
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="performance",
        )

        expected = """import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

public class InnerClassTest__perfonlyinstrumented {
    @Test
    public void testOuter() {
        // Codeflash timing instrumentation
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_iter1 = 1;
        String _cf_mod1 = "InnerClassTest";
        String _cf_cls1 = "InnerClassTest";
        String _cf_fn1 = "testMethod";
        System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + "######$!");
        long _cf_start1 = System.nanoTime();
        try {
            outerMethod();
        } finally {
            long _cf_end1 = System.nanoTime();
            long _cf_dur1 = _cf_end1 - _cf_start1;
            System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_iter1 + ":" + _cf_dur1 + "######!");
        }
    }

    @Nested
    class InnerTests {
        @Test
        public void testInner() {
            // Codeflash timing instrumentation
            int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
            int _cf_iter2 = 2;
            String _cf_mod2 = "InnerClassTest";
            String _cf_cls2 = "InnerClassTest";
            String _cf_fn2 = "testMethod";
            System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + "######$!");
            long _cf_start2 = System.nanoTime();
            try {
                innerMethod();
            } finally {
                long _cf_end2 = System.nanoTime();
                long _cf_dur2 = _cf_end2 - _cf_start2;
                System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_iter2 + ":" + _cf_dur2 + "######!");
            }
        }
    }
}
"""
        assert success is True
        assert result == expected
