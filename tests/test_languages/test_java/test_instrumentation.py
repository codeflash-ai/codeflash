"""Tests for Java code instrumentation.

Tests the instrumentation functions with exact string equality assertions
to ensure the generated code matches expected output exactly.

Also includes end-to-end execution tests that:
1. Instrument Java code
2. Execute with Maven
3. Parse JUnit XML and timing markers from stdout
4. Verify the parsed results are correct
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# Set API key for tests that instantiate Optimizer
os.environ["CODEFLASH_API_KEY"] = "cf-test-key"

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import Language
from codeflash.languages.current import set_current_language
from codeflash.models.function_types import FunctionParent
from codeflash.languages.java.build_tools import find_maven_executable
from codeflash.languages.java.discovery import discover_functions_from_source
from codeflash.languages.java.instrumentation import (
    _add_behavior_instrumentation,
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
        func = FunctionToOptimize(
            function_name="add",
            file_path=Path("Calculator.java"),
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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

        func = FunctionToOptimize(
            function_name="add",
            file_path=tmp_path / "Calculator.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="behavior",
        )

        assert success is True

        # Behavior mode now adds SQLite instrumentation
        # Verify key elements are present
        assert "import java.sql.Connection;" in result
        assert "import java.sql.DriverManager;" in result
        assert "import java.sql.PreparedStatement;" in result
        # Note: java.sql.Statement is used fully qualified to avoid conflicts with other Statement classes
        assert "java.sql.Statement" in result
        assert "class CalculatorTest__perfinstrumented" in result
        assert "CODEFLASH_OUTPUT_FILE" in result
        assert "CREATE TABLE IF NOT EXISTS test_results" in result
        assert "INSERT INTO test_results VALUES" in result
        assert "_cf_loop1" in result
        assert "_cf_iter1" in result
        assert "System.nanoTime()" in result
        assert "com.codeflash.Serializer.serialize((Object)" in result

    def test_instrument_behavior_mode_assert_throws_expression_lambda(self, tmp_path: Path):
        """Test that assertThrows expression lambdas are not broken by behavior instrumentation.

        When a target function call is inside an expression lambda (e.g., () -> Fibonacci.fibonacci(-1)),
        the instrumentation must NOT wrap it in a variable assignment, as that would turn
        the void-compatible lambda into a value-returning lambda and break compilation.
        """
        test_file = tmp_path / "FibonacciTest.java"
        source = """import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeInput_ThrowsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> Fibonacci.fibonacci(-1));
    }

    @Test
    void testZeroInput_ReturnsZero() {
        assertEquals(0L, Fibonacci.fibonacci(0));
    }
}
"""
        test_file.write_text(source)

        func = FunctionToOptimize(
            function_name="fibonacci",
            file_path=tmp_path / "Fibonacci.java",
            starting_line=1,
            ending_line=10,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="behavior",
        )

        assert success is True
        # The assertThrows lambda line should remain unchanged (not wrapped in variable assignment)
        assert "() -> Fibonacci.fibonacci(-1)" in result
        # The non-lambda call should still be wrapped
        assert "_cf_result" in result

    def test_instrument_behavior_mode_assert_throws_block_lambda(self, tmp_path: Path):
        """Test that assertThrows block lambdas are not broken by behavior instrumentation.

        When a target function call is inside a block lambda (e.g., () -> { func(); }),
        the instrumentation must NOT wrap it in a variable assignment.
        """
        test_file = tmp_path / "FibonacciTest.java"
        source = """import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    void testNegativeInput_ThrowsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> {
            Fibonacci.fibonacci(-1);
        });
    }

    @Test
    void testZeroInput_ReturnsZero() {
        assertEquals(0L, Fibonacci.fibonacci(0));
    }
}
"""
        test_file.write_text(source)

        func = FunctionToOptimize(
            function_name="fibonacci",
            file_path=tmp_path / "Fibonacci.java",
            starting_line=1,
            ending_line=10,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="behavior",
        )

        assert success is True
        assert "Fibonacci.fibonacci(-1);" in result
        assert "() -> {" in result
        lines_with_cf_result = [l for l in result.split("\n") if "var _cf_result" in l and "Fibonacci.fibonacci(0)" in l]
        assert len(lines_with_cf_result) > 0, "Non-lambda call to fibonacci(0) should be wrapped"

    def test_instrument_performance_mode_simple(self, tmp_path: Path):
        """Test instrumenting a simple test in performance mode with inner loop."""
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

        func = FunctionToOptimize(
            function_name="add",
            file_path=tmp_path / "Calculator.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "CalculatorTest";
        String _cf_cls1 = "CalculatorTest";
        String _cf_fn1 = "add";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                Calculator calc = new Calculator();
                assertEquals(4, calc.add(2, 2));
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_instrument_performance_mode_multiple_tests(self, tmp_path: Path):
        """Test instrumenting multiple test methods in performance mode with inner loop."""
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

        func = FunctionToOptimize(
            function_name="calculate",
            file_path=tmp_path / "Math.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "MathTest";
        String _cf_cls1 = "MathTest";
        String _cf_fn1 = "calculate";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                assertEquals(4, add(2, 2));
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }

    @Test
    public void testSubtract() {
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations2 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod2 = "MathTest";
        String _cf_cls2 = "MathTest";
        String _cf_fn2 = "calculate";

        for (int _cf_i2 = 0; _cf_i2 < _cf_innerIterations2; _cf_i2++) {
            System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + "######$!");
            long _cf_start2 = System.nanoTime();
            try {
                assertEquals(0, subtract(2, 2));
            } finally {
                long _cf_end2 = System.nanoTime();
                long _cf_dur2 = _cf_end2 - _cf_start2;
                System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + ":" + _cf_dur2 + "######!");
            }
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_instrument_preserves_annotations(self, tmp_path: Path):
        """Test that annotations other than @Test are preserved with inner loop."""
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

        func = FunctionToOptimize(
            function_name="call",
            file_path=tmp_path / "Service.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "ServiceTest";
        String _cf_cls1 = "ServiceTest";
        String _cf_fn1 = "call";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                service.call();
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }

    @Disabled
    @Test
    public void testDisabled() {
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations2 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod2 = "ServiceTest";
        String _cf_cls2 = "ServiceTest";
        String _cf_fn2 = "call";

        for (int _cf_i2 = 0; _cf_i2 < _cf_innerIterations2; _cf_i2++) {
            System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + "######$!");
            long _cf_start2 = System.nanoTime();
            try {
                service.other();
            } finally {
                long _cf_end2 = System.nanoTime();
                long _cf_dur2 = _cf_end2 - _cf_start2;
                System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + ":" + _cf_dur2 + "######!");
            }
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_missing_file(self, tmp_path: Path):
        """Test handling missing test file."""
        test_file = tmp_path / "NonExistent.java"

        func = FunctionToOptimize(
            function_name="add",
            file_path=tmp_path / "Calculator.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )

        success, result = instrument_existing_test(
            test_file,
            call_positions=[],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode="behavior",
        )

        assert success is False


class TestKryoSerializerUsage:
    """Tests for Kryo Serializer usage in behavior mode."""

    def test_serializer_used_for_return_values(self):
        """Test that captured return values use com.codeflash.Serializer.serialize()."""
        source = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    public void testFoo() {
        assertEquals(0, obj.foo());
    }
}
"""
        result = _add_behavior_instrumentation(source, "MyTest", "foo")

        assert "com.codeflash.Serializer.serialize((Object)" in result
        # Should NOT use old _cfSerialize helper
        assert "_cfSerialize" not in result

    def test_byte_array_result_variable(self):
        """Test that the serialized result variable is byte[] not String."""
        source = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    public void testFoo() {
        assertEquals(0, obj.foo());
    }
}
"""
        result = _add_behavior_instrumentation(source, "MyTest", "foo")

        assert "byte[] _cf_serializedResult" in result
        assert "String _cf_serializedResult" not in result

    def test_blob_column_in_schema(self):
        """Test that the SQLite schema uses BLOB for return_value column."""
        source = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    public void testFoo() {
        assertEquals(0, obj.foo());
    }
}
"""
        result = _add_behavior_instrumentation(source, "MyTest", "foo")

        assert "return_value BLOB" in result
        assert "return_value TEXT" not in result

    def test_set_bytes_for_blob_write(self):
        """Test that setBytes is used to write BLOB data to SQLite."""
        source = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    public void testFoo() {
        assertEquals(0, obj.foo());
    }
}
"""
        result = _add_behavior_instrumentation(source, "MyTest", "foo")

        assert "setBytes(8, _cf_serializedResult" in result
        # Should NOT use setString for return value
        assert "setString(8, _cf_serializedResult" not in result

    def test_no_inline_helper_injected(self):
        """Test that no inline _cfSerialize helper method is injected."""
        source = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    public void testFoo() {
        assertEquals(0, obj.foo());
    }
}
"""
        result = _add_behavior_instrumentation(source, "MyTest", "foo")

        assert "private static String _cfSerialize" not in result

    def test_serializer_not_used_in_performance_mode(self):
        """Test that Serializer is NOT used in performance mode (only behavior)."""
        source = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    public void testFoo() {
        assertEquals(0, obj.foo());
    }
}
"""
        result = _add_timing_instrumentation(source, "MyTest", "foo")

        assert "Serializer.serialize" not in result
        assert "_cfSerialize" not in result


class TestAddTimingInstrumentation:
    """Tests for _add_timing_instrumentation helper function with inner loop."""

    def test_single_test_method(self):
        """Test timing instrumentation for a single test method with inner loop."""
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "SimpleTest";
        String _cf_cls1 = "SimpleTest";
        String _cf_fn1 = "targetFunc";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                doSomething();
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }
}
"""
        assert result == expected

    def test_multiple_test_methods(self):
        """Test timing instrumentation for multiple test methods with inner loop."""
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "MultiTest";
        String _cf_cls1 = "MultiTest";
        String _cf_fn1 = "func";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                first();
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }

    @Test
    public void testSecond() {
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations2 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod2 = "MultiTest";
        String _cf_cls2 = "MultiTest";
        String _cf_fn2 = "func";

        for (int _cf_i2 = 0; _cf_i2 < _cf_innerIterations2; _cf_i2++) {
            System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + "######$!");
            long _cf_start2 = System.nanoTime();
            try {
                second();
            } finally {
                long _cf_end2 = System.nanoTime();
                long _cf_dur2 = _cf_end2 - _cf_start2;
                System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + ":" + _cf_dur2 + "######!");
            }
        }
    }
}
"""
        assert result == expected

    def test_timing_markers_format(self):
        """Test that timing markers have the correct format with inner loop."""
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "TestClass";
        String _cf_cls1 = "TestClass";
        String _cf_fn1 = "targetMethod";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                action();
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }
}
"""
        assert result == expected


class TestCreateBenchmarkTest:
    """Tests for create_benchmark_test."""

    def test_create_benchmark(self):
        """Test creating a benchmark test."""
        func = FunctionToOptimize(
            function_name="add",
            file_path=Path("Calculator.java"),
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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
        func = FunctionToOptimize(
            function_name="multiply",
            file_path=Path("Math.java"),
            starting_line=1,
            ending_line=3,
            parents=[],
            is_method=True,
            language="java",
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
        """Test instrumenting generated test in behavior mode.

        Behavior mode should:
        1. Remove assertions containing the target function call
        2. Capture the function return value instead
        3. Rename the class with __perfinstrumented suffix
        """
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

        # Behavior mode transforms assertions to capture return values
        expected = """import org.junit.jupiter.api.Test;

public class CalculatorTest__perfinstrumented {
    @Test
    public void testAdd() {
        Object _cf_result1 = new Calculator().add(2, 2);
    }
}
"""
        assert result == expected

    def test_instrument_generated_test_performance_mode(self):
        """Test instrumenting generated test in performance mode with inner loop."""
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "GeneratedTest";
        String _cf_cls1 = "GeneratedTest";
        String _cf_fn1 = "method";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                target.method();
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
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

    def test_inner_loop_timing_markers(self):
        """Test parsing timing markers from inner loop iterations.

        With the inner loop, each test method produces N timing markers (one per iteration).
        The iterationId (5th field) now represents the inner iteration number (0, 1, 2, ..., N-1).
        """
        # Simulate stdout from 3 inner iterations (inner_iterations=3)
        stdout = """
!$######Module:Class:func:1:0######$!
iteration 0
!######Module:Class:func:1:0:150000######!
!$######Module:Class:func:1:1######$!
iteration 1
!######Module:Class:func:1:1:50000######!
!$######Module:Class:func:1:2######$!
iteration 2
!######Module:Class:func:1:2:45000######!
"""
        start_pattern = re.compile(r"!\$######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+)######\$!")
        end_pattern = re.compile(r"!######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+):([^:]+)######!")

        start_matches = start_pattern.findall(stdout)
        end_matches = end_pattern.findall(stdout)

        # Should have 3 start and 3 end markers (one per inner iteration)
        assert len(start_matches) == 3
        assert len(end_matches) == 3

        # All markers should have the same loopIndex (1) but different iterationIds (0, 1, 2)
        for i, (start, end) in enumerate(zip(start_matches, end_matches)):
            assert start[3] == "1"  # loopIndex
            assert start[4] == str(i)  # iterationId (0, 1, 2)
            assert end[3] == "1"  # loopIndex
            assert end[4] == str(i)  # iterationId (0, 1, 2)

        # Verify durations - iteration 0 is slower (JIT warmup), iterations 1 and 2 are faster
        durations = [int(m[5]) for m in end_matches]
        assert durations == [150000, 50000, 45000]

        # Min runtime logic would select 45000ns (the fastest iteration after JIT warmup)
        min_runtime = min(durations)
        assert min_runtime == 45000


class TestInstrumentedCodeValidity:
    """Tests to verify that instrumented code is syntactically valid Java with inner loop."""

    def test_instrumented_code_has_balanced_braces(self, tmp_path: Path):
        """Test that instrumented code has balanced braces with inner loop."""
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

        func = FunctionToOptimize(
            function_name="process",
            file_path=tmp_path / "Processor.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "BraceTest";
        String _cf_cls1 = "BraceTest";
        String _cf_fn1 = "process";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                if (true) {
                    doSomething();
                }
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }

    @Test
    public void testTwo() {
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations2 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod2 = "BraceTest";
        String _cf_cls2 = "BraceTest";
        String _cf_fn2 = "process";

        for (int _cf_i2 = 0; _cf_i2 < _cf_innerIterations2; _cf_i2++) {
            System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + "######$!");
            long _cf_start2 = System.nanoTime();
            try {
                for (int i = 0; i < 10; i++) {
                    process(i);
                }
            } finally {
                long _cf_end2 = System.nanoTime();
                long _cf_dur2 = _cf_end2 - _cf_start2;
                System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + ":" + _cf_dur2 + "######!");
            }
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_instrumented_code_preserves_imports(self, tmp_path: Path):
        """Test that imports are preserved in instrumented code with inner loop."""
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

        func = FunctionToOptimize(
            function_name="size",
            file_path=tmp_path / "Collection.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "ImportTest";
        String _cf_cls1 = "ImportTest";
        String _cf_fn1 = "size";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                List<String> list = new ArrayList<>();
                assertEquals(0, list.size());
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }
}
"""
        assert success is True
        assert result == expected


class TestEdgeCases:
    """Edge cases for Java instrumentation with inner loop."""

    def test_empty_test_method(self, tmp_path: Path):
        """Test instrumenting an empty test method with inner loop."""
        test_file = tmp_path / "EmptyTest.java"
        source = """import org.junit.jupiter.api.Test;

public class EmptyTest {
    @Test
    public void testEmpty() {
    }
}
"""
        test_file.write_text(source)

        func = FunctionToOptimize(
            function_name="empty",
            file_path=tmp_path / "Empty.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "EmptyTest";
        String _cf_cls1 = "EmptyTest";
        String _cf_fn1 = "empty";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_test_with_nested_braces(self, tmp_path: Path):
        """Test instrumenting code with nested braces with inner loop."""
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

        func = FunctionToOptimize(
            function_name="process",
            file_path=tmp_path / "Processor.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "NestedTest";
        String _cf_cls1 = "NestedTest";
        String _cf_fn1 = "process";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
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
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }
}
"""
        assert success is True
        assert result == expected

    def test_class_with_inner_class(self, tmp_path: Path):
        """Test instrumenting test class with inner class with inner loop."""
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

        func = FunctionToOptimize(
            function_name="testMethod",
            file_path=tmp_path / "Target.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
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
        // Codeflash timing instrumentation with inner loop for JIT warmup
        int _cf_loop1 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
        int _cf_innerIterations1 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
        String _cf_mod1 = "InnerClassTest";
        String _cf_cls1 = "InnerClassTest";
        String _cf_fn1 = "testMethod";

        for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++) {
            System.out.println("!$######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + "######$!");
            long _cf_start1 = System.nanoTime();
            try {
                outerMethod();
            } finally {
                long _cf_end1 = System.nanoTime();
                long _cf_dur1 = _cf_end1 - _cf_start1;
                System.out.println("!######" + _cf_mod1 + ":" + _cf_cls1 + ":" + _cf_fn1 + ":" + _cf_loop1 + ":" + _cf_i1 + ":" + _cf_dur1 + "######!");
            }
        }
    }

    @Nested
    class InnerTests {
        @Test
        public void testInner() {
            // Codeflash timing instrumentation with inner loop for JIT warmup
            int _cf_loop2 = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));
            int _cf_innerIterations2 = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));
            String _cf_mod2 = "InnerClassTest";
            String _cf_cls2 = "InnerClassTest";
            String _cf_fn2 = "testMethod";

            for (int _cf_i2 = 0; _cf_i2 < _cf_innerIterations2; _cf_i2++) {
                System.out.println("!$######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + "######$!");
                long _cf_start2 = System.nanoTime();
                try {
                    innerMethod();
                } finally {
                    long _cf_end2 = System.nanoTime();
                    long _cf_dur2 = _cf_end2 - _cf_start2;
                    System.out.println("!######" + _cf_mod2 + ":" + _cf_cls2 + ":" + _cf_fn2 + ":" + _cf_loop2 + ":" + _cf_i2 + ":" + _cf_dur2 + "######!");
                }
            }
        }
    }
}
"""
        assert success is True
        assert result == expected


# Skip all E2E tests if Maven is not available
requires_maven = pytest.mark.skipif(
    find_maven_executable() is None,
    reason="Maven not found - skipping execution tests",
)


@requires_maven
class TestRunAndParseTests:
    """End-to-end tests using the real run_and_parse_tests entry point."""

    POM_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>codeflash-test</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.junit.platform</groupId>
            <artifactId>junit-platform-console-standalone</artifactId>
            <version>1.9.3</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.xerial</groupId>
            <artifactId>sqlite-jdbc</artifactId>
            <version>3.44.1.0</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>com.google.code.gson</groupId>
            <artifactId>gson</artifactId>
            <version>2.10.1</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>com.codeflash</groupId>
            <artifactId>codeflash-runtime</artifactId>
            <version>1.0.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.1.2</version>
                <configuration>
                    <redirectTestOutputToFile>false</redirectTestOutputToFile>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
"""

    @pytest.fixture
    def java_project(self, tmp_path: Path):
        """Create a temporary Maven project and set up Java language context."""
        from codeflash.languages.base import Language
        from codeflash.languages.current import set_current_language

        # Force set the language to Java (reset the singleton first)
        import codeflash.languages.current as current_module
        current_module._current_language = None
        set_current_language(Language.JAVA)

        # Create Maven project structure
        src_dir = tmp_path / "src" / "main" / "java" / "com" / "example"
        test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
        src_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        (tmp_path / "pom.xml").write_text(self.POM_CONTENT, encoding="utf-8")

        yield tmp_path, src_dir, test_dir

        # Reset language back to Python
        current_module._current_language = None
        set_current_language(Language.PYTHON)

    def test_run_and_parse_behavior_mode(self, java_project):
        """Test run_and_parse_tests in BEHAVIOR mode."""
        from argparse import Namespace

        from codeflash.discovery.functions_to_optimize import FunctionToOptimize
        from codeflash.models.models import TestFile, TestFiles, TestingMode, TestType
        from codeflash.optimization.optimizer import Optimizer

        project_root, src_dir, test_dir = java_project

        # Create source file
        (src_dir / "Calculator.java").write_text("""package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
""", encoding="utf-8")

        # Create and instrument test
        test_source = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
"""
        test_file = test_dir / "CalculatorTest.java"
        test_file.write_text(test_source, encoding="utf-8")

        func_info = FunctionToOptimize(
            function_name="add",
            file_path=src_dir / "Calculator.java",
            starting_line=4,
            ending_line=6,
            parents=[],
            is_method=True,
            language="java",
        )

        success, instrumented = instrument_existing_test(
            test_file, [], func_info, test_dir, mode="behavior"
        )
        assert success

        instrumented_file = test_dir / "CalculatorTest__perfinstrumented.java"
        instrumented_file.write_text(instrumented, encoding="utf-8")

        # Create Optimizer and FunctionOptimizer
        fto = FunctionToOptimize(
            function_name="add",
            file_path=src_dir / "Calculator.java",
            parents=[],
            language="java",
        )

        opt = Optimizer(Namespace(
            project_root=project_root,
            disable_telemetry=True,
            tests_root=test_dir,
            test_project_root=project_root,
            pytest_cmd="pytest",
            experiment_id=None,
        ))

        func_optimizer = opt.create_function_optimizer(fto)
        assert func_optimizer is not None

        func_optimizer.test_files = TestFiles(test_files=[
            TestFile(
                instrumented_behavior_file_path=instrumented_file,
                test_type=TestType.EXISTING_UNIT_TEST,
                original_file_path=test_file,
                benchmarking_file_path=instrumented_file,  # Use same file for behavior tests
            )
        ])

        # Run and parse tests
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Verify results
        assert len(test_results.test_results) >= 1
        result = test_results.test_results[0]
        assert result.did_pass is True
        assert result.runtime is not None
        assert result.runtime > 0

    def test_run_and_parse_performance_mode(self, java_project):
        """Test run_and_parse_tests in PERFORMANCE mode with inner loop timing.

        This test verifies the complete performance benchmarking flow:
        1. Instruments test with inner loop for JIT warmup
        2. Runs with inner_iterations=2 (fast test)
        3. Validates multiple timing markers are produced (one per inner iteration)
        4. Validates parsed results contain timing data
        """
        from argparse import Namespace

        from codeflash.discovery.functions_to_optimize import FunctionToOptimize
        from codeflash.models.models import TestFile, TestFiles, TestingMode, TestType
        from codeflash.optimization.optimizer import Optimizer

        project_root, src_dir, test_dir = java_project

        # Create source file
        (src_dir / "MathUtils.java").write_text("""package com.example;

public class MathUtils {
    public int multiply(int a, int b) {
        return a * b;
    }
}
""", encoding="utf-8")

        # Create and instrument test
        test_source = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MathUtilsTest {
    @Test
    public void testMultiply() {
        MathUtils math = new MathUtils();
        assertEquals(6, math.multiply(2, 3));
    }
}
"""
        test_file = test_dir / "MathUtilsTest.java"
        test_file.write_text(test_source, encoding="utf-8")

        func_info = FunctionToOptimize(
            function_name="multiply",
            file_path=src_dir / "MathUtils.java",
            starting_line=4,
            ending_line=6,
            parents=[],
            is_method=True,
            language="java",
        )

        success, instrumented = instrument_existing_test(
            test_file, [], func_info, test_dir, mode="performance"
        )
        assert success

        # Verify instrumented code contains inner loop for JIT warmup
        assert "CODEFLASH_INNER_ITERATIONS" in instrumented, "Performance mode should use inner loop"
        assert "for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++)" in instrumented

        instrumented_file = test_dir / "MathUtilsTest__perfonlyinstrumented.java"
        instrumented_file.write_text(instrumented, encoding="utf-8")

        # Create Optimizer and FunctionOptimizer
        fto = FunctionToOptimize(
            function_name="multiply",
            file_path=src_dir / "MathUtils.java",
            parents=[],
            language="java",
        )

        opt = Optimizer(Namespace(
            project_root=project_root,
            disable_telemetry=True,
            tests_root=test_dir,
            test_project_root=project_root,
            pytest_cmd="pytest",
            experiment_id=None,
        ))

        func_optimizer = opt.create_function_optimizer(fto)
        assert func_optimizer is not None

        func_optimizer.test_files = TestFiles(test_files=[
            TestFile(
                instrumented_behavior_file_path=test_file,
                test_type=TestType.EXISTING_UNIT_TEST,
                original_file_path=test_file,
                benchmarking_file_path=instrumented_file,
            )
        ])

        # Run performance tests with inner_iterations=2 for fast test
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_INNER_ITERATIONS"] = "2"  # Only 2 inner iterations for fast test

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,  # Only 1 outer loop (Maven invocation)
            testing_time=1.0,
        )

        # Should have 2 results (one per inner iteration)
        assert len(test_results.test_results) >= 2, (
            f"Expected at least 2 results from inner loop (inner_iterations=2), got {len(test_results.test_results)}"
        )

        # All results should pass with valid timing
        runtimes = []
        for result in test_results.test_results:
            assert result.did_pass is True
            assert result.runtime is not None
            assert result.runtime > 0
            runtimes.append(result.runtime)

        # Verify we have multiple timing measurements
        assert len(runtimes) >= 2, f"Expected at least 2 runtimes, got {len(runtimes)}"

        # Log runtime info (min would be selected for benchmarking comparison)
        min_runtime = min(runtimes)
        max_runtime = max(runtimes)
        print(f"Inner loop runtimes: min={min_runtime}ns, max={max_runtime}ns, count={len(runtimes)}")

    def test_run_and_parse_multiple_test_methods(self, java_project):
        """Test run_and_parse_tests with multiple test methods."""
        from argparse import Namespace

        from codeflash.discovery.functions_to_optimize import FunctionToOptimize
        from codeflash.models.models import TestFile, TestFiles, TestingMode, TestType
        from codeflash.optimization.optimizer import Optimizer

        project_root, src_dir, test_dir = java_project

        # Create source file
        (src_dir / "StringUtils.java").write_text("""package com.example;

public class StringUtils {
    public String reverse(String s) {
        return new StringBuilder(s).reverse().toString();
    }
}
""", encoding="utf-8")

        # Create test with multiple methods
        test_source = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class StringUtilsTest {
    @Test
    public void testReverseHello() {
        assertEquals("olleh", new StringUtils().reverse("hello"));
    }

    @Test
    public void testReverseEmpty() {
        assertEquals("", new StringUtils().reverse(""));
    }

    @Test
    public void testReverseSingle() {
        assertEquals("a", new StringUtils().reverse("a"));
    }
}
"""
        test_file = test_dir / "StringUtilsTest.java"
        test_file.write_text(test_source, encoding="utf-8")

        func_info = FunctionToOptimize(
            function_name="reverse",
            file_path=src_dir / "StringUtils.java",
            starting_line=4,
            ending_line=6,
            parents=[],
            is_method=True,
            language="java",
        )

        success, instrumented = instrument_existing_test(
            test_file, [], func_info, test_dir, mode="behavior"
        )
        assert success

        instrumented_file = test_dir / "StringUtilsTest__perfinstrumented.java"
        instrumented_file.write_text(instrumented, encoding="utf-8")

        fto = FunctionToOptimize(
            function_name="reverse",
            file_path=src_dir / "StringUtils.java",
            parents=[],
            language="java",
        )

        opt = Optimizer(Namespace(
            project_root=project_root,
            disable_telemetry=True,
            tests_root=test_dir,
            test_project_root=project_root,
            pytest_cmd="pytest",
            experiment_id=None,
        ))

        func_optimizer = opt.create_function_optimizer(fto)
        func_optimizer.test_files = TestFiles(test_files=[
            TestFile(
                instrumented_behavior_file_path=instrumented_file,
                test_type=TestType.EXISTING_UNIT_TEST,
                original_file_path=test_file,
                benchmarking_file_path=instrumented_file,  # Use same file for behavior tests
            )
        ])

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Should have results for test methods - at least 1 from JUnit XML parsing
        # Note: With behavior mode instrumentation, all 3 tests should be parsed
        assert len(test_results.test_results) >= 1, (
            f"Expected at least 1 test result but got {len(test_results.test_results)}"
        )
        for result in test_results.test_results:
            assert result.did_pass is True, f"Test {result.id.test_function_name} should have passed"

    def test_run_and_parse_failing_test(self, java_project):
        """Test run_and_parse_tests correctly reports failing tests."""
        from argparse import Namespace

        from codeflash.discovery.functions_to_optimize import FunctionToOptimize
        from codeflash.models.models import TestFile, TestFiles, TestingMode, TestType
        from codeflash.optimization.optimizer import Optimizer

        project_root, src_dir, test_dir = java_project

        # Create source file with a bug
        (src_dir / "BrokenCalc.java").write_text("""package com.example;

public class BrokenCalc {
    public int add(int a, int b) {
        return a + b + 1;  // Bug: adds extra 1
    }
}
""", encoding="utf-8")

        # Create test that will fail
        test_source = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class BrokenCalcTest {
    @Test
    public void testAdd() {
        BrokenCalc calc = new BrokenCalc();
        assertEquals(4, calc.add(2, 2));  // Will fail: 5 != 4
    }
}
"""
        test_file = test_dir / "BrokenCalcTest.java"
        test_file.write_text(test_source, encoding="utf-8")

        func_info = FunctionToOptimize(
            function_name="add",
            file_path=src_dir / "BrokenCalc.java",
            starting_line=4,
            ending_line=6,
            parents=[],
            is_method=True,
            language="java",
        )

        success, instrumented = instrument_existing_test(
            test_file, [], func_info, test_dir, mode="behavior"
        )
        assert success

        instrumented_file = test_dir / "BrokenCalcTest__perfinstrumented.java"
        instrumented_file.write_text(instrumented, encoding="utf-8")

        fto = FunctionToOptimize(
            function_name="add",
            file_path=src_dir / "BrokenCalc.java",
            parents=[],
            language="java",
        )

        opt = Optimizer(Namespace(
            project_root=project_root,
            disable_telemetry=True,
            tests_root=test_dir,
            test_project_root=project_root,
            pytest_cmd="pytest",
            experiment_id=None,
        ))

        func_optimizer = opt.create_function_optimizer(fto)
        func_optimizer.test_files = TestFiles(test_files=[
            TestFile(
                instrumented_behavior_file_path=instrumented_file,
                test_type=TestType.EXISTING_UNIT_TEST,
                original_file_path=test_file,
                benchmarking_file_path=instrumented_file,  # Use same file for behavior tests
            )
        ])

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Should have result for the failing test
        assert len(test_results.test_results) >= 1
        result = test_results.test_results[0]
        assert result.did_pass is False

    def test_behavior_mode_writes_to_sqlite(self, java_project):
        """Test that behavior mode correctly writes results to SQLite file."""
        import sqlite3

        from argparse import Namespace

        from codeflash.code_utils.code_utils import get_run_tmp_file
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize
        from codeflash.models.models import TestFile, TestFiles, TestingMode, TestType
        from codeflash.optimization.optimizer import Optimizer

        # Clean up any existing SQLite files from previous tests
        sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
        if sqlite_file.exists():
            sqlite_file.unlink()

        project_root, src_dir, test_dir = java_project

        # Create source file
        (src_dir / "Counter.java").write_text("""package com.example;

public class Counter {
    private int value = 0;

    public int increment() {
        return ++value;
    }
}
""", encoding="utf-8")

        # Create test file - single test method for simplicity
        test_source = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CounterTest {
    @Test
    public void testIncrement() {
        Counter counter = new Counter();
        assertEquals(1, counter.increment());
    }
}
"""
        test_file = test_dir / "CounterTest.java"
        test_file.write_text(test_source, encoding="utf-8")

        # Instrument for BEHAVIOR mode (this should include SQLite writing)
        func_info = FunctionToOptimize(
            function_name="increment",
            file_path=src_dir / "Counter.java",
            starting_line=6,
            ending_line=8,
            parents=[],
            is_method=True,
            language="java",
        )

        success, instrumented = instrument_existing_test(
            test_file, [], func_info, test_dir, mode="behavior"
        )
        assert success

        # Verify SQLite imports were added
        assert "import java.sql.Connection;" in instrumented
        assert "import java.sql.DriverManager;" in instrumented
        assert "import java.sql.PreparedStatement;" in instrumented

        # Verify SQLite writing code was added
        assert "CODEFLASH_OUTPUT_FILE" in instrumented
        assert "CREATE TABLE IF NOT EXISTS test_results" in instrumented
        assert "INSERT INTO test_results VALUES" in instrumented

        instrumented_file = test_dir / "CounterTest__perfinstrumented.java"
        instrumented_file.write_text(instrumented, encoding="utf-8")

        # Create Optimizer and FunctionOptimizer
        fto = FunctionToOptimize(
            function_name="increment",
            file_path=src_dir / "Counter.java",
            parents=[],
            language="java",
        )

        opt = Optimizer(Namespace(
            project_root=project_root,
            disable_telemetry=True,
            tests_root=test_dir,
            test_project_root=project_root,
            pytest_cmd="pytest",
            experiment_id=None,
        ))

        func_optimizer = opt.create_function_optimizer(fto)
        assert func_optimizer is not None

        func_optimizer.test_files = TestFiles(test_files=[
            TestFile(
                instrumented_behavior_file_path=instrumented_file,
                test_type=TestType.EXISTING_UNIT_TEST,
                original_file_path=test_file,
                benchmarking_file_path=instrumented_file,
            )
        ])

        # Run tests
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"

        test_results, _ = func_optimizer.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=func_optimizer.test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Verify tests passed - at least 1 result from JUnit XML parsing
        assert len(test_results.test_results) >= 1, (
            f"Expected at least 1 test result but got {len(test_results.test_results)}"
        )
        for result in test_results.test_results:
            assert result.did_pass is True, f"Test {result.id.test_function_name} should have passed"

        # Find the SQLite file that was created
        # SQLite is created at get_run_tmp_file path
        from codeflash.code_utils.code_utils import get_run_tmp_file
        sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))

        if not sqlite_file.exists():
            # Fall back to checking temp directory for any SQLite files
            import tempfile
            sqlite_files = list(Path(tempfile.gettempdir()).glob("**/test_return_values_*.sqlite"))
            assert len(sqlite_files) >= 1, f"SQLite file should have been created at {sqlite_file} or in temp dir"
            sqlite_file = max(sqlite_files, key=lambda p: p.stat().st_mtime)

        # Verify SQLite contents
        conn = sqlite3.connect(str(sqlite_file))
        cursor = conn.cursor()

        # Check that test_results table exists and has data
        cursor.execute("SELECT COUNT(*) FROM test_results")
        count = cursor.fetchone()[0]
        assert count >= 1, f"Expected at least 1 result in SQLite, got {count}"

        # Check the data structure
        cursor.execute("SELECT * FROM test_results")
        rows = cursor.fetchall()

        for row in rows:
            test_module_path, test_class_name, test_function_name, function_getting_tested, \
                loop_index, iteration_id, runtime, return_value, verification_type = row

            # Verify fields
            assert test_module_path == "CounterTest"
            assert test_class_name == "CounterTest"
            assert function_getting_tested == "increment"
            assert loop_index == 1
            assert runtime > 0, f"Should have a positive runtime, got {runtime}"
            assert verification_type == "function_call"  # Updated from "output"
            assert return_value is not None, "Return value should be serialized, not null"
            assert isinstance(return_value, bytes), f"Expected bytes (Kryo binary), got: {type(return_value)}"
            assert len(return_value) > 0, "Kryo-serialized return value should not be empty"

        conn.close()

    def test_performance_mode_inner_loop_timing_markers(self, java_project):
        """Test that performance mode produces multiple timing markers from inner loop.

        This test verifies that:
        1. Instrumented code runs inner_iterations=2 times
        2. Two timing markers are produced (one per inner iteration)
        3. Each marker has a unique iteration ID (0, 1)
        4. Both markers have valid durations
        """
        from codeflash.languages.java.test_runner import run_benchmarking_tests

        project_root, src_dir, test_dir = java_project

        # Create a simple function to optimize
        (src_dir / "Fibonacci.java").write_text("""package com.example;

public class Fibonacci {
    public int fib(int n) {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }
}
""", encoding="utf-8")

        # Create test file
        test_source = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {
    @Test
    public void testFib() {
        Fibonacci fib = new Fibonacci();
        assertEquals(5, fib.fib(5));
    }
}
"""
        test_file = test_dir / "FibonacciTest.java"
        test_file.write_text(test_source, encoding="utf-8")

        # Instrument for performance mode (adds inner loop)
        func_info = FunctionToOptimize(
            function_name="fib",
            file_path=src_dir / "Fibonacci.java",
            starting_line=4,
            ending_line=7,
            parents=[],
            is_method=True,
            language="java",
        )

        success, instrumented = instrument_existing_test(
            test_file, [], func_info, test_dir, mode="performance"
        )
        assert success

        # Verify instrumented code contains inner loop
        assert "CODEFLASH_INNER_ITERATIONS" in instrumented
        assert "for (int _cf_i1 = 0; _cf_i1 < _cf_innerIterations1; _cf_i1++)" in instrumented

        instrumented_file = test_dir / "FibonacciTest__perfonlyinstrumented.java"
        instrumented_file.write_text(instrumented, encoding="utf-8")

        # Run benchmarking with inner_iterations=2 (fast)
        test_env = os.environ.copy()

        # Use TestFiles-like object
        class MockTestFiles:
            def __init__(self, files):
                self.test_files = files

        class MockTestFile:
            def __init__(self, path):
                self.benchmarking_file_path = path
                self.instrumented_behavior_file_path = path

        test_files = MockTestFiles([MockTestFile(instrumented_file)])

        result_xml_path, result = run_benchmarking_tests(
            test_paths=test_files,
            test_env=test_env,
            cwd=project_root,
            timeout=120,
            project_root=project_root,
            min_loops=1,
            max_loops=1,  # Only 1 outer loop
            target_duration_seconds=1.0,
            inner_iterations=2,  # Only 2 inner iterations for fast test
        )

        # Verify the test ran successfully
        assert result.returncode == 0, f"Maven test failed: {result.stderr}"

        # Parse timing markers from stdout
        stdout = result.stdout
        start_pattern = re.compile(r"!\$######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+)######\$!")
        end_pattern = re.compile(r"!######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+):([^:]+)######!")

        start_matches = start_pattern.findall(stdout)
        end_matches = end_pattern.findall(stdout)

        # Should have 2 timing markers (inner_iterations=2)
        assert len(start_matches) == 2, f"Expected 2 start markers, got {len(start_matches)}: {start_matches}"
        assert len(end_matches) == 2, f"Expected 2 end markers, got {len(end_matches)}: {end_matches}"

        # Verify iteration IDs are 0 and 1
        iteration_ids = [m[4] for m in start_matches]
        assert "0" in iteration_ids, f"Expected iteration ID 0, got: {iteration_ids}"
        assert "1" in iteration_ids, f"Expected iteration ID 1, got: {iteration_ids}"

        # Verify all markers have the same loop index (1)
        loop_indices = [m[3] for m in start_matches]
        assert all(idx == "1" for idx in loop_indices), f"Expected all loop indices to be 1, got: {loop_indices}"

        # Verify durations are positive
        durations = [int(m[5]) for m in end_matches]
        assert all(d > 0 for d in durations), f"Expected positive durations, got: {durations}"

    def test_performance_mode_multiple_methods_inner_loop(self, java_project):
        """Test inner loop with multiple test methods.

        Each test method should run inner_iterations times independently.
        This produces 2 test methods x 2 inner iterations = 4 total timing markers.
        """
        from codeflash.languages.java.test_runner import run_benchmarking_tests

        project_root, src_dir, test_dir = java_project

        # Create a simple math class
        (src_dir / "MathOps.java").write_text("""package com.example;

public class MathOps {
    public int add(int a, int b) {
        return a + b;
    }
}
""", encoding="utf-8")

        # Create test with multiple test methods
        test_source = """package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MathOpsTest {
    @Test
    public void testAddPositive() {
        MathOps math = new MathOps();
        assertEquals(5, math.add(2, 3));
    }

    @Test
    public void testAddNegative() {
        MathOps math = new MathOps();
        assertEquals(-1, math.add(2, -3));
    }
}
"""
        test_file = test_dir / "MathOpsTest.java"
        test_file.write_text(test_source, encoding="utf-8")

        # Instrument for performance mode
        func_info = FunctionToOptimize(
            function_name="add",
            file_path=src_dir / "MathOps.java",
            starting_line=4,
            ending_line=6,
            parents=[],
            is_method=True,
            language="java",
        )

        success, instrumented = instrument_existing_test(
            test_file, [], func_info, test_dir, mode="performance"
        )
        assert success

        instrumented_file = test_dir / "MathOpsTest__perfonlyinstrumented.java"
        instrumented_file.write_text(instrumented, encoding="utf-8")

        # Run benchmarking with inner_iterations=2
        test_env = os.environ.copy()

        class MockTestFiles:
            def __init__(self, files):
                self.test_files = files

        class MockTestFile:
            def __init__(self, path):
                self.benchmarking_file_path = path
                self.instrumented_behavior_file_path = path

        test_files = MockTestFiles([MockTestFile(instrumented_file)])

        result_xml_path, result = run_benchmarking_tests(
            test_paths=test_files,
            test_env=test_env,
            cwd=project_root,
            timeout=120,
            project_root=project_root,
            min_loops=1,
            max_loops=1,
            target_duration_seconds=1.0,
            inner_iterations=2,
        )

        assert result.returncode == 0, f"Maven test failed: {result.stderr}"

        # Parse timing markers
        stdout = result.stdout
        end_pattern = re.compile(r"!######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+):([^:]+)######!")
        end_matches = end_pattern.findall(stdout)

        # Should have 4 timing markers (2 test methods x 2 inner iterations)
        assert len(end_matches) == 4, f"Expected 4 end markers, got {len(end_matches)}: {end_matches}"

        # Count markers per iteration ID
        iter_0_count = sum(1 for m in end_matches if m[4] == "0")
        iter_1_count = sum(1 for m in end_matches if m[4] == "1")

        assert iter_0_count == 2, f"Expected 2 markers for iteration 0, got {iter_0_count}"
        assert iter_1_count == 2, f"Expected 2 markers for iteration 1, got {iter_1_count}"
