"""Tests for Java code instrumentation."""

from pathlib import Path

import pytest

from codeflash.languages.base import FunctionInfo, Language
from codeflash.languages.java.discovery import discover_functions_from_source
from codeflash.languages.java.instrumentation import (
    create_benchmark_test,
    instrument_existing_test,
    instrument_for_behavior,
    instrument_for_benchmarking,
    remove_instrumentation,
)


class TestInstrumentForBehavior:
    """Tests for instrument_for_behavior."""

    def test_adds_import(self):
        """Test that CodeFlash import is added."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        functions = discover_functions_from_source(source)
        result = instrument_for_behavior(source, functions)

        assert "import com.codeflash" in result

    def test_no_functions_unchanged(self):
        """Test that source is unchanged when no functions provided."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        result = instrument_for_behavior(source, [])
        assert result == source


class TestInstrumentForBenchmarking:
    """Tests for instrument_for_benchmarking."""

    def test_adds_benchmark_imports(self):
        """Test that benchmark imports are added."""
        source = """
import org.junit.jupiter.api.Test;

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
        # Should preserve original content
        assert "testAdd" in result


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
        func.__dict__["class_name"] = "Calculator"

        result = create_benchmark_test(
            func,
            test_setup_code="Calculator calc = new Calculator();",
            invocation_code="calc.add(2, 2)",
            iterations=1000,
        )

        assert "benchmark" in result.lower()
        assert "Calculator" in result
        assert "calc.add(2, 2)" in result


class TestRemoveInstrumentation:
    """Tests for remove_instrumentation."""

    def test_removes_codeflash_imports(self):
        """Test removing CodeFlash imports."""
        source = """
import com.codeflash.CodeFlash;
import org.junit.jupiter.api.Test;

public class Test {}
"""
        result = remove_instrumentation(source)
        assert "import com.codeflash" not in result
        assert "org.junit" in result

    def test_preserves_regular_code(self):
        """Test that regular code is preserved."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        result = remove_instrumentation(source)
        assert "add" in result
        assert "return a + b" in result


class TestInstrumentExistingTest:
    """Tests for instrument_existing_test."""

    def test_instrument_behavior_mode(self, tmp_path: Path):
        """Test instrumenting in behavior mode."""
        test_file = tmp_path / "CalculatorTest.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
""")

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

        assert success is True
        assert result is not None

    def test_instrument_performance_mode(self, tmp_path: Path):
        """Test instrumenting in performance mode."""
        test_file = tmp_path / "CalculatorTest.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;

public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
""")

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

        assert success is True
        assert result is not None

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
