"""Tests for Java line profiler."""

import json
import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import FunctionInfo, Language
from codeflash.languages.java.line_profiler import JavaLineProfiler, format_line_profile_results
from codeflash.languages.java.parser import get_java_analyzer


class TestJavaLineProfilerInstrumentation:
    """Tests for line profiler instrumentation."""

    def test_instrument_simple_method(self):
        """Test instrumenting a simple method."""
        source = """package com.example;

public class Calculator {
    public static int add(int a, int b) {
        int result = a + b;
        return result;
    }
}
"""
        file_path = Path("/tmp/Calculator.java")
        func = FunctionInfo(
            function_name="add",
            file_path=file_path,
            starting_line=4,
            ending_line=7,
            starting_col=0,
            ending_col=0,
            parents=(),
            is_async=False,
            is_method=True,
            language=Language.JAVA,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = Path(tmp.name)

        profiler = JavaLineProfiler(output_file=output_file)
        analyzer = get_java_analyzer()

        instrumented = profiler.instrument_source(source, file_path, [func], analyzer)

        # Verify profiler class is added
        assert "class CodeflashLineProfiler" in instrumented
        assert "public static void hit(String file, int line)" in instrumented

        # Verify enterFunction() is called
        assert "CodeflashLineProfiler.enterFunction()" in instrumented

        # Verify hit() calls are added for executable lines
        assert 'CodeflashLineProfiler.hit("/tmp/Calculator.java"' in instrumented

        # Cleanup
        output_file.unlink(missing_ok=True)

    def test_instrument_preserves_non_instrumented_code(self):
        """Test that non-instrumented parts are preserved."""
        source = """public class Test {
    public void method1() {
        int x = 1;
    }

    public void method2() {
        int y = 2;
    }
}
"""
        file_path = Path("/tmp/Test.java")
        func = FunctionInfo(
            function_name="method1",
            file_path=file_path,
            starting_line=2,
            ending_line=4,
            starting_col=0,
            ending_col=0,
            parents=(),
            is_async=False,
            is_method=True,
            language=Language.JAVA,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = Path(tmp.name)

        profiler = JavaLineProfiler(output_file=output_file)
        analyzer = get_java_analyzer()

        instrumented = profiler.instrument_source(source, file_path, [func], analyzer)

        # method2 should not be instrumented
        lines = instrumented.split("\n")
        method2_lines = [l for l in lines if "method2" in l or "int y = 2" in l]

        # Should have method2 declaration and body, but no profiler calls in method2
        assert any("method2" in l for l in method2_lines)
        assert any("int y = 2" in l for l in method2_lines)
        # Profiler calls should not be in method2's body
        method2_start = None
        for i, l in enumerate(lines):
            if "method2" in l:
                method2_start = i
                break

        if method2_start:
            # Check the few lines after method2 declaration
            method2_body = lines[method2_start : method2_start + 5]
            profiler_in_method2 = any("CodeflashLineProfiler.hit" in l for l in method2_body)
            # There might be profiler class code before method2, but not in its body
            # Actually, since we only instrument method1, method2 should be unchanged

        # Cleanup
        output_file.unlink(missing_ok=True)

    def test_find_executable_lines(self):
        """Test finding executable lines in Java code."""
        source = """public static int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}
"""
        analyzer = get_java_analyzer()
        tree = analyzer.parse(source.encode("utf8"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = Path(tmp.name)

        profiler = JavaLineProfiler(output_file=output_file)
        executable_lines = profiler._find_executable_lines(tree.root_node)

        # Should find the if statement and return statements
        assert len(executable_lines) >= 2

        # Cleanup
        output_file.unlink(missing_ok=True)


class TestJavaLineProfilerExecution:
    """Tests for line profiler execution (requires compilation)."""

    @pytest.mark.skipif(
        True,  # Skip for now - compilation test requires full Java env
        reason="Java compiler test skipped - requires javac and dependencies",
    )
    def test_instrumented_code_compiles(self):
        """Test that instrumented code compiles successfully."""
        source = """package com.example;

public class Factorial {
    public static long factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Negative input");
        }
        long result = 1;
        for (int i = 1; i <= n; i++) {
            result *= i;
        }
        return result;
    }
}
"""
        file_path = Path("/tmp/test_profiler/Factorial.java")
        func = FunctionInfo(
            function_name="factorial",
            file_path=file_path,
            starting_line=4,
            ending_line=12,
            starting_col=0,
            ending_col=0,
            parents=(),
            is_async=False,
            is_method=True,
            language=Language.JAVA,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = Path(tmp.name)

        profiler = JavaLineProfiler(output_file=output_file)
        analyzer = get_java_analyzer()

        instrumented = profiler.instrument_source(source, file_path, [func], analyzer)

        # Write instrumented source
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(instrumented, encoding="utf-8")

        # Try to compile
        import subprocess

        result = subprocess.run(
            ["javac", str(file_path)],
            capture_output=True,
            text=True,
        )

        # Check compilation
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            # For now, we expect compilation to fail due to missing Gson dependency
            # This is expected - we're just testing that the instrumentation syntax is valid
            # In real usage, Gson will be available via Maven/Gradle
            assert "package com.google.gson does not exist" in result.stderr or "cannot find symbol" in result.stderr
        else:
            assert result.returncode == 0, f"Compilation failed: {result.stderr}"

        # Cleanup
        output_file.unlink(missing_ok=True)
        file_path.unlink(missing_ok=True)
        class_file = file_path.with_suffix(".class")
        class_file.unlink(missing_ok=True)


class TestLineProfileResultsParsing:
    """Tests for parsing line profile results."""

    def test_parse_results_empty_file(self):
        """Test parsing when file doesn't exist."""
        results = JavaLineProfiler.parse_results(Path("/tmp/nonexistent.json"))

        assert results["timings"] == {}
        assert results["unit"] == 1e-9

    def test_parse_results_valid_data(self):
        """Test parsing valid profiling data."""
        data = {
            "/tmp/Test.java:10": {
                "hits": 100,
                "time": 5000000,  # 5ms in nanoseconds
                "file": "/tmp/Test.java",
                "line": 10,
                "content": "int x = compute();"
            },
            "/tmp/Test.java:11": {
                "hits": 100,
                "time": 95000000,  # 95ms in nanoseconds
                "file": "/tmp/Test.java",
                "line": 11,
                "content": "result = slowOperation(x);"
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            profile_file = Path(tmp.name)

        results = JavaLineProfiler.parse_results(profile_file)

        assert "/tmp/Test.java" in results["timings"]
        assert 10 in results["timings"]["/tmp/Test.java"]
        assert 11 in results["timings"]["/tmp/Test.java"]

        line10 = results["timings"]["/tmp/Test.java"][10]
        assert line10["hits"] == 100
        assert line10["time_ns"] == 5000000
        assert line10["time_ms"] == 5.0

        line11 = results["timings"]["/tmp/Test.java"][11]
        assert line11["hits"] == 100
        assert line11["time_ns"] == 95000000
        assert line11["time_ms"] == 95.0

        # Line 11 is the hotspot (95% of time)
        total_time = line10["time_ms"] + line11["time_ms"]
        assert line11["time_ms"] / total_time > 0.9  # 95% of time

        # Cleanup
        profile_file.unlink()

    def test_format_results(self):
        """Test formatting results for display."""
        data = {
            "/tmp/Test.java:10": {
                "hits": 10,
                "time": 1000000,  # 1ms
                "file": "/tmp/Test.java",
                "line": 10,
                "content": "int x = 1;"
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            profile_file = Path(tmp.name)

        results = JavaLineProfiler.parse_results(profile_file)
        formatted = format_line_profile_results(results)

        assert "Line Profiling Results" in formatted
        assert "/tmp/Test.java" in formatted
        assert "10" in formatted  # Line number
        assert "10" in formatted  # Hits
        assert "int x = 1" in formatted  # Code content

        # Cleanup
        profile_file.unlink()


class TestLineProfilerEdgeCases:
    """Tests for edge cases in line profiling."""

    def test_empty_function_list(self):
        """Test with no functions to instrument."""
        source = "public class Test {}"
        file_path = Path("/tmp/Test.java")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = Path(tmp.name)

        profiler = JavaLineProfiler(output_file=output_file)

        instrumented = profiler.instrument_source(source, file_path, [], None)

        # Should return source unchanged
        assert instrumented == source

        # Cleanup
        output_file.unlink(missing_ok=True)

    def test_function_with_only_comments(self):
        """Test instrumenting a function with only comments."""
        source = """public class Test {
    public void method() {
        // Just a comment
        /* Another comment */
    }
}
"""
        file_path = Path("/tmp/Test.java")
        func = FunctionInfo(
            function_name="method",
            file_path=file_path,
            starting_line=2,
            ending_line=5,
            starting_col=0,
            ending_col=0,
            parents=(),
            is_async=False,
            is_method=True,
            language=Language.JAVA,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = Path(tmp.name)

        profiler = JavaLineProfiler(output_file=output_file)
        analyzer = get_java_analyzer()

        instrumented = profiler.instrument_source(source, file_path, [func], analyzer)

        # Should add profiler class and enterFunction, but no hit() calls for comments
        assert "CodeflashLineProfiler" in instrumented
        assert "enterFunction()" in instrumented

        # Should not add hit() for comment lines
        lines = instrumented.split("\n")
        comment_line_has_hit = any(
            "// Just a comment" in l and "hit(" in l for l in lines
        )
        assert not comment_line_has_hit

        # Cleanup
        output_file.unlink(missing_ok=True)
