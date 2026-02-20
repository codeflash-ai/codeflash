"""Integration tests for Java line profiler with JavaSupport."""

import json
import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import FunctionInfo, Language
from codeflash.languages.java.support import get_java_support


class TestLineProfilerIntegration:
    """Integration tests for line profiler with JavaSupport."""

    def test_instrument_and_parse_results(self):
        """Test full workflow: instrument, parse results."""
        # Create a temporary Java file
        source = """package com.example;

public class Calculator {
    public static int add(int a, int b) {
        int result = a + b;
        return result;
    }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            src_dir = tmppath / "src"
            src_dir.mkdir()

            java_file = src_dir / "Calculator.java"
            java_file.write_text(source, encoding="utf-8")

            # Create profile output file
            profile_output = tmppath / "profile.json"

            func = FunctionInfo(
                function_name="add",
                file_path=java_file,
                starting_line=4,
                ending_line=7,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            # Get JavaSupport and instrument
            support = get_java_support()
            success = support.instrument_source_for_line_profiler(func, profile_output)

            # Should succeed
            assert success, "Instrumentation should succeed"

            # Verify file was modified
            instrumented = java_file.read_text(encoding="utf-8")
            assert "CodeflashLineProfiler" in instrumented
            assert "enterFunction()" in instrumented
            assert "hit(" in instrumented

    def test_parse_empty_results(self):
        """Test parsing results when file doesn't exist."""
        support = get_java_support()

        # Parse non-existent file
        results = support.parse_line_profile_results(Path("/tmp/nonexistent_profile.json"))

        # Should return empty results
        assert results["timings"] == {}
        assert results["unit"] == 1e-9

    def test_parse_valid_results(self):
        """Test parsing valid profiling results."""
        # Create sample profiling data
        data = {
            "/tmp/Test.java:5": {
                "hits": 100,
                "time": 5000000,  # 5ms in nanoseconds
                "file": "/tmp/Test.java",
                "line": 5,
                "content": "int x = compute();"
            },
            "/tmp/Test.java:6": {
                "hits": 100,
                "time": 95000000,  # 95ms in nanoseconds
                "file": "/tmp/Test.java",
                "line": 6,
                "content": "result = slowOperation(x);"
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            profile_file = Path(tmp.name)

        try:
            support = get_java_support()
            results = support.parse_line_profile_results(profile_file)

            # Verify structure
            assert "/tmp/Test.java" in results["timings"]
            assert 5 in results["timings"]["/tmp/Test.java"]
            assert 6 in results["timings"]["/tmp/Test.java"]

            # Verify line 5 data
            line5 = results["timings"]["/tmp/Test.java"][5]
            assert line5["hits"] == 100
            assert line5["time_ns"] == 5000000
            assert line5["time_ms"] == 5.0

            # Verify line 6 is the hotspot (95% of time)
            line6 = results["timings"]["/tmp/Test.java"][6]
            assert line6["hits"] == 100
            assert line6["time_ns"] == 95000000
            assert line6["time_ms"] == 95.0

            # Line 6 should be much slower
            assert line6["time_ms"] > line5["time_ms"] * 10

        finally:
            profile_file.unlink()

    def test_instrument_multiple_functions(self):
        """Test instrumenting multiple functions in same file."""
        source = """public class Test {
    public void method1() {
        int x = 1;
    }

    public void method2() {
        int y = 2;
    }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            java_file = tmppath / "Test.java"
            java_file.write_text(source, encoding="utf-8")

            profile_output = tmppath / "profile.json"

            func1 = FunctionInfo(
                function_name="method1",
                file_path=java_file,
                starting_line=2,
                ending_line=4,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            func2 = FunctionInfo(
                function_name="method2",
                file_path=java_file,
                starting_line=6,
                ending_line=8,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            # Instrument first function
            support = get_java_support()
            success1 = support.instrument_source_for_line_profiler(func1, profile_output)
            assert success1

            # Re-read source and instrument second function
            # Note: In real usage, you'd instrument both at once, but this tests the flow
            source2 = java_file.read_text(encoding="utf-8")

            # Write back original to test multiple instrumentations
            # (In practice, the profiler instruments all functions at once)
