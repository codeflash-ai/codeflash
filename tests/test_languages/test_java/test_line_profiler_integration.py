"""Integration tests for Java line profiler with JavaSupport.
"""

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import FunctionInfo, Language
from codeflash.languages.java.line_profiler import DEFAULT_WARMUP_ITERATIONS, JavaLineProfiler, find_agent_jar
from codeflash.languages.java.support import get_java_support


class TestLineProfilerInstrumentation:
    """Integration tests for line profiler instrumentation through JavaSupport.
    """

    def test_instrument_with_package(self):
        """Test instrumentation for a class with a package declaration.
        """
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
            java_file = tmppath / "Calculator.java"
            java_file.write_text(source, encoding="utf-8")

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

            support = get_java_support()
            success = support.instrument_source_for_line_profiler(func, profile_output)

            assert success, "Profiler config generation should succeed"

            # Source file must NOT be modified (Java uses agent, not source rewriting)
            assert java_file.read_text(encoding="utf-8") == source

            # Config JSON should have been created with correct content
            config_path = profile_output.with_suffix(".config.json")
            assert config_path.exists()
            config = json.loads(config_path.read_text(encoding="utf-8"))

            assert config == {
                "outputFile": str(profile_output),
                "warmupIterations": DEFAULT_WARMUP_ITERATIONS,
                "targets": [
                    {
                        "className": "com/example/Calculator",
                        "methods": [
                            {
                                "name": "add",
                                "startLine": 4,
                                "endLine": 7,
                                "sourceFile": java_file.as_posix(),
                            }
                        ],
                    }
                ],
                "lineContents": {
                    f"{java_file.as_posix()}:4": "public static int add(int a, int b) {",
                    f"{java_file.as_posix()}:5": "int result = a + b;",
                    f"{java_file.as_posix()}:6": "return result;",
                    f"{java_file.as_posix()}:7": "}",
                },
            }

            # javaagent arg should be set on the support instance
            agent_jar = find_agent_jar()
            assert support._line_profiler_agent_arg == f"-javaagent:{agent_jar}=config={config_path}"

            # Warmup iterations should be stored
            assert support._line_profiler_warmup_iterations == DEFAULT_WARMUP_ITERATIONS

    def test_instrument_without_package(self):
        """Test instrumentation for a class without a package declaration.

        Mirrors Python's test_add_decorator_imports_nodeps — simple function with
        no external dependencies.
        """
        source = """public class Sorter {
    public static int[] sort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
        return arr;
    }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            java_file = tmppath / "Sorter.java"
            java_file.write_text(source, encoding="utf-8")

            profile_output = tmppath / "profile.json"

            func = FunctionInfo(
                function_name="sort",
                file_path=java_file,
                starting_line=2,
                ending_line=14,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            support = get_java_support()
            success = support.instrument_source_for_line_profiler(func, profile_output)

            assert success

            # Source not modified
            assert java_file.read_text(encoding="utf-8") == source

            config_path = profile_output.with_suffix(".config.json")
            config = json.loads(config_path.read_text(encoding="utf-8"))

            assert config == {
                "outputFile": str(profile_output),
                "warmupIterations": DEFAULT_WARMUP_ITERATIONS,
                "targets": [
                    {
                        "className": "Sorter",
                        "methods": [
                            {
                                "name": "sort",
                                "startLine": 2,
                                "endLine": 14,
                                "sourceFile": java_file.as_posix(),
                            }
                        ],
                    }
                ],
                "lineContents": {
                    f"{java_file.as_posix()}:2": "public static int[] sort(int[] arr) {",
                    f"{java_file.as_posix()}:3": "int n = arr.length;",
                    f"{java_file.as_posix()}:4": "for (int i = 0; i < n; i++) {",
                    f"{java_file.as_posix()}:5": "for (int j = 0; j < n - i - 1; j++) {",
                    f"{java_file.as_posix()}:6": "if (arr[j] > arr[j + 1]) {",
                    f"{java_file.as_posix()}:7": "int temp = arr[j];",
                    f"{java_file.as_posix()}:8": "arr[j] = arr[j + 1];",
                    f"{java_file.as_posix()}:9": "arr[j + 1] = temp;",
                    f"{java_file.as_posix()}:10": "}",
                    f"{java_file.as_posix()}:11": "}",
                    f"{java_file.as_posix()}:12": "}",
                    f"{java_file.as_posix()}:13": "return arr;",
                    f"{java_file.as_posix()}:14": "}",
                },
            }

    def test_instrument_multiple_methods(self):
        """Test instrumentation with multiple target methods in the same class.

        Mirrors Python's test_add_decorator_imports_helper_outside — multiple
        functions that all need to be profiled.
        """
        source = """public class StringProcessor {
    public static String reverse(String s) {
        char[] chars = s.toCharArray();
        int left = 0;
        int right = chars.length - 1;
        while (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }
        return new String(chars);
    }

    public static boolean isPalindrome(String s) {
        String cleaned = s.toLowerCase().replaceAll("[^a-z0-9]", "");
        String reversed = reverse(cleaned);
        return cleaned.equals(reversed);
    }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            java_file = tmppath / "StringProcessor.java"
            java_file.write_text(source, encoding="utf-8")

            profile_output = tmppath / "profile.json"

            func_reverse = FunctionInfo(
                function_name="reverse",
                file_path=java_file,
                starting_line=2,
                ending_line=14,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )
            func_palindrome = FunctionInfo(
                function_name="isPalindrome",
                file_path=java_file,
                starting_line=16,
                ending_line=20,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            support = get_java_support()
            # Instrument first function
            success = support.instrument_source_for_line_profiler(func_reverse, profile_output)
            assert success

            # Source not modified
            assert java_file.read_text(encoding="utf-8") == source

            config_path = profile_output.with_suffix(".config.json")
            config = json.loads(config_path.read_text(encoding="utf-8"))

            # Both methods should appear as targets when generated together
            profiler = JavaLineProfiler(output_file=profile_output)
            profiler.generate_agent_config(
                source, java_file, [func_reverse, func_palindrome], config_path
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))

            assert config == {
                "outputFile": str(profile_output),
                "warmupIterations": DEFAULT_WARMUP_ITERATIONS,
                "targets": [
                    {
                        "className": "StringProcessor",
                        "methods": [
                            {
                                "name": "reverse",
                                "startLine": 2,
                                "endLine": 14,
                                "sourceFile": java_file.as_posix(),
                            },
                            {
                                "name": "isPalindrome",
                                "startLine": 16,
                                "endLine": 20,
                                "sourceFile": java_file.as_posix(),
                            },
                        ],
                    }
                ],
                "lineContents": {
                    f"{java_file.as_posix()}:2": "public static String reverse(String s) {",
                    f"{java_file.as_posix()}:3": "char[] chars = s.toCharArray();",
                    f"{java_file.as_posix()}:4": "int left = 0;",
                    f"{java_file.as_posix()}:5": "int right = chars.length - 1;",
                    f"{java_file.as_posix()}:6": "while (left < right) {",
                    f"{java_file.as_posix()}:7": "char temp = chars[left];",
                    f"{java_file.as_posix()}:8": "chars[left] = chars[right];",
                    f"{java_file.as_posix()}:9": "chars[right] = temp;",
                    f"{java_file.as_posix()}:10": "left++;",
                    f"{java_file.as_posix()}:11": "right--;",
                    f"{java_file.as_posix()}:12": "}",
                    f"{java_file.as_posix()}:13": "return new String(chars);",
                    f"{java_file.as_posix()}:14": "}",
                    f"{java_file.as_posix()}:16": "public static boolean isPalindrome(String s) {",
                    f"{java_file.as_posix()}:17": 'String cleaned = s.toLowerCase().replaceAll("[^a-z0-9]", "");',
                    f"{java_file.as_posix()}:18": "String reversed = reverse(cleaned);",
                    f"{java_file.as_posix()}:19": "return cleaned.equals(reversed);",
                    f"{java_file.as_posix()}:20": "}",
                },
            }

    def test_instrument_nested_package(self):
        """Test instrumentation for a deeply nested package.

        Mirrors Python's test_add_decorator_imports_helper_in_nested_class —
        verifies correct class name resolution with deep package nesting.
        """
        source = """package org.apache.commons.lang3;

public class StringUtils {
    public static boolean isEmpty(String s) {
        return s == null || s.length() == 0;
    }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            java_file = tmppath / "StringUtils.java"
            java_file.write_text(source, encoding="utf-8")

            profile_output = tmppath / "profile.json"

            func = FunctionInfo(
                function_name="isEmpty",
                file_path=java_file,
                starting_line=4,
                ending_line=6,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            support = get_java_support()
            success = support.instrument_source_for_line_profiler(func, profile_output)

            assert success

            # Source not modified
            assert java_file.read_text(encoding="utf-8") == source

            config_path = profile_output.with_suffix(".config.json")
            config = json.loads(config_path.read_text(encoding="utf-8"))

            assert config == {
                "outputFile": str(profile_output),
                "warmupIterations": DEFAULT_WARMUP_ITERATIONS,
                "targets": [
                    {
                        "className": "org/apache/commons/lang3/StringUtils",
                        "methods": [
                            {
                                "name": "isEmpty",
                                "startLine": 4,
                                "endLine": 6,
                                "sourceFile": java_file.as_posix(),
                            }
                        ],
                    }
                ],
                "lineContents": {
                    f"{java_file.as_posix()}:4": "public static boolean isEmpty(String s) {",
                    f"{java_file.as_posix()}:5": "return s == null || s.length() == 0;",
                    f"{java_file.as_posix()}:6": "}",
                },
            }

    def test_instrument_verifies_line_contents(self):
        """Test that line contents are extracted correctly, skipping comment-only lines.

        Mirrors Python's test_add_decorator_imports_helper_in_dunder_class —
        verifies that instrumentation handles all content in the function body.
        """
        source = """public class Fibonacci {
    public static long fib(int n) {
        if (n <= 1) {
            return n;
        }
        // iterative approach
        long a = 0;
        long b = 1;
        for (int i = 2; i <= n; i++) {
            long temp = b;
            b = a + b;
            a = temp;
        }
        return b;
    }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            java_file = tmppath / "Fibonacci.java"
            java_file.write_text(source, encoding="utf-8")

            profile_output = tmppath / "profile.json"

            func = FunctionInfo(
                function_name="fib",
                file_path=java_file,
                starting_line=2,
                ending_line=15,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            support = get_java_support()
            success = support.instrument_source_for_line_profiler(func, profile_output)

            assert success

            config_path = profile_output.with_suffix(".config.json")
            config = json.loads(config_path.read_text(encoding="utf-8"))

            line_contents = config["lineContents"]
            p = java_file.as_posix()

            # Comment-only line 6 ("// iterative approach") should be excluded
            assert f"{p}:6" not in line_contents

            # Code lines should be present with correct content
            assert line_contents[f"{p}:2"] == "public static long fib(int n) {"
            assert line_contents[f"{p}:3"] == "if (n <= 1) {"
            assert line_contents[f"{p}:4"] == "return n;"
            assert line_contents[f"{p}:7"] == "long a = 0;"
            assert line_contents[f"{p}:9"] == "for (int i = 2; i <= n; i++) {"
            assert line_contents[f"{p}:14"] == "return b;"
            assert line_contents[f"{p}:15"] == "}"


def build_spin_timer_source(spin_durations_ns: list[int]) -> str:
    """Build a SpinTimer Java source that calls spinWait with each given duration."""
    calls = "\n".join(f"        spinWait({d}L);" for d in spin_durations_ns)
    return f"""\
public class SpinTimer {{
    public static long spinWait(long durationNs) {{
        long start = System.nanoTime();
        while (System.nanoTime() - start < durationNs) {{
        }}
        return durationNs;
    }}

    public static void main(String[] args) {{
{calls}
    }}
}}
"""


def run_spin_timer_profiled(tmppath: Path, spin_durations_ns: list[int]) -> dict:
    """Compile and run SpinTimer with the profiler agent, return parsed results."""
    source = build_spin_timer_source(spin_durations_ns)
    java_file = tmppath / "SpinTimer.java"
    java_file.write_text(source, encoding="utf-8")

    profile_output = tmppath / "profile.json"
    config_path = profile_output.with_suffix(".config.json")

    func = FunctionInfo(
        function_name="spinWait",
        file_path=java_file,
        starting_line=2,
        ending_line=7,
        starting_col=0,
        ending_col=0,
        parents=(),
        is_async=False,
        is_method=True,
        language=Language.JAVA,
    )

    profiler = JavaLineProfiler(output_file=profile_output, warmup_iterations=0)
    profiler.generate_agent_config(source, java_file, [func], config_path)
    agent_arg = profiler.build_javaagent_arg(config_path)

    result = subprocess.run(
        ["javac", str(java_file)],
        capture_output=True,
        text=True,
        cwd=str(tmppath),
    )
    assert result.returncode == 0, f"javac failed: {result.stderr}"

    result = subprocess.run(
        ["java", agent_arg, "-cp", str(tmppath), "SpinTimer"],
        capture_output=True,
        text=True,
        cwd=str(tmppath),
        timeout=30,
    )
    assert result.returncode == 0, f"java failed: {result.stderr}"
    assert profile_output.exists(), "Profile output not written"

    return JavaLineProfiler.parse_results(profile_output)


@pytest.mark.skipif(not shutil.which("javac"), reason="Java compiler not available")
class TestSpinTimerProfiling:
    """End-to-end spin-timer tests validating profiler timing accuracy.

    Calls spinWait multiple times with known durations, then verifies the
    profiler-reported total time matches the expected sum of all spin durations.
    """

    @pytest.mark.parametrize(
        "spin_durations_ns",
        [
            [50_000_000, 100_000_000],
            [30_000_000, 40_000_000, 80_000_000],
        ],
    )
    def test_total_time_matches_expected(self, spin_durations_ns):
        """Profiler total time should match the sum of all spin durations."""
        expected_ns = sum(spin_durations_ns)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_spin_timer_profiled(Path(tmpdir), spin_durations_ns)

            assert results["timings"], "No timing data produced"

            line_data = next(iter(results["timings"].values()))
            total_time_ns = sum(t for _, _, t in line_data)

            assert math.isclose(total_time_ns, expected_ns, rel_tol=0.25), (
                f"Measured {total_time_ns}ns, expected ~{expected_ns}ns (25% tolerance)"
            )

    def test_while_line_dominates(self):
        """The while-loop line should account for the majority of self-time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_spin_timer_profiled(Path(tmpdir), [50_000_000, 100_000_000])

            assert results["timings"]

            line_data = next(iter(results["timings"].values()))
            line_times = {lineno: t for lineno, _, t in line_data}
            total_time = sum(line_times.values())

            while_line_time = line_times.get(4, 0)
            assert while_line_time / total_time > 0.80, (
                f"While line has {while_line_time / total_time:.1%} of total time, expected >80%"
            )

    def test_hit_counts_match_call_count(self):
        """Each line in spinWait should have hits equal to the number of calls."""
        spin_durations = [20_000_000, 30_000_000, 50_000_000]

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_spin_timer_profiled(Path(tmpdir), spin_durations)

            assert results["timings"]

            line_data = next(iter(results["timings"].values()))
            line_hits = {lineno: h for lineno, h, _ in line_data}

            # Lines 3 and 6 (start assignment and return) execute once per call
            assert line_hits.get(3, 0) == len(spin_durations), (
                f"Line 3 hits: {line_hits.get(3, 0)}, expected {len(spin_durations)}"
            )
