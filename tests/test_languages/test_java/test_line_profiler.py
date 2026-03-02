"""Tests for Java line profiler (agent-based)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.languages.java.line_profiler import (
    DEFAULT_WARMUP_ITERATIONS,
    JavaLineProfiler,
    find_agent_jar,
    format_line_profile_results,
    resolve_internal_class_name,
)


class TestAgentConfigGeneration:
    """Tests for agent config generation."""

    def test_simple_method(self):
        """Test config generation for a simple method."""
        from codeflash.languages.base import FunctionInfo, Language

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

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"

            profiler = JavaLineProfiler(output_file=output_file)
            profiler.generate_agent_config(source, file_path, [func], config_path)

            assert config_path.exists()
            config = json.loads(config_path.read_text())

            assert config == {
                "outputFile": str(output_file),
                "warmupIterations": DEFAULT_WARMUP_ITERATIONS,
                "targets": [
                    {
                        "className": "com/example/Calculator",
                        "methods": [
                            {
                                "name": "add",
                                "startLine": 4,
                                "endLine": 7,
                                "sourceFile": file_path.as_posix(),
                            }
                        ],
                    }
                ],
                "lineContents": {
                    f"{file_path.as_posix()}:4": "public static int add(int a, int b) {",
                    f"{file_path.as_posix()}:5": "int result = a + b;",
                    f"{file_path.as_posix()}:6": "return result;",
                    f"{file_path.as_posix()}:7": "}",
                },
            }

    def test_line_contents_extraction(self):
        """Test that line contents are extracted correctly."""
        from codeflash.languages.base import FunctionInfo, Language

        source = """public class Test {
    public void method() {
        int x = 1;
        // just a comment
        return;
    }
}
"""
        file_path = Path("/tmp/Test.java")
        func = FunctionInfo(
            function_name="method",
            file_path=file_path,
            starting_line=2,
            ending_line=6,
            starting_col=0,
            ending_col=0,
            parents=(),
            is_async=False,
            is_method=True,
            language=Language.JAVA,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"

            profiler = JavaLineProfiler(output_file=output_file)
            profiler.generate_agent_config(source, file_path, [func], config_path)

            config = json.loads(config_path.read_text())

            assert config["lineContents"] == {
                f"{file_path.as_posix()}:2": "public void method() {",
                f"{file_path.as_posix()}:3": "int x = 1;",
                f"{file_path.as_posix()}:5": "return;",
                f"{file_path.as_posix()}:6": "}",
            }

    def test_multiple_functions(self):
        """Test config with multiple target functions."""
        from codeflash.languages.base import FunctionInfo, Language

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
        func1 = FunctionInfo(
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
        func2 = FunctionInfo(
            function_name="method2",
            file_path=file_path,
            starting_line=6,
            ending_line=8,
            starting_col=0,
            ending_col=0,
            parents=(),
            is_async=False,
            is_method=True,
            language=Language.JAVA,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"

            profiler = JavaLineProfiler(output_file=output_file)
            profiler.generate_agent_config(source, file_path, [func1, func2], config_path)

            config = json.loads(config_path.read_text())

            assert config["targets"][0]["methods"] == [
                {
                    "name": "method1",
                    "startLine": 2,
                    "endLine": 4,
                    "sourceFile": file_path.as_posix(),
                },
                {
                    "name": "method2",
                    "startLine": 6,
                    "endLine": 8,
                    "sourceFile": file_path.as_posix(),
                },
            ]

    def test_empty_function_list(self):
        """Test with no functions produces valid config."""
        source = "public class Test {}"
        file_path = Path("/tmp/Test.java")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"

            profiler = JavaLineProfiler(output_file=output_file)
            profiler.generate_agent_config(source, file_path, [], config_path)

            config = json.loads(config_path.read_text())
            assert config["targets"][0]["methods"] == []


class TestResolveInternalClassName:
    """Tests for JVM class name resolution."""

    def test_with_package(self):
        source = "package com.example;\npublic class Calculator {}"
        result = resolve_internal_class_name(Path("/tmp/Calculator.java"), source)
        assert result == "com/example/Calculator"

    def test_without_package(self):
        source = "public class Calculator {}"
        result = resolve_internal_class_name(Path("/tmp/Calculator.java"), source)
        assert result == "Calculator"

    def test_nested_package(self):
        source = "package org.apache.commons.lang3;\npublic class StringUtils {}"
        result = resolve_internal_class_name(Path("/tmp/StringUtils.java"), source)
        assert result == "org/apache/commons/lang3/StringUtils"


class TestAgentJarLocator:
    """Tests for finding the agent JAR."""

    def test_find_agent_jar(self):
        jar = find_agent_jar()
        # Should find it in either resources or dev build
        assert jar is not None
        assert jar.exists()
        assert jar.name == "codeflash-runtime-1.0.0.jar"

    def test_build_javaagent_arg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{}")

            profiler = JavaLineProfiler(output_file=output_file)
            arg = profiler.build_javaagent_arg(config_path)

            agent_jar = find_agent_jar()
            assert arg == f"-javaagent:{agent_jar}=config={config_path}"

    def test_build_javaagent_arg_missing_jar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{}")

            profiler = JavaLineProfiler(output_file=output_file)

            with patch("codeflash.languages.java.line_profiler.find_agent_jar", return_value=None):
                with pytest.raises(FileNotFoundError):
                    profiler.build_javaagent_arg(config_path)


class TestWarmupConfig:
    """Tests for warmup configuration in agent config generation."""

    def test_default_warmup_iterations(self):
        """Test that default warmup iterations matches the module constant."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            profiler = JavaLineProfiler(output_file=output_file)
            assert profiler.warmup_iterations == DEFAULT_WARMUP_ITERATIONS

    def test_custom_warmup_iterations(self):
        """Test setting custom warmup iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            profiler = JavaLineProfiler(output_file=output_file, warmup_iterations=10)
            assert profiler.warmup_iterations == 10

    def test_warmup_disabled(self):
        """Test warmup can be disabled by setting to 0."""
        from codeflash.languages.base import FunctionInfo, Language

        source = "public class Test {\n    public void method() {\n        return;\n    }\n}"
        file_path = Path("/tmp/Test.java")
        func = FunctionInfo(
            function_name="method",
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

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"

            profiler = JavaLineProfiler(output_file=output_file, warmup_iterations=0)
            profiler.generate_agent_config(source, file_path, [func], config_path)

            config = json.loads(config_path.read_text())
            assert config["warmupIterations"] == 0

    def test_warmup_in_config_json(self):
        """Test that warmupIterations appears in the generated config JSON."""
        from codeflash.languages.base import FunctionInfo, Language

        source = "package com.example;\npublic class Calc {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}"
        file_path = Path("/tmp/Calc.java")
        func = FunctionInfo(
            function_name="add",
            file_path=file_path,
            starting_line=3,
            ending_line=5,
            starting_col=0,
            ending_col=0,
            parents=(),
            is_async=False,
            is_method=True,
            language=Language.JAVA,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"

            profiler = JavaLineProfiler(output_file=output_file, warmup_iterations=7)
            profiler.generate_agent_config(source, file_path, [func], config_path)

            config = json.loads(config_path.read_text())
            assert config["warmupIterations"] == 7


class TestAgentConfigBoundaryConditions:
    """Tests for boundary conditions in agent config generation."""

    def test_start_line_beyond_end_line(self):
        """When starting_line > ending_line, no lines are extracted but config is still valid."""
        from codeflash.languages.base import FunctionInfo, Language

        source = "public class Test {\n    public void foo() { return; }\n}\n"
        file_path = Path("/tmp/Test.java")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"

            func = FunctionInfo(
                function_name="foo",
                file_path=file_path,
                starting_line=5,
                ending_line=2,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            profiler = JavaLineProfiler(output_file=output_file)
            profiler.generate_agent_config(source, file_path, [func], config_path)

            config = json.loads(config_path.read_text())
            assert config["lineContents"] == {}
            assert config["targets"][0]["methods"] == [
                {"name": "foo", "startLine": 5, "endLine": 2, "sourceFile": file_path.as_posix()}
            ]

    def test_line_numbers_beyond_source_length(self):
        """Line numbers beyond the source length are silently skipped."""
        from codeflash.languages.base import FunctionInfo, Language

        source = "public class Test {\n    public void foo() { return; }\n}\n"
        file_path = Path("/tmp/Test.java")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"

            func = FunctionInfo(
                function_name="foo",
                file_path=file_path,
                starting_line=100,
                ending_line=200,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            profiler = JavaLineProfiler(output_file=output_file)
            profiler.generate_agent_config(source, file_path, [func], config_path)

            config = json.loads(config_path.read_text())
            assert config == {
                "outputFile": str(output_file),
                "warmupIterations": DEFAULT_WARMUP_ITERATIONS,
                "targets": [
                    {
                        "className": "Test",
                        "methods": [
                            {
                                "name": "foo",
                                "startLine": 100,
                                "endLine": 200,
                                "sourceFile": file_path.as_posix(),
                            }
                        ],
                    }
                ],
                "lineContents": {},
            }

    def test_negative_line_numbers(self):
        """Negative line numbers produce no line contents (range is empty or out of bounds)."""
        from codeflash.languages.base import FunctionInfo, Language

        source = "public class Test {\n    public void foo() { return; }\n}\n"
        file_path = Path("/tmp/Test.java")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "profile.json"
            config_path = Path(tmpdir) / "config.json"

            func = FunctionInfo(
                function_name="foo",
                file_path=file_path,
                starting_line=-5,
                ending_line=-1,
                starting_col=0,
                ending_col=0,
                parents=(),
                is_async=False,
                is_method=True,
                language=Language.JAVA,
            )

            profiler = JavaLineProfiler(output_file=output_file)
            profiler.generate_agent_config(source, file_path, [func], config_path)

            config = json.loads(config_path.read_text())
            assert config == {
                "outputFile": str(output_file),
                "warmupIterations": DEFAULT_WARMUP_ITERATIONS,
                "targets": [
                    {
                        "className": "Test",
                        "methods": [
                            {
                                "name": "foo",
                                "startLine": -5,
                                "endLine": -1,
                                "sourceFile": file_path.as_posix(),
                            }
                        ],
                    }
                ],
                "lineContents": {},
            }


class TestLineProfileResultsParsing:
    """Tests for parsing line profile results."""

    def test_parse_results_empty_file(self):
        results = JavaLineProfiler.parse_results(Path("/tmp/nonexistent.json"))

        assert results == {"timings": {}, "unit": 1e-9, "str_out": ""}

    def test_parse_results_valid_data(self):
        data = {
            "/tmp/Test.java:10": {
                "hits": 100,
                "time": 5000000,
                "file": "/tmp/Test.java",
                "line": 10,
                "content": "int x = compute();",
            },
            "/tmp/Test.java:11": {
                "hits": 100,
                "time": 95000000,
                "file": "/tmp/Test.java",
                "line": 11,
                "content": "result = slowOperation(x);",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            profile_file = Path(tmp.name)

        results = JavaLineProfiler.parse_results(profile_file)

        assert results["unit"] == 1e-9
        assert results["timings"] == {
            ("/tmp/Test.java", 10, "Test.java"): [(10, 100, 5000000), (11, 100, 95000000)]
        }
        assert results["line_contents"] == {
            ("/tmp/Test.java", 10): "int x = compute();",
            ("/tmp/Test.java", 11): "result = slowOperation(x);",
        }
        assert results["str_out"] == (
            "# Timer unit: 1e-09 s\n"
            "## Function: Test.java\n"
            "## Total time: 0.1 s\n"
            "|   Hits |    Time |   Per Hit |   % Time | Line Contents              |\n"
            "|-------:|--------:|----------:|---------:|:---------------------------|\n"
            "|    100 | 5e+06   |     50000 |        5 | int x = compute();         |\n"
            "|    100 | 9.5e+07 |    950000 |       95 | result = slowOperation(x); |\n"
        )

        profile_file.unlink()

    def test_format_results(self):
        data = {
            "/tmp/Test.java:10": {
                "hits": 10,
                "time": 1000000,
                "file": "/tmp/Test.java",
                "line": 10,
                "content": "int x = 1;",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            profile_file = Path(tmp.name)

        results = JavaLineProfiler.parse_results(profile_file)
        formatted = format_line_profile_results(results)

        expected = (
            "# Timer unit: 1e-09 s\n"
            "## Function: Test.java\n"
            "## Total time: 0.001 s\n"
            "|   Hits |   Time |   Per Hit |   % Time | Line Contents   |\n"
            "|-------:|-------:|----------:|---------:|:----------------|\n"
            "|     10 |  1e+06 |    100000 |      100 | int x = 1;      |\n"
        )
        assert formatted == expected

        profile_file.unlink()

    def test_parse_results_corrupted_json(self):
        """Corrupted/truncated JSON returns empty results instead of crashing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write('{"incomplete": true, "data": [')  # truncated JSON
            profile_file = Path(tmp.name)

        results = JavaLineProfiler.parse_results(profile_file)

        assert results == {"timings": {}, "unit": 1e-9, "str_out": ""}

        profile_file.unlink()

    def test_parse_results_not_a_dict(self):
        """Profile file containing a JSON array instead of object returns empty results."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump([1, 2, 3], tmp)
            profile_file = Path(tmp.name)

        results = JavaLineProfiler.parse_results(profile_file)

        assert results == {"timings": {}, "unit": 1e-9, "str_out": ""}

        profile_file.unlink()

    def test_parse_results_no_config_file_fallback(self):
        """When config.json is missing, parse_results falls back to grouping by file."""
        data = {
            "/tmp/Sorter.java:5": {
                "hits": 10,
                "time": 2000000,
                "file": "/tmp/Sorter.java",
                "line": 5,
                "content": "int n = arr.length;",
            },
            "/tmp/Sorter.java:6": {
                "hits": 10,
                "time": 8000000,
                "file": "/tmp/Sorter.java",
                "line": 6,
                "content": "for (int i = 0; i < n; i++) {",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_file = Path(tmpdir) / "profile.json"
            profile_file.write_text(json.dumps(data), encoding="utf-8")

            # Deliberately do NOT create profile.config.json

            config_path = profile_file.with_suffix(".config.json")
            assert not config_path.exists()

            results = JavaLineProfiler.parse_results(profile_file)

            assert results == {
                "unit": 1e-9,
                "timings": {
                    ("/tmp/Sorter.java", 5, "Sorter.java"): [(5, 10, 2000000), (6, 10, 8000000)]
                },
                "line_contents": {
                    ("/tmp/Sorter.java", 5): "int n = arr.length;",
                    ("/tmp/Sorter.java", 6): "for (int i = 0; i < n; i++) {",
                },
                "str_out": (
                    "# Timer unit: 1e-09 s\n"
                    "## Function: Sorter.java\n"
                    "## Total time: 0.01 s\n"
                    "|   Hits |   Time |   Per Hit |   % Time | Line Contents                 |\n"
                    "|-------:|-------:|----------:|---------:|:------------------------------|\n"
                    "|     10 |  2e+06 |    200000 |       20 | int n = arr.length;           |\n"
                    "|     10 |  8e+06 |    800000 |       80 | for (int i = 0; i < n; i++) { |\n"
                ),
            }
