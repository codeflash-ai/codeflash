"""Tests for JavaScript function tracing.

Tests the JavaScript tracer implementation including:
- Unit tests for Python-side trace parsing and replay test generation
- End-to-end tests for the full tracing pipeline via trace-runner.js
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from codeflash.languages.javascript.replay_test import (
    JavaScriptFunctionModule,
    create_javascript_replay_test,
    get_function_alias,
    get_traced_functions_from_db,
)
from codeflash.languages.javascript.tracer import JavaScriptTracer


def node_available() -> bool:
    """Check if Node.js is available."""
    return shutil.which("node") is not None


def skip_if_node_not_available() -> None:
    """Skip test if Node.js is not available."""
    if not node_available():
        pytest.skip("Node.js not available")


class TestJavaScriptTracerParsing:
    """Unit tests for JavaScriptTracer trace parsing."""

    @pytest.fixture
    def trace_db_with_function_calls(self, tmp_path: Path) -> Path:
        """Create a trace database with function_calls schema."""
        db_path = tmp_path / "trace.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE function_calls (
                type TEXT,
                function TEXT,
                classname TEXT,
                filename TEXT,
                line_number INTEGER,
                last_frame_address INTEGER,
                time_ns INTEGER,
                args BLOB
            )
        """)

        cursor.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        cursor.execute("INSERT INTO metadata (key, value) VALUES ('language', 'javascript')")

        test_args = json.dumps([1, 2, 3])
        cursor.execute(
            """
            INSERT INTO function_calls (type, function, classname, filename, line_number, last_frame_address, time_ns, args)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("call", "add", None, "/project/src/math.js", 10, 1, 1000000, test_args),
        )

        cursor.execute(
            """
            INSERT INTO function_calls (type, function, classname, filename, line_number, last_frame_address, time_ns, args)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("call", "multiply", "Calculator", "/project/src/calc.js", 25, 2, 2000000, json.dumps([5, 10])),
        )

        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def trace_db_legacy_schema(self, tmp_path: Path) -> Path:
        """Create a trace database with legacy traces schema."""
        db_path = tmp_path / "legacy_trace.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE traces (
                id INTEGER PRIMARY KEY,
                call_id INTEGER,
                function TEXT,
                file TEXT,
                args TEXT,
                result TEXT,
                error TEXT,
                runtime_ns TEXT,
                timestamp INTEGER
            )
        """)

        cursor.execute(
            """
            INSERT INTO traces (call_id, function, file, args, result, error, runtime_ns, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (1, "legacyFunc", "/old/path.js", json.dumps(["arg1"]), json.dumps("result"), "null", "5000", 1234567890),
        )

        conn.commit()
        conn.close()
        return db_path

    def test_parse_results_function_calls_schema(self, trace_db_with_function_calls: Path) -> None:
        """Test parsing traces from function_calls schema."""
        traces = JavaScriptTracer.parse_results(trace_db_with_function_calls)

        assert len(traces) == 2

        add_trace = next(t for t in traces if t["function"] == "add")
        assert add_trace["type"] == "call"
        assert add_trace["filename"] == "/project/src/math.js"
        assert add_trace["line_number"] == 10
        assert add_trace["args"] == [1, 2, 3]
        assert add_trace["classname"] is None

        multiply_trace = next(t for t in traces if t["function"] == "multiply")
        assert multiply_trace["classname"] == "Calculator"
        assert multiply_trace["args"] == [5, 10]

    def test_parse_results_legacy_schema(self, trace_db_legacy_schema: Path) -> None:
        """Test parsing traces from legacy traces schema."""
        traces = JavaScriptTracer.parse_results(trace_db_legacy_schema)

        assert len(traces) == 1
        trace = traces[0]
        assert trace["function"] == "legacyFunc"
        assert trace["file"] == "/old/path.js"
        assert trace["args"] == ["arg1"]
        assert trace["result"] == "result"
        assert trace["runtime_ns"] == 5000

    def test_parse_results_nonexistent_file(self, tmp_path: Path) -> None:
        """Test parsing a nonexistent file returns empty list."""
        traces = JavaScriptTracer.parse_results(tmp_path / "nonexistent.sqlite")
        assert traces == []

    def test_parse_results_json_file(self, tmp_path: Path) -> None:
        """Test parsing a JSON trace file."""
        json_path = tmp_path / "trace.json"
        trace_data = [{"function": "jsonFunc", "args": [1, 2], "time_ns": 1000}]
        json_path.write_text(json.dumps(trace_data))

        sqlite_path = tmp_path / "trace.sqlite"
        traces = JavaScriptTracer.parse_results(sqlite_path)

        assert len(traces) == 1
        assert traces[0]["function"] == "jsonFunc"

    def test_get_traced_functions(self, trace_db_with_function_calls: Path) -> None:
        """Test extracting traced function information."""
        functions = JavaScriptTracer.get_traced_functions(trace_db_with_function_calls)

        assert len(functions) == 2

        func_names = {f.function_name for f in functions}
        assert func_names == {"add", "multiply"}

        add_func = next(f for f in functions if f.function_name == "add")
        assert add_func.file_name == "/project/src/math.js"
        assert add_func.class_name is None
        assert add_func.line_number == 10
        assert "math" in add_func.module_path

        multiply_func = next(f for f in functions if f.function_name == "multiply")
        assert multiply_func.class_name == "Calculator"


class TestJavaScriptReplayTestGeneration:
    """Unit tests for replay test generation."""

    def test_get_function_alias_simple(self) -> None:
        """Test generating function aliases."""
        alias = get_function_alias("src/utils", "processData")
        assert alias == "src_utils_processData"

    def test_get_function_alias_with_class(self) -> None:
        """Test generating function aliases with class names."""
        alias = get_function_alias("src/calculator", "add", "Calculator")
        assert alias == "src_calculator_Calculator_add"

    def test_get_function_alias_special_chars(self) -> None:
        """Test generating aliases with special characters in path."""
        alias = get_function_alias("@scope/package/lib", "func")
        assert "_" in alias
        assert "func" in alias

    def test_create_javascript_replay_test_jest(self, tmp_path: Path) -> None:
        """Test generating Jest replay test."""
        functions = [
            JavaScriptFunctionModule(function_name="add", file_name=tmp_path / "math.js", module_name="src/math"),
            JavaScriptFunctionModule(
                function_name="multiply",
                file_name=tmp_path / "calc.js",
                module_name="src/calc",
                class_name="Calculator",
            ),
        ]

        content = create_javascript_replay_test(
            trace_file=str(tmp_path / "trace.sqlite"), functions=functions, max_run_count=50, framework="jest"
        )

        assert "// Auto-generated replay test by Codeflash" in content
        assert "require('codeflash/replay')" in content
        assert "describe('Replay: add'" in content
        assert "describe('Replay: Calculator.multiply'" in content
        assert "test.each" in content
        assert "50" in content

    def test_create_javascript_replay_test_vitest(self, tmp_path: Path) -> None:
        """Test generating Vitest replay test."""
        functions = [
            JavaScriptFunctionModule(function_name="process", file_name=tmp_path / "data.js", module_name="src/data")
        ]

        content = create_javascript_replay_test(
            trace_file=str(tmp_path / "trace.sqlite"), functions=functions, framework="vitest"
        )

        assert "import { describe, test } from 'vitest'" in content
        assert "describe('Replay: process'" in content

    def test_create_javascript_replay_test_skips_constructors(self, tmp_path: Path) -> None:
        """Test that constructors are skipped in replay tests."""
        functions = [
            JavaScriptFunctionModule(
                function_name="constructor",
                file_name=tmp_path / "class.js",
                module_name="src/class",
                class_name="MyClass",
            ),
            JavaScriptFunctionModule(
                function_name="__init__", file_name=tmp_path / "class.js", module_name="src/class"
            ),
            JavaScriptFunctionModule(function_name="doWork", file_name=tmp_path / "class.js", module_name="src/class"),
        ]

        content = create_javascript_replay_test(trace_file=str(tmp_path / "trace.sqlite"), functions=functions)

        assert "constructor" not in content.lower() or "Replay: constructor" not in content
        assert "__init__" not in content or "Replay: __init__" not in content
        assert "describe('Replay: doWork'" in content


class TestJavaScriptTracerCreateReplayTest:
    """Unit tests for JavaScriptTracer.create_replay_test method."""

    @pytest.fixture
    def trace_db(self, tmp_path: Path) -> Path:
        """Create a trace database for testing."""
        db_path = tmp_path / "trace.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE function_calls (
                type TEXT,
                function TEXT,
                classname TEXT,
                filename TEXT,
                line_number INTEGER,
                last_frame_address INTEGER,
                time_ns INTEGER,
                args BLOB
            )
        """)

        cursor.execute(
            """
            INSERT INTO function_calls (type, function, classname, filename, line_number, last_frame_address, time_ns, args)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("call", "fibonacci", None, "./src/math.js", 5, 1, 1000, json.dumps([10])),
        )

        conn.commit()
        conn.close()
        return db_path

    def test_create_replay_test_generates_file(self, trace_db: Path, tmp_path: Path) -> None:
        """Test that create_replay_test generates a test file."""
        tracer = JavaScriptTracer(trace_db)
        output_path = tmp_path / "tests" / "replay.test.js"

        result = tracer.create_replay_test(trace_db, output_path)

        assert result is not None
        assert output_path.exists()

        content = output_path.read_text()
        assert "fibonacci" in content
        assert "describe" in content

    def test_create_replay_test_empty_db(self, tmp_path: Path) -> None:
        """Test that create_replay_test returns None for empty database."""
        db_path = tmp_path / "empty.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE function_calls (
                type TEXT, function TEXT, classname TEXT, filename TEXT,
                line_number INTEGER, last_frame_address INTEGER, time_ns INTEGER, args BLOB
            )
        """)
        conn.commit()
        conn.close()

        tracer = JavaScriptTracer(db_path)
        result = tracer.create_replay_test(db_path, tmp_path / "test.js")

        assert result is None


@pytest.mark.skipif(not node_available(), reason="Node.js not available")
class TestJavaScriptTracerE2E:
    """End-to-end tests for JavaScript tracing."""

    @pytest.fixture
    def js_project(self, tmp_path: Path) -> Path:
        """Create a sample JavaScript project for testing."""
        project_dir = tmp_path / "js_project"
        project_dir.mkdir()

        src_dir = project_dir / "src"
        src_dir.mkdir()

        math_js = src_dir / "math.js"
        math_js.write_text("""
function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}

module.exports = { add, multiply };
""")

        main_js = project_dir / "main.js"
        main_js.write_text("""
const { add, multiply } = require('./src/math.js');

console.log('Running calculations...');
console.log('add(2, 3) =', add(2, 3));
console.log('add(10, 20) =', add(10, 20));
console.log('multiply(4, 5) =', multiply(4, 5));
""")

        package_json = project_dir / "package.json"
        package_json.write_text(json.dumps({"name": "test-project", "version": "1.0.0", "main": "main.js"}))

        return project_dir

    @pytest.fixture
    def trace_runner_path(self) -> Path:
        """Get the path to trace-runner.js."""
        path = Path(__file__).parent.parent.parent / "packages" / "codeflash" / "runtime" / "trace-runner.js"
        if not path.exists():
            pytest.skip("trace-runner.js not found")
        return path

    def test_trace_runner_help(self, trace_runner_path: Path) -> None:
        """Test that trace-runner.js responds to --help."""
        result = subprocess.run(
            ["node", str(trace_runner_path), "--help"], capture_output=True, text=True, timeout=30, check=False
        )

        assert "Usage:" in result.stdout or result.returncode == 0

    def test_trace_javascript_file(self, js_project: Path, trace_runner_path: Path, tmp_path: Path) -> None:
        """Test tracing a JavaScript file end-to-end.

        This test installs required npm dependencies and runs the full tracing pipeline.
        """
        trace_db = tmp_path / "trace.sqlite"

        # Update package.json with dependencies
        package_json = js_project / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test-project",
                    "version": "1.0.0",
                    "main": "main.js",
                    "dependencies": {
                        "@babel/core": "^7.24.0",
                        "@babel/register": "^7.24.0",
                        "better-sqlite3": "^11.0.0",
                    },
                }
            )
        )

        # Install npm dependencies
        npm_install = subprocess.run(
            ["npm", "install", "--silent"], capture_output=True, text=True, timeout=120, cwd=js_project, check=False
        )

        if npm_install.returncode != 0:
            pytest.skip(f"npm install failed: {npm_install.stderr}")

        # Copy codeflash runtime to node_modules
        codeflash_runtime = trace_runner_path.parent
        node_modules = js_project / "node_modules"
        codeflash_pkg = node_modules / "codeflash"
        if codeflash_pkg.exists():
            shutil.rmtree(codeflash_pkg)
        codeflash_pkg.mkdir()
        runtime_dst = codeflash_pkg / "runtime"

        shutil.copytree(codeflash_runtime, runtime_dst)

        # Create package.json with proper exports for codeflash/tracer
        (codeflash_pkg / "package.json").write_text(
            json.dumps(
                {
                    "name": "codeflash",
                    "version": "1.0.0",
                    "main": "runtime/index.js",
                    "exports": {
                        ".": {"require": "./runtime/index.js"},
                        "./tracer": {"require": "./runtime/tracer.js"},
                        "./replay": {"require": "./runtime/replay.js"},
                        "./babel-tracer-plugin": {"require": "./runtime/babel-tracer-plugin.js"},
                    },
                }
            )
        )

        # Set up environment with NODE_PATH to find installed modules
        # Node.js resolves modules relative to the script location by default,
        # so we need to add the test project's node_modules to NODE_PATH
        env = os.environ.copy()
        env["NODE_PATH"] = str(node_modules.resolve())

        # Run the tracer
        result = subprocess.run(
            [
                "node",
                str(trace_runner_path),
                "--trace-db",
                str(trace_db),
                "--project-root",
                str(js_project),
                "--functions",
                json.dumps(["add", "multiply"]),
                str(js_project / "main.js"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=js_project,
            env=env,
            check=False,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        # Verify output
        assert "add(2, 3) =" in result.stdout, f"Expected output not found. stderr: {result.stderr}"

        # Verify trace database was created and contains traces
        assert trace_db.exists(), "Trace database was not created"

        conn = sqlite3.connect(trace_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM function_calls WHERE type = 'call'")
        trace_count = cursor.fetchone()[0]
        conn.close()

        assert trace_count >= 2, f"Expected at least 2 traced calls, got {trace_count}"

    def test_tracer_runner_python_integration(self, js_project: Path, tmp_path: Path) -> None:
        """Test the Python tracer_runner module."""
        from codeflash.languages.javascript.tracer_runner import (
            check_javascript_tracer_available,
            detect_test_framework,
        )

        assert check_javascript_tracer_available() is True

        framework = detect_test_framework(js_project, {})
        assert framework in ("jest", "vitest")

    def test_detect_jest_framework(self, tmp_path: Path) -> None:
        """Test Jest framework detection."""
        from codeflash.languages.javascript.tracer_runner import detect_test_framework

        (tmp_path / "jest.config.js").write_text("module.exports = {};")
        framework = detect_test_framework(tmp_path, {})
        assert framework == "jest"

    def test_detect_vitest_framework(self, tmp_path: Path) -> None:
        """Test Vitest framework detection."""
        from codeflash.languages.javascript.tracer_runner import detect_test_framework

        (tmp_path / "vitest.config.js").write_text("export default {};")
        framework = detect_test_framework(tmp_path, {})
        assert framework == "vitest"

    def test_detect_framework_from_package_json(self, tmp_path: Path) -> None:
        """Test framework detection from package.json."""
        from codeflash.languages.javascript.tracer_runner import detect_test_framework

        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "vitest run"}, "devDependencies": {"vitest": "^1.0.0"}})
        )
        framework = detect_test_framework(tmp_path, {})
        assert framework == "vitest"


class TestGetTracedFunctionsFromDb:
    """Tests for get_traced_functions_from_db function."""

    def test_get_traced_functions_from_db(self, tmp_path: Path) -> None:
        """Test extracting functions from database."""
        db_path = tmp_path / "trace.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE function_calls (
                type TEXT, function TEXT, classname TEXT, filename TEXT,
                line_number INTEGER, last_frame_address INTEGER, time_ns INTEGER, args BLOB
            )
        """)

        cursor.execute(
            "INSERT INTO function_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("call", "testFunc", None, "./src/test.js", 1, 1, 1000, "[]"),
        )

        conn.commit()
        conn.close()

        functions = get_traced_functions_from_db(db_path)

        assert len(functions) == 1
        assert functions[0].function_name == "testFunc"
        assert functions[0].module_name == "src/test"

    def test_get_traced_functions_nonexistent_file(self, tmp_path: Path) -> None:
        """Test with nonexistent database file."""
        functions = get_traced_functions_from_db(tmp_path / "nonexistent.sqlite")
        assert functions == []
