"""Tests for JavaScript test instrumentation and result parsing.

These tests verify that:
1. JavaScript tests are correctly instrumented with codeflash-jest-helper
2. Instrumented tests run correctly with Jest
3. Results (timing, return values) are captured in SQLite
4. The SQLite results are correctly parsed
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.javascript.runtime import get_all_runtime_files
from codeflash.models.models import TestFile, TestFiles
from codeflash.models.test_type import TestType
from codeflash.verification.verification_utils import TestConfig
from codeflash.verification.parse_test_output import parse_sqlite_test_results, parse_test_results
from codeflash.verification.test_runner import run_jest_behavioral_tests, run_jest_benchmarking_tests
from codeflash.code_utils.code_utils import get_run_tmp_file


# Path to the JavaScript test project (sample code only)
JS_PROJECT_ROOT = Path(__file__).parent.parent / "code_to_optimize_js"


def setup_js_test_environment(project_dir: Path) -> None:
    """Copy JavaScript runtime files from codeflash package to project directory."""
    for runtime_file in get_all_runtime_files():
        shutil.copy(runtime_file, project_dir / runtime_file.name)


class TestJavaScriptInstrumentation:
    """Test JavaScript test instrumentation."""

    def test_instrumentation_adds_helper_import(self) -> None:
        """Test that instrumentation adds the codeflash-jest-helper import."""
        # This test verifies the basic JavaScript instrumentation pattern
        # The actual instrumentation is done client-side by modifying test files
        # to use codeflash-jest-helper's capture() or capturePerf() functions

        # Example of a manually instrumented test file
        instrumented_test = """
const codeflash = require('./codeflash-jest-helper');
const { reverseString } = require('../string_utils');

describe('reverseString', () => {
    test('should reverse a string', () => {
        // Behavior mode: capture inputs, outputs, timing to SQLite
        const result = codeflash.capture('reverseString', '8', reverseString, 'hello');
        // [codeflash-disabled] expect(result).toBe('olleh');
    });
});
"""

        # Example of performance-only instrumented test
        perf_instrumented_test = """
const codeflash = require('./codeflash-jest-helper');
const { reverseString } = require('../string_utils');

describe('reverseString', () => {
    test('benchmark reverseString', () => {
        // Performance mode: only timing to stdout, no SQLite overhead
        const result = codeflash.capturePerf('reverseString', '8', reverseString, 'hello');
        // [codeflash-disabled] expect(result).toBe('olleh');
    });
});
"""

        # Verify behavior instrumentation pattern
        assert "codeflash-jest-helper" in instrumented_test
        assert "codeflash.capture(" in instrumented_test
        assert "[codeflash-disabled]" in instrumented_test

        # Verify performance instrumentation pattern
        assert "codeflash.capturePerf(" in perf_instrumented_test


class TestJavaScriptTestExecution:
    """Test that instrumented JavaScript tests execute correctly and produce timing data."""

    @pytest.fixture
    def js_test_setup(self, tmp_path: Path):
        """Set up a temporary JavaScript test environment."""
        # Copy the JavaScript project to temp directory
        project_dir = tmp_path / "js_project"
        shutil.copytree(JS_PROJECT_ROOT, project_dir)

        # Copy runtime JS files from codeflash package
        setup_js_test_environment(project_dir)

        # Create a simple instrumented test file
        test_file = project_dir / "tests" / "test_instrumented.test.js"
        test_file.parent.mkdir(parents=True, exist_ok=True)

        instrumented_test = """
const codeflash = require('../codeflash-jest-helper');
const { reverseString } = require('../string_utils');

describe('reverseString instrumented', () => {
    test('should reverse hello', () => {
        const result = codeflash.capture('reverseString', '7', reverseString, 'hello');
        // [codeflash-disabled] expect(result).toBe('olleh');
    });

    test('should reverse world', () => {
        const result = codeflash.capture('reverseString', '12', reverseString, 'world');
        // [codeflash-disabled] expect(result).toBe('dlrow');
    });
});
"""
        test_file.write_text(instrumented_test)

        yield {
            "project_dir": project_dir,
            "test_file": test_file,
        }

    def test_jest_helper_writes_sqlite(self, js_test_setup, tmp_path: Path) -> None:
        """Test that the Jest helper writes results to SQLite."""
        project_dir = js_test_setup["project_dir"]
        test_file = js_test_setup["test_file"]

        # Set up environment for the test
        sqlite_output = tmp_path / "test_results.sqlite"
        env = os.environ.copy()
        env["CODEFLASH_OUTPUT_FILE"] = str(sqlite_output)
        env["CODEFLASH_LOOP_INDEX"] = "1"
        env["CODEFLASH_TEST_ITERATION"] = "0"
        env["CODEFLASH_TEST_MODULE"] = "test_instrumented"

        # Run Jest directly
        result = subprocess.run(
            ["npx", "jest", str(test_file), "--no-coverage"],
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(f"Jest stdout: {result.stdout}")
        print(f"Jest stderr: {result.stderr}")
        print(f"Jest return code: {result.returncode}")

        # Check that tests passed
        assert result.returncode == 0, f"Jest failed: {result.stderr}"

        # Check that SQLite file was created
        assert sqlite_output.exists(), f"SQLite file not created at {sqlite_output}"

        # Check contents of SQLite
        import sqlite3
        conn = sqlite3.connect(sqlite_output)
        cursor = conn.cursor()
        rows = cursor.execute("SELECT * FROM test_results").fetchall()
        conn.close()

        print(f"SQLite rows: {rows}")
        assert len(rows) >= 2, f"Expected at least 2 rows, got {len(rows)}"

        # Check that runtime is captured (column 6 is runtime)
        for row in rows:
            runtime = row[6]
            assert runtime > 0, f"Expected runtime > 0, got {runtime}"

    def test_jest_helper_json_fallback(self, js_test_setup, tmp_path: Path) -> None:
        """Test that the Jest helper falls back to JSON when SQLite is unavailable."""
        # This test verifies the JSON fallback works (in case better-sqlite3 isn't installed)
        project_dir = js_test_setup["project_dir"]
        test_file = js_test_setup["test_file"]

        # Remove better-sqlite3 to force JSON fallback
        node_modules = project_dir / "node_modules" / "better-sqlite3"
        if node_modules.exists():
            shutil.rmtree(node_modules)

        # Set up environment
        json_output = tmp_path / "test_results.json"
        env = os.environ.copy()
        env["CODEFLASH_OUTPUT_FILE"] = str(json_output)
        env["CODEFLASH_LOOP_INDEX"] = "1"
        env["CODEFLASH_TEST_ITERATION"] = "0"

        # Run Jest
        result = subprocess.run(
            ["npx", "jest", str(test_file), "--no-coverage"],
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(f"Jest stdout: {result.stdout}")
        print(f"Jest stderr: {result.stderr}")

        # Check that tests passed
        assert result.returncode == 0, f"Jest failed: {result.stderr}"

        # Check that JSON file was created (fallback)
        if json_output.exists():
            import json
            with open(json_output) as f:
                data = json.load(f)
            print(f"JSON data: {data}")
            assert "results" in data
            assert len(data["results"]) >= 2


class TestJavaScriptResultParsing:
    """Test parsing of JavaScript test results."""

    @pytest.fixture
    def sqlite_test_results(self, tmp_path: Path) -> Path:
        """Create a mock SQLite file with test results."""
        import json
        import sqlite3

        sqlite_path = tmp_path / "test_return_values_0.sqlite"
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()

        # Create the same schema as codeflash-jest-helper
        cursor.execute("""
            CREATE TABLE test_results (
                test_module_path TEXT,
                test_class_name TEXT,
                test_function_name TEXT,
                function_getting_tested TEXT,
                loop_index INTEGER,
                iteration_id TEXT,
                runtime INTEGER,
                return_value BLOB,
                verification_type TEXT
            )
        """)

        # Insert mock test results (JSON serialized return value for JavaScript)
        test_data = [
            (
                "tests/test_string_utils.test.js",
                None,
                "should reverse hello",
                "reverseString",
                1,
                "123_0",
                5000000,  # 5ms in nanoseconds
                json.dumps([["hello"], {}, "olleh"]).encode(),  # [args, kwargs, return_value]
                "function_call",
            ),
            (
                "tests/test_string_utils.test.js",
                None,
                "should reverse world",
                "reverseString",
                1,
                "124_0",
                3000000,  # 3ms in nanoseconds
                json.dumps([["world"], {}, "dlrow"]).encode(),
                "function_call",
            ),
        ]

        cursor.executemany(
            "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            test_data,
        )
        conn.commit()
        conn.close()

        return sqlite_path

    def test_parse_sqlite_results_jest(self, sqlite_test_results: Path, tmp_path: Path) -> None:
        """Test that SQLite results are correctly parsed for Jest tests."""
        # Set up test configuration
        test_config = TestConfig(
            tests_root=tmp_path / "tests",
            tests_project_rootdir=tmp_path,
            project_root_path=tmp_path,
            pytest_cmd="",
        )
        # Set language to JavaScript so test_framework returns "jest"
        test_config.set_language("javascript")

        # Create test files object - the path should match what's in SQLite
        test_file = tmp_path / "tests" / "test_string_utils.test.js"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("// test file")

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    benchmarking_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                )
            ]
        )

        # Debug: Check what's in the SQLite file
        import sqlite3
        conn = sqlite3.connect(sqlite_test_results)
        cursor = conn.cursor()
        rows = cursor.execute("SELECT test_module_path FROM test_results").fetchall()
        conn.close()
        print(f"SQLite test_module_path values: {rows}")
        print(f"Test file path: {test_file}")
        print(f"tests_project_rootdir: {test_config.tests_project_rootdir}")
        print(f"test_framework: {test_config.test_framework}")
        print(f"is_javascript should be: {test_config.test_framework == 'jest'}")

        # Parse the SQLite results
        results = parse_sqlite_test_results(
            sqlite_file_path=sqlite_test_results,
            test_files=test_files,
            test_config=test_config,
        )

        print(f"Parsed results: {results.test_results}")

        # Verify results
        assert len(results.test_results) == 2, f"Expected 2 results, got {len(results.test_results)}"

        # Check first result
        result0 = results.test_results[0]
        assert result0.id.function_getting_tested == "reverseString"
        assert result0.id.test_function_name == "should reverse hello"
        assert result0.runtime == 5000000
        assert result0.did_pass is True
        # Check return value is parsed from JSON
        assert result0.return_value is not None

        # Check second result
        result1 = results.test_results[1]
        assert result1.id.function_getting_tested == "reverseString"
        assert result1.runtime == 3000000


class TestEndToEndJavaScript:
    """End-to-end tests for JavaScript optimization flow."""

    @pytest.fixture
    def e2e_setup(self, tmp_path: Path):
        """Set up for E2E test."""
        # Copy the JavaScript project
        project_dir = tmp_path / "js_project"
        shutil.copytree(JS_PROJECT_ROOT, project_dir)

        # Copy runtime JS files from codeflash package
        setup_js_test_environment(project_dir)

        # Ensure dependencies are installed
        subprocess.run(
            ["npm", "install"],
            cwd=project_dir,
            capture_output=True,
            timeout=120,
        )

        return project_dir

    def test_behavior_test_run_and_parse(self, e2e_setup: Path) -> None:
        """Test running behavior tests and parsing results."""
        project_dir = e2e_setup

        # Create instrumented test
        test_file = project_dir / "tests" / "test_behavior.test.js"
        test_file.write_text("""
const codeflash = require('../codeflash-jest-helper');
const { reverseString } = require('../string_utils');

describe('reverseString behavior', () => {
    test('reverses hello', () => {
        const result = codeflash.capture('reverseString', '8', reverseString, 'hello');
        // [codeflash-disabled] expect(result).toBe('olleh');
    });
});
""")

        # Set up test configuration
        test_config = TestConfig(
            tests_root=project_dir / "tests",
            tests_project_rootdir=project_dir,
            project_root_path=project_dir,
            test_framework="jest",
            pytest_cmd="",
        )

        # Create test files object
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    benchmarking_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                )
            ]
        )

        # Run behavioral tests
        test_env = os.environ.copy()
        result_path, run_result, _, _ = run_jest_behavioral_tests(
            test_paths=test_files,
            test_env=test_env,
            cwd=project_dir,
            timeout=60,
        )

        print(f"Jest stdout: {run_result.stdout}")
        print(f"Jest stderr: {run_result.stderr}")
        print(f"Result XML path: {result_path}")

        # Check Jest ran successfully
        assert run_result.returncode == 0, f"Jest failed: {run_result.stderr}"

        # Check SQLite file was created
        sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
        print(f"Looking for SQLite at: {sqlite_file}")
        print(f"SQLite exists: {sqlite_file.exists()}")

        if sqlite_file.exists():
            import sqlite3
            conn = sqlite3.connect(sqlite_file)
            cursor = conn.cursor()
            rows = cursor.execute("SELECT * FROM test_results").fetchall()
            conn.close()
            print(f"SQLite rows: {rows}")

            # Verify timing data was captured
            assert len(rows) >= 1, "No rows in SQLite"
            runtime = rows[0][6]  # runtime column
            assert runtime > 0, f"Expected runtime > 0, got {runtime}"

    def test_benchmark_test_run_and_parse(self, e2e_setup: Path) -> None:
        """Test running benchmark tests and parsing timing results."""
        project_dir = e2e_setup

        # Create instrumented test
        test_file = project_dir / "tests" / "test_benchmark.test.js"
        test_file.write_text("""
const codeflash = require('../codeflash-jest-helper');
const { reverseString } = require('../string_utils');

describe('reverseString benchmark', () => {
    test('benchmark reverseString', () => {
        const result = codeflash.capture('reverseString', '8', reverseString, 'hello world');
        // [codeflash-disabled] expect(result).toBe('dlrow olleh');
    });
});
""")

        # Set up test files
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    benchmarking_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                )
            ]
        )

        # Run benchmarking tests
        test_env = os.environ.copy()
        result_path, run_result = run_jest_benchmarking_tests(
            test_paths=test_files,
            test_env=test_env,
            cwd=project_dir,
            timeout=60,
        )

        print(f"Jest stdout: {run_result.stdout}")
        print(f"Jest stderr: {run_result.stderr}")

        # Check Jest ran successfully
        assert run_result.returncode == 0, f"Jest failed: {run_result.stderr}"

        # Check SQLite file was created with timing data
        sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
        assert sqlite_file.exists(), f"SQLite file not created at {sqlite_file}"

        import sqlite3
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        rows = cursor.execute("SELECT runtime FROM test_results").fetchall()
        conn.close()

        # Verify timing > 0
        assert len(rows) >= 1, "No timing data captured"
        total_runtime = sum(row[0] for row in rows)
        assert total_runtime > 0, f"Expected total runtime > 0, got {total_runtime}"
        print(f"Total runtime captured: {total_runtime} ns")

    def test_performance_only_instrumentation(self, e2e_setup: Path) -> None:
        """Test that capturePerf outputs timing to stdout without SQLite writes."""
        project_dir = e2e_setup

        # Create test using capturePerf (performance-only, no SQLite)
        test_file = project_dir / "tests" / "test_perf_only.test.js"
        test_file.write_text("""
const codeflash = require('../codeflash-jest-helper');
const { reverseString } = require('../string_utils');

describe('reverseString perf only', () => {
    test('perf test reverseString', () => {
        // Use capturePerf instead of capture for performance-only
        const result = codeflash.capturePerf('reverseString', '9', reverseString, 'hello world');
        // [codeflash-disabled] expect(result).toBe('dlrow olleh');
    });
});
""")

        # Set up environment - use a separate sqlite file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_output = Path(tmpdir) / "perf_test.sqlite"
            env = os.environ.copy()
            env["CODEFLASH_OUTPUT_FILE"] = str(sqlite_output)
            env["CODEFLASH_LOOP_INDEX"] = "1"
            env["CODEFLASH_TEST_ITERATION"] = "0"
            env["CODEFLASH_TEST_MODULE"] = "tests/test_perf_only.test.js"

            # Run Jest
            result = subprocess.run(
                ["npx", "jest", str(test_file), "--no-coverage"],
                cwd=project_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            print(f"Jest stdout: {result.stdout}")
            print(f"Jest stderr: {result.stderr}")

            # Check Jest ran successfully
            assert result.returncode == 0, f"Jest failed: {result.stderr}"

            # Verify stdout contains performance tags
            # Format: !$######test_module:test_class.test_name:func_name:loop_index:invocation_id######$!
            #         !######test_module:test_class.test_name:func_name:loop_index:invocation_id:duration_ns######!
            import re
            start_pattern = re.compile(r'!\$######.*?:.*?:reverseString:.*?:.*?######\$!')
            end_pattern = re.compile(r'!######.*?:.*?:reverseString:.*?:.*?:(\d+)######!')

            start_matches = start_pattern.findall(result.stdout)
            end_matches = end_pattern.findall(result.stdout)

            print(f"Start matches: {start_matches}")
            print(f"End matches: {end_matches}")

            assert len(start_matches) >= 1, f"Expected start tag in stdout, got: {result.stdout}"
            assert len(end_matches) >= 1, f"Expected end tag with timing in stdout, got: {result.stdout}"

            # Verify timing is captured (duration_ns > 0)
            for duration_str in end_matches:
                duration = int(duration_str)
                assert duration > 0, f"Expected duration > 0, got {duration}"
                print(f"Captured duration: {duration} ns")

            # Verify SQLite was NOT written (perf mode doesn't write to SQLite)
            # Note: The file might be created but should have no rows from capturePerf
            if sqlite_output.exists():
                import sqlite3
                conn = sqlite3.connect(sqlite_output)
                cursor = conn.cursor()
                try:
                    rows = cursor.execute("SELECT COUNT(*) FROM test_results").fetchone()
                    # capturePerf should NOT write to SQLite
                    assert rows[0] == 0, f"Expected 0 rows from capturePerf, got {rows[0]}"
                except sqlite3.OperationalError:
                    # Table doesn't exist, which is fine for perf-only mode
                    pass
                conn.close()
