"""
Comprehensive tests for JavaScript test instrumentation using run_and_parse_tests.

These tests verify the full JavaScript optimization workflow:
1. Behavior mode: instrumented tests capture inputs, outputs, timing to SQLite
2. Performance mode: instrumented tests capture timing to stdout
3. Result parsing via the same path codeflash uses internally
4. Various Jest test patterns (describe, it, test, nested describe, test.each)
5. Special character handling in test names

The tests write un-instrumented JavaScript tests, then use the instrumentation
approach to transform them before running.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.javascript.runtime import get_all_runtime_files
from codeflash.models.models import TestFile, TestFiles, TestingMode, TestType
from codeflash.optimization.optimizer import Optimizer


# Path to the JavaScript test project (sample code only)
JS_PROJECT_ROOT = Path(__file__).parent.parent / "code_to_optimize_js"


def setup_js_test_environment(tmp_path: Path) -> Path:
    """Set up a temporary JavaScript test environment.

    Copies sample code from code_to_optimize_js and runtime files from
    codeflash/languages/javascript/runtime/.

    Args:
        tmp_path: Pytest's temporary path fixture.

    Returns:
        Path to the project directory.
    """
    project_dir = tmp_path / "js_project"
    shutil.copytree(JS_PROJECT_ROOT, project_dir)

    # Copy runtime JS files from codeflash package
    for runtime_file in get_all_runtime_files():
        shutil.copy(runtime_file, project_dir / runtime_file.name)

    # Ensure node_modules exist (npm install)
    if not (project_dir / "node_modules").exists():
        subprocess.run(
            ["npm", "install"],
            cwd=project_dir,
            capture_output=True,
            timeout=120,
        )

    # Create tests directory
    tests_dir = project_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    return project_dir


def instrument_javascript_test(
    test_source: str, function_name: str, mode: str = "behavior"
) -> str:
    """Instrument a JavaScript test file with codeflash helper.

    This transforms un-instrumented Jest tests by:
    1. Adding the codeflash-jest-helper import
    2. Wrapping function calls with capture/capturePerf/capturePerfLooped

    Args:
        test_source: The un-instrumented test source code.
        function_name: The name of the function to instrument.
        mode: The instrumentation mode - 'behavior', 'performance', or 'looped'.

    Returns:
        The instrumented test source code.

    """
    # Add helper import at the top (after any existing imports)
    helper_import = "const codeflash = require('../codeflash-jest-helper');\n"

    if "codeflash-jest-helper" not in test_source:
        # Find the first non-import line to insert the helper import
        lines = test_source.split("\n")
        insert_pos = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("//") and not stripped.startswith("const") and not stripped.startswith("import"):
                insert_pos = i
                break
            if stripped.startswith("const") or stripped.startswith("import"):
                insert_pos = i + 1
        lines.insert(insert_pos, helper_import.rstrip())
        test_source = "\n".join(lines)

    # Choose the capture function based on mode
    if mode == "behavior":
        capture_fn = "codeflash.capture"
    elif mode == "performance":
        capture_fn = "codeflash.capturePerf"
    else:  # looped
        capture_fn = "codeflash.capturePerfLooped"

    # Find function calls and wrap them with capture
    # This is a simplified transformer - in production, you'd use a proper AST parser
    line_id_counter = [0]  # Use list to allow modification in closure

    # Pattern to match function calls: functionName(args) but NOT when preceded by codeflash.
    # and NOT when it's part of require() or a method call
    # Also handles 'await functionName(args)' by including the await in the capture
    # Captures optional "function " or "async function " prefix to skip function definitions
    pattern = rf"((?:async\s+)?function\s+)?(\bawait\s+)?(?<![.\'\"])(?<!/)\b{re.escape(function_name)}\s*\(([^)]*)\)"

    def replace_call(match):
        func_def_prefix = match.group(1)  # "function " or "async function " or None
        # If this is a function definition, don't transform it
        if func_def_prefix:
            return match.group(0)  # Return original match unchanged
        await_prefix = match.group(2) or ""  # "await " or empty string
        args = match.group(3).strip() if match.group(3) else ""
        line_id_counter[0] += 1
        line_id = str(line_id_counter[0])
        if args:
            return f"{await_prefix}{capture_fn}('{function_name}', '{line_id}', {function_name}, {args})"
        return f"{await_prefix}{capture_fn}('{function_name}', '{line_id}', {function_name})"

    # Replace function calls with instrumented versions
    instrumented = re.sub(pattern, replace_call, test_source)

    return instrumented


class TestJavaScriptBehaviorMode:
    """Test JavaScript behavior mode using run_and_parse_tests."""

    @pytest.fixture
    def js_test_setup(self, tmp_path: Path):
        """Set up a temporary JavaScript test environment."""
        return setup_js_test_environment(tmp_path)

    def test_behavior_mode_basic(self, js_test_setup: Path) -> None:
        """Test basic behavior mode captures inputs, outputs, and timing."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented test file
        uninstrumented_source = """
const { reverseString } = require('../string_utils');

describe('reverseString behavior', () => {
    test('reverses hello', () => {
        const result = reverseString('hello');
        // [codeflash-disabled] expect(result).toBe('olleh');
    });

    test('reverses world', () => {
        const result = reverseString('world');
        // [codeflash-disabled] expect(result).toBe('dlrow');
    });
});
"""
        # Instrument the test
        instrumented_source = instrument_javascript_test(uninstrumented_source, "reverseString", mode="behavior")

        # Write the instrumented test to disk
        test_file = tests_dir / "test_behavior_basic.test.js"
        test_file.write_text(instrumented_source)

        # Set up FunctionToOptimize for JavaScript
        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        # Set up Optimizer
        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        # Set up test files
        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        # Set up test environment
        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        # Run and parse tests using the same method codeflash uses
        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, coverage_data = func_opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Verify results
        assert len(test_results) >= 2, f"Expected at least 2 test results, got {len(test_results)}"

        # Check all tests passed
        for result in test_results:
            assert result.did_pass, f"Test {result.id.test_function_name} failed"

        # Check function name was captured
        function_names = [r.id.function_getting_tested for r in test_results]
        assert "reverseString" in function_names, f"Expected reverseString in {function_names}"

        # Check runtime was captured (should be > 0)
        for result in test_results:
            if result.runtime is not None:
                assert result.runtime > 0, f"Expected runtime > 0, got {result.runtime}"

    def test_behavior_mode_multiple_functions(self, js_test_setup: Path) -> None:
        """Test behavior mode with multiple different functions."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented test file testing multiple functions
        uninstrumented_source = """
const { reverseString, countOccurrences, isPalindrome } = require('../string_utils');

describe('string_utils functions', () => {
    test('reverseString works', () => {
        const result = reverseString('abc');
    });

    test('countOccurrences works', () => {
        const result = countOccurrences('hello hello', 'hello');
    });

    test('isPalindrome works', () => {
        const result = isPalindrome('racecar');
    });
});
"""
        # Instrument each function separately and combine
        temp = instrument_javascript_test(uninstrumented_source, "reverseString", mode="behavior")
        temp = instrument_javascript_test(temp, "countOccurrences", mode="behavior")
        instrumented_source = instrument_javascript_test(temp, "isPalindrome", mode="behavior")

        test_file = tests_dir / "test_multi_func.test.js"
        test_file.write_text(instrumented_source)

        # Set up for reverseString as the main function
        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Verify all 3 tests ran
        assert len(test_results) >= 3, f"Expected at least 3 results, got {len(test_results)}"

        # Check different functions were tested
        function_names = {r.id.function_getting_tested for r in test_results}
        assert "reverseString" in function_names
        assert "countOccurrences" in function_names
        assert "isPalindrome" in function_names

    def test_behavior_mode_nested_describe(self, js_test_setup: Path) -> None:
        """Test behavior mode with nested describe blocks."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented test
        uninstrumented_source = """
const { reverseString } = require('../string_utils');

describe('String Utils', () => {
    describe('reverseString', () => {
        describe('basic cases', () => {
            test('reverses simple string', () => {
                const result = reverseString('abc');
            });
        });

        describe('edge cases', () => {
            test('handles empty string', () => {
                const result = reverseString('');
            });
        });
    });
});
"""
        instrumented_source = instrument_javascript_test(uninstrumented_source, "reverseString", mode="behavior")
        test_file = tests_dir / "test_nested.test.js"
        test_file.write_text(instrumented_source)

        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Verify nested tests ran
        assert len(test_results) >= 2, f"Expected at least 2 results, got {len(test_results)}"

        # All should pass
        for result in test_results:
            assert result.did_pass

    def test_behavior_mode_multiple_calls_same_test(self, js_test_setup: Path) -> None:
        """Test behavior mode with multiple function calls in the same test."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented test with multiple calls
        uninstrumented_source = """
const { reverseString } = require('../string_utils');

describe('multiple calls', () => {
    test('calls reverseString multiple times', () => {
        const r1 = reverseString('hello');
        const r2 = reverseString('world');
        const r3 = reverseString('test');
    });
});
"""
        instrumented_source = instrument_javascript_test(uninstrumented_source, "reverseString", mode="behavior")
        test_file = tests_dir / "test_multi_calls.test.js"
        test_file.write_text(instrumented_source)

        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Should have 3 invocations captured
        assert len(test_results) >= 3, f"Expected at least 3 results, got {len(test_results)}"

        # Check unique iteration IDs (different line IDs)
        iteration_ids = [r.id.iteration_id for r in test_results if r.id.iteration_id]
        # Should have at least 3 unique IDs
        assert len(set(iteration_ids)) >= 3, f"Expected 3 unique iteration IDs, got {iteration_ids}"


class TestJavaScriptPerformanceMode:
    """Test JavaScript performance mode using run_and_parse_tests."""

    @pytest.fixture
    def js_test_setup(self, tmp_path: Path):
        """Set up a temporary JavaScript test environment."""
        return setup_js_test_environment(tmp_path)

    def test_performance_mode_basic(self, js_test_setup: Path) -> None:
        """Test performance mode captures timing with limited loops."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented performance test
        uninstrumented_source = """
const { reverseString } = require('../string_utils');

describe('reverseString performance', () => {
    test('benchmark reverseString', () => {
        const result = reverseString('hello world test');
    });
});
"""
        instrumented_source = instrument_javascript_test(uninstrumented_source, "reverseString", mode="performance")
        test_file = tests_dir / "test_perf_basic.test.js"
        test_file.write_text(instrumented_source)

        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,  # Limit to 1-2 loops for fast test
            pytest_max_loops=2,
            testing_time=0.1,
        )

        # Verify performance results
        assert len(test_results) >= 1, f"Expected at least 1 result, got {len(test_results)}"

        # Check timing was captured in stdout
        assert test_results.perf_stdout is not None, "Expected perf_stdout to be captured"

        # Should contain timing markers
        import re
        end_pattern = re.compile(r'!######[^#]+:(\d+)######!')
        timing_matches = end_pattern.findall(test_results.perf_stdout)

        assert len(timing_matches) >= 1, f"Expected timing markers in stdout, got: {test_results.perf_stdout[:500]}"

        # Verify timing values are positive
        for timing in timing_matches:
            assert int(timing) > 0, f"Expected timing > 0, got {timing}"

    def test_performance_mode_looped(self, js_test_setup: Path) -> None:
        """Test performance mode with capturePerfLooped for multiple iterations."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented test
        uninstrumented_source = """
const { reverseString } = require('../string_utils');

describe('reverseString looped perf', () => {
    test('looped benchmark', () => {
        const result = reverseString('test');
    });
});
"""
        instrumented_source = instrument_javascript_test(uninstrumented_source, "reverseString", mode="looped")
        test_file = tests_dir / "test_perf_looped.test.js"
        test_file.write_text(instrumented_source)

        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        # Set loop limits for the test
        test_env["CODEFLASH_MIN_LOOPS"] = "2"
        test_env["CODEFLASH_MAX_LOOPS"] = "2"
        test_env["CODEFLASH_TARGET_DURATION_MS"] = "10"  # Short for fast test

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=2,
            pytest_max_loops=2,
            testing_time=0.1,
        )

        # Verify multiple timing markers (at least 2 iterations)
        import re
        end_pattern = re.compile(r'!######[^#]+:(\d+)######!')
        timing_matches = end_pattern.findall(test_results.perf_stdout or "")

        assert len(timing_matches) >= 2, f"Expected at least 2 timing markers, got {len(timing_matches)}"


class TestJavaScriptSpecialCharacters:
    """Test special character handling in test names."""

    @pytest.fixture
    def js_test_setup(self, tmp_path: Path):
        """Set up a temporary JavaScript test environment."""
        return setup_js_test_environment(tmp_path)

    def test_special_chars_in_describe(self, js_test_setup: Path) -> None:
        """Test that special characters in describe names are sanitized."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented test with special characters in describe name
        uninstrumented_source = """
const { reverseString } = require('../string_utils');

describe('reverseString: special chars! #test (with parens)', () => {
    test('should reverse [brackets]', () => {
        const result = reverseString('hello');
    });
});
"""
        instrumented_source = instrument_javascript_test(uninstrumented_source, "reverseString", mode="performance")
        test_file = tests_dir / "test_special_chars.test.js"
        test_file.write_text(instrumented_source)

        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Test should pass even with special characters
        assert len(test_results) >= 1, "Expected at least 1 result"

        # Verify stdout tags don't contain problematic characters
        import re
        start_pattern = re.compile(r'!\$######([^#]+)######\$!')
        tags = start_pattern.findall(test_results.perf_stdout or "")

        for tag in tags:
            # Split by colon (field separator) and check individual fields
            parts = tag.split(':')
            for part in parts[:-1]:  # Exclude last part which may be numeric
                assert '!' not in part, f"Tag contains unsanitized !: {tag}"
                assert '#' not in part, f"Tag contains unsanitized #: {tag}"
                assert ' ' not in part, f"Tag contains unsanitized space: {tag}"

    def test_parametrized_test_each(self, js_test_setup: Path) -> None:
        """Test test.each parametrized tests work correctly."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented parametrized test
        uninstrumented_source = """
const { reverseString } = require('../string_utils');

describe('reverseString parametrized', () => {
    test.each([
        ['ab', 'ba'],
        ['cd', 'dc'],
    ])('reverses %s to %s', (input, expected) => {
        const result = reverseString(input);
    });
});
"""
        instrumented_source = instrument_javascript_test(uninstrumented_source, "reverseString", mode="performance")
        test_file = tests_dir / "test_each.test.js"
        test_file.write_text(instrumented_source)

        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.PERFORMANCE,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Should have results for both parametrized test cases
        import re
        end_pattern = re.compile(r'!######[^#]+:(\d+)######!')
        timing_matches = end_pattern.findall(test_results.perf_stdout or "")

        assert len(timing_matches) >= 2, f"Expected at least 2 timing results for parametrized test, got {len(timing_matches)}"


class TestJavaScriptEdgeCases:
    """Test edge cases in JavaScript instrumentation."""

    @pytest.fixture
    def js_test_setup(self, tmp_path: Path):
        """Set up a temporary JavaScript test environment."""
        return setup_js_test_environment(tmp_path)

    def test_async_function(self, js_test_setup: Path) -> None:
        """Test async function instrumentation."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented async test
        uninstrumented_source = """
async function asyncDelay(ms, value) {
    return new Promise(resolve => setTimeout(() => resolve(value), ms));
}

describe('async tests', () => {
    test('handles async function', async () => {
        const result = await asyncDelay(5, 'done');
    });
});
"""
        instrumented_source = instrument_javascript_test(uninstrumented_source, "asyncDelay", mode="behavior")
        test_file = tests_dir / "test_async.test.js"
        test_file.write_text(instrumented_source)

        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="asyncDelay",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Async test should pass
        assert len(test_results) >= 1
        for result in test_results:
            assert result.did_pass, f"Async test failed: {result}"

    def test_it_syntax(self, js_test_setup: Path) -> None:
        """Test using 'it' instead of 'test'."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented test using 'it' syntax
        uninstrumented_source = """
const { reverseString } = require('../string_utils');

describe('using it syntax', () => {
    it('should reverse a string', () => {
        const result = reverseString('hello');
    });

    it('should handle empty string', () => {
        const result = reverseString('');
    });
});
"""
        instrumented_source = instrument_javascript_test(uninstrumented_source, "reverseString", mode="behavior")
        test_file = tests_dir / "test_it_syntax.test.js"
        test_file.write_text(instrumented_source)

        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Both 'it' tests should pass
        assert len(test_results) >= 2
        for result in test_results:
            assert result.did_pass

    def test_loop_in_test_code(self, js_test_setup: Path) -> None:
        """Test loop in test code - same call site called multiple times."""
        project_dir = js_test_setup
        tests_dir = project_dir / "tests"

        # Write un-instrumented test with loop
        uninstrumented_source = """
const { reverseString } = require('../string_utils');

describe('loop in test code', () => {
    test('calls in a loop', () => {
        const inputs = ['a', 'bb', 'ccc'];
        for (const input of inputs) {
            reverseString(input);
        }
    });
});
"""
        instrumented_source = instrument_javascript_test(uninstrumented_source, "reverseString", mode="behavior")
        test_file = tests_dir / "test_loop_in_code.test.js"
        test_file.write_text(instrumented_source)

        source_file = project_dir / "string_utils.js"
        function_to_optimize = FunctionToOptimize(
            function_name="reverseString",
            file_path=source_file,
            parents=[],
            language="javascript",
        )

        opt = Optimizer(
            Namespace(
                project_root=project_dir,
                disable_telemetry=True,
                tests_root=tests_dir,
                test_framework="jest",
                pytest_cmd="",
                experiment_id=None,
                test_project_root=project_dir,
            )
        )

        # Set JavaScript-specific config
        opt.test_cfg.set_language("javascript")
        opt.test_cfg.js_project_root = project_dir

        test_files = TestFiles(
            test_files=[
                TestFile(
                    instrumented_behavior_file_path=test_file,
                    test_type=TestType.GENERATED_REGRESSION,
                    original_file_path=test_file,
                    benchmarking_file_path=test_file,
                )
            ]
        )

        test_env = os.environ.copy()
        test_env["CODEFLASH_TEST_ITERATION"] = "0"
        test_env["CODEFLASH_LOOP_INDEX"] = "1"

        func_opt = opt.create_function_optimizer(function_to_optimize)
        test_results, _ = func_opt.run_and_parse_tests(
            testing_type=TestingMode.BEHAVIOR,
            test_env=test_env,
            test_files=test_files,
            optimization_iteration=0,
            pytest_min_loops=1,
            pytest_max_loops=1,
            testing_time=0.1,
        )

        # Should have 3 invocations (loop runs 3 times)
        assert len(test_results) >= 3, f"Expected at least 3 results from loop, got {len(test_results)}"

        # Check incrementing invocation indices (same line ID)
        iteration_ids = [r.id.iteration_id for r in test_results if r.id.iteration_id]

        # Should have indices like 1_0, 1_1, 1_2 (line ID 1 from instrumentation, invocations 0, 1, 2)
        assert any("_0" in str(iter_id) for iter_id in iteration_ids), f"Expected _0 in {iteration_ids}"
        assert any("_1" in str(iter_id) for iter_id in iteration_ids), f"Expected _1 in {iteration_ids}"
        assert any("_2" in str(iter_id) for iter_id in iteration_ids), f"Expected _2 in {iteration_ids}"
