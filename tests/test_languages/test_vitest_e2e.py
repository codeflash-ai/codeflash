"""End-to-end integration tests for Vitest pipeline.

Tests the full optimization pipeline for Vitest projects:
- Function discovery
- Code context extraction
- Test discovery
- Test framework detection
"""

from pathlib import Path

import pytest

from codeflash.code_utils.config_js import detect_test_runner, get_package_json_data
from codeflash.discovery.functions_to_optimize import find_all_functions_in_file, get_files_for_language
from codeflash.languages.base import Language


class TestVitestProjectDiscovery:
    """Tests for function discovery in a Vitest project."""

    @pytest.fixture
    def vitest_project_dir(self):
        """Get the Vitest sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        vitest_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"
        if not vitest_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")
        return vitest_dir

    def test_detects_vitest_as_test_runner(self, vitest_project_dir):
        """Test that Vitest is detected as the test runner."""
        package_json = vitest_project_dir / "package.json"
        package_data = get_package_json_data(package_json)

        assert package_data is not None
        runner = detect_test_runner(vitest_project_dir, package_data)

        assert runner == "vitest"

    def test_discover_functions_in_fibonacci(self, vitest_project_dir):
        """Test discovering functions in fibonacci.ts."""
        fib_file = vitest_project_dir / "fibonacci.ts"
        if not fib_file.exists():
            pytest.skip("fibonacci.ts not found")

        functions = find_all_functions_in_file(fib_file)

        assert fib_file in functions
        func_list = functions[fib_file]

        # Should find the main exported functions
        func_names = {f.function_name for f in func_list}
        assert "fibonacci" in func_names
        assert "isFibonacci" in func_names
        assert "isPerfectSquare" in func_names
        assert "fibonacciSequence" in func_names

        # All should be TypeScript functions
        for func in func_list:
            assert func.language == "typescript"

    def test_discover_functions_in_string_utils(self, vitest_project_dir):
        """Test discovering functions in string_utils.ts."""
        utils_file = vitest_project_dir / "string_utils.ts"
        if not utils_file.exists():
            pytest.skip("string_utils.ts not found")

        functions = find_all_functions_in_file(utils_file)

        assert utils_file in functions
        func_list = functions[utils_file]

        func_names = {f.function_name for f in func_list}
        assert "reverseString" in func_names
        assert "isPalindrome" in func_names
        assert "countVowels" in func_names
        assert "uniqueWords" in func_names

    def test_get_typescript_files(self, vitest_project_dir):
        """Test getting TypeScript files from Vitest project directory."""
        files = get_files_for_language(vitest_project_dir, Language.TYPESCRIPT)

        # Should find .ts files
        ts_files = [f for f in files if f.suffix == ".ts" and "test" not in f.name]
        assert len(ts_files) >= 2  # fibonacci.ts, string_utils.ts

        # Should not include test files in root (they're in tests/)
        root_files = [f for f in ts_files if f.parent == vitest_project_dir]
        assert len(root_files) >= 2


class TestVitestCodeContext:
    """Tests for code context extraction in Vitest project."""

    @pytest.fixture
    def vitest_project_dir(self):
        """Get the Vitest sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        vitest_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"
        if not vitest_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")
        return vitest_dir

    def test_extract_code_context_for_typescript(self, vitest_project_dir):
        """Test extracting code context for a TypeScript function."""
        from codeflash.context.code_context_extractor import get_code_optimization_context
        from codeflash.languages import current as lang_current
        from codeflash.languages.base import Language

        # Force set language to TypeScript for proper context extraction routing
        lang_current._current_language = Language.TYPESCRIPT

        fib_file = vitest_project_dir / "fibonacci.ts"
        if not fib_file.exists():
            pytest.skip("fibonacci.ts not found")

        functions = find_all_functions_in_file(fib_file)
        func_list = functions[fib_file]

        # Find the fibonacci function
        fib_func = next((f for f in func_list if f.function_name == "fibonacci"), None)
        assert fib_func is not None

        # Extract code context
        context = get_code_optimization_context(fib_func, vitest_project_dir)

        # Verify context structure
        assert context.read_writable_code is not None
        assert context.read_writable_code.language == "typescript"
        assert len(context.read_writable_code.code_strings) > 0

        # The code should contain the function
        code = context.read_writable_code.code_strings[0].code
        assert "fibonacci" in code


class TestVitestTestDiscovery:
    """Tests for Vitest test discovery."""

    @pytest.fixture
    def vitest_project_dir(self):
        """Get the Vitest sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        vitest_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"
        if not vitest_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")
        return vitest_dir

    def test_discover_vitest_tests(self, vitest_project_dir):
        """Test discovering Vitest tests for TypeScript functions."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo, Language

        ts_support = get_language_support(Language.TYPESCRIPT)
        test_root = vitest_project_dir / "tests"

        if not test_root.exists():
            pytest.skip("tests directory not found")

        # Create FunctionInfo for fibonacci function
        fib_file = vitest_project_dir / "fibonacci.ts"
        func_info = FunctionInfo(
            name="fibonacci", file_path=fib_file, start_line=11, end_line=16, language=Language.TYPESCRIPT
        )

        # Discover tests
        tests = ts_support.discover_tests(test_root, [func_info])

        # Should find tests for fibonacci
        assert func_info.qualified_name in tests or "fibonacci" in str(tests)


class TestVitestRunnerDispatch:
    """Tests for Vitest runner dispatch in support.py."""

    @pytest.fixture
    def vitest_project_dir(self):
        """Get the Vitest sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        vitest_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"
        if not vitest_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")
        return vitest_dir

    def test_language_support_has_test_framework_property(self):
        """Test that JavaScriptSupport has test_framework property."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import Language

        js_support = get_language_support(Language.JAVASCRIPT)
        ts_support = get_language_support(Language.TYPESCRIPT)

        # Default test framework should be jest
        assert js_support.test_framework == "jest"
        assert ts_support.test_framework == "jest"

    def test_behavioral_tests_accepts_test_framework(self):
        """Test that run_behavioral_tests accepts test_framework parameter."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import Language

        ts_support = get_language_support(Language.TYPESCRIPT)

        # Check signature has test_framework parameter
        import inspect

        sig = inspect.signature(ts_support.run_behavioral_tests)
        params = list(sig.parameters.keys())
        assert "test_framework" in params

    def test_benchmarking_tests_accepts_test_framework(self):
        """Test that run_benchmarking_tests accepts test_framework parameter."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import Language

        ts_support = get_language_support(Language.TYPESCRIPT)

        import inspect

        sig = inspect.signature(ts_support.run_benchmarking_tests)
        params = list(sig.parameters.keys())
        assert "test_framework" in params

    def test_line_profile_tests_accepts_test_framework(self):
        """Test that run_line_profile_tests accepts test_framework parameter."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import Language

        ts_support = get_language_support(Language.TYPESCRIPT)

        import inspect

        sig = inspect.signature(ts_support.run_line_profile_tests)
        params = list(sig.parameters.keys())
        assert "test_framework" in params


class TestVitestVsJestDetection:
    """Tests comparing Vitest and Jest project detection."""

    @pytest.fixture
    def jest_project_dir(self):
        """Get the Jest sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        jest_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_ts"
        if not jest_dir.exists():
            pytest.skip("code_to_optimize_ts directory not found")
        return jest_dir

    @pytest.fixture
    def vitest_project_dir(self):
        """Get the Vitest sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        vitest_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"
        if not vitest_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")
        return vitest_dir

    def test_jest_detected_in_jest_project(self, jest_project_dir):
        """Test that Jest is detected in the Jest project."""
        package_json = jest_project_dir / "package.json"
        package_data = get_package_json_data(package_json)

        assert package_data is not None
        runner = detect_test_runner(jest_project_dir, package_data)

        assert runner == "jest"

    def test_vitest_detected_in_vitest_project(self, vitest_project_dir):
        """Test that Vitest is detected in the Vitest project."""
        package_json = vitest_project_dir / "package.json"
        package_data = get_package_json_data(package_json)

        assert package_data is not None
        runner = detect_test_runner(vitest_project_dir, package_data)

        assert runner == "vitest"

    def test_vitest_prioritized_over_jest(self, tmp_path):
        """Test that Vitest is prioritized when both are present."""
        import json

        package_json = tmp_path / "package.json"
        package_json.write_text(
            json.dumps(
                {
                    "name": "test",
                    "devDependencies": {
                        "vitest": "^2.0.0",
                        "jest": "^29.0.0",
                    },
                }
            )
        )
        package_data = get_package_json_data(package_json)

        runner = detect_test_runner(tmp_path, package_data)

        assert runner == "vitest"
