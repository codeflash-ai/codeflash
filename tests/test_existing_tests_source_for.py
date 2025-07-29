import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from codeflash.result.create_pr import existing_tests_source_for


class TestExistingTestsSourceFor:
    """Test cases for existing_tests_source_for function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock test config
        self.test_cfg = Mock()
        self.test_cfg.tests_root = Path(__file__).resolve().parent
        self.test_cfg.project_root_path = Path(__file__).resolve().parent.parent

        # Mock invocation ID
        self.mock_invocation_id = Mock()
        self.mock_invocation_id.test_module_path = "tests.test_module"
        self.mock_invocation_id.test_class_name = "TestClass"
        self.mock_invocation_id.test_function_name = "test_function"

        # Mock function called in test
        self.mock_function_called_in_test = Mock()
        self.mock_function_called_in_test.tests_in_file = Mock()
        self.mock_function_called_in_test.tests_in_file.test_file = Path(__file__).resolve().parent / "test_module.py"
        #Path to pyproject.toml
        os.chdir(self.test_cfg.project_root_path)
        

    def test_no_test_files_returns_empty_string(self):
        """Test that function returns empty string when no test files exist."""
        
        function_to_tests = {}
        original_runtimes = {}
        optimized_runtimes = {}

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        assert result == ""

    def test_single_test_with_improvement(self):
        """Test single test showing performance improvement."""
        
        function_to_tests = {
            "module.function": {self.mock_function_called_in_test}
        }
        original_runtimes = {
            self.mock_invocation_id: [1000000]  # 1ms in nanoseconds
        }
        optimized_runtimes = {
            self.mock_invocation_id: [500000]   # 0.5ms in nanoseconds
        }

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                  | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------|:--------------|:---------------|:----------|
| `test_module.py::TestClass.test_function` | 1.00ms        | 500μs          | ✅100%    |
"""

        assert result == expected

    def test_single_test_with_regression(self):
        """Test single test showing performance regression."""
        
        function_to_tests = {
            "module.function": {self.mock_function_called_in_test}
        }
        original_runtimes = {
            self.mock_invocation_id: [500000]   # 0.5ms in nanoseconds
        }
        optimized_runtimes = {
            self.mock_invocation_id: [1000000]  # 1ms in nanoseconds
        }

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                  | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------|:--------------|:---------------|:----------|
| `test_module.py::TestClass.test_function` | 500μs         | 1.00ms         | ⚠️-50.0%  |
"""

        assert result == expected

    def test_test_without_class_name(self):
        """Test function without class name (standalone test function)."""
        
        mock_invocation_no_class = Mock()
        mock_invocation_no_class.test_module_path = "tests.test_module"
        mock_invocation_no_class.test_class_name = None
        mock_invocation_no_class.test_function_name = "test_standalone"

        function_to_tests = {
            "module.function": {self.mock_function_called_in_test}
        }
        original_runtimes = {
            mock_invocation_no_class: [1000000]
        }
        optimized_runtimes = {
            mock_invocation_no_class: [800000]
        }

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function          | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:----------------------------------|:--------------|:---------------|:----------|
| `test_module.py::test_standalone` | 1.00ms        | 800μs          | ✅25.0%   |
"""

        assert result == expected

    def test_missing_original_runtime(self):
        """Test when original runtime is missing (shows NaN)."""
        
        function_to_tests = {
            "module.function": {self.mock_function_called_in_test}
        }
        original_runtimes = {}
        optimized_runtimes = {
            self.mock_invocation_id: [500000]
        }

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function   | Original ⏱️   | Optimized ⏱️   | Speedup   |
|----------------------------|---------------|----------------|-----------|
"""

        assert result == expected

    def test_missing_optimized_runtime(self):
        """Test when optimized runtime is missing (shows NaN)."""
        
        function_to_tests = {
            "module.function": {self.mock_function_called_in_test}
        }
        original_runtimes = {
            self.mock_invocation_id: [1000000]
        }
        optimized_runtimes = {}

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function   | Original ⏱️   | Optimized ⏱️   | Speedup   |
|----------------------------|---------------|----------------|-----------|
"""

        assert result == expected

    def test_multiple_tests_sorted_output(self):
        """Test multiple tests with sorted output by filename and function name."""
        # Create second test file
        
        mock_function_called_2 = Mock()
        mock_function_called_2.tests_in_file = Mock()
        mock_function_called_2.tests_in_file.test_file = Path(__file__).resolve().parent / "test_another.py"

        mock_invocation_2 = Mock()
        mock_invocation_2.test_module_path = "tests.test_another"
        mock_invocation_2.test_class_name = "TestAnother"
        mock_invocation_2.test_function_name = "test_another_function"

        function_to_tests = {
            "module.function": {self.mock_function_called_in_test, mock_function_called_2}
        }
        original_runtimes = {
            self.mock_invocation_id: [1000000],
            mock_invocation_2: [2000000]
        }
        optimized_runtimes = {
            self.mock_invocation_id: [800000],
            mock_invocation_2: [1500000]
        }

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                             | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:-----------------------------------------------------|:--------------|:---------------|:----------|
| `test_another.py::TestAnother.test_another_function` | 2.00ms        | 1.50ms         | ✅33.3%   |
| `test_module.py::TestClass.test_function`            | 1.00ms        | 800μs          | ✅25.0%   |
"""

        assert result == expected

    def test_multiple_runtimes_uses_minimum(self):
        """Test that function uses minimum runtime when multiple measurements exist."""
        
        function_to_tests = {
            "module.function": {self.mock_function_called_in_test}
        }
        original_runtimes = {
            self.mock_invocation_id: [1000000, 1200000, 800000]  # min: 800000
        }
        optimized_runtimes = {
            self.mock_invocation_id: [600000, 700000, 500000]    # min: 500000
        }

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                  | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------|:--------------|:---------------|:----------|
| `test_module.py::TestClass.test_function` | 800μs         | 500μs          | ✅60.0%   |
"""

        assert result == expected

    def test_complex_module_path_conversion(self):
        """Test conversion of complex module paths to file paths."""
        
        mock_invocation_complex = Mock()
        mock_invocation_complex.test_module_path = "tests.integration.test_complex_module"
        mock_invocation_complex.test_class_name = "TestComplex"
        mock_invocation_complex.test_function_name = "test_complex_function"

        mock_function_complex = Mock()
        mock_function_complex.tests_in_file = Mock()
        mock_function_complex.tests_in_file.test_file = Path(__file__).resolve().parent / "integration/test_complex_module.py"

        function_to_tests = {
            "module.function": {mock_function_complex}
        }
        original_runtimes = {
            mock_invocation_complex: [1000000]
        }
        optimized_runtimes = {
            mock_invocation_complex: [750000]
        }

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                                                | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------------------------------------|:--------------|:---------------|:----------|
| `integration/test_complex_module.py::TestComplex.test_complex_function` | 1.00ms        | 750μs          | ✅33.3%   |
"""

        assert result == expected

    def test_zero_runtime_values(self):
        """Test handling of zero runtime values."""
        
        function_to_tests = {
            "module.function": {self.mock_function_called_in_test}
        }
        original_runtimes = {
            self.mock_invocation_id: [0]
        }
        optimized_runtimes = {
            self.mock_invocation_id: [0]
        }

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function   | Original ⏱️   | Optimized ⏱️   | Speedup   |
|----------------------------|---------------|----------------|-----------|
"""

        assert result == expected

    def test_filters_out_generated_tests(self):
        """Test that generated tests are filtered out and only non-generated tests are included."""
        # Create a test that would be filtered out (not in non_generated_tests)
        
        mock_generated_test = Mock()
        mock_generated_test.tests_in_file = Mock()
        mock_generated_test.tests_in_file.test_file = "/project/tests/generated_test.py"

        mock_generated_invocation = Mock()
        mock_generated_invocation.test_module_path = "tests.generated_test"
        mock_generated_invocation.test_class_name = "TestGenerated"
        mock_generated_invocation.test_function_name = "test_generated"

        function_to_tests = {
            "module.function": {self.mock_function_called_in_test}
        }
        original_runtimes = {
            self.mock_invocation_id: [1000000],
            mock_generated_invocation: [500000]  # This should be filtered out
        }
        optimized_runtimes = {
            self.mock_invocation_id: [800000],
            mock_generated_invocation: [400000]  # This should be filtered out
        }

        result = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        # Should only include the non-generated test
        expected = """| Test File::Test Function                  | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------|:--------------|:---------------|:----------|
| `test_module.py::TestClass.test_function` | 1.00ms        | 800μs          | ✅25.0%   |
"""

        assert result == expected


