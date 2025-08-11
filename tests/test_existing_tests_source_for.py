from unittest.mock import Mock
import contextlib
import os
import shutil
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

        result, _, _ = existing_tests_source_for(
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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                  | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------|:--------------|:---------------|:----------|
| `test_module.py::TestClass.test_function` | 1.00ms        | 500μs          | 100%✅    |
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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                  | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------|:--------------|:---------------|:----------|
| `test_module.py::TestClass.test_function` | 500μs         | 1.00ms         | -50.0%⚠️  |
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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function          | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:----------------------------------|:--------------|:---------------|:----------|
| `test_module.py::test_standalone` | 1.00ms        | 800μs          | 25.0%✅   |
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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = ""

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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = ""

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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                             | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:-----------------------------------------------------|:--------------|:---------------|:----------|
| `test_another.py::TestAnother.test_another_function` | 2.00ms        | 1.50ms         | 33.3%✅   |
| `test_module.py::TestClass.test_function`            | 1.00ms        | 800μs          | 25.0%✅   |
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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                  | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------|:--------------|:---------------|:----------|
| `test_module.py::TestClass.test_function` | 800μs         | 500μs          | 60.0%✅   |
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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = """| Test File::Test Function                                                | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------------------------------------|:--------------|:---------------|:----------|
| `integration/test_complex_module.py::TestComplex.test_complex_function` | 1.00ms        | 750μs          | 33.3%✅   |
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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        expected = ""

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

        result, _, _ = existing_tests_source_for(
            "module.function",
            function_to_tests,
            self.test_cfg,
            original_runtimes,
            optimized_runtimes
        )

        # Should only include the non-generated test
        expected = """| Test File::Test Function                  | Original ⏱️   | Optimized ⏱️   | Speedup   |
|:------------------------------------------|:--------------|:---------------|:----------|
| `test_module.py::TestClass.test_function` | 1.00ms        | 800μs          | 25.0%✅   |
"""

        assert result == expected

@dataclass(frozen=True)
class MockInvocationId:
    """Mocks codeflash.models.models.InvocationId"""
    test_module_path: str
    test_function_name: str
    test_class_name: Optional[str] = None


@dataclass(frozen=True)
class MockTestsInFile:
    """Mocks a part of codeflash.models.models.FunctionCalledInTest"""
    test_file: Path


@dataclass(frozen=True)
class MockFunctionCalledInTest:
    """Mocks codeflash.models.models.FunctionCalledInTest"""
    tests_in_file: MockTestsInFile


@dataclass(frozen=True)
class MockTestConfig:
    """Mocks codeflash.verification.verification_utils.TestConfig"""
    tests_root: Path


@contextlib.contextmanager
def temp_project_dir():
    """A context manager to create and chdir into a temporary project directory."""
    original_cwd = os.getcwd()
    # Use a unique name to avoid conflicts in /tmp
    project_root = Path(f"/tmp/test_project_{os.getpid()}").resolve()
    try:
        project_root.mkdir(exist_ok=True, parents=True)
        os.chdir(project_root)
        yield project_root
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(project_root, ignore_errors=True)


class ExistingTestsSourceForTests(unittest.TestCase):
    def setUp(self):
        self.func_qual_name = "my_module.my_function"
        # A default test_cfg for tests that don't rely on file system.
        self.test_cfg = MockTestConfig(tests_root=Path("/tmp/tests"))

    def test_no_tests_for_function(self):
        """Test case where no tests are found for the given function."""
        existing, replay, concolic = existing_tests_source_for(
            function_qualified_name_with_modules_from_root=self.func_qual_name,
            function_to_tests={},
            test_cfg=self.test_cfg,
            original_runtimes_all={},
            optimized_runtimes_all={},
        )
        self.assertEqual(existing, "")
        self.assertEqual(replay, "")
        self.assertEqual(concolic, "")

    def test_no_runtime_data(self):
        """Test case where tests exist but there is no runtime data."""
        with temp_project_dir() as project_root:
            tests_dir = project_root / "tests"
            tests_dir.mkdir(exist_ok=True)
            test_file_path = (tests_dir / "test_stuff.py").resolve()
            test_file_path.touch()

            test_cfg = MockTestConfig(tests_root=tests_dir.resolve())
            function_to_tests = {
                self.func_qual_name: {
                    MockFunctionCalledInTest(
                        tests_in_file=MockTestsInFile(test_file=test_file_path)
                    )
                }
            }
            existing, replay, concolic = existing_tests_source_for(
                function_qualified_name_with_modules_from_root=self.func_qual_name,
                function_to_tests=function_to_tests,
                test_cfg=test_cfg,
                original_runtimes_all={},
                optimized_runtimes_all={},
            )
            self.assertEqual(existing, "")
            self.assertEqual(replay, "")
            self.assertEqual(concolic, "")

    def test_with_existing_test_speedup(self):
        """Test with a single existing test that shows a speedup."""
        with temp_project_dir() as project_root:
            tests_dir = project_root / "tests"
            tests_dir.mkdir(exist_ok=True)
            test_file_path = (tests_dir / "test_existing.py").resolve()
            test_file_path.touch()

            test_cfg = MockTestConfig(tests_root=tests_dir.resolve())
            function_to_tests = {
                self.func_qual_name: {
                    MockFunctionCalledInTest(
                        tests_in_file=MockTestsInFile(test_file=test_file_path)
                    )
                }
            }

            invocation_id = MockInvocationId(
                test_module_path="tests.test_existing",
                test_class_name="TestMyStuff",
                test_function_name="test_one",
            )

            original_runtimes = {invocation_id: [200_000_000]}
            optimized_runtimes = {invocation_id: [100_000_000]}

            existing, replay, concolic = existing_tests_source_for(
                function_qualified_name_with_modules_from_root=self.func_qual_name,
                function_to_tests=function_to_tests,
                test_cfg=test_cfg,
                original_runtimes_all=original_runtimes,
                optimized_runtimes_all=optimized_runtimes,
            )

            self.assertIn("| Test File::Test Function", existing)
            self.assertIn("`test_existing.py::TestMyStuff.test_one`", existing)
            self.assertIn("200ms", existing)
            self.assertIn("100ms", existing)
            self.assertIn("100%✅", existing)
            self.assertEqual(replay, "")
            self.assertEqual(concolic, "")

    def test_with_replay_and_concolic_tests_slowdown(self):
        """Test with replay and concolic tests showing a slowdown."""
        with temp_project_dir() as project_root:
            tests_dir = project_root / "tests"
            tests_dir.mkdir(exist_ok=True)
            replay_test_path = (tests_dir / "__replay_test_abc.py").resolve()
            replay_test_path.touch()
            concolic_test_path = (tests_dir / "codeflash_concolic_xyz.py").resolve()
            concolic_test_path.touch()

            test_cfg = MockTestConfig(tests_root=tests_dir.resolve())
            function_to_tests = {
                self.func_qual_name: {
                    MockFunctionCalledInTest(
                        tests_in_file=MockTestsInFile(test_file=replay_test_path)
                    ),
                    MockFunctionCalledInTest(
                        tests_in_file=MockTestsInFile(test_file=concolic_test_path)
                    ),
                }
            }

            replay_inv_id = MockInvocationId(
                test_module_path="tests.__replay_test_abc",
                test_function_name="test_replay_one",
            )
            concolic_inv_id = MockInvocationId(
                test_module_path="tests.codeflash_concolic_xyz",
                test_function_name="test_concolic_one",
            )

            original_runtimes = {
                replay_inv_id: [100_000_000],
                concolic_inv_id: [150_000_000],
            }
            optimized_runtimes = {
                replay_inv_id: [200_000_000],
                concolic_inv_id: [300_000_000],
            }

            existing, replay, concolic = existing_tests_source_for(
                function_qualified_name_with_modules_from_root=self.func_qual_name,
                function_to_tests=function_to_tests,
                test_cfg=test_cfg,
                original_runtimes_all=original_runtimes,
                optimized_runtimes_all=optimized_runtimes,
            )

            self.assertEqual(existing, "")
            self.assertIn("`__replay_test_abc.py::test_replay_one`", replay)
            self.assertIn("-50.0%⚠️", replay)
            self.assertIn("`codeflash_concolic_xyz.py::test_concolic_one`", concolic)
            self.assertIn("-50.0%⚠️", concolic)

    def test_mixed_results_and_min_runtime(self):
        """Test with mixed results and that min() of runtimes is used."""
        with temp_project_dir() as project_root:
            tests_dir = project_root / "tests"
            tests_dir.mkdir(exist_ok=True)
            existing_test_path = (tests_dir / "test_existing.py").resolve()
            existing_test_path.touch()
            replay_test_path = (tests_dir / "__replay_test_mixed.py").resolve()
            replay_test_path.touch()

            test_cfg = MockTestConfig(tests_root=tests_dir.resolve())
            function_to_tests = {
                self.func_qual_name: {
                    MockFunctionCalledInTest(
                        tests_in_file=MockTestsInFile(test_file=existing_test_path)
                    ),
                    MockFunctionCalledInTest(
                        tests_in_file=MockTestsInFile(test_file=replay_test_path)
                    ),
                }
            }

            existing_inv_id = MockInvocationId(
                "tests.test_existing", "test_speedup", "TestExisting"
            )
            replay_inv_id = MockInvocationId(
                "tests.__replay_test_mixed", "test_slowdown"
            )

            original_runtimes = {
                existing_inv_id: [400_000_000, 500_000_000],  # min is 400ms
                replay_inv_id: [100_000_000, 110_000_000],  # min is 100ms
            }
            optimized_runtimes = {
                existing_inv_id: [210_000_000, 200_000_000],  # min is 200ms
                replay_inv_id: [300_000_000, 290_000_000],  # min is 290ms
            }

            existing, replay, concolic = existing_tests_source_for(
                self.func_qual_name,
                function_to_tests,
                test_cfg,
                original_runtimes,
                optimized_runtimes,
            )

            self.assertIn("`test_existing.py::TestExisting.test_speedup`", existing)
            self.assertIn("400ms", existing)
            self.assertIn("200ms", existing)
            self.assertIn("100%✅", existing)
            self.assertIn("`__replay_test_mixed.py::test_slowdown`", replay)
            self.assertIn("100ms", replay)
            self.assertIn("290ms", replay)
            self.assertIn("-65.5%⚠️", replay)
            self.assertEqual(concolic, "")
