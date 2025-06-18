import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from codeflash.code_utils.edit_generated_tests import add_runtime_comments_to_generated_tests
from codeflash.models.models import GeneratedTests, GeneratedTestsList, InvocationId
from codeflash.verification.verification_utils import TestConfig


@pytest.fixture
def test_config():
    """Create a mock TestConfig for testing."""
    config = Mock(spec=TestConfig)
    config.project_root_path = Path("/project")
    config.tests_root = Path("/project/tests")
    return config


@pytest.fixture
def sample_invocation_id():
    """Create a sample InvocationId for testing."""
    return InvocationId(
        test_module_path="tests.test_module",
        test_class_name="TestClass",
        test_function_name="test_function",
    )


@pytest.fixture
def sample_invocation_id_no_class():
    """Create a sample InvocationId without class for testing."""
    return InvocationId(
        test_module_path="tests.test_module",
        test_class_name=None,
        test_function_name="test_function",
    )


class TestAddRuntimeCommentsToGeneratedTests:
    def test_add_runtime_comments_simple_function(self, test_config):
        """Test adding runtime comments to a simple test function."""
        test_source = '''def test_function():
    codeflash_output = some_function()
    assert codeflash_output == expected
'''

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("/project/tests/test_module.py"),
            perf_file_path=Path("/project/tests/test_module_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        invocation_id = InvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_function",
        )

        original_runtimes = {invocation_id: [1000000000, 1200000000]}  # 1s, 1.2s in nanoseconds
        optimized_runtimes = {invocation_id: [500000000, 600000000]}   # 0.5s, 0.6s in nanoseconds

        result = add_runtime_comments_to_generated_tests(
            test_config, generated_tests, original_runtimes, optimized_runtimes
        )

        expected_source = '''def test_function():
    codeflash_output = some_function() # 1.00s -> 500.00ms (50.00%)
    assert codeflash_output == expected
'''

        assert len(result.generated_tests) == 1
        assert result.generated_tests[0].generated_original_test_source == expected_source

    def test_add_runtime_comments_class_method(self, test_config):
        """Test adding runtime comments to a test method within a class."""
        test_source = '''class TestClass:
    def test_function(self):
        codeflash_output = some_function()
        assert codeflash_output == expected
'''

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("/project/tests/test_module.py"),
            perf_file_path=Path("/project/tests/test_module_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        invocation_id = InvocationId(
            test_module_path="tests.test_module",
            test_class_name="TestClass",
            test_function_name="test_function",
        )

        original_runtimes = {invocation_id: [2000000000]}  # 2s in nanoseconds
        optimized_runtimes = {invocation_id: [1000000000]} # 1s in nanoseconds

        result = add_runtime_comments_to_generated_tests(
            test_config, generated_tests, original_runtimes, optimized_runtimes
        )

        expected_source = '''class TestClass:
    def test_function(self):
        codeflash_output = some_function() # 2.00s -> 1.00s (50.00%)
        assert codeflash_output == expected
'''

        assert len(result.generated_tests) == 1
        assert result.generated_tests[0].generated_original_test_source == expected_source

    def test_add_runtime_comments_multiple_assignments(self, test_config):
        """Test adding runtime comments when there are multiple codeflash_output assignments."""
        test_source = '''def test_function():
    setup_data = prepare_test()
    codeflash_output = some_function()
    assert codeflash_output == expected
    codeflash_output = another_function()
    assert codeflash_output == expected2
'''

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("/project/tests/test_module.py"),
            perf_file_path=Path("/project/tests/test_module_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        invocation_id = InvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_function",
        )

        original_runtimes = {invocation_id: [1500000000]}  # 1.5s in nanoseconds
        optimized_runtimes = {invocation_id: [750000000]}  # 0.75s in nanoseconds

        result = add_runtime_comments_to_generated_tests(
            test_config, generated_tests, original_runtimes, optimized_runtimes
        )

        expected_source = '''def test_function():
    setup_data = prepare_test()
    codeflash_output = some_function() # 1.50s -> 750.00ms (50.00%)
    assert codeflash_output == expected
    codeflash_output = another_function() # 1.50s -> 750.00ms (50.00%)
    assert codeflash_output == expected2
'''

        assert len(result.generated_tests) == 1
        assert result.generated_tests[0].generated_original_test_source == expected_source

    def test_add_runtime_comments_no_matching_runtimes(self, test_config):
        """Test that source remains unchanged when no matching runtimes are found."""
        test_source = '''def test_function():
    codeflash_output = some_function()
    assert codeflash_output == expected
'''

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("/project/tests/test_module.py"),
            perf_file_path=Path("/project/tests/test_module_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        # Different invocation ID that won't match
        invocation_id = InvocationId(
            test_module_path="tests.other_module",
            test_class_name=None,
            test_function_name="other_function",
        )

        original_runtimes = {invocation_id: [1000000000]}
        optimized_runtimes = {invocation_id: [500000000]}

        result = add_runtime_comments_to_generated_tests(
            test_config, generated_tests, original_runtimes, optimized_runtimes
        )

        # Source should remain unchanged
        assert len(result.generated_tests) == 1
        assert result.generated_tests[0].generated_original_test_source == test_source

    def test_add_runtime_comments_no_codeflash_output(self, test_config):
        """Test that source remains unchanged when there's no codeflash_output assignment."""
        test_source = '''def test_function():
    result = some_function()
    assert result == expected
'''

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("/project/tests/test_module.py"),
            perf_file_path=Path("/project/tests/test_module_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        invocation_id = InvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_function",
        )

        original_runtimes = {invocation_id: [1000000000]}
        optimized_runtimes = {invocation_id: [500000000]}

        result = add_runtime_comments_to_generated_tests(
            test_config, generated_tests, original_runtimes, optimized_runtimes
        )

        # Source should remain unchanged
        assert len(result.generated_tests) == 1
        assert result.generated_tests[0].generated_original_test_source == test_source

    def test_add_runtime_comments_multiple_tests(self, test_config):
        """Test adding runtime comments to multiple generated tests."""
        test_source1 = '''def test_function1():
    codeflash_output = some_function()
    assert codeflash_output == expected
'''

        test_source2 = '''def test_function2():
    codeflash_output = another_function()
    assert codeflash_output == expected
'''

        generated_test1 = GeneratedTests(
            generated_original_test_source=test_source1,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("/project/tests/test_module1.py"),
            perf_file_path=Path("/project/tests/test_module1_perf.py"),
        )

        generated_test2 = GeneratedTests(
            generated_original_test_source=test_source2,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("/project/tests/test_module2.py"),
            perf_file_path=Path("/project/tests/test_module2_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test1, generated_test2])

        invocation_id1 = InvocationId(
            test_module_path="tests.test_module1",
            test_class_name=None,
            test_function_name="test_function1",
        )

        invocation_id2 = InvocationId(
            test_module_path="tests.test_module2",
            test_class_name=None,
            test_function_name="test_function2",
        )

        original_runtimes = {
            invocation_id1: [1000000000],  # 1s
            invocation_id2: [2000000000],  # 2s
        }
        optimized_runtimes = {
            invocation_id1: [500000000],   # 0.5s
            invocation_id2: [800000000],   # 0.8s
        }

        result = add_runtime_comments_to_generated_tests(
            test_config, generated_tests, original_runtimes, optimized_runtimes
        )

        expected_source1 = '''def test_function1():
    codeflash_output = some_function() # 1.00s -> 500.00ms (50.00%)
    assert codeflash_output == expected
'''

        expected_source2 = '''def test_function2():
    codeflash_output = another_function() # 2.00s -> 800.00ms (60.00%)
    assert codeflash_output == expected
'''

        assert len(result.generated_tests) == 2
        assert result.generated_tests[0].generated_original_test_source == expected_source1
        assert result.generated_tests[1].generated_original_test_source == expected_source2

    def test_add_runtime_comments_performance_regression(self, test_config):
        """Test adding runtime comments when optimized version is slower (negative performance gain)."""
        test_source = '''def test_function():
    codeflash_output = some_function()
    assert codeflash_output == expected
'''

        generated_test = GeneratedTests(
            generated_original_test_source=test_source,
            instrumented_behavior_test_source="",
            instrumented_perf_test_source="",
            behavior_file_path=Path("/project/tests/test_module.py"),
            perf_file_path=Path("/project/tests/test_module_perf.py"),
        )

        generated_tests = GeneratedTestsList(generated_tests=[generated_test])

        invocation_id = InvocationId(
            test_module_path="tests.test_module",
            test_class_name=None,
            test_function_name="test_function",
        )

        original_runtimes = {invocation_id: [1000000000]}  # 1s
        optimized_runtimes = {invocation_id: [1500000000]} # 1.5s (slower!)

        result = add_runtime_comments_to_generated_tests(
            test_config, generated_tests, original_runtimes, optimized_runtimes
        )

        expected_source = '''def test_function():
    codeflash_output = some_function() # 1.00s -> 1.50s (-50.00%)
    assert codeflash_output == expected
'''

        assert len(result.generated_tests) == 1
        assert result.generated_tests[0].generated_original_test_source == expected_source
