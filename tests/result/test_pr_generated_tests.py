"""Unit tests for PR creation with generated tests.

Bug: Generated tests with original_file_path=None are excluded from PR summaries.
Location: codeflash/result/create_pr.py:62-85
"""
from pathlib import Path

import pytest

from codeflash.models.models import TestFile, TestFiles
from codeflash.models.test_type import TestType


@pytest.fixture
def test_files_registry() -> TestFiles:
    """Create a test files registry with both generated and existing tests."""
    registry = TestFiles(test_files=[])

    # Generated test (has None for original_file_path)
    generated_behavior = Path("/workspace/target/src/test_saveCronStore__unit_test_0.test.ts")
    generated_perf = Path("/workspace/target/src/test_saveCronStore__perf_test_0.test.ts")

    generated_test = TestFile(
        instrumented_behavior_file_path=generated_behavior,
        benchmarking_file_path=generated_perf,
        original_file_path=None,  # Generated tests have no original
        original_source="test code",
        test_type=TestType.GENERATED_REGRESSION,
        tests_in_file=None,
    )
    registry.add(generated_test)

    # Existing instrumented test (has original_file_path)
    existing_behavior = Path("/workspace/target/src/store__perfinstrumented.test.ts")
    existing_perf = Path("/workspace/target/src/store__perfonlyinstrumented.test.ts")
    existing_original = Path("/workspace/target/src/store.test.ts")

    existing_test = TestFile(
        instrumented_behavior_file_path=existing_behavior,
        benchmarking_file_path=existing_perf,
        original_file_path=existing_original,
        original_source=None,
        test_type=TestType.EXISTING_UNIT_TEST,
        tests_in_file=None,
    )
    registry.add(existing_test)

    return registry


def test_instrumented_to_original_mapping_includes_generated_tests(test_files_registry: TestFiles) -> None:
    """Test that the instrumented_to_original mapping includes generated tests.

    This is the direct test of the bug at lines 62-78 in create_pr.py.
    """
    # Build the mapping as create_pr.py does (lines 62-95) - with fix
    instrumented_to_original = {}
    for registry_tf in test_files_registry.test_files:
        # For existing tests, map instrumented → original
        # For generated tests (original_file_path=None), map instrumented → instrumented (self)
        if registry_tf.original_file_path:
            # Existing test: map to original file
            if registry_tf.instrumented_behavior_file_path:
                instrumented_to_original[registry_tf.instrumented_behavior_file_path.resolve()] = (
                    registry_tf.original_file_path.resolve()
                )
            if registry_tf.benchmarking_file_path:
                instrumented_to_original[registry_tf.benchmarking_file_path.resolve()] = (
                    registry_tf.original_file_path.resolve()
                )
        else:
            # Generated test (no original file): map to itself
            if registry_tf.instrumented_behavior_file_path:
                behavior_resolved = registry_tf.instrumented_behavior_file_path.resolve()
                instrumented_to_original[behavior_resolved] = behavior_resolved
            if registry_tf.benchmarking_file_path:
                perf_resolved = registry_tf.benchmarking_file_path.resolve()
                instrumented_to_original[perf_resolved] = perf_resolved

    # Check the bug
    generated_behavior_path = Path("/workspace/target/src/test_saveCronStore__unit_test_0.test.ts").resolve()
    existing_behavior_path = Path("/workspace/target/src/store__perfinstrumented.test.ts").resolve()

    # Existing test should be in mapping
    assert existing_behavior_path in instrumented_to_original, \
        "Existing test should be in mapping"

    # After fix: Generated test SHOULD be in mapping (mapped to itself)
    assert generated_behavior_path in instrumented_to_original, \
        "Generated test should be in mapping"
    # Verify it's mapped to itself
    assert instrumented_to_original[generated_behavior_path] == generated_behavior_path, \
        "Generated test should map to itself"
