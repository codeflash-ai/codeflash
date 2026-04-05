"""Test for Issue #9: Excessive logging in create_pr.py

Verifies that function_to_tests logging uses count instead of full key list.
"""

import logging
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codeflash.result.create_pr import existing_tests_source_for
from codeflash.verification.verification_utils import TestConfig


def test_function_to_tests_logging_uses_count_not_full_list():
    """
    Test that function_to_tests debug logging outputs count, not all keys.

    Bug: Line 38 of create_pr.py logs `list(function_to_tests.keys())` which
    creates massive log files (43MB+) when function_to_tests has thousands
    of entries (e.g., budibase monorepo with 1012 functions).

    Fix: Should log only `len(function_to_tests)` instead.
    """
    # Create a large function_to_tests dict (simulate budibase scale)
    function_to_tests = {
        f"package{i}.module{j}.function{k}": set()
        for i in range(10)
        for j in range(10)
        for k in range(10)
    }
    # Total: 1000 keys

    # Capture debug logs
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    # Get the 'rich' logger used by console.py
    logger = logging.getLogger('rich')
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    try:
        # Mock test_cfg
        test_cfg = Mock(spec=TestConfig)
        test_cfg.test_framework = "jest"

        # Call the function
        existing_tests_source_for(
            function_qualified_name_with_modules_from_root="test.function",
            function_to_tests=function_to_tests,
            test_cfg=test_cfg,
            original_runtimes_all={},
            optimized_runtimes_all={},
            test_files_registry=None,
        )

        # Get log output
        log_output = log_stream.getvalue()

        # ASSERTION 1: Should log the count
        assert "function_to_tests" in log_output, "Should mention function_to_tests in logs"
        assert "1000" in log_output or "len" in log_output, \
            "Should log count of function_to_tests, not full list"

        # ASSERTION 2: Should NOT log all keys (would create massive logs)
        # Check that we don't have dozens of "package0.module" strings
        package_mentions = log_output.count("package0.module")
        assert package_mentions < 10, \
            f"Should not log all {len(function_to_tests)} keys. " \
            f"Found {package_mentions} package mentions, which suggests full list logging. " \
            f"Log output size: {len(log_output)} bytes"

        # ASSERTION 3: Log output should be reasonable size (< 10KB for this debug line)
        # The buggy version would produce ~100KB+ for 1000 keys
        assert len(log_output) < 10000, \
            f"Log output too large ({len(log_output)} bytes). " \
            f"This suggests logging full key list instead of count."

    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def test_function_to_tests_logging_with_small_dict():
    """
    Test that logging still works correctly with small function_to_tests dict.

    This ensures the fix doesn't break the normal case.
    """
    # Small dict (< 10 entries)
    function_to_tests = {
        "module.function1": set(),
        "module.function2": set(),
    }

    # Capture debug logs
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    # Get the 'rich' logger used by console.py
    logger = logging.getLogger('rich')
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    try:
        # Mock test_cfg
        test_cfg = Mock(spec=TestConfig)
        test_cfg.test_framework = "jest"

        # Call the function
        existing_tests_source_for(
            function_qualified_name_with_modules_from_root="test.function",
            function_to_tests=function_to_tests,
            test_cfg=test_cfg,
            original_runtimes_all={},
            optimized_runtimes_all={},
            test_files_registry=None,
        )

        # Get log output
        log_output = log_stream.getvalue()

        # Should mention function_to_tests
        assert "function_to_tests" in log_output

        # Log should be reasonable size
        assert len(log_output) < 5000, \
            f"Even with small dict, log output is too large ({len(log_output)} bytes)"

    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
