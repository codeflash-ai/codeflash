"""Test that coverage error messages are framework-agnostic."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeflash.languages.language_enum import Language
from codeflash.models.models import CodeOptimizationContext
from codeflash.verification.coverage_utils import JestCoverageUtils


class TestCoverageUtilsFrameworkAgnostic:
    """Test that error messages don't hardcode 'Jest' when used for Vitest."""

    def test_missing_coverage_file_message_is_framework_agnostic(self, caplog):
        """When coverage file is missing, error message should not say 'Jest' specifically.

        This class is used for both Jest and Vitest (they use the same Istanbul/v8 format).
        Error messages should be generic, not hardcode 'Jest'.
        """
        # Set log level to DEBUG to capture all messages
        caplog.set_level("DEBUG")

        # Create minimal context
        context = MagicMock(spec=CodeOptimizationContext)
        context.language = Language.JAVASCRIPT
        context.target_code = "export function test() {}"
        context.helper_functions = []

        nonexistent_path = Path("/tmp/nonexistent_coverage_12345.json")

        # Load coverage from non-existent file
        result = JestCoverageUtils.load_from_jest_json(
            coverage_json_path=nonexistent_path,
            function_name="testFunc",
            code_context=context,
            source_code_path=Path("/tmp/test.ts")
        )

        # Should return empty coverage data
        assert result.status.name in ("NOT_FOUND", "EMPTY")

        # Error message should NOT hardcode "Jest" - it should be framework-agnostic
        # since this util is used for both Jest and Vitest
        log_messages = [record.message for record in caplog.records]

        # Check that if there's a message about coverage file, it doesn't say "Jest"
        coverage_messages = [msg for msg in log_messages if "coverage file not found" in msg.lower()]
        if coverage_messages:
            # The message should NOT contain "Jest" specifically
            # It should say something like "Coverage file not found" or "JavaScript coverage file not found"
            for msg in coverage_messages:
                assert "Jest" not in msg, (
                    f"Error message should not hardcode 'Jest' since this util is used for Vitest too. "
                    f"Got: {msg}"
                )

    def test_parse_error_message_is_framework_agnostic(self, tmp_path, caplog):
        """When coverage file is malformed, error should not say 'Jest' specifically."""
        # Set log level to capture all messages
        caplog.set_level("DEBUG")

        # Create invalid JSON file
        coverage_file = tmp_path / "invalid_coverage.json"
        coverage_file.write_text("{invalid json")

        context = MagicMock(spec=CodeOptimizationContext)
        context.language = Language.JAVASCRIPT
        context.target_code = "export function test() {}"
        context.helper_functions = []

        result = JestCoverageUtils.load_from_jest_json(
            coverage_json_path=coverage_file,
            function_name="testFunc",
            code_context=context,
            source_code_path=Path("/tmp/test.ts")
        )

        # Should return empty coverage
        assert result.status.name in ("NOT_FOUND", "EMPTY")

        # Check log messages don't hardcode "Jest"
        log_messages = [record.message for record in caplog.records]
        parse_error_messages = [msg for msg in log_messages if "parse" in msg.lower() and "coverage" in msg.lower()]

        for msg in parse_error_messages:
            assert "Jest" not in msg, (
                f"Parse error message should not hardcode 'Jest'. Got: {msg}"
            )
