"""Tests for handling Vitest coverage exclusions.

These tests verify that Codeflash correctly detects and handles files
that are excluded from coverage by vitest.config.ts, preventing false
0% coverage reports.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from codeflash.models.models import CodeOptimizationContext, CoverageStatus
from codeflash.verification.coverage_utils import JestCoverageUtils


class TestVitestCoverageExclusions:
    """Tests for Vitest coverage exclusion handling."""

    def test_missing_coverage_returns_not_found_status(self) -> None:
        """Should return NOT_FOUND status when file is not in coverage data.

        When a file is excluded from Vitest coverage (via coverage.exclude),
        it won't appear in coverage-final.json. Codeflash should return
        NOT_FOUND status (not PARSED_SUCCESSFULLY).

        This test verifies the current behavior is correct at the coverage
        parsing level. The issue is at a higher level (function_optimizer.py)
        where NOT_FOUND status needs better handling.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create mock coverage-final.json that's missing the target file
            coverage_file = tmp_path / "coverage-final.json"
            coverage_data = {
                "/workspace/project/src/utils/helpers.ts": {
                    "fnMap": {},
                    "s": {},
                },
                # src/agents/sandbox/fs-paths.ts is NOT here (excluded by Vitest)
            }
            with coverage_file.open("w") as f:
                json.dump(coverage_data, f)

            # Try to load coverage for a missing file
            missing_file = Path("/workspace/project/src/agents/sandbox/fs-paths.ts")
            from codeflash.models.models import CodeStringsMarkdown

            mock_context = CodeOptimizationContext(
                testgen_context=CodeStringsMarkdown(language="typescript"),
                read_writable_code=CodeStringsMarkdown(language="typescript"),
                helper_functions=[],
                preexisting_objects=set(),
            )

            result = JestCoverageUtils.load_from_jest_json(
                coverage_json_path=coverage_file,
                function_name="parseSandboxBindMount",
                code_context=mock_context,
                source_code_path=missing_file,
            )

            # Should return NOT_FOUND when file not in coverage
            assert result.status == CoverageStatus.NOT_FOUND, (
                f"Expected NOT_FOUND for missing file, got {result.status}"
            )
            assert result.coverage == 0.0

    def test_handles_included_file_normally(self) -> None:
        """Should handle files that ARE included in coverage normally.

        This test verifies that the fix doesn't break normal coverage parsing
        for files that are NOT excluded.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create mock coverage-final.json with a valid file
            coverage_file = tmp_path / "coverage-final.json"
            test_file = "/workspace/project/src/utils/helpers.ts"
            coverage_data = {
                test_file: {
                    "fnMap": {
                        "0": {"name": "someHelper", "loc": {"start": {"line": 1}, "end": {"line": 5}}}
                    },
                    "statementMap": {
                        "0": {"start": {"line": 2}, "end": {"line": 2}},
                        "1": {"start": {"line": 3}, "end": {"line": 3}},
                    },
                    "s": {"0": 5, "1": 5},  # Both statements executed
                    "branchMap": {},
                    "b": {},
                }
            }
            with coverage_file.open("w") as f:
                json.dump(coverage_data, f)

            source_file = Path(test_file)
            from codeflash.models.models import CodeStringsMarkdown

            mock_context = CodeOptimizationContext(
                testgen_context=CodeStringsMarkdown(language="typescript"),
                read_writable_code=CodeStringsMarkdown(language="typescript"),
                helper_functions=[],
                preexisting_objects=set(),
            )

            result = JestCoverageUtils.load_from_jest_json(
                coverage_json_path=coverage_file,
                function_name="someHelper",
                code_context=mock_context,
                source_code_path=source_file,
            )

            # Should parse successfully for non-excluded files
            assert result.status == CoverageStatus.PARSED_SUCCESSFULLY
            assert result.coverage > 0.0  # Should have actual coverage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
