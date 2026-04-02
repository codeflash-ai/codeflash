"""Test that Vitest coverage thresholds are disabled for Codeflash tests.

When running Codeflash-generated tests with coverage enabled, we must disable
the project's global coverage thresholds to prevent false failures. Generated
tests typically cover only a single function (~1-2% of the codebase), which
would fail projects with thresholds like 70% lines/functions.

Related issue: Trace IDs 05a626f3, 932e7799, a145328d, aa9bb63f, d669202e, e6de097a
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.languages.javascript.vitest_runner import run_vitest_behavioral_tests
from codeflash.models.models import TestFile, TestFiles
from codeflash.models.test_type import TestType


class TestVitestCoverageThresholds:
    """Tests for disabling coverage thresholds in Vitest commands."""

    def test_coverage_thresholds_disabled_when_coverage_enabled(self) -> None:
        """Should add coverage threshold flags to disable project thresholds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create mock test file
            test_file_path = tmp_path / "test.test.ts"
            test_file_path.write_text("test('mock', () => {})")

            # Create mock project structure
            (tmp_path / "package.json").write_text('{"name": "test-project"}')
            (tmp_path / "node_modules" / "codeflash").mkdir(parents=True)
            (tmp_path / "node_modules" / "@vitest" / "coverage-v8").mkdir(parents=True)

            # Create TestFiles object
            test_files = TestFiles(
                test_files=[
                    TestFile(
                        instrumented_behavior_file_path=test_file_path,
                        benchmarking_file_path=None,
                        original_file_path=None,
                        test_type=TestType.GENERATED_REGRESSION,
                    )
                ]
            )

            # Mock subprocess.run to capture the command
            captured_cmd = []

            def mock_run(cmd, **kwargs):
                captured_cmd.extend(cmd)
                # Create a minimal JUnit XML file
                result_file = None
                for i, arg in enumerate(cmd):
                    if arg.startswith("--outputFile.junit="):
                        result_file = Path(arg.split("=", 1)[1])
                        break

                if result_file:
                    result_file.parent.mkdir(parents=True, exist_ok=True)
                    result_file.write_text('<?xml version="1.0"?><testsuites><testsuite tests="1" failures="0"></testsuite></testsuites>')

                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout="✓ test.test.ts (1 test)",
                    stderr="",
                )

            with patch("codeflash.languages.javascript.vitest_runner.subprocess.run", side_effect=mock_run):
                with patch("codeflash.languages.javascript.vitest_runner.get_run_tmp_file") as mock_tmp:
                    # Mock temp file location
                    mock_tmp.side_effect = lambda p: tmp_path / p

                    # Run with coverage enabled
                    run_vitest_behavioral_tests(
                        test_paths=test_files,
                        test_env={},
                        cwd=tmp_path,
                        timeout=60,
                        project_root=tmp_path,
                        enable_coverage=True,
                        candidate_index=0,
                    )

            # Verify that coverage threshold flags were added
            cmd_str = " ".join(captured_cmd)

            # The fix should add these flags to disable thresholds
            assert "--coverage.thresholds.lines=0" in captured_cmd, (
                f"Missing --coverage.thresholds.lines=0 in command: {cmd_str}"
            )
            assert "--coverage.thresholds.functions=0" in captured_cmd, (
                f"Missing --coverage.thresholds.functions=0 in command: {cmd_str}"
            )
            assert "--coverage.thresholds.statements=0" in captured_cmd, (
                f"Missing --coverage.thresholds.statements=0 in command: {cmd_str}"
            )
            assert "--coverage.thresholds.branches=0" in captured_cmd, (
                f"Missing --coverage.thresholds.branches=0 in command: {cmd_str}"
            )

    def test_coverage_command_structure_with_thresholds_disabled(self) -> None:
        """Verify the full command structure when coverage is enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create mock setup
            test_file_path = tmp_path / "test.test.ts"
            test_file_path.write_text("test('mock', () => {})")
            (tmp_path / "package.json").write_text('{"name": "test-project"}')
            (tmp_path / "node_modules" / "codeflash").mkdir(parents=True)
            (tmp_path / "node_modules" / "@vitest" / "coverage-v8").mkdir(parents=True)

            test_files = TestFiles(
                test_files=[
                    TestFile(
                        instrumented_behavior_file_path=test_file_path,
                        benchmarking_file_path=None,
                        original_file_path=None,
                        test_type=TestType.GENERATED_REGRESSION,
                    )
                ]
            )

            captured_cmd = []

            def mock_run(cmd, **kwargs):
                captured_cmd.extend(cmd)
                result_file = None
                for arg in cmd:
                    if arg.startswith("--outputFile.junit="):
                        result_file = Path(arg.split("=", 1)[1])
                        break

                if result_file:
                    result_file.parent.mkdir(parents=True, exist_ok=True)
                    result_file.write_text('<?xml version="1.0"?><testsuites><testsuite tests="1" failures="0"></testsuite></testsuites>')

                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="✓ test", stderr=""
                )

            with patch("codeflash.languages.javascript.vitest_runner.subprocess.run", side_effect=mock_run):
                with patch("codeflash.languages.javascript.vitest_runner.get_run_tmp_file") as mock_tmp:
                    mock_tmp.side_effect = lambda p: tmp_path / p

                    run_vitest_behavioral_tests(
                        test_paths=test_files,
                        test_env={},
                        cwd=tmp_path,
                        timeout=60,
                        project_root=tmp_path,
                        enable_coverage=True,
                        candidate_index=0,
                    )

            # Verify command contains both coverage flags AND threshold overrides
            assert "--coverage" in captured_cmd
            assert "--coverage.reporter=json" in captured_cmd
            assert any("--coverage.reportsDirectory=" in arg for arg in captured_cmd)

            # And the threshold overrides
            assert "--coverage.thresholds.lines=0" in captured_cmd
            assert "--coverage.thresholds.functions=0" in captured_cmd
            assert "--coverage.thresholds.statements=0" in captured_cmd
            assert "--coverage.thresholds.branches=0" in captured_cmd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
