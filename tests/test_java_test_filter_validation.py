"""Test that empty test filters are caught and raise errors."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.languages.java.maven_strategy import MavenStrategy
from codeflash.languages.java.test_runner import _build_test_filter
from codeflash.models.models import TestFile, TestFiles, TestType


def test_build_test_filter_with_none_benchmarking_paths():
    """Test that _build_test_filter handles None benchmarking paths correctly."""
    test_files = TestFiles(
        test_files=[
            TestFile(
                instrumented_behavior_file_path=Path("/tmp/test1__perfinstrumented.java"),
                benchmarking_file_path=None,
                original_file_path=Path("/tmp/test1.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            ),
            TestFile(
                instrumented_behavior_file_path=Path("/tmp/test2__perfinstrumented.java"),
                benchmarking_file_path=None,
                original_file_path=Path("/tmp/test2.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            ),
        ]
    )

    result = _build_test_filter(test_files, mode="performance")
    assert result == "", f"Expected empty filter but got: {result}"


def test_build_test_filter_with_valid_paths():
    """Test that _build_test_filter works correctly with valid paths."""
    test_files = TestFiles(
        test_files=[
            TestFile(
                instrumented_behavior_file_path=Path("/project/src/test/java/com/example/Test1__perfinstrumented.java"),
                benchmarking_file_path=Path("/project/src/test/java/com/example/Test1__perfonlyinstrumented.java"),
                original_file_path=Path("/project/src/test/java/com/example/Test1.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            )
        ]
    )

    result = _build_test_filter(test_files, mode="performance")
    assert result != "", "Expected non-empty filter"
    assert "Test1__perfonlyinstrumented" in result


def test_run_tests_via_build_tool_raises_on_empty_filter():
    """Test that MavenStrategy.run_tests_via_build_tool raises ValueError when filter is empty."""
    strategy = MavenStrategy()
    project_root = Path("/tmp/test_project")
    env = {}

    test_files = TestFiles(
        test_files=[
            TestFile(
                instrumented_behavior_file_path=Path("/tmp/test__perfinstrumented.java"),
                benchmarking_file_path=None,
                original_file_path=Path("/tmp/test.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            )
        ]
    )

    with patch.object(MavenStrategy, "find_executable", return_value="mvn"):
        with pytest.raises(ValueError, match="Test filter is EMPTY"):
            strategy.run_tests_via_build_tool(
                project_root,
                test_files,
                env,
                timeout=60,
                mode="performance",
                test_module=None,
            )


def test_run_tests_via_build_tool_succeeds_with_valid_filter():
    """Test that MavenStrategy.run_tests_via_build_tool works correctly when filter is not empty."""
    strategy = MavenStrategy()
    project_root = Path("/tmp/test_project")
    env = {}

    test_files = TestFiles(
        test_files=[
            TestFile(
                instrumented_behavior_file_path=Path("/tmp/src/test/java/com/example/Test__perfinstrumented.java"),
                benchmarking_file_path=Path("/tmp/src/test/java/com/example/Test__perfonlyinstrumented.java"),
                original_file_path=Path("/tmp/src/test/java/com/example/Test.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            )
        ]
    )

    with (
        patch.object(MavenStrategy, "find_executable", return_value="mvn"),
        patch("codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout") as mock_run,
    ):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Tests run: 1, Failures: 0, Errors: 0, Skipped: 0", stderr=""
        )

        result = strategy.run_tests_via_build_tool(
            project_root, test_files, env, timeout=60, mode="performance", test_module=None
        )

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert any("-Dtest=" in arg for arg in cmd), f"Expected -Dtest parameter in command: {cmd}"
