"""Test that empty test filters are caught and raise errors."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from codeflash.languages.java.test_runner import _run_maven_tests, _build_test_filter
from codeflash.models.models import TestFile, TestFiles, TestType


def test_build_test_filter_with_none_benchmarking_paths():
    """Test that _build_test_filter handles None benchmarking paths correctly."""
    # Create TestFiles with None benchmarking_file_path
    test_files = TestFiles(
        test_files=[
            TestFile(
                instrumented_behavior_file_path=Path("/tmp/test1__perfinstrumented.java"),
                benchmarking_file_path=None,  # None path!
                original_file_path=Path("/tmp/test1.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            ),
            TestFile(
                instrumented_behavior_file_path=Path("/tmp/test2__perfinstrumented.java"),
                benchmarking_file_path=None,  # None path!
                original_file_path=Path("/tmp/test2.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            ),
        ]
    )

    # In performance mode with None paths, filter should be empty
    result = _build_test_filter(test_files, mode="performance")
    assert result == "", f"Expected empty filter but got: {result}"


def test_build_test_filter_with_valid_paths():
    """Test that _build_test_filter works correctly with valid paths."""
    # Create TestFiles with valid paths
    test_files = TestFiles(
        test_files=[
            TestFile(
                instrumented_behavior_file_path=Path(
                    "/project/src/test/java/com/example/Test1__perfinstrumented.java"
                ),
                benchmarking_file_path=Path(
                    "/project/src/test/java/com/example/Test1__perfonlyinstrumented.java"
                ),
                original_file_path=Path("/project/src/test/java/com/example/Test1.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            ),
        ]
    )

    # Should produce valid filter
    result = _build_test_filter(test_files, mode="performance")
    assert result != "", "Expected non-empty filter"
    assert "Test1__perfonlyinstrumented" in result


def test_run_maven_tests_raises_on_empty_filter():
    """Test that _run_maven_tests raises ValueError when filter is empty."""
    project_root = Path("/tmp/test_project")
    env = {}

    # Create TestFiles with None paths (will produce empty filter)
    test_files = TestFiles(
        test_files=[
            TestFile(
                instrumented_behavior_file_path=Path("/tmp/test__perfinstrumented.java"),
                benchmarking_file_path=None,  # Will cause empty filter in performance mode
                original_file_path=Path("/tmp/test.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            ),
        ]
    )

    # Mock Maven executable
    with patch("codeflash.languages.java.test_runner.find_maven_executable") as mock_maven:
        mock_maven.return_value = "mvn"

        # Should raise ValueError due to empty filter
        with pytest.raises(ValueError, match="Test filter is EMPTY"):
            _run_maven_tests(
                project_root,
                test_files,
                env,
                timeout=60,
                mode="performance",  # Performance mode with None benchmarking_file_path
            )


def test_run_maven_tests_succeeds_with_valid_filter():
    """Test that _run_maven_tests works correctly when filter is not empty."""
    project_root = Path("/tmp/test_project")
    env = {}

    # Create TestFiles with valid paths
    test_files = TestFiles(
        test_files=[
            TestFile(
                instrumented_behavior_file_path=Path(
                    "/tmp/src/test/java/com/example/Test__perfinstrumented.java"
                ),
                benchmarking_file_path=Path(
                    "/tmp/src/test/java/com/example/Test__perfonlyinstrumented.java"
                ),
                original_file_path=Path("/tmp/src/test/java/com/example/Test.java"),
                test_type=TestType.EXISTING_UNIT_TEST,
            ),
        ]
    )

    # Mock Maven executable and subprocess.run
    with patch("codeflash.languages.java.test_runner.find_maven_executable") as mock_maven, \
         patch("codeflash.languages.java.test_runner.subprocess.run") as mock_run:
        mock_maven.return_value = "mvn"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Tests run: 1, Failures: 0, Errors: 0, Skipped: 0",
            stderr="",
        )

        # Should not raise - filter is valid
        result = _run_maven_tests(
            project_root,
            test_files,
            env,
            timeout=60,
            mode="performance",
        )

        # Verify Maven was called with -Dtest parameter
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert any("-Dtest=" in arg for arg in cmd), f"Expected -Dtest parameter in command: {cmd}"
