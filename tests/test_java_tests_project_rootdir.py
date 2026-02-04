"""Test that tests_project_rootdir is set correctly for Java projects."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.languages.base import Language
from codeflash.languages.current import set_current_language
from codeflash.verification.verification_utils import TestConfig


def test_java_tests_project_rootdir_set_to_tests_root(tmp_path):
    """Test that for Java projects, tests_project_rootdir is set to tests_root."""
    # Create a mock Java project structure
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pom.xml").touch()

    tests_root = project_root / "src" / "test" / "java"
    tests_root.mkdir(parents=True)

    # Create test config with tests_project_rootdir initially set to project root
    # (simulating what happens before the fix)
    test_cfg = TestConfig(
        tests_root=tests_root,
        project_root_path=project_root,
        tests_project_rootdir=project_root,  # Initially set to project root
    )

    # Create a mock Java function to ensure language detection works
    mock_java_function = MagicMock()
    mock_java_function.language = "java"
    file_to_funcs = {Path("dummy.java"): [mock_java_function]}

    # Mock is_python() to return False and is_java() to return True
    # These are imported from codeflash.languages
    with patch("codeflash.languages.is_python", return_value=False), \
         patch("codeflash.languages.is_java", return_value=True), \
         patch("codeflash.discovery.discover_unit_tests.discover_tests_for_language") as mock_discover:
        mock_discover.return_value = ({}, 0, 0)

        # Call discover_unit_tests
        discover_unit_tests(test_cfg, file_to_funcs_to_optimize=file_to_funcs)

    # Verify that tests_project_rootdir was updated to tests_root
    assert test_cfg.tests_project_rootdir == tests_root, (
        f"Expected tests_project_rootdir to be {tests_root}, "
        f"but got {test_cfg.tests_project_rootdir}"
    )


def test_python_tests_project_rootdir_unchanged(tmp_path):
    """Test that for Python projects, tests_project_rootdir behavior is unchanged."""
    # Setup Python environment
    set_current_language(Language.PYTHON)

    # Create a mock Python project structure
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").touch()

    tests_root = project_root / "tests"
    tests_root.mkdir()

    # Create test config
    original_tests_project_rootdir = project_root / "some" / "other" / "dir"
    test_cfg = TestConfig(
        tests_root=tests_root,
        project_root_path=project_root,
        tests_project_rootdir=original_tests_project_rootdir,
    )

    # Mock pytest discovery
    with patch("codeflash.discovery.discover_unit_tests.discover_tests_pytest") as mock_discover:
        mock_discover.return_value = ({}, 0, 0)

        # Call discover_unit_tests
        discover_unit_tests(test_cfg, file_to_funcs_to_optimize={})

    # For Python, tests_project_rootdir should remain unchanged
    # (the function doesn't modify it for Python projects)
    assert test_cfg.tests_project_rootdir == original_tests_project_rootdir
