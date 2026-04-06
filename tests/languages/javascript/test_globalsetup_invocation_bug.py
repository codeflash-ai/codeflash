"""
Test for Issue #18: globalSetup not disabled when tests are inside project root.

When test files are inside the project root (common case), the runtime config that
disables globalSetup/globalTeardown is never created. This causes Jest to use the
project's original config, which may have globalSetup hooks that require
infrastructure (Docker, databases) that isn't available during Codeflash runs.

Example failure:
    Error: Jest: Got error running globalSetup - /workspace/target/globalSetup.ts,
    reason: Command failed: docker context ls --format json
    /bin/sh: 1: docker: not found

Root cause (before fix):
    In test_runner.py, _create_runtime_jest_config was only called when:
        if any(not Path(d).is_relative_to(resolved_root) for d in test_dirs):
            jest_config = _create_runtime_jest_config(...)

    But globalSetup should be disabled for ALL Codeflash test runs, not just when
    tests are outside the project root.

Fix:
    Always call _create_runtime_jest_config when jest_config and test_files exist,
    regardless of whether tests are inside or outside the project root.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from codeflash.languages.javascript.test_runner import _create_runtime_jest_config
from codeflash.models.models import TestFile, TestFiles
from codeflash.models.test_type import TestType


def test_runtime_config_always_created_when_jest_config_exists():
    """
    Test that _create_runtime_jest_config is called even when tests are inside project root.

    This is the KEY fix for Issue #18: we must ALWAYS create the runtime config
    to ensure globalSetup is disabled, not just when tests are outside project root.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "project"
        project_root.mkdir()

        # Create a Jest config
        jest_config = project_root / "jest.config.js"
        jest_config.write_text("module.exports = { globalSetup: './setup.ts' };")

        # Create test file INSIDE project root (common case)
        test_dir = project_root / "src" / "tests"
        test_dir.mkdir(parents=True)
        test_file = test_dir / "test_example.test.ts"
        test_file.write_text("test('example', () => expect(true).toBe(true));")

        # Create package.json
        (project_root / "package.json").write_text('{"name": "test"}')

        # Create node_modules/codeflash
        (project_root / "node_modules" / "codeflash").mkdir(parents=True)

        # Create TestFiles object
        test_file_obj = TestFile(
            instrumented_behavior_file_path=test_file,
            benchmarking_file_path=test_file,
            test_type=TestType.GENERATED_REGRESSION,
        )
        test_paths = TestFiles(test_files=[test_file_obj])

        # Mock _create_runtime_jest_config to track if it's called
        with patch('codeflash.languages.javascript.test_runner._create_runtime_jest_config', wraps=_create_runtime_jest_config) as mock_create_runtime:
            with patch('codeflash.languages.javascript.test_runner.subprocess.run') as mock_run:
                # Mock Jest execution
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                with patch('codeflash.languages.javascript.test_runner._get_jest_config_for_project', return_value=jest_config):
                    from codeflash.languages.javascript.test_runner import run_jest_behavioral_tests
                    try:
                        run_jest_behavioral_tests(
                            test_paths=test_paths,
                            test_env={},
                            cwd=project_root,
                            project_root=project_root,
                            enable_coverage=False,
                            timeout=60,
                        )
                    except Exception:
                        pass  # May fail due to mocking, that's OK

        # THE KEY ASSERTION: _create_runtime_jest_config MUST be called
        # even when tests are inside the project root
        assert mock_create_runtime.call_count > 0, (
            "VULNERABILITY: _create_runtime_jest_config was not called when tests are inside project root. "
            f"This means globalSetup is NOT disabled, causing failures on projects with Docker/DB setup hooks. "
            f"Test file: {test_file}, Project root: {project_root}"
        )

        # Verify it was called with correct arguments
        call_args = mock_create_runtime.call_args
        assert call_args is not None
        assert call_args[0][0] == jest_config  # base_config_path
        assert call_args[0][1] == project_root  # project_root
        # test_dirs should include the test directory
        assert str(test_dir) in call_args[0][2]


def test_runtime_config_disables_globalsetup_for_tests_inside_project():
    """
    Test the actual runtime config file created for tests inside project root.

    Verifies that the config file disables globalSetup/globalTeardown.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "project"
        project_root.mkdir()

        # Create base config with globalSetup
        jest_config = project_root / "jest.config.js"
        jest_config.write_text("""
module.exports = {
  testEnvironment: 'node',
  globalSetup: './globalSetup.ts',
  globalTeardown: './globalTeardown.ts',
};
""")

        # Test directory INSIDE project root
        test_dir = project_root / "src" / "tests" / "codeflash-generated"
        test_dir.mkdir(parents=True)

        # Create runtime config
        test_dirs = {str(test_dir)}
        runtime_config = _create_runtime_jest_config(
            base_config_path=jest_config,
            project_root=project_root,
            test_dirs=test_dirs
        )

        # Verify runtime config was created
        assert runtime_config is not None, (
            "VULNERABILITY: Runtime config not created for tests inside project root"
        )
        assert runtime_config.exists(), (
            f"VULNERABILITY: Runtime config file doesn't exist: {runtime_config}"
        )

        # Verify it disables globalSetup and globalTeardown
        config_content = runtime_config.read_text()
        assert "globalSetup: undefined" in config_content, (
            f"VULNERABILITY: globalSetup not disabled in runtime config.\nContent:\n{config_content}"
        )
        assert "globalTeardown: undefined" in config_content, (
            f"VULNERABILITY: globalTeardown not disabled in runtime config.\nContent:\n{config_content}"
        )


def test_runtime_config_created_for_tests_in_subdirectories():
    """
    Test that runtime config is created even when tests are in subdirectories of project root.

    This is the most common case: tests in packages/server/src/tests/, project root at packages/server/.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "packages" / "server"
        project_root.mkdir(parents=True)

        jest_config = project_root / "jest.config.ts"
        jest_config.write_text("""
export default {
  testEnvironment: 'node',
  globalSetup: './setup.ts',
};
""")

        # Test file in deeply nested subdirectory (still inside project root)
        test_dir = project_root / "src" / "automations" / "tests" / "codeflash-generated"
        test_dir.mkdir(parents=True)
        test_file = test_dir / "test_example.test.ts"
        test_file.write_text("test('example', () => expect(true).toBe(true));")

        # Create the runtime config directly (unit test, not full integration)
        test_dirs = {str(test_dir)}
        runtime_config = _create_runtime_jest_config(
            base_config_path=jest_config,
            project_root=project_root,
            test_dirs=test_dirs
        )

        # Verify runtime config exists and disables globalSetup
        assert runtime_config is not None
        assert runtime_config.exists()

        config_content = runtime_config.read_text()
        assert "globalSetup: undefined" in config_content
        assert "globalTeardown: undefined" in config_content
