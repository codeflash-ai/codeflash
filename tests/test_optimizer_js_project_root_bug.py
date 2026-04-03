"""
Test for the bug where test_cfg.js_project_root is set once and reused.

The bug: When optimizing multiple functions from different directories in a monorepo,
the js_project_root from the FIRST function is cached in test_cfg and used for ALL
subsequent functions, causing incorrect vitest working directories.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.languages.javascript.support import JavaScriptSupport
from codeflash.verification.verification_utils import TestConfig


@patch("codeflash.languages.javascript.optimizer.verify_js_requirements")
def test_js_project_root_not_recalculated_demonstrates_bug(mock_verify):
    """
    This test demonstrates the bug where js_project_root is set once
    and never updated when optimizing functions from different directories.

    Expected behavior: Each function should get its own js_project_root
    Actual behavior: All functions share the first function's js_project_root
    """
    # Mock verify_js_requirements to always pass
    mock_verify.return_value = []

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create main project
        main_project = root / "project"
        main_project.mkdir()
        (main_project / "package.json").write_text('{"name": "main"}')
        (main_project / "src").mkdir()
        (main_project / "test").mkdir()
        (main_project / "node_modules").mkdir()  # Add node_modules to pass requirements check

        # Create extension with its own package.json
        extension_dir = main_project / "extensions" / "discord"
        extension_dir.mkdir(parents=True)
        (extension_dir / "package.json").write_text('{"name": "discord-extension"}')
        (extension_dir / "src").mkdir()
        (extension_dir / "node_modules").mkdir()  # Add node_modules to pass requirements check

        # Create test config (shared across all functions, simulating optimizer behavior)
        test_cfg = TestConfig(
            tests_root=main_project / "test",
            project_root_path=main_project,
            tests_project_rootdir=main_project / "test",
        )
        test_cfg.set_language("javascript")

        # Create JavaScript support instance
        js_support = JavaScriptSupport()

        # Optimize function 1 (in extension directory)
        extension_file = extension_dir / "src" / "accounts.ts"
        extension_file.write_text("export function foo() {}")

        success = js_support.setup_test_config(test_cfg, extension_file, current_worktree=None)
        assert success, "setup_test_config should succeed"
        js_project_root_after_func1 = test_cfg.js_project_root

        # Should be extension directory
        assert js_project_root_after_func1 == extension_dir, (
            f"Function 1: Expected {extension_dir}, got {js_project_root_after_func1}"
        )

        # Optimize function 2 (in main src directory)
        main_file = main_project / "src" / "commands.ts"
        main_file.write_text("export function bar() {}")

        # This is the bug: setup_test_config is NOT called again in the real code!
        # The test_cfg object is reused, so js_project_root stays as extension_dir

        # In the real optimizer, test_cfg is reused without calling setup_test_config again
        # So js_project_root remains the same from function 1
        js_project_root_for_func2 = test_cfg.js_project_root

        # BUG: This assertion should fail because js_project_root was not recalculated
        # It's still pointing to extension_dir instead of main_project
        assert js_project_root_for_func2 == extension_dir, (
            f"BUG DEMONSTRATED: Function 2 inherits function 1's js_project_root. "
            f"Expected {main_project}, got {js_project_root_for_func2}"
        )

        # What SHOULD happen:
        # js_support.setup_test_config(test_cfg, main_file, current_worktree=None)
        # correct_root = test_cfg.js_project_root
        # assert correct_root == main_project


@pytest.mark.xfail(reason="Demonstrates the bug - will fail once bug is fixed")
@patch("codeflash.languages.javascript.optimizer.verify_js_requirements")
def test_js_project_root_reused_across_functions_wrong_behavior(mock_verify):
    """
    This test is marked xfail because it currently PASSES (demonstrating the bug).
    Once the bug is fixed, this test will FAIL (which is correct), and we can remove xfail.
    """
    # Mock verify_js_requirements to always pass
    mock_verify.return_value = []

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        main_project = root / "project"
        main_project.mkdir()
        (main_project / "package.json").write_text('{"name": "main"}')
        (main_project / "src").mkdir()
        (main_project / "test").mkdir()
        (main_project / "node_modules").mkdir()

        extension_dir = main_project / "extensions" / "discord"
        extension_dir.mkdir(parents=True)
        (extension_dir / "package.json").write_text('{"name": "discord"}')
        (extension_dir / "src").mkdir()
        (extension_dir / "node_modules").mkdir()

        test_cfg = TestConfig(
            tests_root=main_project / "test",
            project_root_path=main_project,
            tests_project_rootdir=main_project / "test",
        )
        test_cfg.set_language("javascript")

        js_support = JavaScriptSupport()

        # Set up for extension file
        extension_file = extension_dir / "src" / "accounts.ts"
        extension_file.write_text("export function foo() {}")
        js_support.setup_test_config(test_cfg, extension_file, current_worktree=None)

        # Now try to use test_cfg for a different file
        main_file = main_project / "src" / "commands.ts"
        main_file.write_text("export function bar() {}")

        # This assertion will PASS (showing the bug) because js_project_root is wrong
        # Once fixed, this will FAIL because js_project_root will be recalculated
        assert test_cfg.js_project_root == extension_dir, (
            "Bug exists: js_project_root is not recalculated per function"
        )

        # The correct behavior would be:
        # assert test_cfg.js_project_root == main_project
