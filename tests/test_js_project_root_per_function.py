"""Test that js_project_root is recalculated per function, not cached."""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.javascript.test_runner import find_node_project_root


def test_find_node_project_root_returns_different_roots_for_different_files():
    """Test that find_node_project_root returns the correct root for each file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create main project structure
        main_project = root / "project"
        main_project.mkdir()
        (main_project / "package.json").write_text("{}")
        (main_project / "src").mkdir()
        main_file = main_project / "src" / "main.ts"
        main_file.write_text("// main file")

        # Create extension subdirectory with its own package.json
        extension_dir = main_project / "extensions" / "discord"
        extension_dir.mkdir(parents=True)
        (extension_dir / "package.json").write_text("{}")
        (extension_dir / "src").mkdir()
        extension_file = extension_dir / "src" / "accounts.ts"
        extension_file.write_text("// extension file")

        # Test 1: Extension file should return extension directory
        result1 = find_node_project_root(extension_file)
        assert result1 == extension_dir, (
            f"Expected {extension_dir}, got {result1}"
        )

        # Test 2: Main file should return main project directory
        result2 = find_node_project_root(main_file)
        assert result2 == main_project, (
            f"Expected {main_project}, got {result2}"
        )

        # Test 3: Calling again with extension file should still return extension dir
        result3 = find_node_project_root(extension_file)
        assert result3 == extension_dir, (
            f"Expected {extension_dir}, got {result3}"
        )


def test_js_project_root_should_be_recalculated_per_function():
    """
    Test the actual bug: when optimizing multiple functions from different
    directories, each should get its own js_project_root, not inherit from
    the first function.

    This test simulates the scenario where:
    1. Function #1 is in extensions/discord/src/accounts.ts
    2. Function #2 is in src/plugins/commands.ts
    3. Both should get their correct respective project roots
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create main project
        main_project = root / "project"
        main_project.mkdir()
        (main_project / "package.json").write_text('{"name": "main"}')
        (main_project / "src").mkdir()
        (main_project / "test").mkdir()

        # Create extension with its own package.json
        extension_dir = main_project / "extensions" / "discord"
        extension_dir.mkdir(parents=True)
        (extension_dir / "package.json").write_text('{"name": "discord-extension"}')
        (extension_dir / "src").mkdir()

        # Files to optimize
        extension_file = extension_dir / "src" / "accounts.ts"
        extension_file.write_text("export function foo() {}")

        main_file = main_project / "src" / "commands.ts"
        main_file.write_text("export function bar() {}")

        # Simulate what happens in Codeflash optimizer
        # Function 1 (extension file) sets js_project_root
        js_project_root_1 = find_node_project_root(extension_file)
        assert js_project_root_1 == extension_dir

        # Function 2 (main file) should get its own root, not inherit from function 1
        js_project_root_2 = find_node_project_root(main_file)
        assert js_project_root_2 == main_project, (
            f"Bug reproduced: main file got {js_project_root_2} instead of {main_project}. "
            f"This happens when test_cfg.js_project_root is not recalculated per function."
        )
