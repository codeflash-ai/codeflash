"""Test that runtime config generation handles TypeScript Jest configs correctly.

Reproduces bug where requiring a jest.config.ts file in the generated runtime config
causes a SyntaxError because Node.js cannot directly require TypeScript files.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.javascript.test_runner import _create_runtime_jest_config


def test_runtime_config_with_typescript_base_config():
    """Test that runtime config generation handles jest.config.ts files.

    When the base Jest config is a TypeScript file, the generated runtime config
    should not try to require() it directly, as that would fail with a SyntaxError.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a TypeScript Jest config (common in modern projects)
        base_config = project_root / "jest.config.ts"
        base_config.write_text("""
import { Config } from "jest"

const config: Config = {
  testEnvironment: "node",
  roots: ["<rootDir>/src"],
};

export default config;
""")

        # Create a test directory
        test_dir = project_root / "tests" / "generated"
        test_dir.mkdir(parents=True)

        # Generate runtime config
        runtime_config = _create_runtime_jest_config(
            base_config_path=base_config,
            project_root=project_root,
            test_dirs={test_dir}
        )

        assert runtime_config is not None
        assert runtime_config.exists()

        # Read the generated config
        config_content = runtime_config.read_text()

        # The generated config should NOT try to require a .ts file
        # because Node.js cannot directly require TypeScript files
        assert "require('./jest.config.ts')" not in config_content, (
            "Generated config should not require TypeScript files directly"
        )

        # It should either:
        # 1. Skip the base config and create a standalone config, OR
        # 2. Use a different approach (like ts-node/register)
        # For now, the fix should be to skip the base config when it's TypeScript
        assert "module.exports = {" in config_content


def test_runtime_config_with_javascript_base_config_still_works():
    """Test that JavaScript base configs still work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a JavaScript Jest config
        base_config = project_root / "jest.config.js"
        base_config.write_text("""
module.exports = {
  testEnvironment: "node",
  roots: ["<rootDir>/src"],
};
""")

        # Create a test directory
        test_dir = project_root / "tests" / "generated"
        test_dir.mkdir(parents=True)

        # Generate runtime config
        runtime_config = _create_runtime_jest_config(
            base_config_path=base_config,
            project_root=project_root,
            test_dirs={test_dir}
        )

        assert runtime_config is not None
        assert runtime_config.exists()

        # Read the generated config
        config_content = runtime_config.read_text()

        # JavaScript configs should still be required normally
        assert "require('./jest.config.js')" in config_content
        assert "...baseConfig" in config_content
