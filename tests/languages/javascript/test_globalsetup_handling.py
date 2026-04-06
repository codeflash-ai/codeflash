"""Test that Codeflash properly handles Jest globalSetup/globalTeardown hooks."""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.javascript.test_runner import (
    _create_codeflash_jest_config,
    _create_runtime_jest_config,
)


def test_disables_globalsetup_and_globalteardown():
    """Test that generated Jest config disables globalSetup and globalTeardown from original config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create an original Jest config with globalSetup and globalTeardown
        original_config = project_root / "jest.config.js"
        original_config.write_text("""
module.exports = {
  testEnvironment: 'node',
  globalSetup: './globalSetup.ts',
  globalTeardown: './globalTeardown.ts',
  setupFilesAfterEnv: ['./setupTests.js'],
};
""")

        # Create codeflash config
        codeflash_config = _create_codeflash_jest_config(
            project_root=project_root,
            original_jest_config=original_config,
            for_esm=False
        )

        assert codeflash_config is not None
        assert codeflash_config.exists()

        # Read the generated config
        config_content = codeflash_config.read_text()

        # Should explicitly disable globalSetup and globalTeardown
        assert "globalSetup: undefined" in config_content
        assert "globalTeardown: undefined" in config_content

        # Should NOT reference the original globalSetup/globalTeardown scripts
        assert "./globalSetup.ts" not in config_content
        assert "./globalTeardown.ts" not in config_content


def test_disables_globalsetup_in_minimal_config():
    """Test that minimal config (no original) also disables globalSetup/globalTeardown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create codeflash config without original config
        codeflash_config = _create_codeflash_jest_config(
            project_root=project_root,
            original_jest_config=None,
            for_esm=False
        )

        assert codeflash_config is not None
        assert codeflash_config.exists()

        # Read the generated config
        config_content = codeflash_config.read_text()

        # Should explicitly disable globalSetup and globalTeardown
        assert "globalSetup: undefined" in config_content
        assert "globalTeardown: undefined" in config_content


def test_preserves_setupfilesafterenv():
    """Test that setupFilesAfterEnv is preserved (it's safe, runs per-test-file not globally)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create an original Jest config with setupFilesAfterEnv
        original_config = project_root / "jest.config.js"
        original_config.write_text("""
module.exports = {
  testEnvironment: 'node',
  globalSetup: './globalSetup.ts',
  setupFilesAfterEnv: ['./setupTests.js'],
};
""")

        # Create codeflash config
        codeflash_config = _create_codeflash_jest_config(
            project_root=project_root,
            original_jest_config=original_config,
            for_esm=False
        )

        assert codeflash_config is not None

        # Read the generated config
        config_content = codeflash_config.read_text()

        # Should disable globalSetup but NOT explicitly disable setupFilesAfterEnv
        assert "globalSetup: undefined" in config_content
        assert "setupFilesAfterEnv: undefined" not in config_content


def test_runtime_config_disables_globalsetup_with_base_config():
    """Test that runtime config disables globalSetup when extending a base config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a base Jest config
        base_config = project_root / "jest.config.js"
        base_config.write_text("""
module.exports = {
  testEnvironment: 'node',
  globalSetup: './globalSetup.ts',
};
""")

        # Create runtime config
        test_dirs = {str(project_root / "tests")}
        runtime_config = _create_runtime_jest_config(
            base_config_path=base_config,
            project_root=project_root,
            test_dirs=test_dirs
        )

        assert runtime_config is not None
        assert runtime_config.exists()

        # Read the generated config
        config_content = runtime_config.read_text()

        # Should explicitly disable globalSetup and globalTeardown
        assert "globalSetup: undefined" in config_content
        assert "globalTeardown: undefined" in config_content


def test_runtime_config_disables_globalsetup_standalone():
    """Test that standalone runtime config (no base) disables globalSetup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create runtime config without base config
        test_dirs = {str(project_root / "tests")}
        runtime_config = _create_runtime_jest_config(
            base_config_path=None,
            project_root=project_root,
            test_dirs=test_dirs
        )

        assert runtime_config is not None
        assert runtime_config.exists()

        # Read the generated config
        config_content = runtime_config.read_text()

        # Should explicitly disable globalSetup and globalTeardown
        assert "globalSetup: undefined" in config_content
        assert "globalTeardown: undefined" in config_content


def test_runtime_config_disables_globalsetup_with_typescript_base():
    """Test that runtime config handles TypeScript base configs correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a TypeScript base config
        base_config = project_root / "jest.config.ts"
        base_config.write_text("""
import { Config } from "jest";
export default {
  testEnvironment: 'node',
  globalSetup: './globalSetup.ts',
} as Config;
""")

        # Create runtime config (should use standalone mode for .ts configs)
        test_dirs = {str(project_root / "tests")}
        runtime_config = _create_runtime_jest_config(
            base_config_path=base_config,
            project_root=project_root,
            test_dirs=test_dirs
        )

        assert runtime_config is not None
        assert runtime_config.exists()

        # Read the generated config
        config_content = runtime_config.read_text()

        # Should explicitly disable globalSetup and globalTeardown
        assert "globalSetup: undefined" in config_content
        assert "globalTeardown: undefined" in config_content

        # Should NOT try to require the TypeScript config
        assert "require" not in config_content
