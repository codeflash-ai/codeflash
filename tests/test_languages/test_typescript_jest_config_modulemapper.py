"""Tests for TypeScript Jest config moduleNameMapper preservation in monorepos."""

import json
import tempfile
from pathlib import Path

import pytest


class TestTypeScriptJestConfigModuleMapper:
    """Tests for preserving moduleNameMapper from TypeScript Jest configs."""

    def test_runtime_config_preserves_modulemapper_from_typescript_config(self):
        """Test that moduleNameMapper is extracted and preserved from TypeScript Jest configs.

        This is critical for monorepo workspace packages (e.g., @budibase/backend-core)
        to resolve correctly in generated tests.
        """
        import subprocess

        from codeflash.languages.javascript.test_runner import _create_runtime_jest_config

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()

            # Create monorepo structure
            pkg_dir = tmpdir_path / "packages" / "server"
            pkg_dir.mkdir(parents=True)

            # Monorepo root with node_modules (for _find_monorepo_root)
            (tmpdir_path / "package.json").write_text(json.dumps({
                "name": "test-monorepo",
                "workspaces": {"packages": ["packages/*"]}
            }))
            (tmpdir_path / "node_modules").mkdir()

            # Install ts-node so the TypeScript config can be loaded
            # This simulates a real project environment where ts-node is available
            subprocess.run(
                ["npm", "install", "--no-save", "ts-node@10.9.2", "typescript@5.3.3"],
                cwd=tmpdir_path,
                capture_output=True,
                check=False,
                timeout=120,
            )

            # Package with TypeScript Jest config
            (pkg_dir / "package.json").write_text(json.dumps({
                "name": "@test/server",
                "version": "1.0.0"
            }))

            # TypeScript Jest config with moduleNameMapper for workspace packages
            jest_config_ts = """import { Config } from 'jest';

const config: Config = {
  moduleNameMapper: {
    '@test/backend-core/(.*)': '<rootDir>/../backend-core/$1',
    '@test/backend-core': '<rootDir>/../backend-core/src',
    '@test/shared-core': '<rootDir>/../shared-core/src',
  },
  transform: {
    '^.+\\\\.ts?$': '@swc/jest',
  },
};

export default config;
"""
            jest_config_path = pkg_dir / "jest.config.ts"
            jest_config_path.write_text(jest_config_ts)

            # Create a test directory to include in roots
            test_dir = pkg_dir / "src" / "tests" / "codeflash-generated"
            test_dir.mkdir(parents=True)
            test_dirs = {str(test_dir)}

            # Create runtime config
            runtime_config_path = _create_runtime_jest_config(
                jest_config_path,
                pkg_dir,
                test_dirs
            )

            assert runtime_config_path is not None, "Runtime config should be created"
            assert runtime_config_path.exists(), "Runtime config file should exist"

            content = runtime_config_path.read_text()

            # CRITICAL: moduleNameMapper must be present for workspace packages to resolve
            assert "moduleNameMapper" in content, (
                "Runtime config must include moduleNameMapper for monorepo workspace packages. "
                "Without it, imports like '@test/backend-core' will fail with 'Cannot find module'."
            )

            # Verify the specific mappings are preserved
            assert "@test/backend-core" in content, (
                "Workspace package mapping '@test/backend-core' must be preserved"
            )
            assert "@test/shared-core" in content, (
                "Workspace package mapping '@test/shared-core' must be preserved"
            )

            # Verify other required config is still present
            assert "roots" in content
            assert "globalSetup: undefined" in content
            assert "globalTeardown: undefined" in content

    def test_runtime_config_falls_back_gracefully_when_typescript_config_unreadable(self):
        """Test graceful fallback when TypeScript config can't be loaded."""
        from codeflash.languages.javascript.test_runner import _create_runtime_jest_config

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()

            (tmpdir_path / "package.json").write_text('{"name": "test"}')

            # Invalid TypeScript config that can't be executed
            jest_config_ts = """
import { Config } from 'jest';
// Syntax error - missing semicolon and export
const config: Config = {
  moduleNameMapper: {
    '@test/a': '<rootDir>/../a/src'
  }
}
"""
            jest_config_path = tmpdir_path / "jest.config.ts"
            jest_config_path.write_text(jest_config_ts)

            test_dirs = {str(tmpdir_path / "tests")}

            # Should still create a runtime config (even without moduleNameMapper)
            runtime_config_path = _create_runtime_jest_config(
                jest_config_path,
                tmpdir_path,
                test_dirs
            )

            assert runtime_config_path is not None, "Should create runtime config even if TS config unreadable"
            assert runtime_config_path.exists()

            content = runtime_config_path.read_text()
            # Should have basic config even without moduleNameMapper
            assert "roots" in content
            assert "globalSetup: undefined" in content
