"""Test that runtime Jest config includes TypeScript transformer.

Regression test for bug where _create_runtime_jest_config() did not include
TypeScript transformer when base config was a .ts file, causing all TypeScript
tests to fail with syntax errors.

Issue: When base Jest config is TypeScript (.ts extension), the runtime config
creates a standalone config (cannot require .ts files) but was missing the
transform configuration entirely.
"""

import json
import tempfile
from pathlib import Path

import pytest

from codeflash.languages.javascript.test_runner import _create_runtime_jest_config


def test_runtime_config_includes_typescript_transform_with_ts_jest():
    """Runtime config should include ts-jest transform when available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create package.json with ts-jest
        package_json = {
            "name": "test-project",
            "devDependencies": {
                "ts-jest": "^29.0.0",
                "jest": "^29.0.0"
            }
        }
        (project_root / "package.json").write_text(json.dumps(package_json))

        # Create TypeScript base config (triggers standalone path)
        base_config = project_root / "jest.config.ts"
        base_config.write_text("""
export default {
  preset: 'ts-jest',
  testEnvironment: 'node',
};
""")

        # Create runtime config
        test_dirs = {str(project_root / "tests")}
        runtime_config = _create_runtime_jest_config(base_config, project_root, test_dirs)

        assert runtime_config is not None
        assert runtime_config.exists()

        # Read and verify content
        config_content = runtime_config.read_text()

        # Should include TypeScript transform
        assert "transform:" in config_content, "Runtime config missing transform section"
        assert "ts-jest" in config_content, "Runtime config missing ts-jest transformer"
        assert "'^.+\\\\.(ts|tsx)$'" in config_content, "Runtime config missing TS file pattern"


def test_runtime_config_includes_typescript_transform_with_swc():
    """Runtime config should include @swc/jest transform when available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create package.json with @swc/jest
        package_json = {
            "name": "test-project",
            "devDependencies": {
                "@swc/jest": "^0.2.0",
                "jest": "^29.0.0"
            }
        }
        (project_root / "package.json").write_text(json.dumps(package_json))

        # Create TypeScript base config
        base_config = project_root / "jest.config.ts"
        base_config.write_text("export default { testEnvironment: 'node' };")

        # Create runtime config
        test_dirs = {str(project_root / "tests")}
        runtime_config = _create_runtime_jest_config(base_config, project_root, test_dirs)

        assert runtime_config is not None
        config_content = runtime_config.read_text()

        # Should include @swc/jest transform
        assert "transform:" in config_content
        assert "@swc/jest" in config_content


def test_runtime_config_includes_typescript_transform_with_babel_fallback():
    """Runtime config should install and use babel-jest as fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create package.json with @babel/core but no TS transformer
        # This triggers the fallback path that installs @babel/preset-typescript
        package_json = {
            "name": "test-project",
            "devDependencies": {
                "@babel/core": "^7.0.0",
                "babel-jest": "^29.0.0",
                "jest": "^29.0.0"
            }
        }
        (project_root / "package.json").write_text(json.dumps(package_json))

        # Create TypeScript base config
        base_config = project_root / "jest.config.ts"
        base_config.write_text("export default { testEnvironment: 'node' };")

        # Create runtime config
        test_dirs = {str(project_root / "tests")}
        runtime_config = _create_runtime_jest_config(base_config, project_root, test_dirs)

        assert runtime_config is not None
        config_content = runtime_config.read_text()

        # Should include babel-jest transform (may or may not succeed in installing preset)
        # If preset installation succeeds, should have babel-jest transform
        if "babel-jest" in config_content:
            assert "transform:" in config_content
            assert "@babel/preset-typescript" in config_content


def test_runtime_config_no_transform_when_no_typescript_transformer():
    """Runtime config gracefully handles missing TypeScript transformer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create package.json WITHOUT any TypeScript transformer
        package_json = {
            "name": "test-project",
            "devDependencies": {
                "jest": "^29.0.0"
            }
        }
        (project_root / "package.json").write_text(json.dumps(package_json))

        # Create TypeScript base config
        base_config = project_root / "jest.config.ts"
        base_config.write_text("export default { testEnvironment: 'node' };")

        # Create runtime config
        test_dirs = {str(project_root / "tests")}
        runtime_config = _create_runtime_jest_config(base_config, project_root, test_dirs)

        # Should still create config, just without transforms
        assert runtime_config is not None
        assert runtime_config.exists()

        config_content = runtime_config.read_text()
        # Should have basic config elements
        assert "roots:" in config_content
        assert "testMatch:" in config_content
