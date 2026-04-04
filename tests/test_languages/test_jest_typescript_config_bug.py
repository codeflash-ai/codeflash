"""Test for TypeScript Jest config require bug.

Regression test for the issue where _create_runtime_jest_config generates
code that tries to require('./jest.config.ts'), which fails because Node.js
CommonJS cannot load TypeScript files directly.

Bug: https://github.com/codeflash-ai/codeflash/issues/XXX
Affects: 18 out of 38 optimization runs in initial testing
"""

import subprocess
import tempfile
from pathlib import Path

import pytest


class TestTypeScriptJestConfigRequire:
    """Test that runtime config correctly handles TypeScript base configs."""

    def test_runtime_config_with_typescript_base_config_loads_without_error(self):
        """Runtime config should NOT try to require .ts files directly.

        When base_config_path points to jest.config.ts, the generated runtime
        config must not use require('./jest.config.ts') because Node.js cannot
        parse TypeScript syntax in CommonJS require().

        This test creates a jest.config.ts file and verifies that the generated
        runtime config can be successfully loaded by Node.js without syntax errors.
        """
        from codeflash.languages.javascript.test_runner import _create_runtime_jest_config

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir).resolve()

            # Create a TypeScript Jest config (realistic content with TS syntax)
            ts_config_path = project_path / "jest.config.ts"
            ts_config_content = """import { Config } from "jest"

const config: Config = {
  testEnvironment: 'node',
  testMatch: ['**/*.test.ts'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
}

export default config
"""
            ts_config_path.write_text(ts_config_content, encoding="utf-8")

            # Create runtime config with the TS base config
            test_dirs = {str(project_path / "test")}
            runtime_config_path = _create_runtime_jest_config(
                base_config_path=ts_config_path,
                project_root=project_path,
                test_dirs=test_dirs
            )

            assert runtime_config_path is not None, "Runtime config should be created"
            assert runtime_config_path.exists(), "Runtime config file should exist"

            # Read the generated content
            runtime_content = runtime_config_path.read_text(encoding="utf-8")

            # CRITICAL CHECK: Should NOT contain require('./jest.config.ts')
            # This is the bug we're fixing
            assert "require('./jest.config.ts')" not in runtime_content, (
                "Runtime config should not try to require .ts files directly"
            )

            # The config should handle TypeScript configs appropriately:
            # - Either omit the extension (let Node resolve to .js)
            # - Or use a TypeScript loader (ts-node)
            # - Or skip requiring TS configs entirely

            # Verify the generated config can be loaded by Node.js without errors
            test_script = project_path / "test_load_config.js"
            test_script_content = f"""
try {{
    const config = require('./{runtime_config_path.name}');
    console.log('SUCCESS');
    process.exit(0);
}} catch (err) {{
    console.error('FAILED:', err.message);
    process.exit(1);
}}
"""
            test_script.write_text(test_script_content, encoding="utf-8")

            result = subprocess.run(
                ["node", str(test_script)],
                capture_output=True,
                text=True,
                cwd=project_path,
                timeout=5,
            )

            assert result.returncode == 0, (
                f"Generated runtime config should load without errors.\n"
                f"Config path: {runtime_config_path}\n"
                f"Config content:\n{runtime_content}\n"
                f"Node output:\n{result.stdout}\n{result.stderr}"
            )
            assert "SUCCESS" in result.stdout

    def test_runtime_config_with_js_base_config_works(self):
        """Verify that .js base configs still work correctly (control test)."""
        from codeflash.languages.javascript.test_runner import _create_runtime_jest_config

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir).resolve()

            # Create a JavaScript Jest config
            js_config_path = project_path / "jest.config.js"
            js_config_content = """module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/*.test.js'],
}
"""
            js_config_path.write_text(js_config_content, encoding="utf-8")

            # Create runtime config with the JS base config
            test_dirs = {str(project_path / "test")}
            runtime_config_path = _create_runtime_jest_config(
                base_config_path=js_config_path,
                project_root=project_path,
                test_dirs=test_dirs
            )

            assert runtime_config_path is not None
            assert runtime_config_path.exists()

            # Verify it loads without errors
            test_script = project_path / "test_load_config.js"
            test_script_content = f"""
try {{
    const config = require('./{runtime_config_path.name}');
    console.log('SUCCESS');
    process.exit(0);
}} catch (err) {{
    console.error('FAILED:', err.message);
    process.exit(1);
}}
"""
            test_script.write_text(test_script_content, encoding="utf-8")

            result = subprocess.run(
                ["node", str(test_script)],
                capture_output=True,
                text=True,
                cwd=project_path,
                timeout=5,
            )

            assert result.returncode == 0, f"JS config should load: {result.stderr}"
            assert "SUCCESS" in result.stdout
