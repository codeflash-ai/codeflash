"""Tests for TypeScript transform extraction from jest.config.ts"""

import tempfile
from pathlib import Path

import pytest


class TestTypeScriptTransformExtraction:
    """Test that Codeflash extracts transform directives from TypeScript Jest configs."""

    def test_extracts_swc_transform_from_typescript_config(self):
        """Test that @swc/jest transform is preserved when original config is TypeScript."""
        from codeflash.languages.javascript.test_runner import (
            _create_codeflash_jest_config,
            clear_created_config_files,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()

            # Create package.json without ts-jest (mimics budibase)
            (tmpdir_path / "package.json").write_text('{"name": "test", "devDependencies": {"@swc/jest": "^0.2.0"}}')

            # Create jest.config.ts with @swc/jest transform (mimics budibase)
            jest_config_content = """import { Config } from "jest"

const config: Config = {
  transform: {
    "^.+\\\\.ts?$": "@swc/jest",
  },
}

export default config
"""
            jest_config_path = tmpdir_path / "jest.config.ts"
            jest_config_path.write_text(jest_config_content)

            clear_created_config_files()

            # Call _create_codeflash_jest_config
            result = _create_codeflash_jest_config(tmpdir_path, jest_config_path)

            assert result is not None, "Should create a codeflash config"

            # Read the generated config
            generated_config = result.read_text()

            print(f"Generated config:\n{generated_config}")

            # The generated config MUST include a transform directive for TypeScript files
            # This is critical - without it, Jest cannot parse TypeScript
            assert "transform" in generated_config, f"Generated config must include transform directive. Got:\n{generated_config}"

            # Verify it can actually transform TypeScript files (.ts extension)
            assert (
                "@swc/jest" in generated_config or "ts-jest" in generated_config or '"^.+\\\\.ts' in generated_config
            ), f"Generated config must include TypeScript file transformation. Got:\n{generated_config}"

            clear_created_config_files()

    def test_extracts_ts_jest_transform_from_typescript_config(self):
        """Test that ts-jest transform is preserved when original config is TypeScript."""
        from codeflash.languages.javascript.test_runner import (
            _create_codeflash_jest_config,
            clear_created_config_files,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()

            # Create package.json with ts-jest
            (tmpdir_path / "package.json").write_text('{"name": "test", "devDependencies": {"ts-jest": "^29.0.0"}}')

            # Create jest.config.ts with ts-jest transform
            jest_config_content = """import { Config } from "jest"

const config: Config = {
  transform: {
    "^.+\\\\.(ts|tsx)$": "ts-jest",
  },
}

export default config
"""
            jest_config_path = tmpdir_path / "jest.config.ts"
            jest_config_path.write_text(jest_config_content)

            clear_created_config_files()

            # Call _create_codeflash_jest_config
            result = _create_codeflash_jest_config(tmpdir_path, jest_config_path)

            assert result is not None, "Should create a codeflash config"

            # Read the generated config
            generated_config = result.read_text()

            # Should include ts-jest transform
            assert "ts-jest" in generated_config, f"Generated config should include ts-jest transform. Got:\n{generated_config}"

            clear_created_config_files()
