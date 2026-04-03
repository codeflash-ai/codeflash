"""Test that Codeflash Vitest config properly handles setupFiles from project config.

This test verifies that when creating a custom Vitest config, setupFiles paths
are converted to absolute paths or cleared to prevent resolution issues in nested directories.
"""

from pathlib import Path
import tempfile
import pytest


def test_codeflash_vitest_config_overrides_setupfiles():
    """Test that generated config overrides setupFiles to prevent path resolution issues.

    When a project has setupFiles with relative paths, and Codeflash generates tests
    for functions in nested directories, those relative paths will resolve incorrectly.

    The fix: Convert setupFiles paths to absolute, or disable them for generated tests.
    """
    from codeflash.languages.javascript.vitest_runner import _ensure_codeflash_vitest_config

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a project with setup file
        (project_root / "test").mkdir()
        setup_file = project_root / "test" / "setup.ts"
        setup_file.write_text("// Setup file\n")

        # Create vitest config with relative setupFiles path
        vitest_config = """import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    setupFiles: ["test/setup.ts"],  // Relative path - will cause issues
    include: ["src/**/*.test.ts"],
  },
});
"""
        (project_root / "vitest.config.ts").write_text(vitest_config)

        # Call the function to create Codeflash config
        codeflash_config_path = _ensure_codeflash_vitest_config(project_root)

        # Verify the config was created
        assert codeflash_config_path is not None
        assert codeflash_config_path.exists()

        # Read the generated config
        config_content = codeflash_config_path.read_text()

        # The config should either:
        # 1. Set setupFiles to an empty array (disable setup files for generated tests)
        # 2. OR convert the path to absolute using project root resolution

        # Check that setupFiles is mentioned and handled in the merge
        assert "setupFiles" in config_content, (
            "Generated config must explicitly handle setupFiles to prevent "
            "relative path resolution issues. Current config:\n" + config_content
        )

        # The config should set setupFiles to [] or to absolute paths
        # This prevents the relative path from being resolved incorrectly
        assert ("setupFiles: []" in config_content or
                "setupFiles:" in config_content), (
            "setupFiles must be explicitly set in the merged config"
        )


def test_codeflash_vitest_config_without_setupfiles():
    """Test that configs without setupFiles still work correctly."""
    from codeflash.languages.javascript.vitest_runner import _ensure_codeflash_vitest_config

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create vitest config WITHOUT setupFiles
        vitest_config = """import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
  },
});
"""
        (project_root / "vitest.config.ts").write_text(vitest_config)

        # Call the function to create Codeflash config
        codeflash_config_path = _ensure_codeflash_vitest_config(project_root)

        # Verify the config was created
        assert codeflash_config_path is not None
        assert codeflash_config_path.exists()

        # Config should be created successfully
        config_content = codeflash_config_path.read_text()
        assert "mergeConfig" in config_content or "defineConfig" in config_content
