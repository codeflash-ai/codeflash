"""Test that Codeflash Vitest config properly overrides coverage settings."""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.javascript.vitest_runner import _ensure_codeflash_vitest_config


def test_codeflash_vitest_config_overrides_coverage():
    """Test that generated config overrides coverage reporter to json.

    This is a regression test for the bug where Codeflash would pass
    --coverage.reporter=json on command line, but if the project's
    vitest.config.ts had coverage.reporter set (e.g., ["text", "lcov"]),
    Vitest's mergeConfig wouldn't properly handle the nested coverage
    object merge, resulting in coverage files not being written.

    The fix is to explicitly override coverage settings in the generated
    codeflash.vitest.config.mjs file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a vitest.config.ts with coverage settings like openclaw project
        vitest_config = project_root / "vitest.config.ts"
        vitest_config.write_text("""
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      all: false,
      thresholds: {
        lines: 70,
        functions: 70,
      },
    },
  },
});
""")

        # Generate the codeflash config
        config_path = _ensure_codeflash_vitest_config(project_root)

        assert config_path is not None, "Config should be created"
        assert config_path.exists(), "Config file should exist"

        # Read and verify the generated config
        config_content = config_path.read_text()

        # Check that it merges with original config
        assert "mergeConfig" in config_content, "Should use mergeConfig"
        assert "import originalConfig from './vitest.config.ts'" in config_content

        # CRITICAL: Check that coverage settings are explicitly overridden
        # This is the fix for the bug - without this, coverage files aren't written
        assert "coverage:" in config_content, (
            "Config must explicitly override coverage settings to ensure "
            "json reporter is used regardless of project config"
        )
        assert "reporter:" in config_content, (
            "Config must override coverage.reporter to ['json']"
        )
        # The config should set reporter to json (as array or string)
        # Note: We're checking the config override, not the command-line flag
        assert "['json']" in config_content or '["json"]' in config_content, (
            "Coverage reporter must be set to ['json'] to ensure coverage "
            "files are written in the expected format"
        )


def test_codeflash_vitest_config_without_original_coverage():
    """Test generated config when original has no coverage settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a minimal vitest.config.ts without coverage settings
        vitest_config = project_root / "vitest.config.ts"
        vitest_config.write_text("""
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
  },
});
""")

        # Generate the codeflash config
        config_path = _ensure_codeflash_vitest_config(project_root)

        assert config_path is not None
        assert config_path.exists()

        config_content = config_path.read_text()

        # Should still override coverage settings explicitly
        assert "coverage:" in config_content, (
            "Config must explicitly set coverage even when original doesn't have it"
        )
