"""Test that Vitest coverage config explicitly sets enabled: true.

This test verifies the fix for the issue where Vitest coverage was not
being collected despite passing --coverage flag. The generated config
override must explicitly set coverage.enabled: true to ensure coverage
is enabled when merging with project configs.

Trace IDs affected: 07be59c3-e53c-4350-b874-9d1fee5238d1 and 14 others.
"""

import tempfile
from pathlib import Path

import pytest


def test_vitest_config_has_coverage_enabled():
    """Test that generated codeflash.vitest.config.mjs sets coverage.enabled: true."""
    from codeflash.languages.javascript.vitest_runner import _ensure_codeflash_vitest_config

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a project config file
        vitest_config = project_root / "vitest.config.ts"
        vitest_config.write_text("""
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      include: ['./src/**/*.ts'],
    },
  },
});
""")

        # Create the codeflash config
        _ensure_codeflash_vitest_config(project_root)

        # Check that generated config exists
        codeflash_config = project_root / "codeflash.vitest.config.mjs"
        assert codeflash_config.exists(), "codeflash.vitest.config.mjs should be created"

        # Read and verify config content
        content = codeflash_config.read_text()

        # Verify coverage object is present
        assert "coverage: {" in content, "Config should have coverage object"

        # CRITICAL: Verify enabled: true is set
        # This is the fix - without this, Vitest may not enable coverage
        # even when --coverage flag is passed, due to complex config merging
        assert "enabled: true" in content, (
            "Config must set coverage.enabled: true to explicitly enable "
            "coverage when merging with project configs"
        )

        # Also verify reporter is set to json
        assert "reporter: ['json']" in content or 'reporter: ["json"]' in content, (
            "Config should set reporter to json"
        )


def test_vitest_config_coverage_enabled_with_no_original_config():
    """Test that coverage.enabled: true is set even when no original config exists."""
    from codeflash.languages.javascript.vitest_runner import _ensure_codeflash_vitest_config

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # No vitest.config.ts - should create minimal config
        _ensure_codeflash_vitest_config(project_root)

        codeflash_config = project_root / "codeflash.vitest.config.mjs"
        assert codeflash_config.exists()

        content = codeflash_config.read_text()

        # Even in minimal config, coverage should be explicitly enabled
        assert "coverage: {" in content
        assert "enabled: true" in content
        assert "reporter: ['json']" in content or 'reporter: ["json"]' in content
