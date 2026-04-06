"""Test that Codeflash Vitest config properly overrides coverage settings."""

from pathlib import Path

import pytest

from codeflash.languages.javascript.vitest_runner import _ensure_codeflash_vitest_config


def test_codeflash_vitest_config_overrides_coverage(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    vitest_config = project_root / "vitest.config.ts"
    vitest_config.write_text(
        """
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
""",
        encoding="utf-8",
    )

    config_path = _ensure_codeflash_vitest_config(project_root)

    assert config_path is not None, "Config should be created"
    assert config_path.exists(), "Config file should exist"

    config_content = config_path.read_text(encoding="utf-8")

    assert "mergeConfig" in config_content, "Should use mergeConfig"
    assert "import originalConfig from './vitest.config.ts'" in config_content
    assert "coverage:" in config_content, (
        "Config must explicitly override coverage settings to ensure "
        "json reporter is used regardless of project config"
    )
    assert "reporter:" in config_content, "Config must override coverage.reporter to ['json']"
    assert "['json']" in config_content or '["json"]' in config_content, (
        "Coverage reporter must be set to ['json'] to ensure coverage files are written in the expected format"
    )


def test_codeflash_vitest_config_without_original_coverage(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    vitest_config = project_root / "vitest.config.ts"
    vitest_config.write_text(
        """
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
  },
});
""",
        encoding="utf-8",
    )

    config_path = _ensure_codeflash_vitest_config(project_root)

    assert config_path is not None
    assert config_path.exists()

    config_content = config_path.read_text(encoding="utf-8")

    assert "coverage:" in config_content, "Config must explicitly set coverage even when original doesn't have it"
