from pathlib import Path

import pytest

from codeflash.languages.javascript.vitest_runner import _ensure_codeflash_vitest_config


def test_codeflash_vitest_config_overrides_setupfiles(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    # Create a project with setup file
    (project_root / "test").mkdir()
    (project_root / "test" / "setup.ts").write_text("// Setup file\n", encoding="utf-8")

    vitest_config = """import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    setupFiles: ["test/setup.ts"],  // Relative path - will cause issues
    include: ["src/**/*.test.ts"],
  },
});
"""
    (project_root / "vitest.config.ts").write_text(vitest_config, encoding="utf-8")

    codeflash_config_path = _ensure_codeflash_vitest_config(project_root)

    assert codeflash_config_path is not None
    assert codeflash_config_path.exists()

    config_content = codeflash_config_path.read_text(encoding="utf-8")

    assert "setupFiles" in config_content, (
        "Generated config must explicitly handle setupFiles to prevent "
        "relative path resolution issues. Current config:\n" + config_content
    )
    assert "setupFiles: []" in config_content or "setupFiles:" in config_content, (
        "setupFiles must be explicitly set in the merged config"
    )


def test_codeflash_vitest_config_without_setupfiles(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    vitest_config = """import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
  },
});
"""
    (project_root / "vitest.config.ts").write_text(vitest_config, encoding="utf-8")

    codeflash_config_path = _ensure_codeflash_vitest_config(project_root)

    assert codeflash_config_path is not None
    assert codeflash_config_path.exists()

    config_content = codeflash_config_path.read_text(encoding="utf-8")
    assert "mergeConfig" in config_content or "defineConfig" in config_content
