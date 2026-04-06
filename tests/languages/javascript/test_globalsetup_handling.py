from pathlib import Path

from codeflash.languages.javascript.test_runner import (
    _create_codeflash_jest_config,
    _create_runtime_jest_config,
)


def test_disables_globalsetup_and_globalteardown(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    original_config = project_root / "jest.config.js"
    original_config.write_text(
        """
module.exports = {
  testEnvironment: 'node',
  globalSetup: './globalSetup.ts',
  globalTeardown: './globalTeardown.ts',
  setupFilesAfterEnv: ['./setupTests.js'],
};
""",
        encoding="utf-8",
    )

    codeflash_config = _create_codeflash_jest_config(
        project_root=project_root,
        original_jest_config=original_config,
        for_esm=False,
    )

    assert codeflash_config is not None
    assert codeflash_config.exists()

    config_content = codeflash_config.read_text(encoding="utf-8")

    assert "globalSetup: undefined" in config_content
    assert "globalTeardown: undefined" in config_content

    # The original scripts are not embedded in the wrapper config (spread at runtime)
    assert "./globalSetup.ts" not in config_content
    assert "./globalTeardown.ts" not in config_content


def test_disables_globalsetup_in_minimal_config(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    codeflash_config = _create_codeflash_jest_config(
        project_root=project_root,
        original_jest_config=None,
        for_esm=False,
    )

    assert codeflash_config is not None
    assert codeflash_config.exists()

    config_content = codeflash_config.read_text(encoding="utf-8")

    assert "globalSetup: undefined" in config_content
    assert "globalTeardown: undefined" in config_content


def test_preserves_setupfilesafterenv(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    original_config = project_root / "jest.config.js"
    original_config.write_text(
        """
module.exports = {
  testEnvironment: 'node',
  globalSetup: './globalSetup.ts',
  setupFilesAfterEnv: ['./setupTests.js'],
};
""",
        encoding="utf-8",
    )

    codeflash_config = _create_codeflash_jest_config(
        project_root=project_root,
        original_jest_config=original_config,
        for_esm=False,
    )

    assert codeflash_config is not None

    config_content = codeflash_config.read_text(encoding="utf-8")

    assert "globalSetup: undefined" in config_content
    assert "setupFilesAfterEnv: undefined" not in config_content


def test_runtime_config_disables_globalsetup_with_base_config(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    base_config = project_root / "jest.config.js"
    base_config.write_text(
        """
module.exports = {
  testEnvironment: 'node',
  globalSetup: './globalSetup.ts',
};
""",
        encoding="utf-8",
    )

    test_dirs = {str(project_root / "tests")}
    runtime_config = _create_runtime_jest_config(
        base_config_path=base_config,
        project_root=project_root,
        test_dirs=test_dirs,
    )

    assert runtime_config is not None
    assert runtime_config.exists()

    config_content = runtime_config.read_text(encoding="utf-8")

    assert "globalSetup: undefined" in config_content
    assert "globalTeardown: undefined" in config_content


def test_runtime_config_disables_globalsetup_standalone(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    test_dirs = {str(project_root / "tests")}
    runtime_config = _create_runtime_jest_config(
        base_config_path=None,
        project_root=project_root,
        test_dirs=test_dirs,
    )

    assert runtime_config is not None
    assert runtime_config.exists()

    config_content = runtime_config.read_text(encoding="utf-8")

    assert "globalSetup: undefined" in config_content
    assert "globalTeardown: undefined" in config_content


def test_runtime_config_disables_globalsetup_with_typescript_base(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()

    base_config = project_root / "jest.config.ts"
    base_config.write_text(
        """
import { Config } from "jest";
export default {
  testEnvironment: 'node',
  globalSetup: './globalSetup.ts',
} as Config;
""",
        encoding="utf-8",
    )

    test_dirs = {str(project_root / "tests")}
    runtime_config = _create_runtime_jest_config(
        base_config_path=base_config,
        project_root=project_root,
        test_dirs=test_dirs,
    )

    assert runtime_config is not None
    assert runtime_config.exists()

    config_content = runtime_config.read_text(encoding="utf-8")

    assert "globalSetup: undefined" in config_content
    assert "globalTeardown: undefined" in config_content

    # Should NOT try to require the TypeScript config
    assert "require" not in config_content
