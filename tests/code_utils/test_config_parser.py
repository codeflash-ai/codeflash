from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.code_utils.config_parser import ALL_CONFIG_FILES, PYPROJECT_TOML_CACHE, parse_config_file


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    PYPROJECT_TOML_CACHE.clear()
    ALL_CONFIG_FILES.clear()


PYPROJECT_TOML_CONTENT = """\
[tool.codeflash]
module-root = "src/python"
tests-root = "tests"
"""

CODEFLASH_TOML_CONTENT = """\
[tool.codeflash]
module-root = "src/main/java"
tests-root = "src/test/java"
"""


class TestParseConfigFileTomlPriority:
    def test_codeflash_toml_preferred_over_pyproject_when_same_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "pyproject.toml").write_text(PYPROJECT_TOML_CONTENT, encoding="utf-8")
        (tmp_path / "codeflash.toml").write_text(CODEFLASH_TOML_CONTENT, encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)
        (tmp_path / "src" / "python").mkdir(parents=True)
        (tmp_path / "tests").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        config, config_path = parse_config_file()

        assert config_path == tmp_path / "codeflash.toml"
        assert config["module_root"] == str((tmp_path / "src" / "main" / "java").resolve())
        assert config["tests_root"] == str((tmp_path / "src" / "test" / "java").resolve())

    def test_only_pyproject_toml_still_works(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "pyproject.toml").write_text(PYPROJECT_TOML_CONTENT, encoding="utf-8")
        (tmp_path / "src" / "python").mkdir(parents=True)
        (tmp_path / "tests").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        config, config_path = parse_config_file()

        assert config_path == tmp_path / "pyproject.toml"
        assert config["module_root"] == str((tmp_path / "src" / "python").resolve())

    def test_only_codeflash_toml_still_works(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "codeflash.toml").write_text(CODEFLASH_TOML_CONTENT, encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        config, config_path = parse_config_file()

        assert config_path == tmp_path / "codeflash.toml"
        assert config["module_root"] == str((tmp_path / "src" / "main" / "java").resolve())

    def test_closer_codeflash_toml_preferred_over_parent_pyproject(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # pyproject.toml in parent, codeflash.toml in child (closer to CWD)
        (tmp_path / "pyproject.toml").write_text(PYPROJECT_TOML_CONTENT, encoding="utf-8")
        (tmp_path / "src" / "python").mkdir(parents=True)
        (tmp_path / "tests").mkdir(parents=True)

        child = tmp_path / "subproject"
        child.mkdir()
        (child / "codeflash.toml").write_text(CODEFLASH_TOML_CONTENT, encoding="utf-8")
        (child / "src" / "main" / "java").mkdir(parents=True)
        (child / "src" / "test" / "java").mkdir(parents=True)
        monkeypatch.chdir(child)

        config, config_path = parse_config_file()

        assert config_path == child / "codeflash.toml"
        assert config["module_root"] == str((child / "src" / "main" / "java").resolve())

    def test_closer_pyproject_preferred_over_parent_codeflash_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # codeflash.toml in parent, pyproject.toml in child (closer to CWD)
        (tmp_path / "codeflash.toml").write_text(CODEFLASH_TOML_CONTENT, encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)

        child = tmp_path / "subproject"
        child.mkdir()
        (child / "pyproject.toml").write_text(PYPROJECT_TOML_CONTENT, encoding="utf-8")
        (child / "src" / "python").mkdir(parents=True)
        (child / "tests").mkdir(parents=True)
        monkeypatch.chdir(child)

        config, config_path = parse_config_file()

        assert config_path == child / "pyproject.toml"
        assert config["module_root"] == str((child / "src" / "python").resolve())

    def test_explicit_config_file_path_bypasses_discovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "pyproject.toml").write_text(PYPROJECT_TOML_CONTENT, encoding="utf-8")
        (tmp_path / "codeflash.toml").write_text(CODEFLASH_TOML_CONTENT, encoding="utf-8")
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)
        (tmp_path / "src" / "python").mkdir(parents=True)
        (tmp_path / "tests").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        config, config_path = parse_config_file(config_file_path=tmp_path / "pyproject.toml")

        assert config_path == tmp_path / "pyproject.toml"
        assert config["module_root"] == str((tmp_path / "src" / "python").resolve())
