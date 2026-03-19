from __future__ import annotations

import json
from pathlib import Path

import tomlkit

from codeflash.code_utils.config_parser import LanguageConfig, find_all_config_files
from codeflash.languages.language_enum import Language


def write_toml(path: Path, data: dict) -> None:
    path.write_text(tomlkit.dumps(data), encoding="utf-8")


class TestFindAllConfigFiles:
    def test_finds_pyproject_toml_with_codeflash_section(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 1
        assert result[0].language == Language.PYTHON
        assert result[0].config_path == tmp_path / "pyproject.toml"

    def test_finds_codeflash_toml(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "codeflash.toml", {"tool": {"codeflash": {"module-root": "src/main/java"}}})
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 1
        assert result[0].language == Language.JAVA
        assert result[0].config_path == tmp_path / "codeflash.toml"

    def test_finds_multiple_configs(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        write_toml(tmp_path / "codeflash.toml", {"tool": {"codeflash": {"module-root": "src/main/java"}}})
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 2
        languages = {r.language for r in result}
        assert languages == {Language.PYTHON, Language.JAVA}

    def test_skips_pyproject_without_codeflash_section(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "pyproject.toml", {"tool": {"black": {"line-length": 120}}})
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 0

    def test_finds_config_in_parent_directory(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        subdir = tmp_path / "subproject"
        subdir.mkdir()
        write_toml(subdir / "codeflash.toml", {"tool": {"codeflash": {"module-root": "src/main/java"}}})
        monkeypatch.chdir(subdir)
        result = find_all_config_files()
        assert len(result) == 2
        languages = {r.language for r in result}
        assert languages == {Language.PYTHON, Language.JAVA}

    def test_closest_config_wins_per_language(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "."}}})
        subdir = tmp_path / "sub"
        subdir.mkdir()
        write_toml(subdir / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        monkeypatch.chdir(subdir)
        result = find_all_config_files()
        assert len(result) == 1
        assert result[0].language == Language.PYTHON
        assert result[0].config_path == subdir / "pyproject.toml"

    def test_finds_package_json_with_codeflash_section(self, tmp_path: Path, monkeypatch) -> None:
        pkg = {"codeflash": {"moduleRoot": "src"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 1
        assert result[0].language == Language.JAVASCRIPT
        assert result[0].config_path == tmp_path / "package.json"

    def test_finds_all_three_config_types(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        write_toml(tmp_path / "codeflash.toml", {"tool": {"codeflash": {"module-root": "src/main/java"}}})
        pkg = {"codeflash": {"moduleRoot": "src"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 3
        languages = {r.language for r in result}
        assert languages == {Language.PYTHON, Language.JAVA, Language.JAVASCRIPT}

    def test_malformed_toml_skipped(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "codeflash.toml").write_text("not valid [toml", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 0

    def test_missing_codeflash_section_skipped(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "codeflash.toml", {"tool": {"other": {"key": "value"}}})
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 0


def test_find_all_functions_uses_registry_not_singleton() -> None:
    """DISC-04: Verify find_all_functions_in_file uses per-file registry lookup."""
    import inspect

    from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

    source = inspect.getsource(find_all_functions_in_file)
    assert "get_language_support" in source
    assert "current_language_support" not in source
