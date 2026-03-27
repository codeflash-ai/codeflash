from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import tomlkit

from codeflash.code_utils.config_parser import find_all_config_files
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

    def test_finds_java_via_build_tool_detection(self, tmp_path: Path, monkeypatch) -> None:
        java_config = {"language": "java", "module_root": str(tmp_path / "src/main/java")}
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with patch(
            "codeflash.code_utils.config_parser._parse_java_config_for_dir",
            return_value=java_config,
        ):
            result = find_all_config_files()
        assert len(result) == 1
        assert result[0].language == Language.JAVA
        assert result[0].config_path == tmp_path

    def test_finds_multiple_configs_python_and_java(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        java_config = {"language": "java", "module_root": str(tmp_path / "src/main/java")}
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with patch(
            "codeflash.code_utils.config_parser._parse_java_config_for_dir",
            return_value=java_config,
        ):
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
        java_config = {"language": "java", "module_root": str(subdir / "src/main/java")}
        (subdir / "pom.xml").write_text("<project/>", encoding="utf-8")
        monkeypatch.chdir(subdir)
        with patch(
            "codeflash.code_utils.config_parser._parse_java_config_for_dir",
            return_value=java_config,
        ):
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
        pkg = {"codeflash": {"moduleRoot": "src"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
        java_config = {"language": "java", "module_root": str(tmp_path / "src/main/java")}
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with patch(
            "codeflash.code_utils.config_parser._parse_java_config_for_dir",
            return_value=java_config,
        ):
            result = find_all_config_files()
        assert len(result) == 3
        languages = {r.language for r in result}
        assert languages == {Language.PYTHON, Language.JAVA, Language.JAVASCRIPT}

    def test_no_java_when_no_build_file_exists(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 0

    def test_missing_codeflash_section_skipped(self, tmp_path: Path, monkeypatch) -> None:
        write_toml(tmp_path / "pyproject.toml", {"tool": {"other": {"key": "value"}}})
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 0

    def test_finds_java_in_subdirectory(self, tmp_path: Path, monkeypatch) -> None:
        """Monorepo: Java project in a subdirectory is discovered from the repo root."""
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        java_dir = tmp_path / "java"
        java_dir.mkdir()
        (java_dir / "pom.xml").write_text("<project/>", encoding="utf-8")
        java_config = {"language": "java", "module_root": str(java_dir / "src/main/java")}
        monkeypatch.chdir(tmp_path)
        with patch(
            "codeflash.code_utils.config_parser._parse_java_config_for_dir",
            return_value=java_config,
        ):
            result = find_all_config_files()
        assert len(result) == 2
        languages = {r.language for r in result}
        assert languages == {Language.PYTHON, Language.JAVA}
        java_result = next(r for r in result if r.language == Language.JAVA)
        assert java_result.config_path == java_dir

    def test_finds_js_in_subdirectory(self, tmp_path: Path, monkeypatch) -> None:
        """Monorepo: JS project in a subdirectory is discovered from the repo root."""
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        js_dir = tmp_path / "js"
        js_dir.mkdir()
        pkg = {"codeflash": {"moduleRoot": "src"}}
        (js_dir / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 2
        languages = {r.language for r in result}
        assert languages == {Language.PYTHON, Language.JAVASCRIPT}

    def test_finds_all_three_in_monorepo_subdirs(self, tmp_path: Path, monkeypatch) -> None:
        """Monorepo: Python at root, Java and JS in subdirectories."""
        write_toml(tmp_path / "pyproject.toml", {"tool": {"codeflash": {"module-root": "src"}}})
        java_dir = tmp_path / "java"
        java_dir.mkdir()
        (java_dir / "pom.xml").write_text("<project/>", encoding="utf-8")
        java_config = {"language": "java", "module_root": str(java_dir / "src/main/java")}
        js_dir = tmp_path / "js"
        js_dir.mkdir()
        pkg = {"codeflash": {"moduleRoot": "src"}}
        (js_dir / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with patch(
            "codeflash.code_utils.config_parser._parse_java_config_for_dir",
            return_value=java_config,
        ):
            result = find_all_config_files()
        assert len(result) == 3
        languages = {r.language for r in result}
        assert languages == {Language.PYTHON, Language.JAVA, Language.JAVASCRIPT}

    def test_skips_hidden_and_build_subdirs(self, tmp_path: Path, monkeypatch) -> None:
        """Subdirectory scan skips .git, node_modules, target, etc."""
        for name in [".git", "node_modules", "target", "build", "__pycache__"]:
            d = tmp_path / name
            d.mkdir()
            write_toml(d / "pyproject.toml", {"tool": {"codeflash": {"module-root": "."}}})
        monkeypatch.chdir(tmp_path)
        result = find_all_config_files()
        assert len(result) == 0

    def test_root_config_wins_over_subdir(self, tmp_path: Path, monkeypatch) -> None:
        """Config at CWD (found during upward walk) takes precedence over subdirectory."""
        (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
        java_dir = tmp_path / "java"
        java_dir.mkdir()
        (java_dir / "pom.xml").write_text("<project/>", encoding="utf-8")
        java_config = {"language": "java", "module_root": str(tmp_path / "src/main/java")}
        monkeypatch.chdir(tmp_path)
        with patch(
            "codeflash.code_utils.config_parser._parse_java_config_for_dir",
            return_value=java_config,
        ):
            result = find_all_config_files()
        java_results = [r for r in result if r.language == Language.JAVA]
        assert len(java_results) == 1
        assert java_results[0].config_path == tmp_path


def test_find_all_functions_uses_registry_not_singleton() -> None:
    """DISC-04: Verify find_all_functions_in_file uses per-file registry lookup."""
    import inspect

    from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

    source = inspect.getsource(find_all_functions_in_file)
    assert "get_language_support" in source
    assert "current_language_support" not in source
