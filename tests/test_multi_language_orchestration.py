from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import tomlkit

from codeflash.code_utils.config_parser import LanguageConfig
from codeflash.languages.language_enum import Language


def write_toml(path: Path, data: dict) -> None:
    path.write_text(tomlkit.dumps(data), encoding="utf-8")


def make_base_args(**overrides) -> Namespace:
    defaults = {
        "module_root": None,
        "tests_root": None,
        "benchmarks_root": None,
        "ignore_paths": None,
        "pytest_cmd": None,
        "formatter_cmds": None,
        "disable_telemetry": None,
        "disable_imports_sorting": None,
        "git_remote": None,
        "override_fixtures": None,
        "config_file": None,
        "file": None,
        "function": None,
        "no_pr": False,
        "verbose": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestApplyLanguageConfig:
    def test_sets_module_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        config = {"module_root": str(src)}
        lang_config = LanguageConfig(config=config, config_path=tmp_path / "codeflash.toml", language=Language.JAVA)
        args = make_base_args()

        from codeflash.cli_cmds.cli import apply_language_config

        result = apply_language_config(args, lang_config)
        assert result.module_root == src.resolve()

    def test_sets_tests_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        tests = tmp_path / "src" / "test" / "java"
        tests.mkdir(parents=True)
        config = {"module_root": str(src), "tests_root": str(tests)}
        lang_config = LanguageConfig(config=config, config_path=tmp_path / "codeflash.toml", language=Language.JAVA)
        args = make_base_args()

        from codeflash.cli_cmds.cli import apply_language_config

        result = apply_language_config(args, lang_config)
        assert result.tests_root == tests.resolve()

    def test_resolves_paths_relative_to_config_parent(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        tests = tmp_path / "src" / "test" / "java"
        tests.mkdir(parents=True)
        config = {"module_root": str(src), "tests_root": str(tests)}
        lang_config = LanguageConfig(config=config, config_path=tmp_path / "codeflash.toml", language=Language.JAVA)
        args = make_base_args()

        from codeflash.cli_cmds.cli import apply_language_config

        result = apply_language_config(args, lang_config)
        assert result.module_root.is_absolute()
        assert result.tests_root.is_absolute()

    def test_sets_project_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        tests = tmp_path / "src" / "test" / "java"
        tests.mkdir(parents=True)
        (tmp_path / "pom.xml").touch()
        config = {"module_root": str(src), "tests_root": str(tests)}
        lang_config = LanguageConfig(config=config, config_path=tmp_path / "codeflash.toml", language=Language.JAVA)
        args = make_base_args()

        from codeflash.cli_cmds.cli import apply_language_config

        result = apply_language_config(args, lang_config)
        assert result.project_root == tmp_path.resolve()

    def test_preserves_cli_overrides(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        override_module = tmp_path / "custom"
        override_module.mkdir()
        tests = tmp_path / "src" / "test" / "java"
        tests.mkdir(parents=True)
        config = {"module_root": str(src), "tests_root": str(tests)}
        lang_config = LanguageConfig(config=config, config_path=tmp_path / "codeflash.toml", language=Language.JAVA)
        args = make_base_args(module_root=str(override_module))

        from codeflash.cli_cmds.cli import apply_language_config

        result = apply_language_config(args, lang_config)
        assert result.module_root == override_module.resolve()

    def test_copies_formatter_cmds(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        tests = tmp_path / "tests"
        tests.mkdir()
        config = {"module_root": str(src), "tests_root": str(tests), "formatter_cmds": ["black $file"]}
        lang_config = LanguageConfig(config=config, config_path=tmp_path / "pyproject.toml", language=Language.PYTHON)
        args = make_base_args()

        from codeflash.cli_cmds.cli import apply_language_config

        result = apply_language_config(args, lang_config)
        assert result.formatter_cmds == ["black $file"]

    def test_sets_language_singleton(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        tests = tmp_path / "src" / "test" / "java"
        tests.mkdir(parents=True)
        config = {"module_root": str(src), "tests_root": str(tests)}
        lang_config = LanguageConfig(config=config, config_path=tmp_path / "codeflash.toml", language=Language.JAVA)
        args = make_base_args()

        with patch("codeflash.cli_cmds.cli.set_current_language") as mock_set:
            from codeflash.cli_cmds.cli import apply_language_config

            apply_language_config(args, lang_config)
            mock_set.assert_called_once_with(Language.JAVA)

    def test_handles_python_config(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        tests = tmp_path / "tests"
        tests.mkdir()
        config = {"module_root": str(src), "tests_root": str(tests)}
        lang_config = LanguageConfig(config=config, config_path=tmp_path / "pyproject.toml", language=Language.PYTHON)
        args = make_base_args()

        from codeflash.cli_cmds.cli import apply_language_config

        result = apply_language_config(args, lang_config)
        assert result.module_root == src.resolve()
        assert result.tests_root == tests.resolve()

    def test_java_default_tests_root(self, tmp_path: Path, monkeypatch) -> None:
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        default_tests = tmp_path / "src" / "test" / "java"
        default_tests.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        config = {"module_root": str(src)}
        lang_config = LanguageConfig(config=config, config_path=tmp_path / "codeflash.toml", language=Language.JAVA)
        args = make_base_args()

        from codeflash.cli_cmds.cli import apply_language_config

        result = apply_language_config(args, lang_config)
        assert result.tests_root == default_tests.resolve()
