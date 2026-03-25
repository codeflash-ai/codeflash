from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import tomlkit

from codeflash.code_utils.config_parser import LanguageConfig, normalize_toml_config
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
        "command": None,
        "verify_setup": False,
        "version": False,
        "show_config": False,
        "reset_config": False,
        "previous_checkpoint_functions": [],
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestApplyLanguageConfig:
    def test_sets_module_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "main" / "java"
        src.mkdir(parents=True)
        config = {"module_root": str(src)}
        lang_config = LanguageConfig(config=config, config_path=tmp_path, language=Language.JAVA)
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
        lang_config = LanguageConfig(config=config, config_path=tmp_path, language=Language.JAVA)
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
        lang_config = LanguageConfig(config=config, config_path=tmp_path, language=Language.JAVA)
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
        lang_config = LanguageConfig(config=config, config_path=tmp_path, language=Language.JAVA)
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
        lang_config = LanguageConfig(config=config, config_path=tmp_path, language=Language.JAVA)
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
        lang_config = LanguageConfig(config=config, config_path=tmp_path, language=Language.JAVA)
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
        lang_config = LanguageConfig(config=config, config_path=tmp_path, language=Language.JAVA)
        args = make_base_args()

        from codeflash.cli_cmds.cli import apply_language_config

        result = apply_language_config(args, lang_config)
        assert result.tests_root == default_tests.resolve()


def make_lang_config(tmp_path: Path, language: Language, subdir: str = "") -> LanguageConfig:
    if language == Language.PYTHON:
        src = tmp_path / subdir / "src" if subdir else tmp_path / "src"
        tests = tmp_path / subdir / "tests" if subdir else tmp_path / "tests"
        src.mkdir(parents=True, exist_ok=True)
        tests.mkdir(parents=True, exist_ok=True)
        config_path = tmp_path / subdir / "pyproject.toml" if subdir else tmp_path / "pyproject.toml"
        return LanguageConfig(
            config={"module_root": str(src), "tests_root": str(tests)},
            config_path=config_path,
            language=Language.PYTHON,
        )
    if language == Language.JAVASCRIPT:
        src = tmp_path / subdir / "src" if subdir else tmp_path / "src"
        tests = tmp_path / subdir / "tests" if subdir else tmp_path / "tests"
        src.mkdir(parents=True, exist_ok=True)
        tests.mkdir(parents=True, exist_ok=True)
        config_path = tmp_path / subdir / "package.json" if subdir else tmp_path / "package.json"
        return LanguageConfig(
            config={"module_root": str(src), "tests_root": str(tests)},
            config_path=config_path,
            language=Language.JAVASCRIPT,
        )
    src = tmp_path / subdir / "src" / "main" / "java" if subdir else tmp_path / "src" / "main" / "java"
    tests = tmp_path / subdir / "src" / "test" / "java" if subdir else tmp_path / "src" / "test" / "java"
    src.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / subdir if subdir else tmp_path
    return LanguageConfig(
        config={"module_root": str(src), "tests_root": str(tests)},
        config_path=config_path,
        language=Language.JAVA,
    )


class TestMultiLanguageOrchestration:
    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_sequential_passes_calls_optimizer_per_language(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(disable_telemetry=False)

        from codeflash.main import main

        main()

        assert mock_run.call_count == 2

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    @patch("codeflash.cli_cmds.cli.set_current_language")
    def test_singleton_set_per_pass(
        self,
        mock_set_lang,
        _sentry,
        _posthog,
        _ver,
        _banner,
        mock_parse_args,
        mock_find_configs,
        mock_run,
        _handle_all,
        _fmt,
        _ckpt,
        tmp_path: Path,
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(disable_telemetry=False)

        from codeflash.main import main

        main()

        # set_current_language is called once per language pass via apply_language_config
        lang_calls = [c for c in mock_set_lang.call_args_list if c[0][0] in (Language.PYTHON, Language.JAVA)]
        assert len(lang_calls) >= 2
        called_langs = {c[0][0] for c in lang_calls}
        assert Language.PYTHON in called_langs
        assert Language.JAVA in called_langs

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files", return_value=[])
    @patch("codeflash.main._handle_config_loading")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    @patch("codeflash.main.get_changed_file_paths", return_value=[])
    def test_fallback_to_single_config_when_no_multi_configs(
        self, _changed, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_handle_config, mock_run, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        base = make_base_args(
            disable_telemetry=False, formatter_cmds=[], module_root=str(tmp_path), tests_root=str(tmp_path)
        )
        mock_parse_args.return_value = base
        mock_handle_config.return_value = base

        from codeflash.main import main

        main()

        mock_handle_config.assert_called_once()
        mock_run.assert_called_once()

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_args_deep_copied_between_passes(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(disable_telemetry=False)

        from codeflash.main import main

        main()

        assert mock_run.call_count == 2
        call1_args = mock_run.call_args_list[0][0][0]
        call2_args = mock_run.call_args_list[1][0][0]
        # Args should be different objects (deep copied)
        assert call1_args is not call2_args
        # Module roots should differ between Python and Java configs
        assert call1_args.module_root != call2_args.module_root


    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_error_in_one_language_does_not_block_others(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(disable_telemetry=False)
        # First call (Python) raises, second call (Java) succeeds
        mock_run.side_effect = [RuntimeError("Python optimizer crashed"), None]

        from codeflash.main import main

        main()

        assert mock_run.call_count == 2

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_orchestration_summary_logged(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(disable_telemetry=False)

        with patch("codeflash.main._log_orchestration_summary") as mock_summary:
            from codeflash.main import main

            main()

            mock_summary.assert_called_once()
            results = mock_summary.call_args[0][1]
            assert results["python"] == "success"
            assert results["java"] == "success"

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_summary_reports_failure_status(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(disable_telemetry=False)
        mock_run.side_effect = [RuntimeError("boom"), None]

        with patch("codeflash.main._log_orchestration_summary") as mock_summary:
            from codeflash.main import main

            main()

            results = mock_summary.call_args[0][1]
            assert results["python"] == "failed"
            assert results["java"] == "success"


class TestOrchestrationSummaryLogging:
    def test_summary_format_all_success(self) -> None:
        import logging

        from codeflash.main import _log_orchestration_summary

        with patch.object(logging.Logger, "info") as mock_info:
            logger = logging.getLogger("codeflash.test")
            _log_orchestration_summary(logger, {"python": "success", "java": "success"})
            mock_info.assert_called_once()
            msg = mock_info.call_args[0][0] % mock_info.call_args[0][1:]
            assert "python: success" in msg
            assert "java: success" in msg

    def test_summary_format_mixed_statuses(self) -> None:
        import logging

        from codeflash.main import _log_orchestration_summary

        with patch.object(logging.Logger, "info") as mock_info:
            logger = logging.getLogger("codeflash.test")
            _log_orchestration_summary(logger, {"python": "failed", "java": "success", "javascript": "skipped"})
            mock_info.assert_called_once()
            msg = mock_info.call_args[0][0] % mock_info.call_args[0][1:]
            assert "python: failed" in msg
            assert "java: success" in msg
            assert "javascript: skipped" in msg

    def test_summary_no_results_no_log(self) -> None:
        import logging

        from codeflash.main import _log_orchestration_summary

        with patch.object(logging.Logger, "info") as mock_info:
            logger = logging.getLogger("codeflash.test")
            _log_orchestration_summary(logger, {})
            mock_info.assert_not_called()

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed")
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_summary_reports_skipped_status(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, mock_fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(disable_telemetry=False)
        # Python formatter check fails (skipped), Java succeeds
        mock_fmt.side_effect = [False, True]

        with patch("codeflash.main._log_orchestration_summary") as mock_summary:
            from codeflash.main import main

            main()

            results = mock_summary.call_args[0][1]
            assert results["python"] == "skipped"
            assert results["java"] == "success"
            assert mock_run.call_count == 1


class TestCLIPathRouting:
    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_file_flag_filters_to_matching_language(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(file="path/to/Foo.java", disable_telemetry=False)

        from codeflash.main import main

        main()

        assert mock_run.call_count == 1

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_file_flag_python_file_filters_to_python(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(file="module.py", disable_telemetry=False)

        from codeflash.main import main

        main()

        assert mock_run.call_count == 1

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_file_flag_unknown_extension_runs_all(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(file="Foo.rs", disable_telemetry=False)

        from codeflash.main import main

        main()

        assert mock_run.call_count == 2

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_file_flag_no_matching_config_runs_all(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        mock_find_configs.return_value = [py_config]
        mock_parse_args.return_value = make_base_args(file="Foo.java", disable_telemetry=False)

        from codeflash.main import main

        main()

        assert mock_run.call_count == 1

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_all_flag_sets_module_root_per_language(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(all="", disable_telemetry=False)

        from codeflash.main import main

        main()

        assert mock_run.call_count == 2
        for call in mock_run.call_args_list:
            passed_args = call[0][0]
            assert passed_args.all == passed_args.module_root

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_no_flags_runs_all_language_passes(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        java_config = make_lang_config(tmp_path, Language.JAVA)
        mock_find_configs.return_value = [py_config, java_config]
        mock_parse_args.return_value = make_base_args(disable_telemetry=False)

        from codeflash.main import main

        main()

        assert mock_run.call_count == 2

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_file_flag_typescript_extension(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        # .tsx maps to Language.TYPESCRIPT, which is distinct from Language.JAVASCRIPT.
        # When no TYPESCRIPT config exists, all configs run (fallback behavior).
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        js_config = make_lang_config(tmp_path, Language.JAVASCRIPT, subdir="js-proj")
        mock_find_configs.return_value = [py_config, js_config]
        mock_parse_args.return_value = make_base_args(file="path/to/Component.tsx", disable_telemetry=False)

        from codeflash.main import main

        main()

        # No TYPESCRIPT config exists, so all configs run (same as unknown extension)
        assert mock_run.call_count == 2

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_file_flag_jsx_extension(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        # .jsx maps to Language.JAVASCRIPT, so it correctly filters to the JS config.
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        js_config = make_lang_config(tmp_path, Language.JAVASCRIPT, subdir="js-proj")
        mock_find_configs.return_value = [py_config, js_config]
        mock_parse_args.return_value = make_base_args(file="path/to/Widget.jsx", disable_telemetry=False)

        from codeflash.main import main

        main()

        assert mock_run.call_count == 1


class TestDirectFunctionCoverage:
    @patch("subprocess.run")
    def test_get_changed_file_paths_returns_diff_files(self, mock_subprocess) -> None:
        from codeflash.main import get_changed_file_paths

        mock_subprocess.return_value = MagicMock(returncode=0, stdout="src/main.py\nsrc/App.java\n")
        result = get_changed_file_paths()
        assert len(result) == 2
        assert Path("src/main.py") in result
        assert Path("src/App.java") in result

    @patch("subprocess.run")
    def test_get_changed_file_paths_returns_empty_on_failure(self, mock_subprocess) -> None:
        from codeflash.main import get_changed_file_paths

        mock_subprocess.return_value = MagicMock(returncode=1, stdout="")
        result = get_changed_file_paths()
        assert result == []

    def test_detect_project_for_language_java(self, tmp_path: Path) -> None:
        from codeflash.main import detect_project_for_language

        with (
            patch(
                "codeflash.setup.detector._detect_java_module_root",
                return_value=(tmp_path / "src/main/java", "pom.xml"),
            ),
            patch(
                "codeflash.setup.detector._detect_tests_root",
                return_value=(tmp_path / "src/test/java", "maven"),
            ),
            patch("codeflash.setup.detector._detect_test_runner", return_value=("maven", "pom.xml")),
            patch("codeflash.setup.detector._detect_formatter", return_value=([], None)),
            patch("codeflash.setup.detector._detect_ignore_paths", return_value=([], None)),
        ):
            result = detect_project_for_language(Language.JAVA, tmp_path)
            assert result is not None
            assert result.language == "java"

    def test_detect_project_for_language_unsupported(self) -> None:
        from codeflash.main import detect_project_for_language

        mock_lang = MagicMock()
        mock_lang.value = "rust"
        try:
            detect_project_for_language(mock_lang, Path("/tmp"))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No auto-detection available" in str(e)

    def test_empty_config_no_module_root(self, tmp_path: Path) -> None:
        config: dict = {}
        result = normalize_toml_config(config, tmp_path / "pyproject.toml")
        assert result["formatter_cmds"] == []
        assert result["disable_telemetry"] is False
        assert "module_root" not in result


class TestNormalizeTomlConfig:
    def test_converts_hyphenated_keys_to_underscored(self, tmp_path: Path) -> None:
        config = {"module-root": "src", "tests-root": "tests"}
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        result = normalize_toml_config(config, tmp_path / "pyproject.toml")
        assert "module_root" in result
        assert "tests_root" in result
        assert "module-root" not in result
        assert "tests-root" not in result

    def test_resolves_paths_relative_to_config_parent(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        config = {"module-root": "src"}
        result = normalize_toml_config(config, tmp_path / "pyproject.toml")
        assert result["module_root"] == str(src.resolve())

    def test_applies_default_values(self, tmp_path: Path) -> None:
        config: dict = {}
        result = normalize_toml_config(config, tmp_path / "pyproject.toml")
        assert result["formatter_cmds"] == []
        assert result["disable_telemetry"] is False
        assert result["override_fixtures"] is False
        assert result["git_remote"] == "origin"
        assert result["pytest_cmd"] == "pytest"

    def test_preserves_explicit_values(self, tmp_path: Path) -> None:
        config = {"disable-telemetry": True, "formatter-cmds": ["prettier $file"]}
        result = normalize_toml_config(config, tmp_path / "pyproject.toml")
        assert result["disable_telemetry"] is True
        assert result["formatter_cmds"] == ["prettier $file"]

    def test_resolves_ignore_paths(self, tmp_path: Path) -> None:
        config = {"ignore-paths": ["build", "dist"]}
        result = normalize_toml_config(config, tmp_path / "pyproject.toml")
        assert result["ignore_paths"] == [
            str((tmp_path / "build").resolve()),
            str((tmp_path / "dist").resolve()),
        ]

    def test_empty_ignore_paths_default(self, tmp_path: Path) -> None:
        config: dict = {}
        result = normalize_toml_config(config, tmp_path / "pyproject.toml")
        assert result["ignore_paths"] == []


class TestUnconfiguredLanguageDetection:
    def test_detects_unconfigured_java_from_changed_files(self) -> None:
        from codeflash.main import detect_unconfigured_languages

        configs = [LanguageConfig(config={}, config_path=Path("pyproject.toml"), language=Language.PYTHON)]
        changed = [Path("src/main/java/Foo.java"), Path("src/Bar.py")]
        result = detect_unconfigured_languages(configs, changed)
        assert Language.JAVA in result
        assert Language.PYTHON not in result

    def test_no_unconfigured_when_all_configured(self) -> None:
        from codeflash.main import detect_unconfigured_languages

        configs = [
            LanguageConfig(config={}, config_path=Path("pyproject.toml"), language=Language.PYTHON),
            LanguageConfig(config={}, config_path=Path(), language=Language.JAVA),
        ]
        changed = [Path("Foo.java"), Path("bar.py")]
        result = detect_unconfigured_languages(configs, changed)
        assert result == set()

    def test_ignores_unsupported_extensions(self) -> None:
        from codeflash.main import detect_unconfigured_languages

        changed = [Path("main.rs"), Path("lib.go")]
        result = detect_unconfigured_languages([], changed)
        assert result == set()

    @patch("codeflash.main.find_all_config_files")
    def test_auto_config_adds_language_config_on_success(self, mock_find_configs, tmp_path: Path) -> None:
        from codeflash.main import auto_configure_language

        new_lc = LanguageConfig(config={}, config_path=tmp_path, language=Language.JAVA)
        mock_find_configs.return_value = [new_lc]

        logger = logging.getLogger("codeflash.test")
        with (
            patch("codeflash.main.write_config", return_value=(True, "Created config")) as mock_write,
            patch("codeflash.main.detect_project_for_language") as mock_detect,
        ):
            mock_detect.return_value = MagicMock()
            result = auto_configure_language(Language.JAVA, tmp_path, logger)

        assert result is not None
        assert result.language == Language.JAVA
        mock_write.assert_called_once()

    def test_auto_config_failure_logs_warning(self, tmp_path: Path, caplog: object) -> None:
        from codeflash.main import auto_configure_language

        logger = logging.getLogger("codeflash.test")
        with (
            patch("codeflash.main.detect_project_for_language", side_effect=RuntimeError("detection failed")),
            caplog.at_level(logging.WARNING),  # type: ignore[union-attr]
        ):
            result = auto_configure_language(Language.JAVA, tmp_path, logger)

        assert result is None

    @patch("codeflash.main.ask_should_use_checkpoint_get_functions", return_value=[])
    @patch("codeflash.main.env_utils.check_formatter_installed", return_value=True)
    @patch("codeflash.main.handle_optimize_all_arg_parsing", side_effect=lambda args: args)
    @patch("codeflash.optimization.optimizer.run_with_args")
    @patch("codeflash.main.find_all_config_files")
    @patch("codeflash.main.parse_args")
    @patch("codeflash.main.print_codeflash_banner")
    @patch("codeflash.main.check_for_newer_minor_version")
    @patch("codeflash.telemetry.posthog_cf.initialize_posthog")
    @patch("codeflash.telemetry.sentry.init_sentry")
    def test_per_language_logging_shows_config_path(
        self, _sentry, _posthog, _ver, _banner, mock_parse_args, mock_find_configs, mock_run, _handle_all, _fmt, _ckpt, tmp_path: Path
    ) -> None:
        py_config = make_lang_config(tmp_path, Language.PYTHON)
        mock_find_configs.return_value = [py_config]
        mock_parse_args.return_value = make_base_args(disable_telemetry=False)

        with patch("codeflash.main._log_orchestration_summary"):
            from codeflash.main import main

            with patch("logging.Logger.info") as mock_log_info:
                main()
                logged_messages = [str(call) for call in mock_log_info.call_args_list]
                processing_logs = [m for m in logged_messages if "Processing" in m and "config:" in m]
                assert len(processing_logs) >= 1
