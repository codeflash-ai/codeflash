"""Tests for the first-run experience."""

import json
import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.setup.first_run import (
    _handle_api_key,
    _prompt_confirmation,
    _show_detected_settings,
    _show_welcome,
    handle_first_run,
    is_first_run,
)


class TestIsFirstRun:
    """Tests for is_first_run function."""

    def test_returns_true_when_no_config(self, tmp_path):
        """Should return True when no codeflash config exists."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')

        result = is_first_run(tmp_path)
        assert result is True

    def test_returns_false_when_pyproject_config_exists(self, tmp_path):
        """Should return False when codeflash config exists in pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text(
            '[tool.codeflash]\nmodule-root = "src"'
        )

        result = is_first_run(tmp_path)
        assert result is False

    def test_returns_false_when_package_json_config_exists(self, tmp_path):
        """Should return False when codeflash config exists in package.json."""
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "test",
            "codeflash": {"moduleRoot": "src"}
        }))

        result = is_first_run(tmp_path)
        assert result is False

    def test_returns_true_for_empty_directory(self, tmp_path):
        """Should return True for empty directory."""
        result = is_first_run(tmp_path)
        assert result is True


class TestHandleFirstRun:
    """Tests for handle_first_run function."""

    def test_returns_args_on_success(self, tmp_path, monkeypatch):
        """Should return updated args on successful first run."""
        # Create Python project
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "__init__.py").write_text("")
        (tmp_path / "tests").mkdir()

        # Mock user input and API key
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key")

        # Skip confirmation
        result = handle_first_run(skip_confirm=True, skip_api_key=True)

        assert result is not None
        assert hasattr(result, "module_root")
        assert hasattr(result, "language")
        assert result.language == "python"

    def test_returns_none_when_user_cancels(self, tmp_path, monkeypatch):
        """Should return None when user cancels."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
        monkeypatch.chdir(tmp_path)

        # Mock user cancellation
        with patch("codeflash.setup.first_run._prompt_confirmation", return_value="n"):
            result = handle_first_run(skip_api_key=True)

        assert result is None

    def test_skips_confirm_with_flag(self, tmp_path, monkeypatch):
        """Should skip confirmation when skip_confirm=True."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "__init__.py").write_text("")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key")

        # Should not prompt for confirmation
        with patch("codeflash.setup.first_run._prompt_confirmation") as mock_prompt:
            result = handle_first_run(skip_confirm=True, skip_api_key=True)

        mock_prompt.assert_not_called()
        assert result is not None

    def test_merges_with_existing_args(self, tmp_path, monkeypatch):
        """Should merge detected settings with existing args."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "__init__.py").write_text("")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key")

        existing_args = Namespace(custom_flag=True, module_root=None)

        result = handle_first_run(
            args=existing_args,
            skip_confirm=True,
            skip_api_key=True,
        )

        assert result is not None
        assert result.custom_flag is True  # Preserved
        assert result.module_root is not None  # Updated


class TestPromptConfirmation:
    """Tests for _prompt_confirmation function."""

    def test_returns_y_for_yes(self, monkeypatch):
        """Should return 'y' for yes input."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        with patch("codeflash.cli_cmds.console.console.input", return_value="y"):
            result = _prompt_confirmation()

        assert result == "y"

    def test_returns_y_for_empty(self, monkeypatch):
        """Should return 'y' for empty input (default)."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        with patch("codeflash.cli_cmds.console.console.input", return_value=""):
            result = _prompt_confirmation()

        assert result == "y"

    def test_returns_n_for_no(self, monkeypatch):
        """Should return 'n' for no input."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        with patch("codeflash.cli_cmds.console.console.input", return_value="n"):
            result = _prompt_confirmation()

        assert result == "n"

    def test_returns_customize_for_c(self, monkeypatch):
        """Should return 'customize' for c input."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        with patch("codeflash.cli_cmds.console.console.input", return_value="c"):
            result = _prompt_confirmation()

        assert result == "customize"

    def test_returns_n_for_non_interactive(self, monkeypatch):
        """Should return 'n' for non-interactive environment."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        result = _prompt_confirmation()

        assert result == "n"


class TestHandleApiKey:
    """Tests for _handle_api_key function."""

    def test_returns_true_when_key_exists(self, monkeypatch):
        """Should return True when API key already exists."""
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-existing-key")

        with patch("codeflash.code_utils.env_utils.get_codeflash_api_key", return_value="cf-existing-key"):
            result = _handle_api_key()

        assert result is True

    def test_accepts_valid_key(self, monkeypatch):
        """Should accept valid API key starting with cf-."""
        monkeypatch.delenv("CODEFLASH_API_KEY", raising=False)

        with patch("codeflash.code_utils.env_utils.get_codeflash_api_key", side_effect=OSError):
            with patch("codeflash.cli_cmds.console.console.input", return_value="cf-valid-key"):
                # Patch the entire import to fail, triggering the exception handler
                with patch.dict("sys.modules", {"codeflash.code_utils.shell_utils": None}):
                    result = _handle_api_key()

        assert result is True
        assert os.environ.get("CODEFLASH_API_KEY") == "cf-valid-key"

    def test_rejects_invalid_key(self, monkeypatch):
        """Should reject API key not starting with cf-."""
        monkeypatch.delenv("CODEFLASH_API_KEY", raising=False)

        with patch("codeflash.code_utils.env_utils.get_codeflash_api_key", side_effect=OSError):
            with patch("codeflash.cli_cmds.console.console.input", return_value="invalid-key"):
                result = _handle_api_key()

        assert result is False


class TestShowFunctions:
    """Tests for display functions (smoke tests)."""

    def test_show_welcome_does_not_crash(self):
        """Should not crash when showing welcome message."""
        # Just verify it doesn't raise an exception
        _show_welcome()

    def test_show_detected_settings_does_not_crash(self, tmp_path):
        """Should not crash when showing detected settings."""
        from codeflash.setup.detector import detect_project

        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
        detected = detect_project(tmp_path)

        # Just verify it doesn't raise an exception
        _show_detected_settings(detected)


class TestFirstRunIntegration:
    """Integration tests for the complete first-run flow."""

    def test_full_python_first_run(self, tmp_path, monkeypatch):
        """Should complete full first-run for Python project."""
        # Create Python project
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "myapp"\n\n[tool.ruff]\nline-length = 120'
        )
        pkg_dir = tmp_path / "myapp"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (tmp_path / "tests").mkdir()

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key")

        result = handle_first_run(skip_confirm=True, skip_api_key=True)

        assert result is not None
        assert result.language == "python"
        assert "myapp" in result.module_root
        assert "tests" in result.tests_root

        # Verify config was written
        import tomlkit

        content = (tmp_path / "pyproject.toml").read_text()
        data = tomlkit.parse(content)
        assert "codeflash" in data.get("tool", {})

    def test_full_javascript_first_run(self, tmp_path, monkeypatch):
        """Should complete full first-run for JavaScript project."""
        # Create JS project
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "myapp",
            "devDependencies": {"jest": "^29.0.0"}
        }, indent=2))
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key")

        result = handle_first_run(skip_confirm=True, skip_api_key=True)

        assert result is not None
        assert result.language == "javascript"
        assert "src" in result.module_root
        assert result.pytest_cmd == "jest"  # test_runner mapped to pytest_cmd

    def test_subsequent_run_uses_saved_config(self, tmp_path, monkeypatch):
        """After first run, subsequent runs should not trigger first-run."""
        # Create project with existing config
        (tmp_path / "pyproject.toml").write_text(
            '[tool.codeflash]\nmodule-root = "src"'
        )

        monkeypatch.chdir(tmp_path)

        # Should not be first run
        assert is_first_run(tmp_path) is False