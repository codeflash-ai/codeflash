from __future__ import annotations

import configparser
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import tomlkit

from codeflash.code_utils.code_utils import custom_addopts

def test_custom_addopts_modifies_and_restores_dotini_file(tmp_path: Path) -> None:
    """Verify that custom_addopts correctly modifies and then restores a pytest.ini file."""
    # Create a dummy pytest.ini file
    config_file = tmp_path / ".pytest.ini"
    original_content = "[pytest]\naddopts = -v --cov=./src -n auto\n"
    config_file.write_text(original_content)

    # Use patch to mock get_all_closest_config_files
    os.chdir(tmp_path)
    with custom_addopts():
        # Check that the file is modified inside the context
        modified_content = config_file.read_text()
        config = configparser.ConfigParser()
        config.read_string(modified_content)
        modified_addopts = config.get("pytest", "addopts", fallback="")
        assert modified_addopts == "-v"

    # Check that the file is restored after exiting the context
    restored_content = config_file.read_text()
    assert restored_content.strip() == original_content.strip()

def test_custom_addopts_modifies_and_restores_ini_file(tmp_path: Path) -> None:
    """Verify that custom_addopts correctly modifies and then restores a pytest.ini file."""
    # Create a dummy pytest.ini file
    config_file = tmp_path / "pytest.ini"
    original_content = "[pytest]\naddopts = -v --cov=./src -n auto\n"
    config_file.write_text(original_content)

    # Use patch to mock get_all_closest_config_files
    os.chdir(tmp_path)
    with custom_addopts():
        # Check that the file is modified inside the context
        modified_content = config_file.read_text()
        config = configparser.ConfigParser()
        config.read_string(modified_content)
        modified_addopts = config.get("pytest", "addopts", fallback="")
        assert modified_addopts == "-v"

    # Check that the file is restored after exiting the context
    restored_content = config_file.read_text()
    assert restored_content.strip() == original_content.strip()


def test_custom_addopts_modifies_and_restores_toml_file(tmp_path: Path) -> None:
    """Verify that custom_addopts correctly modifies and then restores a pyproject.toml file."""
    # Create a dummy pyproject.toml file
    config_file = tmp_path / "pyproject.toml"
    os.chdir(tmp_path)
    original_addopts = "-v --cov=./src --junitxml=report.xml"
    original_content_dict = {
        "tool": {"pytest": {"ini_options": {"addopts": original_addopts}}}
    }
    original_content = tomlkit.dumps(original_content_dict)
    config_file.write_text(original_content)

    # Use patch to mock get_all_closest_config_files
    os.chdir(tmp_path)
    with custom_addopts():
        # Check that the file is modified inside the context
        modified_content = config_file.read_text()
        modified_data = tomlkit.parse(modified_content)
        modified_addopts = modified_data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
        assert modified_addopts == "-v"

    # Check that the file is restored after exiting the context
    restored_content = config_file.read_text()
    assert restored_content.strip() == original_content.strip()


def test_custom_addopts_handles_no_addopts(tmp_path: Path) -> None:
    """Ensure custom_addopts doesn't fail when a config file has no addopts."""
    # Create a dummy pytest.ini file without addopts
    config_file = tmp_path / "pytest.ini"
    original_content = "[pytest]\n# no addopts here\n"
    config_file.write_text(original_content)

    os.chdir(tmp_path)
    with custom_addopts():
        # The file should not be modified
        content_inside_context = config_file.read_text()
        assert content_inside_context == original_content

    # The file should remain unchanged
    content_after_context = config_file.read_text()
    assert content_after_context == original_content

def test_custom_addopts_handles_no_relevant_files(tmp_path: Path) -> None:
    """Ensure custom_addopts runs without error when no config files are found."""
    # No config files created in tmp_path

    os.chdir(tmp_path)
    # This should execute without raising any exceptions
    with custom_addopts():
        pass
    # No assertions needed, the test passes if no exceptions were raised


def test_custom_addopts_toml_without_pytest_section(tmp_path: Path) -> None:
    """Verify custom_addopts doesn't fail with a toml file missing a [tool.pytest] section."""
    config_file = tmp_path / "pyproject.toml"
    original_content_dict = {"tool": {"other_tool": {"key": "value"}}}
    original_content = tomlkit.dumps(original_content_dict)
    config_file.write_text(original_content)

    os.chdir(tmp_path)
    with custom_addopts():
        content_inside_context = config_file.read_text()
        assert content_inside_context == original_content

    content_after_context = config_file.read_text()
    assert content_after_context == original_content


def test_custom_addopts_ini_without_pytest_section(tmp_path: Path) -> None:
    """Verify custom_addopts doesn't fail with an ini file missing a [pytest] section."""
    config_file = tmp_path / "pytest.ini"
    original_content = "[other_section]\nkey = value\n"
    config_file.write_text(original_content)

    os.chdir(tmp_path)
    with custom_addopts():
        content_inside_context = config_file.read_text()
        assert content_inside_context == original_content

    content_after_context = config_file.read_text()
    assert content_after_context == original_content


def test_custom_addopts_with_multiple_config_files(tmp_path: Path) -> None:
    """Verify custom_addopts modifies and restores all found config files."""
    os.chdir(tmp_path)

    # Create pytest.ini
    ini_file = tmp_path / "pytest.ini"
    ini_original_content = "[pytest]\naddopts = -v --cov\n"
    ini_file.write_text(ini_original_content)

    # Create pyproject.toml
    toml_file = tmp_path / "pyproject.toml"
    toml_original_addopts = "-s -n auto"
    toml_original_content_dict = {
        "tool": {"pytest": {"ini_options": {"addopts": toml_original_addopts}}}
    }
    toml_original_content = tomlkit.dumps(toml_original_content_dict)
    toml_file.write_text(toml_original_content)

    with custom_addopts():
        # Check INI file modification
        ini_modified_content = ini_file.read_text()
        config = configparser.ConfigParser()
        config.read_string(ini_modified_content)
        assert config.get("pytest", "addopts", fallback="") == "-v"

        # Check TOML file modification
        toml_modified_content = toml_file.read_text()
        modified_data = tomlkit.parse(toml_modified_content)
        modified_addopts = modified_data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
        assert modified_addopts == "-s"

    # Check that both files are restored
    assert ini_file.read_text().strip() == ini_original_content.strip()
    assert toml_file.read_text().strip() == toml_original_content.strip()


def test_custom_addopts_restores_on_exception(tmp_path: Path) -> None:
    """Ensure config file is restored even if an exception occurs inside the context."""
    config_file = tmp_path / "pytest.ini"
    original_content = "[pytest]\naddopts = -v --cov\n"
    config_file.write_text(original_content)

    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="Test exception"):
        with custom_addopts():
            raise ValueError("Test exception")

    restored_content = config_file.read_text()
    assert restored_content.strip() == original_content.strip()
