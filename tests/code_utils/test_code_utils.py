from __future__ import annotations

import configparser
import os
from pathlib import Path
from unittest.mock import patch

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
