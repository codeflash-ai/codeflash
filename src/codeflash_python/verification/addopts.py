from __future__ import annotations

import configparser
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import tomlkit

if TYPE_CHECKING:
    from collections.abc import Generator

from codeflash_python.code_utils.config_parser import get_all_closest_config_files

logger = logging.getLogger("codeflash_python")

BLACKLIST_ADDOPTS = ("--benchmark", "--sugar", "--codespeed", "--cov", "--profile", "--junitxml", "-n")


def filter_args(addopts_args: list[str]) -> list[str]:
    # Convert BLACKLIST_ADDOPTS to a set for faster lookup of simple matches
    # But keep tuple for startswith
    blacklist = BLACKLIST_ADDOPTS
    # Precompute the length for re-use
    n = len(addopts_args)
    filtered_args = []
    i = 0
    while i < n:
        current_arg = addopts_args[i]
        if current_arg.startswith(blacklist):
            i += 1
            if i < n and not addopts_args[i].startswith("-"):
                i += 1
        else:
            filtered_args.append(current_arg)
            i += 1
    return filtered_args


def modify_addopts(config_file: Path) -> tuple[str, bool]:
    file_type = config_file.suffix.lower()
    filename = config_file.name
    config = None
    if file_type not in {".toml", ".ini", ".cfg"} or not config_file.exists():
        return "", False
    # Read original file
    with Path.open(config_file, encoding="utf-8") as f:
        content = f.read()
    try:
        if filename == "pyproject.toml":
            # use tomlkit
            data = tomlkit.parse(content)
            original_addopts = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
            # nothing to do if no addopts present
            if original_addopts == "":
                return content, False
            if isinstance(original_addopts, list):
                original_addopts = " ".join(original_addopts)
            original_addopts = original_addopts.replace("=", " ")
            addopts_args = (
                original_addopts.split()
            )  # any number of space characters as delimiter, doesn't look at = which is fine
        else:
            # use configparser
            config = configparser.ConfigParser()
            config.read_string(content)
            data = {section: dict(config[section]) for section in config.sections()}
            if config_file.name in {"pytest.ini", ".pytest.ini", "tox.ini"}:
                original_addopts = data.get("pytest", {}).get("addopts", "")  # should only be a string
            else:
                original_addopts = data.get("tool:pytest", {}).get("addopts", "")  # should only be a string
            original_addopts = original_addopts.replace("=", " ")
            addopts_args = original_addopts.split()
        new_addopts_args = filter_args(addopts_args)
        if new_addopts_args == addopts_args:
            return content, False
        # change addopts now
        if file_type == ".toml":
            data["tool"]["pytest"]["ini_options"]["addopts"] = " ".join(new_addopts_args)  # type: ignore[index,call-overload]
            # Write modified file
            with Path.open(config_file, "w", encoding="utf-8") as f:
                f.write(tomlkit.dumps(data))
                return content, True
        elif config_file.name in {"pytest.ini", ".pytest.ini", "tox.ini"}:
            assert config is not None
            config.set("pytest", "addopts", " ".join(new_addopts_args))
            # Write modified file
            with Path.open(config_file, "w", encoding="utf-8") as f:
                config.write(f)
                return content, True
        else:
            assert config is not None
            config.set("tool:pytest", "addopts", " ".join(new_addopts_args))
            # Write modified file
            with Path.open(config_file, "w", encoding="utf-8") as f:
                config.write(f)
                return content, True

    except Exception:
        logger.debug("Trouble parsing")
        return content, False  # not modified


@contextmanager
def custom_addopts() -> Generator[None, None, None]:
    closest_config_files = get_all_closest_config_files()

    original_content = {}

    try:
        for config_file in closest_config_files:
            original_content[config_file] = modify_addopts(config_file)
        yield

    finally:
        # Restore original file
        for file, (content, was_modified) in original_content.items():
            if was_modified:
                with Path.open(file, "w", encoding="utf-8") as f:
                    f.write(content)
