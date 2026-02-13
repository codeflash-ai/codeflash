from __future__ import annotations

from pathlib import Path
from typing import Any

import tomlkit

from codeflash.code_utils.config_js import find_package_json, parse_package_json_config
from codeflash.lsp.helpers import is_LSP_enabled

PYPROJECT_TOML_CACHE: dict[Path, Path] = {}
ALL_CONFIG_FILES: dict[Path, dict[str, Path]] = {}


def find_pyproject_toml(config_file: Path | None = None) -> Path:
    # Find the pyproject.toml or codeflash.toml file on the root of the project

    if config_file is not None:
        config_file = Path(config_file)
        if config_file.suffix.lower() != ".toml":
            msg = f"Config file {config_file} is not a valid toml file. Please recheck the path to pyproject.toml"
            raise ValueError(msg)
        if not config_file.exists():
            msg = f"Config file {config_file} does not exist. Please recheck the path to pyproject.toml"
            raise ValueError(msg)
        return config_file
    dir_path = Path.cwd()
    cur_path = dir_path
    # see if it was encountered before in search
    if cur_path in PYPROJECT_TOML_CACHE:
        return PYPROJECT_TOML_CACHE[cur_path]
    # map current path to closest file - check both pyproject.toml and codeflash.toml
    while dir_path != dir_path.parent:
        # First check pyproject.toml (Python projects)
        config_file = dir_path / "pyproject.toml"
        if config_file.exists():
            PYPROJECT_TOML_CACHE[cur_path] = config_file
            return config_file
        # Then check codeflash.toml (Java/other projects)
        config_file = dir_path / "codeflash.toml"
        if config_file.exists():
            PYPROJECT_TOML_CACHE[cur_path] = config_file
            return config_file
        # Search in parent directories
        dir_path = dir_path.parent
    msg = f"Could not find pyproject.toml or codeflash.toml in the current directory {Path.cwd()} or any of the parent directories. Please create it by running `codeflash init`, or pass the path to the config file with the --config-file argument."

    raise ValueError(msg) from None


def get_all_closest_config_files() -> list[Path]:
    all_closest_config_files = []
    for file_type in ["pyproject.toml", "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg"]:
        closest_config_file = find_closest_config_file(file_type)
        if closest_config_file:
            all_closest_config_files.append(closest_config_file)
    return all_closest_config_files


def find_closest_config_file(file_type: str) -> Path | None:
    # Find the closest pyproject.toml, pytest.ini, tox.ini, or setup.cfg file on the root of the project
    dir_path = Path.cwd()
    cur_path = dir_path
    if cur_path in ALL_CONFIG_FILES and file_type in ALL_CONFIG_FILES[cur_path]:
        return ALL_CONFIG_FILES[cur_path][file_type]
    while dir_path != dir_path.parent:
        config_file = dir_path / file_type
        if config_file.exists():
            if cur_path not in ALL_CONFIG_FILES:
                ALL_CONFIG_FILES[cur_path] = {}
            ALL_CONFIG_FILES[cur_path][file_type] = config_file
            return config_file
        # Search for pyproject.toml in the parent directories
        dir_path = dir_path.parent
    return None


def find_conftest_files(test_paths: list[Path]) -> list[Path]:
    list_of_conftest_files = set()
    for test_path in test_paths:
        # Find the conftest file on the root of the project
        dir_path = Path.cwd()
        cur_path = test_path
        while cur_path != dir_path:
            config_file = cur_path / "conftest.py"
            if config_file.exists():
                list_of_conftest_files.add(config_file)
            # Search for conftest.py in the parent directories
            cur_path = cur_path.parent
    return list(list_of_conftest_files)


# TODO for claude: There should be different functions to parse it per language, which should be chosen during runtime
def parse_config_file(
    config_file_path: Path | None = None, override_formatter_check: bool = False
) -> tuple[dict[str, Any], Path]:
    # First try package.json for JS/TS projects
    package_json_path = find_package_json(config_file_path)
    if package_json_path:
        result = parse_package_json_config(package_json_path)
        if result is not None:
            config, path = result
            # Validate formatter if needed
            if not override_formatter_check and config.get("formatter_cmds"):
                formatter_cmds = config.get("formatter_cmds", [])
                if formatter_cmds and formatter_cmds[0] == "your-formatter $file":
                    raise ValueError(
                        "The formatter command is not set correctly in package.json. Please set the "
                        "formatter command in the 'formatterCmds' key."
                    )
            return config, path

    # Fall back to pyproject.toml
    config_file_path = find_pyproject_toml(config_file_path)
    try:
        with config_file_path.open("rb") as f:
            data = tomlkit.parse(f.read())
    except tomlkit.exceptions.ParseError as e:
        msg = f"Error while parsing the config file {config_file_path}. Please recheck the file for syntax errors. Error: {e}"
        raise ValueError(msg) from None

    lsp_mode = is_LSP_enabled()

    try:
        tool = data["tool"]
        assert isinstance(tool, dict)
        config = tool["codeflash"]
    except tomlkit.exceptions.NonExistentKey as e:
        if lsp_mode:
            # don't fail in lsp mode if codeflash config is not found.
            return {}, config_file_path
        msg = f"Could not find the 'codeflash' block in the config file {config_file_path}. Please run 'codeflash init' to add Codeflash config."
        raise ValueError(msg) from e
    assert isinstance(config, dict)

    if config == {} and lsp_mode:
        return {}, config_file_path

    # Preserve language field if present (important for Java/JS projects using codeflash.toml)
    # default values:
    path_keys = ["module-root", "tests-root", "benchmarks-root"]
    path_list_keys = ["ignore-paths"]
    str_keys = {"pytest-cmd": "pytest", "git-remote": "origin"}
    bool_keys = {
        "override-fixtures": False,
        "disable-telemetry": False,
        "disable-imports-sorting": False,
        "benchmark": False,
    }
    # Note: formatter-cmds defaults to empty list. For Python projects, black is typically
    # detected by the project detector. For Java projects, no formatter is supported yet.
    list_str_keys = {"formatter-cmds": []}

    for key, default_value in str_keys.items():
        if key in config:
            config[key] = str(config[key])
        else:
            config[key] = default_value
    for key, default_value in bool_keys.items():
        if key in config:
            config[key] = bool(config[key])
        else:
            config[key] = default_value
    for key in path_keys:
        if key in config:
            config[key] = str((Path(config_file_path).parent / Path(config[key])).resolve())
    for key, default_value in list_str_keys.items():
        if key in config:
            config[key] = [str(cmd) for cmd in config[key]]
        else:
            config[key] = default_value

    for key in path_list_keys:
        if key in config:
            config[key] = [str((Path(config_file_path).parent / path).resolve()) for path in config[key]]
        else:
            config[key] = []

    # see if this is happening during GitHub actions setup
    if config.get("formatter-cmds") and len(config.get("formatter-cmds")) > 0 and not override_formatter_check:
        assert config.get("formatter-cmds")[0] != "your-formatter $file", (
            "The formatter command is not set correctly in pyproject.toml. Please set the "
            "formatter command in the 'formatter-cmds' key. More info - https://docs.codeflash.ai/configuration"
        )
    for key in list(config.keys()):
        if "-" in key:
            config[key.replace("-", "_")] = config[key]
            del config[key]

    return config, config_file_path
