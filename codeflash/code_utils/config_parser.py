from __future__ import annotations

from pathlib import Path
from typing import Any

import tomlkit


def find_pyproject_toml(config_file: Path | None = None) -> Path:
    # Find the pyproject.toml file on the root of the project

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

    while dir_path != dir_path.parent:
        config_file = dir_path / "pyproject.toml"
        if config_file.exists():
            return config_file
        # Search for pyproject.toml in the parent directories
        dir_path = dir_path.parent
    msg = f"Could not find pyproject.toml in the current directory {Path.cwd()} or any of the parent directories. Please create it by running `codeflash init`, or pass the path to pyproject.toml with the --config-file argument."

    raise ValueError(msg)


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


def parse_config_file(
    config_file_path: Path | None = None,
    override_formatter_check: bool = False,  # noqa: FBT001, FBT002
) -> tuple[dict[str, Any], Path]:
    config_file_path = find_pyproject_toml(config_file_path)
    try:
        with config_file_path.open("rb") as f:
            data = tomlkit.parse(f.read())
    except tomlkit.exceptions.ParseError as e:
        msg = f"Error while parsing the config file {config_file_path}. Please recheck the file for syntax errors. Error: {e}"
        raise ValueError(msg) from e

    try:
        tool = data["tool"]
        assert isinstance(tool, dict)
        config = tool["codeflash"]
    except tomlkit.exceptions.NonExistentKey as e:
        msg = f"Could not find the 'codeflash' block in the config file {config_file_path}. Please run 'codeflash init' to create the config file."
        raise ValueError(msg) from e
    assert isinstance(config, dict)

    # default values:
    path_keys = ["module-root", "tests-root", "benchmarks-root"]
    path_list_keys = ["ignore-paths"]
    str_keys = {"pytest-cmd": "pytest", "git-remote": "origin", "exp-git-remote": "exp-origin"}
    bool_keys = {
        "override-fixtures": False,
        "disable-telemetry": False,
        "disable-imports-sorting": False,
        "benchmark": False,
    }
    list_str_keys = {"formatter-cmds": ["black $file"]}

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

    assert config["test-framework"] in {"pytest", "unittest"}, (
        "In pyproject.toml, Codeflash only supports the 'test-framework' as pytest and unittest."
    )
    # see if this is happening during GitHub actions setup
    if len(config["formatter-cmds"]) > 0 and not override_formatter_check:
        assert config["formatter-cmds"][0] != "your-formatter $file", (
            "The formatter command is not set correctly in pyproject.toml. Please set the "
            "formatter command in the 'formatter-cmds' key. More info - https://docs.codeflash.ai/configuration"
        )
    for key in list(config.keys()):
        if "-" in key:
            config[key.replace("-", "_")] = config[key]
            del config[key]

    return config, config_file_path
