import os
from typing import Any

import tomlkit


def find_pyproject_toml(config_file=None):
    # Find the pyproject.toml file on the root of the project

    if config_file is not None:
        if not config_file.lower().endswith(".toml"):
            raise ValueError(
                f"Config file {config_file} is not a valid toml file. Please recheck the path to pyproject.toml",
            )
        if not os.path.exists(config_file):
            raise ValueError(
                f"Config file {config_file} does not exist. Please recheck the path to pyproject.toml",
            )
        return config_file

    else:
        dir_path = os.getcwd()

        while os.path.dirname(dir_path) != dir_path:
            config_file = os.path.join(dir_path, "pyproject.toml")
            if os.path.exists(config_file):
                return config_file
            # Search for pyproject.toml in the parent directories
            dir_path = os.path.dirname(dir_path)
        raise ValueError(
            f"Could not find pyproject.toml in the current directory {os.getcwd()} or any of the parent directories. Please create it by running `poetry init`, or pass the path to pyproject.toml with the --config-file argument.",
        )


def parse_config_file(config_file_path: str = None) -> tuple[dict[str, Any], str]:
    config_file_path = find_pyproject_toml(config_file_path)
    try:
        with open(config_file_path, "rb") as f:
            data = tomlkit.parse(f.read())
    except tomlkit.exceptions.ParseError as e:
        raise ValueError(
            f"Error while parsing the config file {config_file_path}. Please recheck the file for syntax errors. Error: {e}",
        )

    try:
        tool = data["tool"]
        assert isinstance(tool, dict)
        config = tool["codeflash"]
    except tomlkit.exceptions.NonExistentKey:
        raise ValueError(
            f"Could not find the 'codeflash' block in the config file {config_file_path}. "
            f"Please run 'codeflash init' to create the config file.",
        )
    assert isinstance(config, dict)

    # default values:
    path_keys = ["module-root", "tests-root"]
    path_list_keys = ["ignore-paths"]
    str_keys = {
        "pytest-cmd": "pytest",
    }
    bool_keys = {
        "disable-telemetry": False,
        "disable-imports-sorting": False,
    }
    list_str_keys = {
        "formatter-cmds": ["black $file"],
    }

    for key in str_keys:
        if key in config:
            config[key] = str(config[key])
        else:
            config[key] = str_keys[key]
    for key in bool_keys:
        if key in config:
            config[key] = bool(config[key])
        else:
            config[key] = bool_keys[key]
    for key in path_keys:
        if key in config:
            config[key] = os.path.realpath(
                os.path.join(os.path.dirname(config_file_path), config[key]),
            )
    for key in list_str_keys:
        if key in config:
            config[key] = [str(cmd) for cmd in config[key]]
        else:
            config[key] = list_str_keys[key]

    for key in path_list_keys:
        if key in config:
            config[key] = [
                os.path.realpath(os.path.join(os.path.dirname(config_file_path), path))
                for path in config[key]
            ]
        else:  # Default to empty list
            config[key] = []

    assert config["test-framework"] in [
        "pytest",
        "unittest",
    ], "In pyproject.toml, Codeflash only supports the 'test-framework' as pytest and unittest."
    if len(config["formatter-cmds"]) > 0:
        assert config["formatter-cmds"][0] != "your-formatter $file", (
            "The formatter command is not set correctly in pyproject.toml. Please set the "
            "formatter command in the 'formatter-cmds' key. More info - https://docs.codeflash.ai/configuration"
        )
    for key in list(config.keys()):
        if "-" in key:
            config[key.replace("-", "_")] = config[key]
            del config[key]

    return config, config_file_path
