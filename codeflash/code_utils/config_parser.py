import os
from typing import List

import tomlkit

from codeflash.code_utils.config_consts import MIN_IMPROVEMENT_THRESHOLD


def supported_config_keys() -> List[str]:
    return ["test-framework", "tests-root", "module-root"]


def find_pyproject_toml(config_file=None):
    # Find the pyproject.toml file on the root of the project

    if config_file is not None:
        if not config_file.lower().endswith(".toml"):
            raise ValueError(
                f"Config file {config_file} is not a valid toml file. Please recheck the path to pyproject.toml"
            )
        if not os.path.exists(config_file):
            raise ValueError(
                f"Config file {config_file} does not exist. Please recheck the path to pyproject.toml"
            )
        return config_file

    else:
        dir_path = os.getcwd()

        while not os.path.dirname(dir_path) == dir_path:
            config_file = os.path.join(dir_path, "pyproject.toml")
            if os.path.exists(config_file):
                return config_file
            # Search for pyproject.toml in the parent directories
            dir_path = os.path.dirname(dir_path)
        raise ValueError(
            f"Could not find pyproject.toml in the current directory {os.getcwd()} or any of the parent directories. Please create it by running `poetry init`, or pass the path to pyproject.toml with the --config-file argument."
        )


def parse_config_file(config_file_path=None):
    config_file = find_pyproject_toml(config_file_path)
    try:
        with open(config_file, "rb") as f:
            data = tomlkit.parse(f.read())
    except tomlkit.exceptions.ParseError as e:
        raise ValueError(
            f"Error while parsing the config file {config_file}. Please recheck the file for syntax errors. Error: {e}"
        )

    try:
        tool = data["tool"]
        assert isinstance(tool, dict)
        config = tool["codeflash"]
    except tomlkit.exceptions.NonExistentKey:
        raise ValueError(
            f"Could not find the 'codeflash' block in the config file {config_file}. "
            f"Please run 'codeflash init' to create the config file."
        )
    assert isinstance(config, dict)

    # default values:
    path_keys = ["module-root", "tests-root"]
    path_list_keys = ["ignore-paths"]
    # TODO: minimum-peformance-gain should become a more dynamic auto-detection in the future
    float_keys = {
        "minimum-performance-gain": MIN_IMPROVEMENT_THRESHOLD,
    }  # the value is the default value
    str_keys = {
        "pytest-cmd": "pytest",
        "formatter-cmd": "black",
        "imports-cmd": "isort",
    }
    bool_keys = {
        "disable-telemetry": False,
    }

    for key in float_keys:
        if key in config:
            config[key] = float(config[key])
        else:
            config[key] = float_keys[key]
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
            config[key] = os.path.join(os.path.dirname(config_file), config[key])

    for key in path_list_keys:
        if key in config:
            config[key] = [os.path.join(os.path.dirname(config_file), path) for path in config[key]]
        else:  # Default to empty list
            config[key] = []

    assert config["test-framework"] in [
        "pytest",
        "unittest",
    ], "In pyproject.toml, Codeflash only supports the 'test-framework' as pytest and unittest."
    for key in list(config.keys()):
        if "-" in key:
            config[key.replace("-", "_")] = config[key]
            del config[key]

    return config
